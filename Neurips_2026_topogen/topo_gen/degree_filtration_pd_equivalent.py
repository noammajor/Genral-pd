"""PD-equivalent generation under degree filtration (Algorithm 3 — array-based).

Adapts the _State-based vertex filtration to add a node capacity constraint:

  In a degree filtration, node v appears at time k = degree(v), and each edge
  appears at max(degree_u, degree_v).  Every node has a capacity budget equal
  to its birth time k (its degree).  Each edge added to a node decrements its
  capacity by 1.  An edge may only be added when both endpoints have capacity > 0.

  Constraints (in addition to vertex filtration constraints):
    capacity[i] > 0  AND  capacity[j] > 0  for any edge (i, j)

  Nodes born at k=0 (isolated vertices in the original graph) have capacity 0
  and can never connect to anything — consistent with degree-filtration semantics.

Failure modes:
  NO_NEW_VERTEX_PAIR   — stalemate in the merge queue (same as vertex filtration)
  NO_CYCLE_CANDIDATE   — no valid intra-component new-vertex pair for H1 birth
  H0_MERGE_FAILURE     — no valid merge pair sampled (should not fire in practice)
  CAPACITY_EXHAUSTED   — raised if capacity blocks a necessary merge or cycle edge
                         (subsumed under H0_MERGE_FAILURE / NO_CYCLE_CANDIDATE
                          since _sample2d returns -1 when the indicator is all-False)
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from utils.generation_utils import build_index
from topo_gen.persistence import _UF
from topo_gen.result import Failure, FailureReason, GenerationResult, Success
from topo_gen.generation_helpers import _sample2d, _commit, _do_merge


# ---------------------------------------------------------------------------
# State — flat node arrays with capacity
# ---------------------------------------------------------------------------

class _StateDeg:
    """Per-node arrays tracking existence, birth, component membership, edges,
    and remaining degree capacity.

    Arrays (length N = total nodes across all timesteps):
      exists     bool  (N,)   True if node has been added
      comp_id    int   (N,)   current component id (-1 if not added)
      comp_birth int   (N,)   birth time of node's current component (-1 if not added)
      is_new     bool  (N,)   True if born at the current timestep k
      adj        bool  (N,N)  True if edge present (symmetric)
      capacity   int   (N,)   remaining degree budget (initialized to birth time k)
    """

    def __init__(self, n_nodes: int):
        N = self.N = n_nodes
        self.exists     = np.zeros(N, bool)
        self.comp_id    = np.full(N, -1, int)
        self.comp_birth = np.full(N, -1, int)
        self.is_new     = np.zeros(N, bool)
        self.adj        = np.zeros((N, N), bool)
        self.not_self   = ~np.eye(N, dtype=bool)
        self.capacity   = np.zeros(N, int)

    def add_node(self, v: int, birth: int):
        self.exists[v]     = True
        self.comp_id[v]    = v
        self.comp_birth[v] = birth
        self.is_new[v]     = True
        self.capacity[v]   = birth   # degree budget = birth time (= degree in the original graph)

    def add_edge(self, u: int, v: int):
        self.adj[u, v] = self.adj[v, u] = True
        self.capacity[u] -= 1
        self.capacity[v] -= 1

    def clear_new(self):
        self.is_new[:] = False

    def sync(self, uf: _UF):
        """Re-sync comp_id / comp_birth from UF after a merge."""
        for i in range(self.N):
            if not self.exists[i]:
                continue
            r = uf.find(i)
            self.comp_id[i]    = r
            self.comp_birth[i] = uf.born[r]

    # -- capacity mask --

    def _cap_mask(self) -> np.ndarray:
        """(N,N) bool: both endpoints have remaining capacity > 0."""
        has_cap = self.capacity > 0
        return has_cap[:, None] & has_cap[None, :]

    # -- composite 2-D indicators --

    def _one_new(self) -> np.ndarray:
        """(N,N) bool: is_new[i] or is_new[j]."""
        return self.is_new[:, None] | self.is_new[None, :]

    def ind_merge_pair(self, b: int) -> np.ndarray:
        """(N,N) valid merge pairs for death with comp-birth == b."""
        return (
            self.exists[:, None] * self.exists[None, :] *
            (self.comp_birth[:, None] == b) *      # row: dying side
            (self.comp_birth[None, :] <= b) *      # col: surviving side
            (self.comp_id[:, None] != self.comp_id[None, :]) *
            ~self.adj *
            self._one_new() *
            self.not_self *
            self._cap_mask()
        )

    def ind_cycle_pairs(self) -> np.ndarray:
        """(N,N) valid cycle-edge pairs: same comp, no edge, at least one new, capacity."""
        return (
            self.exists[:, None] * self.exists[None, :] *
            (self.comp_id[:, None] == self.comp_id[None, :]) *
            ~self.adj *
            self._one_new() *
            self.not_self *
            self._cap_mask()
        )



# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def degree_filtration_pd_equivalent(
    pd0: list, pd1: list, seed: int = None
) -> GenerationResult:
    """Generate a PD-equivalent graph under degree filtration.

    Parameters
    ----------
    pd0  : list of (birth, death) pairs for H0; death=inf for survivors
           births = degrees in the source graph (so birth > 0 for non-isolated nodes)
    pd1  : list of (birth, inf) pairs for H1
    seed : random seed

    Returns
    -------
    Success(H, node_time, edge_time)  or  Failure(reason, step, timestep, detail)
    """
    rng = np.random.default_rng(seed)

    if not pd0:
        return Success(nx.Graph(), {}, {})

    n, B0, D0, B1 = build_index(pd0, pd1)

    uf        = _UF()
    state     = _StateDeg(len(pd0))
    H         = nx.Graph()
    node_time = {}
    edge_time = {}
    vertex_id = 0

    for k in range(n + 1):

        # Steps 1 & 2: add new isolated vertices (capacity = k = their degree)
        state.clear_new()
        for (b, d) in B0[k]:
            v = vertex_id; vertex_id += 1
            H.add_node(v)
            node_time[v] = k
            uf.add(v, birth=k)
            state.add_node(v, k)

        # Step 3: shuffle all deaths at k, process as a stalemate queue.
        # Non-trivial merges with no valid pair are deferred; a trivial processed
        # in between may unblock them.  Stalemate → Failure(NO_NEW_VERTEX_PAIR).
        queue = list(D0[k])
        rng.shuffle(queue)

        stale = 0
        while queue:
            b, death = queue.pop(0)
            trivial  = (b == k)

            if trivial or np.any(state.ind_merge_pair(b)):
                result = _do_merge(k, state, uf, H, edge_time, rng,
                                   b=b, death=death)
                if isinstance(result, Failure):
                    return result
                stale = 0
            else:
                queue.append((b, death))
                stale += 1
                if stale >= len(queue):
                    return Failure(
                        reason  = FailureReason.NO_NEW_VERTEX_PAIR,
                        step    = 3, timestep = k,
                        detail  = (f"death (b={b}, d={death}): stalemate — "
                                   f"{stale} deferred merges with no progress"),
                    )

        # Step 4: cycle edges (H1 births at k)
        for _ in range(B1[k]):
            u, v = _sample2d(state.ind_cycle_pairs(), rng)
            if u == -1:
                return Failure(
                    reason  = FailureReason.NO_CYCLE_CANDIDATE,
                    step    = 4, timestep = k,
                    detail  = f"H1 birth at k={k}: no valid intra-component new-vertex pair",
                )
            _commit(u, v, k, state, uf, H, edge_time, merge=False)

    return Success(H, node_time, edge_time)
