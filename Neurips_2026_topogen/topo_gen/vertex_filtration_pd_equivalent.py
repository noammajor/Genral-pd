"""PD-equivalent generation under vertex filtration (Algorithm 2).

In a vertex filtration every edge appears at max(t_u, t_v), so every edge
added at stage k must have at least one endpoint born at k (a "new" vertex).

Sampling decisions are expressed directly over nodes via boolean indicator arrays.
There is no two-step "pick root then pick node" — nodes carry their own
comp_id and comp_birth, so all pair constraints are elementwise products
over (N × N) node-pair space.

  Step 3 — merge pair for death (b, k):
      I[i,j] = (comp_birth[i] == b)               # i's component was born at b
             * (comp_birth[j] <= b)               # j's component born <= b
             * (comp_id[i] != comp_id[j])         # different components
             * (is_new[i] | is_new[j])            # at least one endpoint born at k

  Step 3 queue — deaths are shuffled then processed as a FIFO queue.
      A non-trivial merge with no valid new-vertex pair is deferred to the
      back of the queue; a trivial processed in between may unblock it.
      Stalemate (full pass with no progress) → Failure(NO_NEW_VERTEX_PAIR).

  Step 4 — cycle edge:
      I[i,j] = (comp_id[i] == comp_id[j])        # same component
             * (adj[i,j] == 0)                    # no edge yet
             * (is_new[i] | is_new[j])            # at least one endpoint born at k
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from utils.generation_utils import build_index
from topo_gen.persistence import _UF
from topo_gen.result import Failure, FailureReason, GenerationResult, Success
from topo_gen.generation_helpers import _sample2d, _commit, _do_merge


# ---------------------------------------------------------------------------
# State — flat node arrays
# ---------------------------------------------------------------------------

class _State:
    """Per-node arrays tracking existence, birth, component membership, edges.

    Arrays (length N = total nodes across all timesteps):
      exists     bool  (N,)   True if node has been added
      comp_id    int   (N,)   current component id (-1 if not added)
      comp_birth int   (N,)   birth time of node's current component (-1 if not added)
      is_new     bool  (N,)   True if born at the current timestep k
      adj        bool  (N,N)  True if edge present (symmetric)
    """

    def __init__(self, n_nodes: int):
        N = self.N = n_nodes
        self.exists     = np.zeros(N, bool)
        self.comp_id    = np.full(N, -1, int)
        self.comp_birth = np.full(N, -1, int)
        self.is_new     = np.zeros(N, bool)
        self.adj        = np.zeros((N, N), bool)
        self.not_self   = ~np.eye(N, dtype=bool)

    def add_node(self, v: int, birth: int):
        self.exists[v]     = True
        self.comp_id[v]    = v
        self.comp_birth[v] = birth
        self.is_new[v]     = True

    def add_edge(self, u: int, v: int):
        self.adj[u, v] = self.adj[v, u] = True

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
            self.not_self
        )

    # ind_trivial_pair is subsumed by ind_merge_pair(b=k):
    # - comp_birth == k selects only new nodes on the row (same as is_new)
    # - comp_birth <= k is true for all existing nodes (same as exists on col)
    # - _one_new is redundant since the row node is always new
    # def ind_trivial_pair(self) -> np.ndarray:
    #     return (
    #         self.is_new[:, None] *
    #         self.exists[None, :] *
    #         (self.comp_id[:, None] != self.comp_id[None, :]) *
    #         self.not_self
    #     )

    def ind_cycle_pairs(self) -> np.ndarray:
        """(N,N) valid cycle-edge pairs: same comp, no edge, at least one new."""
        return (
            self.exists[:, None] * self.exists[None, :] *
            (self.comp_id[:, None] == self.comp_id[None, :]) *
            ~self.adj *
            self._one_new() *
            self.not_self
        )


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def vertex_filtration_pd_equivalent(
    pd0: list, pd1: list, seed: int = None
) -> GenerationResult:
    """Generate a PD-equivalent graph under vertex filtration.

    Parameters
    ----------
    pd0  : list of (birth, death) pairs for H0; death=inf for survivors
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
    state     = _State(len(pd0))
    H         = nx.Graph()
    node_time = {}
    edge_time = {}
    vertex_id = 0

    for k in range(n + 1):

        # Steps 1 & 2: add new isolated vertices
        state.clear_new()
        for (b, d) in B0[k]:
            v = vertex_id; vertex_id += 1
            H.add_node(v)
            node_time[v] = k
            uf.add(v, birth=k)
            state.add_node(v, k)

        # Step 3: shuffle all deaths at k, process as a queue
        # Non-trivial merges that have no valid new-vertex pair are deferred
        # to the back; a trivial processed in between may unblock them.
        # Stalemate is detected when a full pass yields no progress.
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

