"""PD-equivalent generation under random filtration (Algorithm 1) — indicator-array style.

In a random filtration edges are not constrained to have a new endpoint; any
non-adjacent pair from different (or same, for cycles) components is valid.

Sampling decisions are expressed directly over nodes via boolean indicator
arrays — the same pattern as vertex_filtration_pd_equivalent and
degree_filtration_pd_equivalent.

  Step 3 — merge pair for death (b, k):
      I[i,j] = exists[i] & exists[j]
             & (comp_birth[i] == b)       # i's component was born at b
             & (comp_birth[j] <= b)       # j's component born <= b
             & (comp_id[i] != comp_id[j]) # different components
             & ~adj[i,j]                  # no existing edge
             & not_self[i,j]

  Step 4 — cycle edge:
      I[i,j] = exists[i] & exists[j]
             & (comp_id[i] == comp_id[j]) # same component
             & ~adj[i,j]                  # no existing edge
             & not_self[i,j]

  No is_new constraint — any existing node may be an endpoint.

Failure modes:
  NO_COMPONENT_BORN_AT_B  — no live component has birth == b at merge time
  NO_OLDER_COMPONENT      — after picking dying component, no valid merge target
  NO_CYCLE_CANDIDATE      — no non-adjacent intra-component pair for H1 birth
  H0_MERGE_FAILURE        — indicator is all-False at sample time (catch-all)
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from utils.generation_utils import build_index
from topo_gen.persistence import _UF
from topo_gen.result import Failure, FailureReason, GenerationResult, Success
from topo_gen.generation_helpers import _sample2d, _commit, _do_merge


# ---------------------------------------------------------------------------
# State — flat node arrays (no is_new, no capacity)
# ---------------------------------------------------------------------------

class _StateRnd:
    """Per-node arrays for random filtration.

    Arrays (length N = total nodes across all timesteps):
      exists     bool  (N,)   True if node has been added
      comp_id    int   (N,)   current component id (-1 if not added)
      comp_birth int   (N,)   birth time of node's current component (-1 if not added)
      adj        bool  (N,N)  True if edge present (symmetric)
    """

    def __init__(self, n_nodes: int):
        N = self.N = n_nodes
        self.exists     = np.zeros(N, bool)
        self.comp_id    = np.full(N, -1, int)
        self.comp_birth = np.full(N, -1, int)
        self.adj        = np.zeros((N, N), bool)
        self.not_self   = ~np.eye(N, dtype=bool)

    def add_node(self, v: int, birth: int):
        self.exists[v]     = True
        self.comp_id[v]    = v
        self.comp_birth[v] = birth

    def add_edge(self, u: int, v: int):
        self.adj[u, v] = self.adj[v, u] = True

    def sync(self, uf: _UF):
        """Re-sync comp_id / comp_birth from UF after a merge."""
        for i in range(self.N):
            if not self.exists[i]:
                continue
            r = uf.find(i)
            self.comp_id[i]    = r
            self.comp_birth[i] = uf.born[r]

    # -- composite 2-D indicators --

    def ind_merge_pair(self, b: int) -> np.ndarray:
        """(N,N) valid merge pairs for death with comp-birth == b.

        No is_new constraint — any node may be an endpoint.
        """
        return (
            self.exists[:, None] & self.exists[None, :] &
            (self.comp_birth[:, None] == b) &       # row: dying side
            (self.comp_birth[None, :] <= b) &       # col: surviving side
            (self.comp_id[:, None] != self.comp_id[None, :]) &
            ~self.adj &
            self.not_self
        )

    def ind_cycle_pairs(self) -> np.ndarray:
        """(N,N) valid cycle-edge pairs: same comp, no edge."""
        return (
            self.exists[:, None] & self.exists[None, :] &
            (self.comp_id[:, None] == self.comp_id[None, :]) &
            ~self.adj &
            self.not_self
        )



# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def random_filtration_pd_equivalent(
    pd0: list, pd1: list, seed: int = None
) -> GenerationResult:
    """Generate a PD-equivalent graph under random filtration (Algorithm 1).

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
    state     = _StateRnd(len(pd0))
    H         = nx.Graph()
    node_time = {}
    edge_time = {}
    vertex_id = 0

    for k in range(n + 1):

        # Steps 1 & 2: add new isolated vertices
        for (b, d) in B0[k]:
            v = vertex_id; vertex_id += 1
            H.add_node(v)
            node_time[v] = k
            uf.add(v, birth=k)
            state.add_node(v, k)

        # Step 3: shuffle deaths at k, process each
        queue = list(D0[k])
        rng.shuffle(queue)

        for b, death in queue:
            result = _do_merge(k, state, uf, H, edge_time, rng, b=b, death=death)
            if isinstance(result, Failure):
                return result

        # Step 4: cycle edges (H1 births at k)
        for _ in range(B1[k]):
            u, v = _sample2d(state.ind_cycle_pairs(), rng)
            if u == -1:
                return Failure(
                    reason  = FailureReason.NO_CYCLE_CANDIDATE,
                    step    = 4, timestep = k,
                    detail  = f"H1 birth at k={k}: no valid intra-component pair",
                )
            _commit(u, v, k, state, uf, H, edge_time, merge=False)

    return Success(H, node_time, edge_time)
