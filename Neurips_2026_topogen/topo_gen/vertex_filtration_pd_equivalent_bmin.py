"""PD-equivalent generation under vertex filtration — b_min heuristic (Algorithm 2b).

Identical to vertex_filtration_pd_equivalent.py except for Step 3:

  Before the main queue, if there are any deaths at k and any trivial deaths
  (b == k) available, we perform one forced trivial merge guided by b_min:

    Let b_min = min birth among all deaths at k (trivial or non-trivial).
    Count n_bmin = number of distinct components currently born at b_min.

    - If n_bmin == 1: randomly pick an old node from a component born at
      t < b_min, connect it to a randomly chosen new vertex (born at k).
    - If n_bmin > 1:  randomly pick an old node from a component born at
      t <= b_min, connect it to a randomly chosen new vertex (born at k).

  This consumes one trivial death from the queue and, by the observation in
  the paper, always has a valid pair — so it never stalemates.  It also
  merges a new vertex into an older component, which can unblock pending
  non-trivial merges for b_min in the remaining queue.

  The rest of the queue is shuffled and handled with the existing stalemate
  logic unchanged.  NO_NEW_VERTEX_PAIR should become structurally rare;
  NO_CYCLE_CANDIDATE (Step 4) is unaffected.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from utils.generation_utils import build_index
from topo_gen.persistence import _UF
from topo_gen.result import Failure, FailureReason, GenerationResult, Success
from topo_gen.vertex_filtration_pd_equivalent import _State
from topo_gen.generation_helpers import _sample2d, _commit, _do_merge


def vertex_filtration_pd_equivalent_bmin(
    pd0: list, pd1: list, seed: int = None
) -> GenerationResult:
    """Generate a PD-equivalent graph under vertex filtration (b_min heuristic).

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

        # Step 3: process deaths at k.
        deaths   = list(D0[k])
        trivials = [(b, d) for b, d in deaths if b == k]

        # b_min forced trivial merge: if there are deaths and a trivial death
        # is available to consume, perform one guided merge before the main queue.
        if deaths and trivials:
            b_min   = min(b for b, _ in deaths)
            n_bmin  = len(set(
                state.comp_id[i]
                for i in range(state.N)
                if state.exists[i] and state.comp_birth[i] == b_min
            ))

            # Build indicator: new vertex (col) × eligible old node (row).
            # Row eligibility: comp_birth < b_min (case 1) or <= b_min (case 2).
            threshold = b_min if n_bmin > 1 else b_min - 1  # strict vs non-strict
            old_mask = (
                state.exists *
                ~state.is_new *
                (state.comp_birth <= threshold)
            )
            new_mask = state.is_new  # nodes born at k

            # ind[i, j]: old node i × new node j, no existing edge, not same node
            ind = (
                old_mask[:, None] * new_mask[None, :] *
                ~state.adj *
                state.not_self
            )

            u, v = _sample2d(ind, rng)
            if u != -1:
                # Consume one trivial death — all trivials are (k, k), so any will do
                deaths.remove(trivials[0])
                _commit(u, v, k, state, uf, H, edge_time, merge=True)

        # Remaining deaths: shuffle and process with stalemate queue
        queue = deaths
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
