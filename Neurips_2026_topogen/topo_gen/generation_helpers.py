"""Shared helpers for indicator-array-style PD-equivalent generators.

All three filtration variants (vertex, degree, random) use the same
_sample2d / _commit / _do_merge pattern — collected here to avoid
duplication.
"""

from __future__ import annotations

import numpy as np

from topo_gen.result import Failure, FailureReason


def _sample2d(ind: np.ndarray, rng):
    """Uniform sample from True (i,j) positions; (-1,-1) if none."""
    rows, cols = np.where(ind)
    if not len(rows):
        return -1, -1
    idx = int(rng.integers(len(rows)))
    return int(rows[idx]), int(cols[idx])


def _commit(u, v, k, state, uf, H, edge_time, *, merge: bool):
    """Add edge to graph and state; if merge, update UF and re-sync state."""
    e = (min(u, v), max(u, v))
    H.add_edge(*e)
    edge_time[e] = k
    state.add_edge(u, v)
    if merge:
        uf.union(u, v)
        state.sync(uf)


def _do_merge(k, state, uf, H, edge_time, rng, *, b, death):
    """Sample and commit a merge edge for one death event."""
    u, v = _sample2d(state.ind_merge_pair(b), rng)
    if u == -1:
        return Failure(
            reason  = FailureReason.H0_MERGE_FAILURE,
            step    = 3, timestep = k,
            detail  = f"death (b={b}, d={death}): no valid merge pair",
        )
    _commit(u, v, k, state, uf, H, edge_time, merge=True)
    return None
