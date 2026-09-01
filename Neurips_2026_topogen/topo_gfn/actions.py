"""Action space for the PD-compliant graph GFlowNet.

Recasts Algorithm 3 (``topo_gen.degree_filtration_pd_equivalent``) as an MDP
whose two uniform choices become a learned policy.

Follows SynFlowNet's hierarchical action-categorical pattern: a primary choice
over the action TYPE, then a secondary choice over the node PAIR conditioned on
that type.  The two levels are normalised separately, so

    log P(a) = log P(type) + log P(pair | type)

Forward types
-------------
MERGE   (u, v)  ordered: u is the dying side, v the surviving side.  Serves one
                H0 death event.  Mask is asymmetric.
CYCLE   (u, v)  unordered: both endpoints already share a component.  Serves one
                H1 birth.  Mask is symmetric.
EXIT            legal only once every death and cycle quota is exhausted.

Backward types
--------------
BCK_UNMERGE      undo a MERGE edge.
BCK_BREAK_CYCLE  undo a CYCLE edge.

Every edge stores ``code = 2 * timestep + is_cycle``.  The exact parent set of a
state is then the edges attaining the maximum code -- within a timestep the
merge edges form a forest over the contracted component graph, so any ordering
of them reaches the same state, and cycle edges never change components.  The
type must come from this stored code: it is NOT recoverable from a bridge test,
because a merge edge stops being a bridge once a later cycle passes through it.

This module is pure numpy on purpose -- it is the reference the batched torch
masks are tested against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class ActionType(Enum):
    MERGE = 0
    CYCLE = 1
    EXIT = 2


class BckActionType(Enum):
    BCK_UNMERGE = auto()
    BCK_BREAK_CYCLE = auto()


N_FWD_TYPES = 3  # width of the primary (type) head


@dataclass(frozen=True)
class Action:
    """A forward action.  ``u``/``v`` are unused for EXIT."""
    type: ActionType
    u: int = -1
    v: int = -1


@dataclass(frozen=True)
class BckAction:
    type: BckActionType
    u: int = -1
    v: int = -1


# ---------------------------------------------------------------------------
# Target schedule
# ---------------------------------------------------------------------------

@dataclass
class PDSchedule:
    """The deterministic part of the process, derived from the target PD.

    node_time[v] is vertex v's birth time, which under a degree filtration is
    also its intended final degree.  Vertices are numbered in the order
    Algorithm 3 creates them (ascending birth time).

    deaths[k] is the multiset of component-birth times that must die at k,
    stored as a count vector indexed by birth time.
    cycles[k] is the number of H1 births at k.
    """
    n: int                      # max timestep
    node_time: np.ndarray       # (N,) int   birth == intended degree
    deaths: np.ndarray          # (n+1, n+1) int  deaths[k, b] = count
    cycles: np.ndarray          # (n+1,) int

    @property
    def num_nodes(self) -> int:
        return int(self.node_time.shape[0])

    @property
    def num_edges(self) -> int:
        """Total edges the process will place: one per death, one per cycle."""
        return int(self.deaths.sum() + self.cycles.sum())

    @classmethod
    def from_pd(cls, pd0: list, pd1: list) -> "PDSchedule":
        """Build from persistence diagrams in ``topo_gen.persistence`` format.

        pd0 : list of (birth, death); death may be ``float('inf')``
        pd1 : list of (birth, inf)
        """
        finite_d = [d for _, d in pd0 if d != float("inf")]
        n = max(
            max(b for b, _ in pd0),
            max(finite_d, default=0),
            max((b for b, _ in pd1), default=0),
        )
        n = int(n)

        # Vertices in creation order: ascending birth, matching Algorithm 3's
        # `for k` loop over B0[k].
        births = sorted(int(b) for b, _ in pd0)
        node_time = np.array(births, dtype=np.int64)

        deaths = np.zeros((n + 1, n + 1), dtype=np.int64)
        for b, d in pd0:
            if d != float("inf"):
                deaths[int(d), int(b)] += 1

        cycles = np.zeros(n + 1, dtype=np.int64)
        for b, _ in pd1:
            cycles[int(b)] += 1

        return cls(n=n, node_time=node_time, deaths=deaths, cycles=cycles)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class State:
    """MDP state.

    Only ``adj``/``edge_code``/``k``/``deaths_rem``/``cycles_rem`` are carried;
    everything else (capacity, components, existence) is a pure function of
    those plus the schedule, and is recomputed on demand.  Keeping the stored
    state minimal is what makes ``backward_step`` exactly invertible.
    """
    sched: PDSchedule
    k: int = 0
    adj: np.ndarray = field(default=None)        # (N,N) bool
    edge_code: np.ndarray = field(default=None)  # (N,N) int, -1 where no edge
    deaths_rem: np.ndarray = field(default=None) # (n+1, n+1) int
    cycles_rem: np.ndarray = field(default=None) # (n+1,) int

    def __post_init__(self):
        N = self.sched.num_nodes
        if self.adj is None:
            self.adj = np.zeros((N, N), dtype=bool)
        if self.edge_code is None:
            self.edge_code = np.full((N, N), -1, dtype=np.int64)
        if self.deaths_rem is None:
            self.deaths_rem = self.sched.deaths.copy()
        if self.cycles_rem is None:
            self.cycles_rem = self.sched.cycles.copy()

    # -- derived quantities -------------------------------------------------

    @property
    def exists(self) -> np.ndarray:
        return self.sched.node_time <= self.k

    @property
    def is_new(self) -> np.ndarray:
        return self.sched.node_time == self.k

    @property
    def degree(self) -> np.ndarray:
        return self.adj.sum(axis=1).astype(np.int64)

    @property
    def capacity(self) -> np.ndarray:
        """Remaining degree budget.  Provably reaches exactly 0 on completion."""
        return self.sched.node_time - self.degree

    def components(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (comp_id, comp_birth) over existing nodes; -1 where absent.

        comp_id is the minimum node index in the component (a canonical label,
        so it is order-independent).  comp_birth is the elder-rule birth time:
        the minimum node_time in the component.
        """
        N = self.sched.num_nodes
        ex = self.exists
        comp_id = np.full(N, -1, dtype=np.int64)
        for v in range(N):
            if not ex[v] or comp_id[v] != -1:
                continue
            stack, members = [v], []
            comp_id[v] = v
            while stack:
                x = stack.pop()
                members.append(x)
                for y in np.flatnonzero(self.adj[x]):
                    if comp_id[y] == -1:
                        comp_id[y] = v
                        stack.append(int(y))
            root = min(members)
            for m in members:
                comp_id[m] = root

        comp_birth = np.full(N, -1, dtype=np.int64)
        for r in np.unique(comp_id[comp_id >= 0]):
            mask = comp_id == r
            comp_birth[mask] = self.sched.node_time[mask].min()
        return comp_id, comp_birth

    @property
    def edges_placed(self) -> int:
        return int(self.adj.sum() // 2)

    @property
    def done(self) -> bool:
        return self.deaths_rem.sum() == 0 and self.cycles_rem.sum() == 0

    def copy(self) -> "State":
        return State(
            sched=self.sched,
            k=self.k,
            adj=self.adj.copy(),
            edge_code=self.edge_code.copy(),
            deaths_rem=self.deaths_rem.copy(),
            cycles_rem=self.cycles_rem.copy(),
        )


# ---------------------------------------------------------------------------
# Forward masks
# ---------------------------------------------------------------------------

def _base_mask(s: State) -> np.ndarray:
    """Conditions common to both move types: both endpoints exist, no edge yet,
    not a self-loop, at least one endpoint born this timestep, both have
    remaining capacity."""
    ex, new, cap = s.exists, s.is_new, s.capacity
    N = s.sched.num_nodes
    has_cap = cap > 0
    return (
        ex[:, None] & ex[None, :]
        & ~s.adj
        & ~np.eye(N, dtype=bool)
        & (new[:, None] | new[None, :])
        & has_cap[:, None] & has_cap[None, :]
    )


def merge_mask(s: State) -> np.ndarray:
    """(N,N) bool, ORDERED: [u, v] means u's component dies into v's.

    This is the union over all still-unserved deaths at the current timestep of
    Algorithm 3's ``ind_merge_pair(b)``.  Because that mask requires
    ``comp_birth[u] == b and comp_birth[v] <= b``, the union over a set S of
    remaining birth-times is exactly ``comp_birth[u] in S and
    comp_birth[v] <= comp_birth[u]`` -- so the death being served is implied by
    the pair, and the death-queue ordering is absorbed into the pair choice.
    """
    comp_id, cb = s.components()
    rem_b = s.deaths_rem[s.k] > 0            # (n+1,) which births still owe a death
    cb_safe = np.where(cb >= 0, cb, 0)
    u_ok = (cb >= 0) & rem_b[cb_safe]        # u's component-birth still owes a death
    return (
        _base_mask(s)
        & (comp_id[:, None] != comp_id[None, :])
        & u_ok[:, None]
        & (cb[None, :] <= cb[:, None])
    )


def cycle_mask(s: State) -> np.ndarray:
    """(N,N) bool, SYMMETRIC: both endpoints already share a component."""
    if s.cycles_rem[s.k] <= 0:
        return np.zeros_like(s.adj)
    comp_id, _ = s.components()
    return (
        _base_mask(s)
        & (comp_id[:, None] == comp_id[None, :])
        & (comp_id[:, None] >= 0)
    )


def type_mask(s: State) -> np.ndarray:
    """(3,) bool over ActionType -- the mask for the primary head.

    A type is legal iff it still has quota at the current timestep AND has at
    least one legal pair.  EXIT is legal iff every quota is exhausted.
    """
    m = np.zeros(N_FWD_TYPES, dtype=bool)
    if not s.done:
        m[ActionType.MERGE.value] = merge_mask(s).any()
        m[ActionType.CYCLE.value] = cycle_mask(s).any()
    else:
        m[ActionType.EXIT.value] = True
    return m


def is_dead_end(s: State) -> bool:
    """True if quotas remain but no move is legal.  These states are terminal
    with zero reward and must be handled explicitly -- they are common."""
    return not s.done and not type_mask(s).any()


# ---------------------------------------------------------------------------
# Backward mask
# ---------------------------------------------------------------------------

def backward_mask(s: State) -> np.ndarray:
    """(N,N) bool over removable edges -- the EXACT parent set.

    An edge is removable iff its code is maximal.  Its backward type is read off
    the code's low bit, never from a bridge test.
    """
    out = np.zeros_like(s.adj)
    if s.edges_placed == 0:
        return out
    top = s.edge_code.max()
    return (s.edge_code == top) & s.adj


def bck_type_of(s: State, u: int, v: int) -> BckActionType:
    code = int(s.edge_code[u, v])
    if code < 0:
        raise ValueError(f"no edge ({u},{v})")
    return BckActionType.BCK_BREAK_CYCLE if (code & 1) else BckActionType.BCK_UNMERGE


def edge_code_of(timestep: int, is_cycle: bool) -> int:
    return 2 * int(timestep) + int(is_cycle)
