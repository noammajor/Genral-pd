"""Action space for the PD-compliant graph GFlowNet.

Recasts Algorithm 3 (``topo_gen.degree_filtration_pd_equivalent``) as an MDP
whose two uniform choices become a learned policy.

Naming and structure deliberately mirror SynFlowNet
(``synflownet/envs/graph_building_env.py``): a ``GraphActionType`` enum with
``cname`` / ``mask_name`` / ``is_backward``, an ``ActionIndex`` NamedTuple of
``(action_type, row_idx, col_idx)``, and a ``GraphAction`` object.

Action types
------------
Forward
    Stop    terminate; legal only once every death and cycle quota is spent.
    Merge   (u, v) ORDERED -- u's component dies into v's.  Serves one H0 death.
    Cycle   (u, v) SYMMETRIC -- both endpoints already share a component.
            Serves one H1 birth.
Backward
    BckMerge   "unmerge": undo a Merge edge.
    BckCycle   "break cycle": undo a Cycle edge.

As in SynFlowNet the policy is hierarchical: a primary categorical over the
action type, then a secondary categorical over the node pair conditioned on it,
normalised separately, so ``log P(a) = log P(type) + log P(pair | type)``.

Backward typing
---------------
Every edge stores ``code = 2 * timestep + is_cycle``.  The exact parent set of a
state is the edges attaining the maximum code: within a timestep the merge edges
form a forest over the contracted component graph, so any ordering of them
reaches the same state, and cycle edges never change components.

The type MUST come from this stored code.  It is NOT recoverable from a bridge
test on the current graph -- a merge edge stops being a bridge once a later
cycle passes through it (measured: 46-54% disagreement on cyclic graphs).

This module is pure numpy on purpose -- it is the reference the batched torch
masks are tested against.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple, Optional

import numpy as np


class GraphActionType(enum.Enum):
    # Forward actions
    Stop = enum.auto()
    Merge = enum.auto()
    Cycle = enum.auto()
    # Backward actions
    BckMerge = enum.auto()   # "unmerge"
    BckCycle = enum.auto()   # "break cycle"

    @cached_property
    def cname(self):
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.name).lower()

    @cached_property
    def mask_name(self):
        return self.cname + "_mask"

    @cached_property
    def is_backward(self):
        return self.name.startswith(("Bck", "Remove"))


class ActionIndex(NamedTuple):
    """Index of an action, mirroring SynFlowNet's ActionIndex.

    Different action types produce logit tensors of different shapes: Stop is
    (1, 1) while Merge and Cycle are (n, n) over node pairs, so it is convenient
    to carry the action as a tuple of indices.
    """

    action_type: int              # index into action_type_order / bck_action_type_order
    row_idx: Optional[int] = None  # u -- for Merge this is the DYING side
    col_idx: Optional[int] = None  # v -- for Merge this is the SURVIVING side


class GraphAction:
    def __init__(self, action: GraphActionType, u: int = None, v: int = None):
        """A single graph-building action.

        Parameters
        ----------
        action : GraphActionType
            the action type
        u : int, optional
            first endpoint.  For Merge this is the dying side; Stop has none.
        v : int, optional
            second endpoint.  For Merge this is the surviving side.
        """
        self.action: GraphActionType = action
        self.u: Optional[int] = u
        self.v: Optional[int] = v

    def __repr__(self):
        attrs = ", ".join(str(i) for i in [self.u, self.v] if i is not None)
        return f"<{self.action}, {attrs}>"

    def __eq__(self, other):
        return (isinstance(other, GraphAction) and self.action is other.action
                and self.u == other.u and self.v == other.v)

    def __hash__(self):
        return hash((self.action, self.u, self.v))


# Order fixes the layout the model's logit list must match, exactly as
# SynFlowNet's ReactionTemplateEnvContext.action_type_order does.
DEFAULT_ACTION_TYPE_ORDER = [
    GraphActionType.Stop,
    GraphActionType.Merge,
    GraphActionType.Cycle,
]
DEFAULT_BCK_ACTION_TYPE_ORDER = [
    GraphActionType.BckMerge,
    GraphActionType.BckCycle,
]


# ---------------------------------------------------------------------------
# Target schedule
# ---------------------------------------------------------------------------

@dataclass
class PDSchedule:
    """The deterministic part of the process, derived from the target PD.

    ``node_time[v]`` is vertex v's birth time, which under a degree filtration
    is also its intended final degree.  Vertices are numbered in the order
    Algorithm 3 creates them (ascending birth time).

    ``deaths[k, b]`` is how many components of birth-time b must die at k.
    ``cycles[k]`` is the number of H1 births at k.
    """
    n: int                  # max timestep
    node_time: np.ndarray   # (N,) int   birth == intended degree
    deaths: np.ndarray      # (n+1, n+1) int
    cycles: np.ndarray      # (n+1,) int

    @property
    def num_nodes(self) -> int:
        return int(self.node_time.shape[0])

    @property
    def num_edges(self) -> int:
        """Total edges the process places: one per death, one per cycle."""
        return int(self.deaths.sum() + self.cycles.sum())

    @classmethod
    def from_pd(cls, pd0: list, pd1: list) -> "PDSchedule":
        """Build from diagrams in ``topo_gen.persistence`` format.

        pd0 : list of (birth, death); death may be ``float('inf')``
        pd1 : list of (birth, inf)
        """
        finite_d = [d for _, d in pd0 if d != float("inf")]
        n = int(max(
            max(b for b, _ in pd0),
            max(finite_d, default=0),
            max((b for b, _ in pd1), default=0),
        ))

        node_time = np.array(sorted(int(b) for b, _ in pd0), dtype=np.int64)

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

    Only ``adj`` / ``edge_code`` / ``k`` / ``deaths_rem`` / ``cycles_rem`` are
    carried.  Capacity, components and existence are pure functions of those
    plus the schedule and are recomputed on demand -- that minimality is what
    makes ``backward_step`` exactly invertible.
    """
    sched: PDSchedule
    k: int = 0
    adj: np.ndarray = field(default=None)         # (N,N) bool
    edge_code: np.ndarray = field(default=None)   # (N,N) int, -1 where no edge
    deaths_rem: np.ndarray = field(default=None)  # (n+1, n+1) int
    cycles_rem: np.ndarray = field(default=None)  # (n+1,) int

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
        """(comp_id, comp_birth) over existing nodes; -1 where absent.

        comp_id is the minimum node index in the component -- a canonical,
        order-independent label.  comp_birth is the elder-rule birth: the
        minimum node_time in the component.
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
            m = comp_id == r
            comp_birth[m] = self.sched.node_time[m].min()
        return comp_id, comp_birth

    @property
    def edges_placed(self) -> int:
        return int(self.adj.sum() // 2)

    @property
    def done(self) -> bool:
        return self.deaths_rem.sum() == 0 and self.cycles_rem.sum() == 0

    def edge_list(self) -> list:
        iu, iv = np.triu_indices(self.sched.num_nodes, 1)
        sel = self.adj[iu, iv]
        return list(zip(iu[sel].tolist(), iv[sel].tolist()))

    def copy(self) -> "State":
        return State(
            sched=self.sched, k=self.k,
            adj=self.adj.copy(), edge_code=self.edge_code.copy(),
            deaths_rem=self.deaths_rem.copy(), cycles_rem=self.cycles_rem.copy(),
        )


# ---------------------------------------------------------------------------
# Masks -- one function per action type, named to match GraphActionType.cname
# ---------------------------------------------------------------------------

def _base_mask(s: State) -> np.ndarray:
    """Shared by both move types: both endpoints exist, no edge yet, not a
    self-loop, at least one endpoint born this timestep, both have capacity."""
    ex, new, cap = s.exists, s.is_new, s.capacity
    has_cap = cap > 0
    return (
        ex[:, None] & ex[None, :]
        & ~s.adj
        & ~np.eye(s.sched.num_nodes, dtype=bool)
        & (new[:, None] | new[None, :])
        & has_cap[:, None] & has_cap[None, :]
    )


def merge_mask(s: State) -> np.ndarray:
    """(N,N) bool, ORDERED: [u, v] means u's component dies into v's.

    This is the union over all still-unserved deaths at the current timestep of
    Algorithm 3's ``ind_merge_pair(b)``.  That mask requires
    ``comp_birth[u] == b and comp_birth[v] <= b``, so the union over a set S of
    remaining birth-times is exactly ``comp_birth[u] in S and
    comp_birth[v] <= comp_birth[u]``.  The death being served is therefore
    implied by the pair, which absorbs Algorithm 3's death-queue ordering
    (its line 175 ``rng.shuffle``) into the pair choice.
    """
    comp_id, cb = s.components()
    rem_b = s.deaths_rem[s.k] > 0
    cb_safe = np.where(cb >= 0, cb, 0)
    u_ok = (cb >= 0) & rem_b[cb_safe]
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


def stop_mask(s: State) -> np.ndarray:
    """(1,1) bool -- Stop is legal only once every quota is spent."""
    return np.array([[s.done]], dtype=bool)


def type_mask(s: State, action_type_order=None) -> np.ndarray:
    """Mask for the PRIMARY head, one entry per forward action type.

    A move type is legal iff it still has quota at the current timestep AND has
    at least one legal pair.
    """
    order = action_type_order or DEFAULT_ACTION_TYPE_ORDER
    out = np.zeros(len(order), dtype=bool)
    for i, t in enumerate(order):
        if t is GraphActionType.Stop:
            out[i] = bool(s.done)
        elif t is GraphActionType.Merge:
            out[i] = (not s.done) and bool(merge_mask(s).any())
        elif t is GraphActionType.Cycle:
            out[i] = (not s.done) and bool(cycle_mask(s).any())
    return out


def is_dead_end(s: State) -> bool:
    """Quotas remain but no move is legal.  Terminal with zero reward; common
    enough that the policy needs an explicit always-available fallback."""
    return not s.done and not type_mask(s).any()


# -- backward -----------------------------------------------------------------

def _removable(s: State) -> np.ndarray:
    """(N,N) bool: edges attaining the maximum code -- the exact parent set."""
    if s.edges_placed == 0:
        return np.zeros_like(s.adj)
    return (s.edge_code == s.edge_code.max()) & s.adj


def bck_merge_mask(s: State) -> np.ndarray:
    """Removable edges that were placed as merges (even code)."""
    return _removable(s) & (s.edge_code % 2 == 0)


def bck_cycle_mask(s: State) -> np.ndarray:
    """Removable edges that were placed as cycles (odd code)."""
    return _removable(s) & (s.edge_code % 2 == 1)


def backward_mask(s: State) -> np.ndarray:
    """All removable edges, regardless of type."""
    return _removable(s)


def bck_type_of(s: State, u: int, v: int) -> GraphActionType:
    code = int(s.edge_code[u, v])
    if code < 0:
        raise ValueError(f"no edge ({u},{v})")
    return GraphActionType.BckCycle if (code & 1) else GraphActionType.BckMerge


def edge_code_of(timestep: int, is_cycle: bool) -> int:
    return 2 * int(timestep) + int(is_cycle)


def timestep_of_code(code: int) -> int:
    return int(code) // 2
