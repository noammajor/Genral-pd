"""The PD-compliant graph MDP.

Structured like SynFlowNet's ``synthesis_building_env``: a ``TopoEnv`` that
implements stepping forward and backward, and a ``TopoEnvContext`` that sits
between the agent and the environment -- mapping states to model inputs,
GraphActions to ActionIndices, and building masks.

Unlike SynFlowNet, the MDP is parameterised by a *target*: ``PDSchedule``
determines the node set, the per-timestep death/cycle quotas and hence the
trajectory length (exactly ``sched.num_edges`` moves, then Stop).

Timestep advance
----------------
Algorithm 3 loops ``for k in range(n+1)``, and at each k marks the vertices born
at k as "new", serves that timestep's deaths, then its H1 births.  A vertex is
"new" only during its own birth timestep.  Timesteps with no quota are no-ops
there, so we skip them: after every move we advance k past any timestep whose
quotas are already empty.  That is behaviourally identical and keeps the
trajectory length equal to the number of edges.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from topo_gfn.actions import (
    DEFAULT_ACTION_TYPE_ORDER,
    DEFAULT_BCK_ACTION_TYPE_ORDER,
    ActionIndex,
    GraphAction,
    GraphActionType,
    PDSchedule,
    State,
    bck_cycle_mask,
    bck_merge_mask,
    cycle_mask,
    edge_code_of,
    merge_mask,
    stop_mask,
    timestep_of_code,
    type_mask,
)


class TopoEnvContext:
    """Interface between the agent and the environment.

    Mirrors ``ReactionTemplateEnvContext``: it owns the action-type ordering,
    converts between ``GraphAction`` and ``ActionIndex``, and creates masks.
    """

    def __init__(self, sched: PDSchedule):
        self.sched = sched
        self.num_nodes = sched.num_nodes
        self.action_type_order = list(DEFAULT_ACTION_TYPE_ORDER)
        self.bck_action_type_order = list(DEFAULT_BCK_ACTION_TYPE_ORDER)

    # -- action type <-> index ---------------------------------------------

    def aidx_to_action_type(self, aidx: ActionIndex, fwd: bool = True) -> GraphActionType:
        order = self.action_type_order if fwd else self.bck_action_type_order
        return order[aidx.action_type]

    def action_type_to_aidx(self, action_type: GraphActionType, fwd: bool = True) -> int:
        order = self.action_type_order if fwd else self.bck_action_type_order
        return order.index(action_type)

    def ActionIndex_to_GraphAction(self, aidx: ActionIndex, fwd: bool = True) -> GraphAction:
        t = self.aidx_to_action_type(aidx, fwd)
        if t is GraphActionType.Stop:
            return GraphAction(t)
        return GraphAction(t, u=aidx.row_idx, v=aidx.col_idx)

    def GraphAction_to_ActionIndex(self, action: GraphAction, fwd: bool = True) -> ActionIndex:
        t = self.action_type_to_aidx(action.action, fwd)
        if action.action is GraphActionType.Stop:
            return ActionIndex(action_type=t, row_idx=0, col_idx=0)
        return ActionIndex(action_type=t, row_idx=action.u, col_idx=action.v)

    # -- masks --------------------------------------------------------------

    def create_masks(self, s: State, action_type: GraphActionType) -> np.ndarray:
        """Mask for one action type, keyed like SynFlowNet's create_masks."""
        if action_type is GraphActionType.Stop:
            return stop_mask(s)
        if action_type is GraphActionType.Merge:
            return merge_mask(s)
        if action_type is GraphActionType.Cycle:
            return cycle_mask(s)
        if action_type is GraphActionType.BckMerge:
            return bck_merge_mask(s)
        if action_type is GraphActionType.BckCycle:
            return bck_cycle_mask(s)
        raise ValueError(f"unknown action type {action_type}")

    def all_masks(self, s: State, fwd: bool = True) -> dict:
        """Every mask for the given direction, keyed by ``mask_name``."""
        order = self.action_type_order if fwd else self.bck_action_type_order
        return {t.mask_name: self.create_masks(s, t) for t in order}

    def type_mask(self, s: State) -> np.ndarray:
        """Primary-head mask over ``action_type_order``."""
        return type_mask(s, self.action_type_order)


class TopoEnv:
    """PD-compliant graph environment.

    New states are the empty graph over the schedule's node set; the process
    ends when every H0 death and H1 birth in the target diagram has been served.
    """

    def __init__(self, sched: PDSchedule):
        self.sched = sched
        self.ctx = TopoEnvContext(sched)

    # -- construction -------------------------------------------------------

    def new(self, soft: bool = False) -> State:
        return self.empty_graph(soft=soft)

    def empty_graph(self, soft: bool = False) -> State:
        s = State(sched=self.sched)
        self._advance(s, soft=soft)
        return s

    def _advance(self, s: State, soft: bool = False) -> None:
        """Move the filtration clock forward (in place).

        Hard mode: skip timesteps whose quotas are already spent.

        Soft mode: quota satisfaction can no longer drive the clock, because a
        violating move consumes no quota -- the clock would freeze at the first
        timestep the policy declines to serve, no later vertex would ever be
        born, and the rollout would pile edges onto one stage forever (measured:
        k stuck at 4, only 12 of 14 vertices ever existing).  So the clock is
        driven by EDGE COUNT instead: timestep k gets exactly the number of
        edges the PD prescribes for it, whichever edges the policy picks, and
        the mismatch is what `violations` records.
        """
        if soft:
            from topo_gfn.actions import edges_due_at
            while s.k < s.sched.n and s.k_edges >= edges_due_at(s.sched, s.k):
                s.k += 1
                s.k_edges = 0
            return
        while (s.k < s.sched.n
               and s.deaths_rem[s.k].sum() == 0
               and s.cycles_rem[s.k] == 0):
            s.k += 1

    # -- forward ------------------------------------------------------------

    def step(self, s: State, action: GraphAction, soft: bool = False) -> State:
        """Apply an action, returning the next state.  Does not mutate ``s``.

        Under ``soft=True`` the PD-compliance requirements (capacity, quota,
        born-this-timestep) are no longer enforced: the move is executed
        anyway, the corresponding quota is decremented only if one is actually
        available, and ``s.violations`` counts what was overspent.  The result
        is a well-formed graph whose persistence diagram may differ from the
        target -- the reward, not the mask, is what discourages that.
        """
        if action.action is GraphActionType.Stop:
            if not (s.done or soft):
                raise ValueError("Stop is only legal once all quotas are spent")
            return s.copy()

        u, v = int(action.u), int(action.v)
        out = s.copy()

        if action.action is GraphActionType.Merge:
            if not soft and not merge_mask(s)[u, v]:
                raise ValueError(f"illegal Merge ({u},{v}) at k={s.k}")
            _, cb = s.components()
            b = int(cb[u])                       # u is the DYING side
            if not soft or (b >= 0 and out.deaths_rem[s.k, b] > 0):
                out.deaths_rem[s.k, b] -= 1
            else:
                out.violations += 1              # no death quota to spend
            code = edge_code_of(s.k, is_cycle=False)
        elif action.action is GraphActionType.Cycle:
            if not soft and not cycle_mask(s)[u, v]:
                raise ValueError(f"illegal Cycle ({u},{v}) at k={s.k}")
            if not soft or out.cycles_rem[s.k] > 0:
                out.cycles_rem[s.k] -= 1
            else:
                out.violations += 1              # no cycle quota to spend
            code = edge_code_of(s.k, is_cycle=True)
        else:
            raise ValueError(f"{action.action} is not a forward action")

        if soft and (s.capacity[u] <= 0 or s.capacity[v] <= 0):
            out.violations += 1                  # degree will exceed node_time

        out.adj[u, v] = out.adj[v, u] = True
        out.edge_code[u, v] = out.edge_code[v, u] = code
        out.k_edges += 1
        self._advance(out, soft=soft)
        return out

    # -- backward -----------------------------------------------------------

    def backward_step(self, s: State, action: GraphAction) -> State:
        """Apply a backward action, returning the parent state.

        The removed edge's stored code gives both the timestep to rewind to and
        the quota to restore, so no history is needed.
        """
        u, v = int(action.u), int(action.v)
        code = int(s.edge_code[u, v])
        if code < 0:
            raise ValueError(f"no edge ({u},{v}) to remove")

        expected = self.ctx.create_masks(s, action.action)
        if not expected[u, v]:
            raise ValueError(f"illegal {action.action} ({u},{v})")

        out = s.copy()
        out.adj[u, v] = out.adj[v, u] = False
        out.edge_code[u, v] = out.edge_code[v, u] = -1
        out.k = timestep_of_code(code)

        if code % 2 == 1:                        # was a Cycle
            out.cycles_rem[out.k] += 1
        else:                                    # was a Merge
            # adj is symmetric, so the merge's orientation is not stored.  It is
            # recoverable: removal splits the component in two, and by the elder
            # rule the side that died is the one with the LARGER birth time
            # (merge_mask requires comp_birth[v] <= comp_birth[u]).
            dying = self._dying_side(out, u, v)
            _, cb = out.components()
            out.deaths_rem[out.k, int(cb[dying])] += 1
        return out

    @staticmethod
    def _dying_side(parent: State, u: int, v: int) -> int:
        """Which endpoint's component died, in the state before the merge."""
        _, cb = parent.components()
        return u if cb[u] >= cb[v] else v

    def reverse(self, s: State, action: GraphAction) -> GraphAction:
        """The action undoing ``action``.

        ``s`` is always the state the FORWARD action applies to (i.e. the
        parent), because orienting a Merge requires its component births.
        """
        if action.action is GraphActionType.Merge:
            return GraphAction(GraphActionType.BckMerge, u=action.u, v=action.v)
        if action.action is GraphActionType.Cycle:
            return GraphAction(GraphActionType.BckCycle, u=action.u, v=action.v)
        if action.action is GraphActionType.BckMerge:
            # orient so u is the dying side, as merge_mask requires
            d = self._dying_side(s, action.u, action.v)
            o = action.v if d == action.u else action.u
            return GraphAction(GraphActionType.Merge, u=d, v=o)
        if action.action is GraphActionType.BckCycle:
            return GraphAction(GraphActionType.Cycle, u=action.u, v=action.v)
        raise ValueError(f"cannot reverse {action.action}")

    def parents(self, s: State) -> List[GraphAction]:
        """Backward actions leading to each parent of ``s``."""
        out = []
        for t in self.ctx.bck_action_type_order:
            m = self.ctx.create_masks(s, t)
            iu, iv = np.triu_indices(self.sched.num_nodes, 1)
            for u, v in zip(iu[m[iu, iv]].tolist(), iv[m[iu, iv]].tolist()):
                out.append(GraphAction(t, u=u, v=v))
        return out

    def count_backward_transitions(self, s: State) -> int:
        """|Parents(s)|.  Exact: the edges attaining the maximum edge code."""
        return len(self.parents(s))

    # -- termination --------------------------------------------------------

    def is_terminal(self, s: State) -> bool:
        return s.done

    def is_dead_end(self, s: State) -> bool:
        """Quotas remain but no move is legal -- terminal with zero reward."""
        return (not s.done) and (not self.ctx.type_mask(s).any())
