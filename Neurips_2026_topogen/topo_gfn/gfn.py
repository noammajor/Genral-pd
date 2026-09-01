"""Trajectory balance for the PD-compliant graph MDP.

    loss = ( log Z + sum_t log P_F(a_t | s_t)
             - log R(x)
             - sum_t log P_B(s_t | s_{t+1}) ) ^ 2

P_B is UNIFORM over the parent set, which here is exact and parameter-free:
every edge carries ``code = 2 * timestep + is_cycle``, and the parents of a
state are the edges attaining the maximum code.  Within a timestep the merge
edges form a forest over the contracted component graph so any ordering of them
reaches the same state, and cycle edges never change components.  So

    log P_B(s_t | s_{t+1}) = -log |parents(s_{t+1})|

Measured branching is about 2.9 parents per state, i.e. the DAG is genuinely not
a tree -- P_B is load-bearing and cannot be set to 1.

Reward
------
    log R(x) = beta * s(x)   if the trajectory completed
             = FAIL_LOGR     if it dead-ended (quotas left, no legal move)

Completion already implies exact PD compliance (proven, and verified over
~10k runs), so the indicator carries all the topological constraint and s(x)
only shapes the density inside the compliant set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from topo_gfn.actions import GraphActionType, State
from topo_gfn.env import TopoEnv
from topo_gfn.policy import TopoActionCategorical, batch_masks, batch_states

FAIL_LOGR = -50.0


@dataclass
class Trajectory:
    """One rollout.  ``log_pf`` and ``log_pb`` are summed over the trajectory."""
    env: TopoEnv
    log_pf: torch.Tensor                    # scalar, differentiable
    log_pb: float                           # scalar, constant (uniform P_B)
    terminal: State
    completed: bool
    n_steps: int
    log_reward: float = 0.0
    states: List[State] = field(default_factory=list)


class TopoSampler:
    """Rolls out a batch of trajectories, one per target environment."""

    def __init__(self, model, max_steps_slack: int = 4):
        self.model = model
        self.max_steps_slack = max_steps_slack

    @torch.no_grad()
    def _noop(self):
        pass

    def sample(self, envs: List[TopoEnv], cond: Optional[torch.Tensor] = None,
               keep_states: bool = False) -> List[Trajectory]:
        B = len(envs)
        states = [e.new() for e in envs]
        log_pf = [torch.zeros((), device=self._device()) for _ in range(B)]
        log_pb = [0.0] * B
        steps = [0] * B
        active = [True] * B
        completed = [False] * B
        seen = [[s] for s in states] if keep_states else [[] for _ in range(B)]

        # every trajectory is exactly sched.num_edges moves plus Stop
        budget = max(e.sched.num_edges for e in envs) + self.max_steps_slack

        for _ in range(budget):
            idx = [i for i in range(B) if active[i]]
            if not idx:
                break

            # a dead end terminates immediately with zero reward
            still = []
            for i in idx:
                if envs[i].is_dead_end(states[i]):
                    active[i] = False
                else:
                    still.append(i)
            if not still:
                break

            batch = [states[i] for i in still]
            x, adj, nm = batch_states(batch)
            tm, mm, cm = batch_masks(batch)
            dev = self._device()
            tl, ml, cl = self.model(x.to(dev), adj.to(dev), nm.to(dev),
                                    None if cond is None else cond[still].to(dev))
            cat = TopoActionCategorical(tl, ml, cl, tm, mm, cm)
            aidx = cat.sample()
            lp = cat.log_prob(aidx)

            for j, i in enumerate(still):
                env = envs[i]
                a = env.ctx.ActionIndex_to_GraphAction(aidx[j])
                log_pf[i] = log_pf[i] + lp[j]
                if a.action is GraphActionType.Stop:
                    active[i] = False
                    completed[i] = True
                    continue
                nxt = env.step(states[i], a)
                # uniform P_B over the exact parent set of the CHILD
                npar = env.count_backward_transitions(nxt)
                log_pb[i] += -np.log(max(1, npar))
                states[i] = nxt
                steps[i] += 1
                if keep_states:
                    seen[i].append(nxt)

        return [
            Trajectory(env=envs[i], log_pf=log_pf[i], log_pb=log_pb[i],
                       terminal=states[i], completed=completed[i],
                       n_steps=steps[i], states=seen[i])
            for i in range(B)
        ]

    def _device(self):
        return next(self.model.parameters()).device


class TrajectoryBalance:
    """The TB objective."""

    def __init__(self, model, scorer, beta: float = 1.0,
                 fail_logr: float = FAIL_LOGR):
        self.model = model
        self.scorer = scorer
        self.beta = beta
        self.fail_logr = fail_logr

    def log_reward(self, traj: Trajectory) -> float:
        if not traj.completed:
            return self.fail_logr
        n = traj.env.sched.num_nodes
        adj = traj.terminal.adj[:n, :n]
        return float(self.beta * self.scorer.score(adj))

    def compute_loss(self, trajs: List[Trajectory],
                     cond: Optional[torch.Tensor] = None):
        dev = next(self.model.parameters()).device
        logZ = self.model.logZ(cond).squeeze(-1)          # (B,) or (1,)
        if logZ.numel() == 1 and len(trajs) > 1:
            logZ = logZ.expand(len(trajs))

        logr, lpf, lpb = [], [], []
        for t in trajs:
            t.log_reward = self.log_reward(t)
            logr.append(t.log_reward)
            lpf.append(t.log_pf)
            lpb.append(t.log_pb)

        log_r = torch.tensor(logr, dtype=torch.float32, device=dev)
        log_pf = torch.stack(lpf).to(dev)
        log_pb = torch.tensor(lpb, dtype=torch.float32, device=dev)

        resid = logZ + log_pf - log_r - log_pb
        loss = (resid ** 2).mean()

        info = {
            "loss": float(loss.detach()),
            "logZ": float(logZ.detach().mean()),
            "log_pf": float(log_pf.detach().mean()),
            "log_pb": float(log_pb.mean()),
            "log_r": float(log_r.mean()),
            "completion_rate": float(np.mean([t.completed for t in trajs])),
            "mean_steps": float(np.mean([t.n_steps for t in trajs])),
        }
        done = [t for t in trajs if t.completed]
        if done:
            info["log_r_completed"] = float(np.mean([t.log_reward for t in done]))
        return loss, info
