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
    """One rollout.

    Rollout happens under no_grad and only RECORDS (state, action) pairs;
    log P_F is recomputed in the loss.  Building the autograd graph during the
    rollout instead retains every step's activations -- at 149 steps x batch 64
    x (126x126) pair logits that is several GB, and it OOM-killed every enzymes
    job at 8GB.  torchgfn and SynFlowNet both recompute for the same reason.
    """
    env: TopoEnv
    log_pb: float                           # constant (uniform P_B)
    terminal: State
    completed: bool
    n_steps: int
    states: List[State] = field(default_factory=list)   # visited, pre-action
    actions: List[object] = field(default_factory=list)  # ActionIndex per state
    log_reward: float = 0.0


class TopoSampler:
    """Rolls out a batch of trajectories, one per target environment."""

    def __init__(self, model, max_steps_slack: int = 4):
        self.model = model
        self.max_steps_slack = max_steps_slack

    @torch.no_grad()
    def sample(self, envs: List[TopoEnv], cond: Optional[torch.Tensor] = None,
               keep_states: bool = True) -> List[Trajectory]:
        """Roll out under no_grad, recording (state, action) for the loss."""
        B = len(envs)
        states = [e.new() for e in envs]
        log_pb = [0.0] * B
        steps = [0] * B
        active = [True] * B
        completed = [False] * B
        seen = [[] for _ in range(B)]
        acts_taken = [[] for _ in range(B)]

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

            for j, i in enumerate(still):
                env = envs[i]
                a = env.ctx.ActionIndex_to_GraphAction(aidx[j])
                seen[i].append(states[i])
                acts_taken[i].append(aidx[j])
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

        return [
            Trajectory(env=envs[i], log_pb=log_pb[i], terminal=states[i],
                       completed=completed[i], n_steps=steps[i],
                       states=seen[i], actions=acts_taken[i])
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

    def _log_pf(self, trajs: List[Trajectory], chunk: int = 256) -> torch.Tensor:
        """Recompute sum_t log P_F(a_t | s_t) per trajectory.

        Flattens every (state, action) pair across the given trajectories and
        re-runs the model in chunks, scattering the per-step log-probs back to
        their trajectory with a differentiable index_add.
        """
        dev = next(self.model.parameters()).device
        flat_s, flat_a, owner = [], [], []
        for ti, t in enumerate(trajs):
            for s, a in zip(t.states, t.actions):
                flat_s.append(s)
                flat_a.append(a)
                owner.append(ti)

        out = torch.zeros(len(trajs), device=dev)
        if not flat_s:
            return out
        idx = torch.tensor(owner, dtype=torch.long, device=dev)

        for i in range(0, len(flat_s), chunk):
            bs = flat_s[i:i + chunk]
            ba = flat_a[i:i + chunk]
            x, adj, nm = batch_states(bs)
            tm, mm, cm = batch_masks(bs)
            tl, ml, cl = self.model(x.to(dev), adj.to(dev), nm.to(dev))
            cat = TopoActionCategorical(tl, ml, cl, tm, mm, cm)
            out = out.index_add(0, idx[i:i + chunk], cat.log_prob(ba))
        return out

    def backward_and_info(self, trajs: List[Trajectory], micro_bs: int = 8,
                          chunk: int = 256, cond: Optional[torch.Tensor] = None):
        """Accumulate the TB gradient over micro-batches of trajectories.

        The loss is per-trajectory, so sum log P_F for one trajectory must live
        in the graph all at once -- but different trajectories need not.  Doing
        backward per micro-batch therefore bounds peak memory by micro_bs
        rollouts instead of the full batch, which is what the enzymes OOM was.
        """
        dev = next(self.model.parameters()).device
        for t in trajs:
            t.log_reward = self.log_reward(t)

        n = len(trajs)
        tot_loss = 0.0
        pf_all, resid_all = [], []
        for i in range(0, n, micro_bs):
            mb = trajs[i:i + micro_bs]
            logZ = self.model.logZ(cond).squeeze(-1)
            if logZ.numel() == 1:
                logZ = logZ.expand(len(mb))
            log_pf = self._log_pf(mb, chunk=chunk)
            log_r = torch.tensor([t.log_reward for t in mb], dtype=torch.float32, device=dev)
            log_pb = torch.tensor([t.log_pb for t in mb], dtype=torch.float32, device=dev)

            resid = logZ + log_pf - log_r - log_pb
            loss_mb = (resid ** 2).mean() * (len(mb) / n)
            loss_mb.backward()

            tot_loss += float(loss_mb.detach())
            pf_all.append(log_pf.detach())
            resid_all.append(resid.detach())

        log_pf_d = torch.cat(pf_all) if pf_all else torch.zeros(0)
        info = {
            "loss": tot_loss,
            "logZ": float(self.model.logZ(cond).detach().mean()),
            "log_pf": float(log_pf_d.mean()) if len(log_pf_d) else 0.0,
            "log_pb": float(np.mean([t.log_pb for t in trajs])),
            "log_r": float(np.mean([t.log_reward for t in trajs])),
            "completion_rate": float(np.mean([t.completed for t in trajs])),
            "mean_steps": float(np.mean([t.n_steps for t in trajs])),
        }
        done = [t for t in trajs if t.completed]
        if done:
            info["log_r_completed"] = float(np.mean([t.log_reward for t in done]))
        return tot_loss, info
