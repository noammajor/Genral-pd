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

from topo_gfn.actions import GraphActionType, PDSchedule, State
from topo_gfn.env import TopoEnv
from topo_gfn.policy import (TopoActionCategorical, batch_bck_masks,
                             batch_masks, batch_states)
from topo_gfn.actions import DEFAULT_BCK_ACTION_TYPE_ORDER, GraphAction

FAIL_LOGR = -50.0


def PDSchedule_to_pd(sched: PDSchedule):
    """Recover the target diagrams (pd0, pd1) the schedule was built from."""
    pd0 = []
    counted = np.zeros(sched.n + 1, dtype=np.int64)
    for d in range(sched.deaths.shape[0]):
        for b in range(sched.deaths.shape[1]):
            for _ in range(int(sched.deaths[d, b])):
                pd0.append((b, d))
                counted[b] += 1
    # births with no recorded death survive to infinity
    births, cnt = np.unique(sched.node_time, return_counts=True)
    for b, c in zip(births.tolist(), cnt.tolist()):
        for _ in range(c - int(counted[b])):
            pd0.append((b, float("inf")))
    pd1 = [(k, float("inf"))
           for k in range(len(sched.cycles))
           for _ in range(int(sched.cycles[k]))]
    return pd0, pd1


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
    sim: Optional[float] = None      # embedding similarity to the reference
    violations: int = 0              # soft mode: quota/capacity overspends


class TopoSampler:
    """Rolls out a batch of trajectories, one per target environment."""

    def __init__(self, model, max_steps_slack: int = 4, soft: bool = False):
        self.model = model
        self.max_steps_slack = max_steps_slack
        # soft=True widens the support to every well-formed edge; the cost of
        # a PD violation is charged in the reward, not here.
        self.soft = soft

    @torch.no_grad()
    def sample(self, envs: List[TopoEnv], cond: Optional[torch.Tensor] = None,
               keep_states: bool = True) -> List[Trajectory]:
        """Roll out under no_grad, recording (state, action) for the loss."""
        B = len(envs)
        states = [e.new(soft=self.soft) for e in envs]
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
                if (not self.soft) and envs[i].is_dead_end(states[i]):
                    active[i] = False
                else:
                    still.append(i)
            if not still:
                break

            batch = [states[i] for i in still]
            x, adj, nm = batch_states(batch)
            tm, mm, cm = batch_masks(batch, soft=self.soft)
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
                nxt = env.step(states[i], a, soft=self.soft)
                # uniform P_B over the exact parent set of the CHILD
                npar = env.count_backward_transitions(nxt)
                log_pb[i] += -np.log(max(1, npar))
                states[i] = nxt
                steps[i] += 1

        return [
            Trajectory(env=envs[i], log_pb=log_pb[i], terminal=states[i],
                       completed=completed[i], n_steps=steps[i],
                       states=seen[i], actions=acts_taken[i],
                       violations=int(states[i].violations))
            for i in range(B)
        ]

    def _device(self):
        return next(self.model.parameters()).device


class TrajectoryBalance:
    """The TB objective.

    Fine-tuning regulariser (``sim_fn`` = an ``EmbedSim``): pull completed
    graphs toward their reference realisation in embedding space.  Two
    placements, both touching COMPLETED trajectories only:
      reward : log R += sim_lambda * sim  (multiplicative reward shaping,
               frozen pretrained encoder on both sides -- stationary reward)
      loss   : TB loss -= sim_lambda * sim with gradients through the live
               encoder (reference side stays frozen as an anchor)
    """

    def __init__(self, model, scorer, beta: float = 1.0,
                 fail_logr: float = FAIL_LOGR,
                 sim_fn=None, sim_lambda: float = 0.0,
                 sim_place: str = "reward", sim_center: float = 0.0,
                 score_floor: Optional[float] = None,
                 pd_penalty: float = 0.0, soft: bool = False,
                 violation_penalty: float = 0.0, learn_pb: bool = False):
        assert sim_place in ("reward", "loss")
        self.model = model
        self.scorer = scorer
        self.beta = beta
        self.fail_logr = fail_logr
        self.sim_fn = sim_fn
        self.sim_lambda = sim_lambda
        self.sim_place = sim_place
        # Centering keeps the completed-vs-failed reward gap unchanged: with
        # sim ~ 0.9 for every compliant graph, an uncentered lambda*sim bonus
        # would mostly just inflate all completed rewards.  lambda*(sim-c)
        # only reshapes preference WITHIN the compliant set.
        self.sim_center = sim_center
        # Floor on the completed-graph reward.  s(H) = -mean(z^2) is unbounded
        # below, and on datasets whose descriptor spread is tight it can fall
        # far past FAIL_LOGR -- measured -599 on planar against a -50 failure
        # penalty, which trains the policy to dead-end on purpose.  Clamping
        # keeps completion strictly better than failure; it flattens shaping
        # among graphs at the floor, which is the right trade (learn to
        # complete first, shape second).
        self.score_floor = score_floor
        # Soft-constraint mode: a trajectory can finish with a graph whose PD
        # differs from the target, so compliance stops being all-or-nothing.
        # log R loses pd_penalty * W1(PD(x), PD_target) -- zero for an exactly
        # compliant graph, growing smoothly with the deviation.  This is what
        # replaces the hard mask as the thing that enforces the diagram.
        self.pd_penalty = pd_penalty
        # Soft constraints: the support is every well-formed edge and each PD
        # violation costs `violation_penalty` nats of reward, so the target is
        # P(x) proportional to exp(s(x) - T * violations(x)).  `soft` MUST match
        # the sampler: log P_F is recomputed here, and scoring a violating
        # action against the strict mask would give -inf.
        self.soft = soft
        self.violation_penalty = violation_penalty
        # Learned backward policy.  Default False keeps log P_B = -log|Parents|,
        # computed by counting during the rollout.  When set, P_B comes from the
        # model's backward heads and carries gradients, so the pair
        # (P_F, P_B) is fitted jointly as SynFlowNet does.
        self.learn_pb = learn_pb

    def _sim_of(self, traj: Trajectory) -> Optional[float]:
        ref = getattr(traj.env, "ref_adj", None)
        if self.sim_fn is None or ref is None:
            return None
        n = traj.env.sched.num_nodes
        adj = np.asarray(traj.terminal.adj[:n, :n], dtype=np.float64)
        return self.sim_fn.sim(traj.env.sched, adj, ref)

    def pd_deviation(self, traj: Trajectory) -> float:
        """1-Wasserstein distance from the realised PD to the target."""
        from topo_gen.filtrations import degree_filtration
        from topo_gen.persistence import persistence_diagrams
        from topo_gen.pdm_utils import _wasserstein1_pd
        import networkx as nx

        sched = traj.env.sched
        n = sched.num_nodes
        adj = np.asarray(traj.terminal.adj[:n, :n])
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from((int(u), int(v))
                         for u, v in zip(*np.nonzero(np.triu(adj))))
        nt, et, _ = degree_filtration(G)
        pd0, pd1 = persistence_diagrams(nt, et)
        tgt = PDSchedule_to_pd(sched)
        return (_wasserstein1_pd(pd0, tgt[0])
                + _wasserstein1_pd(pd1, tgt[1]))

    def log_reward(self, traj: Trajectory) -> float:
        if not traj.completed:
            return self.fail_logr
        n = traj.env.sched.num_nodes
        adj = traj.terminal.adj[:n, :n]
        lr = float(self.beta * self.scorer.score(adj))
        if self.score_floor is not None:
            lr = max(lr, self.score_floor)
        if self.violation_penalty:
            lr -= self.violation_penalty * traj.violations
        if self.pd_penalty:
            lr -= self.pd_penalty * self.pd_deviation(traj)
        if self.sim_lambda and self.sim_place == "reward":
            s = self._sim_of(traj)
            if s is not None:
                traj.sim = s
                lr += self.sim_lambda * (s - self.sim_center)
        return lr

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
            tm, mm, cm = batch_masks(bs, soft=self.soft)
            tl, ml, cl = self.model(x.to(dev), adj.to(dev), nm.to(dev))
            cat = TopoActionCategorical(tl, ml, cl, tm, mm, cm)
            out = out.index_add(0, idx[i:i + chunk], cat.log_prob(ba))
        return out

    def _log_pb(self, trajs: List[Trajectory], chunk: int = 256) -> torch.Tensor:
        """Recompute sum_t log P_B(s_t | s_{t+1}) from the backward heads.

        For a forward transition s_t --a--> s_{t+1}, the backward policy is
        scored AT THE CHILD on the action that undoes a, which ``env.reverse``
        supplies.  Stop moves no edge and has no backward counterpart, so it is
        skipped.  Structure matches _log_pf: flatten every (child, bck-action)
        pair, re-run the model in chunks, scatter back with index_add.
        """
        dev = next(self.model.parameters()).device
        flat_s, flat_a, owner = [], [], []
        for ti, t in enumerate(trajs):
            n_seen = len(t.states)
            for j, (s, aidx) in enumerate(zip(t.states, t.actions)):
                ga = t.env.ctx.ActionIndex_to_GraphAction(aidx)
                if ga.action is GraphActionType.Stop:
                    continue
                child = t.states[j + 1] if j + 1 < n_seen else t.terminal
                bck = t.env.reverse(s, ga)          # s is the PARENT, as required
                flat_s.append(child)
                flat_a.append(t.env.ctx.GraphAction_to_ActionIndex(bck, fwd=False))
                owner.append(ti)

        out = torch.zeros(len(trajs), device=dev)
        if not flat_s:
            return out
        idx = torch.tensor(owner, dtype=torch.long, device=dev)
        for i in range(0, len(flat_s), chunk):
            bs, ba = flat_s[i:i + chunk], flat_a[i:i + chunk]
            x, adj, nm = batch_states(bs)
            tm, mm, cm = batch_bck_masks(bs)
            tl, ml, cl = self.model.forward_bck(x.to(dev), adj.to(dev), nm.to(dev))
            cat = TopoActionCategorical(tl, ml, cl, tm, mm, cm,
                                        action_type_order=DEFAULT_BCK_ACTION_TYPE_ORDER)
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
        pf_all, resid_all, sim_vals, pb_all = [], [], [], []
        for i in range(0, n, micro_bs):
            mb = trajs[i:i + micro_bs]
            logZ = self.model.logZ(cond).squeeze(-1)
            if logZ.numel() == 1:
                logZ = logZ.expand(len(mb))
            log_pf = self._log_pf(mb, chunk=chunk)
            log_r = torch.tensor([t.log_reward for t in mb], dtype=torch.float32, device=dev)
            log_pb = (self._log_pb(mb, chunk=chunk) if self.learn_pb
                      else torch.tensor([t.log_pb for t in mb],
                                        dtype=torch.float32, device=dev))
            if self.learn_pb:
                pb_all.append(log_pb.detach())

            resid = logZ + log_pf - log_r - log_pb
            loss_mb = (resid ** 2).mean() * (len(mb) / n)
            if self.sim_lambda and self.sim_place == "loss":
                done_mb = [t for t in mb if t.completed
                           and getattr(t.env, "ref_adj", None) is not None]
                if done_mb:
                    aux = self.sim_fn.live_sim(self.model.encoder, done_mb)
                    sim_vals.append((float(aux.detach()), len(done_mb)))
                    loss_mb = loss_mb - self.sim_lambda * aux * (len(done_mb) / n)
            loss_mb.backward()

            tot_loss += float(loss_mb.detach())
            pf_all.append(log_pf.detach())
            resid_all.append(resid.detach())

        log_pf_d = torch.cat(pf_all) if pf_all else torch.zeros(0)
        info = {
            "loss": tot_loss,
            "logZ": float(self.model.logZ(cond).detach().mean()),
            "log_pf": float(log_pf_d.mean()) if len(log_pf_d) else 0.0,
            "log_pb": (float(torch.cat(pb_all).mean()) if pb_all
                       else float(np.mean([t.log_pb for t in trajs]))),
            "log_r": float(np.mean([t.log_reward for t in trajs])),
            "completion_rate": float(np.mean([t.completed for t in trajs])),
            "mean_steps": float(np.mean([t.n_steps for t in trajs])),
        }
        info["violations"] = float(np.mean([t.violations for t in trajs]))
        if self.pd_penalty:
            info["pd_dev"] = float(np.mean(
                [self.pd_deviation(t) for t in trajs if t.completed] or [0.0]))
        done = [t for t in trajs if t.completed]
        if done:
            info["log_r_completed"] = float(np.mean([t.log_reward for t in done]))
        # mean embedding similarity over completed trajectories, whichever
        # placement produced it
        rw_sims = [t.sim for t in done if t.sim is not None]
        if rw_sims:
            info["sim"] = float(np.mean(rw_sims))
        elif sim_vals:
            tot = sum(c for _, c in sim_vals)
            info["sim"] = float(sum(s * c for s, c in sim_vals) / tot)
        return tot_loss, info
