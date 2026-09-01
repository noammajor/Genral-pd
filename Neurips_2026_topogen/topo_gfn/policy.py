"""Policy network and hierarchical action categorical.

Mirrors SynFlowNet: an ``mlp`` helper with the same signature, a model that
carries one head per action type, and a ``TopoActionCategorical`` that samples
in two levels with two SEPARATELY NORMALISED partition functions, so

    log P(a) = log P(type) + log P(pair | type)

exactly as ``ActionCategorical.log_prob`` sums ``bireact_log_probs[i, row] +
addreactant_log_probs[i, col]`` in synthesis_building_env.py.

Level 1 (primary)   MLP_type   -> 3 logits over {Stop, Merge, Cycle},
                                  masked by actions.type_mask.
Level 2 (secondary) MLP_merge  -> (N,N) ASYMMETRIC pair scores, masked by
                                  actions.merge_mask (u = dying side).
                    MLP_cycle  -> (N,N) SYMMETRIC pair scores, masked by
                                  actions.cycle_mask.

Pair scores are low-rank bilinear rather than an MLP over every concatenated
pair: an MLP would be O(N^2) forward passes, the bilinear form is one matmul.
Merge uses two distinct projections (asymmetric); Cycle uses one projection
against itself (symmetric by construction).

The encoder is dense message passing over the (N,N) adjacency rather than a
torch_geometric GNN: our states already ARE dense boolean matrices, N is at
most ~175 on these benchmarks, and it keeps the module dependency-free.

NOTE: this module needs torch, which is not installed on the dev laptop.  It is
exercised on the cluster, where the venv has torch 2.8.0.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from topo_gfn.actions import (
    DEFAULT_ACTION_TYPE_ORDER,
    ActionIndex,
    GraphActionType,
    State,
    cycle_mask,
    merge_mask,
    type_mask,
)

# number of per-node input features produced by state_features()
NUM_NODE_FEATURES = 9


def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    """Fully-connected net with no activation after the last layer.

    Identical to synflownet/models/graph_transformer.py:17 -- with n_layer=0
    this is exactly nn.Linear(n_in, n_out).
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1])


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------

def state_features(s: State) -> np.ndarray:
    """(N, NUM_NODE_FEATURES) float32 node features for one state.

    Everything here is derivable from the state, but handing the network the
    quantities the masks are built from means it does not have to rediscover
    them.  Capacity is the important one: it is what the winning heuristic keys
    on, and what running out of causes NO_CYCLE_CANDIDATE dead ends.
    """
    N = s.sched.num_nodes
    nt = s.sched.node_time.astype(np.float32)
    deg = s.degree.astype(np.float32)
    cap = s.capacity.astype(np.float32)
    comp_id, cb = s.components()
    _, counts = np.unique(comp_id[comp_id >= 0], return_counts=True)
    size_of = dict(zip(np.unique(comp_id[comp_id >= 0]).tolist(), counts.tolist()))
    comp_size = np.array([size_of.get(int(c), 0) for c in comp_id], dtype=np.float32)

    scale = max(1.0, float(s.sched.n))
    feats = np.stack([
        nt / scale,                                  # intended degree (= birth)
        deg / scale,                                 # current degree
        cap / scale,                                 # remaining capacity  <-- key
        (cap > 0).astype(np.float32),                # has any capacity left
        s.exists.astype(np.float32),
        s.is_new.astype(np.float32),                 # born at the current k
        np.where(cb >= 0, cb, -1).astype(np.float32) / scale,   # component birth
        comp_size / max(1.0, N),                     # component size
        np.full(N, s.k / scale, dtype=np.float32),   # current timestep
    ], axis=1)
    return feats.astype(np.float32)


def batch_states(states: List[State]):
    """Pad a list of states to a common N.

    Returns (x, adj, node_mask) with shapes (B,Nmax,F), (B,Nmax,Nmax), (B,Nmax).
    Padding is needed once we condition on many target PDs, since N varies per
    target (enzymes runs n=10..124).
    """
    B = len(states)
    Nmax = max(s.sched.num_nodes for s in states)
    x = np.zeros((B, Nmax, NUM_NODE_FEATURES), dtype=np.float32)
    adj = np.zeros((B, Nmax, Nmax), dtype=np.float32)
    nm = np.zeros((B, Nmax), dtype=bool)
    for i, s in enumerate(states):
        n = s.sched.num_nodes
        x[i, :n] = state_features(s)
        adj[i, :n, :n] = s.adj.astype(np.float32)
        nm[i, :n] = True
    return (torch.from_numpy(x), torch.from_numpy(adj), torch.from_numpy(nm))


def batch_masks(states: List[State], action_type_order=None):
    """Padded (type, merge, cycle) masks for a list of states, as bool tensors."""
    order = action_type_order or DEFAULT_ACTION_TYPE_ORDER
    B = len(states)
    Nmax = max(s.sched.num_nodes for s in states)
    tm = np.zeros((B, len(order)), dtype=bool)
    mm = np.zeros((B, Nmax, Nmax), dtype=bool)
    cm = np.zeros((B, Nmax, Nmax), dtype=bool)
    for i, s in enumerate(states):
        n = s.sched.num_nodes
        tm[i] = type_mask(s, order)
        mm[i, :n, :n] = merge_mask(s)
        cm[i, :n, :n] = cycle_mask(s)
    return (torch.from_numpy(tm), torch.from_numpy(mm), torch.from_numpy(cm))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class DenseGNN(nn.Module):
    """Dense message passing: h <- act(h W_self + A_norm h W_neigh + c W_cond)."""

    def __init__(self, num_in: int, num_emb: int = 128, num_layers: int = 4,
                 num_cond: int = 0):
        super().__init__()
        self.inp = nn.Linear(num_in, num_emb)
        self.self_w = nn.ModuleList(nn.Linear(num_emb, num_emb) for _ in range(num_layers))
        self.neigh_w = nn.ModuleList(nn.Linear(num_emb, num_emb) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(num_emb) for _ in range(num_layers))
        self.cond_w = nn.ModuleList(
            nn.Linear(num_cond, num_emb) if num_cond else nn.Identity()
            for _ in range(num_layers)
        )
        self.num_cond = num_cond
        self.act = nn.LeakyReLU()

    def forward(self, x, adj, node_mask, cond=None):
        """x (B,N,F)  adj (B,N,N)  node_mask (B,N) -> (B,N,E) nodes, (B,E) graph."""
        m = node_mask.unsqueeze(-1).float()
        h = self.inp(x) * m
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        a = adj / deg                                  # row-normalised
        for sw, nw, ln, cw in zip(self.self_w, self.neigh_w, self.norms, self.cond_w):
            msg = sw(h) + nw(torch.bmm(a, h))
            if self.num_cond and cond is not None:
                msg = msg + cw(cond).unsqueeze(1)
            h = ln(self.act(msg) + h) * m
        denom = m.sum(1).clamp(min=1.0)
        h_g = (h * m).sum(1) / denom                   # masked mean pool
        return h, h_g


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TopoGFN(nn.Module):
    """Policy for the PD-compliant graph MDP.

    One head per action type, as GraphTransformerSynGFN does, plus a logZ head
    over the conditioning vector (SynFlowNet parameterises logZ as an MLP of the
    conditional, not a scalar -- graph_transformer.py:263).
    """

    def __init__(self, num_emb: int = 128, num_layers: int = 4,
                 num_mlp_layers: int = 1, rank: int = 32, num_cond: int = 0):
        super().__init__()
        self.encoder = DenseGNN(NUM_NODE_FEATURES, num_emb, num_layers, num_cond)
        self.num_cond = num_cond

        # primary head: one logit per forward action type
        self.mlp_type = mlp(num_emb, num_emb, len(DEFAULT_ACTION_TYPE_ORDER), num_mlp_layers)

        # secondary heads: low-rank bilinear pair scores
        self.merge_l = mlp(num_emb, num_emb, rank, num_mlp_layers)   # dying side
        self.merge_r = mlp(num_emb, num_emb, rank, num_mlp_layers)   # surviving side
        self.cycle_p = mlp(num_emb, num_emb, rank, num_mlp_layers)   # symmetric
        self.rank = rank

        self.mlp_logZ = mlp(max(1, num_cond), num_emb, 1, 2)

    def forward(self, x, adj, node_mask, cond=None):
        h, h_g = self.encoder(x, adj, node_mask, cond)
        type_logits = self.mlp_type(h_g)                             # (B, 3)
        ml, mr = self.merge_l(h), self.merge_r(h)                    # (B,N,r)
        merge_logits = torch.bmm(ml, mr.transpose(1, 2)) / self.rank ** 0.5
        cp = self.cycle_p(h)
        cycle_logits = torch.bmm(cp, cp.transpose(1, 2)) / self.rank ** 0.5
        return type_logits, merge_logits, cycle_logits

    def logZ(self, cond=None):
        if self.num_cond and cond is not None:
            return self.mlp_logZ(cond)
        dev = next(self.parameters()).device
        return self.mlp_logZ(torch.zeros(1, 1, device=dev))


# ---------------------------------------------------------------------------
# Hierarchical categorical
# ---------------------------------------------------------------------------

class TopoActionCategorical:
    """Two-level categorical over (action type, node pair).

    Follows ActionCategorical in synthesis_building_env.py: the primary and
    secondary levels get their OWN partition functions, and log_prob is the sum
    of the two.  Masked entries are -inf before either softmax, so an illegal
    action can never be sampled and never contributes to either logZ.
    """

    def __init__(self, type_logits, merge_logits, cycle_logits,
                 type_mask_t, merge_mask_t, cycle_mask_t,
                 action_type_order=None):
        self.order = action_type_order or DEFAULT_ACTION_TYPE_ORDER
        self.dev = type_logits.device
        NEG = -torch.inf
        self.type_logits = type_logits.masked_fill(~type_mask_t.to(self.dev), NEG)
        self.merge_logits = merge_logits.masked_fill(~merge_mask_t.to(self.dev), NEG)
        self.cycle_logits = cycle_logits.masked_fill(~cycle_mask_t.to(self.dev), NEG)
        self.i_merge = self.order.index(GraphActionType.Merge)
        self.i_cycle = self.order.index(GraphActionType.Cycle)
        self.i_stop = self.order.index(GraphActionType.Stop)

    # -- level 1 ------------------------------------------------------------

    def type_logprobs(self):
        return torch.log_softmax(self.type_logits, dim=-1)

    # -- level 2 ------------------------------------------------------------

    def _pair_logprobs(self, logits):
        B = logits.shape[0]
        flat = logits.reshape(B, -1)
        return torch.log_softmax(flat, dim=-1).reshape(logits.shape)

    def merge_logprobs(self):
        return self._pair_logprobs(self.merge_logits)

    def cycle_logprobs(self):
        return self._pair_logprobs(self.cycle_logits)

    # -- sampling -----------------------------------------------------------

    @staticmethod
    def _gumbel_like(t: torch.Tensor) -> torch.Tensor:
        """Standard Gumbel noise, -log(-log(U)).

        Parenthesised explicitly: writing ``-torch.log(-torch.log(u).clamp_min(e))``
        binds the method call tighter than the unary minus, which clamps the
        NEGATIVE inner log up to +e, negates it, and then takes the log of a
        negative number -> NaN.  That silently makes every logit NaN and argmax
        return index 0.
        """
        u = torch.rand_like(t).clamp_(min=1e-20, max=1.0 - 1e-7)
        return -torch.log(-torch.log(u))

    def sample(self) -> List[ActionIndex]:
        """Gumbel-argmax at each level, as SynFlowNet does."""
        tlp = self.type_logprobs()
        t_idx = (tlp + self._gumbel_like(tlp)).argmax(dim=-1)

        out = []
        mlp_ = self.merge_logprobs()
        clp = self.cycle_logprobs()
        N = self.merge_logits.shape[-1]
        for i, t in enumerate(t_idx.tolist()):
            if t == self.i_stop:
                out.append(ActionIndex(action_type=t, row_idx=0, col_idx=0))
                continue
            lp = (mlp_ if t == self.i_merge else clp)[i]
            flat = (lp + self._gumbel_like(lp)).reshape(-1).argmax()
            out.append(ActionIndex(action_type=t,
                                   row_idx=int(flat // N), col_idx=int(flat % N)))
        return out

    # -- log-probability ----------------------------------------------------

    def log_prob(self, actions: List[ActionIndex]) -> torch.Tensor:
        """log P(type) + log P(pair | type), one entry per batch element."""
        tlp = self.type_logprobs()
        mlp_ = self.merge_logprobs()
        clp = self.cycle_logprobs()
        out = []
        for i, a in enumerate(actions):
            lp = tlp[i, a.action_type]
            if a.action_type == self.i_merge:
                lp = lp + mlp_[i, a.row_idx, a.col_idx]
            elif a.action_type == self.i_cycle:
                lp = lp + clp[i, a.row_idx, a.col_idx]
            out.append(lp)
        return torch.stack(out)
