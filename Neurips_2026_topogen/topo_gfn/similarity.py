"""Embedding-similarity regularizer for fine-tuning.

The flow (only ever applied to COMPLETED trajectories; failed rollouts keep
FAIL_LOGR untouched):

  1. run the generated graph through the encoder -> node embeddings (N, d).
  2. run the reference graph (the dataset realisation of the PD) through the
     SAME encoder under torch.no_grad() -- it anchors, it never sends
     gradients back.
  3. optionally CENTRE both by the running mean embedding of that PD's
     equivalence class (see below), then dot the two matrices (rows
     L2-normalised, so entries are cosines and lambda has a stable scale) and
     reduce to a scalar:
       match : maximum-weight bipartite assignment over the (N, N) dot matrix,
               mean of the matched entries.  Node-level, permutation-safe --
               the unlabeled setting gives no canonical row correspondence.
       pool  : dot of the two mean-pooled graph embeddings.
       ged   : NANL's GED scoring layer.  A Sinkhorn-normalised SOFT transport
               plan P replaces the hard assignment, and the score is the
               negative L1 feature alignment -mean_i ||h_i - (P h_ref)_i||_1.
               Fully differentiable (no detached assignment).
  4. add it to the reward -- either placement (see TrajectoryBalance):
       reward : log R += lambda * sim
       loss   : TB loss -= lambda * sim, gradients through the generated side.

Centering
---------
At a terminal state every PD-compliant graph has a bit-identical feature
matrix, so their embeddings share a large common component -- measured at 95.5%
of the embedding norm, leaving the structural signal at 4.5%.  Raw cosine is
dominated by that shared part and pins at ~0.999 for every compliant graph,
which makes the regulariser a constant and hence a no-op.  Subtracting a
running mean of the class restores the full range (measured: node-wise cosine
0.994 -> -0.089 on the same embeddings).

The mean needs a POPULATION -- centering two vectors by their own midpoint just
makes them antipodal -- so it is accumulated per PD across training: a running
mean for the first ``center_warmup`` completed graphs of that PD, an EMA after.
Until the warmup count is reached the uncentered score is used.

``source`` picks the reference/reward encoder:
  live   : the training encoder itself (the flow above).  Reward-mode sims are
           then nonstationary -- they drift as the encoder trains.
  frozen : a frozen copy of the pretrained encoder (stationary reward,
           collapse-proof anchor).  Kept as an ablation.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from scipy.optimize import linear_sum_assignment

from topo_gfn.actions import PDSchedule, State
from topo_gfn.policy import batch_states


def terminal_state(sched: PDSchedule, adj: np.ndarray) -> State:
    """Wrap a finished graph as the terminal state of its schedule.

    Nodes are reordered by degree so deg[i] == sched.node_time[i] (node_time
    is the ascending degree list).  Generated graphs already satisfy this, so
    the stable argsort is the identity for them; the reference realisation
    arrives in the dataset's arbitrary node order, and without the reorder the
    wrapped state would hand the encoder inconsistent features (nonzero
    capacities on a finished graph).
    """
    n = sched.num_nodes
    a = adj[:n, :n].astype(bool)
    perm = np.argsort(a.sum(1), kind="stable")
    a = a[np.ix_(perm, perm)]
    return State(sched=sched, k=sched.n, adj=a,
                 deaths_rem=np.zeros_like(sched.deaths),
                 cycles_rem=np.zeros_like(sched.cycles))


def stage_states(sched: PDSchedule, adj: np.ndarray) -> list:
    """The graph's filtration history: (k, State) at each informative stage.

    At stage k the partial graph is the nodes with node_time <= k and the edges
    whose later endpoint is among them.  Two PD-compliant graphs have the same
    node set AND the same edge count at every k -- that is what the deaths and
    cycles quotas fix -- so their stage-k states are directly comparable.

    This is where the discriminative signal lives.  A completed graph has
    degree == node_time and capacity == 0 by definition, so its terminal
    features are a function of the PD alone and are identical across the whole
    equivalence class (measured: max abs difference 0.0).  Mid-filtration the
    partial degrees, capacities and components genuinely differ (measured: 1.0
    at the middle stage of a comm20 target, with node cosine 0.919 vs 0.994 at
    the terminal state).

    Stages with no edges, and the final stage (the complete graph), are dropped:
    both are identical across the class by construction.
    """
    n = sched.num_nodes
    nt = sched.node_time
    a = adj[:n, :n].astype(bool)
    full = int(a.sum() // 2)
    out = []
    for k in range(int(nt.min()), int(nt.max()) + 1):
        alive = nt <= k
        keep = np.outer(alive, alive) & a
        m = int(keep.sum() // 2)
        if m == 0 or m == full:
            continue
        out.append((k, State(sched=sched, k=int(k), adj=keep,
                             deaths_rem=np.zeros_like(sched.deaths),
                             cycles_rem=np.zeros_like(sched.cycles))))
    return out


def cycle_spectrum_at(adj: np.ndarray, lengths=(3, 4, 5)) -> np.ndarray:
    """Exact cycle counts at the requested lengths (subset of 3, 4, 5)."""
    full = cycle_spectrum(adj, 5)
    return np.array([full[k - 3] for k in lengths], dtype=np.float64)


def cycle_spectrum(adj: np.ndarray, kmax: int = 5) -> np.ndarray:
    """EXACT counts of cycles of length 3, 4, ..., kmax.

    The TOTAL cycle count is not usable here: beta_1 = |E| - n + beta_0 is
    fixed by the PD, so every compliant graph has exactly the same number of
    independent cycles.  What differs is how those cycles are distributed over
    LENGTHS -- triangles vs 4-cycles vs longer rings -- which is the
    chemically and biologically meaningful part (ring size in molecules, loop
    size in ENZYMES binding cavities).

    Closed-walk traces overcount, so the standard corrections are applied:
        C3 = tr(A^3) / 6
        C4 = [tr(A^4) - 2m - 2*sum_i d_i(d_i - 1)] / 8
        C5 = [tr(A^5) - 5*tr(A^3) - 5*sum_i (d_i - 2) * (A^3)_ii] / 10
    Verified against C_4 and C_5 (each gives exactly 1).  O(n^3) via three
    matmuls, so it stays cheap even on the dense SBM graphs where enumerating
    cycles outright would blow up.
    """
    A = adj.astype(np.float64)
    d = A.sum(1)
    m = d.sum() / 2.0
    A2 = A @ A
    A3 = A2 @ A
    t3, t4, t5 = np.trace(A3), np.trace(A2 @ A2), np.trace(A3 @ A2)
    out = [t3 / 6.0]
    if kmax >= 4:
        out.append((t4 - 2 * m - 2 * (d * (d - 1)).sum()) / 8.0)
    if kmax >= 5:
        out.append((t5 - 5 * t3 - 5 * ((d - 2) * np.diag(A3)).sum()) / 10.0)
    return np.array(out[:max(1, kmax - 2)])


def cycle_score(adj_gen: np.ndarray, adj_ref: np.ndarray,
                lengths=(3, 4, 5), mode: str = "match",
                clip: float = 2.0) -> float:
    """Cycle-length penalty against the reference; 0 is best, negative is worse.

        match  : -mean_k |C_k(gen) - C_k(ref)| / max(1, C_k(ref))
                 two-sided -- too FEW k-cycles is penalised as much as too many.
        excess : -mean_k max(0, C_k(gen) - C_k(ref)) / max(1, C_k(ref))
                 one-sided -- only having MORE k-cycles than the reference
                 costs anything.  With lengths=(3,) this is a pure triangle
                 penalty, which is the right shape when the policy is known to
                 over-produce triangles (measured on comm20: 13-16 generated
                 vs 11 in the reference).

    Relative rather than absolute error, so one length cannot dominate purely
    by being more numerous; max(1, .) guards references with no k-cycles.

    The per-length error is CLIPPED at ``clip``.  Without it a reference with
    zero k-cycles turns the ratio into a raw count, so a graph with 60 spurious
    triangles would score -60 -- past the -50 dead-end penalty, which would
    teach the policy that failing outright beats completing badly.  Sparse
    datasets (ENZYMES averages ~2 edges per node) make zero-triangle
    references common, so the clip is what keeps the reward bounded there.  At
    clip=2 it never binds on comm20, where the observed worst case is 0.9.
    """
    g = cycle_spectrum_at(adj_gen, lengths)
    r = cycle_spectrum_at(adj_ref, lengths)
    d = (g - r) if mode == "excess" else np.abs(g - r)
    if mode == "excess":
        d = np.maximum(0.0, d)
    rel = np.minimum(d / np.maximum(1.0, np.abs(r)), clip)
    return -float(np.mean(rel))


SPATIAL_TERMS = ("diameter", "radius", "avg_spl")


def spatial_descriptors(adj: np.ndarray, terms=("diameter",)) -> np.ndarray:
    """Distance-geometry descriptors: how SPREAD OUT the graph is.

    None of these is pinned by the PD, unlike the component count: beta_0 is
    the number of infinite-death bars in H0, so every compliant graph has the
    identical number of components at every filtration stage (measured std
    0.0000 across 251 comm20 and 172 enzymes compliant graphs -- a constant,
    and therefore useless as a reward).  Diameter, radius and average shortest
    path length all vary within a PD class (std 0.45 / 0.28 / 0.06 on comm20)
    while sitting far from the reference (1.39 / 0.75 / 0.48 on comm20, and
    7.69 / 3.32 / 2.39 on enzymes), so the error is a systematic bias rather
    than noise.

    Computed on the largest connected component -- diameter is infinite for a
    disconnected graph, and many enzymes targets are disconnected.
    """
    import networkx as nx
    n = adj.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(u), int(v))
                     for u, v in zip(*np.nonzero(np.triu(adj))))
    comps = list(nx.connected_components(G))
    big = G.subgraph(max(comps, key=len)) if comps else G
    small = big.number_of_nodes() <= 1
    out = []
    for t in terms:
        if t == "diameter":
            out.append(0.0 if small else float(nx.diameter(big)))
        elif t == "radius":
            out.append(0.0 if small else float(nx.radius(big)))
        elif t == "avg_spl":
            out.append(0.0 if small
                       else float(nx.average_shortest_path_length(big)))
        else:
            raise ValueError(f"unknown spatial term {t!r}")
    return np.array(out, dtype=np.float64)


def spatial_score(adj_gen: np.ndarray, adj_ref: np.ndarray,
                  terms=("diameter",), clip: float = 2.0) -> float:
    """Negative mean clipped relative error of the spread descriptors."""
    g = spatial_descriptors(adj_gen, terms)
    r = spatial_descriptors(adj_ref, terms)
    rel = np.minimum(np.abs(g - r) / np.maximum(1.0, np.abs(r)), clip)
    return -float(np.mean(rel))


def _unit(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return h / h.norm(dim=-1, keepdim=True).clamp(min=eps)


def sinkhorn(log_alpha: torch.Tensor, temp: float = 0.1,
             iters: int = 20) -> torch.Tensor:
    """Soft doubly-stochastic transport plan (Sinkhorn normalisation).

    Same construction as NANL.get_transport_plan in the reference GED model:
    alternate row and column log-normalisation of the scaled score matrix.
    ``temp`` -> 0 sharpens toward a hard permutation, so this is the
    differentiable relaxation of the Hungarian assignment used by ``match``.
    """
    p = log_alpha / temp
    for _ in range(iters):
        p = p - torch.logsumexp(p, dim=-1, keepdim=True)
        p = p - torch.logsumexp(p, dim=-2, keepdim=True)
    return torch.exp(p)


class EmbedSim:
    """Similarity between a generated graph and its reference realisation."""

    def __init__(self, encoder, agg: str = "match", source: str = "live",
                 sinkhorn_temp: float = 0.1, sinkhorn_iters: int = 20,
                 centered: bool = False, center_warmup: int = 8,
                 center_decay: float = 0.99, stages: bool = False,
                 cycle_lengths=(3, 4, 5), cycle_mode: str = "match",
                 cycle_clip: float = 2.0, spatial_terms=("diameter",)):
        assert agg in ("match", "pool", "ged", "cycles", "spatial")
        assert source in ("live", "frozen")
        assert cycle_mode in ("match", "excess")
        self.agg = agg
        self.stages = stages
        self.cycle_lengths = tuple(cycle_lengths)
        self.cycle_mode = cycle_mode
        self.cycle_clip = cycle_clip
        self.spatial_terms = tuple(spatial_terms)
        self.source = source
        self.sinkhorn_temp = sinkhorn_temp
        self.sinkhorn_iters = sinkhorn_iters
        self.centered = centered
        self.center_warmup = center_warmup
        self.center_decay = center_decay
        if source == "frozen":
            self.anchor = copy.deepcopy(encoder).eval()
            for p in self.anchor.parameters():
                p.requires_grad_(False)
        else:
            self.anchor = encoder            # same module, called under no_grad
        self._ref_cache: dict[bytes, tuple] = {}   # frozen source only
        self._mu: dict[bytes, list] = {}           # per-PD running mean + count

    # -- embedding ----------------------------------------------------------

    @torch.no_grad()
    def _embed_nograd(self, sched: PDSchedule, adj: np.ndarray):
        x, a, nm = batch_states([terminal_state(sched, adj)])
        h, h_g = self.anchor(x, a, nm)
        return h[0], h_g[0]

    def ref_embedding(self, sched: PDSchedule, adj_ref: np.ndarray):
        """Raw reference node/graph embeddings, gradient-free.

        Cacheable only for a frozen anchor; the live encoder moves every
        gradient step, so its reference embeddings are recomputed.
        """
        if self.source == "frozen":
            key = adj_ref.astype(np.uint8).tobytes()
            if key not in self._ref_cache:
                self._ref_cache[key] = self._embed_nograd(sched, adj_ref)
            return self._ref_cache[key]
        return self._embed_nograd(sched, adj_ref)

    # -- centering ----------------------------------------------------------

    @staticmethod
    def _pd_key(sched: PDSchedule, k: int = -1) -> bytes:
        return (sched.node_time.tobytes() + sched.deaths.tobytes()
                + sched.cycles.tobytes() + bytes([k & 0xFF]))

    def _centre(self, sched: PDSchedule, h_gen: torch.Tensor,
                h_ref: torch.Tensor, k: int = -1):
        """Subtract the class mean; returns None while still warming up.

        The mean is kept per (PD, stage): each stage has its own shared
        component, so one pooled mean would not remove either cleanly.
        """
        key = self._pd_key(sched, k)
        st = self._mu.get(key)
        if st is None:
            st = [torch.zeros_like(h_ref), 0]
            self._mu[key] = st
        mu, cnt = st
        g = h_gen.detach()
        # running mean while warming up, EMA afterwards
        st[0] = (mu + (g - mu) / (cnt + 1) if cnt < self.center_warmup
                 else self.center_decay * mu + (1 - self.center_decay) * g)
        st[1] = cnt + 1
        if st[1] < self.center_warmup:
            return None
        return h_gen - st[0], h_ref - st[0]

    # -- aggregation --------------------------------------------------------

    def _agg_sim(self, h_gen, hg_gen, h_ref, hg_ref) -> torch.Tensor:
        """Scalar similarity; differentiable in the *_gen inputs."""
        if self.agg == "pool":
            return (_unit(hg_gen.unsqueeze(0))[0]
                    * _unit(hg_ref.unsqueeze(0))[0]).sum()
        if self.agg == "ged":
            # NANL-style: soft transport plan, then negative L1 feature
            # alignment.  Unit rows keep the residual bounded so lambda has a
            # stable scale, and the whole path stays differentiable.
            Hg, Hr = _unit(h_gen), _unit(h_ref)
            P = sinkhorn(Hg @ Hr.T, temp=self.sinkhorn_temp,
                         iters=self.sinkhorn_iters)
            return -(Hg - P @ Hr).abs().sum(-1).mean()
        C = _unit(h_gen) @ _unit(h_ref).T                  # (N, N) cosines
        r, c = linear_sum_assignment(-C.detach().numpy())  # max-weight match
        return C[r, c].mean()

    def _score(self, sched, h_gen, hg_gen, h_ref, hg_ref,
               k: int = -1) -> torch.Tensor:
        if self.centered:
            cen = self._centre(sched, h_gen, h_ref, k)
            if cen is not None:
                h_gen, h_ref = cen
                hg_gen, hg_ref = h_gen.mean(0), h_ref.mean(0)
        return self._agg_sim(h_gen, hg_gen, h_ref, hg_ref)

    def _stage_pairs(self, sched, adj_gen, adj_ref):
        """(k, gen_state, ref_state) over the stages both graphs share."""
        sg = stage_states(sched, adj_gen)
        sr = dict(stage_states(sched, adj_ref))
        return [(k, g, sr[k]) for k, g in sg if k in sr]

    # -- entry points -------------------------------------------------------

    @torch.no_grad()
    def sim(self, sched: PDSchedule, adj_gen: np.ndarray,
            adj_ref: np.ndarray) -> float:
        """Similarity for reward shaping (no gradients anywhere)."""
        if self.agg == "cycles":
            return cycle_score(adj_gen, adj_ref, self.cycle_lengths,
                               self.cycle_mode, self.cycle_clip)
        if self.agg == "spatial":
            return spatial_score(adj_gen, adj_ref, self.spatial_terms,
                                 self.cycle_clip)
        if self.stages:
            pairs = self._stage_pairs(sched, adj_gen, adj_ref)
            if not pairs:
                return 0.0
            n, vals = sched.num_nodes, []
            for k, g, r in pairs:
                x, a, nm = batch_states([g, r])
                h, hg = self.anchor(x, a, nm)
                vals.append(float(self._score(sched, h[0, :n], hg[0],
                                              h[1, :n], hg[1], k)))
            return float(np.mean(vals))
        h, h_g = self._embed_nograd(sched, adj_gen)
        hr, hgr = self.ref_embedding(sched, adj_ref)
        return float(self._score(sched, h, h_g, hr, hgr))

    def live_sim(self, encoder, trajs) -> torch.Tensor:
        """Mean similarity with gradients through the generated side only.

        The generated graphs are embedded by the training encoder WITH grad;
        the reference side is embedded under no_grad, so no gradient flows
        back from the reference direction.
        """
        if self.agg in ("cycles", "spatial"):
            raise ValueError(f"--sim-agg {self.agg} is not differentiable; use "
                             "--sim-place reward (a GFlowNet reward need not "
                             "be differentiable)")
        if self.stages:
            sims = []
            for t in trajs:
                sched = t.env.sched
                n = sched.num_nodes
                pairs = self._stage_pairs(
                    sched, np.asarray(t.terminal.adj), t.env.ref_adj)
                if not pairs:
                    continue
                vals = []
                for k, g, r in pairs:
                    x, a, nm = batch_states([g])
                    h, hg = encoder(x, a, nm)                  # with grad
                    with torch.no_grad():
                        xr, ar, nmr = batch_states([r])
                        hr, hgr = self.anchor(xr, ar, nmr)     # anchor
                    vals.append(self._score(sched, h[0, :n], hg[0],
                                            hr[0, :n], hgr[0], k))
                sims.append(torch.stack(vals).mean())
            return (torch.stack(sims).mean() if sims
                    else torch.zeros((), requires_grad=True))
        states = [terminal_state(t.env.sched,
                                 np.asarray(t.terminal.adj)) for t in trajs]
        x, a, nm = batch_states(states)
        h, h_g = encoder(x, a, nm)
        sims = []
        for i, t in enumerate(trajs):
            n = t.env.sched.num_nodes
            hr, hgr = self.ref_embedding(t.env.sched, t.env.ref_adj)
            sims.append(self._score(t.env.sched, h[i, :n], h_g[i], hr, hgr))
        return torch.stack(sims).mean()
