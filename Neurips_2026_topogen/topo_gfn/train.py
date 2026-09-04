"""Train the PD-compliant graph GFlowNet with trajectory balance.

    python -m topo_gfn.train --dataset comm20  --n-iterations 10000
    python -m topo_gfn.train --dataset enzymes --n-iterations 10000

Target PDs come from a benchmark split, so this is the CONDITIONAL setting
(mode 2): one policy across many targets, evaluated on held-out ones.
``--single-target`` pins one target for the mode-1 sanity check.

On "epochs": a GFlowNet generates its own training data, so the only thing that
is passed over is the set of target PDs.  One epoch = batch_size * iterations /
n_train_targets.  comm20 has 64 train targets and enzymes 376, so at batch 64
an iteration is one comm20 epoch and about a sixth of an enzymes epoch.  The
script logs both so the number is never ambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topo_gen.filtrations import degree_filtration
from topo_gen.persistence import persistence_diagrams
from topo_gfn.actions import PDSchedule
from topo_gfn.env import TopoEnv
from topo_gfn.gfn import FAIL_LOGR, TopoSampler, TrajectoryBalance
from topo_gfn.policy import TopoGFN, num_node_features, set_feature_mode
from topo_gfn.score import ConstantScorer, DescriptorScorer


def load_targets(dataset: str, split: str, data_root: str, max_nodes: int = None):
    """Benchmark split -> (PDSchedule, dense adjacency) per graph.

    ``dataset`` may be a comma-separated list ("comm20,enzymes"), in which case
    the splits are concatenated.  Handling it here rather than in main() means
    a mixed checkpoint records its spec in args["dataset"] and every
    downstream script (eval, iso_test, digress_metrics) reloads it unchanged.
    """
    if "," in dataset:
        out = []
        for one in dataset.split(","):
            out.extend(load_targets(one.strip(), split, data_root, max_nodes))
        return out
    from utils.dataset_utils import load_split, pyg_to_nx
    out = []
    for d in load_split(data_root, dataset, split):
        G = pyg_to_nx(d)
        n = G.number_of_nodes()
        if max_nodes and n > max_nodes:
            continue
        nt, et, _ = degree_filtration(G)
        pd0, pd1 = persistence_diagrams(nt, et)
        sched = PDSchedule.from_pd(pd0, pd1)
        nodes = sorted(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in G.edges():
            A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1.0
        out.append((sched, A))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="comm20")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n-epochs", type=int, default=None,
                    help="number of FULL PASSES over the training targets. Each "
                         "epoch shuffles all targets and visits every one exactly "
                         "once, in batches. Takes precedence over --n-iterations.")
    ap.add_argument("--n-iterations", type=int, default=10000,
                    help="total gradient steps; used only if --n-epochs is unset")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="trajectories per backward pass. Peak memory scales "
                         "with this, not --batch-size: log P_F for ONE trajectory "
                         "must be in the graph at once, but different "
                         "trajectories need not be. Lower this if OOM.")
    ap.add_argument("--recompute-chunk", type=int, default=256,
                    help="states per forward pass when recomputing log P_F")
    ap.add_argument("--size-buckets", action="store_true", default=True,
                    help="batch targets of similar node count together, so "
                         "padding to the batch max does not blow up the (N,N) "
                         "tensors (enzymes spans n=10..126)")
    ap.add_argument("--no-size-buckets", dest="size_buckets", action="store_false")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="inverse temperature on s(H); 0 = completion-only reward")
    ap.add_argument("--constant-scorer", action="store_true",
                    help="ablation: s(H)=0, reward is completion only")
    ap.add_argument("--single-target", type=int, default=None,
                    help="mode-1: pin one target index")
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--num-emb", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-mlp-layers", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-z", type=float, default=1e-3,
                    help="logZ gets its own higher LR, as SynFlowNet does")
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--log-every", type=int, default=0,
                    help="log every N gradient steps; 0 = per-epoch lines only")
    ap.add_argument("--ckpt-every", type=int, default=50,
                    help="checkpoint every N EPOCHS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--violation-penalty", type=float, default=None,
                    help="SOFT constraints. Instead of masking PD-violating "
                         "moves out, allow any well-formed edge and charge T "
                         "nats of reward per violation: "
                         "log R = beta*s(x) - T*violations. The target becomes "
                         "P(x) ~ exp(s(x) - T*v(x)), so a graph with v "
                         "violations is e^(-T*v) less likely. T=0 relaxes the "
                         "PD entirely (ablation); unset keeps the hard mask. "
                         "Well-formedness (node exists, no duplicate edge, no "
                         "self-loop) is always enforced.")
    ap.add_argument("--pd-penalty", type=float, default=0.0,
                    help="weight on W1(PD(x), PD_target) in the reward. Only "
                         "meaningful with --mask-temp, where compliance is no "
                         "longer guaranteed; 0 leaves deviation unpunished.")
    ap.add_argument("--learn-pb", action="store_true",
                    help="parameterise the BACKWARD policy instead of using a "
                         "uniform P_B over the exact parent set. Adds one head "
                         "per backward action type on the shared encoder, as "
                         "SynFlowNet's GraphTransformerSynGFN does under "
                         "do_bck, and fits (P_F, P_B) jointly through the TB "
                         "residual. Off by default.")
    ap.add_argument("--score-floor", type=float, default=None,
                    help="floor on the completed-graph reward beta*s(H). "
                         "s(H) = -mean(z^2) is unbounded below; on tight "
                         "datasets it can fall past the -50 dead-end penalty "
                         "(measured -599 on planar), which trains the policy "
                         "to fail on purpose. -40 keeps completion strictly "
                         "better than failing. Off by default.")
    ap.add_argument("--init-from", default=None,
                    help="path to a ckpt.pt to fine-tune from")
    ap.add_argument("--sim-lambda", type=float, default=0.0,
                    help="weight of the embedding-similarity regulariser "
                         "(completed graphs vs their reference realisation); "
                         "0 = off. Requires --init-from: the frozen pretrained "
                         "encoder anchors the similarity.")
    ap.add_argument("--spatial-terms", default="diameter",
                    help="spatial only: which distance-geometry descriptors "
                         "enter the score (subset of diameter,radius,avg_spl). "
                         "Component count is deliberately absent: beta_0 is "
                         "pinned by the PD, so it is constant across a class.")
    ap.add_argument("--sim-agg",
                    choices=["match", "pool", "ged", "cycles", "spatial"],
                    default="match",
                    help="match: Hungarian-matched cosine of the (N,d) node "
                         "embeddings. pool: cosine of the pooled graph "
                         "embedding. ged: NANL-style Sinkhorn soft transport "
                         "plan + negative L1 feature alignment (a GED proxy). "
                         "cycles: no embeddings at all -- exact counts of "
                         "cycles of length 3..kmax vs the reference. spatial: "
                         "distance geometry (diameter / radius / average "
                         "shortest path) vs the reference. Neither is "
                         "differentiable, so reward placement only.")
    ap.add_argument("--sim-stages", action="store_true",
                    help="compare the two graphs' FILTRATION HISTORIES stage "
                         "by stage instead of only the finished graph. A "
                         "completed graph has degree == node_time and zero "
                         "capacity by definition, so terminal features are "
                         "identical across a PD class; mid-filtration states "
                         "genuinely differ. Ignored by --sim-agg cycles.")
    ap.add_argument("--cycle-lengths", default="3,4,5",
                    help="cycles only: which cycle lengths enter the score "
                         "(subset of 3,4,5). '3' alone is a pure triangle term.")
    ap.add_argument("--cycle-clip", type=float, default=2.0,
                    help="cycles only: cap on the per-length relative error. "
                         "Keeps the term bounded when a reference has zero "
                         "k-cycles (common on sparse data like ENZYMES), so "
                         "it can never exceed the -50 dead-end penalty.")
    ap.add_argument("--cycle-mode", choices=["match", "excess"], default="match",
                    help="cycles only: match penalises any deviation from the "
                         "reference count; excess penalises only having MORE "
                         "than the reference (a one-sided penalty)")
    ap.add_argument("--sinkhorn-temp", type=float, default=0.1,
                    help="ged only: Sinkhorn temperature; ->0 sharpens the "
                         "soft plan toward a hard permutation")
    ap.add_argument("--sinkhorn-iters", type=int, default=20,
                    help="ged only: Sinkhorn normalisation iterations")
    ap.add_argument("--sim-centered", action="store_true",
                    help="subtract the running class-mean embedding before "
                         "comparing. Terminal features are identical across a "
                         "PD class, so ~95%% of the embedding is a shared "
                         "component and raw cosine pins at ~0.999; centering "
                         "restores the range.")
    ap.add_argument("--center-warmup", type=int, default=8,
                    help="completed graphs per PD to accumulate before the "
                         "class mean is used (uncentered until then)")
    ap.add_argument("--features", choices=["basic", "rich"], default="basic",
                    help="basic: the original 9 node features, which are fixed "
                         "by the PD once a trajectory completes. rich: adds "
                         "triangles, clustering, 2-hop reach and mean "
                         "neighbour degree, which vary within a PD class. "
                         "NOTE rich changes the input width, so a rich model "
                         "cannot be initialised from a basic checkpoint.")
    ap.add_argument("--sim-encoder", choices=["live", "frozen"], default="live",
                    help="live: the training encoder embeds both sides, the "
                         "reference under no_grad (stop-gradient anchor). "
                         "frozen: a frozen copy of the pretrained encoder "
                         "(stationary reward; ablation).")
    ap.add_argument("--sim-place", choices=["reward", "loss"], default="reward",
                    help="reward: log R += lambda*(sim - center) "
                         "(multiplicative reward shaping, frozen encoder). "
                         "loss: TB loss -= lambda*sim with gradients through "
                         "the live encoder.")
    ap.add_argument("--sim-center", type=float, default=0.0,
                    help="baseline subtracted from sim in reward mode. Measured "
                         "pretrained-policy sim is 0.898 +/- 0.018 on comm20, "
                         "so 0.9 keeps the completion incentive unchanged while "
                         "lambda scales within-class preference.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch CPU threads. Keep at 1: our tensors are tiny "
                         "(B x N x N with N <= ~175) and thread sync dominates. "
                         "Measured on a 40-core node at load 43: 955 ms/iter at "
                         "40 threads vs 0.01 ms/iter at 1 -- a ~95,000x difference.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    set_feature_mode(args.features)
    rng = np.random.default_rng(args.seed)
    root = args.data_root or str(Path(__file__).resolve().parents[1] / "data")
    out = Path(args.out or f"runs/{args.dataset}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading {args.dataset} from {root}", flush=True)
    train = load_targets(args.dataset, "train", root, args.max_nodes)
    val = load_targets(args.dataset, "test", root, args.max_nodes)
    if args.single_target is not None:
        train = [train[args.single_target]]
    print(f"[data] {len(train)} train targets, {len(val)} held-out", flush=True)
    ns = [s.num_nodes for s, _ in train]
    es = [s.num_edges for s, _ in train]
    print(f"[data] n in [{min(ns)},{max(ns)}]  |E| in [{min(es)},{max(es)}]  "
          f"mean |E| = {np.mean(es):.1f}  (= mean trajectory length)", flush=True)

    scorer = (ConstantScorer() if args.constant_scorer
              else DescriptorScorer.fit([A for _, A in train]))
    if not args.constant_scorer:
        (out / "scorer.json").write_text(json.dumps(scorer.to_dict(), indent=1))

    model = TopoGFN(num_emb=args.num_emb, num_layers=args.num_layers,
                    num_mlp_layers=args.num_mlp_layers, rank=args.rank,
                    do_bck=args.learn_pb).to(args.device)
    if args.init_from:
        state = torch.load(args.init_from, map_location="cpu", weights_only=False)
        prev = state.get("args", {}).get("features", "basic")
        if prev != args.features:
            raise SystemExit(
                f"--init-from checkpoint was trained with --features {prev}, "
                f"but this run asks for {args.features}. The input width "
                f"differs ({num_node_features(prev)} vs "
                f"{num_node_features(args.features)}), so the weights do not "
                f"transfer -- pretrain from scratch with --features "
                f"{args.features} first.")
        model.load_state_dict(state["model"])
        print(f"[init] fine-tuning from {args.init_from} "
              f"(epoch {state.get('epoch', '?')})", flush=True)

    sim_fn = None
    if args.sim_lambda:
        if not args.init_from:
            raise SystemExit("--sim-lambda needs --init-from (the frozen "
                             "pretrained encoder anchors the similarity)")
        from topo_gfn.similarity import EmbedSim
        if args.sim_agg in ("cycles", "spatial") and args.sim_place == "loss":
            raise SystemExit(f"--sim-agg {args.sim_agg} is not "
                             "differentiable; use "
                             "--sim-place reward (a GFlowNet reward need not "
                             "be differentiable)")
        sim_fn = EmbedSim(model.encoder, agg=args.sim_agg,
                          source=args.sim_encoder,
                          sinkhorn_temp=args.sinkhorn_temp,
                          sinkhorn_iters=args.sinkhorn_iters,
                          centered=args.sim_centered,
                          center_warmup=args.center_warmup,
                          stages=args.sim_stages,
                          cycle_lengths=[int(x) for x in
                                         args.cycle_lengths.split(",")],
                          cycle_mode=args.cycle_mode,
                          cycle_clip=args.cycle_clip,
                          spatial_terms=[x for x in
                                         args.spatial_terms.split(",") if x])
        extra = (f" lengths={args.cycle_lengths} mode={args.cycle_mode}"
                 if args.sim_agg == "cycles" else
                 f" terms={args.spatial_terms}"
                 if args.sim_agg == "spatial" else
                 f" centered={args.sim_centered} stages={args.sim_stages}"
                 f" encoder={args.sim_encoder}")
        print(f"[sim] lambda={args.sim_lambda} agg={args.sim_agg} "
              f"place={args.sim_place}{extra}", flush=True)

    z_params = list(model.mlp_logZ.parameters())
    z_ids = {id(p) for p in z_params}
    body = [p for p in model.parameters() if id(p) not in z_ids]
    opt = torch.optim.Adam(body, lr=args.lr)
    opt_z = torch.optim.Adam(z_params, lr=args.lr_z)

    soft = args.violation_penalty is not None
    sampler = TopoSampler(model, soft=soft)
    algo = TrajectoryBalance(model, scorer, beta=args.beta,
                             sim_fn=sim_fn, sim_lambda=args.sim_lambda,
                             sim_place=args.sim_place,
                             sim_center=args.sim_center,
                             score_floor=args.score_floor,
                             pd_penalty=args.pd_penalty, soft=soft,
                             violation_penalty=(args.violation_penalty or 0.0),
                             learn_pb=args.learn_pb)

    # An epoch is a FULL PASS over the training targets: shuffle, then visit
    # every target exactly once in batches (the last batch may be short).
    iters_per_epoch = int(np.ceil(len(train) / args.batch_size))
    if args.n_epochs is not None:
        n_epochs = args.n_epochs
        total_iters = n_epochs * iters_per_epoch
    else:
        total_iters = args.n_iterations
        n_epochs = int(np.ceil(total_iters / iters_per_epoch))

    print(f"[train] {n_epochs} epochs over {len(train)} targets, batch "
          f"{args.batch_size}  ->  {iters_per_epoch} iteration(s)/epoch, "
          f"{total_iters} gradient steps total", flush=True)
    if args.learn_pb:
        print("[train] learned backward policy (P_B parameterised, not uniform)",
              flush=True)
    if soft:
        print(f"[train] SOFT constraints: violation_penalty="
              f"{args.violation_penalty} pd_penalty={args.pd_penalty}",
              flush=True)
    print(f"[train] device={args.device}  beta={args.beta}  "
          f"scorer={'constant' if args.constant_scorer else 'descriptor'}  "
          f"features={args.features} ({num_node_features()} per node)", flush=True)

    hist, t0, it = [], time.time(), 0
    stop = False
    for epoch in range(1, n_epochs + 1):
        if stop:
            break
        perm = rng.permutation(len(train))          # full pass, no replacement
        if args.size_buckets:
            # Group targets of similar N into a batch. batch_states pads to the
            # batch's largest N, so mixing n=10 with n=126 (enzymes' full range)
            # inflates every small graph 150x in the (N,N) pair tensors. Sorting
            # by size first keeps each batch homogeneous; the shuffle above plus
            # the batch-order shuffle below keep it stochastic, and it is still
            # a full pass over every target exactly once.
            perm = perm[np.argsort([train[i][0].num_nodes for i in perm],
                                   kind="stable")]
        batches = [perm[b0:b0 + args.batch_size]
                   for b0 in range(0, len(perm), args.batch_size)]
        if args.size_buckets:
            rng.shuffle(batches)
        ep_stats = []
        for pick in batches:
            it += 1
            if it > total_iters:
                stop = True
                break
            envs = [TopoEnv(train[i][0]) for i in pick]
            for e, i in zip(envs, pick):
                e.ref_adj = train[i][1]      # reference realisation of the PD

            trajs = sampler.sample(envs)

            opt.zero_grad(set_to_none=True)
            opt_z.zero_grad(set_to_none=True)
            _, info = algo.backward_and_info(
                trajs, micro_bs=args.micro_batch, chunk=args.recompute_chunk)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt_z.step()

            info["iter"] = it
            info["epoch"] = epoch
            info["n_targets"] = len(pick)
            info["grad_norm"] = float(gn)
            hist.append(info)
            ep_stats.append(info)

            if args.log_every and it % args.log_every == 0:
                el = time.time() - t0
                print(f"  it {it:7d} | loss {info['loss']:10.3f} "
                      f"| logZ {info['logZ']:8.3f} "
                      f"| complete {info['completion_rate']:5.2f} "
                      f"| {el/it:.2f}s/it", flush=True)

        if ep_stats:
            el = time.time() - t0
            agg = lambda k: float(np.mean([s[k] for s in ep_stats if k in s]))
            sim_str = (f"| sim {agg('sim'):6.3f} "
                       if any("sim" in s for s in ep_stats) else "")
            if any(s.get("violations") for s in ep_stats):
                sim_str += f"| viol {agg('violations'):5.1f} "
            if any("pd_dev" in s for s in ep_stats):
                sim_str += f"| pdW1 {agg('pd_dev'):6.2f} "
            print(f"epoch {epoch:6d}/{n_epochs} | it {it:7d} "
                  f"| loss {agg('loss'):10.3f} | logZ {agg('logZ'):8.3f} "
                  f"| complete {agg('completion_rate'):5.2f} "
                  f"| logR {agg('log_r'):8.2f} | steps {agg('mean_steps'):5.1f} "
                  f"{sim_str}| {el/max(1,epoch):.2f}s/ep", flush=True)

        if epoch % args.ckpt_every == 0 or epoch == n_epochs or stop:
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "iter": it, "epoch": epoch}, out / "ckpt.pt")
            (out / "history.json").write_text(json.dumps(hist))

    print(f"[done] {epoch} epochs, {it} steps, {time.time()-t0:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
