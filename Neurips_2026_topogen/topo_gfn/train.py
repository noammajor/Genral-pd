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
from topo_gfn.policy import TopoGFN
from topo_gfn.score import ConstantScorer, DescriptorScorer


def load_targets(dataset: str, split: str, data_root: str, max_nodes: int = None):
    """Benchmark split -> (PDSchedule, dense adjacency) per graph."""
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
    ap.add_argument("--n-iterations", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
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
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
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
                    num_mlp_layers=args.num_mlp_layers, rank=args.rank).to(args.device)
    z_params = list(model.mlp_logZ.parameters())
    z_ids = {id(p) for p in z_params}
    body = [p for p in model.parameters() if id(p) not in z_ids]
    opt = torch.optim.Adam(body, lr=args.lr)
    opt_z = torch.optim.Adam(z_params, lr=args.lr_z)

    sampler = TopoSampler(model)
    algo = TrajectoryBalance(model, scorer, beta=args.beta)

    per_epoch = max(1, len(train) / args.batch_size)
    print(f"[train] {args.n_iterations} iterations x batch {args.batch_size}"
          f"  =  {args.n_iterations / per_epoch:.0f} epochs over {len(train)} targets"
          f"  ({per_epoch:.1f} iterations/epoch)", flush=True)
    print(f"[train] device={args.device}  beta={args.beta}  "
          f"scorer={'constant' if args.constant_scorer else 'descriptor'}", flush=True)

    hist, t0 = [], time.time()
    for it in range(1, args.n_iterations + 1):
        pick = rng.integers(0, len(train), size=args.batch_size)
        envs = [TopoEnv(train[i][0]) for i in pick]

        trajs = sampler.sample(envs)
        loss, info = algo.compute_loss(trajs)

        opt.zero_grad(set_to_none=True)
        opt_z.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt_z.step()

        info["iter"] = it
        info["epoch"] = it / per_epoch
        info["grad_norm"] = float(gn)
        hist.append(info)

        if it % args.log_every == 0 or it == 1:
            el = time.time() - t0
            print(f"it {it:6d} | ep {info['epoch']:8.1f} | loss {info['loss']:10.3f} "
                  f"| logZ {info['logZ']:8.3f} | complete {info['completion_rate']:5.2f} "
                  f"| logR {info['log_r']:8.2f} | steps {info['mean_steps']:5.1f} "
                  f"| {el/it:.2f}s/it", flush=True)

        if it % args.ckpt_every == 0 or it == args.n_iterations:
            torch.save({"model": model.state_dict(), "args": vars(args), "iter": it},
                       out / "ckpt.pt")
            (out / "history.json").write_text(json.dumps(hist))

    print(f"[done] {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
