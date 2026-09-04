"""Evaluate a trained GFlowNet checkpoint on held-out targets.

    python -m topo_gfn.eval --run runs/comm20_ep2500_seed0_20026585
    python -m topo_gfn.eval --run runs/... --split test --samples 128

Samples K trajectories per held-out target and reports:
  - solved@K: fraction of targets with at least one completed rollout
  - completion rate: completed rollouts / all rollouts
  - PD compliance, verified from scratch: every completed graph's degree-PD
    is recomputed and compared to the target multiset (completion implies
    compliance by construction; this checks it end to end)
  - diversity: distinct graphs (by adjacency) among a target's compliant samples
  - scorer value of compliant samples (the shaping term the policy trained on)

Writes eval_<split>_K<samples>.json next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from topo_gen.filtrations import degree_filtration
from topo_gen.persistence import persistence_diagrams
from topo_gfn.env import TopoEnv
from topo_gfn.gfn import TopoSampler
from topo_gfn.policy import TopoGFN, set_feature_mode
from topo_gfn.score import ConstantScorer, DescriptorScorer
from topo_gfn.train import load_targets


def pd_of_adj(adj: np.ndarray) -> tuple:
    """Sorted degree-PD multisets of a dense adjacency matrix."""
    n = adj.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(u), int(v)) for u, v in zip(*np.nonzero(np.triu(adj))))
    nt, et, _ = degree_filtration(G)
    pd0, pd1 = persistence_diagrams(nt, et)
    return sorted(map(tuple, pd0)), sorted(map(tuple, pd1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing ckpt.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=128, help="rollouts per target")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    run = Path(args.run)
    ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    ta = ckpt["args"]
    set_feature_mode(ta.get("features", "basic"))
    print(f"[ckpt] {run} @ epoch {ckpt['epoch']} (iter {ckpt['iter']}), "
          f"trained on {ta['dataset']}", flush=True)

    model = TopoGFN(num_emb=ta["num_emb"], num_layers=ta["num_layers"],
                    num_mlp_layers=ta["num_mlp_layers"], rank=ta["rank"])
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()

    scorer_file = run / "scorer.json"
    scorer = (DescriptorScorer.from_dict(json.loads(scorer_file.read_text()))
              if scorer_file.exists() else ConstantScorer())

    root = args.data_root or str(Path(__file__).resolve().parents[1] / "data")
    targets = load_targets(ta["dataset"], args.split, root, ta.get("max_nodes"))
    T, K = len(targets), args.samples
    print(f"[data] {T} {args.split} targets x {K} samples", flush=True)

    sampler = TopoSampler(model)
    per = [defaultdict(float) for _ in range(T)]      # per-target tallies
    uniq = [set() for _ in range(T)]                  # distinct compliant graphs
    scores = [[] for _ in range(T)]
    target_pd = [pd_of_adj(A) for _, A in targets]
    mismatches = 0

    jobs = [(ti, r) for ti in range(T) for r in range(K)]
    t0 = time.time()
    for b0 in range(0, len(jobs), args.batch_size):
        chunk = jobs[b0:b0 + args.batch_size]
        envs = [TopoEnv(targets[ti][0]) for ti, _ in chunk]
        trajs = sampler.sample(envs, keep_states=False)
        for (ti, _), tr in zip(chunk, trajs):
            per[ti]["n"] += 1
            per[ti]["steps"] += tr.n_steps
            if not tr.completed:
                continue
            per[ti]["done"] += 1
            n = targets[ti][0].num_nodes
            adj = np.asarray(tr.terminal.adj[:n, :n], dtype=np.float64)
            if pd_of_adj(adj) == target_pd[ti]:
                per[ti]["compliant"] += 1
            else:
                mismatches += 1
            uniq[ti].add(adj.astype(np.uint8).tobytes())
            scores[ti].append(float(scorer.score(adj)))
        done = b0 + len(chunk)
        el = time.time() - t0
        print(f"  {done}/{len(jobs)} rollouts | {el:.0f}s "
              f"({el/done*1000:.0f} ms/rollout)", flush=True)

    comp = np.array([p["done"] / p["n"] for p in per])
    solved = comp > 0
    res = {
        "run": str(run), "epoch": ckpt["epoch"], "split": args.split,
        "samples_per_target": K, "n_targets": T,
        "solved_at_k": float(solved.mean()),
        "completion_rate": float(comp.mean()),
        "compliance_mismatches": mismatches,
        "mean_distinct_per_solved": float(np.mean(
            [len(u) for u, s in zip(uniq, solved) if s])) if solved.any() else 0.0,
        "mean_score_completed": float(np.mean(
            [s for ss in scores for s in ss])) if any(scores) else None,
        "per_target": [
            {"n_nodes": targets[ti][0].num_nodes,
             "n_edges": targets[ti][0].num_edges,
             "completion": comp[ti], "distinct": len(uniq[ti]),
             "mean_steps": per[ti]["steps"] / per[ti]["n"]}
            for ti in range(T)
        ],
    }

    out = run / f"eval_{args.split}_K{K}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\n[eval] solved@{K}: {res['solved_at_k']:.2%} of targets  "
          f"| completion rate: {res['completion_rate']:.2%}", flush=True)
    print(f"[eval] compliance mismatches among completed: {mismatches} "
          f"(0 expected -- completion implies exact PD compliance)", flush=True)
    print(f"[eval] distinct compliant graphs per solved target: "
          f"{res['mean_distinct_per_solved']:.1f} / {K}", flush=True)
    print(f"[eval] -> {out}", flush=True)


if __name__ == "__main__":
    main()
