"""Per-PD isomorphism-aware test of a trained checkpoint.

    python -m topo_gfn.iso_test --run runs/comm20_ep2500_seed0_20026585

For every held-out PD: sample exactly --samples rollouts and report
  1. completion successes (completed = exact PD compliance, re-verified)
  2. number of pairwise NON-ISOMORPHIC graphs among the compliant samples
  3. one PNG per PD drawing the target plus every isomorphism class found,
     each labelled with how many of the rollouts produced it.

Writes iso_test_<split>_K<samples>/{target_NN.png, summary.json}.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from topo_gfn.env import TopoEnv
from topo_gen.pdm_utils import _wasserstein1_pd
from topo_gfn.eval import pd_of_adj
from topo_gfn.gallery import draw, to_graph
from topo_gfn.gfn import TopoSampler
from topo_gfn.policy import TopoGFN, set_feature_mode
from topo_gfn.train import load_targets


def iso_classes(graphs: list[nx.Graph]) -> list[tuple[nx.Graph, int]]:
    """Group graphs into isomorphism classes -> (representative, count).

    Buckets by WL hash first, so is_isomorphic only runs within a bucket.
    """
    buckets: dict[str, list[list]] = {}
    for G in graphs:
        h = nx.weisfeiler_lehman_graph_hash(G)
        for entry in buckets.setdefault(h, []):
            if nx.is_isomorphic(entry[0], G):
                entry[1] += 1
                break
        else:
            buckets[h].append([G, 1])
    out = [tuple(e) for bs in buckets.values() for e in bs]
    return sorted(out, key=lambda e: -e[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=100, help="rollouts per PD")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-panels", type=int, default=100,
                    help="cap on drawn isomorphism classes per PD, per group")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    run = Path(args.run)
    ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    ta = ckpt["args"]
    set_feature_mode(ta.get("features", "basic"))
    # Soft-constraint checkpoints generate graphs that MAY miss the target
    # diagram, so compliance becomes a measured rate rather than an invariant.
    soft = ta.get("violation_penalty") is not None
    if soft:
        print(f"[iso-test] soft checkpoint (violation_penalty="
              f"{ta['violation_penalty']}): compliance is measured, not assumed",
              flush=True)
    model = TopoGFN(num_emb=ta["num_emb"], num_layers=ta["num_layers"],
                    num_mlp_layers=ta["num_mlp_layers"], rank=ta["rank"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    sampler = TopoSampler(model, soft=soft)

    root = args.data_root or str(Path(__file__).resolve().parents[1] / "data")
    targets = load_targets(ta["dataset"], args.split, root, ta.get("max_nodes"))
    target_pd = [pd_of_adj(A) for _, A in targets]
    T, K = len(targets), args.samples
    print(f"[iso-test] {T} {args.split} PDs x {K} rollouts", flush=True)

    outdir = run / f"iso_test_{args.split}_K{K}"
    outdir.mkdir(exist_ok=True)

    # -- sample exactly K rollouts per PD -----------------------------------
    compliant: list[list[tuple]] = [[] for _ in range(T)]   # (adj, violations)
    offtarget: list[list[tuple]] = [[] for _ in range(T)]   # (adj, W1, viol)
    completed = [0] * T
    noncompliant = [0] * T
    viol_all: list[list[int]] = [[] for _ in range(T)]      # every finished graph
    jobs = [(ti, r) for ti in range(T) for r in range(K)]
    t0 = time.time()
    for b0 in range(0, len(jobs), args.batch_size):
        chunk = jobs[b0:b0 + args.batch_size]
        envs = [TopoEnv(targets[ti][0]) for ti, _ in chunk]
        trajs = sampler.sample(envs)
        for (ti, _), tr in zip(chunk, trajs):
            if not tr.completed:
                continue
            n = targets[ti][0].num_nodes
            adj = np.asarray(tr.terminal.adj[:n, :n], dtype=np.uint8)
            viol_all[ti].append(int(tr.violations))
            got = pd_of_adj(adj.astype(float))
            if got != target_pd[ti]:
                # only reachable under soft constraints; in hard mode
                # completion provably implies compliance
                assert soft, f"PD mismatch on completed rollout, target {ti}"
                noncompliant[ti] += 1
                d = (_wasserstein1_pd(list(got[0]), list(target_pd[ti][0]))
                     + _wasserstein1_pd(list(got[1]), list(target_pd[ti][1])))
                offtarget[ti].append((adj, d, int(tr.violations)))
                continue
            completed[ti] += 1
            compliant[ti].append((adj, int(tr.violations)))
        print(f"  {b0 + len(chunk)}/{len(jobs)} rollouts | "
              f"{time.time()-t0:.0f}s", flush=True)

    # -- group into isomorphism classes and draw ----------------------------
    summary = []
    for ti, (sched, A) in enumerate(targets):
        Gt = to_graph(A)
        vmap = {}
        for a, v in compliant[ti]:
            vmap.setdefault(a.tobytes(), v)
        classes = iso_classes([to_graph(a) for a, _ in compliant[ti]])
        iso_to_target = sum(c for G, c in classes if nx.is_isomorphic(G, Gt))

        off_classes = []
        if offtarget[ti]:
            byd = {}
            for a, d, v in offtarget[ti]:
                byd.setdefault(a.tobytes(), [to_graph(a), d, 0, v])[2] += 1
            off_classes = sorted(byd.values(), key=lambda e: (e[1], -e[2]))
            off_classes = off_classes[:args.max_panels]
        classes = classes[:args.max_panels]

        panels = 1 + len(classes) + len(off_classes)
        cols = min(8, max(4, panels))
        rows = int(np.ceil(panels / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(2.1 * cols, 2.1 * rows), squeeze=False)
        flat = axes.ravel()
        vmax = max(dict(Gt.degree()).values())
        draw(flat[0], Gt, f"TARGET  n={sched.num_nodes} |E|={sched.num_edges}",
             vmax)
        for j, (G, cnt) in enumerate(classes):
            title = f"x{cnt}" + (" (= target)" if nx.is_isomorphic(G, Gt) else "")
            draw(flat[j + 1], G, title, vmax)
            flat[j + 1].set_title(title, fontsize=9, color="darkgreen")
        base = 1 + len(classes)
        for j, (G, d, cnt, v) in enumerate(off_classes):
            draw(flat[base + j], G, "", vmax)
            flat[base + j].set_title(f"x{cnt}  W1={d:.1f}  v={v}", fontsize=9,
                                     color="crimson")
        for ax in flat[panels:]:
            ax.set_axis_off()
        extra = (f"  |  RED = off-target ({noncompliant[ti]} of {K}, "
                 f"{len(off_classes)} classes shown)" if noncompliant[ti] else "")
        fig.suptitle(
            f"PD #{ti}: GREEN = PD-compliant ({completed[ti]}/{K}, "
            f"{len(classes)} classes){extra}"
            f"  |  mean violations/graph "
            f"{(float(np.mean(viol_all[ti])) if viol_all[ti] else 0.0):.2f}"
            + (f", {iso_to_target} isomorphic to target" if iso_to_target else ""),
            fontsize=11)
        fig.tight_layout()
        fig.savefig(outdir / f"target_{ti:02d}.png", dpi=115)
        plt.close(fig)

        summary.append({
            "target": ti, "n": sched.num_nodes, "E": sched.num_edges,
            "completed": completed[ti], "samples": K,
            "non_compliant": noncompliant[ti],
            "off_target_classes": len(off_classes),
            "mean_violations": (float(np.mean(viol_all[ti]))
                                if viol_all[ti] else 0.0),
            "non_isomorphic": len(classes),
            "isomorphic_to_target": iso_to_target,
        })
        print(f"PD {ti:2d}: {completed[ti]:3d}/{K} completed | "
              f"{len(classes):3d} iso-classes | "
              f"{iso_to_target} = target", flush=True)

    res = {
        "run": str(run), "epoch": ckpt["epoch"], "split": args.split,
        "samples_per_pd": K,
        "completion_rate": sum(completed) / (T * K),
        "compliance_rate": (sum(completed) / max(1, sum(completed) + sum(noncompliant))),
        "non_compliant": sum(noncompliant),
        "mean_violations": float(np.mean(
            [v for vs in viol_all for v in vs] or [0.0])),
        "solved": sum(c > 0 for c in completed),
        "mean_non_isomorphic": float(np.mean(
            [s["non_isomorphic"] for s in summary])),
        "per_target": summary,
    }
    (outdir / "summary.json").write_text(json.dumps(res, indent=1))
    if soft:
        print(f"[iso-test] compliance among finished graphs: "
              f"{res['compliance_rate']:.2%} "
              f"({res['non_compliant']} off-target)", flush=True)
    print(f"[iso-test] mean violations per generated graph: "
          f"{res['mean_violations']:.2f}", flush=True)
    print(f"\n[iso-test] completion {res['completion_rate']:.2%} | "
          f"solved {res['solved']}/{T} | "
          f"mean iso-classes {res['mean_non_isomorphic']:.1f}", flush=True)
    print(f"[iso-test] -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
