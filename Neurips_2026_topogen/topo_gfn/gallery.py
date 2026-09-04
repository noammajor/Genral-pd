"""Render held-out targets next to compliant graphs sampled from a checkpoint.

    python -m topo_gfn.gallery --run runs/comm20_ep2500_seed0_20026585

One PNG per target (target graph + up to --keep distinct compliant samples,
nodes coloured by degree = filtration value) plus a montage of every row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from topo_gfn.env import TopoEnv
from topo_gfn.eval import pd_of_adj
from topo_gfn.gfn import TopoSampler
from topo_gfn.policy import TopoGFN, set_feature_mode
from topo_gfn.train import load_targets


def to_graph(adj: np.ndarray) -> nx.Graph:
    n = adj.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(u), int(v)) for u, v in zip(*np.nonzero(np.triu(adj))))
    return G


def draw(ax, G: nx.Graph, title: str, vmax: int):
    pos = nx.spring_layout(G, seed=7)
    deg = [G.degree(v) for v in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.55, width=1.1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=90, node_color=deg,
                           cmap="viridis", vmin=0, vmax=vmax,
                           edgecolors="black", linewidths=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--samples", type=int, default=256, help="rollouts per target")
    ap.add_argument("--keep", type=int, default=4, help="distinct samples to draw")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    run = Path(args.run)
    ckpt = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    ta = ckpt["args"]
    set_feature_mode(ta.get("features", "basic"))
    model = TopoGFN(num_emb=ta["num_emb"], num_layers=ta["num_layers"],
                    num_mlp_layers=ta["num_mlp_layers"], rank=ta["rank"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    sampler = TopoSampler(model)

    root = args.data_root or str(Path(__file__).resolve().parents[1] / "data")
    targets = load_targets(ta["dataset"], args.split, root, ta.get("max_nodes"))
    target_pd = [pd_of_adj(A) for _, A in targets]

    outdir = run / f"gallery_{args.split}"
    outdir.mkdir(exist_ok=True)

    kept = [dict() for _ in targets]        # adj-bytes -> adjacency
    tried = [0] * len(targets)
    # keep sampling targets that still want more distinct compliant graphs
    while True:
        want = [ti for ti in range(len(targets))
                if len(kept[ti]) < args.keep and tried[ti] < args.samples]
        if not want:
            break
        batch = (want * ((args.batch_size // len(want)) + 1))[:args.batch_size]
        envs = [TopoEnv(targets[ti][0]) for ti in batch]
        trajs = sampler.sample(envs)
        for ti, tr in zip(batch, trajs):
            tried[ti] += 1
            if not tr.completed:
                continue
            n = targets[ti][0].num_nodes
            adj = np.asarray(tr.terminal.adj[:n, :n], dtype=np.uint8)
            if pd_of_adj(adj.astype(float)) == target_pd[ti]:
                kept[ti].setdefault(adj.tobytes(), adj)
        print(f"  pending {len(want)} targets | tried {sum(tried)} rollouts",
              flush=True)

    cols = args.keep + 1
    fig_all, axes_all = plt.subplots(len(targets), cols,
                                     figsize=(2.4 * cols, 2.2 * len(targets)))
    summary = []
    for ti, ((sched, A), pds) in enumerate(zip(targets, target_pd)):
        Gt = to_graph(A)
        vmax = max(dict(Gt.degree()).values())
        samples = list(kept[ti].values())[:args.keep]
        rate = f"{len(kept[ti])} distinct in {tried[ti]}"
        fig, axes = plt.subplots(1, cols, figsize=(2.6 * cols, 2.6))
        for axs in (axes, axes_all[ti]):
            draw(axs[0], Gt, f"target #{ti}  n={sched.num_nodes} "
                             f"|E|={sched.num_edges}", vmax)
            for j in range(args.keep):
                ax = axs[j + 1]
                if j < len(samples):
                    draw(ax, to_graph(samples[j]), f"sample {j+1}", vmax)
                else:
                    ax.text(0.5, 0.5, "no compliant\nsample",
                            ha="center", va="center", fontsize=9, color="crimson")
                    ax.set_axis_off()
        fig.suptitle(f"target #{ti}: {rate} compliant rollouts", fontsize=10)
        fig.tight_layout()
        fig.savefig(outdir / f"target_{ti:02d}.png", dpi=140)
        plt.close(fig)
        summary.append({"target": ti, "n": sched.num_nodes,
                        "E": sched.num_edges, "distinct": len(kept[ti]),
                        "tried": tried[ti]})
        print(f"target {ti:2d}: {rate}", flush=True)

    fig_all.tight_layout()
    fig_all.savefig(outdir / "montage.png", dpi=110)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[gallery] -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
