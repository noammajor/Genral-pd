"""Pre-compute per-graph persistence landscape vectors (µ_G₀) for all splits
and save the per-graph arrays + dataset-level mean µ̄.

Output (per dataset, written to data/{dataset}/processed/):
  mu_train.npy  — float32 (N_train, 2*L*S)  one µ_G₀ per training graph
  mu_val.npy    — float32 (N_val,   2*L*S)  one µ_G₀ per val graph
  mu_test.npy   — float32 (N_test,  2*L*S)  one µ_G₀ per test graph
  mu_mean.npy   — float32 (2*L*S,)          mean µ̄ over training graphs
                                             (fallback for unconditional gen)

Usage
-----
  python scripts/precompute_mu.py                          # all four datasets
  python scripts/precompute_mu.py --datasets comm20 planar # specific datasets
  python scripts/precompute_mu.py --num_landscapes 5 --resolution 100
"""

import argparse
import time
from pathlib import Path

import numpy as np
import networkx as nx
import torch

from topo_gen.pdm_utils import graph_mu

DATASETS  = ["comm20", "planar", "enzymes", "sbm"]
SPLITS    = ["train", "val", "test"]
DATA_ROOT = Path(__file__).parent.parent / "data"


def load_graphs(dataset: str, split: str) -> list[nx.Graph]:
    """Load graphs from packed {split}.pt into a list of nx.Graph."""
    pt_path = DATA_ROOT / dataset / "processed" / f"{split}.pt"
    data, meta = torch.load(pt_path, weights_only=False)

    x_offsets  = meta["x"]
    ei_offsets = meta["edge_index"]
    n_graphs   = len(x_offsets) - 1

    graphs = []
    for g in range(n_graphs):
        n_nodes = int(x_offsets[g + 1] - x_offsets[g])
        e_start = int(ei_offsets[g])
        e_end   = int(ei_offsets[g + 1])
        ei      = data.edge_index[:, e_start:e_end]  # (2, m)

        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        src = ei[0].tolist()
        dst = ei[1].tolist()
        for u, v in zip(src, dst):
            if u < v:
                G.add_edge(u, v)
        graphs.append(G)

    return graphs


def compute_mu_array(
    graphs: list,
    num_landscapes: int,
    resolution: int,
    label: str,
) -> np.ndarray:
    """Compute µ_G₀ for every graph in the list, with progress printing."""
    n   = len(graphs)
    dim = 2 * num_landscapes * resolution
    mu  = np.zeros((n, dim), dtype=np.float32)
    t0  = time.time()
    for i, G in enumerate(graphs):
        mu[i] = graph_mu(G, num_landscapes=num_landscapes, resolution=resolution)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"    {label} [{i+1:4d}/{n}]  {time.time()-t0:.1f}s")
    return mu


def compute_and_save(
    dataset: str,
    num_landscapes: int,
    resolution: int,
) -> None:
    out_dir = DATA_ROOT / dataset / "processed"
    dim = 2 * num_landscapes * resolution

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}  |  µ dim = {dim}")
    t_ds = time.time()

    mu_train = None
    for split in SPLITS:
        graphs = load_graphs(dataset, split)
        mu = compute_mu_array(graphs, num_landscapes, resolution, label=split)
        out_path = out_dir / f"mu_{split}.npy"
        np.save(out_path, mu)
        print(f"  mu_{split}.npy : {mu.shape}  → {out_path}")
        if split == "train":
            mu_train = mu

    # mean over training graphs only
    mu_mean = mu_train.mean(axis=0)
    mean_path = out_dir / "mu_mean.npy"
    np.save(mean_path, mu_mean)
    nonzero = (mu_mean != 0).sum()
    print(f"  mu_mean.npy  : {mu_mean.shape}  nonzero={nonzero}  → {mean_path}")
    print(f"  Dataset done in {time.time()-t_ds:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS,
        choices=DATASETS, metavar="DATASET",
        help="Datasets to process (default: all four)",
    )
    parser.add_argument("--num_landscapes", type=int, default=5)
    parser.add_argument("--resolution",     type=int, default=100)
    args = parser.parse_args()

    print(f"num_landscapes={args.num_landscapes}  resolution={args.resolution}")
    print(f"Output µ dim = {2 * args.num_landscapes * args.resolution}")

    for ds in args.datasets:
        compute_and_save(ds, args.num_landscapes, args.resolution)

    print("\nAll done.")


if __name__ == "__main__":
    main()
