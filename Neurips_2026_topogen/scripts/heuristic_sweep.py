"""Step 1 falsification gate: how much of Algorithm 3's failure rate is fixed by
a one-line heuristic, and does that heuristic distort the output distribution?

Algorithm 3 picks uniformly from a boolean (N,N) candidate mask
(``topo_gen.generation_helpers._sample2d``).  This sweeps a weighted pick,

    p(u,v)  proportional to  (capacity[u] * capacity[v]) ** alpha

with independent exponents for the merge and cycle call sites, and reports
BOTH the completion rate and the descriptor distribution of what completes.

Why both: if some fixed alpha gives high completion everywhere AND leaves the
descriptor distribution undistorted, a learned policy has nothing to add and the
GFlowNet project should stop.  If the best alpha is family-dependent, the
per-target spread is wide, or high-completion alphas skew the descriptors, that
is the headroom a state-conditioned policy exists to exploit.

Descriptors are deliberately DEGREE-ORTHOGONAL.  Every PD-compliant graph has
the identical degree sequence and edge count by construction, so degree-based
statistics carry no signal and degree MMD is identically zero.

The algorithm itself is NOT modified: the sampler is monkeypatched at runtime.

Examples
--------
    python scripts/heuristic_sweep.py --family tree --sizes 20 40 60 --out sweep_tree.json
    python scripts/heuristic_sweep.py --family dataset --dataset planar --out sweep_planar.json
    # Slurm array shard 3 of 16:
    python scripts/heuristic_sweep.py --family er --shard 3 --num-shards 16 --out sweep_er_3.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from topo_gen.filtrations import degree_filtration
from topo_gen.persistence import persistence_diagrams
from topo_gen.result import Success
import topo_gen.generation_helpers as gh
import topo_gen.degree_filtration_pd_equivalent as alg3
import topo_gen.degree_filtration_pd_equivalent_bmin as alg3_bmin


# ---------------------------------------------------------------------------
# Weighted sampler (monkeypatched over _sample2d)
# ---------------------------------------------------------------------------

_HOOK = {"alpha_merge": 0.0, "alpha_cycle": 0.0, "capacity": None, "site": "merge"}
_ORIG_SAMPLE2D = gh._sample2d


def _weighted_sample2d(ind: np.ndarray, rng):
    rows, cols = np.where(ind)
    if not len(rows):
        return -1, -1
    alpha = _HOOK["alpha_merge"] if _HOOK["site"] == "merge" else _HOOK["alpha_cycle"]
    cap = _HOOK["capacity"]
    if alpha == 0.0 or cap is None:
        i = int(rng.integers(len(rows)))
        return int(rows[i]), int(cols[i])
    w = (cap[rows].astype(np.float64) * cap[cols].astype(np.float64)) ** alpha
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        i = int(rng.integers(len(rows)))
        return int(rows[i]), int(cols[i])
    i = int(rng.choice(len(rows), p=w / tot))
    return int(rows[i]), int(cols[i])


def _install_hook():
    """Patch the sampler and tag which call site is active.

    ``_do_merge`` (generation_helpers) is the merge site; the Step-4 call inside
    degree_filtration_pd_equivalent is the cycle site.  We tag by wrapping the
    two callers rather than inspecting the stack.
    """
    gh._sample2d = _weighted_sample2d
    alg3._sample2d = _weighted_sample2d
    alg3_bmin._sample2d = _weighted_sample2d

    orig_do_merge = gh._do_merge

    def do_merge(*a, **kw):
        _HOOK["site"] = "merge"
        return orig_do_merge(*a, **kw)

    gh._do_merge = do_merge
    alg3._do_merge = do_merge
    alg3_bmin._do_merge = do_merge

    # capacity spy: _cap_mask is called on every mask build
    for mod, cls_name in ((alg3, "_StateDeg"),):
        cls = getattr(mod, cls_name)
        orig_cap = cls._cap_mask

        def cap_mask(self, _orig=orig_cap):
            _HOOK["capacity"] = self.capacity
            _HOOK["site"] = "cycle" if _HOOK["site"] == "merge_done" else _HOOK["site"]
            return _orig(self)

        cls._cap_mask = cap_mask


# ---------------------------------------------------------------------------
# Descriptors (degree-orthogonal)
# ---------------------------------------------------------------------------

def descriptors(G: nx.Graph, n: int) -> np.ndarray:
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    if A.shape[0] < n:
        pad = n - A.shape[0]
        A = np.pad(A, ((0, pad), (0, pad)))
    deg = A.sum(1)
    A2 = A @ A
    tri = float(np.trace(A2 @ A) / 6.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = deg * (deg - 1)
        loc = np.where(den > 0, (A2 @ A).diagonal() / np.where(den > 0, den, 1), 0.0)
    clust = float(loc.mean())
    d = np.where(deg > 0, deg, 1.0)
    L = np.eye(A.shape[0]) - A / np.sqrt(np.outer(d, d))
    ev = np.sort(np.linalg.eigvalsh(L))
    src, dst = np.nonzero(A)
    if len(src):
        x, y = deg[src], deg[dst]
        s = x.std() * y.std()
        assort = float(((x * y).mean() - x.mean() * y.mean()) / s) if s > 1e-12 else 0.0
    else:
        assort = 0.0
    sq = float((np.triu(A2, 1) ** 2).sum())
    return np.array([tri, clust, ev[1], ev[-1], float(ev[1:].mean()), assort, sq])


# ---------------------------------------------------------------------------
# Target corpora
# ---------------------------------------------------------------------------

def make_tree(n, rng):
    G = nx.Graph(); G.add_node(0)
    for v in range(1, n):
        G.add_edge(int(rng.integers(0, v)), v)
    return G


def make_er(n, rng, p=0.2):
    G = nx.Graph(); G.add_nodes_from(range(n))
    for u, v in itertools.combinations(range(n), 2):
        if rng.random() < p:
            G.add_edge(u, v)
    return G


def load_dataset_targets(dataset, data_root, split="test", limit=None):
    from utils.dataset_utils import load_split, pyg_to_nx
    graphs = [pyg_to_nx(d) for d in load_split(data_root, dataset, split)]
    return graphs if limit is None else graphs[:limit]


def target_pd(G):
    nt, et, _ = degree_filtration(G)
    return persistence_diagrams(nt, et)


def norm_pd(pd0, pd1):
    return (sorted(map(tuple, pd0)), sorted(map(tuple, pd1)))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def evaluate(G, alpha_m, alpha_c, trials, use_bmin, rng):
    """Return (completion_rate, descriptor_matrix, pd_ok_count)."""
    n = G.number_of_nodes()
    pd0, pd1 = target_pd(G)
    tgt = norm_pd(pd0, pd1)
    solver = (alg3_bmin.degree_filtration_pd_equivalent_bmin if use_bmin
              else alg3.degree_filtration_pd_equivalent)

    _HOOK["alpha_merge"], _HOOK["alpha_cycle"] = alpha_m, alpha_c
    ok, pd_ok, descs = 0, 0, []
    for _ in range(trials):
        _HOOK["capacity"] = None
        r = solver(pd0, pd1, seed=int(rng.integers(1 << 30)))
        if not isinstance(r, Success):
            continue
        ok += 1
        if norm_pd(*target_pd(r.H)) == tgt:
            pd_ok += 1
        descs.append(descriptors(r.H, n))
    return ok / trials, (np.array(descs) if descs else np.zeros((0, 7))), pd_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["tree", "er", "dataset"], default="tree")
    ap.add_argument("--dataset", default="planar")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--er-p", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("--alphas-merge", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 4.0, 6.0, 10.0])
    ap.add_argument("--alphas-cycle", type=float, nargs="+",
                    default=[0.0, 2.0, 6.0])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--n-targets", type=int, default=10)
    ap.add_argument("--bmin", action="store_true", help="also sweep the bmin variant")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sweep.json")
    args = ap.parse_args()

    _install_hook()

    # ---- build the target list -------------------------------------------
    targets = []  # (label, graph)
    if args.family == "dataset":
        root = args.data_root or str(Path(__file__).resolve().parents[1] / "data")
        for i, G in enumerate(load_dataset_targets(args.dataset, root,
                                                   limit=args.n_targets)):
            targets.append((f"{args.dataset}[{i}]", G))
    else:
        for n in args.sizes:
            ps = args.er_p if args.family == "er" else [None]
            for p in ps:
                for i in range(args.n_targets):
                    rng = np.random.default_rng(args.seed + 977 * i + 31 * n)
                    G = (make_tree(n, rng) if args.family == "tree"
                         else make_er(n, rng, p))
                    lbl = f"{args.family}_n{n}" + (f"_p{p}" if p else "") + f"[{i}]"
                    targets.append((lbl, G))

    # ---- enumerate the work, then shard ----------------------------------
    variants = [False] + ([True] if args.bmin else [])
    jobs = [(lbl, G, am, ac, bm)
            for (lbl, G) in targets
            for am in args.alphas_merge
            for ac in args.alphas_cycle
            for bm in variants]
    jobs = jobs[args.shard::args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(jobs)} jobs "
          f"({len(targets)} targets)", flush=True)

    rng = np.random.default_rng(args.seed + 7919 * args.shard)
    rows, t0 = [], time.time()
    for j, (lbl, G, am, ac, bm) in enumerate(jobs):
        rate, descs, pd_ok = evaluate(G, am, ac, args.trials, bm, rng)
        ref = descriptors(G, G.number_of_nodes())
        rows.append({
            "target": lbl,
            "n": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "alpha_merge": am,
            "alpha_cycle": ac,
            "bmin": bm,
            "completion_rate": rate,
            "pd_exact": int(pd_ok),
            "n_completed": int(descs.shape[0]),
            "desc_mean": descs.mean(0).tolist() if len(descs) else None,
            "desc_std": descs.std(0).tolist() if len(descs) else None,
            "desc_source": ref.tolist(),
        })
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=1)
    print(f"wrote {args.out}  ({len(rows)} rows, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
