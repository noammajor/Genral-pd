"""CP-SAT v4 Solver Evaluation on Benchmark Datasets (script version).

For every graph in every dataset's test split, runs the CP-SAT v4 degree-filtration
solver and verifies PH equivalence of the generated graph.

Processes one graph at a time, giving all cp_sat_workers threads to CP-SAT for
that graph. With 120 CPU cores: set --cp-sat-workers 120.

Dumps one .pkl per dataset to:
    dumps/cp_v4/{dataset}_cpv4_tl{time_limit}s_cw{cp_sat_workers}cw.pkl

Usage:
    python cp_v4_dataset_eval.py [--datasets comm20 sbm planar enzymes]
                                 [--time-limit 1800]
                                 [--cp-sat-workers 120]
                                 [--split test]
                                 [--max-graphs N]
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
import pickle
import time

import networkx as nx

TOPOGEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOPOGEN_ROOT not in sys.path:
    sys.path.insert(0, TOPOGEN_ROOT)

DATA_ROOT = os.path.join(TOPOGEN_ROOT, "data")
DUMP_ROOT = os.path.join(TOPOGEN_ROOT, "dumps", "cp_v4")

from utils.dataset_utils import load_split, pyg_to_nx
from topo_gen.degree_filtration_cp_v4 import solve, degree_filtration_ph, ph_equal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    datasets       = ["comm20", "sbm", "planar", "enzymes"],
    split          = "test",
    time_limit     = 1800.0,   # 30 min per graph
    cp_sat_workers = 120,
    max_graphs     = None,
)

# ---------------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------------

def _dump_path(dump_root, ds_name, cfg):
    cap_str = f"_max{cfg.max_graphs}" if cfg.max_graphs is not None else ""
    fname = (
        f"{ds_name}"
        f"_cpv4"
        f"_tl{int(cfg.time_limit)}s"
        f"_cw{cfg.cp_sat_workers}cw"
        f"{cap_str}.pkl"
    )
    return os.path.join(dump_root, fname)


def _save_dump(path, ds_name, results, cfg):
    to_save = dict(
        cp_version      = "v4",
        dataset         = ds_name,
        time_limit      = cfg.time_limit,
        cp_sat_workers  = cfg.cp_sat_workers,
        max_graphs      = cfg.max_graphs,
        split           = cfg.split,
        results         = results,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run(cfg):
    os.makedirs(DUMP_ROOT, exist_ok=True)

    for ds_name in cfg.datasets:
        print(f"\n{'='*65}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*65}")

        try:
            pyg_graphs = load_split(DATA_ROOT, ds_name, cfg.split)
        except Exception as exc:
            print(f"  LOAD ERROR: {exc}")
            continue

        graphs = [pyg_to_nx(g) for g in pyg_graphs]
        if cfg.max_graphs is not None:
            graphs = graphs[:cfg.max_graphs]

        n_graphs = len(graphs)
        sizes = [g.number_of_nodes() for g in graphs]
        print(f"  {n_graphs} graphs, n: min={min(sizes)} max={max(sizes)} "
              f"mean={sum(sizes)/len(sizes):.1f}")
        print(f"  CP-SAT threads per graph: {cfg.cp_sat_workers}")
        print(f"  Time limit per graph: {cfg.time_limit:.0f}s")
        print(f"\n  {'idx':>5}  {'n':>4}  {'e':>5}  {'deg':>4}  "
              f"{'outcome':<8}  {'time(s)':>8}  {'done':>6}")
        print(f"  {'-'*55}")

        dump_path = _dump_path(DUMP_ROOT, ds_name, cfg)
        results   = []
        t_ds_start = time.perf_counter()

        for idx, G_src in enumerate(graphs):
            G = nx.convert_node_labels_to_integers(G_src)
            p0, p1 = degree_filtration_ph(G)

            t0 = time.perf_counter()
            H = solve(p0, p1, time_limit=cfg.time_limit, verbose=False,
                      cp_sat_workers=cfg.cp_sat_workers)
            elapsed = time.perf_counter() - t0

            if H is None:
                outcome   = "UNKNOWN"
                ph_ok     = False
                cp_status = "UNKNOWN"
            else:
                p0_H, p1_H = degree_filtration_ph(H)
                ph_ok      = ph_equal(p0, p1, p0_H, p1_H)
                outcome    = "PASS" if ph_ok else "FAIL"
                cp_status  = outcome

            max_deg = max(d for _, d in G.degree()) if G.number_of_nodes() > 0 else 0

            r = dict(
                graph_idx  = idx,
                ref_graph  = G,
                gen_graph  = H,
                n          = G.number_of_nodes(),
                e          = G.number_of_edges(),
                max_deg    = max_deg,
                outcome    = outcome,
                ph_ok      = ph_ok,
                cp_status  = cp_status,
                elapsed    = elapsed,
                p0_ref     = p0,
                p1_ref     = p1,
            )
            results.append(r)

            print(f"  {idx:>5}  {r['n']:>4}  {r['e']:>5}  {max_deg:>4}  "
                  f"{outcome:<8}  {elapsed:>8.2f}  "
                  f"{len(results):>3}/{n_graphs}")

            _save_dump(dump_path, ds_name, results, cfg)

        total = time.perf_counter() - t_ds_start
        n_pass = sum(1 for r in results if r["outcome"] == "PASS")
        print(f"\n  => {ds_name}: {n_pass}/{n_graphs} PASS  "
              f"total wall-clock {total:.1f}s")
        print(f"  => Dumped -> {dump_path}")

    print("\n\nAll done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets",        nargs="+", default=DEFAULTS["datasets"])
    p.add_argument("--split",           default=DEFAULTS["split"])
    p.add_argument("--time-limit",      type=float, default=DEFAULTS["time_limit"],
                   help="Per-graph CP-SAT time limit in seconds (default: 1800)")
    p.add_argument("--cp-sat-workers",  type=int, default=DEFAULTS["cp_sat_workers"],
                   help="CP-SAT internal search threads (default: 120)")
    p.add_argument("--max-graphs",      type=int, default=DEFAULTS["max_graphs"],
                   help="Cap graphs per dataset (default: all)")
    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
