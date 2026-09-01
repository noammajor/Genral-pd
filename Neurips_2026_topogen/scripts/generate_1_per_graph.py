"""Generate 1 PD-Equivalent Graph per Source Graph (script version).

For every source graph in every dataset's test split, across all 5 filtration types:
- Attempt generation until 1 success is collected
- Record: total attempts, wall-clock time per successful graph
- Run PH round-trip and filtration constraint checks on the success
- Dump intermediate results to a pkl file after each source graph completes

Dump format (one .pkl per filtration x dataset x graph index):
    dumps/generate_1/{filtration}/{dataset}/graph_{idx:04d}.pkl

Usage:
    python generate_1_per_graph.py [--filtrations random vertex ...] [--datasets planar ego ...]
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
import time
import pickle
import concurrent.futures
from collections import Counter

import networkx as nx
import numpy as np

TOPOGEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if TOPOGEN_ROOT not in sys.path:
    sys.path.insert(0, TOPOGEN_ROOT)

TESTS_DIR = os.path.join(TOPOGEN_ROOT, "tests")
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

DATA_ROOT = os.path.join(TOPOGEN_ROOT, "data")
DUMP_ROOT = os.path.join(TOPOGEN_ROOT, "dumps", "generate_1")

# --- filtrations ---
from topo_gen.filtrations import random_filtration, vertex_filtration, degree_filtration
from topo_gen.persistence import persistence_diagrams, check_match

# --- generation algorithms ---
from topo_gen.random_filtration_pd_equivalent         import random_filtration_pd_equivalent
from topo_gen.vertex_filtration_pd_equivalent         import vertex_filtration_pd_equivalent
from topo_gen.vertex_filtration_pd_equivalent_bmin    import vertex_filtration_pd_equivalent_bmin
from topo_gen.degree_filtration_pd_equivalent         import degree_filtration_pd_equivalent
from topo_gen.degree_filtration_pd_equivalent_bmin    import degree_filtration_pd_equivalent_bmin

# --- checker functions from test modules ---
from test_random_filtration_pd_equivalent          import check_pd_roundtrip as rnd_check_ph,  check_edge_filtration as rnd_check_edge
from test_vertex_filtration_pd_equivalent          import check_pd_roundtrip as vtx_check_ph,  check_edge_filtration as vtx_check_edge
from test_vertex_filtration_pd_equivalent_bmin     import check_pd_roundtrip as vtxb_check_ph, check_edge_filtration as vtxb_check_edge
from test_degree_filtration_pd_equivalent          import check_pd_roundtrip as deg_check_ph,  check_edge_filtration as deg_check_edge,  check_degree_constraint as deg_check_deg
from test_degree_filtration_pd_equivalent_bmin     import check_pd_roundtrip as degb_check_ph, check_edge_filtration as degb_check_edge, check_degree_constraint as degb_check_deg

# --- utilities ---
from utils.dataset_utils import load_split, pyg_to_nx

# ---------------------------------------------------------------------------
# Config (overridable via CLI args)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    datasets        = ["planar", "sbm", "comm20", "enzymes"],
    split           = "test",
    batch_size      = 8,
    max_attempts    = 20000,
    n_workers       = 4,
    filtration_seed = 0,
    base_seed       = 0,
)

# ---------------------------------------------------------------------------
# Filtration specs
# ---------------------------------------------------------------------------

def _make_constraint_checker(ph_fn, edge_fn, deg_fn=None):
    def _check(r):
        try:
            edge_fn(r, "bench")
            if deg_fn is not None:
                deg_fn(r, "bench")
            return True
        except AssertionError:
            return False
    return _check


def build_filtration_specs(filtration_seed):
    return [
        dict(
            name               = "random",
            filtration_fn      = lambda G: random_filtration(G, seed=filtration_seed),
            gen_fn             = random_filtration_pd_equivalent,
            constraint_checker = _make_constraint_checker(rnd_check_ph,  rnd_check_edge),
        ),
        dict(
            name               = "vertex",
            filtration_fn      = lambda G: vertex_filtration(G, seed=filtration_seed),
            gen_fn             = vertex_filtration_pd_equivalent,
            constraint_checker = _make_constraint_checker(vtx_check_ph,  vtx_check_edge),
        ),
        dict(
            name               = "vertex_bmin",
            filtration_fn      = lambda G: vertex_filtration(G, seed=filtration_seed),
            gen_fn             = vertex_filtration_pd_equivalent_bmin,
            constraint_checker = _make_constraint_checker(vtxb_check_ph, vtxb_check_edge),
        ),
        dict(
            name               = "degree",
            filtration_fn      = degree_filtration,
            gen_fn             = degree_filtration_pd_equivalent,
            constraint_checker = _make_constraint_checker(deg_check_ph,  deg_check_edge, deg_check_deg),
        ),
        dict(
            name               = "degree_bmin",
            filtration_fn      = degree_filtration,
            gen_fn             = degree_filtration_pd_equivalent_bmin,
            constraint_checker = _make_constraint_checker(degb_check_ph, degb_check_edge, degb_check_deg),
        ),
    ]

# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------

def _call_one(gen_fn, pd0, pd1, seed):
    t0  = time.perf_counter()
    res = gen_fn(pd0, pd1, seed=seed)
    return res, seed, time.perf_counter() - t0


def generate_one(
    gen_fn, pd0, pd1, constraint_checker,
    batch_size=8, max_attempts=20000,
    n_workers=4, base_seed=0,
):
    failure_reasons = Counter()
    n_attempts      = 0
    seed            = base_seed
    t_start         = time.perf_counter()
    success         = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Pre-fill the pool with n_workers futures
        pending = {
            executor.submit(_call_one, gen_fn, pd0, pd1, seed + i): seed + i
            for i in range(n_workers)
        }
        seed += n_workers

        while pending and success is None and n_attempts < max_attempts:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                if success is not None:
                    continue
                res, s, elapsed = fut.result()
                n_attempts += 1
                if res.ok:
                    pd0_gen, pd1_gen = persistence_diagrams(res.node_time, res.edge_time)
                    ph_ok = check_match(pd0, pd1, pd0_gen, pd1_gen, verbose=False)
                    constraint_ok = constraint_checker(res)
                    success = dict(
                        H             = res.H,
                        node_time     = res.node_time,
                        edge_time     = res.edge_time,
                        seed          = s,
                        gen_time_s    = elapsed,
                        ph_ok         = ph_ok,
                        constraint_ok = constraint_ok,
                    )
                else:
                    failure_reasons[res.reason.name] += 1
                    # Refill: submit one new future to replace the completed one
                    if success is None and n_attempts < max_attempts:
                        new_fut = executor.submit(_call_one, gen_fn, pd0, pd1, seed)
                        pending.add(new_fut)
                        seed += 1

        # Cancel any remaining futures
        for fut in pending:
            fut.cancel()

    total_time_s = time.perf_counter() - t_start
    status = "complete" if success is not None else "failed"
    return dict(
        success         = success,
        n_attempts      = n_attempts,
        total_time_s    = total_time_s,
        failure_reasons = dict(failure_reasons),
        status          = status,
    )

# ---------------------------------------------------------------------------
# Dump helper
# ---------------------------------------------------------------------------

def save_dump(dump_root, spec_name, dataset, graph_idx, payload):
    out_dir = os.path.join(dump_root, spec_name, dataset)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"graph_{graph_idx:04d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg):
    filtration_specs = build_filtration_specs(cfg.filtration_seed)

    specs    = [s for s in filtration_specs if s['name'] in cfg.filtrations]
    datasets = cfg.datasets

    grand_stats = {}

    for spec in specs:
        fname = spec['name']
        grand_stats[fname] = {}
        print(f"\n{'='*65}")
        print(f" Filtration: {fname}  |  target=1 success per graph")
        print(f"{'='*65}")

        for ds in datasets:
            try:
                graphs = load_split(DATA_ROOT, ds, cfg.split)
            except Exception as exc:
                print(f"  [{ds}] LOAD ERROR: {exc}")
                grand_stats[fname][ds] = []
                continue

            n_graphs   = len(graphs)
            ds_records = []
            grand_stats[fname][ds] = ds_records

            print(f"\n  Dataset: {ds}  ({n_graphs} graphs)")
            print(f"  {'idx':>5}  {'V':>4}  {'E':>5}  {'attempts':>9}  {'time(s)':>8}  status")
            print(f"  {'-'*50}")

            for idx in range(n_graphs):
                G = pyg_to_nx(graphs[idx])
                try:
                    nt, et, T = spec['filtration_fn'](G)
                except Exception as exc:
                    print(f"  [{ds}][{idx}] filtration error: {exc}")
                    continue

                pd0, pd1 = persistence_diagrams(nt, et)

                result = generate_one(
                    gen_fn             = spec['gen_fn'],
                    pd0                = pd0,
                    pd1                = pd1,
                    constraint_checker = spec['constraint_checker'],
                    batch_size         = cfg.batch_size,
                    max_attempts       = cfg.max_attempts,
                    n_workers          = cfg.n_workers,
                    base_seed          = cfg.base_seed,
                )

                succ = result['success']

                record = dict(
                    filtration     = fname,
                    dataset        = ds,
                    graph_idx      = idx,
                    ref_graph      = G,
                    ref_node_time  = nt,
                    ref_edge_time  = et,
                    pd0            = pd0,
                    pd1            = pd1,
                    success        = succ,
                    n_attempts     = result['n_attempts'],
                    total_time_s   = result['total_time_s'],
                    gen_time_s     = succ['gen_time_s'] if succ else float('nan'),
                    failure_reasons= result['failure_reasons'],
                    status         = result['status'],
                    # run config
                    cfg_batch_size      = cfg.batch_size,
                    cfg_max_attempts    = cfg.max_attempts,
                    cfg_n_workers       = cfg.n_workers,
                    cfg_filtration_seed = cfg.filtration_seed,
                    cfg_base_seed       = cfg.base_seed,
                    cfg_split           = cfg.split,
                )
                ds_records.append(record)

                save_dump(DUMP_ROOT, fname, ds, idx, record)

                print(f"  {idx:>5}  {G.number_of_nodes():>4}  {G.number_of_edges():>5}  "
                      f"{result['n_attempts']:>9}  "
                      f"{result['total_time_s']:>8.2f}  {result['status']}")

    print("\n\nAll done.")
    return grand_stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--filtrations", nargs="+",
                   default=["random", "vertex", "vertex_bmin", "degree", "degree_bmin"],
                   help="Which filtrations to run (default: all 5)")
    p.add_argument("--datasets", nargs="+",
                   default=DEFAULTS["datasets"],
                   help="Which datasets to run (default: all 4)")
    p.add_argument("--split",           default=DEFAULTS["split"])
    p.add_argument("--batch-size",      type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--max-attempts",    type=int, default=DEFAULTS["max_attempts"])
    p.add_argument("--n-workers",       type=int, default=DEFAULTS["n_workers"])
    p.add_argument("--filtration-seed", type=int, default=DEFAULTS["filtration_seed"])
    p.add_argument("--base-seed",       type=int, default=DEFAULTS["base_seed"])
    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
