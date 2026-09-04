"""DiGress-style metrics for a trained topo_gfn checkpoint.

    python3 scripts/digress_metrics.py runs/comm20_full [n_per_target]

Metric definitions are transcribed from the vendored SPECTRE/DiGress code
(see scripts/graph_metrics.py for why it cannot be imported directly here), so
the numbers are comparable to the tables in those papers.

Reports the two families that apply to abstract graphs:

  1. V.U.N.  fraction of generated graphs that are Unique (distinct isomorphism
     class within the sample), Novel (not isomorphic to any TRAINING graph) and
     Valid.  Validity is dataset-specific, as in DiGress:
        planar   : planar and connected
        sbm      : not statistically distinguishable from the generating SBM
        comm20 / : DiGress has no predicate for these, so we use THIS paper's
        enzymes    predicate -- the graph realises the target persistence
                   diagram exactly.  Reported separately as "PD-valid".

  2. MMD ratios  r = MMD(gen, test)^2 / MMD(train, test)^2 over degree,
     clustering, orbit (size-4 substructures, via the compiled orca binary) and
     spectral distributions.  r = 1 means the generated set is as close to the
     test set as the training set is; that calibration is why DiGress reports
     ratios rather than raw MMD.

Not reported: NLL/ELBO (family 3).  A GFlowNet has no ELBO -- log Z is learned
as part of trajectory balance, not a variational bound on the data likelihood,
so the number would not be comparable to DiGress's 69.6 on QM9.
"""
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_metrics import (  # noqa: E402
    clustering_stats,
    degree_stats,
    is_planar_graph,
    orbit_stats_all,
    spectral_stats,
    vun,
)
from topo_gfn.env import TopoEnv  # noqa: E402
from topo_gfn.eval import pd_of_adj  # noqa: E402
from topo_gfn.gfn import TopoSampler  # noqa: E402
from topo_gfn.policy import TopoGFN, set_feature_mode  # noqa: E402
from topo_gfn.train import load_targets  # noqa: E402


def to_nx(adj):
    n = adj.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((int(u), int(v)) for u, v in zip(*np.nonzero(np.triu(adj))))
    return G


def main():
    run = Path(sys.argv[1])
    per_target = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    ta = ck["args"]
    set_feature_mode(ta.get("features", "basic"))
    soft = ta.get("violation_penalty") is not None
    ds = ta["dataset"]
    model = TopoGFN(num_emb=ta["num_emb"], num_layers=ta["num_layers"],
                    num_mlp_layers=ta["num_mlp_layers"], rank=ta["rank"])
    model.load_state_dict(ck["model"])
    model.eval()
    torch.set_num_threads(1)
    torch.manual_seed(0)

    train = load_targets(ds, "train", str(ROOT / "data"))
    test = load_targets(ds, "test", str(ROOT / "data"))
    train_nx = [to_nx(A) for _, A in train]
    test_nx = [to_nx(A) for _, A in test]
    tgt_pd = [pd_of_adj(A) for _, A in test]

    print(f"[{run.name}] dataset={ds} soft={soft} "
          f"| {len(test)} test PDs x {per_target} samples", flush=True)

    sam = TopoSampler(model, soft=soft)
    gen, pd_ok = [], 0
    for _ in range(per_target):
        idx = list(range(len(test)))
        for ti, t in zip(idx, sam.sample([TopoEnv(test[ti][0]) for ti in idx])):
            if not t.completed:
                continue
            n = test[ti][0].num_nodes
            A = np.asarray(t.terminal.adj[:n, :n], dtype=np.float64)
            gen.append(to_nx(A))
            pd_ok += (pd_of_adj(A) == tgt_pd[ti])
    if not gen:
        print("no graphs generated"); return
    print(f"generated {len(gen)} graphs\n", flush=True)

    # ---- family 1: V.U.N. --------------------------------------------------
    if ds == "planar":
        valid_fn, vname = is_planar_graph, "planar+connected"
    else:
        # sbm needs graph_tool's blockmodel fit, which SIGILLs on this cluster
        valid_fn, vname = None, None

    frac_u, frac_un, frac_unv = vun(gen, train_nx, valid_fn)
    print("1. PER-SAMPLE VALIDITY (DiGress family 1)")
    print(f"   unique (distinct iso-class in sample) : {frac_u:.3f}")
    print(f"   unique & novel (not in train)         : {frac_un:.3f}")

    if valid_fn:
        print(f"   V.U.N. [{vname}]                  : {frac_unv:.3f}")
        print(f"   validity alone                        : "
              f"{np.mean([valid_fn(g) for g in gen]):.3f}")
    print(f"   PD-valid (this paper's predicate)     : {pd_ok/len(gen):.3f}")

    # ---- family 2: MMD ratios ---------------------------------------------
    print("\n2. DISTRIBUTIONAL SIMILARITY (DiGress family 2)")
    print(f"   {'metric':>11} {'MMD(gen,test)':>14} {'MMD(train,test)':>16} "
          f"{'ratio':>8}")
    for name, fn in [("degree", degree_stats), ("clustering", clustering_stats),
                     ("orbit", orbit_stats_all), ("spectral", spectral_stats)]:
        try:
            m_gen = fn(test_nx, gen)
            m_tr = fn(test_nx, train_nx)
            r = m_gen / m_tr if m_tr > 1e-12 else float("nan")
            print(f"   {name:>11} {m_gen:>14.5f} {m_tr:>16.5f} {r:>8.2f}")
        except Exception as e:
            print(f"   {name:>11}  failed: {type(e).__name__}: {e}")

    print("\n3. LIKELIHOOD: not applicable (a GFlowNet has no ELBO; log Z is a "
          "trajectory-balance normaliser, not a likelihood bound)")


if __name__ == "__main__":
    main()
