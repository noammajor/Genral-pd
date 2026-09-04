"""Distribution of embedding similarity over completed rollouts.

    python3 scripts/probe_sim.py [pool|match|ged] [ckpt_dir] [sinkhorn_temp]

Samples 6 passes over the comm20 train targets with the pretrained policy,
computes sim(generated, reference) for every completed rollout under the given
aggregation, and prints the spread plus the lambda that turns the p5->p95 sim
gap into a ~3 nat reward preference.  Run from the repo root.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from topo_gfn.env import TopoEnv
from topo_gfn.gfn import TopoSampler
from topo_gfn.policy import TopoGFN, set_feature_mode
from topo_gfn.similarity import EmbedSim
from topo_gfn.train import load_targets

agg = sys.argv[1] if len(sys.argv) > 1 else "match"
run = sys.argv[2] if len(sys.argv) > 2 else "runs/comm20_ep2500_seed0_20026585"
temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

torch.set_num_threads(1)
torch.manual_seed(0)
ck = torch.load(f"{run}/ckpt.pt", map_location="cpu", weights_only=False)
ta = ck["args"]
set_feature_mode(ta.get("features", "basic"))
m = TopoGFN(num_emb=ta["num_emb"], num_layers=ta["num_layers"],
            num_mlp_layers=ta["num_mlp_layers"], rank=ta["rank"])
m.load_state_dict(ck["model"])
m.eval()

train = load_targets(ta["dataset"], "train", "data")
sim = EmbedSim(m.encoder, agg=agg, source="live", sinkhorn_temp=temp)
sampler = TopoSampler(m)

sims = []
for rep in range(6):
    envs = [TopoEnv(s) for s, _ in train]
    for (s, A), t in zip(train, sampler.sample(envs)):
        if t.completed:
            n = s.num_nodes
            adj = np.asarray(t.terminal.adj[:n, :n], dtype=np.float64)
            sims.append(sim.sim(s, adj, A))

s = np.array(sims)
p5, p95 = np.percentile(s, 5), np.percentile(s, 95)
print(f"agg={agg}{f' temp={temp}' if agg == 'ged' else ''} n={len(s)} "
      f"mean={s.mean():.4f} std={s.std():.4f} "
      f"min={s.min():.4f} p5={p5:.4f} p95={p95:.4f} max={s.max():.4f}")
print(f"suggested lambda (3 nats over p5->p95): {3.0 / max(1e-6, p95 - p5):.0f}")
