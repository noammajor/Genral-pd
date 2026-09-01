# Exact Topological Compliance: Generating Persistence-Equivalent Graphs — NeurIPS 2026 Anonymous Submission

This repository accompanies the paper **"Exact Topological Compliance: Generating Persistence-Equivalent Graphs"**. It contains the code, configurations, and scripts needed to reproduce the experiments reported in  the paper.

The submission contrasts two families of approaches for generating graphs whose persistent-homology (PH) signature matches a target persistence diagram (PD):

1. **A constraint-programming (CP-SAT) solver** that produces graphs whose degree-filtration PD is *exactly* the target PD.
2. **Topology-aware neural diffusion baselines** (ConStruct and DiGress, each with  the TAGG conditioning module) that approximate the target PD via a learned reverse-diffusion process.

All experiments are run on four standard graph-generation benchmarks: **comm20**, **planar**, **sbm**, and **enzymes**.

---

## Repository layout

```
.
├── topo_gen/               # Core algorithmic library (filtrations, PH, CP solvers)
├── baselines/ConStruct/    # Vendored ConStruct/DiGress diffusion backbones
├── configs/                # Hydra configuration tree (datasets, models, experiments)
├── scripts/                # CLI entry points for evaluation and preprocessing
├── train_construct.py      # Train/eval ConStruct (no topology conditioning)
├── train_construct_tagg.py # Train/eval ConStruct + TAGG (PDM loss + TAM bias)
├── train_digress.py        # Train/eval DiGress (no topology conditioning)
├── train_digress_tagg.py   # Train/eval DiGress + TAGG
├── utils/                  # Dataset / generation / visualization helpers
├── dump_data.py            # Dump benchmark splits into a flat pickle format
├── environment.yml         # Conda environment specification
└── README.md               # This file
```


## Installation

Create the Conda environment from the pinned spec:

```bash
conda env create -f environment.yml
conda activate topogen
```

The environment pulls in `pytorch`, `pytorch-lightning`, `torch-geometric`, `gudhi`, `ortools` (CP-SAT), `hydra-core`, and the standard scientific stack. A CUDA-enabled GPU is required to retrain the diffusion baselines; the CP-SAT solver runs on CPU only.
