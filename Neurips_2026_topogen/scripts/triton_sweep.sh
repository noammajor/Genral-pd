#!/bin/bash
#SBATCH --job-name=pd-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.out
#SBATCH --array=0-15
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# CPU only -- this stage is pure numpy/networkx, no GPU.
# Triton partitions: batch-csl batch-skl batch-hsw batch-bdw(default) batch-milan hugemem
# Left unset so Slurm picks the default; uncomment to pin one.
##SBATCH --partition=batch-milan

# Step 1 falsification gate, as a Slurm array.
# Each task takes one shard of the (target x alpha_merge x alpha_cycle) grid.
#
#   sbatch scripts/triton_sweep.sh tree
#   sbatch scripts/triton_sweep.sh er
#   sbatch scripts/triton_sweep.sh dataset comm20
#   sbatch scripts/triton_sweep.sh dataset planar
#
# Merge shards afterwards with scripts/merge_sweep.py

set -euo pipefail

FAMILY="${1:-tree}"
DATASET="${2:-planar}"

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

# Verified environment on Triton (2026-09):
#   scicomp-python-env gives numpy 2.2.6 / networkx 3.5 / scipy 1.16.1 / torch 2.8.0+cu128
#   ~/venvs/topogfn adds gudhi 3.13.0 and torch_geometric 2.8.0 on top of it.
module load scicomp-python-env
source ~/venvs/topogfn/bin/activate

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs sweep_out

echo "host=$(hostname) family=$FAMILY dataset=$DATASET shard=$SHARD/$NUM_SHARDS"
python3 -c "import numpy, networkx, gudhi; print('deps ok')"

if [ "$FAMILY" = "dataset" ]; then
  EXTRA=(--dataset "$DATASET" --data-root "$PWD/data")
  OUT="sweep_out/${FAMILY}_${DATASET}_${SHARD}.json"
  # real benchmarks are big (planar n=64, sbm n<=174): fewer targets, fewer trials
  TRIALS=100
  NTARGETS=20
else
  EXTRA=(--sizes 20 40 60)
  OUT="sweep_out/${FAMILY}_${SHARD}.json"
  TRIALS=200
  NTARGETS=10
fi

srun python3 scripts/heuristic_sweep.py \
  --family "$FAMILY" "${EXTRA[@]}" \
  --alphas-merge 0 1 2 4 6 10 \
  --alphas-cycle 0 2 6 \
  --trials "$TRIALS" \
  --n-targets "$NTARGETS" \
  --bmin \
  --shard "$SHARD" \
  --num-shards "$NUM_SHARDS" \
  --out "$OUT"

echo "done -> $OUT"
