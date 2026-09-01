#!/bin/bash
#SBATCH --job-name=pd-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.out
#SBATCH --array=0-15
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
# NOTE: no GPU needed -- this stage is pure numpy/networkx on CPU.
# Set your partition/account if Triton requires them:
##SBATCH --partition=batch
##SBATCH --account=<your-account>

# Step 1 falsification gate, as a Slurm array.
# Each task takes one shard of the (target x alpha_merge x alpha_cycle) grid.
#
#   sbatch scripts/triton_sweep.sh tree
#   sbatch scripts/triton_sweep.sh er
#   sbatch scripts/triton_sweep.sh dataset planar
#
# Merge the shards afterwards with scripts/merge_sweep.py

set -euo pipefail

FAMILY="${1:-tree}"
DATASET="${2:-planar}"

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

module load mamba 2>/dev/null || module load miniconda 2>/dev/null || true
source activate topogfn 2>/dev/null || conda activate topogfn

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs sweep_out

echo "host=$(hostname) family=$FAMILY shard=$SHARD/$NUM_SHARDS"
python -c "import numpy, networkx, gudhi; print('deps ok')"

EXTRA=()
if [ "$FAMILY" = "dataset" ]; then
  EXTRA=(--dataset "$DATASET")
  OUT="sweep_out/${FAMILY}_${DATASET}_${SHARD}.json"
else
  OUT="sweep_out/${FAMILY}_${SHARD}.json"
fi

srun python scripts/heuristic_sweep.py \
  --family "$FAMILY" "${EXTRA[@]}" \
  --sizes 20 40 60 \
  --alphas-merge 0 1 2 4 6 10 \
  --alphas-cycle 0 2 6 \
  --trials 200 \
  --n-targets 10 \
  --bmin \
  --shard "$SHARD" \
  --num-shards "$NUM_SHARDS" \
  --out "$OUT"

echo "done -> $OUT"
