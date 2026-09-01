#!/bin/bash
#SBATCH --job-name=pd-gfn
#SBATCH --output=logs/gfn_%j.out
#SBATCH --error=logs/gfn_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# CPU only. The tensors here are tiny (B x N x N with N <= 175) and the loop is
# Python-bound in the env, so a GPU buys little. What DOES matter is thread
# count: on a 40-core node torch defaulted to 40 threads and ran ~95,000x
# slower than single-threaded, because sync overhead dwarfs the arithmetic.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Train the PD-compliant graph GFlowNet with trajectory balance.
#
#   sbatch scripts/triton_train.sh comm20  10000
#   sbatch scripts/triton_train.sh enzymes 10000
#   sbatch scripts/triton_train.sh comm20  10000 --constant-scorer   # ablation
#
# Measured: comm20 ~2.45 s/iteration at batch 32 on one CPU core.
# enzymes has mean |E| 63.5 vs comm20's 35.7, so expect ~1.8x that.

set -euo pipefail

DATASET="${1:-comm20}"
NITER="${2:-10000}"
shift 2 2>/dev/null || true
EXTRA=("$@")

module load scicomp-python-env
source ~/venvs/topogfn/bin/activate

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs

OUT="runs/${DATASET}_${SLURM_JOB_ID:-local}"
echo "host=$(hostname) dataset=$DATASET iters=$NITER out=$OUT"
echo "extra: ${EXTRA[*]:-<none>}"

srun python3 -m topo_gfn.train \
  --dataset "$DATASET" \
  --n-iterations "$NITER" \
  --batch-size 64 \
  --num-emb 128 --num-layers 4 --rank 32 \
  --lr 1e-4 --lr-z 1e-3 \
  --threads 1 \
  --log-every 50 --ckpt-every 500 \
  --device cpu \
  --out "$OUT" \
  "${EXTRA[@]}"

echo "done -> $OUT"
