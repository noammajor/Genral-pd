#!/bin/bash
#SBATCH --job-name=pd-gfn
#SBATCH --output=logs/gfn_%j.out
#SBATCH --error=logs/gfn_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
# NOTE: more MEMORY helps, more CORES does not -- we pin torch to a single
# thread on purpose (see below), so extra cores sit idle. Raise memory with
# sbatch --mem=64G ... (shell vars do NOT expand in #SBATCH lines; the
# command-line flag overrides the directive). Lower MICRO= if it still OOMs.
#
# CPU only. The tensors here are tiny (B x N x N with N <= 175) and the loop is
# Python-bound in the env, so a GPU buys little. What DOES matter is thread
# count: on a 40-core node torch defaulted to 40 threads and ran ~95,000x
# slower than single-threaded, because sync overhead dwarfs the arithmetic.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Train the PD-compliant graph GFlowNet with trajectory balance.
#
# The second argument is EPOCHS = full passes over the training targets.
# An epoch shuffles all targets and visits each exactly once.
#   comm20 :  64 train targets / batch 64 =  1 iteration per epoch
#   enzymes: 376 train targets / batch 64 =  6 iterations per epoch
#
#   sbatch scripts/triton_train.sh comm20  10000          # full
#   sbatch scripts/triton_train.sh comm20   2500          # 25%, base results
#   sbatch scripts/triton_train.sh enzymes  1700          # full
#   sbatch scripts/triton_train.sh enzymes   425          # 25%
#   SEED=1 sbatch scripts/triton_train.sh comm20 2500     # different seed
#   sbatch scripts/triton_train.sh comm20 10000 --constant-scorer  # ablation
#
# Measured: comm20 ~2.45 s/iteration at batch 32 on one CPU core.
# enzymes has mean |E| 63.5 vs comm20's 35.7, so expect ~1.8x that.

set -euo pipefail

DATASET="${1:-comm20}"
NEPOCH="${2:-10000}"
shift 2 2>/dev/null || true
EXTRA=("$@")

module load scicomp-python-env
source ~/venvs/topogfn/bin/activate

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs

# Output dir carries dataset, epoch count and seed, so a short "base results"
# run never collides with the full run.
SEED="${SEED:-0}"
OUT="runs/${DATASET}_ep${NEPOCH}_seed${SEED}_${SLURM_JOB_ID:-local}"
echo "host=$(hostname) dataset=$DATASET epochs=$NEPOCH out=$OUT"
echo "extra: ${EXTRA[*]:-<none>}"

srun python3 -m topo_gfn.train \
  --dataset "$DATASET" \
  --n-epochs "$NEPOCH" \
  --batch-size 64 \
  --num-emb 128 --num-layers 4 --rank 32 \
  --lr 1e-4 --lr-z 1e-3 \
  --threads 1 \
  --micro-batch "${MICRO:-8}" \
  --seed "$SEED" \
  --log-every 0 --ckpt-every 50 \
  --device cpu \
  --out "$OUT" \
  "${EXTRA[@]}"

echo "done -> $OUT"
