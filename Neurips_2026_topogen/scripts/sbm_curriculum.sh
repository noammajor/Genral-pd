#!/bin/bash
#SBATCH --job-name=sbm-curriculum
#SBATCH --output=logs/curriculum_%j.out
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#
# Constraint-temperature curriculum for SBM.
#
# SBM never completes under the hard mask: both arms sat at exactly 0.00
# completion for 16+ hours, because a fresh policy essentially never finishes a
# ~512-edge trajectory by chance, so there is no reward signal to learn from.
#
# Soft constraints remove that cliff -- a soft rollout ALWAYS terminates, so
# every trajectory carries signal -- and the violation penalty T controls how
# hard compliance is pushed.  So walk T up instead of starting at infinity:
#
#   T=1   loose: learn to place edges at all, dense reward
#   T=5   tighten
#   T=10  tighten again
#   T=inf hard mask restored (no --violation-penalty)
#
# Each stage warm-starts from the previous stage's checkpoint.  Run as one job
# so stage k+1 can name stage k's output directory.
set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load scicomp-python-env
source ~/venvs/topogfn/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

EPOCHS="${EPOCHS:-40}"
BASE="runs/sbm_curriculum_${SLURM_JOB_ID:-local}"
COMMON="--dataset sbm --batch-size 64 --micro-batch 4 --threads 1 --device cpu
        --num-emb 128 --num-layers 4 --rank 32 --lr 1e-4 --lr-z 1e-3
        --log-every 0 --ckpt-every 10"

prev=""
for stage in 1 5 10 inf; do
  out="${BASE}/T${stage}"
  mkdir -p "$out"
  init=""
  [ -n "$prev" ] && init="--init-from $prev/ckpt.pt"
  if [ "$stage" = "inf" ]; then
    pen=""            # hard mask: no violation penalty at all
  else
    pen="--violation-penalty $stage"
  fi
  echo "=============== stage T=$stage -> $out ${init:+(from $prev)}"
  srun python3 -m topo_gfn.train $COMMON --n-epochs "$EPOCHS" \
       --out "$out" $init $pen || { echo "stage T=$stage FAILED"; exit 1; }
  prev="$out"
done
echo "curriculum done -> $BASE"
