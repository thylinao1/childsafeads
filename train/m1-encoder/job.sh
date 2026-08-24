#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --time=12:00:00
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/childsafeads/logs/%x-%j.out
#SBATCH --propagate=NONE
#SBATCH --exclude=xgpj0
# Usage: sbatch -J csa-m1d-s42 job.sh deberta 42
# --exclude=xgpj0: node ENOEXECs ~/miniconda3 python entrypoints (2026-08-10, jobs 723422/723429/723430)
set -eo pipefail
echo "[job] host=$(hostname) ulimit-v=$(ulimit -v)"
# fail fast with a clear message if conda entrypoints are broken on this node
if ! ~/miniconda3/bin/python -V >/dev/null 2>&1; then
  echo "[job] FATAL: ~/miniconda3/bin/python does not exec on $(hostname)"; exit 17
fi
# node-local scratch for all volatile caches (home is at its ~500GB quota edge)
SCRATCH=/tmp/csa-${SLURM_JOB_ID:-manual}
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_HOME="$SCRATCH/hf"
export XDG_CACHE_HOME="$SCRATCH/xdg"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor"
export TRITON_CACHE_DIR="$SCRATCH/triton"
export VLLM_CACHE_ROOT="$SCRATCH/vllm"
export TOKENIZERS_PARALLELISM=false
ARCH=$1
SEED=$2
LANE=~/childsafeads/train/m1-encoder
mkdir -p ~/childsafeads/logs ~/childsafeads/preds

# CSA_WAIT_TEST=<epoch>: the job may be queued before the eval test set is
# released; if it starts early, poll for data/test.jsonl until the deadline
# so the retrain always includes the test predict. No-op when unset.
if [ -n "$CSA_WAIT_TEST" ]; then
  while [ ! -f ~/childsafeads/data/test.jsonl ] && [ "$(date +%s)" -lt "$CSA_WAIT_TEST" ]; do
    echo "[job] waiting for data/test.jsonl (deadline $CSA_WAIT_TEST)"; sleep 120
  done
  [ -f ~/childsafeads/data/test.jsonl ] || { echo "[job] FATAL: test.jsonl never arrived by deadline"; exit 19; }
fi

flock ~/.csa-m1-env.lock bash "$LANE/setup_env.sh"
source ~/miniconda3/bin/activate csa-m1

if [ "$ARCH" = "modernbert" ]; then
  EXTRA="--batch 8 --accum 2"
else
  EXTRA="--batch 16 --accum 1"
fi

cd "$LANE"
python train.py --arch "$ARCH" --seed "$SEED" --repo ~/childsafeads $EXTRA
python merge_seeds.py ~/childsafeads/preds
echo "[job] done arch=$ARCH seed=$SEED"
