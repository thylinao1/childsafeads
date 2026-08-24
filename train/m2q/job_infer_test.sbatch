#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/childsafeads/logs/%x-%j.out
#SBATCH --propagate=NONE
#SBATCH --exclude=xgpj0
# Usage: sbatch --job-name=csa-m2q-inf-st3 job_infer_test.sbatch st3
# Adapter-only test inference (~60-75 min): requires work/<task>/fold{0..4}_adapter
# from the adapter-saving training run AND data/test.jsonl in the repo.
set -eo pipefail
echo "[job] host=$(hostname) ulimit-v=$(ulimit -v)"
if ! ~/miniconda3/bin/python -V >/dev/null 2>&1; then
  echo "[job] FATAL: ~/miniconda3/bin/python does not exec on $(hostname)"; exit 17
fi
SCRATCH=/tmp/csa-${SLURM_JOB_ID:-manual}
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_HOME="$SCRATCH/hf"
export XDG_CACHE_HOME="$SCRATCH/xdg"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor"
export TRITON_CACHE_DIR="$SCRATCH/triton"
export TOKENIZERS_PARALLELISM=false
TASK=$1
[ -n "$TASK" ] || { echo "[job] FATAL: pass task (st1|st2|st3)"; exit 18; }
[ -f ~/childsafeads/data/test.jsonl ] || { echo "[job] FATAL: data/test.jsonl not present"; exit 18; }
source ~/miniconda3/bin/activate csa-m2
cd ~/childsafeads/train/m2q
python infer_test_from_adapters.py --task "$TASK" --repo ~/childsafeads
echo "[job] done task=$TASK"
