#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/childsafeads/logs/%x-%j.out

# Usage: sbatch --job-name=csa-m2q-st1 job.sbatch st1
set -e
TASK=$1
[ -z "$TASK" ] && { echo "need task arg"; exit 1; }
set --  # clear positional args: sourcing conda activate inherits them (st1 was read as an env name)
ulimit -f unlimited 2>/dev/null || true
source ~/miniconda3/bin/activate
bash ~/childsafeads/train/m2q/setup_env.sh
conda activate csa-m2
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
mkdir -p ~/childsafeads/logs ~/childsafeads/preds
echo "[job] task=$TASK host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python ~/childsafeads/train/m2q/train_m2q.py --task "$TASK"
echo "[job] DONE $TASK"
