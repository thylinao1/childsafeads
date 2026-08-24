#!/bin/bash
#SBATCH --job-name=csa-m3-think
#SBATCH --partition=gpu
#SBATCH --gpus=h200-141:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/childsafeads/logs/%x-%j.out
#SBATCH --propagate=NONE

# Test-split inference (503 rows). If a100-80 pends long on the gpu partition,
# fall back to --gpus=h100-96:1.
set -e
cd ~/childsafeads
# CSA_WAIT_TEST=<epoch>: the job may be submitted before the test set is
# released; poll for the file until the deadline instead of aborting.
if [ ! -f data/test.jsonl ] && [ -n "$CSA_WAIT_TEST" ]; then
  while [ ! -f data/test.jsonl ] && [ "$(date +%s)" -lt "$CSA_WAIT_TEST" ]; do
    echo "waiting for data/test.jsonl (deadline $CSA_WAIT_TEST)"; sleep 120
  done
fi
if [ ! -f data/test.jsonl ]; then
  echo "data/test.jsonl not present - aborting"; exit 1
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
echo "host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID"
nvidia-smi -L
bash train/m3q32/env_setup.sh
source ~/miniconda3/bin/activate csa-vllm
# flashinfer JIT fails on these nodes (ninja build error, smoke 2026-08-10):
# force native torch sampler + FlashAttention backend, both precompiled.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
python train/m3q32/run_infer.py \
  --model Qwen/Qwen3-32B \
  --variants C \
  --member m3q32TH --thinking --tasks st3 \
  --splits oof,dev,test \
  --chunk 250 \
  --preds-dir ~/childsafeads/preds \
  --ckpt-dir ~/childsafeads/train/m3q32/ckpt \
  --tp 1
source ~/miniconda3/bin/activate csa-vllm
python train/m3q32/verify_npz.py ~/childsafeads/preds m3q32TH oof:2353 dev:504 test:503
echo "JOB DONE $(date -Is)"
