#!/bin/bash
#SBATCH --job-name=csa-m3q32
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=logs/%x-%j.out

# Submit from ~/childsafeads (so logs/ resolves). mkdir -p logs first.
set -e
cd ~/childsafeads
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
  --variants A,B \
  --tasks st1,st2,st3 \
  --chunk 250 \
  --preds-dir ~/childsafeads/preds \
  --ckpt-dir ~/childsafeads/train/m3q32/ckpt
source ~/miniconda3/bin/activate csa-vllm
python train/m3q32/verify_npz.py ~/childsafeads/preds m3q32 2353 504
echo "JOB DONE $(date -Is)"
