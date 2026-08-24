#!/bin/bash
# Smoke: small model, 8 instances, full pipeline, contract assertions.
# Run under: srun --partition=gpu --gpus=a100-40:1 --time=00:30:00 ...
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
bash train/m3q32/env_setup.sh
source ~/miniconda3/bin/activate csa-vllm
# flashinfer JIT fails on these nodes (ninja build error, smoke 2026-08-10):
# force native torch sampler + FlashAttention backend, both precompiled.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
SMOKE_PREDS=~/childsafeads/train/m3q32/smoke_preds
SMOKE_CKPT=~/childsafeads/train/m3q32/smoke_ckpt
rm -rf "$SMOKE_PREDS" "$SMOKE_CKPT"
python train/m3q32/run_infer.py \
  --model Qwen/Qwen3-4B \
  --limit 8 --chunk 8 \
  --member m3q32smoke \
  --preds-dir "$SMOKE_PREDS" \
  --ckpt-dir "$SMOKE_CKPT" \
  --gpu-mem-util 0.85
python train/m3q32/verify_npz.py "$SMOKE_PREDS" m3q32smoke 8 8
echo "SMOKE GREEN"
