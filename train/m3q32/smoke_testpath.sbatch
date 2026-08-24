#!/bin/bash
#SBATCH --job-name=csa-m3-testpath
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --gpus=a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/childsafeads/logs/%x-%j.out
#SBATCH --propagate=NONE
#SBATCH --exclude=xgpj0
# Test-path smoke: Qwen3-4B, --splits test --limit 8 against the smoketest repo's
# fake unlabeled test.jsonl. Proves run_infer's new split plumbing + verify_npz's
# split:count form before the real 32B test job.
set -eo pipefail
echo "[job] host=$(hostname) ulimit-v=$(ulimit -v)"
if ! ~/miniconda3/bin/python -V >/dev/null 2>&1; then
  echo "[job] FATAL: ~/miniconda3/bin/python does not exec on $(hostname)"; exit 17
fi
cd ~/childsafeads
SCRATCH=/tmp/csa-${SLURM_JOB_ID:-manual}
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
export HF_HOME="$SCRATCH/hf"
export XDG_CACHE_HOME="$SCRATCH/xdg"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/inductor"
export TRITON_CACHE_DIR="$SCRATCH/triton"
export VLLM_CACHE_ROOT="$SCRATCH/vllm"
SREPO=~/childsafeads_smoketest
[ -f "$SREPO/data/test.jsonl" ] || { echo "[job] FATAL: $SREPO/data/test.jsonl missing"; exit 18; }
bash train/m3q32/env_setup.sh
source ~/miniconda3/bin/activate csa-vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
SMOKE_PREDS="$SREPO/preds_m3smoke"
SMOKE_CKPT="$SREPO/ckpt_m3smoke"
rm -rf "$SMOKE_PREDS" "$SMOKE_CKPT"
python train/m3q32/run_infer.py \
  --model Qwen/Qwen3-4B \
  --splits test --limit 8 --chunk 8 \
  --member m3q32smoke \
  --data-dir "$SREPO/data" \
  --preds-dir "$SMOKE_PREDS" \
  --ckpt-dir "$SMOKE_CKPT" \
  --gpu-mem-util 0.85
python train/m3q32/verify_npz.py "$SMOKE_PREDS" m3q32smoke test:8
echo "SMOKE-TESTPATH PASS"
