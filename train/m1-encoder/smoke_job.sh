#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=00:40:00
#SBATCH --gpus=a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=e1506804@comp.nus.edu.sg
#SBATCH --output=/home/e/e1506804/childsafeads/logs/%x-%j.out
#SBATCH --propagate=NONE
#SBATCH --exclude=xgpj0
# GPU smoke: real models, --smoke (2 folds x 1 epoch x 4 steps) for both archs.
set -eo pipefail
echo "[job] host=$(hostname) ulimit-v=$(ulimit -v)"
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
LANE=~/childsafeads/train/m1-encoder
mkdir -p ~/childsafeads/logs ~/childsafeads/preds ~/childsafeads/preds_smoke

flock ~/.csa-m1-env.lock bash "$LANE/setup_env.sh"
source ~/miniconda3/bin/activate csa-m1
cd "$LANE"
python train.py --arch deberta --seed 99 --repo ~/childsafeads --smoke --preds ~/childsafeads/preds_smoke
python train.py --arch modernbert --seed 99 --repo ~/childsafeads --smoke --batch 8 --accum 2 --preds ~/childsafeads/preds_smoke
python - <<'EOF'
import numpy as np
for m in ["m1d","m1m"]:
    d = np.load(f"/home/e/e1506804/childsafeads/preds_smoke/{m}-s99_st3_dev.npz", allow_pickle=True)
    p = d["probs"]
    nan = int(np.sum(~np.isfinite(p)))
    print(f"[verify] {m}: shape={p.shape} folds={d['dev_folds'].shape[0]} "
          f"nonfinite={nan} min={np.nanmin(p):.4f} max={np.nanmax(p):.4f}")
    assert p.shape == (504, 8) and d["dev_folds"].shape[0] == 2, f"{m}: bad shape"
    assert nan == 0, f"{m}: {nan} non-finite probabilities"
    assert p.min() >= 0 and p.max() <= 1, f"{m}: probs out of range"
    print(m, "dev st3 OK")
EOF
echo "SMOKE-GPU PASS"
