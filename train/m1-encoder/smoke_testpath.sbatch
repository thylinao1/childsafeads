#!/bin/bash
#SBATCH --job-name=csa-m1-testpath
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
# Test-path smoke: deberta --smoke against the smoketest repo (which carries a fake
# unlabeled data/test.jsonl). Proves the new test predict + npz write + seed merge
# before the full 2.5h retrains depend on it. Expects ~/childsafeads_smoketest
# to exist with data/{train,dev,test}.jsonl + eval/folds.json.
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
LANE=~/childsafeads/train/m1-encoder
SREPO=~/childsafeads_smoketest
[ -f "$SREPO/data/test.jsonl" ] || { echo "[job] FATAL: $SREPO/data/test.jsonl missing"; exit 18; }
mkdir -p ~/childsafeads/logs "$SREPO/preds"
rm -f "$SREPO"/preds/*.npz "$SREPO"/preds/*_meta.json

flock ~/.csa-m1-env.lock bash "$LANE/setup_env.sh"
source ~/miniconda3/bin/activate csa-m1
cd "$LANE"
python train.py --arch deberta --seed 99 --repo "$SREPO" --smoke --preds "$SREPO/preds"
python merge_seeds.py "$SREPO/preds"
python - <<'EOF'
import numpy as np, os
P = os.path.expanduser("~/childsafeads_smoketest/preds")
for tag in ["m1d-s99", "m1d"]:
    for t in ["st1", "st2", "st3"]:
        f = f"{P}/{tag}_{t}_test.npz"
        assert os.path.exists(f), f"missing {f}"
        z = np.load(f, allow_pickle=True)
        p = z["probs"]
        assert p.shape[0] == 504 and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1, (f, p.shape)
        assert p.std() > 1e-4, f"{f}: degenerate probs"
        print(f"[verify] {tag} {t} test: shape={p.shape} ok")
# oof/dev must also still exist (guard did not break the normal path)
for t in ["st1", "st2", "st3"]:
    for sp in ["oof", "dev"]:
        assert os.path.exists(f"{P}/m1d_{t}_{sp}.npz"), f"missing merged {t} {sp}"
print("all contracts ok")
EOF
echo "SMOKE-TESTPATH PASS"
