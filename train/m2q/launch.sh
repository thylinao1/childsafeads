#!/bin/bash
# One-shot m2q lane launcher. Run from the local machine; requires working `ssh soc`.
# Does: rsync lane -> cluster srun smoke (a100-40, 25 min) -> submit 3 sbatch jobs.
set -e
echo "== rsync lane =="
ssh -o BatchMode=yes soc 'mkdir -p ~/childsafeads/train/m2q ~/childsafeads/logs ~/childsafeads/preds'
rsync -a ~/Developer/childsafeads/train/m2q/ soc:~/childsafeads/train/m2q/

echo "== job-count guard =="
N=$(ssh -o BatchMode=yes soc 'squeue --me -h | wc -l')
echo "jobs in system: $N"
[ "$N" -gt 26 ] && { echo "too many jobs in system, aborting"; exit 1; }

echo "== env (idempotent, on compute node; login node has 1GB vmem ulimit) =="
ssh -o BatchMode=yes soc 'srun --partition=normal --time=00:40:00 --mem=24G --cpus-per-task=4 --job-name=csa-m2q-env bash ~/childsafeads/train/m2q/setup_env.sh 2>&1 | tail -3'

echo "== smoke (srun gpu a100-40 25min): model+LoRA load, tokenize 8 rows, 1 train step =="
ssh -o BatchMode=yes soc 'srun --partition=gpu --gpus=a100-40 --time=00:25:00 --mem=48G --cpus-per-task=8 --job-name=csa-m2q-smoke bash ~/childsafeads/train/m2q/smoke_wrapper.sh 2>&1 | tail -15' | tee /tmp/m2q_smoke.out
grep -q "SMOKE PASS" /tmp/m2q_smoke.out || { echo "SMOKE FAILED - not submitting"; exit 1; }

echo "== submit 3 task jobs =="
for T in st1 st2 st3; do
  ssh -o BatchMode=yes soc "cd ~/childsafeads/train/m2q && sbatch --job-name=csa-m2q-$T job.sbatch $T"
done
ssh -o BatchMode=yes soc 'squeue --me -o "%.10i %.15j %.8T %.10M"'
echo "== done. Expect preds at soc:~/childsafeads/preds/m2q_{st1,st2,st3}_{oof,dev}.npz =="
