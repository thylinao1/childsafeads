#!/bin/bash
# srun target: node-local caches + the M2 smoke. No nested-quoting hazards.
set -e
S=/tmp/csa-smoke-$$
mkdir -p "$S"
trap 'rm -rf "$S"' EXIT
export HF_HOME="$S/hf" XDG_CACHE_HOME="$S/xdg"
source ~/miniconda3/bin/activate csa-m2
python ~/childsafeads/train/m2q/smoke.py
