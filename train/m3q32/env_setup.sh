#!/bin/bash
# Idempotent csa-vllm env setup. Run on a COMPUTE node (login node has ulimit issues).
set -e
source ~/miniconda3/bin/activate
if ! conda env list | awk '{print $1}' | grep -qx csa-vllm; then
  echo "creating conda env csa-vllm"
  conda create -y -n csa-vllm python=3.11
fi
conda activate csa-vllm
python -c "import vllm, numpy" 2>/dev/null || pip install --no-cache-dir vllm numpy
python -c "import vllm; print('vllm', vllm.__version__)"
