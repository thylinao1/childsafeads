#!/bin/bash
# Idempotent conda env setup for the m2q lane. Safe to re-run.
set -e
source ~/miniconda3/bin/activate
if ! conda env list | awk '{print $1}' | grep -qx csa-m2; then
  conda create -y -n csa-m2 -c conda-forge python=3.11
fi
conda activate csa-m2
python -c "import torch, transformers, peft, accelerate, sklearn, numpy" 2>/dev/null && { echo "csa-m2 env ready"; exit 0; }
pip install --no-input torch transformers peft accelerate scikit-learn numpy
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
echo "csa-m2 env ready"
