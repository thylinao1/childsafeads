#!/bin/bash
# Idempotent conda env setup for the m1 lane. Run under flock so concurrent jobs serialize.
set -e
source ~/miniconda3/bin/activate
if ! conda env list | awk '{print $1}' | grep -qx "csa-m1"; then
  echo "[env] creating csa-m1"
  conda create -y -n csa-m1 python=3.11
fi
conda activate csa-m1
python - <<'EOF' 2>/dev/null && exit 0
import torch, transformers, sklearn, numpy, sentencepiece  # noqa
print("[env] csa-m1 already complete:", torch.__version__, transformers.__version__)
EOF
echo "[env] installing packages into csa-m1"
pip install --no-input torch transformers accelerate scikit-learn numpy sentencepiece protobuf tiktoken
