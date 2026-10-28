#!/usr/bin/env bash
# 在 pi400 上建一个只用于「模型转换」的 Python 环境（torch 只在这里用，转完就删）。
set -euo pipefail

VENV="${VENV:-$HOME/venv-conv}"
PIP_MIRROR="${PIP_MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --no-cache-dir --upgrade pip -i "$PIP_MIRROR"
"$VENV/bin/pip" install --no-cache-dir --index-url "$TORCH_INDEX" "torch==2.9.1+cpu"
"$VENV/bin/pip" install --no-cache-dir -i "$PIP_MIRROR" transformers sentencepiece safetensors gguf numpy

"$VENV/bin/python" - <<'PY'
import torch, transformers, gguf
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("gguf", gguf.__version__)
PY
df -h / | tail -1
echo "venv ready"
