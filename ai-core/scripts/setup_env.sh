#!/usr/bin/env bash
set -e

echo "=> Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

echo "=> Initializing Python 3.11 virtual environment..."
uv venv --python 3.11 .venv

echo "=> Syncing dependencies at lightning speed (via pyproject.toml)..."
uv sync

echo "=> Verifying PyTorch CUDA environment..."
.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"

echo "=> Environment setup complete! Activate with: source .venv/bin/activate"
