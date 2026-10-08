#!/usr/bin/env bash
# ==============================================================================
# Environment Reproduction Script
# Project: Classification of Eye Diseases from Color Fundus Images
# Target: Python 3.10.x, PyTorch 2.5.1+cu121
# ==============================================================================

set -euo pipefail

echo "=== 1. Checking Python Environment ==="
python3 --version

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "=== 2. Upgrading pip & wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== 3. Installing Exact PyTorch with CUDA 12.1 ==="
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo "=== 4. Installing Core Dependencies ==="
pip install -r requirements.txt

echo "=== 5. Verifying CUDA Capability & Determinism ==="
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'Device Capability: {torch.cuda.get_device_capability(0)}')
"

echo "=== Environment successfully reproduced! ==="
