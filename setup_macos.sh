#!/bin/bash
# Setup script for macOS Apple Silicon (M1/M2/M3/M4)
# Usage: bash setup_macos.sh [env_name]
set -euo pipefail

echo "=== CosyVoice macOS Apple Silicon Setup ==="

ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "Warning: This script is designed for Apple Silicon (arm64), detected: $ARCH"
fi

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniforge or Miniconda first."
    echo "  brew install miniforge"
    exit 1
fi

ENV_NAME="${1:-cosyvoice}"

echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing pynini via conda-forge..."
conda install -c conda-forge pynini==2.1.5 -y

echo "Installing PyTorch with MPS support..."
pip install torch torchaudio

echo "Installing remaining dependencies..."
pip install -r requirements.txt

if [ -f .gitmodules ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate $ENV_NAME"
echo ""
echo "Verify MPS device:"
echo "  python -c \"import torch; print('MPS available:', torch.backends.mps.is_available())\""
echo ""
echo "Notes:"
echo "  - TensorRT is not available on Apple Silicon (CUDA-only)"
echo "  - vLLM is not available on Apple Silicon (CUDA-only)"
echo "  - Training (DeepSpeed/DDP) is not supported on MPS"
echo "  - Inference uses MPS acceleration (faster than CPU)"
