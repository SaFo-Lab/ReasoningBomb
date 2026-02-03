#!/bin/bash
# =============================================================================
# Setup Script for Reasoning Bomb
# =============================================================================
#
# This script sets up the environment for running the adversarial puzzle
# generation pipeline.
#
# Usage:
#   bash setup.sh
#
# =============================================================================

set -e

echo "================================================================================"
echo "Setting up Reasoning Bomb"
echo "================================================================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: ${PYTHON_VERSION}"

if [[ "${PYTHON_VERSION}" < "3.10" ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Warning: CUDA not found. GPU training will not work."
fi

# Install dependencies
echo ""
echo "Installing dependencies..."

# Core PyTorch (adjust CUDA version as needed)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# verl framework
echo ""
echo "Installing verl..."
pip install verl

# Other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Optional: Flash Attention (uncomment if needed)
# echo ""
# echo "Installing Flash Attention 2..."
# pip install flash-attn --no-build-isolation

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/{predictor,warmstart,checkpoints,logs,data}

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import transformers
import vllm
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
"

echo ""
echo "================================================================================"
echo "Setup complete!"
echo "================================================================================"
echo ""
echo "Quick Start:"
echo "  1. Edit configs/default.yaml with your model paths"
echo "  2. Run: bash scripts/1_train_predictor.sh"
echo "  3. Run: bash scripts/4_train_grpo.sh"
echo ""

