#!/bin/bash
# =============================================================================
# Step 4: Train with GRPO
# =============================================================================
#
# Main GRPO training loop for adversarial puzzle generation.
#
# Requirements:
#   - 8 GPUs recommended (2 predictor + 2 embedding + 4 training)
#   - Trained MLP predictor from Step 1
#
# Time: ~24-48 hours for 250 epochs
#
# Usage:
#   bash scripts/4_train_grpo.sh [OPTIONS]
#
# Options:
#   --puzzle_max_len N      Maximum puzzle length (default: 128)
#   --diversity_weight W    Diversity reward weight (default: 0.5)
#   --disable_diversity     Disable diversity reward
#
# Examples:
#   bash scripts/4_train_grpo.sh
#   bash scripts/4_train_grpo.sh --puzzle_max_len 256
#   bash scripts/4_train_grpo.sh --disable_diversity
#
# =============================================================================

set -e

cd "$(dirname "$0")/.."

# Check for predictor
PREDICTOR_MODEL="outputs/predictor/mlp_predictor.pt"
if [ ! -f "${PREDICTOR_MODEL}" ]; then
    echo "Error: Predictor model not found: ${PREDICTOR_MODEL}"
    echo "Run: bash scripts/1_train_predictor.sh first"
    exit 1
fi

echo "================================================================================"
echo "GRPO Training for Adversarial Puzzle Generation"
echo "================================================================================"
echo ""

# Forward all arguments to the training script
bash src/training/run.sh "$@"

