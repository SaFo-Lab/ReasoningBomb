#!/bin/bash
# =============================================================================
# Step 1: Train Length Predictor
# =============================================================================
#
# This script trains the MLP predictor that estimates reasoning length
# from victim model hidden states.
#
# Requirements:
#   - 8 GPUs (for victim model inference with tensor parallelism)
#   - ~200GB disk space (for cached puzzles and hidden states)
#
# Time: ~4-8 hours depending on num_puzzles
#
# Usage:
#   bash scripts/1_train_predictor.sh [--reuse]
#
# =============================================================================

set -e

cd "$(dirname "$0")/.."

CONFIG_FILE="${CONFIG_FILE:-configs/default.yaml}"
REUSE_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reuse)
            REUSE_FLAG="--reuse"
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "Training Length Predictor"
echo "================================================================================"
echo ""
echo "Config: ${CONFIG_FILE}"
echo "Reuse cached data: $([ -n \"${REUSE_FLAG}\" ] && echo 'Yes' || echo 'No')"
echo ""

# Run training
python -m src.predictor.train \
    --config ${CONFIG_FILE} \
    ${REUSE_FLAG}

echo ""
echo "================================================================================"
echo "Predictor training complete!"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  - outputs/predictor/mlp_predictor.pt      (trained model)"
echo "  - outputs/predictor/puzzles_with_lengths.json"
echo "  - outputs/predictor/hidden_states.pt"
echo ""
echo "Next step: bash scripts/2_collect_warmstart.sh"

