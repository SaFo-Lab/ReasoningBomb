#!/bin/bash
# =============================================================================
# Step 3: Train SFT Warm Start Models (Optional)
# =============================================================================
#
# Fine-tunes the attacker model on collected warm start data.
# Creates separate models for each puzzle length category.
#
# Requirements:
#   - 2+ GPUs (for model fine-tuning with FSDP)
#
# Time: ~2-4 hours per category
#
# Usage:
#   bash scripts/3_train_sft.sh [--category length_128]
#
# =============================================================================

set -e

cd "$(dirname "$0")/.."

CONFIG_FILE="${CONFIG_FILE:-configs/default.yaml}"
CATEGORY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --category)
            CATEGORY="--category $2"
            shift 2
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
echo "Training SFT Warm Start Models"
echo "================================================================================"
echo ""
echo "Config: ${CONFIG_FILE}"
[ -n "${CATEGORY}" ] && echo "Category: ${CATEGORY}"
echo ""

# Check for dataset
DATASET="outputs/warmstart/warmstart_dataset.json"
if [ ! -f "${DATASET}" ]; then
    echo "Error: Dataset not found: ${DATASET}"
    echo "Run: bash scripts/2_collect_warmstart.sh first"
    exit 1
fi

# Run training
python -m src.warmstart.train \
    --config ${CONFIG_FILE} \
    ${CATEGORY}

echo ""
echo "================================================================================"
echo "SFT training complete!"
echo "================================================================================"
echo ""
echo "Output models:"
echo "  - outputs/warmstart/sft_length_128/"
echo "  - outputs/warmstart/sft_length_256/"
echo "  - outputs/warmstart/sft_length_512/"
echo ""
echo "Next step: bash scripts/4_train_grpo.sh"

