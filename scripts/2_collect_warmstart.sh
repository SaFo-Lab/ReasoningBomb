#!/bin/bash
# =============================================================================
# Step 2: Collect Warm Start Data (Optional)
# =============================================================================
#
# Collects high-quality puzzle examples for SFT warm start.
# This step is optional but recommended for better GRPO training stability.
#
# Requirements:
#   - 1 GPU (for puzzle generation)
#
# Time: ~1-2 hours
#
# Usage:
#   bash scripts/2_collect_warmstart.sh
#
# =============================================================================

set -e

cd "$(dirname "$0")/.."

CONFIG_FILE="${CONFIG_FILE:-configs/default.yaml}"

echo "================================================================================"
echo "Collecting Warm Start Data"
echo "================================================================================"
echo ""
echo "Config: ${CONFIG_FILE}"
echo ""

# Run collection
python -m src.warmstart.collect \
    --config ${CONFIG_FILE}

echo ""
echo "================================================================================"
echo "Data collection complete!"
echo "================================================================================"
echo ""
echo "Output files:"
echo "  - outputs/warmstart/warmstart_dataset.json"
echo ""
echo "Next step: bash scripts/3_train_sft.sh"

