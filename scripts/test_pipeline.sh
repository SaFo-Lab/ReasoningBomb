#!/bin/bash
# =============================================================================
# Quick Pipeline Test
# =============================================================================
#
# Runs through all pipeline steps with tiny settings to verify code works.
# Uses tiny_test.yaml config with minimal models and data.
#
# Usage:
#   bash scripts/test_pipeline.sh
#
# Expected time: ~5-10 minutes on a single GPU
#
# =============================================================================

set -e

cd "$(dirname "$0")/.."

export CONFIG_FILE="configs/tiny_test.yaml"

echo "================================================================================"
echo "ReasoningBomb - Pipeline Test with Tiny Configuration"
echo "================================================================================"
echo ""
echo "Config: ${CONFIG_FILE}"
echo "This test uses:"
echo "  - Small 0.5B models"
echo "  - Only 10 puzzles"
echo "  - 2 training epochs"
echo "  - Single GPU"
echo ""

# Create test output directory
mkdir -p outputs_test/{predictor,warmstart,checkpoints,logs,data}

# -----------------------------------------------------------------------------
# Test 1: Verify imports
# -----------------------------------------------------------------------------
echo "[Test 1] Verifying imports..."
python3 -c "
from src.predictor.model import LengthPredictorMLP
from src.utils.common import load_config, setup_logging
print('  ✓ Imports successful')
"

# -----------------------------------------------------------------------------
# Test 2: Load config
# -----------------------------------------------------------------------------
echo ""
echo "[Test 2] Loading configuration..."
python3 -c "
from src.utils.common import load_config
config = load_config('${CONFIG_FILE}')
print(f'  ✓ Config loaded')
print(f'    Attacker: {config[\"models\"][\"attacker\"]}')
print(f'    Victim: {config[\"models\"][\"victim\"]}')
print(f'    Num puzzles: {config[\"predictor\"][\"num_puzzles\"]}')
"

# -----------------------------------------------------------------------------
# Test 3: Test MLP model
# -----------------------------------------------------------------------------
echo ""
echo "[Test 3] Testing MLP model..."
python3 -c "
import torch
from src.predictor.model import LengthPredictorMLP

# Create model
model = LengthPredictorMLP(input_dim=1024, hidden_dim=256)
print(f'  ✓ Model created')

# Test forward pass
x = torch.randn(4, 1024)
y = model(x)
print(f'  ✓ Forward pass: input {x.shape} -> output {y.shape}')

# Test save/load
import tempfile
with tempfile.NamedTemporaryFile(suffix='.pt') as f:
    model.save(f.name)
    loaded = LengthPredictorMLP.load(f.name, device='cpu')
    print(f'  ✓ Save/load successful')
"

# -----------------------------------------------------------------------------
# Test 4: Test tokenizer loading
# -----------------------------------------------------------------------------
echo ""
echo "[Test 4] Testing tokenizer..."
python3 -c "
from transformers import AutoTokenizer
from src.utils.common import load_config

config = load_config('${CONFIG_FILE}')
tokenizer = AutoTokenizer.from_pretrained(
    config['models']['attacker'],
    trust_remote_code=True
)
print(f'  ✓ Tokenizer loaded: {config[\"models\"][\"attacker\"]}')

# Test chat template
messages = [{'role': 'user', 'content': 'Hello'}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(f'  ✓ Chat template works')
"

# -----------------------------------------------------------------------------
# Test 5: Test reward function structure
# -----------------------------------------------------------------------------
echo ""
echo "[Test 5] Testing reward function structure..."
python3 -c "
# Test that reward module can be imported and has correct interface
import sys
sys.path.insert(0, '.')

# Mock environment variables
import os
os.environ['PUZZLE_MAX_LEN'] = '64'
os.environ['DIVERSITY_WEIGHT'] = '0.0'
os.environ['ENABLE_DIVERSITY_REWARD'] = '0'
os.environ['ROLLOUT_BATCH_SIZE'] = '2'

from src.training.reward import compute_score_batch, extract_puzzle

# Test extract_puzzle
puzzle, valid = extract_puzzle('<think>test</think>What is 2+2?')
assert valid == True
assert puzzle == 'What is 2+2?'
print('  ✓ extract_puzzle works')

# Test with invalid input
puzzle, valid = extract_puzzle('No think tags here')
assert valid == False
print('  ✓ Invalid detection works')

print('  ✓ Reward function structure OK')
"

echo ""
echo "================================================================================"
echo "All basic tests passed!"
echo "================================================================================"
echo ""
echo "To run the full pipeline test (requires GPU):"
echo ""
echo "  # Step 1: Train predictor (simplified)"
echo "  CONFIG_FILE=configs/tiny_test.yaml python -m src.predictor.train"
echo ""
echo "  # Step 2: Test server (in separate terminal)"
echo "  CONFIG_FILE=configs/tiny_test.yaml python -m src.predictor.server"
echo ""

