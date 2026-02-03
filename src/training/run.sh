#!/bin/bash
# =============================================================================
# GRPO Training Script
# =============================================================================
#
# Usage:
#   bash src/training/run.sh [OPTIONS]
#
# Options:
#   --puzzle_max_len N      Maximum puzzle length (default: 128)
#   --diversity_weight W    Diversity reward weight (default: 0.5)
#   --disable_diversity     Disable diversity reward
#   --predictor_gpus GPUS   GPUs for predictor server (default: 0,1)
#   --embedding_gpus GPUS   GPUs for embedding server (default: 2,3)
#   --training_gpus GPUS    GPUs for training (default: 4,5,6,7)
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------

# Paper defaults
PUZZLE_MAX_LEN=128
DIVERSITY_WEIGHT=1.0  # Paper: "We set the diversity weight w_div = 1.0"
ENABLE_DIVERSITY=1

PREDICTOR_GPUS="0,1"
EMBEDDING_GPUS="2,3"
TRAINING_GPUS="4,5,6,7"

CONFIG_FILE="configs/default.yaml"

# Paths (adjust for your setup)
OUTPUT_DIR="./outputs"
PREDICTOR_MODEL="${OUTPUT_DIR}/predictor/mlp_predictor.pt"
WARMSTART_MODEL="${OUTPUT_DIR}/warmstart/sft_length_${PUZZLE_MAX_LEN}"
BASE_MODEL="Qwen/Qwen3-8B"

# Training hyperparameters (from paper)
# Paper: "group size N_sample=8 samples per meta-prompt, 150 training iterations"
TRAIN_BATCH_SIZE=48
ROLLOUT_BATCH_SIZE=8  # Paper: N_sample=8
PPO_MINI_BATCH_SIZE=24
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=2048  # Paper: "max attacker generation length of 2048 tokens"
LEARNING_RATE=5e-6  # Paper: "AdamW (learning rate 5×10^-6)"
KL_LOSS_COEF=0.04  # Paper: "KL coefficient β=0.04"
CLIP_PARAM=0.2  # Paper: "clip parameter ε=0.2"
TOTAL_EPOCHS=150  # Paper: "150 training iterations"

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --puzzle_max_len)
            PUZZLE_MAX_LEN="$2"
            shift 2
            ;;
        --diversity_weight)
            DIVERSITY_WEIGHT="$2"
            shift 2
            ;;
        --disable_diversity)
            ENABLE_DIVERSITY=0
            shift
            ;;
        --predictor_gpus)
            PREDICTOR_GPUS="$2"
            shift 2
            ;;
        --embedding_gpus)
            EMBEDDING_GPUS="$2"
            shift 2
            ;;
        --training_gpus)
            TRAINING_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            head -30 "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Export Configuration
# -----------------------------------------------------------------------------

export PUZZLE_MAX_LEN
export DIVERSITY_WEIGHT
export ENABLE_DIVERSITY_REWARD=${ENABLE_DIVERSITY}
export ROLLOUT_BATCH_SIZE

# Experiment naming
PROJECT_NAME="reasoningbomb"
DIVERSITY_STR=$(echo "${DIVERSITY_WEIGHT}" | tr -d '.')
if [ "$ENABLE_DIVERSITY" = "1" ]; then
    EXPERIMENT_NAME="puz${PUZZLE_MAX_LEN}_div${DIVERSITY_STR}"
else
    EXPERIMENT_NAME="puz${PUZZLE_MAX_LEN}_nodiv"
fi

# Data paths
DATA_DIR="${OUTPUT_DIR}/data"
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/test.parquet"

# Select attacker model
if [ -d "${WARMSTART_MODEL}" ]; then
    ATTACKER_MODEL="${WARMSTART_MODEL}"
    echo "Using warm start model: ${ATTACKER_MODEL}"
else
    ATTACKER_MODEL="${BASE_MODEL}"
    echo "Using base model: ${ATTACKER_MODEL}"
fi

CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR} logs

# -----------------------------------------------------------------------------
# Display Configuration
# -----------------------------------------------------------------------------

echo "================================================================================"
echo "ReasoningBomb - GRPO Training (Stage 2)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Experiment: ${EXPERIMENT_NAME}"
echo "  Puzzle Max Length: ${PUZZLE_MAX_LEN} tokens"
echo "  Diversity: $([ "$ENABLE_DIVERSITY" = "1" ] && echo "ON (w=${DIVERSITY_WEIGHT})" || echo "OFF")"
echo "  Attacker Model: ${ATTACKER_MODEL}"
echo ""
echo "GPU Allocation:"
echo "  Predictor: GPUs ${PREDICTOR_GPUS} (port 8000)"
echo "  Embedding: GPUs ${EMBEDDING_GPUS} (port 8001)"
echo "  Training: GPUs ${TRAINING_GPUS}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Start Servers
# -----------------------------------------------------------------------------

echo "[Step 1] Starting Servers..."

# Cleanup
pkill -f "predictor.server" 2>/dev/null || true
pkill -f "embedding_server" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

# Start predictor server
echo "Starting Predictor Server (GPUs ${PREDICTOR_GPUS})..."
CUDA_VISIBLE_DEVICES=${PREDICTOR_GPUS} nohup python -m src.predictor.server \
    --config ${CONFIG_FILE} \
    --port 8000 \
    --mlp_model ${PREDICTOR_MODEL} \
    > logs/predictor.log 2>&1 &
PREDICTOR_PID=$!
echo "  PID: ${PREDICTOR_PID}"

# Start embedding server (if diversity enabled)
if [ "$ENABLE_DIVERSITY" = "1" ]; then
    echo "Starting Embedding Server (GPUs ${EMBEDDING_GPUS})..."
    CUDA_VISIBLE_DEVICES=${EMBEDDING_GPUS} nohup python -m src.training.embedding_server \
        --config ${CONFIG_FILE} \
        --port 8001 \
        > logs/embedding.log 2>&1 &
    EMBEDDING_PID=$!
    echo "  PID: ${EMBEDDING_PID}"
fi

# -----------------------------------------------------------------------------
# Step 2: Wait for Servers
# -----------------------------------------------------------------------------

echo ""
echo "[Step 2] Waiting for Servers..."

wait_for_server() {
    local url=$1
    local name=$2
    local max_wait=600
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo "  ✓ ${name} ready"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  Waiting for ${name}... (${waited}s)"
    done
    
    echo "  ✗ ${name} timeout"
    return 1
}

wait_for_server "http://localhost:8000" "Predictor"

if [ "$ENABLE_DIVERSITY" = "1" ]; then
    wait_for_server "http://localhost:8001" "Embedding"
fi

# -----------------------------------------------------------------------------
# Step 3: Verify Setup
# -----------------------------------------------------------------------------

echo ""
echo "[Step 3] Verifying Setup..."

# Check data files
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Data not found. Creating placeholder data..."
    mkdir -p ${DATA_DIR}
    python -c "
import pandas as pd
data = {'prompt': ['Generate a puzzle'] * 100}
pd.DataFrame(data).to_parquet('${TRAIN_DATA}')
pd.DataFrame(data[:20]).to_parquet('${VAL_DATA}')
print('Created placeholder data')
"
fi

echo "  ✓ Data files ready"

# Count training GPUs
NUM_TRAINING_GPUS=$(echo "${TRAINING_GPUS}" | tr ',' '\n' | wc -l | xargs)
echo "  ✓ Training GPUs: ${NUM_TRAINING_GPUS}"

# -----------------------------------------------------------------------------
# Step 4: Launch Training
# -----------------------------------------------------------------------------

echo ""
echo "[Step 4] Launching GRPO Training..."
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down..."
    kill ${PREDICTOR_PID} 2>/dev/null || true
    [ -n "${EMBEDDING_PID}" ] && kill ${EMBEDDING_PID} 2>/dev/null || true
    pkill -f "predictor.server" 2>/dev/null || true
    pkill -f "embedding_server" 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

# Launch GRPO training via verl
export CUDA_VISIBLE_DEVICES=${TRAINING_GPUS}

NUM_CPUS=$(nproc)
[ "$NUM_CPUS" -gt 32 ] && NUM_CPUS=32

REWARD_FUNCTION="src/training/reward.py"

python3 -m verl.trainer.main_ppo \
    ray_kwargs.ray_init.num_cpus=${NUM_CPUS} \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA} \
    data.val_files=${VAL_DATA} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${ATTACKER_MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${ROLLOUT_BATCH_SIZE} \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=${REWARD_FUNCTION} \
    custom_reward_function.name=compute_score_batch \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NUM_TRAINING_GPUS} \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.resume_mode='auto'

echo ""
echo "Training complete!"
echo "Checkpoints: ${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/"

cleanup

