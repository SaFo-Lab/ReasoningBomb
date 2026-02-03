"""
Reward Function for GRPO Training (ReasoningBomb).

Implements constant-time surrogate reward from the paper:
    R(p) = r_len(p) + w_div × r_div(p)

Where:
    - r_len(p): Length prediction reward from MLP predictor
      Paper: "normalized length reward using fixed constants μ_len=6.0 and σ_len=2.0"
      r_len = (log_pred - 6.0) / 2.0
      
    - r_div(p): Per-group pairwise diversity reward
      Paper: "We set the diversity weight w_div = 1.0"
      r_div = 1 - avg_pairwise_cosine_similarity

Usage:
    This module is called by verl during GRPO training.
    Configure via environment variables:
        - PUZZLE_MAX_LEN: Maximum puzzle length in tokens (default: 128)
        - DIVERSITY_WEIGHT: Weight for diversity reward (default: 1.0)
        - ENABLE_DIVERSITY_REWARD: Enable/disable diversity (default: 1)
        - ROLLOUT_BATCH_SIZE: GRPO group size (default: 8)
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple
from time import sleep

import requests
import numpy as np
from transformers import AutoTokenizer


# =============================================================================
# Configuration (from environment variables)
# =============================================================================

PREDICTOR_URL = os.environ.get('PREDICTOR_URL', 'http://localhost:8000')
EMBEDDING_URL = os.environ.get('EMBEDDING_URL', 'http://localhost:8001')

MAX_PUZZLE_TOKENS = int(os.environ.get('PUZZLE_MAX_LEN', 128))
# Paper: "We set the diversity weight w_div = 1.0"
DIVERSITY_WEIGHT = float(os.environ.get('DIVERSITY_WEIGHT', 1.0))
ENABLE_DIVERSITY = os.environ.get('ENABLE_DIVERSITY_REWARD', '1') == '1'
# Paper: "group size N_sample=8 samples per meta-prompt"
ROLLOUT_BATCH_SIZE = int(os.environ.get('ROLLOUT_BATCH_SIZE', 8))

INVALID_PENALTY = 0.0
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3

# Lazy tokenizer loading
_tokenizer = None
_tokenizer_lock = threading.Lock()
TOKENIZER_MODEL = os.environ.get('TOKENIZER_MODEL', 'Qwen/Qwen3-8B')
HF_CACHE = os.environ.get('HF_HOME', None)


def get_tokenizer():
    """Get cached tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                _tokenizer = AutoTokenizer.from_pretrained(
                    TOKENIZER_MODEL,
                    cache_dir=HF_CACHE,
                    trust_remote_code=True
                )
    return _tokenizer


# =============================================================================
# Server Communication
# =============================================================================

def call_predictor(puzzle: str) -> Dict:
    """Call predictor server for length prediction."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{PREDICTOR_URL}/predict",
                json={"puzzle": puzzle},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
            raise RuntimeError(f"Status {response.status_code}")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                sleep(2 ** attempt)
            else:
                raise


def call_embedding(puzzles: List[str]) -> Dict:
    """Call embedding server for text embeddings."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{EMBEDDING_URL}/embed",
                json={"texts": puzzles},
                timeout=REQUEST_TIMEOUT
            )
            if response.status_code == 200:
                return response.json()
            raise RuntimeError(f"Status {response.status_code}")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                sleep(2 ** attempt)
            else:
                raise


# =============================================================================
# Puzzle Extraction
# =============================================================================

def extract_puzzle(raw_output: str) -> Tuple[str, bool]:
    """
    Extract puzzle from reasoning model output.
    
    Input format: <think>reasoning...</think>actual_puzzle
    
    Args:
        raw_output: Raw model output
        
    Returns:
        Tuple of (puzzle text, is_valid)
    """
    if "</think>" not in raw_output:
        return "Hi.", False
    
    puzzle = raw_output.split("</think>", 1)[1].strip()
    
    if not puzzle:
        return "Hi.", False
    
    # Check token length
    tokenizer = get_tokenizer()
    token_count = len(tokenizer.encode(puzzle, add_special_tokens=False))
    
    if token_count > MAX_PUZZLE_TOKENS:
        return "Hi.", False
    
    return puzzle, True


# =============================================================================
# Diversity Computation
# =============================================================================

def compute_per_group_diversity(
    puzzles: List[str],
    valid_flags: List[bool],
    group_size: int
) -> List[float]:
    """
    Compute per-group diversity using pairwise cosine similarity.
    
    For each sample, diversity = 1 - (avg similarity to other samples in group).
    This penalizes samples similar to many others.
    
    Args:
        puzzles: All puzzles in batch
        valid_flags: Validity flags
        group_size: GRPO group size
        
    Returns:
        List of diversity rewards
    """
    num_puzzles = len(puzzles)
    rewards = [0.0] * num_puzzles
    
    num_groups = (num_puzzles + group_size - 1) // group_size
    
    # Collect valid puzzles for embedding
    valid_puzzles = []
    valid_to_idx = []
    
    for i, (puzzle, valid) in enumerate(zip(puzzles, valid_flags)):
        if valid:
            valid_puzzles.append(puzzle)
            valid_to_idx.append(i)
    
    if len(valid_puzzles) == 0:
        return rewards
    
    # Get embeddings
    try:
        result = call_embedding(valid_puzzles)
        if not result.get('success', False):
            return rewards
        embeddings = np.array(result['embeddings'])
    except Exception as e:
        print(f"[Reward] Embedding error: {e}")
        return rewards
    
    # Build index mapping
    idx_to_embed = {orig: emb for emb, orig in enumerate(valid_to_idx)}
    
    # Process each group
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(start + group_size, num_puzzles)
        
        # Get valid samples in this group
        group_valid_idx = []
        group_embed_idx = []
        
        for i in range(start, end):
            if valid_flags[i]:
                group_valid_idx.append(i)
                group_embed_idx.append(idx_to_embed[i])
        
        if len(group_valid_idx) <= 1:
            continue
        
        # Get group embeddings
        group_emb = embeddings[group_embed_idx]
        n = len(group_emb)
        
        # Pairwise similarity matrix
        sim_matrix = group_emb @ group_emb.T  # [n, n]
        
        # Average similarity to OTHER samples
        # avg_sim[i] = (sum(row) - 1) / (n - 1)  [exclude self-similarity]
        row_sums = sim_matrix.sum(axis=1)
        avg_sims = (row_sums - 1.0) / (n - 1)
        
        # Diversity = 1 - avg_similarity
        diversity_scores = 1.0 - avg_sims
        
        for j, orig_idx in enumerate(group_valid_idx):
            rewards[orig_idx] = DIVERSITY_WEIGHT * diversity_scores[j]
    
    return rewards


# =============================================================================
# Main Reward Function (verl interface)
# =============================================================================

def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict]
) -> List[Dict]:
    """
    Compute rewards for a batch of generated puzzles.
    
    Called by verl during GRPO training.
    
    Args:
        data_sources: Data source identifiers (unused)
        solution_strs: Generated puzzle outputs
        ground_truths: Ground truth (unused)
        extra_infos: Extra information (unused)
        
    Returns:
        List of reward dicts with:
            - "score": Final reward (required by verl)
            - "length_reward": Length component
            - "diversity_reward": Diversity component
            - "is_valid": Validity flag
    """
    num_puzzles = len(solution_strs)
    
    print(f"[Reward] Processing {num_puzzles} puzzles...")
    print(f"[Reward] Diversity: {'ON' if ENABLE_DIVERSITY else 'OFF'} "
          f"(weight={DIVERSITY_WEIGHT})")
    
    # Extract puzzles
    puzzles = []
    valid_flags = []
    
    for output in solution_strs:
        puzzle, is_valid = extract_puzzle(output)
        puzzles.append(puzzle)
        valid_flags.append(is_valid)
    
    valid_count = sum(valid_flags)
    valid_indices = [i for i, v in enumerate(valid_flags) if v]
    
    print(f"[Reward] Valid: {valid_count}/{num_puzzles}")
    
    # Get length rewards (parallel)
    length_rewards = [None] * num_puzzles
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        
        for i, (puzzle, valid) in enumerate(zip(puzzles, valid_flags)):
            if not valid:
                length_rewards[i] = INVALID_PENALTY
            else:
                futures[executor.submit(call_predictor, puzzle)] = i
        
        for future in futures:
            idx = futures[future]
            try:
                result = future.result()
                if result.get('success', False):
                    log_pred = result.get('log_prediction', 6.0)
                    length_rewards[idx] = (log_pred - 6.0) / 2.0
                else:
                    length_rewards[idx] = 0.0
            except Exception as e:
                print(f"[Reward] Predictor error {idx}: {e}")
                length_rewards[idx] = 0.0
    
    # Get diversity rewards
    if ENABLE_DIVERSITY and valid_count > 1:
        diversity_rewards = compute_per_group_diversity(
            puzzles, valid_flags, ROLLOUT_BATCH_SIZE
        )
    else:
        diversity_rewards = [0.0] * num_puzzles
    
    # Combine rewards
    results = []
    for i in range(num_puzzles):
        if valid_flags[i]:
            score = length_rewards[i] + diversity_rewards[i]
            results.append({
                "score": score,
                "length_reward": length_rewards[i],
                "diversity_reward": diversity_rewards[i],
                "is_valid": 1.0,
            })
        else:
            results.append({
                "score": INVALID_PENALTY,
                "length_reward": INVALID_PENALTY,
                "diversity_reward": 0.0,
                "is_valid": 0.0,
            })
    
    # Log statistics
    if valid_indices:
        valid_len = [length_rewards[i] for i in valid_indices]
        valid_div = [diversity_rewards[i] for i in valid_indices]
        valid_scores = [results[i]["score"] for i in valid_indices]
        
        print(f"[Reward] Length: avg={np.mean(valid_len):.3f}")
        if ENABLE_DIVERSITY:
            print(f"[Reward] Diversity: avg={np.mean(valid_div):.3f}")
        print(f"[Reward] Total: avg={np.mean(valid_scores):.3f}")
    
    return results

