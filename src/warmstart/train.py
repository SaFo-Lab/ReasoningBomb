"""
Train SFT warm start models.

Fine-tunes separate attacker models for each puzzle length category.

Usage:
    python -m src.warmstart.train --config configs/default.yaml
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from ..utils import load_config, setup_logging, set_hf_cache


class WarmstartDataset(Dataset):
    """Dataset for SFT training on warm start data."""
    
    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        prompt: str,
        max_length: int = 8192
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        response = sample['raw_output']
        
        # Format as chat
        messages = [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": response}
        ]
        
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        input_ids = tokenized['input_ids']
        labels = input_ids.copy()
        
        # Mask prompt tokens (only train on response)
        user_messages = [{"role": "user", "content": self.prompt}]
        user_text = self.tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        user_tokens = self.tokenizer(user_text, add_special_tokens=True)['input_ids']
        prompt_length = len(user_tokens)
        
        labels[:prompt_length] = [-100] * prompt_length
        tokenized['labels'] = labels
        
        return tokenized


class DataCollatorWithPadding:
    """Pads batches and handles labels."""
    
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        
        batch = self.tokenizer.pad(
            features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        
        for i, lbl in enumerate(labels):
            seq_len = min(len(lbl), max_len)
            padded_labels[i, :seq_len] = torch.tensor(lbl[:seq_len], dtype=torch.long)
        
        batch["labels"] = padded_labels
        return batch


class LogCallback(TrainerCallback):
    """Log training progress to console."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            lr = logs.get("learning_rate", 0)
            print(f"[step {state.global_step}] loss={logs['loss']:.4f} lr={lr:.2e}")


def train_category_model(
    category: str,
    samples: List[Dict],
    prompt: str,
    base_model: str,
    output_dir: str,
    config: Dict,
    logger=None
):
    """
    Train a model for a specific length category.
    
    Args:
        category: Category name (e.g., "length_128")
        samples: List of samples for this category
        prompt: User prompt used for generation
        base_model: Path to base model
        output_dir: Output directory for fine-tuned model
        config: Training configuration
        logger: Logger instance
    """
    log = logger.info if logger else print
    
    log(f"\n{'='*80}")
    log(f"Training model for: {category}")
    log(f"{'='*80}")
    log(f"Samples: {len(samples)}")
    log(f"Output: {output_dir}")
    
    cache_dir = config['paths']['hf_cache']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # Create dataset
    # Extract just the user prompt content from the full formatted prompt
    import re
    # The prompt contains the user content between specific markers
    # For simplicity, we use the PUZZLE_PROMPT constant
    from .collect import PUZZLE_PROMPT
    
    dataset = WarmstartDataset(
        samples=samples,
        tokenizer=tokenizer,
        prompt=PUZZLE_PROMPT,
        max_length=8192
    )
    
    log(f"Dataset size: {len(dataset)}")
    
    # Training arguments
    warmstart_config = config['warmstart']
    
    # Paper: "train for 20 epochs with per-device batch size 2 and 4 gradient 
    #         accumulation steps, learning rate 1×10^-5 with cosine schedule 
    #         and warmup ratio 0.03, AdamW with weight decay 0.01"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=warmstart_config['sft_epochs'],
        per_device_train_batch_size=warmstart_config['sft_batch_size'],
        gradient_accumulation_steps=warmstart_config['gradient_accumulation_steps'],
        learning_rate=warmstart_config['sft_learning_rate'],
        warmup_ratio=warmstart_config.get('warmup_ratio', 0.03),
        lr_scheduler_type=warmstart_config.get('lr_scheduler_type', "cosine"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=warmstart_config.get('max_grad_norm', 1.0),
        weight_decay=warmstart_config.get('weight_decay', 0.01),
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[LogCallback()],
    )
    
    # Train
    log("Starting training...")
    result = trainer.train()
    
    log(f"Training complete!")
    log(f"Final loss: {result.training_loss:.4f}")
    
    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metrics
    with open(os.path.join(output_dir, "training_metrics.json"), 'w') as f:
        json.dump({
            'category': category,
            'num_samples': len(samples),
            'final_loss': result.training_loss,
        }, f, indent=2)
    
    log(f"Saved model to: {output_dir}")
    
    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Train Warm Start Models')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--category', type=str, default=None,
                       help='Train specific category only')
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_hf_cache(config['paths']['hf_cache'])
    
    warmstart_dir = config['paths']['warmstart_dir']
    dataset_path = os.path.join(warmstart_dir, "warmstart_dataset.json")
    
    logger = setup_logging(
        name="warmstart_train",
        level=config['logging']['log_level'],
        log_file=os.path.join(warmstart_dir, "train.log")
    )
    
    # Load dataset
    logger.info("Loading warm start dataset...")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Run: python -m src.warmstart.collect first")
        return
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    dataset = data['dataset']
    prompt = data['metadata']['prompt']
    base_model = config['models']['attacker']
    
    # Categories to train
    categories = list(config['warmstart']['categories'].keys())
    if args.category:
        if args.category not in categories:
            logger.error(f"Unknown category: {args.category}")
            return
        categories = [args.category]
    
    logger.info(f"Training {len(categories)} models: {categories}")
    
    # Train each category
    results = {}
    for category in categories:
        samples = dataset.get(category, [])
        
        if not samples:
            logger.warning(f"No samples for {category}, skipping")
            continue
        
        output_dir = os.path.join(warmstart_dir, f"sft_{category}")
        
        try:
            result = train_category_model(
                category=category,
                samples=samples,
                prompt=prompt,
                base_model=base_model,
                output_dir=output_dir,
                config=config,
                logger=logger
            )
            results[category] = "SUCCESS"
        except Exception as e:
            logger.error(f"Error training {category}: {e}")
            results[category] = f"FAILED: {e}"
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Training Summary")
    logger.info("="*80)
    for cat, status in results.items():
        logger.info(f"  {cat}: {status}")


if __name__ == '__main__':
    main()

