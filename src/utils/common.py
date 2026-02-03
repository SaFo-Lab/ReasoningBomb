"""Common utility functions."""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve output directories
    output_dir = Path(config['paths']['output_dir'])
    for key in ['predictor_dir', 'warmstart_dir', 'checkpoints_dir', 'logs_dir']:
        if config['paths'][key]:
            config['paths'][key] = str(output_dir / config['paths'][key])
    
    # Create directories
    for key in ['predictor_dir', 'warmstart_dir', 'checkpoints_dir', 'logs_dir']:
        if config['paths'][key]:
            Path(config['paths'][key]).mkdir(parents=True, exist_ok=True)
    
    return config


def setup_logging(
    name: str = "rbomb",
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device_map(gpu_string: str) -> Dict[str, Any]:
    """
    Convert GPU string to device map for model loading.
    
    Args:
        gpu_string: Comma-separated GPU indices (e.g., "0,1,2")
        
    Returns:
        Device map configuration
    """
    gpus = [int(g.strip()) for g in gpu_string.split(',')]
    
    if len(gpus) == 1:
        return {'': f'cuda:{gpus[0]}'}
    else:
        return 'auto'


def set_hf_cache(cache_dir: Optional[str]) -> None:
    """Set HuggingFace cache directory."""
    if cache_dir:
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_DATASETS_CACHE'] = cache_dir


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_chat_prompt(
    content: str,
    tokenizer,
    system_prompt: Optional[str] = None
) -> str:
    """
    Format content as chat prompt using tokenizer's chat template.
    
    Args:
        content: User message content
        tokenizer: Tokenizer with chat template
        system_prompt: Optional system prompt
        
    Returns:
        Formatted prompt string
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

