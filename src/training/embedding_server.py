"""
Embedding Server for Diversity Computation.

Provides text embeddings using Qwen3-Embedding-8B for computing
diversity scores in the reward function.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python -m src.training.embedding_server --port 8001
"""

import os
import argparse
import threading
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify
from typing import List, Dict

from ..utils import load_config, set_hf_cache


class EmbeddingServer:
    """Server for computing text embeddings."""
    
    def __init__(
        self,
        model_path: str,
        embedding_dim: int = 1024,
        cache_dir: str = None
    ):
        """
        Initialize embedding server.
        
        Args:
            model_path: Path to embedding model
            embedding_dim: Output embedding dimension
            cache_dir: HuggingFace cache directory
        """
        print(f"[Server] Initializing...")
        print(f"[Server] Model: {model_path}")
        print(f"[Server] Embedding dim: {embedding_dim}")
        
        self.embedding_dim = embedding_dim
        
        try:
            from sentence_transformers import SentenceTransformer
            
            print("[Server] Loading with sentence-transformers...")
            
            self.model = SentenceTransformer(
                model_path,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,
                    "device_map": "auto",
                },
                cache_folder=cache_dir,
                trust_remote_code=True,
            )
            self.model.truncate_dim = embedding_dim
            self.use_st = True
            
            print("[Server] ✓ Loaded with sentence-transformers")
            
        except ImportError:
            print("[Server] sentence-transformers not available, using transformers...")
            self._load_with_transformers(model_path, cache_dir)
            self.use_st = False
        
        self._lock = threading.Lock()
        print("[Server] Initialization complete!")
    
    def _load_with_transformers(self, model_path: str, cache_dir: str):
        """Fallback loader using transformers."""
        from transformers import AutoModel, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.model.eval()
    
    def get_embeddings(self, texts: List[str]) -> Dict:
        """
        Compute embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dict with embeddings and metadata
        """
        try:
            with self._lock:
                if self.use_st:
                    embeddings = self.model.encode(
                        texts,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                else:
                    # Transformers fallback
                    batch = self.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=8192,
                        return_tensors='pt'
                    )
                    
                    device = next(self.model.parameters()).device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**batch)
                        # Last token pooling
                        mask = batch['attention_mask']
                        seq_lens = mask.sum(dim=1) - 1
                        embeddings = outputs.last_hidden_state[
                            torch.arange(len(texts), device=device),
                            seq_lens
                        ]
                        
                        if embeddings.shape[-1] > self.embedding_dim:
                            embeddings = embeddings[:, :self.embedding_dim]
                        
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                        embeddings = embeddings.cpu().float().numpy()
                
                torch.cuda.empty_cache()
            
            return {
                'success': True,
                'embeddings': embeddings.tolist(),
                'num_texts': len(texts),
                'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else self.embedding_dim,
            }
            
        except Exception as e:
            print(f"[Server] Error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e), 'embeddings': []}


# Flask app
app = Flask(__name__)
server = None


@app.route('/health', methods=['GET'])
def health():
    if server is None:
        return jsonify({'status': 'not_ready'}), 503
    return jsonify({
        'status': 'healthy',
        'type': 'embedding_server',
        'embedding_dim': server.embedding_dim,
    })


@app.route('/embed', methods=['POST'])
def embed():
    """Get embeddings for texts."""
    if server is None:
        return jsonify({'error': 'Server not initialized'}), 503
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Missing texts', 'success': False}), 400
        
        result = server.get_embeddings(texts)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


def main():
    global server
    
    parser = argparse.ArgumentParser(description='Embedding Server')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--embedding_dim', type=int, default=1024)
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_hf_cache(config['paths']['hf_cache'])
    
    port = args.port or config['training']['embedding_port']
    model = args.model or config['models']['embedding']
    
    print("="*80)
    print("Embedding Server")
    print("="*80)
    print(f"Port: {port}")
    print(f"Model: {model}")
    print(f"Embedding Dim: {args.embedding_dim}")
    print("="*80)
    
    server = EmbeddingServer(
        model_path=model,
        embedding_dim=args.embedding_dim,
        cache_dir=config['paths']['hf_cache']
    )
    
    print(f"\n[Server] Starting on port {port}...")
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == '__main__':
    main()

