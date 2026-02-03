"""
Length Predictor Server.

Provides HTTP endpoint for predicting reasoning length from puzzles.
Uses victim model hidden states + trained MLP predictor.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python -m src.predictor.server --port 8000
"""

import os
import argparse
import threading
import torch
from flask import Flask, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM

from .model import LengthPredictorMLP
from ..utils import load_config, setup_logging, set_hf_cache


class PredictorServer:
    """Server for predicting reasoning length from puzzles."""
    
    def __init__(
        self,
        victim_model_path: str,
        mlp_model_path: str,
        cache_dir: str = None
    ):
        """
        Initialize the predictor server.
        
        Args:
            victim_model_path: Path to victim model
            mlp_model_path: Path to trained MLP predictor
            cache_dir: HuggingFace cache directory
        """
        print(f"[Server] Initializing...")
        print(f"[Server] Victim model: {victim_model_path}")
        print(f"[Server] MLP model: {mlp_model_path}")
        
        # Load tokenizer
        print("[Server] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            victim_model_path,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Load victim model with automatic device mapping
        print("[Server] Loading victim model...")
        self.victim_model = AutoModelForCausalLM.from_pretrained(
            victim_model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        self.victim_model.eval()
        
        self.hidden_dim = self.victim_model.config.hidden_size
        print(f"[Server] Hidden dimension: {self.hidden_dim}")
        
        # Thread lock for serialized forward passes
        self._forward_lock = threading.Lock()
        
        # Load MLP predictor
        print("[Server] Loading MLP predictor...")
        self.mlp = LengthPredictorMLP.load(mlp_model_path, device='cuda')
        
        print("[Server] Initialization complete!")
    
    def predict(self, puzzle: str) -> dict:
        """
        Predict reasoning length for a puzzle.
        
        Args:
            puzzle: Puzzle text
            
        Returns:
            Dict with prediction results
        """
        try:
            # Format with chat template
            messages = [{"role": "user", "content": puzzle}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.victim_model.device)
            input_length = input_ids.shape[1]
            
            # Extract hidden states (serialized for memory safety)
            with self._forward_lock:
                with torch.no_grad():
                    outputs = self.victim_model(
                        input_ids,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    
                    # Get last layer, last position hidden state
                    last_hidden = outputs.hidden_states[-1][0, -1, :]
                    last_hidden = last_hidden.float().unsqueeze(0)
                    
                    # Predict with MLP
                    log_pred = self.mlp(last_hidden.cuda())
                    predicted_length = torch.expm1(log_pred).item()
                    log_pred_value = log_pred.item()
                
                # Cleanup
                del outputs, last_hidden
                torch.cuda.empty_cache()
            
            return {
                'success': True,
                'predicted_length': predicted_length,
                'log_prediction': log_pred_value,
                'input_length': input_length,
            }
            
        except Exception as e:
            print(f"[Server] Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'predicted_length': 0.0,
            }


# Flask application
app = Flask(__name__)
server = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if server is None:
        return jsonify({'status': 'not_ready'}), 503
    return jsonify({'status': 'healthy', 'type': 'length_predictor'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict reasoning length for a puzzle.
    
    Request: {"puzzle": "puzzle text"}
    Response: {"success": true, "predicted_length": 5000.0, ...}
    """
    if server is None:
        return jsonify({'error': 'Server not initialized'}), 503
    
    try:
        data = request.get_json()
        puzzle = data.get('puzzle', '')
        
        if not puzzle:
            return jsonify({'error': 'Missing puzzle', 'success': False}), 400
        
        result = server.predict(puzzle)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


def main():
    global server
    
    parser = argparse.ArgumentParser(description='Length Predictor Server')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--port', type=int, default=None,
                       help='Server port (overrides config)')
    parser.add_argument('--victim_model', type=str, default=None,
                       help='Victim model path (overrides config)')
    parser.add_argument('--mlp_model', type=str, default=None,
                       help='MLP model path (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_hf_cache(config['paths']['hf_cache'])
    
    # Get parameters (CLI overrides config)
    port = args.port or config['predictor']['server_port']
    victim_model = args.victim_model or config['models']['victim']
    mlp_model = args.mlp_model or os.path.join(
        config['paths']['predictor_dir'], 'mlp_predictor.pt'
    )
    
    print("="*80)
    print("Length Predictor Server")
    print("="*80)
    print(f"Port: {port}")
    print(f"Victim Model: {victim_model}")
    print(f"MLP Model: {mlp_model}")
    print("="*80)
    
    # Initialize server
    server = PredictorServer(
        victim_model_path=victim_model,
        mlp_model_path=mlp_model,
        cache_dir=config['paths']['hf_cache']
    )
    
    # Run Flask app
    print(f"\n[Server] Starting on port {port}...")
    app.run(host='0.0.0.0', port=port, threaded=True)


if __name__ == '__main__':
    main()

