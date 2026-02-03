"""MLP model for reasoning length prediction."""

import torch
import torch.nn as nn


class LengthPredictorMLP(nn.Module):
    """
    MLP that predicts reasoning length from victim model's hidden states.
    
    Architecture (from paper):
        "MLP with ReLU activations and dropout (0.1): R^d → 1024 → 512 → R"
        
        Input (hidden_dim) → Linear(1024) → ReLU → Dropout(0.1)
        → Linear(512) → ReLU → Dropout(0.1)
        → Linear(1) → Output
    
    The model predicts log-scale reasoning length log(1 + L_rp),
    which is converted back via expm1() for the actual token count.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 1024,
        intermediate_dim: int = 512
    ):
        """
        Initialize the MLP predictor.
        
        Args:
            input_dim: Dimension of input hidden states (from victim model)
            hidden_dim: First hidden layer dimension (paper: 1024)
            intermediate_dim: Second hidden layer dimension (paper: 512)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Hidden states tensor of shape (batch_size, input_dim)
            
        Returns:
            Log-scale predicted length of shape (batch_size,)
        """
        return self.net(x).squeeze(-1)
    
    def predict_length(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict actual reasoning length (not log-scale).
        
        Args:
            x: Hidden states tensor
            
        Returns:
            Predicted token counts
        """
        log_pred = self.forward(x)
        return torch.expm1(log_pred)  # exp(x) - 1, inverse of log1p
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda') -> 'LengthPredictorMLP':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to saved model
            device: Device to load model on
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint.get('hidden_dim', 1024),
            intermediate_dim=checkpoint.get('intermediate_dim', 512)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        return model
    
    def save(self, path: str) -> None:
        """
        Save model to checkpoint.
        
        Args:
            path: Path to save model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'intermediate_dim': self.intermediate_dim,
        }, path)

