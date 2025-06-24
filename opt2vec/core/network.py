"""
Tiny Opt2Vec network for resource-constrained environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TinyOpt2VecNetwork(nn.Module):
    """
    Ultra-lightweight version of Opt2Vec network for resource-constrained environments.

    Uses simple MLP instead of Transformer to reduce memory and computation by ~90%.

    Architecture:
    - Input: [batch_size, history_length, 3] - sequence of optimization history
    - Flatten: [batch_size, history_length * 3]
    - Hidden: 32 neurons with ReLU
    - Output: [batch_size, embedding_dim] - compressed representation
    """

    def __init__(self,
                 input_dim: int = 3,  # [loss, grad_norm, lr]
                 embedding_dim: int = 16,  # Reduced from 32 for memory efficiency
                 history_length: int = 5):  # Reduced from 10
        """
        Initialize TinyOpt2Vec network.

        Args:
            input_dim: Number of features per history step
            embedding_dim: Dimension of output embedding
            history_length: Number of history steps to process
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.history_length = history_length

        # Simple MLP instead of Transformer
        flattened_input = input_dim * history_length
        self.network = nn.Sequential(
            nn.Linear(flattened_input, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, embedding_dim),
            nn.Tanh()  # Bound output to [-1, 1] for stability
        )

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            history: Input tensor of shape [batch_size, history_length, input_dim]

        Returns:
            embedding: Output embedding of shape [batch_size, embedding_dim]
        """
        batch_size = history.shape[0]

        # Flatten history sequence
        x = history.view(batch_size, -1)  # [batch, history_length * input_dim]

        # Apply network
        embedding = self.network(x)

        return embedding

    def get_embedding_stats(self, history: torch.Tensor) -> dict:
        """
        Get statistics about the generated embedding.

        Args:
            history: Input history tensor

        Returns:
            Dictionary with embedding statistics
        """
        with torch.no_grad():
            embedding = self.forward(history)
            return {
                'mean': embedding.mean().item(),
                'std': embedding.std().item(),
                'min': embedding.min().item(),
                'max': embedding.max().item(),
                'norm': torch.norm(embedding, dim=1).mean().item()
            }
