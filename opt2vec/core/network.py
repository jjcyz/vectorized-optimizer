"""
Enhanced Opt2Vec network with advanced architectural features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal information in optimization history.
    """

    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class ResidualBlock(nn.Module):
    """
    Residual block with normalization and activation.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 activation: str = 'gelu',
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm

        # Activation function selection
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU is the same as Swish
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        # Main layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

        # Normalization
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
        else:
            self.norm1 = nn.BatchNorm1d(input_dim)
            self.norm2 = nn.BatchNorm1d(input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Skip connection (if dimensions don't match)
        self.skip_connection = nn.Linear(input_dim, input_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor

        Returns:
            Output tensor with residual connection
        """
        identity = x

        # First layer
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Second layer
        x = self.fc2(x)
        x = self.dropout(x)

        # Residual connection
        x = x + identity

        return x


class LightweightAttention(nn.Module):
    """
    Lightweight attention mechanism for history encoding.
    """

    def __init__(self,
                 input_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention mechanism.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Output tensor with attention applied
        """
        batch_size, seq_len, _ = x.shape

        # Linear transformations
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.input_dim)

        # Output projection
        output = self.output(context)

        return output


class TinyOpt2VecNetwork(nn.Module):
    """
    Enhanced Opt2Vec network with advanced architectural features.

    Features:
    - Configurable embedding dimensions: [32, 64, 128]
    - Extended history windows: [8, 16, 32] steps
    - Residual connections and normalization
    - Attention mechanism for history encoding
    - Extended feature set with parameter norms, gradient diversity, loss curvature
    - Multiple activation functions (Swish, GELU, ReLU)
    """

    def __init__(self,
                 input_dim: int = 7,  # Extended features: [loss, grad_norm, lr, param_norm, grad_diversity, loss_curvature, step]
                 embedding_dim: int = 64,  # Configurable: [32, 64, 128]
                 history_length: int = 16,  # Extended: [8, 16, 32]
                 hidden_dim: int = 128,
                 num_residual_blocks: int = 3,
                 activation: str = 'gelu',  # 'gelu', 'swish', 'relu'
                 dropout: float = 0.1,
                 use_layer_norm: bool = True,
                 use_attention: bool = True,
                 use_positional_encoding: bool = True,
                 num_attention_heads: int = 4):
        """
        Initialize Enhanced Opt2Vec network.

        Args:
            input_dim: Number of features per history step
            embedding_dim: Dimension of output embedding [32, 64, 128]
            history_length: Number of history steps to process [8, 16, 32]
            hidden_dim: Hidden dimension for residual blocks
            num_residual_blocks: Number of residual blocks
            activation: Activation function ('gelu', 'swish', 'relu')
            dropout: Dropout rate for regularization
            use_layer_norm: Whether to use LayerNorm (True) or BatchNorm (False)
            use_attention: Whether to use attention mechanism
            use_positional_encoding: Whether to use positional encoding
            num_attention_heads: Number of attention heads
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.history_length = history_length
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding

        # Feature projection
        self.feature_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_dim, history_length)

        # Attention mechanism
        if use_attention:
            self.attention = LightweightAttention(hidden_dim, num_attention_heads, dropout)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, activation, dropout, use_layer_norm)
            for _ in range(num_residual_blocks)
        ])

        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU() if activation == 'gelu' else (nn.SiLU() if activation == 'swish' else nn.ReLU()),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim),
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhanced network.

        Args:
            history: Input tensor of shape [batch_size, history_length, input_dim]

        Returns:
            embedding: Output embedding of shape [batch_size, embedding_dim]
        """
        batch_size = history.shape[0]

        # Feature projection
        x = self.feature_projection(history)  # [batch_size, history_length, hidden_dim]

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = x.transpose(0, 1)  # [history_length, batch_size, hidden_dim]
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # [batch_size, history_length, hidden_dim]

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)

        # Apply residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        # Global pooling
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, history_length]
        x = self.global_pool(x).squeeze(-1)  # [batch_size, hidden_dim]

        # Final projection
        embedding = self.final_projection(x)

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
                'norm': torch.norm(embedding, dim=1).mean().item(),
                'embedding_dim': self.embedding_dim,
                'history_length': self.history_length
            }


class NetworkConfig:
    """
    Configuration class for different network architectures.
    """

    @staticmethod
    def get_config(embedding_dim: int = 64,
                   history_length: int = 16,
                   activation: str = 'gelu') -> dict:
        """
        Get network configuration for given parameters.

        Args:
            embedding_dim: Embedding dimension (32, 64, 128)
            history_length: History length (8, 16, 32)
            activation: Activation function ('gelu', 'swish', 'relu')

        Returns:
            Configuration dictionary
        """
        return {
            'input_dim': 7,  # Extended features
            'embedding_dim': embedding_dim,
            'history_length': history_length,
            'hidden_dim': max(64, embedding_dim * 2),
            'num_residual_blocks': 3,
            'activation': activation,
            'dropout': 0.1,
            'use_layer_norm': True,
            'use_attention': True,
            'use_positional_encoding': True,
            'num_attention_heads': 4
        }
