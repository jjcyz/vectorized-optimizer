"""
Enhanced Opt2Vec optimizer with advanced architectural features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import warnings

from .history import LightweightOptimizationHistory
from .network import TinyOpt2VecNetwork, NetworkConfig

logger = logging.getLogger(__name__)


class LightweightOpt2VecOptimizer:
    """
    Opt2Vec optimizer with architectural features.

    Features:
    - Configurable embedding dimensions: [32, 64, 128]
    - Extended history windows: [8, 16, 32] steps
    - Residual connections and normalization
    - Attention mechanism for history encoding
    - Extended feature set with parameter norms, gradient diversity, loss curvature
    - Multiple activation functions (Swish, GELU, ReLU)
    """

    def __init__(self,
                 parameters,
                 base_lr: float = 0.01,
                 embedding_dim: int = 64,  # Configurable: [32, 64, 128]
                 history_length: int = 8,  # Extended: [8, 16, 32]
                 activation: str = 'gelu',  # 'gelu', 'swish', 'relu'
                 device: torch.device = torch.device('cpu'),
                 debug_mode: bool = False,
                 max_grad_norm: float = 1.0,
                 lr_bounds: tuple = (1e-6, 1e2),
                 momentum_bounds: tuple = (0.0, 0.99),
                 use_extended_features: bool = True,
                 normalize_features: bool = True,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True,
                 use_attention: bool = True,
                 use_positional_encoding: bool = True,
                 num_attention_heads: int = 4):
        """
        Initialize Opt2Vec optimizer.

        Args:
            parameters: Model parameters to optimize
            base_lr: Base learning rate
            embedding_dim: Dimension of optimization embeddings [32, 64, 128]
            history_length: Number of history steps to track [8, 16, 32]
            activation: Activation function ('gelu', 'swish', 'relu')
            device: Target device for computation
            debug_mode: Enable debug logging
            max_grad_norm: Maximum gradient norm for clipping
            lr_bounds: (min_lr, max_lr) bounds for learning rate
            momentum_bounds: (min_momentum, max_momentum) bounds for momentum
            use_extended_features: Whether to use extended feature set
            normalize_features: Whether to normalize features
            dropout: Dropout rate for regularization
            use_layer_norm: Whether to use LayerNorm
            use_attention: Whether to use attention mechanism
            use_positional_encoding: Whether to use positional encoding
            num_attention_heads: Number of attention heads
        """
        self.param_groups = [{'params': list(parameters)}]
        self.base_lr = base_lr
        self.embedding_dim = embedding_dim
        self.history_length = history_length
        self.device = device
        self.debug_mode = debug_mode
        self.max_grad_norm = max_grad_norm
        self.lr_bounds = lr_bounds
        self.momentum_bounds = momentum_bounds
        self.use_extended_features = use_extended_features

        # Get network configuration
        network_config = NetworkConfig.get_config(
            embedding_dim=embedding_dim,
            history_length=history_length,
            activation=activation
        )

        # Update config with user preferences
        network_config.update({
            'dropout': dropout,
            'use_layer_norm': use_layer_norm,
            'use_attention': use_attention,
            'use_positional_encoding': use_positional_encoding,
            'num_attention_heads': num_attention_heads
        })

        # Initialize the embedding network
        self.opt2vec_net = TinyOpt2VecNetwork(**network_config).to(device)

        # Enhanced history tracker
        self.history = LightweightOptimizationHistory(
            history_length=history_length,
            use_extended_features=use_extended_features,
            normalize_features=normalize_features
        )

        # Adaptation networks with residual connections
        self.lr_adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU() if activation == 'gelu' else (nn.SiLU() if activation == 'swish' else nn.ReLU()),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.GELU() if activation == 'gelu' else (nn.SiLU() if activation == 'swish' else nn.ReLU()),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid()
        ).to(device)

        self.momentum_adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU() if activation == 'gelu' else (nn.SiLU() if activation == 'swish' else nn.ReLU()),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.GELU() if activation == 'gelu' else (nn.SiLU() if activation == 'swish' else nn.ReLU()),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 4, 1),
            nn.Sigmoid()
        ).to(device)

        # Momentum buffers (only store when needed)
        self.momentum_buffers = {}
        self.step_count = 0

        # Debugging and monitoring
        self.debug_stats = {
            'grad_norms': [],
            'lr_multipliers': [],
            'momentum_factors': [],
            'embedding_stats': [],
            'update_magnitudes': [],
            'parameter_norms': [],
            'loss_values': [],
            'gradient_diversities': [],
            'loss_curvatures': [],
            'feature_stats': []
        }

        # Initialize adapters
        self._init_adapters()

        if debug_mode:
            logger.info(f"Enhanced Opt2Vec initialized with:")
            logger.info(f"  - Embedding dim: {embedding_dim}")
            logger.info(f"  - History length: {history_length}")
            logger.info(f"  - Activation: {activation}")
            logger.info(f"  - Extended features: {use_extended_features}")
            logger.info(f"  - Attention: {use_attention}")
            logger.info(f"  - Positional encoding: {use_positional_encoding}")

    def _init_adapters(self):
        """Initialize adaptation networks for stability."""
        for adapter in [self.lr_adapter, self.momentum_adapter]:
            for m in adapter.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.constant_(m.bias, 0)

    def zero_grad(self):
        """Zero gradients for all parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    def compute_grad_norm(self) -> float:
        """
        Compute total gradient norm across all parameters.

        Returns:
            Total gradient norm
        """
        total_norm = 0.0
        param_count = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

        grad_norm = (total_norm ** 0.5) if param_count > 0 else 0.0
        return grad_norm

    def compute_parameter_norm(self) -> float:
        """
        Compute total parameter norm across all parameters.

        Returns:
            Total parameter norm
        """
        total_norm = 0.0
        param_count = 0

        for group in self.param_groups:
            for p in group['params']:
                param_norm = p.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        param_norm = (total_norm ** 0.5) if param_count > 0 else 0.0
        return param_norm

    def compute_gradient_diversity(self) -> float:
        """
        Compute gradient diversity across all parameters.

        Returns:
            Gradient diversity score
        """
        gradients = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    gradients.append(p.grad.data)

        return self.history.compute_gradient_diversity(gradients)

    def compute_loss_curvature(self) -> float:
        """
        Compute loss curvature from recent history.

        Returns:
            Loss curvature score
        """
        if len(self.history.losses) < 3:
            return 0.0

        recent_losses = list(self.history.losses)[-3:]
        return self.history.compute_loss_curvature(recent_losses)

    def _check_numerical_stability(self, tensor: torch.Tensor, name: str) -> bool:
        """
        Check for numerical stability issues in a tensor.

        Args:
            tensor: Tensor to check
            name: Name for logging

        Returns:
            True if stable, False if issues detected
        """
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name}")
            return False
        if tensor.abs().max() > 1e6:
            logger.warning(f"Large values detected in {name}: max={tensor.abs().max().item()}")
            return False
        return True

    def _clip_gradients(self) -> float:
        """
        Apply gradient clipping and return the clipping factor.

        Returns:
            Clipping factor applied
        """
        total_norm = 0.0
        param_count = 0

        # Compute total norm
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

        total_norm = (total_norm ** 0.5) if param_count > 0 else 0.0

        # Apply clipping if needed
        clip_coef = min(self.max_grad_norm / (total_norm + 1e-6), 1.0)

        if clip_coef < 1.0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)

        return clip_coef

    def _validate_embedding(self, embedding: torch.Tensor) -> bool:
        """
        Validate embedding for collapse or instability.

        Args:
            embedding: Embedding tensor to validate

        Returns:
            True if embedding is valid, False otherwise
        """
        # Check for numerical issues
        if not self._check_numerical_stability(embedding, "embedding"):
            return False

        # Check for embedding collapse (low variance)
        embedding_std = embedding.std().item()
        if embedding_std < 1e-6:
            logger.warning(f"Embedding collapse detected: std={embedding_std}")
            return False

        # Check embedding norm
        embedding_norm = torch.norm(embedding).item()
        if embedding_norm < 1e-6 or embedding_norm > 1e3:
            logger.warning(f"Abnormal embedding norm: {embedding_norm}")
            return False

        return True

    def _compute_parameter_stats(self) -> Dict[str, float]:
        """
        Compute statistics about current parameters.

        Returns:
            Dictionary with parameter statistics
        """
        total_norm = 0.0
        param_count = 0
        max_param = 0.0
        min_param = float('inf')

        for group in self.param_groups:
            for p in group['params']:
                param_norm = p.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1
                max_param = max(max_param, p.data.abs().max().item())
                min_param = min(min_param, p.data.abs().min().item())

        return {
            'total_norm': (total_norm ** 0.5) if param_count > 0 else 0.0,
            'max_param': max_param,
            'min_param': min_param,
            'param_count': param_count
        }

    def step(self, loss: float) -> Optional[np.ndarray]:
        """
        Perform optimization step with enhanced features.

        Args:
            loss: Current loss value

        Returns:
            Optional array with adaptation statistics
        """
        # Compute extended features
        grad_norm = self.compute_grad_norm()
        param_norm = self.compute_parameter_norm()
        grad_diversity = self.compute_gradient_diversity()
        loss_curvature = self.compute_loss_curvature()

        # Get current learning rate
        current_lr = self.base_lr

        # Add step to history with extended features
        if self.use_extended_features:
            self.history.add_step(
                loss=loss,
                grad_norm=grad_norm,
                lr=current_lr,
                param_norm=param_norm,
                grad_diversity=grad_diversity,
                loss_curvature=loss_curvature
            )
        else:
            self.history.add_step(
                loss=loss,
                grad_norm=grad_norm,
                lr=current_lr
            )

        # Skip adaptation if not enough history
        if len(self.history) < 2:
            self._apply_updates(current_lr, 0.0)
            self.step_count += 1
            return None

        # Get history tensor
        history_tensor = self.history.get_history_tensor(self.device)
        history_tensor = history_tensor.unsqueeze(0)  # Add batch dimension

        # Generate embedding
        with torch.no_grad():
            embedding = self.opt2vec_net(history_tensor)

            if not self._validate_embedding(embedding):
                logger.warning("Invalid embedding detected, using fallback values")
                adaptive_lr = current_lr
                momentum_factor = 0.0
            else:
                # Compute adaptive learning rate and momentum
                lr_multiplier = self.lr_adapter(embedding).squeeze()
                momentum_factor = self.momentum_adapter(embedding).squeeze()

                # Apply bounds
                lr_multiplier = torch.clamp(lr_multiplier, 0.1, 10.0)
                momentum_factor = torch.clamp(momentum_factor, 0.0, 0.99)

                adaptive_lr = current_lr * lr_multiplier.item()
                momentum_factor = momentum_factor.item()

        # Apply gradient clipping
        clip_coef = self._clip_gradients()

        # Apply updates
        update_magnitude = self._apply_updates(adaptive_lr, momentum_factor)

        # Store debug statistics
        if self.debug_mode:
            self.debug_stats['grad_norms'].append(grad_norm)
            self.debug_stats['lr_multipliers'].append(adaptive_lr / current_lr)
            self.debug_stats['momentum_factors'].append(momentum_factor)
            self.debug_stats['update_magnitudes'].append(update_magnitude)
            self.debug_stats['parameter_norms'].append(param_norm)
            self.debug_stats['loss_values'].append(loss)
            self.debug_stats['gradient_diversities'].append(grad_diversity)
            self.debug_stats['loss_curvatures'].append(loss_curvature)
            self.debug_stats['feature_stats'].append(self.history.get_feature_stats())

            # Get embedding statistics
            embedding_stats = self.opt2vec_net.get_embedding_stats(history_tensor)
            self.debug_stats['embedding_stats'].append(embedding_stats)

        self.step_count += 1

        # Return adaptation statistics
        return np.array([
            adaptive_lr / current_lr,  # LR multiplier
            momentum_factor,           # Momentum factor
            grad_norm,                 # Gradient norm
            param_norm,                # Parameter norm
            grad_diversity,            # Gradient diversity
            loss_curvature,            # Loss curvature
            update_magnitude           # Update magnitude
        ])

    def _apply_updates(self, adaptive_lr: float, momentum_factor: float) -> float:
        """
        Apply parameter updates with adaptive learning rate and momentum.

        Args:
            adaptive_lr: Adaptive learning rate
            momentum_factor: Adaptive momentum factor

        Returns:
            Total magnitude of parameter updates
        """
        # Apply gradient clipping
        clip_coef = self._clip_gradients()

        total_update_magnitude = 0.0
        update_count = 0

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_state = id(p)

                # Check gradient stability
                if not self._check_numerical_stability(grad, f"gradient for param {param_state}"):
                    logger.warning(f"Unstable gradient detected for parameter {param_state}")
                    continue

                # Simple momentum update
                if param_state not in self.momentum_buffers:
                    self.momentum_buffers[param_state] = torch.zeros_like(p.data)

                buf = self.momentum_buffers[param_state]

                # Check momentum buffer stability
                if not self._check_numerical_stability(buf, f"momentum buffer {param_state}"):
                    logger.warning(f"Unstable momentum buffer detected for parameter {param_state}")
                    buf.zero_()

                buf.mul_(momentum_factor).add_(grad)

                # Compute update
                update = -adaptive_lr * buf

                # Check update stability
                if not self._check_numerical_stability(update, f"update for param {param_state}"):
                    logger.warning(f"Unstable update detected for parameter {param_state}")
                    continue

                # Apply update: θ_{t+1} = θ_t + u_t
                p.data.add_(update)

                # Track update magnitude
                update_magnitude = torch.norm(update).item()
                total_update_magnitude += update_magnitude ** 2
                update_count += 1

        return (total_update_magnitude ** 0.5) if update_count > 0 else 0.0

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """
        Get adaptation statistics.

        Returns:
            Dictionary with adaptation statistics
        """
        if not self.debug_stats['lr_multipliers']:
            return {}

        return {
            'lr_multiplier_mean': np.mean(self.debug_stats['lr_multipliers']),
            'lr_multiplier_std': np.std(self.debug_stats['lr_multipliers']),
            'momentum_factor_mean': np.mean(self.debug_stats['momentum_factors']),
            'momentum_factor_std': np.std(self.debug_stats['momentum_factors']),
            'grad_norm_mean': np.mean(self.debug_stats['grad_norms']),
            'grad_norm_std': np.std(self.debug_stats['grad_norms']),
            'update_magnitude_mean': np.mean(self.debug_stats['update_magnitudes']),
            'update_magnitude_std': np.std(self.debug_stats['update_magnitudes']),
            'parameter_norm_mean': np.mean(self.debug_stats['parameter_norms']),
            'parameter_norm_std': np.std(self.debug_stats['parameter_norms']),
            'gradient_diversity_mean': np.mean(self.debug_stats['gradient_diversities']),
            'gradient_diversity_std': np.std(self.debug_stats['gradient_diversities']),
            'loss_curvature_mean': np.mean(self.debug_stats['loss_curvatures']),
            'loss_curvature_std': np.std(self.debug_stats['loss_curvatures']),
            'total_steps': self.step_count,
            'history_length': len(self.history),
            'embedding_dim': self.embedding_dim,
            'use_extended_features': self.use_extended_features
        }

    def get_parameters(self) -> List[torch.Tensor]:
        """Get all trainable parameters of the optimizer networks."""
        params = []
        params.extend(self.opt2vec_net.parameters())
        params.extend(self.lr_adapter.parameters())
        params.extend(self.momentum_adapter.parameters())
        return params

    def get_debug_summary(self) -> Dict[str, Any]:
        """
        Get debug summary.

        Returns:
            Dictionary with debug information
        """
        return {
            'optimizer_config': {
                'embedding_dim': self.embedding_dim,
                'history_length': self.history_length,
                'base_lr': self.base_lr,
                'use_extended_features': self.use_extended_features,
                'device': str(self.device)
            },
            'network_config': {
                'activation': self.opt2vec_net.activation_name,
                'use_attention': self.opt2vec_net.use_attention,
                'use_positional_encoding': self.opt2vec_net.use_positional_encoding
            },
            'adaptation_stats': self.get_adaptation_stats(),
            'feature_stats': self.history.get_feature_stats(),
            'recent_embedding_stats': self.debug_stats['embedding_stats'][-1] if self.debug_stats['embedding_stats'] else None
        }
