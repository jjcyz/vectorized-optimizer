"""
Lightweight Opt2Vec optimizer with adaptive learning rate and momentum.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import warnings

from .history import LightweightOptimizationHistory
from .network import TinyOpt2VecNetwork

logger = logging.getLogger(__name__)


class LightweightOpt2VecOptimizer:
    """
    Memory-efficient version of the Opt2Vec optimizer.

    Uses embeddings to adaptively modify optimization behavior:
    - Learning rate adaptation based on optimization history
    - Momentum adaptation based on optimization history
    - Efficient memory usage with Python float storage
    """

    def __init__(self,
                 parameters,
                 base_lr: float = 0.01,  # Increased base LR for faster convergence
                 embedding_dim: int = 16,
                 history_length: int = 5,
                 device: torch.device = torch.device('cpu'),
                 debug_mode: bool = False,
                 max_grad_norm: float = 1.0,
                 lr_bounds: tuple = (1e-6, 1e2),
                 momentum_bounds: tuple = (0.0, 0.99)):
        """
        Initialize LightweightOpt2Vec optimizer.

        Args:
            parameters: Model parameters to optimize
            base_lr: Base learning rate
            embedding_dim: Dimension of optimization embeddings
            history_length: Number of history steps to track
            device: Target device for computation
            debug_mode: Enable debug logging
            max_grad_norm: Maximum gradient norm for clipping
            lr_bounds: (min_lr, max_lr) bounds for learning rate
            momentum_bounds: (min_momentum, max_momentum) bounds for momentum
        """
        self.param_groups = [{'params': list(parameters)}]
        self.base_lr = base_lr
        self.embedding_dim = embedding_dim
        self.device = device
        self.debug_mode = debug_mode
        self.max_grad_norm = max_grad_norm
        self.lr_bounds = lr_bounds
        self.momentum_bounds = momentum_bounds

        # Initialize the embedding network
        self.opt2vec_net = TinyOpt2VecNetwork(
            embedding_dim=embedding_dim,
            history_length=history_length
        ).to(device)

        # History tracker
        self.history = LightweightOptimizationHistory(history_length)

        # Simple adaptation networks
        self.lr_adapter = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        ).to(device)

        self.momentum_adapter = nn.Sequential(
            nn.Linear(embedding_dim, 1),
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
            'loss_values': []
        }

        # Initialize adapters
        self._init_adapters()

    def _init_adapters(self):
        """Initialize adaptation networks for stability."""
        for m in self.lr_adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)

        for m in self.momentum_adapter.modules():
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
        Perform optimization step with adaptive parameters.

        Args:
            loss: Current loss value

        Returns:
            Current embedding vector (if available)
        """
        # Store loss for debugging
        self.debug_stats['loss_values'].append(loss)

        # Calculate gradient norm efficiently
        grad_norm = self.compute_grad_norm()
        self.debug_stats['grad_norms'].append(grad_norm)

        # Add to history
        self.history.add_step(loss, grad_norm, self.base_lr)

        # Get optimization embedding (only after we have some history)
        if self.step_count >= 2:  # Start adapting after a few steps
            history_tensor = self.history.get_history_tensor(self.device).unsqueeze(0)

            with torch.no_grad():
                embedding = self.opt2vec_net(history_tensor)

                # Validate embedding
                if not self._validate_embedding(embedding):
                    logger.warning("Invalid embedding detected, using default values")
                    adaptive_lr = self.base_lr
                    momentum_factor = 0.9
                    embedding = torch.zeros(self.embedding_dim, device=self.device)
                else:
                    # Get adaptive parameters
                    lr_multiplier = self.lr_adapter(embedding).item()
                    momentum_factor = self.momentum_adapter(embedding).item()

                    # Apply bounds checking
                    lr_multiplier = np.clip(lr_multiplier, 0.0, 1.0)
                    momentum_factor = np.clip(momentum_factor, 0.0, 1.0)

                    # Scale to reasonable ranges with bounds
                    min_lr, max_lr = self.lr_bounds
                    min_momentum, max_momentum = self.momentum_bounds

                    adaptive_lr = min_lr + (max_lr - min_lr) * lr_multiplier
                    momentum_factor = min_momentum + (max_momentum - min_momentum) * momentum_factor

                    # Store debugging info
                    self.debug_stats['lr_multipliers'].append(lr_multiplier)
                    self.debug_stats['momentum_factors'].append(momentum_factor)
                    self.debug_stats['embedding_stats'].append({
                        'mean': embedding.mean().item(),
                        'std': embedding.std().item(),
                        'norm': torch.norm(embedding).item()
                    })

                if self.debug_mode:
                    logger.debug(f"Step {self.step_count}: LR={adaptive_lr:.6f}, "
                               f"Momentum={momentum_factor:.4f}, "
                               f"GradNorm={grad_norm:.4f}")
        else:
            # Use default values initially
            adaptive_lr = self.base_lr
            momentum_factor = 0.9
            embedding = torch.zeros(self.embedding_dim, device=self.device)
            self.debug_stats['lr_multipliers'].append(1.0)
            self.debug_stats['momentum_factors'].append(0.9)

        # Apply updates with comprehensive monitoring
        update_magnitude = self._apply_updates(adaptive_lr, momentum_factor)
        self.debug_stats['update_magnitudes'].append(update_magnitude)

        # Track parameter statistics
        param_stats = self._compute_parameter_stats()
        self.debug_stats['parameter_norms'].append(param_stats['total_norm'])

        self.step_count += 1
        return embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding

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
        Get statistics about current adaptation behavior.

        Returns:
            Dictionary with adaptation statistics
        """
        if self.step_count < 2:
            return {
                'lr_multiplier': 1.0,
                'momentum_factor': 0.9,
                'adaptive_lr': self.base_lr,
                'step_count': self.step_count,
                'debug_stats': self.debug_stats
            }

        history_tensor = self.history.get_history_tensor(self.device).unsqueeze(0)

        with torch.no_grad():
            embedding = self.opt2vec_net(history_tensor)
            lr_multiplier = self.lr_adapter(embedding).item()
            momentum_factor = self.momentum_adapter(embedding).item()

            # Apply bounds
            lr_multiplier = np.clip(lr_multiplier, 0.0, 1.0)
            momentum_factor = np.clip(momentum_factor, 0.0, 1.0)

            min_lr, max_lr = self.lr_bounds
            min_momentum, max_momentum = self.momentum_bounds

            adaptive_lr = min_lr + (max_lr - min_lr) * lr_multiplier
            momentum_factor = min_momentum + (max_momentum - min_momentum) * momentum_factor

        return {
            'lr_multiplier': lr_multiplier,
            'momentum_factor': momentum_factor,
            'adaptive_lr': adaptive_lr,
            'step_count': self.step_count,
            'embedding_stats': self.opt2vec_net.get_embedding_stats(history_tensor),
            'debug_stats': self.debug_stats
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
        Get a comprehensive debug summary of the optimizer state.

        Returns:
            Dictionary with debug information
        """
        if not self.debug_stats['grad_norms']:
            return {'message': 'No debug data available yet'}

        return {
            'step_count': self.step_count,
            'recent_grad_norms': self.debug_stats['grad_norms'][-10:],
            'recent_lr_multipliers': self.debug_stats['lr_multipliers'][-10:],
            'recent_momentum_factors': self.debug_stats['momentum_factors'][-10:],
            'recent_update_magnitudes': self.debug_stats['update_magnitudes'][-10:],
            'recent_parameter_norms': self.debug_stats['parameter_norms'][-10:],
            'recent_losses': self.debug_stats['loss_values'][-10:],
            'embedding_stats': self.debug_stats['embedding_stats'][-5:] if self.debug_stats['embedding_stats'] else [],
            'grad_norm_stats': {
                'mean': np.mean(self.debug_stats['grad_norms']),
                'std': np.std(self.debug_stats['grad_norms']),
                'max': np.max(self.debug_stats['grad_norms']),
                'min': np.min(self.debug_stats['grad_norms'])
            },
            'lr_multiplier_stats': {
                'mean': np.mean(self.debug_stats['lr_multipliers']),
                'std': np.std(self.debug_stats['lr_multipliers']),
                'max': np.max(self.debug_stats['lr_multipliers']),
                'min': np.min(self.debug_stats['lr_multipliers'])
            }
        }
