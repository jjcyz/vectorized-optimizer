"""
Lightweight Opt2Vec optimizer with adaptive learning rate and momentum.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
import logging

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
                 debug_mode: bool = False):
        """
        Initialize LightweightOpt2Vec optimizer.

        Args:
            parameters: Model parameters to optimize
            base_lr: Base learning rate
            embedding_dim: Dimension of optimization embeddings
            history_length: Number of history steps to track
            device: Target device for computation
            debug_mode: Enable debug logging
        """
        self.param_groups = [{'params': list(parameters)}]
        self.base_lr = base_lr
        self.embedding_dim = embedding_dim
        self.device = device
        self.debug_mode = debug_mode

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

    def step(self, loss: float) -> Optional[np.ndarray]:
        """
        Perform optimization step with adaptive parameters.

        Args:
            loss: Current loss value

        Returns:
            Current embedding vector (if available)
        """
        # Calculate gradient norm efficiently
        grad_norm = self.compute_grad_norm()

        # Add to history
        self.history.add_step(loss, grad_norm, self.base_lr)

        # Get optimization embedding (only after we have some history)
        if self.step_count >= 2:  # Start adapting after a few steps
            history_tensor = self.history.get_history_tensor(self.device).unsqueeze(0)

            with torch.no_grad():
                embedding = self.opt2vec_net(history_tensor)

                # Get adaptive parameters
                lr_multiplier = self.lr_adapter(embedding).item()
                momentum_factor = self.momentum_adapter(embedding).item()

                # Scale to reasonable ranges
                adaptive_lr = self.base_lr * (0.5 + 1.0 * lr_multiplier)  # Range: [0.5*base, 1.5*base]
                momentum_factor = 0.1 + 0.8 * momentum_factor  # Range: [0.1, 0.9]

                if self.debug_mode:
                    logger.debug(f"Step {self.step_count}: LR={adaptive_lr:.6f}, "
                               f"Momentum={momentum_factor:.4f}, "
                               f"GradNorm={grad_norm:.4f}")
        else:
            # Use default values initially
            adaptive_lr = self.base_lr
            momentum_factor = 0.9
            embedding = torch.zeros(self.embedding_dim, device=self.device)

        # Apply updates
        self._apply_updates(adaptive_lr, momentum_factor)

        self.step_count += 1
        return embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding

    def _apply_updates(self, adaptive_lr: float, momentum_factor: float):
        """
        Apply parameter updates with adaptive learning rate and momentum.

        Args:
            adaptive_lr: Adaptive learning rate
            momentum_factor: Adaptive momentum factor
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                param_state = id(p)

                # Simple momentum update
                if param_state not in self.momentum_buffers:
                    self.momentum_buffers[param_state] = torch.zeros_like(p.data)

                buf = self.momentum_buffers[param_state]
                buf.mul_(momentum_factor).add_(grad)

                # Parameter update
                p.data.add_(buf, alpha=-adaptive_lr)

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
                'step_count': self.step_count
            }

        history_tensor = self.history.get_history_tensor(self.device).unsqueeze(0)

        with torch.no_grad():
            embedding = self.opt2vec_net(history_tensor)
            lr_multiplier = self.lr_adapter(embedding).item()
            momentum_factor = self.momentum_adapter(embedding).item()
            adaptive_lr = self.base_lr * (0.5 + 1.0 * lr_multiplier)
            momentum_factor = 0.1 + 0.8 * momentum_factor

        return {
            'lr_multiplier': lr_multiplier,
            'momentum_factor': momentum_factor,
            'adaptive_lr': adaptive_lr,
            'step_count': self.step_count,
            'embedding_stats': self.opt2vec_net.get_embedding_stats(history_tensor)
        }

    def get_parameters(self) -> List[torch.Tensor]:
        """Get all trainable parameters of the optimizer networks."""
        params = []
        params.extend(self.opt2vec_net.parameters())
        params.extend(self.lr_adapter.parameters())
        params.extend(self.momentum_adapter.parameters())
        return params
