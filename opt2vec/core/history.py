"""
Enhanced optimization history tracker with extended features.
"""

import torch
import numpy as np
from collections import deque
from typing import Optional, Dict, List, Tuple
import math


class LightweightOptimizationHistory:
    """
    Enhanced optimization history tracker with extended features.

    Features:
    - Extended history window: [8, 16, 32] steps
    - Additional features: parameter norms, gradient diversity, loss curvature
    - Temporal encoding for step information
    - Feature normalization and stability checks
    """

    def __init__(self,
                 history_length: int = 16,
                 use_extended_features: bool = True,
                 normalize_features: bool = True):
        """
        Initialize enhanced history tracker.

        Args:
            history_length: Number of recent steps to track (8, 16, 32)
            use_extended_features: Whether to use extended feature set
            normalize_features: Whether to normalize features
        """
        self.history_length = history_length
        self.use_extended_features = use_extended_features
        self.normalize_features = normalize_features

        # Core features
        self.losses = deque(maxlen=history_length)
        self.grad_norms = deque(maxlen=history_length)
        self.learning_rates = deque(maxlen=history_length)

        # Extended features
        if use_extended_features:
            self.param_norms = deque(maxlen=history_length)
            self.grad_diversities = deque(maxlen=history_length)
            self.loss_curvatures = deque(maxlen=history_length)
            self.steps = deque(maxlen=history_length)

        # Statistics for normalization
        self.feature_stats = {
            'loss': {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0},
            'grad_norm': {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0},
            'lr': {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0},
            'param_norm': {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0},
            'grad_diversity': {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0},
            'loss_curvature': {'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0}
        }

        self.step_count = 0

    def add_step(self,
                 loss: float,
                 grad_norm: float,
                 lr: float,
                 param_norm: Optional[float] = None,
                 grad_diversity: Optional[float] = None,
                 loss_curvature: Optional[float] = None):
        """
        Add a training step to history.

        Args:
            loss: Current loss value
            grad_norm: Current gradient norm
            lr: Current learning rate
            param_norm: Current parameter norm (optional)
            grad_diversity: Current gradient diversity (optional)
            loss_curvature: Current loss curvature (optional)
        """
        # Convert to Python floats to save memory
        self.losses.append(float(loss))
        self.grad_norms.append(float(grad_norm))
        self.learning_rates.append(float(lr))

        if self.use_extended_features:
            # Use provided values or compute defaults
            if param_norm is None:
                param_norm = 1.0  # Default parameter norm
            if grad_diversity is None:
                grad_diversity = 0.5  # Default gradient diversity
            if loss_curvature is None:
                loss_curvature = 0.0  # Default loss curvature

            self.param_norms.append(float(param_norm))
            self.grad_diversities.append(float(grad_diversity))
            self.loss_curvatures.append(float(loss_curvature))
            self.steps.append(self.step_count)

        self.step_count += 1

        # Update feature statistics for normalization
        if self.normalize_features and len(self.losses) > 1:
            self._update_feature_stats()

    def _update_feature_stats(self):
        """Update feature statistics for normalization."""
        if len(self.losses) < 2:
            return

        # Update loss statistics
        loss_array = np.array(list(self.losses))
        self.feature_stats['loss']['mean'] = float(np.mean(loss_array))
        self.feature_stats['loss']['std'] = float(np.std(loss_array) + 1e-8)
        self.feature_stats['loss']['min'] = float(np.min(loss_array))
        self.feature_stats['loss']['max'] = float(np.max(loss_array))

        # Update gradient norm statistics
        grad_array = np.array(list(self.grad_norms))
        self.feature_stats['grad_norm']['mean'] = float(np.mean(grad_array))
        self.feature_stats['grad_norm']['std'] = float(np.std(grad_array) + 1e-8)
        self.feature_stats['grad_norm']['min'] = float(np.min(grad_array))
        self.feature_stats['grad_norm']['max'] = float(np.max(grad_array))

        # Update learning rate statistics
        lr_array = np.array(list(self.learning_rates))
        self.feature_stats['lr']['mean'] = float(np.mean(lr_array))
        self.feature_stats['lr']['std'] = float(np.std(lr_array) + 1e-8)
        self.feature_stats['lr']['min'] = float(np.min(lr_array))
        self.feature_stats['lr']['max'] = float(np.max(lr_array))

        if self.use_extended_features:
            # Update extended feature statistics
            param_array = np.array(list(self.param_norms))
            self.feature_stats['param_norm']['mean'] = float(np.mean(param_array))
            self.feature_stats['param_norm']['std'] = float(np.std(param_array) + 1e-8)
            self.feature_stats['param_norm']['min'] = float(np.min(param_array))
            self.feature_stats['param_norm']['max'] = float(np.max(param_array))

            diversity_array = np.array(list(self.grad_diversities))
            self.feature_stats['grad_diversity']['mean'] = float(np.mean(diversity_array))
            self.feature_stats['grad_diversity']['std'] = float(np.std(diversity_array) + 1e-8)
            self.feature_stats['grad_diversity']['min'] = float(np.min(diversity_array))
            self.feature_stats['grad_diversity']['max'] = float(np.max(diversity_array))

            curvature_array = np.array(list(self.loss_curvatures))
            self.feature_stats['loss_curvature']['mean'] = float(np.mean(curvature_array))
            self.feature_stats['loss_curvature']['std'] = float(np.std(curvature_array) + 1e-8)
            self.feature_stats['loss_curvature']['min'] = float(np.min(curvature_array))
            self.feature_stats['loss_curvature']['max'] = float(np.max(curvature_array))

    def _normalize_feature(self, value: float, feature_name: str) -> float:
        """
        Normalize a feature value using stored statistics.

        Args:
            value: Feature value to normalize
            feature_name: Name of the feature

        Returns:
            Normalized feature value
        """
        if not self.normalize_features:
            return value

        stats = self.feature_stats[feature_name]

        # Z-score normalization with clipping
        normalized = (value - stats['mean']) / stats['std']
        normalized = np.clip(normalized, -3.0, 3.0)  # Clip to prevent extreme values

        return float(normalized)

    def get_history_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert history to tensor for neural network input.

        Args:
            device: Target device for tensor

        Returns:
            History tensor of shape [history_length, feature_dim]
        """
        if len(self.losses) < self.history_length:
            # Pad with the first available value or sensible defaults
            first_loss = self.losses[0] if self.losses else 0.0
            first_grad = self.grad_norms[0] if self.grad_norms else 0.0
            first_lr = self.learning_rates[0] if self.learning_rates else 0.001

            padding_length = self.history_length - len(self.losses)
            losses = [first_loss] * padding_length + list(self.losses)
            grad_norms = [first_grad] * padding_length + list(self.grad_norms)
            learning_rates = [first_lr] * padding_length + list(self.learning_rates)

            if self.use_extended_features:
                first_param = self.param_norms[0] if self.param_norms else 1.0
                first_diversity = self.grad_diversities[0] if self.grad_diversities else 0.5
                first_curvature = self.loss_curvatures[0] if self.loss_curvatures else 0.0
                first_step = self.steps[0] if self.steps else 0

                param_norms = [first_param] * padding_length + list(self.param_norms)
                grad_diversities = [first_diversity] * padding_length + list(self.grad_diversities)
                loss_curvatures = [first_curvature] * padding_length + list(self.loss_curvatures)
                steps = [first_step] * padding_length + list(self.steps)
        else:
            losses = list(self.losses)
            grad_norms = list(self.grad_norms)
            learning_rates = list(self.learning_rates)

            if self.use_extended_features:
                param_norms = list(self.param_norms)
                grad_diversities = list(self.grad_diversities)
                loss_curvatures = list(self.loss_curvatures)
                steps = list(self.steps)

        # Normalize features
        if self.normalize_features:
            losses = [self._normalize_feature(l, 'loss') for l in losses]
            grad_norms = [self._normalize_feature(g, 'grad_norm') for g in grad_norms]
            learning_rates = [self._normalize_feature(lr, 'lr') for lr in learning_rates]

            if self.use_extended_features:
                param_norms = [self._normalize_feature(p, 'param_norm') for p in param_norms]
                grad_diversities = [self._normalize_feature(d, 'grad_diversity') for d in grad_diversities]
                loss_curvatures = [self._normalize_feature(c, 'loss_curvature') for c in loss_curvatures]

        # Create feature vectors
        if self.use_extended_features:
            # Extended feature set: [loss, grad_norm, lr, param_norm, grad_diversity, loss_curvature, step]
            history = torch.tensor([
                [l, g, lr, p, d, c, s]
                for l, g, lr, p, d, c, s in zip(losses, grad_norms, learning_rates,
                                              param_norms, grad_diversities, loss_curvatures, steps)
            ], dtype=torch.float32, device=device)
        else:
            # Basic feature set: [loss, grad_norm, lr]
            history = torch.tensor([
                [l, g, lr] for l, g, lr in zip(losses, grad_norms, learning_rates)
            ], dtype=torch.float32, device=device)

        return history

    def compute_gradient_diversity(self, gradients: List[torch.Tensor]) -> float:
        """
        Compute gradient diversity metric.

        Args:
            gradients: List of gradient tensors

        Returns:
            Gradient diversity score
        """
        if not gradients:
            return 0.5

        # Flatten gradients and group by size
        flat_grads = [g.flatten() for g in gradients if g is not None]
        if not flat_grads:
            return 0.5

        # Group gradients by size
        grad_groups = {}
        for grad in flat_grads:
            size = grad.numel()
            if size not in grad_groups:
                grad_groups[size] = []
            grad_groups[size].append(grad)

        # Compute cosine similarities within each group
        similarities = []
        for size, grads in grad_groups.items():
            if len(grads) < 2:
                continue

            for i in range(len(grads)):
                for j in range(i + 1, len(grads)):
                    cos_sim = torch.cosine_similarity(grads[i], grads[j], dim=0)
                    similarities.append(cos_sim.item())

        if not similarities:
            return 0.5

        # Diversity is 1 - average similarity
        diversity = 1.0 - np.mean(similarities)
        return float(np.clip(diversity, 0.0, 1.0))

    def compute_loss_curvature(self, losses: List[float]) -> float:
        """
        Compute loss curvature metric.

        Args:
            losses: List of recent loss values

        Returns:
            Loss curvature score
        """
        if len(losses) < 3:
            return 0.0

        # Compute second derivative approximation
        recent_losses = losses[-3:]
        curvature = recent_losses[2] - 2 * recent_losses[1] + recent_losses[0]

        # Normalize by the average loss magnitude
        avg_loss = np.mean(np.abs(recent_losses))
        if avg_loss > 1e-8:
            curvature = curvature / avg_loss

        return float(np.clip(curvature, -1.0, 1.0))

    def clear(self):
        """Clear all history."""
        self.losses.clear()
        self.grad_norms.clear()
        self.learning_rates.clear()

        if self.use_extended_features:
            self.param_norms.clear()
            self.grad_diversities.clear()
            self.loss_curvatures.clear()
            self.steps.clear()

        self.step_count = 0

    def __len__(self) -> int:
        """Return number of recorded steps."""
        return len(self.losses)

    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current feature statistics."""
        return self.feature_stats.copy()
