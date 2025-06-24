"""
Lightweight optimization history tracker for Opt2Vec.
"""

import torch
from collections import deque
from typing import Optional


class LightweightOptimizationHistory:
    """
    Memory-efficient optimization history tracker.

    Tracks last N steps of [loss, gradient_norm, learning_rate] using Python floats
    instead of tensors for storage efficiency.
    """

    def __init__(self, history_length: int = 5):
        """
        Initialize history tracker.

        Args:
            history_length: Number of recent steps to track
        """
        self.history_length = history_length
        self.losses = deque(maxlen=history_length)
        self.grad_norms = deque(maxlen=history_length)
        self.learning_rates = deque(maxlen=history_length)

    def add_step(self, loss: float, grad_norm: float, lr: float):
        """
        Add a training step to history.

        Args:
            loss: Current loss value
            grad_norm: Current gradient norm
            lr: Current learning rate
        """
        # Convert to Python floats to save memory
        self.losses.append(float(loss))
        self.grad_norms.append(float(grad_norm))
        self.learning_rates.append(float(lr))

    def get_history_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Convert history to tensor for neural network input.

        Args:
            device: Target device for tensor

        Returns:
            History tensor of shape [history_length, 3]
        """
        if len(self.losses) < self.history_length:
            # Pad with the first available value or zero
            first_loss = self.losses[0] if self.losses else 0.0
            first_grad = self.grad_norms[0] if self.grad_norms else 0.0
            first_lr = self.learning_rates[0] if self.learning_rates else 0.001

            padding_length = self.history_length - len(self.losses)
            losses = [first_loss] * padding_length + list(self.losses)
            grad_norms = [first_grad] * padding_length + list(self.grad_norms)
            learning_rates = [first_lr] * padding_length + list(self.learning_rates)
        else:
            losses = list(self.losses)
            grad_norms = list(self.grad_norms)
            learning_rates = list(self.learning_rates)

        # Normalize values to prevent numerical issues
        if losses:
            max_loss = max(max(losses), 1e-6)
            losses = [l / max_loss for l in losses]
        if grad_norms:
            max_grad = max(max(grad_norms), 1e-6)
            grad_norms = [g / max_grad for g in grad_norms]

        # Stack into tensor
        history = torch.tensor([
            [l, g, lr] for l, g, lr in zip(losses, grad_norms, learning_rates)
        ], dtype=torch.float32, device=device)

        return history

    def clear(self):
        """Clear all history."""
        self.losses.clear()
        self.grad_norms.clear()
        self.learning_rates.clear()

    def __len__(self) -> int:
        """Return number of recorded steps."""
        return len(self.losses)
