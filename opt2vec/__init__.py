"""
Opt2Vec: Meta-Learning Optimizer (Resource-Constrained Version)

A novel meta-learning optimizer that learns to optimize by creating embedding vectors
from optimization history. Optimized for resource-constrained environments.

Enhanced version includes:
- Configurable embedding dimensions: [32, 64, 128]
- Extended history windows: [8, 16, 32] steps
- Residual connections and normalization
- Attention mechanism for history encoding
- Extended feature set with parameter norms, gradient diversity, loss curvature
- Multiple activation functions (Swish, GELU, ReLU)
"""
from .core.history import LightweightOptimizationHistory
from .core.network import TinyOpt2VecNetwork, NetworkConfig
from .core.optimizer import LightweightOpt2VecOptimizer
from .core.trainer import EfficientMetaLearningTrainer

__all__ = [
    # Core components
    "LightweightOptimizationHistory",
    "TinyOpt2VecNetwork",
    "LightweightOpt2VecOptimizer",
    "EfficientMetaLearningTrainer",
    "NetworkConfig",
]
