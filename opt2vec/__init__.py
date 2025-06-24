"""
Opt2Vec: Meta-Learning Optimizer (Resource-Constrained Version)

A novel meta-learning optimizer that learns to optimize by creating embedding vectors
from optimization history. Optimized for resource-constrained environments.
"""

__version__ = "0.1.0"
__author__ = "Opt2Vec Team"

from .core.history import LightweightOptimizationHistory
from .core.network import TinyOpt2VecNetwork
from .core.optimizer import LightweightOpt2VecOptimizer
from .core.trainer import EfficientMetaLearningTrainer

__all__ = [
    "LightweightOptimizationHistory",
    "TinyOpt2VecNetwork",
    "LightweightOpt2VecOptimizer",
    "EfficientMetaLearningTrainer",
]
