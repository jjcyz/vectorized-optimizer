# Opt2Vec: Meta-Learning Optimizer

A lightweight meta-learning optimizer that learns to optimize by creating embedding vectors from optimization history. Designed for resource-constrained environments.

## Overview

Opt2Vec learns to adapt learning rates and momentum based on training dynamics by encoding optimization history into neural embeddings. It uses a lightweight architecture optimized for low-memory environments.

## Key Features

- **Memory Efficient**: <500MB memory usage
- **Lightweight Architecture**: MLP-based instead of Transformer
- **Adaptive Optimization**: Jointly learns learning rate and momentum adaptation
- **Configurable**: Multiple embedding dimensions, history lengths, and activation functions
- **Extended Features**: Parameter norms, gradient diversity, loss curvature tracking

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from opt2vec import LightweightOpt2VecOptimizer

model = torch.nn.Linear(10, 1)
criterion = torch.nn.MSELoss()

optimizer = LightweightOpt2VecOptimizer(
    model.parameters(),
    base_lr=0.01,
    embedding_dim=64,
    history_length=16,
    activation='gelu',
    use_extended_features=True
)

for epoch in range(10):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(loss.item())
```

### Meta-Learning Training

```python
from opt2vec import EfficientMetaLearningTrainer

trainer = EfficientMetaLearningTrainer(device='cpu')
trainer.meta_train(num_meta_steps=50, num_tasks_per_step=3, inner_steps=5)
```

## Examples

- `example_enhanced_usage.py` - Complete training example with visualization
- `test_meta_learning.py` - Meta-learning training example
- `test_debug_optimizer_simple.py` - Debugging and testing utilities

## Experiments

Run experiments from the `opt2vec/experiments/` directory:

- `mnist_baseline.py` - Compare with Adam/SGD on MNIST
- `meta_learning.py` - Meta-learning training
- `analysis.py` - Embedding analysis and visualization
- `enhanced_experiment_runner.py` - Systematic configuration sweeps

## Architecture

```
Training History → TinyOpt2VecNetwork → Embedding Vector → Adaptive Updates
[loss, grad_norm, lr] → MLP → [embedding] → [lr_scale, momentum] → [updates]
```

## Requirements

- Python 3.8+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy, matplotlib, scikit-learn

## Performance

- **Memory**: <500MB peak usage
- **Training Time**: 2-5 minutes (Colab), 5-15 minutes (Mac)
- **Convergence**: Competitive (matches) with Adam on MNIST classification
