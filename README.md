# Opt2Vec: Meta-Learning Optimizer (Resource-Constrained Version)

A novel meta-learning optimizer that learns to optimize by creating embedding vectors from optimization history. This implementation is optimized for **Google Colab** and **low-spec machines** (like Mac with 1.6GHz i5, 8GB RAM).

## 🎯 Project Overview

Opt2Vec learns "intuition" about how to adjust learning rates and momentum based on past training dynamics. It uses a lightweight neural network to create embeddings from optimization history and adaptively modifies optimization parameters.

### Key Features
- **Memory Efficient**: <100MB memory usage, designed for resource-constrained environments
- **Fast Training**: 2-5 minutes on Colab, 5-15 minutes on Mac
- **Lightweight Architecture**: MLP-based instead of Transformer (90% memory reduction)
- **Adaptive Optimization**: Jointly learns learning rate and momentum adaptation

## 🏗️ System Architecture

```
Training History → TinyOpt2VecNetwork → Embedding Vector → Adaptive Updates → Model Training
[loss, grad_norm, lr] → Simple MLP → [16-dim vector] → [lr_scale, momentum] → [parameter updates]
```

## 🚀 Enhanced Features

- **Configurable Embedding Dimensions:** [32, 64, 128]
- **Residual Connections & Normalization:** LayerNorm/BatchNorm for stability
- **Dropout & Multiple Activations:** Swish, GELU, ReLU
- **Lightweight Attention & Positional Encoding:** For history encoding
- **Extended History Windows:** [8, 16, 32] steps
- **Extended Feature Set:** Parameter norms, gradient diversity, loss curvature, temporal step
- **Systematic Experimentation:** Easy config and experiment runner

## 🛠️ Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from opt2vec.core.optimizer import LightweightOpt2VecOptimizer
from opt2vec.core.network import TinyOpt2VecNetwork
import torch

model = torch.nn.Linear(10, 1)
criterion = torch.nn.MSELoss()

optimizer = LightweightOpt2VecOptimizer(
    model.parameters(),
    base_lr=0.01,
    embedding_dim=64,           # [32, 64, 128]
    history_length=16,          # [8, 16, 32]
    activation='gelu',          # 'gelu', 'swish', 'relu'
    use_extended_features=True, # Use parameter norms, gradient diversity, etc.
    use_attention=True,         # Use attention mechanism
    use_positional_encoding=True, # Use positional encoding
    device=torch.device('cpu')
)

for epoch in range(10):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(loss.item())
```

### Meta-Learning Example

```python
from opt2vec.core.trainer import EfficientMetaLearningTrainer

# Initialize meta-trainer
trainer = EfficientMetaLearningTrainer(device='cpu')

# Meta-train the optimizer
trainer.meta_train(
    num_meta_steps=50,
    num_tasks_per_step=3,
    inner_steps=5
)
```

## 🧑‍🔬 Debugging & Monitoring
- **Comprehensive logging:** Track gradient norms, learning rates, parameter/embedding stats, and more
- **Gradient clipping & stability checks:** Prevents explosion/instability
- **Debug summaries:** `optimizer.get_debug_summary()` for stepwise stats
- **Visualization:** 6-panel plots for loss, learning rate, momentum, norms, and diversity

## 🧪 Experimentation
- **Configurable via arguments or config objects** (see `NetworkConfig`)
- **Experiment runner** for systematic sweeps (see `ENHANCED_FEATURES.md`)
- **Compare with Adam/SGD** by swapping optimizers in your training loop

## 📚 References & Further Info
- See `ENHANCED_FEATURES.md` for architecture details and advanced usage
- See `DEBUGGING_GUIDE.md` for debugging and stability monitoring
- See `example_enhanced_usage.py` for a full demonstration

## 📊 Performance Expectations

### Resource Usage
- **Memory**: <500MB peak usage
- **Training Time**: 2-5 minutes (Colab), 5-15 minutes (Mac)
- **Dataset Size**: 200-1000 samples (instead of full MNIST)
- **Convergence**: Clear differences visible in 10-20 steps

### Baseline Comparison
Opt2Vec typically matches or exceeds Adam performance on MNIST while providing adaptive optimization behavior.

## 🧠 Core Components

### 1. LightweightOptimizationHistory
Tracks last 5 steps of [loss, gradient_norm, learning_rate] using Python floats for memory efficiency.

### 2. TinyOpt2VecNetwork
Simple MLP architecture:
- Input: [batch_size, 5, 3] - sequence of optimization history
- Hidden layer: 32 neurons with ReLU
- Output: [batch_size, 16] embedding
- Memory: ~10KB parameters vs 100KB+ for Transformer

### 3. LightweightOpt2VecOptimizer
- `lr_adapter`: Single linear layer embedding → lr multiplier
- `momentum_adapter`: Single linear layer embedding → momentum
- Efficient momentum buffers
- Update rule: `θₜ₊₁ = θₜ - adaptive_lr * momentum_buffer`

## 🔬 Experiments

### MNIST Baseline Comparison
```bash
python experiments/mnist_baseline.py
```

### Meta-Learning Training
```bash
python3.11 experiments/meta_learning.py
```

### Embedding Analysis
```bash
python experiments/analysis.py
```

## 📈 Results

The system provides:
- **Adaptive Optimization**: Learning rate and momentum automatically adjust based on training dynamics
- **Memory Efficiency**: 90% reduction in memory usage compared to Transformer-based approaches
- **Fast Convergence**: Competitive performance with standard optimizers
- **Interpretable Embeddings**: Visualization of optimization state evolution

## 🎛️ Configuration

Key hyperparameters:
- `embedding_dim`: 16 (reduced from 32 for memory efficiency)
- `history_length`: 5 (reduced from 10)
- `base_lr`: 0.01 (increased for faster convergence)
- `num_tasks_per_step`: 3 (reduced for memory efficiency)

## 🐛 Troubleshooting

### Memory Issues
- Reduce `num_tasks_per_step` or `inner_steps`
- Use smaller `embedding_dim` or `history_length`
- Enable gradient checkpointing for large models

### Training Instabilities
- Check gradient norms and clip if necessary
- Verify learning rate ranges are reasonable
- Monitor embedding diversity to prevent collapse

## 📚 Research Connections

This work builds on:
- **Learning to Learn by Gradient Descent by Gradient Descent** (Andrychowicz et al., 2016)
- **Optimization as a Model for Few-Shot Learning** (Ravi & Larochelle, 2017)
- **Meta-Learning with Implicit Gradients** (Rajeswaran et al., 2019)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 🎯 Success Metrics

- [x] **Functionality**: Opt2Vec trains without crashing, produces embeddings
- [x] **Performance**: Matches or exceeds Adam baseline on MNIST
- [x] **Memory Efficiency**: <500MB peak memory usage
- [x] **Speed**: Fast training suitable for resource-constrained environments
- [ ] **Generalization**: Works across different tasks and architectures
- [ ] **Interpretability**: Embedding visualizations show meaningful patterns
- [ ] **Stability**: Training converges consistently across multiple runs
