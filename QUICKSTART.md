# Opt2Vec Quick Start Guide

Get up and running with Opt2Vec in minutes!

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/jjcyz/vectorized-optimizer
cd opt2vec

# Install dependencies
pip3.11 install -r requirements.txt

# Or install in development mode
pip3.11 install -e .
```

## ⚡ Quick Test

Run the test script to verify everything works:

```bash
python3.11 test_opt2vec.py
```

You should see:
```
🎉 All tests passed! Opt2Vec is working correctly.
```

## 🎯 Basic Usage

### Simple Example

```python
import torch
import torch.nn as nn
from opt2vec.core.optimizer import LightweightOpt2VecOptimizer

# Create a model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Create data
x = torch.randn(100, 10)
y = torch.randn(100, 1)

# Initialize Opt2Vec optimizer
optimizer = LightweightOpt2VecOptimizer(
    model.parameters(),
    base_lr=0.01,
    embedding_dim=16,
    history_length=5
)

# Training loop
criterion = nn.MSELoss()
for step in range(50):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # Opt2Vec automatically adapts learning rate and momentum
    embedding = optimizer.step(loss.item())

    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")
```

### Comparison with Standard Optimizers

```python
import torch.optim as optim

# Test different optimizers
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'Adam': optim.Adam(model.parameters(), lr=0.01),
    'Opt2Vec': LightweightOpt2VecOptimizer(
        model.parameters(), base_lr=0.01, embedding_dim=16, history_length=5
    )
}

# Train with each optimizer and compare results
```

## 🧠 Meta-Learning Example

```python
from opt2vec.core.trainer import EfficientMetaLearningTrainer

# Initialize meta-trainer
trainer = EfficientMetaLearningTrainer(device='cpu')

# Run meta-training
results = trainer.meta_train(
    num_meta_steps=50,
    num_tasks_per_step=3,
    meta_lr=1e-3,
    inner_steps=5
)

print(f"Meta-loss improvement: {results['meta_losses'][0] - results['meta_losses'][-1]:.4f}")
```

## 📊 Experiments

### MNIST Baseline Comparison

```bash
python3.11 -m opt2vec.experiments.mnist_baseline
```

### Meta-Learning Training

```bash
python3.11 -m opt2vec.experiments.meta_learning
```

### Embedding Analysis

```bash
python3.11 -m opt2vec.experiments.analysis
```

## 🎨 Visualization

Opt2Vec includes built-in visualization tools:

```python
from opt2vec.utils.visualization import plot_training_curves, plot_embedding_evolution

# Plot training curves
plot_training_curves({'Opt2Vec': losses}, "Training Progress")

# Plot embedding evolution
plot_embedding_evolution(embeddings, losses, "Embedding Evolution")
```

## 🔧 Configuration

Key hyperparameters to tune:

```python
optimizer = LightweightOpt2VecOptimizer(
    parameters,
    base_lr=0.01,           # Base learning rate
    embedding_dim=16,       # Embedding dimension (16, 32, 64)
    history_length=5,       # History steps (5, 10, 20)
    device='cpu'            # Device for computation
)
```

## 📈 Performance Expectations

### Resource Usage
- **Memory**: <500MB peak usage
- **Training Time**: 2-5 minutes (Colab), 5-15 minutes (Mac)
- **Dataset Size**: 200-1000 samples (instead of full MNIST)

### Performance
- **MNIST**: Typically matches or exceeds Adam performance
- **Convergence**: Clear differences visible in 10-20 steps
- **Adaptation**: Automatic learning rate and momentum adjustment

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Make sure you've installed the package
   ```bash
   pip install -e .
   ```

2. **Memory Issues**: Reduce batch size or use smaller models
   ```python
   # Use smaller embedding dimension
   embedding_dim=8  # instead of 16
   ```

3. **Slow Training**: Use GPU if available
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

### Getting Help

- Check the test script: `python test_opt2vec.py`
- Run examples: `python3.11 example_usage.py`
- Check memory usage: `from opt2vec.utils.memory import get_memory_usage`

## 🎯 Next Steps

1. **Run Examples**: `python example_usage.py`
2. **Try Experiments**: Run MNIST baseline comparison
3. **Explore Meta-Learning**: Train your own Opt2Vec optimizer
4. **Analyze Embeddings**: Understand optimization behavior
5. **Customize**: Adapt for your specific use case

