
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import copy
from collections import deque
import logging
import time
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory management utilities
def clear_memory():
    """Clear GPU/CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2  # MB

class LightweightOptimizationHistory:
    """Memory-efficient optimization history tracker"""

    def __init__(self, history_length: int = 5):  # Reduced from 10
        self.history_length = history_length
        self.losses = deque(maxlen=history_length)
        self.grad_norms = deque(maxlen=history_length)
        self.learning_rates = deque(maxlen=history_length)

    def add_step(self, loss: float, grad_norm: float, lr: float):
        """Add a training step to history"""
        # Convert to Python floats to save memory
        self.losses.append(float(loss))
        self.grad_norms.append(float(grad_norm))
        self.learning_rates.append(float(lr))

    def get_history_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert history to tensor for neural network input"""
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

class TinyOpt2VecNetwork(nn.Module):
    """
    Ultra-lightweight version of Opt2Vec network for resource-constrained environments
    Uses simple MLP instead of Transformer to reduce memory and computation
    """

    def __init__(self,
                 input_dim: int = 3,  # [loss, grad_norm, lr]
                 embedding_dim: int = 16,  # Reduced from 32
                 history_length: int = 5):  # Reduced from 10
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.history_length = history_length

        # Simple MLP instead of Transformer
        flattened_input = input_dim * history_length
        self.network = nn.Sequential(
            nn.Linear(flattened_input, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, embedding_dim),
            nn.Tanh()
        )

        # Initialize weights for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: [batch_size, history_length, input_dim]
        Returns:
            embedding: [batch_size, embedding_dim]
        """
        batch_size = history.shape[0]

        # Flatten history sequence
        x = history.view(batch_size, -1)  # [batch, history_length * input_dim]

        # Apply network
        embedding = self.network(x)

        return embedding

class LightweightOpt2VecOptimizer:
    """
    Memory-efficient version of the Opt2Vec optimizer
    """

    def __init__(self,
                 parameters,
                 base_lr: float = 0.01,  # Increased base LR for faster convergence
                 embedding_dim: int = 16,
                 history_length: int = 5,
                 device: torch.device = torch.device('cpu')):

        self.param_groups = [{'params': list(parameters)}]
        self.base_lr = base_lr
        self.embedding_dim = embedding_dim
        self.device = device

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

    def zero_grad(self):
        """Zero gradients for all parameters"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    def step(self, loss: float):
        """Perform optimization step"""
        # Calculate gradient norm efficiently
        total_norm = 0.0
        param_count = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1

        grad_norm = (total_norm ** 0.5) if param_count > 0 else 0.0

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
        else:
            # Use default values initially
            adaptive_lr = self.base_lr
            momentum_factor = 0.9
            embedding = torch.zeros(self.embedding_dim, device=self.device)

        # Apply updates
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

        self.step_count += 1
        return embedding.detach().cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding

# Tiny CNN for MNIST (much smaller than original)
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 2)  # Reduced channels and larger stride
        self.conv2 = nn.Conv2d(8, 16, 3, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)  # Much smaller
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class EfficientMetaLearningTrainer:
    """
    Memory-efficient meta-learning trainer
    """

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device

    def create_tiny_task(self, task_size: int = 50):  # Much smaller tasks
        """Create tiny optimization tasks"""
        # Simple 2D quadratic: f(x) = (x-a)^2 + (y-b)^2
        dim = 2  # Reduced from 10
        X = torch.randn(task_size, dim, device=self.device) * 2
        a = torch.randn(dim, device=self.device)

        # Compute targets
        X_centered = X - a.unsqueeze(0)
        y = torch.sum(X_centered ** 2, dim=1)

        return X, y, {'center': a}

    def quick_inner_loop(self, model, data, targets, optimizer, steps: int = 5):  # Reduced steps
        """Quick inner loop training"""
        model.train()
        losses = []

        for step in range(steps):
            optimizer.zero_grad()

            pred = model(data).squeeze()
            loss = F.mse_loss(pred, targets)

            loss.backward()

            if isinstance(optimizer, LightweightOpt2VecOptimizer):
                embedding = optimizer.step(loss.item())
            else:
                optimizer.step()

            losses.append(loss.item())

            # Early stopping if loss is very small
            if loss.item() < 1e-4:
                break

        return losses

    def meta_train_step(self, opt2vec_components, meta_opt, num_tasks: int = 3):
        """Efficient meta-training step"""
        meta_losses = []

        for task_idx in range(num_tasks):
            # Create tiny task
            data, targets, _ = self.create_tiny_task(task_size=30)

            # Create tiny model
            model = nn.Linear(2, 1).to(self.device)

            # Create optimizer with shared components
            task_optimizer = LightweightOpt2VecOptimizer(
                model.parameters(),
                base_lr=0.05,
                device=self.device
            )

            # Share the networks
            task_optimizer.opt2vec_net = opt2vec_components['opt2vec_net']
            task_optimizer.lr_adapter = opt2vec_components['lr_adapter']
            task_optimizer.momentum_adapter = opt2vec_components['momentum_adapter']

            # Quick inner loop
            losses = self.quick_inner_loop(model, data, targets, task_optimizer, steps=5)

            # Meta-objective
            final_loss = losses[-1] if losses else 1.0
            meta_losses.append(final_loss)

            # Clear memory
            del model, task_optimizer, data, targets

        # Outer loop update
        if meta_losses:
            meta_loss = torch.tensor(meta_losses, device=self.device).mean()

            meta_opt.zero_grad()
            meta_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(opt2vec_components['opt2vec_net'].parameters()) +
                list(opt2vec_components['lr_adapter'].parameters()) +
                list(opt2vec_components['momentum_adapter'].parameters()),
                max_norm=1.0
            )

            meta_opt.step()

            return meta_loss.item(), meta_losses

        return 0.0, []

def create_tiny_mnist_dataset(num_samples: int = 1000):
    """Create a tiny MNIST subset for testing"""
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load full dataset
        full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

        # Create small subset
        indices = torch.randperm(len(full_dataset))[:num_samples]
        subset_data = []
        subset_targets = []

        for i in indices:
            data, target = full_dataset[i]
            subset_data.append(data)
            subset_targets.append(target)

        subset_data = torch.stack(subset_data)
        subset_targets = torch.tensor(subset_targets)

        return TensorDataset(subset_data, subset_targets)

    except Exception as e:
        logger.warning(f"Could not load MNIST: {e}. Creating synthetic data.")
        # Create synthetic 28x28 data
        data = torch.randn(num_samples, 1, 28, 28)
        targets = torch.randint(0, 10, (num_samples,))
        return TensorDataset(data, targets)

def train_baseline_efficient(device, num_samples=500, epochs=2):
    """Train baseline model efficiently"""
    logger.info(f"Training baseline on {num_samples} samples...")

    # Create tiny dataset
    dataset = create_tiny_mnist_dataset(num_samples)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model and optimizer
    model = TinyCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    model.train()

    start_time = time.time()

    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # Memory management
            if batch_idx % 10 == 0:
                clear_memory()

        avg_loss = np.mean(epoch_losses)
        losses.extend(epoch_losses)
        logger.info(f'Epoch {epoch}, Avg Loss: {avg_loss:.4f}')

    end_time = time.time()
    logger.info(f"Baseline training completed in {end_time - start_time:.2f} seconds")

    return losses

def train_with_opt2vec_efficient(device, num_samples=500, epochs=2):
    """Train model with Opt2Vec optimizer efficiently"""
    logger.info(f"Training with Opt2Vec on {num_samples} samples...")

    # Create tiny dataset
    dataset = create_tiny_mnist_dataset(num_samples)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model and optimizer
    model = TinyCNN().to(device)
    optimizer = LightweightOpt2VecOptimizer(model.parameters(), base_lr=0.01, device=device)

    losses = []
    embeddings = []
    model.train()

    start_time = time.time()

    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # Get embedding from optimizer step
            embedding = optimizer.step(loss.item())
            if isinstance(embedding, np.ndarray):
                embeddings.append(embedding)

            epoch_losses.append(loss.item())

            # Memory management
            if batch_idx % 10 == 0:
                clear_memory()

        avg_loss = np.mean(epoch_losses)
        losses.extend(epoch_losses)
        logger.info(f'Epoch {epoch}, Avg Loss: {avg_loss:.4f}')

    end_time = time.time()
    logger.info(f"Opt2Vec training completed in {end_time - start_time:.2f} seconds")

    return losses, embeddings

def quick_embedding_analysis(embeddings, max_points=100):
    """Quick embedding analysis for resource-constrained environments"""
    if not embeddings or len(embeddings) < 5:
        logger.info("Not enough embeddings for analysis")
        return

    embeddings = np.array(embeddings)

    # Sample embeddings if too many
    if len(embeddings) > max_points:
        indices = np.linspace(0, len(embeddings)-1, max_points, dtype=int)
        embeddings = embeddings[indices]

    logger.info(f"Embedding Analysis:")
    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    logger.info(f"  Std norm: {np.linalg.norm(embeddings, axis=1).std():.4f}")

    # Simple 2D visualization if embedding dim is small
    if embeddings.shape[1] <= 2:
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=10)
        plt.title('Embedding Space')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        plt.subplot(1, 2, 2)
        plt.plot(np.linalg.norm(embeddings, axis=1))
        plt.title('Embedding Norm Evolution')
        plt.xlabel('Step')
        plt.ylabel('Norm')

        plt.tight_layout()
        plt.show()

def main_efficient():
    """Efficient main experiment for resource-constrained environments"""
    # Use CPU by default, GPU if available and not memory constrained
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For very constrained environments, force CPU
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory < 2.0:  # Less than 2GB GPU memory
            device = torch.device('cpu')
            logger.info("GPU memory low, using CPU")

    logger.info(f"Using device: {device}")
    logger.info(f"Memory usage at start: {get_memory_usage():.1f} MB")

    # Small-scale experiments
    num_samples = 200  # Very small dataset
    epochs = 2

    try:
        # 1. Train baseline
        logger.info("=" * 50)
        logger.info("Training baseline model...")
        baseline_losses = train_baseline_efficient(device, num_samples, epochs)
        clear_memory()

        # 2. Quick meta-learning demo
        logger.info("=" * 50)
        logger.info("Quick meta-learning demo...")

        trainer = EfficientMetaLearningTrainer(device)

        # Create tiny meta-optimizer components
        opt2vec_components = {
            'opt2vec_net': TinyOpt2VecNetwork(embedding_dim=8).to(device),  # Even smaller
            'lr_adapter': nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()).to(device),
            'momentum_adapter': nn.Sequential(nn.Linear(8, 1), nn.Sigmoid()).to(device)
        }

        # Meta-optimizer
        meta_params = list(opt2vec_components['opt2vec_net'].parameters()) + \
                     list(opt2vec_components['lr_adapter'].parameters()) + \
                     list(opt2vec_components['momentum_adapter'].parameters())
        meta_optimizer = optim.Adam(meta_params, lr=0.01)

        # Quick meta-training
        meta_losses = []
        for meta_step in range(10):  # Very few steps
            meta_loss, task_losses = trainer.meta_train_step(opt2vec_components, meta_optimizer, num_tasks=2)
            meta_losses.append(meta_loss)

            if meta_step % 5 == 0:
                logger.info(f"Meta-step {meta_step}, Meta-loss: {meta_loss:.4f}")

            clear_memory()

        # 3. Train with learned optimizer
        logger.info("=" * 50)
        logger.info("Training with learned Opt2Vec...")

        # Create fresh Opt2Vec optimizer with learned components
        dataset = create_tiny_mnist_dataset(num_samples)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)  # Even smaller batches

        model = TinyCNN().to(device)
        optimizer = LightweightOpt2VecOptimizer(model.parameters(), base_lr=0.01, device=device, embedding_dim=8)

        # Use learned components
        optimizer.opt2vec_net = opt2vec_components['opt2vec_net']
        optimizer.lr_adapter = opt2vec_components['lr_adapter']
        optimizer.momentum_adapter = opt2vec_components['momentum_adapter']

        opt2vec_losses = []
        embeddings = []

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()

                embedding = optimizer.step(loss.item())
                if isinstance(embedding, np.ndarray):
                    embeddings.append(embedding)

                opt2vec_losses.append(loss.item())

                if batch_idx % 5 == 0:
                    clear_memory()

        # 4. Compare and analyze
        logger.info("=" * 50)
        logger.info("Results comparison:")

        if baseline_losses and opt2vec_losses:
            logger.info(f"Baseline final loss: {np.mean(baseline_losses[-5:]):.4f}")
            logger.info(f"Opt2Vec final loss: {np.mean(opt2vec_losses[-5:]):.4f}")

            # Simple plot
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.plot(baseline_losses[:len(opt2vec_losses)], label='Baseline', alpha=0.7)
            plt.plot(opt2vec_losses, label='Opt2Vec', alpha=0.7)
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Training Comparison')
            plt.legend()
            plt.yscale('log')

            plt.subplot(1, 2, 2)
            plt.plot(meta_losses)
            plt.xlabel('Meta-training Steps')
            plt.ylabel('Meta Loss')
            plt.title('Meta-Learning Progress')

            plt.tight_layout()
            plt.show()

        # 5. Quick embedding analysis
        logger.info("=" * 50)
        logger.info("Embedding analysis:")
        quick_embedding_analysis(embeddings)

        logger.info(f"Final memory usage: {get_memory_usage():.1f} MB")
        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.info("This might be due to memory constraints. Try reducing num_samples further.")
        raise

if __name__ == "__main__":
    main_efficient()
