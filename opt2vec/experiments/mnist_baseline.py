"""
MNIST baseline comparison experiment for Opt2Vec.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple

from ..core.optimizer import LightweightOpt2VecOptimizer
from ..utils.memory import clear_memory, get_memory_usage
from ..utils.visualization import plot_training_curves, plot_optimizer_comparison
from ..utils.metrics import compute_optimization_metrics, compare_optimizers

logger = logging.getLogger(__name__)


class TinyCNN(nn.Module):
    """Tiny CNN for MNIST classification (memory-efficient)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 2)  # Reduced channels and larger stride
        self.conv2 = nn.Conv2d(8, 16, 3, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)  # Much smaller
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_tiny_mnist_dataset(num_samples: int = 1000) -> Tuple[DataLoader, DataLoader]:
    """
    Create a small MNIST dataset for resource-constrained environments.

    Args:
        num_samples: Number of samples to use

    Returns:
        Tuple of (train_loader, test_loader)
    """
    try:
        from torchvision import datasets, transforms

        # Download MNIST if not available
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)

        # Use only a subset for memory efficiency
        if num_samples < len(train_dataset):
            indices = torch.randperm(len(train_dataset))[:num_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    except ImportError:
        logger.warning("torchvision not available, creating synthetic data")
        return create_synthetic_data(num_samples)


def create_synthetic_data(num_samples: int = 1000) -> Tuple[DataLoader, DataLoader]:
    """
    Create synthetic data for testing when MNIST is not available.

    Args:
        num_samples: Number of samples to create

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create synthetic image data
    train_data = torch.randn(num_samples, 1, 28, 28)
    train_labels = torch.randint(0, 10, (num_samples,))

    test_data = torch.randn(num_samples // 5, 1, 28, 28)
    test_labels = torch.randint(0, 10, (num_samples // 5,))

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def train_with_optimizer(model: nn.Module,
                        train_loader: DataLoader,
                        test_loader: DataLoader,
                        optimizer: Any,
                        optimizer_name: str,
                        device: torch.device,
                        epochs: int = 2) -> Dict[str, List[float]]:
    """
    Train model with specified optimizer and return results.

    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer to use
        optimizer_name: Name of optimizer for logging
        device: Target device
        epochs: Number of training epochs

    Returns:
        Dictionary with training results
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []
    embeddings = []
    adaptation_stats = []
    memory_usage = []

    logger.info(f"Training with {optimizer_name}...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Handle different optimizer types
            if isinstance(optimizer, LightweightOpt2VecOptimizer):
                embedding = optimizer.step(loss.item())
                if embedding is not None:
                    embeddings.append(embedding)

                # Get adaptation stats
                stats = optimizer.get_adaptation_stats()
                adaptation_stats.append(stats)
            else:
                optimizer.step()

            epoch_losses.append(loss.item())

            # Monitor memory usage
            memory_usage.append(get_memory_usage())

            # Clear memory periodically
            if batch_idx % 10 == 0:
                clear_memory()

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        test_accuracy = correct / total
        avg_loss = np.mean(epoch_losses)

        train_losses.extend(epoch_losses)
        test_accuracies.append(test_accuracy)

        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={test_accuracy:.4f}")

    training_time = time.time() - start_time
    logger.info(f"{optimizer_name} training completed in {training_time:.2f} seconds")

    return {
        'losses': train_losses,
        'test_accuracies': test_accuracies,
        'embeddings': embeddings,
        'adaptation_stats': adaptation_stats,
        'memory_usage': memory_usage,
        'training_time': training_time
    }


def run_mnist_baseline_experiment(device: torch.device = torch.device('cpu'),
                                 num_samples: int = 1000,
                                 epochs: int = 2) -> Dict[str, Any]:
    """
    Run MNIST baseline comparison experiment.

    Args:
        device: Target device for computation
        num_samples: Number of MNIST samples to use
        epochs: Number of training epochs

    Returns:
        Dictionary with experiment results
    """
    logger.info("Starting MNIST baseline comparison experiment...")

    # Create dataset
    train_loader, test_loader = create_tiny_mnist_dataset(num_samples)
    logger.info(f"Created dataset with {num_samples} training samples")

    # Test different optimizers
    optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        'Opt2Vec': LightweightOpt2VecOptimizer
    }

    results = {}

    for name, optimizer_class in optimizers.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {name}")
        logger.info(f"{'='*50}")

        # Create fresh model for each optimizer
        model = TinyCNN()

        # Initialize optimizer
        if name == 'Opt2Vec':
            optimizer = optimizer_class(
                model.parameters(),
                base_lr=0.01,
                embedding_dim=16,
                history_length=5,
                device=device
            )
        elif name == 'SGD':
            optimizer = optimizer_class(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = optimizer_class(model.parameters(), lr=0.01)

        # Train model
        result = train_with_optimizer(
            model, train_loader, test_loader, optimizer, name, device, epochs
        )

        results[name] = result

        # Clear memory between optimizers
        clear_memory()

    # Analyze results
    logger.info("\n" + "="*50)
    logger.info("EXPERIMENT RESULTS")
    logger.info("="*50)

    # Compute metrics for each optimizer
    for name, result in results.items():
        metrics = compute_optimization_metrics(
            result['losses'],
            result.get('embeddings'),
            result.get('adaptation_stats')
        )

        logger.info(f"\n{name}:")
        logger.info(f"  Final Loss: {metrics['final_loss']:.4f}")
        logger.info(f"  Total Improvement: {metrics['total_improvement']:.4f}")
        logger.info(f"  Convergence Rate: {metrics['convergence_rate']:.4f}")
        logger.info(f"  Training Time: {result['training_time']:.2f}s")
        logger.info(f"  Final Test Accuracy: {result['test_accuracies'][-1]:.4f}")

    # Compare optimizers
    comparison = compare_optimizers(results)

    logger.info(f"\nOverall Ranking:")
    for i, (name, score) in enumerate(comparison['overall_ranking']):
        logger.info(f"  {i+1}. {name}: {score:.4f}")

    # Create visualizations
    logger.info("\nCreating visualizations...")

    # Training curves
    loss_data = {name: result['losses'] for name, result in results.items()}
    plot_training_curves(loss_data, "MNIST Training Loss Curves")

    # Comprehensive comparison
    plot_optimizer_comparison(results, "MNIST Optimizer Comparison")

    # Embedding analysis for Opt2Vec
    if 'Opt2Vec' in results and results['Opt2Vec']['embeddings']:
        from ..utils.visualization import plot_embedding_evolution
        plot_embedding_evolution(
            results['Opt2Vec']['embeddings'],
            results['Opt2Vec']['losses'],
            "Opt2Vec Embedding Evolution"
        )

    return {
        'results': results,
        'comparison': comparison,
        'experiment_config': {
            'device': str(device),
            'num_samples': num_samples,
            'epochs': epochs
        }
    }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = run_mnist_baseline_experiment(device=device, num_samples=1000, epochs=2)

    print("\nExperiment completed successfully!")
