"""
Example usage of enhanced Opt2Vec optimizer with advanced features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Import from the enhanced opt2vec
from opt2vec import LightweightOpt2VecOptimizer


def get_mnist_loaders(batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class SimpleModel(nn.Module):
    """Simple MLP for MNIST classification."""
    def __init__(self, input_dim: int = 28*28, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.network(x)


def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.view(batch_X.size(0), -1).to(device)
            batch_y = batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_X.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_with_enhanced_opt2vec(
    embedding_dim: int = 64,  # Best: 64 (vs 128, 32)
    history_length: int = 8,  # Best: 8 (vs 16, 32)
    activation: str = 'gelu',  # Best: gelu (vs swish, relu)
    use_extended_features: bool = True,
    num_epochs: int = 10,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu')
):
    print(f"Training with enhanced Opt2Vec configuration:")
    print(f"  - Embedding dim: {embedding_dim}")
    print(f"  - History length: {history_length}")
    print(f"  - Activation: {activation}")
    print(f"  - Extended features: {use_extended_features}")
    print(f"  - Model: Simple 2-layer MLP, output_dim=10 (classification)")
    print(f"  - Loss: CrossEntropyLoss (MNIST classification)")
    print(f"  - Batch size: {batch_size}, Epochs: {num_epochs}")
    print("\nEstimated time: 3-6 minutes on your Mac (1.6GHz i5, 8GB RAM)")
    print()
    start_time = time.time()

    # Model and data
    model = SimpleModel(input_dim=28*28, hidden_dim=64, output_dim=10).to(device)
    train_loader, test_loader = get_mnist_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()

    optimizer = LightweightOpt2VecOptimizer(
        parameters=model.parameters(),
        base_lr=0.01,
        embedding_dim=embedding_dim,
        history_length=history_length,
        activation=activation,
        device=device,
        debug_mode=True,
        max_grad_norm=1.0,
        lr_bounds=(1e-6, 1e2),
        momentum_bounds=(0.0, 0.99),
        use_extended_features=use_extended_features,
        normalize_features=True,
        dropout=0.1,
        use_layer_norm=True,
        use_attention=True,
        use_positional_encoding=True
    )

    history = {'losses': [], 'lr_multipliers': [], 'momentum_factors': [], 'grad_norms': [], 'param_norms': [], 'grad_diversities': [], 'loss_curvatures': []}

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.view(batch_X.size(0), -1).to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step(loss.item())
            epoch_losses.append(loss.item())
        avg_epoch_loss = np.mean(epoch_losses)
        history['losses'].append(avg_epoch_loss)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.1f} minutes")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Opt2Vec Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
    return history, test_loss, test_acc


def plot_training_history(history: Dict[str, List[float]], config: Dict[str, Any]):
    """Plot training history."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Enhanced Opt2Vec Training History\n'
                f'Embedding: {config["embedding_dim"]}, History: {config["history_length"]}, '
                f'Activation: {config["activation"]}')

    # Loss
    axes[0, 0].plot(history['losses'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # Learning rate multipliers
    if history['lr_multipliers']:
        axes[0, 1].plot(history['lr_multipliers'])
        axes[0, 1].set_title('Learning Rate Multipliers')
        axes[0, 1].set_ylabel('LR Multiplier')
        axes[0, 1].grid(True)

    # Momentum factors
    if history['momentum_factors']:
        axes[0, 2].plot(history['momentum_factors'])
        axes[0, 2].set_title('Momentum Factors')
        axes[0, 2].set_ylabel('Momentum')
        axes[0, 2].grid(True)

    # Gradient norms
    if history['grad_norms']:
        axes[1, 0].plot(history['grad_norms'])
        axes[1, 0].set_title('Gradient Norms')
        axes[1, 0].set_ylabel('Grad Norm')
        axes[1, 0].grid(True)

    # Parameter norms
    if history['param_norms']:
        axes[1, 1].plot(history['param_norms'])
        axes[1, 1].set_title('Parameter Norms')
        axes[1, 1].set_ylabel('Param Norm')
        axes[1, 1].grid(True)

    # Gradient diversity
    if history['grad_diversities']:
        axes[1, 2].plot(history['grad_diversities'])
        axes[1, 2].set_title('Gradient Diversity')
        axes[1, 2].set_ylabel('Diversity')
        axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(f'enhanced_opt2vec_training_{config["embedding_dim"]}_{config["history_length"]}_{config["activation"]}.png')
    plt.show()


def compare_configurations():
    """Compare different enhanced configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test different embedding dimensions
    print("\n" + "="*60)
    print("Testing different embedding dimensions")
    print("="*60)

    embedding_configs = [
        {'embedding_dim': 32, 'history_length': 16, 'activation': 'gelu'},
        {'embedding_dim': 64, 'history_length': 16, 'activation': 'gelu'},
        {'embedding_dim': 128, 'history_length': 16, 'activation': 'gelu'}
    ]

    embedding_results = []

    for config in embedding_configs:
        print(f"\nTesting embedding dimension: {config['embedding_dim']}")
        history, test_loss, test_acc = train_with_enhanced_opt2vec(
            embedding_dim=config['embedding_dim'],
            history_length=config['history_length'],
            activation=config['activation'],
            num_epochs=30
        )

        final_loss = history['losses'][-1]
        embedding_results.append({
            'embedding_dim': config['embedding_dim'],
            'final_loss': final_loss,
            'history': history
        })

        print(f"Final loss: {final_loss:.6f}")

    # Test different history lengths
    print("\n" + "="*60)
    print("Testing different history lengths")
    print("="*60)

    history_configs = [
        {'embedding_dim': 64, 'history_length': 8, 'activation': 'gelu'},
        {'embedding_dim': 64, 'history_length': 16, 'activation': 'gelu'},
        {'embedding_dim': 64, 'history_length': 32, 'activation': 'gelu'}
    ]

    history_results = []

    for config in history_configs:
        print(f"\nTesting history length: {config['history_length']}")
        history, test_loss, test_acc = train_with_enhanced_opt2vec(
            embedding_dim=config['embedding_dim'],
            history_length=config['history_length'],
            activation=config['activation'],
            num_epochs=30
        )

        final_loss = history['losses'][-1]
        history_results.append({
            'history_length': config['history_length'],
            'final_loss': final_loss,
            'history': history
        })

        print(f"Final loss: {final_loss:.6f}")

    # Test different activation functions
    print("\n" + "="*60)
    print("Testing different activation functions")
    print("="*60)

    activation_configs = [
        {'embedding_dim': 64, 'history_length': 16, 'activation': 'relu'},
        {'embedding_dim': 64, 'history_length': 16, 'activation': 'gelu'},
        {'embedding_dim': 64, 'history_length': 16, 'activation': 'swish'}
    ]

    activation_results = []

    for config in activation_configs:
        print(f"\nTesting activation: {config['activation']}")
        history, test_loss, test_acc = train_with_enhanced_opt2vec(
            embedding_dim=config['embedding_dim'],
            history_length=config['history_length'],
            activation=config['activation'],
            num_epochs=30
        )

        final_loss = history['losses'][-1]
        activation_results.append({
            'activation': config['activation'],
            'final_loss': final_loss,
            'history': history
        })

        print(f"Final loss: {final_loss:.6f}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nEmbedding Dimension Results:")
    for result in embedding_results:
        print(f"  {result['embedding_dim']}: {result['final_loss']:.6f}")

    print("\nHistory Length Results:")
    for result in history_results:
        print(f"  {result['history_length']}: {result['final_loss']:.6f}")

    print("\nActivation Function Results:")
    for result in activation_results:
        print(f"  {result['activation']}: {result['final_loss']:.6f}")

    # Find best configurations
    best_embedding = min(embedding_results, key=lambda x: x['final_loss'])
    best_history = min(history_results, key=lambda x: x['final_loss'])
    best_activation = min(activation_results, key=lambda x: x['final_loss'])

    print(f"\nBest configurations:")
    print(f"  Embedding dim: {best_embedding['embedding_dim']} (loss: {best_embedding['final_loss']:.6f})")
    print(f"  History length: {best_history['history_length']} (loss: {best_history['final_loss']:.6f})")
    print(f"  Activation: {best_activation['activation']} (loss: {best_activation['final_loss']:.6f})")


def train_with_adam(num_epochs=10, batch_size=32, device=torch.device('cpu')):
    print("\nTraining with Adam optimizer...")
    start_time = time.time()
    model = SimpleModel(input_dim=28*28, hidden_dim=64, output_dim=10).to(device)
    train_loader, test_loader = get_mnist_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.view(batch_X.size(0), -1).to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_epoch_loss = np.mean(epoch_losses)
        history.append(avg_epoch_loss)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.1f} minutes")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Adam Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
    return history, test_loss, test_acc

def train_with_sgd(num_epochs=10, batch_size=32, device=torch.device('cpu')):
    print("\nTraining with SGD optimizer...")
    start_time = time.time()
    model = SimpleModel(input_dim=28*28, hidden_dim=64, output_dim=10).to(device)
    train_loader, test_loader = get_mnist_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.view(batch_X.size(0), -1).to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_epoch_loss = np.mean(epoch_losses)
        history.append(avg_epoch_loss)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
    elapsed = time.time() - start_time
    print(f"Total elapsed time: {elapsed/60:.1f} minutes")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"SGD Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
    return history, test_loss, test_acc


def main():
    print("Enhanced Opt2Vec Optimizer Demonstration")
    print("=" * 50)

    # Use best configurations based on experiments
    config = {
        'embedding_dim': 64,  # Best: 64 (vs 128, 32)
        'history_length': 8,  # Best: 8 (vs 16, 32)
        'activation': 'gelu',  # Best: gelu (vs swish, relu)
        'use_extended_features': True
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Opt2Vec
    history, opt2vec_test_loss, opt2vec_test_acc = train_with_enhanced_opt2vec(
        embedding_dim=config['embedding_dim'],
        history_length=config['history_length'],
        activation=config['activation'],
        use_extended_features=config['use_extended_features'],
        num_epochs=50,
        batch_size=32,
        device=device
    )
    # Adam
    adam_history, adam_test_loss, adam_test_acc = train_with_adam(num_epochs=10, batch_size=32, device=device)
    # SGD
    sgd_history, sgd_test_loss, sgd_test_acc = train_with_sgd(num_epochs=10, batch_size=32, device=device)

    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.plot(history['losses'], label=f'Opt2Vec (Test Acc: {opt2vec_test_acc*100:.1f}%)')
    plt.plot(adam_history, label=f'Adam (Test Acc: {adam_test_acc*100:.1f}%)')
    plt.plot(sgd_history, label=f'SGD (Test Acc: {sgd_test_acc*100:.1f}%)')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nCompleted!")

    # Plot training history
    plot_training_history(history, config)
    plot_training_history(adam_history, config)
    plot_training_history(sgd_history, config)

    # # Hyperparameter sweep section
    # print("Hyperparameter Sweep: Opt2Vec Configurations")
    # print("="*50)
    # compare_configurations()

if __name__ == "__main__":
    main()
