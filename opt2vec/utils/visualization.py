"""
Visualization utilities for Opt2Vec project.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Optional
import torch


def plot_training_curves(losses: Dict[str, List[float]],
                        title: str = "Training Curves",
                        save_path: Optional[str] = None):
    """
    Plot training loss curves for different optimizers.

    Args:
        losses: Dictionary mapping optimizer names to loss lists
        title: Plot title
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))

    for name, loss_list in losses.items():
        plt.plot(loss_list, label=name, alpha=0.8)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_embedding_evolution(embeddings: List[np.ndarray],
                           losses: Optional[List[float]] = None,
                           title: str = "Embedding Evolution",
                           save_path: Optional[str] = None):
    """
    Plot embedding evolution over training steps.

    Args:
        embeddings: List of embedding vectors
        losses: Optional list of corresponding loss values
        title: Plot title
        save_path: Optional path to save the plot
    """
    if len(embeddings) == 0:
        print("No embeddings to plot")
        return

    # Flatten and pad embeddings to consistent shape
    flattened_embeddings = []
    for emb in embeddings:
        if emb is not None:
            flattened_embeddings.append(emb.flatten())
        else:
            if flattened_embeddings:
                flattened_embeddings.append(np.zeros_like(flattened_embeddings[0]))
            else:
                continue
    if len(flattened_embeddings) < 2:
        print("Not enough embeddings to plot")
        return
    max_dim = max(emb.shape[0] for emb in flattened_embeddings)
    padded_embeddings = []
    for emb in flattened_embeddings:
        if emb.shape[0] < max_dim:
            padded = np.zeros(max_dim)
            padded[:emb.shape[0]] = emb
            padded_embeddings.append(padded)
        else:
            padded_embeddings.append(emb)
    embeddings_array = np.array(padded_embeddings)

    # Use t-SNE for dimensionality reduction if needed
    if embeddings_array.shape[1] > 2:
        print("Reducing embedding dimensionality with t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_array)
    else:
        embeddings_2d = embeddings_array

    plt.figure(figsize=(12, 8))

    # Create scatter plot
    if losses is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=losses, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Loss')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

    # Add trajectory lines
    plt.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'k-', alpha=0.3, linewidth=0.5)

    # Mark start and end points
    plt.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1],
               c='green', s=100, marker='o', label='Start', edgecolors='black')
    plt.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1],
               c='red', s=100, marker='s', label='End', edgecolors='black')

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_adaptation_patterns(adaptation_stats: List[Dict[str, Any]],
                           title: str = "Adaptation Patterns",
                           save_path: Optional[str] = None):
    """
    Plot learning rate and momentum adaptation patterns.

    Args:
        adaptation_stats: List of adaptation statistics dictionaries
        title: Plot title
        save_path: Optional path to save the plot
    """
    if len(adaptation_stats) == 0:
        print("No adaptation stats to plot")
        return

    steps = [stats['step_count'] for stats in adaptation_stats]
    lr_multipliers = [stats['lr_multiplier'] for stats in adaptation_stats]
    momentum_factors = [stats['momentum_factor'] for stats in adaptation_stats]
    adaptive_lrs = [stats['adaptive_lr'] for stats in adaptation_stats]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Learning rate multiplier
    ax1.plot(steps, lr_multipliers, 'b-', label='LR Multiplier')
    ax1.set_ylabel('LR Multiplier')
    ax1.set_title('Learning Rate Adaptation')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Momentum factor
    ax2.plot(steps, momentum_factors, 'r-', label='Momentum Factor')
    ax2.set_ylabel('Momentum Factor')
    ax2.set_title('Momentum Adaptation')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Adaptive learning rate
    ax3.plot(steps, adaptive_lrs, 'g-', label='Adaptive LR')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Adaptive Learning Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_optimizer_comparison(results: Dict[str, Dict[str, List[float]]],
                            title: str = "Optimizer Comparison",
                            save_path: Optional[str] = None):
    """
    Plot comprehensive comparison of different optimizers.

    Args:
        results: Dictionary mapping optimizer names to result dictionaries
        title: Plot title
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss curves
    ax1 = axes[0, 0]
    for name, data in results.items():
        ax1.plot(data['losses'], label=name, alpha=0.8)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final loss comparison
    ax2 = axes[0, 1]
    final_losses = [data['losses'][-1] for data in results.values()]
    names = list(results.keys())
    bars = ax2.bar(names, final_losses, alpha=0.7)
    ax2.set_ylabel('Final Loss')
    ax2.set_title('Final Loss Comparison')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, final_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')

    # Convergence speed (steps to reach 90% of final improvement)
    ax3 = axes[1, 0]
    convergence_steps = []
    for name, data in results.items():
        losses = data['losses']
        initial_loss = losses[0]
        final_loss = losses[-1]
        target_loss = initial_loss - 0.9 * (initial_loss - final_loss)

        # Find step where loss reaches target
        for i, loss in enumerate(losses):
            if loss <= target_loss:
                convergence_steps.append(i)
                break
        else:
            convergence_steps.append(len(losses))

    bars = ax3.bar(names, convergence_steps, alpha=0.7)
    ax3.set_ylabel('Steps to 90% Convergence')
    ax3.set_title('Convergence Speed')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, convergence_steps):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value}', ha='center', va='bottom')

    # Memory usage (if available)
    ax4 = axes[1, 1]
    if 'memory_usage' in next(iter(results.values())):
        memory_usage = [data.get('memory_usage', [0])[-1] for data in results.values()]
        bars = ax4.bar(names, memory_usage, alpha=0.7)
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory Usage')
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, memory_usage):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'Memory usage data not available',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Memory Usage')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
