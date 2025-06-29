"""
Debugging utilities for Opt2Vec optimizer.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)


class Opt2VecDebugger:
    """
    Comprehensive debugging and monitoring utility for Opt2Vec optimizer.
    """

    def __init__(self, save_dir: str = "./debug_outputs"):
        """
        Initialize debugger.

        Args:
            save_dir: Directory to save debug outputs
        """
        self.save_dir = save_dir
        self.debug_history = {
            'grad_norms': [],
            'lr_multipliers': [],
            'momentum_factors': [],
            'embedding_stats': [],
            'update_magnitudes': [],
            'parameter_norms': [],
            'loss_values': [],
            'meta_grad_norms': [],
            'meta_losses': [],
            'task_improvements': [],
            'stability_events': []
        }

    def log_optimizer_step(self, optimizer_debug_stats: Dict[str, Any], step: int):
        """
        Log optimizer step statistics.

        Args:
            optimizer_debug_stats: Debug statistics from optimizer
            step: Current step number
        """
        if not optimizer_debug_stats:
            return

        # Extract recent values
        if optimizer_debug_stats.get('grad_norms'):
            self.debug_history['grad_norms'].extend(optimizer_debug_stats['grad_norms'][-1:])
        if optimizer_debug_stats.get('lr_multipliers'):
            self.debug_history['lr_multipliers'].extend(optimizer_debug_stats['lr_multipliers'][-1:])
        if optimizer_debug_stats.get('momentum_factors'):
            self.debug_history['momentum_factors'].extend(optimizer_debug_stats['momentum_factors'][-1:])
        if optimizer_debug_stats.get('update_magnitudes'):
            self.debug_history['update_magnitudes'].extend(optimizer_debug_stats['update_magnitudes'][-1:])
        if optimizer_debug_stats.get('parameter_norms'):
            self.debug_history['parameter_norms'].extend(optimizer_debug_stats['parameter_norms'][-1:])
        if optimizer_debug_stats.get('loss_values'):
            self.debug_history['loss_values'].extend(optimizer_debug_stats['loss_values'][-1:])
        if optimizer_debug_stats.get('embedding_stats'):
            self.debug_history['embedding_stats'].extend(optimizer_debug_stats['embedding_stats'][-1:])

    def log_meta_step(self, meta_stats: Dict[str, Any], step: int):
        """
        Log meta-training step statistics.

        Args:
            meta_stats: Meta-training statistics
            step: Current meta step number
        """
        if meta_stats.get('meta_loss'):
            self.debug_history['meta_losses'].append(meta_stats['meta_loss'])
        if meta_stats.get('avg_improvement'):
            self.debug_history['task_improvements'].append(meta_stats['avg_improvement'])
        if meta_stats.get('grad_stats', {}).get('total_norm'):
            self.debug_history['meta_grad_norms'].append(meta_stats['grad_stats']['total_norm'])

    def log_stability_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Log stability-related events.

        Args:
            event_type: Type of stability event
            event_data: Event data
        """
        self.debug_history['stability_events'].append({
            'type': event_type,
            'data': event_data,
            'timestamp': len(self.debug_history['grad_norms'])
        })

    def check_gradient_stability(self, grad_norms: List[float], window: int = 10) -> Dict[str, Any]:
        """
        Check for gradient stability issues.

        Args:
            grad_norms: List of gradient norms
            window: Window size for analysis

        Returns:
            Dictionary with stability analysis
        """
        if len(grad_norms) < window:
            return {'status': 'insufficient_data'}

        recent_norms = grad_norms[-window:]

        # Check for gradient explosion
        max_norm = max(recent_norms)
        explosion_threshold = 10.0
        is_exploding = max_norm > explosion_threshold

        # Check for gradient vanishing
        min_norm = min(recent_norms)
        vanishing_threshold = 1e-8
        is_vanishing = min_norm < vanishing_threshold

        # Check for gradient oscillation
        norm_std = np.std(recent_norms)
        oscillation_threshold = 5.0
        is_oscillating = norm_std > oscillation_threshold

        return {
            'status': 'stable' if not (is_exploding or is_vanishing or is_oscillating) else 'unstable',
            'is_exploding': is_exploding,
            'is_vanishing': is_vanishing,
            'is_oscillating': is_oscillating,
            'max_norm': max_norm,
            'min_norm': min_norm,
            'norm_std': norm_std,
            'recent_norms': recent_norms
        }

    def check_embedding_stability(self, embedding_stats: List[Dict[str, float]], window: int = 5) -> Dict[str, Any]:
        """
        Check for embedding stability issues.

        Args:
            embedding_stats: List of embedding statistics
            window: Window size for analysis

        Returns:
            Dictionary with embedding stability analysis
        """
        if len(embedding_stats) < window:
            return {'status': 'insufficient_data'}

        recent_stats = embedding_stats[-window:]

        # Check for embedding collapse
        std_values = [stat['std'] for stat in recent_stats]
        min_std = min(std_values)
        is_collapsed = min_std < 1e-6

        # Check for embedding explosion
        norm_values = [stat['norm'] for stat in recent_stats]
        max_norm = max(norm_values)
        is_exploded = max_norm > 100.0

        # Check for embedding oscillation
        mean_values = [stat['mean'] for stat in recent_stats]
        mean_std = np.std(mean_values)
        is_oscillating = mean_std > 10.0

        return {
            'status': 'stable' if not (is_collapsed or is_exploded or is_oscillating) else 'unstable',
            'is_collapsed': is_collapsed,
            'is_exploded': is_exploded,
            'is_oscillating': is_oscillating,
            'min_std': min_std,
            'max_norm': max_norm,
            'mean_std': mean_std,
            'recent_stats': recent_stats
        }

    def check_learning_rate_stability(self, lr_multipliers: List[float], window: int = 10) -> Dict[str, Any]:
        """
        Check for learning rate stability issues.

        Args:
            lr_multipliers: List of learning rate multipliers
            window: Window size for analysis

        Returns:
            Dictionary with learning rate stability analysis
        """
        if len(lr_multipliers) < window:
            return {'status': 'insufficient_data'}

        recent_lrs = lr_multipliers[-window:]

        # Check for extreme learning rates
        min_lr = min(recent_lrs)
        max_lr = max(recent_lrs)
        is_extreme = min_lr < 0.01 or max_lr > 0.99

        # Check for learning rate oscillation
        lr_std = np.std(recent_lrs)
        is_oscillating = lr_std > 0.3

        # Check for learning rate saturation
        is_saturated = min_lr > 0.9 or max_lr < 0.1

        return {
            'status': 'stable' if not (is_extreme or is_oscillating or is_saturated) else 'unstable',
            'is_extreme': is_extreme,
            'is_oscillating': is_oscillating,
            'is_saturated': is_saturated,
            'min_lr': min_lr,
            'max_lr': max_lr,
            'lr_std': lr_std,
            'recent_lrs': recent_lrs
        }

    def create_stability_report(self) -> Dict[str, Any]:
        """
        Create a comprehensive stability report.

        Returns:
            Dictionary with stability analysis
        """
        report = {
            'gradient_stability': self.check_gradient_stability(self.debug_history['grad_norms']),
            'embedding_stability': self.check_embedding_stability(self.debug_history['embedding_stats']),
            'lr_stability': self.check_learning_rate_stability(self.debug_history['lr_multipliers']),
            'stability_events': self.debug_history['stability_events'],
            'summary': {}
        }

        # Overall stability assessment
        overall_stable = (
            report['gradient_stability'].get('status') == 'stable' and
            report['embedding_stability'].get('status') == 'stable' and
            report['lr_stability'].get('status') == 'stable'
        )

        report['summary'] = {
            'overall_stable': overall_stable,
            'total_steps': len(self.debug_history['grad_norms']),
            'total_events': len(self.debug_history['stability_events']),
            'gradient_issues': report['gradient_stability'].get('status') != 'stable',
            'embedding_issues': report['embedding_stability'].get('status') != 'stable',
            'lr_issues': report['lr_stability'].get('status') != 'stable'
        }

        return report

    def plot_optimization_trajectory(self, save_path: Optional[str] = None):
        """
        Plot optimization trajectory with various metrics.

        Args:
            save_path: Path to save the plot
        """
        if not self.debug_history['grad_norms']:
            logger.warning("No data available for plotting")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Opt2Vec Optimization Trajectory Analysis', fontsize=16)

        # Gradient norms
        if self.debug_history['grad_norms']:
            axes[0, 0].plot(self.debug_history['grad_norms'])
            axes[0, 0].set_title('Gradient Norms')
            axes[0, 0].set_ylabel('Gradient Norm')
            axes[0, 0].grid(True)

        # Learning rate multipliers
        if self.debug_history['lr_multipliers']:
            axes[0, 1].plot(self.debug_history['lr_multipliers'])
            axes[0, 1].set_title('Learning Rate Multipliers')
            axes[0, 1].set_ylabel('LR Multiplier')
            axes[0, 1].grid(True)

        # Momentum factors
        if self.debug_history['momentum_factors']:
            axes[0, 2].plot(self.debug_history['momentum_factors'])
            axes[0, 2].set_title('Momentum Factors')
            axes[0, 2].set_ylabel('Momentum')
            axes[0, 2].grid(True)

        # Update magnitudes
        if self.debug_history['update_magnitudes']:
            axes[1, 0].plot(self.debug_history['update_magnitudes'])
            axes[1, 0].set_title('Parameter Update Magnitudes')
            axes[1, 0].set_ylabel('Update Magnitude')
            axes[1, 0].grid(True)

        # Loss values
        if self.debug_history['loss_values']:
            axes[1, 1].plot(self.debug_history['loss_values'])
            axes[1, 1].set_title('Loss Values')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)

        # Parameter norms
        if self.debug_history['parameter_norms']:
            axes[1, 2].plot(self.debug_history['parameter_norms'])
            axes[1, 2].set_title('Parameter Norms')
            axes[1, 2].set_ylabel('Parameter Norm')
            axes[1, 2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_embedding_analysis(self, embeddings: List[np.ndarray], save_path: Optional[str] = None):
        """
        Plot embedding analysis using dimensionality reduction.

        Args:
            embeddings: List of embedding vectors
            save_path: Path to save the plot
        """
        if not embeddings:
            logger.warning("No embeddings available for analysis")
            return

        # Convert to numpy array
        embedding_array = np.array(embeddings)

        # Use PCA for dimensionality reduction
        if embedding_array.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embedding_array)
        else:
            reduced_embeddings = embedding_array

        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                   c=range(len(reduced_embeddings)), cmap='viridis', alpha=0.7)
        plt.colorbar(label='Step')
        plt.title('Opt2Vec Embedding Trajectory (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_meta_training_analysis(self, save_path: Optional[str] = None):
        """
        Plot meta-training analysis.

        Args:
            save_path: Path to save the plot
        """
        if not self.debug_history['meta_losses']:
            logger.warning("No meta-training data available for plotting")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Meta-Training Analysis', fontsize=16)

        # Meta losses
        if self.debug_history['meta_losses']:
            axes[0].plot(self.debug_history['meta_losses'])
            axes[0].set_title('Meta Losses')
            axes[0].set_ylabel('Meta Loss')
            axes[0].grid(True)

        # Task improvements
        if self.debug_history['task_improvements']:
            axes[1].plot(self.debug_history['task_improvements'])
            axes[1].set_title('Task Improvements')
            axes[1].set_ylabel('Improvement')
            axes[1].grid(True)

        # Meta gradient norms
        if self.debug_history['meta_grad_norms']:
            axes[2].plot(self.debug_history['meta_grad_norms'])
            axes[2].set_title('Meta Gradient Norms')
            axes[2].set_ylabel('Gradient Norm')
            axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def save_debug_report(self, filename: str = "opt2vec_debug_report.txt"):
        """
        Save a comprehensive debug report to file.

        Args:
            filename: Name of the report file
        """
        report = self.create_stability_report()

        with open(f"{self.save_dir}/{filename}", 'w') as f:
            f.write("Opt2Vec Debug Report\n")
            f.write("=" * 50 + "\n\n")

            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Stable: {report['summary']['overall_stable']}\n")
            f.write(f"Total Steps: {report['summary']['total_steps']}\n")
            f.write(f"Total Events: {report['summary']['total_events']}\n")
            f.write(f"Gradient Issues: {report['summary']['gradient_issues']}\n")
            f.write(f"Embedding Issues: {report['summary']['embedding_issues']}\n")
            f.write(f"Learning Rate Issues: {report['summary']['lr_issues']}\n\n")

            # Detailed analysis
            f.write("DETAILED ANALYSIS\n")
            f.write("-" * 20 + "\n")

            # Gradient stability
            grad_stability = report['gradient_stability']
            f.write(f"Gradient Stability: {grad_stability.get('status', 'unknown')}\n")
            if grad_stability.get('status') != 'insufficient_data':
                f.write(f"  Exploding: {grad_stability.get('is_exploding', False)}\n")
                f.write(f"  Vanishing: {grad_stability.get('is_vanishing', False)}\n")
                f.write(f"  Oscillating: {grad_stability.get('is_oscillating', False)}\n")
                f.write(f"  Max Norm: {grad_stability.get('max_norm', 0):.4f}\n")
                f.write(f"  Min Norm: {grad_stability.get('min_norm', 0):.4e}\n\n")

            # Embedding stability
            emb_stability = report['embedding_stability']
            f.write(f"Embedding Stability: {emb_stability.get('status', 'unknown')}\n")
            if emb_stability.get('status') != 'insufficient_data':
                f.write(f"  Collapsed: {emb_stability.get('is_collapsed', False)}\n")
                f.write(f"  Exploded: {emb_stability.get('is_exploded', False)}\n")
                f.write(f"  Oscillating: {emb_stability.get('is_oscillating', False)}\n")
                f.write(f"  Min Std: {emb_stability.get('min_std', 0):.4e}\n")
                f.write(f"  Max Norm: {emb_stability.get('max_norm', 0):.4f}\n\n")

            # Learning rate stability
            lr_stability = report['lr_stability']
            f.write(f"Learning Rate Stability: {lr_stability.get('status', 'unknown')}\n")
            if lr_stability.get('status') != 'insufficient_data':
                f.write(f"  Extreme: {lr_stability.get('is_extreme', False)}\n")
                f.write(f"  Oscillating: {lr_stability.get('is_oscillating', False)}\n")
                f.write(f"  Saturated: {lr_stability.get('is_saturated', False)}\n")
                f.write(f"  Min LR: {lr_stability.get('min_lr', 0):.4f}\n")
                f.write(f"  Max LR: {lr_stability.get('max_lr', 0):.4f}\n\n")

            # Stability events
            f.write("STABILITY EVENTS\n")
            f.write("-" * 20 + "\n")
            for event in report['stability_events']:
                f.write(f"Step {event['timestamp']}: {event['type']} - {event['data']}\n")

        logger.info(f"Debug report saved to {self.save_dir}/{filename}")


def create_loss_landscape_2d(model: torch.nn.Module,
                           data: torch.Tensor,
                           targets: torch.Tensor,
                           param_names: List[str],
                           resolution: int = 50) -> Dict[str, np.ndarray]:
    """
    Create 2D loss landscape projections for visualization.

    Args:
        model: Neural network model
        data: Input data
        targets: Target values
        param_names: Names of parameters to visualize
        resolution: Resolution of the landscape grid

    Returns:
        Dictionary with loss landscape data
    """
    model.eval()
    criterion = torch.nn.MSELoss()

    landscapes = {}

    for param_name in param_names:
        param = None
        for name, p in model.named_parameters():
            if name == param_name:
                param = p
                break

        if param is None:
            continue

        # Store original parameter values
        original_values = param.data.clone()

        # Create grid around current parameter values
        param_min = param.data.min().item() - 0.1
        param_max = param.data.max().item() + 0.1

        x = np.linspace(param_min, param_max, resolution)
        y = np.linspace(param_min, param_max, resolution)
        X, Y = np.meshgrid(x, y)

        loss_grid = np.zeros((resolution, resolution))

        # Compute loss for each grid point
        for i in range(resolution):
            for j in range(resolution):
                # Set parameter values
                param.data = torch.tensor([[X[i, j], Y[i, j]]], dtype=param.dtype, device=param.device)

                # Compute loss
                with torch.no_grad():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss_grid[i, j] = loss.item()

        # Restore original parameter values
        param.data = original_values

        landscapes[param_name] = {
            'X': X,
            'Y': Y,
            'loss': loss_grid
        }

    return landscapes
