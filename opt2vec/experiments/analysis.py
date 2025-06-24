"""
Embedding analysis experiment for understanding Opt2Vec behavior.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging
from typing import Dict, List, Any, Optional, Tuple

from ..core.optimizer import LightweightOpt2VecOptimizer
from ..core.network import TinyOpt2VecNetwork
from ..utils.memory import clear_memory
from ..utils.visualization import plot_embedding_evolution, plot_adaptation_patterns

logger = logging.getLogger(__name__)


def run_embedding_analysis(device: torch.device = torch.device('cpu'),
                          num_tasks: int = 10,
                          steps_per_task: int = 20) -> Dict[str, Any]:
    """
    Run comprehensive embedding analysis experiment.

    Args:
        device: Target device for computation
        num_tasks: Number of tasks to analyze
        steps_per_task: Number of optimization steps per task

    Returns:
        Dictionary with analysis results
    """
    logger.info("Starting Opt2Vec embedding analysis...")

    # Collect embeddings from multiple tasks
    all_embeddings = []
    all_losses = []
    all_adaptation_stats = []
    task_metadata = []

    for task_idx in range(num_tasks):
        logger.info(f"Analyzing task {task_idx + 1}/{num_tasks}...")

        # Create task
        data, targets = create_analysis_task(task_size=50)
        model = create_analysis_model()

        # Initialize optimizer
        optimizer = LightweightOpt2VecOptimizer(
            model.parameters(),
            base_lr=0.01,
            embedding_dim=16,
            history_length=5,
            device=device
        )

        # Train and collect data
        task_embeddings = []
        task_losses = []
        task_stats = []

        criterion = nn.MSELoss()

        for step in range(steps_per_task):
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Opt2Vec step
            embedding = optimizer.step(loss.item())

            # Collect data
            if embedding is not None:
                task_embeddings.append(embedding)
                task_losses.append(loss.item())

                # Get adaptation stats
                stats = optimizer.get_adaptation_stats()
                task_stats.append(stats)

        # Store task data
        all_embeddings.extend(task_embeddings)
        all_losses.extend(task_losses)
        all_adaptation_stats.extend(task_stats)

        task_metadata.extend([{
            'task_idx': task_idx,
            'step': step,
            'total_steps': steps_per_task
        } for step in range(len(task_embeddings))])

        # Clear memory
        clear_memory()

    # Analyze embeddings
    logger.info("Analyzing embeddings...")
    embedding_analysis = analyze_embedding_space(all_embeddings, all_losses, task_metadata)

    # Analyze adaptation patterns
    logger.info("Analyzing adaptation patterns...")
    adaptation_analysis = analyze_adaptation_patterns(all_adaptation_stats)

    # Create visualizations
    logger.info("Creating visualizations...")
    create_analysis_visualizations(
        all_embeddings, all_losses, all_adaptation_stats,
        embedding_analysis, adaptation_analysis
    )

    return {
        'embeddings': all_embeddings,
        'losses': all_losses,
        'adaptation_stats': all_adaptation_stats,
        'task_metadata': task_metadata,
        'embedding_analysis': embedding_analysis,
        'adaptation_analysis': adaptation_analysis
    }


def create_analysis_task(task_size: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a task for embedding analysis.

    Args:
        task_size: Number of data points

    Returns:
        Tuple of (data, targets)
    """
    # Generate random quadratic function with varying complexity
    x = torch.randn(task_size, 1) * 2.0
    a = torch.randn(1) * 0.5
    b = torch.randn(1) * 0.5
    c = torch.randn(1) * 0.5
    noise = torch.randn(task_size, 1) * 0.1

    y = a * x**2 + b * x + c + noise

    return x, y


def create_analysis_model() -> nn.Module:
    """
    Create a model for embedding analysis.

    Returns:
        Neural network model
    """
    return nn.Sequential(
        nn.Linear(1, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )


def analyze_embedding_space(embeddings: List[np.ndarray],
                          losses: List[float],
                          task_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the embedding space structure and properties.

    Args:
        embeddings: List of embedding vectors
        losses: List of corresponding loss values
        task_metadata: List of task metadata

    Returns:
        Dictionary with embedding analysis results
    """
    if len(embeddings) == 0:
        return {'error': 'No embeddings to analyze'}

    # Ensure all embeddings are numpy arrays with consistent shapes
    processed_embeddings = []
    filtered_losses = []
    filtered_metadata = []

    for i, emb in enumerate(embeddings):
        if emb is not None:
            # Convert to numpy array if it's not already
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            elif not isinstance(emb, np.ndarray):
                emb = np.array(emb)

            # Ensure it's 1D and has the expected shape
            if emb.ndim > 1:
                emb = emb.flatten()

            processed_embeddings.append(emb)

            # Add corresponding loss and metadata
            if i < len(losses):
                filtered_losses.append(losses[i])
            if i < len(task_metadata):
                filtered_metadata.append(task_metadata[i])
        else:
            # Skip None embeddings
            continue

    if len(processed_embeddings) == 0:
        return {'error': 'No valid embeddings to analyze'}

    # Check if all embeddings have the same shape
    embedding_shapes = [emb.shape for emb in processed_embeddings]
    if len(set(embedding_shapes)) > 1:
        logger.warning(f"Found embeddings with different shapes: {set(embedding_shapes)}")
        # Pad or truncate to the most common shape
        most_common_shape = max(set(embedding_shapes), key=embedding_shapes.count)
        for i, emb in enumerate(processed_embeddings):
            if emb.shape != most_common_shape:
                if emb.shape[0] < most_common_shape[0]:
                    # Pad with zeros
                    padded = np.zeros(most_common_shape)
                    padded[:emb.shape[0]] = emb
                    processed_embeddings[i] = padded
                else:
                    # Truncate
                    processed_embeddings[i] = emb[:most_common_shape[0]]

    embeddings_array = np.array(processed_embeddings)

    analysis = {}

    # Basic statistics
    analysis['embedding_stats'] = {
        'mean': np.mean(embeddings_array),
        'std': np.std(embeddings_array),
        'min': np.min(embeddings_array),
        'max': np.max(embeddings_array),
        'norm_mean': np.mean([np.linalg.norm(emb) for emb in processed_embeddings]),
        'norm_std': np.std([np.linalg.norm(emb) for emb in processed_embeddings])
    }

    # Dimensionality analysis
    analysis['dimensionality'] = analyze_embedding_dimensions(embeddings_array)

    # Clustering analysis
    analysis['clustering'] = analyze_embedding_clusters(embeddings_array, filtered_losses)

    # Temporal analysis
    analysis['temporal'] = analyze_temporal_patterns(processed_embeddings, filtered_metadata)

    # Loss correlation analysis
    analysis['loss_correlation'] = analyze_loss_correlations(embeddings_array, filtered_losses)

    return analysis


def analyze_embedding_dimensions(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the importance of different embedding dimensions.

    Args:
        embeddings: Embedding array

    Returns:
        Dictionary with dimensionality analysis
    """
    # PCA analysis
    pca = PCA()
    pca.fit(embeddings)

    # Variance explained by each component
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find number of components for different variance thresholds
    n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

    return {
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'n_components_90': n_components_90,
        'n_components_95': n_components_95,
        'n_components_99': n_components_99,
        'effective_dimensionality': n_components_95  # Use 95% variance as effective dim
    }


def analyze_embedding_clusters(embeddings: np.ndarray, losses: List[float]) -> Dict[str, Any]:
    """
    Analyze clustering patterns in the embedding space.

    Args:
        embeddings: Embedding array
        losses: List of loss values

    Returns:
        Dictionary with clustering analysis
    """
    # K-means clustering
    n_clusters = min(5, len(embeddings) // 10)  # Adaptive number of clusters
    if n_clusters < 2:
        return {'error': 'Insufficient data for clustering'}

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Analyze clusters
    cluster_analysis = {}
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_losses = [loss for j, loss in enumerate(losses) if cluster_mask[j]]

        cluster_analysis[f'cluster_{i}'] = {
            'size': np.sum(cluster_mask),
            'mean_loss': np.mean(cluster_losses),
            'std_loss': np.std(cluster_losses),
            'min_loss': np.min(cluster_losses),
            'max_loss': np.max(cluster_losses)
        }

    return {
        'n_clusters': n_clusters,
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_,
        'cluster_analysis': cluster_analysis,
        'inertia': kmeans.inertia_
    }


def analyze_temporal_patterns(embeddings: List[np.ndarray],
                            task_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze temporal patterns in embeddings across tasks and steps.

    Args:
        embeddings: List of embedding vectors
        task_metadata: List of task metadata

    Returns:
        Dictionary with temporal analysis
    """
    if len(embeddings) == 0:
        return {'error': 'No embeddings for temporal analysis'}

    embeddings_array = np.array(embeddings)

    # Analyze embedding evolution within tasks
    task_evolution = {}

    # Since we filtered out None embeddings, we need to reconstruct task metadata
    # that matches the filtered embeddings list
    if len(task_metadata) != len(embeddings):
        logger.warning(f"Task metadata length ({len(task_metadata)}) doesn't match embeddings length ({len(embeddings)})")
        # Create synthetic task metadata for the embeddings we have
        synthetic_metadata = []
        for i in range(len(embeddings)):
            synthetic_metadata.append({
                'task_idx': i // 20,  # Assuming 20 steps per task
                'step': i % 20,
                'total_steps': 20
            })
        task_metadata = synthetic_metadata

    unique_tasks = set(meta['task_idx'] for meta in task_metadata)

    for task_idx in unique_tasks:
        task_mask = [meta['task_idx'] == task_idx for meta in task_metadata]
        task_embeddings = embeddings_array[task_mask]

        if len(task_embeddings) > 1:
            # Compute embedding trajectory
            trajectory_length = np.sum(np.linalg.norm(np.diff(task_embeddings, axis=0), axis=1))
            final_distance = np.linalg.norm(task_embeddings[-1] - task_embeddings[0])

            task_evolution[task_idx] = {
                'trajectory_length': trajectory_length,
                'final_distance': final_distance,
                'efficiency': final_distance / trajectory_length if trajectory_length > 0 else 0,
                'embedding_variance': np.var(task_embeddings)
            }

    # Analyze cross-task patterns
    cross_task_analysis = analyze_cross_task_patterns(embeddings_array, task_metadata)

    return {
        'task_evolution': task_evolution,
        'cross_task_patterns': cross_task_analysis
    }


def analyze_cross_task_patterns(embeddings: np.ndarray,
                               task_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns across different tasks.

    Args:
        embeddings: Embedding array
        task_metadata: List of task metadata

    Returns:
        Dictionary with cross-task analysis
    """
    unique_tasks = list(set(meta['task_idx'] for meta in task_metadata))

    # Compute task centroids
    task_centroids = {}
    for task_idx in unique_tasks:
        task_mask = [meta['task_idx'] == task_idx for meta in task_metadata]
        task_embeddings = embeddings[task_mask]
        task_centroids[task_idx] = np.mean(task_embeddings, axis=0)

    # Compute inter-task distances
    inter_task_distances = {}
    for i, task1 in enumerate(unique_tasks):
        for j, task2 in enumerate(unique_tasks):
            if i < j:
                distance = np.linalg.norm(task_centroids[task1] - task_centroids[task2])
                inter_task_distances[f'{task1}_{task2}'] = distance

    return {
        'task_centroids': task_centroids,
        'inter_task_distances': inter_task_distances,
        'mean_inter_task_distance': np.mean(list(inter_task_distances.values()))
    }


def analyze_loss_correlations(embeddings: np.ndarray, losses: List[float]) -> Dict[str, Any]:
    """
    Analyze correlations between embedding dimensions and loss values.

    Args:
        embeddings: Embedding array
        losses: List of loss values

    Returns:
        Dictionary with loss correlation analysis
    """
    correlations = []
    for i in range(embeddings.shape[1]):
        correlation = np.corrcoef(embeddings[:, i], losses)[0, 1]
        correlations.append(correlation)

    # Find most correlated dimensions
    abs_correlations = np.abs(correlations)
    most_correlated_dims = np.argsort(abs_correlations)[::-1]

    return {
        'dimension_correlations': correlations,
        'most_correlated_dimensions': most_correlated_dims,
        'max_correlation': max(abs_correlations),
        'mean_correlation': np.mean(abs_correlations)
    }


def analyze_adaptation_patterns(adaptation_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze learning rate and momentum adaptation patterns.

    Args:
        adaptation_stats: List of adaptation statistics

    Returns:
        Dictionary with adaptation analysis
    """
    if len(adaptation_stats) == 0:
        return {'error': 'No adaptation stats to analyze'}

    # Extract adaptation values
    lr_multipliers = [stats['lr_multiplier'] for stats in adaptation_stats]
    momentum_factors = [stats['momentum_factor'] for stats in adaptation_stats]
    adaptive_lrs = [stats['adaptive_lr'] for stats in adaptation_stats]

    analysis = {}

    # Statistical analysis
    analysis['lr_multiplier'] = {
        'mean': np.mean(lr_multipliers),
        'std': np.std(lr_multipliers),
        'min': np.min(lr_multipliers),
        'max': np.max(lr_multipliers),
        'range': np.max(lr_multipliers) - np.min(lr_multipliers)
    }

    analysis['momentum_factor'] = {
        'mean': np.mean(momentum_factors),
        'std': np.std(momentum_factors),
        'min': np.min(momentum_factors),
        'max': np.max(momentum_factors),
        'range': np.max(momentum_factors) - np.min(momentum_factors)
    }

    analysis['adaptive_lr'] = {
        'mean': np.mean(adaptive_lrs),
        'std': np.std(adaptive_lrs),
        'min': np.min(adaptive_lrs),
        'max': np.max(adaptive_lrs),
        'range': np.max(adaptive_lrs) - np.min(adaptive_lrs)
    }

    # Adaptation stability
    analysis['stability'] = {
        'lr_stability': np.var(lr_multipliers),
        'momentum_stability': np.var(momentum_factors),
        'lr_momentum_correlation': np.corrcoef(lr_multipliers, momentum_factors)[0, 1]
    }

    return analysis


def create_analysis_visualizations(embeddings: List[np.ndarray],
                                 losses: List[float],
                                 adaptation_stats: List[Dict[str, Any]],
                                 embedding_analysis: Dict[str, Any],
                                 adaptation_analysis: Dict[str, Any]):
    """
    Create comprehensive visualizations for embedding analysis.

    Args:
        embeddings: List of embedding vectors
        losses: List of loss values
        adaptation_stats: List of adaptation statistics
        embedding_analysis: Embedding analysis results
        adaptation_analysis: Adaptation analysis results
    """
    # 1. Embedding evolution
    plot_embedding_evolution(embeddings, losses, "Opt2Vec Embedding Analysis")

    # 2. Adaptation patterns
    plot_adaptation_patterns(adaptation_stats, "Opt2Vec Adaptation Analysis")

    # 3. Dimensionality analysis
    if 'dimensionality' in embedding_analysis:
        plot_dimensionality_analysis(embedding_analysis['dimensionality'])

    # 4. Clustering analysis
    if 'clustering' in embedding_analysis:
        plot_clustering_analysis(embeddings, embedding_analysis['clustering'])

    # 5. Loss correlation analysis
    if 'loss_correlation' in embedding_analysis:
        plot_loss_correlation_analysis(embedding_analysis['loss_correlation'])


def plot_dimensionality_analysis(dimensionality_analysis: Dict[str, Any]):
    """Plot dimensionality analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Explained variance ratio
    explained_variance = dimensionality_analysis['explained_variance_ratio']
    ax1.bar(range(len(explained_variance)), explained_variance)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('PCA Explained Variance')
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    cumulative_variance = dimensionality_analysis['cumulative_variance']
    ax2.plot(range(len(cumulative_variance)), cumulative_variance, 'b-', linewidth=2)
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90%')
    ax2.axhline(y=0.95, color='g', linestyle='--', label='95%')
    ax2.axhline(y=0.99, color='orange', linestyle='--', label='99%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_clustering_analysis(embeddings: List[np.ndarray], clustering_analysis: Dict[str, Any]):
    """Plot clustering analysis results."""
    if 'error' in clustering_analysis:
        return

    # Process embeddings the same way as in analyze_embedding_space
    processed_embeddings = []
    for emb in embeddings:
        if emb is not None:
            # Convert to numpy array if it's not already
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            elif not isinstance(emb, np.ndarray):
                emb = np.array(emb)

            # Ensure it's 1D and has the expected shape
            if emb.ndim > 1:
                emb = emb.flatten()

            processed_embeddings.append(emb)
        else:
            # Skip None embeddings
            continue

    if len(processed_embeddings) == 0:
        logger.warning("No valid embeddings for clustering visualization")
        return

    # Check if all embeddings have the same shape
    embedding_shapes = [emb.shape for emb in processed_embeddings]
    if len(set(embedding_shapes)) > 1:
        logger.warning(f"Found embeddings with different shapes: {set(embedding_shapes)}")
        # Pad or truncate to the most common shape
        most_common_shape = max(set(embedding_shapes), key=embedding_shapes.count)
        for i, emb in enumerate(processed_embeddings):
            if emb.shape != most_common_shape:
                if emb.shape[0] < most_common_shape[0]:
                    # Pad with zeros
                    padded = np.zeros(most_common_shape)
                    padded[:emb.shape[0]] = emb
                    processed_embeddings[i] = padded
                else:
                    # Truncate
                    processed_embeddings[i] = emb[:most_common_shape[0]]

    embeddings_array = np.array(processed_embeddings)

    # Reduce dimensionality for visualization
    if embeddings_array.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_array)
    else:
        embeddings_2d = embeddings_array

    # Plot clusters
    plt.figure(figsize=(10, 8))

    cluster_labels = clustering_analysis['cluster_labels']
    cluster_centers = clustering_analysis['cluster_centers']

    # Plot data points
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=cluster_labels, cmap='viridis', alpha=0.7)

    # Plot cluster centers (if 2D)
    if cluster_centers.shape[1] > 2:
        centers_2d = tsne.fit_transform(cluster_centers)
    else:
        centers_2d = cluster_centers

    plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='red', s=200, marker='x', linewidths=3, label='Cluster Centers')

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f'Embedding Clusters (K={clustering_analysis["n_clusters"]})')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_loss_correlation_analysis(correlation_analysis: Dict[str, Any]):
    """Plot loss correlation analysis results."""
    correlations = correlation_analysis['dimension_correlations']
    most_correlated = correlation_analysis['most_correlated_dimensions']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Correlation heatmap
    correlation_matrix = np.array(correlations).reshape(1, -1)
    im = ax1.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto')
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss-Embedding Correlations')
    ax1.set_xticks(range(len(correlations)))
    plt.colorbar(im, ax=ax1)

    # Top correlated dimensions
    top_k = min(5, len(most_correlated))
    top_dims = most_correlated[:top_k]
    top_correlations = [correlations[i] for i in top_dims]

    bars = ax2.bar(range(top_k), top_correlations, alpha=0.7)
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Correlation with Loss')
    ax2.set_title(f'Top {top_k} Most Correlated Dimensions')
    ax2.set_xticks(range(top_k))
    ax2.set_xticklabels([f'Dim {i}' for i in top_dims])
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, top_correlations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run analysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = run_embedding_analysis(device=device, num_tasks=10, steps_per_task=20)

    print("\nEmbedding analysis completed successfully!")
