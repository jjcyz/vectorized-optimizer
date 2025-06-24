"""
Metrics computation utilities for optimization evaluation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def compute_optimization_metrics(losses: List[float],
                               embeddings: Optional[List[np.ndarray]] = None,
                               adaptation_stats: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
    """
    Compute comprehensive optimization metrics.

    Args:
        losses: List of loss values during training
        embeddings: Optional list of embedding vectors
        adaptation_stats: Optional list of adaptation statistics

    Returns:
        Dictionary with computed metrics
    """
    if len(losses) < 2:
        return {'error': 'Insufficient data for metrics computation'}

    metrics = {}

    # Basic loss metrics
    metrics['initial_loss'] = losses[0]
    metrics['final_loss'] = losses[-1]
    metrics['total_improvement'] = losses[0] - losses[-1]
    metrics['relative_improvement'] = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0

    # Convergence metrics
    metrics['convergence_rate'] = compute_convergence_rate(losses)
    metrics['steps_to_90_percent'] = compute_steps_to_percentage(losses, 0.9)
    metrics['steps_to_95_percent'] = compute_steps_to_percentage(losses, 0.95)

    # Stability metrics
    metrics['loss_variance'] = np.var(losses)
    metrics['loss_std'] = np.std(losses)
    metrics['smoothness'] = compute_smoothness(losses)

    # Embedding metrics (if available)
    if embeddings is not None and len(embeddings) > 0:
        embedding_metrics = compute_embedding_metrics(embeddings)
        metrics.update(embedding_metrics)

    # Adaptation metrics (if available)
    if adaptation_stats is not None and len(adaptation_stats) > 0:
        adaptation_metrics = compute_adaptation_metrics(adaptation_stats)
        metrics.update(adaptation_metrics)

    return metrics


def compute_convergence_rate(losses: List[float]) -> float:
    """
    Compute convergence rate based on exponential decay fit.

    Args:
        losses: List of loss values

    Returns:
        Convergence rate (higher = faster convergence)
    """
    if len(losses) < 3:
        return 0.0

    # Fit exponential decay: loss(t) = a * exp(-b*t) + c
    try:
        t = np.arange(len(losses))
        log_losses = np.log(np.maximum(losses, 1e-8))

        # Simple linear fit to log(losses)
        coeffs = np.polyfit(t, log_losses, 1)
        convergence_rate = -coeffs[0]  # Negative slope = convergence rate

        return max(0.0, convergence_rate)  # Ensure non-negative
    except:
        return 0.0


def compute_steps_to_percentage(losses: List[float], percentage: float) -> int:
    """
    Compute number of steps to reach a certain percentage of total improvement.

    Args:
        losses: List of loss values
        percentage: Target percentage (0.0 to 1.0)

    Returns:
        Number of steps to reach target
    """
    if len(losses) < 2:
        return len(losses)

    initial_loss = losses[0]
    final_loss = losses[-1]
    target_loss = initial_loss - percentage * (initial_loss - final_loss)

    for i, loss in enumerate(losses):
        if loss <= target_loss:
            return i

    return len(losses)


def compute_smoothness(losses: List[float]) -> float:
    """
    Compute smoothness of loss curve (lower = smoother).

    Args:
        losses: List of loss values

    Returns:
        Smoothness metric (mean absolute difference between consecutive losses)
    """
    if len(losses) < 2:
        return 0.0

    differences = np.abs(np.diff(losses))
    return np.mean(differences)


def compute_embedding_metrics(embeddings: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute metrics related to embedding vectors.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Dictionary with embedding metrics
    """
    if len(embeddings) < 2:
        return {}

    # Handle embeddings with inconsistent shapes by flattening them
    flattened_embeddings = []
    for emb in embeddings:
        if emb is not None:
            flattened_embeddings.append(emb.flatten())
        else:
            # Handle None embeddings by using zeros
            if flattened_embeddings:
                flattened_embeddings.append(np.zeros_like(flattened_embeddings[0]))
            else:
                # If this is the first embedding and it's None, skip
                continue

    if len(flattened_embeddings) < 2:
        return {}

    # Ensure all embeddings have the same shape
    max_dim = max(emb.shape[0] for emb in flattened_embeddings)
    padded_embeddings = []
    for emb in flattened_embeddings:
        if emb.shape[0] < max_dim:
            # Pad with zeros if needed
            padded = np.zeros(max_dim)
            padded[:emb.shape[0]] = emb
            padded_embeddings.append(padded)
        else:
            padded_embeddings.append(emb)

    embeddings_array = np.array(padded_embeddings)

    metrics = {}

    # Embedding statistics
    metrics['embedding_mean'] = np.mean(embeddings_array)
    metrics['embedding_std'] = np.std(embeddings_array)
    metrics['embedding_norm_mean'] = np.mean([np.linalg.norm(emb) for emb in padded_embeddings])
    metrics['embedding_norm_std'] = np.std([np.linalg.norm(emb) for emb in padded_embeddings])

    # Embedding diversity (how much embeddings change)
    embedding_diffs = np.diff(embeddings_array, axis=0)
    metrics['embedding_change_mean'] = np.mean(np.linalg.norm(embedding_diffs, axis=1))
    metrics['embedding_change_std'] = np.std(np.linalg.norm(embedding_diffs, axis=1))

    # Embedding stability (lower = more stable)
    metrics['embedding_stability'] = np.var(embeddings_array)

    return metrics


def compute_adaptation_metrics(adaptation_stats: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute metrics related to adaptation behavior.

    Args:
        adaptation_stats: List of adaptation statistics dictionaries

    Returns:
        Dictionary with adaptation metrics
    """
    if len(adaptation_stats) < 2:
        return {}

    metrics = {}

    # Extract adaptation values
    lr_multipliers = [stats['lr_multiplier'] for stats in adaptation_stats]
    momentum_factors = [stats['momentum_factor'] for stats in adaptation_stats]
    adaptive_lrs = [stats['adaptive_lr'] for stats in adaptation_stats]

    # Learning rate adaptation metrics
    metrics['lr_multiplier_mean'] = np.mean(lr_multipliers)
    metrics['lr_multiplier_std'] = np.std(lr_multipliers)
    metrics['lr_multiplier_range'] = max(lr_multipliers) - min(lr_multipliers)

    # Momentum adaptation metrics
    metrics['momentum_factor_mean'] = np.mean(momentum_factors)
    metrics['momentum_factor_std'] = np.std(momentum_factors)
    metrics['momentum_factor_range'] = max(momentum_factors) - min(momentum_factors)

    # Adaptive learning rate metrics
    metrics['adaptive_lr_mean'] = np.mean(adaptive_lrs)
    metrics['adaptive_lr_std'] = np.std(adaptive_lrs)
    metrics['adaptive_lr_range'] = max(adaptive_lrs) - min(adaptive_lrs)

    # Adaptation stability
    metrics['lr_adaptation_stability'] = np.var(lr_multipliers)
    metrics['momentum_adaptation_stability'] = np.var(momentum_factors)

    return metrics


def compare_optimizers(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """
    Compare multiple optimizers and rank their performance.

    Args:
        results: Dictionary mapping optimizer names to result dictionaries

    Returns:
        Dictionary with comparison results
    """
    comparison = {}

    # Compute metrics for each optimizer
    optimizer_metrics = {}
    for name, data in results.items():
        metrics = compute_optimization_metrics(
            data['losses'],
            data.get('embeddings'),
            data.get('adaptation_stats')
        )
        optimizer_metrics[name] = metrics

    # Rank optimizers by different criteria
    comparison['final_loss_ranking'] = rank_by_metric(optimizer_metrics, 'final_loss', ascending=True)
    comparison['convergence_rate_ranking'] = rank_by_metric(optimizer_metrics, 'convergence_rate', ascending=False)
    comparison['steps_to_90_percent_ranking'] = rank_by_metric(optimizer_metrics, 'steps_to_90_percent', ascending=True)
    comparison['smoothness_ranking'] = rank_by_metric(optimizer_metrics, 'smoothness', ascending=True)

    # Overall ranking (average of normalized rankings)
    comparison['overall_ranking'] = compute_overall_ranking(optimizer_metrics)

    # Statistical significance (if multiple runs available)
    if all('losses' in data for data in results.values()):
        comparison['statistical_tests'] = perform_statistical_tests(results)

    comparison['detailed_metrics'] = optimizer_metrics

    return comparison


def rank_by_metric(metrics: Dict[str, Dict[str, float]],
                  metric_name: str,
                  ascending: bool = True) -> List[Tuple[str, float]]:
    """
    Rank optimizers by a specific metric.

    Args:
        metrics: Dictionary of optimizer metrics
        metric_name: Name of metric to rank by
        ascending: Whether lower values are better

    Returns:
        List of (optimizer_name, metric_value) tuples, sorted by rank
    """
    valid_metrics = {name: data[metric_name]
                    for name, data in metrics.items()
                    if metric_name in data}

    if not valid_metrics:
        return []

    sorted_items = sorted(valid_metrics.items(),
                         key=lambda x: x[1],
                         reverse=not ascending)

    return sorted_items


def compute_overall_ranking(metrics: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
    """
    Compute overall ranking based on multiple metrics.

    Args:
        metrics: Dictionary of optimizer metrics

    Returns:
        List of (optimizer_name, overall_score) tuples, sorted by rank
    """
    # Key metrics for overall ranking
    key_metrics = ['final_loss', 'convergence_rate', 'steps_to_90_percent', 'smoothness']

    # Normalize each metric to [0, 1] range
    normalized_scores = {}

    for metric_name in key_metrics:
        values = [data.get(metric_name, 0) for data in metrics.values()]
        if not values or all(v == 0 for v in values):
            continue

        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            continue

        for optimizer_name, data in metrics.items():
            if optimizer_name not in normalized_scores:
                normalized_scores[optimizer_name] = []

            value = data.get(metric_name, 0)
            normalized = (value - min_val) / (max_val - min_val)

            # Invert metrics where lower is better
            if metric_name in ['final_loss', 'steps_to_90_percent', 'smoothness']:
                normalized = 1 - normalized

            normalized_scores[optimizer_name].append(normalized)

    # Compute average normalized score
    overall_scores = {}
    for optimizer_name, scores in normalized_scores.items():
        overall_scores[optimizer_name] = np.mean(scores)

    # Sort by overall score (higher is better)
    sorted_items = sorted(overall_scores.items(),
                         key=lambda x: x[1],
                         reverse=True)

    return sorted_items


def perform_statistical_tests(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    """
    Perform statistical tests to compare optimizers.

    Args:
        results: Dictionary mapping optimizer names to result dictionaries

    Returns:
        Dictionary with statistical test results
    """
    # This is a placeholder for statistical tests
    # In a real implementation, you might use t-tests, ANOVA, etc.

    return {
        'note': 'Statistical tests not implemented in this version',
        'suggested_tests': ['paired_t_test', 'wilcoxon_signed_rank', 'friedman_test']
    }
