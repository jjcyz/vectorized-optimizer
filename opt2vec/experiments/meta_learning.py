"""
Meta-learning experiment for training Opt2Vec optimizer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional

from ..core.trainer import EfficientMetaLearningTrainer
from ..core.optimizer import LightweightOpt2VecOptimizer
from ..utils.memory import clear_memory, get_memory_usage
from ..utils.visualization import plot_training_curves, plot_embedding_evolution
from ..utils.metrics import compute_optimization_metrics

logger = logging.getLogger(__name__)


def run_meta_learning_experiment(device: torch.device = torch.device('cpu'),
                                num_meta_steps: int = 50,
                                num_tasks_per_step: int = 3,
                                meta_lr: float = 1e-3,
                                inner_steps: int = 5) -> Dict[str, Any]:
    """
    Run meta-learning experiment to train Opt2Vec optimizer.

    Args:
        device: Target device for computation
        num_meta_steps: Number of meta-training steps
        num_tasks_per_step: Number of tasks per meta-step
        meta_lr: Learning rate for meta-optimizer
        inner_steps: Number of inner loop steps

    Returns:
        Dictionary with experiment results
    """
    logger.info("Starting Opt2Vec meta-learning experiment...")
    logger.info(f"Device: {device}")
    logger.info(f"Meta-steps: {num_meta_steps}")
    logger.info(f"Tasks per step: {num_tasks_per_step}")
    logger.info(f"Meta learning rate: {meta_lr}")
    logger.info(f"Inner steps: {inner_steps}")

    # Initialize meta-trainer
    trainer = EfficientMetaLearningTrainer(device=device)

    # Monitor memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

    # Run meta-training
    start_time = time.time()
    meta_results = trainer.meta_train(
        num_meta_steps=num_meta_steps,
        num_tasks_per_step=num_tasks_per_step,
        meta_lr=meta_lr,
        inner_steps=inner_steps
    )
    total_time = time.time() - start_time

    # Final memory usage
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory

    logger.info(f"Meta-training completed in {total_time:.2f} seconds")
    logger.info(f"Final memory usage: {final_memory:.2f} MB (Δ: {memory_increase:+.2f} MB)")

    # Analyze results
    logger.info("\n" + "="*50)
    logger.info("META-LEARNING RESULTS")
    logger.info("="*50)

    meta_losses = meta_results['meta_losses']
    improvements = meta_results['improvements']

    # Filter out NaN values for analysis
    valid_meta_losses = [loss for loss in meta_losses if not (np.isnan(loss) or np.isinf(loss))]
    valid_improvements = [imp for imp in improvements if not (np.isnan(imp) or np.isinf(imp))]

    logger.info(f"Initial meta-loss: {valid_meta_losses[0]:.4f}")
    logger.info(f"Final meta-loss: {valid_meta_losses[-1]:.4f}")
    logger.info(f"Meta-loss improvement: {valid_meta_losses[0] - valid_meta_losses[-1]:.4f}")
    logger.info(f"Average task improvement: {np.mean(valid_improvements):.4f}")

    # Compute convergence metrics
    convergence_rate = compute_convergence_rate(meta_losses)
    logger.info(f"Meta-learning convergence rate: {convergence_rate:.4f}")

    # Create visualizations
    logger.info("\nCreating visualizations...")

    # Meta-loss curve
    plot_training_curves(
        {'Meta-Loss': meta_losses},
        "Opt2Vec Meta-Learning Loss"
    )

    # Task improvement curve
    plot_training_curves(
        {'Task Improvement': improvements},
        "Task Improvement Over Meta-Steps"
    )

    # Test meta-learned optimizer
    logger.info("\nTesting meta-learned optimizer...")
    test_results = test_meta_learned_optimizer(
        meta_results['opt2vec_components'],
        device=device
    )

    return {
        'meta_results': meta_results,
        'test_results': test_results,
        'experiment_config': {
            'device': str(device),
            'num_meta_steps': num_meta_steps,
            'num_tasks_per_step': num_tasks_per_step,
            'meta_lr': meta_lr,
            'inner_steps': inner_steps,
            'total_time': total_time,
            'memory_usage': {
                'initial': initial_memory,
                'final': final_memory,
                'increase': memory_increase
            }
        }
    }


def compute_convergence_rate(losses: List[float]) -> float:
    """
    Compute convergence rate for meta-learning.

    Args:
        losses: List of meta-loss values

    Returns:
        Convergence rate
    """
    if len(losses) < 3:
        return 0.0

    try:
        # Filter out NaN and extreme values
        valid_losses = [loss for loss in losses if not (np.isnan(loss) or np.isinf(loss) or loss > 1000)]

        if len(valid_losses) < 3:
            return 0.0

        # Fit exponential decay
        t = np.arange(len(valid_losses))
        log_losses = np.log(np.maximum(valid_losses, 1e-8))
        coeffs = np.polyfit(t, log_losses, 1)
        convergence_rate = -coeffs[0]
        return max(0.0, convergence_rate)
    except:
        return 0.0


def test_meta_learned_optimizer(opt2vec_components: Dict[str, nn.Module],
                               device: torch.device = torch.device('cpu'),
                               num_test_tasks: int = 8) -> Dict[str, Any]:
    """
    Test the meta-learned optimizer on new tasks.

    Args:
        opt2vec_components: Meta-learned Opt2Vec components
        device: Target device
        num_test_tasks: Number of test tasks

    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing on {num_test_tasks} new tasks...")

    test_results = []

    for task_idx in range(num_test_tasks):
        # Create test task
        trainer = EfficientMetaLearningTrainer(device=device)
        data, targets = trainer.create_tiny_task(task_size=60)
        model = trainer.create_tiny_model()

        # Initialize optimizer with meta-learned parameters
        optimizer = LightweightOpt2VecOptimizer(
            model.parameters(),
            base_lr=0.01,
            embedding_dim=16,
            history_length=5,
            device=device
        )

        # Copy meta-learned parameters
        trainer._copy_meta_parameters(opt2vec_components, optimizer)

        # Train on test task
        losses = trainer.quick_inner_loop(model, data, targets, optimizer, steps=12)

        # Compute metrics
        metrics = compute_optimization_metrics(losses)

        test_results.append({
            'task_idx': task_idx,
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'improvement': losses[0] - losses[-1],
            'metrics': metrics
        })

        logger.info(f"Task {task_idx}: Loss {losses[0]:.4f} → {losses[-1]:.4f} "
                   f"(improvement: {losses[0] - losses[-1]:.4f})")

        # Clear memory
        clear_memory()

    # Aggregate results
    avg_improvement = np.mean([r['improvement'] for r in test_results])
    avg_final_loss = np.mean([r['final_loss'] for r in test_results])

    logger.info(f"\nTest Results Summary:")
    logger.info(f"Average improvement: {avg_improvement:.4f}")
    logger.info(f"Average final loss: {avg_final_loss:.4f}")

    return {
        'task_results': test_results,
        'avg_improvement': avg_improvement,
        'avg_final_loss': avg_final_loss
    }


def compare_meta_learning_variants(device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
    """
    Compare different meta-learning configurations.

    Args:
        device: Target device

    Returns:
        Dictionary with comparison results
    """
    logger.info("Comparing meta-learning variants...")

    variants = {
        'Fast': {'num_meta_steps': 20, 'num_tasks_per_step': 2, 'inner_steps': 3},
        'Standard': {'num_meta_steps': 50, 'num_tasks_per_step': 3, 'inner_steps': 5},
        'Thorough': {'num_meta_steps': 100, 'num_tasks_per_step': 5, 'inner_steps': 8}
    }

    results = {}

    for name, config in variants.items():
        logger.info(f"\nTesting {name} variant...")

        try:
            result = run_meta_learning_experiment(
                device=device,
                **config
            )
            results[name] = result
        except Exception as e:
            logger.error(f"Error in {name} variant: {e}")
            results[name] = {'error': str(e)}

        # Clear memory between variants
        clear_memory()

    # Compare results
    logger.info("\n" + "="*50)
    logger.info("VARIANT COMPARISON")
    logger.info("="*50)

    for name, result in results.items():
        if 'error' in result:
            logger.info(f"{name}: ERROR - {result['error']}")
        else:
            meta_losses = result['meta_results']['meta_losses']
            final_loss = meta_losses[-1] if meta_losses else float('inf')
            training_time = result['experiment_config']['total_time']

            logger.info(f"{name}:")
            logger.info(f"  Final meta-loss: {final_loss:.4f}")
            logger.info(f"  Training time: {training_time:.2f}s")

    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run improved meta-learning experiment with 60 steps
    results = run_meta_learning_experiment(
        device=device,
        num_meta_steps=60,  # Increased from 50
        num_tasks_per_step=3,  # Reduced from 4 for stability
        meta_lr=2e-4,  # Further reduced from 5e-4 for better stability
        inner_steps=6  # Reduced from 8 for stability
    )

    print("\nMeta-learning experiment completed successfully!")

    # Optionally run variant comparison
    # variant_results = compare_meta_learning_variants(device)
