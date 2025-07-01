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
                                num_meta_steps: int = 60,  # Increased from 50
                                num_tasks_per_step: int = 3,
                                meta_lr: float = 2e-4,  # Reduced for stability
                                inner_steps: int = 6) -> Dict[str, Any]:  # Increased from 5
    """
    Run meta-learning experiment to train Opt2Vec optimizer with BEST configurations.

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
    logger.info("Using BEST configurations:")
    logger.info("  - Embedding dim: 64")
    logger.info("  - History length: 8")
    logger.info("  - Activation: gelu")
    logger.info("  - Extended features: True")
    logger.info("  - Attention: True")
    logger.info("  - Positional encoding: True")

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
    Test the meta-learned optimizer against handcrafted optimizers on the same tasks.

    Args:
        opt2vec_components: Meta-learned Opt2Vec components
        device: Target device
        num_test_tasks: Number of test tasks

    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing on {num_test_tasks} tasks with Opt2Vec vs Adam vs SGD...")

    test_results = []
    all_learning_curves = {'Opt2Vec': [], 'Adam': [], 'SGD': []}

    for task_idx in range(num_test_tasks):
        logger.info(f"\nTask {task_idx + 1}/{num_test_tasks}")

        # Create test task (same for all optimizers)
        trainer = EfficientMetaLearningTrainer(device=device)
        data, targets = trainer.create_tiny_task(task_size=60)

        task_results = {
            'task_idx': task_idx,
            'optimizers': {}
        }

        # Test Opt2Vec
        logger.info("  Testing Opt2Vec...")
        model_opt2vec = trainer.create_tiny_model()
        optimizer_opt2vec = LightweightOpt2VecOptimizer(
            model_opt2vec.parameters(),
            base_lr=0.01,
            embedding_dim=64,
            history_length=8,
            activation='gelu',
            device=device,
            use_extended_features=True,
            normalize_features=True,
            dropout=0.1,
            use_layer_norm=True,
            use_attention=True,
            use_positional_encoding=True
        )
        trainer._copy_meta_parameters(opt2vec_components, optimizer_opt2vec)

        start_time = time.time()
        opt2vec_losses = trainer.quick_inner_loop(model_opt2vec, data, targets, optimizer_opt2vec, steps=12)
        opt2vec_time = time.time() - start_time

        opt2vec_metrics = compute_optimization_metrics(opt2vec_losses)
        task_results['optimizers']['Opt2Vec'] = {
            'initial_loss': opt2vec_losses[0],
            'final_loss': opt2vec_losses[-1],
            'improvement': opt2vec_losses[0] - opt2vec_losses[-1],
            'training_time': opt2vec_time,
            'metrics': opt2vec_metrics,
            'losses': opt2vec_losses
        }
        all_learning_curves['Opt2Vec'].append(opt2vec_losses)

        # Test Adam
        logger.info("  Testing Adam...")
        model_adam = trainer.create_tiny_model()
        adam_results = trainer.quick_inner_loop_with_handcrafted(
            model_adam, data, targets, 'adam', steps=12
        )
        adam_metrics = compute_optimization_metrics(adam_results['losses'])
        task_results['optimizers']['Adam'] = {
            'initial_loss': adam_results['initial_loss'],
            'final_loss': adam_results['final_loss'],
            'improvement': adam_results['improvement'],
            'training_time': adam_results['training_time'],
            'metrics': adam_metrics,
            'losses': adam_results['losses']
        }
        all_learning_curves['Adam'].append(adam_results['losses'])

        # Test SGD
        logger.info("  Testing SGD...")
        model_sgd = trainer.create_tiny_model()
        sgd_results = trainer.quick_inner_loop_with_handcrafted(
            model_sgd, data, targets, 'sgd', steps=12
        )
        sgd_metrics = compute_optimization_metrics(sgd_results['losses'])
        task_results['optimizers']['SGD'] = {
            'initial_loss': sgd_results['initial_loss'],
            'final_loss': sgd_results['final_loss'],
            'improvement': sgd_results['improvement'],
            'training_time': sgd_results['training_time'],
            'metrics': sgd_metrics,
            'losses': sgd_results['losses']
        }
        all_learning_curves['SGD'].append(sgd_results['losses'])

        # Log results for this task
        logger.info(f"  Task {task_idx} Results:")
        for opt_name, opt_results in task_results['optimizers'].items():
            logger.info(f"    {opt_name}: Loss {opt_results['initial_loss']:.4f} → {opt_results['final_loss']:.4f} "
                       f"(improvement: {opt_results['improvement']:.4f}, time: {opt_results['training_time']:.3f}s)")

        test_results.append(task_results)

        # Clear memory
        clear_memory()

    # Aggregate results
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS SUMMARY")
    logger.info("="*60)

    summary = {}
    for opt_name in ['Opt2Vec', 'Adam', 'SGD']:
        improvements = [r['optimizers'][opt_name]['improvement'] for r in test_results]
        final_losses = [r['optimizers'][opt_name]['final_loss'] for r in test_results]
        training_times = [r['optimizers'][opt_name]['training_time'] for r in test_results]

        avg_improvement = np.mean(improvements)
        avg_final_loss = np.mean(final_losses)
        avg_training_time = np.mean(training_times)

        summary[opt_name] = {
            'avg_improvement': avg_improvement,
            'avg_final_loss': avg_final_loss,
            'avg_training_time': avg_training_time,
            'std_improvement': np.std(improvements),
            'std_final_loss': np.std(final_losses)
        }

        logger.info(f"{opt_name}:")
        logger.info(f"  Avg improvement: {avg_improvement:.4f} ± {np.std(improvements):.4f}")
        logger.info(f"  Avg final loss: {avg_final_loss:.4f} ± {np.std(final_losses):.4f}")
        logger.info(f"  Avg training time: {avg_training_time:.3f}s")

    # Create visualization
    logger.info("\nCreating learning curve comparison...")
    create_optimizer_comparison_plot(all_learning_curves, test_results)

    return {
        'task_results': test_results,
        'summary': summary,
        'learning_curves': all_learning_curves
    }


def compare_meta_learning_variants(device: torch.device = torch.device('cpu')) -> Dict[str, Any]:
    """
    Compare different meta-learning configurations.

    Args:
        device: Target device

    Returns:
        Dictionary with comparison results
    """
    logger.info("Comparing meta-learning variants with BEST configurations...")
    logger.info("All variants use: embedding_dim=64, history_length=8, activation=gelu")

    variants = {
        'Fast': {'num_meta_steps': 30, 'num_tasks_per_step': 2, 'inner_steps': 4},
        'Standard': {'num_meta_steps': 60, 'num_tasks_per_step': 3, 'inner_steps': 6},
        'Thorough': {'num_meta_steps': 100, 'num_tasks_per_step': 4, 'inner_steps': 8}
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


def create_optimizer_comparison_plot(learning_curves: Dict[str, List[List[float]]],
                                   test_results: List[Dict[str, Any]]):
    """
    Create comprehensive visualization comparing optimizers.

    Args:
        learning_curves: Dictionary with learning curves for each optimizer
        test_results: Detailed test results
    """
    import matplotlib.pyplot as plt

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Opt2Vec vs Handcrafted Optimizers Comparison', fontsize=16)

    # 1. Average learning curves
    ax1 = axes[0, 0]
    colors = {'Opt2Vec': 'blue', 'Adam': 'red', 'SGD': 'green'}

    for opt_name, curves in learning_curves.items():
        if curves:
            # Compute average curve
            max_len = max(len(curve) for curve in curves)
            padded_curves = []
            for curve in curves:
                if len(curve) < max_len:
                    # Pad with last value
                    padded_curve = curve + [curve[-1]] * (max_len - len(curve))
                else:
                    padded_curve = curve
                padded_curves.append(padded_curve)

            avg_curve = np.mean(padded_curves, axis=0)
            std_curve = np.std(padded_curves, axis=0)
            steps = np.arange(len(avg_curve))

            ax1.plot(steps, avg_curve, label=opt_name, color=colors[opt_name], linewidth=2)
            ax1.fill_between(steps, avg_curve - std_curve, avg_curve + std_curve,
                           alpha=0.2, color=colors[opt_name])

    ax1.set_title('Average Learning Curves')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Final loss comparison
    ax2 = axes[0, 1]
    opt_names = ['Opt2Vec', 'Adam', 'SGD']
    final_losses = []
    final_loss_stds = []

    for opt_name in opt_names:
        losses = [r['optimizers'][opt_name]['final_loss'] for r in test_results]
        final_losses.append(np.mean(losses))
        final_loss_stds.append(np.std(losses))

    bars = ax2.bar(opt_names, final_losses, yerr=final_loss_stds,
                   color=[colors[name] for name in opt_names], alpha=0.7)
    ax2.set_title('Final Loss Comparison')
    ax2.set_ylabel('Final Loss')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, final_losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')

    # 3. Improvement comparison
    ax3 = axes[1, 0]
    improvements = []
    improvement_stds = []

    for opt_name in opt_names:
        imps = [r['optimizers'][opt_name]['improvement'] for r in test_results]
        improvements.append(np.mean(imps))
        improvement_stds.append(np.std(imps))

    bars = ax3.bar(opt_names, improvements, yerr=improvement_stds,
                   color=[colors[name] for name in opt_names], alpha=0.7)
    ax3.set_title('Loss Improvement Comparison')
    ax3.set_ylabel('Loss Improvement')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, improvements):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')

    # 4. Training time comparison
    ax4 = axes[1, 1]
    training_times = []

    for opt_name in opt_names:
        times = [r['optimizers'][opt_name]['training_time'] for r in test_results]
        training_times.append(np.mean(times))

    bars = ax4.bar(opt_names, training_times,
                   color=[colors[name] for name in opt_names], alpha=0.7)
    ax4.set_title('Training Time Comparison')
    ax4.set_ylabel('Training Time (seconds)')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, training_times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('optimizer_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("DETAILED COMPARISON SUMMARY")
    print("="*60)

    for opt_name in opt_names:
        print(f"\n{opt_name}:")
        print(f"  Final Loss: {final_losses[opt_names.index(opt_name)]:.4f} ± {final_loss_stds[opt_names.index(opt_name)]:.4f}")
        print(f"  Improvement: {improvements[opt_names.index(opt_name)]:.4f} ± {improvement_stds[opt_names.index(opt_name)]:.4f}")
        print(f"  Training Time: {training_times[opt_names.index(opt_name)]:.3f}s")

    # Determine winner
    best_final_loss = min(final_losses)
    best_improvement = max(improvements)
    fastest_time = min(training_times)

    print(f"\nFinal Results:")
    print(f"  Best Final Loss: {opt_names[final_losses.index(best_final_loss)]} ({best_final_loss:.4f})")
    print(f"  Best Improvement: {opt_names[improvements.index(best_improvement)]} ({best_improvement:.4f})")
    print(f"  Fastest Training: {opt_names[training_times.index(fastest_time)]} ({fastest_time:.3f}s)")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Uses: embedding_dim=64, history_length=8, activation='gelu', extended_features=True
    results = run_meta_learning_experiment(device=device)

    print("\nMeta-learning experiment completed successfully!")
    print("\nNow testing meta-learned optimizer against handcrafted optimizers...")

    # The test_meta_learned_optimizer function is already called within run_meta_learning_experiment
    # and will automatically compare Opt2Vec vs Adam vs SGD on the same tasks

    print("\nCOMPARISON COMPLETED!")
    print("Check 'optimizer_comparison_results.png' for detailed visualizations.")

    # Optionally run variant comparison
    # variant_results = compare_meta_learning_variants(device)
