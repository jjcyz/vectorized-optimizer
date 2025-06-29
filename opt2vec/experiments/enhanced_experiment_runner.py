"""
Enhanced experiment runner for systematic testing of Opt2Vec architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from ..core.enhanced_optimizer import EnhancedOpt2VecOptimizer
from ..configs.enhanced_configs import EnhancedNetworkConfig, EnhancedExperimentConfigs

logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """
    Simple test model for experiments.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 50, output_dim: int = 1):
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


class EnhancedExperimentRunner:
    """
    Runner for systematic experiments with enhanced Opt2Vec.
    """

    def __init__(self,
                 output_dir: str = "experiment_results",
                 device: torch.device = torch.device('cpu')):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory to save experiment results
            device: Device for experiments
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.output_dir.mkdir(exist_ok=True)

        # Experiment tracking
        self.experiment_results = []
        self.current_experiment = None

    def run_single_experiment(self,
                            config: EnhancedNetworkConfig,
                            num_steps: int = 1000,
                            batch_size: int = 32,
                            input_dim: int = 10) -> Dict[str, Any]:
        """
        Run a single experiment with given configuration.

        Args:
            config: Network configuration
            num_steps: Number of training steps
            batch_size: Batch size for training
            input_dim: Input dimension for test model

        Returns:
            Experiment results
        """
        logger.info(f"Starting experiment with config: {config}")

        # Create test model
        model = SimpleTestModel(input_dim=input_dim).to(self.device)

        # Create enhanced optimizer
        optimizer = EnhancedOpt2VecOptimizer(
            parameters=model.parameters(),
            base_lr=config.base_lr,
            embedding_dim=config.embedding_dim,
            history_length=config.history_length,
            activation=config.activation,
            device=self.device,
            debug_mode=True,
            max_grad_norm=config.max_grad_norm,
            lr_bounds=config.lr_bounds,
            momentum_bounds=config.momentum_bounds,
            use_extended_features=config.use_extended_features,
            normalize_features=config.normalize_features,
            dropout=config.dropout,
            use_layer_norm=config.use_layer_norm,
            use_attention=config.use_attention,
            use_positional_encoding=config.use_positional_encoding
        )

        # Training data
        x = torch.randn(batch_size, input_dim).to(self.device)
        y = torch.randn(batch_size, 1).to(self.device)

        # Training loop
        losses = []
        adaptation_stats = []
        step_times = []

        start_time = time.time()

        for step in range(num_steps):
            step_start = time.time()

            # Forward pass
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)

            # Backward pass
            loss.backward()

            # Optimization step
            adaptation_stat = optimizer.step(loss.item())

            # Record metrics
            losses.append(loss.item())
            if adaptation_stat is not None:
                adaptation_stats.append(adaptation_stat)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Log progress
            if step % 100 == 0:
                logger.info(f"Step {step}/{num_steps}, Loss: {loss.item():.6f}")

        total_time = time.time() - start_time

        # Compute final statistics
        final_loss = losses[-1]
        loss_convergence = np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses)

        # Adaptation statistics
        if adaptation_stats:
            adaptation_stats = np.array(adaptation_stats)
            lr_multipliers = adaptation_stats[:, 0]
            momentum_factors = adaptation_stats[:, 1]
            grad_norms = adaptation_stats[:, 2]
            param_norms = adaptation_stats[:, 3]
            grad_diversities = adaptation_stats[:, 4]
            loss_curvatures = adaptation_stats[:, 5]
            update_magnitudes = adaptation_stats[:, 6]
        else:
            lr_multipliers = momentum_factors = grad_norms = param_norms = []
            grad_diversities = loss_curvatures = update_magnitudes = []

        # Compile results
        results = {
            'config': config.__dict__,
            'training_metrics': {
                'final_loss': final_loss,
                'loss_convergence': loss_convergence,
                'total_time': total_time,
                'avg_step_time': np.mean(step_times),
                'num_steps': num_steps
            },
            'adaptation_metrics': {
                'lr_multiplier_mean': float(np.mean(lr_multipliers)) if len(lr_multipliers) > 0 else 1.0,
                'lr_multiplier_std': float(np.std(lr_multipliers)) if len(lr_multipliers) > 0 else 0.0,
                'momentum_factor_mean': float(np.mean(momentum_factors)) if len(momentum_factors) > 0 else 0.0,
                'momentum_factor_std': float(np.std(momentum_factors)) if len(momentum_factors) > 0 else 0.0,
                'grad_norm_mean': float(np.mean(grad_norms)) if len(grad_norms) > 0 else 0.0,
                'param_norm_mean': float(np.mean(param_norms)) if len(param_norms) > 0 else 0.0,
                'grad_diversity_mean': float(np.mean(grad_diversities)) if len(grad_diversities) > 0 else 0.0,
                'loss_curvature_mean': float(np.mean(loss_curvatures)) if len(loss_curvatures) > 0 else 0.0,
                'update_magnitude_mean': float(np.mean(update_magnitudes)) if len(update_magnitudes) > 0 else 0.0
            },
            'loss_history': losses,
            'step_times': step_times
        }

        # Get optimizer debug summary
        debug_summary = optimizer.get_debug_summary()
        results['optimizer_debug'] = debug_summary

        logger.info(f"Experiment completed. Final loss: {final_loss:.6f}")

        return results

    def run_embedding_dim_experiments(self, num_steps: int = 1000) -> List[Dict[str, Any]]:
        """Run experiments with different embedding dimensions."""
        logger.info("Running embedding dimension experiments")

        configs = EnhancedExperimentConfigs.get_embedding_dim_experiments()
        results = []

        for i, config in enumerate(configs):
            logger.info(f"Embedding dim experiment {i+1}/{len(configs)}: dim={config.embedding_dim}")
            result = self.run_single_experiment(config, num_steps=num_steps)
            result['experiment_type'] = 'embedding_dimension'
            result['experiment_id'] = f"embedding_dim_{config.embedding_dim}"
            results.append(result)

        return results

    def run_history_length_experiments(self, num_steps: int = 1000) -> List[Dict[str, Any]]:
        """Run experiments with different history lengths."""
        logger.info("Running history length experiments")

        configs = EnhancedExperimentConfigs.get_history_length_experiments()
        results = []

        for i, config in enumerate(configs):
            logger.info(f"History length experiment {i+1}/{len(configs)}: length={config.history_length}")
            result = self.run_single_experiment(config, num_steps=num_steps)
            result['experiment_type'] = 'history_length'
            result['experiment_id'] = f"history_length_{config.history_length}"
            results.append(result)

        return results

    def run_activation_experiments(self, num_steps: int = 1000) -> List[Dict[str, Any]]:
        """Run experiments with different activation functions."""
        logger.info("Running activation function experiments")

        configs = EnhancedExperimentConfigs.get_activation_experiments()
        results = []

        for i, config in enumerate(configs):
            logger.info(f"Activation experiment {i+1}/{len(configs)}: activation={config.activation}")
            result = self.run_single_experiment(config, num_steps=num_steps)
            result['experiment_type'] = 'activation_function'
            result['experiment_id'] = f"activation_{config.activation}"
            results.append(result)

        return results

    def run_architecture_experiments(self, num_steps: int = 1000) -> List[Dict[str, Any]]:
        """Run experiments with different architectural features."""
        logger.info("Running architecture experiments")

        configs = EnhancedExperimentConfigs.get_architecture_experiments()
        results = []

        for i, config in enumerate(configs):
            logger.info(f"Architecture experiment {i+1}/{len(configs)}")
            result = self.run_single_experiment(config, num_steps=num_steps)
            result['experiment_type'] = 'architecture'
            result['experiment_id'] = f"arch_{i}"
            results.append(result)

        return results

    def run_feature_experiments(self, num_steps: int = 1000) -> List[Dict[str, Any]]:
        """Run experiments with different feature sets."""
        logger.info("Running feature experiments")

        configs = EnhancedExperimentConfigs.get_feature_experiments()
        results = []

        for i, config in enumerate(configs):
            logger.info(f"Feature experiment {i+1}/{len(configs)}: extended={config.use_extended_features}")
            result = self.run_single_experiment(config, num_steps=num_steps)
            result['experiment_type'] = 'feature_set'
            result['experiment_id'] = f"features_{i}"
            results.append(result)

        return results

    def run_comprehensive_experiments(self, num_steps: int = 1000) -> List[Dict[str, Any]]:
        """Run comprehensive set of experiments."""
        logger.info("Running comprehensive experiments")

        all_results = []

        # Run all experiment types
        all_results.extend(self.run_embedding_dim_experiments(num_steps))
        all_results.extend(self.run_history_length_experiments(num_steps))
        all_results.extend(self.run_activation_experiments(num_steps))
        all_results.extend(self.run_architecture_experiments(num_steps))
        all_results.extend(self.run_feature_experiments(num_steps))

        return all_results

    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save experiment results to file."""
        filepath = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_result[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            serializable_result[key][k] = v.tolist()
                        else:
                            serializable_result[key][k] = v
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze experiment results and generate summary."""
        analysis = {
            'total_experiments': len(results),
            'experiment_types': {},
            'best_configs': {},
            'performance_summary': {}
        }

        # Group by experiment type
        for result in results:
            exp_type = result['experiment_type']
            if exp_type not in analysis['experiment_types']:
                analysis['experiment_types'][exp_type] = []
            analysis['experiment_types'][exp_type].append(result)

        # Find best configurations for each type
        for exp_type, exp_results in analysis['experiment_types'].items():
            # Sort by final loss (lower is better)
            sorted_results = sorted(exp_results, key=lambda x: x['training_metrics']['final_loss'])
            best_result = sorted_results[0]

            analysis['best_configs'][exp_type] = {
                'config': best_result['config'],
                'final_loss': best_result['training_metrics']['final_loss'],
                'experiment_id': best_result['experiment_id']
            }

        # Overall performance summary
        all_final_losses = [r['training_metrics']['final_loss'] for r in results]
        analysis['performance_summary'] = {
            'mean_final_loss': np.mean(all_final_losses),
            'std_final_loss': np.std(all_final_losses),
            'min_final_loss': np.min(all_final_losses),
            'max_final_loss': np.max(all_final_losses),
            'best_overall_config': analysis['best_configs'][min(analysis['best_configs'].keys(),
                                                              key=lambda k: analysis['best_configs'][k]['final_loss'])]
        }

        return analysis

    def generate_report(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Generate a comprehensive experiment report."""
        report_path = self.output_dir / "experiment_report.txt"

        with open(report_path, 'w') as f:
            f.write("Enhanced Opt2Vec Experiment Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total Experiments: {analysis['total_experiments']}\n\n")

            f.write("Performance Summary:\n")
            f.write(f"  Mean Final Loss: {analysis['performance_summary']['mean_final_loss']:.6f}\n")
            f.write(f"  Std Final Loss: {analysis['performance_summary']['std_final_loss']:.6f}\n")
            f.write(f"  Min Final Loss: {analysis['performance_summary']['min_final_loss']:.6f}\n")
            f.write(f"  Max Final Loss: {analysis['performance_summary']['max_final_loss']:.6f}\n\n")

            f.write("Best Configurations by Experiment Type:\n")
            for exp_type, best_config in analysis['best_configs'].items():
                f.write(f"\n{exp_type.upper()}:\n")
                f.write(f"  Final Loss: {best_config['final_loss']:.6f}\n")
                f.write(f"  Experiment ID: {best_config['experiment_id']}\n")
                f.write(f"  Config: {best_config['config']}\n")

            f.write(f"\nBest Overall Configuration:\n")
            best_overall = analysis['performance_summary']['best_overall_config']
            f.write(f"  Type: {best_overall['experiment_id']}\n")
            f.write(f"  Final Loss: {best_overall['final_loss']:.6f}\n")
            f.write(f"  Config: {best_overall['config']}\n")

        logger.info(f"Report generated at {report_path}")


def main():
    """Main function to run comprehensive experiments."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create experiment runner
    runner = EnhancedExperimentRunner(output_dir="enhanced_experiment_results")

    # Run comprehensive experiments
    results = runner.run_comprehensive_experiments(num_steps=500)

    # Save results
    runner.save_results(results, "comprehensive_experiments.json")

    # Analyze results
    analysis = runner.analyze_results(results)

    # Generate report
    runner.generate_report(results, analysis)

    print("Experiments completed! Check the output directory for results.")


if __name__ == "__main__":
    main()
