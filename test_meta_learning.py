#!/usr/bin/env python3
"""
Test script for the updated meta-learning experiment with BEST configurations.
Now includes comparison against handcrafted optimizers (Adam and SGD).
"""

import torch
import logging
from opt2vec.experiments.meta_learning import run_meta_learning_experiment

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("TESTING UPDATED META-LEARNING EXPERIMENT")
    print("=" * 60)
    print("This will test the meta-learning experiment with the BEST configurations:")
    print("  - Embedding dim: 64 (vs old 16)")
    print("  - History length: 8 (vs old 5)")
    print("  - Activation: gelu")
    print("  - Extended features: True")
    print("  - Attention: True")
    print("  - Positional encoding: True")
    print()
    print("Will compare Opt2Vec vs Adam vs SGD on the same tasks!")
    print("=" * 60)

    # Run experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Run a shorter version for testing
        results = run_meta_learning_experiment(
            device=device,
            num_meta_steps=20,  # Shorter for testing
            num_tasks_per_step=2,
            meta_lr=2e-4,
            inner_steps=4
        )

        print("\n" + "=" * 60)
        print("META-LEARNING TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Print summary
        meta_losses = results['meta_results']['meta_losses']
        if meta_losses:
            print(f"Initial meta-loss: {meta_losses[0]:.4f}")
            print(f"Final meta-loss: {meta_losses[-1]:.4f}")
            print(f"Improvement: {meta_losses[0] - meta_losses[-1]:.4f}")

        test_results = results['test_results']
        if test_results and 'summary' in test_results:
            print(f"\nOPTIMIZER COMPARISON RESULTS:")
            for opt_name, stats in test_results['summary'].items():
                print(f"  {opt_name}:")
                print(f"    Avg improvement: {stats['avg_improvement']:.4f} ± {stats['std_improvement']:.4f}")
                print(f"    Avg final loss: {stats['avg_final_loss']:.4f} ± {stats['std_final_loss']:.4f}")
                print(f"    Avg training time: {stats['avg_training_time']:.3f}s")

        print("\nThe meta-learning experiment is working with the BEST configurations!")

    except Exception as e:
        print(f"\nError during meta-learning experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
