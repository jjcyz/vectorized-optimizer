#!/usr/bin/env python3
"""
Simple example demonstrating Opt2Vec usage.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def simple_example():
    """Simple example of using Opt2Vec optimizer."""
    print("🚀 Opt2Vec Simple Example")
    print("=" * 50)

    # Import Opt2Vec components
    from opt2vec.core.optimizer import LightweightOpt2VecOptimizer

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    # Create synthetic data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    # Initialize Opt2Vec optimizer
    optimizer = LightweightOpt2VecOptimizer(
        model.parameters(),
        base_lr=0.01,
        embedding_dim=16,
        history_length=5
    )

    criterion = nn.MSELoss()
    losses = []
    embeddings = []

    print("Training with Opt2Vec...")

    # Training loop
    for step in range(50):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()

        # Opt2Vec automatically adapts learning rate and momentum
        embedding = optimizer.step(loss.item())

        losses.append(loss.item())
        if embedding is not None:
            embeddings.append(embedding)

        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Collected {len(embeddings)} embeddings")

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Opt2Vec Training Curve')
    plt.grid(True, alpha=0.3)
    plt.show()

    return losses, embeddings

def comparison_example():
    """Compare Opt2Vec with standard optimizers."""
    print("\n🔍 Opt2Vec vs Standard Optimizers")
    print("=" * 50)

    from opt2vec.core.optimizer import LightweightOpt2VecOptimizer
    import torch.optim as optim

    # Create model and data
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    x = torch.randn(50, 5)
    y = torch.randn(50, 1)
    criterion = nn.MSELoss()

    # Test different optimizers
    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=0.01),
        'Adam': optim.Adam(model.parameters(), lr=0.01),
        'Opt2Vec': LightweightOpt2VecOptimizer(
            model.parameters(), base_lr=0.01, embedding_dim=16, history_length=5
        )
    }

    results = {}

    for name, optimizer in optimizers.items():
        print(f"Testing {name}...")

        # Reset model weights
        for param in model.parameters():
            param.data.normal_()

        losses = []

        for step in range(30):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            if name == 'Opt2Vec':
                optimizer.step(loss.item())
            else:
                optimizer.step()

            losses.append(loss.item())

        results[name] = losses
        print(f"  Final loss: {losses[-1]:.4f}")

    # Plot comparison
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name, alpha=0.8)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return results

def meta_learning_example():
    """Example of meta-learning with Opt2Vec."""
    print("\n🧠 Opt2Vec Meta-Learning Example")
    print("=" * 50)

    from opt2vec.core.trainer import EfficientMetaLearningTrainer

    # Initialize meta-trainer
    trainer = EfficientMetaLearningTrainer(device=torch.device('cpu'))

    print("Running meta-learning...")

    # Run meta-training (small scale for demo)
    results = trainer.meta_train(
        num_meta_steps=20,  # Small number for demo
        num_tasks_per_step=2,
        meta_lr=1e-3,
        inner_steps=3
    )

    meta_losses = results['meta_losses']
    improvements = results['improvements']

    print(f"Initial meta-loss: {meta_losses[0]:.4f}")
    print(f"Final meta-loss: {meta_losses[-1]:.4f}")
    print(f"Average task improvement: {np.mean(improvements):.4f}")

    # Plot meta-learning progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(meta_losses)
    ax1.set_xlabel('Meta Steps')
    ax1.set_ylabel('Meta Loss')
    ax1.set_title('Meta-Learning Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(improvements)
    ax2.set_xlabel('Meta Steps')
    ax2.set_ylabel('Task Improvement')
    ax2.set_title('Task Improvement')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results

def main():
    """Run all examples."""
    print("🎯 Opt2Vec Examples")
    print("=" * 60)

    try:
        # Simple example
        losses, embeddings = simple_example()

        # Comparison example
        comparison_results = comparison_example()

        # Meta-learning example
        meta_results = meta_learning_example()

        print("\n✅ All examples completed successfully!")
        print("\nKey takeaways:")
        print("- Opt2Vec automatically adapts learning rate and momentum")
        print("- It can be used as a drop-in replacement for standard optimizers")
        print("- Meta-learning enables learning optimization strategies")
        print("- Memory-efficient design suitable for resource-constrained environments")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure you have installed all dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
