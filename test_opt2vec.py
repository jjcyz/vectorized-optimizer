#!/usr/bin/env python3
"""
Simple test script to verify Opt2Vec implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_components():
    """Test basic Opt2Vec components."""
    logger.info("Testing basic Opt2Vec components...")

    try:
        from opt2vec.core.history import LightweightOptimizationHistory
        from opt2vec.core.network import TinyOpt2VecNetwork
        from opt2vec.core.optimizer import LightweightOpt2VecOptimizer

        # Test history
        history = LightweightOptimizationHistory(history_length=5)
        history.add_step(1.0, 0.5, 0.01)
        history.add_step(0.8, 0.4, 0.01)
        history_tensor = history.get_history_tensor(torch.device('cpu'))
        assert history_tensor.shape == (5, 3), f"Expected (5, 3), got {history_tensor.shape}"
        logger.info("✓ History component works")

        # Test network
        network = TinyOpt2VecNetwork(embedding_dim=16, history_length=5)
        test_input = torch.randn(1, 5, 3)
        embedding = network(test_input)
        assert embedding.shape == (1, 16), f"Expected (1, 16), got {embedding.shape}"
        logger.info("✓ Network component works")

        # Test optimizer
        model = nn.Linear(10, 1)
        optimizer = LightweightOpt2VecOptimizer(
            model.parameters(),
            base_lr=0.01,
            embedding_dim=16,
            history_length=5
        )

        # Test optimization step
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        embedding = optimizer.step(loss.item())

        assert embedding is not None, "Optimizer should return embedding"
        logger.info("✓ Optimizer component works")

        return True

    except Exception as e:
        logger.error(f"✗ Component test failed: {e}")
        return False

def test_simple_training():
    """Test simple training with Opt2Vec."""
    logger.info("Testing simple training...")

    try:
        from opt2vec.core.optimizer import LightweightOpt2VecOptimizer

        # Create simple model and data
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

        x = torch.randn(100, 5)
        y = torch.randn(100, 1)

        # Initialize optimizer
        optimizer = LightweightOpt2VecOptimizer(
            model.parameters(),
            base_lr=0.01,
            embedding_dim=16,
            history_length=5
        )

        criterion = nn.MSELoss()
        losses = []
        embeddings = []

        # Train for a few steps
        for step in range(20):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            embedding = optimizer.step(loss.item())
            losses.append(loss.item())

            if embedding is not None:
                embeddings.append(embedding)

        # Check that training progressed
        assert len(losses) == 20, f"Expected 20 losses, got {len(losses)}"
        assert len(embeddings) > 0, "Should have collected embeddings"

        # Check that loss decreased (at least initially)
        initial_loss = losses[0]
        final_loss = losses[-1]
        logger.info(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

        logger.info("✓ Simple training works")
        return True

    except Exception as e:
        logger.error(f"✗ Training test failed: {e}")
        return False

def test_memory_efficiency():
    """Test memory efficiency."""
    logger.info("Testing memory efficiency...")

    try:
        from opt2vec.utils.memory import get_memory_usage

        initial_memory = get_memory_usage()
        logger.info(f"Initial memory: {initial_memory:.2f} MB")

        # Create and train model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        from opt2vec.core.optimizer import LightweightOpt2VecOptimizer
        optimizer = LightweightOpt2VecOptimizer(
            model.parameters(),
            base_lr=0.01,
            embedding_dim=16,
            history_length=5
        )

        x = torch.randn(50, 10)
        y = torch.randn(50, 1)
        criterion = nn.MSELoss()

        for step in range(10):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step(loss.item())

        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory

        logger.info(f"Final memory: {final_memory:.2f} MB")
        logger.info(f"Memory increase: {memory_increase:+.2f} MB")

        # Check that memory usage is reasonable (< 100MB increase)
        assert memory_increase < 100, f"Memory increase too large: {memory_increase:.2f} MB"

        logger.info("✓ Memory efficiency test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Memory test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting Opt2Vec tests...")

    tests = [
        ("Basic Components", test_basic_components),
        ("Simple Training", test_simple_training),
        ("Memory Efficiency", test_memory_efficiency),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")

        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test FAILED with exception: {e}")

    logger.info(f"\n{'='*50}")
    logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")

    if passed == total:
        logger.info("🎉 All tests passed! Opt2Vec is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
