"""
Memory management utilities for resource-constrained environments.
"""

import torch
import gc
import psutil
from typing import Optional


def clear_memory():
    """Clear GPU/CPU memory to prevent memory leaks."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    else:
        return psutil.Process().memory_info().rss / 1024**2  # MB


def get_gpu_memory_info() -> Optional[dict]:
    """
    Get detailed GPU memory information.

    Returns:
        Dictionary with GPU memory stats or None if no GPU
    """
    if not torch.cuda.is_available():
        return None

    return {
        'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
        'cached': torch.cuda.memory_reserved() / 1024**2,      # MB
        'total': torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
    }


def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage of a function.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with memory monitoring
    """
    def wrapper(*args, **kwargs):
        initial_memory = get_memory_usage()
        result = func(*args, **kwargs)
        final_memory = get_memory_usage()

        print(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB "
              f"(Δ: {final_memory - initial_memory:+.2f}MB)")

        return result

    return wrapper
