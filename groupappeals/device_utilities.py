"""
Device utility functions for GroupAppeals package.

This module provides shared utilities for determining and configuring
the optimal compute device (CUDA, MPS, or CPU) across all GroupAppeals modules.
"""

import torch


def determine_compute_device():
    """
    Determine the optimal device for model inference.
    
    Returns:
        str: The optimal device ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def convert_device_to_pipeline_id(device):
    """
    Convert device string to device ID for transformers pipeline.
    
    Args:
        device (str): Device string ('cuda', 'mps', or 'cpu')
        
    Returns:
        int or str: Device ID for pipeline (0 for cuda, 'mps' for mps, -1 for cpu)
    """
    if device == "cuda":
        return 0
    elif device == "mps":
        return "mps"
    else:
        return -1