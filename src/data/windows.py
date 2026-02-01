"""Window construction utilities."""
from typing import Tuple

import numpy as np


def build_windows(
    features: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows from features.
    
    Args:
        features: (T, F) feature array
        lookback: window size W
        
    Returns:
        windows: (N, W, F) array where N = T - W + 1
        indices: (N,) array of original time indices (last index in each window)
    """
    assert features.ndim == 2, f"Expected 2D array, got {features.ndim}D"
    T, F = features.shape
    assert lookback > 0, f"lookback must be positive, got {lookback}"
    assert lookback <= T, f"lookback ({lookback}) > T ({T})"
    
    N = T - lookback + 1
    windows = np.zeros((N, lookback, F), dtype=features.dtype)
    indices = np.arange(lookback - 1, T)
    
    for i in range(N):
        windows[i] = features[i:i + lookback]
    
    assert windows.shape == (N, lookback, F), f"Unexpected shape: {windows.shape}"
    assert len(indices) == N, f"Index length mismatch: {len(indices)} != {N}"
    
    return windows, indices


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    """
    Flatten windows for tabular models.
    
    Args:
        windows: (N, W, F) array
        
    Returns:
        flat: (N, W*F) array
    """
    assert windows.ndim == 3, f"Expected 3D array, got {windows.ndim}D"
    N, W, F = windows.shape
    return windows.reshape(N, W * F)


def compute_window_stats(windows: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics for each window.
    
    Computes per-feature: mean, std, last, delta (last - first)
    
    Args:
        windows: (N, W, F) array
        
    Returns:
        stats: (N, 4*F) array
    """
    assert windows.ndim == 3, f"Expected 3D array, got {windows.ndim}D"
    N, W, F = windows.shape
    
    mean = np.mean(windows, axis=1)  # (N, F)
    std = np.std(windows, axis=1)    # (N, F)
    last = windows[:, -1, :]         # (N, F)
    delta = windows[:, -1, :] - windows[:, 0, :]  # (N, F)
    
    stats = np.concatenate([mean, std, last, delta], axis=1)  # (N, 4*F)
    
    assert stats.shape == (N, 4 * F), f"Unexpected shape: {stats.shape}"
    
    return stats


def transform_windows(
    windows: np.ndarray,
    mode: str
) -> np.ndarray:
    """
    Transform windows based on mode.
    
    Args:
        windows: (N, W, F) array
        mode: 'sequence', 'tabular_flat', or 'tabular_stats'
        
    Returns:
        transformed features
    """
    if mode == "sequence":
        return windows
    elif mode == "tabular_flat":
        return flatten_windows(windows)
    elif mode == "tabular_stats":
        return compute_window_stats(windows)
    else:
        raise ValueError(f"Unknown mode: {mode}")
