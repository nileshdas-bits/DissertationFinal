"""Train/validation/test split utilities with purge boundary."""
from typing import Dict, Tuple

import numpy as np

from ..utils.logging import get_logger


def chronological_split(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices chronologically.
    
    Args:
        n_samples: total number of samples
        train_ratio: fraction for training
        val_ratio: fraction for validation
        test_ratio: fraction for testing
        
    Returns:
        train_idx, val_idx, test_idx: index arrays
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)
    
    return train_idx, val_idx, test_idx


def purge_boundary_samples(
    indices: np.ndarray,
    time_indices: np.ndarray,
    tau: int,
    split_boundaries: Dict[str, int]
) -> np.ndarray:
    """
    Remove samples whose label horizon crosses split boundaries.
    
    A sample at time t uses label from t+tau. If t+tau crosses into
    the next split, the sample must be purged.
    
    Args:
        indices: sample indices to filter
        time_indices: original time index for each sample
        tau: prediction horizon
        split_boundaries: dict with 'train_end', 'val_end' boundaries
        
    Returns:
        purged_indices: filtered indices
    """
    mask = np.ones(len(indices), dtype=bool)
    
    train_end = split_boundaries.get("train_end")
    val_end = split_boundaries.get("val_end")
    
    for i, idx in enumerate(indices):
        t = time_indices[idx]
        future_t = t + tau
        
        if train_end is not None and t < train_end <= future_t:
            mask[i] = False
        if val_end is not None and t < val_end <= future_t:
            mask[i] = False
    
    return indices[mask]


def create_splits(
    n_samples: int,
    time_indices: np.ndarray,
    tau: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    purge: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Create train/val/test splits with optional purge boundary.
    
    Args:
        n_samples: number of samples
        time_indices: original time index for each sample
        tau: prediction horizon
        train_ratio: training set ratio
        val_ratio: validation set ratio
        test_ratio: test set ratio
        purge: whether to purge boundary samples
        
    Returns:
        train_idx, val_idx, test_idx, boundaries
    """
    logger = get_logger()
    
    train_idx, val_idx, test_idx = chronological_split(
        n_samples, train_ratio, val_ratio, test_ratio
    )
    
    boundaries = {
        "train_end": len(train_idx),
        "val_end": len(train_idx) + len(val_idx)
    }
    
    original_sizes = (len(train_idx), len(val_idx), len(test_idx))
    
    if purge:
        time_train_end = time_indices[train_idx[-1]] if len(train_idx) > 0 else 0
        time_val_end = time_indices[val_idx[-1]] if len(val_idx) > 0 else 0
        
        time_boundaries = {
            "train_end": time_train_end + 1,
            "val_end": time_val_end + 1
        }
        
        train_idx = purge_boundary_samples(
            train_idx, time_indices, tau, time_boundaries
        )
        val_idx = purge_boundary_samples(
            val_idx, time_indices, tau, time_boundaries
        )
    
    purged_sizes = (len(train_idx), len(val_idx), len(test_idx))
    
    logger.info(
        f"Split sizes - train: {original_sizes[0]} -> {purged_sizes[0]}, "
        f"val: {original_sizes[1]} -> {purged_sizes[1]}, "
        f"test: {original_sizes[2]} -> {purged_sizes[2]}"
    )
    
    return train_idx, val_idx, test_idx, boundaries


def verify_no_leakage(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    time_indices: np.ndarray,
    tau: int
) -> bool:
    """
    Verify that no sample's label horizon crosses into future splits.
    
    Args:
        train_idx, val_idx, test_idx: split indices
        time_indices: original time index for each sample
        tau: prediction horizon
        
    Returns:
        True if no leakage detected
        
    Raises:
        AssertionError if leakage detected
    """
    if len(train_idx) == 0 or len(val_idx) == 0:
        return True
    
    train_times = time_indices[train_idx]
    val_times = time_indices[val_idx]
    test_times = time_indices[test_idx] if len(test_idx) > 0 else np.array([])
    
    val_start = val_times.min()
    train_max_future = train_times.max() + tau
    
    assert train_max_future < val_start, \
        f"Train leakage: train_max_future={train_max_future} >= val_start={val_start}"
    
    if len(test_idx) > 0:
        test_start = test_times.min()
        val_max_future = val_times.max() + tau
        
        assert val_max_future < test_start, \
            f"Val leakage: val_max_future={val_max_future} >= test_start={test_start}"
    
    return True
