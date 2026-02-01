"""Label computation utilities."""
from typing import Tuple

import numpy as np


def compute_regression_labels(
    mid: np.ndarray,
    tau: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute regression labels: y[t] = mid[t+tau] - mid[t].
    
    Args:
        mid: mid-price array
        tau: prediction horizon
        
    Returns:
        labels: regression labels
        valid_mask: boolean mask for valid samples
    """
    assert mid.ndim == 1, f"Expected 1D array, got {mid.ndim}D"
    assert tau > 0, f"tau must be positive, got {tau}"
    
    n = len(mid)
    labels = np.zeros(n)
    valid_mask = np.zeros(n, dtype=bool)
    
    valid_end = n - tau
    if valid_end > 0:
        labels[:valid_end] = mid[tau:] - mid[:valid_end]
        valid_mask[:valid_end] = True
    
    return labels, valid_mask


def compute_classification_labels(
    reg_labels: np.ndarray,
    epsilon: float
) -> np.ndarray:
    """
    Convert regression labels to classification labels.
    
    y_cls = 1 if y_reg > eps
    y_cls = -1 if y_reg < -eps
    y_cls = 0 otherwise
    
    Args:
        reg_labels: regression labels
        epsilon: threshold for classification
        
    Returns:
        cls_labels: classification labels (-1, 0, 1)
    """
    assert reg_labels.ndim == 1, f"Expected 1D array, got {reg_labels.ndim}D"
    assert epsilon >= 0, f"epsilon must be non-negative, got {epsilon}"
    
    cls_labels = np.zeros_like(reg_labels, dtype=np.int64)
    cls_labels[reg_labels > epsilon] = 1
    cls_labels[reg_labels < -epsilon] = -1
    
    return cls_labels


def get_labels(
    mid: np.ndarray,
    tau: int,
    task: str,
    epsilon: float = 0.0002
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get labels for specified task.
    
    Args:
        mid: mid-price array
        tau: prediction horizon
        task: 'regression' or 'classification'
        epsilon: threshold for classification
        
    Returns:
        labels: labels array
        valid_mask: boolean mask for valid samples
    """
    reg_labels, valid_mask = compute_regression_labels(mid, tau)
    
    if task == "regression":
        return reg_labels, valid_mask
    elif task == "classification":
        cls_labels = compute_classification_labels(reg_labels, epsilon)
        return cls_labels, valid_mask
    else:
        raise ValueError(f"Unknown task: {task}")
