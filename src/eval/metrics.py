"""Evaluation metrics with uniform output format."""
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


CLASS_NAMES = ["up", "stationary", "down"]  # indices 0, 1, 2
STATIONARY_CLASS = 1  # class index for stationary


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    mask = sign_true != 0
    if mask.sum() > 0:
        directional_accuracy = float(np.mean(sign_true[mask] == sign_pred[mask]))
    else:
        directional_accuracy = None
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "directional_accuracy": directional_accuracy,
        "accuracy": None,
        "macro_f1": None,
        "balanced_accuracy": None,
        "accuracy_no_stationary": None,
        "roc_auc_ovr": None,
        "precision_up": None,
        "recall_up": None,
        "f1_up": None,
        "precision_stationary": None,
        "recall_stationary": None,
        "f1_stationary": None,
        "precision_down": None,
        "recall_down": None,
        "f1_down": None,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    stationary_class: int = STATIONARY_CLASS
) -> Dict[str, Any]:
    """
    Compute uniform classification metrics.
    
    Args:
        y_true: true class indices {0, 1, 2}
        y_pred: predicted class indices {0, 1, 2}
        y_proba: predicted probabilities shape (N, 3), optional
        stationary_class: class index for stationary (default 1)
        
    Returns:
        Dict with all metric keys (None if not computable)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    mask_non_stationary = y_true != stationary_class
    if mask_non_stationary.sum() > 0:
        accuracy_no_stat = float(accuracy_score(
            y_true[mask_non_stationary],
            y_pred[mask_non_stationary]
        ))
    else:
        accuracy_no_stat = None
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    roc_auc = None
    if y_proba is not None:
        try:
            unique_classes = np.unique(y_true)
            if len(unique_classes) >= 2 and y_proba.shape[1] == 3:
                roc_auc = float(roc_auc_score(
                    y_true, y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=[0, 1, 2]
                ))
        except (ValueError, IndexError):
            roc_auc = None
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "accuracy_no_stationary": accuracy_no_stat,
        "roc_auc_ovr": roc_auc,
        "precision_up": float(precision[0]),
        "recall_up": float(recall[0]),
        "f1_up": float(f1[0]),
        "precision_stationary": float(precision[1]),
        "recall_stationary": float(recall[1]),
        "f1_stationary": float(f1[1]),
        "precision_down": float(precision[2]),
        "recall_down": float(recall[2]),
        "f1_down": float(f1[2]),
        "support_up": int(support[0]),
        "support_stationary": int(support[1]),
        "support_down": int(support[2]),
        "mae": None,
        "rmse": None,
        "r2": None,
        "directional_accuracy": None,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int] = [0, 1, 2]
) -> np.ndarray:
    """Compute confusion matrix with fixed label order."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    """Compute sklearn classification report as dict."""
    return classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute metrics based on task.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        task: 'regression' or 'classification'
        y_proba: predicted probabilities (classification only)
        
    Returns:
        Dict with uniform metric keys
    """
    if task == "regression":
        return compute_regression_metrics(y_true, y_pred)
    else:
        return compute_classification_metrics(y_true, y_pred, y_proba)


def convert_labels_to_class_indices(
    labels: np.ndarray,
    source_format: str = "signed"
) -> np.ndarray:
    """
    Convert labels to class indices {0, 1, 2}.
    
    Mappings:
    - "signed": {-1, 0, 1} -> {2, 1, 0} (down=-1->2, stationary=0->1, up=1->0)
    - "raw": {1, 2, 3} -> {0, 1, 2} (up=1->0, stationary=2->1, down=3->2)
    
    Args:
        labels: input labels
        source_format: "signed" or "raw"
        
    Returns:
        class indices {0, 1, 2}
    """
    labels = np.asarray(labels)
    
    if source_format == "signed":
        # -1 (down) -> 2, 0 (stationary) -> 1, 1 (up) -> 0
        mapping = {-1: 2, 0: 1, 1: 0}
        result = np.zeros_like(labels, dtype=np.int64)
        for src, dst in mapping.items():
            result[labels == src] = dst
        return result
    
    elif source_format == "raw":
        # 1 (up) -> 0, 2 (stationary) -> 1, 3 (down) -> 2
        return (labels - 1).astype(np.int64)
    
    else:
        raise ValueError(f"Unknown source_format: {source_format}")


def convert_class_indices_to_signed(indices: np.ndarray) -> np.ndarray:
    """
    Convert class indices {0, 1, 2} back to signed {1, 0, -1}.
    
    0 (up) -> 1, 1 (stationary) -> 0, 2 (down) -> -1
    """
    mapping = np.array([1, 0, -1], dtype=np.int64)
    return mapping[indices.astype(int)]
