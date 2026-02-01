"""Plotting utilities."""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Predictions vs Actual",
    max_points: int = 1000
) -> None:
    """
    Scatter plot of predictions vs actual values.
    
    Args:
        y_true: true values
        y_pred: predicted values
        output_path: path to save figure
        title: plot title
        max_points: max points to plot (for performance)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=10)
    
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Residuals Distribution"
) -> None:
    """
    Histogram of residuals.
    
    Args:
        y_true: true values
        y_pred: predicted values
        output_path: path to save figure
        title: plot title
    """
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Residual Distribution")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    output_path: Path,
    title: str = "Training Curves"
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: training losses per epoch
        val_losses: validation losses per epoch (optional)
        output_path: path to save figure
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label="Train Loss")
    
    if val_losses:
        ax.plot(epochs, val_losses, "r-", label="Val Loss")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    labels: List[str] = ["Up", "Stationary", "Down"],
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: true labels (class indices {0,1,2})
        y_pred: predicted labels (class indices {0,1,2})
        output_path: path to save figure
        labels: class labels (0=Up, 1=Stationary, 2=Down)
        title: plot title
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm_norm, cmap="Blues")
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{cm[i, j]}\n({cm_norm[i, j]:.2%})"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    output_dir: Path,
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None
) -> Dict[str, Path]:
    """
    Create all relevant plots.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        task: 'regression' or 'classification'
        output_dir: directory to save plots
        train_losses: training losses (optional)
        val_losses: validation losses (optional)
        
    Returns:
        dict of plot name -> path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = {}
    
    if task == "regression":
        pred_path = output_dir / "predictions_vs_actual.png"
        plot_predictions_vs_actual(y_true, y_pred, pred_path)
        plots["predictions_vs_actual"] = pred_path
        
        resid_path = output_dir / "residuals.png"
        plot_residuals(y_true, y_pred, resid_path)
        plots["residuals"] = resid_path
    else:
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, cm_path)
        plots["confusion_matrix"] = cm_path
    
    if train_losses:
        curves_path = output_dir / "training_curves.png"
        plot_training_curves(train_losses, val_losses, curves_path)
        plots["training_curves"] = curves_path
    
    return plots
