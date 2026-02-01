"""Trainer for sklearn-compatible models."""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..models.base import BaseModel
from ..utils.logging import get_logger


class SklearnTrainer:
    """Trainer for sklearn-compatible models."""
    
    def __init__(self, model: BaseModel, task: str = "regression"):
        self.model = model
        self.task = task
        self.logger = get_logger()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: training features
            y_train: training labels
            X_val: validation features (optional)
            y_val: validation labels (optional)
            
        Returns:
            training info dict
        """
        self.logger.info(f"Training {self.model.__class__.__name__}")
        self.logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape if X_val is not None else 'N/A'}")
        
        self.model.fit(X_train, y_train, X_val, y_val)
        
        train_preds = self.model.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_preds)
        
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_metrics = self._compute_metrics(y_val, val_preds)
            val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
        
        self.logger.info(f"Train metrics: {train_metrics}")
        if val_metrics:
            self.logger.info(f"Val metrics: {val_metrics}")
        
        return {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities (classification only)."""
        return self.model.predict_proba(X)
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute basic metrics."""
        if self.task == "regression":
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            return {"mae": mae, "rmse": rmse}
        else:
            accuracy = np.mean(y_true == y_pred)
            return {"accuracy": accuracy}
