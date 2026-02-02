"""Ridge model for classification and regression."""
import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier

from .base import BaseModel

logger = logging.getLogger(__name__)


def _flatten_if_needed(X: np.ndarray) -> np.ndarray:
    """Flatten 3D input to 2D for sklearn compatibility."""
    if X.ndim == 3:
        n_samples, lookback, n_features = X.shape
        return X.reshape(n_samples, lookback * n_features)
    elif X.ndim == 2:
        return X
    else:
        raise ValueError(f"Expected 2D or 3D input, got {X.ndim}D with shape {X.shape}")


class RidgeModel(BaseModel):
    """Ridge regression/classification wrapper.
    
    For classification, uses RidgeClassifier which performs multi-class
    classification using a one-vs-rest scheme with Ridge regression.
    """
    
    def __init__(
        self,
        task: str = "classification",
        alpha: float = 1.0,
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.task = task
        self.alpha = alpha
        self.random_state = random_state
        self.kwargs = kwargs
        
        if task == "regression":
            self.model = Ridge(alpha=alpha, **kwargs)
        else:
            self.model = RidgeClassifier(
                alpha=alpha,
                random_state=random_state,
                **kwargs
            )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "RidgeModel":
        X_train = _flatten_if_needed(X_train)
        
        if self.task == "classification":
            y_train = y_train.astype(int)
        
        logger.info(f"Training RidgeModel (task={self.task}, alpha={self.alpha})")
        logger.info(f"Training data shape: {X_train.shape}")
        
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _flatten_if_needed(X)
        preds = self.model.predict(X)
        if self.task == "classification":
            preds = preds.astype(int)
        return preds
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        # RidgeClassifier does not provide proper probability estimates
        return None
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "alpha": self.alpha,
            "random_state": self.random_state,
            **self.kwargs
        }
