"""Ridge regression model."""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier

from .base import BaseModel


class RidgeModel(BaseModel):
    """Ridge regression/classification wrapper."""
    
    def __init__(
        self,
        task: str = "regression",
        alpha: float = 1.0,
        **kwargs
    ):
        self.task = task
        self.alpha = alpha
        self.kwargs = kwargs
        
        if task == "regression":
            self.model = Ridge(alpha=alpha, **kwargs)
        else:
            self.model = RidgeClassifier(alpha=alpha, **kwargs)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "RidgeModel":
        assert X_train.ndim == 2, f"Expected 2D input, got {X_train.ndim}D"
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D"
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.task == "classification":
            if hasattr(self.model, "decision_function"):
                return self.model.decision_function(X)
        return None
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "alpha": self.alpha,
            **self.kwargs
        }
