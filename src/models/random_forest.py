"""Random Forest model."""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest regression/classification wrapper."""
    
    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[str] = None,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        **kwargs
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **kwargs
        }
        
        if task == "regression":
            self.model = RandomForestRegressor(**params)
        else:
            params["class_weight"] = class_weight
            self.model = RandomForestClassifier(**params)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "RandomForestModel":
        assert X_train.ndim == 2, f"Expected 2D input, got {X_train.ndim}D"
        if self.task == "classification":
            y_train = y_train.astype(int)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D"
        preds = self.model.predict(X)
        if self.task == "classification":
            preds = preds.astype(int)
        return preds
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.task == "classification":
            return self.model.predict_proba(X)
        return None
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from fitted model."""
        return self.model.feature_importances_
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            **self.kwargs
        }
