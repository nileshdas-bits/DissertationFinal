"""Logistic Regression classifier."""
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """Logistic Regression multi-class classifier."""
    
    def __init__(
        self,
        task: str = "classification",
        max_iter: int = 1000,
        solver: str = "saga",
        multi_class: str = "multinomial",
        class_weight: str = "balanced",
        C: float = 1.0,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        **kwargs
    ):
        assert task == "classification", "LogisticRegression only supports classification"
        
        self.task = task
        self.max_iter = max_iter
        self.solver = solver
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.C = C
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        
        self.model = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            multi_class=multi_class,
            class_weight=class_weight,
            C=C,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "LogisticRegressionModel":
        assert X_train.ndim == 2, f"Expected 2D input, got {X_train.ndim}D"
        y_train = y_train.astype(int)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2, f"Expected 2D input, got {X.ndim}D"
        return self.model.predict(X).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "max_iter": self.max_iter,
            "solver": self.solver,
            "multi_class": self.multi_class,
            "class_weight": self.class_weight,
            "C": self.C,
            "random_state": self.random_state,
            **self.kwargs
        }
