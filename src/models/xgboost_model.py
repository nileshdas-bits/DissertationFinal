"""XGBoost model."""
import inspect
import logging
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

from .base import BaseModel

logger = logging.getLogger(__name__)


def _supports_fit_param(estimator, param_name: str) -> bool:
    """Check if estimator.fit() accepts a given parameter."""
    try:
        sig = inspect.signature(estimator.fit)
        return param_name in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        return False


def _get_early_stopping_callback(rounds: int):
    """Get early stopping callback if available."""
    try:
        from xgboost.callback import EarlyStopping
        return EarlyStopping(rounds=rounds, save_best=True)
    except ImportError:
        return None


class XGBoostModel(BaseModel):
    """XGBoost regression/classification wrapper."""
    
    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tree_method: str = "hist",
        random_state: Optional[int] = None,
        early_stopping_rounds: Optional[int] = 10,
        n_jobs: int = -1,
        **kwargs
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.evals_result_: Dict[str, Any] = {}
        self._early_stopping_enabled = False
        
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": tree_method,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }
        
        # Remove None values and add kwargs
        params = {k: v for k, v in params.items() if v is not None}
        # Remove early_stopping_rounds from constructor params if present
        params.pop("early_stopping_rounds", None)
        params.update({k: v for k, v in kwargs.items() if k != "early_stopping_rounds"})
        
        if task == "regression":
            self.model = xgb.XGBRegressor(**params)
        else:
            # Classification-specific params
            params["objective"] = "multi:softprob"
            params["num_class"] = 3
            params["eval_metric"] = "mlogloss"
            self.model = xgb.XGBClassifier(**params)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "XGBoostModel":
        assert X_train.ndim == 2, f"Expected 2D input, got {X_train.ndim}D"
        
        # Ensure labels are int for classification
        if self.task == "classification":
            y_train = y_train.astype(int)
            if y_val is not None:
                y_val = y_val.astype(int)
        
        fit_params = {}
        self._early_stopping_enabled = False
        
        if X_val is not None and y_val is not None:
            if _supports_fit_param(self.model, "eval_set"):
                fit_params["eval_set"] = [(X_train, y_train), (X_val, y_val)]
            
            if _supports_fit_param(self.model, "verbose"):
                fit_params["verbose"] = False
            
            if self.early_stopping_rounds:
                self._early_stopping_enabled = self._configure_early_stopping(fit_params)
        
        self.model.fit(X_train, y_train, **fit_params)
        
        # Store evals result if available
        if hasattr(self.model, 'evals_result_'):
            self.evals_result_ = self.model.evals_result()
        
        return self
    
    def _configure_early_stopping(self, fit_params: Dict[str, Any]) -> bool:
        """Configure early stopping in a version-safe manner.
        
        Returns True if early stopping was successfully configured.
        """
        rounds = self.early_stopping_rounds
        
        # Try callback-based early stopping first (preferred for newer versions)
        callback = _get_early_stopping_callback(rounds)
        if callback is not None and _supports_fit_param(self.model, "callbacks"):
            fit_params["callbacks"] = [callback]
            logger.info(f"XGBoost early stopping enabled: rounds={rounds} (callback)")
            return True
        
        # Try direct early_stopping_rounds parameter (older versions)
        if _supports_fit_param(self.model, "early_stopping_rounds"):
            fit_params["early_stopping_rounds"] = rounds
            logger.info(f"XGBoost early stopping enabled: rounds={rounds}")
            return True
        
        logger.warning(
            "Early stopping not supported by installed xgboost; "
            "proceeding without early stopping"
        )
        return False
    
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
    
    def get_evals_result(self) -> Dict[str, Any]:
        """Get evaluation results per iteration."""
        return self.evals_result_
    
    def get_best_iteration(self) -> Optional[int]:
        """Get best iteration from early stopping."""
        if hasattr(self.model, 'best_iteration'):
            return self.model.best_iteration
        return None
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "early_stopping_rounds": self.early_stopping_rounds,
            **self.kwargs
        }
