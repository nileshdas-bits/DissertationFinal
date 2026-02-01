"""Normalization/standardization utilities."""
from typing import Optional, Tuple

import numpy as np


class Scaler:
    """Base scaler class."""
    
    def fit(self, X: np.ndarray) -> "Scaler":
        raise NotImplementedError
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZScoreScaler(Scaler):
    """Z-score (standardization) scaler."""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        if X.ndim == 3:
            N, W, F = X.shape
            X_flat = X.reshape(-1, F)
        else:
            X_flat = X
        
        self.mean_ = np.mean(X_flat, axis=0)
        self.std_ = np.std(X_flat, axis=0)
        self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Scaler not fitted"
        
        original_shape = X.shape
        
        if X.ndim == 3:
            N, W, F = X.shape
            X_flat = X.reshape(-1, F)
            X_scaled = (X_flat - self.mean_) / self.std_
            return X_scaled.reshape(original_shape)
        else:
            return (X - self.mean_) / self.std_
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Scaler not fitted"
        
        original_shape = X.shape
        
        if X.ndim == 3:
            N, W, F = X.shape
            X_flat = X.reshape(-1, F)
            X_orig = X_flat * self.std_ + self.mean_
            return X_orig.reshape(original_shape)
        else:
            return X * self.std_ + self.mean_


class MinMaxScaler(Scaler):
    """Min-max scaler."""
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1), epsilon: float = 1e-8):
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        if X.ndim == 3:
            N, W, F = X.shape
            X_flat = X.reshape(-1, F)
        else:
            X_flat = X
        
        self.min_ = np.min(X_flat, axis=0)
        self.max_ = np.max(X_flat, axis=0)
        
        data_range = self.max_ - self.min_
        data_range = np.where(data_range < self.epsilon, 1.0, data_range)
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.min_ is not None, "Scaler not fitted"
        
        original_shape = X.shape
        feature_min, _ = self.feature_range
        
        if X.ndim == 3:
            N, W, F = X.shape
            X_flat = X.reshape(-1, F)
            X_scaled = (X_flat - self.min_) * self.scale_ + feature_min
            return X_scaled.reshape(original_shape)
        else:
            return (X - self.min_) * self.scale_ + feature_min
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        assert self.min_ is not None, "Scaler not fitted"
        
        original_shape = X.shape
        feature_min, _ = self.feature_range
        
        if X.ndim == 3:
            N, W, F = X.shape
            X_flat = X.reshape(-1, F)
            X_orig = (X_flat - feature_min) / self.scale_ + self.min_
            return X_orig.reshape(original_shape)
        else:
            return (X - feature_min) / self.scale_ + self.min_


class IdentityScaler(Scaler):
    """Identity scaler (no transformation)."""
    
    def fit(self, X: np.ndarray) -> "IdentityScaler":
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X


def create_scaler(method: str) -> Scaler:
    """Create scaler by method name."""
    if method == "zscore":
        return ZScoreScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "none":
        return IdentityScaler()
    else:
        raise ValueError(f"Unknown scaler method: {method}")
