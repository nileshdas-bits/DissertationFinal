"""Feature extraction utilities."""
from typing import Tuple

import numpy as np


def extract_lob_features(
    data: np.ndarray,
    bid_price_col: int = 0,
    ask_price_col: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract LOB features from raw data.
    
    Standard FI-2010 format:
    - Columns 0,1: bid_price_1, bid_vol_1
    - Columns 2,3: ask_price_1, ask_vol_1
    - And so on for deeper levels
    
    Returns:
        features: raw features array
        best_bid: best bid prices
        best_ask: best ask prices
    """
    assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"
    assert data.shape[1] >= 4, f"Need at least 4 columns for LOB, got {data.shape[1]}"
    
    best_bid = data[:, bid_price_col].copy()
    best_ask = data[:, ask_price_col].copy()
    
    assert np.all(np.isfinite(best_bid)), "Non-finite values in best_bid"
    assert np.all(np.isfinite(best_ask)), "Non-finite values in best_ask"
    
    return data, best_bid, best_ask


def compute_mid_price(best_bid: np.ndarray, best_ask: np.ndarray) -> np.ndarray:
    """Compute mid-price from bid and ask."""
    assert best_bid.shape == best_ask.shape, "Shape mismatch"
    
    mid = (best_bid + best_ask) / 2.0
    
    assert np.all(np.isfinite(mid)), "Non-finite values in mid-price"
    
    return mid


def compute_spread(best_bid: np.ndarray, best_ask: np.ndarray) -> np.ndarray:
    """Compute bid-ask spread."""
    return best_ask - best_bid


def compute_returns(mid: np.ndarray, periods: int = 1) -> np.ndarray:
    """Compute mid-price returns."""
    returns = np.zeros_like(mid)
    returns[periods:] = (mid[periods:] - mid[:-periods]) / mid[:-periods]
    return returns
