"""PyTorch Dataset and DataLoader utilities."""
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: Optional[np.ndarray] = None
    ):
        """
        Args:
            X: features array (N, W, F) for sequence or (N, D) for flat
            y: labels array (N,)
            indices: original time indices (N,)
        """
        assert len(X) == len(y), f"X/y length mismatch: {len(X)} != {len(y)}"
        
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    indices: Optional[np.ndarray] = None,
    num_workers: int = 0
) -> DataLoader:
    """Create DataLoader from numpy arrays."""
    dataset = TimeSeriesDataset(X, y, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
