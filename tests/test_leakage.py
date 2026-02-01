"""Tests for data leakage prevention."""
import numpy as np
import pytest

from src.data.splits import (
    chronological_split,
    create_splits,
    purge_boundary_samples,
    verify_no_leakage,
)


def test_chronological_split_ratios():
    """Test that chronological split respects ratios."""
    n_samples = 1000
    train_idx, val_idx, test_idx = chronological_split(
        n_samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    assert len(train_idx) == 700
    assert len(val_idx) == 150
    assert len(test_idx) == 150
    
    assert train_idx[-1] < val_idx[0]
    assert val_idx[-1] < test_idx[0]


def test_chronological_split_no_overlap():
    """Test that splits have no overlap."""
    n_samples = 1000
    train_idx, val_idx, test_idx = chronological_split(
        n_samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0


def test_purge_boundary_removes_leaking_samples():
    """Test that boundary purge removes samples that would leak."""
    indices = np.arange(100)
    time_indices = np.arange(100)
    tau = 10
    
    split_boundaries = {"train_end": 70, "val_end": 85}
    
    train_indices = np.arange(0, 70)
    purged = purge_boundary_samples(
        train_indices, time_indices, tau, split_boundaries
    )
    
    for idx in purged:
        t = time_indices[idx]
        assert t + tau < 70, f"Sample at t={t} with tau={tau} crosses boundary"


def test_create_splits_with_purge():
    """Test full split creation with purge."""
    n_samples = 1000
    time_indices = np.arange(n_samples)
    tau = 10
    
    train_idx, val_idx, test_idx, _ = create_splits(
        n_samples=n_samples,
        time_indices=time_indices,
        tau=tau,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        purge=True
    )
    
    assert len(train_idx) < 700
    assert len(val_idx) < 150
    
    if len(train_idx) > 0 and len(val_idx) > 0:
        train_max_time = time_indices[train_idx].max()
        val_min_time = time_indices[val_idx].min()
        assert train_max_time + tau < val_min_time


def test_verify_no_leakage_passes_valid_splits():
    """Test that verification passes for valid splits."""
    n_samples = 1000
    time_indices = np.arange(n_samples)
    tau = 10
    
    train_idx = np.arange(0, 680)
    val_idx = np.arange(700, 840)
    test_idx = np.arange(860, 1000)
    
    result = verify_no_leakage(train_idx, val_idx, test_idx, time_indices, tau)
    assert result is True


def test_verify_no_leakage_fails_invalid_splits():
    """Test that verification fails for invalid splits."""
    n_samples = 1000
    time_indices = np.arange(n_samples)
    tau = 10
    
    train_idx = np.arange(0, 700)
    val_idx = np.arange(700, 850)
    test_idx = np.arange(850, 1000)
    
    with pytest.raises(AssertionError):
        verify_no_leakage(train_idx, val_idx, test_idx, time_indices, tau)


def test_no_future_information_in_train():
    """Test that training samples don't use future data."""
    n_samples = 500
    time_indices = np.arange(n_samples)
    tau = 20
    
    train_idx, val_idx, test_idx, _ = create_splits(
        n_samples=n_samples,
        time_indices=time_indices,
        tau=tau,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        purge=True
    )
    
    if len(train_idx) > 0 and len(val_idx) > 0:
        train_times = time_indices[train_idx]
        val_start = time_indices[val_idx].min()
        
        for t in train_times:
            label_time = t + tau
            assert label_time < val_start, \
                f"Train sample at t={t} uses label from t+tau={label_time} >= val_start={val_start}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
