"""Tests for label computation."""
import numpy as np
import pytest

from src.data.labels import (
    compute_classification_labels,
    compute_regression_labels,
    get_labels,
)


def test_regression_labels_basic():
    """Test basic regression label computation."""
    mid = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    tau = 2
    
    labels, valid_mask = compute_regression_labels(mid, tau)
    
    assert labels[0] == 102.0 - 100.0  # 2.0
    assert labels[1] == 103.0 - 101.0  # 2.0
    assert labels[2] == 104.0 - 102.0  # 2.0
    
    assert valid_mask[0] == True
    assert valid_mask[1] == True
    assert valid_mask[2] == True
    assert valid_mask[3] == False
    assert valid_mask[4] == False


def test_regression_labels_valid_mask():
    """Test that valid mask correctly marks invalid samples."""
    mid = np.arange(100, dtype=float)
    tau = 10
    
    labels, valid_mask = compute_regression_labels(mid, tau)
    
    assert valid_mask[:90].all()
    assert not valid_mask[90:].any()


def test_classification_labels_positive():
    """Test classification labels for positive movements."""
    reg_labels = np.array([0.001, 0.0003, 0.0005])
    epsilon = 0.0002
    
    cls_labels = compute_classification_labels(reg_labels, epsilon)
    
    assert cls_labels[0] == 1  # 0.001 > 0.0002
    assert cls_labels[1] == 1  # 0.0003 > 0.0002
    assert cls_labels[2] == 1  # 0.0005 > 0.0002


def test_classification_labels_negative():
    """Test classification labels for negative movements."""
    reg_labels = np.array([-0.001, -0.0003, -0.0005])
    epsilon = 0.0002
    
    cls_labels = compute_classification_labels(reg_labels, epsilon)
    
    assert cls_labels[0] == -1  # -0.001 < -0.0002
    assert cls_labels[1] == -1  # -0.0003 < -0.0002
    assert cls_labels[2] == -1  # -0.0005 < -0.0002


def test_classification_labels_neutral():
    """Test classification labels for neutral movements."""
    reg_labels = np.array([0.0001, -0.0001, 0.0, 0.0002, -0.0002])
    epsilon = 0.0002
    
    cls_labels = compute_classification_labels(reg_labels, epsilon)
    
    assert cls_labels[0] == 0  # 0.0001 is within epsilon
    assert cls_labels[1] == 0  # -0.0001 is within epsilon
    assert cls_labels[2] == 0  # 0 is within epsilon
    assert cls_labels[3] == 0  # 0.0002 is exactly at epsilon (not greater)
    assert cls_labels[4] == 0  # -0.0002 is exactly at -epsilon (not less)


def test_classification_labels_boundary():
    """Test classification labels at epsilon boundaries."""
    epsilon = 0.0002
    reg_labels = np.array([
        epsilon + 1e-10,   # just above epsilon -> 1
        epsilon,           # exactly epsilon -> 0
        epsilon - 1e-10,   # just below epsilon -> 0
        -epsilon + 1e-10,  # just above -epsilon -> 0
        -epsilon,          # exactly -epsilon -> 0
        -epsilon - 1e-10,  # just below -epsilon -> -1
    ])
    
    cls_labels = compute_classification_labels(reg_labels, epsilon)
    
    assert cls_labels[0] == 1   # just above
    assert cls_labels[1] == 0   # exactly at
    assert cls_labels[2] == 0   # just below
    assert cls_labels[3] == 0   # just above -epsilon
    assert cls_labels[4] == 0   # exactly at -epsilon
    assert cls_labels[5] == -1  # just below -epsilon


def test_get_labels_regression():
    """Test get_labels for regression task."""
    mid = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    
    labels, valid_mask = get_labels(mid, tau=1, task="regression")
    
    assert labels[0] == 1.0
    assert labels[1] == 1.0
    assert labels[2] == 1.0
    assert labels[3] == 1.0


def test_get_labels_classification():
    """Test get_labels for classification task."""
    mid = np.array([100.0, 100.001, 100.0, 99.999, 100.0])
    
    labels, valid_mask = get_labels(
        mid, tau=1, task="classification", epsilon=0.0005
    )
    
    expected_reg = np.array([0.001, -0.001, -0.001, 0.001, 0.0])
    expected_cls = np.array([1, -1, -1, 1, 0])
    
    np.testing.assert_array_equal(labels[:4], expected_cls[:4])


def test_labels_shape_consistency():
    """Test that labels have consistent shape with input."""
    mid = np.random.randn(1000).cumsum() + 100
    
    for tau in [1, 5, 10, 50]:
        labels, valid_mask = compute_regression_labels(mid, tau)
        
        assert len(labels) == len(mid)
        assert len(valid_mask) == len(mid)
        assert valid_mask.sum() == len(mid) - tau


def test_labels_no_nans():
    """Test that labels don't contain NaNs in valid region."""
    mid = np.random.randn(1000).cumsum() + 100
    tau = 10
    
    labels, valid_mask = compute_regression_labels(mid, tau)
    
    assert not np.isnan(labels[valid_mask]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
