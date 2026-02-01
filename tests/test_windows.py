"""Tests for window construction."""
import numpy as np
import pytest

from src.data.windows import (
    build_windows,
    compute_window_stats,
    flatten_windows,
    transform_windows,
)


def test_build_windows_shape():
    """Test that windows have correct shape."""
    T, F = 100, 10
    features = np.random.randn(T, F)
    lookback = 20
    
    windows, indices = build_windows(features, lookback)
    
    expected_n = T - lookback + 1
    assert windows.shape == (expected_n, lookback, F)
    assert len(indices) == expected_n


def test_build_windows_indices():
    """Test that window indices are correct."""
    T, F = 50, 5
    features = np.random.randn(T, F)
    lookback = 10
    
    windows, indices = build_windows(features, lookback)
    
    assert indices[0] == lookback - 1
    assert indices[-1] == T - 1
    np.testing.assert_array_equal(indices, np.arange(lookback - 1, T))


def test_build_windows_content():
    """Test that window content is correct."""
    T, F = 20, 3
    features = np.arange(T * F, dtype=float).reshape(T, F)
    lookback = 5
    
    windows, indices = build_windows(features, lookback)
    
    expected_window_0 = features[0:5]
    np.testing.assert_array_equal(windows[0], expected_window_0)
    
    expected_window_5 = features[5:10]
    np.testing.assert_array_equal(windows[5], expected_window_5)


def test_flatten_windows_shape():
    """Test that flattened windows have correct shape."""
    N, W, F = 100, 20, 10
    windows = np.random.randn(N, W, F)
    
    flat = flatten_windows(windows)
    
    assert flat.shape == (N, W * F)


def test_flatten_windows_content():
    """Test that flattened content is correct."""
    windows = np.array([
        [[1, 2], [3, 4], [5, 6]],
        [[7, 8], [9, 10], [11, 12]]
    ])
    
    flat = flatten_windows(windows)
    
    expected = np.array([
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12]
    ])
    np.testing.assert_array_equal(flat, expected)


def test_compute_window_stats_shape():
    """Test that window stats have correct shape."""
    N, W, F = 100, 20, 10
    windows = np.random.randn(N, W, F)
    
    stats = compute_window_stats(windows)
    
    assert stats.shape == (N, 4 * F)


def test_compute_window_stats_content():
    """Test that window stats are computed correctly."""
    windows = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    ])
    
    stats = compute_window_stats(windows)
    
    expected_mean = np.array([3.0, 4.0])
    expected_std = np.array([np.std([1, 3, 5]), np.std([2, 4, 6])])
    expected_last = np.array([5.0, 6.0])
    expected_delta = np.array([4.0, 4.0])
    
    np.testing.assert_array_almost_equal(stats[0, :2], expected_mean)
    np.testing.assert_array_almost_equal(stats[0, 2:4], expected_std)
    np.testing.assert_array_almost_equal(stats[0, 4:6], expected_last)
    np.testing.assert_array_almost_equal(stats[0, 6:8], expected_delta)


def test_transform_windows_sequence():
    """Test sequence mode returns unchanged windows."""
    windows = np.random.randn(100, 20, 10)
    
    result = transform_windows(windows, "sequence")
    
    np.testing.assert_array_equal(result, windows)


def test_transform_windows_tabular_flat():
    """Test tabular_flat mode flattens windows."""
    N, W, F = 100, 20, 10
    windows = np.random.randn(N, W, F)
    
    result = transform_windows(windows, "tabular_flat")
    
    assert result.shape == (N, W * F)


def test_transform_windows_tabular_stats():
    """Test tabular_stats mode computes statistics."""
    N, W, F = 100, 20, 10
    windows = np.random.randn(N, W, F)
    
    result = transform_windows(windows, "tabular_stats")
    
    assert result.shape == (N, 4 * F)


def test_transform_windows_invalid_mode():
    """Test that invalid mode raises error."""
    windows = np.random.randn(100, 20, 10)
    
    with pytest.raises(ValueError):
        transform_windows(windows, "invalid_mode")


def test_sequence_windowing_label_alignment():
    """
    Test that sequence windowing aligns labels correctly.
    
    For each t where t >= W-1:
        X_seq[t] = X_stream[t-W+1 : t+1] -> shape (W, F)
        y[t] = label_stream[t]
    
    The index returned by build_windows corresponds to the LAST timestep in each window,
    which is the correct timestep for the associated label.
    """
    T, F = 100, 144
    W = 10
    
    X_stream = np.arange(T * F, dtype=float).reshape(T, F)
    y_stream = np.arange(T)  # y[t] = t
    
    windows, indices = build_windows(X_stream, W)
    
    # Number of valid samples
    N = T - W + 1
    assert windows.shape == (N, W, F)
    assert len(indices) == N
    
    # Check first window: t=9 (W-1), X_seq[0] = X_stream[0:10]
    assert indices[0] == W - 1  # First valid t is W-1 = 9
    np.testing.assert_array_equal(windows[0], X_stream[0:W])
    
    # The label for windows[0] should be y_stream[indices[0]] = y_stream[9] = 9
    y_aligned = y_stream[indices]
    assert y_aligned[0] == W - 1
    
    # Check last window: t=99, X_seq[-1] = X_stream[90:100]
    assert indices[-1] == T - 1
    np.testing.assert_array_equal(windows[-1], X_stream[T-W:T])
    assert y_aligned[-1] == T - 1
    
    # Check arbitrary window: t=50, X_seq should be X_stream[41:51]
    sample_idx = 50 - (W - 1)  # Index in windowed array
    assert indices[sample_idx] == 50
    np.testing.assert_array_equal(windows[sample_idx], X_stream[50-W+1:51])
    assert y_aligned[sample_idx] == 50


def test_sequence_windowing_time_order_preserved():
    """Test that time order is preserved after windowing."""
    T, F = 1000, 10
    W = 50
    
    X_stream = np.random.randn(T, F)
    
    windows, indices = build_windows(X_stream, W)
    
    # Indices should be strictly increasing (time order)
    assert np.all(np.diff(indices) == 1)
    
    # First index should be W-1, last should be T-1
    assert indices[0] == W - 1
    assert indices[-1] == T - 1


def test_windowing_no_cross_boundary():
    """
    Test that windowing applied separately to splits prevents boundary crossing.
    
    When train and test are windowed separately, no window from test
    can include data from train.
    """
    # Simulate separate train/test splits
    T_train, T_test, F = 500, 200, 10
    W = 50
    
    X_train = np.random.randn(T_train, F)
    X_test = np.random.randn(T_test, F)
    
    windows_train, idx_train = build_windows(X_train, W)
    windows_test, idx_test = build_windows(X_test, W)
    
    # Train windows only contain train data (indices 0 to T_train-1)
    assert idx_train[0] == W - 1
    assert idx_train[-1] == T_train - 1
    
    # Test windows only contain test data (indices 0 to T_test-1, relative to test)
    assert idx_test[0] == W - 1
    assert idx_test[-1] == T_test - 1
    
    # No overlap: windows_train uses X_train, windows_test uses X_test
    # This is guaranteed by applying build_windows to separate arrays


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
