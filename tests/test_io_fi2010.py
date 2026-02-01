"""Tests for FI-2010 data loading."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.io import (
    list_fi2010_files,
    load_fi2010_file,
    load_fi2010_split,
    read_numeric_matrix,
)


def create_synthetic_fi2010_file(path: Path, n_features: int = 144, n_labels: int = 5, n_timesteps: int = 10):
    """Create a synthetic FI-2010-like file."""
    np.random.seed(42)
    features = np.random.randn(n_features, n_timesteps)
    labels = np.random.choice([1, 2, 3], size=(n_labels, n_timesteps))
    M = np.vstack([features, labels])
    np.savetxt(path, M, fmt="%.6f", delimiter=" ")
    return M


class TestReadNumericMatrix:
    
    def test_whitespace_delimited(self, tmp_path):
        fpath = tmp_path / "test.txt"
        M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.savetxt(fpath, M, fmt="%.2f")
        
        result = read_numeric_matrix(fpath)
        np.testing.assert_array_almost_equal(result, M)
    
    def test_comma_delimited(self, tmp_path):
        fpath = tmp_path / "test.csv"
        M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.savetxt(fpath, M, fmt="%.2f", delimiter=",")
        
        result = read_numeric_matrix(fpath)
        np.testing.assert_array_almost_equal(result, M)
    
    def test_nonexistent_file_raises(self, tmp_path):
        fpath = tmp_path / "nonexistent.txt"
        with pytest.raises(ValueError, match="does not exist"):
            read_numeric_matrix(fpath)


class TestListFi2010Files:
    
    def test_basic_listing(self, tmp_path):
        (tmp_path / "Train_CF_1.txt").write_text("1 2 3\n4 5 6")
        (tmp_path / "Train_CF_2.txt").write_text("1 2 3\n4 5 6")
        (tmp_path / "Test_CF_1.txt").write_text("1 2 3\n4 5 6")
        
        files = list_fi2010_files(str(tmp_path), "Train_CF_*.txt")
        
        assert len(files) == 2
        assert files[0].name == "Train_CF_1.txt"
        assert files[1].name == "Train_CF_2.txt"
    
    def test_no_match_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No files matched"):
            list_fi2010_files(str(tmp_path), "*.txt")
    
    def test_nonexistent_dir_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            list_fi2010_files("/nonexistent/path", "*.txt")


class TestLoadFi2010File:
    
    def test_basic_load_features_only(self, tmp_path):
        fpath = tmp_path / "test.txt"
        M = create_synthetic_fi2010_file(fpath, n_features=144, n_labels=5, n_timesteps=10)
        
        result = load_fi2010_file(fpath, return_provided_labels=False)
        
        assert "X" in result
        assert "file" in result
        assert result["X"].shape == (10, 144)
        assert result["X"].dtype == np.float32
        assert "Y_provided" not in result
    
    def test_load_with_labels_raw(self, tmp_path):
        fpath = tmp_path / "test.txt"
        M = create_synthetic_fi2010_file(fpath, n_features=144, n_labels=5, n_timesteps=10)
        
        result = load_fi2010_file(fpath, return_provided_labels=True, label_format="raw")
        
        assert "Y_provided" in result
        assert result["Y_provided"].shape == (10, 5)
        assert result["Y_provided"].dtype == np.int64
        
        unique_vals = set(np.unique(result["Y_provided"]))
        assert unique_vals.issubset({1, 2, 3})
    
    def test_load_with_labels_signed(self, tmp_path):
        fpath = tmp_path / "test.txt"
        M = create_synthetic_fi2010_file(fpath, n_features=144, n_labels=5, n_timesteps=10)
        
        result = load_fi2010_file(fpath, return_provided_labels=True, label_format="signed")
        
        assert "Y_provided" in result
        assert result["Y_provided"].shape == (10, 5)
        assert result["Y_provided"].dtype == np.int8
        
        unique_vals = set(np.unique(result["Y_provided"]))
        assert unique_vals.issubset({-1, 0, 1})
    
    def test_signed_mapping_correctness(self, tmp_path):
        fpath = tmp_path / "test.txt"
        n_timesteps = 3
        features = np.zeros((144, n_timesteps))
        labels = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ])
        M = np.vstack([features, labels])
        np.savetxt(fpath, M, fmt="%.6f")
        
        result = load_fi2010_file(fpath, return_provided_labels=True, label_format="signed")
        Y = result["Y_provided"]
        
        assert Y[0, 0] == 1
        assert Y[1, 0] == 0
        assert Y[2, 0] == -1
    
    def test_insufficient_rows_raises(self, tmp_path):
        fpath = tmp_path / "short.txt"
        M = np.random.randn(148, 10)
        np.savetxt(fpath, M, fmt="%.6f")
        
        with pytest.raises(ValueError, match="expected at least 149"):
            load_fi2010_file(fpath)
    
    def test_invalid_label_values_raises(self, tmp_path):
        fpath = tmp_path / "bad_labels.txt"
        features = np.zeros((144, 5))
        labels = np.array([[1, 2, 3, 4, 5]] * 5)
        M = np.vstack([features, labels])
        np.savetxt(fpath, M, fmt="%.6f")
        
        with pytest.raises(ValueError, match="Invalid label values"):
            load_fi2010_file(fpath, return_provided_labels=True, label_format="raw")


class TestLoadFi2010Split:
    
    def test_basic_split_load(self, tmp_path):
        for i in range(1, 3):
            create_synthetic_fi2010_file(
                tmp_path / f"Train_CF_{i}.txt",
                n_timesteps=100
            )
            create_synthetic_fi2010_file(
                tmp_path / f"Test_CF_{i}.txt",
                n_timesteps=50
            )
        
        data = load_fi2010_split(
            data_dir=str(tmp_path),
            train_glob="Train_CF_*.txt",
            test_glob="Test_CF_*.txt",
            return_provided_labels=False
        )
        
        assert "train" in data
        assert "test" in data
        
        assert data["train"]["X"].shape == (200, 144)
        assert data["test"]["X"].shape == (100, 144)
        
        assert len(data["train"]["files"]) == 2
        assert len(data["test"]["files"]) == 2
    
    def test_split_with_labels(self, tmp_path):
        for i in range(1, 3):
            create_synthetic_fi2010_file(
                tmp_path / f"Train_CF_{i}.txt",
                n_timesteps=100
            )
            create_synthetic_fi2010_file(
                tmp_path / f"Test_CF_{i}.txt",
                n_timesteps=50
            )
        
        data = load_fi2010_split(
            data_dir=str(tmp_path),
            train_glob="Train_CF_*.txt",
            test_glob="Test_CF_*.txt",
            return_provided_labels=True,
            label_format="signed"
        )
        
        assert "Y_provided" in data["train"]
        assert "Y_provided" in data["test"]
        
        assert data["train"]["Y_provided"].shape == (200, 5)
        assert data["test"]["Y_provided"].shape == (100, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
