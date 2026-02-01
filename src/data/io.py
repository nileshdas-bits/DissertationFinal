"""File discovery and parsing utilities for FI-2010 dataset."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger


def discover_files(
    data_dir: Path,
    patterns: Optional[List[str]] = None
) -> List[Path]:
    """Discover data files in directory."""
    if patterns is None:
        patterns = ["*.txt", "*.csv"]
    
    files = []
    for pattern in patterns:
        files.extend(data_dir.glob(pattern))
    
    files = sorted([f for f in files if f.is_file() and not f.name.startswith(".")])
    
    logger = get_logger()
    logger.info(f"Discovered {len(files)} data files in {data_dir}")
    
    return files


def detect_delimiter(line: str) -> Optional[str]:
    """Detect delimiter from a line."""
    comma_count = line.count(",")
    tab_count = line.count("\t")
    
    if comma_count > 0 and comma_count >= tab_count:
        return ","
    elif tab_count > 0:
        return "\t"
    else:
        return None


def parse_file(file_path: Path) -> np.ndarray:
    """Parse a data file into numpy array."""
    logger = get_logger()
    
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
    
    delimiter = detect_delimiter(first_line)
    
    try:
        if delimiter:
            data = np.loadtxt(file_path, delimiter=delimiter)
        else:
            data = np.loadtxt(file_path)
    except ValueError:
        try:
            if delimiter:
                df = pd.read_csv(file_path, delimiter=delimiter, header=None)
            else:
                df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.dropna(axis=1, how="all")
            data = df.values
        except Exception as e:
            raise ValueError(f"Failed to parse {file_path}: {e}")
    
    assert data.ndim == 2, f"Expected 2D array, got {data.ndim}D"
    assert data.shape[0] > 0, "No rows in data"
    assert data.shape[1] > 0, "No columns in data"
    
    logger.info(f"Loaded {file_path.name}: shape={data.shape}")
    
    return data


def load_all_data(
    data_dir: Path,
    patterns: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Load and concatenate all data files."""
    files = discover_files(data_dir, patterns)
    
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    arrays = []
    file_names = []
    
    for file_path in files:
        arr = parse_file(file_path)
        arrays.append(arr)
        file_names.append(file_path.name)
    
    if len(arrays) == 1:
        combined = arrays[0]
    else:
        n_cols = arrays[0].shape[1]
        for i, arr in enumerate(arrays):
            assert arr.shape[1] == n_cols, \
                f"Column mismatch: {files[0].name} has {n_cols}, {files[i].name} has {arr.shape[1]}"
        combined = np.vstack(arrays)
    
    logger = get_logger()
    logger.info(f"Combined data shape: {combined.shape}")
    
    return combined, file_names


# =============================================================================
# FI-2010 specific loaders
# =============================================================================

def list_fi2010_files(data_dir: str, glob_pattern: str) -> List[Path]:
    """
    List FI-2010 files matching a glob pattern.
    
    Args:
        data_dir: Path to data directory
        glob_pattern: Glob pattern to match files
        
    Returns:
        Lexicographically sorted list of paths by filename
        
    Raises:
        ValueError: If data_dir does not exist or no files matched
    """
    path = Path(data_dir)
    if not path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")
    
    files = list(path.glob(glob_pattern))
    files = [f for f in files if f.is_file()]
    files = sorted(files, key=lambda p: p.name)
    
    if not files:
        raise ValueError(f"No files matched pattern '{glob_pattern}' in {data_dir}")
    
    return files


def read_numeric_matrix(path: Path) -> np.ndarray:
    """
    Read a numeric matrix from file with robust parsing.
    
    Supports whitespace-delimited and comma-delimited formats.
    Handles empty lines and multiple spaces.
    
    Args:
        path: Path to the file
        
    Returns:
        2D float64 numpy array
        
    Raises:
        ValueError: If parsing fails
    """
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")
    
    with open(path, "r") as f:
        lines = f.readlines()
    
    non_empty_lines = [ln for ln in lines if ln.strip()]
    if not non_empty_lines:
        raise ValueError(f"File is empty or contains only whitespace: {path.name}")
    
    first_line = non_empty_lines[0].strip()
    has_comma = "," in first_line
    
    try:
        if has_comma:
            data = np.loadtxt(path, delimiter=",", dtype=np.float64)
        else:
            data = np.loadtxt(path, dtype=np.float64)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
        
    except Exception as e1:
        try:
            if has_comma:
                data = np.loadtxt(path, dtype=np.float64)
            else:
                data = np.loadtxt(path, delimiter=",", dtype=np.float64)
            
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data
            
        except Exception as e2:
            try:
                if has_comma:
                    df = pd.read_csv(path, header=None, dtype=np.float64)
                else:
                    df = pd.read_csv(path, header=None, delim_whitespace=True, dtype=np.float64)
                
                df = df.dropna(axis=0, how="all")
                df = df.dropna(axis=1, how="all")
                return df.values.astype(np.float64)
                
            except Exception as e3:
                raise ValueError(
                    f"Failed to parse {path.name}: "
                    f"comma-delim error: {e1}, whitespace error: {e2}, pandas error: {e3}"
                )


def load_fi2010_file(
    path: Path,
    feature_rows: int = 144,
    label_rows: int = 5,
    return_provided_labels: bool = False,
    label_format: str = "raw"
) -> Dict:
    """
    Load a single FI-2010 file.
    
    FI-2010 files have shape (R, C) where:
    - Rows 0..143 are features (144 rows)
    - Rows 144..148 are provided classification labels (5 rows)
    - Time is along columns
    
    The loader transposes features to (T, 144) where T = number of columns.
    
    Args:
        path: Path to FI-2010 file
        feature_rows: Number of feature rows (default 144)
        label_rows: Number of label rows (default 5)
        return_provided_labels: Whether to include provided labels
        label_format: "raw" for {1,2,3} or "signed" for {+1,0,-1}
        
    Returns:
        Dict with keys:
        - "X": features array (T, feature_rows), float32
        - "file": filename string
        - "Y_provided": labels array (T, label_rows) if return_provided_labels=True
    """
    M = read_numeric_matrix(path)
    
    if M.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {M.ndim}D from {path.name}")
    
    min_rows = feature_rows + label_rows
    if M.shape[0] < min_rows:
        raise ValueError(
            f"Matrix has {M.shape[0]} rows, expected at least {min_rows} "
            f"({feature_rows} features + {label_rows} labels) in {path.name}"
        )
    
    X = M[0:feature_rows, :].T.astype(np.float32)
    
    result = {
        "X": X,
        "file": path.name
    }
    
    if return_provided_labels:
        Y = M[feature_rows:feature_rows + label_rows, :].T
        
        if label_format == "raw":
            Y_out = Y.astype(np.int64)
            unique_vals = set(np.unique(Y_out))
            valid_vals = {1, 2, 3}
            if not unique_vals.issubset(valid_vals):
                invalid = unique_vals - valid_vals
                raise ValueError(
                    f"Invalid label values {invalid} in {path.name}. "
                    f"Expected only {valid_vals}"
                )
            result["Y_provided"] = Y_out
            
        elif label_format == "signed":
            Y_int = Y.astype(np.int64)
            unique_vals = set(np.unique(Y_int))
            valid_vals = {1, 2, 3}
            if not unique_vals.issubset(valid_vals):
                invalid = unique_vals - valid_vals
                raise ValueError(
                    f"Invalid label values {invalid} in {path.name}. "
                    f"Expected only {valid_vals}"
                )
            mapping = np.array([0, 1, 0, -1], dtype=np.int8)
            Y_signed = mapping[Y_int]
            result["Y_provided"] = Y_signed
            
        else:
            raise ValueError(f"Unknown label_format: {label_format}. Use 'raw' or 'signed'")
    
    return result


def load_fi2010_split(
    data_dir: str,
    train_glob: str,
    test_glob: str,
    sort_files: bool = True,
    return_provided_labels: bool = False,
    label_format: str = "raw"
) -> Dict:
    """
    Load FI-2010 train/test split from predefined file sets.
    
    Args:
        data_dir: Path to data directory
        train_glob: Glob pattern for training files
        test_glob: Glob pattern for test files
        sort_files: Whether to sort files lexicographically (default True)
        return_provided_labels: Whether to include provided labels
        label_format: "raw" for {1,2,3} or "signed" for {+1,0,-1}
        
    Returns:
        Dict with structure:
        {
            "train": {
                "X": np.ndarray (T_train, 144),
                "Y_provided": np.ndarray (T_train, 5) if enabled,
                "files": list of filenames
            },
            "test": {
                "X": np.ndarray (T_test, 144),
                "Y_provided": np.ndarray (T_test, 5) if enabled,
                "files": list of filenames
            }
        }
    """
    train_files = list_fi2010_files(data_dir, train_glob)
    test_files = list_fi2010_files(data_dir, test_glob)
    
    def load_split(files: List[Path]) -> Dict:
        loaded = []
        for f in files:
            d = load_fi2010_file(
                f,
                return_provided_labels=return_provided_labels,
                label_format=label_format
            )
            loaded.append(d)
        
        X_list = [d["X"] for d in loaded]
        X_concat = np.concatenate(X_list, axis=0)
        
        result = {
            "X": X_concat,
            "files": [d["file"] for d in loaded]
        }
        
        if return_provided_labels:
            Y_list = [d["Y_provided"] for d in loaded]
            Y_concat = np.concatenate(Y_list, axis=0)
            result["Y_provided"] = Y_concat
        
        return result
    
    return {
        "train": load_split(train_files),
        "test": load_split(test_files)
    }
