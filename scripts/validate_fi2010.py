#!/usr/bin/env python
"""Validate FI-2010 dataset files and print a copy-pastable report."""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


def fmt_float(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def fmt_list(lst: List, max_items: int = 10) -> str:
    if len(lst) <= max_items:
        return str(lst)
    return str(lst[:max_items]) + f"... ({len(lst)} total)"


def validate_file(
    path: Path,
    read_numeric_matrix,
    feature_rows: int = 144,
    label_rows: int = 5,
    return_labels: bool = False,
    label_format: str = "raw"
) -> Dict[str, Any]:
    """Validate a single FI-2010 file."""
    result = {
        "file": path.name,
        "ok": True,
        "error": None,
        "raw_shape": None,
        "X_shape": None,
        "X_nan": 0,
        "X_inf": 0,
        "X_min": None,
        "X_max": None,
        "X_mean": None,
        "X_std": None,
        "labels_enabled": return_labels,
        "Y_shape": None,
        "Y_unique_cols": None,
    }
    
    try:
        M = read_numeric_matrix(path)
    except Exception as e:
        result["ok"] = False
        result["error"] = f"parse_error: {e}"
        return result
    
    result["raw_shape"] = M.shape
    
    min_rows = feature_rows + label_rows
    if M.shape[0] < min_rows:
        result["ok"] = False
        result["error"] = f"insufficient_rows: {M.shape[0]} < {min_rows}"
        return result
    
    X = M[0:feature_rows, :].T.astype(np.float32)
    result["X_shape"] = X.shape
    
    if X.shape[1] != feature_rows:
        result["ok"] = False
        result["error"] = f"feature_dim_mismatch: got {X.shape[1]}, expected {feature_rows}"
        return result
    
    result["X_nan"] = int(np.isnan(X).sum())
    result["X_inf"] = int(np.isinf(X).sum())
    result["X_min"] = float(np.nanmin(X))
    result["X_max"] = float(np.nanmax(X))
    result["X_mean"] = float(np.nanmean(X))
    result["X_std"] = float(np.nanstd(X))
    
    if return_labels:
        Y_raw = M[feature_rows:feature_rows + label_rows, :].T
        result["Y_shape"] = Y_raw.shape
        
        unique_per_col = []
        for col in range(Y_raw.shape[1]):
            unique_vals = sorted(set(np.unique(Y_raw[:, col]).astype(int).tolist()))
            unique_per_col.append(unique_vals)
        result["Y_unique_cols"] = unique_per_col
        
        all_unique = set()
        for u in unique_per_col:
            all_unique.update(u)
        
        valid_raw = {1, 2, 3}
        if not all_unique.issubset(valid_raw):
            invalid = all_unique - valid_raw
            result["ok"] = False
            result["error"] = f"invalid_label_values: {invalid}"
            return result
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate FI-2010 dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_cfg = config.get("dataset", {})
    data_dir = dataset_cfg.get("data_dir", "data/raw/fi2010")
    train_glob = dataset_cfg.get("train_glob", "Train_Dst_NoAuction_ZScore_CF_*.txt")
    test_glob = dataset_cfg.get("test_glob", "Test_Dst_NoAuction_ZScore_CF_*.txt")
    return_provided_labels = dataset_cfg.get("return_provided_labels", False)
    label_format = dataset_cfg.get("label_format", "raw")
    
    project_root = Path(__file__).resolve().parent.parent
    if not Path(data_dir).is_absolute():
        data_dir_resolved = str(project_root / data_dir)
    else:
        data_dir_resolved = data_dir
    
    sys.path.insert(0, str(project_root))
    from src.data.io import list_fi2010_files, load_fi2010_split, read_numeric_matrix
    
    fail_reasons = []
    warnings = []
    
    # List files
    try:
        train_files = list_fi2010_files(data_dir_resolved, train_glob)
    except ValueError as e:
        train_files = []
        fail_reasons.append(f"train_files: {e}")
    
    try:
        test_files = list_fi2010_files(data_dir_resolved, test_glob)
    except ValueError as e:
        test_files = []
        fail_reasons.append(f"test_files: {e}")
    
    train_count = len(train_files)
    test_count = len(test_files)
    
    if train_count == 0:
        fail_reasons.append("no_train_files")
    if test_count == 0:
        fail_reasons.append("no_test_files")
    
    train_names = [f.name for f in train_files]
    test_names = [f.name for f in test_files]
    
    train_first3 = train_names[:3]
    train_last3 = train_names[-3:] if len(train_names) >= 3 else train_names
    test_first3 = test_names[:3]
    test_last3 = test_names[-3:] if len(test_names) >= 3 else test_names
    
    # Per-file validation
    train_results = []
    test_results = []
    
    for f in train_files:
        r = validate_file(
            f, read_numeric_matrix,
            return_labels=return_provided_labels,
            label_format=label_format
        )
        train_results.append(r)
        if not r["ok"]:
            fail_reasons.append(f"train_file_error: {r['file']}: {r['error']}")
    
    for f in test_files:
        r = validate_file(
            f, read_numeric_matrix,
            return_labels=return_provided_labels,
            label_format=label_format
        )
        test_results.append(r)
        if not r["ok"]:
            fail_reasons.append(f"test_file_error: {r['file']}: {r['error']}")
    
    # Concatenated validation
    concat_train = {"shape": None, "dtype": None, "nan": 0, "inf": 0, "min": None, "max": None, "mean": None, "std": None}
    concat_test = {"shape": None, "dtype": None, "nan": 0, "inf": 0, "min": None, "max": None, "mean": None, "std": None}
    Y_train_unique_cols = None
    Y_test_unique_cols = None
    
    if train_count > 0 and test_count > 0 and not fail_reasons:
        try:
            data = load_fi2010_split(
                data_dir=data_dir_resolved,
                train_glob=train_glob,
                test_glob=test_glob,
                return_provided_labels=return_provided_labels,
                label_format=label_format
            )
            
            X_train = data["train"]["X"]
            X_test = data["test"]["X"]
            
            concat_train["shape"] = X_train.shape
            concat_train["dtype"] = str(X_train.dtype)
            concat_train["nan"] = int(np.isnan(X_train).sum())
            concat_train["inf"] = int(np.isinf(X_train).sum())
            concat_train["min"] = float(np.nanmin(X_train))
            concat_train["max"] = float(np.nanmax(X_train))
            concat_train["mean"] = float(np.nanmean(X_train))
            concat_train["std"] = float(np.nanstd(X_train))
            
            concat_test["shape"] = X_test.shape
            concat_test["dtype"] = str(X_test.dtype)
            concat_test["nan"] = int(np.isnan(X_test).sum())
            concat_test["inf"] = int(np.isinf(X_test).sum())
            concat_test["min"] = float(np.nanmin(X_test))
            concat_test["max"] = float(np.nanmax(X_test))
            concat_test["mean"] = float(np.nanmean(X_test))
            concat_test["std"] = float(np.nanstd(X_test))
            
            if return_provided_labels:
                Y_train = data["train"]["Y_provided"]
                Y_test = data["test"]["Y_provided"]
                
                Y_train_unique_cols = []
                for col in range(Y_train.shape[1]):
                    Y_train_unique_cols.append(sorted(set(np.unique(Y_train[:, col]).tolist())))
                
                Y_test_unique_cols = []
                for col in range(Y_test.shape[1]):
                    Y_test_unique_cols.append(sorted(set(np.unique(Y_test[:, col]).tolist())))
            
            # Orientation sanity
            if X_train.shape[1] != 144:
                fail_reasons.append(f"orientation_error: X_train.shape[1]={X_train.shape[1]}, expected 144")
            if X_test.shape[1] != 144:
                fail_reasons.append(f"orientation_error: X_test.shape[1]={X_test.shape[1]}, expected 144")
            
        except Exception as e:
            fail_reasons.append(f"concat_error: {e}")
    
    # Heuristics
    zscore_train_mean = concat_train["mean"]
    zscore_train_std = concat_train["std"]
    
    if zscore_train_mean is not None:
        if abs(zscore_train_mean) > 0.5:
            warnings.append(f"zscore_mean_warning: |mean|={abs(zscore_train_mean):.4f} > 0.5")
    if zscore_train_std is not None:
        if zscore_train_std < 0.5 or zscore_train_std > 2.0:
            warnings.append(f"zscore_std_warning: std={zscore_train_std:.4f} outside [0.5, 2.0]")
    
    # Print report
    status = "PASS" if not fail_reasons else "FAIL"
    
    print("=== FI-2010 VALIDATION REPORT ===")
    print("Config:")
    print(f"  data_dir: {data_dir}")
    print(f"  train_glob: {train_glob}")
    print(f"  test_glob: {test_glob}")
    print(f"  return_provided_labels: {return_provided_labels}")
    print(f"  label_format: {label_format}")
    
    print("Files:")
    print(f"  train_count: {train_count}")
    print(f"  test_count: {test_count}")
    print(f"  train_first3: {train_first3}")
    print(f"  train_last3: {train_last3}")
    print(f"  test_first3: {test_first3}")
    print(f"  test_last3: {test_last3}")
    
    print("Per-file summary (first 3 train + first 3 test only):")
    for r in train_results[:3]:
        print(f"  {r['file']}")
        print(f"    raw_shape: {r['raw_shape']}")
        print(f"    X_shape: {r['X_shape']}")
        print(f"    X_nan: {r['X_nan']}")
        print(f"    X_inf: {r['X_inf']}")
        if r['X_min'] is not None:
            print(f"    X_min/max/mean/std: {fmt_float(r['X_min'])}/{fmt_float(r['X_max'])}/{fmt_float(r['X_mean'])}/{fmt_float(r['X_std'])}")
        print(f"    labels_enabled: {'yes' if r['labels_enabled'] else 'no'}")
        if r['labels_enabled'] and r['Y_shape']:
            print(f"    Y_shape: {r['Y_shape']}")
            print(f"    Y_unique_cols: {r['Y_unique_cols']}")
        if r['error']:
            print(f"    error: {r['error']}")
    
    for r in test_results[:3]:
        print(f"  {r['file']}")
        print(f"    raw_shape: {r['raw_shape']}")
        print(f"    X_shape: {r['X_shape']}")
        print(f"    X_nan: {r['X_nan']}")
        print(f"    X_inf: {r['X_inf']}")
        if r['X_min'] is not None:
            print(f"    X_min/max/mean/std: {fmt_float(r['X_min'])}/{fmt_float(r['X_max'])}/{fmt_float(r['X_mean'])}/{fmt_float(r['X_std'])}")
        print(f"    labels_enabled: {'yes' if r['labels_enabled'] else 'no'}")
        if r['labels_enabled'] and r['Y_shape']:
            print(f"    Y_shape: {r['Y_shape']}")
            print(f"    Y_unique_cols: {r['Y_unique_cols']}")
        if r['error']:
            print(f"    error: {r['error']}")
    
    print("Concatenated:")
    if concat_train["shape"]:
        print(f"  X_train: shape={concat_train['shape']}, dtype={concat_train['dtype']}, nan={concat_train['nan']}, inf={concat_train['inf']}, min/max/mean/std={fmt_float(concat_train['min'])}/{fmt_float(concat_train['max'])}/{fmt_float(concat_train['mean'])}/{fmt_float(concat_train['std'])}")
    else:
        print("  X_train: N/A")
    
    if concat_test["shape"]:
        print(f"  X_test: shape={concat_test['shape']}, dtype={concat_test['dtype']}, nan={concat_test['nan']}, inf={concat_test['inf']}, min/max/mean/std={fmt_float(concat_test['min'])}/{fmt_float(concat_test['max'])}/{fmt_float(concat_test['mean'])}/{fmt_float(concat_test['std'])}")
    else:
        print("  X_test: N/A")
    
    print(f"  labels_enabled: {'yes' if return_provided_labels else 'no'}")
    if return_provided_labels and Y_train_unique_cols:
        print(f"  Y_train_unique_cols: {Y_train_unique_cols}")
    if return_provided_labels and Y_test_unique_cols:
        print(f"  Y_test_unique_cols: {Y_test_unique_cols}")
    
    print("Heuristics:")
    print(f"  zscore_train_mean: {fmt_float(zscore_train_mean) if zscore_train_mean is not None else 'N/A'}")
    print(f"  zscore_train_std: {fmt_float(zscore_train_std) if zscore_train_std is not None else 'N/A'}")
    print(f"  warnings: {warnings if warnings else '[]'}")
    
    print("Status:")
    print(f"  {status}")
    if fail_reasons:
        print(f"  fail_reasons: {fail_reasons}")
    
    print("=== END REPORT ===")
    
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
