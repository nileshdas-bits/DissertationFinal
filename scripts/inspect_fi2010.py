#!/usr/bin/env python
"""Inspect FI-2010 dataset files and print summary statistics."""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="Inspect FI-2010 dataset")
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
        data_dir = str(project_root / data_dir)
    
    print(f"Data directory: {data_dir}")
    print(f"Train glob: {train_glob}")
    print(f"Test glob: {test_glob}")
    print(f"Return provided labels: {return_provided_labels}")
    print(f"Label format: {label_format}")
    print()
    
    sys.path.insert(0, str(project_root))
    from src.data.io import load_fi2010_split
    
    try:
        data = load_fi2010_split(
            data_dir=data_dir,
            train_glob=train_glob,
            test_glob=test_glob,
            return_provided_labels=return_provided_labels,
            label_format=label_format
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    train_data = data["train"]
    test_data = data["test"]
    
    print("=" * 60)
    print("TRAIN SPLIT")
    print("=" * 60)
    print(f"Number of files: {len(train_data['files'])}")
    print(f"First 2 files: {train_data['files'][:2]}")
    print(f"X shape: {train_data['X'].shape}")
    print(f"X dtype: {train_data['X'].dtype}")
    print(f"X min: {train_data['X'].min():.6f}")
    print(f"X max: {train_data['X'].max():.6f}")
    print(f"X mean: {train_data['X'].mean():.6f}")
    
    if return_provided_labels and "Y_provided" in train_data:
        Y = train_data["Y_provided"]
        print(f"Y_provided shape: {Y.shape}")
        print(f"Y_provided dtype: {Y.dtype}")
        print("Unique values per label column:")
        for col in range(Y.shape[1]):
            unique = np.unique(Y[:, col])
            print(f"  Column {col}: {sorted(unique.tolist())}")
    
    print()
    print("=" * 60)
    print("TEST SPLIT")
    print("=" * 60)
    print(f"Number of files: {len(test_data['files'])}")
    print(f"First 2 files: {test_data['files'][:2]}")
    print(f"X shape: {test_data['X'].shape}")
    print(f"X dtype: {test_data['X'].dtype}")
    print(f"X min: {test_data['X'].min():.6f}")
    print(f"X max: {test_data['X'].max():.6f}")
    print(f"X mean: {test_data['X'].mean():.6f}")
    
    if return_provided_labels and "Y_provided" in test_data:
        Y = test_data["Y_provided"]
        print(f"Y_provided shape: {Y.shape}")
        print(f"Y_provided dtype: {Y.dtype}")
        print("Unique values per label column:")
        for col in range(Y.shape[1]):
            unique = np.unique(Y[:, col])
            print(f"  Column {col}: {sorted(unique.tolist())}")


if __name__ == "__main__":
    main()
