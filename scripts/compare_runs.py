#!/usr/bin/env python
"""Compare runs and generate metrics tables."""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def parse_run_id(run_id: str) -> Dict[str, str]:
    """Parse run_id to extract components."""
    # Format: {model}_{h}{horizon}_f{fold}_w{window}_{timestamp}
    pattern = r"^(.+?)_h(\d+)_f(.+?)_w(\d+)_(\d{8}_\d{6})$"
    match = re.match(pattern, run_id)
    if match:
        return {
            "model_name": match.group(1),
            "label_horizon_index": int(match.group(2)),
            "fold_id": match.group(3),
            "window_size": int(match.group(4)),
            "timestamp": match.group(5),
        }
    return {}


def load_run_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metrics from a run directory."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    # Add run info from directory name
    run_id = run_dir.name
    parsed = parse_run_id(run_id)
    
    result = {"run_id": run_id}
    result.update(parsed)
    result.update(metrics)
    
    return result


def collect_all_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Collect metrics from all runs."""
    runs = []
    
    if not runs_dir.exists():
        return runs
    
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics = load_run_metrics(run_dir)
        if metrics:
            runs.append(metrics)
    
    return runs


def filter_runs(
    runs: List[Dict[str, Any]],
    model: Optional[str] = None,
    fold: Optional[str] = None,
    horizon: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Filter runs by criteria."""
    filtered = runs
    
    if model:
        filtered = [r for r in filtered if r.get("model_name", "").lower() == model.lower()]
    
    if fold:
        filtered = [r for r in filtered if r.get("fold_id") == fold]
    
    if horizon is not None:
        filtered = [r for r in filtered if r.get("label_horizon_index") == horizon]
    
    return filtered


def create_metrics_table(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create metrics DataFrame from runs."""
    columns = [
        "run_id", "model_name", "fold_id", "label_horizon_index", 
        "window_size", "accuracy", "macro_f1", "balanced_accuracy",
        "accuracy_no_stationary", "roc_auc_ovr"
    ]
    
    rows = []
    for run in runs:
        row = {col: run.get(col) for col in columns}
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by macro_f1 descending
    if "macro_f1" in df.columns and not df["macro_f1"].isna().all():
        df = df.sort_values("macro_f1", ascending=False, na_position="last")
    
    return df


def create_summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table with mean/std by model."""
    metric_cols = ["accuracy", "macro_f1", "balanced_accuracy", 
                   "accuracy_no_stationary", "roc_auc_ovr"]
    
    existing_cols = [c for c in metric_cols if c in df.columns]
    
    if not existing_cols or "model_name" not in df.columns:
        return pd.DataFrame()
    
    summary_rows = []
    for model in df["model_name"].unique():
        if pd.isna(model):
            continue
        model_df = df[df["model_name"] == model]
        row = {"model_name": model, "n_runs": len(model_df)}
        
        for col in existing_cols:
            values = model_df[col].dropna()
            if len(values) > 0:
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std() if len(values) > 1 else 0.0
            else:
                row[f"{col}_mean"] = None
                row[f"{col}_std"] = None
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    if "macro_f1_mean" in summary_df.columns:
        summary_df = summary_df.sort_values("macro_f1_mean", ascending=False, na_position="last")
    
    return summary_df


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table."""
    if df.empty:
        return "No data available."
    
    # Round float columns
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df_display = df.copy()
    for col in float_cols:
        df_display[col] = df_display[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
    
    return df_display.to_markdown(index=False)


def main():
    parser = argparse.ArgumentParser(description="Compare runs and generate metrics tables")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Runs directory")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    parser.add_argument("--fold", type=str, default=None, help="Filter by fold ID")
    parser.add_argument("--horizon", type=int, default=None, help="Filter by horizon index")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    runs_dir = project_root / args.runs_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Collecting runs from: {runs_dir}")
    
    all_runs = collect_all_runs(runs_dir)
    
    if not all_runs:
        print("No runs found.")
        sys.exit(0)
    
    print(f"Found {len(all_runs)} runs")
    
    # Apply filters
    filtered_runs = filter_runs(all_runs, args.model, args.fold, args.horizon)
    print(f"After filtering: {len(filtered_runs)} runs")
    
    if not filtered_runs:
        print("No runs match filters.")
        sys.exit(0)
    
    # Create metrics table
    metrics_df = create_metrics_table(filtered_runs)
    
    # Save CSV
    csv_path = output_dir / "metrics_table.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Save Markdown
    md_path = output_dir / "metrics_table.md"
    with open(md_path, "w") as f:
        f.write("# Run Comparison\n\n")
        f.write(f"Total runs: {len(filtered_runs)}\n\n")
        if args.model:
            f.write(f"Model filter: {args.model}\n")
        if args.fold:
            f.write(f"Fold filter: {args.fold}\n")
        if args.horizon is not None:
            f.write(f"Horizon filter: {args.horizon}\n")
        f.write("\n## Metrics Table\n\n")
        f.write(df_to_markdown(metrics_df))
        f.write("\n")
    print(f"Saved: {md_path}")
    
    # Create summary by model
    summary_df = create_summary_by_model(metrics_df)
    if not summary_df.empty:
        summary_csv_path = output_dir / "metrics_summary_by_model.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved: {summary_csv_path}")
        
        with open(md_path, "a") as f:
            f.write("\n## Summary by Model\n\n")
            f.write(df_to_markdown(summary_df))
            f.write("\n")
    
    # Print summary to stdout
    print("\n" + "=" * 60)
    print("METRICS TABLE (sorted by macro_f1)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    
    if not summary_df.empty:
        print("\n" + "=" * 60)
        print("SUMMARY BY MODEL")
        print("=" * 60)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
