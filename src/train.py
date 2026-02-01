"""Main training entrypoint with uniform run logging."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml

from .config import Config
from .data.dataset_torch import create_dataloader
from .data.io import load_fi2010_split
from .data.scalers import create_scaler
from .data.windows import build_windows, transform_windows
from .eval.metrics import (
    compute_classification_report,
    compute_confusion_matrix,
    compute_metrics,
    convert_labels_to_class_indices,
)
from .eval.plots import create_plots
from .utils.logging import get_logger, setup_logging
from .utils.paths import get_project_root
from .utils.seed import set_seed


def generate_run_id(config: Config) -> str:
    """Generate structured run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model.name
    horizon = config.label.horizon_index
    fold = config.data.fold_id.replace("*", "all")
    window = config.window.lookback
    return f"{model_name}_h{horizon}_f{fold}_w{window}_{timestamp}"


def get_runs_dir() -> Path:
    """Get runs directory."""
    runs_dir = get_project_root() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def load_fi2010_data(config: Config) -> Dict[str, Any]:
    """Load FI-2010 data with provided labels."""
    logger = get_logger()
    
    raw_dir = get_project_root() / config.data.raw_dir
    
    data = load_fi2010_split(
        data_dir=str(raw_dir),
        train_glob=config.data.train_glob,
        test_glob=config.data.test_glob,
        return_provided_labels=True,
        label_format="raw"
    )
    
    X_train_raw = data["train"]["X"]
    X_test_raw = data["test"]["X"]
    
    horizon_idx = config.label.horizon_index
    y_train_raw = data["train"]["Y_provided"][:, horizon_idx]
    y_test_raw = data["test"]["Y_provided"][:, horizon_idx]
    
    y_train = convert_labels_to_class_indices(y_train_raw, source_format="raw")
    y_test = convert_labels_to_class_indices(y_test_raw, source_format="raw")
    
    logger.info(f"Loaded FI-2010: X_train={X_train_raw.shape}, X_test={X_test_raw.shape}")
    logger.info(f"Labels horizon_index={horizon_idx}: train classes={np.unique(y_train)}, test classes={np.unique(y_test)}")
    
    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "train_files": data["train"]["files"],
        "test_files": data["test"]["files"],
    }


def preprocess_features(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply windowing and normalization."""
    logger = get_logger()
    
    # Build windows
    X_train_win, train_indices = build_windows(X_train_raw, config.window.lookback)
    X_test_win, test_indices = build_windows(X_test_raw, config.window.lookback)
    
    # Normalization
    if config.normalization.method != "none":
        scaler = create_scaler(config.normalization.method)
        scaler.fit(X_train_win)
        X_train_win = scaler.transform(X_train_win)
        X_test_win = scaler.transform(X_test_win)
    
    # Transform based on mode
    mode = config.window.mode
    if mode == "flat":
        mode = "tabular_flat"
    X_train = transform_windows(X_train_win, mode)
    X_test = transform_windows(X_test_win, mode)
    
    logger.info(f"Preprocessed: X_train={X_train.shape}, X_test={X_test.shape}")
    
    return X_train, X_test, train_indices, test_indices


def create_model(config: Config, input_shape: Tuple[int, ...]):
    """Create model based on config."""
    model_name = config.model.name.lower()
    params = config.model.params.copy()
    task = config.label.task
    
    if model_name == "ridge":
        from .models.ridge import RidgeModel
        return RidgeModel(task=task, **params)
    
    elif model_name == "random_forest":
        from .models.random_forest import RandomForestModel
        return RandomForestModel(task=task, random_state=config.seed, **params)
    
    elif model_name == "xgboost":
        from .models.xgboost_model import XGBoostModel
        return XGBoostModel(task=task, random_state=config.seed, **params)
    
    elif model_name == "lstm":
        from .models.lstm import create_lstm
        input_size = input_shape[-1]
        return create_lstm(input_size=input_size, task=task, **params)
    
    elif model_name == "gru":
        from .models.gru import create_gru
        input_size = input_shape[-1]
        return create_gru(input_size=input_size, task=task, **params)
    
    elif model_name == "tcn":
        from .models.tcn import create_tcn
        input_size = input_shape[-1]
        return create_tcn(input_size=input_size, task=task, **params)
    
    elif model_name == "autoencoder":
        from .models.autoencoder import create_autoencoder
        input_size = input_shape[-1]
        return create_autoencoder(input_size=input_size, task=task, **params)
    
    elif model_name == "logistic_regression":
        from .models.logistic_regression import LogisticRegressionModel
        return LogisticRegressionModel(task=task, random_state=config.seed, **params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def is_torch_model(model_name: str) -> bool:
    """Check if model requires PyTorch training."""
    return model_name.lower() in ["lstm", "gru", "tcn", "autoencoder"]


def train_sklearn_model(model, X_train, y_train, X_val, y_val, config):
    """Train sklearn-compatible model."""
    from .training.trainer_sklearn import SklearnTrainer
    trainer = SklearnTrainer(model, task=config.label.task)
    history = trainer.train(X_train, y_train, X_val, y_val)
    return {"trainer": trainer, "history": history}


def train_torch_model(model, X_train, y_train, X_val, y_val, config):
    """Train PyTorch model."""
    from .training.trainer_torch import TorchTrainer
    
    train_loader = create_dataloader(X_train, y_train, config.training.batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, config.training.batch_size, shuffle=False)
    
    trainer = TorchTrainer(
        model,
        task=config.label.task,
        learning_rate=config.training.learning_rate,
        device=config.training.device,
        early_stopping_patience=config.training.early_stopping_patience
    )
    
    # Check for two-stage autoencoder training
    is_autoencoder = config.model.name.lower() == "autoencoder"
    two_stage = getattr(config.training, 'two_stage', False)
    
    if is_autoencoder and two_stage:
        history = trainer.train_two_stage(
            train_loader, val_loader,
            pretrain_epochs=config.training.pretrain_epochs,
            finetune_epochs=config.training.finetune_epochs,
            freeze_encoder=config.training.freeze_encoder
        )
    else:
        history = trainer.train(train_loader, val_loader, epochs=config.training.epochs)
    
    return {"trainer": trainer, "history": history}


def evaluate_model(trainer, X_test, y_test, config, is_torch: bool):
    """Evaluate model on test set. Returns predictions as class indices {0,1,2}."""
    if is_torch:
        test_loader = create_dataloader(X_test, y_test, config.training.batch_size, shuffle=False)
        y_pred = trainer.predict(test_loader)
        y_proba = trainer.predict_proba(test_loader)
    else:
        y_pred = trainer.predict(X_test)
        y_proba = trainer.predict_proba(X_test)
    
    y_pred = np.asarray(y_pred).astype(int)
    metrics = compute_metrics(y_test, y_pred, config.label.task, y_proba)
    return y_pred, y_proba, metrics


def save_run_artifacts(
    run_dir: Path,
    config: Config,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    test_indices: np.ndarray,
    metrics: Dict[str, Any],
    train_losses: Optional[list] = None,
    val_losses: Optional[list] = None
) -> Dict[str, Path]:
    """Save all run artifacts to run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    
    # config_resolved.yaml
    config_path = run_dir / "config_resolved.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    artifacts["config"] = config_path
    
    # metrics.json
    metrics_clean = {}
    for k, v in metrics.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            metrics_clean[k] = None
        else:
            metrics_clean[k] = v
    
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_clean, f, indent=2)
    artifacts["metrics"] = metrics_path
    
    # predictions_test.csv
    pred_data = {
        "t_index": test_indices,
        "y_true": y_test,
        "y_pred": y_pred,
    }
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 3:
        pred_data["p0"] = y_proba[:, 0]
        pred_data["p1"] = y_proba[:, 1]
        pred_data["p2"] = y_proba[:, 2]
    
    pred_df = pd.DataFrame(pred_data)
    pred_path = run_dir / "predictions_test.csv"
    pred_df.to_csv(pred_path, index=False)
    artifacts["predictions"] = pred_path
    
    # confusion_matrix_test.csv
    if config.label.task == "classification":
        cm = compute_confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["true_up", "true_stationary", "true_down"],
            columns=["pred_up", "pred_stationary", "pred_down"]
        )
        cm_path = run_dir / "confusion_matrix_test.csv"
        cm_df.to_csv(cm_path)
        artifacts["confusion_matrix"] = cm_path
        
        # classification_report_test.json
        report = compute_classification_report(y_test, y_pred)
        report_path = run_dir / "classification_report_test.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        artifacts["classification_report"] = report_path
    
    # Plots
    plots = create_plots(
        y_test, y_pred,
        task=config.label.task,
        output_dir=run_dir,
        train_losses=train_losses,
        val_losses=val_losses
    )
    artifacts.update(plots)
    
    return artifacts


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="FI-2010 Classification")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = get_project_root() / config_path
    
    config = Config.from_yaml(config_path)
    config.validate()
    
    setup_logging()
    logger = get_logger()
    
    run_id = generate_run_id(config)
    run_dir = get_runs_dir() / run_id
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")
    
    set_seed(config.seed)
    
    # Load data
    if config.label.source == "provided":
        data = load_fi2010_data(config)
        X_train_raw = data["X_train_raw"]
        X_test_raw = data["X_test_raw"]
        y_train_all = data["y_train"]
        y_test_all = data["y_test"]
    else:
        raise ValueError("Only 'provided' label source is supported for FI-2010")
    
    # Preprocess
    X_train, X_test, train_indices, test_indices = preprocess_features(
        X_train_raw, X_test_raw, config
    )
    
    # Align labels with windowed samples
    y_train = y_train_all[train_indices]
    y_test = y_test_all[test_indices]
    
    # Split train into train/val (time-ordered)
    n_train = len(X_train)
    val_split = int(n_train * 0.85)
    X_train_split, X_val = X_train[:val_split], X_train[val_split:]
    y_train_split, y_val = y_train[:val_split], y_train[val_split:]
    
    logger.info(f"Train/Val split: {len(X_train_split)}/{len(X_val)}, Test: {len(X_test)}")
    
    # Create and train model
    model = create_model(config, X_train_split.shape)
    is_torch = is_torch_model(config.model.name)
    
    if is_torch:
        result = train_torch_model(model, X_train_split, y_train_split, X_val, y_val, config)
        train_losses = result["history"].get("train_losses", [])
        val_losses = result["history"].get("val_losses", [])
    else:
        result = train_sklearn_model(model, X_train_split, y_train_split, X_val, y_val, config)
        train_losses = None
        val_losses = None
    
    trainer = result["trainer"]
    
    # Evaluate
    y_pred, y_proba, metrics = evaluate_model(trainer, X_test, y_test, config, is_torch)
    
    logger.info(f"Test metrics: accuracy={metrics.get('accuracy'):.4f}, macro_f1={metrics.get('macro_f1'):.4f}")
    
    # Save artifacts
    artifacts = save_run_artifacts(
        run_dir, config,
        y_test, y_pred, y_proba,
        test_indices, metrics,
        train_losses, val_losses
    )
    
    # Save feature importances for tree-based models
    if hasattr(trainer.model, 'get_feature_importances'):
        importances = trainer.model.get_feature_importances()
        n_features = len(importances)
        feat_imp_df = pd.DataFrame({
            'feature_idx': range(n_features),
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save top 20
        top20_path = run_dir / "feature_importance_top20.csv"
        feat_imp_df.head(20).to_csv(top20_path, index=False)
        artifacts["feature_importance"] = top20_path
        logger.info(f"Saved feature importances to {top20_path}")
    
    # Save XGBoost evals result and best iteration
    if hasattr(trainer.model, 'get_evals_result'):
        evals_result = trainer.model.get_evals_result()
        if evals_result:
            evals_path = run_dir / "evals_result.json"
            # Convert numpy arrays to lists for JSON serialization
            evals_serializable = {}
            for key, val in evals_result.items():
                evals_serializable[key] = {}
                for metric_name, metric_vals in val.items():
                    evals_serializable[key][metric_name] = [float(v) for v in metric_vals]
            with open(evals_path, "w") as f:
                json.dump(evals_serializable, f, indent=2)
            artifacts["evals_result"] = evals_path
            logger.info(f"Saved evals result to {evals_path}")
    
    if hasattr(trainer.model, 'get_best_iteration'):
        best_iter = trainer.model.get_best_iteration()
        if best_iter is not None:
            metrics["best_iteration"] = best_iter
            # Update metrics.json with best_iteration
            metrics_path = run_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Best iteration: {best_iter}")
    
    # MLflow logging
    mlflow.set_tracking_uri(f"file://{get_project_root() / 'mlruns'}")
    mlflow.set_experiment(config.experiment_name)
    
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({
            "model_name": config.model.name,
            "task": config.label.task,
            "label_source": config.label.source,
            "label_horizon_index": config.label.horizon_index,
            "fold_id": config.data.fold_id,
            "feature_mode": config.window.mode,
            "window_size": config.window.lookback,
            "seed": config.seed,
            "normalization": config.normalization.method,
        })
        
        # Log metrics (only numeric)
        for k, v in metrics.items():
            if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                mlflow.log_metric(k, v)
        
        # Log run directory as artifact
        mlflow.log_artifacts(str(run_dir))
    
    logger.info(f"Results saved to {run_dir}")
    logger.info("Training complete")


if __name__ == "__main__":
    main()
