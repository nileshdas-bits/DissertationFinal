# FI-2010 Short-Horizon Price Sensitivity Prediction

A dissertation-grade pipeline for predicting short-horizon price movements from limit order book data using multiple model architectures.

## Features

- Multiple model types: Ridge, Random Forest, XGBoost, LSTM, GRU, TCN, Autoencoder
- Leakage-safe chronological splits with boundary purging
- Both regression and 3-class classification tasks
- Train-only normalization
- MLflow experiment tracking
- Deterministic runs with fixed seeds
- Compatible with macOS and Google Colab

## Project Structure

```
├── configs/                    # YAML configuration files
│   ├── ridge_reg.yaml
│   ├── xgboost_reg.yaml
│   ├── xgboost_cls.yaml
│   ├── lstm_reg.yaml
│   ├── gru_reg.yaml
│   ├── tcn_reg.yaml
│   └── autoencoder_reg.yaml
├── data/
│   ├── raw/fi2010/            # Place dataset files here
│   └── processed/             # Generated processed data
├── notebooks/
│   └── colab_run.ipynb        # Colab notebook
├── src/
│   ├── config.py              # Configuration management
│   ├── train.py               # Main training entrypoint
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model implementations
│   ├── training/              # Training utilities
│   ├── eval/                  # Evaluation metrics and plots
│   └── utils/                 # Utilities (seed, paths, logging)
├── tests/                     # Test suite
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.10+
- numpy, pandas, scikit-learn, xgboost, torch, pyyaml, mlflow, matplotlib, tqdm

## Dataset Placement

1. Obtain the FI-2010 dataset files
2. Place all data files in `data/raw/fi2010/`
3. Files can be `.txt` or `.csv` format (whitespace or comma delimited)

### FI-2010 File Format

FI-2010 files have a specific orientation:
- **Shape**: (149 rows, T columns) where T is the number of time steps
- **Rows 0-143**: 144 LOB features
- **Rows 144-148**: 5 provided classification labels (for different horizons)
- **Time is along columns**, not rows

The loader transposes the feature matrix to produce shape `(T, 144)` for downstream processing.

### File Naming Convention

Standard FI-2010 files follow this pattern:
- Train files: `Train_Dst_NoAuction_ZScore_CF_*.txt` (9 cross-fold files)
- Test files: `Test_Dst_NoAuction_ZScore_CF_*.txt` (9 cross-fold files)

### Provided Labels (Optional)

The dataset includes pre-computed classification labels encoded as:
- 1 = up movement
- 2 = stationary
- 3 = down movement

These can be optionally loaded with `return_provided_labels=True` and mapped to signed format (+1, 0, -1) with `label_format="signed"`.

### Inspect Dataset

```bash
python scripts/inspect_fi2010.py --config configs/inspect_fi2010.yaml
```

### Validate Dataset

Run the validation script to check file integrity, shapes, and statistics:

```bash
python scripts/validate_fi2010.py --config configs/inspect_fi2010.yaml
```

The output is a structured report that can be copy-pasted for review. It checks:
- File presence and parseability
- Matrix shape (must have >= 149 rows)
- Feature dimension after transpose (must be 144)
- NaN/inf counts
- Label value validity ({1,2,3})
- Z-score distribution heuristics

## Usage

### Local Execution

```bash
# Run FI-2010 classification with XGBoost
python -m src.train --config configs/xgboost_fi2010_cls.yaml

# Run FI-2010 classification with LSTM
python -m src.train --config configs/lstm_fi2010_cls.yaml

# Run all tests
pytest tests/ -v
```

### Google Colab

1. Open `notebooks/colab_run.ipynb` in Colab
2. Upload dataset files to `data/raw/fi2010/`
3. Run cells sequentially

### Run Outputs

Each training run saves artifacts to `runs/<run_id>/`:
- `config_resolved.yaml` - resolved configuration
- `metrics.json` - all computed metrics
- `predictions_test.csv` - predictions with columns: t_index, y_true, y_pred, p0, p1, p2
- `confusion_matrix_test.csv` - 3x3 confusion matrix
- `classification_report_test.json` - per-class precision/recall/f1
- Plots (confusion matrix, training curves if applicable)

Run ID format: `{model}_{h}{horizon}_f{fold}_w{window}_{timestamp}`

### Compare Runs

Generate cross-model comparison reports:

```bash
# Compare all runs
python scripts/compare_runs.py

# Filter by model
python scripts/compare_runs.py --model xgboost

# Filter by horizon and fold
python scripts/compare_runs.py --horizon 0 --fold all

# Custom output directory
python scripts/compare_runs.py --output-dir my_reports
```

Outputs saved to `reports/`:
- `metrics_table.csv` - all runs with metrics
- `metrics_table.md` - markdown formatted table
- `metrics_summary_by_model.csv` - mean/std by model

### MLflow Tracking

```bash
# Start MLflow UI (local)
mlflow ui

# View at http://localhost:5000
```

## Configuration

All experiments are configured via YAML files. Key parameters:

```yaml
experiment_name: my_experiment
seed: 42

data:
  raw_dir: ./data/raw/fi2010
  file_patterns: ["*.txt", "*.csv"]

label:
  tau: 10                      # Prediction horizon (ticks)
  task: regression             # regression or classification
  epsilon: 0.0002              # Threshold for classification

window:
  lookback: 50                 # Window size W
  mode: sequence               # sequence, tabular_flat, or tabular_stats

split:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  purge_boundary: true         # Remove samples crossing split boundaries

normalization:
  method: zscore               # zscore or minmax

model:
  name: lstm
  params:
    hidden_size: 64
    num_layers: 2
    dropout: 0.2

training:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 15
  device: auto                 # auto, cpu, cuda, mps
```

## Labels

### Regression
```
y_reg[t] = mid[t+tau] - mid[t]
```
where `mid[t] = (best_bid[t] + best_ask[t]) / 2`

### Classification
```
y_cls = 1   if y_reg > epsilon
y_cls = -1  if y_reg < -epsilon
y_cls = 0   otherwise
```

## Windowing

Each sample uses a lookback window:
- Input: `X[t] = features[t-W+1 : t]` (inclusive)
- Target: `y[t]` based on `mid[t+tau] - mid[t]`

Three modes:
1. `sequence`: Keep shape (N, W, F) for RNNs/TCN
2. `tabular_flat`: Flatten to (N, W*F)
3. `tabular_stats`: Compute per-feature stats (N, 4*F): mean, std, last, delta

## Leakage Prevention

1. **Chronological split**: Data split by time index (no shuffle)
2. **Boundary purge**: Samples where `t+tau` crosses split boundary are removed
3. **Train-only normalization**: Scaler fit only on training data

## Evaluation Metrics

### Regression
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Directional Accuracy: sign(pred) == sign(true)
- Magnitude Error: |pred - true|

### Classification
- Accuracy
- Macro F1
- ROC-AUC (one-vs-rest)
- Per-class accuracy (down, neutral, up)

## Output Artifacts

Each run produces:
- `metrics.json`: Evaluation metrics
- `predictions.csv`: Predictions with time indices
- `config.json`: Run configuration
- `predictions_vs_actual.png` (regression)
- `residuals.png` (regression)
- `confusion_matrix.png` (classification)
- `training_curves.png` (deep learning models)

## Models

| Model | Type | Mode | Description |
|-------|------|------|-------------|
| Ridge | Sklearn | tabular_flat | L2-regularized linear regression |
| Random Forest | Sklearn | tabular_stats | Ensemble of decision trees |
| XGBoost | Sklearn | tabular_flat | Gradient boosting |
| LSTM | PyTorch | sequence | Long Short-Term Memory |
| GRU | PyTorch | sequence | Gated Recurrent Unit |
| TCN | PyTorch | sequence | Temporal Convolutional Network |
| Autoencoder | PyTorch | sequence | LSTM autoencoder with prediction head |

## Reproducibility

- Fixed random seeds (numpy, torch, python)
- Deterministic torch operations when possible
- All parameters logged to MLflow
- Configuration saved with each run

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_leakage.py -v
pytest tests/test_labels.py -v
pytest tests/test_windows.py -v
```

## License

MIT
