"""Configuration management."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Data configuration."""
    raw_dir: str = "./data/raw/fi2010"
    processed_dir: str = "./data/processed"
    file_patterns: List[str] = field(default_factory=lambda: ["*.txt", "*.csv"])
    train_glob: str = "Train_Dst_NoAuction_ZScore_CF_*.txt"
    test_glob: str = "Test_Dst_NoAuction_ZScore_CF_*.txt"
    fold_id: str = "all"  # "all" or specific fold like "CF_9"


@dataclass
class LabelConfig:
    """Label configuration."""
    source: str = "provided"  # "provided" (FI-2010 labels) or "midprice"
    horizon_index: int = 0  # 0-4 for FI-2010 provided labels (k=10,20,30,50,100)
    task: str = "classification"
    tau: int = 10  # prediction horizon for midprice labels
    epsilon: float = 0.0002  # threshold for midprice classification


@dataclass
class WindowConfig:
    """Window configuration."""
    lookback: int = 50
    mode: str = "sequence"  # sequence, tabular_flat, tabular_stats


@dataclass
class SplitConfig:
    """Split configuration."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_boundary: bool = True


@dataclass
class NormalizationConfig:
    """Normalization configuration."""
    method: str = "none"  # zscore, minmax, or none (FI-2010 is pre-normalized)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "xgboost"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    device: str = "auto"
    # Autoencoder two-stage training
    pretrain_epochs: int = 50
    finetune_epochs: int = 50
    freeze_encoder: bool = True
    two_stage: bool = False  # Enable two-stage training for autoencoder


@dataclass
class OutputConfig:
    """Output configuration."""
    runs_dir: Optional[str] = None


@dataclass
class Config:
    """Main configuration."""
    experiment_name: str = "default"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(
            experiment_name=d.get("experiment_name", "default"),
            seed=d.get("seed", 42),
            data=DataConfig(**d.get("data", {})),
            label=LabelConfig(**d.get("label", {})),
            window=WindowConfig(**d.get("window", {})),
            split=SplitConfig(**d.get("split", {})),
            normalization=NormalizationConfig(**d.get("normalization", {})),
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            output=OutputConfig(**d.get("output", {})),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "data": self.data.__dict__,
            "label": self.label.__dict__,
            "window": self.window.__dict__,
            "split": self.split.__dict__,
            "normalization": self.normalization.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "output": self.output.__dict__,
        }
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.label.source == "midprice":
            assert self.split.train_ratio + self.split.val_ratio + self.split.test_ratio == 1.0, \
                "Split ratios must sum to 1.0"
        assert self.label.task in ("regression", "classification"), \
            f"Invalid task: {self.label.task}"
        assert self.window.mode in ("sequence", "tabular_flat", "tabular_stats", "flat"), \
            f"Invalid window mode: {self.window.mode}"
        assert self.normalization.method in ("zscore", "minmax", "none"), \
            f"Invalid normalization method: {self.normalization.method}"
        assert self.label.source in ("provided", "midprice"), \
            f"Invalid label source: {self.label.source}"
        if self.label.source == "provided":
            assert 0 <= self.label.horizon_index <= 4, \
                f"horizon_index must be 0-4, got {self.label.horizon_index}"
        
        # Sequence models require sequence mode
        sequence_models = {"lstm", "gru", "tcn", "autoencoder"}
        tabular_models = {"ridge", "random_forest", "xgboost", "logistic_regression"}
        model_name = self.model.name.lower()
        
        if model_name in sequence_models:
            assert self.window.mode == "sequence", \
                f"Model '{model_name}' requires window.mode='sequence', got '{self.window.mode}'"
            assert self.window.lookback > 1, \
                f"Sequence model requires window.lookback > 1, got {self.window.lookback}"
        
        if model_name in tabular_models:
            assert self.window.mode in ("tabular_flat", "tabular_stats", "flat"), \
                f"Model '{model_name}' requires tabular window mode, got '{self.window.mode}'"
