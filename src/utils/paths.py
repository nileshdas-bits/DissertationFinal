"""Path utilities for cross-platform compatibility."""
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get project root directory."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").exists() and (parent / "data").exists():
            return parent
    return Path.cwd()


def get_data_dir() -> Path:
    """Get data directory."""
    return get_project_root() / "data"


def get_raw_data_dir() -> Path:
    """Get raw data directory."""
    return get_data_dir() / "raw" / "fi2010"


def get_processed_data_dir() -> Path:
    """Get processed data directory."""
    path = get_data_dir() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_configs_dir() -> Path:
    """Get configs directory."""
    return get_project_root() / "configs"


def get_outputs_dir(run_name: Optional[str] = None) -> Path:
    """Get outputs directory for a run."""
    base = get_project_root() / "outputs"
    if run_name:
        path = base / run_name
    else:
        path = base
    path.mkdir(parents=True, exist_ok=True)
    return path
