"""
Proteus Configuration Package

Provides:
- Dataset definitions via YAML (config/datasets/*.yaml)
- Experiment specifications via frozen dataclasses (experiment_spec.py)
"""

from .dataset_config import DatasetConfig, load_dataset_config, list_datasets
from .runtime_paths import (
    DEFAULT_LABELS_DIR,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_REFERENCE_PATH,
    DEFAULT_ZARR_BASE,
    PROJECT_ROOT,
    normalize_path,
    resolve_labels_dir,
    resolve_raw_data_dir,
    resolve_reference_path,
    resolve_zarr_base,
)

__all__ = [
    'DatasetConfig',
    'load_dataset_config',
    'list_datasets',
    'PROJECT_ROOT',
    'DEFAULT_ZARR_BASE',
    'DEFAULT_RAW_DATA_DIR',
    'DEFAULT_REFERENCE_PATH',
    'DEFAULT_LABELS_DIR',
    'normalize_path',
    'resolve_zarr_base',
    'resolve_raw_data_dir',
    'resolve_reference_path',
    'resolve_labels_dir',
]
