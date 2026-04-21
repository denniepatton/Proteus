"""
Proteus Configuration Package

Provides:
- Dataset definitions via YAML (config/datasets/*.yaml)
- Experiment specifications via frozen dataclasses (experiment_spec.py)
"""

from .dataset_config import DatasetConfig, load_dataset_config, list_datasets

__all__ = [
    'DatasetConfig',
    'load_dataset_config',
    'list_datasets',
]
