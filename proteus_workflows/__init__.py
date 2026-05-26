"""Workflow entry points for Proteus train/run/data operations."""

from .data import ensure_datasets_exist, generate_data, verify_dataset_stores
from .inference import run_model, run_saved_model_on_dataset
from .train import (
    build_training_configs_from_spec,
    full_train_from_splits,
    train_model_cv,
)

__all__ = [
    'ensure_datasets_exist',
    'generate_data',
    'verify_dataset_stores',
    'run_model',
    'run_saved_model_on_dataset',
    'build_training_configs_from_spec',
    'full_train_from_splits',
    'train_model_cv',
]
