"""Hyperparameter tuning utilities for Proteus.

Submodules
----------
param_resolution
    Parameter suggestion, config resolution, and fixed-param replay.
metrics
    Fold-metric collection and cross-fold objective computation.
promotion
    Cross-fold promotion of top hyperparameter configurations.
reporting
    Study reporting, phase-transition analysis, and fold stability.
optuna_tuner
    Public entry points: CV tuning, fold-parallel tuning,
    coordinated ask/tell/run.
"""

from .optuna_tuner import (
    coordinated_ask,
    coordinated_tell,
    run_coordinated_fold,
    run_optuna_cv_tuning,
    run_optuna_fold_tuning,
)

__all__ = [
    "run_optuna_cv_tuning",
    "run_optuna_fold_tuning",
    "coordinated_ask",
    "coordinated_tell",
    "run_coordinated_fold",
]
