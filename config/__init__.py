"""
Proteus Configuration Package

Provides YAML-based configuration management for:
- Dataset definitions (data sources, labels, properties)
- Cohort definitions (dataset combinations, CV settings, modes)
- Experiment definitions (model architecture, training hyperparameters)
- Path management (environment-specific paths)

Usage:
    from config import ConfigLoader, load_run_config
    
    # Load configuration for a training run
    config = load_run_config(
        cohort="prostate_cv",
        experiment="default",
        environment="cluster"
    )
    
    # Access components
    paths = config['paths']
    cohort = config['cohort']
    experiment = config['experiment']
    datasets = config['datasets']
    
    # Or use the loader directly for more control
    loader = ConfigLoader()
    dataset = loader.load_dataset("PC-PDX")
    cohort = loader.load_cohort("prostate_cv")
"""

from .config_loader import (
    ConfigLoader,
    DatasetConfig,
    CohortConfig,
    ExperimentConfig,
    PathsConfig,
    load_run_config,
    get_default_loader,
    resolve_paths
)

__all__ = [
    'ConfigLoader',
    'DatasetConfig',
    'CohortConfig',
    'ExperimentConfig',
    'PathsConfig',
    'load_run_config',
    'get_default_loader',
    'resolve_paths'
]
