"""
Configuration loader for Proteus.

This module provides utilities for loading and validating YAML configuration files
for datasets, cohorts, experiments, and paths.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class DatasetConfig:
    """
    Configuration for a single dataset.
    
    Required YAML schema:
        name: str
        sources:
            triton_tss: str  # Relative path to TSS results dir
            triton_gb: str   # Relative path to GB results dir
        labels:
            tfx: Optional[str]            # TFX labels filename (null for unlabeled datasets)
            tumor_exp: Optional[str]      # Tumor expression filename
            background_exp: Optional[str] # Background expression filename
    """
    name: str
    triton_tss: str
    triton_gb: str
    tfx_file: Optional[str]
    tumor_exp_file: Optional[str]
    background_exp_file: Optional[str]


@dataclass
class CohortConfig:
    """Configuration for a training/inference cohort."""
    name: str
    description: str
    mode: str  # cv, full_train, inference
    datasets: List[Dict[str, Any]]
    cv: Dict[str, Any]
    output: Dict[str, str]
    holdout_datasets: List[Dict[str, Any]] = field(default_factory=list)
    model: Optional[Dict[str, str]] = None  # For inference mode
    transfer: Optional[Dict[str, Any]] = None  # For transfer learning
    
    @property
    def is_cv(self) -> bool:
        return self.mode == 'cv'
    
    @property
    def is_inference(self) -> bool:
        return self.mode == 'inference'
    
    @property
    def n_folds(self) -> Optional[int]:
        return self.cv.get('n_folds')
    
    def get_datasets_by_role(self, role: str) -> List[str]:
        """Get dataset names with a specific role."""
        return [d['name'] for d in self.datasets if d.get('role') == role]


@dataclass
class ExperimentConfig:
    """Configuration for model and training hyperparameters."""
    name: str
    description: str
    model: Dict[str, Any]
    training: Dict[str, Any]
    dataloader: Dict[str, Any]
    checkpointing: Dict[str, Any]
    logging: Dict[str, Any]
    hp_search: Optional[Dict[str, Any]] = None
    
    @property
    def is_hp_search(self) -> bool:
        return self.hp_search is not None and self.hp_search.get('enabled', False)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Extract model architecture config, resolving any search spaces to defaults."""
        if not self.is_hp_search:
            return self.model
        # For HP search configs, return first/default value
        return self._resolve_search_spaces(self.model)
    
    def _resolve_search_spaces(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert search space definitions to default values."""
        resolved = {}
        for key, value in config.items():
            if isinstance(value, dict):
                if 'search' in value:
                    # This is a search space, return first/default value
                    if 'values' in value:
                        resolved[key] = value['values'][0]
                    elif 'low' in value:
                        resolved[key] = value['low']
                    else:
                        resolved[key] = value
                else:
                    resolved[key] = self._resolve_search_spaces(value)
            else:
                resolved[key] = value
        return resolved


@dataclass
class PathsConfig:
    """Configuration for base paths and environment settings."""
    base: Dict[str, str]
    reference: Dict[str, str]
    environments: Dict[str, Dict[str, str]]
    active_environment: str = 'local'
    
    def get_path(self, key: str) -> str:
        """Get a path, with environment variable expansion."""
        # Check base paths
        if key in self.base:
            return self._expand_path(self.base[key])
        # Check reference paths
        if key in self.reference:
            return self._expand_path(self.reference[key])
        # Check environment-specific paths
        env_paths = self.environments.get(self.active_environment, {})
        if key in env_paths:
            return self._expand_path(env_paths[key])
        raise KeyError(f"Path key '{key}' not found in configuration")
    
    def _expand_path(self, path: str) -> str:
        """Expand environment variables and references to other paths."""
        # First expand environment variables
        path = os.path.expandvars(path)
        
        # Then expand references like ${data_dir}
        for key, value in self.base.items():
            path = path.replace(f"${{{key}}}", os.path.expandvars(value))
        for key, value in self.reference.items():
            path = path.replace(f"${{{key}}}", os.path.expandvars(value))
            
        return path


# =============================================================================
# Configuration Loader Class
# =============================================================================

class ConfigLoader:
    """
    Main configuration loader for Proteus.
    
    Handles loading, validation, and resolution of YAML configuration files.
    
    Usage:
        loader = ConfigLoader(config_dir="/path/to/config")
        
        # Load individual configs
        dataset = loader.load_dataset("PC-PDX")
        cohort = loader.load_cohort("prostate_cv")
        experiment = loader.load_experiment("default")
        
        # Load complete run configuration
        run_config = loader.load_run_config(
            cohort="prostate_cv",
            experiment="default"
        )
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Path to config directory. If None, uses default location
                       relative to this file.
        """
        if config_dir is None:
            # Default: config/ directory at project root
            config_dir = Path(__file__).parent
        
        self.config_dir = Path(config_dir)
        self.datasets_dir = self.config_dir / "datasets"
        self.cohorts_dir = self.config_dir / "cohorts"
        self.experiments_dir = self.config_dir / "experiments"
        
        # Cache for loaded configs
        self._paths_config: Optional[PathsConfig] = None
        self._dataset_cache: Dict[str, DatasetConfig] = {}
        self._cohort_cache: Dict[str, CohortConfig] = {}
        self._experiment_cache: Dict[str, ExperimentConfig] = {}
        
    def load_paths(self, environment: str = 'local') -> PathsConfig:
        """Load the paths configuration."""
        if self._paths_config is not None:
            return self._paths_config
        
        paths_file = self.config_dir / "paths.yaml"
        if not paths_file.exists():
            # Try template
            paths_file = self.config_dir / "paths.yaml.template"
            if not paths_file.exists():
                raise FileNotFoundError(
                    f"No paths.yaml or paths.yaml.template found in {self.config_dir}"
                )
            logger.warning("Using paths.yaml.template - create paths.yaml for your environment")
        
        with open(paths_file) as f:
            data = yaml.safe_load(f)
        
        self._paths_config = PathsConfig(
            base=data.get('base', {}),
            reference=data.get('reference', {}),
            environments=data.get('environments', {}),
            active_environment=environment
        )
        return self._paths_config
    
    def load_dataset(self, name: str) -> DatasetConfig:
        """
        Load a dataset configuration by name.
        
        Args:
            name: Dataset name (filename without .yaml extension)
            
        Returns:
            DatasetConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required fields are missing
        """
        if name in self._dataset_cache:
            return self._dataset_cache[name]
        
        dataset_file = self.datasets_dir / f"{name}.yaml"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_file}")
        
        with open(dataset_file) as f:
            data = yaml.safe_load(f)
        
        # Validate required fields
        if 'name' not in data:
            raise ValueError(f"Dataset config {dataset_file} missing required field: name")
        sources = data.get('sources', {})
        if 'triton_tss' not in sources:
            raise ValueError(f"Dataset config {dataset_file} missing required field: sources.triton_tss")
        if 'triton_gb' not in sources:
            raise ValueError(f"Dataset config {dataset_file} missing required field: sources.triton_gb")
        
        labels = data.get('labels', {})
        
        config = DatasetConfig(
            name=data['name'],
            triton_tss=sources['triton_tss'],
            triton_gb=sources['triton_gb'],
            tfx_file=labels.get('tfx'),
            tumor_exp_file=labels.get('tumor_exp'),
            background_exp_file=labels.get('background_exp'),
        )
        
        self._dataset_cache[name] = config
        return config
    
    def load_cohort(self, name: str) -> CohortConfig:
        """
        Load a cohort configuration by name.
        
        Args:
            name: Cohort name (filename without .yaml extension)
            
        Returns:
            CohortConfig object
        """
        if name in self._cohort_cache:
            return self._cohort_cache[name]
        
        cohort_file = self.cohorts_dir / f"{name}.yaml"
        if not cohort_file.exists():
            raise FileNotFoundError(f"Cohort config not found: {cohort_file}")
        
        with open(cohort_file) as f:
            data = yaml.safe_load(f)
        
        config = CohortConfig(
            name=data.get('name', name),
            description=data.get('description', ''),
            mode=data.get('mode', 'cv'),
            datasets=data.get('datasets', []),
            cv=data.get('cv', {}),
            output=data.get('output', {}),
            holdout_datasets=data.get('holdout_datasets', []),
            model=data.get('model'),
            transfer=data.get('transfer')
        )
        
        self._cohort_cache[name] = config
        return config
    
    def load_experiment(self, name: str) -> ExperimentConfig:
        """
        Load an experiment configuration by name.
        
        Args:
            name: Experiment name (filename without .yaml extension)
            
        Returns:
            ExperimentConfig object
        """
        if name in self._experiment_cache:
            return self._experiment_cache[name]
        
        experiment_file = self.experiments_dir / f"{name}.yaml"
        if not experiment_file.exists():
            raise FileNotFoundError(f"Experiment config not found: {experiment_file}")
        
        with open(experiment_file) as f:
            data = yaml.safe_load(f)
        
        config = ExperimentConfig(
            name=data.get('name', name),
            description=data.get('description', ''),
            model=data.get('model', {}),
            training=data.get('training', {}),
            dataloader=data.get('dataloader', {}),
            checkpointing=data.get('checkpointing', {}),
            logging=data.get('logging', {}),
            hp_search=data.get('hp_search')
        )
        
        self._experiment_cache[name] = config
        return config
    
    def load_run_config(
        self,
        cohort: str,
        experiment: str = 'default',
        environment: str = 'local'
    ) -> Dict[str, Any]:
        """
        Load a complete run configuration.
        
        This combines cohort, experiment, and paths configs into a single
        dictionary suitable for running training or inference.
        
        Args:
            cohort: Cohort name
            experiment: Experiment name (default: 'default')
            environment: Environment name for paths (default: 'local')
            
        Returns:
            Dictionary with all configuration needed for a run
        """
        paths_config = self.load_paths(environment)
        cohort_config = self.load_cohort(cohort)
        experiment_config = self.load_experiment(experiment)
        
        # Load all referenced datasets
        datasets = {}
        for ds in cohort_config.datasets:
            ds_name = ds['name']
            datasets[ds_name] = self.load_dataset(ds_name)
        
        # Also load holdout datasets
        for ds in cohort_config.holdout_datasets:
            ds_name = ds['name']
            if ds_name not in datasets:
                datasets[ds_name] = self.load_dataset(ds_name)
        
        return {
            'paths': paths_config,
            'cohort': cohort_config,
            'experiment': experiment_config,
            'datasets': datasets
        }
    
    def list_datasets(self) -> List[str]:
        """List all available dataset configurations."""
        if not self.datasets_dir.exists():
            return []
        return [f.stem for f in self.datasets_dir.glob("*.yaml")]
    
    def list_cohorts(self) -> List[str]:
        """List all available cohort configurations."""
        if not self.cohorts_dir.exists():
            return []
        return [f.stem for f in self.cohorts_dir.glob("*.yaml")]
    
    def list_experiments(self) -> List[str]:
        """List all available experiment configurations."""
        if not self.experiments_dir.exists():
            return []
        return [f.stem for f in self.experiments_dir.glob("*.yaml")]
    
    def validate_cohort(self, name: str) -> List[str]:
        """
        Validate a cohort configuration.
        
        Checks that all referenced datasets exist and have required fields.
        
        Args:
            name: Cohort name to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            cohort = self.load_cohort(name)
        except Exception as e:
            return [f"Failed to load cohort '{name}': {e}"]
        
        # Check all referenced datasets exist
        for ds in cohort.datasets:
            ds_name = ds.get('name')
            if not ds_name:
                errors.append(f"Dataset entry missing 'name' field in cohort '{name}'")
                continue
                
            try:
                ds_config = self.load_dataset(ds_name)
                
                # Check inference mode doesn't require truth labels from unlabeled data
                if cohort.mode != 'inference' and ds.get('role') in ['train', 'train_val']:
                    if not ds_config.has_truth_labels:
                        errors.append(
                            f"Dataset '{ds_name}' has no truth labels but is used for "
                            f"training in cohort '{name}'"
                        )
            except FileNotFoundError:
                errors.append(f"Dataset config not found: '{ds_name}'")
        
        # Validate mode-specific requirements
        if cohort.mode == 'cv':
            if not cohort.n_folds or cohort.n_folds < 2:
                errors.append(f"CV cohort '{name}' requires n_folds >= 2")
        elif cohort.mode == 'inference':
            if not cohort.model:
                errors.append(f"Inference cohort '{name}' requires 'model' specification")
        
        return errors
    
    def clear_cache(self):
        """Clear all cached configurations."""
        self._paths_config = None
        self._dataset_cache.clear()
        self._cohort_cache.clear()
        self._experiment_cache.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def get_default_loader() -> ConfigLoader:
    """Get a ConfigLoader with default config directory."""
    return ConfigLoader()


def load_run_config(
    cohort: str,
    experiment: str = 'default',
    environment: str = 'local',
    config_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load a complete run configuration.
    
    Args:
        cohort: Cohort name
        experiment: Experiment name
        environment: Environment name for paths
        config_dir: Optional custom config directory
        
    Returns:
        Complete run configuration dictionary
    """
    loader = ConfigLoader(config_dir)
    return loader.load_run_config(cohort, experiment, environment)


def resolve_paths(
    config: Dict[str, Any],
    paths_config: PathsConfig
) -> Dict[str, Any]:
    """
    Resolve all path references in a configuration dictionary.
    
    Recursively processes the config and expands any path references.
    
    Args:
        config: Configuration dictionary
        paths_config: PathsConfig object for resolution
        
    Returns:
        Config with resolved paths
    """
    if isinstance(config, dict):
        resolved = {}
        for key, value in config.items():
            if isinstance(value, str) and (
                value.startswith('${') or 
                key.endswith('_dir') or 
                key.endswith('_path') or 
                key.endswith('_file')
            ):
                resolved[key] = paths_config._expand_path(value)
            elif isinstance(value, (dict, list)):
                resolved[key] = resolve_paths(value, paths_config)
            else:
                resolved[key] = value
        return resolved
    elif isinstance(config, list):
        return [resolve_paths(item, paths_config) for item in config]
    return config
