"""Minimal dataset configuration loader.

Reads per-dataset YAML files from config/datasets/ and returns frozen
dataclass instances.  Replaces the heavier config_loader.py that
carried unused CohortConfig / ExperimentConfig / PathsConfig classes.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Dataset YAML directory, relative to this file's parent (config/).
_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a single Triton dataset.

    Fields map 1-to-1 to the YAML schema::

        name: <str>
        sources:
          triton_tss: <relative path>
          triton_gb:  <relative path>
        labels:
          tfx:            <filename | null>
          tumor_exp:      <filename | null>
          background_exp: <filename | null>
    """
    name: str
    triton_tss: str
    triton_gb: str
    tfx_file: Optional[str]
    tumor_exp_file: Optional[str]
    background_exp_file: Optional[str]


def load_dataset_config(name: str) -> DatasetConfig:
    """Load a dataset configuration by name.

    Args:
        name: Dataset identifier (matches ``config/datasets/<name>.yaml``).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required fields are missing.
    """
    path = _DATASETS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if "name" not in data:
        raise ValueError(f"{path} missing required field: name")
    sources = data.get("sources", {})
    if "triton_tss" not in sources:
        raise ValueError(f"{path} missing required field: sources.triton_tss")
    if "triton_gb" not in sources:
        raise ValueError(f"{path} missing required field: sources.triton_gb")

    labels = data.get("labels", {})
    return DatasetConfig(
        name=data["name"],
        triton_tss=sources["triton_tss"],
        triton_gb=sources["triton_gb"],
        tfx_file=labels.get("tfx"),
        tumor_exp_file=labels.get("tumor_exp"),
        background_exp_file=labels.get("background_exp"),
    )


def list_datasets() -> list[str]:
    """Return sorted names of all configured datasets."""
    return sorted(p.stem for p in _DATASETS_DIR.glob("*.yaml"))
