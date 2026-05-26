"""Runtime path resolution for Proteus.

The public framework should run with sensible repository-local defaults while
still allowing site-specific overrides through environment variables or
config/paths.yaml.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZARR_BASE = str((PROJECT_ROOT / 'data' / 'zarr').resolve())
DEFAULT_RAW_DATA_DIR = str((PROJECT_ROOT / 'data' / 'raw_triton').resolve())
DEFAULT_REFERENCE_PATH = str(
    (PROJECT_ROOT / 'reference' / 'MANE.GRCh38.v1.3.UniversalReference.tsv').resolve()
)
DEFAULT_LABELS_DIR = str((PROJECT_ROOT / 'labels').resolve())

_OPTIONAL_PATHS_FILE = PROJECT_ROOT / 'config' / 'paths.yaml'
_TOKEN_RE = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


def _expand_tokens(value: str, context: Dict[str, str]) -> str:
    expanded = str(value)
    for _ in range(8):
        changed = False

        def _replace(match: re.Match[str]) -> str:
            nonlocal changed

            key = match.group(1)
            fallback = match.group(3)
            if key in context:
                changed = True
                return context[key]

            env_value = os.environ.get(key)
            if env_value:
                changed = True
                return env_value

            if fallback is not None:
                changed = True
                return fallback

            return match.group(0)

        new_value = _TOKEN_RE.sub(_replace, expanded)
        new_value = os.path.expandvars(os.path.expanduser(new_value))
        if new_value == expanded and not changed:
            break
        expanded = new_value
    return expanded


def normalize_path(
    path_value: str,
    *,
    base_context: Optional[Dict[str, str]] = None,
    relative_to_project_root: bool = True,
) -> str:
    context = {
        'project_dir': str(PROJECT_ROOT),
        'PROJECT_ROOT': str(PROJECT_ROOT),
    }
    if base_context:
        context.update(base_context)

    expanded = _expand_tokens(path_value, context).strip()
    if not expanded:
        return expanded

    if expanded.startswith((
        'sqlite://',
        'postgresql://',
        'mysql://',
        's3://',
        'gs://',
        'az://',
        'http://',
        'https://',
    )):
        return expanded

    path = Path(expanded)
    if relative_to_project_root and not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def _load_paths_doc() -> Dict[str, Any]:
    if not _OPTIONAL_PATHS_FILE.exists():
        return {}

    try:
        with open(_OPTIONAL_PATHS_FILE, 'r') as handle:
            doc = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}

    return doc if isinstance(doc, dict) else {}


def _load_base_paths() -> Dict[str, str]:
    doc = _load_paths_doc()
    raw_base = doc.get('base', {})
    if not isinstance(raw_base, dict):
        return {}

    resolved: Dict[str, str] = {}
    for key, raw_value in raw_base.items():
        if raw_value is None:
            continue
        resolved[str(key)] = normalize_path(str(raw_value), base_context=resolved)
    return resolved


def resolve_zarr_base(explicit: Optional[str] = None) -> str:
    if explicit:
        return normalize_path(explicit)

    configured = _load_base_paths().get('zarr_base_dir')
    if configured:
        return configured

    env_value = os.environ.get('PROTEUS_ZARR_BASE')
    if env_value:
        return normalize_path(env_value)

    return DEFAULT_ZARR_BASE


def resolve_raw_data_dir(explicit: Optional[str] = None) -> str:
    if explicit:
        return normalize_path(explicit)

    configured = _load_base_paths().get('raw_data_dir')
    if configured:
        return configured

    env_value = os.environ.get('PROTEUS_RAW_DATA_DIR')
    if env_value:
        return normalize_path(env_value)

    return DEFAULT_RAW_DATA_DIR


def resolve_labels_dir() -> str:
    configured = _load_base_paths().get('labels_dir')
    if configured:
        return configured
    return DEFAULT_LABELS_DIR


def resolve_reference_path(explicit: Optional[str] = None) -> str:
    if explicit:
        return normalize_path(explicit)

    doc = _load_paths_doc()
    base_paths = _load_base_paths()
    reference_section = doc.get('reference', {})
    if isinstance(reference_section, dict):
        universal_ref = reference_section.get('universal_ref')
        if universal_ref:
            return normalize_path(str(universal_ref), base_context=base_paths)

    return DEFAULT_REFERENCE_PATH
