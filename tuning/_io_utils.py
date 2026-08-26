"""Shared internal I/O helpers for the tuning package.

Small utilities used by multiple tuning submodules (Optuna coordination,
promotion ledger, reporting). Kept in one place so the fold-parsing and atomic
JSON-write logic cannot drift between modules.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, List

__all__ = ["folds_from_tsv", "write_json_atomic"]


def folds_from_tsv(training_set_tsv: str) -> List[int]:
    """Return the sorted unique fold ids present in a sample-splits TSV."""
    folds: set[int] = set()
    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            folds.add(int(row['Fold']))
    return sorted(folds)


def write_json_atomic(path: str, obj: Any) -> None:
    """Write ``obj`` as JSON to ``path`` atomically via a temp-file rename."""
    tmp = f"{path}.tmp"
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
