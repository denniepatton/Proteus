"""Data loading and gene-weight utilities for XGBoost baseline.

Reads from the same Zarr stores and BTS_sample-splits.tsv that Proteus uses,
guaranteeing identical feature values, target values, and CV fold assignments.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# Project root (two levels up from this file → Proteus/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Zarr path resolution (mirrors Proteus._collect_*_zarr_paths)
# ---------------------------------------------------------------------------

def _resolve_zarr_path(cohort: str, sample: str, zarr_base: str) -> str:
    """Build ``{zarr_base}/{cohort}/zarr_files/{sample_name}.zarr``."""
    sample_name = sample
    if sample.startswith(f"{cohort}_"):
        sample_name = sample[len(cohort) + 1:]
    return os.path.join(zarr_base, cohort, "zarr_files", f"{sample_name}.zarr")


def parse_splits_tsv(
    tsv_path: str,
) -> List[Dict[str, Any]]:
    """Parse BTS_sample-splits.tsv → list of {Cohort, Sample, Fold}."""
    rows: List[Dict[str, Any]] = []
    with open(tsv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({
                "Cohort": row["Cohort"].strip(),
                "Sample": row["Sample"].strip(),
                "Fold": int(row["Fold"]),
            })
    return rows


def collect_zarr_paths(
    tsv_path: str,
    zarr_base: str,
    fold: Optional[int] = None,
    exclude_fold: Optional[int] = None,
    exclude_cohorts: Optional[List[str]] = None,
    include_cohorts: Optional[List[str]] = None,
) -> List[str]:
    """Collect Zarr paths filtered by fold and/or cohort.

    Parameters
    ----------
    fold:
        If given, include *only* this fold.
    exclude_fold:
        If given, include everything *except* this fold.
    exclude_cohorts:
        If given, skip samples whose Cohort is in this list.
    include_cohorts:
        If given, include *only* samples whose Cohort is in this list.
    Neither fold filter → all samples.
    """
    rows = parse_splits_tsv(tsv_path)
    paths: List[str] = []
    for r in rows:
        if fold is not None and r["Fold"] != fold:
            continue
        if exclude_fold is not None and r["Fold"] == exclude_fold:
            continue
        if exclude_cohorts and r["Cohort"] in exclude_cohorts:
            continue
        if include_cohorts and r["Cohort"] not in include_cohorts:
            continue
        zp = _resolve_zarr_path(r["Cohort"], r["Sample"], zarr_base)
        if os.path.isdir(zp):
            paths.append(zp)
    return paths


# ---------------------------------------------------------------------------
# Feature / label loading
# ---------------------------------------------------------------------------

def load_sample_arrays(
    zarr_path: str,
) -> Dict[str, np.ndarray]:
    """Load scalar features + labels from a single sample Zarr store.

    Returns dict with keys:
        TSS_features   (n_genes, 11)  float32
        GB_features    (n_genes, 11)  float32
        ref_scalars    (n_genes, 3)   float32
        tumor_exp      (n_genes,)     float32
        gene_names     (n_genes,)     str
        sample_name    str (scalar)
    """
    store = zarr.open(zarr_path, mode="r")
    return {
        "TSS_features": store["flat/TSS_features"][:],
        "GB_features": store["flat/GB_features"][:],
        "ref_scalars": store["flat/ref_scalars"][:],
        "tumor_exp": store["flat/tumor_exp"][:],
        "gene_names": np.array([str(g) for g in store["gene_names"][:]]),
        "sample_name": str(store.attrs.get("sample_name", Path(zarr_path).stem)),
    }


def load_dataset(
    zarr_paths: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all samples into flat arrays, masking NaN targets.

    Returns
    -------
    X : (N, 25) float32
        Concatenated [TSS_features(11) | GB_features(11) | ref_scalars(3)].
    y : (N,) float32
        tumor_exp for valid (non-NaN) genes.
    gene_names : (N,) object
        Gene name per row.
    sample_ids : (N,) object
        Sample name per row.
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_genes: List[np.ndarray] = []
    all_samples: List[np.ndarray] = []

    for zp in zarr_paths:
        arrs = load_sample_arrays(zp)
        n = arrs["TSS_features"].shape[0]
        X_sample = np.concatenate(
            [arrs["TSS_features"], arrs["GB_features"], arrs["ref_scalars"]],
            axis=1,
        )  # (n_genes, 25)
        y_sample = arrs["tumor_exp"]  # (n_genes,)

        # Mask out NaN targets
        valid = np.isfinite(y_sample)
        if valid.sum() == 0:
            continue

        all_X.append(X_sample[valid])
        all_y.append(y_sample[valid])
        all_genes.append(arrs["gene_names"][valid])
        all_samples.append(
            np.full(int(valid.sum()), arrs["sample_name"], dtype=object)
        )

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    gene_names = np.concatenate(all_genes, axis=0)
    sample_ids = np.concatenate(all_samples, axis=0)
    return X, y, gene_names, sample_ids


# ---------------------------------------------------------------------------
# Gene variance weights (mirrors proteus_utils.compute_gene_weights_from_zarr)
# ---------------------------------------------------------------------------

def compute_gene_variances(
    zarr_paths: List[str],
    label_key: str = "flat/tumor_exp",
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-gene variance (raw + clipped) identical to Proteus.

    Returns (raw_weights, clipped_weights) each mapping gene_name → float.
    """
    gene_sum: Dict[str, float] = {}
    gene_sum_sq: Dict[str, float] = {}
    gene_count: Dict[str, int] = {}

    for zp in zarr_paths:
        store = zarr.open(zp, mode="r")
        gnames = [str(g) for g in store["gene_names"][:]]
        labels = store[label_key][:]

        for i, gname in enumerate(gnames):
            val = float(labels[i])
            if not np.isfinite(val):
                continue
            gene_sum[gname] = gene_sum.get(gname, 0.0) + val
            gene_sum_sq[gname] = gene_sum_sq.get(gname, 0.0) + val * val
            gene_count[gname] = gene_count.get(gname, 0) + 1

    raw: Dict[str, float] = {}
    clipped: Dict[str, float] = {}
    for g in gene_sum:
        n = gene_count[g]
        if n < 2:
            raw[g] = 0.0
            clipped[g] = clip_min
            continue
        mean = gene_sum[g] / n
        var = max(0.0, gene_sum_sq[g] / n - mean * mean)
        raw[g] = float(var)
        clipped[g] = float(np.clip(var, clip_min, clip_max))

    return raw, clipped


def gene_weight_vector(
    gene_names: np.ndarray,
    weight_map: Dict[str, float],
    default: float = 1.0,
) -> np.ndarray:
    """Map a gene-name array to a weight vector via a dict lookup."""
    return np.array(
        [weight_map.get(str(g), default) for g in gene_names],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Default paths helper
# ---------------------------------------------------------------------------

def default_zarr_base() -> str:
    """Return the default zarr_base from env or paths.yaml."""
    env_val = os.environ.get("PROTEUS_ZARR_BASE")
    if env_val:
        return env_val
    return "/hpc/temp/ha_g/user/rpatton/ProteusZarrs"


def default_splits_tsv() -> str:
    return str(PROJECT_ROOT / "data" / "BTS_sample-splits.tsv")


def default_panft_splits_tsv() -> str:
    return str(PROJECT_ROOT / "data" / "PAN-FT_sample-splits.tsv")
