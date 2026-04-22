"""Fold-metric collection and cross-fold objective computation.

Reads per-fold ``validation-fold_<k>_metrics.json`` files produced by
the training loop and aggregates them into a single CV objective dict
(written to ``cv_objective.json``).  Also provides the scalar
extraction helper used by every tuning entry point.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

from config.experiment_spec import ExperimentSpecError

__all__ = [
    "collect_fold_metrics",
    "nanmean",
    "nanstd",
    "compute_cv_objective",
    "objective_value",
]


def collect_fold_metrics(run_dir: str) -> List[Dict[str, Any]]:
    """Collect metrics from all CV fold directories."""
    out: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(run_dir)):
        if not entry.startswith("CV-fold_"):
            continue
        fold_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(fold_dir):
            continue
        try:
            fold_num = int(entry.split("_")[-1])
        except ValueError:
            continue
        metrics_path = os.path.join(fold_dir, f"validation-fold_{fold_num}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                out.append(json.load(f))
    return out


def nanmean(xs: List[float]) -> float:
    """Mean of finite numeric values, ignoring None and non-finite."""
    vals = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(sum(float(x) for x in vals) / len(vals))


def nanstd(xs: List[float]) -> float:
    """Sample standard deviation of finite numeric values."""
    vals = [float(x) for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return float(math.sqrt(var))


def compute_cv_objective(run_dir: str) -> Dict[str, Any]:
    """Compute CV objective from fold metrics."""
    rows = collect_fold_metrics(run_dir)
    if not rows:
        raise RuntimeError(f"No fold metrics found under: {run_dir}")

    tumor_mae = [float(r.get("tumor", {}).get("mae")) for r in rows if r.get("tumor", {}).get("mae") is not None]
    tumor_r2 = [float(r.get("tumor", {}).get("r2")) for r in rows if r.get("tumor", {}).get("r2") is not None]
    tumor_gvw_crps = [
        float(r.get("tumor", {}).get("gvw_crps"))
        for r in rows
        if r.get("tumor", {}).get("gvw_crps") is not None
    ]
    bg_mae = [float(r.get("background", {}).get("mae")) for r in rows if r.get("background", {}).get("mae") is not None]
    bg_r2 = [float(r.get("background", {}).get("r2")) for r in rows if r.get("background", {}).get("r2") is not None]
    bg_gvw_crps = [
        float(r.get("background", {}).get("gvw_crps"))
        for r in rows
        if r.get("background", {}).get("gvw_crps") is not None
    ]

    obj = {
        "folds_with_metrics": len(rows),
        "tumor_mae_mean": nanmean(tumor_mae),
        "tumor_mae_std": nanstd(tumor_mae),
        "tumor_r2_mean": nanmean(tumor_r2),
        "tumor_r2_std": nanstd(tumor_r2),
        "tumor_gvw_crps_mean": nanmean(tumor_gvw_crps),
        "tumor_gvw_crps_std": nanstd(tumor_gvw_crps),
        "background_mae_mean": nanmean(bg_mae),
        "background_mae_std": nanstd(bg_mae),
        "background_r2_mean": nanmean(bg_r2),
        "background_r2_std": nanstd(bg_r2),
        "background_gvw_crps_mean": nanmean(bg_gvw_crps),
        "background_gvw_crps_std": nanstd(bg_gvw_crps),
        # Combined objective: average of tumor and background MAE
        "val_loss": nanmean([nanmean(tumor_mae), nanmean(bg_mae)]),
    }

    # Balanced fine-tuning objective: per-fold balanced_tumor_gvw_crps
    # (equal-weight average of per-source GVW-CRPS when source_groups is
    # active, or per-cohort GVW-CRPS otherwise; computed in
    # _save_fold_predictions).
    balanced_crps = [
        float(r["balanced_tumor_gvw_crps"])
        for r in rows
        if r.get("balanced_tumor_gvw_crps") is not None
    ]
    if balanced_crps:
        obj["balanced_tumor_gvw_crps_mean"] = nanmean(balanced_crps)
        obj["balanced_tumor_gvw_crps_std"] = nanstd(balanced_crps)

    weighted_crps = [
        float(r["weighted_tumor_gvw_crps"])
        for r in rows
        if r.get("weighted_tumor_gvw_crps") is not None
    ]
    if weighted_crps:
        obj["weighted_tumor_gvw_crps_mean"] = nanmean(weighted_crps)
        obj["weighted_tumor_gvw_crps_std"] = nanstd(weighted_crps)

    # Balanced GVW-r: per-fold balanced_tumor_gvw_r
    # (equal-weight average of per-source GVW-r when source_groups is
    # active, or per-cohort GVW-r otherwise).
    balanced_gvw_r = [
        float(r["balanced_tumor_gvw_r"])
        for r in rows
        if r.get("balanced_tumor_gvw_r") is not None
    ]
    if balanced_gvw_r:
        obj["balanced_tumor_gvw_r_mean"] = nanmean(balanced_gvw_r)
        obj["balanced_tumor_gvw_r_std"] = nanstd(balanced_gvw_r)

    # Per-fold balanced_tumor_gvw_crps_std (stdev across cohorts within each fold)
    balanced_crps_std = [
        float(r["balanced_tumor_gvw_crps_std"])
        for r in rows
        if r.get("balanced_tumor_gvw_crps_std") is not None
    ]
    if balanced_crps_std:
        obj["balanced_tumor_gvw_crps_std_mean"] = nanmean(balanced_crps_std)

    # Per-fold balanced_tumor_gvw_r_std (stdev across cohorts within each fold)
    balanced_gvw_r_std = [
        float(r["balanced_tumor_gvw_r_std"])
        for r in rows
        if r.get("balanced_tumor_gvw_r_std") is not None
    ]
    if balanced_gvw_r_std:
        obj["balanced_tumor_gvw_r_std_mean"] = nanmean(balanced_gvw_r_std)

    # Composite lower-bound / upper-bound objectives
    # GVW-r lower bound: mean − std (maximize to raise worst-source floor)
    _r_m = obj.get("balanced_tumor_gvw_r_mean")
    _r_s = obj.get("balanced_tumor_gvw_r_std")
    if (_r_m is not None and _r_s is not None
            and math.isfinite(float(_r_m)) and math.isfinite(float(_r_s))):
        obj["balanced_tumor_gvw_r_lower_bound"] = float(_r_m) - float(_r_s)
    # GVW-CRPS upper bound: mean + std (minimize to lower worst-source ceiling)
    _c_m = obj.get("balanced_tumor_gvw_crps_mean")
    _c_s = obj.get("balanced_tumor_gvw_crps_std")
    if (_c_m is not None and _c_s is not None
            and math.isfinite(float(_c_m)) and math.isfinite(float(_c_s))):
        obj["balanced_tumor_gvw_crps_upper_bound"] = float(_c_m) + float(_c_s)

    # Min-source GVW-r: per-fold minimum across sources, averaged over folds.
    min_r_per_fold = [
        float(r["min_tumor_gvw_r"])
        for r in rows
        if r.get("min_tumor_gvw_r") is not None
           and math.isfinite(float(r["min_tumor_gvw_r"]))
    ]
    if min_r_per_fold:
        obj["min_tumor_gvw_r"] = nanmean(min_r_per_fold)

    out_path = os.path.join(run_dir, "cv_objective.json")
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)

    return obj


def objective_value(obj: Dict[str, Any], metric: str, direction: str) -> float:
    """Extract objective value from computed metrics."""
    if metric not in obj:
        raise ExperimentSpecError(
            f"Objective metric '{metric}' not found. Available: {sorted(obj.keys())}"
        )
    val = obj[metric]
    if val is None or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        return float("inf") if direction == "minimize" else float("-inf")
    return float(val)
