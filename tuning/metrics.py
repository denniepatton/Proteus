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
from proteus_utils.data_helpers import (
    compute_cohort_objective,
    compute_plcav4_2_objective,
    compute_soft_objective,
)

__all__ = [
    "collect_fold_metrics",
    "compute_cohort_objective",
    "compute_plcav4_2_objective",
    "compute_soft_objective",
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
        obj["balanced_tumor_gvw_crps_max"] = float(max(balanced_crps))

    weighted_crps = [
        float(r["weighted_tumor_gvw_crps"])
        for r in rows
        if r.get("weighted_tumor_gvw_crps") is not None
    ]
    if weighted_crps:
        obj["weighted_tumor_gvw_crps_mean"] = nanmean(weighted_crps)
        obj["weighted_tumor_gvw_crps_std"] = nanstd(weighted_crps)
        obj["weighted_tumor_gvw_crps_max"] = float(max(weighted_crps))

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

    cohort_gvw_r_mean = [
        float(r["cohort_tumor_gvw_r_mean"])
        for r in rows
        if r.get("cohort_tumor_gvw_r_mean") is not None
    ]
    if cohort_gvw_r_mean:
        obj["cohort_tumor_gvw_r_mean"] = nanmean(cohort_gvw_r_mean)

    cohort_gvw_r_std = [
        float(r["cohort_tumor_gvw_r_std"])
        for r in rows
        if r.get("cohort_tumor_gvw_r_std") is not None
    ]
    if cohort_gvw_r_std:
        obj["cohort_tumor_gvw_r_std"] = nanmean(cohort_gvw_r_std)

    cohort_gvw_r_balanced = [
        float(r["cohort_tumor_gvw_r_balanced"])
        for r in rows
        if r.get("cohort_tumor_gvw_r_balanced") is not None
    ]
    if cohort_gvw_r_balanced:
        obj["cohort_tumor_gvw_r_balanced"] = nanmean(cohort_gvw_r_balanced)
        obj["cohort_tumor_gvw_r_balanced_fold_std"] = nanstd(cohort_gvw_r_balanced)

    cohort_gvw_crps_mean = [
        float(r["cohort_tumor_gvw_crps_mean"])
        for r in rows
        if r.get("cohort_tumor_gvw_crps_mean") is not None
    ]
    if cohort_gvw_crps_mean:
        obj["cohort_tumor_gvw_crps_mean"] = nanmean(cohort_gvw_crps_mean)

    cohort_gvw_crps_std = [
        float(r["cohort_tumor_gvw_crps_std"])
        for r in rows
        if r.get("cohort_tumor_gvw_crps_std") is not None
    ]
    if cohort_gvw_crps_std:
        obj["cohort_tumor_gvw_crps_std"] = nanmean(cohort_gvw_crps_std)

    cohort_gvw_crps_balanced = [
        float(r["cohort_tumor_gvw_crps_balanced"])
        for r in rows
        if r.get("cohort_tumor_gvw_crps_balanced") is not None
    ]
    if cohort_gvw_crps_balanced:
        obj["cohort_tumor_gvw_crps_balanced"] = nanmean(cohort_gvw_crps_balanced)
        obj["cohort_tumor_gvw_crps_balanced_fold_std"] = nanstd(cohort_gvw_crps_balanced)

    robust_fold_objectives: List[float] = []
    for row in rows:
        row_value = row.get("cohort_tumor_gvw_v4_2_objective")
        if isinstance(row_value, (int, float)) and math.isfinite(float(row_value)):
            robust_fold_objectives.append(float(row_value))
            continue
        row_r = row.get("cohort_tumor_gvw_r_balanced")
        row_crps = row.get("cohort_tumor_gvw_crps_balanced")
        if not (
            isinstance(row_r, (int, float))
            and isinstance(row_crps, (int, float))
            and math.isfinite(float(row_r))
            and math.isfinite(float(row_crps))
        ):
            continue
        row_objective, _, _ = compute_plcav4_2_objective(
            float(row_r),
            float(row_crps),
        )
        if math.isfinite(row_objective):
            robust_fold_objectives.append(row_objective)
    if robust_fold_objectives:
        fold_mean = nanmean(robust_fold_objectives)
        fold_std = nanstd(robust_fold_objectives)
        if not math.isfinite(fold_std):
            fold_std = 0.0
        obj["cohort_tumor_gvw_v4_2_objective_mean"] = fold_mean
        obj["cohort_tumor_gvw_v4_2_objective_fold_std"] = fold_std
        obj["cohort_tumor_gvw_v4_2_objective"] = fold_mean - 0.5 * fold_std

    soft_fold_objectives: List[float] = []
    for row in rows:
        row_value = row.get("cohort_tumor_soft_objective")
        if not (isinstance(row_value, (int, float)) and math.isfinite(float(row_value))):
            row_value = row.get("cohort_tumor_gvw_v4_3_objective")
        if isinstance(row_value, (int, float)) and math.isfinite(float(row_value)):
            soft_fold_objectives.append(float(row_value))
            continue
        row_source_crps = {
            str(key[len("tumor_gvw_crps_"):]): float(value)
            for key, value in row.items()
            if isinstance(key, str)
            and key.startswith("tumor_gvw_crps_")
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
        row_source_r = {
            str(key[len("tumor_gvw_r_"):]): float(value)
            for key, value in row.items()
            if isinstance(key, str)
            and key.startswith("tumor_gvw_r_")
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
        if set(row_source_crps) != set(row_source_r) or not row_source_crps:
            continue
        row_objective, _, _ = compute_soft_objective(
            row_source_r,
            row_source_crps,
            required_names=sorted(row_source_r),
        )
        if math.isfinite(row_objective):
            soft_fold_objectives.append(row_objective)
    if soft_fold_objectives:
        fold_mean = nanmean(soft_fold_objectives)
        fold_std = nanstd(soft_fold_objectives)
        if not math.isfinite(fold_std):
            fold_std = 0.0
        obj["cohort_tumor_soft_objective_mean"] = fold_mean
        obj["cohort_tumor_soft_objective_fold_std"] = fold_std
        obj["cohort_tumor_soft_objective"] = fold_mean

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
