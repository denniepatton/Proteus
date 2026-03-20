"""Training, evaluation, and prediction saving for the XGBoost baseline.

Mirrors the Proteus 5-fold CV pipeline:
  - Same zarr features (25 scalars), same fold assignments, same gene-variance
    weighting.  Single-regressor point predictions evaluated with RMSE,
    GVW-R², per-gene / per-sample Spearman R².
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from .data_utils import (
    PROJECT_ROOT,
    collect_zarr_paths,
    compute_gene_variances,
    default_splits_tsv,
    default_zarr_base,
    gene_weight_vector,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARLY_STOPPING_ROUNDS = 30


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    fold_num: int,
    xgb_params: Dict[str, Any],
    n_estimators: int = 1000,
    tsv_path: Optional[str] = None,
    zarr_base: Optional[str] = None,
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """Train a single XGBoost regressor for one CV fold.

    Returns (model, fold_info).
    """
    tsv_path = tsv_path or default_splits_tsv()
    zarr_base = zarr_base or default_zarr_base()

    # ---- Collect zarr paths ----
    train_zarr = collect_zarr_paths(tsv_path, zarr_base, exclude_fold=fold_num)
    val_zarr = collect_zarr_paths(tsv_path, zarr_base, fold=fold_num)
    all_zarr = collect_zarr_paths(tsv_path, zarr_base)

    # ---- Gene variance weights ----
    _, clipped_weights = compute_gene_variances(train_zarr)
    raw_weights_full, _ = compute_gene_variances(all_zarr)

    # ---- Load data ----
    X_train, y_train, genes_train, samples_train = load_dataset(train_zarr)
    X_val, y_val, genes_val, samples_val = load_dataset(val_zarr)

    w_train = gene_weight_vector(genes_train, clipped_weights, default=0.1)
    w_val = gene_weight_vector(genes_val, clipped_weights, default=0.1)

    # ---- Build DMatrix ----
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    # ---- Train (weighted squared error) ----
    params = {**xgb_params, "objective": "reg:squarederror"}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    fold_info = {
        "fold_num": fold_num,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "best_iteration": int(model.best_iteration),
        "X_val": X_val,
        "y_val": y_val,
        "genes_val": genes_val,
        "samples_val": samples_val,
        "raw_weights_full": raw_weights_full,
    }
    return model, fold_info


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _pearson_r(y: np.ndarray, yhat: np.ndarray) -> float:
    y_c = y - float(np.mean(y))
    yhat_c = yhat - float(np.mean(yhat))
    denom = float(np.sqrt(np.sum(y_c * y_c) * np.sum(yhat_c * yhat_c)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.clip(np.sum(y_c * yhat_c) / denom, -1.0, 1.0))


def _rankdata(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)


def _spearman_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    r = _pearson_r(_rankdata(y), _rankdata(yhat))
    return r * r if np.isfinite(r) else float("nan")


def _weighted_r2(
    y: np.ndarray,
    yhat: np.ndarray,
    genes: np.ndarray,
    weight_map: Optional[Dict[str, float]],
) -> float:
    """Gene-variance-weighted R² — identical to Proteus ``_weighted_r2``."""
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 2:
        return float("nan")
    y_m = y[mask]
    yhat_m = yhat[mask]
    if weight_map is not None:
        w = np.array(
            [float(weight_map.get(str(g), 1.0)) for g in genes[mask]],
            dtype=np.float64,
        )
    else:
        w = np.ones(int(mask.sum()), dtype=np.float64)
    ss_res = float(np.sum(w * (y_m - yhat_m) ** 2))
    w_sum = float(np.sum(w))
    w_mean_y = float(np.sum(w * y_m)) / w_sum
    ss_tot = float(np.sum(w * (y_m - w_mean_y) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def evaluate_fold(
    model: xgb.Booster,
    fold_info: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compute evaluation metrics for one fold.

    Returns (metrics_dict, predictions_df).
    """
    X_val = fold_info["X_val"]
    y_val = fold_info["y_val"]
    genes_val = fold_info["genes_val"]
    samples_val = fold_info["samples_val"]
    raw_weights = fold_info["raw_weights_full"]
    fold_num = fold_info["fold_num"]

    dval = xgb.DMatrix(X_val)
    pred = model.predict(dval)

    y = y_val.astype(np.float64)
    yhat = pred.astype(np.float64)

    # ---- Basic regression metrics ----
    mask = np.isfinite(y) & np.isfinite(yhat)
    y_m, yhat_m = y[mask], yhat[mask]
    err = yhat_m - y_m
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    ss_tot = float(np.sum((y_m - float(np.mean(y_m))) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - np.sum(err ** 2) / ss_tot)

    # ---- GVW-R² ----
    gvw_mean_r2 = _weighted_r2(y, yhat, genes_val, raw_weights)

    # ---- Per-gene and per-sample Spearman R² ----
    df_pred = pd.DataFrame({
        "truth": y,
        "prediction": yhat,
        "gene": genes_val.astype(str),
        "sample": samples_val.astype(str),
    })

    def _group_spearman_r2(df: pd.DataFrame, group_col: str) -> np.ndarray:
        out = []
        for _, sub in df.groupby(group_col):
            ys = sub["truth"].to_numpy(dtype=np.float64)
            yhs = sub["prediction"].to_numpy(dtype=np.float64)
            m = np.isfinite(ys) & np.isfinite(yhs)
            if int(m.sum()) < 3:
                continue
            out.append(_spearman_r2(ys[m], yhs[m]))
        arr = np.asarray(out, dtype=np.float64)
        return arr[np.isfinite(arr)]

    gene_r2_arr = _group_spearman_r2(df_pred, "gene")
    sample_r2_arr = _group_spearman_r2(df_pred, "sample")

    metrics = {
        "fold": fold_num,
        "tumor": {
            "n": int(mask.sum()),
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "gvw_mean_r2": gvw_mean_r2,
            "gene_r2_mean": float(np.nanmean(gene_r2_arr)) if gene_r2_arr.size else float("nan"),
            "gene_r2_median": float(np.nanmedian(gene_r2_arr)) if gene_r2_arr.size else float("nan"),
            "gene_r2_n": int(gene_r2_arr.size),
            "sample_r2_mean": float(np.nanmean(sample_r2_arr)) if sample_r2_arr.size else float("nan"),
            "sample_r2_median": float(np.nanmedian(sample_r2_arr)) if sample_r2_arr.size else float("nan"),
            "sample_r2_n": int(sample_r2_arr.size),
        },
    }

    # ---- Predictions DataFrame ----
    df_out = pd.DataFrame({
        "truth": y,
        "prediction": yhat,
        "sample": samples_val.astype(str),
    }, index=genes_val.astype(str))

    return metrics, df_out


# ---------------------------------------------------------------------------
# Full 5-fold CV run
# ---------------------------------------------------------------------------

def run_cv(
    xgb_params: Dict[str, Any],
    n_estimators: int = 1000,
    out_dir: Optional[str] = None,
    tsv_path: Optional[str] = None,
    zarr_base: Optional[str] = None,
    folds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run 5-fold CV, save predictions + metrics, return summary.

    Output structure:
        out_dir/
            CV-fold_1/ validation-fold_1_metrics.json
                       validation-fold_1_predictions_tumor.tsv
                       model.ubj
            ...
            cv_objective.json
    """
    tsv_path = tsv_path or default_splits_tsv()
    zarr_base = zarr_base or default_zarr_base()
    if out_dir is None:
        out_dir = str(PROJECT_ROOT / "analysis" / "XGBoost" / "results_cv")
    os.makedirs(out_dir, exist_ok=True)

    folds = folds or [1, 2, 3, 4, 5]
    all_metrics: List[Dict[str, Any]] = []

    for fold_num in folds:
        print(f"\n{'='*60}")
        print(f"  Fold {fold_num}")
        print(f"{'='*60}")

        model, fold_info = train_fold(
            fold_num, xgb_params, n_estimators, tsv_path, zarr_base,
        )
        metrics, df_pred = evaluate_fold(model, fold_info)

        # ---- Save everything ----
        fold_dir = os.path.join(out_dir, f"CV-fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, f"validation-fold_{fold_num}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        df_pred.to_csv(
            os.path.join(fold_dir, f"validation-fold_{fold_num}_predictions_tumor.tsv"),
            sep="\t",
        )

        model.save_model(os.path.join(fold_dir, "model.ubj"))

        all_metrics.append(metrics)

        t = metrics["tumor"]
        print(
            f"  RMSE: {t['rmse']:.4f}  "
            f"R²: {t['r2']:.4f}  "
            f"GVW-R²: {t['gvw_mean_r2']:.4f}  "
            f"gene_r2: {t['gene_r2_mean']:.4f}  "
            f"sample_r2: {t['sample_r2_mean']:.4f}"
        )

    # ---- Aggregate ----
    summary = _aggregate_cv_metrics(all_metrics)
    with open(os.path.join(out_dir, "cv_objective.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  CV Summary")
    print(f"{'='*60}")
    print(f"  Mean RMSE:        {summary['tumor_rmse_mean']:.4f} ± {summary['tumor_rmse_std']:.4f}")
    print(f"  Mean R²:          {summary['tumor_r2_mean']:.4f} ± {summary['tumor_r2_std']:.4f}")
    print(f"  Mean GVW-R²:      {summary['tumor_gvw_mean_r2_mean']:.4f} ± {summary['tumor_gvw_mean_r2_std']:.4f}")
    print(f"  Mean gene_r2:     {summary['tumor_gene_r2_mean_mean']:.4f}")
    print(f"  Mean sample_r2:   {summary['tumor_sample_r2_mean_mean']:.4f}")

    return summary


def _nanmean(xs: List[float]) -> float:
    vals = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _nanstd(xs: List[float]) -> float:
    vals = [float(x) for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return float(math.sqrt(var))


def _aggregate_cv_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-fold metrics into mean ± std."""
    gvw_r2 = [m["tumor"]["gvw_mean_r2"] for m in all_metrics]
    r2 = [m["tumor"]["r2"] for m in all_metrics]
    rmse = [m["tumor"]["rmse"] for m in all_metrics]
    mae = [m["tumor"]["mae"] for m in all_metrics]
    gene_r2 = [m["tumor"]["gene_r2_mean"] for m in all_metrics]
    sample_r2 = [m["tumor"]["sample_r2_mean"] for m in all_metrics]

    return {
        "folds_with_metrics": len(all_metrics),
        "tumor_rmse_mean": _nanmean(rmse),
        "tumor_rmse_std": _nanstd(rmse),
        "tumor_gvw_mean_r2_mean": _nanmean(gvw_r2),
        "tumor_gvw_mean_r2_std": _nanstd(gvw_r2),
        "tumor_r2_mean": _nanmean(r2),
        "tumor_r2_std": _nanstd(r2),
        "tumor_mae_mean": _nanmean(mae),
        "tumor_mae_std": _nanstd(mae),
        "tumor_gene_r2_mean_mean": _nanmean(gene_r2),
        "tumor_gene_r2_mean_std": _nanstd(gene_r2),
        "tumor_sample_r2_mean_mean": _nanmean(sample_r2),
        "tumor_sample_r2_mean_std": _nanstd(sample_r2),
    }
