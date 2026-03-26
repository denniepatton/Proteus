"""Statistical metrics for Proteus fold evaluation.

Functions extracted from ``_save_fold_predictions`` so they can be
reused across fold-level, CV-aggregation, and tuning code paths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Primitive correlations
# ---------------------------------------------------------------------------

def rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie rank transform (delegates to pandas for tie handling)."""
    return pd.Series(x).rank(method='average').to_numpy(dtype=np.float64)


def pearson_r(y: np.ndarray, yhat: np.ndarray) -> float:
    """Numerically-safe Pearson correlation (finite inputs only)."""
    y_c = y - float(np.mean(y))
    yhat_c = yhat - float(np.mean(yhat))
    denom = float(np.sqrt(np.sum(y_c * y_c) * np.sum(yhat_c * yhat_c)))
    if denom <= 1e-12:
        return float('nan')
    return float(np.clip(np.sum(y_c * yhat_c) / denom, -1.0, 1.0))


def spearman_r(y: np.ndarray, yhat: np.ndarray) -> float:
    """Spearman rank correlation."""
    return pearson_r(rankdata(y), rankdata(yhat))


def spearman_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Squared Spearman rank correlation."""
    r = spearman_r(y, yhat)
    return r * r if np.isfinite(r) else float('nan')


# ---------------------------------------------------------------------------
# Group-level correlations
# ---------------------------------------------------------------------------

def group_spearman_r2(
    df: pd.DataFrame,
    group_col: str,
    pred_col: str = 'prediction_mean',
) -> np.ndarray:
    """Per-group Spearman r² (groups with < 3 valid obs are skipped)."""
    out: List[float] = []
    for _, sub in df.groupby(group_col):
        y = sub['truth'].to_numpy(dtype=np.float64)
        yhat = sub[pred_col].to_numpy(dtype=np.float64)
        mask = np.isfinite(y) & np.isfinite(yhat)
        if int(mask.sum()) < 3:
            continue
        out.append(spearman_r2(y[mask], yhat[mask]))
    arr = np.asarray(out, dtype=np.float64)
    return arr[np.isfinite(arr)]


def group_pearson_r_named(
    df: pd.DataFrame,
    group_col: str,
    pred_col: str,
) -> Tuple[np.ndarray, List[str]]:
    """Per-group Pearson r with associated group names.

    Returns ``(r_values, names)`` arrays.  Groups with < 3 valid
    observations or non-finite r are excluded.
    """
    names: List[str] = []
    vals: List[float] = []
    for gname, sub in df.groupby(group_col):
        y = sub['truth'].to_numpy(dtype=np.float64)
        yhat = sub[pred_col].to_numpy(dtype=np.float64)
        mask = np.isfinite(y) & np.isfinite(yhat)
        if int(mask.sum()) < 3:
            continue
        r = pearson_r(y[mask], yhat[mask])
        if np.isfinite(r):
            names.append(str(gname))
            vals.append(r)
    return np.asarray(vals, dtype=np.float64), names


# ---------------------------------------------------------------------------
# Coverage & regression summary
# ---------------------------------------------------------------------------

def coverage(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of valid observations inside the quantile interval [lo, hi]."""
    mask = np.isfinite(truth)
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean((truth[mask] >= lo[mask]) & (truth[mask] <= hi[mask])))


def compute_regression_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics (MAE, MSE, RMSE, R²) with NaN-safe masking."""
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(truth) & np.isfinite(pred)
    if mask.sum() == 0:
        return {
            'n': 0,
            'mae': float('nan'),
            'mse': float('nan'),
            'rmse': float('nan'),
            'r2': float('nan'),
        }
    y = truth[mask]
    yhat = pred[mask]
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(np.square(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum(np.square(y - float(np.mean(y)))))
    r2 = float('nan') if denom <= 1e-12 else float(1.0 - (np.sum(np.square(err)) / denom))
    return {
        'n': int(mask.sum()),
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
    }


# ---------------------------------------------------------------------------
# Gene-variance–weighted Pearson r (GVW-r)
# ---------------------------------------------------------------------------

def gvw_r(
    truth_vec: np.ndarray,
    pred_vec: np.ndarray,
    genes_vec: List[str],
    external_var: Optional[Dict[str, float]] = None,
) -> float:
    """Per-gene Pearson r, variance-weighted average.

    If *external_var* (gene-name → variance dict) is provided, those
    variances are used as weights; otherwise variance is computed from
    the truth values for each gene within this group.
    """
    df_tmp = pd.DataFrame({
        'truth': truth_vec,
        'pred': pred_vec,
        'gene': genes_vec,
    })
    gene_r_vals: List[float] = []
    gene_var_vals: List[float] = []
    for gname, sub in df_tmp.groupby('gene'):
        y = sub['truth'].to_numpy(dtype=np.float64)
        yhat = sub['pred'].to_numpy(dtype=np.float64)
        m = np.isfinite(y) & np.isfinite(yhat)
        if int(m.sum()) < 3:
            continue
        r = pearson_r(y[m], yhat[m])
        if external_var is not None:
            v = external_var.get(str(gname), 0.0)
        else:
            v = float(np.var(y[m]))
        if np.isfinite(r) and v > 0:
            gene_r_vals.append(r)
            gene_var_vals.append(v)
    if not gene_r_vals:
        return float('nan')
    r_arr = np.asarray(gene_r_vals, dtype=np.float64)
    v_arr = np.asarray(gene_var_vals, dtype=np.float64)
    return float(np.average(r_arr, weights=v_arr))


# ---------------------------------------------------------------------------
# Scatter annotation stats (used by fold-level plots)
# ---------------------------------------------------------------------------

def scatter_stats(
    truth: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    """Compute global regression stats for scatter-plot annotations."""
    mask = np.isfinite(truth) & np.isfinite(pred)
    y, yhat = truth[mask], pred[mask]
    if y.size < 3:
        return {}
    r = pearson_r(y, yhat)
    sp = spearman_r(y, yhat)
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
    return {'R²': r2, 'Pearson r': r, 'Spearman r': sp, 'MAE': mae}
