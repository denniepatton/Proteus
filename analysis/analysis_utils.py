"""
analysis_utils.py
Proteus analysis framework — data loading, stats, and metrics.

All public functions work on *wide* DataFrames (genes × samples) unless
explicitly noted.  Long-format prediction TSVs are pivoted to wide on load.
"""

from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Reference / label loaders
# ---------------------------------------------------------------------------

def load_truth(cohort: str) -> pd.DataFrame:
    """Load truth expression: labels/<cohort>_Log2CPMplus1.tsv → wide DataFrame.

    Returns DataFrame with gene names as index, sample names as columns.
    """
    path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df


def load_subtypes(cohort: str) -> pd.DataFrame:
    """Load labels/<cohort>_TFX_Subtype.tsv → DataFrame indexed by sample.

    Columns: tfx (float), subtype (str), plus any extras present in the file.
    """
    path = PROJECT_ROOT / "labels" / f"{cohort}_TFX_Subtype.tsv"
    df = pd.read_csv(path, sep="\t")
    df = df.set_index("sample")
    df["tfx"] = pd.to_numeric(df["tfx"], errors="coerce")
    return df


def load_metadata(name: str) -> pd.DataFrame:
    """Load reference/<name>.tsv → DataFrame indexed by first column."""
    path = PROJECT_ROOT / "reference" / f"{name}.tsv"
    df = pd.read_csv(path, sep="\t", index_col=0)
    return df


def load_gene_list(name: str) -> List[str]:
    """Load reference/<name>.txt — one gene per line."""
    path = PROJECT_ROOT / "reference" / f"{name}.txt"
    if not path.exists():
        # Try .tsv with first column
        path = PROJECT_ROOT / "reference" / f"{name}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Gene list not found: {name}")
        df = pd.read_csv(path, sep="\t", header=None)
        return df.iloc[:, 0].astype(str).tolist()
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_palette() -> Dict[str, str]:
    """Load reference/palette.tsv → {subtype: hex_color}."""
    path = PROJECT_ROOT / "reference" / "palette.tsv"
    df = pd.read_csv(path, sep="\t", header=None)
    return dict(zip(df[0], df[1]))


def load_fold_splits() -> pd.DataFrame:
    """Load data/BTS_sample-splits.tsv → DataFrame with Cohort, Sample, Fold."""
    path = PROJECT_ROOT / "data" / "BTS_sample-splits.tsv"
    return pd.read_csv(path, sep="\t")


# ---------------------------------------------------------------------------
# Gene-list registry (commonly used sets)
# ---------------------------------------------------------------------------

# Lazy-loaded cache
_GENE_LISTS: Dict[str, List[str]] = {}


def get_gene_list(name: str) -> List[str]:
    """Get a named gene list, loading from disk on first access."""
    if name == "all":
        return []  # sentinel: means no filtering
    if name not in _GENE_LISTS:
        _GENE_LISTS[name] = load_gene_list(name)
    return _GENE_LISTS[name]


STANDARD_GENE_SETS: Dict[str, str] = {
    "PC-Phenotyping": "PC_PhenotypingGenes",
    "Targets_PC": "Targets_PC",
    "Targets_SCLC": "Targets_SCLC",
    "Targets_BC": "Targets_BC",
    "PAM50": "PAM50",
    "CCP31": "CCP31",
    "DruggableGenome": "DruggableGenome_Tier1",
    "CancerSurfaceome": "CancerSurfaceome",
    "GavinsSet": "GavinsSet",
    "Housekeeping": "Housekeeping_HumanProteinAtlas",
    "NonHousekeeping": "NonHousekeeping_HumanProteinAtlas",
}

# Gene set breaks: row indices for visual dividers in heatmaps.
# PC-Phenotyping has 3 sub-groups: NEURO I (0-7), NEURO II (8-13), AR (14-21).
GENE_SET_BREAKS: Dict[str, List[int]] = {
    "PC-Phenotyping": [8, 14],
}


# ---------------------------------------------------------------------------
# Prediction loading — long → wide
# ---------------------------------------------------------------------------

def _long_to_wide(df_long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot a long-format predictions DF to wide (genes × samples).

    Expects df_long to have gene names as index, a 'sample' column,
    and ``value_col`` column with numeric values.
    """
    df = df_long[[value_col, "sample"]].copy()
    df.index.name = "gene"
    df = df.reset_index()
    wide = df.pivot(index="gene", columns="sample", values=value_col)
    return wide


def load_cv_predictions_proteus(
    trial_dir: str,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load Proteus CV fold predictions and merge.

    Returns (pred_wide, truth_wide, ci_lo_wide, ci_hi_wide).
    pred/truth are genes × samples.  ci_lo/ci_hi may be None if unavailable.
    """
    pred_parts, truth_parts = [], []
    ci_lo_parts, ci_hi_parts = [], []

    for fold in folds:
        fold_dir = os.path.join(trial_dir, f"CV-fold_{fold}")
        path = os.path.join(fold_dir, f"validation-fold_{fold}_predictions_tumor.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Proteus CV prediction: {path}")

        df = pd.read_csv(path, sep="\t", index_col=0)
        pred_parts.append(_long_to_wide(df, "prediction_mean"))
        truth_parts.append(_long_to_wide(df, "truth"))

        if "ci_q5" in df.columns and "ci_q95" in df.columns:
            ci_lo_parts.append(_long_to_wide(df, "ci_q5"))
            ci_hi_parts.append(_long_to_wide(df, "ci_q95"))

    pred_wide = _merge_wide_folds(pred_parts)
    truth_wide = _merge_wide_folds(truth_parts)

    ci_lo_wide = _merge_wide_folds(ci_lo_parts) if ci_lo_parts else None
    ci_hi_wide = _merge_wide_folds(ci_hi_parts) if ci_hi_parts else None

    return pred_wide, truth_wide, ci_lo_wide, ci_hi_wide


def load_cv_predictions_xgboost(
    results_dir: str,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load XGBoost CV fold predictions and merge.

    Returns (pred_wide, truth_wide).
    """
    pred_parts, truth_parts = [], []

    for fold in folds:
        fold_dir = os.path.join(results_dir, f"CV-fold_{fold}")
        path = os.path.join(fold_dir, f"validation-fold_{fold}_predictions_tumor.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing XGBoost CV prediction: {path}")

        df = pd.read_csv(path, sep="\t", index_col=0)
        pred_parts.append(_long_to_wide(df, "prediction"))
        truth_parts.append(_long_to_wide(df, "truth"))

    return _merge_wide_folds(pred_parts), _merge_wide_folds(truth_parts)


def load_holdout_predictions_proteus(
    path: str,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load a single Proteus holdout/inference predictions_tumor.tsv.

    Returns (pred_wide, ci_lo_wide, ci_hi_wide).
    ci_lo/ci_hi are None if columns not present.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    pred_wide = _long_to_wide(df, "prediction_mean")

    ci_lo = _long_to_wide(df, "ci_q5") if "ci_q5" in df.columns else None
    ci_hi = _long_to_wide(df, "ci_q95") if "ci_q95" in df.columns else None

    return pred_wide, ci_lo, ci_hi


def load_holdout_predictions_xgboost(path: str) -> pd.DataFrame:
    """Load a single XGBoost holdout predictions_tumor.tsv → pred_wide."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    return _long_to_wide(df, "prediction")


def _merge_wide_folds(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge a list of wide DataFrames (genes × samples) across CV folds.

    Each fold contributes distinct samples, so we outer-join on gene index
    and concat on the sample axis.
    """
    if len(parts) == 1:
        return parts[0]

    # Concat along columns (samples). Genes union across all folds.
    merged = pd.concat(parts, axis=1)
    # If a gene appears in multiple folds for the same sample, keep first.
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def pearson_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Pearson r with bootstrap CI.

    Returns (r, p_value, ci_lo, ci_hi).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    r, p = pearsonr(x, y)

    # Bootstrap CI
    data = (x, y)

    def _stat(x_arr, y_arr, axis=None):
        # bootstrap passes 1-D resampled arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pearsonr(x_arr, y_arr)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bootstrap(
            data, _stat, n_resamples=n_boot,
            confidence_level=confidence, method="percentile",
            random_state=np.random.default_rng(rng_seed), paired=True,
        )
    return (r, p, float(res.confidence_interval.low), float(res.confidence_interval.high))


def spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Spearman rho with bootstrap CI.

    Returns (rho, p_value, ci_lo, ci_hi).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.nan)

    rho, p = spearmanr(x, y)

    data = (x, y)

    def _stat(x_arr, y_arr, axis=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return spearmanr(x_arr, y_arr)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = bootstrap(
            data, _stat, n_resamples=n_boot,
            confidence_level=confidence, method="percentile",
            random_state=np.random.default_rng(rng_seed), paired=True,
        )
    return (rho, p, float(res.confidence_interval.low), float(res.confidence_interval.high))


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(pearsonr(x, y)[0])


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(x, y)[0])


# --------------- Parallelised per-gene / per-sample metrics ----------------

def _gene_metrics_single(args: Tuple[str, np.ndarray, np.ndarray]) -> Dict[str, Any]:
    """Compute metrics for a single gene (designed for ProcessPoolExecutor)."""
    gene, truth, pred = args
    mask = np.isfinite(truth) & np.isfinite(pred)
    t, p = truth[mask], pred[mask]
    n = int(mask.sum())
    if n < 3:
        return {"gene": gene, "n": n, "pearson_r": np.nan, "spearman_rho": np.nan,
                "rmse": np.nan, "mae": np.nan, "truth_var": np.nan}

    err = p - t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = float(pearsonr(t, p)[0])
        sr = float(spearmanr(t, p)[0])
    return {
        "gene": gene,
        "n": n,
        "pearson_r": pr,
        "spearman_rho": sr,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "truth_var": float(np.var(t)),
    }


def gene_level_metrics(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Per-gene Pearson / Spearman / RMSE / MAE across samples.

    Uses multiprocessing for speed.
    Returns DataFrame indexed by gene name.
    """
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    args_list = [
        (gene, tw.loc[gene].to_numpy(dtype=np.float64),
         pw.loc[gene].to_numpy(dtype=np.float64))
        for gene in common_genes
    ]

    if n_workers is None:
        n_workers = _default_workers()

    if n_workers > 1 and len(args_list) > 100:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_gene_metrics_single, args_list, chunksize=500))
    else:
        results = [_gene_metrics_single(a) for a in args_list]

    df = pd.DataFrame(results).set_index("gene")
    return df


def _sample_metrics_single(args: Tuple[str, np.ndarray, np.ndarray]) -> Dict[str, Any]:
    """Compute metrics for a single sample across genes."""
    sample, truth, pred = args
    mask = np.isfinite(truth) & np.isfinite(pred)
    t, p = truth[mask], pred[mask]
    n = int(mask.sum())
    if n < 3:
        return {"sample": sample, "n": n, "pearson_r": np.nan, "spearman_rho": np.nan,
                "rmse": np.nan, "mae": np.nan}

    err = p - t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = float(pearsonr(t, p)[0])
        sr = float(spearmanr(t, p)[0])
    return {
        "sample": sample,
        "n": n,
        "pearson_r": pr,
        "spearman_rho": sr,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }


def sample_level_metrics(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Per-sample Pearson / Spearman / RMSE / MAE across genes.

    Uses multiprocessing for speed.
    Returns DataFrame indexed by sample name.
    """
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    args_list = [
        (sample, tw[sample].to_numpy(dtype=np.float64),
         pw[sample].to_numpy(dtype=np.float64))
        for sample in common_samples
    ]

    if n_workers is None:
        n_workers = _default_workers()

    if n_workers > 1 and len(args_list) > 10:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_sample_metrics_single, args_list, chunksize=10))
    else:
        results = [_sample_metrics_single(a) for a in args_list]

    df = pd.DataFrame(results).set_index("sample")
    return df


# --------------- GVW-r (gene-variance-weighted correlation) ----------------

def compute_gene_variance(truth_wide: pd.DataFrame) -> Dict[str, float]:
    """Per-gene variance from truth across samples → {gene: var}."""
    return truth_wide.var(axis=1, ddof=1).to_dict()


def gvw_r(
    truth_flat: np.ndarray,
    pred_flat: np.ndarray,
    genes_flat: np.ndarray,
    var_map: Dict[str, float],
) -> float:
    """Gene-variance-weighted Pearson r.

    Each observation (gene×sample) is weighted by the variance of that gene
    across truth samples.  Returns the weighted correlation.
    """
    mask = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    t = truth_flat[mask]
    p = pred_flat[mask]
    g = genes_flat[mask]
    w = np.array([var_map.get(str(gene), 0.0) for gene in g], dtype=np.float64)
    np.nan_to_num(w, nan=0.0, copy=False)

    total_w = w.sum()
    if total_w < 1e-12 or len(t) < 3:
        return np.nan

    mean_t = np.average(t, weights=w)
    mean_p = np.average(p, weights=w)
    cov_tp = np.sum(w * (t - mean_t) * (p - mean_p)) / total_w
    var_t = np.sum(w * (t - mean_t) ** 2) / total_w
    var_p = np.sum(w * (p - mean_p) ** 2) / total_w

    denom = np.sqrt(var_t * var_p)
    if denom < 1e-12:
        return np.nan
    return float(cov_tp / denom)


def gvw_rho(
    truth_flat: np.ndarray,
    pred_flat: np.ndarray,
    genes_flat: np.ndarray,
    var_map: Dict[str, float],
) -> float:
    """Gene-variance-weighted Spearman rho.

    Ranks truth and pred separately, then computes GVW-weighted Pearson on
    the ranks.
    """
    from scipy.stats import rankdata

    mask = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    t = truth_flat[mask]
    p = pred_flat[mask]
    g = genes_flat[mask]

    # Replace values with ranks
    t_ranked = rankdata(t).astype(np.float64)
    p_ranked = rankdata(p).astype(np.float64)

    w = np.array([var_map.get(str(gene), 0.0) for gene in g], dtype=np.float64)
    np.nan_to_num(w, nan=0.0, copy=False)
    total_w = w.sum()
    if total_w < 1e-12 or len(t) < 3:
        return np.nan

    mean_t = np.average(t_ranked, weights=w)
    mean_p = np.average(p_ranked, weights=w)
    cov_tp = np.sum(w * (t_ranked - mean_t) * (p_ranked - mean_p)) / total_w
    var_t = np.sum(w * (t_ranked - mean_t) ** 2) / total_w
    var_p = np.sum(w * (p_ranked - mean_p) ** 2) / total_w

    denom = np.sqrt(var_t * var_p)
    if denom < 1e-12:
        return np.nan
    return float(cov_tp / denom)


def _flatten_wide_pair(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten two wide DataFrames into aligned 1-D arrays + gene labels.

    Returns (truth_flat, pred_flat, genes_flat).
    """
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    n_genes = len(common_genes)
    n_samples = len(common_samples)
    genes_flat = np.repeat(np.array(common_genes, dtype=object), n_samples)
    truth_flat = tw.values.flatten().astype(np.float64)
    pred_flat = pw.values.flatten().astype(np.float64)
    return truth_flat, pred_flat, genes_flat


# --------------- Variance-group breakdown ----------------------------------

VARIANCE_BINS = {"low": (0.0, 1.0), "medium": (1.0, 3.0), "high": (3.0, np.inf)}


def variance_group_metrics(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    var_map: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute per-gene Pearson/Spearman broken down by variance group.

    Returns summary DataFrame with rows = variance groups.
    """
    if var_map is None:
        var_map = compute_gene_variance(truth_wide)

    gene_met = gene_level_metrics(truth_wide, pred_wide)
    gene_met["variance"] = gene_met.index.map(lambda g: var_map.get(g, 0.0))

    rows = []
    for grp, (lo, hi) in VARIANCE_BINS.items():
        sub = gene_met[(gene_met["variance"] >= lo) & (gene_met["variance"] < hi)]
        rows.append({
            "group": grp,
            "n_genes": len(sub),
            "mean_pearson_r": sub["pearson_r"].mean(),
            "median_pearson_r": sub["pearson_r"].median(),
            "mean_spearman_rho": sub["spearman_rho"].mean(),
            "median_spearman_rho": sub["spearman_rho"].median(),
            "mean_rmse": sub["rmse"].mean(),
        })
    rows.append({
        "group": "all",
        "n_genes": len(gene_met),
        "mean_pearson_r": gene_met["pearson_r"].mean(),
        "median_pearson_r": gene_met["pearson_r"].median(),
        "mean_spearman_rho": gene_met["spearman_rho"].mean(),
        "median_spearman_rho": gene_met["spearman_rho"].median(),
        "mean_rmse": gene_met["rmse"].mean(),
    })
    return pd.DataFrame(rows)


# --------------- Cohort summary table (the big printable table) ------------

def cohort_summary_table(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    label: str = "",
    n_workers: Optional[int] = None,
    var_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute the full cohort summary metrics dict.

    Includes: GVW-r, GVW-rho, mean/median gene-level r/rho,
    mean/median transcriptome-level r/rho.

    Parameters
    ----------
    var_map : optional pre-computed gene variance dict.  When supplied the
        GVW metrics and variance-group breakdowns use these values instead
        of recomputing from *truth_wide* (which may only be a subset of
        the full cohort).  Callers should obtain this from the full
        canonical RNA-Seq truth labels.
    """
    if var_map is None:
        var_map = compute_gene_variance(truth_wide)
    t_flat, p_flat, g_flat = _flatten_wide_pair(truth_wide, pred_wide)

    gvw_r_val = gvw_r(t_flat, p_flat, g_flat, var_map)
    gvw_rho_val = gvw_rho(t_flat, p_flat, g_flat, var_map)

    gene_met = gene_level_metrics(truth_wide, pred_wide, n_workers=n_workers)
    sample_met = sample_level_metrics(truth_wide, pred_wide, n_workers=n_workers)

    return {
        "label": label,
        "n_genes": len(gene_met),
        "n_samples": len(sample_met),
        "gvw_r": gvw_r_val,
        "gvw_rho": gvw_rho_val,
        "mean_gene_r": gene_met["pearson_r"].mean(),
        "median_gene_r": gene_met["pearson_r"].median(),
        "mean_gene_rho": gene_met["spearman_rho"].mean(),
        "median_gene_rho": gene_met["spearman_rho"].median(),
        "mean_transcriptome_r": sample_met["pearson_r"].mean(),
        "median_transcriptome_r": sample_met["pearson_r"].median(),
        "mean_transcriptome_rho": sample_met["spearman_rho"].mean(),
        "median_transcriptome_rho": sample_met["spearman_rho"].median(),
        "mean_gene_rmse": gene_met["rmse"].mean(),
        "mean_sample_rmse": sample_met["rmse"].mean(),
        "_gene_metrics": gene_met,
        "_sample_metrics": sample_met,
    }


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    """Pretty-print a list of cohort_summary_table() results."""
    header = (
        f"{'Label':<20s} {'n_g':>6s} {'n_s':>5s} "
        f"{'GVW-r':>7s} {'GVW-ρ':>7s} "
        f"{'μ_g_r':>7s} {'M_g_r':>7s} {'μ_g_ρ':>7s} {'M_g_ρ':>7s} "
        f"{'μ_t_r':>7s} {'M_t_r':>7s} {'μ_t_ρ':>7s} {'M_t_ρ':>7s}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['label']:<20s} {s['n_genes']:>6d} {s['n_samples']:>5d} "
            f"{s['gvw_r']:>7.4f} {s['gvw_rho']:>7.4f} "
            f"{s['mean_gene_r']:>7.4f} {s['median_gene_r']:>7.4f} "
            f"{s['mean_gene_rho']:>7.4f} {s['median_gene_rho']:>7.4f} "
            f"{s['mean_transcriptome_r']:>7.4f} {s['median_transcriptome_r']:>7.4f} "
            f"{s['mean_transcriptome_rho']:>7.4f} {s['median_transcriptome_rho']:>7.4f}"
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_pval(pval: float) -> str:
    if np.isnan(pval):
        return "p = NaN"
    if pval < 1e-10:
        return "p < 1e-10"
    exponent = int(np.floor(np.log10(pval)))
    mantissa = pval / 10 ** exponent
    if exponent < -2 or exponent > 2:
        return f"p = {mantissa:.2f}e{exponent}"
    return f"p = {pval:.3g}"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _default_workers() -> int:
    """Return a sensible default worker count."""
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns path."""
    os.makedirs(path, exist_ok=True)
    return path
