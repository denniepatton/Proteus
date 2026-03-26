"""
analysis_utils.py
Proteus analysis framework — data loading, stats, and metrics.

All public functions work on *wide* DataFrames (genes × samples) unless
explicitly noted.  Long-format prediction TSVs are pivoted to wide on load.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

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
    df = pd.read_csv(path, sep="\t", header=None, encoding="mac_roman").dropna()
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


def gene_list_available(name: str) -> bool:
    """Check whether a named gene list file exists on disk."""
    if name == "all":
        return True
    txt = PROJECT_ROOT / "reference" / f"{name}.txt"
    tsv = PROJECT_ROOT / "reference" / f"{name}.tsv"
    return txt.exists() or tsv.exists()


def get_gene_list(name: str) -> List[str]:
    """Get a named gene list, loading from disk on first access."""
    if name == "all":
        return []  # sentinel: means no filtering
    if name not in _GENE_LISTS:
        _GENE_LISTS[name] = load_gene_list(name)
    return _GENE_LISTS[name]


STANDARD_GENE_SETS: Dict[str, str] = {
    "PC-Phenotyping": "PC_PhenotypingGenes",
    "Targets_CRPC": "Targets_CRPC",
    "Targets_SCLC": "Targets_SCLC",
    "Targets_BLCA": "Targets_BLCA",
    "PAM50": "PAM50",
    "CCP31": "CCP31",
    "DruggableGenome": "DruggableGenome_Tier1",
    "CancerSurfaceome": "CancerSurfaceome",
    "Housekeeping": "Housekeeping_HumanProteinAtlas",
    "NonHousekeeping": "NonHousekeeping_HumanProteinAtlas",
}

# Gene set breaks: row indices for visual dividers in heatmaps.
# PC-Phenotyping has 3 sub-groups: NEURO I (0-7), NEURO II (8-13), AR (14-21).
GENE_SET_BREAKS: Dict[str, List[int]] = {
    "PC-Phenotyping": [8, 14],
}


# ---------------------------------------------------------------------------
# Sample name alignment helpers
# ---------------------------------------------------------------------------


def strip_cohort_prefix(columns: pd.Index, cohort: str) -> pd.Index:
    """Strip '{cohort}_' prefix from column names when present.

    Truth label files use prefixed sample names (e.g. 'BLCA-RA_15-109')
    while prediction files use bare IDs (e.g. '15-109').  Stripping the
    prefix allows ``pd.Index.intersection`` to find the overlap.
    """
    prefix = f"{cohort}_"
    plen = len(prefix)
    return pd.Index(
        [col[plen:] if col.startswith(prefix) else col for col in columns]
    )


def clean_epicseq_sample_names(columns: pd.Index) -> pd.Index:
    """Normalise EPIC-Seq sample IDs by stripping '_<digits>.cram' suffix.

    EPIC-Seq pivot columns look like '15-109_1107.cram'; Proteus/XGBoost
    predictions (and stripped truth columns) use '15-109'.
    """
    return pd.Index([re.sub(r'_\d+\.cram$', '', str(c)) for c in columns])


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


def _extract_donor_id(pht_sample: str) -> Optional[str]:
    """Extract the HD-L donor ID (e.g. 'BP1124576_S78') from a PHT sample name.

    PHT sample names look like 'PHT-Fold-1_LuCaP_96-BP1124576_S78'.
    The donor ID is the portion after the last hyphen-separated PDX name,
    matching the pattern ``BP\\d+_S\\d+``.
    """
    m = re.search(r"(BP\d+_S\d+)$", pht_sample)
    return m.group(1) if m else None


def load_cv_predictions_background(
    trial_dir: str,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load background (HD-L) predictions from CV folds, averaged per donor.

    Each fold's ``predictions_background.tsv`` contains the model's
    background-head predictions for every PHT admixture sample.  Multiple
    PHT admixtures share the same HD-L donor within a fold, so we
    average prediction_mean across all PHT samples that share a donor
    to produce a single predicted profile per donor.

    Returns (pred_wide, truth_wide) — both genes × HD-L donors.
    """
    pred_parts: List[pd.DataFrame] = []
    truth_parts: List[pd.DataFrame] = []

    for fold in folds:
        fold_dir = os.path.join(trial_dir, f"CV-fold_{fold}")
        path = os.path.join(fold_dir, f"validation-fold_{fold}_predictions_background.tsv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path, sep="\t", index_col=0)
        df["donor"] = df["sample"].map(_extract_donor_id)
        df = df.dropna(subset=["donor"])

        # Average across PHT admixtures sharing the same donor
        df.index.name = "gene"
        df = df.reset_index()
        pred_avg = df.groupby(["gene", "donor"])["prediction_mean"].mean().reset_index()
        pred_avg = pred_avg.pivot(index="gene", columns="donor", values="prediction_mean")

        truth_avg = df.groupby(["gene", "donor"])["truth"].mean().reset_index()
        truth_avg = truth_avg.pivot(index="gene", columns="donor", values="truth")

        pred_parts.append(pred_avg)
        truth_parts.append(truth_avg)

    if not pred_parts:
        return pd.DataFrame(), pd.DataFrame()

    pred_wide = _merge_wide_folds(pred_parts)
    truth_wide = _merge_wide_folds(truth_parts)

    return pred_wide, truth_wide


def load_cv_predictions_proteus_full(
    trial_dir: str,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
) -> Dict[str, Any]:
    """Load Proteus CV fold predictions with full MDN parameter columns.

    Unlike ``load_cv_predictions_proteus`` which returns only pred/truth/CIs,
    this function loads all MDN output columns, tracks per-fold data
    separately, and maps samples to fold indices.

    Needed for the reliability model (full MDN feature computation).

    Returns dict with keys:
        ``pred_wide``         — genes × samples merged point predictions
        ``truth_wide``        — genes × samples merged truth
        ``ci_lo_wide``        — genes × samples merged lower quantile (or None)
        ``ci_hi_wide``        — genes × samples merged upper quantile (or None)
        ``p0_wide``           — genes × samples merged pi_delta (or None)
        ``pi_gamma_wide``     — genes × samples merged pi_gamma (or None)
        ``pi_normal_wide``    — genes × samples merged pi_normal (or None)
        ``gamma_beta_wide``   — genes × samples merged gamma_beta (or None)
        ``normal_mean_wide``  — genes × samples merged normal_mean (or None)
        ``normal_stdev_wide`` — genes × samples merged normal_stdev (or None)
        ``fold_data``         — {fold_int: dict of per-fold wide DFs}
        ``sample_fold``       — {sample_name: fold_int}
    """
    # Column name in TSV → key in result dict
    _OPTIONAL_COLS = {
        "ci_q5": "ci_lo_wide",
        "ci_q95": "ci_hi_wide",
        "pi_delta": "p0_wide",
        "pi_gamma": "pi_gamma_wide",
        "pi_normal": "pi_normal_wide",
        "gamma_beta": "gamma_beta_wide",
        "normal_mean": "normal_mean_wide",
        "normal_stdev": "normal_stdev_wide",
    }

    pred_parts: List[pd.DataFrame] = []
    truth_parts: List[pd.DataFrame] = []
    optional_parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in _OPTIONAL_COLS.values()}
    fold_data: Dict[int, Dict[str, pd.DataFrame]] = {}
    sample_fold: Dict[str, int] = {}

    for fold in folds:
        fold_dir = os.path.join(trial_dir, f"CV-fold_{fold}")
        path = os.path.join(fold_dir, f"validation-fold_{fold}_predictions_tumor.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing Proteus CV prediction: {path}")

        df = pd.read_csv(path, sep="\t", index_col=0)

        fold_pred_wide = _long_to_wide(df, "prediction_mean")
        fold_truth_wide = _long_to_wide(df, "truth")
        pred_parts.append(fold_pred_wide)
        truth_parts.append(fold_truth_wide)

        # Per-fold dict
        fd: Dict[str, pd.DataFrame] = {
            "pred_wide": fold_pred_wide,
            "truth_wide": fold_truth_wide,
        }

        for tsv_col, key in _OPTIONAL_COLS.items():
            if tsv_col in df.columns:
                w = _long_to_wide(df, tsv_col)
                optional_parts[key].append(w)
                fd[key] = w

        fold_data[fold] = fd

        for sample in fold_pred_wide.columns:
            sample_fold[sample] = fold

    result: Dict[str, Any] = {
        "pred_wide": _merge_wide_folds(pred_parts),
        "truth_wide": _merge_wide_folds(truth_parts),
        "fold_data": fold_data,
        "sample_fold": sample_fold,
    }
    for key, parts in optional_parts.items():
        result[key] = _merge_wide_folds(parts) if parts else None

    return result


def load_holdout_predictions_proteus_full(
    path: str,
) -> Dict[str, Any]:
    """Load a single Proteus holdout/inference TSV with full MDN columns.

    Returns dict with keys:
        ``pred_wide``         — genes × samples point predictions
        ``ci_lo_wide``        — genes × samples lower quantile (or None)
        ``ci_hi_wide``        — genes × samples upper quantile (or None)
        ``p0_wide``           — genes × samples pi_delta (or None)
        ``pi_gamma_wide``     — genes × samples pi_gamma (or None)
        ``pi_normal_wide``    — genes × samples pi_normal (or None)
        ``gamma_beta_wide``   — genes × samples gamma_beta (or None)
        ``normal_mean_wide``  — genes × samples normal_mean (or None)
        ``normal_stdev_wide`` — genes × samples normal_stdev (or None)
    """
    _OPTIONAL_COLS = {
        "ci_q5": "ci_lo_wide",
        "ci_q95": "ci_hi_wide",
        "pi_delta": "p0_wide",
        "pi_gamma": "pi_gamma_wide",
        "pi_normal": "pi_normal_wide",
        "gamma_beta": "gamma_beta_wide",
        "normal_mean": "normal_mean_wide",
        "normal_stdev": "normal_stdev_wide",
    }
    df = pd.read_csv(path, sep="\t", index_col=0)
    result: Dict[str, Any] = {"pred_wide": _long_to_wide(df, "prediction_mean")}
    for tsv_col, key in _OPTIONAL_COLS.items():
        result[key] = _long_to_wide(df, tsv_col) if tsv_col in df.columns else None
    return result


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


def load_cv_predictions_xgboost(
    trial_dir: str,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load XGBoost CV fold predictions and merge.

    XGBoost predictions use column ``prediction`` (no CIs).

    Returns (pred_wide, truth_wide).
    Both are genes × samples DataFrames.
    """
    pred_parts, truth_parts = [], []

    for fold in folds:
        fold_dir = os.path.join(trial_dir, f"CV-fold_{fold}")
        path = os.path.join(fold_dir, f"validation-fold_{fold}_predictions_tumor.tsv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing XGBoost CV prediction: {path}")

        df = pd.read_csv(path, sep="\t", index_col=0)
        pred_parts.append(_long_to_wide(df, "prediction"))
        truth_parts.append(_long_to_wide(df, "truth"))

    pred_wide = _merge_wide_folds(pred_parts)
    truth_wide = _merge_wide_folds(truth_parts)

    return pred_wide, truth_wide


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


# --------------- Vectorised per-gene / per-sample metrics ------------------


def _rankdata_columns(arr: np.ndarray) -> np.ndarray:
    """Rank each column independently (axis=0), average ties.

    arr : (n_rows, n_cols) — each column is ranked independently.
    Returns same shape with float64 ranks (1-based, ties averaged).

    Uses scipy.stats.rankdata with ``axis=0`` for a fully-vectorised
    C-level implementation (~100× faster than the previous pure-Python
    loop on wide arrays like 10 × 19,000).

    NaN values receive the highest ranks (sorted to end), matching the
    behaviour of the previous argsort-based implementation.
    """
    from scipy.stats import rankdata as _scipy_rankdata

    if np.isnan(arr).any():
        arr = np.where(np.isnan(arr), np.inf, arr)
    return _scipy_rankdata(arr, axis=0, method='average').astype(np.float64)


def gene_level_metrics(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Per-gene Pearson / Spearman / RMSE / MAE across samples.

    Fully vectorised — computes all ~19K genes in a single pass using
    NumPy broadcasting.  No per-gene Python loops or multiprocessing.
    Returns DataFrame indexed by gene name.
    """
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    T = tw.values.astype(np.float64)  # (G, S)
    P = pw.values.astype(np.float64)

    n_samples = T.shape[1]
    genes = tw.index

    if n_samples < 3:
        return pd.DataFrame({
            "n": n_samples, "pearson_r": np.nan, "spearman_rho": np.nan,
            "rmse": np.nan, "mae": np.nan, "truth_var": np.nan,
        }, index=genes)

    # ── Pearson r (vectorised) ──
    # Mask NaN: use per-gene valid counts so that samples with NaN for
    # a given gene are excluded from that gene's sums.
    finite_mask = np.isfinite(T) & np.isfinite(P)  # (G, S)
    T_clean = np.where(finite_mask, T, 0.0)
    P_clean = np.where(finite_mask, P, 0.0)
    n_valid = finite_mask.sum(axis=1).astype(np.float64)  # per-gene
    n_valid = np.where(n_valid >= 3, n_valid, np.nan)     # need >=3
    sum_t = T_clean.sum(axis=1)
    sum_p = P_clean.sum(axis=1)
    sum_tt = (T_clean * T_clean).sum(axis=1)
    sum_pp = (P_clean * P_clean).sum(axis=1)
    sum_tp = (T_clean * P_clean).sum(axis=1)
    num = n_valid * sum_tp - sum_t * sum_p
    den_t = n_valid * sum_tt - sum_t ** 2
    den_p = n_valid * sum_pp - sum_p ** 2
    with np.errstate(invalid="ignore"):
        den = np.sqrt(den_t * den_p)
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson_r = np.where(den > 0, num / den, np.nan)

    # ── Spearman rho (vectorised via NaN-aware ranking) ──
    # Rank each gene's truth/pred across samples, excluding NaN entries
    # so that NaN values don't inflate ranks for valid observations.
    T_masked_sp = np.where(finite_mask, T, np.nan)
    P_masked_sp = np.where(finite_mask, P, np.nan)
    T_ranked = _rankdata_columns_nan(T_masked_sp.T)  # (S, G)
    P_ranked = _rankdata_columns_nan(P_masked_sp.T)
    # Replace NaN ranks with 0; use per-gene valid count for Pearson-on-ranks
    rank_valid = np.isfinite(T_ranked) & np.isfinite(P_ranked)  # (S, G)
    T_rk = np.where(rank_valid, T_ranked, 0.0)
    P_rk = np.where(rank_valid, P_ranked, 0.0)
    n_rk = rank_valid.sum(axis=0).astype(np.float64)  # per-gene valid count
    n_rk_safe = np.where(n_rk >= 3, n_rk, np.nan)
    sum_tr = T_rk.sum(axis=0)
    sum_pr = P_rk.sum(axis=0)
    sum_trtr = (T_rk * T_rk).sum(axis=0)
    sum_prpr = (P_rk * P_rk).sum(axis=0)
    sum_trpr = (T_rk * P_rk).sum(axis=0)
    num_s = n_rk_safe * sum_trpr - sum_tr * sum_pr
    den_tr = n_rk_safe * sum_trtr - sum_tr ** 2
    den_pr = n_rk_safe * sum_prpr - sum_pr ** 2
    with np.errstate(invalid="ignore"):
        den_s = np.sqrt(den_tr * den_pr)
    with np.errstate(divide="ignore", invalid="ignore"):
        spearman_rho = np.where(den_s > 0, num_s / den_s, np.nan)

    # ── RMSE, MAE, truth variance (vectorised, NaN-safe) ──
    # Genes where all samples are NaN produce empty slices; handle
    # explicitly to avoid RuntimeWarning from nanmean/nanvar.
    any_valid = finite_mask.any(axis=1)  # (G,) — True if gene has ≥1 valid pair
    any_truth = np.isfinite(T).any(axis=1)
    err = P - T  # NaN propagates naturally
    rmse = np.full(T.shape[0], np.nan)
    mae = np.full(T.shape[0], np.nan)
    truth_var = np.full(T.shape[0], np.nan)
    if any_valid.any():
        rmse[any_valid] = np.sqrt(np.nanmean(err[any_valid] ** 2, axis=1))
        mae[any_valid] = np.nanmean(np.abs(err[any_valid]), axis=1)
    if any_truth.any():
        truth_var[any_truth] = np.nanvar(T[any_truth], axis=1, ddof=0)

    return pd.DataFrame({
        "n": n_samples,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "rmse": rmse,
        "mae": mae,
        "truth_var": truth_var,
    }, index=genes)


def sample_level_metrics(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Per-sample Pearson / Spearman / RMSE / MAE across genes.

    Vectorised — computes all samples in a single pass.
    Returns DataFrame indexed by sample name.
    """
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    T = tw.values.astype(np.float64)  # (G, S)
    P = pw.values.astype(np.float64)

    n_genes = T.shape[0]
    samples = tw.columns

    if n_genes < 3:
        return pd.DataFrame({
            "n": n_genes, "pearson_r": np.nan, "spearman_rho": np.nan,
            "rmse": np.nan, "mae": np.nan,
        }, index=samples)

    # ── Pearson r per sample (vectorised along gene axis) ──
    # Mask NaN: use nansum so that genes with NaN in a given sample
    # are excluded from that sample's sums rather than poisoning them.
    finite_mask = np.isfinite(T) & np.isfinite(P)  # (G, S)
    T_clean = np.where(finite_mask, T, 0.0)
    P_clean = np.where(finite_mask, P, 0.0)
    n_valid = finite_mask.sum(axis=0).astype(np.float64)  # per-sample
    n_valid = np.where(n_valid >= 3, n_valid, np.nan)     # need >=3
    sum_t = (T_clean).sum(axis=0)
    sum_p = (P_clean).sum(axis=0)
    sum_tt = (T_clean * T_clean).sum(axis=0)
    sum_pp = (P_clean * P_clean).sum(axis=0)
    sum_tp = (T_clean * P_clean).sum(axis=0)
    num = n_valid * sum_tp - sum_t * sum_p
    den = np.sqrt((n_valid * sum_tt - sum_t ** 2) * (n_valid * sum_pp - sum_p ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson_r = np.where(den > 0, num / den, np.nan)

    # ── Spearman rho per sample — NaN-aware rank along gene axis ──
    T_masked_sp = np.where(finite_mask, T, np.nan)
    P_masked_sp = np.where(finite_mask, P, np.nan)
    T_ranked = _rankdata_columns_nan(T_masked_sp)   # (G, S) — rank each column along genes
    P_ranked = _rankdata_columns_nan(P_masked_sp)
    rank_valid = np.isfinite(T_ranked) & np.isfinite(P_ranked)  # (G, S)
    T_rk = np.where(rank_valid, T_ranked, 0.0)
    P_rk = np.where(rank_valid, P_ranked, 0.0)
    n_rk = rank_valid.sum(axis=0).astype(np.float64)  # per-sample valid count
    n_rk_safe = np.where(n_rk >= 3, n_rk, np.nan)
    sum_tr = T_rk.sum(axis=0)
    sum_pr = P_rk.sum(axis=0)
    sum_trtr = (T_rk * T_rk).sum(axis=0)
    sum_prpr = (P_rk * P_rk).sum(axis=0)
    sum_trpr = (T_rk * P_rk).sum(axis=0)
    num_s = n_rk_safe * sum_trpr - sum_tr * sum_pr
    with np.errstate(invalid="ignore"):
        den_s = np.sqrt((n_rk_safe * sum_trtr - sum_tr ** 2) * (n_rk_safe * sum_prpr - sum_pr ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        spearman_rho = np.where(den_s > 0, num_s / den_s, np.nan)

    # ── RMSE, MAE (NaN-safe) ──
    any_valid = finite_mask.any(axis=0)  # (S,) — True if sample has ≥1 valid pair
    err = P - T
    rmse = np.full(T.shape[1], np.nan)
    mae = np.full(T.shape[1], np.nan)
    if any_valid.any():
        rmse[any_valid] = np.sqrt(np.nanmean(err[:, any_valid] ** 2, axis=0))
        mae[any_valid] = np.nanmean(np.abs(err[:, any_valid]), axis=0)

    return pd.DataFrame({
        "n": n_genes,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "rmse": rmse,
        "mae": mae,
    }, index=samples)


# --------------- GVW-r (gene-variance-weighted correlation) ----------------

def compute_gene_variance(truth_wide: pd.DataFrame) -> Dict[str, float]:
    """Per-gene variance from truth across samples → {gene: var}."""
    return truth_wide.var(axis=1, ddof=1).to_dict()


def _rankdata_columns_nan(arr: np.ndarray) -> np.ndarray:
    """Rank each column independently, skipping NaN entries.

    arr : (n_rows, n_cols) — each column is ranked independently.
    Returns same shape with float64 ranks (1-based, ties averaged).
    NaN entries in the input receive NaN ranks (excluded from ranking).
    Only the finite entries within each column participate in ranking.
    """
    from scipy.stats import rankdata as _scipy_rankdata

    result = np.full_like(arr, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(arr)

    # Fast path: no NaN at all — rank everything in one vectorised call
    if finite_mask.all():
        return _scipy_rankdata(arr, axis=0, method='average').astype(np.float64)

    # Per-column ranking of only finite values
    for j in range(arr.shape[1]):
        col_mask = finite_mask[:, j]
        if col_mask.sum() < 2:
            # 0 or 1 finite values — rank is trivially 1 (or leave NaN)
            if col_mask.any():
                result[col_mask, j] = 1.0
            continue
        result[col_mask, j] = _scipy_rankdata(
            arr[col_mask, j], method='average',
        ).astype(np.float64)
    return result


def _gvw_rho_from_wide(
    T: np.ndarray,
    P: np.ndarray,
    w_gene: np.ndarray,
) -> float:
    """Gene-variance-weighted Spearman rho from (G, S) arrays.

    Parameters
    ----------
    T, P : (n_genes, n_samples) truth and prediction arrays.
    w_gene : (n_genes,) per-gene variance weights.

    For each gene, computes the exact Spearman rho across samples,
    dropping any NaN observations.  Genes with fewer than 5 valid
    sample-pairs are excluded entirely.  The final result is the
    variance-weighted average of per-gene rho values.
    """
    from scipy.stats import spearmanr as _spearmanr

    n_genes, n_samples = T.shape
    if n_samples < 2 or n_genes < 1:
        return np.nan

    rho_per_gene = np.full(n_genes, np.nan)
    for g in range(n_genes):
        valid = np.isfinite(T[g]) & np.isfinite(P[g])
        if valid.sum() < 5:
            continue
        tv, pv = T[g, valid], P[g, valid]
        if np.ptp(tv) == 0 or np.ptp(pv) == 0:
            continue
        rho_per_gene[g] = _spearmanr(tv, pv).statistic

    # Keep only genes with a valid rho and positive variance weight
    usable = np.isfinite(rho_per_gene) & (w_gene > 0)
    if not usable.any():
        return np.nan

    w = w_gene[usable]
    rho = rho_per_gene[usable]
    return float(np.dot(w, rho) / w.sum())


def _gvw_r_from_wide(
    T: np.ndarray,
    P: np.ndarray,
    w_gene: np.ndarray,
) -> float:
    """Gene-variance-weighted Pearson r from (G, S) arrays.

    Parameters
    ----------
    T, P : (n_genes, n_samples) truth and prediction arrays.
    w_gene : (n_genes,) per-gene variance weights.

    For each gene, computes the exact Pearson r across samples,
    dropping any NaN observations.  Genes with fewer than 5 valid
    sample-pairs are excluded entirely.  The final result is the
    variance-weighted average of per-gene r values.
    """
    n_genes, n_samples = T.shape
    if n_samples < 2 or n_genes < 1:
        return np.nan

    r_per_gene = np.full(n_genes, np.nan)
    for g in range(n_genes):
        valid = np.isfinite(T[g]) & np.isfinite(P[g])
        n = valid.sum()
        if n < 5:
            continue
        t = T[g, valid]
        p = P[g, valid]
        mean_t = t.mean()
        mean_p = p.mean()
        dt = t - mean_t
        dp = p - mean_p
        denom = np.sqrt(np.dot(dt, dt) * np.dot(dp, dp))
        if denom < 1e-12:
            continue
        r_per_gene[g] = np.dot(dt, dp) / denom

    # Keep only genes with a valid r and positive variance weight
    usable = np.isfinite(r_per_gene) & (w_gene > 0)
    if not usable.any():
        return np.nan

    w = w_gene[usable]
    r = r_per_gene[usable]
    return float(np.dot(w, r) / w.sum())


def gvw_r(
    truth_flat: np.ndarray,
    pred_flat: np.ndarray,
    genes_flat: np.ndarray,
    var_map: Dict[str, float],
) -> float:
    """Gene-variance-weighted Pearson r (flat-array interface).

    Reshapes into (G, S) arrays and delegates to ``_gvw_r_from_wide``.
    """
    mask = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    t = truth_flat[mask]
    p = pred_flat[mask]
    g = genes_flat[mask]
    if len(t) < 3:
        return np.nan

    unique_genes = np.unique(g)
    n_genes = len(unique_genes)
    # Build (G, S) arrays — pad with NaN for ragged counts
    gene_to_idx = {ug: i for i, ug in enumerate(unique_genes)}
    max_count = max(np.sum(g == ug) for ug in unique_genes)
    T = np.full((n_genes, max_count), np.nan)
    P = np.full((n_genes, max_count), np.nan)
    counters = np.zeros(n_genes, dtype=int)
    for k in range(len(t)):
        gi = gene_to_idx[g[k]]
        T[gi, counters[gi]] = t[k]
        P[gi, counters[gi]] = p[k]
        counters[gi] += 1
    w_gene = np.array([var_map.get(str(ug), 0.0) for ug in unique_genes],
                      dtype=np.float64)
    return _gvw_r_from_wide(T, P, w_gene)


def gvw_rho(
    truth_flat: np.ndarray,
    pred_flat: np.ndarray,
    genes_flat: np.ndarray,
    var_map: Dict[str, float],
) -> float:
    """Gene-variance-weighted Spearman rho (flat-array interface).

    Reshapes into (G, S) arrays and delegates to ``_gvw_rho_from_wide``.
    """
    mask = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    t = truth_flat[mask]
    p = pred_flat[mask]
    g = genes_flat[mask]
    if len(t) < 3:
        return np.nan

    unique_genes = np.unique(g)
    n_genes = len(unique_genes)
    gene_to_idx = {ug: i for i, ug in enumerate(unique_genes)}
    max_count = max(np.sum(g == ug) for ug in unique_genes)
    T = np.full((n_genes, max_count), np.nan)
    P = np.full((n_genes, max_count), np.nan)
    counters = np.zeros(n_genes, dtype=int)
    for k in range(len(t)):
        gi = gene_to_idx[g[k]]
        T[gi, counters[gi]] = t[k]
        P[gi, counters[gi]] = p[k]
        counters[gi] += 1
    w_gene = np.array([var_map.get(str(ug), 0.0) for ug in unique_genes],
                      dtype=np.float64)
    return _gvw_rho_from_wide(T, P, w_gene)


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

# Expression bounds (log2(CPM+1) scale)
EXPRESSION_FLOOR = 0.0
EXPRESSION_CEIL = 19.93  # log2(10^6 + 1)


def variance_group_metrics(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    var_map: Optional[Dict[str, float]] = None,
    gene_met: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute per-gene Pearson/Spearman broken down by variance group.

    Returns summary DataFrame with rows = variance groups.
    """
    if var_map is None:
        var_map = compute_gene_variance(truth_wide)

    if gene_met is None:
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
        GVW metrics and variance-group breakdowns use these weights.
        When *None*, variance is computed from the truth–prediction
        intersection (only samples visible to the model contribute).
    """
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    if var_map is None:
        var_map = compute_gene_variance(tw)

    # GVW metrics via wide-format vectorised helpers (no flattening)
    T_arr = tw.values.astype(np.float64)
    P_arr = pw.values.astype(np.float64)
    w_gene = np.array([var_map.get(g, 0.0) for g in common_genes], dtype=np.float64)
    print(f"    GVW ...", end=" ", flush=True)
    gvw_r_val = _gvw_r_from_wide(T_arr, P_arr, w_gene)
    gvw_rho_val = _gvw_rho_from_wide(T_arr, P_arr, w_gene)

    print("gene metrics ...", end=" ", flush=True)
    gene_met = gene_level_metrics(tw, pw, n_workers=n_workers)
    print("sample metrics ...", end=" ", flush=True)
    sample_met = sample_level_metrics(tw, pw, n_workers=n_workers)
    print("done", flush=True)

    # Suppress benign "Mean of empty slice" from pandas .mean() on
    # columns that may be all-NaN (e.g. constant genes with 5 HO samples).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice",
                                category=RuntimeWarning)
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
    return max(1, os.cpu_count() or 1)


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns path."""
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# PHS Benchmarking — PDX truth and EPIC-Seq loaders
# ---------------------------------------------------------------------------

# Prediction PDX names that differ from truth label column names.
_PDX_NAME_MAP: Dict[str, str] = {
    "DIVE_CDX172": "DIVE_CDX17",
}


def parse_phs_sample(sample: str) -> Dict[str, str]:
    """Parse a PHS sample name into components.

    Format: ``PHS_{PDX}-{HD}_{coverage}x_{TFx}_merge``

    Returns dict with keys: pdx, hd, coverage, tfx.
    """
    m = re.match(r'^PHS_(.+)-(KP\w+)_(\d+x)_(TF[\d.]+)_merge$', sample)
    if m is None:
        raise ValueError(f"Cannot parse PHS sample name: {sample}")
    return {
        "pdx": m.group(1), "hd": m.group(2),
        "coverage": m.group(3), "tfx": m.group(4),
    }


def load_pdx_truth_for_phs(samples: Sequence[str]) -> pd.DataFrame:
    """Build truth (genes x samples) for PHS samples from PDX expression labels.

    Each PHS sample's truth is the pure PDX tumor expression looked up from
    ``labels/PC-PDX_Log2CPMplus1.tsv`` and ``labels/SCLC-PDX_Log2CPMplus1.tsv``.
    """
    pc_pdx = pd.read_csv(
        PROJECT_ROOT / "labels" / "PC-PDX_Log2CPMplus1.tsv", sep="\t", index_col=0,
    )
    sclc_pdx = pd.read_csv(
        PROJECT_ROOT / "labels" / "SCLC-PDX_Log2CPMplus1.tsv", sep="\t", index_col=0,
    )
    all_genes = pc_pdx.index.union(sclc_pdx.index)
    pdx_pool = pd.concat(
        [pc_pdx.reindex(all_genes), sclc_pdx.reindex(all_genes)], axis=1,
    )

    cols: Dict[str, pd.Series] = {}
    for sample in samples:
        info = parse_phs_sample(sample)
        truth_name = _PDX_NAME_MAP.get(info["pdx"], info["pdx"])
        if truth_name not in pdx_pool.columns:
            raise ValueError(
                f"PDX '{info['pdx']}' (mapped -> '{truth_name}') not in truth labels. "
                f"Available: {sorted(pdx_pool.columns.tolist())}"
            )
        cols[sample] = pdx_pool[truth_name]

    return pd.DataFrame(cols)


def load_holdout_predictions_epicseq(
    path: str,
    score_cols: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Load EPIC-Seq long-format TSV and pivot each score to wide (genes x samples).

    Reads the file once, replaces Inf/-Inf with NaN, aggregates multiple
    TSS IDs per gene (mean), and pivots each requested score column.

    NDR (nucleosome depletion ratio) is inverted at load time because it is
    expected to be inversely correlated with expression: higher NDR ⇒ more
    open chromatin ⇒ higher expression, but the raw score is a *depletion*
    ratio so the sign must be flipped for comparison with log₂(CPM+1).

    Returns dict mapping score_col -> wide DataFrame.
    """
    df = pd.read_csv(path, sep="\t")
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # NDR is inversely correlated with expression — flip sign at the outset
    _NDR_COLS = {"NDR_relative_2k"}
    for col in score_cols:
        if col in _NDR_COLS:
            df[col] = -df[col]

    grouped = df.groupby(["gene", "Sample"])[list(score_cols)].mean().reset_index()
    result: Dict[str, pd.DataFrame] = {}
    for col in score_cols:
        result[col] = grouped.pivot(index="gene", columns="Sample", values=col)
    return result


def generate_null_predictions(
    truth_wide: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Null model: shuffle sample values independently within each gene (row).

    For each gene, the expression values across samples are permuted,
    destroying per-gene cross-sample correlations.  This is the correct
    null for per-gene Spearman rho (shuffling within columns instead
    would preserve per-gene structure and produce a bimodal null).
    """
    rng = np.random.default_rng(seed)
    shuffled = truth_wide.copy()
    arr = shuffled.values  # (G, S)
    for g in range(arr.shape[0]):
        row = arr[g]
        finite = np.isfinite(row)
        finite_vals = row[finite].copy()
        rng.shuffle(finite_vals)
        row[finite] = finite_vals
    return shuffled


def compute_subset_metrics(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    n_workers: Optional[int] = None,
    var_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute the benchmark metrics on an aligned (truth, pred) pair.

    Parameters
    ----------
    var_map : optional pre-computed gene variance dict.  When supplied,
        GVW metrics use these weights.  When *None*, variance is computed
        from the truth–prediction intersection (only samples visible to
        the model contribute).

    Returns dict with:
      sample_rho  : np.ndarray of per-sample transcriptome-wide Spearman rho
      mean_gene_rho  : float
      median_gene_rho: float
      gvw_rho        : float
      rmse           : float (mean across genes of per-gene RMSE)
      mean_top10pct_gene_rho : float (mean Spearman rho in top 10% most variable genes)
      gene_rho_array : np.ndarray of per-gene Spearman rho
      n_genes_rho_ge070 : int (genes with Spearman rho >= 0.70)
    """
    empty = {
        "sample_rho": np.array([]),
        "mean_gene_rho": np.nan,
        "median_gene_rho": np.nan,
        "gvw_rho": np.nan,
        "rmse": np.nan,
        "mean_top10pct_gene_rho": np.nan,
        "gene_rho_array": np.array([]),
        "gene_rho_genes": np.array([], dtype=object),
        "top10pct_gene_rho_array": np.array([]),
        "n_genes_rho_ge070": 0,
    }

    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    if len(common_samples) < 2 or len(common_genes) < 3:
        return empty

    # Suppress benign warnings from np.nanmean / np.nanpercentile on
    # all-NaN slices (the NaN result is the desired outcome).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice",
                                category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                category=RuntimeWarning)

        if var_map is None:
            var_map = compute_gene_variance(tw)

        T_arr = tw.values.astype(np.float64)
        P_arr = pw.values.astype(np.float64)

        # Per-gene valid pair count; exclude genes with < 5 valid samples
        n_valid_per_gene = (np.isfinite(T_arr) & np.isfinite(P_arr)).sum(axis=1)
        usable_genes = n_valid_per_gene >= 5

        # Fully-vectorised gene-level and sample-level metrics
        gmet = gene_level_metrics(tw, pw)
        smet = sample_level_metrics(tw, pw)

        # Filter gene rho to genes with >= 5 valid samples
        gene_rho_raw = gmet["spearman_rho"].values.copy()
        gene_rho_raw[~usable_genes] = np.nan
        gene_rho_s = pd.Series(gene_rho_raw, index=common_genes).dropna()
        gene_rho = gene_rho_s.values
        sample_rho = smet["spearman_rho"].dropna().values

        # GVW-rho: weighted average of per-gene rho (no Python loop —
        # reuses the vectorised rho from gene_level_metrics above)
        w_gene = np.array([var_map.get(g, 0.0) for g in common_genes],
                          dtype=np.float64)
        gvw_ok = np.isfinite(gene_rho_raw) & (w_gene > 0)
        gvw_rho_val = (float(np.dot(w_gene[gvw_ok], gene_rho_raw[gvw_ok])
                             / w_gene[gvw_ok].sum())
                       if gvw_ok.any() else np.nan)

        # Mean Spearman rho in the top 10% most variable genes
        gene_vars = pd.Series(
            {g: var_map.get(g, 0.0) for g in common_genes[usable_genes]})
        n_top = max(1, int(np.ceil(len(gene_vars) * 0.10)))
        top_genes = gene_vars.nlargest(n_top).index
        top_rho_vals = gene_rho_s.reindex(top_genes).dropna().values
        mean_top10 = float(np.nanmean(top_rho_vals)) if len(top_rho_vals) else np.nan

        # RMSE: mean across usable genes only
        gene_rmse = gmet["rmse"].values.copy()
        gene_rmse[~usable_genes] = np.nan
        rmse_val = float(np.nanmean(gene_rmse))

        result: Dict[str, Any] = {
            "sample_rho": sample_rho,
            "mean_gene_rho": float(np.nanmean(gene_rho)) if len(gene_rho) else np.nan,
            "median_gene_rho": float(np.nanmedian(gene_rho)) if len(gene_rho) else np.nan,
            "gvw_rho": gvw_rho_val,
            "rmse": rmse_val,
            "mean_top10pct_gene_rho": mean_top10,
            "gene_rho_array": gene_rho,
            "gene_rho_genes": gene_rho_s.index.values,
            "top10pct_gene_rho_array": top_rho_vals,
            "n_genes_rho_ge070": int((gene_rho >= 0.70).sum()) if len(gene_rho) else 0,
        }

    return result


# ---------------------------------------------------------------------------
# Study-defined sample sets (BTS / PHT / PHS)
# ---------------------------------------------------------------------------

# The 25 PC-PDX samples in the study: 20 in BTS-CV + 5 holdout
_PC_PDX_CV = [
    "LuCaP_145-1", "LuCaP_189-4", "LuCaP_70CR", "LuCaP_96",       # F1
    "LuCaP_173-1", "LuCaP_136CR", "LuCaP_167CR", "LuCaP_170-3",   # F2
    "LuCaP_208-4", "LuCaP_147CR", "LuCaP_78", "LuCaP_96CR",       # F3
    "LuCaP_145-2", "LuCaP_141", "LuCaP_35", "LuCaP_70",           # F4
    "LuCaP_49", "LuCaP_35CR", "LuCaP_58", "LuCaP_176",            # F5
]
_PC_PDX_HO = ["LuCaP_136", "LuCaP_167", "LuCaP_176CR", "LuCaP_92", "LuCaP_93"]
PC_PDX_STUDY = sorted(_PC_PDX_CV + _PC_PDX_HO)

# The 25 SCLC-PDX samples in the study: 20 in BTS-CV + 5 holdout
_SCLC_PDX_CV = [
    "SCLC152", "LU86", "DIVE_CDX30P", "FHSC14",                    # F1
    "MSK_LX631", "MSK_LX1042", "DIVE_CDX25", "LX110",             # F2
    "MSK_LX640", "DIVE_CDX29", "FHSC36", "FHSC47",                # F3
    "MSK_LX1322", "LU100", "FHSC46", "LX48",                      # F4
    "H211", "LX33B", "SCLC150", "FHSC04",                         # F5
]
_SCLC_PDX_HO = ["DIVE_CDX17", "H526", "MSK_LX465", "NCI109B", "SCLC151"]
SCLC_PDX_STUDY = sorted(_SCLC_PDX_CV + _SCLC_PDX_HO)

# Holdout sets for easy look-up
PDX_HOLDOUT_SAMPLES = set(_PC_PDX_HO + _SCLC_PDX_HO)


def build_study_truth_pool() -> pd.DataFrame:
    """Build the canonical truth pool restricted to the 25+25 study PDX
    tumor samples (no healthy donors).

    Returns a wide DataFrame (genes × samples) used as the single source of
    gene-variance weights throughout the analysis pipeline.
    """
    parts = []
    for cohort, study_set in (("PC-PDX", PC_PDX_STUDY), ("SCLC-PDX", SCLC_PDX_STUDY)):
        path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
        if not path.exists():
            continue
        t = load_truth(cohort)
        keep = [s for s in study_set if s in t.columns]
        parts.append(t[keep])
    if not parts:
        raise FileNotFoundError("No study truth labels found")
    combined = pd.concat(parts, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


def build_study_var_map() -> Dict[str, float]:
    """Gene variance from the 25+25 study PDX tumor truth pool."""
    return compute_gene_variance(build_study_truth_pool())


# ---------------------------------------------------------------------------
# Interval coverage evaluation
# ---------------------------------------------------------------------------

def empirical_coverage(
    truth_flat: np.ndarray,
    ci_lo_flat: np.ndarray,
    ci_hi_flat: np.ndarray,
) -> float:
    """Fraction of truth values inside [ci_lo, ci_hi]."""
    mask = np.isfinite(truth_flat) & np.isfinite(ci_lo_flat) & np.isfinite(ci_hi_flat)
    y, lo, hi = truth_flat[mask], ci_lo_flat[mask], ci_hi_flat[mask]
    if len(y) == 0:
        return np.nan
    return float(((y >= lo) & (y <= hi)).sum() / len(y))


def compute_coverage_curve(
    truth_flat: np.ndarray,
    ci_lo_flat: np.ndarray,
    ci_hi_flat: np.ndarray,
    pred_flat: np.ndarray,
    nominal_levels: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
) -> Dict[str, List[float]]:
    """Compute empirical coverage at multiple nominal levels.

    Assumes ci_lo, ci_hi correspond to the widest band (e.g. 95%).
    Narrower bands are interpolated by shrinking the interval symmetrically
    around the prediction mean.

    Returns dict with keys 'nominal' and 'empirical'.
    """
    mask = (np.isfinite(truth_flat) & np.isfinite(ci_lo_flat)
            & np.isfinite(ci_hi_flat) & np.isfinite(pred_flat))
    y = truth_flat[mask]
    lo = ci_lo_flat[mask]
    hi = ci_hi_flat[mask]
    mu = pred_flat[mask]
    if len(y) < 10:
        return {"nominal": list(nominal_levels),
                "empirical": [np.nan] * len(nominal_levels)}

    # Original interval corresponds to the widest nominal (assume 95%)
    max_nominal = max(nominal_levels)

    empiricals = []
    for level in nominal_levels:
        scale = level / max_nominal
        adj_lo = mu - scale * (mu - lo)
        adj_hi = mu + scale * (hi - mu)
        coverage = float(((y >= adj_lo) & (y <= adj_hi)).sum() / len(y))
        empiricals.append(coverage)
    return {"nominal": list(nominal_levels), "empirical": empiricals}
