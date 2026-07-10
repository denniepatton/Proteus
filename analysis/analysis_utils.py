
"""Core analysis utilities: data loading, metrics, statistics, and constants."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIN_CORRELATION_N = 5

_GENE_LISTS: Dict[str, List[str]] = {}
_GENEFU_BRIDGE_CACHE: Optional[Tuple[Any, Any, Any]] = None

# Targeted alias support only. These are not broad synonym expansions; they
# exist to preserve the paper-path scoring behavior for a small number of known
# mismatches between curated gene sets and some expression matrices.
_TARGETED_GENE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "TARP": ("CD3G",),
}


def benjamini_hochberg(p_values: Union[pd.Series, np.ndarray, Sequence[float]]) -> Union[pd.Series, np.ndarray]:
    """Benjamini-Hochberg FDR correction preserving Series index when present."""
    if isinstance(p_values, pd.Series):
        values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=np.float64)
        index = p_values.index
        is_series = True
    else:
        values = pd.to_numeric(pd.Series(p_values), errors="coerce").to_numpy(dtype=np.float64)
        index = None
        is_series = False

    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(values)
    if valid_mask.any():
        valid_values = values[valid_mask]
        order = np.argsort(valid_values, kind="mergesort")
        ranked = valid_values[order]
        n_valid = ranked.size
        scale = float(n_valid) / np.arange(1, n_valid + 1, dtype=np.float64)
        adjusted_ranked = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
        adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
        restored = np.empty_like(adjusted_ranked)
        restored[order] = adjusted_ranked
        adjusted[valid_mask] = restored

    if is_series:
        return pd.Series(adjusted, index=index)
    return adjusted


def _augment_missing_gene_aliases(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Add a small set of deterministic fallback gene rows when absent.

    The current analysis contract only needs one targeted fallback: if `TARP`
    is absent but `CD3G` is present, expose a synthetic `TARP` row by copying
    `CD3G`. This preserves older scoring expectations without introducing a
    broad alias system that could silently duplicate many genes.
    """
    if expr_df.empty:
        return expr_df.copy()

    augmented = expr_df.copy()
    augmented.index = pd.Index([str(gene_name) for gene_name in augmented.index])
    for target_gene, source_candidates in _TARGETED_GENE_ALIASES.items():
        if target_gene in augmented.index:
            continue
        for source_gene in source_candidates:
            if source_gene in augmented.index:
                augmented.loc[target_gene] = augmented.loc[source_gene]
                break
    return augmented


def _resolve_expression_gene_rows(expr_df: pd.DataFrame, gene_list: Sequence[str]) -> List[str]:
    """Resolve requested gene names against an expression matrix.

    Resolution is exact-name first, then targeted fallback aliases. Returned
    row labels are unique and preserve the requested gene-list order so mean
    gene-set scores remain deterministic.
    """
    resolved_rows: List[str] = []
    seen_rows: set[str] = set()
    available_rows = {str(row_name) for row_name in expr_df.index}

    for requested_gene in gene_list:
        requested_name = str(requested_gene)
        candidate_rows = (requested_name,) + _TARGETED_GENE_ALIASES.get(requested_name, ())
        for candidate_row in candidate_rows:
            if candidate_row in available_rows and candidate_row not in seen_rows:
                resolved_rows.append(candidate_row)
                seen_rows.add(candidate_row)
                break
    return resolved_rows


def load_truth(cohort: str) -> pd.DataFrame:
    """Load truth expression: labels/<cohort>_Log2CPMplus1.tsv → wide DataFrame.

    Returns DataFrame with gene names as index, sample names as columns.
    """
    path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
    df = pd.read_csv(path, sep="\t", index_col=0)
    return _augment_missing_gene_aliases(df)


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


def clean_epicseq_sample_names_wcdt(columns: pd.Index) -> pd.Index:
    """Normalise CRPC-WCDT EPIC-Seq sample IDs to match truth column names.

    EPIC-Seq names look like ``DTB-097-Progression-cfDNA_recalibrated``;
    truth labels use ``DTB-097-PRO``.  Mapping:
      - Strip ``(-cfDNA)?_recalibrated`` suffix
      - ``Baseline`` → ``BL``
      - ``Progression`` → ``PRO``
    """
    _TIMEPOINT_MAP = {"Baseline": "BL", "Progression": "PRO"}
    cleaned = []
    for c in columns:
        s = re.sub(r'(-cfDNA)?_recalibrated$', '', str(c))
        for long, short in _TIMEPOINT_MAP.items():
            s = s.replace(f'-{long}', f'-{short}')
        cleaned.append(s)
    return pd.Index(cleaned)


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
        ``activity_odds_ratio_wide`` — genes × samples log2((pi_normal + eps) / (1 - pi_normal + eps)) (or None)
        ``active_confidence_wide`` — genes × samples P(Y >= 2.0) from the raw MDN (or None)
        ``fold_data``         — {fold_int: dict of per-fold wide DFs}
        ``sample_fold``       — {sample_name: fold_int}
    """
    fold_prediction_paths = {
        int(fold): os.path.join(
            trial_dir,
            f"CV-fold_{int(fold)}",
            f"validation-fold_{int(fold)}_predictions_tumor.tsv",
        )
        for fold in folds
    }
    return load_cv_predictions_proteus_full_from_paths(fold_prediction_paths)


def load_cv_predictions_proteus_full_from_paths(
    fold_prediction_paths: Dict[int, Union[str, Path]],
) -> Dict[str, Any]:
    """Load Proteus CV predictions from an explicit fold→file path mapping."""
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

    for fold, path_like in sorted(fold_prediction_paths.items()):
        path = os.fspath(path_like)
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

    if (
        result.get("p0_wide") is not None
        and result.get("pi_gamma_wide") is not None
        and result.get("pi_normal_wide") is not None
    ):
        result["activity_odds_ratio_wide"] = mdn_activity_odds_ratio_wide(
            result["p0_wide"],
            result["pi_gamma_wide"],
            result["pi_normal_wide"],
        )
    else:
        result["activity_odds_ratio_wide"] = None

    required_mdn_keys = (
        "p0_wide",
        "pi_gamma_wide",
        "pi_normal_wide",
        "gamma_beta_wide",
        "normal_mean_wide",
        "normal_stdev_wide",
    )
    if all(result.get(key) is not None for key in required_mdn_keys):
        result["active_confidence_wide"] = mdn_on_probability_wide(
            result["p0_wide"],
            result["pi_gamma_wide"],
            result["pi_normal_wide"],
            result["gamma_beta_wide"],
            result["normal_mean_wide"],
            result["normal_stdev_wide"],
        )
    else:
        result["active_confidence_wide"] = None

    return result


# CRPC-WCDT canonical evaluation set. Only these 13 samples have complete,
# cross-model matched results; every analysis that loads CRPC-WCDT predictions
# is restricted to this set at load time so the n=13 cap propagates naturally.
CRPC_WCDT_COMPLETE_SAMPLES: Tuple[str, ...] = (
    "DTB-097-PRO",
    "DTB-102-PRO",
    "DTB-119-PRO",
    "DTB-149-BL",
    "DTB-156-BL",
    "DTB-183-BL",
    "DTB-205-BL",
    "DTB-210-BL",
    "DTB-214-BL",
    "DTB-216-PRO",
    "DTB-258-BL",
    "DTB-261-BL",
    "DTB-266-BL",
)
_CRPC_WCDT_PATH_TOKEN: str = "CRPC-WCDT"


def _restrict_wcdt_wide_by_path(
    wide: Optional[pd.DataFrame], path: str,
) -> Optional[pd.DataFrame]:
    """Restrict a genes × samples table to the CRPC-WCDT canonical 13 samples.

    Applied only when *path* refers to a CRPC-WCDT prediction file; otherwise
    the table is returned unchanged. Column order follows the canonical list.
    """
    if wide is None or _CRPC_WCDT_PATH_TOKEN not in str(path):
        return wide
    keep = [s for s in CRPC_WCDT_COMPLETE_SAMPLES if s in wide.columns]
    return wide[keep]


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
        ``activity_odds_ratio_wide`` — genes × samples log2((pi_normal + eps) / (1 - pi_normal + eps)) (or None)
        ``active_confidence_wide`` — genes × samples P(Y >= 2.0) from the raw MDN (or None)
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
    if (
        result.get("p0_wide") is not None
        and result.get("pi_gamma_wide") is not None
        and result.get("pi_normal_wide") is not None
    ):
        result["activity_odds_ratio_wide"] = mdn_activity_odds_ratio_wide(
            result["p0_wide"],
            result["pi_gamma_wide"],
            result["pi_normal_wide"],
        )
    else:
        result["activity_odds_ratio_wide"] = None
    required_mdn_keys = (
        "p0_wide",
        "pi_gamma_wide",
        "pi_normal_wide",
        "gamma_beta_wide",
        "normal_mean_wide",
        "normal_stdev_wide",
    )
    if all(result.get(key) is not None for key in required_mdn_keys):
        result["active_confidence_wide"] = mdn_on_probability_wide(
            result["p0_wide"],
            result["pi_gamma_wide"],
            result["pi_normal_wide"],
            result["gamma_beta_wide"],
            result["normal_mean_wide"],
            result["normal_stdev_wide"],
        )
    else:
        result["active_confidence_wide"] = None
    for _key, _table in list(result.items()):
        if isinstance(_table, pd.DataFrame):
            result[_key] = _restrict_wcdt_wide_by_path(_table, path)
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

    pred_wide = _restrict_wcdt_wide_by_path(pred_wide, path)
    ci_lo = _restrict_wcdt_wide_by_path(ci_lo, path)
    ci_hi = _restrict_wcdt_wide_by_path(ci_hi, path)
    return pred_wide, ci_lo, ci_hi


def load_holdout_predictions_xgboost(path: str) -> pd.DataFrame:
    """Load a single XGBoost holdout predictions_tumor.tsv → pred_wide."""
    try:
        df = pd.read_csv(path, sep="\t", index_col=0)
    except pd.errors.ParserError:
        # Some HPC-backed filesystems intermittently fail the C parser on
        # otherwise valid TSVs; the Python engine is slower but more robust.
        df = pd.read_csv(path, sep="\t", index_col=0, engine="python")
    return _restrict_wcdt_wide_by_path(_long_to_wide(df, "prediction"), path)


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

    Genes with fewer than ``MIN_CORRELATION_N`` valid truth/prediction
    pairs remain undefined (NaN). Genes with insufficient truth variance
    are excluded (NaN). Genes with sufficient valid pairs and variable
    truth but zero spread in predictions are treated as correlation
    failures and assigned 0.0 so aggregate summaries compare the same
    gene set across models.
    """
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    T = tw.values.astype(np.float64)  # (G, S)
    P = pw.values.astype(np.float64)

    n_samples = T.shape[1]
    genes = tw.index

    if n_samples < MIN_CORRELATION_N:
        return pd.DataFrame({
            "n": n_samples, "pearson_r": np.nan, "spearman_rho": np.nan,
            "rmse": np.nan, "mae": np.nan, "truth_var": np.nan,
        }, index=genes)

    # ── Pearson r (vectorised) ──
    # Mask NaN: use per-gene valid counts so that samples with NaN for
    # a given gene are excluded from that gene's sums.
    finite_mask = np.isfinite(T) & np.isfinite(P)  # (G, S)
    T_valid = np.where(finite_mask, T, np.nan)
    P_valid = np.where(finite_mask, P, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        truth_variable = np.nanmax(T_valid, axis=1) > np.nanmin(T_valid, axis=1)
        pred_variable = np.nanmax(P_valid, axis=1) > np.nanmin(P_valid, axis=1)
    T_clean = np.where(finite_mask, T, 0.0)
    P_clean = np.where(finite_mask, P, 0.0)
    n_valid = finite_mask.sum(axis=1).astype(np.float64)  # per-gene
    n_valid = np.where(n_valid >= MIN_CORRELATION_N, n_valid, np.nan)
    sufficient_pairs = np.isfinite(n_valid)
    truth_defined = sufficient_pairs & truth_variable
    pred_constant_pair = truth_defined & ~pred_variable
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
    pearson_r[pred_constant_pair] = 0.0
    pearson_r[~truth_defined] = np.nan

    # ── Spearman rho (vectorised via NaN-aware ranking) ──
    # Rank each gene's truth/pred across samples, excluding NaN entries
    # so that NaN values don't inflate ranks for valid observations.
    T_masked_sp = T_valid
    P_masked_sp = P_valid
    T_ranked = _rankdata_columns_nan(T_masked_sp.T)  # (S, G)
    P_ranked = _rankdata_columns_nan(P_masked_sp.T)
    # Replace NaN ranks with 0; use per-gene valid count for Pearson-on-ranks
    rank_valid = np.isfinite(T_ranked) & np.isfinite(P_ranked)  # (S, G)
    T_rk = np.where(rank_valid, T_ranked, 0.0)
    P_rk = np.where(rank_valid, P_ranked, 0.0)
    n_rk = rank_valid.sum(axis=0).astype(np.float64)  # per-gene valid count
    n_rk_safe = np.where(n_rk >= MIN_CORRELATION_N, n_rk, np.nan)
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
    spearman_rho[pred_constant_pair] = 0.0
    spearman_rho[~truth_defined] = np.nan

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

    if n_genes < MIN_CORRELATION_N:
        return pd.DataFrame({
            "n": n_genes, "pearson_r": np.nan, "spearman_rho": np.nan,
            "rmse": np.nan, "mae": np.nan,
        }, index=samples)

    # ── Pearson r per sample (vectorised along gene axis) ──
    # Mask NaN: use nansum so that genes with NaN in a given sample
    # are excluded from that sample's sums rather than poisoning them.
    finite_mask = np.isfinite(T) & np.isfinite(P)  # (G, S)
    T_valid = np.where(finite_mask, T, np.nan)
    P_valid = np.where(finite_mask, P, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        truth_variable = np.nanmax(T_valid, axis=0) > np.nanmin(T_valid, axis=0)
        pred_variable = np.nanmax(P_valid, axis=0) > np.nanmin(P_valid, axis=0)
    T_clean = np.where(finite_mask, T, 0.0)
    P_clean = np.where(finite_mask, P, 0.0)
    n_valid = finite_mask.sum(axis=0).astype(np.float64)  # per-sample
    n_valid = np.where(n_valid >= MIN_CORRELATION_N, n_valid, np.nan)
    sufficient_pairs = np.isfinite(n_valid)
    truth_defined = sufficient_pairs & truth_variable
    pred_constant_pair = truth_defined & ~pred_variable
    sum_t = (T_clean).sum(axis=0)
    sum_p = (P_clean).sum(axis=0)
    sum_tt = (T_clean * T_clean).sum(axis=0)
    sum_pp = (P_clean * P_clean).sum(axis=0)
    sum_tp = (T_clean * P_clean).sum(axis=0)
    num = n_valid * sum_tp - sum_t * sum_p
    den = np.sqrt((n_valid * sum_tt - sum_t ** 2) * (n_valid * sum_pp - sum_p ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson_r = np.where(den > 0, num / den, np.nan)
    pearson_r[pred_constant_pair] = 0.0
    pearson_r[~truth_defined] = np.nan

    # ── Spearman rho per sample — NaN-aware rank along gene axis ──
    T_masked_sp = T_valid
    P_masked_sp = P_valid
    T_ranked = _rankdata_columns_nan(T_masked_sp)   # (G, S) — rank each column along genes
    P_ranked = _rankdata_columns_nan(P_masked_sp)
    rank_valid = np.isfinite(T_ranked) & np.isfinite(P_ranked)  # (G, S)
    T_rk = np.where(rank_valid, T_ranked, 0.0)
    P_rk = np.where(rank_valid, P_ranked, 0.0)
    n_rk = rank_valid.sum(axis=0).astype(np.float64)  # per-sample valid count
    n_rk_safe = np.where(n_rk >= MIN_CORRELATION_N, n_rk, np.nan)
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
    spearman_rho[pred_constant_pair] = 0.0
    spearman_rho[~truth_defined] = np.nan

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
    dropping any NaN observations. Genes with fewer than
    ``MIN_CORRELATION_N`` valid sample-pairs are excluded entirely.
    Genes with zero truth variance are excluded. Genes with enough valid
    pairs and variable truth but zero prediction spread contribute 0.0.
    The final result is the variance-weighted average of per-gene rho values.
    """
    from scipy.stats import spearmanr as _spearmanr

    n_genes, n_samples = T.shape
    if n_samples < 2 or n_genes < 1:
        return np.nan

    rho_per_gene = np.full(n_genes, np.nan)
    for g in range(n_genes):
        valid = np.isfinite(T[g]) & np.isfinite(P[g])
        if valid.sum() < MIN_CORRELATION_N:
            continue
        tv, pv = T[g, valid], P[g, valid]
        if np.ptp(tv) == 0:
            continue
        if np.ptp(pv) == 0:
            rho_per_gene[g] = 0.0
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
    dropping any NaN observations. Genes with fewer than
    ``MIN_CORRELATION_N`` valid sample-pairs are excluded entirely.
    Genes with zero truth variance are excluded. Genes with enough valid
    pairs and variable truth but zero prediction spread contribute 0.0.
    The final result is the variance-weighted average of per-gene r values.
    """
    n_genes, n_samples = T.shape
    if n_samples < 2 or n_genes < 1:
        return np.nan

    r_per_gene = np.full(n_genes, np.nan)
    for g in range(n_genes):
        valid = np.isfinite(T[g]) & np.isfinite(P[g])
        n = valid.sum()
        if n < MIN_CORRELATION_N:
            continue
        t = T[g, valid]
        p = P[g, valid]
        mean_t = t.mean()
        mean_p = p.mean()
        dt = t - mean_t
        dp = p - mean_p
        if np.dot(dt, dt) < 1e-12:
            continue
        denom = np.sqrt(np.dot(dt, dt) * np.dot(dp, dp))
        if denom < 1e-12:
            r_per_gene[g] = 0.0
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
    if len(t) < MIN_CORRELATION_N:
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
    if len(t) < MIN_CORRELATION_N:
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
    verbose: bool = True,
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
    if verbose:
        print(f"    GVW ...", end=" ", flush=True)
    gvw_r_val = _gvw_r_from_wide(T_arr, P_arr, w_gene)
    gvw_rho_val = _gvw_rho_from_wide(T_arr, P_arr, w_gene)

    if verbose:
        print("gene metrics ...", end=" ", flush=True)
    gene_met = gene_level_metrics(tw, pw, n_workers=n_workers)
    if verbose:
        print("sample metrics ...", end=" ", flush=True)
    sample_met = sample_level_metrics(tw, pw, n_workers=n_workers)
    if verbose:
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
    label_width = 28
    header = (
        f"{'Label':<{label_width}s} {'n_g':>6s} {'n_s':>5s} "
        f"{'GVW-r':>7s} {'GVW-ρ':>7s} "
        f"{'μ_g_r':>7s} {'M_g_r':>7s} {'μ_g_ρ':>7s} {'M_g_ρ':>7s} "
        f"{'μ_t_r':>7s} {'M_t_r':>7s} {'μ_t_ρ':>7s} {'M_t_ρ':>7s}"
    )
    print(header)
    print("-" * len(header))

    def _print_summary_row(summary: Dict[str, Any]) -> None:
        print(
            f"{summary['label']:<{label_width}s} {summary['n_genes']:>6d} {summary['n_samples']:>5d} "
            f"{summary['gvw_r']:>7.4f} {summary['gvw_rho']:>7.4f} "
            f"{summary['mean_gene_r']:>7.4f} {summary['median_gene_r']:>7.4f} "
            f"{summary['mean_gene_rho']:>7.4f} {summary['median_gene_rho']:>7.4f} "
            f"{summary['mean_transcriptome_r']:>7.4f} {summary['median_transcriptome_r']:>7.4f} "
            f"{summary['mean_transcriptome_rho']:>7.4f} {summary['median_transcriptome_rho']:>7.4f}"
        )

    for s in summaries:
        _print_summary_row(s)
        median_summary = s.get("_median_point_summary")
        if isinstance(median_summary, dict):
            _print_summary_row(median_summary)


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


# Keep these analysis-side MDN constants aligned with the stored model outputs.
_MDN_GAMMA_SHAPE = 0.75
_AOR_EPSILON = 1e-6


def _mdn_on_probability(
    p0_flat: np.ndarray,
    pi_gamma_flat: np.ndarray,
    pi_normal_flat: np.ndarray,
    gamma_beta_flat: np.ndarray,
    normal_mean_flat: np.ndarray,
    normal_stdev_flat: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Compute P(Y >= threshold) from the Proteus hurdle-gamma-normal MDN."""
    from scipy.special import erf, gammainc

    beta = np.clip(gamma_beta_flat.astype(np.float64), 1e-12, None)
    sigma = np.clip(normal_stdev_flat.astype(np.float64), 1e-12, None)
    gamma_cdf = gammainc(_MDN_GAMMA_SHAPE, beta * max(threshold, 0.0))
    normal_cdf = 0.5 * (1.0 + erf((threshold - normal_mean_flat) / (sigma * np.sqrt(2.0))))
    cdf_zero = 1.0 if threshold >= 0.0 else 0.0
    cdf = (
        p0_flat.astype(np.float64) * cdf_zero
        + pi_gamma_flat.astype(np.float64) * gamma_cdf
        + pi_normal_flat.astype(np.float64) * normal_cdf
    )
    return np.clip(1.0 - cdf, 0.0, 1.0)


def _mdn_activity_odds_ratio(
    p0_flat: np.ndarray,
    pi_gamma_flat: np.ndarray,
    pi_normal_flat: np.ndarray,
    *,
    epsilon: float = _AOR_EPSILON,
) -> np.ndarray:
    """Compute log2((pi_normal + eps) / (1 - pi_normal + eps)).

    The MDN mixture weights should sum to 1, so the off mass is represented by
    ``1 - pi_normal`` rather than recomputing ``pi_delta + pi_gamma``.
    """
    del p0_flat, pi_gamma_flat

    normal_mass = np.clip(pi_normal_flat.astype(np.float64), 0.0, 1.0)
    off_mass = np.clip(1.0 - normal_mass, 0.0, 1.0)
    numerator = normal_mass + float(epsilon)
    denominator = off_mass + float(epsilon)
    return np.log2(numerator / denominator)


def _dedupe_wide_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated row/column labels from a wide genes x samples matrix.

    Gene-set selections can repeat the same gene label, which materializes
    duplicate rows after ``.loc[...]`` even when the source matrix itself is
    unique. MDN alignment helpers operate on one value per gene-sample pair,
    so they collapse duplicate labels to the first occurrence.
    """
    deduped = df
    if deduped.index.has_duplicates:
        deduped = deduped.loc[~deduped.index.duplicated(keep="first")]
    if deduped.columns.has_duplicates:
        deduped = deduped.loc[:, ~deduped.columns.duplicated(keep="first")]
    return deduped


def mdn_on_probability_wide(
    p0_wide: pd.DataFrame,
    pi_gamma_wide: pd.DataFrame,
    pi_normal_wide: pd.DataFrame,
    gamma_beta_wide: pd.DataFrame,
    normal_mean_wide: pd.DataFrame,
    normal_stdev_wide: pd.DataFrame,
    *,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Compute genes x samples P(Y >= threshold) from the raw MDN outputs."""
    p0_wide = _dedupe_wide_axes(p0_wide)
    pi_gamma_wide = _dedupe_wide_axes(pi_gamma_wide)
    pi_normal_wide = _dedupe_wide_axes(pi_normal_wide)
    gamma_beta_wide = _dedupe_wide_axes(gamma_beta_wide)
    normal_mean_wide = _dedupe_wide_axes(normal_mean_wide)
    normal_stdev_wide = _dedupe_wide_axes(normal_stdev_wide)

    common_genes = (
        p0_wide.index
        .intersection(pi_gamma_wide.index)
        .intersection(pi_normal_wide.index)
        .intersection(gamma_beta_wide.index)
        .intersection(normal_mean_wide.index)
        .intersection(normal_stdev_wide.index)
    )
    common_samples = (
        p0_wide.columns
        .intersection(pi_gamma_wide.columns)
        .intersection(pi_normal_wide.columns)
        .intersection(gamma_beta_wide.columns)
        .intersection(normal_mean_wide.columns)
        .intersection(normal_stdev_wide.columns)
    )
    if len(common_genes) == 0 or len(common_samples) == 0:
        return pd.DataFrame(index=common_genes, columns=common_samples, dtype=np.float64)

    p_on_flat = _mdn_on_probability(
        p0_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
        pi_gamma_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
        pi_normal_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
        gamma_beta_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
        normal_mean_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
        normal_stdev_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
        threshold,
    )
    p_on = p_on_flat.reshape(len(common_genes), len(common_samples))
    return pd.DataFrame(p_on, index=common_genes, columns=common_samples)


def mdn_activity_odds_ratio_wide(
    p0_wide: pd.DataFrame,
    pi_gamma_wide: pd.DataFrame,
    pi_normal_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Compute genes x samples activity odds ratio from MDN mixture weights."""
    p0_wide = _dedupe_wide_axes(p0_wide)
    pi_gamma_wide = _dedupe_wide_axes(pi_gamma_wide)
    pi_normal_wide = _dedupe_wide_axes(pi_normal_wide)

    common_genes = (
        p0_wide.index
        .intersection(pi_gamma_wide.index)
        .intersection(pi_normal_wide.index)
    )
    common_samples = (
        p0_wide.columns
        .intersection(pi_gamma_wide.columns)
        .intersection(pi_normal_wide.columns)
    )
    if len(common_genes) == 0 or len(common_samples) == 0:
        return pd.DataFrame(index=common_genes, columns=common_samples, dtype=np.float64)

    aor_flat = _mdn_activity_odds_ratio(
        p0_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
        pi_gamma_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
        pi_normal_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel(),
    )
    aor = aor_flat.reshape(len(common_genes), len(common_samples))
    return pd.DataFrame(aor, index=common_genes, columns=common_samples)


def mdn_pit_values_wide(
    truth_wide: pd.DataFrame,
    pi_delta: pd.DataFrame,
    pi_gamma: pd.DataFrame,
    pi_normal: pd.DataFrame,
    gamma_beta: pd.DataFrame,
    normal_mean: pd.DataFrame,
    normal_stdev: pd.DataFrame,
    *,
    random_seed: int = 0,
) -> np.ndarray:
    """Compute randomized PIT values from the hurdle-gamma-normal MDN.

    Because the MDN contains a point mass at zero, this uses randomized PIT
    at the jump: for observations with y = 0, returns a value sampled
    uniformly on [F(y-), F(y)].  All randomness is seeded deterministically.
    """
    from scipy.special import erf, gammainc

    common_genes = (
        truth_wide.index.intersection(pi_delta.index)
        .intersection(pi_gamma.index)
        .intersection(pi_normal.index)
        .intersection(gamma_beta.index)
        .intersection(normal_mean.index)
        .intersection(normal_stdev.index)
    )
    common_samples = (
        truth_wide.columns.intersection(pi_delta.columns)
        .intersection(pi_gamma.columns)
        .intersection(pi_normal.columns)
        .intersection(gamma_beta.columns)
        .intersection(normal_mean.columns)
        .intersection(normal_stdev.columns)
    )
    if len(common_genes) == 0 or len(common_samples) == 0:
        return np.asarray([], dtype=np.float64)

    truth_arr = truth_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64)
    p0_arr = pi_delta.loc[common_genes, common_samples].to_numpy(dtype=np.float64)
    pg_arr = pi_gamma.loc[common_genes, common_samples].to_numpy(dtype=np.float64)
    pn_arr = pi_normal.loc[common_genes, common_samples].to_numpy(dtype=np.float64)
    beta_arr = gamma_beta.loc[common_genes, common_samples].to_numpy(dtype=np.float64)
    mu_arr = normal_mean.loc[common_genes, common_samples].to_numpy(dtype=np.float64)
    sigma_arr = normal_stdev.loc[common_genes, common_samples].to_numpy(dtype=np.float64)

    prob_sum = p0_arr + pg_arr + pn_arr
    valid = (
        np.isfinite(truth_arr)
        & np.isfinite(p0_arr)
        & np.isfinite(pg_arr)
        & np.isfinite(pn_arr)
        & np.isfinite(beta_arr)
        & np.isfinite(mu_arr)
        & np.isfinite(sigma_arr)
        & (prob_sum > 0.0)
    )
    if int(valid.sum()) == 0:
        return np.asarray([], dtype=np.float64)

    truth_flat = truth_arr[valid]
    p0_flat = p0_arr[valid] / prob_sum[valid]
    pg_flat = pg_arr[valid] / prob_sum[valid]
    pn_flat = pn_arr[valid] / prob_sum[valid]
    beta_flat = np.clip(beta_arr[valid], 1e-12, None)
    mu_flat = mu_arr[valid]
    sigma_flat = np.clip(sigma_arr[valid], 1e-12, None)

    gamma_cdf = gammainc(_MDN_GAMMA_SHAPE, beta_flat * np.maximum(truth_flat, 0.0))
    normal_cdf = 0.5 * (1.0 + erf((truth_flat - mu_flat) / (sigma_flat * np.sqrt(2.0))))
    cdf_right = (
        p0_flat * (truth_flat >= 0.0).astype(np.float64)
        + pg_flat * gamma_cdf
        + pn_flat * normal_cdf
    )

    zero_mask = np.isclose(truth_flat, 0.0, atol=1e-12)
    pit = cdf_right.copy()
    if bool(zero_mask.any()):
        cdf_left = (
            pg_flat[zero_mask] * 0.0
            + pn_flat[zero_mask] * normal_cdf[zero_mask]
        )
        rng = np.random.default_rng(random_seed)
        u = rng.uniform(size=int(zero_mask.sum()))
        pit[zero_mask] = cdf_left + u * (cdf_right[zero_mask] - cdf_left)

    return np.clip(pit, 0.0, 1.0)


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

    if len(common_samples) < MIN_CORRELATION_N or len(common_genes) < MIN_CORRELATION_N:
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

        # Per-gene valid pair count; exclude genes with < MIN_CORRELATION_N valid samples
        n_valid_per_gene = (np.isfinite(T_arr) & np.isfinite(P_arr)).sum(axis=1)
        usable_genes = n_valid_per_gene >= MIN_CORRELATION_N

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

# The 25 PC-PDX samples in the study: 20 CV/development + 5 holdout
_PC_PDX_CV = [
    "LuCaP_145-1", "LuCaP_189-4", "LuCaP_70CR", "LuCaP_96",       # F1
    "LuCaP_173-1", "LuCaP_136CR", "LuCaP_167CR", "LuCaP_170-3",   # F2
    "LuCaP_208-4", "LuCaP_147CR", "LuCaP_78", "LuCaP_96CR",       # F3
    "LuCaP_145-2", "LuCaP_141", "LuCaP_35", "LuCaP_70",           # F4
    "LuCaP_49", "LuCaP_35CR", "LuCaP_58", "LuCaP_176",            # F5
]
_PC_PDX_HO = ["LuCaP_136", "LuCaP_167", "LuCaP_176CR", "LuCaP_92", "LuCaP_93"]
PC_PDX_STUDY = sorted(_PC_PDX_CV + _PC_PDX_HO)

# The 25 SCLC-PDX samples in the study: 20 CV/development + 5 holdout
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


def _build_plca_training_truth_by_source() -> Dict[str, pd.DataFrame]:
    """Return the direct truth matrices for the four PLCA training sources."""
    truth_by_source: Dict[str, pd.DataFrame] = {}
    for cohort in ("CRPC-RA", "SCLC", "PC-PDX", "SCLC-PDX"):
        truth_wide = load_truth(cohort)
        if cohort == "PC-PDX":
            keep = [sample for sample in PC_PDX_STUDY if sample in truth_wide.columns]
            truth_wide = truth_wide.loc[:, keep]
        elif cohort == "SCLC-PDX":
            keep = [sample for sample in SCLC_PDX_STUDY if sample in truth_wide.columns]
            truth_wide = truth_wide.loc[:, keep]
        truth_by_source[cohort] = truth_wide
    return truth_by_source


def _compute_truth_dispersion_table(truth_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute direct per-gene dispersion statistics from a truth matrix."""
    numeric_truth = truth_wide.apply(pd.to_numeric, errors="coerce")
    q1 = numeric_truth.quantile(0.25, axis=1)
    q3 = numeric_truth.quantile(0.75, axis=1)
    return pd.DataFrame(
        {
            "stdev": numeric_truth.std(axis=1, ddof=1),
            "variance": numeric_truth.var(axis=1, ddof=1),
            "range": numeric_truth.max(axis=1) - numeric_truth.min(axis=1),
            "mid_quartile_range": q3 - q1,
        }
    )


def export_plca_training_truth_dispersion_table(
    out_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Export direct truth-data dispersion stats for the PLCA training pool.

    The aggregate `training` columns combine all truth samples from CRPC-RA,
    SCLC, PC-PDX, and SCLC-PDX. The PDX cohorts are restricted to the
    canonical study sample lists so the export stays aligned with the training
    sources used throughout the analysis pipeline.

    Args:
        out_path: Optional output TSV path. Defaults to
            ``analysis/results/plca_training_truth_dispersion.tsv``.

    Returns:
        DataFrame with one row per gene and per-group columns for
        standard deviation, variance, range, and mid-quartile range.
    """
    truth_by_source = _build_plca_training_truth_by_source()
    training_truth = pd.concat(
        [truth_by_source[cohort] for cohort in ("CRPC-RA", "SCLC", "PC-PDX", "SCLC-PDX")],
        axis=1,
    )
    training_truth = training_truth.loc[:, ~training_truth.columns.duplicated()]

    groups: Dict[str, pd.DataFrame] = {
        "training": training_truth,
        "CRPC-RA": truth_by_source["CRPC-RA"],
        "SCLC": truth_by_source["SCLC"],
        "PC-PDX": truth_by_source["PC-PDX"],
        "SCLC-PDX": truth_by_source["SCLC-PDX"],
    }

    table = pd.DataFrame(index=training_truth.index.sort_values())
    for group_name, truth_wide in groups.items():
        prefix = group_name.lower().replace("-", "_")
        group_table = _compute_truth_dispersion_table(truth_wide).add_prefix(f"{prefix}_")
        table = table.join(group_table, how="left")

    table.index.name = "gene"
    table = table.reset_index()

    if out_path is None:
        out_path = PROJECT_ROOT / "analysis" / "results" / "plca_training_truth_dispersion.tsv"
    out_path = Path(out_path)
    ensure_dir(str(out_path.parent))
    table.to_csv(out_path, sep="\t", index=False)

    return table


def export_plca_training_variance_table(
    out_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Export the exact PLCA tumor-expression variance table used in training.

    This follows the full-train PLCA recipe in
    ``config/experiments/plca_final-training.yaml``:
    non-PHT PLCA samples are collected from the splits TSV, tumor-expression
    variance is computed from ``flat/tumor_exp``, and the loss weights are the
    same variances clipped to ``[0.1, 10.0]``.

    Args:
        out_path: Optional output TSV path. Defaults to
            ``analysis/results/plca_training_gene_variance.tsv``.

    Returns:
        DataFrame with columns ``gene``, ``variance_raw``, and
        ``variance_clipped``.
    """
    from config.experiment_spec import load_experiment_spec
    from proteus_utils.data_helpers import collect_zarr_paths
    from proteus_utils.proteus_utils import _compute_gene_variances

    spec = load_experiment_spec(str(PROJECT_ROOT / "config" / "experiments" / "plca_final-training.yaml"))
    training_set_tsv = PROJECT_ROOT / spec.data.training_set
    zarr_base = spec.data.zarr_base

    zarr_paths = collect_zarr_paths(
        str(training_set_tsv),
        zarr_base,
        exclude_cohorts={"PHT"},
    )
    if not zarr_paths:
        raise FileNotFoundError(
            "No PLCA training zarr stores were found for variance export under "
            f"{zarr_base} using {training_set_tsv}"
        )

    raw_variance, clipped_variance = _compute_gene_variances(
        zarr_paths,
        label_key="flat/tumor_exp",
        clip_min=0.1,
        clip_max=10.0,
    )
    genes = sorted(raw_variance)
    table = pd.DataFrame(
        {
            "gene": genes,
            "variance_raw": [raw_variance[g] for g in genes],
            "variance_clipped": [clipped_variance[g] for g in genes],
        }
    )

    if out_path is None:
        out_path = PROJECT_ROOT / "analysis" / "results" / "plca_training_gene_variance.tsv"
    out_path = Path(out_path)
    ensure_dir(str(out_path.parent))
    table.to_csv(out_path, sep="\t", index=False)

    return table


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
    nominal_levels: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90),
    interval_nominal: float = 0.90,
) -> Dict[str, List[float]]:
    """Compute empirical coverage at multiple nominal levels.

    Assumes ``ci_lo`` and ``ci_hi`` correspond to a directly modelled
    interval with nominal coverage ``interval_nominal``. Narrower bands are
    interpolated by shrinking the interval symmetrically around the
    prediction mean as a diagnostic only.

    Returns dict with keys 'nominal' and 'empirical'.
    """
    if not 0.0 < interval_nominal <= 1.0:
        raise ValueError(
            f"interval_nominal must lie in (0, 1], got {interval_nominal!r}"
        )
    if any(level > interval_nominal for level in nominal_levels):
        raise ValueError(
            "nominal_levels cannot exceed interval_nominal for a single "
            "directly modelled interval"
        )

    mask = (np.isfinite(truth_flat) & np.isfinite(ci_lo_flat)
            & np.isfinite(ci_hi_flat) & np.isfinite(pred_flat))
    y = truth_flat[mask]
    lo = ci_lo_flat[mask]
    hi = ci_hi_flat[mask]
    mu = pred_flat[mask]
    if len(y) < 10:
        return {"nominal": list(nominal_levels),
                "empirical": [np.nan] * len(nominal_levels)}

    empiricals = []
    for level in nominal_levels:
        scale = level / interval_nominal
        adj_lo = mu - scale * (mu - lo)
        adj_hi = mu + scale * (hi - mu)
        coverage = float(((y >= adj_lo) & (y <= adj_hi)).sum() / len(y))
        empiricals.append(coverage)
    return {"nominal": list(nominal_levels), "empirical": empiricals}


# ---------------------------------------------------------------------------
# Risk coverage curve computation (on/off gene calling)
# ---------------------------------------------------------------------------

_ONOFF_THRESHOLD_DEFAULT = 2.0  # log2(CPM+1) on/off boundary


def compute_risk_coverage_data(
    truth_flat: np.ndarray,
    pred_flat: np.ndarray,
    ci_lo_flat: Optional[np.ndarray] = None,
    ci_hi_flat: Optional[np.ndarray] = None,
    p0_flat: Optional[np.ndarray] = None,
    pi_gamma_flat: Optional[np.ndarray] = None,
    pi_normal_flat: Optional[np.ndarray] = None,
    gamma_beta_flat: Optional[np.ndarray] = None,
    normal_mean_flat: Optional[np.ndarray] = None,
    normal_stdev_flat: Optional[np.ndarray] = None,
    threshold: float = _ONOFF_THRESHOLD_DEFAULT,
    n_steps: int = 200,
) -> Dict[str, np.ndarray]:
    """Compute coverage-vs-PPV and coverage-vs-FPR as confidence varies.

    For Proteus MDN outputs, confidence is computed from the full predictive
    distribution: first derive ``P(Y >= threshold)`` from the hurdle-gamma-
    normal mixture, then rank observations by posterior confidence
    ``max(P_on, 1 - P_on)``.  At each retention level the retained subset is
    evaluated for PPV and FPR of the binary on/off call implied by
    ``P_on >= 0.5``.

    If full MDN parameters are unavailable but CIs are present, this falls
    back to the legacy CI-width ranking.  Models without either uncertainty
    representation return a single point at 100% coverage.

    For models without CIs, returns a single (coverage=1.0) point.

    Returns
    -------
    dict with keys ``coverage``, ``ppv``, ``fpr``, ``ppv_auc``,
    ``fpr_auc``.
    """
    valid = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    if ci_lo_flat is not None and ci_hi_flat is not None:
        valid &= np.isfinite(ci_lo_flat) & np.isfinite(ci_hi_flat)
    has_mdn = all(
        arr is not None
        for arr in (
            p0_flat,
            pi_gamma_flat,
            pi_normal_flat,
            gamma_beta_flat,
            normal_mean_flat,
            normal_stdev_flat,
        )
    )
    if has_mdn:
        valid &= (
            np.isfinite(p0_flat)
            & np.isfinite(pi_gamma_flat)
            & np.isfinite(pi_normal_flat)
            & np.isfinite(gamma_beta_flat)
            & np.isfinite(normal_mean_flat)
            & np.isfinite(normal_stdev_flat)
        )
    t = truth_flat[valid]
    p = pred_flat[valid]

    def _build_curve(
        coverage_vals: np.ndarray,
        ppv_vals: np.ndarray,
        fpr_vals: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        order = np.argsort(coverage_vals)
        coverage_sorted = coverage_vals[order]
        ppv_sorted = ppv_vals[order]
        fpr_sorted = fpr_vals[order]
        if len(coverage_sorted) > 1:
            ppv_auc = float(np.trapezoid(ppv_sorted, coverage_sorted))
            fpr_auc = float(np.trapezoid(fpr_sorted, coverage_sorted))
        else:
            ppv_auc = float("nan")
            fpr_auc = float("nan")
        return {
            "coverage": coverage_sorted,
            "ppv": ppv_sorted,
            "fpr": fpr_sorted,
            "ppv_auc": ppv_auc,
            "fpr_auc": fpr_auc,
        }

    y_true = (t >= threshold).astype(int)

    if has_mdn:
        p_on = _mdn_on_probability(
            p0_flat[valid],
            pi_gamma_flat[valid],
            pi_normal_flat[valid],
            gamma_beta_flat[valid],
            normal_mean_flat[valid],
            normal_stdev_flat[valid],
            threshold,
        )
        confidence = np.maximum(p_on, 1.0 - p_on)
        if len(confidence) < 10:
            pred_hard = (p_on >= 0.5).astype(np.float64)
            ppv, fpr = _ppv_fpr(y_true, pred_hard, threshold=0.5)
            return _build_curve(
                np.array([1.0]),
                np.array([ppv]),
                np.array([fpr]),
            )

        order = np.argsort(-confidence, kind="mergesort")
        y_true = y_true[order]
        p_on = p_on[order]

        min_count = min(10, len(y_true))
        retained_counts = np.unique(
            np.linspace(min_count, len(y_true), n_steps).round().astype(int)
        )

        coverages, ppvs, fprs = [], [], []
        n_total = len(y_true)
        for retained in retained_counts:
            if retained < 10:
                continue
            ppv, fpr_val = _ppv_fpr(
                y_true[:retained],
                p_on[:retained],
                threshold=0.5,
            )
            coverages.append(retained / n_total)
            ppvs.append(ppv)
            fprs.append(fpr_val)

        return _build_curve(
            np.asarray(coverages, dtype=np.float64),
            np.asarray(ppvs, dtype=np.float64),
            np.asarray(fprs, dtype=np.float64),
        )

    # No CI → single point at full coverage
    if ci_lo_flat is None or ci_hi_flat is None:
        ppv, fpr = _ppv_fpr(y_true, p, threshold)
        return _build_curve(
            np.array([1.0]),
            np.array([ppv]),
            np.array([fpr]),
        )

    lo = ci_lo_flat[valid]
    hi = ci_hi_flat[valid]
    widths = hi - lo

    if len(widths) < 10:
        ppv, fpr = _ppv_fpr(y_true, p, threshold)
        return _build_curve(
            np.array([1.0]),
            np.array([ppv]),
            np.array([fpr]),
        )

    order = np.argsort(widths, kind="mergesort")
    y_true = y_true[order]
    p = p[order]

    min_count = min(10, len(y_true))
    retained_counts = np.unique(
        np.linspace(min_count, len(y_true), n_steps).round().astype(int)
    )

    coverages, ppvs, fprs = [], [], []
    n_total = len(y_true)
    for retained in retained_counts:
        if retained < 10:
            continue
        ppv, fpr_val = _ppv_fpr(y_true[:retained], p[:retained], threshold)
        coverages.append(retained / n_total)
        ppvs.append(ppv)
        fprs.append(fpr_val)

    return _build_curve(
        np.asarray(coverages, dtype=np.float64),
        np.asarray(ppvs, dtype=np.float64),
        np.asarray(fprs, dtype=np.float64),
    )


def compute_active_confidence_curve(
    truth_flat: np.ndarray,
    active_confidence_flat: np.ndarray,
    *,
    threshold: float = _ONOFF_THRESHOLD_DEFAULT,
    target_ppv: float = 0.90,
) -> Dict[str, Any]:
    """Sweep active-confidence thresholds for ON calling against truth.

    ``active_confidence_flat`` is interpreted as ``P(Y >= threshold)``.
    The selected threshold is the lowest posterior-on threshold whose
    retained ON calls achieve ``PPV >= target_ppv``. If no threshold meets
    the target, the threshold with the highest achievable PPV is selected,
    breaking ties toward lower thresholds to preserve coverage.
    """
    valid = np.isfinite(truth_flat) & np.isfinite(active_confidence_flat)
    t = truth_flat[valid]
    p_on = np.clip(active_confidence_flat[valid].astype(np.float64), 0.0, 1.0)
    n_total = len(t)
    if n_total == 0:
        return {
            "threshold": np.array([], dtype=np.float64),
            "coverage": np.array([], dtype=np.float64),
            "ppv": np.array([], dtype=np.float64),
            "fdr": np.array([], dtype=np.float64),
            "n_predicted_on": np.array([], dtype=np.int64),
            "selected_threshold": float("nan"),
            "selected_coverage": float("nan"),
            "selected_ppv": float("nan"),
            "selected_fdr": float("nan"),
            "selected_n_predicted_on": 0,
            "target_ppv": float(target_ppv),
            "target_met": False,
            "n_total": 0,
            "n_true_on": 0,
        }

    y_true = (t >= threshold).astype(np.int64)
    order = np.argsort(-p_on, kind="mergesort")
    p_sorted = p_on[order]
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    last_idx = np.flatnonzero(np.r_[p_sorted[1:] != p_sorted[:-1], True])
    thresholds = p_sorted[last_idx]
    tp = tp_cum[last_idx].astype(np.float64)
    fp = fp_cum[last_idx].astype(np.float64)
    n_predicted_on = (tp + fp).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppv = np.divide(tp, n_predicted_on, out=np.full_like(tp, np.nan), where=n_predicted_on > 0)
    fdr = 1.0 - ppv
    coverage = n_predicted_on.astype(np.float64) / float(n_total)

    meets_target = np.isfinite(ppv) & (ppv >= float(target_ppv)) & (n_predicted_on > 0)
    if meets_target.any():
        selected_idx = int(np.flatnonzero(meets_target)[-1])
        target_met = True
    else:
        valid_ppv = np.isfinite(ppv) & (n_predicted_on > 0)
        if valid_ppv.any():
            best_ppv = float(np.nanmax(ppv[valid_ppv]))
            selected_idx = int(np.flatnonzero(valid_ppv & np.isclose(ppv, best_ppv))[-1])
        else:
            selected_idx = 0
        target_met = False

    return {
        "threshold": thresholds.astype(np.float64),
        "coverage": coverage.astype(np.float64),
        "ppv": ppv.astype(np.float64),
        "fdr": fdr.astype(np.float64),
        "n_predicted_on": n_predicted_on,
        "selected_threshold": float(thresholds[selected_idx]),
        "selected_coverage": float(coverage[selected_idx]),
        "selected_ppv": float(ppv[selected_idx]),
        "selected_fdr": float(fdr[selected_idx]),
        "selected_n_predicted_on": int(n_predicted_on[selected_idx]),
        "target_ppv": float(target_ppv),
        "target_met": bool(target_met),
        "n_total": int(n_total),
        "n_true_on": int(y_true.sum()),
    }


def compute_activity_odds_ratio_on_curve(
    truth_flat: np.ndarray,
    activity_odds_ratio_flat: np.ndarray,
    *,
    threshold: float = _ONOFF_THRESHOLD_DEFAULT,
    target_ppv: float = 0.90,
) -> Dict[str, Any]:
    """Sweep upper AOR thresholds for ON calling against truth."""
    valid = np.isfinite(truth_flat) & np.isfinite(activity_odds_ratio_flat)
    t = truth_flat[valid]
    aor = activity_odds_ratio_flat[valid].astype(np.float64)
    n_total = len(t)
    if n_total == 0:
        return {
            "threshold": np.array([], dtype=np.float64),
            "coverage": np.array([], dtype=np.float64),
            "ppv": np.array([], dtype=np.float64),
            "fdr": np.array([], dtype=np.float64),
            "n_predicted_on": np.array([], dtype=np.int64),
            "selected_threshold": float("nan"),
            "selected_coverage": float("nan"),
            "selected_ppv": float("nan"),
            "selected_fdr": float("nan"),
            "selected_n_predicted_on": 0,
            "target_ppv": float(target_ppv),
            "target_met": False,
            "n_total": 0,
            "n_true_on": 0,
        }

    y_true = (t >= threshold).astype(np.int64)
    order = np.argsort(-aor, kind="mergesort")
    aor_sorted = aor[order]
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    last_idx = np.flatnonzero(np.r_[aor_sorted[1:] != aor_sorted[:-1], True])
    thresholds = aor_sorted[last_idx]
    tp = tp_cum[last_idx].astype(np.float64)
    fp = fp_cum[last_idx].astype(np.float64)
    n_predicted_on = (tp + fp).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppv = np.divide(tp, n_predicted_on, out=np.full_like(tp, np.nan), where=n_predicted_on > 0)
    fdr = 1.0 - ppv
    coverage = n_predicted_on.astype(np.float64) / float(n_total)

    meets_target = np.isfinite(ppv) & (ppv >= float(target_ppv)) & (n_predicted_on > 0)
    if meets_target.any():
        selected_idx = int(np.flatnonzero(meets_target)[-1])
        target_met = True
    else:
        valid_ppv = np.isfinite(ppv) & (n_predicted_on > 0)
        if valid_ppv.any():
            best_ppv = float(np.nanmax(ppv[valid_ppv]))
            selected_idx = int(np.flatnonzero(valid_ppv & np.isclose(ppv, best_ppv))[-1])
        else:
            selected_idx = 0
        target_met = False

    return {
        "threshold": thresholds.astype(np.float64),
        "coverage": coverage.astype(np.float64),
        "ppv": ppv.astype(np.float64),
        "fdr": fdr.astype(np.float64),
        "n_predicted_on": n_predicted_on,
        "selected_threshold": float(thresholds[selected_idx]),
        "selected_coverage": float(coverage[selected_idx]),
        "selected_ppv": float(ppv[selected_idx]),
        "selected_fdr": float(fdr[selected_idx]),
        "selected_n_predicted_on": int(n_predicted_on[selected_idx]),
        "target_ppv": float(target_ppv),
        "target_met": bool(target_met),
        "n_total": int(n_total),
        "n_true_on": int(y_true.sum()),
    }


def compute_activity_odds_ratio_off_curve(
    truth_flat: np.ndarray,
    activity_odds_ratio_flat: np.ndarray,
    *,
    threshold: float = _ONOFF_THRESHOLD_DEFAULT,
    target_npv: float = 0.90,
) -> Dict[str, Any]:
    """Sweep lower AOR thresholds for OFF calling against truth."""
    valid = np.isfinite(truth_flat) & np.isfinite(activity_odds_ratio_flat)
    t = truth_flat[valid]
    aor = activity_odds_ratio_flat[valid].astype(np.float64)
    n_total = len(t)
    if n_total == 0:
        return {
            "threshold": np.array([], dtype=np.float64),
            "coverage": np.array([], dtype=np.float64),
            "npv": np.array([], dtype=np.float64),
            "for": np.array([], dtype=np.float64),
            "n_predicted_off": np.array([], dtype=np.int64),
            "selected_threshold": float("nan"),
            "selected_coverage": float("nan"),
            "selected_npv": float("nan"),
            "selected_for": float("nan"),
            "selected_n_predicted_off": 0,
            "target_npv": float(target_npv),
            "target_met": False,
            "n_total": 0,
            "n_true_off": 0,
        }

    truth_off = (t < threshold).astype(np.int64)
    order = np.argsort(aor, kind="mergesort")
    aor_sorted = aor[order]
    truth_off_sorted = truth_off[order]

    tn_cum = np.cumsum(truth_off_sorted == 1)
    fn_cum = np.cumsum(truth_off_sorted == 0)
    last_idx = np.flatnonzero(np.r_[aor_sorted[1:] != aor_sorted[:-1], True])
    thresholds = aor_sorted[last_idx]
    tn = tn_cum[last_idx].astype(np.float64)
    fn = fn_cum[last_idx].astype(np.float64)
    n_predicted_off = (tn + fn).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        npv = np.divide(tn, n_predicted_off, out=np.full_like(tn, np.nan), where=n_predicted_off > 0)
    false_omission_rate = 1.0 - npv
    coverage = n_predicted_off.astype(np.float64) / float(n_total)

    meets_target = np.isfinite(npv) & (npv >= float(target_npv)) & (n_predicted_off > 0)
    if meets_target.any():
        selected_idx = int(np.flatnonzero(meets_target)[-1])
        target_met = True
    else:
        valid_npv = np.isfinite(npv) & (n_predicted_off > 0)
        if valid_npv.any():
            best_npv = float(np.nanmax(npv[valid_npv]))
            selected_idx = int(np.flatnonzero(valid_npv & np.isclose(npv, best_npv))[-1])
        else:
            selected_idx = 0
        target_met = False

    return {
        "threshold": thresholds.astype(np.float64),
        "coverage": coverage.astype(np.float64),
        "npv": npv.astype(np.float64),
        "for": false_omission_rate.astype(np.float64),
        "n_predicted_off": n_predicted_off,
        "selected_threshold": float(thresholds[selected_idx]),
        "selected_coverage": float(coverage[selected_idx]),
        "selected_npv": float(npv[selected_idx]),
        "selected_for": float(false_omission_rate[selected_idx]),
        "selected_n_predicted_off": int(n_predicted_off[selected_idx]),
        "target_npv": float(target_npv),
        "target_met": bool(target_met),
        "n_total": int(n_total),
        "n_true_off": int(truth_off.sum()),
    }


def _ppv_fpr(
    y_true: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    """Compute PPV and FPR for a binary on/off call."""
    y_pred = (pred >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    ppv = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    return ppv, fpr


# ---------------------------------------------------------------------------
# On/off confusion summary (3-way: ON / OFF / INDETERMINATE)
# ---------------------------------------------------------------------------

def compute_onoff_confusion(
    truth_flat: np.ndarray,
    pred_flat: np.ndarray,
    ci_lo_flat: Optional[np.ndarray] = None,
    ci_hi_flat: Optional[np.ndarray] = None,
    activity_odds_ratio_flat: Optional[np.ndarray] = None,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    active_confidence_flat: Optional[np.ndarray] = None,
    active_threshold: float = 0.5,
    threshold: float = _ONOFF_THRESHOLD_DEFAULT,
) -> Dict[str, Any]:
    """Compute 3-way on/off/indeterminate confusion summary.

    Classification rules (using the on/off threshold, default 2.0):

    * **Models without CIs** — every gene-sample pair is called ON
      (pred >= threshold) or OFF (pred < threshold).  No indeterminate.
                * **Models with activity odds ratio thresholds** — a pair is
                    INDETERMINATE when ``lower_threshold <= AOR <= upper_threshold``.
                    Confident ON requires ``AOR > upper_threshold``; confident OFF
                    requires ``AOR < lower_threshold``.
        * **Models with active-confidence** — a pair is INDETERMINATE when
            ``1 - active_threshold < P_on < active_threshold``. Confident ON
            requires ``P_on >= active_threshold``; confident OFF requires
            ``P_on <= 1 - active_threshold``.
        * **Models with 90% CIs but no active-confidence** — falls back to the
            legacy CI-straddle rule.

    Returns a dict with:
        n_total, n_on, n_off, n_indet,
        tp, fp, tn, fn  (on the confident subset only),
        accuracy, ppv, npv, fpr, fnr, coverage (fraction not indet).
    """
    valid = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    if activity_odds_ratio_flat is not None:
        valid &= np.isfinite(activity_odds_ratio_flat)
    if active_confidence_flat is not None:
        valid &= np.isfinite(active_confidence_flat)
    if ci_lo_flat is not None and ci_hi_flat is not None:
        valid &= np.isfinite(ci_lo_flat) & np.isfinite(ci_hi_flat)
    t = truth_flat[valid]
    p = pred_flat[valid]
    n_total = len(t)
    if n_total == 0:
        return {"n_total": 0}

    truth_on = t >= threshold
    n_truth_on = int(truth_on.sum())
    n_truth_off = int((~truth_on).sum())

    if (
        activity_odds_ratio_flat is not None
        and lower_threshold is not None
        and upper_threshold is not None
        and np.isfinite(lower_threshold)
        and np.isfinite(upper_threshold)
    ):
        aor = activity_odds_ratio_flat[valid].astype(np.float64)
        conf_on = aor > float(upper_threshold)
        conf_off = aor < float(lower_threshold)
        indet = ~(conf_on | conf_off)

        n_indet = int(indet.sum())
        n_on = int(conf_on.sum())
        n_off = int(conf_off.sum())

        confident = ~indet
        n_conf = int(confident.sum())
        if n_conf == 0:
            return {
                "n_total": n_total,
                "n_truth_on": n_truth_on,
                "n_truth_off": n_truth_off,
                "n_on": n_on, "n_off": n_off, "n_indet": n_indet,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "accuracy": float("nan"), "ppv": float("nan"),
                "npv": float("nan"), "fpr": float("nan"), "fnr": float("nan"),
                "sensitivity": float("nan"),
                "coverage": 0.0,
            }

        pred_on_conf = conf_on[confident]
        truth_on_conf = truth_on[confident]
        tp = int((pred_on_conf & truth_on_conf).sum())
        fp = int((pred_on_conf & (~truth_on_conf)).sum())
        fn = int(((~pred_on_conf) & truth_on_conf).sum())
        tn = int(((~pred_on_conf) & (~truth_on_conf)).sum())
        return {
            "n_total": n_total,
            "n_truth_on": n_truth_on,
            "n_truth_off": n_truth_off,
            "n_on": n_on, "n_off": n_off, "n_indet": n_indet,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / max(n_conf, 1),
            "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1),
            "fpr": fp / max(fp + tn, 1),
            "fnr": fn / max(fn + tp, 1),
            "sensitivity": tp / max(tp + fn, 1),
            "coverage": n_conf / max(n_total, 1),
        }

    if active_confidence_flat is not None:
        p_on = np.clip(active_confidence_flat[valid].astype(np.float64), 0.0, 1.0)
        confidence_threshold = float(np.clip(active_threshold, 0.5, 1.0))
        conf_on = p_on >= confidence_threshold
        conf_off = p_on <= (1.0 - confidence_threshold)
        indet = ~(conf_on | conf_off)

        n_indet = int(indet.sum())
        n_on = int(conf_on.sum())
        n_off = int(conf_off.sum())

        confident = ~indet
        n_conf = int(confident.sum())
        if n_conf == 0:
            return {
                "n_total": n_total,
                "n_truth_on": n_truth_on,
                "n_truth_off": n_truth_off,
                "n_on": n_on, "n_off": n_off, "n_indet": n_indet,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "accuracy": float("nan"), "ppv": float("nan"),
                "npv": float("nan"), "fpr": float("nan"), "fnr": float("nan"),
                "sensitivity": float("nan"),
                "coverage": 0.0,
            }

        pred_on_conf = conf_on[confident]
        truth_on_conf = truth_on[confident]
        tp = int((pred_on_conf & truth_on_conf).sum())
        fp = int((pred_on_conf & (~truth_on_conf)).sum())
        fn = int(((~pred_on_conf) & truth_on_conf).sum())
        tn = int(((~pred_on_conf) & (~truth_on_conf)).sum())
        return {
            "n_total": n_total,
            "n_truth_on": n_truth_on,
            "n_truth_off": n_truth_off,
            "n_on": n_on, "n_off": n_off, "n_indet": n_indet,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / max(n_conf, 1),
            "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1),
            "fpr": fp / max(fp + tn, 1),
            "fnr": fn / max(fn + tp, 1),
            "sensitivity": tp / max(tp + fn, 1),
            "coverage": n_conf / max(n_total, 1),
        }

    if ci_lo_flat is None or ci_hi_flat is None:
        # No CIs: binary ON/OFF
        pred_on = p >= threshold
        tp = int(((pred_on) & (truth_on)).sum())
        fp = int(((pred_on) & (~truth_on)).sum())
        fn = int(((~pred_on) & (truth_on)).sum())
        tn = int(((~pred_on) & (~truth_on)).sum())
        return {
            "n_total": n_total,
            "n_truth_on": n_truth_on,
            "n_truth_off": n_truth_off,
            "n_on": int(pred_on.sum()),
            "n_off": int((~pred_on).sum()),
            "n_indet": 0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / max(n_total, 1),
            "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1),
            "fpr": fp / max(fp + tn, 1),
            "fnr": fn / max(fn + tp, 1),
            "sensitivity": tp / max(tp + fn, 1),
            "coverage": 1.0,
        }

    lo = ci_lo_flat[valid]
    hi = ci_hi_flat[valid]

    # 3-way classification
    indet = (lo < threshold) & (hi > threshold)
    conf_on = lo >= threshold
    conf_off = hi < threshold

    n_indet = int(indet.sum())
    n_on = int(conf_on.sum())
    n_off = int(conf_off.sum())

    # Metrics on confident subset only
    confident = ~indet
    n_conf = int(confident.sum())
    if n_conf == 0:
        return {
            "n_total": n_total,
            "n_truth_on": n_truth_on,
            "n_truth_off": n_truth_off,
            "n_on": n_on, "n_off": n_off, "n_indet": n_indet,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "accuracy": float("nan"), "ppv": float("nan"),
            "npv": float("nan"), "fpr": float("nan"), "fnr": float("nan"),
            "sensitivity": float("nan"),
            "coverage": 0.0,
        }

    # Among confident pairs, the call is ON if ci_lo >= threshold, OFF if ci_hi < threshold
    pred_on_conf = conf_on[confident]
    truth_on_conf = truth_on[confident]
    tp = int(((pred_on_conf) & (truth_on_conf)).sum())
    fp = int(((pred_on_conf) & (~truth_on_conf)).sum())
    fn = int(((~pred_on_conf) & (truth_on_conf)).sum())
    tn = int(((~pred_on_conf) & (~truth_on_conf)).sum())

    return {
        "n_total": n_total,
        "n_truth_on": n_truth_on,
        "n_truth_off": n_truth_off,
        "n_on": n_on, "n_off": n_off, "n_indet": n_indet,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / max(n_conf, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "fnr": fn / max(fn + tp, 1),
        "sensitivity": tp / max(tp + fn, 1),
        "coverage": n_conf / max(n_total, 1),
    }


# ---------------------------------------------------------------------------
# PAM50 centroid classifier (Parker et al. 2009, JCO 27:1160-1167)
# ---------------------------------------------------------------------------

# Centroid values from Parker et al. 2009, Supplementary Table 3.
# Each entry: gene → [Basal, HER2, LumA, LumB, Normal]
# Normal is retained in the raw centroids for completeness but excluded
# from classification (see _PAM50_CLINICAL_SUBTYPES) because it
# represents normal tissue contamination / low-TFx artefacts and is not
# clinically actionable.
_PAM50_SUBTYPES = ("Basal", "HER2", "LumA", "LumB", "Normal")
_PAM50_CLINICAL_SUBTYPES = ("Basal", "HER2", "LumA", "LumB")

_PAM50_CENTROIDS_RAW: Dict[str, List[float]] = {
    "UBE2T":   [ 0.635,  0.286, -0.505,  0.215, -0.462],
    "BIRC5":   [ 0.577,  0.315, -0.577,  0.290, -0.400],
    "NUF2":    [ 0.558,  0.285, -0.524,  0.251, -0.383],
    "CDC6":    [ 0.508,  0.190, -0.371,  0.156, -0.326],
    "CCNB1":   [ 0.608,  0.257, -0.528,  0.244, -0.383],
    "TYMS":    [ 0.502,  0.192, -0.408,  0.180, -0.291],
    "MYBL2":   [ 0.534,  0.250, -0.480,  0.224, -0.341],
    "CEP55":   [ 0.582,  0.258, -0.521,  0.254, -0.370],
    "MELK":    [ 0.599,  0.226, -0.504,  0.218, -0.349],
    "NDC80":   [ 0.534,  0.253, -0.491,  0.240, -0.339],
    "RRM2":    [ 0.485,  0.179, -0.408,  0.171, -0.256],
    "UBE2C":   [ 0.591,  0.268, -0.535,  0.256, -0.370],
    "CENPF":   [ 0.525,  0.207, -0.440,  0.192, -0.304],
    "PTTG1":   [ 0.520,  0.264, -0.510,  0.258, -0.337],
    "EXO1":    [ 0.528,  0.214, -0.451,  0.213, -0.323],
    "ORC6":    [ 0.440,  0.181, -0.374,  0.155, -0.236],
    "ANLN":    [ 0.550,  0.212, -0.459,  0.208, -0.325],
    "CCNE1":   [ 0.394,  0.170, -0.300,  0.086, -0.219],
    "CDC20":   [ 0.560,  0.254, -0.505,  0.241, -0.350],
    "MKI67":   [ 0.553,  0.226, -0.482,  0.223, -0.329],
    "KIF2C":   [ 0.527,  0.246, -0.480,  0.233, -0.331],
    "ACTR3B":  [ 0.283,  0.063, -0.187,  0.032, -0.126],
    "MYC":     [ 0.273,  0.017, -0.109, -0.019, -0.101],
    "EGFR":    [ 0.418, -0.105, -0.145, -0.143, -0.004],
    "KRT5":    [ 0.449, -0.145, -0.157, -0.163,  0.029],
    "PHGDH":   [ 0.342, -0.055, -0.159, -0.096, -0.006],
    "CDH3":    [ 0.427, -0.038, -0.225, -0.089, -0.045],
    "MIA":     [ 0.148, -0.025, -0.055, -0.029, -0.022],
    "KRT17":   [ 0.444, -0.143, -0.155, -0.157,  0.018],
    "FOXC1":   [ 0.375, -0.109, -0.136, -0.130,  0.005],
    "SFRP1":   [-0.095, -0.174,  0.064, -0.093,  0.276],
    "KRT14":   [ 0.382, -0.126, -0.146, -0.147,  0.049],
    "ESR1":    [-0.381,  0.017,  0.289,  0.083,  0.015],
    "SLC39A6": [-0.261,  0.041,  0.181,  0.046, -0.010],
    "BAG1":    [-0.151,  0.013,  0.111,  0.022,  0.017],
    "MAPT":    [-0.234,  0.009,  0.178,  0.014,  0.031],
    "PGR":     [-0.259, -0.049,  0.243,  0.002,  0.066],
    "CXXC5":   [-0.204,  0.045,  0.123,  0.039, -0.004],
    "MLPH":    [-0.303,  0.046,  0.222,  0.040, -0.001],
    "BCL2":    [-0.272, -0.006,  0.233,  0.001,  0.048],
    "MDM2":    [-0.115,  0.016,  0.087,  0.007,  0.000],
    "NAT1":    [-0.254,  0.048,  0.173,  0.013,  0.023],
    "FOXA1":   [-0.349,  0.070,  0.245,  0.049, -0.024],
    "BLVRA":   [-0.082,  0.028,  0.047,  0.008, -0.003],
    "MMP11":   [-0.026,  0.084, -0.003,  0.014, -0.057],
    "GPR160":  [-0.184,  0.055,  0.113,  0.031, -0.016],
    "FGFR4":   [-0.054,  0.038,  0.016, -0.006,  0.002],
    "GRB7":    [-0.101,  0.343, -0.066,  0.000, -0.088],
    "TMEM45B": [-0.083,  0.132, -0.015,  0.007, -0.020],
    "ERBB2":   [-0.146,  0.317, -0.050, -0.017, -0.036],
}

_PAM50_LABEL_MAP: Dict[str, str] = {
    "Basal": "Basal",
    "Basal-like": "Basal",
    "HER2": "HER2",
    "HER2-enriched": "HER2",
    "Her2": "HER2",
    "LumA": "LumA",
    "Luminal A": "LumA",
    "LumB": "LumB",
    "Luminal B": "LumB",
    "Normal": "Normal",
    "Normal-like": "Normal",
}

_PAM50_GENEFU_MODEL_NAMES: Tuple[str, ...] = (
    "pam50.scale",
)


def _normalize_pam50_label(label: Any) -> str:
    """Normalize genefu PAM50 labels to the repo's display convention."""
    return _PAM50_LABEL_MAP.get(str(label), str(label))


def _load_genefu_bridge() -> Tuple[Any, Any, Any]:
    """Load the R bridge that wraps genefu PAM50 classification."""
    global _GENEFU_BRIDGE_CACHE
    if _GENEFU_BRIDGE_CACHE is not None:
        return _GENEFU_BRIDGE_CACHE

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r(
        """
                proteus_load_genefu <- function() {
                    if ("genefu" %in% loadedNamespaces()) {
                        return(invisible(TRUE))
                    }

                    withCallingHandlers(
                        loadNamespace("genefu"),
                        warning = function(w) {
                            msg <- conditionMessage(w)
                            if (grepl("replacing previous import.*logsumexp.*when loading 'genefu'", msg)) {
                                invokeRestart("muffleWarning")
                            }
                        }
                    )

                    invisible(TRUE)
                }

        genefu_pam50_classify <- function(expr_df, model_name) {
                    proteus_load_genefu()

          data(list = model_name, package = 'genefu', envir = environment())
          sbt.model <- get(model_name, envir = environment())

          expr_matrix <- as.matrix(expr_df)
          common_probes <- intersect(rownames(expr_matrix), sbt.model$centroids.map$probe)
          if (length(common_probes) < 10) {
            return(data.frame())
          }

          expr_sub <- expr_matrix[common_probes, , drop = FALSE]
          data_sub <- t(expr_sub)
          model_map <- sbt.model$centroids.map[match(common_probes, sbt.model$centroids.map$probe), , drop = FALSE]
          annot <- data.frame(EntrezGene.ID = model_map$EntrezGene.ID, row.names = common_probes)

          pred <- genefu::intrinsic.cluster.predict(
            sbt.model = sbt.model,
            data = data_sub,
            annot = annot,
            do.mapping = FALSE,
            do.prediction.strength = FALSE,
            verbose = FALSE
          )

          out <- data.frame(
            sample = rownames(data_sub),
            predicted_subtype = as.character(pred$subtype),
            stringsAsFactors = FALSE,
            row.names = rownames(data_sub)
          )

          if (!is.null(pred$cor)) {
            cor_df <- as.data.frame(pred$cor)
            cor_df <- cor_df[rownames(data_sub), , drop = FALSE]
            colnames(cor_df) <- paste0('corr_', colnames(cor_df))
            out <- cbind(out, cor_df)
          }
          if (!is.null(pred$subtype.proba)) {
            proba_df <- as.data.frame(pred$subtype.proba)
            proba_df <- proba_df[rownames(data_sub), , drop = FALSE]
            colnames(proba_df) <- paste0('proba_', colnames(proba_df))
            out <- cbind(out, proba_df)
          }

          rownames(out) <- NULL
          out
        }
        """
    )
    _GENEFU_BRIDGE_CACHE = (ro, pandas2ri, localconverter)
    return _GENEFU_BRIDGE_CACHE


def _pam50_classify_single_genefu(
    expr_df: pd.DataFrame,
    *,
    model_name: str,
    bridge: Optional[Tuple[Any, Any, Any]] = None,
) -> pd.DataFrame:
    """Classify one expression matrix with one genefu PAM50 model variant."""
    ro, pandas2ri, localconverter = _load_genefu_bridge() if bridge is None else bridge
    converter = ro.default_converter + pandas2ri.converter
    with localconverter(converter):
        r_expr = ro.conversion.get_conversion().py2rpy(expr_df)
        result_r = ro.globalenv["genefu_pam50_classify"](r_expr, model_name)
        result = ro.conversion.get_conversion().rpy2py(result_r)

    result_df = pd.DataFrame(result)
    if result_df.empty:
        return result_df

    result_df["predicted_subtype"] = result_df["predicted_subtype"].map(_normalize_pam50_label)
    rename_map = {
        column: column.replace("Her2", "HER2")
        for column in result_df.columns
        if "Her2" in column
    }
    if rename_map:
        result_df = result_df.rename(columns=rename_map)
    return result_df


def pam50_classify_genefu(
    expr_df: pd.DataFrame,
    *,
    model_names: Sequence[str] = _PAM50_GENEFU_MODEL_NAMES,
) -> Dict[str, pd.DataFrame]:
    """Classify samples with each requested genefu PAM50 model variant."""
    bridge = _load_genefu_bridge()
    results: Dict[str, pd.DataFrame] = {}
    for model_name in model_names:
        results[model_name] = _pam50_classify_single_genefu(
            expr_df,
            model_name=model_name,
            bridge=bridge,
        )
    return results


# ---------------------------------------------------------------------------
# Pluvicto clinical data loaders
# ---------------------------------------------------------------------------

def extract_pluvicto_patient_id(sample_name: str) -> str:
    """Extract patient ID from a Pluvicto sample name.

    ``FH0084_E_C1`` → ``FH0084``.  Splits on ``_E_`` or ``_S_`` and
    returns the prefix.
    """
    for sep in ("_E_", "_S_"):
        if sep in sample_name:
            return sample_name.split(sep)[0]
    return sample_name


def load_pluvicto_outcomes() -> pd.DataFrame:
    """Load reference/CRPC-Pluvicto_outcomes.csv → DataFrame indexed by Sample_ID."""
    path = PROJECT_ROOT / "reference" / "CRPC-Pluvicto_outcomes.csv"
    df = pd.read_csv(path)
    df = df.set_index("Sample_ID")
    # Coerce NA strings to NaN for numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_pluvicto_clinical() -> pd.DataFrame:
    """Load reference/CRPC-Pluvicto_baseline-clinical.csv → DataFrame indexed by Sample_ID.

    Handles ``>60`` and ``N/A`` string values by coercing to NaN.
    """
    path = PROJECT_ROOT / "reference" / "CRPC-Pluvicto_baseline-clinical.csv"
    df = pd.read_csv(path, na_values=["N/A", "NA", ""])
    df = df.set_index("Sample_ID")
    for col in df.columns:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.startswith(">") else v
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_geneset_score(
    expr_df: pd.DataFrame,
    gene_list: List[str],
) -> pd.Series:
    """Mean expression of *gene_list* genes for each sample.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Genes × samples expression matrix.
    gene_list : list of str
        Gene symbols to average.

    Returns
    -------
    pd.Series
        Per-sample mean expression score, indexed by sample name.
    """
    if expr_df.empty:
        return pd.Series(index=expr_df.columns, dtype=float)

    augmented_expr = _augment_missing_gene_aliases(expr_df)
    present = _resolve_expression_gene_rows(augmented_expr, gene_list)
    if not present:
        return pd.Series(index=augmented_expr.columns, dtype=float)

    gene_frame = augmented_expr.loc[present]
    gene_frame = gene_frame.loc[~gene_frame.isna().any(axis=1)]
    if gene_frame.empty:
        return pd.Series(index=augmented_expr.columns, dtype=float)

    return gene_frame.mean(axis=0)


def compute_cox_univariate(
    time: pd.Series,
    event: pd.Series,
    covariates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fit univariate Cox PH for each column in *covariates_df*.

    Parameters
    ----------
    time : pd.Series
        Survival time.
    event : pd.Series
        Event indicator (1 = event, 0 = censored).
    covariates_df : pd.DataFrame
        One column per covariate, rows aligned with *time* / *event*.

    Returns
    -------
    pd.DataFrame
        Rows = covariates, columns include ``HR``, ``HR_lo``, ``HR_hi``, ``p``,
        ``n``, ``n_events``, and proportional-hazards diagnostic ``ph_p``.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    results = []
    for col in covariates_df.columns:
        cph_df = pd.DataFrame({
            "T": time.values,
            "E": event.values,
            col: covariates_df[col].values,
        }).dropna()
        if cph_df.shape[0] < 10 or cph_df["E"].sum() < 3:
            results.append({
                "variable": col, "HR": np.nan, "HR_lo": np.nan,
                "HR_hi": np.nan, "p": np.nan,
                "n": int(cph_df.shape[0]), "n_events": int(cph_df["E"].sum()),
                "ph_p": np.nan,
            })
            continue
        # Standardise covariate to unit variance for comparable HRs
        std = cph_df[col].std()
        if std < 1e-12:
            results.append({
                "variable": col, "HR": np.nan, "HR_lo": np.nan,
                "HR_hi": np.nan, "p": np.nan,
                "n": int(cph_df.shape[0]), "n_events": int(cph_df["E"].sum()),
                "ph_p": np.nan,
            })
            continue
        cph_df[col] = cph_df[col] / std

        cph = CoxPHFitter()
        cph.fit(cph_df, duration_col="T", event_col="E")
        summary = cph.summary
        ph_test = proportional_hazard_test(cph, cph_df, time_transform="rank")
        ph_summary = ph_test.summary
        ph_p = float(ph_summary.loc[col, "p"]) if col in ph_summary.index else np.nan
        results.append({
            "variable": col,
            "HR": float(summary.loc[col, "exp(coef)"]),
            "HR_lo": float(summary.loc[col, "exp(coef) lower 95%"]),
            "HR_hi": float(summary.loc[col, "exp(coef) upper 95%"]),
            "p": float(summary.loc[col, "p"]),
            "n": int(cph_df.shape[0]),
            "n_events": int(cph_df["E"].sum()),
            "ph_p": ph_p,
        })
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["q_value"] = benjamini_hochberg(result_df["p"])
    return result_df


def compute_cox_multivariate(
    time: pd.Series,
    event: pd.Series,
    covariates_df: pd.DataFrame,
    adjust_for: pd.Series,
    adjust_name: str = "TFx",
) -> pd.DataFrame:
    """Fit multivariate Cox PH for each column in *covariates_df*.

    Each model includes the target covariate plus a shared adjustment term.
    For the Pluvicto survival workflow that adjustment term is TFx, which the
    user explicitly requested to preserve outside the limma path.

    Parameters
    ----------
    time : pd.Series
        Survival time.
    event : pd.Series
        Event indicator (1 = event, 0 = censored).
    covariates_df : pd.DataFrame
        One column per covariate, rows aligned with *time* / *event*.
    adjust_for : pd.Series
        Confounding covariate included in every fit.
    adjust_name : str
        Column name for the adjustment variable.

    Returns
    -------
    pd.DataFrame
        Rows = covariates, columns include ``HR``, ``HR_lo``, ``HR_hi``, ``p``,
        ``n``, ``n_events``, and proportional-hazards diagnostics for both the
        target term and the adjustment term.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    results = []
    for col in covariates_df.columns:
        if col == adjust_name:
            continue
        cph_df = pd.DataFrame({
            "T": time.values,
            "E": event.values,
            col: covariates_df[col].values,
            adjust_name: adjust_for.values,
        }).dropna()
        if cph_df.shape[0] < 10 or cph_df["E"].sum() < 3:
            results.append({
                "variable": col,
                "HR": np.nan,
                "HR_lo": np.nan,
                "HR_hi": np.nan,
                "p": np.nan,
                "n": int(cph_df.shape[0]),
                "n_events": int(cph_df["E"].sum()),
                "ph_p": np.nan,
                "ph_adjust_p": np.nan,
                "ph_min_p": np.nan,
            })
            continue

        # Standardize both terms so hazard ratios remain comparable across
        # biomarkers while keeping the TFx-adjusted survival path numerically
        # stable when one term has a narrow range.
        for covariate_name in (col, adjust_name):
            std = cph_df[covariate_name].std()
            if std < 1e-12:
                cph_df[covariate_name] = 0.0
            else:
                cph_df[covariate_name] = cph_df[covariate_name] / std

        cph = CoxPHFitter()
        cph.fit(cph_df, duration_col="T", event_col="E")
        summary = cph.summary
        ph_test = proportional_hazard_test(cph, cph_df, time_transform="rank")
        ph_summary = ph_test.summary
        ph_p = float(ph_summary.loc[col, "p"]) if col in ph_summary.index else np.nan
        ph_adjust_p = (
            float(ph_summary.loc[adjust_name, "p"])
            if adjust_name in ph_summary.index else np.nan
        )
        ph_min_p = float(pd.to_numeric(ph_summary["p"], errors="coerce").min())
        results.append({
            "variable": col,
            "HR": float(summary.loc[col, "exp(coef)"]),
            "HR_lo": float(summary.loc[col, "exp(coef) lower 95%"]),
            "HR_hi": float(summary.loc[col, "exp(coef) upper 95%"]),
            "p": float(summary.loc[col, "p"]),
            "n": int(cph_df.shape[0]),
            "n_events": int(cph_df["E"].sum()),
            "ph_p": ph_p,
            "ph_adjust_p": ph_adjust_p,
            "ph_min_p": ph_min_p,
        })
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df["q_value"] = benjamini_hochberg(result_df["p"])
    return result_df


def compute_logrank_p(
    time_lo: pd.Series,
    event_lo: pd.Series,
    time_hi: pd.Series,
    event_hi: pd.Series,
) -> float:
    """Return the two-group log-rank p-value for exploratory KM plots."""
    from lifelines.statistics import logrank_test

    return float(
        logrank_test(
            time_lo.values.astype(float),
            time_hi.values.astype(float),
            event_observed_A=event_lo.values.astype(int),
            event_observed_B=event_hi.values.astype(int),
        ).p_value
    )


# ---------------------------------------------------------------------------
# MDN Monte Carlo sampling
# ---------------------------------------------------------------------------

# Fixed constants matching model_utils.py
_K_GAMMA = 0.75   # Gamma shape parameter (fixed)
_Y_MAX = 19.93    # Expression ceiling


def mdn_monte_carlo_sample(
    pi_delta: pd.DataFrame,
    pi_gamma: pd.DataFrame,
    pi_normal: pd.DataFrame,
    gamma_beta: pd.DataFrame,
    normal_mean: pd.DataFrame,
    normal_stdev: pd.DataFrame,
    M: int = 100,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """Draw *M* independent expression matrices from the MDN posterior.

    Each input is a genes x samples wide DataFrame sharing the same index
    (gene names) and columns (sample names).

    Returns a list of *M* DataFrames, each genes x samples, where each
    element is an independent draw from the 3-component mixture
    (hurdle at 0, Gamma(k=0.75, beta), Normal(mu, sigma^2)),
    clamped to [0, Y_MAX].
    """
    genes = pi_delta.index
    samples = pi_delta.columns
    G, S = len(genes), len(samples)

    # Convert to numpy arrays (G, S)
    pd_arr = pi_delta.values.astype(np.float64)
    pg_arr = pi_gamma.values.astype(np.float64)
    pn_arr = pi_normal.values.astype(np.float64)
    gb_arr = gamma_beta.values.astype(np.float64)
    nm_arr = normal_mean.values.astype(np.float64)
    ns_arr = normal_stdev.values.astype(np.float64)

    rng = np.random.default_rng(seed)
    draws: List[pd.DataFrame] = []

    for m in range(M):
        # Draw component indices: 0=hurdle, 1=gamma, 2=normal
        u = rng.random((G, S))
        cum_delta = pd_arr
        cum_gamma = cum_delta + pg_arr
        comp = np.where(u < cum_delta, 0, np.where(u < cum_gamma, 1, 2))

        # Gamma: shape=K_GAMMA, scale=1/beta (numpy uses scale, not rate)
        gamma_scale = np.where(gb_arr > 0, 1.0 / gb_arr, 1.0)
        gamma_draw = rng.gamma(shape=_K_GAMMA, scale=gamma_scale, size=(G, S))

        # Normal
        normal_draw = rng.normal(loc=nm_arr, scale=np.maximum(ns_arr, 1e-8),
                                 size=(G, S))

        # Assemble: hurdle -> 0, gamma -> gamma_draw, normal -> normal_draw
        x = np.where(comp == 0, 0.0,
                      np.where(comp == 1, gamma_draw, normal_draw))
        x = np.clip(x, 0.0, _Y_MAX)

        draws.append(pd.DataFrame(x, index=genes, columns=samples))

    return draws


def mdn_predictive_variance(
    pi_delta: pd.DataFrame,
    pi_gamma: pd.DataFrame,
    pi_normal: pd.DataFrame,
    gamma_beta: pd.DataFrame,
    normal_mean: pd.DataFrame,
    normal_stdev: pd.DataFrame,
) -> pd.DataFrame:
    """Compute analytical predictive variance of the MDN mixture.

    Var[X] = E[X^2] - (E[X])^2 where:
      E[X]   = pi_gamma * K/beta + pi_normal * mu
      E[X^2] = pi_gamma * K(K+1)/beta^2 + pi_normal * (mu^2 + sigma^2)

    Returns a genes x samples DataFrame of variances (same shape as inputs).
    """
    pd_arr = pi_delta.values.astype(np.float64)
    pg_arr = pi_gamma.values.astype(np.float64)
    pn_arr = pi_normal.values.astype(np.float64)
    gb_arr = np.maximum(gamma_beta.values.astype(np.float64), 1e-8)
    nm_arr = normal_mean.values.astype(np.float64)
    ns_arr = normal_stdev.values.astype(np.float64)

    # E[X] (hurdle component contributes 0)
    ex = pg_arr * (_K_GAMMA / gb_arr) + pn_arr * nm_arr

    # E[X^2]
    ex2 = (pg_arr * (_K_GAMMA * (_K_GAMMA + 1.0) / (gb_arr ** 2))
           + pn_arr * (nm_arr ** 2 + ns_arr ** 2))

    var = np.maximum(ex2 - ex ** 2, 1e-12)

    return pd.DataFrame(var, index=pi_delta.index, columns=pi_delta.columns)


def mdn_quantile_interval_wide(
    pi_delta: pd.DataFrame,
    pi_gamma: pd.DataFrame,
    pi_normal: pd.DataFrame,
    gamma_beta: pd.DataFrame,
    normal_mean: pd.DataFrame,
    normal_stdev: pd.DataFrame,
    quantiles: Tuple[float, float] = (0.25, 0.75),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute wide lower/upper MDN quantile surfaces from stored mixture params."""
    import torch

    from models.model_utils import mixture_quantile

    if len(quantiles) != 2:
        raise ValueError("quantiles must contain exactly two probability levels")

    pi_delta = _dedupe_wide_axes(pi_delta)
    pi_gamma = _dedupe_wide_axes(pi_gamma)
    pi_normal = _dedupe_wide_axes(pi_normal)
    gamma_beta = _dedupe_wide_axes(gamma_beta)
    normal_mean = _dedupe_wide_axes(normal_mean)
    normal_stdev = _dedupe_wide_axes(normal_stdev)

    genes = (
        pi_delta.index
        .intersection(pi_gamma.index)
        .intersection(pi_normal.index)
        .intersection(gamma_beta.index)
        .intersection(normal_mean.index)
        .intersection(normal_stdev.index)
    )
    samples = (
        pi_delta.columns
        .intersection(pi_gamma.columns)
        .intersection(pi_normal.columns)
        .intersection(gamma_beta.columns)
        .intersection(normal_mean.columns)
        .intersection(normal_stdev.columns)
    )
    if len(genes) == 0 or len(samples) == 0:
        empty = pd.DataFrame(index=genes, columns=samples, dtype=np.float64)
        return empty.copy(), empty

    p0_arr = pi_delta.reindex(index=genes, columns=samples).values.astype(np.float64)
    pg_arr = pi_gamma.reindex(index=genes, columns=samples).values.astype(np.float64)
    pn_arr = pi_normal.reindex(index=genes, columns=samples).values.astype(np.float64)
    beta_arr = gamma_beta.reindex(index=genes, columns=samples).values.astype(np.float64)
    mu_arr = normal_mean.reindex(index=genes, columns=samples).values.astype(np.float64)
    sigma_arr = normal_stdev.reindex(index=genes, columns=samples).values.astype(np.float64)

    lo_arr = np.full(p0_arr.shape, np.nan, dtype=np.float64)
    hi_arr = np.full(p0_arr.shape, np.nan, dtype=np.float64)

    p0_flat = p0_arr.reshape(-1)
    pg_flat = pg_arr.reshape(-1)
    pn_flat = pn_arr.reshape(-1)
    beta_flat = beta_arr.reshape(-1)
    mu_flat = mu_arr.reshape(-1)
    sigma_flat = sigma_arr.reshape(-1)

    prob_sum = p0_flat + pg_flat + pn_flat
    valid_flat = (
        np.isfinite(p0_flat)
        & np.isfinite(pg_flat)
        & np.isfinite(pn_flat)
        & np.isfinite(beta_flat)
        & np.isfinite(mu_flat)
        & np.isfinite(sigma_flat)
        & (prob_sum > 0.0)
    )

    if valid_flat.any():
        probs = np.stack([p0_flat[valid_flat], pg_flat[valid_flat], pn_flat[valid_flat]], axis=1)
        probs /= probs.sum(axis=1, keepdims=True)

        mdn_params = {
            "log_pi": torch.from_numpy(np.log(np.clip(probs, 1e-12, None))).float(),
            "gamma_beta": torch.from_numpy(np.clip(beta_flat[valid_flat], 1e-12, None)).float().unsqueeze(-1),
            "normal_mu": torch.from_numpy(mu_flat[valid_flat]).float().unsqueeze(-1),
            "normal_sigma": torch.from_numpy(np.clip(sigma_flat[valid_flat], 1e-12, None)).float().unsqueeze(-1),
        }
        q = mixture_quantile(mdn_params, quantiles=quantiles).cpu().numpy()

        lo_flat = lo_arr.reshape(-1)
        hi_flat = hi_arr.reshape(-1)
        lo_flat[valid_flat] = q[:, 0]
        hi_flat[valid_flat] = q[:, 1]

    return (
        pd.DataFrame(lo_arr, index=genes, columns=samples),
        pd.DataFrame(hi_arr, index=genes, columns=samples),
    )
