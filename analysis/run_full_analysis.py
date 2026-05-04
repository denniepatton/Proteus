#!/usr/bin/env python3
"""
run_full_analysis.py
Proteus analysis framework — main orchestrator.

Usage examples:
    python analysis/run_full_analysis.py --group Validation_PDX
    python analysis/run_full_analysis.py --group Validation_Patient
    python analysis/run_full_analysis.py --group Discovery_Patient
    python analysis/run_full_analysis.py --group Benchmarking_Synthetic
    python analysis/run_full_analysis.py --group all

Run ``--help`` for full CLI reference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analysis_utils import (
    _flatten_wide_pair,
    _augment_missing_gene_aliases,
    clean_epicseq_sample_names,
    clean_epicseq_sample_names_wcdt,
    compute_activity_odds_ratio_off_curve,
    compute_activity_odds_ratio_on_curve,
    cohort_summary_table,
    compute_cox_univariate,
    compute_cox_multivariate,
    compute_coverage_curve,
    compute_gene_variance,
    compute_geneset_score,
    compute_onoff_confusion,
    compute_subset_metrics,
    empirical_coverage,
    ensure_dir,
    extract_pluvicto_patient_id,
    format_pval,
    gene_level_metrics,
    gene_list_available,
    generate_null_predictions,
    get_gene_list,
    GENE_SET_BREAKS,
    load_cv_predictions_proteus_full,
    load_holdout_predictions_epicseq,
    load_holdout_predictions_proteus_full,
    load_holdout_predictions_xgboost,
    load_palette,
    load_pdx_truth_for_phs,
    load_pluvicto_clinical,
    load_pluvicto_outcomes,
    load_subtypes,
    load_truth,
    mdn_activity_odds_ratio_wide,
    mdn_on_probability_wide,
    mdn_pit_values_wide,
    mdn_monte_carlo_sample,
    mdn_quantile_interval_wide,
    pam50_classify_genefu,
    pam50_classify_mc_genefu,
    parse_phs_sample,
    print_summary_table,
    sample_level_metrics,
    SCLC_PDX_STUDY,
    STANDARD_GENE_SETS,
    strip_cohort_prefix,
    variance_group_metrics,
)
from analysis.calibration import (
    compute_gene_regimes,
    RegimeCalibrator,
)
from analysis.examples import run_examples as run_examples_workflow
from analysis.pipeline_config import SECTION_OUTPUT_DIRS
from analysis.pipeline_config import (
    BENCHMARK_COMPLETE_MODELS,
    BENCHMARK_SUBSET_MODELS,
    DISCOVERY_COHORTS as CONFIG_DISCOVERY_COHORTS,
    DISCOVERY_MODEL_PATHS as CONFIG_DISCOVERY_MODEL_PATHS,
    EPIC_DISPLAY_NAMES as CONFIG_EPIC_DISPLAY_NAMES,
    EPIC_SCORE_COLUMNS as CONFIG_EPIC_SCORE_COLUMNS,
    PATIENT_BENCHMARK_COHORTS,
    PATIENT_EPIC_PATHS,
    PATIENT_PROTEUS_MODEL_PATHS,
    PATIENT_XGB_PATHS,
    SYNTHETIC_EPIC_PATH,
    SYNTHETIC_PROTEUS_MODEL_PATHS,
    SYNTHETIC_XGB_PATH,
    VALIDATION_PATIENT_CV_COHORTS,
    VALIDATION_PATIENT_HOLDOUT_COHORTS,
    VALIDATION_PATIENT_HOLDOUT_PATH_TEMPLATE,
    VALIDATION_PATIENT_CV_RESULTS_DIR,
    VALIDATION_PDX_COHORTS,
    VALIDATION_PDX_CV_RESULTS_DIR,
    VALIDATION_PDX_HOLDOUT_PATHS,
    benchmark_model_views,
)
from analysis.reliability import (
    compute_gene_features,
    ReliabilityModel,
    retention_curve,
    _compute_gene_pearson_r,
)
from analysis.plot_utils import (
    batch_gene_scatters,
    batch_sample_scatters,
    inactive_confidence_threshold_plot,
    benchmark_cohort_boxplots,
    benchmark_cohort_histograms,
    benchmark_cohort_metrics_lines,
    benchmark_depth_lines,
    benchmark_grid_boxplots,
    benchmark_grid_histograms,
    binary_status_boxplot,
    clinical_correlation_scatter,
    confusion_matrix_heatmap,
    double_heatmap,
    forest_plot,
    gene_target_scatter,
    kaplan_meier_plot,
    metric_distribution,
    metric_distribution_by_variance,
    plot_calibration_coverage,
    plot_error_vs_width,
    plot_pit_histogram,
    plot_regime_coverage_heatmap,
    plot_regime_qhat_heatmap,
    plot_reliability_scatter,
    plot_reliability_weights,
    plot_retention_curve,
    onoff_confusion_summary_plot,
    active_confidence_threshold_plot,
    roc_gene_calling,
    sample_expression_kde,
    sample_scatter,
    single_heatmap,
    volcano_plot,
    enrichment_dot_plot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GENE_SETS_FOR_SCATTER = ["PC-Phenotyping", "DruggableGenome", "CancerSurfaceome"]
DEFAULT_GOI_PC = {
    "ARPC": ["AR", "NKX3-1", "FOLH1", "KLK3"],
    "NEPC": ["ASCL1", "CHGA", "SOX2", "SYP"],
}
DEFAULT_GOI_SCLC = {
    "POU2F3": ["POU2F3"],
    "ASCL1": ["ASCL1"],
    "NEUROD1": ["NEUROD1"],
    "ATOH1": ["ATOH1"],
}
DEFAULT_GOI_BLCA = {
    "UC": ["FGFR3", "GATA3", "PPARG"],
    "UC/SQ": ["KRT5", "KRT14", "EGFR"],
    "NE": ["ASCL1", "CHGA", "SYP"],
    "PUC": ["CDH1", "EPCAM", "ERBB2"],
}

_MDN_PARAM_KEYS: Tuple[str, ...] = (
    "p0_wide",
    "pi_gamma_wide",
    "pi_normal_wide",
    "gamma_beta_wide",
    "normal_mean_wide",
    "normal_stdev_wide",
)
_PROTEUS_BUNDLE_WIDE_KEYS: Tuple[str, ...] = (
    "pred_wide",
    "truth_wide",
    "ci_lo_wide",
    "ci_hi_wide",
    "activity_odds_ratio_wide",
    "active_confidence_wide",
    *_MDN_PARAM_KEYS,
)

_CALIBRATION_INTERVAL_NOMINAL = 0.90
_CALIBRATION_LEVELS: Tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90)
_DEFAULT_MONTE_CARLO_DRAWS = 100
_ACTIVE_CALL_TARGET_PPV = 0.90
_INACTIVE_CALL_TARGET_NPV = 0.90
_DOWNSTREAM_ACTIVE_DIM_THRESHOLD = 0.95

_PATIENT_COHORTS = frozenset(
    list(VALIDATION_PATIENT_HOLDOUT_COHORTS)
    + list(VALIDATION_PATIENT_CV_COHORTS)
    + [cohort_name for cohort_name, _ in CONFIG_DISCOVERY_COHORTS]
)


def _section_results_path(section_name: str) -> str:
    """Return the top-level results directory path for one analysis section."""
    return str(PROJECT_ROOT / "analysis" / "results" / SECTION_OUTPUT_DIRS[section_name])


def _section_results_dir(section_name: str) -> str:
    return ensure_dir(_section_results_path(section_name))


def _should_skip_section(section_name: str, args: argparse.Namespace) -> bool:
    """Return True when a top-level section folder already exists and overwrite is off."""
    if bool(getattr(args, "overwrite", False)):
        return False
    out_dir = _section_results_path(section_name)
    if os.path.isdir(out_dir):
        print(f"  [skip] {section_name}: top-level output folder already exists ({out_dir})")
        print("         Pass --overwrite to rerun this analysis group.")
        return True
    return False


def _write_run_manifest(section_name: str, out_dir: str, args: argparse.Namespace) -> None:
    """Write a minimal reproducibility manifest for one analysis section."""
    manifest = {
        "section": section_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "project_root": str(PROJECT_ROOT),
        "working_directory": os.getcwd(),
        "python_version": sys.version,
        "arguments": vars(args).copy(),
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _is_patient_cohort(cohort_name: str) -> bool:
    """Return True when a cohort represents patient-derived samples."""
    return cohort_name in _PATIENT_COHORTS


def _filter_wide_to_samples(
    table: Optional[pd.DataFrame],
    keep_samples: Optional[Sequence[str]],
) -> Optional[pd.DataFrame]:
    """Restrict a wide matrix to the requested sample columns in order."""
    if table is None or keep_samples is None:
        return table

    keep = [str(sample_name) for sample_name in keep_samples if str(sample_name) in table.columns]
    return table.loc[:, keep].copy()


def _filter_prediction_bundle_to_samples(
    data_bundle: Dict[str, Any],
    keep_samples: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """Restrict all wide prediction tables in a bundle to the same sample list."""
    if keep_samples is None:
        return dict(data_bundle)

    filtered_bundle = dict(data_bundle)
    for key in _PROTEUS_BUNDLE_WIDE_KEYS:
        filtered_bundle[key] = _filter_wide_to_samples(data_bundle.get(key), keep_samples)
    return filtered_bundle


def _align_sample_metadata_to_reference(
    metadata_df: pd.DataFrame,
    reference_columns: pd.Index,
    *,
    cohort: Optional[str] = None,
) -> pd.DataFrame:
    """Rename metadata sample indices to match a reference sample axis."""
    reference_index = pd.Index([str(column) for column in reference_columns])
    sample_map = {str(column): str(column) for column in reference_index}
    if cohort is not None:
        stripped_index = strip_cohort_prefix(reference_index, cohort)
        for canonical_name, short_name in zip(reference_index, stripped_index):
            sample_map[str(short_name)] = str(canonical_name)

    aligned = metadata_df.copy()
    aligned.index = pd.Index([sample_map.get(str(sample_name), str(sample_name)) for sample_name in aligned.index])
    aligned = aligned.loc[~aligned.index.duplicated(keep="first")]
    return aligned.loc[aligned.index.intersection(reference_index)]


def _load_filtered_subtypes_for_reference(
    cohort: str,
    reference_columns: pd.Index,
    *,
    minimum_tfx: float,
    label_name: Optional[str] = None,
    report_context: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load subtype metadata aligned to a reference sample axis and apply TFx filtering."""
    resolved_label = label_name or cohort
    subtype_path = PROJECT_ROOT / "labels" / f"{resolved_label}_TFX_Subtype.tsv"
    if not subtype_path.exists():
        if minimum_tfx > 0.0 and _is_patient_cohort(cohort):
            raise ValueError(
                f"minimum_tfx={minimum_tfx} requested but TFx metadata is missing for patient cohort {resolved_label}"
            )
        return None

    subtypes_df = _align_sample_metadata_to_reference(
        load_subtypes(resolved_label),
        reference_columns,
        cohort=cohort,
    )
    if minimum_tfx <= 0.0 or not _is_patient_cohort(cohort):
        return subtypes_df

    if "tfx" not in subtypes_df.columns:
        raise ValueError(
            f"minimum_tfx={minimum_tfx} requested but TFx metadata is missing a 'tfx' column for {resolved_label}"
        )

    reference_index = pd.Index([str(column) for column in reference_columns])
    missing_samples = [sample_name for sample_name in reference_index if sample_name not in subtypes_df.index]
    if missing_samples:
        raise ValueError(
            f"minimum_tfx={minimum_tfx} requested but {resolved_label} is missing TFx metadata for samples: "
            + ", ".join(missing_samples)
        )

    low_tfx_mask = subtypes_df["tfx"].lt(float(minimum_tfx)).fillna(False)
    dropped_samples = subtypes_df.index[low_tfx_mask].tolist()
    if dropped_samples:
        context = report_context or cohort
        print(
            f"  Minimum TFx filter ({context}): dropped {len(dropped_samples)} patient samples below {minimum_tfx:.3f}"
        )
        print("    " + ", ".join(dropped_samples))
    return subtypes_df.loc[~low_tfx_mask].copy()


def _concat_common_gene_frames(
    parts: Sequence[Optional[pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """Concatenate wide matrices after intersecting to a shared gene index."""
    non_empty = [part for part in parts if part is not None and not part.empty]
    if not non_empty:
        return None

    common_genes = non_empty[0].index
    for part in non_empty[1:]:
        common_genes = common_genes.intersection(part.index)
    if len(common_genes) == 0:
        return None

    return pd.concat([part.loc[common_genes] for part in non_empty], axis=1)


def _flatten_calibration_comparison(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    ci_lo_wide: pd.DataFrame,
    ci_hi_wide: pd.DataFrame,
    ci_lo_adj_wide: pd.DataFrame,
    ci_hi_adj_wide: pd.DataFrame,
) -> Optional[Dict[str, np.ndarray]]:
    """Align and flatten interval data for before/after calibration summaries."""
    common_genes = (
        truth_wide.index.intersection(pred_wide.index)
        .intersection(ci_lo_wide.index)
        .intersection(ci_hi_wide.index)
        .intersection(ci_lo_adj_wide.index)
        .intersection(ci_hi_adj_wide.index)
    )
    common_samples = (
        truth_wide.columns.intersection(pred_wide.columns)
        .intersection(ci_lo_wide.columns)
        .intersection(ci_hi_wide.columns)
        .intersection(ci_lo_adj_wide.columns)
        .intersection(ci_hi_adj_wide.columns)
    )
    if len(common_genes) == 0 or len(common_samples) == 0:
        return None

    truth = truth_wide.loc[common_genes, common_samples].values
    pred = pred_wide.loc[common_genes, common_samples].values
    ci_lo = ci_lo_wide.loc[common_genes, common_samples].values
    ci_hi = ci_hi_wide.loc[common_genes, common_samples].values
    ci_lo_adj = ci_lo_adj_wide.loc[common_genes, common_samples].values
    ci_hi_adj = ci_hi_adj_wide.loc[common_genes, common_samples].values

    valid = (
        np.isfinite(truth)
        & np.isfinite(pred)
        & np.isfinite(ci_lo)
        & np.isfinite(ci_hi)
        & np.isfinite(ci_lo_adj)
        & np.isfinite(ci_hi_adj)
    )
    if int(valid.sum()) < 10:
        return None

    return {
        "truth": truth[valid],
        "pred": pred[valid],
        "ci_lo": ci_lo[valid],
        "ci_hi": ci_hi[valid],
        "ci_lo_adj": ci_lo_adj[valid],
        "ci_hi_adj": ci_hi_adj[valid],
    }


def _summarize_calibration_comparison(
    flat: Dict[str, np.ndarray],
    *,
    nominal_levels: Sequence[float] = _CALIBRATION_LEVELS,
    interval_nominal: float = _CALIBRATION_INTERVAL_NOMINAL,
) -> Dict[str, Any]:
    """Summarize coverage and interval-width changes before/after calibration."""
    truth = flat["truth"]
    pred = flat["pred"]
    ci_lo = flat["ci_lo"]
    ci_hi = flat["ci_hi"]
    ci_lo_adj = flat["ci_lo_adj"]
    ci_hi_adj = flat["ci_hi_adj"]

    base_curve = compute_coverage_curve(
        truth,
        ci_lo,
        ci_hi,
        pred,
        nominal_levels,
        interval_nominal=interval_nominal,
    )
    adjusted_curve = compute_coverage_curve(
        truth,
        ci_lo_adj,
        ci_hi_adj,
        pred,
        nominal_levels,
        interval_nominal=interval_nominal,
    )

    base_width = ci_hi - ci_lo
    adjusted_width = ci_hi_adj - ci_lo_adj
    width_delta = adjusted_width - base_width
    width_ratio = adjusted_width / np.maximum(base_width, 1e-6)

    summary: Dict[str, Any] = {
        "n_observations": int(len(truth)),
        "base_mean_width": float(np.mean(base_width)),
        "adjusted_mean_width": float(np.mean(adjusted_width)),
        "base_median_width": float(np.median(base_width)),
        "adjusted_median_width": float(np.median(adjusted_width)),
        "mean_width_delta": float(np.mean(width_delta)),
        "median_width_delta": float(np.median(width_delta)),
        "mean_width_ratio": float(np.mean(width_ratio)),
        "median_width_ratio": float(np.median(width_ratio)),
        "pct_widened": float(100.0 * np.mean(width_delta > 1e-9)),
        "pct_narrowed": float(100.0 * np.mean(width_delta < -1e-9)),
        "pct_unchanged": float(100.0 * np.mean(np.abs(width_delta) <= 1e-9)),
        "base_curve": base_curve,
        "adjusted_curve": adjusted_curve,
    }
    for level, base_cov, adjusted_cov in zip(
        nominal_levels,
        base_curve["empirical"],
        adjusted_curve["empirical"],
    ):
        level_tag = int(round(100 * level))
        summary[f"base_coverage_{level_tag}"] = float(base_cov)
        summary[f"adjusted_coverage_{level_tag}"] = float(adjusted_cov)
        summary[f"coverage_delta_{level_tag}"] = float(adjusted_cov - base_cov)
    return summary


def _save_calibration_summary(summary: Dict[str, Any], path: str) -> None:
    """Persist scalar calibration diagnostics as a one-row TSV."""
    row = {
        key: value
        for key, value in summary.items()
        if key not in {"base_curve", "adjusted_curve"}
    }
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False)


def _save_regime_qhats(calibrator: RegimeCalibrator, out_dir: str) -> None:
    """Persist per-regime lower/upper q-hats as a TSV for downstream review."""
    qhat_df = pd.DataFrame(
        [
            {
                "regime": regime,
                "q_hat_lower": calibrator.q_hats_lower.get(regime, 0.0),
                "q_hat_upper": calibrator.q_hats_upper.get(regime, 0.0),
            }
            for regime in sorted(set(calibrator.q_hats_lower) | set(calibrator.q_hats_upper))
        ]
    ).sort_values("regime")
    qhat_df.to_csv(os.path.join(out_dir, "regime_qhats.tsv"), sep="\t", index=False)


def _save_pit_diagnostics(
    truth_wide: pd.DataFrame,
    mdn_tables: Dict[str, Optional[pd.DataFrame]],
    out_dir: str,
    *,
    overwrite: bool = False,
    title: str,
) -> None:
    """Persist PIT diagnostics from the full MDN predictive distribution."""
    required_keys = (
        "p0_wide",
        "pi_gamma_wide",
        "pi_normal_wide",
        "gamma_beta_wide",
        "normal_mean_wide",
        "normal_stdev_wide",
    )
    if any(mdn_tables.get(key) is None for key in required_keys):
        return

    pit_values = mdn_pit_values_wide(
        truth_wide,
        mdn_tables["p0_wide"],
        mdn_tables["pi_gamma_wide"],
        mdn_tables["pi_normal_wide"],
        mdn_tables["gamma_beta_wide"],
        mdn_tables["normal_mean_wide"],
        mdn_tables["normal_stdev_wide"],
    )
    if len(pit_values) < 10:
        return

    from scipy.stats import kstest

    ks_result = kstest(pit_values, "uniform")
    pd.DataFrame(
        [
            {
                "n_observations": int(len(pit_values)),
                "pit_mean": float(np.mean(pit_values)),
                "pit_variance": float(np.var(pit_values)),
                "pit_ks_statistic": float(ks_result.statistic),
                "pit_ks_pvalue": float(ks_result.pvalue),
            }
        ]
    ).to_csv(os.path.join(out_dir, "pit_summary.tsv"), sep="\t", index=False)
    _maybe(
        plot_pit_histogram,
        overwrite,
        pit_values,
        os.path.join(out_dir, "pit_histogram"),
        title=title,
    )


def _maybe_apply_interval_calibration(
    ci_lo: Optional[pd.DataFrame],
    ci_hi: Optional[pd.DataFrame],
    calibrator: Optional[RegimeCalibrator],
    *,
    enabled: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Apply conformal interval calibration only when requested and available."""
    if ci_lo is None or ci_hi is None:
        return ci_lo, ci_hi
    if enabled and calibrator is not None:
        return calibrator.apply(ci_lo, ci_hi)
    return ci_lo, ci_hi


def _prepare_proteus_analysis_bundle(
    full_data: Dict[str, Any],
    *,
    interval_calibrator: Optional[RegimeCalibrator],
    calibrate_cis: bool,
) -> Dict[str, Any]:
    """Return a shallow copy of a Proteus bundle with requested interval surfaces applied."""
    analysis_data = dict(full_data)
    analysis_data["ci_lo_wide"], analysis_data["ci_hi_wide"] = _maybe_apply_interval_calibration(
        full_data.get("ci_lo_wide"),
        full_data.get("ci_hi_wide"),
        interval_calibrator,
        enabled=calibrate_cis,
    )
    return analysis_data


def _compute_activity_odds_ratio_summary_wide(
    truth_wide: pd.DataFrame,
    activity_odds_ratio_wide: Optional[pd.DataFrame],
    *,
    target_ppv: float = _ACTIVE_CALL_TARGET_PPV,
    target_npv: float = _INACTIVE_CALL_TARGET_NPV,
    threshold: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Compute pooled lower/upper AOR threshold summaries on overlapping truth pairs."""
    if activity_odds_ratio_wide is None or truth_wide.empty:
        return None

    common_genes = truth_wide.index.intersection(activity_odds_ratio_wide.index)
    common_samples = truth_wide.columns.intersection(activity_odds_ratio_wide.columns)
    if len(common_genes) == 0 or len(common_samples) == 0:
        return None

    truth_flat = truth_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel()
    aor_flat = activity_odds_ratio_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel()
    on_curve = compute_activity_odds_ratio_on_curve(
        truth_flat,
        aor_flat,
        threshold=threshold,
        target_ppv=target_ppv,
    )
    off_curve = compute_activity_odds_ratio_off_curve(
        truth_flat,
        aor_flat,
        threshold=threshold,
        target_npv=target_npv,
    )
    if len(on_curve.get("threshold", [])) == 0 or len(off_curve.get("threshold", [])) == 0:
        return None
    return {
        "on_curve": on_curve,
        "off_curve": off_curve,
        "upper_threshold": float(on_curve["selected_threshold"]),
        "lower_threshold": float(off_curve["selected_threshold"]),
    }


def _compute_patient_cv_activity_thresholds(
    cv_full: Dict[str, Any],
    cohort_map: Dict[str, str],
    *,
    target_ppv: float = _ACTIVE_CALL_TARGET_PPV,
    target_npv: float = _INACTIVE_CALL_TARGET_NPV,
    threshold: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """Compute the shared UB/LB AOR thresholds from pooled patient CV OOF data."""
    truth_wide = cv_full.get("truth_wide")
    activity_odds_ratio_wide = cv_full.get("activity_odds_ratio_wide")
    if truth_wide is None or activity_odds_ratio_wide is None:
        return None

    keep_samples = [
        sample_name
        for sample_name in truth_wide.columns
        if cohort_map.get(str(sample_name)) in VALIDATION_PATIENT_CV_COHORTS
    ]
    if not keep_samples:
        return None

    truth_patient = truth_wide.loc[:, keep_samples]
    aor_patient = activity_odds_ratio_wide.loc[:, [sample for sample in keep_samples if sample in activity_odds_ratio_wide.columns]]
    return _compute_activity_odds_ratio_summary_wide(
        truth_patient,
        aor_patient,
        target_ppv=target_ppv,
        target_npv=target_npv,
        threshold=threshold,
    )


def _compute_mdn_median_prediction_wide(
    mdn_tables: Dict[str, Optional[pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """Derive the MDN point-median surface from the stored mixture parameters."""
    if not all(mdn_tables.get(key) is not None for key in _MDN_PARAM_KEYS):
        return None

    median_pred_wide, _ = mdn_quantile_interval_wide(
        mdn_tables["p0_wide"],
        mdn_tables["pi_gamma_wide"],
        mdn_tables["pi_normal_wide"],
        mdn_tables["gamma_beta_wide"],
        mdn_tables["normal_mean_wide"],
        mdn_tables["normal_stdev_wide"],
        quantiles=(0.5, 0.5),
    )
    return median_pred_wide


def _attach_median_point_summary(
    summary: Dict[str, Any],
    truth_wide: pd.DataFrame,
    mdn_tables: Dict[str, Optional[pd.DataFrame]],
    *,
    n_workers: int,
    var_map: Optional[Dict[str, float]] = None,
) -> None:
    """Attach a CLI-only median-point summary row when MDN parameters are available."""
    median_pred_wide = _compute_mdn_median_prediction_wide(mdn_tables)
    if median_pred_wide is None:
        return

    summary["_median_point_summary"] = cohort_summary_table(
        truth_wide,
        median_pred_wide,
        label=f"{summary['label']} [med]",
        n_workers=n_workers,
        var_map=var_map,
        verbose=False,
    )


def _gene_level_ci_inputs(
    ci_lo: Optional[pd.DataFrame],
    ci_hi: Optional[pd.DataFrame],
    *,
    scatter_ci: bool,
    dim_ambiguous: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Return CI inputs for gene-level plots when interval overlays are requested."""
    if scatter_ci:
        return ci_lo, ci_hi
    return None, None


def _ambiguous_sample_mask_for_gene(
    activity_odds_ratio_wide: Optional[pd.DataFrame],
    gene_name: str,
    sample_names: Sequence[str],
    *,
    lower_threshold: Optional[float],
    upper_threshold: Optional[float],
) -> pd.Series:
    """Return a boolean ambiguity mask for one gene across the requested samples."""
    sample_index = pd.Index(sample_names)
    if (
        activity_odds_ratio_wide is not None
        and gene_name in activity_odds_ratio_wide.index
        and lower_threshold is not None
        and upper_threshold is not None
        and np.isfinite(lower_threshold)
        and np.isfinite(upper_threshold)
    ):
        activity_gene = pd.to_numeric(
            activity_odds_ratio_wide.loc[gene_name].reindex(sample_index),
            errors="coerce",
        )
        return (
            (activity_gene >= float(lower_threshold))
            & (activity_gene <= float(upper_threshold))
        ).fillna(False)
    return pd.Series(False, index=sample_index)


def _fit_global_calibrator(
    canonical_truth: pd.DataFrame,
    cv_full: Dict[str, Any],
    *,
    overwrite: bool = False,
    out_dir: Optional[str] = None,
) -> Optional[RegimeCalibrator]:
    """Fit a pooled conformal calibrator on truth-bearing OOF PLCA samples."""
    ci_lo_wide = cv_full.get("ci_lo_wide")
    ci_hi_wide = cv_full.get("ci_hi_wide")
    if ci_lo_wide is None or ci_hi_wide is None or canonical_truth.empty:
        return None

    common_genes = canonical_truth.index.intersection(ci_lo_wide.index).intersection(ci_hi_wide.index)
    common_samples = [
        sample
        for sample in canonical_truth.columns.intersection(ci_lo_wide.columns).intersection(ci_hi_wide.columns)
        if not str(sample).startswith("PHT")
    ]
    if len(common_genes) == 0 or len(common_samples) < 10:
        return None

    cal_truth = canonical_truth.loc[common_genes, common_samples]
    cal_ci_lo = ci_lo_wide.loc[common_genes, common_samples]
    cal_ci_hi = ci_hi_wide.loc[common_genes, common_samples]
    regime_df = compute_gene_regimes(cal_truth)
    calibrator = RegimeCalibrator.fit(
        cal_truth,
        cal_ci_lo,
        cal_ci_hi,
        regime_df,
        alpha=0.10,
    )
    if out_dir is not None:
        _save_regime_qhats(calibrator, out_dir)
        _maybe(
            plot_regime_qhat_heatmap,
            overwrite,
            calibrator.q_hats_lower,
            os.path.join(out_dir, "regime_qhat_heatmap_lower"),
            title="Asymmetric CQR Lower-Tail q̂ by Gene Regime",
        )
        _maybe(
            plot_regime_qhat_heatmap,
            overwrite,
            calibrator.q_hats_upper,
            os.path.join(out_dir, "regime_qhat_heatmap_upper"),
            title="Asymmetric CQR Upper-Tail q̂ by Gene Regime",
        )
    return calibrator


def _load_validation_pdx_holdout_data(
    canonical_truth: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Load truth pool and holdout prediction tables for Validation_PDX cohorts."""
    truth_parts: List[pd.DataFrame] = [canonical_truth]
    for cohort_name in VALIDATION_PDX_COHORTS:
        cohort_truth = load_truth(cohort_name)
        extra_cols = [c for c in cohort_truth.columns if c not in canonical_truth.columns]
        if extra_cols:
            truth_parts.append(cohort_truth[extra_cols])
    common_truth_genes = truth_parts[0].index
    for part in truth_parts[1:]:
        common_truth_genes = common_truth_genes.intersection(part.index)
    truth_pool = pd.concat([part.loc[common_truth_genes] for part in truth_parts], axis=1)

    ho_preds: Dict[str, Dict[str, Any]] = {}
    for cohort_name, rel_path in VALIDATION_PDX_HOLDOUT_PATHS.items():
        ho_path = PROJECT_ROOT / rel_path
        if not ho_path.exists():
            print(f"  [skip] Holdout predictions not found for {cohort_name}: {rel_path}")
            continue

        cohort_truth = load_truth(cohort_name)
        pmap = _build_prefix_map(cohort_truth.columns, cohort_name)
        ho_full = load_holdout_predictions_proteus_full(str(ho_path))
        for key in ("pred_wide", "ci_lo_wide", "ci_hi_wide", *_MDN_PARAM_KEYS):
            table = ho_full.get(key)
            if table is not None:
                _apply_prefix_map(table, pmap)

        ho_samples = [s for s in ho_full["pred_wide"].columns if s not in canonical_truth.columns]
        if not ho_samples:
            ho_samples = list(ho_full["pred_wide"].columns)
        ho_preds[cohort_name] = {
            key: (ho_full[key][ho_samples] if ho_full.get(key) is not None else None)
            for key in ("pred_wide", "ci_lo_wide", "ci_hi_wide", *_MDN_PARAM_KEYS)
        }

    return truth_pool, ho_preds


def _run_cv_crossfit_calibration(
    cv_full: Dict[str, Any],
    out_dir: Optional[str],
    *,
    overwrite: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Evaluate calibration on OOF data via leave-one-fold-out cross-fitting."""
    fold_data = cv_full.get("fold_data") or {}
    if not fold_data:
        print("  [skip] CV cross-fit calibration: no fold data available")
        return None, None

    cv_out = ensure_dir(os.path.join(out_dir, "CV_CrossFit")) if out_dir is not None else None
    fold_rows: List[Dict[str, Any]] = []
    pooled_truth: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []
    pooled_lo: List[np.ndarray] = []
    pooled_hi: List[np.ndarray] = []
    pooled_lo_adj: List[np.ndarray] = []
    pooled_hi_adj: List[np.ndarray] = []
    adjusted_lo_parts: List[pd.DataFrame] = []
    adjusted_hi_parts: List[pd.DataFrame] = []

    for heldout_fold in sorted(fold_data):
        test_fd = fold_data[heldout_fold]
        test_ci_lo = test_fd.get("ci_lo_wide")
        test_ci_hi = test_fd.get("ci_hi_wide")
        if test_ci_lo is None or test_ci_hi is None:
            continue

        train_truth_parts: List[pd.DataFrame] = []
        train_lo_parts: List[pd.DataFrame] = []
        train_hi_parts: List[pd.DataFrame] = []
        for fold_idx, train_fd in sorted(fold_data.items()):
            if fold_idx == heldout_fold:
                continue
            train_ci_lo = train_fd.get("ci_lo_wide")
            train_ci_hi = train_fd.get("ci_hi_wide")
            if train_ci_lo is None or train_ci_hi is None:
                continue
            keep_samples = [s for s in train_fd["truth_wide"].columns if not str(s).startswith("PHT")]
            if not keep_samples:
                continue
            train_truth_parts.append(train_fd["truth_wide"][keep_samples])
            train_lo_parts.append(train_ci_lo[[s for s in keep_samples if s in train_ci_lo.columns]])
            train_hi_parts.append(train_ci_hi[[s for s in keep_samples if s in train_ci_hi.columns]])

        train_truth = _concat_common_gene_frames(train_truth_parts)
        train_ci_lo = _concat_common_gene_frames(train_lo_parts)
        train_ci_hi = _concat_common_gene_frames(train_hi_parts)
        if train_truth is None or train_ci_lo is None or train_ci_hi is None:
            continue

        regime_df = compute_gene_regimes(train_truth)
        calibrator = RegimeCalibrator.fit(
            train_truth,
            train_ci_lo,
            train_ci_hi,
            regime_df,
            alpha=0.10,
        )

        test_keep = [s for s in test_fd["truth_wide"].columns if not str(s).startswith("PHT")]
        if not test_keep:
            continue
        test_truth = test_fd["truth_wide"][test_keep]
        test_pred = test_fd["pred_wide"][[s for s in test_keep if s in test_fd["pred_wide"].columns]]
        test_ci_lo = test_ci_lo[[s for s in test_keep if s in test_ci_lo.columns]]
        test_ci_hi = test_ci_hi[[s for s in test_keep if s in test_ci_hi.columns]]
        test_ci_lo_adj, test_ci_hi_adj = calibrator.apply(test_ci_lo, test_ci_hi)
        adjusted_lo_parts.append(test_ci_lo_adj)
        adjusted_hi_parts.append(test_ci_hi_adj)

        flat = _flatten_calibration_comparison(
            test_truth,
            test_pred,
            test_ci_lo,
            test_ci_hi,
            test_ci_lo_adj,
            test_ci_hi_adj,
        )
        if flat is None:
            continue

        summary = _summarize_calibration_comparison(flat)
        lower_q_vals = list(calibrator.q_hats_lower.values())
        upper_q_vals = list(calibrator.q_hats_upper.values())
        summary.update(
            {
                "fold": heldout_fold,
                "n_regimes": calibrator.n_regimes,
                "qhat_lower_min": float(min(lower_q_vals)) if lower_q_vals else np.nan,
                "qhat_lower_median": float(np.median(lower_q_vals)) if lower_q_vals else np.nan,
                "qhat_lower_mean": float(np.mean(lower_q_vals)) if lower_q_vals else np.nan,
                "qhat_lower_max": float(max(lower_q_vals)) if lower_q_vals else np.nan,
                "qhat_upper_min": float(min(upper_q_vals)) if upper_q_vals else np.nan,
                "qhat_upper_median": float(np.median(upper_q_vals)) if upper_q_vals else np.nan,
                "qhat_upper_mean": float(np.mean(upper_q_vals)) if upper_q_vals else np.nan,
                "qhat_upper_max": float(max(upper_q_vals)) if upper_q_vals else np.nan,
            }
        )
        fold_rows.append(
            {key: value for key, value in summary.items() if key not in {"base_curve", "adjusted_curve"}}
        )

        pooled_truth.append(flat["truth"])
        pooled_pred.append(flat["pred"])
        pooled_lo.append(flat["ci_lo"])
        pooled_hi.append(flat["ci_hi"])
        pooled_lo_adj.append(flat["ci_lo_adj"])
        pooled_hi_adj.append(flat["ci_hi_adj"])

    if not fold_rows:
        print("  [skip] CV cross-fit calibration: no evaluable folds")
        return None, None

    if cv_out is not None:
        pd.DataFrame(fold_rows).sort_values("fold").to_csv(
            os.path.join(cv_out, "cv_crossfit_by_fold.tsv"), sep="\t", index=False)

    pooled_flat = {
        "truth": np.concatenate(pooled_truth),
        "pred": np.concatenate(pooled_pred),
        "ci_lo": np.concatenate(pooled_lo),
        "ci_hi": np.concatenate(pooled_hi),
        "ci_lo_adj": np.concatenate(pooled_lo_adj),
        "ci_hi_adj": np.concatenate(pooled_hi_adj),
    }
    pooled_summary = _summarize_calibration_comparison(pooled_flat)
    if cv_out is not None:
        _save_calibration_summary(pooled_summary, os.path.join(cv_out, "cv_crossfit_summary.tsv"))
        pd.DataFrame(
            {
                "nominal": pooled_summary["base_curve"]["nominal"],
                "base_empirical": pooled_summary["base_curve"]["empirical"],
                "adjusted_empirical": pooled_summary["adjusted_curve"]["empirical"],
            }
        ).to_csv(os.path.join(cv_out, "cv_crossfit_coverage.tsv"), sep="\t", index=False)

        _maybe(
            plot_calibration_coverage,
            overwrite,
            pooled_summary["base_curve"],
            pooled_summary["adjusted_curve"],
            os.path.join(cv_out, "coverage_comparison"),
            title="CQR Calibration Coverage — CV Cross-Fit OOF",
        )
        pit_truth = cv_full.get("truth_wide")
        if pit_truth is not None:
            pit_keep = [sample for sample in pit_truth.columns if not str(sample).startswith("PHT")]
            if pit_keep:
                _save_pit_diagnostics(
                    pit_truth[pit_keep],
                    {
                        key: (
                            cv_full.get(key)[[sample for sample in pit_keep if sample in cv_full.get(key).columns]]
                            if cv_full.get(key) is not None else None
                        )
                        for key in _MDN_PARAM_KEYS
                    },
                    cv_out,
                    overwrite=overwrite,
                    title="MDN PIT — CV Cross-Fit OOF",
                )

    adjusted_lo = pd.concat(adjusted_lo_parts, axis=1) if adjusted_lo_parts else None
    adjusted_hi = pd.concat(adjusted_hi_parts, axis=1) if adjusted_hi_parts else None
    if adjusted_lo is not None:
        adjusted_lo = adjusted_lo.loc[:, ~adjusted_lo.columns.duplicated()]
    if adjusted_hi is not None:
        adjusted_hi = adjusted_hi.loc[:, ~adjusted_hi.columns.duplicated()]
    return adjusted_lo, adjusted_hi

def run_examples(args: argparse.Namespace) -> None:
    """Generate the consolidated Examples outputs.

    Uses the full PLCA truth pool for the RNA-seq mixture fit and the
    existing PC-PDX study subset for the naive-example plots.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Examples")
    print("=" * 70)

    if _should_skip_section("Examples", args):
        return

    out_dir = _section_results_dir("Examples")
    _write_run_manifest("Examples", out_dir, args)
    plca_truth = _build_plca_truth(args.minimum_tfx)
    if plca_truth.empty:
        raise ValueError("PLCA truth pool is empty; cannot build Examples outputs")
    run_examples_workflow(out_dir, plca_truth, args.overwrite)
    print("\n  Done")
def _run_calibration_evaluation(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    ho_ci_lo: pd.DataFrame,
    ho_ci_hi: pd.DataFrame,
    calibrator: RegimeCalibrator,
    label: str,
    out_dir: str,
    *,
    overwrite: bool = False,
) -> None:
    """Evaluate CQR regime-based calibration on holdout data.

    Compares base (uncalibrated) intervals against the CQR-calibrated
    intervals using per-regime normalised scores.  Produces:
      - Coverage comparison plot (base vs calibrated)
      - Regime coverage heatmap (per-regime empirical coverage on HO)
      - Error-vs-width diagnostic
      - Calibrated prediction TSV
      - Per-sample expression KDEs

    Parameters
    ----------
    truth_wide : pd.DataFrame
        Truth expression values for the evaluation set.
    pred_wide : pd.DataFrame
        Predictions for the evaluation set.
    ho_ci_lo, ho_ci_hi : pd.DataFrame
        Raw (uncalibrated) lower/upper CI bounds from the model.
    calibrator : RegimeCalibrator
        CQR calibrator previously fit on OOF data.
    label : str
        Label for titles/prints (e.g. "BTS").
    out_dir : str
        Output directory (files saved directly here).
    overwrite : bool
        Whether to overwrite existing plot files.
    """
    common = truth_wide.columns.intersection(pred_wide.columns)
    if len(common) < 2:
        return

    tw = truth_wide[common]
    pw = pred_wide[common]
    lo = ho_ci_lo.reindex_like(tw)
    hi = ho_ci_hi.reindex_like(tw)

    # Apply CQR calibration
    lo_adj, hi_adj = calibrator.apply(lo, hi)

    # Save calibrated predictions TSV
    cal_path = os.path.join(out_dir, "calibrated_predictions_ho.tsv")
    cal_df = pd.DataFrame({
        "prediction_mean": pw.stack(),
        "ci_lo_calibrated": lo_adj.reindex_like(pw).stack(),
        "ci_hi_calibrated": hi_adj.reindex_like(pw).stack(),
    })
    cal_df.index.names = ["gene", "sample"]
    cal_df = cal_df.reset_index()
    cal_df.to_csv(cal_path, sep="\t", index=False)
    print(f"    Calibrated predictions saved ({len(cal_df)} rows)")

    flat = _flatten_calibration_comparison(tw, pw, lo, hi, lo_adj, hi_adj)
    if flat is None:
        print("    [skip] Too few valid observations for calibration evaluation")
        return
    summary = _summarize_calibration_comparison(flat)
    _save_calibration_summary(summary, os.path.join(out_dir, "calibration_summary.tsv"))

    # Diagnostic: CI widths before/after
    base_widths = flat["ci_hi"] - flat["ci_lo"]
    adj_widths = flat["ci_hi_adj"] - flat["ci_lo_adj"]
    base_median_w = summary["base_median_width"]
    adj_median_w = summary["adjusted_median_width"]
    n_collapsed = int(np.sum(adj_widths == 0))
    print(f"    CI median width: base={base_median_w:.3f}  CQR-adjusted={adj_median_w:.3f}")
    if n_collapsed > 0:
        pct = 100 * n_collapsed / len(adj_widths)
        print(f"    [warn] {n_collapsed} intervals collapsed to zero-width ({pct:.1f}%)")

    print(f"  CQR calibration evaluation ({label} HO):")
    target_pct = int(round(100 * _CALIBRATION_INTERVAL_NOMINAL))
    print(f"    Base  {target_pct}% coverage: {summary[f'base_coverage_{target_pct}']:.3f}")
    print(f"    CQR   {target_pct}% coverage: {summary[f'adjusted_coverage_{target_pct}']:.3f}")

    _maybe(plot_calibration_coverage, overwrite,
           summary["base_curve"], summary["adjusted_curve"],
           os.path.join(out_dir, "coverage_comparison"),
           title=f"CQR Calibration Coverage — {label} HO")

    # Regime coverage heatmap on HO data
    ho_eval = calibrator.evaluate(tw, pw, lo, hi)
    if "regime_coverage" in ho_eval:
        _maybe(plot_regime_coverage_heatmap, overwrite,
               ho_eval["regime_coverage"],
               os.path.join(out_dir, "regime_coverage_heatmap"),
               title=f"CQR Regime Coverage — {label} HO")

    # Error-vs-width diagnostic: per-gene interval width vs gene-level metrics
    gmet = gene_level_metrics(tw, pw, n_workers=1)
    width = (hi - lo).mean(axis=1)
    width_adj = (hi_adj - lo_adj).mean(axis=1)

    _maybe(plot_error_vs_width, overwrite,
           width, width_adj, gmet,
           os.path.join(out_dir, "error_vs_width"),
           title=f"Error vs Width — {label} HO")

    # Per-sample prediction KDEs
    _maybe(sample_expression_kde, overwrite, pw,
           os.path.join(out_dir, "expression_profiles_ho"),
           subtypes_df=None)


def _run_reliability_evaluation(
    ho_data: Dict[str, Any],
    dev_truth_wide: pd.DataFrame,
    truth_pool_wide: pd.DataFrame,
    reliability_model: ReliabilityModel,
    cohort_name: str,
    out_dir: str,
    *,
    overwrite: bool = False,
) -> None:
    """Evaluate the gene-level reliability model on holdout data.

    Computes gene-level features from HO predictions (using all available
    MDN columns), uses the pre-trained ridge model to predict per-gene
    reliability r̂, and compares against true per-gene Pearson r on HO
    samples.

    Parameters
    ----------
    ho_data : dict from load_holdout_predictions_proteus_full (subset to
        holdout samples).  Keys: pred_wide, ci_lo_wide, ci_hi_wide,
        p0_wide, pi_gamma_wide, pi_normal_wide, gamma_beta_wide,
        normal_stdev_wide.
    dev_truth_wide : Development truth pool used for static gene features.
    truth_pool_wide : Truth pool that includes the evaluated holdout samples.
    reliability_model : Pre-trained reliability model (fitted on OOF data).
    cohort_name : Label for the cohort.
    out_dir : Output directory for this cohort.
    overwrite : Whether to overwrite existing plot files.
    """
    rel_out = ensure_dir(os.path.join(out_dir, "reliability"))

    ho_pred = ho_data["pred_wide"]
    ho_ci_lo = ho_data.get("ci_lo_wide")
    ho_ci_hi = ho_data.get("ci_hi_wide")

    if ho_ci_lo is None or ho_ci_hi is None:
        print(f"    [skip] No CIs for reliability evaluation on {cohort_name} HO")
        return

    ho_truth = truth_pool_wide.reindex(index=ho_pred.index, columns=ho_pred.columns)
    common = ho_truth.columns.intersection(ho_pred.columns)
    if len(common) < 2:
        return

    pw = ho_pred[common]
    tw = ho_truth[common]
    lo = ho_ci_lo.reindex_like(pw)
    hi = ho_ci_hi.reindex_like(pw)

    # Build kwargs for compute_gene_features with all available MDN columns
    feat_kwargs: Dict[str, Any] = {}
    for mk in ("p0_wide", "pi_gamma_wide", "pi_normal_wide",
                "gamma_beta_wide", "normal_stdev_wide"):
        v = ho_data.get(mk)
        if v is not None:
            feat_kwargs[mk] = v.reindex_like(pw)

    ho_features = compute_gene_features(
        pw, lo, hi, dev_truth_wide, **feat_kwargs,
    )
    ho_gene_r = _compute_gene_pearson_r(tw, pw)
    print(f"    HO features: {ho_features.shape[0]} genes × {ho_features.shape[1]} features")
    print(f"    HO gene r: {ho_gene_r.notna().sum()} genes with valid Pearson r")

    # Predict reliability and evaluate
    predicted_r = reliability_model.predict(ho_features)
    eval_result = reliability_model.evaluate(predicted_r, ho_gene_r)
    print(f"    Reliability evaluation ({cohort_name} HO):")
    print(f"      Spearman ρ = {eval_result['spearman_rho']:.4f}")
    print(f"      RMSE(z)    = {eval_result['rmse_z']:.4f}")

    # Scatter: predicted r̂ vs true r
    _maybe(plot_reliability_scatter, overwrite,
           predicted_r, ho_gene_r,
           os.path.join(rel_out, "reliability_scatter"),
           title=f"Reliability — {cohort_name} HO")

    # Retention curve: gene retention by predicted reliability
    ret = retention_curve(predicted_r, ho_gene_r)
    baseline_mean_r = float(ho_gene_r.dropna().mean()) if ho_gene_r.notna().any() else None
    _maybe(plot_retention_curve, overwrite,
           ret,
           os.path.join(rel_out, "retention_curve"),
           baseline_mean_r=baseline_mean_r,
           title=f"Retention Curve — {cohort_name} HO")

    # Save gene reliability scores
    rel_path = os.path.join(rel_out, "gene_reliability_scores.tsv")
    rel_df = pd.DataFrame({
        "predicted_r": predicted_r,
        "true_r": ho_gene_r,
    })
    rel_df.index.name = "gene"
    rel_df = rel_df.sort_values("predicted_r", ascending=False)
    rel_df.to_csv(rel_path, sep="\t")
    print(f"    Gene reliability scores saved ({len(rel_df)} rows)")


def _run_cv_suite(
    pred_wide: pd.DataFrame,
    truth_wide: pd.DataFrame,
    label: str,
    out_dir: str,
    *,
    ci_lo: Optional[pd.DataFrame] = None,
    ci_hi: Optional[pd.DataFrame] = None,
    activity_odds_ratio_wide: Optional[pd.DataFrame] = None,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    active_confidence_wide: Optional[pd.DataFrame] = None,
    active_threshold: Optional[float] = None,
    goi: Optional[Dict[str, List[str]]] = None,
    subtypes_df: Optional[pd.DataFrame] = None,
    heatmap_gene_sets: Optional[List[str]] = None,
    gene_scatter_sets: Optional[List[str]] = None,
    var_map: Optional[Dict[str, float]] = None,
    gene_met: Optional[pd.DataFrame] = None,
    sample_met: Optional[pd.DataFrame] = None,
    n_workers: int = 1,
    overwrite: bool = False,
    scatter_ci: bool = False,
    dim_ambiguous: bool = False,
) -> None:
    """Full plot/output suite for one cohort's combined CV+HO results.

    Parameters
    ----------
    goi : dict mapping subtype → gene list for sample scatter highlights.
    subtypes_df : DataFrame(index=sample, columns=[subtype, tfx, status, ...]).
        If ``status`` column is present, holdout samples get hollow markers.
    heatmap_gene_sets : which gene set names to make double-heatmaps for.
    gene_scatter_sets : gene set names to generate batch gene scatters for.
    var_map : pre-computed gene variance from full study cohort.
    """
    t0 = time.time()
    print(f"\n── {label} (plots + outputs) ──")

    if heatmap_gene_sets is None:
        heatmap_gene_sets = ["PC-Phenotyping", "PAM50", "Targets_CRPC"]
    if goi is None:
        goi = DEFAULT_GOI_PC
    if var_map is None:
        var_map = compute_gene_variance(truth_wide)

    # Gene- and sample-level metrics (reuse from cohort_summary_table if provided)
    if gene_met is None:
        gene_met = gene_level_metrics(truth_wide, pred_wide, n_workers=n_workers)
    if sample_met is None:
        sample_met = sample_level_metrics(truth_wide, pred_wide, n_workers=n_workers)

    # Save metric TSVs
    gene_met.to_csv(os.path.join(out_dir, "gene_metrics.tsv"), sep="\t")
    sample_met.to_csv(os.path.join(out_dir, "sample_metrics.tsv"), sep="\t")

    # Variance group summary
    vg = variance_group_metrics(truth_wide, pred_wide, var_map, gene_met=gene_met)
    vg.to_csv(os.path.join(out_dir, "variance_group_metrics.tsv"), sep="\t", index=False)
    print(f"  Variance groups:\n{vg.to_string(index=False)}")

    # ── Gene metric distributions (square, dashed zero line) ──
    _maybe(metric_distribution, overwrite, gene_met["pearson_r"], "pearson_r",
           os.path.join(out_dir, "gene_pearson_r_dist"), xlim=(-1, 1), square=True)

    _maybe(metric_distribution, overwrite, gene_met["spearman_rho"], "spearman_rho",
           os.path.join(out_dir, "gene_spearman_rho_dist"), xlim=(-1, 1), square=True)

    _maybe(metric_distribution, overwrite, gene_met["rmse"], "gene_rmse",
           os.path.join(out_dir, "gene_rmse_dist"), square=True)

    # Variance-colored metric distributions
    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "pearson_r",
           os.path.join(out_dir, "gene_pearson_r_dist_by_variance"), xlim=(-1, 1),
           title=f"{label} — Pearson r by Variance Group", square=True)

    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "spearman_rho",
           os.path.join(out_dir, "gene_spearman_rho_dist_by_variance"), xlim=(-1, 1),
           title=f"{label} — Spearman ρ by Variance Group", square=True)

    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "rmse",
           os.path.join(out_dir, "gene_rmse_dist_by_variance"),
           title=f"{label} — RMSE by Variance Group", square=True)

    # Double heatmap for key gene sets (with breaks and subtypes)
    for gs_name in heatmap_gene_sets:
        resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
        if not gene_list_available(resolved):
            continue
        genes = get_gene_list(resolved)
        breaks = GENE_SET_BREAKS.get(gs_name)
        truth_sort_genes = None
        if gs_name == "PC-Phenotyping" and breaks:
            truth_sort_genes = genes[:breaks[-1]]
        _maybe(double_heatmap, overwrite, truth_wide, pred_wide, genes,
               os.path.join(out_dir, f"double_heatmap_{gs_name}"),
               subtypes=subtypes_df, breaks=breaks, truth_sort_genes=truth_sort_genes)

    # Per-gene target scatters for GOI sets (tumor only)
    if gene_scatter_sets:
        for gs_name in gene_scatter_sets:
            resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
            if not gene_list_available(resolved):
                continue
            genes = get_gene_list(resolved)
            gs_dir = ensure_dir(os.path.join(out_dir, f"gene_tumor_scatters_{gs_name}"))
            _ci_lo_gs, _ci_hi_gs = _gene_level_ci_inputs(
                ci_lo,
                ci_hi,
                scatter_ci=scatter_ci,
                dim_ambiguous=dim_ambiguous,
            )
            batch_gene_scatters(
                truth_wide, pred_wide, genes, gs_dir,
                subtypes_df=subtypes_df,
                ci_lo_wide=_ci_lo_gs, ci_hi_wide=_ci_hi_gs,
                activity_odds_ratio_wide=activity_odds_ratio_wide,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
                active_confidence_wide=active_confidence_wide,
                active_threshold=active_threshold,
                dim_threshold_genes=dim_ambiguous,
                n_workers=n_workers,
            )

    # Per-sample scatters (parallelised) — exclude PHT admix samples
    sample_dir = ensure_dir(os.path.join(out_dir, "sample_scatters"))
    non_pht_samples = [s for s in truth_wide.columns if not str(s).startswith("PHT")]
    tw_no_pht = truth_wide[non_pht_samples]
    pw_no_pht = pred_wide[[s for s in non_pht_samples if s in pred_wide.columns]]
    batch_sample_scatters(
        tw_no_pht, pw_no_pht, sample_dir,
        goi=goi, subtypes_df=subtypes_df, n_workers=n_workers,
    )

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")


def _run_cv_suite_hdl(
    pred_wide: pd.DataFrame,
    truth_wide: pd.DataFrame,
    label: str,
    out_dir: str,
    *,
    var_map: Optional[Dict[str, float]] = None,
    gene_met: Optional[pd.DataFrame] = None,
    sample_met: Optional[pd.DataFrame] = None,
    n_workers: int = 1,
    overwrite: bool = False,
) -> None:
    """Simplified plot/output suite for HD-L background predictions.

    Produces metric distributions and sample scatters only (no gene
    scatters, heatmaps, or reliability).
    """
    t0 = time.time()
    print(f"\n── {label} (plots + outputs) ──")

    if var_map is None:
        var_map = compute_gene_variance(truth_wide)

    if gene_met is None:
        gene_met = gene_level_metrics(truth_wide, pred_wide, n_workers=n_workers)
    if sample_met is None:
        sample_met = sample_level_metrics(truth_wide, pred_wide, n_workers=n_workers)

    gene_met.to_csv(os.path.join(out_dir, "gene_metrics.tsv"), sep="\t")
    sample_met.to_csv(os.path.join(out_dir, "sample_metrics.tsv"), sep="\t")

    vg = variance_group_metrics(truth_wide, pred_wide, var_map, gene_met=gene_met)
    vg.to_csv(os.path.join(out_dir, "variance_group_metrics.tsv"), sep="\t", index=False)
    print(f"  Variance groups:\n{vg.to_string(index=False)}")

    # Gene metric distributions
    _maybe(metric_distribution, overwrite, gene_met["pearson_r"], "pearson_r",
           os.path.join(out_dir, "gene_pearson_r_dist"), xlim=(-1, 1), square=True)
    _maybe(metric_distribution, overwrite, gene_met["spearman_rho"], "spearman_rho",
           os.path.join(out_dir, "gene_spearman_rho_dist"), xlim=(-1, 1), square=True)
    _maybe(metric_distribution, overwrite, gene_met["rmse"], "gene_rmse",
           os.path.join(out_dir, "gene_rmse_dist"), square=True)

    # Variance-colored distributions
    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "pearson_r",
           os.path.join(out_dir, "gene_pearson_r_dist_by_variance"), xlim=(-1, 1),
           title=f"{label} — Pearson r by Variance Group", square=True)
    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "spearman_rho",
           os.path.join(out_dir, "gene_spearman_rho_dist_by_variance"), xlim=(-1, 1),
           title=f"{label} — Spearman ρ by Variance Group", square=True)
    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "rmse",
           os.path.join(out_dir, "gene_rmse_dist_by_variance"),
           title=f"{label} — RMSE by Variance Group", square=True)

    # Sample scatters
    sample_dir = ensure_dir(os.path.join(out_dir, "sample_scatters"))
    batch_sample_scatters(
        truth_wide, pred_wide, sample_dir,
        goi=None, subtypes_df=None, n_workers=n_workers,
    )

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Analysis group: Patient_Validation (Holdout — Patient Biopsies)
# ---------------------------------------------------------------------------


def _run_holdout_suite(
    pred_wide: pd.DataFrame,
    truth_wide: pd.DataFrame,
    label: str,
    out_dir: str,
    *,
    ci_lo: Optional[pd.DataFrame] = None,
    ci_hi: Optional[pd.DataFrame] = None,
    subtypes_df: Optional[pd.DataFrame] = None,
    heatmap_gene_sets: Optional[List[str]] = None,
    goi: Optional[Dict[str, List[str]]] = None,
    n_workers: int = 1,
    overwrite: bool = False,
    summaries: Optional[List[Dict]] = None,
    scatter_ci: bool = False,
) -> Dict[str, Any]:
    """Full analysis suite for holdout predictions against truth."""
    t0 = time.time()
    print(f"\n── {label} ──")

    # Align
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    print(f"  Common: {len(common_genes)} genes × {len(common_samples)} samples")

    if len(common_samples) < 2:
        print(f"  [skip] Too few common samples for {label}")
        return {}

    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    summary = cohort_summary_table(tw, pw, label=label, n_workers=n_workers)
    if summaries is not None:
        summaries.append(summary)

    gene_met: pd.DataFrame = summary["_gene_metrics"]
    sample_met: pd.DataFrame = summary["_sample_metrics"]
    var_map = compute_gene_variance(tw)

    gene_met.to_csv(os.path.join(out_dir, "gene_metrics.tsv"), sep="\t")
    sample_met.to_csv(os.path.join(out_dir, "sample_metrics.tsv"), sep="\t")

    vg = variance_group_metrics(tw, pw, var_map, gene_met=gene_met)
    vg.to_csv(os.path.join(out_dir, "variance_group_metrics.tsv"), sep="\t", index=False)
    print(f"  Variance groups:\n{vg.to_string(index=False)}")

    # Plots
    _maybe(metric_distribution, overwrite, gene_met["pearson_r"], "pearson_r",
           os.path.join(out_dir, "gene_pearson_r_dist"), xlim=(-1, 1), square=True)

    _maybe(metric_distribution, overwrite, gene_met["spearman_rho"], "spearman_rho",
           os.path.join(out_dir, "gene_spearman_rho_dist"), xlim=(-1, 1), square=True)

    _maybe(metric_distribution, overwrite, gene_met["rmse"], "gene_rmse",
           os.path.join(out_dir, "gene_rmse_dist"), square=True)

    # Variance-colored metric distributions
    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "pearson_r",
           os.path.join(out_dir, "gene_pearson_r_dist_by_variance"), xlim=(-1, 1), square=True)

    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "spearman_rho",
           os.path.join(out_dir, "gene_spearman_rho_dist_by_variance"), xlim=(-1, 1), square=True)

    _maybe(metric_distribution_by_variance, overwrite, gene_met, var_map, "rmse",
           os.path.join(out_dir, "gene_rmse_dist_by_variance"), square=True)

    # Double heatmaps
    _heatmap_sets = heatmap_gene_sets if heatmap_gene_sets is not None else [
        "PC-Phenotyping", "PAM50", "Targets_CRPC",
    ]
    for gs_name in _heatmap_sets:
        resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
        if not gene_list_available(resolved):
            continue
        genes = get_gene_list(resolved)
        breaks = GENE_SET_BREAKS.get(gs_name)
        truth_sort_genes = None
        if gs_name == "PC-Phenotyping" and breaks:
            truth_sort_genes = genes[:breaks[-1]]
        _maybe(double_heatmap, overwrite, tw, pw, genes,
               os.path.join(out_dir, f"double_heatmap_{gs_name}"),
               subtypes=subtypes_df, breaks=breaks, truth_sort_genes=truth_sort_genes)

    # Per-sample scatters
    sample_dir = ensure_dir(os.path.join(out_dir, "sample_scatters"))

    _goi = goi if goi is not None else DEFAULT_GOI_PC
    batch_sample_scatters(
        tw, pw, sample_dir,
        goi=_goi, subtypes_df=subtypes_df, n_workers=n_workers,
    )

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")
    return summary


# Per-cohort gene-set config: heatmap gene sets, scatter gene sets, and GOI.
_VALIDATION_COHORT_CONFIG: Dict[str, Dict[str, Any]] = {
    "CRPC-WCDT": {
        "heatmap": ["PC-Phenotyping", "Targets_CRPC"],
        "scatter": ["Targets_CRPC"],
        "goi": DEFAULT_GOI_PC,
    },
    "BLCA-RA": {
        "heatmap": ["Targets_BLCA"],
        "scatter": ["Targets_BLCA"],
        "goi": DEFAULT_GOI_BLCA,
    },
    "CRPC-RA": {
        "heatmap": ["PC-Phenotyping", "Targets_CRPC"],
        "scatter": ["Targets_CRPC"],
        "goi": DEFAULT_GOI_PC,
    },
    "SCLC": {
        "heatmap": ["Targets_SCLC"],
        "scatter": ["Targets_SCLC"],
        "goi": DEFAULT_GOI_SCLC,
    },
}

# Composite target gene list for Collected gene scatters (union across cancers).
_COLLECTED_TARGET_SETS = ["Targets_CRPC", "Targets_BLCA", "Targets_SCLC"]


def _run_patient_validation_cohort(
    cohort: str,
    model_path_template: str,
    out_dir: str,
    args: argparse.Namespace,
    *,
    interval_calibrator: Optional[RegimeCalibrator] = None,
    activity_threshold_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run Patient_Validation analysis for a single cohort.

    Loads holdout predictions from ``model_path_template``, runs the full
    per-cohort suite (metrics + plots), and returns data for Collected.

    Returns a dict with cohort-level data for the Collected analysis,
    or None if the cohort is skipped.
    """
    truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
    if not truth_path.exists():
        print(f"  [skip] Truth not found: {cohort}")
        return None
    truth_wide = load_truth(cohort)
    subtypes_df = _load_filtered_subtypes_for_reference(
        cohort,
        truth_wide.columns,
        minimum_tfx=args.minimum_tfx,
        report_context=f"Validation_Patient {cohort}",
    )
    if args.minimum_tfx > 0.0 and subtypes_df is not None:
        truth_wide = _filter_wide_to_samples(truth_wide, subtypes_df.index)
        if truth_wide is not None and truth_wide.shape[1] == 0:
            print(f"  [skip] No samples remain for {cohort} after minimum TFx filtering")
            return None

    # Build short→prefixed map and add prefixes to prediction columns
    pmap = _build_prefix_map(truth_wide.columns, cohort)

    tfx_dict = None
    if subtypes_df is not None and "tfx" in subtypes_df.columns:
        tfx_dict = subtypes_df["tfx"].to_dict()

    cohort_cfg = _VALIDATION_COHORT_CONFIG.get(cohort, {
        "heatmap": ["PC-Phenotyping", "Targets_CRPC"],
        "scatter": ["Targets_CRPC"],
        "goi": DEFAULT_GOI_PC,
    })

    # Output directly into cohort folder (no model subfolder)
    cohort_out = ensure_dir(os.path.join(out_dir, cohort))
    summaries: List[Dict[str, Any]] = []

    path = PROJECT_ROOT / model_path_template.format(cohort=cohort)
    if not path.exists():
        print(f"    [skip] Predictions not found: {path.name}")
        return None

    full_data = load_holdout_predictions_proteus_full(str(path))
    for key in ("pred_wide", "ci_lo_wide", "ci_hi_wide", "activity_odds_ratio_wide", "active_confidence_wide", *_MDN_PARAM_KEYS):
        _apply_prefix_map(full_data.get(key), pmap)
    if args.minimum_tfx > 0.0 and subtypes_df is not None:
        full_data = _filter_prediction_bundle_to_samples(full_data, subtypes_df.index)
    analysis_data = _prepare_proteus_analysis_bundle(
        full_data,
        interval_calibrator=interval_calibrator,
        calibrate_cis=args.calibrate_cis,
    )
    pred = analysis_data["pred_wide"]
    ci_lo = analysis_data.get("ci_lo_wide")
    ci_hi = analysis_data.get("ci_hi_wide")
    activity_odds_ratio_wide = analysis_data.get("activity_odds_ratio_wide")
    active_confidence_wide = analysis_data.get("active_confidence_wide")
    activity_lower_threshold = (
        float(activity_threshold_summary["lower_threshold"])
        if activity_threshold_summary is not None and activity_odds_ratio_wide is not None
        else None
    )
    activity_upper_threshold = (
        float(activity_threshold_summary["upper_threshold"])
        if activity_threshold_summary is not None and activity_odds_ratio_wide is not None
        else None
    )
    active_threshold = (
        _DOWNSTREAM_ACTIVE_DIM_THRESHOLD
        if active_confidence_wide is not None
        else None
    )
    label = f"{cohort}"

    suite_result = _run_holdout_suite(
        pred, truth_wide, label, cohort_out,
        ci_lo=ci_lo, ci_hi=ci_hi, subtypes_df=subtypes_df,
        heatmap_gene_sets=cohort_cfg["heatmap"],
        goi=cohort_cfg["goi"],
        n_workers=args.n_workers, overwrite=args.overwrite,
        summaries=summaries, scatter_ci=args.scatter_ci,
    )
    if suite_result:
        _attach_median_point_summary(
            suite_result,
            truth_wide,
            {key: full_data.get(key) for key in _MDN_PARAM_KEYS},
            n_workers=args.n_workers,
        )

    # Gene target scatters
    _ci_lo_tgt, _ci_hi_tgt = _gene_level_ci_inputs(
        ci_lo,
        ci_hi,
        scatter_ci=args.scatter_ci,
        dim_ambiguous=args.dim_ambiguous,
    )
    for gs_name in cohort_cfg.get("scatter", []):
        resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
        if not gene_list_available(resolved):
            continue
        genes = get_gene_list(resolved)
        target_dir = ensure_dir(
            os.path.join(cohort_out, f"target_scatters_{gs_name}"))
        batch_gene_scatters(
            truth_wide, pred, genes, target_dir,
            subtypes_df=subtypes_df, tfx_dict=tfx_dict,
            ci_lo_wide=_ci_lo_tgt, ci_hi_wide=_ci_hi_tgt,
            activity_odds_ratio_wide=activity_odds_ratio_wide,
            lower_threshold=activity_lower_threshold,
            upper_threshold=activity_upper_threshold,
            active_confidence_wide=active_confidence_wide,
            active_threshold=active_threshold,
            dim_threshold_genes=args.dim_ambiguous,
            n_workers=args.n_workers,
        )

    if summaries:
        print(f"\n  ── {cohort} Summary ──")
        print_summary_table(summaries)
        _save_summary_tsv(summaries, os.path.join(cohort_out, "summary_metrics.tsv"))

    # IHC correlation analysis (for cohorts with IHC scoring data)
    _run_ihc_analysis(
        truth_wide,
        pred,
        cohort,
        cohort_out,
        ci_lo_wide=ci_lo,
        ci_hi_wide=ci_hi,
        activity_odds_ratio_wide=activity_odds_ratio_wide,
        lower_threshold=activity_lower_threshold,
        upper_threshold=activity_upper_threshold,
        active_confidence_wide=active_confidence_wide,
        active_threshold=active_threshold,
        subtypes_df=subtypes_df,
        tfx_dict=tfx_dict,
        dim_ambiguous=args.dim_ambiguous,
        overwrite=args.overwrite,
    )

    _run_pam50_analysis(
        pred,
        cohort_out,
        args.overwrite,
        cohort_label=cohort,
        truth_wide=truth_wide,
    )

    # Return data for Collected analysis
    return {
        "cohort": cohort,
        "truth_wide": truth_wide,
        "pred_wide": pred,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "activity_odds_ratio_wide": activity_odds_ratio_wide,
        "active_confidence_wide": active_confidence_wide,
        "subtypes_df": subtypes_df,
        "tfx_dict": tfx_dict,
        "status": "holdout",
        **{key: full_data.get(key) for key in _MDN_PARAM_KEYS},
    }


def _run_patient_validation_cv_cohort(
    cohort: str,
    cv_pred: pd.DataFrame,
    cv_ci_lo: Optional[pd.DataFrame],
    cv_ci_hi: Optional[pd.DataFrame],
    cv_full: Dict[str, Any],
    cohort_map_cv: Dict[str, str],
    out_dir: str,
    args: argparse.Namespace,
    activity_threshold_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run patient-validation analysis for one cohort using OOF CV predictions."""
    truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
    if not truth_path.exists():
        print(f"    [skip] Truth not found: {cohort}")
        return None

    truth_wide = load_truth(cohort)
    subtypes_df = _load_filtered_subtypes_for_reference(
        cohort,
        truth_wide.columns,
        minimum_tfx=args.minimum_tfx,
        report_context=f"Validation_Patient {cohort}",
    )
    if args.minimum_tfx > 0.0 and subtypes_df is not None:
        truth_wide = _filter_wide_to_samples(truth_wide, subtypes_df.index)
    pmap = _build_prefix_map(truth_wide.columns, cohort)
    allowed_samples = set(subtypes_df.index) if args.minimum_tfx > 0.0 and subtypes_df is not None else None
    cohort_samples = [
        s for s in cv_pred.columns
        if cohort_map_cv.get(s) == cohort and (allowed_samples is None or s in allowed_samples)
    ]
    if not cohort_samples:
        print(f"    [skip] No CV samples for {cohort}")
        return None

    pred_c = cv_pred[cohort_samples].copy()
    _apply_prefix_map(pred_c, pmap)
    ci_lo_c = (
        cv_ci_lo[[s for s in cohort_samples if s in cv_ci_lo.columns]].copy()
        if cv_ci_lo is not None else None
    )
    ci_hi_c = (
        cv_ci_hi[[s for s in cohort_samples if s in cv_ci_hi.columns]].copy()
        if cv_ci_hi is not None else None
    )
    _apply_prefix_map(ci_lo_c, pmap)
    _apply_prefix_map(ci_hi_c, pmap)

    common_samples = truth_wide.columns.intersection(pred_c.columns)
    if len(common_samples) == 0:
        print(f"    [skip] No overlapping truth/pred samples for {cohort}")
        return None

    truth_c = truth_wide[common_samples]
    pred_c = pred_c[common_samples]
    if ci_lo_c is not None:
        ci_lo_c = ci_lo_c[[s for s in common_samples if s in ci_lo_c.columns]]
    if ci_hi_c is not None:
        ci_hi_c = ci_hi_c[[s for s in common_samples if s in ci_hi_c.columns]]
    mdn_tables: Dict[str, Optional[pd.DataFrame]] = {}
    for key in _MDN_PARAM_KEYS:
        table = cv_full.get(key)
        if table is None:
            mdn_tables[key] = None
            continue
        table_c = table[[s for s in cohort_samples if s in table.columns]].copy()
        _apply_prefix_map(table_c, pmap)
        mdn_tables[key] = table_c[[s for s in common_samples if s in table_c.columns]]
    activity_odds_ratio_wide = None
    activity_odds_ratio_table = cv_full.get("activity_odds_ratio_wide")
    if activity_odds_ratio_table is not None:
        activity_odds_ratio_c = activity_odds_ratio_table[[s for s in cohort_samples if s in activity_odds_ratio_table.columns]].copy()
        _apply_prefix_map(activity_odds_ratio_c, pmap)
        activity_odds_ratio_wide = activity_odds_ratio_c[[s for s in common_samples if s in activity_odds_ratio_c.columns]]
    elif mdn_tables.get("p0_wide") is not None and mdn_tables.get("pi_gamma_wide") is not None and mdn_tables.get("pi_normal_wide") is not None:
        activity_odds_ratio_wide = mdn_activity_odds_ratio_wide(
            mdn_tables["p0_wide"],
            mdn_tables["pi_gamma_wide"],
            mdn_tables["pi_normal_wide"],
        )
    active_confidence_wide = None
    if all(mdn_tables.get(key) is not None for key in _MDN_PARAM_KEYS):
        active_confidence_wide = mdn_on_probability_wide(
            mdn_tables["p0_wide"],
            mdn_tables["pi_gamma_wide"],
            mdn_tables["pi_normal_wide"],
            mdn_tables["gamma_beta_wide"],
            mdn_tables["normal_mean_wide"],
            mdn_tables["normal_stdev_wide"],
        )
    activity_lower_threshold = (
        float(activity_threshold_summary["lower_threshold"])
        if activity_threshold_summary is not None and activity_odds_ratio_wide is not None
        else None
    )
    activity_upper_threshold = (
        float(activity_threshold_summary["upper_threshold"])
        if activity_threshold_summary is not None and activity_odds_ratio_wide is not None
        else None
    )
    active_threshold = (
        _DOWNSTREAM_ACTIVE_DIM_THRESHOLD
        if active_confidence_wide is not None
        else None
    )

    tfx_dict = None
    if subtypes_df is not None and "tfx" in subtypes_df.columns:
        tfx_dict = subtypes_df["tfx"].to_dict()

    cohort_cfg = _VALIDATION_COHORT_CONFIG.get(cohort, {
        "heatmap": ["PC-Phenotyping", "Targets_CRPC"],
        "scatter": ["Targets_CRPC"],
        "goi": DEFAULT_GOI_PC,
    })
    cohort_out = ensure_dir(os.path.join(out_dir, cohort))

    summary = cohort_summary_table(
        truth_c, pred_c, label=f"Proteus CV — {cohort}", n_workers=args.n_workers,
    )
    _attach_median_point_summary(
        summary,
        truth_c,
        mdn_tables,
        n_workers=args.n_workers,
    )
    _run_cv_suite(
        pred_c,
        truth_c,
        f"Proteus CV ({cohort})",
        cohort_out,
        ci_lo=ci_lo_c,
        ci_hi=ci_hi_c,
        activity_odds_ratio_wide=activity_odds_ratio_wide,
        lower_threshold=activity_lower_threshold,
        upper_threshold=activity_upper_threshold,
        active_confidence_wide=active_confidence_wide,
        active_threshold=active_threshold,
        subtypes_df=subtypes_df,
        heatmap_gene_sets=cohort_cfg["heatmap"],
        goi=cohort_cfg["goi"],
        gene_scatter_sets=cohort_cfg.get("scatter", []),
        gene_met=summary["_gene_metrics"],
        sample_met=summary["_sample_metrics"],
        n_workers=args.n_workers,
        overwrite=args.overwrite,
        scatter_ci=args.scatter_ci,
        dim_ambiguous=args.dim_ambiguous,
    )

    if cohort == "CRPC-RA":
        _run_crpc_ra_metastasis_composition_analysis(
            truth_c,
            pred_c,
            cohort_out,
            subtypes_df=subtypes_df,
            overwrite=args.overwrite,
        )

    if cohort in _IHC_FILES:
        _run_ihc_analysis(
            truth_c,
            pred_c,
            cohort,
            cohort_out,
            ci_lo_wide=ci_lo_c,
            ci_hi_wide=ci_hi_c,
            activity_odds_ratio_wide=activity_odds_ratio_wide,
            lower_threshold=activity_lower_threshold,
            upper_threshold=activity_upper_threshold,
            active_confidence_wide=active_confidence_wide,
            active_threshold=active_threshold,
            subtypes_df=subtypes_df,
            tfx_dict=tfx_dict,
            dim_ambiguous=args.dim_ambiguous,
            overwrite=args.overwrite,
        )

    _run_pam50_analysis(
        pred_c,
        cohort_out,
        args.overwrite,
        cohort_label=cohort,
        truth_wide=truth_c,
    )

    _save_summary_tsv([summary], os.path.join(cohort_out, "summary_metrics.tsv"))
    return {
        "cohort": cohort,
        "truth_wide": truth_c,
        "pred_wide": pred_c,
        "ci_lo": ci_lo_c,
        "ci_hi": ci_hi_c,
        "activity_odds_ratio_wide": activity_odds_ratio_wide,
        "active_confidence_wide": active_confidence_wide,
        "subtypes_df": subtypes_df,
        "tfx_dict": tfx_dict,
        "status": "validation",
        **mdn_tables,
    }


# ---------------------------------------------------------------------------
# IHC correlation analysis
# ---------------------------------------------------------------------------

# Cohorts with IHC protein scoring data in reference/.
_IHC_FILES: Dict[str, str] = {
    "BLCA-RA": "reference/BLCA-RA_IHC-Scores.tsv",
    "CRPC-RA": "reference/CRPC-RA_IHC-Scores.tsv",
}

# Protein → gene name mapping for cases where they differ.
_IHC_PROTEIN_TO_GENE: Dict[str, str] = {
    "PSA": "KLK3",
}

_CRPC_RA_METS_METADATA_PATH = PROJECT_ROOT / "reference" / "CRPC-RA_METS.tsv"
_CRPC_RA_METS_EXPRESSION_PATH = PROJECT_ROOT / "labels" / "CRPC-RA-METS_Log2CPMplus1.tsv"
_CRPC_RA_METS_ANALYSIS_DIRNAME = "MetastasisComposition"
_CRPC_RA_BOOTSTRAP_RESAMPLES = 500
_CRPC_RA_PERMUTATION_RESAMPLES = 2000
_CRPC_RA_COMPOSITION_ORDER: Tuple[str, ...] = (
    "LN-only",
    "Bone-only",
    "Liver-containing",
    "Bone-containing",
    "Visceral non-liver",
    "Mixed",
    "Other",
)
_CRPC_RA_SHORTLIST_DISPLAY: Dict[str, str] = {
    "ARG.10": "AR.10",
    "NE.10": "NE.10",
    "CCP.31": "CCP.31",
}


def _ihc_scatter(
    ihc_vals: np.ndarray,
    expr_vals: np.ndarray,
    sample_names: Sequence[str],
    protein: str,
    gene: str,
    expr_label: str,
    name: str,
    *,
    subtypes_df: Optional[pd.DataFrame] = None,
    tfx_dict: Optional[Dict[str, float]] = None,
    ambiguous_mask: Optional[pd.Series] = None,
) -> None:
    """Scatter: IHC protein score vs RNA-Seq expression (truth or predicted)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, t as t_dist

    if ambiguous_mask is not None:
        mask_series = ambiguous_mask.reindex(pd.Index(sample_names)).fillna(False)
        confident = ~mask_series.to_numpy(dtype=bool)
    else:
        mask_series = pd.Series(False, index=pd.Index(sample_names))
        confident = np.ones(len(sample_names), dtype=bool)

    if int(confident.sum()) >= 3:
        rho_result = spearmanr(ihc_vals[confident], expr_vals[confident])
        rho_val = float(rho_result.statistic)
        p_val = float(rho_result.pvalue)
        fit_x = ihc_vals[confident]
        fit_y = expr_vals[confident]
    else:
        rho_val = np.nan
        p_val = np.nan
        fit_x = np.array([], dtype=np.float64)
        fit_y = np.array([], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    pal = load_palette()
    seen_subtypes: set[str] = set()
    point_x: List[float] = []
    point_y: List[float] = []
    for sample_name, ihc_val, expr_val in zip(sample_names, ihc_vals, expr_vals):
        is_ambiguous = bool(mask_series.get(sample_name, False))
        subtype = "Unknown"
        color = "#999999"
        marker = "o"
        if subtypes_df is not None and sample_name in subtypes_df.index:
            if "subtype" in subtypes_df.columns:
                subtype = str(subtypes_df.loc[sample_name, "subtype"])
            color = pal.get(subtype, "#999999")
            if "status" in subtypes_df.columns and str(subtypes_df.loc[sample_name, "status"]) == "holdout":
                marker = "D"
        seen_subtypes.add(subtype)
        size = max(30, (tfx_dict or {}).get(sample_name, 1.0) * 300)
        if is_ambiguous:
            ax.scatter(
                ihc_val,
                expr_val,
                s=size,
                facecolors="none",
                edgecolors=color,
                marker=marker,
                linewidths=2.0,
                alpha=1.0,
                zorder=4,
            )
        else:
            ax.scatter(
                ihc_val,
                expr_val,
                s=size,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.78,
                zorder=3,
            )
        point_x.append(float(ihc_val))
        point_y.append(float(expr_val))

    if len(fit_x) > 2 and float(np.ptp(fit_x)) > 0.0:
        coeffs = np.polyfit(fit_x, fit_y, 1)
        x_line = np.linspace(float(fit_x.min()), float(fit_x.max()), 100)
        fit_fn = np.poly1d(coeffs)
        ax.plot(x_line, np.polyval(coeffs, x_line), "k-", lw=1, zorder=1)

        n = len(fit_x)
        if n > 3:
            x_vals = np.asarray(fit_x, dtype=np.float64)
            y_vals = np.asarray(fit_y, dtype=np.float64)
            residuals = y_vals - fit_fn(x_vals)
            s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
            mean_x = np.mean(x_vals)
            sxx = np.sum((x_vals - mean_x) ** 2)
            if sxx > 0:
                t_val = t_dist.ppf(0.975, n - 2)
                y_line = fit_fn(x_line)
                ci = t_val * s_err * np.sqrt(1 / n + (x_line - mean_x) ** 2 / sxx)
                ax.fill_between(
                    x_line,
                    y_line - ci,
                    y_line + ci,
                    color="gray",
                    alpha=0.25,
                    zorder=0,
                )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=pal.get(subtype, "#999999"), markeredgecolor="black",
            markersize=8, label=subtype,
        )
        for subtype in sorted(seen_subtypes)
    ]
    if handles:
        ax.legend(handles=handles, title="Subtype", bbox_to_anchor=(1.02, 1),
                  loc="upper left", fontsize=8)

    ax.set_xlabel(f"IHC Score ({protein})")
    ax.set_ylabel(f"{expr_label} log\u2082(CPM+1) ({gene})")
    ax.set_title(f"{protein} IHC vs {expr_label} — {gene}")
    if np.isfinite(rho_val) and np.isfinite(p_val):
        spearman_line = f"Spearman ρ = {rho_val:.3f}  ({format_pval(p_val)})"
    else:
        spearman_line = "Spearman ρ = NA  (p=NA)"
    note_lines = [spearman_line, f"n = {int(confident.sum()) if ambiguous_mask is not None else len(ihc_vals)}"]
    if ambiguous_mask is not None and int(mask_series.sum()) > 0:
        note_lines.append(f"{int(mask_series.sum())} low-confidence excluded from fit/stats")
    ax.text(
        0.05,
        0.95,
        "\n".join(note_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    marker_note = []
    if tfx_dict is not None:
        marker_note.append("Marker area ∝ TFx")
    if ambiguous_mask is not None and int(mask_series.sum()) > 0:
        marker_note.append("Hollow = low-confidence")
    if marker_note:
        ax.text(0.95, 0.05, " ; ".join(marker_note), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
    ax.set_box_aspect(1)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _ihc_heatmap(
    rho_df: pd.DataFrame,
    name: str,
    title: str = "IHC Correlation",
) -> None:
    """Heatmap: 3 columns of Spearman rho (truth vs IHC, pred vs IHC, truth vs pred)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colors import TwoSlopeNorm

    proteins = rho_df["protein"].tolist()
    data = rho_df[["rho_truth_ihc", "rho_pred_ihc", "rho_truth_pred"]].values

    fig, ax = plt.subplots(figsize=(4, max(3, 0.4 * len(proteins) + 1)))

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = LinearSegmentedColormap.from_list(
        "proteus_black_center_diverging",
        ["#2166AC", "#000000", "#B2182B"],
        N=256,
    )
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Truth vs IHC", "Pred vs IHC", "Truth vs Pred"],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels(proteins, fontsize=9)

    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, label="Spearman \u03c1", shrink=0.8)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_ihc_analysis(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    cohort: str,
    out_dir: str,
    *,
    ci_lo_wide: Optional[pd.DataFrame] = None,
    ci_hi_wide: Optional[pd.DataFrame] = None,
    activity_odds_ratio_wide: Optional[pd.DataFrame] = None,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    active_confidence_wide: Optional[pd.DataFrame] = None,
    active_threshold: Optional[float] = None,
    subtypes_df: Optional[pd.DataFrame] = None,
    tfx_dict: Optional[Dict[str, float]] = None,
    dim_ambiguous: bool = False,
    overwrite: bool = False,
) -> None:
    """IHC correlation analysis for cohorts with IHC protein scoring data.

    For each protein with IHC data that maps to a gene in the truth/prediction
    set, produces scatter plots (IHC vs truth, IHC vs pred) and a summary
    heatmap showing three columns of Spearman rho per protein:
      - Truth RNA-Seq vs IHC score
      - Predicted expression vs IHC score
      - Truth RNA-Seq vs predicted expression
    """
    from scipy.stats import spearmanr

    ihc_rel = _IHC_FILES.get(cohort)
    if ihc_rel is None:
        return
    ihc_path = PROJECT_ROOT / ihc_rel
    if not ihc_path.exists():
        return

    ihc_out = ensure_dir(os.path.join(out_dir, "IHC"))
    print(f"    ── IHC correlation ({cohort}) ──")

    ihc_df = pd.read_csv(ihc_path, sep="\t")
    # Strip whitespace from sample names (some IHC files have trailing spaces)
    ihc_df["sample"] = ihc_df["sample"].str.strip()
    ihc_df = ihc_df.set_index("sample")

    # Parse protein columns: use _mean columns (or _final for PSA).
    proteins: List[str] = []
    ihc_col_map: Dict[str, str] = {}
    for col in ihc_df.columns:
        if col.endswith("_mean"):
            protein = col[:-5]
            proteins.append(protein)
            ihc_col_map[protein] = col
        elif col == "PSA_final":
            proteins.append("PSA")
            ihc_col_map["PSA"] = col

    # Map bare IHC sample names → prefixed truth/pred column names
    pmap = _build_prefix_map(truth_wide.columns, cohort)

    rows: List[Dict[str, Any]] = []
    for protein in proteins:
        gene = _IHC_PROTEIN_TO_GENE.get(protein, protein)
        if gene not in truth_wide.index or gene not in pred_wide.index:
            continue

        col = ihc_col_map[protein]
        ihc_series = pd.to_numeric(ihc_df[col], errors="coerce")

        # Align samples across IHC, truth, and pred
        ihc_vals, truth_vals, pred_vals = [], [], []
        aligned_samples: List[str] = []
        for sample_bare, ihc_val in ihc_series.items():
            if not np.isfinite(ihc_val):
                continue
            prefixed = pmap.get(str(sample_bare), str(sample_bare))
            if prefixed in truth_wide.columns and prefixed in pred_wide.columns:
                t_val = float(truth_wide.loc[gene, prefixed])
                p_val = float(pred_wide.loc[gene, prefixed])
                if np.isfinite(t_val) and np.isfinite(p_val):
                    ihc_vals.append(ihc_val)
                    truth_vals.append(t_val)
                    pred_vals.append(p_val)
                    aligned_samples.append(prefixed)

        if len(ihc_vals) < 3:
            continue

        ihc_arr = np.array(ihc_vals, dtype=np.float64)
        truth_arr = np.array(truth_vals, dtype=np.float64)
        pred_arr = np.array(pred_vals, dtype=np.float64)

        ambiguous_mask = None
        if dim_ambiguous:
            ambiguous_mask = _ambiguous_sample_mask_for_gene(
                activity_odds_ratio_wide,
                gene,
                aligned_samples,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
            confident = ~ambiguous_mask.to_numpy(dtype=bool)
        else:
            confident = np.ones(len(aligned_samples), dtype=bool)

        if int(confident.sum()) >= 3:
            rho_truth_ihc = float(spearmanr(truth_arr[confident], ihc_arr[confident]).statistic)
            rho_pred_ihc = float(spearmanr(pred_arr[confident], ihc_arr[confident]).statistic)
            rho_truth_pred = float(spearmanr(truth_arr[confident], pred_arr[confident]).statistic)
            n_eval = int(confident.sum())
        else:
            rho_truth_ihc = np.nan
            rho_pred_ihc = np.nan
            rho_truth_pred = np.nan
            n_eval = int(confident.sum())

        rows.append({
            "protein": protein,
            "gene": gene,
            "n": n_eval,
            "n_low_confidence": int(ambiguous_mask.sum()) if ambiguous_mask is not None else 0,
            "rho_truth_ihc": rho_truth_ihc,
            "rho_pred_ihc": rho_pred_ihc,
            "rho_truth_pred": rho_truth_pred,
        })

        # Per-protein scatter plots
        _maybe(_ihc_scatter, overwrite,
             ihc_arr, truth_arr, aligned_samples, protein, gene, "Truth",
             os.path.join(ihc_out, f"truth_vs_ihc_{protein}"),
               subtypes_df=subtypes_df, tfx_dict=tfx_dict, ambiguous_mask=ambiguous_mask)
        _maybe(_ihc_scatter, overwrite,
             ihc_arr, pred_arr, aligned_samples, protein, gene, "Predicted",
             os.path.join(ihc_out, f"pred_vs_ihc_{protein}"),
               subtypes_df=subtypes_df, tfx_dict=tfx_dict, ambiguous_mask=ambiguous_mask)

    if not rows:
        print("    [skip] No IHC-gene matches found")
        return

    rho_df = pd.DataFrame(rows)
    rho_df.to_csv(os.path.join(ihc_out, "ihc_correlations.tsv"),
                  sep="\t", index=False)
    print(f"    IHC correlations: {len(rows)} proteins")

    # Summary heatmap
    _maybe(_ihc_heatmap, overwrite, rho_df,
           os.path.join(ihc_out, "ihc_correlation_heatmap"),
           title=f"IHC Correlation — {cohort}")


def _crpc_ra_safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()


def _crpc_ra_patient_id_from_validation_sample(sample_name: Any) -> Optional[str]:
    match = re.search(r"(\d{2}-\d{3})$", str(sample_name).strip())
    return match.group(1) if match else None


def _parse_crpc_ra_met_column(column_name: Any) -> Optional[Dict[str, str]]:
    match = re.match(r"^(?P<patient_id>\d{2}-\d{3})(?P<lesion>[A-Za-z0-9]+)_(?P<tissue>.+)$", str(column_name).strip())
    if match is None:
        return None
    tissue_label = _normalize_crpc_ra_tissue_label(match.group("tissue"))
    return {
        "patient_id": match.group("patient_id"),
        "lesion_code": match.group("lesion"),
        "tissue_label": tissue_label,
        "tissue_bucket": _crpc_ra_tissue_bucket(tissue_label),
    }


def _normalize_crpc_ra_tissue_label(tissue_label: Any) -> str:
    cleaned = str(tissue_label).strip().upper().replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned


def _crpc_ra_tissue_bucket(tissue_label: Any) -> str:
    label = _normalize_crpc_ra_tissue_label(tissue_label)
    if label == "LN" or label.endswith("_LN") or label.startswith("LN_"):
        return "LN"
    if any(token in label for token in ("LIVER", "HEPATIC")):
        return "LIVER"
    if any(token in label for token in ("BONE", "RIB", "SCAPULA", "SPINAL")):
        return "BONE"
    if "ADRENAL" in label:
        return "ADRENAL"
    if any(token in label for token in ("LUNG", "PLEURAL")):
        return "LUNG_PLEURAL"
    if "PANCREAS" in label:
        return "PANCREAS"
    if "BLADDER" in label:
        return "BLADDER"
    if "PROSTATE" in label:
        return "PROSTATE"
    if "SPLEEN" in label:
        return "SPLEEN"
    if "BRAIN" in label:
        return "BRAIN"
    if any(token in label for token in (
        "OMENTUM",
        "PERITONE",
        "DIAPHRAGM",
        "MESENTERIC",
        "ABDOMINAL",
        "PELVIC",
        "FASCIA",
        "CHEST_WALL",
        "FAT",
        "MEDIASTINUM",
        "MEDIASTINAL",
        "PERIHEPATIC",
        "PERIESOPHAGEAL",
        "POSTERIOR",
        "SOFT_TISSUE",
        "SKIN",
    )):
        return "SOFT_OTHER"
    return label


def _crpc_ra_is_visceral_bucket(bucket: str) -> bool:
    return bucket in {
        "LIVER",
        "LUNG_PLEURAL",
        "ADRENAL",
        "PANCREAS",
        "BLADDER",
        "SPLEEN",
        "BRAIN",
        "PROSTATE",
    }


def _parse_crpc_ra_distribution_counts(distribution: Any) -> Dict[str, int]:
    if pd.isna(distribution):
        return {}
    counts: Dict[str, int] = {}
    raw_distribution = str(distribution).strip().strip('"')
    if not raw_distribution:
        return counts
    for piece in raw_distribution.split(","):
        part = piece.strip().strip('"')
        if not part:
            continue
        match = re.match(r"(?P<count>\d+)\s+(?P<tissue>.+)", part)
        if match is None:
            continue
        tissue_label = _normalize_crpc_ra_tissue_label(match.group("tissue"))
        counts[tissue_label] = counts.get(tissue_label, 0) + int(match.group("count"))
    return counts


def _crpc_ra_entropy_from_counts(counts: Dict[str, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    fractions = np.asarray([count / total for count in counts.values() if count > 0], dtype=np.float64)
    if len(fractions) == 0:
        return 0.0
    return float(-np.sum(fractions * np.log(fractions)))


def _crpc_ra_composition_class(counts: Dict[str, int]) -> str:
    broad_counts: Dict[str, int] = {}
    for tissue_label, count in counts.items():
        bucket = _crpc_ra_tissue_bucket(tissue_label)
        broad_counts[bucket] = broad_counts.get(bucket, 0) + int(count)
    if not broad_counts:
        return "Other"
    if set(broad_counts) == {"LN"}:
        return "LN-only"
    if set(broad_counts) == {"BONE"}:
        return "Bone-only"
    if "LIVER" in broad_counts:
        return "Liver-containing"
    if "BONE" in broad_counts:
        return "Bone-containing"
    if any(_crpc_ra_is_visceral_bucket(bucket) for bucket in broad_counts):
        return "Visceral non-liver"
    if len(broad_counts) > 1:
        return "Mixed"
    return "Other"


def _load_crpc_ra_sampling_metadata() -> Optional[pd.DataFrame]:
    if not _CRPC_RA_METS_METADATA_PATH.exists():
        return None

    metadata_df = pd.read_csv(_CRPC_RA_METS_METADATA_PATH, sep="\t")
    if "sample" not in metadata_df.columns:
        raise ValueError("CRPC-RA_METS.tsv must contain a 'sample' column")

    patient_counts: Dict[str, Dict[str, int]] = {}
    all_tissues: set[str] = set()
    for row in metadata_df.itertuples(index=False):
        patient_id = str(getattr(row, "sample")).strip()
        counts = _parse_crpc_ra_distribution_counts(getattr(row, "distribution", ""))
        patient_counts[patient_id] = counts
        all_tissues.update(counts.keys())

    sorted_tissues = sorted(all_tissues)
    records: List[Dict[str, Any]] = []
    for row in metadata_df.itertuples(index=False):
        patient_id = str(getattr(row, "sample")).strip()
        counts = patient_counts.get(patient_id, {})
        total_mets = int(getattr(row, "n_mets", sum(counts.values()) or 0))
        broad_counts: Dict[str, int] = {}
        for tissue_label, count in counts.items():
            bucket = _crpc_ra_tissue_bucket(tissue_label)
            broad_counts[bucket] = broad_counts.get(bucket, 0) + int(count)
        record: Dict[str, Any] = {
            "patient_id": patient_id,
            "distribution": getattr(row, "distribution", ""),
            "n_mets": total_mets,
            "log1p_n_mets": float(np.log1p(total_mets)),
            "n_tissue_types": int(sum(count > 0 for count in counts.values())),
            "tissue_entropy": _crpc_ra_entropy_from_counts(counts),
            "dominant_tissue_fraction": float(max(counts.values()) / total_mets) if counts and total_mets > 0 else np.nan,
            "dominant_tissue": max(counts, key=counts.get) if counts else "",
            "LN_only": bool(broad_counts) and set(broad_counts) == {"LN"},
            "bone_only": bool(broad_counts) and set(broad_counts) == {"BONE"},
            "bone_containing": "BONE" in broad_counts,
            "liver_containing": "LIVER" in broad_counts,
            "visceral_containing": any(_crpc_ra_is_visceral_bucket(bucket) for bucket in broad_counts),
            "visceral_non_liver_containing": any(
                _crpc_ra_is_visceral_bucket(bucket) and bucket != "LIVER" for bucket in broad_counts
            ),
            "mixed_tissue": int(sum(count > 0 for count in counts.values())) > 1,
            "composition_class": _crpc_ra_composition_class(counts),
        }
        for tissue_label in sorted_tissues:
            safe_token = _crpc_ra_safe_token(tissue_label)
            tissue_count = int(counts.get(tissue_label, 0))
            record[f"count_{safe_token}"] = tissue_count
            record[f"fraction_{safe_token}"] = (tissue_count / total_mets) if total_mets > 0 else np.nan
            record[f"any_{safe_token}"] = tissue_count > 0
        records.append(record)

    return pd.DataFrame.from_records(records).set_index("patient_id")


def _load_crpc_ra_met_expression() -> Optional[pd.DataFrame]:
    if not _CRPC_RA_METS_EXPRESSION_PATH.exists():
        return None
    met_expr = pd.read_csv(_CRPC_RA_METS_EXPRESSION_PATH, sep="\t", index_col=0)
    met_expr.index = pd.Index([str(gene_name) for gene_name in met_expr.index])
    met_expr.columns = pd.Index([str(column_name).strip() for column_name in met_expr.columns])
    return _augment_missing_gene_aliases(met_expr)


def _build_crpc_ra_met_info(columns: Sequence[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for column_name in columns:
        parsed = _parse_crpc_ra_met_column(column_name)
        if parsed is None:
            continue
        records.append({
            "met_sample": str(column_name),
            "patient_id": parsed["patient_id"],
            "lesion_code": parsed["lesion_code"],
            "tissue_label": parsed["tissue_label"],
            "tissue_bucket": parsed["tissue_bucket"],
        })
    if not records:
        empty_df = pd.DataFrame(columns=["patient_id", "lesion_code", "tissue_label", "tissue_bucket"])
        empty_df.index = pd.Index([], name="met_sample")
        return empty_df
    return pd.DataFrame.from_records(records).set_index("met_sample")


def _weighted_series_mean(values: pd.Series, weights: pd.Series) -> float:
    aligned_weights = pd.to_numeric(weights.reindex(values.index), errors="coerce")
    aligned_values = pd.to_numeric(values, errors="coerce")
    valid = np.isfinite(aligned_values.values) & np.isfinite(aligned_weights.values) & (aligned_weights.values > 0)
    if not bool(valid.any()):
        return np.nan
    return float(np.average(aligned_values.values[valid], weights=aligned_weights.values[valid]))


def _crpc_ra_bootstrap_correlation_ci(
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    method: str,
    n_resamples: int = _CRPC_RA_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> Tuple[float, float]:
    valid = np.isfinite(pd.to_numeric(x_values, errors="coerce").values) & np.isfinite(pd.to_numeric(y_values, errors="coerce").values)
    x = pd.to_numeric(x_values, errors="coerce").values[valid]
    y = pd.to_numeric(y_values, errors="coerce").values[valid]
    if len(x) < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    for _ in range(n_resamples):
        sample_idx = rng.integers(0, len(x), size=len(x))
        x_boot = pd.Series(x[sample_idx])
        y_boot = pd.Series(y[sample_idx])
        est = x_boot.corr(y_boot, method=method)
        if np.isfinite(est):
            estimates.append(float(est))
    if not estimates:
        return (np.nan, np.nan)
    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def _crpc_ra_group_difference_test(
    summary_df: pd.DataFrame,
    *,
    metric_col: str,
    group_col: str,
    group_label: str,
    n_bootstrap: int = _CRPC_RA_BOOTSTRAP_RESAMPLES,
    n_permutations: int = _CRPC_RA_PERMUTATION_RESAMPLES,
    seed: int = 42,
) -> Dict[str, Any]:
    test_df = summary_df[[metric_col, group_col]].copy()
    test_df[metric_col] = pd.to_numeric(test_df[metric_col], errors="coerce")
    test_df[group_col] = test_df[group_col].astype(bool)
    test_df = test_df.dropna(subset=[metric_col])
    group_true = test_df.loc[test_df[group_col], metric_col].values.astype(np.float64)
    group_false = test_df.loc[~test_df[group_col], metric_col].values.astype(np.float64)
    if len(group_true) < 2 or len(group_false) < 2:
        return {
            "comparison": group_label,
            "metric": metric_col,
            "n_group_true": int(len(group_true)),
            "n_group_false": int(len(group_false)),
            "difference_mean": np.nan,
            "permutation_p": np.nan,
            "bootstrap_ci_lo": np.nan,
            "bootstrap_ci_hi": np.nan,
        }

    observed = float(np.mean(group_true) - np.mean(group_false))
    combined = np.concatenate([group_true, group_false])
    true_count = len(group_true)
    rng = np.random.default_rng(seed)

    permutation_stats = np.empty(n_permutations, dtype=np.float64)
    for perm_idx in range(n_permutations):
        permuted = rng.permutation(combined)
        permutation_stats[perm_idx] = float(np.mean(permuted[:true_count]) - np.mean(permuted[true_count:]))
    permutation_p = float((np.sum(np.abs(permutation_stats) >= abs(observed)) + 1) / (n_permutations + 1))

    bootstrap_stats = np.empty(n_bootstrap, dtype=np.float64)
    for boot_idx in range(n_bootstrap):
        sample_true = group_true[rng.integers(0, len(group_true), size=len(group_true))]
        sample_false = group_false[rng.integers(0, len(group_false), size=len(group_false))]
        bootstrap_stats[boot_idx] = float(np.mean(sample_true) - np.mean(sample_false))

    return {
        "comparison": group_label,
        "metric": metric_col,
        "n_group_true": int(len(group_true)),
        "n_group_false": int(len(group_false)),
        "difference_mean": observed,
        "permutation_p": permutation_p,
        "bootstrap_ci_lo": float(np.percentile(bootstrap_stats, 2.5)),
        "bootstrap_ci_hi": float(np.percentile(bootstrap_stats, 97.5)),
    }


def _crpc_ra_fit_huber_regressions(
    summary_df: pd.DataFrame,
    *,
    n_bootstrap: int = _CRPC_RA_BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> pd.DataFrame:
    from sklearn.linear_model import HuberRegressor

    specs = [
        ("gvw_mae", ["log1p_n_mets"]),
        ("pearson_r", ["log1p_n_mets"]),
        ("gvw_mae", ["log1p_n_mets", "tissue_entropy", "liver_containing", "bone_containing"]),
        ("pearson_r", ["log1p_n_mets", "tissue_entropy", "liver_containing", "bone_containing"]),
    ]
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    for outcome, predictors in specs:
        model_df = summary_df[[outcome, *predictors]].copy()
        for predictor in predictors:
            if model_df[predictor].dtype == bool:
                model_df[predictor] = model_df[predictor].astype(int)
        model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
        for predictor in predictors:
            model_df[predictor] = pd.to_numeric(model_df[predictor], errors="coerce")
        model_df = model_df.dropna()
        if len(model_df) < len(predictors) + 4:
            continue

        model = HuberRegressor(max_iter=1000)
        try:
            model.fit(model_df[predictors].values, model_df[outcome].values)
        except ValueError:
            continue
        term_names = ["intercept", *predictors]
        coefficients = np.concatenate([[model.intercept_], model.coef_]).astype(np.float64)

        boot_coefs: List[np.ndarray] = []
        for _ in range(n_bootstrap):
            sample_idx = rng.integers(0, len(model_df), size=len(model_df))
            boot_df = model_df.iloc[sample_idx]
            if len(boot_df[predictors].drop_duplicates()) < max(2, len(predictors)):
                continue
            boot_model = HuberRegressor(max_iter=1000)
            try:
                boot_model.fit(boot_df[predictors].values, boot_df[outcome].values)
            except ValueError:
                continue
            boot_coefs.append(np.concatenate([[boot_model.intercept_], boot_model.coef_]).astype(np.float64))

        for term_idx, term_name in enumerate(term_names):
            if boot_coefs:
                boot_term = np.asarray([coef[term_idx] for coef in boot_coefs], dtype=np.float64)
                ci_lo = float(np.percentile(boot_term, 2.5))
                ci_hi = float(np.percentile(boot_term, 97.5))
            else:
                ci_lo = np.nan
                ci_hi = np.nan
            rows.append({
                "outcome": outcome,
                "model": " + ".join(predictors),
                "term": term_name,
                "coefficient": float(coefficients[term_idx]),
                "bootstrap_ci_lo": ci_lo,
                "bootstrap_ci_hi": ci_hi,
                "n_patients": int(len(model_df)),
            })

    return pd.DataFrame(rows)


def _build_crpc_ra_patient_gene_table(
    patient_id: str,
    sample_name: str,
    truth_series: pd.Series,
    pred_series: pd.Series,
    met_expr: pd.DataFrame,
    gene_weights: pd.Series,
) -> pd.DataFrame:
    common_genes = truth_series.index.intersection(pred_series.index).intersection(met_expr.index)
    if len(common_genes) == 0 or met_expr.shape[1] == 0:
        return pd.DataFrame()

    truth_aligned = pd.to_numeric(truth_series.loc[common_genes], errors="coerce")
    pred_aligned = pd.to_numeric(pred_series.loc[common_genes], errors="coerce")
    met_aligned = met_expr.loc[common_genes].apply(pd.to_numeric, errors="coerce")

    met_mean = met_aligned.mean(axis=1)
    met_sd = met_aligned.std(axis=1, ddof=0)
    met_min = met_aligned.min(axis=1)
    met_max = met_aligned.max(axis=1)
    met_range = met_max - met_min
    with np.errstate(divide="ignore", invalid="ignore"):
        met_cv = np.divide(met_sd, np.abs(met_mean), out=np.full_like(met_sd, np.nan, dtype=np.float64), where=np.abs(met_mean) > 1e-8)

    gene_df = pd.DataFrame({
        "patient_id": patient_id,
        "sample": sample_name,
        "gene": common_genes,
        "truth": truth_aligned.values.astype(np.float64),
        "pred": pred_aligned.values.astype(np.float64),
        "met_mean": met_mean.values.astype(np.float64),
        "met_sd": met_sd.values.astype(np.float64),
        "met_range": met_range.values.astype(np.float64),
        "met_cv": met_cv.values.astype(np.float64),
        "max_minus_mean": (met_max - met_mean).values.astype(np.float64),
        "min_minus_mean": (met_min - met_mean).values.astype(np.float64),
        "n_mets_expr": int(met_expr.shape[1]),
        "weight": pd.to_numeric(gene_weights.reindex(common_genes), errors="coerce").values.astype(np.float64),
    })
    gene_df["signed_residual"] = gene_df["pred"] - gene_df["truth"]
    gene_df["abs_error"] = np.abs(gene_df["signed_residual"])
    gene_df["squared_error"] = gene_df["signed_residual"] ** 2

    finite_mask = np.isfinite(gene_df[["truth", "pred", "met_mean", "met_sd", "met_range", "abs_error"]]).all(axis=1)
    return gene_df.loc[finite_mask].reset_index(drop=True)


def _summarize_crpc_ra_gene_table(
    gene_df: pd.DataFrame,
    gene_weights: pd.Series,
    *,
    gene_subset: Optional[set[str]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    if gene_subset is not None:
        summary_df = gene_df[gene_df["gene"].isin(gene_subset)].copy()
    else:
        summary_df = gene_df.copy()
    if summary_df.empty:
        return {}

    truth_series = pd.Series(summary_df["truth"].values, index=summary_df["gene"].values)
    pred_series = pd.Series(summary_df["pred"].values, index=summary_df["gene"].values)
    abs_error = pd.Series(summary_df["abs_error"].values, index=summary_df["gene"].values)
    signed_residual = pd.Series(summary_df["signed_residual"].values, index=summary_df["gene"].values)
    squared_error = pd.Series(summary_df["squared_error"].values, index=summary_df["gene"].values)
    met_sd = pd.Series(summary_df["met_sd"].values, index=summary_df["gene"].values)
    met_range = pd.Series(summary_df["met_range"].values, index=summary_df["gene"].values)

    return {
        f"{prefix}n_genes": int(len(summary_df)),
        f"{prefix}mae": float(abs_error.mean()),
        f"{prefix}median_absolute_error": float(abs_error.median()),
        f"{prefix}rmse": float(np.sqrt(squared_error.mean())),
        f"{prefix}gvw_mae": _weighted_series_mean(abs_error, gene_weights),
        f"{prefix}gvw_rmse": float(np.sqrt(_weighted_series_mean(squared_error, gene_weights))),
        f"{prefix}pearson_r": float(truth_series.corr(pred_series, method="pearson")),
        f"{prefix}spearman_rho": float(truth_series.corr(pred_series, method="spearman")),
        f"{prefix}median_signed_residual": float(signed_residual.median()),
        f"{prefix}gvw_signed_residual": _weighted_series_mean(signed_residual, gene_weights),
        f"{prefix}median_met_sd": float(met_sd.median()),
        f"{prefix}mean_met_sd": float(met_sd.mean()),
        f"{prefix}gvw_met_sd": _weighted_series_mean(met_sd, gene_weights),
        f"{prefix}median_met_range": float(met_range.median()),
        f"{prefix}gvw_met_range": _weighted_series_mean(met_range, gene_weights),
        f"{prefix}fraction_genes_met_range_gt1": float((met_range > 1.0).mean()),
        f"{prefix}fraction_genes_met_range_gt2": float((met_range > 2.0).mean()),
    }


def _build_crpc_ra_closest_truth_rows(
    patient_id: str,
    sample_name: str,
    pred_series: pd.Series,
    truth_series: pd.Series,
    met_expr: pd.DataFrame,
    met_info: pd.DataFrame,
) -> List[Dict[str, Any]]:
    aligned_pred = pd.to_numeric(pred_series.reindex(met_expr.index), errors="coerce")
    aligned_truth = pd.to_numeric(truth_series.reindex(met_expr.index), errors="coerce")
    rows: List[Dict[str, Any]] = []

    def _mae_against(target_series: pd.Series) -> float:
        valid = np.isfinite(aligned_pred.values) & np.isfinite(target_series.values)
        if int(valid.sum()) == 0:
            return np.nan
        return float(np.mean(np.abs(aligned_pred.values[valid] - target_series.values[valid])))

    rows.append({
        "patient_id": patient_id,
        "sample": sample_name,
        "comparison_group": "reference",
        "comparison_label": "Average met truth",
        "mae": _mae_against(aligned_truth),
    })

    individual_rows: List[Dict[str, Any]] = []
    for met_sample in met_expr.columns:
        target_series = pd.to_numeric(met_expr[met_sample], errors="coerce")
        individual_rows.append({
            "patient_id": patient_id,
            "sample": sample_name,
            "comparison_group": "individual_met",
            "comparison_label": str(met_sample),
            "mae": _mae_against(target_series),
        })
    rows.extend(individual_rows)
    valid_individual = [row for row in individual_rows if np.isfinite(row["mae"])]
    if valid_individual:
        rows.append({
            "patient_id": patient_id,
            "sample": sample_name,
            "comparison_group": "summary",
            "comparison_label": "Best individual met",
            "mae": float(min(valid_individual, key=lambda row: row["mae"])["mae"]),
        })

    tissue_rows: List[Dict[str, Any]] = []
    for tissue_bucket, tissue_samples in met_info.groupby("tissue_bucket"):
        bucket_cols = [column_name for column_name in tissue_samples.index if column_name in met_expr.columns]
        if not bucket_cols:
            continue
        tissue_series = pd.to_numeric(met_expr[bucket_cols].mean(axis=1), errors="coerce")
        tissue_rows.append({
            "patient_id": patient_id,
            "sample": sample_name,
            "comparison_group": "tissue_mean",
            "comparison_label": f"{tissue_bucket} mean",
            "mae": _mae_against(tissue_series),
        })
    rows.extend(tissue_rows)
    valid_tissue = [row for row in tissue_rows if np.isfinite(row["mae"])]
    if valid_tissue:
        rows.append({
            "patient_id": patient_id,
            "sample": sample_name,
            "comparison_group": "summary",
            "comparison_label": "Best tissue mean",
            "mae": float(min(valid_tissue, key=lambda row: row["mae"])["mae"]),
        })

    rows.append({
        "patient_id": patient_id,
        "sample": sample_name,
        "comparison_group": "summary",
        "comparison_label": "Max-expression met",
        "mae": _mae_against(pd.to_numeric(met_expr.max(axis=1), errors="coerce")),
    })
    rows.append({
        "patient_id": patient_id,
        "sample": sample_name,
        "comparison_group": "summary",
        "comparison_label": "Min-expression met",
        "mae": _mae_against(pd.to_numeric(met_expr.min(axis=1), errors="coerce")),
    })
    return rows


def _plot_crpc_ra_summary_heatmap(summary_df: pd.DataFrame, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = summary_df.copy().sort_values(["composition_class", "patient_id"])
    if plot_df.empty:
        return

    binary_cols = [
        "LN_only",
        "bone_only",
        "bone_containing",
        "liver_containing",
        "visceral_containing",
        "mixed_tissue",
    ]
    metric_cols = [
        "n_mets",
        "n_tissue_types",
        "tissue_entropy",
        "dominant_tissue_fraction",
        "gvw_mae",
        "pearson_r",
        "gvw_met_sd",
        "gvw_met_range",
    ]

    binary_frame = plot_df[binary_cols].astype(float)
    metric_frame = plot_df[metric_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    for column in metric_frame.columns:
        finite = np.isfinite(metric_frame[column].values)
        if int(finite.sum()) == 0:
            continue
        column_min = float(metric_frame.loc[finite, column].min())
        column_max = float(metric_frame.loc[finite, column].max())
        if np.isclose(column_min, column_max):
            metric_frame.loc[finite, column] = 0.5
        else:
            metric_frame.loc[finite, column] = (metric_frame.loc[finite, column] - column_min) / (column_max - column_min)

    figure_height = max(6.0, 0.28 * len(plot_df) + 2.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, figure_height),
        gridspec_kw={"width_ratios": [1.2, 1.8]},
    )

    sns.heatmap(
        binary_frame,
        ax=axes[0],
        cmap="Greys",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        yticklabels=plot_df["patient_id"],
    )
    axes[0].set_title("Tissue Composition Flags", fontsize=11)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Patient")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        metric_frame,
        ax=axes[1],
        cmap="viridis",
        cbar_kws={"label": "Scaled value"},
        linewidths=0.5,
        linecolor="white",
        yticklabels=False,
    )
    axes[1].set_title("Sampling / Accuracy / Heterogeneity", fontsize=11)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("CRPC-RA Metastasis Composition vs Label Representativeness", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_crpc_ra_metric_scatter(
    summary_df: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    y_specs: Sequence[Tuple[str, str]],
    name: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    plot_df = summary_df.copy()
    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
    palette = dict(zip(
        _CRPC_RA_COMPOSITION_ORDER,
        sns.color_palette("Set2", n_colors=len(_CRPC_RA_COMPOSITION_ORDER)),
    ))

    fig, axes = plt.subplots(1, len(y_specs), figsize=(6 * len(y_specs), 5.5), squeeze=False)
    axes_flat = axes.ravel()
    for axis, (y_col, y_label) in zip(axes_flat, y_specs):
        local_df = plot_df[["patient_id", "composition_class", x_col, y_col]].copy()
        local_df[y_col] = pd.to_numeric(local_df[y_col], errors="coerce")
        local_df = local_df.dropna(subset=[x_col, y_col])
        if local_df.empty:
            axis.axis("off")
            continue
        sns.scatterplot(
            data=local_df,
            x=x_col,
            y=y_col,
            hue="composition_class",
            hue_order=[label for label in _CRPC_RA_COMPOSITION_ORDER if label in set(local_df["composition_class"])],
            palette=palette,
            ax=axis,
            s=70,
            edgecolor="black",
            linewidth=0.5,
        )
        for row in local_df.itertuples(index=False):
            axis.text(getattr(row, x_col), getattr(row, y_col), row.patient_id, fontsize=6, alpha=0.8)
        if len(local_df) >= 3 and float(np.ptp(local_df[x_col].values)) > 0.0:
            coeffs = np.polyfit(local_df[x_col].values.astype(np.float64), local_df[y_col].values.astype(np.float64), 1)
            x_line = np.linspace(float(local_df[x_col].min()), float(local_df[x_col].max()), 100)
            axis.plot(x_line, np.polyval(coeffs, x_line), color="#444444", lw=1.2, ls="--")
            rho_result = spearmanr(local_df[x_col].values, local_df[y_col].values)
            axis.text(
                0.03,
                0.97,
                f"Spearman ρ = {float(rho_result.statistic):.3f}\n{format_pval(float(rho_result.pvalue))}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), framealpha=0.8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_crpc_ra_boxstrip(
    summary_df: pd.DataFrame,
    *,
    category_col: str,
    y_specs: Sequence[Tuple[str, str]],
    name: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = summary_df.copy()
    order = [label for label in _CRPC_RA_COMPOSITION_ORDER if label in set(plot_df[category_col])]
    if not order:
        return

    fig, axes = plt.subplots(1, len(y_specs), figsize=(5.5 * len(y_specs), 5.5), squeeze=False)
    for axis, (y_col, y_label) in zip(axes.ravel(), y_specs):
        local_df = plot_df[[category_col, y_col]].copy()
        local_df[y_col] = pd.to_numeric(local_df[y_col], errors="coerce")
        local_df = local_df.dropna()
        if local_df.empty:
            axis.axis("off")
            continue
        sns.boxplot(data=local_df, x=category_col, y=y_col, order=order, ax=axis, color="#D8D8D8", fliersize=0)
        sns.stripplot(data=local_df, x=category_col, y=y_col, order=order, ax=axis, color="#2E5EAA", size=6, jitter=0.18, alpha=0.75)
        axis.set_xlabel("")
        axis.set_ylabel(y_label)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_crpc_ra_quartile_error(quartile_df: pd.DataFrame, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if quartile_df.empty:
        return
    group_order = ["All", *[_CRPC_RA_SHORTLIST_DISPLAY.get(name, name) for name in _CRPC_RA_SHORTLIST_DISPLAY]]
    available_groups = [group_name for group_name in group_order if group_name in set(quartile_df["gene_group"])]
    if not available_groups:
        return

    fig, axes = plt.subplots(1, len(available_groups), figsize=(5.0 * len(available_groups), 5.0), squeeze=False)
    for axis, group_name in zip(axes.ravel(), available_groups):
        group_df = quartile_df[quartile_df["gene_group"] == group_name]
        if group_df.empty:
            axis.axis("off")
            continue
        for _, patient_df in group_df.groupby("patient_id"):
            axis.plot(patient_df["quartile"], patient_df["mean_abs_error"], color="#BBBBBB", alpha=0.35, lw=1.0)
        overall = group_df.groupby("quartile", as_index=False)["mean_abs_error"].mean()
        axis.plot(overall["quartile"], overall["mean_abs_error"], color="#CC3311", marker="o", lw=2.0)
        axis.set_title(group_name)
        axis.set_xlabel("Met-SD Quartile")
        axis.set_ylabel("Mean Absolute Error")
        axis.set_xticks([1, 2, 3, 4])
        axis.grid(True, alpha=0.25)
    fig.suptitle("Prediction Error vs Within-Patient Met Heterogeneity", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_crpc_ra_closest_truth_heatmap(closest_df: pd.DataFrame, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    selected_labels = [
        "Average met truth",
        "Best individual met",
        "LN mean",
        "BONE mean",
        "LIVER mean",
        "Best tissue mean",
        "Max-expression met",
        "Min-expression met",
    ]
    plot_df = closest_df[closest_df["comparison_label"].isin(selected_labels)].copy()
    if plot_df.empty:
        return
    wide = plot_df.pivot(index="patient_id", columns="comparison_label", values="mae")
    available_labels = [label for label in selected_labels if label in wide.columns]
    if not available_labels:
        return
    wide = wide[available_labels]
    rank_wide = wide.rank(axis=1, method="min", na_option="keep")
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(available_labels)), max(6, 0.28 * len(rank_wide) + 2.0)))
    sns.heatmap(rank_wide, cmap="viridis_r", linewidths=0.5, linecolor="white", cbar_kws={"label": "MAE rank (1 = closest)"}, ax=ax)
    ax.set_title("Proteus Closeness to Average vs Specific Met Truths", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Patient")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_crpc_ra_pathway_truth_scatter(pathway_df: pd.DataFrame, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    if pathway_df.empty:
        return
    pathways = [label for label in (_CRPC_RA_SHORTLIST_DISPLAY.get("ARG.10", "ARG.10"), _CRPC_RA_SHORTLIST_DISPLAY.get("NE.10", "NE.10"), _CRPC_RA_SHORTLIST_DISPLAY.get("CCP.31", "CCP.31")) if label in set(pathway_df["pathway_display"])]
    if not pathways:
        return
    palette = dict(zip(
        _CRPC_RA_COMPOSITION_ORDER,
        sns.color_palette("Set2", n_colors=len(_CRPC_RA_COMPOSITION_ORDER)),
    ))
    fig, axes = plt.subplots(1, len(pathways), figsize=(5.5 * len(pathways), 5.2), squeeze=False)
    for axis, pathway_label in zip(axes.ravel(), pathways):
        local_df = pathway_df[pathway_df["pathway_display"] == pathway_label].dropna(subset=["truth_score", "pred_score"]).copy()
        if local_df.empty:
            axis.axis("off")
            continue
        sns.scatterplot(data=local_df, x="truth_score", y="pred_score", hue="composition_class", palette=palette, ax=axis, s=70, edgecolor="black", linewidth=0.5)
        for row in local_df.itertuples(index=False):
            axis.text(row.truth_score, row.pred_score, row.patient_id, fontsize=6, alpha=0.75)
        lo = float(min(local_df["truth_score"].min(), local_df["pred_score"].min()))
        hi = float(max(local_df["truth_score"].max(), local_df["pred_score"].max()))
        axis.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="#888888")
        rho_result = spearmanr(local_df["truth_score"].values, local_df["pred_score"].values)
        axis.text(0.03, 0.97, f"Spearman ρ = {float(rho_result.statistic):.3f}\n{format_pval(float(rho_result.pvalue))}", transform=axis.transAxes, ha="left", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))
        axis.set_title(pathway_label)
        axis.set_xlabel("Average-met score")
        axis.set_ylabel("Proteus score")
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), framealpha=0.8)
    fig.suptitle("Proteus vs Average-Met Pathway Scores", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_crpc_ra_pathway_scatter(
    pathway_df: pd.DataFrame,
    *,
    x_col: str,
    x_label: str,
    name: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if pathway_df.empty:
        return
    pathways = [label for label in (_CRPC_RA_SHORTLIST_DISPLAY.get("ARG.10", "ARG.10"), _CRPC_RA_SHORTLIST_DISPLAY.get("NE.10", "NE.10"), _CRPC_RA_SHORTLIST_DISPLAY.get("CCP.31", "CCP.31")) if label in set(pathway_df["pathway_display"])]
    if not pathways:
        return
    palette = dict(zip(
        _CRPC_RA_COMPOSITION_ORDER,
        sns.color_palette("Set2", n_colors=len(_CRPC_RA_COMPOSITION_ORDER)),
    ))
    fig, axes = plt.subplots(1, len(pathways), figsize=(5.5 * len(pathways), 5.2), squeeze=False)
    for axis, pathway_label in zip(axes.ravel(), pathways):
        local_df = pathway_df[pathway_df["pathway_display"] == pathway_label].copy()
        local_df[x_col] = pd.to_numeric(local_df[x_col], errors="coerce")
        local_df = local_df.dropna(subset=[x_col, "abs_residual"])
        if local_df.empty:
            axis.axis("off")
            continue
        sns.scatterplot(data=local_df, x=x_col, y="abs_residual", hue="composition_class", palette=palette, ax=axis, s=70, edgecolor="black", linewidth=0.5)
        for row in local_df.itertuples(index=False):
            axis.text(getattr(row, x_col), row.abs_residual, row.patient_id, fontsize=6, alpha=0.75)
        axis.set_title(pathway_label)
        axis.set_xlabel(x_label)
        axis.set_ylabel("Absolute pathway residual")
        axis.grid(True, alpha=0.25)
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), framealpha=0.8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_crpc_ra_pathway_boxstrip(pathway_df: pd.DataFrame, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if pathway_df.empty:
        return
    order = [label for label in _CRPC_RA_COMPOSITION_ORDER if label in set(pathway_df["composition_class"])]
    if not order:
        return
    pathways = [label for label in (_CRPC_RA_SHORTLIST_DISPLAY.get("ARG.10", "ARG.10"), _CRPC_RA_SHORTLIST_DISPLAY.get("NE.10", "NE.10"), _CRPC_RA_SHORTLIST_DISPLAY.get("CCP.31", "CCP.31")) if label in set(pathway_df["pathway_display"])]
    if not pathways:
        return
    fig, axes = plt.subplots(2, len(pathways), figsize=(5.0 * len(pathways), 9.0), squeeze=False)
    for col_idx, pathway_label in enumerate(pathways):
        pathway_local = pathway_df[pathway_df["pathway_display"] == pathway_label]
        for row_idx, metric_col in enumerate(("abs_residual", "signed_residual")):
            axis = axes[row_idx, col_idx]
            local_df = pathway_local[["composition_class", metric_col]].copy()
            local_df[metric_col] = pd.to_numeric(local_df[metric_col], errors="coerce")
            local_df = local_df.dropna()
            if local_df.empty:
                axis.axis("off")
                continue
            sns.boxplot(data=local_df, x="composition_class", y=metric_col, order=order, ax=axis, color="#D8D8D8", fliersize=0)
            sns.stripplot(data=local_df, x="composition_class", y=metric_col, order=order, ax=axis, color="#2E5EAA", size=6, jitter=0.18, alpha=0.75)
            axis.set_title(f"{pathway_label} — {'Absolute' if metric_col == 'abs_residual' else 'Signed'} residual")
            axis.set_xlabel("")
            axis.set_ylabel("Residual")
            axis.tick_params(axis="x", rotation=45)
            axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Pathway Residuals by Tissue Composition", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_crpc_ra_metastasis_composition_analysis(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    cohort_out: str,
    *,
    subtypes_df: Optional[pd.DataFrame],
    overwrite: bool,
) -> None:
    analysis_out = ensure_dir(os.path.join(cohort_out, _CRPC_RA_METS_ANALYSIS_DIRNAME))

    truth_wide = truth_wide.loc[~truth_wide.index.duplicated(keep="first")]
    pred_wide = pred_wide.loc[~pred_wide.index.duplicated(keep="first")]

    sampling_meta = _load_crpc_ra_sampling_metadata()
    if sampling_meta is None:
        print("    [skip] CRPC-RA metastasis composition: sampling metadata not found")
        return
    met_expr = _load_crpc_ra_met_expression()
    if met_expr is None:
        print("    [skip] CRPC-RA metastasis composition: met-level truth matrix not found")
        return
    met_expr = met_expr.loc[~met_expr.index.duplicated(keep="first")]

    met_info = _build_crpc_ra_met_info(met_expr.columns)
    if met_info.empty:
        print("    [skip] CRPC-RA metastasis composition: no met-level sample columns could be parsed")
        return

    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    patient_sample_pairs: List[Tuple[str, str]] = []
    for sample_name in common_samples:
        patient_id = _crpc_ra_patient_id_from_validation_sample(sample_name)
        if patient_id is None:
            continue
        if patient_id not in sampling_meta.index:
            continue
        if patient_id not in set(met_info["patient_id"]):
            continue
        patient_sample_pairs.append((patient_id, str(sample_name)))

    if not patient_sample_pairs:
        print("    [skip] CRPC-RA metastasis composition: no evaluable patients after alignment")
        return

    shared_genes = truth_wide.index.intersection(pred_wide.index).intersection(met_expr.index)
    if len(shared_genes) < 10:
        print("    [skip] CRPC-RA metastasis composition: too few common genes")
        return

    selected_samples = [sample_name for _, sample_name in patient_sample_pairs]
    truth_aligned = truth_wide.loc[shared_genes, selected_samples]
    pred_aligned = pred_wide.loc[shared_genes, selected_samples]
    met_expr = met_expr.loc[shared_genes]
    gene_weights = pd.Series(compute_gene_variance(truth_aligned), dtype=float)
    shortlist_sets = _load_shortlist_sets()
    shortlist_gene_sets = {
        set_name: set(gene_list).intersection(shared_genes)
        for set_name, gene_list in shortlist_sets.items()
    }

    truth_shortlist_scores = _compute_ssgsea_score_matrix(truth_aligned[selected_samples], shortlist_sets, min_size=3) if shortlist_sets else pd.DataFrame()
    pred_shortlist_scores = _compute_ssgsea_score_matrix(pred_aligned[selected_samples], shortlist_sets, min_size=3) if shortlist_sets else pd.DataFrame()

    patient_rows: List[Dict[str, Any]] = []
    gene_rows_parts: List[pd.DataFrame] = []
    comparator_rows: List[Dict[str, Any]] = []
    pathway_rows: List[Dict[str, Any]] = []

    for patient_id, sample_name in patient_sample_pairs:
        patient_met_cols = [column_name for column_name in met_info.index if met_info.loc[column_name, "patient_id"] == patient_id]
        if not patient_met_cols:
            continue
        patient_met_expr = met_expr[patient_met_cols]
        gene_df = _build_crpc_ra_patient_gene_table(
            patient_id,
            sample_name,
            truth_aligned[sample_name],
            pred_aligned[sample_name],
            patient_met_expr,
            gene_weights,
        )
        if gene_df.empty:
            continue
        gene_rows_parts.append(gene_df)

        patient_row: Dict[str, Any] = {
            "patient_id": patient_id,
            "sample": sample_name,
        }
        if patient_id in sampling_meta.index:
            patient_row.update(sampling_meta.loc[patient_id].to_dict())
        patient_row["n_mets_expr"] = int(patient_met_expr.shape[1])
        if subtypes_df is not None and sample_name in subtypes_df.index:
            if "tfx" in subtypes_df.columns:
                patient_row["tfx"] = float(pd.to_numeric(pd.Series([subtypes_df.loc[sample_name, "tfx"]]), errors="coerce").iloc[0])
            if "subtype" in subtypes_df.columns:
                patient_row["subtype"] = str(subtypes_df.loc[sample_name, "subtype"])

        patient_row.update(_summarize_crpc_ra_gene_table(gene_df, gene_weights))
        for set_name, genes in shortlist_gene_sets.items():
            if not genes:
                continue
            set_prefix = f"{_crpc_ra_safe_token(set_name)}_"
            patient_row.update(_summarize_crpc_ra_gene_table(gene_df, gene_weights, gene_subset=genes, prefix=set_prefix))
        patient_rows.append(patient_row)

        patient_met_info = met_info.loc[patient_met_cols]
        comparator_rows.extend(
            _build_crpc_ra_closest_truth_rows(
                patient_id,
                sample_name,
                gene_df.set_index("gene")["pred"],
                gene_df.set_index("gene")["truth"],
                patient_met_expr,
                patient_met_info,
            )
        )

        if shortlist_sets and not truth_shortlist_scores.empty and not pred_shortlist_scores.empty and sample_name in truth_shortlist_scores.columns and sample_name in pred_shortlist_scores.columns:
            patient_met_scores = _compute_ssgsea_score_matrix(patient_met_expr, shortlist_sets, min_size=3)
            if not patient_met_scores.empty:
                for pathway_name in patient_met_scores.index.intersection(truth_shortlist_scores.index).intersection(pred_shortlist_scores.index):
                    display_name = _CRPC_RA_SHORTLIST_DISPLAY.get(pathway_name, pathway_name)
                    met_score_values = pd.to_numeric(patient_met_scores.loc[pathway_name], errors="coerce")
                    pred_score = pd.to_numeric(pd.Series([pred_shortlist_scores.loc[pathway_name, sample_name]]), errors="coerce").iloc[0]
                    truth_score = pd.to_numeric(pd.Series([truth_shortlist_scores.loc[pathway_name, sample_name]]), errors="coerce").iloc[0]
                    if not np.isfinite(pred_score) or not np.isfinite(truth_score):
                        continue
                    path_row: Dict[str, Any] = {
                        "patient_id": patient_id,
                        "sample": sample_name,
                        "pathway": pathway_name,
                        "pathway_display": display_name,
                        "pred_score": float(pred_score),
                        "truth_score": float(truth_score),
                        "signed_residual": float(pred_score - truth_score),
                        "abs_residual": float(abs(pred_score - truth_score)),
                        "met_score_mean": float(met_score_values.mean()),
                        "met_score_sd": float(met_score_values.std(ddof=0)),
                        "met_score_range": float(met_score_values.max() - met_score_values.min()),
                    }
                    if patient_id in sampling_meta.index:
                        path_row.update(sampling_meta.loc[patient_id].to_dict())
                    pathway_rows.append(path_row)

    if not patient_rows or not gene_rows_parts:
        print("    [skip] CRPC-RA metastasis composition: no evaluable patient summaries")
        return

    patient_summary = pd.DataFrame.from_records(patient_rows)
    patient_summary = patient_summary.sort_values(["composition_class", "patient_id"]).reset_index(drop=True)
    gene_rows_df = pd.concat(gene_rows_parts, ignore_index=True)
    comparator_df = pd.DataFrame.from_records(comparator_rows)
    pathway_df = pd.DataFrame.from_records(pathway_rows)

    if not pathway_df.empty:
        patient_summary = patient_summary.merge(
            pathway_df.pivot(index="patient_id", columns="pathway_display", values="abs_residual").add_prefix("pathway_abs_residual_"),
            how="left",
            left_on="patient_id",
            right_index=True,
        )

    quartile_rows: List[Dict[str, Any]] = []
    group_sets: Dict[str, Optional[set[str]]] = {"All": None}
    for set_name, genes in shortlist_gene_sets.items():
        group_sets[_CRPC_RA_SHORTLIST_DISPLAY.get(set_name, set_name)] = genes
    for group_name, genes in group_sets.items():
        group_df = gene_rows_df if genes is None else gene_rows_df[gene_rows_df["gene"].isin(genes)].copy()
        for patient_id, patient_gene_df in group_df.groupby("patient_id"):
            if len(patient_gene_df) < 4:
                continue
            ranked = patient_gene_df["met_sd"].rank(method="first")
            try:
                quartiles = pd.qcut(ranked, 4, labels=[1, 2, 3, 4])
            except ValueError:
                continue
            patient_gene_df = patient_gene_df.assign(heterogeneity_quartile=pd.to_numeric(quartiles, errors="coerce"))
            for quartile_idx in [1, 2, 3, 4]:
                quartile_df = patient_gene_df[patient_gene_df["heterogeneity_quartile"] == quartile_idx]
                if quartile_df.empty:
                    continue
                quartile_rows.append({
                    "gene_group": group_name,
                    "patient_id": patient_id,
                    "quartile": int(quartile_idx),
                    "mean_abs_error": float(quartile_df["abs_error"].mean()),
                    "mean_met_sd": float(quartile_df["met_sd"].mean()),
                    "n_genes": int(len(quartile_df)),
                })
    quartile_summary = pd.DataFrame.from_records(quartile_rows)

    correlation_rows: List[Dict[str, Any]] = []
    correlation_specs = [
        ("log1p_n_mets", "log1p(n_mets)"),
        ("tissue_entropy", "Tissue entropy"),
        ("dominant_tissue_fraction", "Dominant tissue fraction"),
        ("gvw_met_sd", "GVW met SD"),
        ("median_met_sd", "Median met SD"),
        ("gvw_met_range", "GVW met range"),
    ]
    for outcome, outcome_label in (("gvw_mae", "GVW-MAE"), ("pearson_r", "Pearson r")):
        for predictor, predictor_label in correlation_specs:
            local_df = patient_summary[[outcome, predictor]].copy()
            local_df[outcome] = pd.to_numeric(local_df[outcome], errors="coerce")
            local_df[predictor] = pd.to_numeric(local_df[predictor], errors="coerce")
            local_df = local_df.dropna()
            if len(local_df) < 4:
                continue
            from scipy.stats import spearmanr

            result = spearmanr(local_df[predictor].values, local_df[outcome].values)
            ci_lo, ci_hi = _crpc_ra_bootstrap_correlation_ci(local_df[predictor], local_df[outcome], method="spearman")
            correlation_rows.append({
                "outcome": outcome_label,
                "predictor": predictor_label,
                "spearman_rho": float(result.statistic),
                "p_value": float(result.pvalue),
                "bootstrap_ci_lo": ci_lo,
                "bootstrap_ci_hi": ci_hi,
                "n_patients": int(len(local_df)),
            })
    correlation_df = pd.DataFrame.from_records(correlation_rows)

    regression_df = _crpc_ra_fit_huber_regressions(patient_summary)

    group_test_rows: List[Dict[str, Any]] = []
    comparison_specs = [
        ("liver_containing", "Liver-containing vs non-liver"),
        ("bone_containing", "Bone-containing vs non-bone"),
        ("LN_only", "LN-only vs non-LN-only"),
        ("visceral_containing", "Visceral-containing vs non-visceral"),
    ]
    for metric_col in ("gvw_mae", "pearson_r", "median_signed_residual", "gvw_signed_residual"):
        for group_col, group_label in comparison_specs:
            group_test_rows.append(
                _crpc_ra_group_difference_test(
                    patient_summary,
                    metric_col=metric_col,
                    group_col=group_col,
                    group_label=group_label,
                )
            )
    group_test_df = pd.DataFrame.from_records(group_test_rows)

    sampling_meta.loc[patient_summary["patient_id"].tolist()].to_csv(os.path.join(analysis_out, "sampling_metadata.tsv"), sep="\t")
    met_info[met_info["patient_id"].isin(set(patient_summary["patient_id"]))].to_csv(os.path.join(analysis_out, "met_metadata.tsv"), sep="\t")
    patient_summary.to_csv(os.path.join(analysis_out, "patient_summary.tsv"), sep="\t", index=False)
    gene_rows_df.to_csv(os.path.join(analysis_out, "patient_gene_heterogeneity.tsv"), sep="\t", index=False)
    comparator_df.to_csv(os.path.join(analysis_out, "closest_truth_mae.tsv"), sep="\t", index=False)
    correlation_df.to_csv(os.path.join(analysis_out, "correlation_tests.tsv"), sep="\t", index=False)
    regression_df.to_csv(os.path.join(analysis_out, "robust_regression.tsv"), sep="\t", index=False)
    group_test_df.to_csv(os.path.join(analysis_out, "composition_group_tests.tsv"), sep="\t", index=False)
    if not quartile_summary.empty:
        quartile_summary.to_csv(os.path.join(analysis_out, "heterogeneity_quartiles.tsv"), sep="\t", index=False)
    if not pathway_df.empty:
        pathway_df.to_csv(os.path.join(analysis_out, "pathway_summary.tsv"), sep="\t", index=False)

    _maybe(
        _plot_crpc_ra_summary_heatmap,
        overwrite,
        patient_summary,
        os.path.join(analysis_out, "patient_tissue_accuracy_heatmap"),
    )
    _maybe(
        _plot_crpc_ra_metric_scatter,
        overwrite,
        patient_summary,
        x_col="log1p_n_mets",
        x_label="log(1 + n_mets)",
        y_specs=(("gvw_mae", "GVW-MAE"), ("pearson_r", "Pearson r")),
        name=os.path.join(analysis_out, "accuracy_vs_n_mets"),
        title="Patient Accuracy vs Number of Sampled Metastases",
    )
    _maybe(
        _plot_crpc_ra_metric_scatter,
        overwrite,
        patient_summary,
        x_col="tissue_entropy",
        x_label="Tissue entropy",
        y_specs=(("gvw_mae", "GVW-MAE"), ("pearson_r", "Pearson r")),
        name=os.path.join(analysis_out, "accuracy_vs_tissue_entropy"),
        title="Patient Accuracy vs Tissue-Source Heterogeneity",
    )
    _maybe(
        _plot_crpc_ra_metric_scatter,
        overwrite,
        patient_summary,
        x_col="dominant_tissue_fraction",
        x_label="Dominant tissue fraction",
        y_specs=(("gvw_mae", "GVW-MAE"), ("pearson_r", "Pearson r")),
        name=os.path.join(analysis_out, "accuracy_vs_dominant_tissue_fraction"),
        title="Patient Accuracy vs Dominant Tissue Fraction",
    )
    heterogeneity_plot_summary = patient_summary.loc[
        pd.to_numeric(patient_summary["n_mets_expr"], errors="coerce") > 1
    ].copy()
    _maybe(
        _plot_crpc_ra_metric_scatter,
        overwrite,
        heterogeneity_plot_summary,
        x_col="gvw_met_sd",
        x_label="GVW-weighted met SD",
        y_specs=(("gvw_mae", "GVW-MAE"), ("pearson_r", "Pearson r")),
        name=os.path.join(analysis_out, "accuracy_vs_met_heterogeneity"),
        title="Patient Accuracy vs Met-Level Expression Heterogeneity",
    )
    _maybe(
        _plot_crpc_ra_boxstrip,
        overwrite,
        patient_summary,
        category_col="composition_class",
        y_specs=(("gvw_mae", "GVW-MAE"), ("pearson_r", "Pearson r"), ("gvw_signed_residual", "GVW signed residual")),
        name=os.path.join(analysis_out, "accuracy_by_composition_class"),
        title="Patient Accuracy by Tissue Composition",
    )
    _maybe(
        _plot_crpc_ra_boxstrip,
        overwrite,
        patient_summary,
        category_col="composition_class",
        y_specs=(("median_signed_residual", "Median signed residual"), ("gvw_signed_residual", "GVW signed residual")),
        name=os.path.join(analysis_out, "signed_residual_by_composition_class"),
        title="Signed Residual Direction by Tissue Composition",
    )
    if not quartile_summary.empty:
        _maybe(
            _plot_crpc_ra_quartile_error,
            overwrite,
            quartile_summary,
            os.path.join(analysis_out, "absolute_error_vs_met_sd_quartiles"),
        )
    if not comparator_df.empty:
        _maybe(
            _plot_crpc_ra_closest_truth_heatmap,
            overwrite,
            comparator_df,
            os.path.join(analysis_out, "closest_truth_comparator_rank_heatmap"),
        )
    if not pathway_df.empty:
        _maybe(
            _plot_crpc_ra_pathway_truth_scatter,
            overwrite,
            pathway_df,
            os.path.join(analysis_out, "pathway_scores_vs_average_truth"),
        )
        _maybe(
            _plot_crpc_ra_pathway_scatter,
            overwrite,
            pathway_df,
            x_col="tissue_entropy",
            x_label="Tissue entropy",
            name=os.path.join(analysis_out, "pathway_residual_vs_tissue_entropy"),
            title="Pathway Residuals vs Tissue Entropy",
        )
        _maybe(
            _plot_crpc_ra_pathway_scatter,
            overwrite,
            pathway_df,
            x_col="met_score_sd",
            x_label="Met-level pathway SD",
            name=os.path.join(analysis_out, "pathway_residual_vs_met_score_sd"),
            title="Pathway Residuals vs Pathway Heterogeneity",
        )
        _maybe(
            _plot_crpc_ra_pathway_boxstrip,
            overwrite,
            pathway_df,
            os.path.join(analysis_out, "pathway_residuals_by_composition_class"),
        )

    print(
        f"    CRPC-RA metastasis composition: {len(patient_summary)} patients, "
        f"{len(gene_rows_df)} patient-gene rows"
    )


def _crosscohort_score_scatter(
    truth_scores: pd.Series,
    pred_scores: pd.Series,
    sample_meta: pd.DataFrame,
    name: str,
    *,
    title: str,
    x_label: str = "Truth ssGSEA Score",
    y_label: str = "Predicted ssGSEA Score",
) -> None:
    """Scatter pooled truth-vs-predicted score values across cohorts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    shared = truth_scores.index.intersection(pred_scores.index).intersection(sample_meta.index)
    if len(shared) < 3:
        return

    truth_vals = pd.to_numeric(truth_scores.loc[shared], errors="coerce")
    pred_vals = pd.to_numeric(pred_scores.loc[shared], errors="coerce")
    valid = np.isfinite(truth_vals.values) & np.isfinite(pred_vals.values)
    if int(valid.sum()) < 3:
        return

    shared = shared[valid]
    truth_vals = truth_vals.loc[shared]
    pred_vals = pred_vals.loc[shared]
    meta = sample_meta.loc[shared]
    palette = load_palette()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    seen_cohorts: Dict[str, str] = {}
    for sample in shared:
        cohort = str(meta.loc[sample, "cohort"]) if "cohort" in meta.columns else "Unknown"
        status = str(meta.loc[sample, "status"]) if "status" in meta.columns else "validation"
        tfx_val = pd.to_numeric(pd.Series([meta.loc[sample, "tfx"]]), errors="coerce").iloc[0] \
            if "tfx" in meta.columns else np.nan
        color = palette.get(cohort, "#999999")
        marker = "D" if status == "holdout" else "o"
        size = max(40.0, 260.0 * float(tfx_val)) if np.isfinite(tfx_val) else 60.0
        ax.scatter(
            float(truth_vals.loc[sample]),
            float(pred_vals.loc[sample]),
            s=size,
            c=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.78,
        )
        seen_cohorts[cohort] = color

    lo = float(min(truth_vals.min(), pred_vals.min()))
    hi = float(max(truth_vals.max(), pred_vals.max()))
    pad = 0.05 * max(hi - lo, 1e-6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="#BBBBBB", lw=1.2)

    can_fit = (
        len(shared) >= 3
        and float(np.ptp(truth_vals.values)) > 0.0
        and float(np.ptp(pred_vals.values)) > 0.0
    )
    if can_fit:
        coeffs = np.polyfit(truth_vals.values, pred_vals.values, 1)
        x_line = np.linspace(lo - pad, hi + pad, 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), color="#CC3311", lw=1.5, alpha=0.8)

    if can_fit:
        r_p, p_p = pearsonr(truth_vals.values, pred_vals.values)
        r_s, p_s = spearmanr(truth_vals.values, pred_vals.values)
        metric_text = (
            f"Pearson r = {r_p:.3f}  ({format_pval(p_p)})\n"
            f"Spearman ρ = {r_s:.3f}  ({format_pval(p_s)})\n"
            f"n = {len(shared)}"
        )
    else:
        metric_text = f"Pearson r = NA\nSpearman ρ = NA\nn = {len(shared)}"
    ax.text(
        0.03,
        0.97,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=cohort,
            markerfacecolor=color, markeredgecolor="black", markersize=8,
        )
        for cohort, color in sorted(seen_cohorts.items())
    ]
    if handles:
        ax.legend(handles=handles, title="Cohort", bbox_to_anchor=(1.02, 1.0),
                  loc="upper left", fontsize=8)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(0.98, 0.03, "Circle=validation, diamond=holdout; size ∝ TFx",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_box_aspect(1)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _is_ne_like_subtype(cohort: str, subtype: Any) -> bool:
    """Map cohort-specific subtype labels to the requested NE-like labels."""
    subtype_str = str(subtype).strip().upper()
    if cohort in {"CRPC-RA", "CRPC-WCDT"}:
        return ("NEPC" in subtype_str) or ("MIXED" in subtype_str)
    if cohort == "BLCA-RA":
        return subtype_str == "NE"
    return False


def _compute_score_roc_curve(
    labels: Sequence[int],
    scores: Sequence[float],
) -> Optional[Tuple[np.ndarray, np.ndarray, float, int]]:
    """Compute an ROC curve for binary labels and continuous scores."""
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(labels, dtype=np.int32)
    y_score = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(y_score)
    if int(valid.sum()) < 3:
        return None
    y_true = y_true[valid]
    y_score = y_score[valid]
    if np.unique(y_true).size < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, float(auc(fpr, tpr)), int(len(y_true))


def _concat_score_matrices(score_parts: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Concatenate score matrices after intersecting to a common set index."""
    non_empty = [part for part in score_parts if part is not None and not part.empty]
    if not non_empty:
        return None

    common_index = non_empty[0].index
    for part in non_empty[1:]:
        common_index = common_index.intersection(part.index)
    if len(common_index) == 0:
        return None

    combined = pd.concat([part.loc[common_index] for part in non_empty], axis=1)
    ordered_index = [name for name in ("ARG.10", "NE.10", "CCP.31") if name in combined.index]
    return combined.loc[ordered_index] if ordered_index else combined


def _run_collected_validation(
    cohort_data: List[Dict[str, Any]],
    out_dir: str,
    args: argparse.Namespace,
    label: str = "CrossCohort",
    *,
    cohort_order: Optional[Sequence[str]] = None,
    target_sets: Optional[Sequence[str]] = None,
    include_ne10_roc: bool = True,
    activity_threshold_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Cross-cohort validation analysis combining all cohorts with truth.

    Produces:
      - Summary metrics, gene-level and sample-level metric TSVs
      - Gene metric distribution plots (Pearson r, Spearman ρ, RMSE)
      - Gene scatter plots on a composite target gene list (union of
        Targets_CRPC, Targets_BLCA, Targets_SCLC), colored by cohort
        (palette) and sized by TFx, with circle (validation/CV) or
        diamond (holdout) markers based on each cohort entry's status
      - No per-sample scatters (redundant with per-cohort output)

    Variance is pooled across all cohorts for GVW metrics, reflecting
    the pan-cancer / cross-cohort evaluation intent of this analysis.
    """
    collected_out = ensure_dir(os.path.join(out_dir, "CrossCohort"))
    print(f"\n  ── {label} (all validation cohorts) ──")

    if cohort_order is None:
        cohort_order = VALIDATION_PATIENT_HOLDOUT_COHORTS + VALIDATION_PATIENT_CV_COHORTS
    if target_sets is None:
        target_sets = _COLLECTED_TARGET_SETS

    expected_cohorts = set(cohort_order)
    ordered_cohort_data = [
        cd for cohort in cohort_order
        for cd in cohort_data
        if cd["cohort"] == cohort
    ]
    cohort_data = [cd for cd in ordered_cohort_data if cd["cohort"] in expected_cohorts]
    missing_cohorts = [
        cohort for cohort in cohort_order
        if cohort not in {cd["cohort"] for cd in cohort_data}
    ]
    if missing_cohorts:
        print(f"    [warn] Missing cohorts in {label}: {', '.join(missing_cohorts)}")

    # Intersect genes across all cohorts
    all_genes = None
    for cd in cohort_data:
        idx = cd["truth_wide"].index.intersection(cd["pred_wide"].index)
        all_genes = idx if all_genes is None else all_genes.intersection(idx)

    if all_genes is None or len(all_genes) < 10:
        print("    [skip] Too few common genes across cohorts")
        return

    # Build combined truth and pred on the common gene index
    truth_parts, pred_parts = [], []
    ci_lo_parts, ci_hi_parts = [], []
    for cd in cohort_data:
        tw = cd["truth_wide"].loc[all_genes]
        pw = cd["pred_wide"].loc[all_genes]
        common_samples = tw.columns.intersection(pw.columns)
        if len(common_samples) > 0:
            truth_parts.append(tw[common_samples])
            pred_parts.append(pw[common_samples])
            _lo = cd.get("ci_lo")
            _hi = cd.get("ci_hi")
            if _lo is not None and _hi is not None:
                lo_genes = all_genes.intersection(_lo.index)
                hi_genes = all_genes.intersection(_hi.index)
                ci_genes = lo_genes.intersection(hi_genes)
                lo_samp = [s for s in common_samples if s in _lo.columns]
                hi_samp = [s for s in common_samples if s in _hi.columns]
                ci_samp = sorted(set(lo_samp) & set(hi_samp))
                if len(ci_genes) > 0 and len(ci_samp) > 0:
                    ci_lo_parts.append(_lo.loc[ci_genes, ci_samp])
                    ci_hi_parts.append(_hi.loc[ci_genes, ci_samp])

    if not truth_parts:
        print("    [skip] No overlapping samples")
        return

    combined_truth = pd.concat(truth_parts, axis=1)
    combined_pred = pd.concat(pred_parts, axis=1)
    combined_ci_lo: Optional[pd.DataFrame] = None
    combined_ci_hi: Optional[pd.DataFrame] = None
    combined_activity_odds_ratio: Optional[pd.DataFrame] = None
    combined_active_confidence: Optional[pd.DataFrame] = None
    if ci_lo_parts and ci_hi_parts:
        combined_ci_lo = pd.concat(ci_lo_parts, axis=1).reindex(
            index=all_genes, columns=combined_pred.columns)
        combined_ci_hi = pd.concat(ci_hi_parts, axis=1).reindex(
            index=all_genes, columns=combined_pred.columns)
    active_conf_parts: List[pd.DataFrame] = []
    for cd in cohort_data:
        active_conf = cd.get("active_confidence_wide")
        if active_conf is None:
            continue
        active_genes = all_genes.intersection(active_conf.index)
        active_samples = [sample for sample in combined_pred.columns if sample in active_conf.columns]
        if len(active_genes) > 0 and len(active_samples) > 0:
            active_conf_parts.append(active_conf.loc[active_genes, active_samples])
    if active_conf_parts:
        combined_active_confidence = pd.concat(active_conf_parts, axis=1).reindex(
            index=all_genes, columns=combined_pred.columns,
        )
    activity_odds_parts: List[pd.DataFrame] = []
    for cd in cohort_data:
        activity_odds = cd.get("activity_odds_ratio_wide")
        if activity_odds is None:
            continue
        activity_genes = all_genes.intersection(activity_odds.index)
        activity_samples = [sample for sample in combined_pred.columns if sample in activity_odds.columns]
        if len(activity_genes) > 0 and len(activity_samples) > 0:
            activity_odds_parts.append(activity_odds.loc[activity_genes, activity_samples])
    if activity_odds_parts:
        combined_activity_odds_ratio = pd.concat(activity_odds_parts, axis=1).reindex(
            index=all_genes, columns=combined_pred.columns,
        )
    combined_mdn_tables: Dict[str, Optional[pd.DataFrame]] = {}
    for key in _MDN_PARAM_KEYS:
        mdn_parts: List[pd.DataFrame] = []
        for cd in cohort_data:
            table = cd.get(key)
            if table is None:
                continue
            mdn_genes = all_genes.intersection(table.index)
            mdn_samples = [sample for sample in combined_pred.columns if sample in table.columns]
            if len(mdn_genes) > 0 and len(mdn_samples) > 0:
                mdn_parts.append(table.loc[mdn_genes, mdn_samples])
        combined_mdn_tables[key] = (
            pd.concat(mdn_parts, axis=1).reindex(index=all_genes, columns=combined_pred.columns)
            if mdn_parts else None
        )
    combined_active_threshold = (
        _DOWNSTREAM_ACTIVE_DIM_THRESHOLD
        if combined_active_confidence is not None
        else None
    )
    combined_lower_threshold = (
        float(activity_threshold_summary["lower_threshold"])
        if activity_threshold_summary is not None and combined_activity_odds_ratio is not None
        else None
    )
    combined_upper_threshold = (
        float(activity_threshold_summary["upper_threshold"])
        if activity_threshold_summary is not None and combined_activity_odds_ratio is not None
        else None
    )
    print(f"    Combined: {combined_truth.shape[0]} genes × "
          f"{combined_truth.shape[1]} samples")

    # Pooled variance across all cohorts for pan-cancer GVW weighting.
    # This gives equal influence per gene across the combined sample pool,
    # appropriate for cross-cohort performance evaluation.
    var_map = compute_gene_variance(combined_truth)
    summary = cohort_summary_table(
        combined_truth, combined_pred, label=label,
        n_workers=args.n_workers, var_map=var_map,
    )
    _attach_median_point_summary(
        summary,
        combined_truth,
        combined_mdn_tables,
        n_workers=args.n_workers,
        var_map=var_map,
    )
    gene_met: pd.DataFrame = summary["_gene_metrics"]
    sample_met: pd.DataFrame = summary["_sample_metrics"]

    gene_met.to_csv(os.path.join(collected_out, "gene_metrics.tsv"), sep="\t")
    sample_met.to_csv(os.path.join(collected_out, "sample_metrics.tsv"), sep="\t")

    vg = variance_group_metrics(combined_truth, combined_pred, var_map,
                                gene_met=gene_met)
    vg.to_csv(os.path.join(collected_out, "variance_group_metrics.tsv"),
              sep="\t", index=False)
    print(f"    Variance groups:\n{vg.to_string(index=False)}")

    # Gene metric distributions
    _maybe(metric_distribution, args.overwrite, gene_met["pearson_r"], "pearson_r",
           os.path.join(collected_out, "gene_pearson_r_dist"), xlim=(-1, 1), square=True)
    _maybe(metric_distribution, args.overwrite, gene_met["spearman_rho"], "spearman_rho",
           os.path.join(collected_out, "gene_spearman_rho_dist"), xlim=(-1, 1), square=True)
    _maybe(metric_distribution, args.overwrite, gene_met["rmse"], "gene_rmse",
           os.path.join(collected_out, "gene_rmse_dist"), square=True)

    _maybe(metric_distribution_by_variance, args.overwrite, gene_met, var_map, "pearson_r",
           os.path.join(collected_out, "gene_pearson_r_dist_by_variance"),
           xlim=(-1, 1), title=f"{label} — Pearson r by Variance Group", square=True)
    _maybe(metric_distribution_by_variance, args.overwrite, gene_met, var_map, "spearman_rho",
           os.path.join(collected_out, "gene_spearman_rho_dist_by_variance"),
           xlim=(-1, 1), title=f"{label} — Spearman ρ by Variance Group", square=True)
    _maybe(metric_distribution_by_variance, args.overwrite, gene_met, var_map, "rmse",
           os.path.join(collected_out, "gene_rmse_dist_by_variance"),
           title=f"{label} — RMSE by Variance Group", square=True)

    # ── Composite gene scatter plots (colored by cohort, sized by TFx) ──
    # Build synthetic subtypes_df: "subtype" = cohort name for palette coloring,
    # "status" = "validation" (circle) or "holdout" (diamond).
    collected_subtypes_rows: List[Dict[str, Any]] = []
    merged_tfx: Dict[str, float] = {}
    for cd in cohort_data:
        status = cd.get("status", "holdout")
        tfx_dict = cd.get("tfx_dict") or {}
        sample_subtypes = cd.get("subtypes_df")
        for sample in cd["pred_wide"].columns:
            if sample in combined_pred.columns:
                sample_status = status
                if (
                    sample_subtypes is not None
                    and "status" in sample_subtypes.columns
                    and sample in sample_subtypes.index
                ):
                    sample_status = str(sample_subtypes.loc[sample, "status"])
                collected_subtypes_rows.append({
                    "sample": sample,
                    "cohort": cd["cohort"],
                    "subtype": cd["cohort"],
                    "tfx": tfx_dict.get(sample, np.nan),
                    "status": sample_status,
                })
                if sample in tfx_dict:
                    merged_tfx[sample] = tfx_dict[sample]

    collected_subtypes = pd.DataFrame(collected_subtypes_rows).set_index("sample")
    collected_tfx = merged_tfx if merged_tfx else None

    # Composite gene list: union of per-cancer target sets (deduped).
    composite_genes: List[str] = []
    seen_genes: set = set()
    for gs_name in target_sets:
        resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
        if not gene_list_available(resolved):
            continue
        for g in get_gene_list(resolved):
            if g not in seen_genes:
                composite_genes.append(g)
                seen_genes.add(g)

    if composite_genes:
        target_dir = ensure_dir(os.path.join(collected_out, "gene_target_scatters"))
        gene_ci_lo, gene_ci_hi = _gene_level_ci_inputs(
            combined_ci_lo,
            combined_ci_hi,
            scatter_ci=args.scatter_ci,
            dim_ambiguous=args.dim_ambiguous,
        )
        batch_gene_scatters(
            combined_truth, combined_pred, composite_genes, target_dir,
            subtypes_df=collected_subtypes, tfx_dict=collected_tfx,
            ci_lo_wide=gene_ci_lo, ci_hi_wide=gene_ci_hi,
            activity_odds_ratio_wide=combined_activity_odds_ratio,
            lower_threshold=combined_lower_threshold,
            upper_threshold=combined_upper_threshold,
            active_confidence_wide=combined_active_confidence,
            active_threshold=combined_active_threshold,
            dim_threshold_genes=args.dim_ambiguous,
            n_workers=args.n_workers,
        )

    shortlist_sets = _load_shortlist_sets()
    if shortlist_sets:
        shortlist_out = ensure_dir(os.path.join(collected_out, "ShortlistScores"))
        truth_score_parts: List[pd.DataFrame] = []
        raw_score_parts: List[pd.DataFrame] = []
        adjusted_score_parts: List[pd.DataFrame] = []

        for cd in cohort_data:
            common_samples = cd["truth_wide"].columns.intersection(cd["pred_wide"].columns)
            if len(common_samples) == 0:
                continue

            truth_scores = _compute_ssgsea_score_matrix(
                cd["truth_wide"][common_samples], shortlist_sets, min_size=3)
            raw_scores = _compute_ssgsea_score_matrix(
                cd["pred_wide"][common_samples], shortlist_sets, min_size=3)
            if not truth_scores.empty:
                truth_score_parts.append(truth_scores)
            if not raw_scores.empty:
                raw_score_parts.append(raw_scores)

            has_mdn = all(cd.get(key) is not None for key in _MDN_PARAM_KEYS)
            if not has_mdn:
                print(f"    [warn] Shortlist ssGSEA adjusted scores unavailable for {cd['cohort']}")
                continue

            print(f"    Shortlist ssGSEA MC: {cd['cohort']} ({len(common_samples)} samples)")
            mc_draws = mdn_monte_carlo_sample(
                cd["p0_wide"][common_samples],
                cd["pi_gamma_wide"][common_samples],
                cd["pi_normal_wide"][common_samples],
                cd["gamma_beta_wide"][common_samples],
                cd["normal_mean_wide"][common_samples],
                cd["normal_stdev_wide"][common_samples],
                M=args.montecarlo_draws,
                seed=42,
            )
            adjusted_scores, _, _ = _aggregate_ssgsea_score_matrices(
                mc_draws, shortlist_sets, min_size=3)
            if adjusted_scores is not None and not adjusted_scores.empty:
                adjusted_score_parts.append(adjusted_scores)

        truth_scores_all = _concat_score_matrices(truth_score_parts)
        raw_scores_all = _concat_score_matrices(raw_score_parts)
        adjusted_scores_all = _concat_score_matrices(adjusted_score_parts)

        if truth_scores_all is not None:
            truth_scores_all.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_truth.tsv"), sep="\t")
        if raw_scores_all is not None:
            raw_scores_all.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_raw.tsv"), sep="\t")
        if adjusted_scores_all is not None:
            adjusted_scores_all.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_uncertainty-adjusted.tsv"), sep="\t")

        if truth_scores_all is not None and raw_scores_all is not None:
            scatter_out = ensure_dir(os.path.join(shortlist_out, "scatter_plots"))
            for set_name in truth_scores_all.index.intersection(raw_scores_all.index):
                _maybe(
                    _crosscohort_score_scatter,
                    args.overwrite,
                    truth_scores_all.loc[set_name],
                    raw_scores_all.loc[set_name],
                    collected_subtypes,
                    os.path.join(scatter_out, f"{set_name}_raw"),
                    title=f"{label} — {set_name} ssGSEA (Truth vs Predicted Raw)",
                )

        if truth_scores_all is not None and adjusted_scores_all is not None:
            scatter_out = ensure_dir(os.path.join(shortlist_out, "scatter_plots"))
            for set_name in truth_scores_all.index.intersection(adjusted_scores_all.index):
                _maybe(
                    _crosscohort_score_scatter,
                    args.overwrite,
                    truth_scores_all.loc[set_name],
                    adjusted_scores_all.loc[set_name],
                    collected_subtypes,
                    os.path.join(scatter_out, f"{set_name}_uncertainty-adjusted"),
                    title=f"{label} — {set_name} ssGSEA (Truth vs Predicted Adjusted)",
                )

        if include_ne10_roc:
            ne10_out = ensure_dir(os.path.join(shortlist_out, "NE10_ROC"))
            roc_cohort_order = [cohort for cohort in cohort_order if cohort != "SCLC"]

            def _build_ne10_curves(score_df: Optional[pd.DataFrame]) -> Tuple[Dict[str, Tuple], pd.DataFrame]:
                if score_df is None or "NE.10" not in score_df.index:
                    return {}, pd.DataFrame()
                curves: Dict[str, Tuple] = {}
                rows: List[Dict[str, Any]] = []
                for cd in cohort_data:
                    cohort = cd["cohort"]
                    if cohort == "SCLC":
                        continue
                    subtypes_df = cd.get("subtypes_df")
                    if subtypes_df is None or "subtype" not in subtypes_df.columns:
                        continue
                    shared_samples = [
                        sample for sample in score_df.columns
                        if sample in cd["pred_wide"].columns and sample in subtypes_df.index
                    ]
                    if not shared_samples:
                        continue

                    labels: List[int] = []
                    scores: List[float] = []
                    for sample in shared_samples:
                        score_val = pd.to_numeric(pd.Series([score_df.loc["NE.10", sample]]), errors="coerce").iloc[0]
                        if not np.isfinite(score_val):
                            continue
                        subtype_val = subtypes_df.loc[sample, "subtype"]
                        ne_like = int(_is_ne_like_subtype(cohort, subtype_val))
                        labels.append(ne_like)
                        scores.append(float(score_val))
                        rows.append({
                            "sample": sample,
                            "cohort": cohort,
                            "subtype": subtype_val,
                            "ne_like": ne_like,
                            "NE.10_score": float(score_val),
                        })

                    curve = _compute_score_roc_curve(labels, scores)
                    if curve is not None:
                        curves[cohort] = curve
                    else:
                        print(f"    [warn] Skipping NE.10 ROC for {cohort}: insufficient class balance")

                return curves, pd.DataFrame(rows)

            raw_curves, raw_roc_df = _build_ne10_curves(raw_scores_all)
            if not raw_roc_df.empty:
                raw_roc_df.to_csv(os.path.join(ne10_out, "ne10_roc_inputs_raw.tsv"), sep="\t", index=False)
            if raw_curves:
                _maybe(
                    roc_gene_calling,
                    args.overwrite,
                    raw_curves,
                    os.path.join(ne10_out, "ne10_roc_raw"),
                    title="NE.10 ROC — NE-like detection (Raw)",
                    model_order=roc_cohort_order,
                )

            adjusted_curves, adjusted_roc_df = _build_ne10_curves(adjusted_scores_all)
            if not adjusted_roc_df.empty:
                adjusted_roc_df.to_csv(
                    os.path.join(ne10_out, "ne10_roc_inputs_uncertainty-adjusted.tsv"),
                    sep="\t",
                    index=False,
                )
            if adjusted_curves:
                _maybe(
                    roc_gene_calling,
                    args.overwrite,
                    adjusted_curves,
                    os.path.join(ne10_out, "ne10_roc_uncertainty-adjusted"),
                    title="NE.10 ROC — NE-like detection (Adjusted)",
                    model_order=roc_cohort_order,
                )

    # Save summary
    print_summary_table([summary])
    _save_summary_tsv([summary], os.path.join(collected_out, "summary_metrics.tsv"))




# ---------------------------------------------------------------------------
# Analysis group: Discovery_Patient (predictions only — no truth)
# ---------------------------------------------------------------------------

# Discovery cohorts: predictions available but no RNA-Seq truth labels.
# Format: (cohort_predictions_dir_name, subtype_label_name_or_None).
# subtype_label_name maps to labels/{name}_TFX_Subtype.tsv for annotation.
_DISCOVERY_COHORTS: List[Tuple[str, Optional[str]]] = [
    *CONFIG_DISCOVERY_COHORTS,
]

# Models to check for each discovery cohort.
_DISCOVERY_MODELS = [
    *CONFIG_DISCOVERY_MODEL_PATHS,
]

# GMT files for GSEA
_GSEA_HALLMARK_GMT = PROJECT_ROOT / "reference" / "h.all.v2026.1.Hs.symbols.gmt"
_GSEA_SHORTLIST_GMT = PROJECT_ROOT / "reference" / "gsva_genesets_shortlist.gmt"
_GSEA_SHORTLIST_SETS = {"ARG.10", "NE.10", "CCP.31"}
_ENTREZ_SYMBOL_OVERRIDES: Dict[str, str] = {
    "25859": "PART1",
}
_PFIZER_VISIT_ORDER = ["SCN", "C3D1", "EOT"]
_PFIZER_RESPONSE_COLOR_MAP: Dict[str, str] = {
    "Progression": "#B63A3A",
    "Disease Control / Non-Progression": "#2C7A4B",
}


def _load_gmt(path: Path) -> Dict[str, List[str]]:
    """Parse a GMT file → {set_name: [gene_symbols]}."""
    genesets: Dict[str, List[str]] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            # parts[1] is description/URL, parts[2:] are genes
            genesets[name] = [g for g in parts[2:] if g]
    return genesets


def _prepare_expression_matrix_for_scores(
    expr_df: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Apply targeted gene aliasing and drop NaN-bearing rows before scoring."""
    prepared = _augment_missing_gene_aliases(expr_df)
    if prepared.empty:
        return prepared

    invalid_rows = prepared.isna().any(axis=1)
    if bool(invalid_rows.any()):
        prepared = prepared.loc[~invalid_rows]
    return prepared


def _build_entrez_to_symbol() -> Dict[str, str]:
    """Build Entrez ID → gene symbol map from MANE reference."""
    ref_path = PROJECT_ROOT / "reference" / "MANE.GRCh38.v1.3.UniversalReference.tsv"
    mapping: Dict[str, str] = {}
    with open(ref_path) as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            # Column 0: "GeneID:123", Column 3: symbol
            gene_id_raw = parts[0]  # e.g. "GeneID:354"
            symbol = parts[3] or (parts[14] if len(parts) > 14 else "")
            entrez = gene_id_raw.replace("GeneID:", "")
            if entrez and symbol:
                mapping[entrez] = symbol
    mapping.update(_ENTREZ_SYMBOL_OVERRIDES)
    return mapping


def _convert_entrez_gmt(
    genesets: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Convert Entrez-ID gene sets to gene symbols via MANE reference."""
    e2s = _build_entrez_to_symbol()
    converted: Dict[str, List[str]] = {}
    for name, entrez_ids in genesets.items():
        symbols = []
        unmapped = []
        for eid in entrez_ids:
            sym = e2s.get(eid)
            if sym:
                symbols.append(sym)
            else:
                unmapped.append(eid)
        if unmapped:
            print(f"    GSEA: {name}: {len(unmapped)} unmapped Entrez IDs: "
                  f"{unmapped[:5]}{'...' if len(unmapped) > 5 else ''}")
        converted[name] = symbols
    return converted


def _run_gsea(
    de_results: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    contrast: str = "NEPC-ARPC",
    tag: str = "",
) -> None:
    """Run GSEA (prerank) on limma DE results using gseapy.

    Runs against Hallmark pathways and the CRPC shortlist gene sets
    (ARG.10, NE.10, CCP.31).  Saves results TSVs and enrichment dot plots.

    Parameters
    ----------
    tag : str
        Suffix appended to output subdirectory names.
    """
    import gseapy as gp

    # Build ranked gene list: gene → limma t-statistic (better ranking than logFC
    # because it accounts for variance, recommended for GSEA prerank)
    if "t" in de_results.columns:
        rank_col = "t"
    else:
        rank_col = "logFC"
    rank_series = de_results[rank_col].dropna()
    rank_series = rank_series[~rank_series.index.duplicated(keep="first")]
    rank_series = rank_series.sort_values(ascending=False)

    gsea_out = ensure_dir(os.path.join(out_dir, f"GSEA{tag}"))

    # --- (1) Hallmark pathways ---
    if _GSEA_HALLMARK_GMT.exists():
        hallmark_sets = _load_gmt(_GSEA_HALLMARK_GMT)
        print(f"    GSEA Hallmark{tag}: {len(hallmark_sets)} gene sets, "
              f"{len(rank_series)} ranked genes")

        hallmark_out = ensure_dir(os.path.join(gsea_out, "Hallmark"))
        hallmark_tsv = os.path.join(hallmark_out, f"gsea_hallmark_results{tag}.tsv")

        if overwrite or not os.path.exists(hallmark_tsv):
            res_h = gp.prerank(
                rnk=rank_series,
                gene_sets=hallmark_sets,
                outdir=hallmark_out,
                min_size=10,
                max_size=500,
                permutation_num=1000,
                seed=42,
                no_plot=True,
                verbose=False,
            )
            res_h_df = res_h.res2d
            res_h_df.to_csv(hallmark_tsv, sep="\t", index=False)
            print("    Hallmark GSEA results saved")
        else:
            res_h_df = pd.read_csv(hallmark_tsv, sep="\t")

        # Enrichment dot plot
        _maybe(enrichment_dot_plot, overwrite, res_h_df,
               os.path.join(hallmark_out, "enrichment_dotplot_hallmark"),
               contrast_name=contrast, title="GSEA Hallmark Pathways")

        sig_h = res_h_df[pd.to_numeric(res_h_df.get("FDR q-val", res_h_df.get("fdr", pd.Series())),
                                       errors="coerce") < 0.25]
        print(f"    Hallmark: {len(sig_h)} sets at FDR<0.25")
    else:
        print(f"    [skip] Hallmark GMT not found: {_GSEA_HALLMARK_GMT}")

    # --- (2) CRPC shortlist (Entrez → Symbol) ---
    if _GSEA_SHORTLIST_GMT.exists():
        raw_sets = _load_gmt(_GSEA_SHORTLIST_GMT)
        # Filter to requested sets
        raw_sets = {k: v for k, v in raw_sets.items() if k in _GSEA_SHORTLIST_SETS}
        if raw_sets:
            converted_sets = _convert_entrez_gmt(raw_sets)
            for name, genes in converted_sets.items():
                print(f"    GSEA shortlist: {name} → {len(genes)} genes "
                      f"(e.g. {', '.join(genes[:4])})")

            shortlist_out = ensure_dir(os.path.join(gsea_out, "Shortlist"))
            shortlist_tsv = os.path.join(shortlist_out, f"gsea_shortlist_results{tag}.tsv")

            if overwrite or not os.path.exists(shortlist_tsv):
                res_s = gp.prerank(
                    rnk=rank_series,
                    gene_sets=converted_sets,
                    outdir=shortlist_out,
                    min_size=3,
                    max_size=500,
                    permutation_num=1000,
                    seed=42,
                    no_plot=True,
                    verbose=False,
                )
                res_s_df = res_s.res2d
                res_s_df.to_csv(shortlist_tsv, sep="\t", index=False)
                print("    Shortlist GSEA results saved")
            else:
                res_s_df = pd.read_csv(shortlist_tsv, sep="\t")

            # Enrichment dot plot
            _maybe(enrichment_dot_plot, overwrite, res_s_df,
                   os.path.join(shortlist_out, "enrichment_dotplot_shortlist"),
                   contrast_name=contrast,
                   title="GSEA: ARG.10 / NE.10 / CCP.31")

            # Print per-set summary
            for _, row in res_s_df.iterrows():
                fdr_col = "FDR q-val" if "FDR q-val" in res_s_df.columns else "fdr"
                nes_col = "NES" if "NES" in res_s_df.columns else "nes"
                term_col = "Term" if "Term" in res_s_df.columns else "Name"
                print(f"      {row.get(term_col, '?')}: "
                      f"NES={float(row.get(nes_col, float('nan'))):.3f}, "
                      f"FDR={float(row.get(fdr_col, float('nan'))):.3g}")
    else:
        print(f"    [skip] Shortlist GMT not found: {_GSEA_SHORTLIST_GMT}")


def _run_limma_de(
    pred_wide: pd.DataFrame,
    subtypes_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    contrast: str = "NEPC-ARPC",
    padj_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    tag: str = "",
    tfx_series: Optional[pd.Series] = None,
) -> Optional[pd.DataFrame]:
    """Run limma DE on one expression matrix via rpy2."""
    prepared = _prepare_limma_inputs(
        pred_wide,
        subtypes_df,
        contrast,
        tfx_series=tfx_series,
    )
    if prepared is None:
        return None

    expr = prepared["expr"]
    groups = prepared["groups"]
    groups_in_contrast = prepared["groups_in_contrast"]
    counts = {group_name: int((groups == group_name).sum()) for group_name in groups_in_contrast}
    print(
        f"    limma-trend DE: {' vs '.join(f'{counts[g]} {g}' for g in groups_in_contrast)} "
        f"({expr.shape[0]} genes)"
    )

    de_results = _execute_limma_de(
        expr,
        groups,
        contrast,
        tfx_values=prepared["tfx_values"],
    )

    contrast_tag = contrast.replace("-", "_vs_")
    de_path = os.path.join(out_dir, f"limma_DE_{contrast_tag}{tag}.tsv")
    de_results.to_csv(de_path, sep="\t")
    print("    DE results saved")

    _maybe(
        volcano_plot,
        overwrite,
        de_results,
        os.path.join(out_dir, f"volcano_{contrast_tag}{tag}"),
        padj_thresh=padj_thresh,
        lfc_thresh=lfc_thresh,
        contrast_name=contrast,
    )

    sig = de_results[
        (de_results["adj.P.Val"] < padj_thresh)
        & (de_results["logFC"].abs() > lfc_thresh)
    ]
    n_up = int((sig["logFC"] > 0).sum())
    n_down = int((sig["logFC"] < 0).sum())
    print(
        f"    DE summary: {len(sig)} significant genes "
        f"({n_up} {groups_in_contrast[0]}-up, {n_down} {groups_in_contrast[1]}-up) "
        f"at padj<{padj_thresh}, |logFC|>{lfc_thresh}"
    )

    return de_results


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg adjusted p-values for a numeric Series."""
    values = pd.to_numeric(p_values, errors="coerce")
    adjusted = np.full(len(values), np.nan, dtype=np.float64)
    valid_mask = np.isfinite(values.values)
    if not bool(valid_mask.any()):
        return pd.Series(adjusted, index=p_values.index, dtype=float)

    valid_values = values.values[valid_mask]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    n_tests = len(ranked)
    bh_ranked = np.minimum.accumulate(
        ((ranked * n_tests) / np.arange(1, n_tests + 1, dtype=np.float64))[::-1]
    )[::-1]
    bh_ranked = np.clip(bh_ranked, 0.0, 1.0)

    valid_adjusted = np.empty_like(valid_values)
    valid_adjusted[order] = bh_ranked
    adjusted[valid_mask] = valid_adjusted
    return pd.Series(adjusted, index=p_values.index, dtype=float)


def _prepare_limma_inputs(
    pred_wide: pd.DataFrame,
    subtypes_df: pd.DataFrame,
    contrast: str,
    *,
    tfx_series: Optional[pd.Series] = None,
) -> Optional[Dict[str, Any]]:
    """Filter and align limma inputs once for both point and MC analyses."""
    groups_in_contrast = contrast.replace(" ", "").split("-")
    if len(groups_in_contrast) != 2:
        raise ValueError(f"Contrast must be 'GroupA-GroupB', got: {contrast}")
    required_groups = set(groups_in_contrast)

    valid = subtypes_df[subtypes_df["subtype"].isin(required_groups)]
    common = [sample for sample in pred_wide.columns if sample in valid.index]

    aligned_tfx: Optional[pd.Series] = None
    if tfx_series is not None:
        aligned_tfx = pd.to_numeric(tfx_series.reindex(common), errors="coerce")
        missing_tfx_samples = aligned_tfx.index[aligned_tfx.isna()].tolist()
        if missing_tfx_samples:
            print(f"    limma DE: dropped {len(missing_tfx_samples)} samples with missing TFx")
            missing_tfx = set(missing_tfx_samples)
            common = [sample for sample in common if sample not in missing_tfx]
            aligned_tfx = aligned_tfx.reindex(common)

    if len(common) < 3:
        print(f"    [skip] limma DE: only {len(common)} samples in {required_groups}")
        return None

    expr = pred_wide[common]
    n_total = expr.shape[0]

    na_mask = expr.isna().any(axis=1)
    n_na = int(na_mask.sum())
    expr = expr.loc[~na_mask]

    expressed_mask = (expr > 1.0).sum(axis=1) >= 3
    n_low = int((~expressed_mask).sum())
    expr = expr.loc[expressed_mask]

    print(
        f"    Gene filter: {n_total} total → dropped {n_na} NA, "
        f"{n_low} low-expression → {expr.shape[0]} genes retained"
    )

    if expr.shape[0] == 0:
        print("    [skip] limma DE: no genes passed filtering")
        return None

    groups = valid.loc[common, "subtype"].values
    counts = {group_name: int((groups == group_name).sum()) for group_name in groups_in_contrast}
    if any(count == 0 for count in counts.values()):
        print(f"    [skip] limma DE: missing one contrast group after filtering: {counts}")
        return None

    return {
        "expr": expr,
        "common_samples": common,
        "groups": groups,
        "groups_in_contrast": groups_in_contrast,
        "tfx_values": None if aligned_tfx is None else aligned_tfx.astype(np.float64),
    }


def _load_limma_bridge() -> Tuple[Any, Any, Any]:
    """Load the rpy2 bridge objects and source the limma helper script."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    r_script = str(PROJECT_ROOT / "analysis" / "limma_de.R")
    ro.r.source(r_script)
    return ro, pandas2ri, localconverter


def _execute_limma_de(
    expr: pd.DataFrame,
    groups: Sequence[str],
    contrast: str,
    *,
    tfx_values: Optional[pd.Series] = None,
    bridge: Optional[Tuple[Any, Any, Any]] = None,
) -> pd.DataFrame:
    """Execute one limma differential-expression fit through the R bridge."""
    ro, pandas2ri, localconverter = _load_limma_bridge() if bridge is None else bridge
    converter = ro.default_converter + pandas2ri.converter
    with localconverter(converter):
        r_expr = ro.conversion.get_conversion().py2rpy(expr)
        r_groups = ro.StrVector(list(groups))
        r_tfx = ro.NULL
        if tfx_values is not None:
            r_tfx = ro.FloatVector(pd.to_numeric(tfx_values, errors="coerce").values.tolist())
        de_r = ro.globalenv["run_limma_de"](r_expr, r_groups, contrast, r_tfx)
        de_results = ro.conversion.get_conversion().rpy2py(de_r)
    return pd.DataFrame(de_results)


def _aggregate_limma_draw_results(draw_results: List[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate limma results across Monte Carlo draws with Rubin pooling."""
    from scipy.stats import norm, t as student_t

    if not draw_results:
        raise ValueError("draw_results must contain at least one limma result table")

    template = draw_results[0].copy()
    aggregated = pd.DataFrame(index=template.index)

    logfc_stack = np.stack(
        [
            pd.to_numeric(result["logFC"], errors="coerce").reindex(template.index).values.astype(np.float64)
            for result in draw_results
        ],
        axis=0,
    )
    se_stack = np.stack(
        [
            pd.to_numeric(result["standard_error"], errors="coerce").reindex(template.index).values.astype(np.float64)
            for result in draw_results
        ],
        axis=0,
    )
    within_var_stack = se_stack ** 2
    n_draws = logfc_stack.shape[0]

    pooled_logfc = np.nanmean(logfc_stack, axis=0)
    mean_within_var = np.nanmean(within_var_stack, axis=0)
    if n_draws > 1:
        between_var = np.nanvar(logfc_stack, axis=0, ddof=1)
    else:
        between_var = np.zeros(logfc_stack.shape[1], dtype=np.float64)

    total_var = mean_within_var + (1.0 + 1.0 / n_draws) * between_var
    pooled_se = np.sqrt(np.maximum(total_var, 0.0))
    pooled_t = np.divide(
        pooled_logfc,
        pooled_se,
        out=np.zeros_like(pooled_logfc),
        where=pooled_se > 0.0,
    )

    df_total = np.full_like(pooled_logfc, np.inf, dtype=np.float64)
    between_term = (1.0 + 1.0 / n_draws) * between_var
    valid_between = between_term > 0.0
    df_total[valid_between] = (n_draws - 1.0) * (1.0 + (mean_within_var[valid_between] / between_term[valid_between])) ** 2

    pooled_p = np.empty_like(pooled_logfc)
    finite_df = np.isfinite(df_total)
    pooled_p[finite_df] = 2.0 * student_t.sf(np.abs(pooled_t[finite_df]), df_total[finite_df])
    pooled_p[~finite_df] = 2.0 * norm.sf(np.abs(pooled_t[~finite_df]))

    crit = np.full_like(pooled_logfc, norm.ppf(0.975), dtype=np.float64)
    crit[finite_df] = student_t.ppf(0.975, df_total[finite_df])
    aggregated["logFC"] = pooled_logfc
    aggregated["standard_error"] = pooled_se
    aggregated["df_total"] = df_total
    aggregated["t"] = pooled_t
    aggregated["P.Value"] = pooled_p
    aggregated["adj.P.Val"] = _benjamini_hochberg(pd.Series(pooled_p, index=template.index)).values
    aggregated["logFC_ci_lo"] = pooled_logfc - crit * pooled_se
    aggregated["logFC_ci_hi"] = pooled_logfc + crit * pooled_se

    for column in template.columns:
        if column in {"logFC", "standard_error", "df_total", "t", "P.Value", "adj.P.Val"}:
            continue
        numeric_stack = np.stack(
            [
                pd.to_numeric(result[column], errors="coerce").reindex(template.index).values.astype(np.float64)
                for result in draw_results
            ],
            axis=0,
        )
        aggregated[column] = np.nanmean(numeric_stack, axis=0)

    return aggregated


def _run_limma_de_mc(
    draws: List[pd.DataFrame],
    reference_pred_wide: pd.DataFrame,
    subtypes_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    contrast: str = "NEPC-ARPC",
    padj_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    tag: str = "_uncertainty-adjusted",
    tfx_series: Optional[pd.Series] = None,
) -> Optional[pd.DataFrame]:
    """Run limma DE on MC draws and aggregate gene-level summaries."""
    if not draws:
        print("    [skip] limma DE MC: no Monte Carlo draws available")
        return None

    prepared = _prepare_limma_inputs(
        reference_pred_wide,
        subtypes_df,
        contrast,
        tfx_series=tfx_series,
    )
    if prepared is None:
        return None

    expr = prepared["expr"]
    common_samples = prepared["common_samples"]
    groups = prepared["groups"]
    groups_in_contrast = prepared["groups_in_contrast"]
    print(
        f"    limma-trend DE MC: {' vs '.join(f'{int((groups == g).sum())} {g}' for g in groups_in_contrast)} "
        f"({expr.shape[0]} genes, {len(draws)} draws)"
    )

    bridge = _load_limma_bridge()
    draw_results: List[pd.DataFrame] = []
    for draw_idx, draw in enumerate(draws, start=1):
        draw_expr = draw.reindex(index=expr.index, columns=common_samples)
        if draw_expr.isna().any().any():
            raise ValueError("Monte Carlo limma draw contains missing values after alignment")
        draw_result = _execute_limma_de(
            draw_expr,
            groups,
            contrast,
            tfx_values=prepared["tfx_values"],
            bridge=bridge,
        )
        draw_results.append(draw_result.reindex(expr.index))
        if draw_idx % 10 == 0 or draw_idx == len(draws):
            print(f"      limma MC draw {draw_idx}/{len(draws)} complete")

    de_results = _aggregate_limma_draw_results(draw_results)

    contrast_tag = contrast.replace("-", "_vs_")
    de_path = os.path.join(out_dir, f"limma_DE_{contrast_tag}{tag}.tsv")
    de_results.to_csv(de_path, sep="\t")
    print("    MC DE results saved")

    _maybe(
        volcano_plot,
        overwrite,
        de_results,
        os.path.join(out_dir, f"volcano_{contrast_tag}{tag}"),
        padj_thresh=padj_thresh,
        lfc_thresh=lfc_thresh,
        contrast_name=contrast,
    )

    sig = de_results[
        (de_results["adj.P.Val"] < padj_thresh)
        & (de_results["logFC"].abs() > lfc_thresh)
    ]
    n_up = int((sig["logFC"] > 0).sum())
    n_down = int((sig["logFC"] < 0).sum())
    print(
        f"    MC DE summary: {len(sig)} significant genes "
        f"({n_up} {groups_in_contrast[0]}-up, {n_down} {groups_in_contrast[1]}-up) "
        f"at padj<{padj_thresh}, |logFC|>{lfc_thresh}"
    )
    return de_results


# ---------------------------------------------------------------------------
# Discovery_Patient: PAM50 and BC-Ghana receptor expression
# ---------------------------------------------------------------------------

_BC_GHANA_RECEPTOR_PAIRS = [
    ("HER2_status", "ERBB2", "HER2"),
    ("ER_status",   "ESR1",  "ER"),
    ("PR_stats",    "PGR",   "PR"),
]


def _normalize_pam50_reference_label(label: Any) -> Optional[str]:
    """Normalize external PAM50-like labels to the genefu display convention."""
    if pd.isna(label):
        return None
    label_str = str(label)
    if label_str == "Her2":
        return "HER2"
    return label_str


def _select_pam50_result(
    pam50_results: Dict[str, pd.DataFrame],
) -> Optional[Tuple[str, pd.DataFrame]]:
    """Select the single active genefu PAM50 result table."""
    for variant_name, variant_df in pam50_results.items():
        if variant_df is not None and not variant_df.empty and "sample" in variant_df.columns:
            return variant_name, variant_df
    return None


def _pam50_primary_score_columns(score_columns: Sequence[str]) -> List[str]:
    """Choose the primary continuous PAM50 score family for downstream comparison."""
    corr_cols = sorted(column for column in score_columns if column.startswith("corr_"))
    if corr_cols:
        return corr_cols
    return sorted(column for column in score_columns if column.startswith("proba_"))


def _plot_pam50_score_scatter(
    truth_scores: pd.DataFrame,
    pred_scores: pd.DataFrame,
    out_path: str,
    *,
    title: str,
) -> None:
    """Plot truth-vs-predicted PAM50 score agreement for each subtype score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score_columns = [column for column in truth_scores.columns if column in pred_scores.columns]
    if not score_columns:
        return

    ncols = min(3, len(score_columns))
    nrows = int(np.ceil(len(score_columns) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False)

    for axis, column in zip(axes.ravel(), score_columns):
        truth_values = pd.to_numeric(truth_scores[column], errors="coerce")
        pred_values = pd.to_numeric(pred_scores[column], errors="coerce")
        valid = truth_values.notna() & pred_values.notna()
        subtype_name = column.split("_", 1)[1] if "_" in column else column

        if bool(valid.any()):
            axis.scatter(
                truth_values.loc[valid],
                pred_values.loc[valid],
                s=32,
                alpha=0.8,
                color="#3A6EA5",
                edgecolors="white",
                linewidths=0.4,
            )
            if int(valid.sum()) >= 2:
                xy_min = float(min(truth_values.loc[valid].min(), pred_values.loc[valid].min()))
                xy_max = float(max(truth_values.loc[valid].max(), pred_values.loc[valid].max()))
                if not np.isclose(xy_min, xy_max):
                    axis.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--", color="#7A7A7A", linewidth=1.0)
                    axis.set_xlim(xy_min, xy_max)
                    axis.set_ylim(xy_min, xy_max)

        pearson_r = np.nan
        spearman_rho = np.nan
        if int(valid.sum()) >= 3:
            pearson_r = float(truth_values.loc[valid].corr(pred_values.loc[valid], method="pearson"))
            spearman_rho = float(truth_values.loc[valid].corr(pred_values.loc[valid], method="spearman"))

        axis.set_title(subtype_name)
        axis.set_xlabel("Truth PAM50 score")
        axis.set_ylabel("Predicted PAM50 score")
        axis.text(
            0.03,
            0.97,
            f"n = {int(valid.sum())}\nPearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_rho:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
        )

    for axis in axes.ravel()[len(score_columns):]:
        axis.set_visible(False)

    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(f"{out_path}.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_pam50_analysis(
    pred_wide: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    *,
    cohort_label: str,
    tag: str = "",
    truth_wide: Optional[pd.DataFrame] = None,
    mc_draws: Optional[List[pd.DataFrame]] = None,
    reference_subtypes: Optional[pd.Series] = None,
    reference_label: Optional[str] = None,
) -> None:
    """Run PAM50 classification and score comparisons for a cohort."""
    pred_for_scores = _prepare_expression_matrix_for_scores(
        pred_wide,
        context=f"PAM50 predicted {cohort_label}",
    )
    if pred_for_scores.empty:
        print(f"    [skip] PAM50 {cohort_label}: no scored genes available")
        return

    if mc_draws is not None:
        draws_for_scores = [
            _prepare_expression_matrix_for_scores(
                draw,
                context=f"PAM50 MC draw {cohort_label}",
            )
            for draw in mc_draws
        ]
        pam50_results = pam50_classify_mc_genefu(draws_for_scores, pred_for_scores)
    else:
        pam50_results = pam50_classify_genefu(pred_for_scores)

    selected_pred = _select_pam50_result(pam50_results)
    if selected_pred is None:
        print(f"    [skip] PAM50 {cohort_label}: genefu returned no classifications")
        return

    variant_name, pred_classified = selected_pred
    variant_safe = _analysis_safe_name(variant_name)
    variant_out = ensure_dir(os.path.join(out_dir, f"PAM50{tag}", variant_safe))

    pred_indexed = pred_classified.drop_duplicates(subset="sample").set_index("sample")
    pred_indexed = pred_indexed.reindex(pred_for_scores.columns)
    pred_indexed.index.name = "sample"
    pred_indexed.reset_index().to_csv(
        os.path.join(variant_out, "pam50_classification_predicted.tsv"),
        sep="\t",
        index=False,
    )

    pred_score_columns = _pam50_primary_score_columns(pred_indexed.columns)
    if pred_score_columns:
        pred_indexed[pred_score_columns].to_csv(
            os.path.join(variant_out, "pam50_scores_predicted.tsv"),
            sep="\t",
        )

    print(
        f"    PAM50 {variant_name}: saved predicted classifications for {int(pred_indexed['predicted_subtype'].notna().sum())} samples"
    )

    if truth_wide is not None:
        truth_for_scores = _prepare_expression_matrix_for_scores(
            truth_wide,
            context=f"PAM50 truth {cohort_label}",
        )
        if truth_for_scores.empty:
            print(f"    [skip] PAM50 {cohort_label}: truth matrix has no scored genes")
            return

        selected_truth = _select_pam50_result(pam50_classify_genefu(truth_for_scores))
        if selected_truth is None:
            print(f"    [skip] PAM50 {cohort_label}: truth genefu classification failed")
            return

        _, truth_classified = selected_truth
        truth_indexed = truth_classified.drop_duplicates(subset="sample").set_index("sample")
        matched_samples = [sample for sample in truth_indexed.index if sample in pred_indexed.index]
        if not matched_samples:
            print(f"    [skip] PAM50 {cohort_label}: no overlapping truth/predicted classifications")
            return

        truth_common = truth_indexed.reindex(matched_samples)
        pred_common = pred_indexed.reindex(matched_samples)
        truth_common.index.name = "sample"
        truth_common.reset_index().to_csv(
            os.path.join(variant_out, "pam50_classification_truth.tsv"),
            sep="\t",
            index=False,
        )

        shared_score_columns = _pam50_primary_score_columns(
            [column for column in truth_common.columns if column in pred_common.columns]
        )
        if shared_score_columns:
            truth_scores = truth_common[shared_score_columns].apply(pd.to_numeric, errors="coerce")
            pred_scores = pred_common[shared_score_columns].apply(pd.to_numeric, errors="coerce")
            truth_scores.to_csv(os.path.join(variant_out, "pam50_scores_truth.tsv"), sep="\t")

            score_rows: List[Dict[str, Any]] = []
            flat_truth: List[np.ndarray] = []
            flat_pred: List[np.ndarray] = []
            score_type = shared_score_columns[0].split("_", 1)[0]
            for column in shared_score_columns:
                truth_values = truth_scores[column]
                pred_values = pred_scores[column]
                valid = truth_values.notna() & pred_values.notna()
                row: Dict[str, Any] = {
                    "score_type": score_type,
                    "score_column": column,
                    "subtype": column.split("_", 1)[1] if "_" in column else column,
                    "n_samples": int(valid.sum()),
                    "pearson_r": np.nan,
                    "spearman_rho": np.nan,
                    "mae": np.nan,
                }
                if int(valid.sum()) >= 3:
                    truth_valid = truth_values.loc[valid]
                    pred_valid = pred_values.loc[valid]
                    row["pearson_r"] = float(truth_valid.corr(pred_valid, method="pearson"))
                    row["spearman_rho"] = float(truth_valid.corr(pred_valid, method="spearman"))
                    row["mae"] = float((truth_valid - pred_valid).abs().mean())
                    flat_truth.append(truth_valid.to_numpy(dtype=float))
                    flat_pred.append(pred_valid.to_numpy(dtype=float))
                score_rows.append(row)

            if flat_truth and flat_pred:
                flat_truth_values = np.concatenate(flat_truth)
                flat_pred_values = np.concatenate(flat_pred)
                if flat_truth_values.size >= 3 and flat_pred_values.size >= 3:
                    score_rows.append({
                        "score_type": score_type,
                        "score_column": "all_scores",
                        "subtype": "All",
                        "n_samples": int(flat_truth_values.size),
                        "pearson_r": float(pd.Series(flat_truth_values).corr(pd.Series(flat_pred_values), method="pearson")),
                        "spearman_rho": float(pd.Series(flat_truth_values).corr(pd.Series(flat_pred_values), method="spearman")),
                        "mae": float(np.mean(np.abs(flat_truth_values - flat_pred_values))),
                    })

            pd.DataFrame(score_rows).to_csv(
                os.path.join(variant_out, "pam50_score_correlations.tsv"),
                sep="\t",
                index=False,
            )
            _maybe(
                _plot_pam50_score_scatter,
                overwrite,
                truth_scores,
                pred_scores,
                os.path.join(variant_out, "pam50_score_scatter_truth_vs_pred"),
                title=f"PAM50 truth vs predicted scores — {cohort_label} ({variant_name})",
            )

        comparison_df = pd.DataFrame({
            "sample": matched_samples,
            "truth_subtype": truth_common["predicted_subtype"].values,
            "predicted_subtype": pred_common["predicted_subtype"].values,
        }).dropna()
        if comparison_df.empty:
            print(f"    [skip] PAM50 {cohort_label}: no truth/pred subtype pairs to compare")
            return

        comparison_df["match"] = comparison_df["truth_subtype"] == comparison_df["predicted_subtype"]
        comparison_df.to_csv(
            os.path.join(variant_out, "pam50_classification_truth_vs_pred.tsv"),
            sep="\t",
            index=False,
        )
        pd.DataFrame([
            {
                "comparison": "truth_vs_predicted",
                "n_samples": int(len(comparison_df)),
                "accuracy": float(comparison_df["match"].mean()),
            }
        ]).to_csv(
            os.path.join(variant_out, "pam50_classification_summary.tsv"),
            sep="\t",
            index=False,
        )

        class_order = sorted(set(comparison_df["truth_subtype"]) | set(comparison_df["predicted_subtype"]))
        _maybe(
            confusion_matrix_heatmap,
            overwrite,
            comparison_df["truth_subtype"].values,
            comparison_df["predicted_subtype"].values,
            os.path.join(variant_out, "pam50_confusion_matrix_truth_vs_pred"),
            class_order=class_order,
            title=f"PAM50 truth vs predicted — {cohort_label} ({variant_name})",
        )
        print(
            f"    PAM50 {cohort_label}: {int(comparison_df['match'].sum())}/{len(comparison_df)} truth-vs-pred subtype matches"
        )
        return

    if reference_subtypes is None:
        return

    reference_name = reference_label or "Reference subtype"
    reference_series = reference_subtypes.map(_normalize_pam50_reference_label)
    matched_reference_samples = [sample for sample in pred_indexed.index if sample in reference_series.index]
    if not matched_reference_samples:
        print(f"    [skip] PAM50 {cohort_label}: no {reference_name.lower()} labels overlap predictions")
        return

    reference_common = reference_series.reindex(matched_reference_samples)
    pred_reference = pred_indexed.reindex(matched_reference_samples)
    comparison_df = pd.DataFrame({
        "sample": matched_reference_samples,
        "reference_subtype": reference_common.values,
        "predicted_subtype": pred_reference["predicted_subtype"].values,
    }).dropna()
    if comparison_df.empty:
        print(f"    [skip] PAM50 {cohort_label}: no {reference_name.lower()} subtype pairs to compare")
        return

    comparison_df["match"] = comparison_df["reference_subtype"] == comparison_df["predicted_subtype"]
    comparison_df.to_csv(
        os.path.join(variant_out, "pam50_classification_reference_vs_pred.tsv"),
        sep="\t",
        index=False,
    )
    pd.DataFrame([
        {
            "comparison": "reference_vs_predicted",
            "reference_label": reference_name,
            "n_samples": int(len(comparison_df)),
            "accuracy": float(comparison_df["match"].mean()),
        }
    ]).to_csv(
        os.path.join(variant_out, "pam50_classification_summary.tsv"),
        sep="\t",
        index=False,
    )

    class_order = sorted(set(comparison_df["reference_subtype"]) | set(comparison_df["predicted_subtype"]))
    _maybe(
        confusion_matrix_heatmap,
        overwrite,
        comparison_df["reference_subtype"].values,
        comparison_df["predicted_subtype"].values,
        os.path.join(variant_out, "pam50_confusion_matrix_reference_vs_pred"),
        class_order=class_order,
        title=f"PAM50 {reference_name} vs predicted — {cohort_label} ({variant_name})",
    )
    print(
        f"    PAM50 {cohort_label}: {int(comparison_df['match'].sum())}/{len(comparison_df)} {reference_name.lower()} subtype matches"
    )


def _run_bc_ghana_analysis(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
) -> None:
    """Receptor-gene expression boxplots for BC-Ghana."""
    meta_path = PROJECT_ROOT / "reference" / "BC-Ghana_metadata.tsv"
    if not meta_path.exists():
        print("    [skip] BC-Ghana metadata not found")
        return

    meta = pd.read_csv(meta_path, sep="\t").set_index("sample")
    common_samples = [s for s in pred.columns if s in meta.index]
    if len(common_samples) < 3:
        print(f"    [skip] BC-Ghana: only {len(common_samples)} matched samples")
        return
    print(f"    BC-Ghana: {len(common_samples)} samples matched to metadata")

    # --- Receptor-gene expression boxplots ---
    receptor_out = ensure_dir(os.path.join(model_out, "receptor_expression"))
    for status_col, gene, receptor_name in _BC_GHANA_RECEPTOR_PAIRS:
        if gene not in pred.index:
            print(f"    [skip] {gene} not in predictions")
            continue
        if status_col not in meta.columns:
            print(f"    [skip] {status_col} not in metadata")
            continue

        expr_vals = pred.loc[gene, common_samples].astype(float)
        status_vals = meta.loc[common_samples, status_col].astype(int)

        vals_neg = expr_vals[status_vals == 0].values
        vals_pos = expr_vals[status_vals == 1].values
        # Mann-Whitney U p-value
        from scipy.stats import mannwhitneyu
        if len(vals_neg) >= 2 and len(vals_pos) >= 2:
            _, mwu_p = mannwhitneyu(vals_neg, vals_pos, alternative="two-sided")
        else:
            mwu_p = None
        _maybe(
            binary_status_boxplot, overwrite,
            vals_neg, vals_pos,
            os.path.join(receptor_out,
                         f"{receptor_name}_status_vs_{gene}_expression"),
            xlabel=f"{receptor_name} IHC status",
            ylabel=f"{gene} predicted expression (log₂CPM+1)",
            title=f"{gene} expression by {receptor_name} status — BC-Ghana",
            pval=mwu_p,
        )


# ---------------------------------------------------------------------------
# Discovery_Patient: CRPC-Pluvicto ssGSEA + survival + clinical
# ---------------------------------------------------------------------------

def _run_pluvicto_ssgsea(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    tag: str = "",
) -> Optional[pd.DataFrame]:
    """Run ssGSEA (Hallmark) on per-sample Pluvicto predictions.

    Parameters
    ----------
    tag : str
        Suffix appended to output filenames.

    Returns a pathway-score DataFrame (pathways x samples) or None.
    """
    import gseapy as gp

    if not _GSEA_HALLMARK_GMT.exists():
        print(f"    [skip] Hallmark GMT not found: {_GSEA_HALLMARK_GMT}")
        return None

    ssgsea_out = ensure_dir(os.path.join(model_out, f"ssGSEA{tag}"))
    scores_tsv = os.path.join(ssgsea_out, f"ssgsea_hallmark_scores{tag}.tsv")

    if overwrite or not os.path.exists(scores_tsv):
        hallmark_sets = _load_gmt(_GSEA_HALLMARK_GMT)
        pred_for_scores = _prepare_expression_matrix_for_scores(
            pred,
            context="ssGSEA Hallmark",
        )
        if pred_for_scores.empty:
            print("    [skip] ssGSEA Hallmark: no genes available after NaN filtering")
            return None
        print(f"    ssGSEA Hallmark: {len(hallmark_sets)} gene sets, "
              f"{pred_for_scores.shape[1]} samples")
        res = gp.ssgsea(
            data=pred_for_scores,
            gene_sets=hallmark_sets,
            outdir=None,
            min_size=10,
            max_size=500,
            no_plot=True,
            verbose=False,
        )
        scores = res.res2d.pivot(
            index="Term", columns="Name", values="NES",
        ).apply(pd.to_numeric, errors="coerce")
        scores.to_csv(scores_tsv, sep="\t")
        print("    ssGSEA score matrix saved")
    else:
        scores = pd.read_csv(scores_tsv, sep="\t", index_col=0)

    # Summary: top/bottom 5 pathways by mean NES
    mean_nes = scores.mean(axis=1).sort_values()
    n_show = min(5, len(mean_nes))
    print(f"    Top {n_show} enriched: "
          + ", ".join(mean_nes.tail(n_show).index.tolist()))
    print(f"    Bottom {n_show} depleted: "
          + ", ".join(mean_nes.head(n_show).index.tolist()))

    return scores


def _run_pluvicto_ssgsea_mc(
    draws: List[pd.DataFrame],
    model_out: str,
    overwrite: bool,
) -> Optional[pd.DataFrame]:
    """Run ssGSEA on each MC draw and aggregate (mean NES + 95% CI).

    Returns mean pathway scores (pathways x samples) or None.
    Saves mean, CI-lo, and CI-hi score matrices.
    """
    import gseapy as gp

    if not _GSEA_HALLMARK_GMT.exists():
        print(f"    [skip] ssGSEA MC: Hallmark GMT not found")
        return None

    tag = "_uncertainty-adjusted"
    ssgsea_out = ensure_dir(os.path.join(model_out, f"ssGSEA{tag}"))
    mean_tsv = os.path.join(ssgsea_out, f"ssgsea_hallmark_scores{tag}.tsv")

    if not overwrite and os.path.exists(mean_tsv):
        return pd.read_csv(mean_tsv, sep="\t", index_col=0)

    hallmark_sets = _load_gmt(_GSEA_HALLMARK_GMT)
    M = len(draws)
    print(f"    ssGSEA MC: {M} draws × {len(hallmark_sets)} gene sets")

    all_scores: List[pd.DataFrame] = []
    for i, draw in enumerate(draws):
        draw_for_scores = _prepare_expression_matrix_for_scores(
            draw,
            context=f"ssGSEA MC draw {i + 1}",
        )
        if draw_for_scores.empty:
            continue
        res = gp.ssgsea(
            data=draw_for_scores,
            gene_sets=hallmark_sets,
            outdir=None,
            min_size=10,
            max_size=500,
            no_plot=True,
            verbose=False,
        )
        scores = res.res2d.pivot(
            index="Term", columns="Name", values="NES",
        ).apply(pd.to_numeric, errors="coerce")
        all_scores.append(scores)
        if (i + 1) % 10 == 0:
            print(f"      ssGSEA MC draw {i + 1}/{M} complete")

    if not all_scores:
        print("    [skip] ssGSEA MC: no genes available after NaN filtering")
        return None

    # Stack and compute statistics
    stacked = np.stack([s.values for s in all_scores], axis=0)  # (M, P, S)
    mean_scores = pd.DataFrame(
        np.mean(stacked, axis=0),
        index=all_scores[0].index, columns=all_scores[0].columns)
    ci_lo = pd.DataFrame(
        np.percentile(stacked, 2.5, axis=0),
        index=all_scores[0].index, columns=all_scores[0].columns)
    ci_hi = pd.DataFrame(
        np.percentile(stacked, 97.5, axis=0),
        index=all_scores[0].index, columns=all_scores[0].columns)

    mean_scores.to_csv(mean_tsv, sep="\t")
    ci_lo.to_csv(os.path.join(ssgsea_out, f"ssgsea_hallmark_ci_lo{tag}.tsv"),
                 sep="\t")
    ci_hi.to_csv(os.path.join(ssgsea_out, f"ssgsea_hallmark_ci_hi{tag}.tsv"),
                 sep="\t")
    print("    MC ssGSEA summary saved")

    return mean_scores


def _compute_ssgsea_score_matrix(
    expr_df: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    *,
    min_size: int,
    max_size: int = 500,
) -> pd.DataFrame:
    """Compute per-sample ssGSEA scores for a set collection."""
    import gseapy as gp

    if expr_df.empty or not gene_sets:
        return pd.DataFrame()

    prepared_expr = _prepare_expression_matrix_for_scores(
        expr_df,
        context="ssGSEA score matrix",
    )
    if prepared_expr.empty:
        return pd.DataFrame()

    res = gp.ssgsea(
        data=prepared_expr,
        gene_sets=gene_sets,
        outdir=None,
        min_size=min_size,
        max_size=max_size,
        no_plot=True,
        verbose=False,
    )
    if getattr(res, "res2d", None) is None or res.res2d.empty:
        return pd.DataFrame()
    return res.res2d.pivot(index="Term", columns="Name", values="NES").apply(
        pd.to_numeric, errors="coerce")


def _aggregate_ssgsea_score_matrices(
    draws: List[pd.DataFrame],
    gene_sets: Dict[str, List[str]],
    *,
    min_size: int,
    max_size: int = 500,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Aggregate ssGSEA score matrices across Monte Carlo draws."""
    all_scores: List[pd.DataFrame] = []
    for draw in draws:
        draw_scores = _compute_ssgsea_score_matrix(
            draw, gene_sets, min_size=min_size, max_size=max_size)
        if not draw_scores.empty:
            all_scores.append(draw_scores)

    if not all_scores:
        return None, None, None

    template = all_scores[0]
    stacked = np.stack([
        score.reindex(index=template.index, columns=template.columns).values
        for score in all_scores
    ], axis=0)
    mean_scores = pd.DataFrame(
        np.mean(stacked, axis=0), index=template.index, columns=template.columns)
    ci_lo = pd.DataFrame(
        np.percentile(stacked, 2.5, axis=0), index=template.index, columns=template.columns)
    ci_hi = pd.DataFrame(
        np.percentile(stacked, 97.5, axis=0), index=template.index, columns=template.columns)
    return mean_scores, ci_lo, ci_hi


def _load_shortlist_sets() -> Dict[str, List[str]]:
    """Load the configured shortlist sets used for per-sample scoring."""
    if not _GSEA_SHORTLIST_GMT.exists():
        return {}
    raw_sets = _load_gmt(_GSEA_SHORTLIST_GMT)
    raw_sets = {k: v for k, v in raw_sets.items() if k in _GSEA_SHORTLIST_SETS}
    if not raw_sets:
        return {}
    return _convert_entrez_gmt(raw_sets)


def _run_pluvicto_shortlist_scores(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    tag: str = "",
) -> Optional[pd.DataFrame]:
    """Compute ARG.10 / NE.10 / CCP.31 ssGSEA score matrices for Pluvicto."""
    shortlist_sets = _load_shortlist_sets()
    if not shortlist_sets:
        print(f"    [skip] Shortlist GMT not found or no shortlist sets available")
        return None

    shortlist_out = ensure_dir(os.path.join(model_out, f"ShortlistScores{tag}"))
    scores_tsv = os.path.join(shortlist_out, f"shortlist_scores{tag}.tsv")

    if overwrite or not os.path.exists(scores_tsv):
        scores = _compute_ssgsea_score_matrix(pred, shortlist_sets, min_size=3)
        if scores.empty:
            print(f"    [skip] Shortlist scores{tag}: no genes present in predictions")
            return None
        scores.to_csv(scores_tsv, sep="\t")
        print("    Shortlist scores saved")
    else:
        scores = pd.read_csv(scores_tsv, sep="\t", index_col=0)

    mean_scores = scores.mean(axis=1).sort_values(ascending=False)
    print(
        "    Shortlist mean scores: "
        + ", ".join(f"{name}={mean_scores[name]:.3f}" for name in mean_scores.index)
    )
    return scores


def _run_pluvicto_shortlist_scores_mc(
    draws: List[pd.DataFrame],
    model_out: str,
    overwrite: bool,
) -> Optional[pd.DataFrame]:
    """Aggregate shortlist score matrices across MC draws for Pluvicto."""
    shortlist_sets = _load_shortlist_sets()
    if not shortlist_sets:
        print("    [skip] Shortlist MC: no shortlist sets available")
        return None

    tag = "_uncertainty-adjusted"
    shortlist_out = ensure_dir(os.path.join(model_out, f"ShortlistScores{tag}"))
    mean_tsv = os.path.join(shortlist_out, f"shortlist_scores{tag}.tsv")

    if not overwrite and os.path.exists(mean_tsv):
        return pd.read_csv(mean_tsv, sep="\t", index_col=0)

    mean_scores, ci_lo, ci_hi = _aggregate_ssgsea_score_matrices(
        draws, shortlist_sets, min_size=3)
    if mean_scores is None or ci_lo is None or ci_hi is None:
        print("    [skip] Shortlist MC: no genes present in predictions")
        return None

    mean_scores.to_csv(mean_tsv, sep="\t")
    ci_lo.to_csv(os.path.join(shortlist_out, f"shortlist_ci_lo{tag}.tsv"), sep="\t")
    ci_hi.to_csv(os.path.join(shortlist_out, f"shortlist_ci_hi{tag}.tsv"), sep="\t")
    print("    MC shortlist scores saved")
    return mean_scores


def _analysis_safe_name(name: str) -> str:
    """Convert an arbitrary analysis label into a filesystem-safe stem."""
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)
    ).strip("_")


def _summarize_cox_ph_diagnostics(
    cox_table: Optional[pd.DataFrame],
    label: str,
    *,
    ph_column: str,
) -> None:
    """Print a compact proportional-hazards diagnostic summary for a Cox result table."""
    if cox_table is None or cox_table.empty or ph_column not in cox_table.columns:
        return

    ph_p = pd.to_numeric(cox_table[ph_column], errors="coerce")
    valid = ph_p.notna()
    if not bool(valid.any()):
        return

    n_flagged = int((ph_p[valid] < 0.05).sum())
    print(
        f"    Cox PH diagnostics ({label}): {n_flagged}/{int(valid.sum())} fits with {ph_column} < 0.05"
    )


def _plot_pluvicto_significant_multivariate_kms(
    covariates: pd.DataFrame,
    time_s: pd.Series,
    event_s: pd.Series,
    cox_scores_mv: Optional[pd.DataFrame],
    out_prefix: str,
    overwrite: bool,
    *,
    analysis_label: str,
) -> None:
    """Plot KM curves for Pluvicto covariates with significant TFx-adjusted Cox p-values."""
    if cox_scores_mv is None or cox_scores_mv.empty:
        return

    mv_rows = cox_scores_mv.copy()
    mv_rows["p"] = pd.to_numeric(mv_rows["p"], errors="coerce")
    mv_sig = mv_rows.loc[
        mv_rows["p"].notna()
        & (mv_rows["p"] < 0.05)
        & mv_rows["variable"].isin(covariates.columns)
    ].sort_values(["p", "variable"])
    if mv_sig.empty:
        print(f"    [skip] {analysis_label}: no TFx-adjusted Cox associations with p < 0.05")
        return

    from lifelines.statistics import logrank_test

    for variable in mv_sig["variable"].drop_duplicates().astype(str):
        values = pd.to_numeric(covariates[variable], errors="coerce")
        valid_values = values.dropna()
        if len(valid_values) < 10:
            continue

        median_val = valid_values.median()
        is_hi = valid_values >= median_val
        if bool(is_hi.all()) or bool((~is_hi).all()):
            continue

        time_km = time_s.loc[valid_values.index]
        event_km = event_s.loc[valid_values.index]
        lr = logrank_test(
            time_km[~is_hi], time_km[is_hi],
            event_km[~is_hi], event_km[is_hi],
        )
        _maybe(
            kaplan_meier_plot, overwrite,
            time_km[~is_hi].values, event_km[~is_hi].values,
            time_km[is_hi].values, event_km[is_hi].values,
            f"{out_prefix}_{_analysis_safe_name(variable)}",
            title=f"Overall survival — {variable}",
            logrank_p=lr.p_value,
        )


def _run_pluvicto_score_survival(
    score_df: Optional[pd.DataFrame],
    score_label: str,
    output_stem: str,
    survival_out: str,
    time_v: pd.Series,
    event_v: pd.Series,
    samples_v: List[str],
    pids_v: List[str],
    tfx_series_v: Optional[pd.Series],
    overwrite: bool,
    tag: str,
) -> None:
    """Run Cox PH summaries and plots for one Pluvicto score family."""
    if score_df is None or score_df.empty:
        print(f"    [skip] {score_label}: score matrix unavailable")
        return

    available_pairs = [
        (sample, pid)
        for sample, pid in zip(samples_v, pids_v)
        if sample in score_df.columns
    ]
    if len(available_pairs) < 10:
        print(f"    [skip] {score_label}: only {len(available_pairs)} matched samples")
        return

    matched_samples = [sample for sample, _ in available_pairs]
    matched_pids = [pid for _, pid in available_pairs]
    covariates = score_df[matched_samples].T
    covariates.index = matched_pids

    time_s = time_v.loc[matched_pids]
    event_s = event_v.loc[matched_pids]
    cox_scores = compute_cox_univariate(time_s, event_s, covariates)
    cox_scores.to_csv(
        os.path.join(survival_out, f"cox_{output_stem}_results{tag}.tsv"),
        sep="\t", index=False,
    )
    _summarize_cox_ph_diagnostics(cox_scores, f"{score_label} univariate", ph_column="ph_p")

    cox_scores_mv = None
    if tfx_series_v is not None:
        tfx_s = tfx_series_v.reindex(matched_pids)
        if tfx_s.notna().any():
            cox_scores_mv = compute_cox_multivariate(time_s, event_s, covariates, tfx_s)
            cox_scores_mv.to_csv(
                os.path.join(survival_out, f"cox_{output_stem}_results_mv{tag}.tsv"),
                sep="\t", index=False,
            )
            _summarize_cox_ph_diagnostics(
                cox_scores_mv,
                f"{score_label} TFx-adjusted",
                ph_column="ph_min_p",
            )

    cox_sig = cox_scores.dropna(subset=["HR"])
    if cox_sig.empty:
        print(f"    [skip] {score_label}: no finite Cox estimates")
        return

    cox_sig = cox_sig.sort_values("HR")
    n_show = min(10, len(cox_sig))
    show_rows = pd.concat([cox_sig.head(n_show), cox_sig.tail(n_show)])
    show_rows = show_rows.drop_duplicates(subset=["variable"])
    show_mv = None
    if cox_scores_mv is not None:
        show_mv = cox_scores_mv[cox_scores_mv["variable"].isin(show_rows["variable"])]

    _maybe(
        forest_plot, overwrite,
        show_rows,
        os.path.join(survival_out, f"forest_{output_stem}{tag}"),
        title=f"Cox PH — {score_label}{tag}",
        hr_table_mv=show_mv,
    )
    _plot_pluvicto_significant_multivariate_kms(
        covariates,
        time_s,
        event_s,
        cox_scores_mv,
        os.path.join(survival_out, f"KM_{output_stem}{tag}"),
        overwrite,
        analysis_label=score_label,
    )


def _run_pluvicto_survival(
    pred: pd.DataFrame,
    ssgsea_scores: Optional[pd.DataFrame],
    shortlist_scores: Optional[pd.DataFrame],
    model_out: str,
    overwrite: bool,
    minimum_tfx: float = 0.0,
    tag: str = "",
) -> None:
    """Forest plots and KM curves for Pluvicto survival analysis."""
    outcomes = load_pluvicto_outcomes()
    clinical = load_pluvicto_clinical()

    # Match prediction samples to outcomes via patient ID
    pid_map = {s: extract_pluvicto_patient_id(s) for s in pred.columns}
    matched_pids = {pid for pid in pid_map.values() if pid in outcomes.index}
    if len(matched_pids) < 10:
        print(f"    [skip] Pluvicto survival: only {len(matched_pids)} matched")
        return
    # Use one sample per patient (first encountered)
    pid_to_sample: Dict[str, str] = {}
    for s, pid in pid_map.items():
        if pid in matched_pids and pid not in pid_to_sample:
            pid_to_sample[pid] = s
    pids = sorted(pid_to_sample.keys())
    samples = [pid_to_sample[p] for p in pids]

    time = outcomes.loc[pids, "survival_days"]
    event = outcomes.loc[pids, "Death"]
    valid = time.notna() & event.notna()
    pids_v = [p for p, v in zip(pids, valid) if v]
    samples_v = [pid_to_sample[p] for p in pids_v]
    time_v = time.loc[pids_v].astype(float)
    event_v = event.loc[pids_v].astype(int)
    print(f"    Pluvicto survival: {len(pids_v)} patients with events "
          f"({int(event_v.sum())} deaths)")

    survival_out = ensure_dir(os.path.join(model_out, f"survival{tag}"))

    # --- Load TFx for multivariate adjustment (used by both forest plots) ---
    tfx_by_pid: Dict[str, float] = {}
    sub_df = _load_filtered_subtypes_for_reference(
        "CRPC-Pluvicto",
        pred.columns,
        minimum_tfx=minimum_tfx,
        report_context="Discovery_Patient CRPC-Pluvicto",
    )
    if sub_df is not None and "tfx" in sub_df.columns:
        for sname, tfx_val in sub_df["tfx"].items():
            pid = extract_pluvicto_patient_id(str(sname))
            if pid in pids_v and pid not in tfx_by_pid:
                tfx_by_pid[pid] = float(tfx_val)
    tfx_series_v = pd.Series(tfx_by_pid).reindex(pids_v) if tfx_by_pid else None

    _run_pluvicto_score_survival(
        ssgsea_scores,
        "Hallmark pathway scores",
        "pathway",
        survival_out,
        time_v,
        event_v,
        samples_v,
        pids_v,
        tfx_series_v,
        overwrite,
        tag,
    )
    _run_pluvicto_score_survival(
        shortlist_scores,
        "ARG.10 / NE.10 / CCP.31 shortlist scores",
        "shortlist",
        survival_out,
        time_v,
        event_v,
        samples_v,
        pids_v,
        tfx_series_v,
        overwrite,
        tag,
    )

    # --- Forest plot 2: clinical / imaging variables ---
    imaging_cols = [
        "PSMA-PET_SUVmax", "PSMA-PET_SUVmean",
        "PSMA-PET_PSMAexpression", "PSMA-PET_Volume",
        "FDG_SUVmax", "FDG_SUVmean", "FDG_TLG", "FDG_Volume",
    ]
    extra_cols = ["Baseline_PSA"]
    clin_cols = [c for c in imaging_cols + extra_cols if c in clinical.columns]
    clin_pids = [p for p in pids_v if p in clinical.index]
    if len(clin_pids) >= 10 and clin_cols:
        clin_covs = clinical.loc[clin_pids, clin_cols]
        time_c = time_v.loc[clin_pids]
        event_c = event_v.loc[clin_pids]

        # Add TFx from pre-loaded series
        if tfx_series_v is not None:
            tfx_clin = tfx_series_v.reindex(clin_pids).dropna()
            if not tfx_clin.empty:
                clin_covs = clin_covs.copy()
                clin_covs["TFx"] = tfx_clin

        cox_clin = compute_cox_univariate(time_c, event_c, clin_covs)
        cox_clin.to_csv(
            os.path.join(survival_out, f"cox_clinical_results{tag}.tsv"),
            sep="\t", index=False)
        _summarize_cox_ph_diagnostics(cox_clin, "clinical univariate", ph_column="ph_p")

        # Multivariate (TFx-adjusted) — only non-TFx covariates
        cox_clin_mv = None
        if tfx_series_v is not None:
            tfx_c = tfx_series_v.reindex(clin_pids)
            non_tfx_cols = [c for c in clin_covs.columns if c != "TFx"]
            if non_tfx_cols:
                clin_covs_notfx = clin_covs[non_tfx_cols]
                cox_clin_mv = compute_cox_multivariate(
                    time_c, event_c, clin_covs_notfx, tfx_c)
                cox_clin_mv.to_csv(
                    os.path.join(survival_out, f"cox_clinical_results_mv{tag}.tsv"),
                    sep="\t", index=False)
                _summarize_cox_ph_diagnostics(
                    cox_clin_mv,
                    "clinical TFx-adjusted",
                    ph_column="ph_min_p",
                )

        cox_clin_valid = cox_clin.dropna(subset=["HR"])
        if not cox_clin_valid.empty:
            # Match multivariate rows
            clin_mv_for_plot = None
            if cox_clin_mv is not None:
                clin_mv_for_plot = cox_clin_mv[
                    cox_clin_mv["variable"].isin(cox_clin_valid["variable"])
                ]
            _maybe(
                forest_plot, overwrite,
                cox_clin_valid,
                os.path.join(survival_out, f"forest_clinical{tag}"),
                title=f"Cox PH — clinical & imaging variables{tag}",
                hr_table_mv=clin_mv_for_plot,
            )

            _plot_pluvicto_significant_multivariate_kms(
                clin_covs,
                time_c,
                event_c,
                cox_clin_mv,
                os.path.join(survival_out, f"KM_clinical{tag}"),
                overwrite,
                analysis_label="Clinical & imaging variables",
            )


def _run_pluvicto_clinical_correlations(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    *,
    shortlist_scores: Optional[pd.DataFrame] = None,
    tag: str = "",
) -> None:
    """Scatter plots: FOLH1 vs PSMA-PET, CCP31 vs FDG imaging variables."""
    clinical = load_pluvicto_clinical()

    # Match prediction samples to clinical data via patient ID
    pid_map = {s: extract_pluvicto_patient_id(s) for s in pred.columns}
    pid_to_sample: Dict[str, str] = {}
    for s, pid in pid_map.items():
        if pid in clinical.index and pid not in pid_to_sample:
            pid_to_sample[pid] = s
    pids = sorted(pid_to_sample.keys())
    samples = [pid_to_sample[p] for p in pids]
    if len(pids) < 5:
        print(f"    [skip] Pluvicto clinical correlations: "
              f"only {len(pids)} matched")
        return
    print(f"    Pluvicto clinical correlations: {len(pids)} patients matched")

    corr_dir = "clinical_correlations" if not tag else f"clinical_correlations{tag}"
    corr_out = ensure_dir(os.path.join(model_out, corr_dir))

    # --- FOLH1 vs PSMA-PET ---
    if "FOLH1" in pred.index:
        folh1_vals = pred.loc["FOLH1", samples].values.astype(float)
        folh1_series = pd.Series(folh1_vals, index=pids)
        for psma_col in ["PSMA-PET_SUVmax", "PSMA-PET_SUVmean",
                         "PSMA-PET_PSMAexpression", "PSMA-PET_Volume"]:
            if psma_col not in clinical.columns:
                continue
            clin_vals = clinical.loc[pids, psma_col]
            valid = folh1_series.notna() & clin_vals.notna()
            if valid.sum() < 5:
                continue
            _maybe(
                clinical_correlation_scatter, overwrite,
                folh1_series[valid].values, clin_vals[valid].values,
                os.path.join(corr_out, f"FOLH1_vs_{psma_col}"),
                xlabel="FOLH1 predicted expression (log₂CPM+1)",
                ylabel=psma_col,
                title=f"FOLH1 vs {psma_col}",
                y_log_scale=psma_col.endswith("_Volume") or psma_col.endswith("_TLG"),
            )
    else:
        print("    [skip] FOLH1 not in predictions")

    # --- CCP31 score vs FDG ---
    ccp31_series: Optional[pd.Series] = None
    ccp31_xlabel = "CCP31 score"
    if shortlist_scores is not None and "CCP.31" in shortlist_scores.index:
        ccp31_series = pd.to_numeric(shortlist_scores.loc["CCP.31"].reindex(samples), errors="coerce")
        ccp31_series.index = pids
        ccp31_xlabel = "CCP31 ssGSEA score"
    else:
        ccp31_name = STANDARD_GENE_SETS.get("CCP31", "CCP31")
        if gene_list_available(ccp31_name):
            ccp31_genes = get_gene_list(ccp31_name)
            ccp31_score = compute_geneset_score(pred[samples], ccp31_genes)
            ccp31_series = pd.Series(ccp31_score.values, index=pids)
            ccp31_xlabel = "CCP31 score (mean log₂CPM+1)"

    if ccp31_series is not None:
        for fdg_col in ["FDG_SUVmax", "FDG_SUVmean", "FDG_TLG", "FDG_Volume"]:
            if fdg_col not in clinical.columns:
                continue
            clin_vals = clinical.loc[pids, fdg_col]
            valid = ccp31_series.notna() & clin_vals.notna()
            if valid.sum() < 5:
                continue
            _maybe(
                clinical_correlation_scatter, overwrite,
                ccp31_series[valid].values, clin_vals[valid].values,
                os.path.join(corr_out, f"CCP31_vs_{fdg_col}"),
                xlabel=ccp31_xlabel,
                ylabel=fdg_col,
                title=f"CCP31 vs {fdg_col}",
                y_log_scale=fdg_col.endswith("_Volume") or fdg_col.endswith("_TLG"),
            )
    else:
        print("    [skip] CCP31 gene set not available")


def _pfizer_patient_id(sample_id: str) -> str:
    """Extract the patient identifier from a CRPC-Pfizer sample ID."""
    return str(sample_id).split("_", 1)[0]


def _pfizer_normalize_best_response(response_value: Any) -> str:
    """Normalize Pfizer response labels into a canonical set."""
    response = str(response_value).strip()
    if response == "PR &PSA50":
        return "PR/PSA50"
    return response


def _pfizer_pretty_response(code: str) -> str:
    """Map response scheme codes to display labels."""
    return {
        "Progression": "Progression",
        "DiseaseControl": "Disease Control / Non-Progression",
        "PD": "Progression",
        "NonCRPD": "Disease Control / Non-Progression",
        "SDPR": "Disease Control / Non-Progression",
    }.get(code, code)


def _pfizer_response_scheme_specs() -> List[Dict[str, Any]]:
    """Return the single Pfizer binary response encoding used throughout discovery."""
    return [
        {
            "name": "Progression_vs_DiseaseControl",
            "column": "response_binary",
            "levels": ["DiseaseControl", "Progression"],
            "binary": True,
        },
    ]


def _pfizer_safe_name(name: str) -> str:
    """Convert an arbitrary marker label into a filesystem-safe stem."""
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)
    ).strip("_")


def _load_pfizer_metadata(pred_columns: Sequence[str]) -> pd.DataFrame:
    """Load and match CRPC-Pfizer metadata to prediction sample names."""
    path = PROJECT_ROOT / "reference" / "CRPC-Pfizer_metadata.tsv"
    meta = pd.read_csv(path, sep="\t")
    meta.columns = [str(column).strip() for column in meta.columns]
    meta = meta.rename(
        columns={
            "sample_ID": "sample_id",
            "EZFDCAT": "enzalutamide_group",
            "PF-1497_BID": "treatment_arm",
            "Best_Overall_Resp": "best_response",
            "Pfizer_Visit": "visit",
            "Days_from_SCN": "days_from_scn",
        }
    )
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["best_response"] = meta["best_response"].map(_pfizer_normalize_best_response)
    meta["visit"] = pd.Categorical(
        meta["visit"].astype(str).str.strip(),
        categories=_PFIZER_VISIT_ORDER,
        ordered=True,
    )
    meta["days_from_scn"] = pd.to_numeric(meta["days_from_scn"], errors="coerce")
    meta["patient_id"] = meta["sample_id"].map(_pfizer_patient_id)

    response_series = meta["best_response"].astype("string")

    meta["response_binary"] = pd.Series(pd.NA, index=meta.index, dtype="string")
    meta.loc[response_series.eq("PD"), "response_binary"] = "Progression"
    meta.loc[
        response_series.isin(["Non-CR/PD", "SD", "PR/PSA50"]),
        "response_binary",
    ] = "DiseaseControl"

    meta_index = set(meta["sample_id"])
    matched_rows: List[pd.Series] = []
    used_meta_ids: set[str] = set()
    for pred_sample in pred_columns:
        matched_sample = None
        if pred_sample in meta_index:
            matched_sample = pred_sample
        else:
            stripped_sample = str(strip_cohort_prefix(pd.Index([pred_sample]), "CRPC-Pfizer")[0])
            if stripped_sample in meta_index and stripped_sample not in used_meta_ids:
                matched_sample = stripped_sample
        if matched_sample is None:
            continue
        row = meta.loc[meta["sample_id"] == matched_sample].iloc[0].copy()
        row["prediction_sample"] = pred_sample
        used_meta_ids.add(matched_sample)
        matched_rows.append(row)

    if not matched_rows:
        return pd.DataFrame()

    matched = pd.DataFrame(matched_rows).set_index("prediction_sample")
    ordered_samples = [sample for sample in pred_columns if sample in matched.index]
    return matched.loc[ordered_samples]


def _pfizer_select_hallmarks_from_gsea(model_out: str, tag: str, *, n_top: int = 4) -> List[str]:
    """Read the top hallmark enrichments from a saved GSEA result table."""
    path = os.path.join(model_out, f"GSEA{tag}", "Hallmark", f"gsea_hallmark_results{tag}.tsv")
    if not os.path.exists(path):
        return []

    gsea_df = pd.read_csv(path, sep="\t")
    if gsea_df.empty:
        return []

    term_col = "Term" if "Term" in gsea_df.columns else "Name"
    fdr_col = "FDR q-val" if "FDR q-val" in gsea_df.columns else "fdr"
    nes_col = "NES" if "NES" in gsea_df.columns else "nes"
    if term_col not in gsea_df.columns or fdr_col not in gsea_df.columns:
        return []

    working = gsea_df[[term_col, fdr_col] + ([nes_col] if nes_col in gsea_df.columns else [])].copy()
    working[fdr_col] = pd.to_numeric(working[fdr_col], errors="coerce")
    if nes_col in working.columns:
        working[nes_col] = pd.to_numeric(working[nes_col], errors="coerce")
        working["abs_nes"] = working[nes_col].abs()
    else:
        working["abs_nes"] = 0.0
    working = working.dropna(subset=[fdr_col])
    if working.empty:
        return []

    sig = working.loc[working[fdr_col] < 0.25].copy()
    ranked = sig if not sig.empty else working
    ranked = ranked.sort_values([fdr_col, "abs_nes"], ascending=[True, False])
    return ranked[term_col].head(n_top).astype(str).tolist()


def _pfizer_run_limma_and_gsea(
    pred: pd.DataFrame,
    pfizer_meta: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    tag_prefix: str,
    *,
    mc_draws: Optional[List[pd.DataFrame]] = None,
) -> List[str]:
    """Run timepoint-specific response limma/GSEA contrasts for CRPC-Pfizer."""
    interesting_hallmarks: List[str] = []
    analyses = [
        ("SCN", "response_binary", "Progression-DiseaseControl"),
        ("C3D1", "response_binary", "Progression-DiseaseControl"),
    ]

    for visit_name, response_column, contrast in analyses:
        visit_meta = pfizer_meta.loc[
            (pfizer_meta["visit"].astype(str) == visit_name)
            & pfizer_meta[response_column].notna()
        ].copy()
        if visit_meta.empty:
            continue

        subtype_df = visit_meta[[response_column]].rename(columns={response_column: "subtype"})
        contrast_tag = f"{tag_prefix}_{visit_name}_{response_column}"
        pred_visit = pred[visit_meta.index]
        if mc_draws is None:
            de_results = _run_limma_de(
                pred_visit,
                subtype_df,
                model_out,
                overwrite,
                contrast=contrast,
                tag=contrast_tag,
            )
        else:
            visit_draws = [draw[visit_meta.index] for draw in mc_draws]
            de_results = _run_limma_de_mc(
                visit_draws,
                pred_visit,
                subtype_df,
                model_out,
                overwrite,
                contrast=contrast,
                tag=contrast_tag,
            )
        if de_results is None:
            continue

        _run_gsea(de_results, model_out, overwrite, contrast=contrast, tag=contrast_tag)
        if visit_name == "C3D1" and response_column == "response_binary":
            interesting_hallmarks.extend(_pfizer_select_hallmarks_from_gsea(model_out, contrast_tag))

    deduped: List[str] = []
    seen: set[str] = set()
    for hallmark_name in interesting_hallmarks:
        if hallmark_name in seen:
            continue
        deduped.append(hallmark_name)
        seen.add(hallmark_name)
    return deduped


def _build_pfizer_marker_matrix(
    pred: pd.DataFrame,
    shortlist_scores: Optional[pd.DataFrame],
    hallmark_scores: Optional[pd.DataFrame],
    interesting_hallmarks: Sequence[str],
) -> pd.DataFrame:
    """Assemble the requested Pfizer marker panel as a markers × samples matrix."""
    marker_rows: Dict[str, pd.Series] = {}
    shortlist_sets = _load_shortlist_sets()

    if shortlist_scores is not None:
        for score_name in ("NE.10", "ARG.10"):
            if score_name in shortlist_scores.index:
                marker_rows[f"{score_name} score"] = shortlist_scores.loc[score_name]

    if "EZH2" in pred.index:
        marker_rows["EZH2"] = pred.loc["EZH2"]

    for gene_name in shortlist_sets.get("NE.10", []):
        if gene_name in pred.index:
            marker_rows[gene_name] = pred.loc[gene_name]

    if hallmark_scores is not None:
        for hallmark_name in interesting_hallmarks:
            if hallmark_name in hallmark_scores.index:
                marker_rows[hallmark_name] = hallmark_scores.loc[hallmark_name]

    if not marker_rows:
        return pd.DataFrame()

    marker_df = pd.DataFrame(marker_rows).T
    marker_df = marker_df.loc[~marker_df.index.duplicated(keep="first")]
    return marker_df


def _pfizer_plot_spaghetti(
    marker_name: str,
    values: pd.Series,
    pfizer_meta: pd.DataFrame,
    out_path: str,
    *,
    title_suffix: str,
) -> None:
    """Plot per-patient longitudinal trajectories using actual follow-up days."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker_values = pd.to_numeric(values.reindex(pfizer_meta.index), errors="coerce")
    valid = pfizer_meta.loc[marker_values.notna()].copy()
    if valid.empty:
        return
    valid["value"] = marker_values.loc[valid.index].astype(float)
    valid = valid.loc[valid["days_from_scn"].notna()].copy()
    if valid.empty:
        return
    valid = valid.sort_values(["patient_id", "days_from_scn", "visit"])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for patient_id, patient_df in valid.groupby("patient_id"):
        patient_df = patient_df.sort_values(["days_from_scn", "visit"])
        response_code = str(patient_df["response_binary"].iloc[0])
        color = _PFIZER_RESPONSE_COLOR_MAP.get(_pfizer_pretty_response(response_code), "#888888")
        x_vals = patient_df["days_from_scn"].astype(float).tolist()
        y_vals = patient_df["value"].tolist()
        if len(x_vals) > 1:
            ax.plot(x_vals, y_vals, color=color, alpha=0.55, linewidth=1.3, zorder=1)
        for visit_name, x_val, y_val in zip(patient_df["visit"].astype(str), x_vals, y_vals):
            ax.scatter(
                x_val,
                y_val,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                s=38,
                marker={"SCN": "o", "C3D1": "s", "EOT": "D"}.get(visit_name, "o"),
                alpha=0.9,
                zorder=3,
            )

    visit_reference_days = []
    visit_reference_labels = []
    for visit_name in _PFIZER_VISIT_ORDER:
        visit_days = pd.to_numeric(
            valid.loc[valid["visit"].astype(str) == visit_name, "days_from_scn"],
            errors="coerce",
        ).dropna()
        if visit_days.empty:
            continue
        ref_day = float(np.median(visit_days))
        visit_reference_days.append(ref_day)
        visit_reference_labels.append(visit_name)
        ax.axvline(ref_day, color="#D7D7D7", linestyle="--", linewidth=0.9, zorder=0)

    legend_handles = []
    available_codes = set(valid["response_binary"].dropna().astype(str))
    for code in ("Progression", "DiseaseControl"):
        if code not in available_codes:
            continue
        label = _pfizer_pretty_response(code)
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=_PFIZER_RESPONSE_COLOR_MAP[label],
                marker="o",
                markerfacecolor=_PFIZER_RESPONSE_COLOR_MAP[label],
                markeredgecolor="black",
                linewidth=1.2,
                label=label,
            )
        )

    if visit_reference_days:
        ax.set_xticks(visit_reference_days)
        ax.set_xticklabels(visit_reference_labels)
    ax.set_xlabel("Days from SCN")
    ax.set_ylabel(marker_name)
    ax.set_title(f"{marker_name} longitudinal trajectories — {title_suffix}", fontsize=12, fontweight="bold")
    if legend_handles:
        ax.legend(handles=legend_handles, title="Best response", fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_plot_timepoint_distribution(
    marker_name: str,
    values: pd.Series,
    pfizer_meta: pd.DataFrame,
    out_path: str,
    *,
    title_suffix: str,
) -> None:
    """Plot a marker across visits with response-stratified boxplots per timepoint."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker_values = pd.to_numeric(values.reindex(pfizer_meta.index), errors="coerce")
    valid = pfizer_meta.loc[marker_values.notna()].copy()
    if valid.empty:
        return
    valid["value"] = marker_values.loc[valid.index].astype(float)
    valid = valid.sort_values(["visit", "patient_id", "days_from_scn"])

    response_codes = ["DiseaseControl", "Progression"]
    response_labels = {code: _pfizer_pretty_response(code) for code in response_codes}
    visit_centers = np.arange(len(_PFIZER_VISIT_ORDER), dtype=float)
    response_offsets = {"DiseaseControl": -0.16, "Progression": 0.16}
    box_width = 0.28

    box_data: List[np.ndarray] = []
    box_positions: List[float] = []
    box_colors: List[str] = []
    point_groups: List[Tuple[float, np.ndarray, str]] = []
    nonempty_groups = 0
    for visit_idx, visit_name in enumerate(_PFIZER_VISIT_ORDER):
        visit_df = valid.loc[valid["visit"].astype(str) == visit_name]
        for response_code in response_codes:
            group_df = visit_df.loc[visit_df["response_binary"].astype(str) == response_code]
            group_vals = group_df["value"].astype(float).values
            if len(group_vals) == 0:
                continue
            nonempty_groups += 1
            position = float(visit_centers[visit_idx] + response_offsets[response_code])
            box_data.append(group_vals)
            box_positions.append(position)
            box_colors.append(_PFIZER_RESPONSE_COLOR_MAP[response_labels[response_code]])
            point_groups.append((position, group_vals, _PFIZER_RESPONSE_COLOR_MAP[response_labels[response_code]]))

    if nonempty_groups < 2:
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(markersize=0),
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    rng = np.random.default_rng(42)
    for position, group_vals, color in point_groups:
        x_vals = position + rng.uniform(-0.055, 0.055, size=len(group_vals))
        ax.scatter(
            x_vals,
            group_vals,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            s=34,
            alpha=0.9,
            zorder=3,
        )

    for divider in visit_centers[:-1] + 0.5:
        ax.axvline(divider, color="#DDDDDD", linewidth=0.8, zorder=0)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=_PFIZER_RESPONSE_COLOR_MAP[response_labels[code]],
            markeredgecolor="black",
            markersize=8,
            linewidth=0,
            label=response_labels[code],
        )
        for code in response_codes
        if any(valid["response_binary"].astype(str) == code)
    ]

    ax.set_xticks(visit_centers)
    ax.set_xticklabels(_PFIZER_VISIT_ORDER)
    ax.set_xlabel("Visit")
    ax.set_ylabel(marker_name)
    ax.set_title(f"{marker_name} across visits — {title_suffix}", fontsize=12, fontweight="bold")
    if legend_handles:
        ax.legend(handles=legend_handles, title="Best response", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_build_delta_table(marker_df: pd.DataFrame, pfizer_meta: pd.DataFrame) -> pd.DataFrame:
    """Compute patient-level SCN→C3D1 deltas for every marker."""
    scn_meta = pfizer_meta.loc[pfizer_meta["visit"].astype(str) == "SCN"].copy().reset_index()
    scn_meta = scn_meta.rename(columns={"index": "prediction_sample"})
    c3d1_meta = pfizer_meta.loc[pfizer_meta["visit"].astype(str) == "C3D1"].copy().reset_index()
    c3d1_meta = c3d1_meta.rename(columns={"index": "prediction_sample"})
    if scn_meta.empty or c3d1_meta.empty:
        return pd.DataFrame()

    scn_meta = scn_meta.drop_duplicates(subset=["patient_id"], keep="first").set_index("patient_id")
    c3d1_meta = c3d1_meta.drop_duplicates(subset=["patient_id"], keep="first").set_index("patient_id")
    common_patients = scn_meta.index.intersection(c3d1_meta.index)
    if len(common_patients) == 0:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for patient_id in common_patients:
        scn_sample = scn_meta.loc[patient_id, "prediction_sample"]
        c3d1_sample = c3d1_meta.loc[patient_id, "prediction_sample"]
        row: Dict[str, Any] = {
            "patient_id": patient_id,
            "best_response_display": _pfizer_pretty_response(str(c3d1_meta.loc[patient_id, "response_binary"])),
            "response_binary": c3d1_meta.loc[patient_id, "response_binary"],
        }
        for marker_name in marker_df.index:
            scn_val = pd.to_numeric(pd.Series([marker_df.loc[marker_name, scn_sample]]), errors="coerce").iloc[0]
            c3d1_val = pd.to_numeric(pd.Series([marker_df.loc[marker_name, c3d1_sample]]), errors="coerce").iloc[0]
            row[marker_name] = np.nan if (not np.isfinite(scn_val) or not np.isfinite(c3d1_val)) else float(c3d1_val - scn_val)
        rows.append(row)
    return pd.DataFrame(rows).set_index("patient_id")


def _pfizer_plot_delta_slope(
    marker_name: str,
    values: pd.Series,
    pfizer_meta: pd.DataFrame,
    out_path: str,
    *,
    title_suffix: str,
) -> None:
    """Plot paired SCN→C3D1 trajectories for one marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marker_values = pd.to_numeric(values.reindex(pfizer_meta.index), errors="coerce")
    valid = pfizer_meta.loc[marker_values.notna()].copy()
    if valid.empty:
        return
    valid["value"] = marker_values.loc[valid.index].astype(float)
    valid = valid.loc[valid["visit"].astype(str).isin(["SCN", "C3D1"])]

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    x_positions = {"SCN": 0, "C3D1": 1}
    for patient_id, patient_df in valid.groupby("patient_id"):
        patient_df = patient_df.sort_values(["visit", "days_from_scn"])
        if set(patient_df["visit"].astype(str)) != {"SCN", "C3D1"}:
            continue
        response_code = str(patient_df["response_binary"].iloc[0])
        color = _PFIZER_RESPONSE_COLOR_MAP.get(_pfizer_pretty_response(response_code), "#888888")
        x_vals = [x_positions[str(visit)] for visit in patient_df["visit"].astype(str)]
        y_vals = patient_df["value"].tolist()
        ax.plot(x_vals, y_vals, color=color, alpha=0.55, linewidth=1.3)
        ax.scatter(x_vals, y_vals, color=color, edgecolor="black", linewidth=0.4, s=40, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SCN", "C3D1"])
    ax.set_xlabel("Visit")
    ax.set_ylabel(marker_name)
    ax.set_title(f"{marker_name}: SCN → C3D1 — {title_suffix}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_plot_delta_distribution(
    marker_name: str,
    delta_df: pd.DataFrame,
    response_column: str,
    response_levels: Sequence[str],
    out_path: str,
    *,
    title_suffix: str,
) -> None:
    """Plot a response-stratified delta distribution for one marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if marker_name not in delta_df.columns or response_column not in delta_df.columns:
        return
    working = delta_df[[marker_name, response_column]].dropna().copy()
    if working.empty:
        return

    grouped = [
        working.loc[working[response_column] == level, marker_name].astype(float).values
        for level in response_levels
    ]
    if sum(len(vals) > 0 for vals in grouped) < 2:
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    positions = list(range(len(response_levels)))
    bp = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(markersize=0),
        showfliers=False,
    )
    for patch, level in zip(bp["boxes"], response_levels):
        patch.set_facecolor(_PFIZER_RESPONSE_COLOR_MAP.get(_pfizer_pretty_response(level), "#BBBBBB"))
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")

    for pos, level, vals in zip(positions, response_levels, grouped):
        color = _PFIZER_RESPONSE_COLOR_MAP.get(_pfizer_pretty_response(level), "#888888")
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos + jitter, vals, color=color, edgecolor="black", linewidth=0.4, s=34, alpha=0.85, zorder=3)

    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([_pfizer_pretty_response(level) for level in response_levels])
    ax.set_xlabel("Best response")
    ax.set_ylabel(f"Δ {marker_name} (C3D1 - SCN)")
    ax.set_title(f"Δ {marker_name} by response — {title_suffix}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_plot_state_space(
    ne_scores: pd.Series,
    ar_scores: pd.Series,
    pfizer_meta: pd.DataFrame,
    out_path: str,
    *,
    title_suffix: str,
) -> None:
    """Plot NE-like vs AR-like scores as a longitudinal state space with arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ne_vals = pd.to_numeric(ne_scores.reindex(pfizer_meta.index), errors="coerce")
    ar_vals = pd.to_numeric(ar_scores.reindex(pfizer_meta.index), errors="coerce")
    valid = pfizer_meta.loc[ne_vals.notna() & ar_vals.notna()].copy()
    if valid.empty:
        return
    valid["ne_score"] = ne_vals.loc[valid.index].astype(float)
    valid["ar_score"] = ar_vals.loc[valid.index].astype(float)
    valid = valid.sort_values(["patient_id", "visit", "days_from_scn"])

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    visit_markers = {"SCN": "o", "C3D1": "s", "EOT": "D"}
    for patient_id, patient_df in valid.groupby("patient_id"):
        patient_df = patient_df.sort_values(["visit", "days_from_scn"])
        response_code = str(patient_df["response_binary"].iloc[0])
        color = _PFIZER_RESPONSE_COLOR_MAP.get(_pfizer_pretty_response(response_code), "#888888")
        x_vals = patient_df["ar_score"].astype(float).tolist()
        y_vals = patient_df["ne_score"].astype(float).tolist()
        for idx in range(len(patient_df) - 1):
            ax.annotate(
                "",
                xy=(x_vals[idx + 1], y_vals[idx + 1]),
                xytext=(x_vals[idx], y_vals[idx]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3, alpha=0.75),
            )
        for visit_name, x_val, y_val in zip(patient_df["visit"].astype(str), x_vals, y_vals):
            ax.scatter(
                x_val,
                y_val,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                s=52,
                marker=visit_markers.get(visit_name, "o"),
                alpha=0.9,
            )

    ax.set_xlabel("ARG.10 score")
    ax.set_ylabel("NE.10 score")
    ax.set_title(f"AR-like vs NE-like state space — {title_suffix}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_pfizer_mixed_models(
    marker_df: pd.DataFrame,
    pfizer_meta: pd.DataFrame,
    out_dir: str,
    *,
    overwrite: bool,
    title_suffix: str,
) -> None:
    """Fit visit × response random-intercept mixed models for the Pfizer marker panel."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    marker_long_rows: List[Dict[str, Any]] = []
    meta_scn_c3d1 = pfizer_meta.loc[pfizer_meta["visit"].astype(str).isin(["SCN", "C3D1"])]
    for marker_name in marker_df.index:
        marker_values = pd.to_numeric(marker_df.loc[marker_name].reindex(meta_scn_c3d1.index), errors="coerce")
        for sample_name, value in marker_values.items():
            if not np.isfinite(value):
                continue
            row = meta_scn_c3d1.loc[sample_name]
            marker_long_rows.append(
                {
                    "marker": marker_name,
                    "patient": row["patient_id"],
                    "visit": str(row["visit"]),
                    "value": float(value),
                    "response_binary": row["response_binary"],
                }
            )

    if not marker_long_rows:
        print("    [skip] CRPC-Pfizer mixed models: no SCN/C3D1 marker data")
        return

    marker_long = pd.DataFrame(marker_long_rows)
    ro.r(
        """
        run_pfizer_mixed_models <- function(df, response_levels) {
          suppressPackageStartupMessages(library(nlme))
          if (nrow(df) == 0) {
            return(data.frame())
          }
          df$visit <- factor(df$visit, levels = c('SCN', 'C3D1'))
          df$response <- factor(df$response, levels = response_levels)
          out <- list()
          for (marker_name in unique(df$marker)) {
            dsub <- droplevels(df[df$marker == marker_name, ])
            if (nrow(dsub) < 4) {
              next
            }
            if (length(unique(dsub$patient)) < 3 || length(unique(dsub$visit)) < 2 || length(unique(dsub$response)) < 2) {
              next
            }
            fit <- try(
              nlme::lme(
                value ~ visit * response,
                random = ~1 | patient,
                data = dsub,
                method = 'REML',
                na.action = na.omit,
                control = nlme::lmeControl(msMaxIter = 200)
              ),
              silent = TRUE
            )
            if (inherits(fit, 'try-error')) {
              next
            }
            tt <- as.data.frame(summary(fit)$tTable)
            tt$term <- rownames(tt)
            tt$marker <- marker_name
            tt$n_obs <- nrow(dsub)
            tt$n_patients <- length(unique(dsub$patient))
            rownames(tt) <- NULL
            out[[length(out) + 1]] <- tt
          }
          if (length(out) == 0) {
            return(data.frame())
          }
          do.call(rbind, out)
        }
        """
    )

    converter = ro.default_converter + pandas2ri.converter
    for scheme in _pfizer_response_scheme_specs():
        scheme_df = marker_long[["marker", "patient", "visit", "value", scheme["column"]]].rename(
            columns={scheme["column"]: "response"}
        )
        scheme_df = scheme_df.dropna(subset=["response"])
        if scheme_df.empty:
            continue

        with localconverter(converter):
            r_df = ro.conversion.get_conversion().py2rpy(scheme_df)
            r_levels = ro.StrVector(list(scheme["levels"]))
            result_r = ro.globalenv["run_pfizer_mixed_models"](r_df, r_levels)
            result_df = ro.conversion.get_conversion().rpy2py(result_r)

        if result_df is None or len(result_df) == 0:
            print(f"    [skip] CRPC-Pfizer mixed models ({scheme['name']}): no fits")
            continue

        result_df = pd.DataFrame(result_df).rename(
            columns={
                "Value": "estimate",
                "Std.Error": "std_error",
                "DF": "df",
                "t-value": "t_value",
                "p-value": "p_value",
            }
        )
        result_df["scheme"] = scheme["name"]
        interaction_df = result_df.loc[
            result_df["term"].astype(str).str.contains("visitC3D1:response")
        ].copy()
        if not interaction_df.empty:
            interaction_df["response_level"] = interaction_df["term"].astype(str).str.split("response", n=1).str[-1]
            interaction_df["response_level"] = interaction_df["response_level"].map(_pfizer_pretty_response)

        stem = os.path.join(out_dir, f"mixed_models_{scheme['name']}_{title_suffix}")
        full_path = f"{stem}_full.tsv"
        interaction_path = f"{stem}_interaction.tsv"
        if overwrite or not os.path.exists(full_path):
            result_df.to_csv(full_path, sep="\t", index=False)
        if not interaction_df.empty and (overwrite or not os.path.exists(interaction_path)):
            interaction_df.to_csv(interaction_path, sep="\t", index=False)


def _run_pfizer_delta_response_models(
    delta_df: pd.DataFrame,
    out_dir: str,
    *,
    overwrite: bool,
    title_suffix: str,
) -> None:
    """Model C3D1-SCN deltas as predictors of response categories."""
    from scipy.stats import kruskal, mannwhitneyu
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if delta_df.empty:
        return

    meta_columns = {"best_response_display", "response_binary"}
    marker_columns = [column for column in delta_df.columns if column not in meta_columns]
    for scheme in _pfizer_response_scheme_specs():
        rows: List[Dict[str, Any]] = []
        response_column = scheme["column"]
        for marker_name in marker_columns:
            working = delta_df[[marker_name, response_column]].dropna().copy()
            if working.empty or working[response_column].nunique() < 2:
                continue

            row: Dict[str, Any] = {
                "scheme": scheme["name"],
                "marker": marker_name,
                "n": int(len(working)),
            }
            if scheme["binary"]:
                progression_code = scheme["levels"][-1]
                y = (working[response_column] == progression_code).astype(int).values
                X = working[[marker_name]].astype(float).values
                if np.unique(y).size < 2:
                    continue
                model = LogisticRegression(solver="liblinear", random_state=42, max_iter=1000)
                model.fit(X, y)
                auc_val = roc_auc_score(y, model.predict_proba(X)[:, 1])
                neg_vals = working.loc[working[response_column] != progression_code, marker_name].astype(float).values
                pos_vals = working.loc[working[response_column] == progression_code, marker_name].astype(float).values
                _, mw_p = mannwhitneyu(neg_vals, pos_vals, alternative="two-sided")
                row.update(
                    {
                        "coefficient": float(model.coef_[0][0]),
                        "intercept": float(model.intercept_[0]),
                        "auc": float(auc_val),
                        "mannwhitney_p": float(mw_p),
                    }
                )
            else:
                grouped = [
                    working.loc[working[response_column] == level, marker_name].astype(float).values
                    for level in scheme["levels"]
                    if (working[response_column] == level).any()
                ]
                if len(grouped) < 2:
                    continue
                stat, kw_p = kruskal(*grouped)
                row.update(
                    {
                        "kruskal_statistic": float(stat),
                        "kruskal_p": float(kw_p),
                    }
                )
            rows.append(row)

        if not rows:
            continue

        result_df = pd.DataFrame(rows).sort_values("marker")
        out_path = os.path.join(out_dir, f"delta_response_models_{scheme['name']}_{title_suffix}.tsv")
        if overwrite or not os.path.exists(out_path):
            result_df.to_csv(out_path, sep="\t", index=False)


def _run_crpc_pfizer_analysis(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    *,
    mc_draws: Optional[List[pd.DataFrame]] = None,
) -> None:
    """Run longitudinal metadata-driven analyses for CRPC-Pfizer."""
    pfizer_meta = _load_pfizer_metadata(pred.columns)
    if pfizer_meta.empty:
        print("    [skip] CRPC-Pfizer metadata: no matched samples")
        return

    print(
        f"    CRPC-Pfizer: matched {len(pfizer_meta)} samples across "
        f"{pfizer_meta['patient_id'].nunique()} patients"
    )

    pfizer_out = ensure_dir(os.path.join(model_out, "Pfizer_Longitudinal"))
    pfizer_meta.to_csv(os.path.join(pfizer_out, "metadata_matched.tsv"), sep="\t")

    pred_subset = pred[pfizer_meta.index]
    ssgsea_raw = _run_pluvicto_ssgsea(pred_subset, model_out, overwrite, tag="_raw")
    shortlist_raw = _run_pluvicto_shortlist_scores(pred_subset, model_out, overwrite, tag="_raw")
    interesting_hallmarks_raw = _pfizer_run_limma_and_gsea(
        pred_subset,
        pfizer_meta,
        model_out,
        overwrite,
        tag_prefix="_raw",
    )
    marker_matrix_raw = _build_pfizer_marker_matrix(
        pred_subset,
        shortlist_raw,
        ssgsea_raw,
        interesting_hallmarks_raw,
    )

    analysis_specs: List[Tuple[str, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]] = [
        ("raw", marker_matrix_raw, shortlist_raw, ssgsea_raw, interesting_hallmarks_raw),
    ]

    if mc_draws is not None:
        mc_draws_subset = [draw[pfizer_meta.index] for draw in mc_draws]
        ssgsea_adj = _run_pluvicto_ssgsea_mc(mc_draws_subset, model_out, overwrite)
        shortlist_adj = _run_pluvicto_shortlist_scores_mc(mc_draws_subset, model_out, overwrite)
        interesting_hallmarks_adj = _pfizer_run_limma_and_gsea(
            pred_subset,
            pfizer_meta,
            model_out,
            overwrite,
            tag_prefix="_uncertainty-adjusted",
            mc_draws=mc_draws_subset,
        )
        marker_matrix_adj = _build_pfizer_marker_matrix(
            pred_subset,
            shortlist_adj,
            ssgsea_adj,
            interesting_hallmarks_adj,
        )
        analysis_specs.append(
            ("uncertainty-adjusted", marker_matrix_adj, shortlist_adj, ssgsea_adj, interesting_hallmarks_adj)
        )

    for analysis_name, marker_matrix, shortlist_scores, _, interesting_hallmarks in analysis_specs:
        if marker_matrix.empty:
            print(f"    [skip] CRPC-Pfizer {analysis_name}: no marker panel available")
            continue

        analysis_out = ensure_dir(os.path.join(pfizer_out, analysis_name))
        marker_matrix.to_csv(os.path.join(analysis_out, "marker_panel.tsv"), sep="\t")
        pd.DataFrame({"hallmark": interesting_hallmarks}).to_csv(
            os.path.join(analysis_out, "selected_hallmarks.tsv"),
            sep="\t",
            index=False,
        )

        spaghetti_out = ensure_dir(os.path.join(analysis_out, "spaghetti"))
        timepoint_out = ensure_dir(os.path.join(analysis_out, "timepoint_markers"))
        paired_out = ensure_dir(os.path.join(analysis_out, "paired_delta"))
        for marker_name in marker_matrix.index:
            stem = _pfizer_safe_name(marker_name)
            _maybe(
                _pfizer_plot_timepoint_distribution,
                overwrite,
                marker_name,
                marker_matrix.loc[marker_name],
                pfizer_meta,
                os.path.join(timepoint_out, stem),
                title_suffix=analysis_name,
            )
            _maybe(
                _pfizer_plot_spaghetti,
                overwrite,
                marker_name,
                marker_matrix.loc[marker_name],
                pfizer_meta,
                os.path.join(spaghetti_out, stem),
                title_suffix=analysis_name,
            )
            _maybe(
                _pfizer_plot_delta_slope,
                overwrite,
                marker_name,
                marker_matrix.loc[marker_name],
                pfizer_meta,
                os.path.join(paired_out, stem),
                title_suffix=analysis_name,
            )

        delta_df = _pfizer_build_delta_table(marker_matrix, pfizer_meta)
        if not delta_df.empty:
            delta_df.to_csv(os.path.join(analysis_out, "marker_deltas.tsv"), sep="\t")
            delta_plot_out = ensure_dir(os.path.join(analysis_out, "delta_by_response"))
            for scheme in _pfizer_response_scheme_specs():
                scheme_out = ensure_dir(os.path.join(delta_plot_out, scheme["name"]))
                for marker_name in marker_matrix.index:
                    stem = _pfizer_safe_name(marker_name)
                    _maybe(
                        _pfizer_plot_delta_distribution,
                        overwrite,
                        marker_name,
                        delta_df,
                        scheme["column"],
                        scheme["levels"],
                        os.path.join(scheme_out, stem),
                        title_suffix=analysis_name,
                    )
            _run_pfizer_delta_response_models(
                delta_df,
                analysis_out,
                overwrite=overwrite,
                title_suffix=analysis_name,
            )

        mixed_out = ensure_dir(os.path.join(analysis_out, "mixed_models"))
        _run_pfizer_mixed_models(
            marker_matrix,
            pfizer_meta,
            mixed_out,
            overwrite=overwrite,
            title_suffix=analysis_name,
        )

        if shortlist_scores is not None and {"NE.10", "ARG.10"}.issubset(shortlist_scores.index):
            state_out = ensure_dir(os.path.join(analysis_out, "state_space"))
            _maybe(
                _pfizer_plot_state_space,
                overwrite,
                shortlist_scores.loc["NE.10"],
                shortlist_scores.loc["ARG.10"],
                pfizer_meta,
                os.path.join(state_out, "ne10_vs_arg10"),
                title_suffix=analysis_name,
            )


def _run_discovery_cohort(
    cohort: str,
    subtype_label: Optional[str],
    out_dir: str,
    args: argparse.Namespace,
    *,
    interval_calibrator: Optional[RegimeCalibrator] = None,
) -> None:
    """Run Discovery_Patient analysis for a single cohort (no truth)."""
    cohort_out = ensure_dir(os.path.join(out_dir, cohort))

    label_name = subtype_label or cohort

    for model_name, path_pattern in _DISCOVERY_MODELS:
        path = PROJECT_ROOT / path_pattern.format(cohort=cohort)
        if not path.exists():
            print(f"    [skip] {model_name}: not found")
            continue

        # Load full MDN parameters for uncertainty propagation
        full_data = load_holdout_predictions_proteus_full(str(path))
        subtypes_df = _load_filtered_subtypes_for_reference(
            cohort,
            full_data["pred_wide"].columns,
            minimum_tfx=args.minimum_tfx,
            label_name=label_name,
            report_context=f"Discovery_Patient {cohort}",
        )
        if args.minimum_tfx > 0.0 and subtypes_df is not None:
            full_data = _filter_prediction_bundle_to_samples(full_data, subtypes_df.index)
        tfx_dict = subtypes_df["tfx"].to_dict() if subtypes_df is not None and "tfx" in subtypes_df.columns else None
        sub_dict = subtypes_df["subtype"].to_dict() if subtypes_df is not None and "subtype" in subtypes_df.columns else None
        analysis_data = _prepare_proteus_analysis_bundle(
            full_data,
            interval_calibrator=interval_calibrator,
            calibrate_cis=args.calibrate_cis,
        )
        pred = analysis_data["pred_wide"]
        ci_lo = analysis_data.get("ci_lo_wide")
        ci_hi = analysis_data.get("ci_hi_wide")

        # Build MC draws if MDN params available
        mc_draws: Optional[List[pd.DataFrame]] = None
        has_mdn = all(full_data.get(k) is not None for k in (
            "p0_wide", "pi_gamma_wide", "pi_normal_wide",
            "gamma_beta_wide", "normal_mean_wide", "normal_stdev_wide"))
        if has_mdn:
            mc_draws = mdn_monte_carlo_sample(
                full_data["p0_wide"], full_data["pi_gamma_wide"],
                full_data["pi_normal_wide"], full_data["gamma_beta_wide"],
                full_data["normal_mean_wide"], full_data["normal_stdev_wide"],
                M=args.montecarlo_draws, seed=42)

        # CRPC-Pluvicto: restrict to clinical FH samples (not FHL) at cycle 1
        if cohort == "CRPC-Pluvicto":
            keep = [
                s for s in pred.columns
                if s.startswith("FH") and not s.startswith("FHL")
                and s.endswith("C1")
            ]
            n_before = pred.shape[1]
            pred = pred[keep]
            if ci_lo is not None:
                ci_lo = ci_lo[[s for s in keep if s in ci_lo.columns]]
            if ci_hi is not None:
                ci_hi = ci_hi[[s for s in keep if s in ci_hi.columns]]
            if mc_draws is not None:
                mc_draws = [d[keep] for d in mc_draws]
            print(f"    Pluvicto filter: {n_before} → {pred.shape[1]} samples "
                  f"(FH* !FHL*, *C1)")

        model_out = cohort_out if model_name == "Proteus" else ensure_dir(os.path.join(cohort_out, model_name))
        print(f"    {model_name}: {pred.shape[0]} genes × {pred.shape[1]} samples")

        _maybe(sample_expression_kde, args.overwrite, pred,
               os.path.join(model_out, "expression_profiles"),
               subtypes_df=subtypes_df)

        for gs_name in ["PC-Phenotyping", "Targets_CRPC", "PAM50"]:
            resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
            if not gene_list_available(resolved):
                continue
            genes = get_gene_list(resolved)
            _maybe(
                single_heatmap, args.overwrite, pred, genes,
                os.path.join(model_out, f"heatmap_{gs_name}"),
                tfx_dict=tfx_dict, sub_dict=sub_dict,
            )

        pred.to_csv(os.path.join(model_out, "predictions_wide.tsv"), sep="\t")

        pam50_reference_subtypes = None
        pam50_reference_label = None
        if cohort == "BC-Ghana" and subtypes_df is not None and "subtype" in subtypes_df.columns:
            pam50_reference_subtypes = subtypes_df["subtype"]
            pam50_reference_label = "Metadata subtype"
        _run_pam50_analysis(
            pred,
            model_out,
            args.overwrite,
            cohort_label=cohort,
            reference_subtypes=pam50_reference_subtypes,
            reference_label=pam50_reference_label,
        )
        if cohort == "BC-Ghana" and mc_draws is not None:
            _run_pam50_analysis(
                pred,
                model_out,
                args.overwrite,
                cohort_label=cohort,
                tag="_uncertainty-adjusted",
                mc_draws=mc_draws,
                reference_subtypes=pam50_reference_subtypes,
                reference_label=pam50_reference_label,
            )

        # --- Limma DE + GSEA: ARPC vs NEPC for CRPC-UW ---
        if cohort == "CRPC-UW" and sub_dict is not None and subtypes_df is not None:
            # TFx series for uncertainty-adjusted limma
            tfx_for_limma = None
            if subtypes_df is not None and "tfx" in subtypes_df.columns:
                tfx_for_limma = subtypes_df["tfx"]

            # _raw: point predictions, TFx confounder, no precision weights
            de_raw = _run_limma_de(pred, subtypes_df, model_out,
                                   args.overwrite, tag="_raw",
                                   tfx_series=tfx_for_limma)
            if de_raw is not None:
                _run_gsea(de_raw, model_out, args.overwrite, tag="_raw")

            # _uncertainty-adjusted: TFx confounder + Monte Carlo aggregation
            if mc_draws is not None:
                de_adj = _run_limma_de_mc(
                    mc_draws,
                    pred,
                    subtypes_df,
                    model_out,
                    args.overwrite,
                    tag="_uncertainty-adjusted",
                    tfx_series=tfx_for_limma,
                )
                if de_adj is not None:
                    _run_gsea(de_adj, model_out, args.overwrite,
                              tag="_uncertainty-adjusted")

        # --- BC-Ghana: receptor expression boxplots ---
        if cohort == "BC-Ghana":
            _run_bc_ghana_analysis(pred, model_out, args.overwrite)

        # --- CRPC-Pluvicto: ssGSEA, survival, clinical correlations ---
        if cohort == "CRPC-Pluvicto":
            # _raw Hallmark + shortlist scores + survival
            ssgsea_raw = _run_pluvicto_ssgsea(
                pred, model_out, args.overwrite, tag="_raw")
            shortlist_raw = _run_pluvicto_shortlist_scores(
                pred, model_out, args.overwrite, tag="_raw")
            _run_pluvicto_survival(
                pred, ssgsea_raw, shortlist_raw, model_out, args.overwrite, args.minimum_tfx, tag="_raw")
            _run_pluvicto_clinical_correlations(
                pred,
                model_out,
                args.overwrite,
                shortlist_scores=shortlist_raw,
            )

            # _uncertainty-adjusted Hallmark + shortlist scores + survival
            if mc_draws is not None:
                ssgsea_mc = _run_pluvicto_ssgsea_mc(
                    mc_draws, model_out, args.overwrite)
                shortlist_mc = _run_pluvicto_shortlist_scores_mc(
                    mc_draws, model_out, args.overwrite)
                _run_pluvicto_survival(
                    pred, ssgsea_mc, shortlist_mc, model_out, args.overwrite,
                    args.minimum_tfx, tag="_uncertainty-adjusted")
                _run_pluvicto_clinical_correlations(
                    pred,
                    model_out,
                    args.overwrite,
                    shortlist_scores=shortlist_mc,
                    tag="_uncertainty-adjusted",
                )

        if cohort == "CRPC-Pfizer":
            _run_crpc_pfizer_analysis(
                pred,
                model_out,
                args.overwrite,
                mc_draws=mc_draws,
            )


def run_patient_discovery(args: argparse.Namespace) -> None:
    """Discovery cohort analysis: predictions only (no biopsy truth).

    Iterates over all configured discovery cohorts and generates heatmaps,
    expression profiles, and prediction exports for each available model.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Discovery_Patient")
    print("=" * 70)

    if _should_skip_section("Discovery_Patient", args):
        return

    out_dir = _section_results_dir("Discovery_Patient")
    _write_run_manifest("Discovery_Patient", out_dir, args)

    canonical_truth = _build_plca_truth(args.minimum_tfx)
    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
    interval_calibrator: Optional[RegimeCalibrator] = None
    if cv_bundle is not None and not canonical_truth.empty:
        cv_full = cv_bundle["cv_full"]
        if args.calibrate_cis and cv_full.get("ci_lo_wide") is not None and cv_full.get("ci_hi_wide") is not None:
            interval_calibrator = _fit_global_calibrator(canonical_truth, cv_full)
            if interval_calibrator is not None:
                print(f"  {interval_calibrator.summary()}")

    # Default: iterate over all discovery cohorts
    for cohort, subtype_label in _DISCOVERY_COHORTS:
        print(f"\n  ── Cohort: {cohort} ──")
        _run_discovery_cohort(
            cohort,
            subtype_label,
            out_dir,
            args,
            interval_calibrator=interval_calibrator,
        )

    print("\n  Done")


def _load_main_plca_cv_full(minimum_tfx: float = 0.0) -> Optional[Dict[str, Any]]:
    """Load and prefix-normalize the canonical PLCA CV outputs."""
    cv_dir = PROJECT_ROOT / VALIDATION_PATIENT_CV_RESULTS_DIR
    if not os.path.isdir(cv_dir):
        print(f"  [skip] PLCA CV results not found: {VALIDATION_PATIENT_CV_RESULTS_DIR}")
        return None

    fold_splits = _load_plca_fold_splits()
    cohort_map: Dict[str, str] = {}
    prefix_map: Dict[str, str] = {}
    for _, row in fold_splits.iterrows():
        prefixed = row["Sample"]
        cohort = row["Cohort"]
        short = str(strip_cohort_prefix(pd.Index([prefixed]), cohort)[0])
        cohort_map[str(prefixed)] = cohort
        cohort_map[short] = cohort
        prefix_map[short] = prefixed

    cv_full = load_cv_predictions_proteus_full(str(cv_dir))
    table_keys = (
        "pred_wide",
        "truth_wide",
        "ci_lo_wide",
        "ci_hi_wide",
        "p0_wide",
        "pi_gamma_wide",
        "pi_normal_wide",
        "gamma_beta_wide",
        "normal_mean_wide",
        "normal_stdev_wide",
    )

    fold_data = cv_full.get("fold_data") or {}
    for fd in fold_data.values():
        for key in table_keys:
            table = fd.get(key)
            if table is not None:
                _apply_prefix_map(table, prefix_map)

    def _merge_prefixed_fold_tables(key: str) -> Optional[pd.DataFrame]:
        parts = [fd[key] for _, fd in sorted(fold_data.items()) if fd.get(key) is not None]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return pd.concat(parts, axis=1)

    for key in table_keys:
        cv_full[key] = _merge_prefixed_fold_tables(key)

    if minimum_tfx > 0.0 and cv_full.get("pred_wide") is not None:
        allowed_samples_by_cohort: Dict[str, set[str]] = {}
        pred_columns = pd.Index([str(sample_name) for sample_name in cv_full["pred_wide"].columns])
        for cohort_name in VALIDATION_PATIENT_CV_COHORTS:
            cohort_samples = pd.Index([
                sample_name for sample_name in pred_columns if cohort_map.get(str(sample_name)) == cohort_name
            ])
            if len(cohort_samples) == 0:
                continue
            filtered_subtypes = _load_filtered_subtypes_for_reference(
                cohort_name,
                cohort_samples,
                minimum_tfx=minimum_tfx,
                report_context=f"PLCA CV {cohort_name}",
            )
            allowed_samples_by_cohort[cohort_name] = set(filtered_subtypes.index)

        keep_samples = [
            sample_name
            for sample_name in pred_columns
            if cohort_map.get(str(sample_name)) not in allowed_samples_by_cohort
            or str(sample_name) in allowed_samples_by_cohort[cohort_map[str(sample_name)]]
        ]
        keep_sample_set = set(keep_samples)
        for key in table_keys:
            cv_full[key] = _filter_wide_to_samples(cv_full.get(key), keep_samples)
        for fd in fold_data.values():
            for key in table_keys:
                fd[key] = _filter_wide_to_samples(fd.get(key), keep_samples)
        cohort_map = {
            sample_name: cohort_name
            for sample_name, cohort_name in cohort_map.items()
            if sample_name in keep_sample_set or cohort_name not in VALIDATION_PATIENT_CV_COHORTS
        }

    if (
        cv_full.get("p0_wide") is not None
        and cv_full.get("pi_gamma_wide") is not None
        and cv_full.get("pi_normal_wide") is not None
    ):
        cv_full["activity_odds_ratio_wide"] = mdn_activity_odds_ratio_wide(
            cv_full["p0_wide"],
            cv_full["pi_gamma_wide"],
            cv_full["pi_normal_wide"],
        )
    else:
        cv_full["activity_odds_ratio_wide"] = None

    required_mdn_keys = (
        "p0_wide",
        "pi_gamma_wide",
        "pi_normal_wide",
        "gamma_beta_wide",
        "normal_mean_wide",
        "normal_stdev_wide",
    )
    if all(cv_full.get(key) is not None for key in required_mdn_keys):
        cv_full["active_confidence_wide"] = mdn_on_probability_wide(
            cv_full["p0_wide"],
            cv_full["pi_gamma_wide"],
            cv_full["pi_normal_wide"],
            cv_full["gamma_beta_wide"],
            cv_full["normal_mean_wide"],
            cv_full["normal_stdev_wide"],
        )
    else:
        cv_full["active_confidence_wide"] = None

    sample_fold: Dict[str, int] = {}
    for fold, fd in sorted(fold_data.items()):
        pred_wide = fd.get("pred_wide")
        if pred_wide is None:
            continue
        for sample in pred_wide.columns:
            sample_fold[str(sample)] = int(fold)
    cv_full["sample_fold"] = sample_fold

    return {
        "cv_full": cv_full,
        "cohort_map": cohort_map,
        "prefix_map": prefix_map,
    }


def run_calibration(args: argparse.Namespace) -> None:
    """Standalone calibration analysis using pooled truth-bearing OOF data."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Calibration")
    print("=" * 70)

    if _should_skip_section("Calibration", args):
        return

    out_dir = _section_results_dir("Calibration")
    _write_run_manifest("Calibration", out_dir, args)
    canonical_truth = _build_plca_truth(args.minimum_tfx)
    if canonical_truth.empty:
        print("  [skip] No pooled PLCA truth available")
        return

    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
    if cv_bundle is None:
        return

    cv_full = cv_bundle["cv_full"]
    interval_calibrator = _fit_global_calibrator(
        canonical_truth,
        cv_full,
        overwrite=args.overwrite,
        out_dir=out_dir,
    )

    if interval_calibrator is None:
        print("  [skip] Unable to fit any calibration model")
        return

    if interval_calibrator is not None:
        print(f"  {interval_calibrator.summary()}")
        _run_cv_crossfit_calibration(cv_full, out_dir, overwrite=args.overwrite)

    activity_threshold_summary = _compute_patient_cv_activity_thresholds(
        cv_full,
        cv_bundle["cohort_map"],
    )
    if activity_threshold_summary is not None:
        on_curve = activity_threshold_summary["on_curve"]
        off_curve = activity_threshold_summary["off_curve"]
        threshold_rows = pd.DataFrame(
            [
                {
                    "target_ppv": float(on_curve["target_ppv"]),
                    "upper_threshold": float(activity_threshold_summary["upper_threshold"]),
                    "upper_target_met": bool(on_curve["target_met"]),
                    "upper_selected_coverage": float(on_curve["selected_coverage"]),
                    "upper_selected_ppv": float(on_curve["selected_ppv"]),
                    "upper_selected_fdr": float(on_curve["selected_fdr"]),
                    "upper_selected_n_predicted_on": int(on_curve["selected_n_predicted_on"]),
                    "target_npv": float(off_curve["target_npv"]),
                    "lower_threshold": float(activity_threshold_summary["lower_threshold"]),
                    "lower_target_met": bool(off_curve["target_met"]),
                    "lower_selected_coverage": float(off_curve["selected_coverage"]),
                    "lower_selected_npv": float(off_curve["selected_npv"]),
                    "lower_selected_for": float(off_curve["selected_for"]),
                    "lower_selected_n_predicted_off": int(off_curve["selected_n_predicted_off"]),
                    "n_total": int(on_curve["n_total"]),
                    "n_true_on": int(on_curve["n_true_on"]),
                    "n_true_off": int(off_curve["n_true_off"]),
                }
            ]
        )
        threshold_rows.to_csv(
            os.path.join(out_dir, "PatientCV_ActivityOddsRatioThresholds.tsv"),
            sep="\t",
            index=False,
        )

        active_curve_rows = pd.DataFrame(
            {
                "activity_odds_ratio_threshold": on_curve["threshold"],
                "coverage": on_curve["coverage"],
                "ppv": on_curve["ppv"],
                "fdr": on_curve["fdr"],
                "n_predicted_on": on_curve["n_predicted_on"],
            }
        )
        active_curve_rows.to_csv(
            os.path.join(out_dir, "ActiveConfidenceCurve_PatientCV_OOF.tsv"),
            sep="\t",
            index=False,
        )
        inactive_curve_rows = pd.DataFrame(
            {
                "activity_odds_ratio_threshold": off_curve["threshold"],
                "coverage": off_curve["coverage"],
                "npv": off_curve["npv"],
                "for": off_curve["for"],
                "n_predicted_off": off_curve["n_predicted_off"],
            }
        )
        inactive_curve_rows.to_csv(
            os.path.join(out_dir, "InactiveConfidenceCurve_PatientCV_OOF.tsv"),
            sep="\t",
            index=False,
        )

        active_confidence_threshold_plot(
            {"Proteus": on_curve},
            os.path.join(out_dir, "ActiveConfidenceThreshold_PatientCV_OOF"),
            title="Patient CV OOF Active Threshold (AOR)",
            model_order=["Proteus"],
        )
        inactive_confidence_threshold_plot(
            {"Proteus": off_curve},
            os.path.join(out_dir, "InactiveConfidenceThreshold_PatientCV_OOF"),
            title="Patient CV OOF Inactive Threshold (AOR)",
            model_order=["Proteus"],
        )
        print(
            "  Patient CV OOF activity odds ratio thresholds: "
            f"UB={activity_threshold_summary['upper_threshold']:.3f} "
            f"(PPV={on_curve['selected_ppv']:.3f}, coverage={on_curve['selected_coverage']:.3f}), "
            f"LB={activity_threshold_summary['lower_threshold']:.3f} "
            f"(NPV={off_curve['selected_npv']:.3f}, coverage={off_curve['selected_coverage']:.3f})"
        )

    truth_pool, ho_preds = _load_validation_pdx_holdout_data(canonical_truth)
    holdout_pred_parts: List[pd.DataFrame] = []
    holdout_truth_parts: List[pd.DataFrame] = []
    holdout_lo_parts: List[pd.DataFrame] = []
    holdout_hi_parts: List[pd.DataFrame] = []
    holdout_mdn_parts: Dict[str, List[pd.DataFrame]] = {key: [] for key in _MDN_PARAM_KEYS}
    for cohort_name in VALIDATION_PDX_COHORTS:
        ho_data = ho_preds.get(cohort_name)
        if ho_data is None or ho_data.get("ci_lo_wide") is None or ho_data.get("ci_hi_wide") is None:
            continue
        holdout_pred = ho_data["pred_wide"]
        holdout_truth = truth_pool.reindex(index=holdout_pred.index, columns=holdout_pred.columns)
        holdout_pred_parts.append(holdout_pred)
        holdout_truth_parts.append(holdout_truth)
        holdout_lo_parts.append(ho_data["ci_lo_wide"])
        holdout_hi_parts.append(ho_data["ci_hi_wide"])
        for key in _MDN_PARAM_KEYS:
            table = ho_data.get(key)
            if table is not None:
                holdout_mdn_parts[key].append(table)

    pooled_holdout_pred = _concat_common_gene_frames(holdout_pred_parts)
    pooled_holdout_truth = _concat_common_gene_frames(holdout_truth_parts)
    pooled_holdout_lo = _concat_common_gene_frames(holdout_lo_parts)
    pooled_holdout_hi = _concat_common_gene_frames(holdout_hi_parts)
    pooled_holdout_mdn = {
        key: _concat_common_gene_frames(parts)
        for key, parts in holdout_mdn_parts.items()
    }
    if (
        pooled_holdout_pred is not None
        and pooled_holdout_truth is not None
        and pooled_holdout_lo is not None
        and pooled_holdout_hi is not None
        and interval_calibrator is not None
    ):
        holdout_out = ensure_dir(os.path.join(out_dir, "PDX_Holdout"))
        _run_calibration_evaluation(
            pooled_holdout_truth,
            pooled_holdout_pred,
            pooled_holdout_lo,
            pooled_holdout_hi,
            interval_calibrator,
            "Validation_PDX",
            holdout_out,
            overwrite=args.overwrite,
        )
        _save_pit_diagnostics(
            pooled_holdout_truth,
            pooled_holdout_mdn,
            holdout_out,
            overwrite=args.overwrite,
            title="MDN PIT — Validation_PDX Holdout",
        )


def run_validation_pdx(args: argparse.Namespace) -> None:
    """Validation_PDX analysis using PLCA main-model CV + true holdout PDX samples."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Validation_PDX")
    print("=" * 70)

    if _should_skip_section("Validation_PDX", args):
        return

    out_dir = _section_results_dir("Validation_PDX")
    _write_run_manifest("Validation_PDX", out_dir, args)
    n_workers = args.n_workers

    canonical_truth = _build_plca_truth(args.minimum_tfx)
    if canonical_truth.empty:
        print("  [skip] No canonical PLCA truth available")
        return

    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
    if cv_bundle is None:
        return

    cv_full = cv_bundle["cv_full"]
    cohort_map = cv_bundle["cohort_map"]
    proteus_pred = cv_full["pred_wide"]
    proteus_ci_lo = cv_full.get("ci_lo_wide")
    proteus_ci_hi = cv_full.get("ci_hi_wide")

    def _split_by_cohort(
        pred: pd.DataFrame,
        truth: pd.DataFrame,
        ci_lo: Optional[pd.DataFrame],
        ci_hi: Optional[pd.DataFrame],
        cohort: str,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        samples = sorted(s for s in pred.columns if cohort_map.get(s) == cohort)
        if not samples:
            return None, None, None, None
        pred_c = pred[[s for s in samples if s in pred.columns]]
        truth_c = truth[[s for s in samples if s in truth.columns]]
        ci_lo_c = ci_lo[[s for s in samples if s in ci_lo.columns]] if ci_lo is not None else None
        ci_hi_c = ci_hi[[s for s in samples if s in ci_hi.columns]] if ci_hi is not None else None
        return pred_c, truth_c, ci_lo_c, ci_hi_c

    interval_calibrator: Optional[RegimeCalibrator] = None
    if args.calibrate_cis and proteus_ci_lo is not None and proteus_ci_hi is not None:
        interval_calibrator = _fit_global_calibrator(canonical_truth, cv_full)
        if interval_calibrator is not None:
            print(f"  {interval_calibrator.summary()}")
            crossfit_ci_lo, crossfit_ci_hi = _run_cv_crossfit_calibration(cv_full, None)
            if crossfit_ci_lo is not None and crossfit_ci_hi is not None:
                proteus_ci_lo = crossfit_ci_lo
                proteus_ci_hi = crossfit_ci_hi

    reliability_model: Optional[ReliabilityModel] = None
    if proteus_ci_lo is not None and cv_full.get("fold_data"):
        reliability_model = ReliabilityModel.fit(
            cv_full["fold_data"],
            canonical_truth,
            ridge_alpha=1.0,
        )
        print(f"  {reliability_model.summary()}")
        _maybe(
            plot_reliability_weights,
            args.overwrite,
            reliability_model.feature_columns,
            reliability_model.weights,
            os.path.join(out_dir, "reliability_feature_weights"),
            title="Reliability Model Feature Weights (OOF)",
        )

    truth_pool, raw_ho_preds = _load_validation_pdx_holdout_data(canonical_truth)
    analysis_ho_preds: Dict[str, Dict[str, Any]] = {}
    for cohort_name, ho_data in raw_ho_preds.items():
        analysis_ho_preds[cohort_name] = _prepare_proteus_analysis_bundle(
            ho_data,
            interval_calibrator=interval_calibrator,
            calibrate_cis=args.calibrate_cis,
        )

    all_summaries: List[Dict[str, Any]] = []
    crosscohort_data: List[Dict[str, Any]] = []
    for cohort_name in VALIDATION_PDX_COHORTS:
        print(f"\n  ── Cohort: {cohort_name} ──")
        cv_pred_c, _cv_truth, cv_ci_lo_c, cv_ci_hi_c = _split_by_cohort(
            proteus_pred,
            canonical_truth,
            proteus_ci_lo,
            proteus_ci_hi,
            cohort_name,
        )
        if cv_pred_c is None or cv_pred_c.shape[1] < 2:
            print(f"    [skip] Too few CV samples for {cohort_name}")
            continue

        cohort_out = ensure_dir(os.path.join(out_dir, cohort_name))
        ho_data = analysis_ho_preds.get(cohort_name)
        raw_ho_data = raw_ho_preds.get(cohort_name)
        ho_pred_c = ho_ci_lo_c_only = ho_ci_hi_c_only = None
        ho_samples: List[str] = []
        if ho_data is not None:
            ho_pred_c = ho_data["pred_wide"]
            ho_ci_lo_c_only = ho_data.get("ci_lo_wide")
            ho_ci_hi_c_only = ho_data.get("ci_hi_wide")
            ho_samples = sorted(ho_pred_c.columns)

        sample_status: Dict[str, str] = {s: "validation" for s in cv_pred_c.columns}
        for sample in ho_samples:
            sample_status[sample] = "holdout"

        subtypes_df = None
        subtypes_path = PROJECT_ROOT / "labels" / f"{cohort_name}_TFX_Subtype.tsv"
        if subtypes_path.exists():
            subtypes_df = load_subtypes(cohort_name)
            subtypes_df = subtypes_df.copy()
            subtypes_df["status"] = subtypes_df.index.map(lambda s: sample_status.get(s, "unknown"))

        if ho_pred_c is not None:
            common_genes = cv_pred_c.index.intersection(ho_pred_c.index)
            combined_pred = pd.concat([cv_pred_c.loc[common_genes], ho_pred_c.loc[common_genes]], axis=1)
            combined_ci_lo = None
            combined_ci_hi = None
            if cv_ci_lo_c is not None and ho_ci_lo_c_only is not None:
                combined_ci_lo = pd.concat([cv_ci_lo_c.loc[common_genes], ho_ci_lo_c_only.loc[common_genes]], axis=1)
                combined_ci_hi = pd.concat([cv_ci_hi_c.loc[common_genes], ho_ci_hi_c_only.loc[common_genes]], axis=1)
        else:
            combined_pred = cv_pred_c
            combined_ci_lo = cv_ci_lo_c
            combined_ci_hi = cv_ci_hi_c

        combined_mdn_tables: Dict[str, Optional[pd.DataFrame]] = {}
        for key in _MDN_PARAM_KEYS:
            cv_table = cv_full.get(key)
            cv_table_c = None
            if cv_table is not None:
                cv_samples = [sample for sample in cv_pred_c.columns if sample in cv_table.columns]
                if cv_samples:
                    cv_table_c = cv_table[cv_samples]
            ho_table = raw_ho_data.get(key) if raw_ho_data is not None else None
            if cv_table_c is not None and ho_table is not None:
                mdn_genes = cv_table_c.index.intersection(ho_table.index)
                combined_mdn_tables[key] = pd.concat(
                    [cv_table_c.loc[mdn_genes], ho_table.loc[mdn_genes]],
                    axis=1,
                )
            elif cv_table_c is not None:
                combined_mdn_tables[key] = cv_table_c
            else:
                combined_mdn_tables[key] = ho_table

        cv_truth_c = canonical_truth.reindex(index=cv_pred_c.index, columns=cv_pred_c.columns)
        combined_truth = truth_pool.reindex(index=combined_pred.index, columns=combined_pred.columns)

        cohort_cfg = _PLCA_CV_COHORT_CONFIG[cohort_name]
        cohort_summaries: List[Dict[str, Any]] = []
        cv_summary = cohort_summary_table(
            cv_truth_c,
            cv_pred_c,
            label=f"CV ({cohort_name})",
            n_workers=n_workers,
        )
        _attach_median_point_summary(
            cv_summary,
            cv_truth_c,
            {key: cv_full.get(key) for key in _MDN_PARAM_KEYS},
            n_workers=n_workers,
        )
        cohort_summaries.append(cv_summary)
        all_summaries.append(cv_summary)

        if ho_pred_c is not None and len(ho_samples) >= 2:
            ho_truth_c = truth_pool.reindex(index=ho_pred_c.index, columns=ho_pred_c.columns)
            ho_summary = cohort_summary_table(
                ho_truth_c,
                ho_pred_c,
                label=f"Holdout ({cohort_name})",
                n_workers=n_workers,
            )
            _attach_median_point_summary(
                ho_summary,
                ho_truth_c,
                {key: raw_ho_data.get(key) for key in _MDN_PARAM_KEYS},
                n_workers=n_workers,
            )
            cohort_summaries.append(ho_summary)
            all_summaries.append(ho_summary)

        combined_summary = cohort_summary_table(
            combined_truth,
            combined_pred,
            label=f"Combined ({cohort_name})",
            n_workers=n_workers,
        )
        _attach_median_point_summary(
            combined_summary,
            combined_truth,
            combined_mdn_tables,
            n_workers=n_workers,
        )
        cohort_summaries.append(combined_summary)
        all_summaries.append(combined_summary)

        _run_cv_suite(
            combined_pred,
            combined_truth,
            f"Proteus ({cohort_name})",
            cohort_out,
            ci_lo=combined_ci_lo,
            ci_hi=combined_ci_hi,
            goi=cohort_cfg["goi"],
            subtypes_df=subtypes_df,
            heatmap_gene_sets=cohort_cfg["heatmap_gene_sets"],
            gene_scatter_sets=cohort_cfg.get("gene_scatter_sets", []),
            gene_met=combined_summary["_gene_metrics"],
            sample_met=combined_summary["_sample_metrics"],
            n_workers=n_workers,
            overwrite=args.overwrite,
            scatter_ci=args.scatter_ci,
            dim_ambiguous=args.dim_ambiguous,
        )

        if raw_ho_data is not None and reliability_model is not None:
            _run_reliability_evaluation(
                raw_ho_data,
                canonical_truth,
                truth_pool,
                reliability_model,
                cohort_name,
                cohort_out,
                overwrite=args.overwrite,
            )

        tfx_dict = subtypes_df["tfx"].to_dict() if subtypes_df is not None and "tfx" in subtypes_df.columns else None
        crosscohort_data.append(
            {
                "cohort": cohort_name,
                "truth_wide": combined_truth,
                "pred_wide": combined_pred,
                "ci_lo": combined_ci_lo,
                "ci_hi": combined_ci_hi,
                "subtypes_df": subtypes_df,
                "tfx_dict": tfx_dict,
                "status": "mixed",
                **combined_mdn_tables,
            }
        )

        _save_summary_tsv(cohort_summaries, os.path.join(cohort_out, "summary_metrics.tsv"))

    if all_summaries:
        print("\n" + "─" * 70)
        print_summary_table(all_summaries)
        _save_summary_tsv(all_summaries, os.path.join(out_dir, "summary_metrics.tsv"))

    if len(crosscohort_data) >= 2:
        _run_collected_validation(
            crosscohort_data,
            out_dir,
            args,
            label="CrossCohort",
            cohort_order=VALIDATION_PDX_COHORTS,
            target_sets=["Targets_CRPC", "Targets_SCLC"],
            include_ne10_roc=False,
        )


def run_validation_patient(args: argparse.Namespace) -> None:
    """Validation_Patient analysis using PLCA main-model holdout and OOF CV data."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Validation_Patient")
    print("=" * 70)

    if _should_skip_section("Validation_Patient", args):
        return

    out_dir = _section_results_dir("Validation_Patient")
    _write_run_manifest("Validation_Patient", out_dir, args)
    cohort_data: List[Dict[str, Any]] = []

    canonical_truth = _build_plca_truth(args.minimum_tfx)
    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
    interval_calibrator: Optional[RegimeCalibrator] = None

    cv_full = None
    cv_pred = None
    cv_ci_lo = None
    cv_ci_hi = None
    cohort_map_cv = None
    activity_threshold_summary = None
    if cv_bundle is not None:
        cv_full = cv_bundle["cv_full"]
        cohort_map_cv = cv_bundle["cohort_map"]
        cv_pred = cv_full["pred_wide"]
        cv_ci_lo = cv_full.get("ci_lo_wide")
        cv_ci_hi = cv_full.get("ci_hi_wide")
        activity_threshold_summary = _compute_patient_cv_activity_thresholds(cv_full, cohort_map_cv)

        if not canonical_truth.empty and args.calibrate_cis and cv_ci_lo is not None and cv_ci_hi is not None:
            interval_calibrator = _fit_global_calibrator(canonical_truth, cv_full)
            if interval_calibrator is not None:
                print(f"  {interval_calibrator.summary()}")
                crossfit_ci_lo, crossfit_ci_hi = _run_cv_crossfit_calibration(cv_full, None)
                if crossfit_ci_lo is not None and crossfit_ci_hi is not None:
                    cv_ci_lo = crossfit_ci_lo
                    cv_ci_hi = crossfit_ci_hi

    for cohort in VALIDATION_PATIENT_HOLDOUT_COHORTS:
        print(f"\n  ── Cohort: {cohort} (holdout) ──")
        result = _run_patient_validation_cohort(
            cohort,
            VALIDATION_PATIENT_HOLDOUT_PATH_TEMPLATE,
            out_dir,
            args,
            interval_calibrator=interval_calibrator,
            activity_threshold_summary=activity_threshold_summary,
        )
        if result is not None:
            cohort_data.append(result)

    if cv_bundle is not None and cv_full is not None and cv_pred is not None and cohort_map_cv is not None:
        for cohort in VALIDATION_PATIENT_CV_COHORTS:
            print(f"\n  ── Cohort: {cohort} (CV) ──")
            result = _run_patient_validation_cv_cohort(
                cohort,
                cv_pred,
                cv_ci_lo,
                cv_ci_hi,
                cv_full,
                cohort_map_cv,
                out_dir,
                args,
                activity_threshold_summary=activity_threshold_summary,
            )
            if result is not None:
                cohort_data.append(result)

    if len(cohort_data) >= 2:
        _run_collected_validation(
            cohort_data,
            out_dir,
            args,
            label="CrossCohort",
            activity_threshold_summary=activity_threshold_summary,
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_prefix_map(truth_columns: pd.Index, cohort: str) -> Dict[str, str]:
    """Map short (prefix-stripped) sample names → canonical truth column names.

    For cohorts using prefixed names (e.g. ``CRPC-RA_15-023``), this maps
    ``'15-023' → 'CRPC-RA_15-023'``.  For cohorts whose truth labels are
    already short (e.g. SCLC ``'L-0141'``), the map is identity.
    """
    stripped = strip_cohort_prefix(truth_columns, cohort)
    return dict(zip(stripped, truth_columns))


def _apply_prefix_map(
    df: Optional[pd.DataFrame], pmap: Dict[str, str],
) -> Optional[pd.DataFrame]:
    """Rename DataFrame columns using a short→prefixed mapping (in-place)."""
    if df is None:
        return None
    df.columns = pd.Index([pmap.get(str(s), str(s)) for s in df.columns])
    return df


def _exclude_pht(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    ci_lo: Optional[pd.DataFrame] = None,
    ci_hi: Optional[pd.DataFrame] = None,
) -> Tuple:
    """Drop PHT admix samples (columns starting with 'PHT') from all wide DataFrames."""
    keep = [s for s in pred.columns if not str(s).startswith("PHT")]
    n_dropped = len(pred.columns) - len(keep)
    if n_dropped:
        print(f"  Excluded {n_dropped} PHT admix samples")
    pred = pred[keep]
    keep_t = [s for s in truth.columns if not str(s).startswith("PHT")]
    truth = truth[keep_t]
    if ci_lo is not None:
        ci_lo = ci_lo[[s for s in ci_lo.columns if not str(s).startswith("PHT")]]
    if ci_hi is not None:
        ci_hi = ci_hi[[s for s in ci_hi.columns if not str(s).startswith("PHT")]]
    return pred, truth, ci_lo, ci_hi


def _maybe(fn, overwrite: bool, *args, **kwargs) -> Any:
    """Call ``fn`` only if overwrite is True or the output PDF doesn't exist.

    Assumes the second positional arg (after overwrite) or the first string
    that looks like an output path is the file stem.
    """
    if overwrite:
        return fn(*args, **kwargs)

    # Try to find the output path — it's the first string arg that looks like a path
    for a in args:
        if isinstance(a, str) and ("/" in a or "\\" in a):
            if os.path.exists(f"{a}.pdf"):
                return None
            break
    return fn(*args, **kwargs)


def _save_summary_tsv(summaries: List[Dict[str, Any]], path: str) -> None:
    """Save cohort summaries to a TSV, excluding internal DataFrames."""
    rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if not k.startswith("_")}
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print("  Summary table saved")


# ---------------------------------------------------------------------------
# Analysis group: Benchmarking (PHS holdout — model comparison)
# ---------------------------------------------------------------------------

# TFx/depth grid and models — per README spec
_BENCH_TFX_ORDER = ["TF0.50", "TF0.30", "TF0.10", "TF0.05", "TF0.03", "TF0.01"]
_BENCH_DEPTH_ORDER = ["10x", "5x"]
_BENCH_EXCLUDED_DEPTHS = {"1x", "30x", "50x"}


def _bench_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process-pool worker for one (tfx, depth, model) benchmark combo."""
    m = compute_subset_metrics(
        payload["tw"], payload["pw"], n_workers=1, var_map=payload["var_map"],
    )
    return {
        "tfx": payload["tfx"],
        "depth": payload["depth"],
        "model": payload["model"],
        "m": m,
    }


def _write_synthetic_benchmark_view(
    metrics_table: pd.DataFrame,
    gene_rho_table: pd.DataFrame,
    sample_rho_table: pd.DataFrame,
    out_dir: str,
    view_name: str,
    model_order: Sequence[str],
    overwrite: bool,
    var_map: Dict[str, float],
) -> None:
    """Write one filtered synthetic benchmarking view (COMPLETE or SUBSET)."""
    view_out = ensure_dir(os.path.join(out_dir, view_name))
    metrics_view = metrics_table[metrics_table["model"].isin(model_order)].copy()
    gene_rho_view = gene_rho_table[gene_rho_table["model"].isin(model_order)].copy()
    sample_rho_view = sample_rho_table[sample_rho_table["model"].isin(model_order)].copy()

    if metrics_view.empty:
        return

    metrics_view.to_csv(
        os.path.join(view_out, "summary_benchmarking.tsv"), sep="\t", index=False,
    )

    for metric_col, ylabel in [
        ("gvw_rho", "GVW-ρ"),
        ("mean_gene_rho", "Mean Gene ρ"),
        ("rmse", "Mean Gene RMSE"),
        ("mean_top10pct_gene_rho", "Mean ρ (Top 10%)"),
        ("n_genes_rho_ge070", "Genes with ρ≥0.70"),
    ]:
        _maybe(
            benchmark_grid_boxplots,
            overwrite,
            metrics_view,
            os.path.join(view_out, f"benchmark_grid_{metric_col}"),
            metric_col=metric_col,
            ylabel=ylabel,
            title=f"Benchmarking Synthetic ({view_name}) — {ylabel}",
            tfx_order=_BENCH_TFX_ORDER,
            depth_order=_BENCH_DEPTH_ORDER,
            model_order=model_order,
            naive_model="Naive",
        )

    if not sample_rho_view.empty:
        _maybe(
            benchmark_grid_boxplots,
            overwrite,
            sample_rho_view,
            os.path.join(view_out, "benchmark_grid_sample_rho"),
            metric_col="spearman_rho",
            ylabel="Sample ρ",
            title=f"Benchmarking Synthetic ({view_name}) — Per-Sample Spearman ρ",
            tfx_order=_BENCH_TFX_ORDER,
            depth_order=_BENCH_DEPTH_ORDER,
            model_order=model_order,
            naive_model="Naive",
        )

    if not gene_rho_view.empty:
        _maybe(
            benchmark_grid_histograms,
            overwrite,
            gene_rho_view,
            os.path.join(view_out, "benchmark_grid_gene_rho_hist"),
            tfx_order=_BENCH_TFX_ORDER,
            depth_order=_BENCH_DEPTH_ORDER,
            model_order=model_order,
        )

        var_s = pd.Series(var_map)
        n_top = max(1, int(np.ceil(len(var_s) * 0.10)))
        top10_genes = set(var_s.nlargest(n_top).index)
        top10_rho = gene_rho_view[gene_rho_view["gene"].isin(top10_genes)]
        if not top10_rho.empty:
            _maybe(
                benchmark_grid_histograms,
                overwrite,
                top10_rho,
                os.path.join(view_out, "benchmark_grid_gene_rho_hist_top10pct"),
                tfx_order=_BENCH_TFX_ORDER,
                depth_order=_BENCH_DEPTH_ORDER,
                model_order=model_order,
                title="Per-Gene Spearman ρ (Top 10% Var.)",
            )

    if not metrics_view.empty:
        for depth_val in ["10x", "5x"]:
            _maybe(
                benchmark_depth_lines,
                overwrite,
                metrics_view,
                os.path.join(view_out, f"benchmark_lines_{depth_val}"),
                depth=depth_val,
                model_order=model_order,
            )


def _run_synthetic_hallmark_benchmark(
    truth_wide: pd.DataFrame,
    model_preds: Dict[str, pd.DataFrame],
    proteus_bundles: Dict[str, Dict[str, Any]],
    meta_df: pd.DataFrame,
    out_dir: str,
    *,
    overwrite: bool,
    montecarlo_draws: int,
) -> None:
    """Benchmark Hallmark ssGSEA score recapitulation for the SUBSET model view."""
    if not _GSEA_HALLMARK_GMT.exists():
        print(f"  [skip] Hallmark benchmark: GMT not found: {_GSEA_HALLMARK_GMT}")
        return

    subset_models = [model_name for model_name in BENCHMARK_SUBSET_MODELS if model_name in model_preds]
    if not subset_models:
        print("  [skip] Hallmark benchmark: no SUBSET models available")
        return

    all_samples = sorted(meta_df["sample"].unique())
    hallmark_sets = _load_gmt(_GSEA_HALLMARK_GMT)
    truth_scores = _compute_ssgsea_score_matrix(
        truth_wide[[sample for sample in all_samples if sample in truth_wide.columns]],
        hallmark_sets,
        min_size=10,
    )
    if truth_scores.empty:
        print("  [skip] Hallmark benchmark: no truth Hallmark scores could be computed")
        return

    pred_scores_by_model: Dict[str, pd.DataFrame] = {}
    hallmark_model_order: List[str] = []
    for model_name in subset_models:
        pw = model_preds[model_name]
        common_samples = [sample for sample in all_samples if sample in pw.columns]
        if not common_samples:
            continue
        score_df = _compute_ssgsea_score_matrix(pw[common_samples], hallmark_sets, min_size=10)
        if not score_df.empty:
            pred_scores_by_model[model_name] = score_df
            hallmark_model_order.append(model_name)

    proteus_bundle = proteus_bundles.get("Proteus")
    if proteus_bundle is not None and all(proteus_bundle.get(key) is not None for key in _MDN_PARAM_KEYS):
        common_samples = [sample for sample in all_samples if sample in proteus_bundle["pred_wide"].columns]
        if common_samples:
            print(f"  Hallmark ssGSEA MC: Proteus ({len(common_samples)} samples)")
            mc_draws = mdn_monte_carlo_sample(
                proteus_bundle["p0_wide"][common_samples],
                proteus_bundle["pi_gamma_wide"][common_samples],
                proteus_bundle["pi_normal_wide"][common_samples],
                proteus_bundle["gamma_beta_wide"][common_samples],
                proteus_bundle["normal_mean_wide"][common_samples],
                proteus_bundle["normal_stdev_wide"][common_samples],
                M=montecarlo_draws,
                seed=42,
            )
            adjusted_scores, _, _ = _aggregate_ssgsea_score_matrices(
                mc_draws,
                hallmark_sets,
                min_size=10,
            )
            if adjusted_scores is not None and not adjusted_scores.empty:
                adjusted_label = "Proteus (adjusted)"
                pred_scores_by_model[adjusted_label] = adjusted_scores
                hallmark_model_order.append(adjusted_label)

    if not pred_scores_by_model:
        print("  [skip] Hallmark benchmark: no model Hallmark scores could be computed")
        return

    hallmark_out = ensure_dir(os.path.join(out_dir, "SUBSET", "HallmarkScores"))
    summary_rows: List[Dict[str, Any]] = []

    for tfx in _BENCH_TFX_ORDER:
        for depth in _BENCH_DEPTH_ORDER:
            cell_samples = sorted(
                meta_df[(meta_df["tfx"] == tfx) & (meta_df["depth"] == depth)]["sample"]
            )
            if not cell_samples:
                continue
            truth_cell = truth_scores[[sample for sample in cell_samples if sample in truth_scores.columns]]
            if truth_cell.shape[1] < 3:
                continue

            for model_name in hallmark_model_order:
                pred_scores = pred_scores_by_model.get(model_name)
                if pred_scores is None:
                    continue
                common_samples = [sample for sample in cell_samples if sample in pred_scores.columns and sample in truth_cell.columns]
                if len(common_samples) < 3:
                    continue
                metric_dict = compute_subset_metrics(
                    truth_cell[common_samples],
                    pred_scores[common_samples],
                    n_workers=1,
                )
                summary_rows.append(
                    {
                        "tfx": tfx,
                        "depth": depth,
                        "model": model_name,
                        "mean_pathway_rho": metric_dict["mean_gene_rho"],
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("  [skip] Hallmark benchmark: no evaluable TFx/depth cells")
        return

    summary_df.to_csv(os.path.join(hallmark_out, "summary_hallmark_scores.tsv"), sep="\t", index=False)
    _maybe(
        benchmark_grid_boxplots,
        overwrite,
        summary_df,
        os.path.join(hallmark_out, "hallmark_grid_mean_pathway_rho"),
        metric_col="mean_pathway_rho",
        ylabel="Mean Hallmark ρ",
        title="Benchmarking Synthetic (SUBSET) — Mean Hallmark ρ",
        tfx_order=_BENCH_TFX_ORDER,
        depth_order=_BENCH_DEPTH_ORDER,
        model_order=hallmark_model_order,
        naive_model="Naive",
    )
    for depth_val in ["10x", "5x"]:
        _maybe(
            benchmark_depth_lines,
            overwrite,
            summary_df,
            os.path.join(hallmark_out, f"benchmark_lines_{depth_val}"),
            depth=depth_val,
            model_order=hallmark_model_order,
            metric_defs=[("mean_pathway_rho", "Mean Hallmark ρ")],
            title=f"HallmarkScores — Mean Hallmark ρ vs TFx ({depth_val} depth)",
        )


def _write_patient_benchmark_view(
    metrics_table: pd.DataFrame,
    gene_rho_table: pd.DataFrame,
    sample_rho_table: pd.DataFrame,
    out_dir: str,
    view_name: str,
    cohort_order: Sequence[str],
    model_order: Sequence[str],
    overwrite: bool,
) -> None:
    """Write one filtered patient benchmarking view (COMPLETE or SUBSET)."""
    view_out = ensure_dir(os.path.join(out_dir, view_name))
    metrics_view = metrics_table[metrics_table["model"].isin(model_order)].copy()
    gene_rho_view = gene_rho_table[gene_rho_table["model"].isin(model_order)].copy()
    sample_rho_view = sample_rho_table[sample_rho_table["model"].isin(model_order)].copy()

    if metrics_view.empty:
        return

    metrics_view.to_csv(
        os.path.join(view_out, "summary_benchmarking.tsv"), sep="\t", index=False,
    )

    for metric_col, ylabel in [
        ("gvw_rho", "GVW-ρ"),
        ("mean_gene_rho", "Mean Gene ρ"),
        ("rmse", "Mean Gene RMSE"),
        ("mean_top10pct_gene_rho", "Mean ρ (Top 10%)"),
        ("n_genes_rho_ge070", "Genes with ρ≥0.70"),
    ]:
        _maybe(
            benchmark_cohort_boxplots,
            overwrite,
            metrics_view,
            os.path.join(view_out, f"benchmark_cohort_{metric_col}"),
            metric_col=metric_col,
            ylabel=ylabel,
            title=f"Benchmarking Patient ({view_name}) — {ylabel}",
            cohort_order=cohort_order,
            model_order=model_order,
            naive_model="Naive",
        )

    if not sample_rho_view.empty:
        _maybe(
            benchmark_cohort_boxplots,
            overwrite,
            sample_rho_view,
            os.path.join(view_out, "benchmark_cohort_sample_rho"),
            metric_col="spearman_rho",
            ylabel="Sample ρ",
            title=f"Benchmarking Patient ({view_name}) — Per-Sample Spearman ρ",
            cohort_order=cohort_order,
            model_order=model_order,
            naive_model="Naive",
        )

    if not gene_rho_view.empty:
        _maybe(
            benchmark_cohort_histograms,
            overwrite,
            gene_rho_view,
            os.path.join(view_out, "benchmark_cohort_gene_rho_hist"),
            cohort_order=cohort_order,
            model_order=model_order,
            title=f"Benchmarking Patient ({view_name}) — Per-Gene Spearman ρ",
        )


# EPIC-Seq sample name cleaners per cohort.  CRPC-WCDT uses long-form
# names (e.g. DTB-097-Progression-cfDNA_recalibrated) while BLCA-RA uses
# the standard cram suffix form (e.g. 15-109_1107.cram).
_PANFT_EPIC_CLEANERS = {
    "CRPC-WCDT": clean_epicseq_sample_names_wcdt,
    "BLCA-RA": clean_epicseq_sample_names,
}


# ---------------------------------------------------------------------------
# ROC on/off gene calling helpers
# ---------------------------------------------------------------------------

_ONOFF_THRESHOLD = 2.0  # log2(CPM+1) threshold for on/off calling

_ROC_GENE_SETS = {
    "DruggableGenome_Tier1": "DruggableGenome_Tier1",
    "CancerSurfaceome": "CancerSurfaceome",
}


def _compute_roc_for_model(
    truth_flat: np.ndarray,
    pred_flat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve for binary on/off classification.

    truth_flat and pred_flat are 1-D arrays of matching length.
    Binary labels: truth >= threshold → on (1), else off (0).
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    y_true = (truth_flat >= _ONOFF_THRESHOLD).astype(int)
    # Edge case: all same class
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.array([0, 1]), np.array([0, 1]), 0.5

    auc = roc_auc_score(y_true, pred_flat)
    fpr, tpr, _ = roc_curve(y_true, pred_flat)
    return fpr, tpr, auc


def _run_roc_gene_calling(
    truth_wide: pd.DataFrame,
    model_preds: Dict[str, pd.DataFrame],
    proteus_ci: Dict[str, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    proteus_mdn: Dict[str, Dict[str, Optional[pd.DataFrame]]],
    out_dir: str,
    overwrite: bool,
    model_order: Sequence[str],
    cohort: str,
    activity_threshold_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Compute and plot ROC curves for on/off gene calling.

    Parameters
    ----------
    proteus_ci : dict
        ``{model_name: (ci_lo, ci_hi)}`` for Proteus models that have CIs.
    proteus_mdn : dict
        ``{model_name: {mdn_key: wide_df_or_none}}`` for Proteus models with
        full MDN parameter outputs.
    """
    for gs_display, gs_key in _ROC_GENE_SETS.items():
        if not gene_list_available(gs_key):
            continue
        gene_list = list(dict.fromkeys(get_gene_list(gs_key)))
        common_genes = [g for g in gene_list if g in truth_wide.index]
        if len(common_genes) < 10:
            print(f"    [skip] ROC {gs_display}: only {len(common_genes)} genes "
                  f"overlap with truth")
            continue

        roc_curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

        for model_name in model_order:
            if model_name not in model_preds:
                continue
            # Skip the row-shuffled Naive baseline — the shuffled-truth
            # naive (below) is the sole null reference for ROC.
            if model_name == "Naive":
                continue
            pw = model_preds[model_name]
            common_samples = [s for s in pw.columns if s in truth_wide.columns]
            cg = [g for g in common_genes if g in pw.index]
            if not common_samples or len(cg) < 5:
                continue

            t = truth_wide.loc[cg, common_samples].values.flatten()
            p = pw.loc[cg, common_samples].values.flatten()

            # Drop NaNs
            valid = np.isfinite(t) & np.isfinite(p)
            if valid.sum() < 10:
                continue

            n_valid = int(valid.sum())
            fpr, tpr, auc = _compute_roc_for_model(t[valid], p[valid])
            roc_curves[model_name] = (fpr, tpr, auc, n_valid)

        # Shuffled-truth naive (labelled "Naive")
        first_model = next(iter(model_preds.values()))
        common_samples_naive = [s for s in first_model.columns
                                if s in truth_wide.columns]
        cg_naive = [g for g in common_genes if g in first_model.index]
        if len(common_samples_naive) >= 3 and len(cg_naive) >= 5:
            t_naive = truth_wide.loc[cg_naive, common_samples_naive].values.flatten()
            rng_naive = np.random.default_rng(42)
            t_shuffled = rng_naive.permutation(t_naive)
            p_naive = first_model.loc[cg_naive, common_samples_naive].values.flatten()
            valid_naive = np.isfinite(t_shuffled) & np.isfinite(p_naive)
            if valid_naive.sum() >= 10:
                n_naive = int(valid_naive.sum())
                fn, tn, an = _compute_roc_for_model(
                    t_shuffled[valid_naive], p_naive[valid_naive])
                roc_curves["Naive"] = (fn, tn, an, n_naive)

        if roc_curves:
            _maybe(roc_gene_calling, overwrite,
                   roc_curves,
                   os.path.join(out_dir, f"ROC_{gs_display}_{cohort}"),
                   title=f"ROC Gene On/Off — {gs_display} ({cohort})",
                   model_order=list(model_order))

        print(f"    ROC {gs_display}: {len(roc_curves)} curves")

        # ── Activity odds ratio threshold sweeps: ON PPV/FDR and OFF NPV/FOR ──
        threshold_curves: Dict[str, Dict[str, Any]] = {}
        inactive_threshold_curves: Dict[str, Dict[str, Any]] = {}

        for model_name in model_order:
            if model_name not in model_preds:
                continue
            pw = model_preds[model_name]
            cg_r = [g for g in common_genes if g in pw.index]
            cs_r = [s for s in pw.columns if s in truth_wide.columns]
            if len(cg_r) < 5 or len(cs_r) < 3:
                continue

            mdn_tables = proteus_mdn.get(model_name, {})
            required_keys = (
                "p0_wide",
                "pi_gamma_wide",
                "pi_normal_wide",
            )
            if not all(mdn_tables.get(key) is not None for key in required_keys):
                continue
            cg_mdn = [
                g for g in cg_r
                if all(g in mdn_tables[key].index for key in required_keys)
            ]
            cs_mdn = [
                s for s in cs_r
                if all(s in mdn_tables[key].columns for key in required_keys)
            ]
            if len(cg_mdn) < 5 or len(cs_mdn) < 3:
                continue

            truth_active = truth_wide.loc[cg_mdn, cs_mdn]
            activity_odds_ratio = mdn_activity_odds_ratio_wide(
                mdn_tables["p0_wide"].loc[cg_mdn, cs_mdn],
                mdn_tables["pi_gamma_wide"].loc[cg_mdn, cs_mdn],
                mdn_tables["pi_normal_wide"].loc[cg_mdn, cs_mdn],
            )
            threshold_summary = _compute_activity_odds_ratio_summary_wide(
                truth_active,
                activity_odds_ratio,
                target_ppv=_ACTIVE_CALL_TARGET_PPV,
                target_npv=_INACTIVE_CALL_TARGET_NPV,
                threshold=_ONOFF_THRESHOLD,
            )
            if threshold_summary is not None:
                threshold_curves[model_name] = threshold_summary["on_curve"]
                inactive_threshold_curves[model_name] = threshold_summary["off_curve"]

        if threshold_curves:
            curve_rows: List[Dict[str, Any]] = []
            summary_rows: List[Dict[str, Any]] = []
            for model_name in model_order:
                if model_name not in threshold_curves:
                    continue
                curve = threshold_curves[model_name]
                for threshold_value, coverage_value, ppv_value, fdr_value, n_predicted_on in zip(
                    curve["threshold"],
                    curve["coverage"],
                    curve["ppv"],
                    curve["fdr"],
                    curve["n_predicted_on"],
                ):
                    curve_rows.append(
                        {
                            "model": model_name,
                            "activity_odds_ratio_threshold": float(threshold_value),
                            "coverage": float(coverage_value),
                            "ppv": float(ppv_value),
                            "fdr": float(fdr_value),
                            "n_predicted_on": int(n_predicted_on),
                        }
                    )
                summary_rows.append(
                    {
                        "model": model_name,
                        "target_ppv": float(curve["target_ppv"]),
                        "target_met": bool(curve["target_met"]),
                        "selected_activity_odds_ratio_threshold": float(curve["selected_threshold"]),
                        "selected_coverage": float(curve["selected_coverage"]),
                        "selected_ppv": float(curve["selected_ppv"]),
                        "selected_fdr": float(curve["selected_fdr"]),
                        "selected_n_predicted_on": int(curve["selected_n_predicted_on"]),
                        "n_total": int(curve["n_total"]),
                        "n_true_on": int(curve["n_true_on"]),
                    }
                )

            pd.DataFrame(curve_rows).to_csv(
                os.path.join(out_dir, f"ActiveConfidenceCurve_{gs_display}_{cohort}.tsv"),
                sep="\t",
                index=False,
            )
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, f"ActiveConfidenceThreshold_{gs_display}_{cohort}.tsv"),
                sep="\t",
                index=False,
            )
            _maybe(
                active_confidence_threshold_plot,
                overwrite,
                threshold_curves,
                os.path.join(out_dir, f"ActiveConfidenceThreshold_{gs_display}_{cohort}"),
                title=f"Active Threshold (AOR) — {gs_display} ({cohort})",
                model_order=list(model_order),
            )
            print(f"    ActiveConfidenceThreshold {gs_display}: {len(threshold_curves)} models")

        if inactive_threshold_curves:
            curve_rows = []
            summary_rows = []
            for model_name in model_order:
                if model_name not in inactive_threshold_curves:
                    continue
                curve = inactive_threshold_curves[model_name]
                for threshold_value, coverage_value, npv_value, for_value, n_predicted_off in zip(
                    curve["threshold"],
                    curve["coverage"],
                    curve["npv"],
                    curve["for"],
                    curve["n_predicted_off"],
                ):
                    curve_rows.append(
                        {
                            "model": model_name,
                            "activity_odds_ratio_threshold": float(threshold_value),
                            "coverage": float(coverage_value),
                            "npv": float(npv_value),
                            "for": float(for_value),
                            "n_predicted_off": int(n_predicted_off),
                        }
                    )
                summary_rows.append(
                    {
                        "model": model_name,
                        "target_npv": float(curve["target_npv"]),
                        "target_met": bool(curve["target_met"]),
                        "selected_activity_odds_ratio_threshold": float(curve["selected_threshold"]),
                        "selected_coverage": float(curve["selected_coverage"]),
                        "selected_npv": float(curve["selected_npv"]),
                        "selected_for": float(curve["selected_for"]),
                        "selected_n_predicted_off": int(curve["selected_n_predicted_off"]),
                        "n_total": int(curve["n_total"]),
                        "n_true_off": int(curve["n_true_off"]),
                    }
                )

            pd.DataFrame(curve_rows).to_csv(
                os.path.join(out_dir, f"InactiveConfidenceCurve_{gs_display}_{cohort}.tsv"),
                sep="\t",
                index=False,
            )
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(out_dir, f"InactiveConfidenceThreshold_{gs_display}_{cohort}.tsv"),
                sep="\t",
                index=False,
            )
            _maybe(
                inactive_confidence_threshold_plot,
                overwrite,
                inactive_threshold_curves,
                os.path.join(out_dir, f"InactiveConfidenceThreshold_{gs_display}_{cohort}"),
                title=f"Inactive Threshold (AOR) — {gs_display} ({cohort})",
                model_order=list(model_order),
            )
            print(f"    InactiveConfidenceThreshold {gs_display}: {len(inactive_threshold_curves)} models")

        # ── 3-way confusion summary: ON / OFF / INDETERMINATE ──
        confusion_results: Dict[str, Dict[str, Any]] = {}
        confusion_order: List[str] = []
        shared_lower_threshold = None
        shared_upper_threshold = None
        if activity_threshold_summary is not None:
            shared_lower_threshold = float(activity_threshold_summary["lower_threshold"])
            shared_upper_threshold = float(activity_threshold_summary["upper_threshold"])

        for model_name in model_order:
            if model_name not in model_preds or model_name == "Naive":
                continue
            pw = model_preds[model_name]
            common_samples = [s for s in pw.columns if s in truth_wide.columns]
            cg = [g for g in common_genes if g in pw.index]
            if not common_samples or len(cg) < 5:
                continue

            t = truth_wide.loc[cg, common_samples].to_numpy(dtype=np.float64).ravel()
            p = pw.loc[cg, common_samples].to_numpy(dtype=np.float64).ravel()

            activity_odds_ratio_flat = None
            if (
                shared_lower_threshold is not None
                and shared_upper_threshold is not None
                and model_name in proteus_mdn
            ):
                mdn_tables = proteus_mdn.get(model_name, {})
                required_keys = (
                    "p0_wide",
                    "pi_gamma_wide",
                    "pi_normal_wide",
                )
                if all(mdn_tables.get(key) is not None for key in required_keys):
                    cg_mdn = [
                        g for g in cg
                        if all(g in mdn_tables[key].index for key in required_keys)
                    ]
                    cs_mdn = [
                        s for s in common_samples
                        if all(s in mdn_tables[key].columns for key in required_keys)
                    ]
                    if len(cg_mdn) >= 5 and len(cs_mdn) >= 3:
                        activity_odds_ratio_wide = mdn_activity_odds_ratio_wide(
                            mdn_tables["p0_wide"].loc[cg_mdn, cs_mdn],
                            mdn_tables["pi_gamma_wide"].loc[cg_mdn, cs_mdn],
                            mdn_tables["pi_normal_wide"].loc[cg_mdn, cs_mdn],
                        )
                        activity_odds_ratio_flat = activity_odds_ratio_wide.to_numpy(dtype=np.float64).ravel()
                        t = truth_wide.loc[cg_mdn, cs_mdn].to_numpy(dtype=np.float64).ravel()
                        p = pw.loc[cg_mdn, cs_mdn].to_numpy(dtype=np.float64).ravel()

            if activity_odds_ratio_flat is not None:
                binary_name = f"{model_name} (binary)"
                ua_name = f"{model_name} (uncertainty-informed)"

                binary_confusion = compute_onoff_confusion(
                    t,
                    p,
                    threshold=_ONOFF_THRESHOLD,
                )
                ua_confusion = compute_onoff_confusion(
                    t,
                    p,
                    activity_odds_ratio_flat=activity_odds_ratio_flat,
                    lower_threshold=shared_lower_threshold,
                    upper_threshold=shared_upper_threshold,
                    threshold=_ONOFF_THRESHOLD,
                )

                if binary_confusion.get("n_total", 0) > 0:
                    confusion_results[binary_name] = binary_confusion
                    confusion_order.append(binary_name)
                if ua_confusion.get("n_total", 0) > 0:
                    confusion_results[ua_name] = ua_confusion
                    confusion_order.append(ua_name)
            else:
                d = compute_onoff_confusion(
                    t,
                    p,
                    threshold=_ONOFF_THRESHOLD,
                )
                if d.get("n_total", 0) > 0:
                    confusion_results[model_name] = d
                    confusion_order.append(model_name)

        if confusion_results:
            preferred_confusion_order = [
                "Proteus (binary)",
                "Proteus (uncertainty-informed)",
                "EPIC_GEP",
                "XGBoost",
            ]
            confusion_order = [m for m in preferred_confusion_order if m in confusion_results] + [
                m for m in confusion_order if m not in preferred_confusion_order
            ]
            conf_rows = []
            for m in confusion_order:
                d = confusion_results[m]
                conf_rows.append({
                    "model": m,
                    "n_total": d["n_total"],
                    "n_on": d["n_on"],
                    "n_off": d["n_off"],
                    "n_indet": d["n_indet"],
                    "accuracy": d.get("accuracy", float("nan")),
                    "ppv": d.get("ppv", float("nan")),
                    "npv": d.get("npv", float("nan")),
                    "fpr": d.get("fpr", float("nan")),
                    "specificity": 1.0 - d.get("fpr", float("nan")) if np.isfinite(d.get("fpr", float("nan"))) else float("nan"),
                    "coverage": d.get("coverage", float("nan")),
                    "lower_activity_odds_ratio_threshold": shared_lower_threshold,
                    "upper_activity_odds_ratio_threshold": shared_upper_threshold,
                })
            conf_df = pd.DataFrame(conf_rows)
            conf_df.to_csv(
                os.path.join(out_dir, f"OnOffConfusion_{gs_display}_{cohort}.tsv"),
                sep="\t", index=False,
            )

            _maybe(
                onoff_confusion_summary_plot,
                overwrite,
                confusion_results,
                os.path.join(out_dir, f"OnOffConfusion_{gs_display}_{cohort}"),
                title=f"On/Off Confusion — {gs_display} ({cohort})",
                model_order=confusion_order,
            )
            print(
                f"    OnOffConfusion {gs_display}: "
                f"{len(confusion_results)} models"
            )


def run_benchmarking_panft(args: argparse.Namespace) -> None:
    """Benchmarking_Patient analysis on real patient cohorts.

    Compares models on CRPC-WCDT and BLCA-RA biopsies. Metrics are computed
    per cohort and written into COMPLETE and SUBSET output views.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Benchmarking_Patient")
    print("=" * 70)

    if _should_skip_section("Benchmarking_Patient", args):
        return

    out_dir = _section_results_dir("Benchmarking_Patient")
    _write_run_manifest("Benchmarking_Patient", out_dir, args)

    canonical_truth = _build_plca_truth(args.minimum_tfx)
    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
    interval_calibrator: Optional[RegimeCalibrator] = None
    activity_threshold_summary: Optional[Dict[str, Any]] = None
    if cv_bundle is not None:
        cv_full = cv_bundle["cv_full"]
        activity_threshold_summary = _compute_patient_cv_activity_thresholds(
            cv_full,
            cv_bundle["cohort_map"],
        )
        if not canonical_truth.empty and args.calibrate_cis and cv_full.get("ci_lo_wide") is not None and cv_full.get("ci_hi_wide") is not None:
            interval_calibrator = _fit_global_calibrator(canonical_truth, cv_full)
            if interval_calibrator is not None:
                print(f"  {interval_calibrator.summary()}")

    # Accumulate metrics, per-gene rho, and per-sample rho across cohorts
    all_metrics_rows: List[Dict[str, Any]] = []
    all_gene_rho_rows: List[Dict[str, Any]] = []
    all_sample_rho_rows: List[Dict[str, Any]] = []

    for cohort in PATIENT_BENCHMARK_COHORTS:
        print(f"\n  ── Cohort: {cohort} ──")

        # Load truth
        truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
        if not truth_path.exists():
            print(f"  [skip] Truth not found: {truth_path.name}")
            continue
        truth_wide = load_truth(cohort)
        subtypes_df = _load_filtered_subtypes_for_reference(
            cohort,
            truth_wide.columns,
            minimum_tfx=args.minimum_tfx,
            report_context=f"Benchmarking_Patient {cohort}",
        )
        keep_samples = list(subtypes_df.index) if args.minimum_tfx > 0.0 and subtypes_df is not None else None
        if keep_samples is not None:
            truth_wide = _filter_wide_to_samples(truth_wide, keep_samples)
            if truth_wide is not None and truth_wide.shape[1] == 0:
                print(f"  [skip] No samples remain for {cohort} after minimum TFx filtering")
                continue
        print(f"  Truth: {truth_wide.shape[0]} genes × {truth_wide.shape[1]} samples")

        # Build short→prefixed map from truth column names for this cohort
        pmap = _build_prefix_map(truth_wide.columns, cohort)

        # Collect model predictions for this cohort
        model_preds: Dict[str, pd.DataFrame] = {}
        proteus_ci: Dict[str, Tuple[Optional[pd.DataFrame],
                                     Optional[pd.DataFrame]]] = {}
        proteus_mdn: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}

        # Proteus models
        proteus_paths = PATIENT_PROTEUS_MODEL_PATHS.get(cohort, {})
        for model_name, rel_path in proteus_paths.items():
            path = PROJECT_ROOT / rel_path
            if path.exists():
                full_data = load_holdout_predictions_proteus_full(str(path))
                for key in ("pred_wide", "ci_lo_wide", "ci_hi_wide", *_MDN_PARAM_KEYS):
                    table = full_data.get(key)
                    if table is not None:
                        _apply_prefix_map(table, pmap)
                if keep_samples is not None:
                    full_data = _filter_prediction_bundle_to_samples(full_data, keep_samples)
                analysis_data = full_data
                if model_name == "Proteus":
                    analysis_data = _prepare_proteus_analysis_bundle(
                        full_data,
                        interval_calibrator=interval_calibrator,
                        calibrate_cis=args.calibrate_cis,
                    )
                pred = analysis_data["pred_wide"]
                ci_lo = analysis_data.get("ci_lo_wide")
                ci_hi = analysis_data.get("ci_hi_wide")
                mdn_tables: Dict[str, Optional[pd.DataFrame]] = {}
                for key in (
                    "p0_wide",
                    "pi_gamma_wide",
                    "pi_normal_wide",
                    "gamma_beta_wide",
                    "normal_mean_wide",
                    "normal_stdev_wide",
                ):
                    table = full_data.get(key)
                    mdn_tables[key] = table
                model_preds[model_name] = pred
                proteus_ci[model_name] = (ci_lo, ci_hi)
                proteus_mdn[model_name] = mdn_tables
                print(f"    Loaded {model_name}: {pred.shape[1]} samples")
            else:
                print(f"    [skip] {model_name}: {rel_path} not found")

        # XGBoost
        xgb_rel = PATIENT_XGB_PATHS.get(cohort)
        if xgb_rel:
            xgb_path = PROJECT_ROOT / xgb_rel
            if xgb_path.exists():
                xgb_pred = load_holdout_predictions_xgboost(str(xgb_path))
                _apply_prefix_map(xgb_pred, pmap)
                xgb_pred = _filter_wide_to_samples(xgb_pred, keep_samples)
                model_preds["XGBoost"] = xgb_pred
                print(f"    Loaded XGBoost: {xgb_pred.shape[1]} samples")
            else:
                print(f"    [skip] XGBoost: {xgb_rel} not found")

        # EPIC-Seq
        epic_rel = PATIENT_EPIC_PATHS.get(cohort)
        if epic_rel:
            epic_path = PROJECT_ROOT / epic_rel
            if epic_path.exists():
                epic_preds = load_holdout_predictions_epicseq(
                    str(epic_path), CONFIG_EPIC_SCORE_COLUMNS,
                )
                cleaner = _PANFT_EPIC_CLEANERS.get(cohort, clean_epicseq_sample_names)
                for score_name, epw in epic_preds.items():
                    # PFE here is the raw EPIC-Seq PFE output (column "PFE"),
                    # not the normalized quantity used in GEP construction.
                    display = CONFIG_EPIC_DISPLAY_NAMES.get(score_name, f"EPIC_{score_name}")
                    epw.columns = cleaner(epw.columns)
                    _apply_prefix_map(epw, pmap)
                    epw = _filter_wide_to_samples(epw, keep_samples)
                    model_preds[display] = epw
                    print(f"    Loaded {display}: {epw.shape[1]} samples")
            else:
                print(f"    [skip] EPIC-Seq: {epic_rel} not found")

        if not model_preds:
            print(f"  [skip] No model predictions for {cohort}")
            continue

        # Naive baseline: shuffle sample values within each gene for this cohort.
        naive_pred = generate_null_predictions(truth_wide, seed=42)
        model_preds["Naive"] = naive_pred

        # Canonical order for this cohort
        model_order = [m for m in BENCHMARK_COMPLETE_MODELS if m in model_preds]

        # Compute per-model metrics
        for model_name in model_order:
            pw = model_preds[model_name]
            m = compute_subset_metrics(truth_wide, pw)
            all_metrics_rows.append({
                "cohort": cohort,
                "model": model_name,
                "sample_rho_mean": float(np.nanmean(m["sample_rho"])) if len(m["sample_rho"]) else np.nan,
                "mean_gene_rho": m["mean_gene_rho"],
                "median_gene_rho": m["median_gene_rho"],
                "gvw_rho": m["gvw_rho"],
                "rmse": m["rmse"],
                "mean_top10pct_gene_rho": m["mean_top10pct_gene_rho"],
                "n_genes_rho_ge070": m["n_genes_rho_ge070"],
                "n_samples": len(m["sample_rho"]),
            })

            # Per-gene Spearman rho
            gene_names = m.get("gene_rho_genes", np.array([], dtype=object))
            for gi, rho_val in enumerate(m["gene_rho_array"]):
                all_gene_rho_rows.append({
                    "cohort": cohort, "model": model_name,
                    "gene": gene_names[gi] if gi < len(gene_names) else "",
                    "spearman_rho": rho_val,
                })

            # Per-sample Spearman rho
            for rho_val in m["sample_rho"]:
                all_sample_rho_rows.append({
                    "cohort": cohort, "model": model_name,
                    "spearman_rho": rho_val,
                })

        # ROC on/off gene calling (per cohort)
        for view_name, view_models in benchmark_model_views().items():
            view_order = [m for m in view_models if m in model_preds]
            view_preds = {m: model_preds[m] for m in view_order}
            view_ci = {m: proteus_ci[m] for m in view_order if m in proteus_ci}
            _run_roc_gene_calling(
                truth_wide,
                view_preds,
                view_ci,
                {m: proteus_mdn[m] for m in view_order if m in proteus_mdn},
                ensure_dir(os.path.join(out_dir, view_name)),
                args.overwrite,
                view_order,
                cohort,
                activity_threshold_summary=activity_threshold_summary,
            )

    metrics_table = pd.DataFrame(all_metrics_rows)
    gene_rho_table = pd.DataFrame(all_gene_rho_rows)
    sample_rho_table = pd.DataFrame(all_sample_rho_rows)

    if metrics_table.empty:
        print("  [skip] No metrics computed")
        return

    # Save summary TSV
    metrics_table.to_csv(
        os.path.join(out_dir, "summary_benchmarking_all.tsv"), sep="\t", index=False,
    )
    print(f"\n  Summary: {len(metrics_table)} rows")
    print(metrics_table.to_string(index=False))

    for view_name, view_models in benchmark_model_views().items():
        _write_patient_benchmark_view(
            metrics_table,
            gene_rho_table,
            sample_rho_table,
            out_dir,
            view_name,
            PATIENT_BENCHMARK_COHORTS,
            [m for m in view_models if m in set(metrics_table["model"])],
            args.overwrite,
        )

    print("\n  Done")


def run_benchmarking(args: argparse.Namespace) -> None:
    """Benchmarking_Synthetic analysis on the PHS TFx/depth grid.

    Loads PHS predictions for Proteus, Proteus_NoSeq, Proteus_OnlySeq,
    XGBoost, EPIC-Seq scores, and a Naive (row-shuffled truth) baseline.
    Computes metrics per (TFx, depth, model) cell and produces grid
    boxplots, histograms, and depth-specific line plots.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Benchmarking_Synthetic")
    print("=" * 70)

    if _should_skip_section("Benchmarking_Synthetic", args):
        return

    out_dir = _section_results_dir("Benchmarking_Synthetic")
    _write_run_manifest("Benchmarking_Synthetic", out_dir, args)
    n_workers = args.n_workers

    canonical_truth = _build_plca_truth(args.minimum_tfx)
    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
    interval_calibrator: Optional[RegimeCalibrator] = None
    if cv_bundle is not None and not canonical_truth.empty:
        cv_full = cv_bundle["cv_full"]
        if args.calibrate_cis and cv_full.get("ci_lo_wide") is not None and cv_full.get("ci_hi_wide") is not None:
            interval_calibrator = _fit_global_calibrator(canonical_truth, cv_full)
            if interval_calibrator is not None:
                print(f"  {interval_calibrator.summary()}")

    # ── Load model predictions and build truth ──────────────────────────
    model_preds: Dict[str, pd.DataFrame] = {}  # model_name -> wide (genes × samples)
    proteus_bundles: Dict[str, Dict[str, Any]] = {}

    for model_name, rel_path in SYNTHETIC_PROTEUS_MODEL_PATHS.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"  [skip] {model_name}: {rel_path} not found")
            continue
        full_data = load_holdout_predictions_proteus_full(str(path))
        model_preds[model_name] = full_data["pred_wide"]
        proteus_bundles[model_name] = full_data
        pred_wide = full_data["pred_wide"]
        print(f"  Loaded {model_name}: {pred_wide.shape[0]} genes × {pred_wide.shape[1]} samples")

    # XGBoost uses a different column name ('prediction' not 'prediction_mean')
    xgb_path = PROJECT_ROOT / SYNTHETIC_XGB_PATH
    if xgb_path.exists():
        xgb_pred = load_holdout_predictions_xgboost(str(xgb_path))
        model_preds["XGBoost"] = xgb_pred
        print(f"  Loaded XGBoost: {xgb_pred.shape[0]} genes × {xgb_pred.shape[1]} samples")
    else:
        print(f"  [skip] XGBoost: {SYNTHETIC_XGB_PATH} not found")

    if not model_preds:
        print("  [skip] No model predictions found for benchmarking")
        return

    # Use the first model's sample set as reference
    ref_pred = next(iter(model_preds.values()))
    all_samples = sorted(ref_pred.columns)

    # Build truth from PDX expression labels
    truth_wide = load_pdx_truth_for_phs(all_samples)
    print(f"  Truth: {truth_wide.shape[0]} genes × {truth_wide.shape[1]} samples")

    # Pre-compute gene variance from the 10 unique PDX profiles only
    # (PHS samples are admixtures of the same 10 PDXs at various TFx/depth,
    # so variance from 181 duplicated columns would be misleading).
    unique_pdx_names = set()
    pdx_col_map: Dict[str, str] = {}  # pdx_name -> first PHS sample w/ that PDX
    for s in all_samples:
        info = parse_phs_sample(s)
        pdx_name = info["pdx"]
        if pdx_name not in unique_pdx_names:
            unique_pdx_names.add(pdx_name)
            pdx_col_map[pdx_name] = s
    unique_truth = truth_wide[list(pdx_col_map.values())]
    unique_truth.columns = list(pdx_col_map.keys())
    var_map = compute_gene_variance(unique_truth)
    print(f"  Variance map: {len(var_map)} genes from {len(unique_pdx_names)} unique PDX profiles")

    # Naive model: shuffle sample values within each gene across the PHS grid.
    naive_pred = generate_null_predictions(truth_wide, seed=42)
    model_preds["Naive"] = naive_pred

    # ── Load EPIC-Seq scores ───────────────────────────────────────────
    epic_path = PROJECT_ROOT / SYNTHETIC_EPIC_PATH
    epic_preds: Dict[str, pd.DataFrame] = {}
    if epic_path.exists():
        epic_preds = load_holdout_predictions_epicseq(
            str(epic_path), CONFIG_EPIC_SCORE_COLUMNS,
        )
        for score_name, epw in epic_preds.items():
            # PFE here is the raw EPIC-Seq PFE output (column "PFE"),
            # not the normalized quantity used in GEP construction.
            display = CONFIG_EPIC_DISPLAY_NAMES.get(score_name, f"EPIC_{score_name}")
            model_preds[display] = epw
            print(f"  Loaded {display}: {epw.shape[0]} genes × {epw.shape[1]} samples")
    else:
        print(f"  [skip] EPIC-Seq: {SYNTHETIC_EPIC_PATH} not found")

    # ── Parse sample metadata and build grid ────────────────────────────
    sample_meta = []
    for s in all_samples:
        info = parse_phs_sample(s)
        sample_meta.append({
            "sample": s,
            "pdx": info["pdx"],
            "depth": info["coverage"],
            "tfx": info["tfx"],
        })
    meta_df = pd.DataFrame(sample_meta)

    # Filter to analysis grid (exclude 1x, 30x, 50x)
    meta_df = meta_df[~meta_df["depth"].isin(_BENCH_EXCLUDED_DEPTHS)]
    grid_samples = set(meta_df["sample"])
    print(f"  Grid samples (after depth filter): {len(grid_samples)}")

    # ── Compute metrics per (TFx, depth, model) ────────────────────────
    model_order = [m for m in BENCHMARK_COMPLETE_MODELS if m in model_preds]

    metrics_rows = []     # one row per (tfx, depth, model) with scalar metrics
    gene_rho_rows = []    # one row per (tfx, depth, model, gene) with gene rho
    sample_rho_rows = []  # one row per (tfx, depth, model, sample) with sample rho

    # Build work items for parallel dispatch
    work_items: List[Dict[str, Any]] = []
    for tfx in _BENCH_TFX_ORDER:
        for depth in _BENCH_DEPTH_ORDER:
            cell_samples = sorted(
                meta_df[(meta_df["tfx"] == tfx) & (meta_df["depth"] == depth)]["sample"]
            )
            if not cell_samples:
                continue

            # Subset truth to these samples
            tw = truth_wide[[s for s in cell_samples if s in truth_wide.columns]]
            if tw.shape[1] < 1:
                continue

            for model_name in model_order:
                pw_full = model_preds.get(model_name)
                if pw_full is None:
                    continue
                pw = pw_full[[s for s in cell_samples if s in pw_full.columns]]
                if pw.shape[1] < 1:
                    continue
                work_items.append({
                    "tw": tw, "pw": pw, "var_map": var_map,
                    "tfx": tfx, "depth": depth, "model": model_name,
                })

    # Parallel computation across all (TFx, depth, model) combos
    pool_workers = min(n_workers, len(work_items)) if n_workers > 1 else 1
    print(f"  Computing {len(work_items)} combos across {pool_workers} workers ...")

    results_list: List[Dict[str, Any]] = []
    if pool_workers > 1 and len(work_items) > 1:
        with ProcessPoolExecutor(max_workers=pool_workers) as pool:
            futures = {pool.submit(_bench_worker, item): i
                       for i, item in enumerate(work_items)}
            for fut in as_completed(futures):
                results_list.append(fut.result())
                done = len(results_list)
                if done % 20 == 0 or done == len(work_items):
                    print(f"    [{done}/{len(work_items)}] computed")
    else:
        for item in work_items:
            results_list.append(_bench_worker(item))

    # Unpack results
    for r in results_list:
        m = r["m"]
        metrics_rows.append({
            "tfx": r["tfx"], "depth": r["depth"], "model": r["model"],
            "sample_rho_mean": float(np.nanmean(m["sample_rho"])) if len(m["sample_rho"]) else np.nan,
            "mean_gene_rho": m["mean_gene_rho"],
            "median_gene_rho": m["median_gene_rho"],
            "gvw_rho": m["gvw_rho"],
            "rmse": m["rmse"],
            "mean_top10pct_gene_rho": m["mean_top10pct_gene_rho"],
            "n_genes_rho_ge070": m["n_genes_rho_ge070"],
        })

        # Per-gene Spearman rho for histogram grids
        gene_names = m.get("gene_rho_genes", np.array([], dtype=object))
        for gi, rho_val in enumerate(m["gene_rho_array"]):
            gene_rho_rows.append({
                "tfx": r["tfx"], "depth": r["depth"], "model": r["model"],
                "gene": gene_names[gi] if gi < len(gene_names) else "",
                "spearman_rho": rho_val,
            })

        # Per-sample Spearman rho for boxplot grids
        for rho_val in m["sample_rho"]:
            sample_rho_rows.append({
                "tfx": r["tfx"], "depth": r["depth"], "model": r["model"],
                "spearman_rho": rho_val,
            })

    metrics_table = pd.DataFrame(metrics_rows)
    gene_rho_table = pd.DataFrame(gene_rho_rows)
    sample_rho_table = pd.DataFrame(sample_rho_rows)

    if metrics_table.empty:
        print("  [skip] No metrics computed — no grid cells had data")
        return

    # Save summary TSV
    metrics_table.to_csv(
        os.path.join(out_dir, "summary_benchmarking_all.tsv"), sep="\t", index=False,
    )
    print(f"  Summary: {len(metrics_table)} rows")

    # ── Plots ──────────────────────────────────────────────────────────
    print("  Generating benchmark plots ...")
    for view_name, view_models in benchmark_model_views().items():
        _write_synthetic_benchmark_view(
            metrics_table,
            gene_rho_table,
            sample_rho_table,
            out_dir,
            view_name,
            [m for m in view_models if m in set(metrics_table["model"])],
            args.overwrite,
            var_map,
        )

    _run_synthetic_hallmark_benchmark(
        truth_wide,
        model_preds,
        proteus_bundles,
        meta_df,
        out_dir,
        overwrite=args.overwrite,
        montecarlo_draws=args.montecarlo_draws,
    )

    print("  Done")


# ---------------------------------------------------------------------------
# PLCA truth and cohort configuration helpers
# ---------------------------------------------------------------------------

_PLCA_SPLITS_TSV = "data/PLCA_sample-splits.tsv"

# Cohorts that have ground-truth expression labels and appear in PLCA training.
_PLCA_CV_COHORTS = ["CRPC-RA", "SCLC", "PC-PDX", "SCLC-PDX"]

# Per-cohort analysis config for PLCA-derived validation splits
_PLCA_CV_COHORT_CONFIG: Dict[str, Dict[str, Any]] = {
    "CRPC-RA": {
        "heatmap_gene_sets": ["PC-Phenotyping", "Targets_CRPC"],
        "goi": DEFAULT_GOI_PC,
        "gene_scatter_sets": ["Targets_CRPC"],
        "truth_label": "CRPC-RA",
    },
    "SCLC": {
        "heatmap_gene_sets": ["Targets_SCLC"],
        "goi": DEFAULT_GOI_SCLC,
        "gene_scatter_sets": ["Targets_SCLC"],
        "truth_label": "SCLC",
    },
    "PC-PDX": {
        "heatmap_gene_sets": ["PC-Phenotyping", "Targets_CRPC"],
        "goi": DEFAULT_GOI_PC,
        "gene_scatter_sets": ["Targets_CRPC"],
        "truth_label": "PC-PDX",
    },
    "SCLC-PDX": {
        "heatmap_gene_sets": ["Targets_SCLC"],
        "goi": DEFAULT_GOI_SCLC,
        "gene_scatter_sets": ["Targets_SCLC"],
        "truth_label": "SCLC-PDX",
    },
}


def _load_plca_fold_splits() -> pd.DataFrame:
    """Load data/PLCA_sample-splits.tsv → DataFrame with Cohort, Sample, Fold."""
    path = PROJECT_ROOT / _PLCA_SPLITS_TSV
    return pd.read_csv(path, sep="\t")


def _build_plca_truth(minimum_tfx: float = 0.0) -> pd.DataFrame:
    """Build canonical truth for PLCA CV cohorts (CRPC-RA, SCLC, PC-PDX, SCLC-PDX).

    Loads Log2CPMplus1 labels for all four cohorts, keeps only samples present
    in the PLCA sample-splits table (excludes HO PDX), concatenates on a common
    gene index.  PHT samples are excluded from truth (no independent labels).
    """
    splits = _load_plca_fold_splits()
    split_samples = set(splits["Sample"])

    parts = []
    for cohort in _PLCA_CV_COHORTS:
        truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
        if not truth_path.exists():
            continue
        tw = load_truth(cohort)
        keep = [s for s in tw.columns if s in split_samples]
        if minimum_tfx > 0.0 and _is_patient_cohort(cohort):
            filtered_subtypes = _load_filtered_subtypes_for_reference(
                cohort,
                pd.Index(keep),
                minimum_tfx=minimum_tfx,
                report_context=f"PLCA truth {cohort}",
            )
            keep = [s for s in keep if s in filtered_subtypes.index]
        if keep:
            parts.append(tw[keep])

    if not parts:
        return pd.DataFrame()

    common_genes = parts[0].index
    for p in parts[1:]:
        common_genes = common_genes.intersection(p.index)
    return pd.concat([p.loc[common_genes] for p in parts], axis=1)




# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

GROUPS = {
    "Examples": run_examples,
    "Benchmarking_Synthetic": run_benchmarking,
    "Benchmarking_Patient": run_benchmarking_panft,
    "Calibration": run_calibration,
    "Validation_PDX": run_validation_pdx,
    "Validation_Patient": run_validation_patient,
    "Discovery_Patient": run_patient_discovery,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proteus analysis framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--group", type=str, required=True,
        choices=list(GROUPS.keys()) + ["all"],
        help="Analysis group to run (or 'all').",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate all plots even if PDFs already exist.",
    )
    parser.add_argument(
        "--minimum_tfx", type=float, default=0.0,
        help="Drop patient samples with TFx below this threshold at load time (default: 0.0).",
    )
    parser.add_argument(
        "--calibrate_cis", action="store_true",
           help="Use CV-fitted conformal-calibrated Proteus 90%% CIs in downstream analyses.",
    )
    parser.add_argument(
        "--dim_ambiguous", action="store_true",
           help="Render and exclude single-gene samples whose 90%% CI overlaps the on/off threshold as hollow markers.",
    )
    parser.add_argument(
        "--scatter_ci", action="store_true",
        help="Pass interval surfaces into scatter analyses; single-gene/IHC plots still do not render per-point CI error bars.",
    )
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Worker processes to use (default: all available CPUs).",
    )
    parser.add_argument(
        "--montecarlo_draws", type=int, default=_DEFAULT_MONTE_CARLO_DRAWS,
        help=f"Monte Carlo expression draws for uncertainty-adjusted MDN analyses (default: {_DEFAULT_MONTE_CARLO_DRAWS}).",
    )

    args = parser.parse_args()

    available_cpus = max(1, os.cpu_count() or 1)
    if args.minimum_tfx < 0.0 or args.minimum_tfx > 1.0:
        parser.error("--minimum_tfx must be between 0.0 and 1.0")
    if args.montecarlo_draws < 1:
        parser.error("--montecarlo_draws must be >= 1")
    if args.n_workers is not None and args.n_workers < 1:
        parser.error("--n_workers must be >= 1")
    requested_workers = args.n_workers
    resolved_workers = available_cpus if requested_workers is None else min(requested_workers, available_cpus)
    args.n_workers = resolved_workers
    if requested_workers is None:
        print(f"Available CPUs: {available_cpus}  |  Using all workers")
    elif requested_workers > available_cpus:
        print(
            f"Available CPUs: {available_cpus}  |  Requested {requested_workers}, using {resolved_workers} workers"
        )
    else:
        print(f"Available CPUs: {available_cpus}  |  Using {resolved_workers} workers")

    t_start = time.time()

    if args.group == "all":
        for name, fn in GROUPS.items():
            fn(args)
    else:
        GROUPS[args.group](args)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
