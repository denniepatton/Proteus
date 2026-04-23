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
    cohort_summary_table,
    compute_cox_univariate,
    compute_cox_multivariate,
    compute_coverage_curve,
    compute_gene_variance,
    compute_geneset_score,
    compute_onoff_confusion,
    compute_risk_coverage_data,
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
    load_holdout_predictions_proteus,
    load_holdout_predictions_proteus_full,
    load_holdout_predictions_xgboost,
    load_palette,
    load_pdx_truth_for_phs,
    load_pluvicto_clinical,
    load_pluvicto_outcomes,
    load_subtypes,
    load_truth,
    mdn_monte_carlo_sample,
    mdn_predictive_variance,
    pam50_classify,
    pam50_classify_mc,
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
    benchmark_cohort_boxplots,
    benchmark_cohort_histograms,
    benchmark_cohort_metrics_lines,
    benchmark_depth_lines,
    benchmark_ecdna_lines,
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
    plot_regime_coverage_heatmap,
    plot_regime_qhat_heatmap,
    plot_reliability_scatter,
    plot_reliability_weights,
    plot_retention_curve,
    onoff_confusion_summary_plot,
    risk_coverage_plot,
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

_CALIBRATION_LEVELS: Tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def _section_results_dir(section_name: str) -> str:
    return ensure_dir(
        str(PROJECT_ROOT / "analysis" / "results" / SECTION_OUTPUT_DIRS[section_name])
    )


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
) -> Dict[str, Any]:
    """Summarize coverage and interval-width changes before/after calibration."""
    truth = flat["truth"]
    pred = flat["pred"]
    ci_lo = flat["ci_lo"]
    ci_hi = flat["ci_hi"]
    ci_lo_adj = flat["ci_lo_adj"]
    ci_hi_adj = flat["ci_hi_adj"]

    base_curve = compute_coverage_curve(truth, ci_lo, ci_hi, pred, nominal_levels)
    adjusted_curve = compute_coverage_curve(truth, ci_lo_adj, ci_hi_adj, pred, nominal_levels)

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
    """Persist per-regime q-hats as a TSV for downstream review."""
    qhat_df = pd.DataFrame(
        [{"regime": regime, "q_hat": value} for regime, value in calibrator.q_hats.items()]
    ).sort_values("regime")
    qhat_df.to_csv(os.path.join(out_dir, "regime_qhats.tsv"), sep="\t", index=False)


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
            calibrator.q_hats,
            os.path.join(out_dir, "regime_qhat_heatmap"),
            title="CQR Normalised q̂ by Gene Regime",
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
    out_dir: str,
    *,
    overwrite: bool = False,
) -> None:
    """Evaluate calibration on OOF data via leave-one-fold-out cross-fitting."""
    fold_data = cv_full.get("fold_data") or {}
    if not fold_data:
        print("  [skip] CV cross-fit calibration: no fold data available")
        return

    cv_out = ensure_dir(os.path.join(out_dir, "CV_CrossFit"))
    fold_rows: List[Dict[str, Any]] = []
    pooled_truth: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []
    pooled_lo: List[np.ndarray] = []
    pooled_hi: List[np.ndarray] = []
    pooled_lo_adj: List[np.ndarray] = []
    pooled_hi_adj: List[np.ndarray] = []

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
        summary.update(
            {
                "fold": heldout_fold,
                "n_regimes": calibrator.n_regimes,
                "qhat_min": float(min(calibrator.q_hats.values())) if calibrator.q_hats else np.nan,
                "qhat_median": float(np.median(list(calibrator.q_hats.values()))) if calibrator.q_hats else np.nan,
                "qhat_mean": float(np.mean(list(calibrator.q_hats.values()))) if calibrator.q_hats else np.nan,
                "qhat_max": float(max(calibrator.q_hats.values())) if calibrator.q_hats else np.nan,
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
        return

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

def run_examples(args: argparse.Namespace) -> None:
    """Generate the consolidated Examples outputs.

    Uses the full PLCA truth pool for the RNA-seq mixture fit and the
    existing PC-PDX study subset for the naive-example plots.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Examples")
    print("=" * 70)

    out_dir = _section_results_dir("Examples")
    plca_truth = _build_plca_truth()
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
    print(f"    Base  90% coverage: {summary['base_coverage_90']:.3f}")
    print(f"    CQR   90% coverage: {summary['adjusted_coverage_90']:.3f}")

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
            _ci_lo_gs = ci_lo if scatter_ci else None
            _ci_hi_gs = ci_hi if scatter_ci else None
            batch_gene_scatters(
                truth_wide, pred_wide, genes, gs_dir,
                subtypes_df=subtypes_df,
                ci_lo_wide=_ci_lo_gs, ci_hi_wide=_ci_hi_gs,
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

    # Build short→prefixed map and add prefixes to prediction columns
    pmap = _build_prefix_map(truth_wide.columns, cohort)

    subtypes_df = None
    sub_path = PROJECT_ROOT / "labels" / f"{cohort}_TFX_Subtype.tsv"
    if sub_path.exists():
        subtypes_df = load_subtypes(cohort)

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
    pred = full_data["pred_wide"]
    ci_lo = full_data.get("ci_lo_wide")
    ci_hi = full_data.get("ci_hi_wide")
    for key in ("pred_wide", "ci_lo_wide", "ci_hi_wide", *_MDN_PARAM_KEYS):
        _apply_prefix_map(full_data.get(key), pmap)
    label = f"{cohort}"

    suite_result = _run_holdout_suite(
        pred, truth_wide, label, cohort_out,
        ci_lo=ci_lo, ci_hi=ci_hi, subtypes_df=subtypes_df,
        heatmap_gene_sets=cohort_cfg["heatmap"],
        goi=cohort_cfg["goi"],
        n_workers=args.n_workers, overwrite=args.overwrite,
        summaries=summaries, scatter_ci=args.scatter_ci,
    )

    # Gene target scatters
    _ci_lo_tgt = ci_lo if args.scatter_ci else None
    _ci_hi_tgt = ci_hi if args.scatter_ci else None
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
        subtypes_df=subtypes_df,
        tfx_dict=tfx_dict,
        overwrite=args.overwrite,
    )

    # Return data for Collected analysis
    return {
        "cohort": cohort,
        "truth_wide": truth_wide,
        "pred_wide": pred,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
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
) -> Optional[Dict[str, Any]]:
    """Run patient-validation analysis for one cohort using OOF CV predictions."""
    truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
    if not truth_path.exists():
        print(f"    [skip] Truth not found: {cohort}")
        return None

    truth_wide = load_truth(cohort)
    cohort_samples = [s for s in cv_pred.columns if cohort_map_cv.get(s) == cohort]
    if not cohort_samples:
        print(f"    [skip] No CV samples for {cohort}")
        return None

    truth_c = truth_wide[[s for s in cohort_samples if s in truth_wide.columns]]
    pred_c = cv_pred[cohort_samples]
    ci_lo_c = (
        cv_ci_lo[[s for s in cohort_samples if s in cv_ci_lo.columns]]
        if cv_ci_lo is not None else None
    )
    ci_hi_c = (
        cv_ci_hi[[s for s in cohort_samples if s in cv_ci_hi.columns]]
        if cv_ci_hi is not None else None
    )
    mdn_tables = {
        key: (
            cv_full[key][[s for s in cohort_samples if s in cv_full[key].columns]]
            if cv_full.get(key) is not None else None
        )
        for key in _MDN_PARAM_KEYS
    }

    subtypes_df = None
    sub_path = PROJECT_ROOT / "labels" / f"{cohort}_TFX_Subtype.tsv"
    if sub_path.exists():
        subtypes_df = load_subtypes(cohort)

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
    _run_cv_suite(
        pred_c,
        truth_c,
        f"Proteus CV ({cohort})",
        cohort_out,
        ci_lo=ci_lo_c,
        ci_hi=ci_hi_c,
        subtypes_df=subtypes_df,
        heatmap_gene_sets=cohort_cfg["heatmap"],
        goi=cohort_cfg["goi"],
        gene_scatter_sets=cohort_cfg.get("scatter", []),
        gene_met=summary["_gene_metrics"],
        sample_met=summary["_sample_metrics"],
        n_workers=args.n_workers,
        overwrite=args.overwrite,
        scatter_ci=args.scatter_ci,
    )

    if cohort in _IHC_FILES:
        _run_ihc_analysis(
            truth_c,
            pred_c,
            cohort,
            cohort_out,
            subtypes_df=subtypes_df,
            tfx_dict=tfx_dict,
            overwrite=args.overwrite,
        )

    _save_summary_tsv([summary], os.path.join(cohort_out, "summary_metrics.tsv"))
    return {
        "cohort": cohort,
        "truth_wide": truth_c,
        "pred_wide": pred_c,
        "ci_lo": ci_lo_c,
        "ci_hi": ci_hi_c,
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
) -> None:
    """Scatter: IHC protein score vs RNA-Seq expression (truth or predicted)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, t as t_dist

    rho_val, p_val = spearmanr(ihc_vals, expr_vals)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    pal = load_palette()
    seen_subtypes: set[str] = set()
    point_x: List[float] = []
    point_y: List[float] = []
    for sample_name, ihc_val, expr_val in zip(sample_names, ihc_vals, expr_vals):
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

    if len(ihc_vals) > 2:
        coeffs = np.polyfit(ihc_vals, expr_vals, 1)
        x_line = np.linspace(float(ihc_vals.min()), float(ihc_vals.max()), 100)
        fit_fn = np.poly1d(coeffs)
        ax.plot(x_line, np.polyval(coeffs, x_line), "k-", lw=1, zorder=1)

        n = len(ihc_vals)
        if n > 3:
            x_vals = np.asarray(point_x, dtype=np.float64)
            y_vals = np.asarray(point_y, dtype=np.float64)
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
        ax.text(
        0.05,
        0.95,
        f"Spearman \u03c1 = {rho_val:.3f}\n{format_pval(p_val)}\nn = {len(ihc_vals)}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        )
    ax.text(0.95, 0.05, "Marker area ∝ TFx", transform=ax.transAxes,
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
    subtypes_df: Optional[pd.DataFrame] = None,
    tfx_dict: Optional[Dict[str, float]] = None,
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

        rho_truth_ihc = spearmanr(truth_arr, ihc_arr).statistic
        rho_pred_ihc = spearmanr(pred_arr, ihc_arr).statistic
        rho_truth_pred = spearmanr(truth_arr, pred_arr).statistic

        rows.append({
            "protein": protein,
            "gene": gene,
            "n": len(ihc_vals),
            "rho_truth_ihc": rho_truth_ihc,
            "rho_pred_ihc": rho_pred_ihc,
            "rho_truth_pred": rho_truth_pred,
        })

        # Per-protein scatter plots
        _maybe(_ihc_scatter, overwrite,
             ihc_arr, truth_arr, aligned_samples, protein, gene, "Truth",
             os.path.join(ihc_out, f"truth_vs_ihc_{protein}"),
             subtypes_df=subtypes_df, tfx_dict=tfx_dict)
        _maybe(_ihc_scatter, overwrite,
             ihc_arr, pred_arr, aligned_samples, protein, gene, "Predicted",
             os.path.join(ihc_out, f"pred_vs_ihc_{protein}"),
             subtypes_df=subtypes_df, tfx_dict=tfx_dict)

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


def _crosscohort_score_scatter(
    truth_scores: pd.Series,
    pred_scores: pd.Series,
    sample_meta: pd.DataFrame,
    name: str,
    *,
    title: str,
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

    if len(shared) >= 3:
        coeffs = np.polyfit(truth_vals.values, pred_vals.values, 1)
        x_line = np.linspace(lo - pad, hi + pad, 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), color="#CC3311", lw=1.5, alpha=0.8)

    r_p, p_p = pearsonr(truth_vals.values, pred_vals.values)
    r_s, p_s = spearmanr(truth_vals.values, pred_vals.values)
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {r_p:.3f}  ({format_pval(p_p)})\n"
        f"Spearman ρ = {r_s:.3f}  ({format_pval(p_s)})\n"
        f"n = {len(shared)}",
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

    ax.set_xlabel("Truth ssGSEA Score", fontsize=11)
    ax.set_ylabel("Predicted ssGSEA Score", fontsize=11)
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

    expected_cohorts = set(VALIDATION_PATIENT_HOLDOUT_COHORTS + VALIDATION_PATIENT_CV_COHORTS)
    ordered_cohort_data = [
        cd for cohort in VALIDATION_PATIENT_HOLDOUT_COHORTS + VALIDATION_PATIENT_CV_COHORTS
        for cd in cohort_data
        if cd["cohort"] == cohort
    ]
    cohort_data = [cd for cd in ordered_cohort_data if cd["cohort"] in expected_cohorts]
    missing_cohorts = [
        cohort for cohort in VALIDATION_PATIENT_HOLDOUT_COHORTS + VALIDATION_PATIENT_CV_COHORTS
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
    if ci_lo_parts and ci_hi_parts:
        combined_ci_lo = pd.concat(ci_lo_parts, axis=1).reindex(
            index=all_genes, columns=combined_pred.columns)
        combined_ci_hi = pd.concat(ci_hi_parts, axis=1).reindex(
            index=all_genes, columns=combined_pred.columns)
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
        for sample in cd["pred_wide"].columns:
            if sample in combined_pred.columns:
                collected_subtypes_rows.append({
                    "sample": sample,
                    "cohort": cd["cohort"],
                    "subtype": cd["cohort"],
                    "tfx": tfx_dict.get(sample, np.nan),
                    "status": status,
                })
                if sample in tfx_dict:
                    merged_tfx[sample] = tfx_dict[sample]

    collected_subtypes = pd.DataFrame(collected_subtypes_rows).set_index("sample")
    collected_tfx = merged_tfx if merged_tfx else None

    # Composite gene list: union of per-cancer target sets (deduped).
    composite_genes: List[str] = []
    seen_genes: set = set()
    for gs_name in _COLLECTED_TARGET_SETS:
        resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
        if not gene_list_available(resolved):
            continue
        for g in get_gene_list(resolved):
            if g not in seen_genes:
                composite_genes.append(g)
                seen_genes.add(g)

    if composite_genes:
        target_dir = ensure_dir(os.path.join(collected_out, "gene_target_scatters"))
        batch_gene_scatters(
            combined_truth, combined_pred, composite_genes, target_dir,
            subtypes_df=collected_subtypes, tfx_dict=collected_tfx,
            ci_lo_wide=combined_ci_lo, ci_hi_wide=combined_ci_hi,
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
                M=100,
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

        ne10_out = ensure_dir(os.path.join(shortlist_out, "NE10_ROC"))
        roc_cohort_order = [
            cohort for cohort in VALIDATION_PATIENT_HOLDOUT_COHORTS + VALIDATION_PATIENT_CV_COHORTS
            if cohort != "SCLC"
        ]

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
    weights: Optional[pd.DataFrame] = None,
    tfx_series: Optional[pd.Series] = None,
) -> Optional[pd.DataFrame]:
    """Run limma DE on predicted expression between two subtypes via rpy2.

    Parameters
    ----------
    tag : str
        Suffix appended to output filenames (e.g. "_raw", "_uncertainty-adjusted").
    weights : pd.DataFrame, optional
        Observation-level precision weights (genes x samples), passed to
        ``lmFit(..., weights=)``.
    tfx_series : pd.Series, optional
        Continuous TFx values indexed by sample name, added as a confounder.

    Returns the full topTable DataFrame, or None if skipped.
    """
    # Parse contrast to get the two groups
    groups_in_contrast = contrast.replace(" ", "").split("-")
    if len(groups_in_contrast) != 2:
        raise ValueError(f"Contrast must be 'GroupA-GroupB', got: {contrast}")
    required_groups = set(groups_in_contrast)

    # Filter to samples with a subtype in the contrast
    valid = subtypes_df[subtypes_df["subtype"].isin(required_groups)]
    common = [s for s in pred_wide.columns if s in valid.index]

    if tfx_series is not None:
        aligned_tfx = pd.to_numeric(tfx_series.reindex(common), errors="coerce")
        missing_tfx_samples = aligned_tfx.index[aligned_tfx.isna()].tolist()
        if missing_tfx_samples:
            print(f"    limma DE: dropped {len(missing_tfx_samples)} samples with missing TFx")
            common = [sample for sample in common if sample not in set(missing_tfx_samples)]

    if len(common) < 3:
        print(f"    [skip] limma DE: only {len(common)} samples in {required_groups}")
        return None

    expr = pred_wide[common]

    aligned_weights: Optional[pd.DataFrame] = None
    if weights is not None:
        aligned_weights = weights.reindex(index=pred_wide.index, columns=common)
        missing_weight_samples = aligned_weights.columns[aligned_weights.isna().all(axis=0)].tolist()
        if missing_weight_samples:
            print(
                f"    limma DE: dropped {len(missing_weight_samples)} samples with missing weights"
            )
            common = [sample for sample in common if sample not in set(missing_weight_samples)]
            if len(common) < 3:
                print(f"    [skip] limma DE: only {len(common)} samples in {required_groups}")
                return None
            expr = pred_wide[common]
            aligned_weights = weights.reindex(index=pred_wide.index, columns=common)

    # Gene filtering before limma
    n_total = expr.shape[0]

    # (1) Drop genes with any NA
    na_mask = expr.isna().any(axis=1)
    n_na = int(na_mask.sum())
    expr = expr[~na_mask]

    # (2) Keep genes with log2(CPM+1) > 1.0 in at least 3 samples
    expressed_mask = (expr > 1.0).sum(axis=1) >= 3
    n_low = int((~expressed_mask).sum())
    expr = expr[expressed_mask]

    n_weight_na = 0
    if aligned_weights is not None:
        aligned_weights = aligned_weights.reindex(index=expr.index, columns=common)
        weight_na_mask = aligned_weights.isna().any(axis=1)
        n_weight_na = int(weight_na_mask.sum())
        expr = expr.loc[~weight_na_mask]
        aligned_weights = aligned_weights.loc[~weight_na_mask]

    print(f"    Gene filter: {n_total} total → dropped {n_na} NA, "
          f"{n_low} low-expression, {n_weight_na} missing-weight → {expr.shape[0]} genes retained")

    if expr.shape[0] == 0:
        print("    [skip] limma DE: no genes passed filtering")
        return None

    groups = valid.loc[common, "subtype"].values
    counts = {g: int((groups == g).sum()) for g in groups_in_contrast}
    if any(count == 0 for count in counts.values()):
        print(f"    [skip] limma DE: missing one contrast group after filtering: {counts}")
        return None
    print(f"    limma-trend DE: {' vs '.join(f'{counts[g]} {g}' for g in groups_in_contrast)} "
          f"({expr.shape[0]} genes)")

    # rpy2 bridge
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    r_script = str(PROJECT_ROOT / "analysis" / "limma_de.R")
    ro.r.source(r_script)

    converter = ro.default_converter + pandas2ri.converter
    with localconverter(converter):
        r_expr = ro.conversion.get_conversion().py2rpy(expr)
        r_groups = ro.StrVector(list(groups))

        # Optional weights (genes x samples, aligned to expr)
        r_weights = ro.NULL
        if aligned_weights is not None:
            r_weights = ro.conversion.get_conversion().py2rpy(aligned_weights)

        # Optional TFx confounder (per-sample continuous)
        r_tfx = ro.NULL
        if tfx_series is not None:
            tfx_vals = pd.to_numeric(tfx_series.reindex(common), errors="coerce")
            r_tfx = ro.FloatVector(tfx_vals.values.tolist())

        de_r = ro.globalenv["run_limma_de"](
            r_expr, r_groups, contrast, r_weights, r_tfx)
        de_results = ro.conversion.get_conversion().rpy2py(de_r)

    # Save TSV
    contrast_tag = contrast.replace("-", "_vs_")
    de_path = os.path.join(out_dir, f"limma_DE_{contrast_tag}{tag}.tsv")
    de_results.to_csv(de_path, sep="\t")
    print("    DE results saved")

    # Volcano plot
    _maybe(volcano_plot, overwrite, de_results,
           os.path.join(out_dir, f"volcano_{contrast_tag}{tag}"),
           padj_thresh=padj_thresh, lfc_thresh=lfc_thresh,
           contrast_name=contrast)

    # Summary
    sig = de_results[
        (de_results["adj.P.Val"] < padj_thresh)
        & (de_results["logFC"].abs() > lfc_thresh)
    ]
    n_up = int((sig["logFC"] > 0).sum())
    n_down = int((sig["logFC"] < 0).sum())
    print(f"    DE summary: {len(sig)} significant genes "
          f"({n_up} {groups_in_contrast[0]}-up, {n_down} {groups_in_contrast[1]}-up) "
          f"at padj<{padj_thresh}, |logFC|>{lfc_thresh}")

    return de_results


# ---------------------------------------------------------------------------
# Discovery_Patient: BC-Ghana receptor expression + PAM50
# ---------------------------------------------------------------------------

_BC_GHANA_RECEPTOR_PAIRS = [
    ("HER2_status", "ERBB2", "HER2"),
    ("ER_status",   "ESR1",  "ER"),
    ("PR_stats",    "PGR",   "PR"),
]


def _run_bc_ghana_analysis(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    tag: str = "",
    mc_draws: Optional[List[pd.DataFrame]] = None,
) -> None:
    """Receptor-gene expression boxplots and PAM50 classification for BC-Ghana.

    Parameters
    ----------
    tag : str
        Suffix appended to output filenames.
    mc_draws : list[DataFrame], optional
        MC draws for uncertainty-adjusted PAM50 (uses ``pam50_classify_mc``).
    """
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

    # --- PAM50 classification ---
    pam50_out = ensure_dir(os.path.join(model_out, f"PAM50{tag}"))
    pred_sub = pred[common_samples]

    if mc_draws is not None:
        # Uncertainty-adjusted: classify each MC draw, aggregate
        draws_sub = [d.reindex(columns=common_samples) for d in mc_draws]
        pam50_results = pam50_classify_mc(draws_sub, pred_sub)
        print(f"    PAM50 MC classification saved ({len(pam50_results)} samples)")
    else:
        pam50_results = pam50_classify(pred_sub)
        print(f"    PAM50 classification saved ({len(pam50_results)} samples)")

    pam50_results.to_csv(
        os.path.join(pam50_out, f"pam50_classification{tag}.tsv"),
        sep="\t", index=False)

    if pam50_results.empty or "sample" not in pam50_results.columns:
        print(f"    [skip] PAM50 confusion matrix: no classified samples")
        return

    # Confusion matrix: predicted vs metadata subtype
    truth_subtypes = meta.loc[common_samples, "subtype"]
    pam50_indexed = pam50_results.set_index("sample")
    matched = truth_subtypes.index.intersection(pam50_indexed.index)
    if len(matched) >= 3:
        y_true = truth_subtypes.loc[matched].values
        y_pred = pam50_indexed.loc[matched, "predicted_subtype"].values
        all_labels = sorted(set(y_true) | set(y_pred))
        _maybe(
            confusion_matrix_heatmap, overwrite,
            y_true, y_pred,
            os.path.join(pam50_out, f"pam50_confusion_matrix{tag}"),
            class_order=all_labels,
            title="PAM50 classification — BC-Ghana",
        )
        correct = (y_true == y_pred).sum()
        print(f"    PAM50 accuracy: {correct}/{len(matched)} "
              f"({100 * correct / len(matched):.1f}%)")


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

    cox_scores_mv = None
    if tfx_series_v is not None:
        tfx_s = tfx_series_v.reindex(matched_pids)
        if tfx_s.notna().any():
            cox_scores_mv = compute_cox_multivariate(time_s, event_s, covariates, tfx_s)
            cox_scores_mv.to_csv(
                os.path.join(survival_out, f"cox_{output_stem}_results_mv{tag}.tsv"),
                sep="\t", index=False,
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
    sub_path = PROJECT_ROOT / "labels" / "CRPC-Pluvicto_TFX_Subtype.tsv"
    if sub_path.exists():
        sub_df = load_subtypes("CRPC-Pluvicto")
        if "tfx" in sub_df.columns:
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

    corr_out = ensure_dir(os.path.join(model_out, "clinical_correlations"))

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
            )
    else:
        print("    [skip] FOLH1 not in predictions")

    # --- CCP31 score vs FDG ---
    ccp31_name = STANDARD_GENE_SETS.get("CCP31", "CCP31")
    if gene_list_available(ccp31_name):
        ccp31_genes = get_gene_list(ccp31_name)
        ccp31_score = compute_geneset_score(pred[samples], ccp31_genes)
        ccp31_series = pd.Series(ccp31_score.values, index=pids)
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
                xlabel="CCP31 score (mean log₂CPM+1)",
                ylabel=fdg_col,
                title=f"CCP31 vs {fdg_col}",
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
    weights: Optional[pd.DataFrame] = None,
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
        de_results = _run_limma_de(
            pred[visit_meta.index],
            subtype_df,
            model_out,
            overwrite,
            contrast=contrast,
            tag=contrast_tag,
            weights=weights,
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
    pred_var: Optional[pd.DataFrame] = None,
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
        limma_weights = None
        if pred_var is not None:
            limma_weights = (1.0 / pred_var[pfizer_meta.index].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        interesting_hallmarks_adj = _pfizer_run_limma_and_gsea(
            pred_subset,
            pfizer_meta,
            model_out,
            overwrite,
            tag_prefix="_uncertainty-adjusted",
            weights=limma_weights,
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
) -> None:
    """Run Discovery_Patient analysis for a single cohort (no truth)."""
    cohort_out = ensure_dir(os.path.join(out_dir, cohort))

    subtypes_df = None
    tfx_dict = None
    sub_dict = None
    label_name = subtype_label or cohort
    sub_path = PROJECT_ROOT / "labels" / f"{label_name}_TFX_Subtype.tsv"
    if sub_path.exists():
        subtypes_df = load_subtypes(label_name)
        if "tfx" in subtypes_df.columns:
            tfx_dict = subtypes_df["tfx"].to_dict()
        if "subtype" in subtypes_df.columns:
            sub_dict = subtypes_df["subtype"].to_dict()

    for model_name, path_pattern in _DISCOVERY_MODELS:
        path = PROJECT_ROOT / path_pattern.format(cohort=cohort)
        if not path.exists():
            print(f"    [skip] {model_name}: not found")
            continue

        # Load full MDN parameters for uncertainty propagation
        full_data = load_holdout_predictions_proteus_full(str(path))
        pred = full_data["pred_wide"]
        ci_lo = full_data.get("ci_lo_wide")
        ci_hi = full_data.get("ci_hi_wide")

        # Build MC draws if MDN params available
        mc_draws: Optional[List[pd.DataFrame]] = None
        pred_var: Optional[pd.DataFrame] = None
        has_mdn = all(full_data.get(k) is not None for k in (
            "p0_wide", "pi_gamma_wide", "pi_normal_wide",
            "gamma_beta_wide", "normal_mean_wide", "normal_stdev_wide"))
        if has_mdn:
            mc_draws = mdn_monte_carlo_sample(
                full_data["p0_wide"], full_data["pi_gamma_wide"],
                full_data["pi_normal_wide"], full_data["gamma_beta_wide"],
                full_data["normal_mean_wide"], full_data["normal_stdev_wide"],
                M=100, seed=42)
            pred_var = mdn_predictive_variance(
                full_data["p0_wide"], full_data["pi_gamma_wide"],
                full_data["pi_normal_wide"], full_data["gamma_beta_wide"],
                full_data["normal_mean_wide"], full_data["normal_stdev_wide"])

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
            if pred_var is not None:
                pred_var = pred_var[keep]
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

            # _uncertainty-adjusted: TFx confounder + inverse-variance weights
            if pred_var is not None:
                limma_weights = 1.0 / pred_var
                de_adj = _run_limma_de(
                    pred, subtypes_df, model_out, args.overwrite,
                    tag="_uncertainty-adjusted",
                    weights=limma_weights, tfx_series=tfx_for_limma)
                if de_adj is not None:
                    _run_gsea(de_adj, model_out, args.overwrite,
                              tag="_uncertainty-adjusted")

        # --- BC-Ghana: receptor expression boxplots + PAM50 classification ---
        if cohort == "BC-Ghana":
            _run_bc_ghana_analysis(pred, model_out, args.overwrite,
                                   tag="_raw")
            if mc_draws is not None:
                _run_bc_ghana_analysis(
                    pred, model_out, args.overwrite,
                    tag="_uncertainty-adjusted", mc_draws=mc_draws)

        # --- CRPC-Pluvicto: ssGSEA, survival, clinical correlations ---
        if cohort == "CRPC-Pluvicto":
            # _raw Hallmark + shortlist scores + survival
            ssgsea_raw = _run_pluvicto_ssgsea(
                pred, model_out, args.overwrite, tag="_raw")
            shortlist_raw = _run_pluvicto_shortlist_scores(
                pred, model_out, args.overwrite, tag="_raw")
            _run_pluvicto_survival(
                pred, ssgsea_raw, shortlist_raw, model_out, args.overwrite, tag="_raw")

            # _uncertainty-adjusted Hallmark + shortlist scores + survival
            if mc_draws is not None:
                ssgsea_mc = _run_pluvicto_ssgsea_mc(
                    mc_draws, model_out, args.overwrite)
                shortlist_mc = _run_pluvicto_shortlist_scores_mc(
                    mc_draws, model_out, args.overwrite)
                _run_pluvicto_survival(
                    pred, ssgsea_mc, shortlist_mc, model_out, args.overwrite,
                    tag="_uncertainty-adjusted")

            _run_pluvicto_clinical_correlations(
                pred, model_out, args.overwrite)

        if cohort == "CRPC-Pfizer":
            _run_crpc_pfizer_analysis(
                pred,
                model_out,
                args.overwrite,
                mc_draws=mc_draws,
                pred_var=pred_var,
            )


def run_patient_discovery(args: argparse.Namespace) -> None:
    """Discovery cohort analysis: predictions only (no biopsy truth).

    Iterates over all configured discovery cohorts and generates heatmaps,
    expression profiles, and prediction exports for each available model.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Discovery_Patient")
    print("=" * 70)

    out_dir = _section_results_dir("Discovery_Patient")

    # Default: iterate over all discovery cohorts
    for cohort, subtype_label in _DISCOVERY_COHORTS:
        print(f"\n  ── Cohort: {cohort} ──")
        _run_discovery_cohort(cohort, subtype_label, out_dir, args)

    print("\n  Done")


def _load_main_plca_cv_full() -> Optional[Dict[str, Any]]:
    """Load and prefix-normalize the canonical PLCA CV outputs."""
    cv_dir = PROJECT_ROOT / VALIDATION_PDX_CV_RESULTS_DIR
    if not os.path.isdir(cv_dir):
        print(f"  [skip] PLCA CV results not found: {VALIDATION_PDX_CV_RESULTS_DIR}")
        return None

    fold_splits = _load_plca_fold_splits()
    cohort_map: Dict[str, str] = dict(zip(fold_splits["Sample"], fold_splits["Cohort"]))
    prefix_map: Dict[str, str] = {}
    for _, row in fold_splits.iterrows():
        prefixed = row["Sample"]
        short = str(strip_cohort_prefix(pd.Index([prefixed]), row["Cohort"])[0])
        prefix_map[short] = prefixed

    cv_full = load_cv_predictions_proteus_full(str(cv_dir))
    for key in (
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
    ):
        table = cv_full.get(key)
        if table is not None:
            _apply_prefix_map(table, prefix_map)

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

    out_dir = _section_results_dir("Calibration")
    canonical_truth = _build_plca_truth()
    if canonical_truth.empty:
        print("  [skip] No pooled PLCA truth available")
        return

    cv_bundle = _load_main_plca_cv_full()
    if cv_bundle is None:
        return

    cv_full = cv_bundle["cv_full"]
    calibrator = _fit_global_calibrator(
        canonical_truth,
        cv_full,
        overwrite=args.overwrite,
        out_dir=out_dir,
    )
    if calibrator is None:
        print("  [skip] Unable to fit calibration model")
        return

    print(f"  {calibrator.summary()}")
    _run_cv_crossfit_calibration(cv_full, out_dir, overwrite=args.overwrite)

    truth_pool, ho_preds = _load_validation_pdx_holdout_data(canonical_truth)
    holdout_pred_parts: List[pd.DataFrame] = []
    holdout_truth_parts: List[pd.DataFrame] = []
    holdout_lo_parts: List[pd.DataFrame] = []
    holdout_hi_parts: List[pd.DataFrame] = []
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

    pooled_holdout_pred = _concat_common_gene_frames(holdout_pred_parts)
    pooled_holdout_truth = _concat_common_gene_frames(holdout_truth_parts)
    pooled_holdout_lo = _concat_common_gene_frames(holdout_lo_parts)
    pooled_holdout_hi = _concat_common_gene_frames(holdout_hi_parts)
    if (
        pooled_holdout_pred is not None
        and pooled_holdout_truth is not None
        and pooled_holdout_lo is not None
        and pooled_holdout_hi is not None
    ):
        holdout_out = ensure_dir(os.path.join(out_dir, "PDX_Holdout"))
        _run_calibration_evaluation(
            pooled_holdout_truth,
            pooled_holdout_pred,
            pooled_holdout_lo,
            pooled_holdout_hi,
            calibrator,
            "Validation_PDX",
            holdout_out,
            overwrite=args.overwrite,
        )


def run_validation_pdx(args: argparse.Namespace) -> None:
    """Validation_PDX analysis using PLCA main-model CV + true holdout PDX samples."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Validation_PDX")
    print("=" * 70)

    out_dir = _section_results_dir("Validation_PDX")
    n_workers = args.n_workers

    canonical_truth = _build_plca_truth()
    if canonical_truth.empty:
        print("  [skip] No canonical PLCA truth available")
        return

    cv_bundle = _load_main_plca_cv_full()
    if cv_bundle is None:
        return

    cv_full = cv_bundle["cv_full"]
    cohort_map = cv_bundle["cohort_map"]
    proteus_pred = cv_full["pred_wide"]
    proteus_truth = cv_full.get("truth_wide")
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

    calibrator: Optional[RegimeCalibrator] = None
    if proteus_ci_lo is not None and proteus_ci_hi is not None and proteus_truth is not None:
        calibrator = _fit_global_calibrator(canonical_truth, cv_full)
        if calibrator is not None:
            print(f"  {calibrator.summary()}")

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

    truth_pool, ho_preds = _load_validation_pdx_holdout_data(canonical_truth)

    if calibrator is not None:
        proteus_ci_lo, proteus_ci_hi = calibrator.apply(proteus_ci_lo, proteus_ci_hi)
        for cohort_name, ho_data in ho_preds.items():
            if ho_data.get("ci_lo_wide") is None:
                continue
            ho_data["ci_lo_wide"], ho_data["ci_hi_wide"] = calibrator.apply(
                ho_data["ci_lo_wide"],
                ho_data["ci_hi_wide"],
            )

    all_summaries: List[Dict[str, Any]] = []
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
        ho_data = ho_preds.get(cohort_name)
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
            cohort_summaries.append(ho_summary)
            all_summaries.append(ho_summary)

        combined_summary = cohort_summary_table(
            combined_truth,
            combined_pred,
            label=f"Combined ({cohort_name})",
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
        )

        if ho_data is not None and reliability_model is not None:
            _run_reliability_evaluation(
                ho_data,
                canonical_truth,
                truth_pool,
                reliability_model,
                cohort_name,
                cohort_out,
                overwrite=args.overwrite,
            )

        _save_summary_tsv(cohort_summaries, os.path.join(cohort_out, "summary_metrics.tsv"))

    if all_summaries:
        print("\n" + "─" * 70)
        print_summary_table(all_summaries)
        _save_summary_tsv(all_summaries, os.path.join(out_dir, "summary_metrics.tsv"))


def run_validation_patient(args: argparse.Namespace) -> None:
    """Validation_Patient analysis using PLCA main-model holdout and OOF CV data."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Validation_Patient")
    print("=" * 70)

    out_dir = _section_results_dir("Validation_Patient")
    cohort_data: List[Dict[str, Any]] = []

    for cohort in VALIDATION_PATIENT_HOLDOUT_COHORTS:
        print(f"\n  ── Cohort: {cohort} (holdout) ──")
        result = _run_patient_validation_cohort(
            cohort,
            VALIDATION_PATIENT_HOLDOUT_PATH_TEMPLATE,
            out_dir,
            args,
        )
        if result is not None:
            cohort_data.append(result)

    cv_bundle = _load_main_plca_cv_full()
    if cv_bundle is not None:
        cv_full = cv_bundle["cv_full"]
        cv_pred = cv_full["pred_wide"]
        cv_ci_lo = cv_full.get("ci_lo_wide")
        cv_ci_hi = cv_full.get("ci_hi_wide")
        cohort_map_cv = cv_bundle["cohort_map"]

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
            )
            if result is not None:
                cohort_data.append(result)

    if len(cohort_data) >= 2:
        _run_collected_validation(cohort_data, out_dir, args, label="CrossCohort")

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
    ecdna_table: pd.DataFrame,
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
    ecdna_view = ecdna_table[ecdna_table["model"].isin(model_order)].copy()

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

    if not ecdna_view.empty:
        _maybe(
            benchmark_ecdna_lines,
            overwrite,
            ecdna_view,
            os.path.join(view_out, "benchmark_ecdna_lines"),
            model_order=model_order,
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
        gene_list = get_gene_list(gs_key)
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

        # ── Risk coverage curve: coverage vs PPV / FPR ──
        # Proteus models with CIs get smooth lines (sweep CI width);
        # comparators (XGBoost, EPIC, Naive) get single points at 100%.
        risk_data: Dict[str, Dict[str, np.ndarray]] = {}
        _proteus_model_names: List[str] = []

        for model_name in model_order:
            if model_name not in model_preds:
                continue
            pw = model_preds[model_name]
            cg_r = [g for g in common_genes if g in pw.index]
            cs_r = [s for s in pw.columns if s in truth_wide.columns]
            if len(cg_r) < 5 or len(cs_r) < 3:
                continue

            t_r = truth_wide.loc[cg_r, cs_r].values.flatten()
            p_r = pw.loc[cg_r, cs_r].values.flatten()

            lo_r = hi_r = None
            mdn_flat: Dict[str, Optional[np.ndarray]] = {
                "p0_flat": None,
                "pi_gamma_flat": None,
                "pi_normal_flat": None,
                "gamma_beta_flat": None,
                "normal_mean_flat": None,
                "normal_stdev_flat": None,
            }
            if model_name in proteus_ci:
                ci_lo, ci_hi = proteus_ci[model_name]
                if ci_lo is not None and ci_hi is not None:
                    cg_ci = [g for g in cg_r
                             if g in ci_lo.index and g in ci_hi.index]
                    cs_ci = [s for s in cs_r
                             if s in ci_lo.columns and s in ci_hi.columns]
                    if len(cg_ci) >= 5 and len(cs_ci) >= 3:
                        lo_r = ci_lo.loc[cg_ci, cs_ci].values.flatten()
                        hi_r = ci_hi.loc[cg_ci, cs_ci].values.flatten()
                        t_r = truth_wide.loc[cg_ci, cs_ci].values.flatten()
                        p_r = pw.loc[cg_ci, cs_ci].values.flatten()
                        _proteus_model_names.append(model_name)
                        if model_name in proteus_mdn:
                            mdn_tables = proteus_mdn[model_name]
                            required_keys = (
                                "p0_wide",
                                "pi_gamma_wide",
                                "pi_normal_wide",
                                "gamma_beta_wide",
                                "normal_mean_wide",
                                "normal_stdev_wide",
                            )
                            if all(mdn_tables.get(key) is not None for key in required_keys):
                                cg_mdn = [
                                    g for g in cg_ci
                                    if all(g in mdn_tables[key].index for key in required_keys)
                                ]
                                cs_mdn = [
                                    s for s in cs_ci
                                    if all(s in mdn_tables[key].columns for key in required_keys)
                                ]
                                if len(cg_mdn) >= 5 and len(cs_mdn) >= 3:
                                    t_r = truth_wide.loc[cg_mdn, cs_mdn].values.flatten()
                                    p_r = pw.loc[cg_mdn, cs_mdn].values.flatten()
                                    lo_r = ci_lo.loc[cg_mdn, cs_mdn].values.flatten()
                                    hi_r = ci_hi.loc[cg_mdn, cs_mdn].values.flatten()
                                    mdn_flat = {
                                        "p0_flat": mdn_tables["p0_wide"].loc[cg_mdn, cs_mdn].values.flatten(),
                                        "pi_gamma_flat": mdn_tables["pi_gamma_wide"].loc[cg_mdn, cs_mdn].values.flatten(),
                                        "pi_normal_flat": mdn_tables["pi_normal_wide"].loc[cg_mdn, cs_mdn].values.flatten(),
                                        "gamma_beta_flat": mdn_tables["gamma_beta_wide"].loc[cg_mdn, cs_mdn].values.flatten(),
                                        "normal_mean_flat": mdn_tables["normal_mean_wide"].loc[cg_mdn, cs_mdn].values.flatten(),
                                        "normal_stdev_flat": mdn_tables["normal_stdev_wide"].loc[cg_mdn, cs_mdn].values.flatten(),
                                    }

            rd = compute_risk_coverage_data(
                t_r,
                p_r,
                lo_r,
                hi_r,
                threshold=_ONOFF_THRESHOLD,
                **mdn_flat,
            )
            if len(rd["coverage"]) > 0:
                risk_data[model_name] = rd

        if risk_data:
            _maybe(risk_coverage_plot, overwrite,
                   risk_data,
                   os.path.join(out_dir,
                                f"RiskCoverage_{gs_display}_{cohort}"),
                   title=f"Risk Coverage — {gs_display} ({cohort})",
                   model_order=list(model_order),
                   proteus_models=_proteus_model_names)
            print(f"    RiskCoverage {gs_display}: "
                  f"{len(risk_data)} models")

        # ── 3-way confusion summary: ON / OFF / INDETERMINATE ──
        confusion_results: Dict[str, Dict[str, Any]] = {}
        confusion_order: List[str] = []

        for model_name in model_order:
            if model_name not in model_preds:
                continue
            if model_name == "Naive":
                continue
            pw = model_preds[model_name]
            common_samples = [s for s in pw.columns if s in truth_wide.columns]
            cg = [g for g in common_genes if g in pw.index]
            if not common_samples or len(cg) < 5:
                continue

            t = truth_wide.loc[cg, common_samples].values.flatten()
            p = pw.loc[cg, common_samples].values.flatten()

            lo_c = hi_c = None
            if model_name in proteus_ci:
                ci_lo, ci_hi = proteus_ci[model_name]
                if ci_lo is not None and ci_hi is not None:
                    cg_ci = [g for g in cg
                             if g in ci_lo.index and g in ci_hi.index]
                    cs_ci = [s for s in common_samples
                             if s in ci_lo.columns and s in ci_hi.columns]
                    if len(cg_ci) >= 5 and len(cs_ci) >= 3:
                        lo_c = ci_lo.loc[cg_ci, cs_ci].values.flatten()
                        hi_c = ci_hi.loc[cg_ci, cs_ci].values.flatten()
                        t = truth_wide.loc[cg_ci, cs_ci].values.flatten()
                        p = pw.loc[cg_ci, cs_ci].values.flatten()

            if lo_c is not None and hi_c is not None:
                binary_name = f"{model_name} (binary)"
                ua_name = f"{model_name} (uncertainty-aware)"

                binary_confusion = compute_onoff_confusion(
                    t, p, threshold=_ONOFF_THRESHOLD,
                )
                ua_confusion = compute_onoff_confusion(
                    t, p, lo_c, hi_c, threshold=_ONOFF_THRESHOLD,
                )

                if binary_confusion.get("n_total", 0) > 0:
                    confusion_results[binary_name] = binary_confusion
                    confusion_order.append(binary_name)
                if ua_confusion.get("n_total", 0) > 0:
                    confusion_results[ua_name] = ua_confusion
                    confusion_order.append(ua_name)
            else:
                d = compute_onoff_confusion(
                    t, p, threshold=_ONOFF_THRESHOLD,
                )
                if d.get("n_total", 0) > 0:
                    confusion_results[model_name] = d
                    confusion_order.append(model_name)

        if confusion_results:
            # Save TSV
            conf_rows = []
            for m in confusion_order:
                d = confusion_results[m]
                conf_rows.append({
                    "model": m,
                    "n_total": d["n_total"],
                    "n_on": d["n_on"], "n_off": d["n_off"],
                    "n_indet": d["n_indet"],
                    "accuracy": d.get("accuracy", float("nan")),
                    "ppv": d.get("ppv", float("nan")),
                    "npv": d.get("npv", float("nan")),
                    "fpr": d.get("fpr", float("nan")),
                    "coverage": d.get("coverage", float("nan")),
                })
            conf_df = pd.DataFrame(conf_rows)
            conf_df.to_csv(
                os.path.join(out_dir,
                             f"OnOffConfusion_{gs_display}_{cohort}.tsv"),
                sep="\t", index=False,
            )

            _maybe(onoff_confusion_summary_plot, overwrite,
                   confusion_results,
                   os.path.join(out_dir,
                                f"OnOffConfusion_{gs_display}_{cohort}"),
                     title=f"On/Off Confusion — {gs_display} ({cohort})",
                     model_order=confusion_order)
            print(f"    OnOffConfusion {gs_display}: "
                  f"{len(confusion_results)} models")


def run_benchmarking_panft(args: argparse.Namespace) -> None:
    """Benchmarking_Patient analysis on real patient cohorts.

    Compares models on CRPC-WCDT and BLCA-RA biopsies. Metrics are computed
    per cohort and written into COMPLETE and SUBSET output views.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Benchmarking_Patient")
    print("=" * 70)

    out_dir = _section_results_dir("Benchmarking_Patient")

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
                pred = full_data["pred_wide"]
                ci_lo = full_data.get("ci_lo_wide")
                ci_hi = full_data.get("ci_hi_wide")
                _apply_prefix_map(pred, pmap)
                if ci_lo is not None:
                    _apply_prefix_map(ci_lo, pmap)
                if ci_hi is not None:
                    _apply_prefix_map(ci_hi, pmap)
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
                    if table is not None:
                        _apply_prefix_map(table, pmap)
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
    boxplots, histograms, and eCDNA line plots.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Benchmarking_Synthetic")
    print("=" * 70)

    out_dir = _section_results_dir("Benchmarking_Synthetic")
    n_workers = args.n_workers

    # ── Load model predictions and build truth ──────────────────────────
    model_preds: Dict[str, pd.DataFrame] = {}  # model_name -> wide (genes × samples)

    for model_name, rel_path in SYNTHETIC_PROTEUS_MODEL_PATHS.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"  [skip] {model_name}: {rel_path} not found")
            continue
        pred, _, _ = load_holdout_predictions_proteus(str(path))
        model_preds[model_name] = pred
        print(f"  Loaded {model_name}: {pred.shape[0]} genes × {pred.shape[1]} samples")

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

    # ── Build eCDNA table (aggregate across depth for line plots) ───────
    # eCDNA = TFx_numeric * depth_numeric
    tfx_num = {f"TF{v}": float(v) for v in ["0.50", "0.30", "0.10", "0.05", "0.03", "0.01"]}
    depth_num = {"10x": 10.0, "5x": 5.0}
    ecdna_rows = []
    for _, row in metrics_table.iterrows():
        tfx_val = tfx_num.get(row["tfx"])
        dep_val = depth_num.get(row["depth"])
        if tfx_val is not None and dep_val is not None:
            r = row.to_dict()
            r["ecdna"] = tfx_val * dep_val
            ecdna_rows.append(r)
    ecdna_table = pd.DataFrame(ecdna_rows)

    # ── Plots ──────────────────────────────────────────────────────────
    print("  Generating benchmark plots ...")
    for view_name, view_models in benchmark_model_views().items():
        _write_synthetic_benchmark_view(
            metrics_table,
            gene_rho_table,
            sample_rho_table,
            ecdna_table,
            out_dir,
            view_name,
            [m for m in view_models if m in set(metrics_table["model"])],
            args.overwrite,
            var_map,
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


def _build_plca_truth() -> pd.DataFrame:
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
        "--n_bootstraps", type=int, default=100,
        help="Legacy bootstrap count; retained in the CLI but unused by active section workflows.",
    )
    parser.add_argument(
        "--scatter_ci", action="store_true",
        help="Show CI error bars on sample/gene scatter plots (off by default).",
    )

    args = parser.parse_args()

    available_cpus = max(1, os.cpu_count() or 1)
    args.n_workers = available_cpus
    print(f"Available CPUs: {available_cpus}  |  Using all workers")

    t_start = time.time()

    if args.group == "all":
        for name, fn in GROUPS.items():
            group_dir = PROJECT_ROOT / "analysis" / "results" / name
            if group_dir.is_dir():
                print(f"\n  [skip] {name}: {group_dir} already exists")
                continue
            fn(args)
    else:
        GROUPS[args.group](args)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
