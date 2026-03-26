#!/usr/bin/env python3
"""
run_full_analysis.py
Proteus analysis framework — main orchestrator.

Usage examples:
    python analysis/run_full_analysis.py --group BTS-CV
    python analysis/run_full_analysis.py --group Patient_Validation
    python analysis/run_full_analysis.py --group Patient_Validation \
        --proteus_holdout results_holdout/BioDVN/CRPC-WCDT \
        --truth_cohort CRPC-WCDT
    python analysis/run_full_analysis.py --group Patient_Discovery
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
    build_study_truth_pool,
    clean_epicseq_sample_names,
    cohort_summary_table,
    compute_coverage_curve,
    compute_gene_variance,
    compute_subset_metrics,
    empirical_coverage,
    ensure_dir,
    gene_level_metrics,
    gene_list_available,
    generate_null_predictions,
    get_gene_list,
    GENE_SET_BREAKS,
    load_cv_predictions_proteus,
    load_cv_predictions_proteus_full,
    load_cv_predictions_background,
    load_cv_predictions_xgboost,
    load_fold_splits,
    load_holdout_predictions_epicseq,
    load_holdout_predictions_proteus,
    load_holdout_predictions_proteus_full,
    load_holdout_predictions_xgboost,
    load_pdx_truth_for_phs,
    load_subtypes,
    load_truth,
    parse_phs_sample,
    PC_PDX_STUDY,
    PDX_HOLDOUT_SAMPLES,
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
    benchmark_ecdna_lines,
    benchmark_grid_boxplots,
    benchmark_grid_histograms,
    double_heatmap,
    gene_target_scatter,
    metric_distribution,
    metric_distribution_by_variance,
    plot_calibration_coverage,
    plot_error_vs_width,
    plot_regime_coverage_heatmap,
    plot_regime_qhat_heatmap,
    plot_reliability_scatter,
    plot_reliability_weights,
    plot_retention_curve,
    sample_expression_kde,
    sample_scatter,
    single_heatmap,
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

# Cohort-specific analysis configuration for BTS-CV splits
_COHORT_CONFIG = {
    "PC-PDX": {
        "heatmap_gene_sets": ["PC-Phenotyping"],
        "goi": DEFAULT_GOI_PC,
        "gene_scatter_sets": ["Targets_PC"],
        "study_samples": PC_PDX_STUDY,
    },
    "SCLC-PDX": {
        "heatmap_gene_sets": ["Targets_SCLC"],
        "goi": DEFAULT_GOI_SCLC,
        "gene_scatter_sets": ["Targets_SCLC"],
        "study_samples": SCLC_PDX_STUDY,
    },
}


# ---------------------------------------------------------------------------
# Analysis group: BTS-CV
# ---------------------------------------------------------------------------

_BTS_CV_PATH = "results_training/proteus_base-model_cv"
_BTS_HO_PATHS = {
    "PC-PDX": "results_holdout/BioDVN/PC-PDX/predictions_tumor.tsv",
    "SCLC-PDX": "results_holdout/BioDVN/SCLC-PDX/predictions_tumor.tsv",
}


def run_bts_cv(args: argparse.Namespace) -> None:
    """BTS cross-validation analysis with holdout PDX samples.

    Results are split by cohort (PC-PDX, SCLC-PDX).  Each cohort includes
    both the OOF cross-validation predictions (20 samples) and the true
    holdout PDX predictions (5 samples).  Metrics are reported separately
    for CV, HO, and Combined.

    Pipeline stages:
      1. Load OOF predictions (with full MDN parameters) and holdout data.
      2. CQR regime-based calibration on all OOF samples (PDX + PHT).
      3. Gene-level reliability model fitted on OOF features.
      4. Exclude PHT → per-cohort evaluation + plots.
      5. Calibration evaluation on holdout.
      6. Reliability evaluation on holdout.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: BTS-CV (Cross-Validation + Holdout)")
    print("=" * 70)

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / "BTS-CV"))
    summaries: List[Dict[str, Any]] = []
    n_workers = args.n_workers

    # Load fold-split table to map samples → cohort
    fold_splits = load_fold_splits()
    cohort_map: Dict[str, str] = dict(zip(fold_splits["Sample"], fold_splits["Cohort"]))

    subtypes_df = _build_bts_subtypes()
    canonical_truth = _build_bts_truth()
    print(f"  Canonical truth: {canonical_truth.shape[0]} genes × {canonical_truth.shape[1]} samples")

    def _split_by_cohort(
        pred: pd.DataFrame,
        truth: pd.DataFrame,
        ci_lo: Optional[pd.DataFrame],
        ci_hi: Optional[pd.DataFrame],
        cohort: str,
    ) -> Tuple:
        """Extract only tumor samples matching ``cohort`` (no HD-L)."""
        samples = sorted(s for s in pred.columns if cohort_map.get(s) == cohort)
        if not samples:
            return None, None, None, None
        p = pred[[s for s in samples if s in pred.columns]]
        t = truth[[s for s in samples if s in truth.columns]]
        cl = ci_lo[[s for s in samples if s in ci_lo.columns]] if ci_lo is not None else None
        ch = ci_hi[[s for s in samples if s in ci_hi.columns]] if ci_hi is not None else None
        return p, t, cl, ch

    # ── Load Proteus CV predictions (full: includes p0, per-fold data) ──
    cv_dir = str(PROJECT_ROOT / _BTS_CV_PATH)
    print(f"\nLoading Proteus CV (full) from: {cv_dir}")
    cv_full = load_cv_predictions_proteus_full(cv_dir)
    proteus_pred = cv_full["pred_wide"]
    _pred_truth = cv_full["truth_wide"]
    proteus_ci_lo = cv_full["ci_lo_wide"]
    proteus_ci_hi = cv_full["ci_hi_wide"]
    proteus_p0 = cv_full["p0_wide"]
    sample_fold_map = cv_full["sample_fold"]

    # ──────────────────────────────────────────────────────────────────────
    # CQR regime-based calibration on ALL OOF samples (40 PDX + PHT)
    # ──────────────────────────────────────────────────────────────────────
    # Rationale: computing gene regimes and normalised conformal scores on
    # ALL out-of-fold samples (PDX + PHT admixtures) provides the richest
    # calibration signal across different TFx regimes.  PHT samples are
    # excluded *after* calibration for evaluation and plotting.
    calibrator: Optional[RegimeCalibrator] = None
    regime_df: Optional[pd.DataFrame] = None
    if proteus_ci_lo is not None and proteus_ci_hi is not None:
        # Align truth to the prediction grid for calibration
        cal_truth = _pred_truth.reindex(
            index=proteus_ci_lo.index, columns=proteus_ci_lo.columns,
        )
        # Gene regimes from the canonical development truth (50 PDX samples)
        regime_df = compute_gene_regimes(canonical_truth)
        print(f"  Gene regimes: {regime_df['regime'].nunique()} unique regimes "
              f"from {len(regime_df)} genes")

        # Fit the CQR calibrator on OOF data
        calibrator = RegimeCalibrator.fit(
            cal_truth, proteus_ci_lo, proteus_ci_hi, regime_df, alpha=0.05,
        )
        print(f"  {calibrator.summary()}")

        # Regime q̂ heatmap (saved once, not per-cohort)
        _maybe(plot_regime_qhat_heatmap, args.overwrite,
               calibrator.q_hats,
               os.path.join(out_dir, "regime_qhat_heatmap"),
               title="CQR Normalised q̂ by Gene Regime")

    # ──────────────────────────────────────────────────────────────────────
    # Gene-level reliability model fitted on OOF features
    # ──────────────────────────────────────────────────────────────────────
    # Per-fold features → ridge regression target = Fisher-z(per-fold r).
    # Features f15/f16 (gene zero-rate, variance) use the full canonical
    # 50-sample development truth pool.
    reliability_model: Optional[ReliabilityModel] = None
    if proteus_ci_lo is not None and cv_full.get("fold_data"):
        print("\n  ── Fitting reliability model (per-fold) ──")
        reliability_model = ReliabilityModel.fit(
            cv_full["fold_data"], canonical_truth, ridge_alpha=1.0,
        )
        print(f"    {reliability_model.summary()}")

        # Save feature weights plot
        _maybe(plot_reliability_weights, args.overwrite,
               reliability_model.feature_columns, reliability_model.weights,
               os.path.join(out_dir, "reliability_feature_weights"),
               title="Reliability Model Feature Weights (OOF)")

    # Exclude PHT admix samples from evaluation / plots
    proteus_pred, _, proteus_ci_lo, proteus_ci_hi = _exclude_pht(
        proteus_pred, _pred_truth, proteus_ci_lo, proteus_ci_hi,
    )
    if proteus_p0 is not None:
        proteus_p0 = proteus_p0[[
            s for s in proteus_p0.columns if not str(s).startswith("PHT")
        ]]

    # ── Load holdout PDX predictions (full: with p0) ──
    ho_preds: Dict[str, Dict[str, Any]] = {}
    _MDN_WIDE_KEYS = ("p0_wide", "pi_gamma_wide", "pi_normal_wide",
                       "gamma_beta_wide", "normal_mean_wide", "normal_stdev_wide")
    for cohort_name, rel_path in _BTS_HO_PATHS.items():
        ho_path = PROJECT_ROOT / rel_path
        if ho_path.exists():
            print(f"  Loading HO {cohort_name}: {rel_path}")
            ho_full = load_holdout_predictions_proteus_full(str(ho_path))
            # Keep only the 5 true holdout samples
            ho_samples = [s for s in ho_full["pred_wide"].columns if s in PDX_HOLDOUT_SAMPLES]
            if ho_samples:
                entry: Dict[str, Any] = {
                    "pred_wide": ho_full["pred_wide"][ho_samples],
                    "ci_lo_wide": ho_full["ci_lo_wide"][ho_samples] if ho_full["ci_lo_wide"] is not None else None,
                    "ci_hi_wide": ho_full["ci_hi_wide"][ho_samples] if ho_full["ci_hi_wide"] is not None else None,
                }
                for mk in _MDN_WIDE_KEYS:
                    entry[mk] = ho_full[mk][ho_samples] if ho_full.get(mk) is not None else None
                ho_preds[cohort_name] = entry
                print(f"    HO samples: {ho_samples}")
        else:
            print(f"  [skip] HO {cohort_name} not found: {rel_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Apply CQR calibration to ALL intervals (CV + HO) and evaluate once
    # ──────────────────────────────────────────────────────────────────────
    # The calibrator was fit on the full OOF data (both cohorts). Calibration
    # is applied globally — not per-cohort — because the 5-fold CV trains
    # on both PC-PDX and SCLC-PDX simultaneously.
    if calibrator is not None:
        cal_out = ensure_dir(os.path.join(out_dir, "Calibration"))

        # Evaluate on combined holdout data (raw CIs → base vs calibrated)
        ho_pred_parts, ho_ci_lo_parts, ho_ci_hi_parts = [], [], []
        for cname in _COHORT_CONFIG:
            hd = ho_preds.get(cname)
            if hd is not None and hd["ci_lo_wide"] is not None:
                ho_pred_parts.append(hd["pred_wide"])
                ho_ci_lo_parts.append(hd["ci_lo_wide"])
                ho_ci_hi_parts.append(hd["ci_hi_wide"])

        if ho_pred_parts:
            common_ho_genes = ho_pred_parts[0].index
            for part in ho_pred_parts[1:]:
                common_ho_genes = common_ho_genes.intersection(part.index)
            all_ho_pred = pd.concat([p.loc[common_ho_genes] for p in ho_pred_parts], axis=1)
            all_ho_ci_lo = pd.concat([p.loc[common_ho_genes] for p in ho_ci_lo_parts], axis=1)
            all_ho_ci_hi = pd.concat([p.loc[common_ho_genes] for p in ho_ci_hi_parts], axis=1)

            _run_calibration_evaluation(
                all_ho_pred, canonical_truth, all_ho_ci_lo, all_ho_ci_hi,
                calibrator, "BTS", cal_out,
                overwrite=args.overwrite,
            )

        # Apply calibration to CV OOF intervals in-place
        if proteus_ci_lo is not None:
            proteus_ci_lo, proteus_ci_hi = calibrator.apply(
                proteus_ci_lo, proteus_ci_hi,
            )
            print(f"  Applied CQR calibration to {proteus_ci_lo.shape[1]} CV OOF intervals")

        # Apply calibration to holdout intervals in-place
        for cname in ho_preds:
            hd = ho_preds[cname]
            if hd["ci_lo_wide"] is not None:
                hd["ci_lo_wide"], hd["ci_hi_wide"] = calibrator.apply(
                    hd["ci_lo_wide"], hd["ci_hi_wide"],
                )
        if ho_preds:
            print(f"  Applied CQR calibration to holdout intervals")

    # ── Run per-cohort analyses ──
    for cohort_name, cfg in _COHORT_CONFIG.items():
        cp, ct, ccl, cch = _split_by_cohort(
            proteus_pred, canonical_truth, proteus_ci_lo, proteus_ci_hi, cohort_name,
        )
        if cp is None or cp.shape[1] < 2:
            print(f"  [skip] Too few {cohort_name} samples for Proteus CV")
            continue

        cohort_out = ensure_dir(os.path.join(out_dir, cohort_name))

        # Determine which samples are CV vs HO
        cv_samples = sorted(cp.columns)
        ho_data = ho_preds.get(cohort_name)
        ho_pred_w = ho_ci_lo_w = ho_ci_hi_w = ho_p0_w = None
        ho_samples = []
        if ho_data is not None:
            ho_pred_w = ho_data["pred_wide"]
            ho_ci_lo_w = ho_data["ci_lo_wide"]
            ho_ci_hi_w = ho_data["ci_hi_wide"]
            ho_p0_w = ho_data["p0_wide"]
            ho_samples = sorted(ho_pred_w.columns)

        # Tag sample status in subtypes_df for plotting
        sample_status: Dict[str, str] = {}
        for s in cv_samples:
            sample_status[s] = "validation"
        for s in ho_samples:
            sample_status[s] = "holdout"
        if subtypes_df is not None:
            subtypes_with_status = subtypes_df.copy()
            subtypes_with_status["status"] = subtypes_with_status.index.map(
                lambda s: sample_status.get(s, "unknown")
            )
        else:
            subtypes_with_status = None

        # Merge CV + HO for combined analysis (CIs are already calibrated)
        if ho_pred_w is not None:
            common_genes = cp.index.intersection(ho_pred_w.index)
            combined_pred = pd.concat([cp.loc[common_genes], ho_pred_w.loc[common_genes]], axis=1)
            combined_ci_lo = None
            combined_ci_hi = None
            if ccl is not None and ho_ci_lo_w is not None:
                combined_ci_lo = pd.concat([ccl.loc[common_genes], ho_ci_lo_w.loc[common_genes]], axis=1)
                combined_ci_hi = pd.concat([cch.loc[common_genes], ho_ci_hi_w.loc[common_genes]], axis=1)
        else:
            combined_pred = cp
            combined_ci_lo = ccl
            combined_ci_hi = cch

        combined_truth = canonical_truth.reindex(index=combined_pred.index,
                                                  columns=combined_pred.columns)

        # ── Report metrics separately: CV / HO / Combined ──
        print(f"\n  ── Metrics: {cohort_name} (CV only, {len(cv_samples)} samples) ──")
        cv_truth = canonical_truth.reindex(index=cp.index, columns=cp.columns)
        cv_summary = cohort_summary_table(cv_truth, cp, label=f"CV ({cohort_name})",
                                          n_workers=n_workers)
        summaries.append(cv_summary)

        if ho_pred_w is not None and len(ho_samples) >= 2:
            print(f"  ── Metrics: {cohort_name} (HO only, {len(ho_samples)} samples) ──")
            ho_truth = canonical_truth.reindex(index=ho_pred_w.index,
                                               columns=ho_pred_w.columns)
            ho_summary = cohort_summary_table(ho_truth, ho_pred_w,
                                              label=f"HO ({cohort_name})",
                                              n_workers=n_workers)
            summaries.append(ho_summary)

        print(f"  ── Metrics: {cohort_name} (Combined, {combined_pred.shape[1]} samples) ──")
        comb_summary = cohort_summary_table(combined_truth, combined_pred,
                                            label=f"Combined ({cohort_name})",
                                            n_workers=n_workers)
        summaries.append(comb_summary)

        # ── Generate plots and outputs using combined data ──
        _run_cv_suite(
            combined_pred, combined_truth, f"BTS ({cohort_name})", cohort_out,
            ci_lo=combined_ci_lo, ci_hi=combined_ci_hi,
            goi=cfg["goi"], subtypes_df=subtypes_with_status,
            heatmap_gene_sets=cfg["heatmap_gene_sets"],
            gene_scatter_sets=cfg.get("gene_scatter_sets", []),
            gene_met=comb_summary["_gene_metrics"],
            sample_met=comb_summary["_sample_metrics"],
            n_workers=n_workers, overwrite=args.overwrite,
            scatter_ci=args.scatter_ci,
        )

        # ── Reliability model evaluation (on HO data) ──
        if ho_data is not None and reliability_model is not None:
            _run_reliability_evaluation(
                ho_data, canonical_truth,
                reliability_model,
                cohort_name, cohort_out,
                overwrite=args.overwrite,
            )

        # ── Per-sample expression KDEs ──
        _maybe(sample_expression_kde, args.overwrite, combined_pred,
               os.path.join(cohort_out, "expression_profiles"),
               subtypes_df=subtypes_with_status)

    # ── HD-L background prediction analysis ──
    print("\n  ── HD-L (background predictions) ──")
    bg_pred, bg_truth = load_cv_predictions_background(cv_dir)
    if bg_pred.empty:
        print("  [skip] No background prediction files found")
    else:
        hdl_out = ensure_dir(os.path.join(out_dir, "HD-L"))
        print(f"  Background preds: {bg_pred.shape[0]} genes × {bg_pred.shape[1]} donors")

        hdl_summary = cohort_summary_table(
            bg_truth, bg_pred, label="HD-L (background)",
            n_workers=n_workers,
        )
        summaries.append(hdl_summary)

        _run_cv_suite_hdl(
            bg_pred, bg_truth, "BTS (HD-L)", hdl_out,
            gene_met=hdl_summary["_gene_metrics"],
            sample_met=hdl_summary["_sample_metrics"],
            n_workers=n_workers, overwrite=args.overwrite,
        )

    # ── Print summary ──
    if summaries:
        print("\n" + "─" * 70)
        print_summary_table(summaries)
        _save_summary_tsv(summaries, os.path.join(out_dir, "summary_metrics.tsv"))


def _run_calibration_evaluation(
    ho_pred: pd.DataFrame,
    canonical_truth: pd.DataFrame,
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
    ho_pred : pd.DataFrame
        Holdout predictions (genes × samples).
    canonical_truth : pd.DataFrame
        Canonical truth pool (genes × all study samples).
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
    ho_truth = canonical_truth.reindex(index=ho_pred.index, columns=ho_pred.columns)
    common = ho_truth.columns.intersection(ho_pred.columns)
    if len(common) < 2:
        return

    tw = ho_truth[common]
    pw = ho_pred[common]
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
    print(f"    Calibrated predictions saved: {cal_path}  ({len(cal_df)} rows)")

    # Diagnostic: CI widths before/after
    base_widths = (hi - lo).values.flatten()
    adj_widths = (hi_adj - lo_adj).values.flatten()
    base_median_w = float(np.nanmedian(base_widths))
    adj_median_w = float(np.nanmedian(adj_widths))
    n_collapsed = int(np.sum(adj_widths == 0))
    print(f"    CI median width: base={base_median_w:.3f}  CQR-adjusted={adj_median_w:.3f}")
    if n_collapsed > 0:
        pct = 100 * n_collapsed / len(adj_widths)
        print(f"    [warn] {n_collapsed} intervals collapsed to zero-width ({pct:.1f}%)")

    # Flatten for coverage computation
    t_flat = tw.values.flatten()
    p_flat = pw.values.flatten()
    lo_flat = lo.values.flatten()
    hi_flat = hi.values.flatten()
    lo_adj_flat = lo_adj.values.flatten()
    hi_adj_flat = hi_adj.values.flatten()

    # Coverage curves: base vs CQR-calibrated
    levels = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
    base_curve = compute_coverage_curve(t_flat, lo_flat, hi_flat, p_flat, levels)
    conf_curve = compute_coverage_curve(t_flat, lo_adj_flat, hi_adj_flat, p_flat, levels)

    print(f"  CQR calibration evaluation ({label} HO):")
    print(f"    Base  95% coverage: {empirical_coverage(t_flat, lo_flat, hi_flat):.3f}")
    print(f"    CQR   95% coverage: {empirical_coverage(t_flat, lo_adj_flat, hi_adj_flat):.3f}")

    _maybe(plot_calibration_coverage, overwrite,
           base_curve, conf_curve,
           os.path.join(out_dir, "coverage_comparison"),
           title=f"CQR Calibration Coverage — {label} HO")

    # Regime coverage heatmap on HO data
    ho_eval = calibrator.evaluate(tw, pw, lo_adj, hi_adj)
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
    canonical_truth: pd.DataFrame,
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
    canonical_truth : Canonical truth pool (genes × all study samples).
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

    ho_truth = canonical_truth.reindex(index=ho_pred.index, columns=ho_pred.columns)
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
        pw, lo, hi, canonical_truth, **feat_kwargs,
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
    print(f"    Gene reliability scores saved: {rel_path}  ({len(rel_df)} rows)")


def _build_bts_truth() -> pd.DataFrame:
    """Load canonical truth restricted to the 50 study PDX tumor samples
    (25 PC-PDX + 25 SCLC-PDX).

    Uses ``build_study_truth_pool()`` which loads PC-PDX and SCLC-PDX
    Log2CPMplus1 labels and keeps only the named study samples.
    HD-L healthy donors and any extra samples are excluded.
    """
    return build_study_truth_pool()


def _build_bts_subtypes() -> Optional[pd.DataFrame]:
    """Build a subtypes DataFrame for BTS tumor samples.

    Loads PC-PDX and SCLC-PDX subtype labels.
    Returns DataFrame indexed by sample with columns [subtype, tfx].
    HD-L healthy donors are excluded.
    """
    parts = []
    for cohort in ("PC-PDX", "SCLC-PDX"):
        path = PROJECT_ROOT / "labels" / f"{cohort}_TFX_Subtype.tsv"
        if path.exists():
            parts.append(load_subtypes(cohort))
    if not parts:
        return None
    return pd.concat(parts, axis=0)


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
        heatmap_gene_sets = ["PC-Phenotyping", "PAM50", "Targets_PC"]
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
        _maybe(double_heatmap, overwrite, truth_wide, pred_wide, genes,
               os.path.join(out_dir, f"double_heatmap_{gs_name}"),
               subtypes=subtypes_df, breaks=breaks)

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
    if scatter_ci:
        ci_lo_no_pht = ci_lo[[s for s in non_pht_samples if s in ci_lo.columns]] if ci_lo is not None else None
        ci_hi_no_pht = ci_hi[[s for s in non_pht_samples if s in ci_hi.columns]] if ci_hi is not None else None
    else:
        ci_lo_no_pht = None
        ci_hi_no_pht = None
    batch_sample_scatters(
        tw_no_pht, pw_no_pht, sample_dir,
        ci_lo_wide=ci_lo_no_pht, ci_hi_wide=ci_hi_no_pht, goi=goi,
        subtypes_df=subtypes_df, n_workers=n_workers,
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
        "PC-Phenotyping", "PAM50", "Targets_PC",
    ]
    for gs_name in _heatmap_sets:
        resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
        if not gene_list_available(resolved):
            continue
        genes = get_gene_list(resolved)
        breaks = GENE_SET_BREAKS.get(gs_name)
        _maybe(double_heatmap, overwrite, tw, pw, genes,
               os.path.join(out_dir, f"double_heatmap_{gs_name}"),
               subtypes=subtypes_df, breaks=breaks)

    # Per-sample scatters
    sample_dir = ensure_dir(os.path.join(out_dir, "sample_scatters"))

    if scatter_ci:
        ci_lo_aligned = ci_lo.reindex_like(tw) if ci_lo is not None else None
        ci_hi_aligned = ci_hi.reindex_like(tw) if ci_hi is not None else None
    else:
        ci_lo_aligned = None
        ci_hi_aligned = None

    _goi = goi if goi is not None else DEFAULT_GOI_PC
    batch_sample_scatters(
        tw, pw, sample_dir,
        ci_lo_wide=ci_lo_aligned, ci_hi_wide=ci_hi_aligned,
        goi=_goi, subtypes_df=subtypes_df, n_workers=n_workers,
    )

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")
    return summary


# ---------------------------------------------------------------------------
# Analysis group: Patient_Validation (holdout predictions vs biopsy truth)
# ---------------------------------------------------------------------------

# Cohorts with RNA-Seq truth labels and PAN-FT holdout results.
_VALIDATION_COHORTS = ["CRPC-WCDT", "BLCA-RA", "CRPC-RA", "SCLC"]

# Per-cohort gene-set config: heatmap gene sets, scatter gene sets, and GOI.
_VALIDATION_COHORT_CONFIG: Dict[str, Dict[str, Any]] = {
    "CRPC-WCDT": {
        "heatmap": ["PC-Phenotyping", "Targets_PC"],
        "scatter": ["Targets_PC"],
        "goi": DEFAULT_GOI_PC,
    },
    "BLCA-RA": {
        "heatmap": ["Targets_BC"],
        "scatter": ["Targets_BC"],
        "goi": DEFAULT_GOI_PC,
    },
    "CRPC-RA": {
        "heatmap": ["PC-Phenotyping", "Targets_PC"],
        "scatter": ["Targets_PC"],
        "goi": DEFAULT_GOI_PC,
    },
    "SCLC": {
        "heatmap": ["Targets_SCLC"],
        "scatter": ["Targets_SCLC"],
        "goi": DEFAULT_GOI_SCLC,
    },
}

# Single model: PAN-FT fine-tuned Proteus.
_VALIDATION_MODEL_PATH = "results_holdout/BioDVN_PAN-FT/{cohort}/predictions_tumor.tsv"


def _run_patient_validation_cohort(
    cohort: str,
    out_dir: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Run Patient_Validation analysis for a single cohort (PAN-FT model only).

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
        "heatmap": ["PC-Phenotyping", "Targets_PC"],
        "scatter": ["Targets_PC"],
        "goi": DEFAULT_GOI_PC,
    })

    # Output directly into cohort folder (no model subfolder)
    cohort_out = ensure_dir(os.path.join(out_dir, cohort))
    summaries: List[Dict[str, Any]] = []

    path = PROJECT_ROOT / _VALIDATION_MODEL_PATH.format(cohort=cohort)
    if not path.exists():
        print(f"    [skip] PAN-FT predictions not found: {path.name}")
        return None

    pred, ci_lo, ci_hi = load_holdout_predictions_proteus(str(path))
    _apply_prefix_map(pred, pmap)
    _apply_prefix_map(ci_lo, pmap)
    _apply_prefix_map(ci_hi, pmap)
    label = f"Proteus PAN-FT — {cohort}"

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

    # Return data for Collected analysis
    return {
        "cohort": cohort,
        "truth_wide": truth_wide,
        "pred_wide": pred,
        "subtypes_df": subtypes_df,
        "tfx_dict": tfx_dict,
    }


def _run_collected_validation(
    cohort_data: List[Dict[str, Any]],
    out_dir: str,
    args: argparse.Namespace,
) -> None:
    """Run the Collected validation analysis combining all cohorts.

    Produces gene-level scatter plots across all validation cohorts on
    the intersection of genes available in every cohort.  Summary metrics
    are computed on the combined pool.
    """
    collected_out = ensure_dir(os.path.join(out_dir, "Collected"))
    print("\n  ── Collected (all validation cohorts) ──")

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
    for cd in cohort_data:
        tw = cd["truth_wide"].loc[all_genes]
        pw = cd["pred_wide"].loc[all_genes]
        common_samples = tw.columns.intersection(pw.columns)
        if len(common_samples) > 0:
            truth_parts.append(tw[common_samples])
            pred_parts.append(pw[common_samples])

    if not truth_parts:
        print("    [skip] No overlapping samples")
        return

    combined_truth = pd.concat(truth_parts, axis=1)
    combined_pred = pd.concat(pred_parts, axis=1)
    print(f"    Combined: {combined_truth.shape[0]} genes × "
          f"{combined_truth.shape[1]} samples")

    # Compute combined metrics
    var_map = compute_gene_variance(combined_truth)
    summary = cohort_summary_table(
        combined_truth, combined_pred, label="Collected",
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
           xlim=(-1, 1), title="Collected — Pearson r by Variance Group", square=True)
    _maybe(metric_distribution_by_variance, args.overwrite, gene_met, var_map, "spearman_rho",
           os.path.join(collected_out, "gene_spearman_rho_dist_by_variance"),
           xlim=(-1, 1), title="Collected — Spearman ρ by Variance Group", square=True)
    _maybe(metric_distribution_by_variance, args.overwrite, gene_met, var_map, "rmse",
           os.path.join(collected_out, "gene_rmse_dist_by_variance"),
           title="Collected — RMSE by Variance Group", square=True)

    # Per-sample scatters
    sample_dir = ensure_dir(os.path.join(collected_out, "sample_scatters"))
    batch_sample_scatters(
        combined_truth, combined_pred, sample_dir,
        goi=DEFAULT_GOI_PC, subtypes_df=None, n_workers=args.n_workers,
    )

    # Save summary
    print_summary_table([summary])
    _save_summary_tsv([summary], os.path.join(collected_out, "summary_metrics.tsv"))


def run_patient_validation(args: argparse.Namespace) -> None:
    """Patient validation analysis: PAN-FT model predictions vs biopsy truth.

    Iterates over all configured cohorts with truth labels and PAN-FT holdout
    predictions.  Output is flat per cohort (no model subfolders).  A
    "Collected" subfolder combines all cohorts for cross-cohort analysis.

    Override behaviour: if ``--proteus_holdout`` and ``--truth_cohort`` are
    given, runs a single custom model/cohort pair instead.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Patient_Validation")
    print("=" * 70)

    out_dir = ensure_dir(
        str(PROJECT_ROOT / "analysis" / "results" / "Patient_Validation"))

    # Single-cohort CLI override
    if args.proteus_holdout and args.truth_cohort:
        cohort = args.truth_cohort
        truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
        if not truth_path.exists():
            print(f"  [skip] Truth not found: {cohort}")
            return
        truth_wide = load_truth(cohort)
        pmap = _build_prefix_map(truth_wide.columns, cohort)
        subtypes_df = None
        sub_path = PROJECT_ROOT / "labels" / f"{cohort}_TFX_Subtype.tsv"
        if sub_path.exists():
            subtypes_df = load_subtypes(cohort)
        ho_path = os.path.join(_resolve(args.proteus_holdout),
                               "predictions_tumor.tsv")
        if not os.path.exists(ho_path):
            print(f"  [skip] {ho_path} not found")
            return
        pred, ci_lo, ci_hi = load_holdout_predictions_proteus(ho_path)
        _apply_prefix_map(pred, pmap)
        _apply_prefix_map(ci_lo, pmap)
        _apply_prefix_map(ci_hi, pmap)
        custom_out = ensure_dir(os.path.join(out_dir, cohort))
        summaries: List[Dict[str, Any]] = []
        _run_holdout_suite(
            pred, truth_wide, f"Custom — {cohort}", custom_out,
            ci_lo=ci_lo, ci_hi=ci_hi, subtypes_df=subtypes_df,
            n_workers=args.n_workers, overwrite=args.overwrite,
            summaries=summaries, scatter_ci=args.scatter_ci,
        )
        if summaries:
            print_summary_table(summaries)
        return

    # Default: iterate over all validation cohorts (PAN-FT model only)
    cohort_data: List[Dict[str, Any]] = []
    for cohort in _VALIDATION_COHORTS:
        print(f"\n  ── Cohort: {cohort} ──")
        result = _run_patient_validation_cohort(cohort, out_dir, args)
        if result is not None:
            cohort_data.append(result)

    # Collected analysis across all validation cohorts
    if len(cohort_data) >= 2:
        _run_collected_validation(cohort_data, out_dir, args)


# ---------------------------------------------------------------------------
# Analysis group: Patient_Discovery (predictions only — no truth)
# ---------------------------------------------------------------------------

# Discovery cohorts: predictions available but no RNA-Seq truth labels.
# Format: (cohort_predictions_dir_name, subtype_label_name_or_None).
# subtype_label_name maps to labels/{name}_TFX_Subtype.tsv for annotation.
_DISCOVERY_COHORTS: List[Tuple[str, Optional[str]]] = [
    ("CRPC-Pluvicto", "Pluvicto"),
    ("CRPC-UW", "UW"),
    ("BC-Ghana", None),
]

# Models to check for each discovery cohort (ordered by priority).
_DISCOVERY_MODELS = [
    ("Proteus_Tuned", "results_holdout/BioDVN_PAN-FT/{cohort}/predictions_tumor.tsv"),
    ("Proteus_Base", "results_holdout/BioDVN/{cohort}/predictions_tumor.tsv"),
]


def _run_discovery_cohort(
    cohort: str,
    subtype_label: Optional[str],
    out_dir: str,
    args: argparse.Namespace,
) -> None:
    """Run Patient_Discovery analysis for a single cohort (no truth)."""
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

        pred, ci_lo, ci_hi = load_holdout_predictions_proteus(str(path))
        model_out = ensure_dir(os.path.join(cohort_out, model_name))
        print(f"    {model_name}: {pred.shape[0]} genes × {pred.shape[1]} samples")

        _maybe(sample_expression_kde, args.overwrite, pred,
               os.path.join(model_out, "expression_profiles"),
               subtypes_df=subtypes_df)

        for gs_name in ["PC-Phenotyping", "Targets_PC", "PAM50"]:
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


def run_patient_discovery(args: argparse.Namespace) -> None:
    """Discovery cohort analysis: predictions only (no biopsy truth).

    Iterates over all configured discovery cohorts and generates heatmaps,
    expression profiles, and prediction exports for each available model.

    Override: if ``--proteus_holdout`` is given, runs only that path.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Patient_Discovery")
    print("=" * 70)

    out_dir = ensure_dir(
        str(PROJECT_ROOT / "analysis" / "results" / "Patient_Discovery"))

    # CLI override: single path
    if args.proteus_holdout:
        cohort_tag = args.truth_cohort or Path(args.proteus_holdout).name
        _run_discovery_cohort(cohort_tag, None, out_dir, args)
        print("  Done")
        return

    # Default: iterate over all discovery cohorts
    for cohort, subtype_label in _DISCOVERY_COHORTS:
        print(f"\n  ── Cohort: {cohort} ──")
        _run_discovery_cohort(cohort, subtype_label, out_dir, args)

    print("\n  Done")


# ---------------------------------------------------------------------------
# Group: PAN-FT-CV (fine-tune cross-validation, mirrors BTS-CV pipeline)
# ---------------------------------------------------------------------------

_PANFT_CV_PATH = "results_training/PAN-FT_cv"

# Per-cohort analysis config for PAN-FT-CV splits
_PANFT_CV_COHORT_CONFIG: Dict[str, Dict[str, Any]] = {
    "CRPC-RA": {
        "heatmap_gene_sets": ["PC-Phenotyping", "Targets_PC"],
        "goi": DEFAULT_GOI_PC,
        "gene_scatter_sets": ["Targets_PC"],
        "truth_label": "CRPC-RA",
    },
    "SCLC": {
        "heatmap_gene_sets": ["Targets_SCLC"],
        "goi": DEFAULT_GOI_SCLC,
        "gene_scatter_sets": ["Targets_SCLC"],
        "truth_label": "SCLC",
    },
}


def _load_panft_fold_splits() -> pd.DataFrame:
    """Load data/PAN-FT_sample-splits.tsv → DataFrame with Cohort, Sample, Fold."""
    path = PROJECT_ROOT / "data" / "PAN-FT_sample-splits.tsv"
    return pd.read_csv(path, sep="\t")


def _build_panft_truth() -> pd.DataFrame:
    """Build canonical truth for PAN-FT CV cohorts (CRPC-RA + SCLC).

    Loads Log2CPMplus1 labels for both cohorts, keeps only samples present
    in the PAN-FT sample-splits table, and concatenates on a common gene index.
    Column names retain their native form (e.g. ``CRPC-RA_15-023``).
    """
    splits = _load_panft_fold_splits()
    split_samples = set(splits["Sample"])

    parts = []
    for cohort in ("CRPC-RA", "SCLC"):
        truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
        if not truth_path.exists():
            continue
        tw = load_truth(cohort)
        # Keep only samples that are in the PAN-FT splits
        keep = [s for s in tw.columns if s in split_samples]
        if keep:
            parts.append(tw[keep])

    if not parts:
        return pd.DataFrame()

    common_genes = parts[0].index
    for p in parts[1:]:
        common_genes = common_genes.intersection(p.index)
    return pd.concat([p.loc[common_genes] for p in parts], axis=1)


def _build_panft_subtypes() -> Optional[pd.DataFrame]:
    """Build a subtypes DataFrame for PAN-FT CV samples (CRPC-RA + SCLC).

    Index retains the native prefixed sample names from the label files.
    """
    parts = []
    for cohort in ("CRPC-RA", "SCLC"):
        path = PROJECT_ROOT / "labels" / f"{cohort}_TFX_Subtype.tsv"
        if path.exists():
            parts.append(load_subtypes(cohort))
    if not parts:
        return None
    return pd.concat(parts, axis=0)


def run_panft_cv(args: argparse.Namespace) -> None:
    """PAN-FT cross-validation analysis (mirrors BTS-CV pipeline).

    Loads full MDN predictions from PAN-FT 5-fold CV, splits by cohort
    (CRPC-RA, SCLC), and runs the full analysis suite per cohort including:
      1. CQR regime-based calibration on all OOF samples.
      2. Gene-level reliability model fitted on OOF features.
      3. Per-cohort evaluation + plots with cohort-specific gene sets.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: PAN-FT-CV (Cross-Validation)")
    print("=" * 70)

    cv_dir = str(PROJECT_ROOT / _PANFT_CV_PATH)
    if not os.path.isdir(cv_dir):
        print(f"  [skip] CV results not found: {_PANFT_CV_PATH}")
        return

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / "PAN-FT-CV"))
    summaries: List[Dict[str, Any]] = []
    n_workers = args.n_workers

    # Load fold-split table to map samples → cohort (prefixed names).
    fold_splits = _load_panft_fold_splits()
    cohort_map: Dict[str, str] = dict(
        zip(fold_splits["Sample"], fold_splits["Cohort"])
    )

    # Build short→prefixed map for renaming prediction columns.
    _prefix_map: Dict[str, str] = {}
    for _, row in fold_splits.iterrows():
        prefixed = row["Sample"]
        short = str(strip_cohort_prefix(
            pd.Index([prefixed]), row["Cohort"],
        )[0])
        _prefix_map[short] = prefixed

    subtypes_df = _build_panft_subtypes()
    canonical_truth = _build_panft_truth()
    if canonical_truth.empty:
        print("  [skip] No canonical truth could be built for PAN-FT")
        return
    print(f"  Canonical truth: {canonical_truth.shape[0]} genes × "
          f"{canonical_truth.shape[1]} samples")

    def _split_by_cohort(
        pred: pd.DataFrame,
        truth: pd.DataFrame,
        ci_lo: Optional[pd.DataFrame],
        ci_hi: Optional[pd.DataFrame],
        cohort: str,
    ) -> Tuple:
        """Extract only samples matching ``cohort``."""
        samples = sorted(s for s in pred.columns if cohort_map.get(s) == cohort)
        if not samples:
            return None, None, None, None
        p = pred[[s for s in samples if s in pred.columns]]
        t = truth[[s for s in samples if s in truth.columns]]
        cl = ci_lo[[s for s in samples if s in ci_lo.columns]] if ci_lo is not None else None
        ch = ci_hi[[s for s in samples if s in ci_hi.columns]] if ci_hi is not None else None
        return p, t, cl, ch

    # ── Load Proteus CV predictions (full: includes p0, per-fold data) ──
    print(f"\n  Loading Proteus CV (full) from: {cv_dir}")
    cv_full = load_cv_predictions_proteus_full(cv_dir)

    # Rename prediction columns from short IDs to prefixed canonical names
    # (e.g. '15-023' → 'CRPC-RA_15-023'; 'L-0141' → 'L-0141').
    for key in ("pred_wide", "truth_wide", "ci_lo_wide", "ci_hi_wide",
                "p0_wide", "pi_gamma_wide", "pi_normal_wide",
                "gamma_beta_wide", "normal_mean_wide", "normal_stdev_wide"):
        cv_full[key] = _apply_prefix_map(cv_full.get(key), _prefix_map)
    for fold_id, fd in cv_full.get("fold_data", {}).items():
        for fk in list(fd.keys()):
            if isinstance(fd[fk], pd.DataFrame):
                fd[fk] = _apply_prefix_map(fd[fk], _prefix_map)
    cv_full["sample_fold"] = {
        _prefix_map.get(s, s): f for s, f in cv_full["sample_fold"].items()
    }

    proteus_pred = cv_full["pred_wide"]
    _pred_truth = cv_full["truth_wide"]
    proteus_ci_lo = cv_full["ci_lo_wide"]
    proteus_ci_hi = cv_full["ci_hi_wide"]

    # ──────────────────────────────────────────────────────────────────────
    # CQR regime-based calibration on ALL OOF samples
    # ──────────────────────────────────────────────────────────────────────
    calibrator: Optional[RegimeCalibrator] = None
    regime_df: Optional[pd.DataFrame] = None
    if proteus_ci_lo is not None and proteus_ci_hi is not None:
        cal_truth = _pred_truth.reindex(
            index=proteus_ci_lo.index, columns=proteus_ci_lo.columns,
        )
        regime_df = compute_gene_regimes(canonical_truth)
        print(f"  Gene regimes: {regime_df['regime'].nunique()} unique regimes "
              f"from {len(regime_df)} genes")

        calibrator = RegimeCalibrator.fit(
            cal_truth, proteus_ci_lo, proteus_ci_hi, regime_df, alpha=0.05,
        )
        print(f"  {calibrator.summary()}")

        _maybe(plot_regime_qhat_heatmap, args.overwrite,
               calibrator.q_hats,
               os.path.join(out_dir, "regime_qhat_heatmap"),
               title="CQR Normalised q̂ by Gene Regime (PAN-FT)")

    # ──────────────────────────────────────────────────────────────────────
    # Gene-level reliability model fitted on OOF features
    # ──────────────────────────────────────────────────────────────────────
    reliability_model: Optional[ReliabilityModel] = None
    if proteus_ci_lo is not None and cv_full.get("fold_data"):
        print("\n  ── Fitting reliability model (per-fold) ──")
        reliability_model = ReliabilityModel.fit(
            cv_full["fold_data"], canonical_truth, ridge_alpha=1.0,
        )
        print(f"    {reliability_model.summary()}")

        _maybe(plot_reliability_weights, args.overwrite,
               reliability_model.feature_columns, reliability_model.weights,
               os.path.join(out_dir, "reliability_feature_weights"),
               title="Reliability Model Feature Weights (PAN-FT OOF)")

    # Apply CQR calibration to CV OOF intervals
    if calibrator is not None and proteus_ci_lo is not None:
        proteus_ci_lo, proteus_ci_hi = calibrator.apply(
            proteus_ci_lo, proteus_ci_hi,
        )
        print(f"  Applied CQR calibration to {proteus_ci_lo.shape[1]} CV OOF intervals")

    # ── Run per-cohort analyses ──
    for cohort_name, cfg in _PANFT_CV_COHORT_CONFIG.items():
        cp, ct, ccl, cch = _split_by_cohort(
            proteus_pred, canonical_truth, proteus_ci_lo, proteus_ci_hi, cohort_name,
        )
        if cp is None or cp.shape[1] < 2:
            print(f"  [skip] Too few {cohort_name} samples for PAN-FT CV")
            continue

        cohort_out = ensure_dir(os.path.join(out_dir, cohort_name))

        print(f"\n  ── Metrics: {cohort_name} ({cp.shape[1]} samples) ──")
        cv_truth = canonical_truth.reindex(index=cp.index, columns=cp.columns)
        cv_summary = cohort_summary_table(cv_truth, cp, label=f"PAN-FT CV ({cohort_name})",
                                          n_workers=n_workers)
        summaries.append(cv_summary)

        # Build subtypes for this cohort's samples
        cohort_subtypes = None
        if subtypes_df is not None:
            cohort_samples_sub = [s for s in cp.columns if s in subtypes_df.index]
            if cohort_samples_sub:
                cohort_subtypes = subtypes_df.loc[cohort_samples_sub]

        _run_cv_suite(
            cp, cv_truth, f"PAN-FT CV ({cohort_name})", cohort_out,
            ci_lo=ccl, ci_hi=cch,
            goi=cfg["goi"], subtypes_df=cohort_subtypes,
            heatmap_gene_sets=cfg["heatmap_gene_sets"],
            gene_scatter_sets=cfg.get("gene_scatter_sets", []),
            gene_met=cv_summary["_gene_metrics"],
            sample_met=cv_summary["_sample_metrics"],
            n_workers=n_workers, overwrite=args.overwrite,
            scatter_ci=args.scatter_ci,
        )

    # ── Print summary ──
    if summaries:
        print("\n" + "─" * 70)
        print_summary_table(summaries)
        _save_summary_tsv(summaries, os.path.join(out_dir, "summary_metrics.tsv"))


# ---------------------------------------------------------------------------
# Group: CRPC-FT-CV (legacy fine-tune CV — simple version)
# ---------------------------------------------------------------------------

_CRPC_FT_CV_PATH = "results_training/proteus_crpc-finetune_cv"


def run_crpc_finetune_cv(args: argparse.Namespace) -> None:
    """CRPC-FT-CV analysis (simple version without calibration/reliability)."""
    group_name = "CRPC-FT-CV"
    print(f"\n{'='*70}")
    print(f"  ANALYSIS GROUP: {group_name}")
    print("=" * 70)

    cv_dir = str(PROJECT_ROOT / _CRPC_FT_CV_PATH)
    if not os.path.isdir(cv_dir):
        print(f"  [skip] CV results not found: {_CRPC_FT_CV_PATH}")
        return

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / group_name))
    n_workers = args.n_workers

    print(f"  Loading Proteus CV from: {cv_dir}")
    pred, truth, ci_lo, ci_hi = load_cv_predictions_proteus(cv_dir)
    pred, truth, ci_lo, ci_hi = _exclude_pht(pred, truth, ci_lo, ci_hi)

    ft_out = ensure_dir(os.path.join(out_dir, "Proteus"))
    summary = cohort_summary_table(truth, pred, label=group_name, n_workers=n_workers)

    _run_cv_suite(
        pred, truth, group_name, ft_out,
        ci_lo=ci_lo, ci_hi=ci_hi,
        gene_met=summary["_gene_metrics"],
        sample_met=summary["_sample_metrics"],
        n_workers=n_workers, overwrite=args.overwrite,
        scatter_ci=args.scatter_ci,
    )

    print("\n" + "─" * 70)
    print_summary_table([summary])
    _save_summary_tsv([summary], os.path.join(out_dir, "summary_metrics.tsv"))


# ---------------------------------------------------------------------------
# Group: XGB-FT-CV (XGBoost fine-tune CV)
# ---------------------------------------------------------------------------

_XGB_FT_CV_PATH = "results_training/xgboost_fine-tune_cv"


def run_xgb_finetune_cv(args: argparse.Namespace) -> None:
    """XGBoost fine-tune CV analysis."""
    group_name = "XGB-FT-CV"
    print(f"\n{'='*70}")
    print(f"  ANALYSIS GROUP: {group_name}")
    print("=" * 70)

    cv_dir = str(PROJECT_ROOT / _XGB_FT_CV_PATH)
    if not os.path.isdir(cv_dir):
        print(f"  [skip] CV results not found: {_XGB_FT_CV_PATH}")
        return

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / group_name))
    n_workers = args.n_workers

    print(f"  Loading XGBoost CV from: {cv_dir}")
    pred, truth = load_cv_predictions_xgboost(cv_dir)
    # Exclude PHT admix samples
    pred, truth, _, _ = _exclude_pht(pred, truth)

    xgb_out = ensure_dir(os.path.join(out_dir, "XGBoost"))
    summary = cohort_summary_table(truth, pred, label=group_name, n_workers=n_workers)

    _run_cv_suite(
        pred, truth, group_name, xgb_out,
        gene_met=summary["_gene_metrics"],
        sample_met=summary["_sample_metrics"],
        n_workers=n_workers, overwrite=args.overwrite,
        scatter_ci=args.scatter_ci,
    )

    print("\n" + "─" * 70)
    print_summary_table([summary])
    _save_summary_tsv([summary], os.path.join(out_dir, "summary_metrics.tsv"))


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


def _resolve(path: str) -> str:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


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
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Analysis group: Benchmarking (PHS holdout — model comparison)
# ---------------------------------------------------------------------------

# TFx/depth grid and models — per README spec
_BENCH_TFX_ORDER = ["TF0.50", "TF0.30", "TF0.10", "TF0.05", "TF0.03", "TF0.01"]
_BENCH_DEPTH_ORDER = ["10x", "5x"]
_BENCH_EXCLUDED_DEPTHS = {"1x", "30x", "50x"}

# Model directories under results_holdout/
_BENCH_PROTEUS_MODELS: Dict[str, str] = {
    "Proteus_Base": "results_holdout/BioDVN/PHS/predictions_tumor.tsv",
    "Proteus_NoSeq": "results_holdout/BioDVN_NoSeq/PHS/predictions_tumor.tsv",
    "Proteus_OnlySeq": "results_holdout/BioDVN_OnlySeq/PHS/predictions_tumor.tsv",
}
_BENCH_XGB_PATH = "results_holdout/XGBoost/PHS/predictions_tumor.tsv"
_EPIC_SEQ_PATH = "results_holdout/EPIC-Seq/PHS/EPICSeqInferredExpressionValues.txt"
_EPIC_SCORE_COLS = ["inferredGEP", "PFE", "NDR_relative_2k"]
_EPIC_DISPLAY_NAMES = {
    "inferredGEP": "EPIC_GEP",
    "PFE": "EPIC_PFE",
    "NDR_relative_2k": "EPIC_NDR",
}


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


# ---------------------------------------------------------------------------
# Analysis group: Benchmarking_PAN-FT (real patient cohorts)
# ---------------------------------------------------------------------------

_PANFT_COHORTS = ["CRPC-WCDT", "BLCA-RA"]

_PANFT_MODEL_ORDER = [
    "Proteus_Tuned", "Proteus_Base", "XGBoost",
    "EPIC_GEP", "EPIC_PFE", "EPIC_NDR", "Naïve",
]

# Per-cohort model paths (relative to PROJECT_ROOT).
_PANFT_PROTEUS_PATHS: Dict[str, Dict[str, str]] = {
    "CRPC-WCDT": {
        "Proteus_Tuned": "results_holdout/BioDVN_PAN-FT/CRPC-WCDT/predictions_tumor.tsv",
        "Proteus_Base": "results_holdout/BioDVN/CRPC-WCDT/predictions_tumor.tsv",
    },
    "BLCA-RA": {
        "Proteus_Tuned": "results_holdout/BioDVN_PAN-FT/BLCA-RA/predictions_tumor.tsv",
        "Proteus_Base": "results_holdout/BioDVN/BLCA-RA/predictions_tumor.tsv",
    },
}

_PANFT_XGB_PATHS: Dict[str, str] = {
    "CRPC-WCDT": "results_holdout/XGBoost/CRPC-WCDT/predictions_tumor.tsv",
    "BLCA-RA": "results_holdout/XGBoost/BLCA-RA/predictions_tumor.tsv",
}

# EPIC-Seq paths — only BLCA-RA has EPIC-Seq results.
_PANFT_EPIC_PATHS: Dict[str, str] = {
    "BLCA-RA": "results_holdout/EPIC-Seq/BLCA-RA/EPICSeqInferredExpressionValues.txt",
}


def run_benchmarking_panft(args: argparse.Namespace) -> None:
    """Benchmarking on real patient cohorts (CRPC-WCDT, BLCA-RA).

    Compares models on actual patient biopsies.  Metrics are computed per
    cohort and presented side-by-side in twin-panel plots (one panel per
    cohort).  Output is flat — no per-cohort subfolders.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Benchmarking_PAN-FT")
    print("=" * 70)

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / "Benchmarking_PAN-FT"))

    # Accumulate metrics, per-gene rho, and per-sample rho across cohorts
    all_metrics_rows: List[Dict[str, Any]] = []
    all_gene_rho_rows: List[Dict[str, Any]] = []
    all_sample_rho_rows: List[Dict[str, Any]] = []

    for cohort in _PANFT_COHORTS:
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

        # Proteus models
        proteus_paths = _PANFT_PROTEUS_PATHS.get(cohort, {})
        for model_name, rel_path in proteus_paths.items():
            path = PROJECT_ROOT / rel_path
            if path.exists():
                pred, _, _ = load_holdout_predictions_proteus(str(path))
                _apply_prefix_map(pred, pmap)
                model_preds[model_name] = pred
                print(f"    Loaded {model_name}: {pred.shape[1]} samples")
            else:
                print(f"    [skip] {model_name}: {rel_path} not found")

        # XGBoost
        xgb_rel = _PANFT_XGB_PATHS.get(cohort)
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
        epic_rel = _PANFT_EPIC_PATHS.get(cohort)
        if epic_rel:
            epic_path = PROJECT_ROOT / epic_rel
            if epic_path.exists():
                epic_preds = load_holdout_predictions_epicseq(
                    str(epic_path), _EPIC_SCORE_COLS,
                )
                for score_name, epw in epic_preds.items():
                    display = _EPIC_DISPLAY_NAMES.get(score_name, f"EPIC_{score_name}")
                    epw.columns = clean_epicseq_sample_names(epw.columns)
                    _apply_prefix_map(epw, pmap)
                    model_preds[display] = epw
                    print(f"    Loaded {display}: {epw.shape[1]} samples")
            else:
                print(f"    [skip] EPIC-Seq: {epic_rel} not found")

        if not model_preds:
            print(f"  [skip] No model predictions for {cohort}")
            continue

        # Naïve baseline (row-shuffled truth)
        naive_pred = generate_null_predictions(truth_wide, seed=42)
        model_preds["Naïve"] = naive_pred

        # Canonical order for this cohort
        model_order = [m for m in _PANFT_MODEL_ORDER if m in model_preds]

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

    metrics_table = pd.DataFrame(all_metrics_rows)
    gene_rho_table = pd.DataFrame(all_gene_rho_rows)
    sample_rho_table = pd.DataFrame(all_sample_rho_rows)

    if metrics_table.empty:
        print("  [skip] No metrics computed")
        return

    # Save summary TSV
    metrics_table.to_csv(
        os.path.join(out_dir, "summary_benchmarking.tsv"), sep="\t", index=False,
    )
    print(f"\n  Summary: {len(metrics_table)} rows")
    print(metrics_table.to_string(index=False))

    # Resolve global model order (models that appear in at least one cohort)
    model_order = [m for m in _PANFT_MODEL_ORDER if m in set(metrics_table["model"])]

    # ── Twin-panel cohort boxplots for key scalar metrics ──
    for metric_col, ylabel in [
        ("gvw_rho", "GVW-ρ"),
        ("mean_gene_rho", "Mean Gene ρ"),
        ("rmse", "Mean Gene RMSE"),
        ("mean_top10pct_gene_rho", "Mean ρ (Top 10%)"),
        ("n_genes_rho_ge070", "Genes with ρ≥0.70"),
    ]:
        _maybe(benchmark_cohort_boxplots, args.overwrite,
               metrics_table, os.path.join(out_dir, f"benchmark_cohort_{metric_col}"),
               metric_col=metric_col, ylabel=ylabel,
               title=f"Benchmarking PAN-FT — {ylabel}",
               cohort_order=_PANFT_COHORTS, model_order=model_order)

    # Per-sample Spearman rho boxplots
    if not sample_rho_table.empty:
        _maybe(benchmark_cohort_boxplots, args.overwrite,
               sample_rho_table,
               os.path.join(out_dir, "benchmark_cohort_sample_rho"),
               metric_col="spearman_rho", ylabel="Sample ρ",
               title="Benchmarking PAN-FT — Per-Sample Spearman ρ",
               cohort_order=_PANFT_COHORTS, model_order=model_order)

    # ── Twin-panel cohort histograms of per-gene Spearman rho ──
    if not gene_rho_table.empty:
        _maybe(benchmark_cohort_histograms, args.overwrite,
               gene_rho_table,
               os.path.join(out_dir, "benchmark_cohort_gene_rho_hist"),
               cohort_order=_PANFT_COHORTS, model_order=model_order)

    # ── Multi-metric line plots (cohort on x-axis) ──
    _maybe(benchmark_cohort_metrics_lines, args.overwrite,
           metrics_table, os.path.join(out_dir, "benchmark_cohort_metrics"),
           cohort_order=_PANFT_COHORTS, model_order=model_order)

    print("\n  Done")


def _generate_naive_example(
    out_dir: str,
    overwrite: bool,
) -> None:
    """Produce NaiveExample illustrative plots.

    Uses all 25 LuCaP PC-PDX truth profiles to illustrate two baselines
    at both the single-transcriptome (scatter) and per-gene (distribution)
    levels:

    1. Scatter: LuCaP_136 vs 136CR (biological replicate pair)
    2. Scatter: LuCaP_136 vs population-mean expression (regression to mean)
    3. Distribution: per-gene Pearson r, truth vs row-shuffled truth
       across all 25 LuCaP samples (na\u00efve shuffle baseline)
    4. Distribution: per-gene Pearson r, truth vs truth + strong noise
       across all 25 LuCaP samples (synthetic replicate baseline)
    """
    ne_dir = ensure_dir(os.path.join(out_dir, "NaiveExample"))
    N_RESAMPLES = 200  # re-draws for smoother distributions

    # Load all 25 PC-PDX truth profiles
    pc_truth = pd.read_csv(
        PROJECT_ROOT / "labels" / "PC-PDX_Log2CPMplus1.tsv", sep="\t", index_col=0,
    )
    keep = [s for s in PC_PDX_STUDY if s in pc_truth.columns]
    if len(keep) < 3:
        print("  [skip] NaiveExample: fewer than 3 LuCaP samples in truth")
        return
    T = pc_truth[keep]  # genes × 25 samples
    T_arr = T.values.astype(np.float64)
    n_genes, n_samp = T_arr.shape

    # --- (1) LuCaP_136 vs LuCaP_136CR scatter (biological replicate) ---
    if "LuCaP_136" in T.columns and "LuCaP_136CR" in T.columns:
        s136 = T["LuCaP_136"]
        s136cr = T["LuCaP_136CR"]
        common = s136.dropna().index.intersection(s136cr.dropna().index)
        path_a = os.path.join(ne_dir, "LuCaP_136_vs_136CR")
        if overwrite or not os.path.exists(f"{path_a}.pdf"):
            sample_scatter(
                s136cr.loc[common], s136.loc[common],
                "LuCaP_136 vs 136CR", path_a, goi=None,
            )
    else:
        print("  [skip] NaiveExample: LuCaP_136 or 136CR not in labels")

    # --- (2) LuCaP_136 vs population-mean expression ---
    if "LuCaP_136" in T.columns:
        s136 = T["LuCaP_136"]
        pop_mean = T.mean(axis=1)  # mean across all 25 samples per gene
        common = s136.dropna().index.intersection(pop_mean.dropna().index)
        path_b = os.path.join(ne_dir, "LuCaP_136_vs_population_mean")
        if overwrite or not os.path.exists(f"{path_b}.pdf"):
            sample_scatter(
                s136.loc[common], pop_mean.loc[common],
                "LuCaP_136 vs Population Mean", path_b, goi=None,
            )

    # Helper: vectorised gene-level Pearson r across genes × samples
    def _gene_pearson_r(truth_arr, pred_arr):
        finite = np.isfinite(truth_arr) & np.isfinite(pred_arr)
        n_valid = finite.sum(axis=1)
        n_g = truth_arr.shape[0]
        result = np.full(n_g, np.nan)
        ok = n_valid >= 5
        if not ok.any():
            return result
        Tc = np.where(finite, truth_arr, 0.0)
        Pc = np.where(finite, pred_arr, 0.0)
        nv = n_valid[ok].astype(np.float64)
        st = Tc[ok].sum(axis=1)
        sp = Pc[ok].sum(axis=1)
        stt = (Tc[ok] ** 2).sum(axis=1)
        spp = (Pc[ok] ** 2).sum(axis=1)
        stp = (Tc[ok] * Pc[ok]).sum(axis=1)
        num = nv * stp - st * sp
        with np.errstate(invalid="ignore"):
            den = np.sqrt((nv * stt - st ** 2) * (nv * spp - sp ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            result[ok] = np.where(den > 0, num / den, np.nan)
        return result

    # --- (3) Na\u00efve per-gene Pearson r (row-shuffled baseline) ---
    # Average over N_RESAMPLES shuffles for a smoother distribution.
    all_naive_r = []
    for seed_i in range(N_RESAMPLES):
        naive_pred = generate_null_predictions(T, seed=seed_i)
        r_vals = _gene_pearson_r(T_arr, naive_pred.values.astype(np.float64))
        all_naive_r.append(r_vals[np.isfinite(r_vals)])
    naive_r_concat = np.concatenate(all_naive_r)
    naive_r_series = pd.Series(naive_r_concat, name="pearson_r")
    path_c = os.path.join(ne_dir, "naive_gene_pearson_r_distribution")
    if overwrite or not os.path.exists(f"{path_c}.pdf"):
        metric_distribution(
            naive_r_series, "Per-Gene Pearson r (Na\u00efve)", path_c,
            xlim=(-1.0, 1.0), square=True,
        )

    # --- (4) Synthetic replicate per-gene Pearson r ---
    # pred = truth + Gaussian noise sized so the mean per-gene Pearson r
    # approximates real technical replicates.  The relationship between
    # noise fraction f and E[r] ≈ 1/sqrt(1+f²) at large n; f≈0.48
    # gives r ~ 0.90, representative of replicate concordance.
    NOISE_FRAC = 0.48
    all_noisy_r = []
    for seed_i in range(N_RESAMPLES):
        rng = np.random.default_rng(42 + seed_i)
        gene_std = np.nanstd(T_arr, axis=1, keepdims=True)
        noise = rng.normal(0, 1.0, size=T_arr.shape) * (NOISE_FRAC * gene_std)
        noisy_pred = T_arr + noise
        noisy_pred[~np.isfinite(T_arr)] = np.nan
        r_vals = _gene_pearson_r(T_arr, noisy_pred)
        all_noisy_r.append(r_vals[np.isfinite(r_vals)])
    noisy_r_concat = np.concatenate(all_noisy_r)
    noisy_r_series = pd.Series(noisy_r_concat, name="pearson_r")
    path_d = os.path.join(ne_dir, "synthetic_replicate_gene_pearson_r_distribution")
    if overwrite or not os.path.exists(f"{path_d}.pdf"):
        metric_distribution(
            noisy_r_series, "Per-Gene Pearson r (Synthetic Replicate)", path_d,
            xlim=(-1.0, 1.0), square=True,
        )


def run_benchmarking(args: argparse.Namespace) -> None:
    """Benchmarking analysis: compare models on PHS holdout across TFx/depth grid.

    Loads PHS predictions for Proteus_Base, NoSeq, OnlySeq, XGBoost,
    EPIC-Seq scores, and a Naïve (row-shuffled truth) baseline.
    Computes metrics per (TFx, depth, model) cell and produces grid
    boxplots, histograms, and eCDNA line plots.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Benchmarking_Base")
    print("=" * 70)

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / "Benchmarking_Base"))
    n_workers = args.n_workers

    # ── Load model predictions and build truth ──────────────────────────
    model_preds: Dict[str, pd.DataFrame] = {}  # model_name -> wide (genes × samples)

    for model_name, rel_path in _BENCH_PROTEUS_MODELS.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"  [skip] {model_name}: {rel_path} not found")
            continue
        pred, _, _ = load_holdout_predictions_proteus(str(path))
        model_preds[model_name] = pred
        print(f"  Loaded {model_name}: {pred.shape[0]} genes × {pred.shape[1]} samples")

    # XGBoost uses a different column name ('prediction' not 'prediction_mean')
    xgb_path = PROJECT_ROOT / _BENCH_XGB_PATH
    if xgb_path.exists():
        xgb_pred = load_holdout_predictions_xgboost(str(xgb_path))
        model_preds["XGBoost"] = xgb_pred
        print(f"  Loaded XGBoost: {xgb_pred.shape[0]} genes × {xgb_pred.shape[1]} samples")
    else:
        print(f"  [skip] XGBoost: {_BENCH_XGB_PATH} not found")

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

    # Na\u00efve model: column-wise shuffled truth
    naive_pred = generate_null_predictions(truth_wide, seed=42)
    model_preds["Na\u00efve"] = naive_pred

    # ── Load EPIC-Seq scores ───────────────────────────────────────────
    epic_path = PROJECT_ROOT / _EPIC_SEQ_PATH
    epic_preds: Dict[str, pd.DataFrame] = {}
    if epic_path.exists():
        epic_preds = load_holdout_predictions_epicseq(
            str(epic_path), _EPIC_SCORE_COLS,
        )
        for score_name, epw in epic_preds.items():
            display = _EPIC_DISPLAY_NAMES.get(score_name, f"EPIC_{score_name}")
            model_preds[display] = epw
            print(f"  Loaded {display}: {epw.shape[0]} genes × {epw.shape[1]} samples")
    else:
        print(f"  [skip] EPIC-Seq: {_EPIC_SEQ_PATH} not found")

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
    # Canonical model order for Benchmarking_Base (excludes Proteus_Tuned)
    _BASE_MODEL_ORDER = [
        "Proteus_Base", "Proteus_NoSeq", "Proteus_OnlySeq",
        "XGBoost", "EPIC_GEP", "EPIC_PFE", "EPIC_NDR", "Na\u00efve",
    ]
    model_order = [m for m in _BASE_MODEL_ORDER if m in model_preds]

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
        os.path.join(out_dir, "summary_benchmarking.tsv"), sep="\t", index=False,
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

    # Grid boxplots for key scalar metrics
    for metric_col, ylabel in [
        ("gvw_rho", "GVW-ρ"),
        ("mean_gene_rho", "Mean Gene ρ"),
        ("rmse", "Mean Gene RMSE"),
        ("mean_top10pct_gene_rho", "Mean ρ (Top 10%)"),
        ("n_genes_rho_ge070", "Genes with ρ≥0.70"),
    ]:
        _maybe(benchmark_grid_boxplots, args.overwrite,
               metrics_table, os.path.join(out_dir, f"benchmark_grid_{metric_col}"),
               metric_col=metric_col, ylabel=ylabel,
               title=f"Benchmarking — {ylabel}",
               tfx_order=_BENCH_TFX_ORDER, depth_order=_BENCH_DEPTH_ORDER,
               model_order=model_order)

    # Grid boxplots for per-sample Spearman rho (one box per model)
    if not sample_rho_table.empty:
        _maybe(benchmark_grid_boxplots, args.overwrite,
               sample_rho_table,
               os.path.join(out_dir, "benchmark_grid_sample_rho"),
               metric_col="spearman_rho", ylabel="Sample ρ",
               title="Benchmarking — Per-Sample Spearman ρ",
               tfx_order=_BENCH_TFX_ORDER, depth_order=_BENCH_DEPTH_ORDER,
               model_order=model_order)

    # Grid histograms of per-gene Spearman rho (all genes)
    if not gene_rho_table.empty:
        _maybe(benchmark_grid_histograms, args.overwrite,
               gene_rho_table, os.path.join(out_dir, "benchmark_grid_gene_rho_hist"),
               tfx_order=_BENCH_TFX_ORDER, depth_order=_BENCH_DEPTH_ORDER,
               model_order=model_order)

        # Top 10% most variable genes only
        if var_map:
            var_s = pd.Series(var_map)
            n_top = max(1, int(np.ceil(len(var_s) * 0.10)))
            top10_genes = set(var_s.nlargest(n_top).index)
            top10_rho = gene_rho_table[gene_rho_table["gene"].isin(top10_genes)]
            if not top10_rho.empty:
                _maybe(benchmark_grid_histograms, args.overwrite,
                       top10_rho,
                       os.path.join(out_dir, "benchmark_grid_gene_rho_hist_top10pct"),
                       tfx_order=_BENCH_TFX_ORDER, depth_order=_BENCH_DEPTH_ORDER,
                       model_order=model_order,
                       title="Per-Gene Spearman ρ (Top 10% Var.)")


    # eCDNA line plots
    if not ecdna_table.empty:
        _maybe(benchmark_ecdna_lines, args.overwrite,
               ecdna_table, os.path.join(out_dir, "benchmark_ecdna_lines"),
               model_order=model_order)

    # ── NaiveExample: illustrative scatter / distribution baseline plots ──
    _generate_naive_example(out_dir, args.overwrite)

    print("  Done")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GROUPS = {
    "BTS-CV": run_bts_cv,
    "Benchmarking_Base": run_benchmarking,
    "Benchmarking_PAN-FT": run_benchmarking_panft,
    "PAN-FT-CV": run_panft_cv,
    "CRPC-FT-CV": run_crpc_finetune_cv,
    "XGB-FT-CV": run_xgb_finetune_cv,
    "Patient_Validation": run_patient_validation,
    "Patient_Discovery": run_patient_discovery,
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
        "--proteus_holdout", type=str, default=None,
        help="Path to Proteus holdout/inference output dir (has predictions_tumor.tsv).",
    )
    parser.add_argument(
        "--xgb_holdout", type=str, default=None,
        help="Path to XGBoost holdout output dir (has predictions_tumor.tsv).",
    )
    parser.add_argument(
        "--truth_cohort", type=str, default=None,
        help="Cohort name for truth labels (e.g. PC-PDX, PHS-tumor).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate all plots even if PDFs already exist.",
    )
    parser.add_argument(
        "--n_workers", type=int, default=0,
        help="Number of CPU workers for parallel operations (0 = all CPUs).",
    )
    parser.add_argument(
        "--n_bootstraps", type=int, default=100,
        help="Bootstrap resamples for scalar metric CIs (0 = off).",
    )
    parser.add_argument(
        "--scatter_ci", action="store_true",
        help="Show CI error bars on sample/gene scatter plots (off by default).",
    )

    args = parser.parse_args()

    # Resolve n_workers
    available_cpus = os.cpu_count() or 1
    if args.n_workers <= 0:
        args.n_workers = max(1, available_cpus)
    print(f"Available CPUs: {available_cpus}  |  Workers requested: {args.n_workers}")

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
