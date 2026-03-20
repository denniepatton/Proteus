#!/usr/bin/env python3
"""
run_full_analysis.py
Proteus analysis framework — main orchestrator.

Usage examples:
    python analysis/run_full_analysis.py --group BTS-CV
    python analysis/run_full_analysis.py --group Patient-HO
    python analysis/run_full_analysis.py --group Patient-HO \
        --proteus_holdout results_holdout/BioDVN/CRPC-WCDT \
        --truth_cohort CRPC-WCDT
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
    build_study_var_map,
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

    # Build the single study-wide variance map (25+25 PDX tumor only)
    study_var_map = build_study_var_map()
    print(f"  Study variance map: {len(study_var_map)} genes")

    # Per-cohort variance maps using study tumor samples only
    cohort_var_maps: Dict[str, Dict[str, float]] = {}
    for cohort_name, cfg in _COHORT_CONFIG.items():
        cohort_samples = [s for s in cfg["study_samples"] if s in canonical_truth.columns]
        if cohort_samples:
            cohort_var_maps[cohort_name] = compute_gene_variance(canonical_truth[cohort_samples])
            print(f"  Variance map for {cohort_name}: {len(cohort_samples)} samples")
        else:
            cohort_var_maps[cohort_name] = study_var_map

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
        var_map = cohort_var_maps[cohort_name]

        print(f"\n  ── Metrics: {cohort_name} (CV only, {len(cv_samples)} samples) ──")
        cv_truth = canonical_truth.reindex(index=cp.index, columns=cp.columns)
        cv_summary = cohort_summary_table(cv_truth, cp, label=f"CV ({cohort_name})",
                                          n_workers=n_workers, var_map=var_map)
        summaries.append(cv_summary)

        if ho_pred_w is not None and len(ho_samples) >= 2:
            print(f"  ── Metrics: {cohort_name} (HO only, {len(ho_samples)} samples) ──")
            ho_truth = canonical_truth.reindex(index=ho_pred_w.index,
                                               columns=ho_pred_w.columns)
            ho_summary = cohort_summary_table(ho_truth, ho_pred_w,
                                              label=f"HO ({cohort_name})",
                                              n_workers=n_workers, var_map=var_map)
            summaries.append(ho_summary)

        print(f"  ── Metrics: {cohort_name} (Combined, {combined_pred.shape[1]} samples) ──")
        comb_summary = cohort_summary_table(combined_truth, combined_pred,
                                            label=f"Combined ({cohort_name})",
                                            n_workers=n_workers, var_map=var_map)
        summaries.append(comb_summary)

        # ── Generate plots and outputs using combined data ──
        _run_cv_suite(
            combined_pred, combined_truth, f"BTS ({cohort_name})", cohort_out,
            ci_lo=combined_ci_lo, ci_hi=combined_ci_hi,
            goi=cfg["goi"], subtypes_df=subtypes_with_status,
            heatmap_gene_sets=cfg["heatmap_gene_sets"],
            gene_scatter_sets=cfg.get("gene_scatter_sets", []),
            var_map=var_map,
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

    # Per-gene target scatters for GOI sets
    if gene_scatter_sets:
        for gs_name in gene_scatter_sets:
            resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
            if not gene_list_available(resolved):
                continue
            genes = get_gene_list(resolved)
            gs_dir = ensure_dir(os.path.join(out_dir, f"gene_scatters_{gs_name}"))
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


# ---------------------------------------------------------------------------
# Analysis group: Patient-HO (Holdout — Patient Biopsies)
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
    for gs_name in ["PC-Phenotyping", "PAM50", "Targets_PC"]:
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

    batch_sample_scatters(
        tw, pw, sample_dir,
        ci_lo_wide=ci_lo_aligned, ci_hi_wide=ci_hi_aligned,
        goi=DEFAULT_GOI_PC, subtypes_df=subtypes_df, n_workers=n_workers,
    )

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")
    return summary


# ---------------------------------------------------------------------------
# Analysis group: Patient-HO (patient holdout with biopsy truth)
# ---------------------------------------------------------------------------

# Default patient holdout cohorts: (truth_cohort, holdout_predictions_dir)
# Uses base model (BioDVN) results by default.
_PATIENT_HO_DEFAULTS = [
    ("CRPC-WCDT", "results_holdout/BioDVN/CRPC-WCDT"),
    ("BLCA-RA",   "results_holdout/BioDVN/BLCA-RA"),
]


def _run_patient_ho_cohort(
    truth_cohort: str,
    ho_dir: str,
    out_dir: str,
    args: argparse.Namespace,
    summaries: List[Dict[str, Any]],
) -> None:
    """Run Patient-HO analysis for a single cohort."""
    truth_path = PROJECT_ROOT / "labels" / f"{truth_cohort}_Log2CPMplus1.tsv"
    if not truth_path.exists():
        print(f"  [skip] Truth not found: {truth_cohort}")
        return
    truth_wide = load_truth(truth_cohort)

    subtypes_df = None
    sub_path = PROJECT_ROOT / "labels" / f"{truth_cohort}_TFX_Subtype.tsv"
    if sub_path.exists():
        subtypes_df = load_subtypes(truth_cohort)

    tfx_dict = None
    if subtypes_df is not None and "tfx" in subtypes_df.columns:
        tfx_dict = subtypes_df["tfx"].to_dict()

    ho_path = os.path.join(_resolve(ho_dir), "predictions_tumor.tsv")
    if not os.path.exists(ho_path):
        print(f"  [skip] Predictions not found: {ho_path}")
        return

    proteus_pred, ci_lo, ci_hi = load_holdout_predictions_proteus(ho_path)
    cohort_tag = truth_cohort.replace("/", "_")
    p_out = ensure_dir(os.path.join(out_dir, cohort_tag, "Proteus"))

    _run_holdout_suite(
        proteus_pred, truth_wide, f"Proteus-{cohort_tag}", p_out,
        ci_lo=ci_lo, ci_hi=ci_hi, subtypes_df=subtypes_df,
        n_workers=args.n_workers, overwrite=args.overwrite, summaries=summaries,
        scatter_ci=args.scatter_ci,
    )

    # Gene target scatters
    _ci_lo_tgt = ci_lo if args.scatter_ci else None
    _ci_hi_tgt = ci_hi if args.scatter_ci else None
    for gs_name in ["Targets_PC", "Targets_SCLC", "Targets_BC"]:
        resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
        if not gene_list_available(resolved):
            continue
        genes = get_gene_list(resolved)
        target_dir = ensure_dir(os.path.join(p_out, f"target_scatters_{gs_name}"))
        batch_gene_scatters(
            truth_wide, proteus_pred, genes, target_dir,
            subtypes_df=subtypes_df, tfx_dict=tfx_dict,
            ci_lo_wide=_ci_lo_tgt, ci_hi_wide=_ci_hi_tgt,
            n_workers=args.n_workers,
        )

    # XGBoost holdout (optional)
    if args.xgb_holdout:
        xgb_ho_path = os.path.join(_resolve(args.xgb_holdout), "predictions_tumor.tsv")
        if os.path.exists(xgb_ho_path):
            xgb_pred = load_holdout_predictions_xgboost(xgb_ho_path)
            x_out = ensure_dir(os.path.join(out_dir, cohort_tag, "XGBoost"))
            _run_holdout_suite(
                xgb_pred, truth_wide, f"XGBoost-{cohort_tag}", x_out,
                n_workers=args.n_workers, overwrite=args.overwrite, summaries=summaries,
            )

    if summaries:
        print("\n" + "─" * 70)
        print_summary_table(summaries)
        _save_summary_tsv(summaries, os.path.join(out_dir, cohort_tag, "summary_metrics.tsv"))


def run_patient_ho(args: argparse.Namespace) -> None:
    """Patient holdout analysis: predictions vs biopsy truth.

    If --proteus_holdout and --truth_cohort are given, runs for that single
    cohort.  Otherwise, defaults to CRPC-WCDT and BLCA-RA using the base
    model (BioDVN) holdout predictions.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Patient-HO")
    print("=" * 70)

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / "Patient-HO"))

    # Explicit single-cohort mode
    if args.proteus_holdout and args.truth_cohort:
        summaries: List[Dict[str, Any]] = []
        _run_patient_ho_cohort(
            args.truth_cohort, args.proteus_holdout, out_dir, args, summaries,
        )
        return

    # Default: iterate over standard patient holdout cohorts
    for truth_cohort, ho_dir in _PATIENT_HO_DEFAULTS:
        resolved = str(PROJECT_ROOT / ho_dir)
        if not os.path.isdir(resolved):
            print(f"  [skip] Holdout dir not found: {ho_dir}")
            continue
        print(f"\n  ── {truth_cohort} (base model) ──")
        summaries = []
        _run_patient_ho_cohort(truth_cohort, ho_dir, out_dir, args, summaries)


# ---------------------------------------------------------------------------
# Analysis group: Patient-Discovery (no truth)
# ---------------------------------------------------------------------------

def run_patient_discovery(args: argparse.Namespace) -> None:
    """Discovery cohort analysis: predictions only (no biopsy truth)."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Patient-Discovery")
    print("=" * 70)

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / "Patient-Discovery"))

    if not args.proteus_holdout:
        print("  [skip] --proteus_holdout required")
        return

    ho_path = os.path.join(_resolve(args.proteus_holdout), "predictions_tumor.tsv")
    if not os.path.exists(ho_path):
        print(f"  [skip] {ho_path} not found")
        return

    cohort_tag = args.truth_cohort or Path(args.proteus_holdout).name
    p_out = ensure_dir(os.path.join(out_dir, cohort_tag))

    pred, ci_lo, ci_hi = load_holdout_predictions_proteus(ho_path)
    print(f"  Predictions: {pred.shape[0]} genes × {pred.shape[1]} samples")

    # Heatmaps for key gene sets
    subtypes_df = None
    tfx_dict = None
    sub_dict = None
    sub_path = PROJECT_ROOT / "labels" / f"{cohort_tag}_TFX_Subtype.tsv"
    if sub_path.exists():
        subtypes_df = load_subtypes(cohort_tag)
        tfx_dict = subtypes_df["tfx"].to_dict()
        if "subtype" in subtypes_df.columns:
            sub_dict = subtypes_df["subtype"].to_dict()

    # Expression profiles (colored by subtype)
    _maybe(sample_expression_kde, args.overwrite, pred,
           os.path.join(p_out, "expression_profiles"), subtypes_df=subtypes_df)

    for gs_name in ["PC-Phenotyping", "Targets_PC", "PAM50"]:
        resolved = STANDARD_GENE_SETS.get(gs_name, gs_name)
        if not gene_list_available(resolved):
            continue
        genes = get_gene_list(resolved)
        _maybe(
            single_heatmap, args.overwrite, pred, genes,
            os.path.join(p_out, f"heatmap_{gs_name}"),
            tfx_dict=tfx_dict, sub_dict=sub_dict,
        )

    pred.to_csv(os.path.join(p_out, "predictions_wide.tsv"), sep="\t")
    print("  Done")


# ---------------------------------------------------------------------------
# Group: PAN-FT-CV / CRPC-FT-CV (fine-tune CV)
# ---------------------------------------------------------------------------

_FINETUNE_CV_PATHS: Dict[str, str] = {
    "PAN-FT-CV": "results_training/proteus_pan-finetune_cv",
    "CRPC-FT-CV": "results_training/proteus_crpc-finetune_cv",
}


def run_finetune_cv(args: argparse.Namespace, group_name: str) -> None:
    """Fine-tune CV analysis (PAN-FT-CV or CRPC-FT-CV)."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS GROUP: {group_name}")
    print("=" * 70)

    cv_rel = _FINETUNE_CV_PATHS.get(group_name)
    if cv_rel is None:
        raise ValueError(f"Unknown finetune CV group: {group_name}")
    cv_dir = str(PROJECT_ROOT / cv_rel)
    if not os.path.isdir(cv_dir):
        print(f"  [skip] CV results not found: {cv_rel}")
        return

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / group_name))
    summaries: List[Dict[str, Any]] = []
    n_workers = args.n_workers

    print(f"  Loading Proteus CV from: {cv_dir}")
    pred, truth, ci_lo, ci_hi = load_cv_predictions_proteus(cv_dir)
    # Exclude PHT admix samples from CV analyses
    pred, truth, ci_lo, ci_hi = _exclude_pht(pred, truth, ci_lo, ci_hi)

    ft_out = ensure_dir(os.path.join(out_dir, "Proteus"))
    summary = cohort_summary_table(truth, pred, label=group_name, n_workers=n_workers)
    summaries.append(summary)

    _run_cv_suite(
        pred, truth, group_name, ft_out,
        ci_lo=ci_lo, ci_hi=ci_hi,
        gene_met=summary["_gene_metrics"],
        sample_met=summary["_sample_metrics"],
        n_workers=n_workers, overwrite=args.overwrite,
        scatter_ci=args.scatter_ci,
    )

    if summaries:
        print("\n" + "─" * 70)
        print_summary_table(summaries)
        _save_summary_tsv(summaries, os.path.join(out_dir, "summary_metrics.tsv"))


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
    "Proteus-Base": "results_holdout/BioDVN/PHS/predictions_tumor.tsv",
    "Proteus-NoSeq": "results_holdout/BioDVN_NoSeq/PHS/predictions_tumor.tsv",
    "Proteus-OnlySeq": "results_holdout/BioDVN_OnlySeq/PHS/predictions_tumor.tsv",
}
_BENCH_XGB_PATH = "results_holdout/XGBoost/PHS/predictions_tumor.tsv"
_EPIC_SEQ_PATH = "results_holdout/EPIC-Seq/PHS/EPICSeqInferredExpressionValues.txt"
_EPIC_SCORE_COLS = ["inferredGEP", "PFE", "NDR_relative_2k", "Gini", "Shannon", "OCF_Ratio"]


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


def _generate_null_example(
    out_dir: str,
    overwrite: bool,
) -> None:
    """Produce NullExample illustrative plots.

    Uses all 25 LuCaP PC-PDX truth profiles to illustrate two baselines
    at both the single-transcriptome (scatter) and per-gene (distribution)
    levels:

    1. Scatter: LuCaP_136 vs 136CR (biological replicate pair)
    2. Scatter: LuCaP_136 vs population-mean expression (regression to mean)
    3. Distribution: per-gene Pearson r, truth vs row-shuffled truth
       across all 25 LuCaP samples (null baseline, same as benchmarking)
    4. Distribution: per-gene Pearson r, truth vs truth + faint noise
       across all 25 LuCaP samples (near-replicate baseline)
    """
    ne_dir = ensure_dir(os.path.join(out_dir, "NullExample"))

    # Load all 25 PC-PDX truth profiles
    pc_truth = pd.read_csv(
        PROJECT_ROOT / "labels" / "PC-PDX_Log2CPMplus1.tsv", sep="\t", index_col=0,
    )
    keep = [s for s in PC_PDX_STUDY if s in pc_truth.columns]
    if len(keep) < 3:
        print("  [skip] NullExample: fewer than 3 LuCaP samples in truth")
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
        print("  [skip] NullExample: LuCaP_136 or 136CR not in labels")

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

    # --- (3) Null per-gene Pearson r (row-shuffled baseline) ---
    # Same approach as generate_null_predictions: shuffle within each row
    # (gene) to break per-gene correlations, then compute per-gene r.
    null_pred = generate_null_predictions(T, seed=42)
    N_arr = null_pred.values.astype(np.float64)

    # Per-gene Pearson r between truth and null (vectorised across axis=1)
    finite = np.isfinite(T_arr) & np.isfinite(N_arr)
    n_valid = finite.sum(axis=1)
    null_gene_r = np.full(n_genes, np.nan)
    ok = n_valid >= 5
    if ok.any():
        Tc = np.where(finite, T_arr, 0.0)
        Nc = np.where(finite, N_arr, 0.0)
        nv = n_valid[ok].astype(np.float64)
        st = Tc[ok].sum(axis=1)
        sn = Nc[ok].sum(axis=1)
        stt = (Tc[ok] ** 2).sum(axis=1)
        snn = (Nc[ok] ** 2).sum(axis=1)
        stn = (Tc[ok] * Nc[ok]).sum(axis=1)
        num = nv * stn - st * sn
        with np.errstate(invalid="ignore"):
            den = np.sqrt((nv * stt - st ** 2) * (nv * snn - sn ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            null_gene_r[ok] = np.where(den > 0, num / den, np.nan)

    null_r_series = pd.Series(null_gene_r[np.isfinite(null_gene_r)], name="pearson_r")
    path_c = os.path.join(ne_dir, "null_gene_pearson_r_distribution")
    if overwrite or not os.path.exists(f"{path_c}.pdf"):
        metric_distribution(
            null_r_series, "Per-Gene Pearson r (Null)", path_c,
            xlim=(-1.0, 1.0), square=True,
        )

    # --- (4) Near-replicate per-gene Pearson r ---
    # pred = truth + Gaussian noise (sigma = 50% of per-gene std)
    rng = np.random.default_rng(42)
    gene_std = np.nanstd(T_arr, axis=1, keepdims=True)  # (G, 1)
    noise = rng.normal(0, 1.0, size=T_arr.shape) * (0.50 * gene_std)
    noisy_pred = T_arr + noise
    noisy_pred[~np.isfinite(T_arr)] = np.nan

    # Per-gene Pearson r between truth and near-replicate (axis=1)
    finite_n = np.isfinite(T_arr) & np.isfinite(noisy_pred)
    nv_n = finite_n.sum(axis=1)
    noisy_gene_r = np.full(n_genes, np.nan)
    ok_n = nv_n >= 5
    if ok_n.any():
        Tc_n = np.where(finite_n, T_arr, 0.0)
        Pc_n = np.where(finite_n, noisy_pred, 0.0)
        nvn = nv_n[ok_n].astype(np.float64)
        st_n = Tc_n[ok_n].sum(axis=1)
        sp_n = Pc_n[ok_n].sum(axis=1)
        stt_n = (Tc_n[ok_n] ** 2).sum(axis=1)
        spp_n = (Pc_n[ok_n] ** 2).sum(axis=1)
        stp_n = (Tc_n[ok_n] * Pc_n[ok_n]).sum(axis=1)
        num_n = nvn * stp_n - st_n * sp_n
        with np.errstate(invalid="ignore"):
            den_n = np.sqrt((nvn * stt_n - st_n ** 2) * (nvn * spp_n - sp_n ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            noisy_gene_r[ok_n] = np.where(den_n > 0, num_n / den_n, np.nan)

    noisy_r_series = pd.Series(noisy_gene_r[np.isfinite(noisy_gene_r)], name="pearson_r")
    path_d = os.path.join(ne_dir, "near_replicate_gene_pearson_r_distribution")
    if overwrite or not os.path.exists(f"{path_d}.pdf"):
        metric_distribution(
            noisy_r_series, "Per-Gene Pearson r (Near Replicate)", path_d,
            xlim=(-1.0, 1.0), square=True,
        )


def run_benchmarking(args: argparse.Namespace) -> None:
    """Benchmarking analysis: compare models on PHS holdout across TFx/depth grid.

    Loads PHS predictions for Proteus-Base, NoSeq, OnlySeq, XGBoost,
    EPIC-Seq scores, and a Null (column-shuffled truth) baseline.
    Computes metrics per (TFx, depth, model) cell and produces grid
    boxplots, histograms, and eCDNA line plots.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Benchmarking")
    print("=" * 70)

    out_dir = ensure_dir(str(PROJECT_ROOT / "analysis" / "results" / "Benchmarking"))
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

    # Null model: column-wise shuffled truth
    null_pred = generate_null_predictions(truth_wide, seed=42)
    model_preds["Null"] = null_pred

    # ── Load EPIC-Seq scores ───────────────────────────────────────────
    epic_path = PROJECT_ROOT / _EPIC_SEQ_PATH
    epic_preds: Dict[str, pd.DataFrame] = {}
    if epic_path.exists():
        epic_preds = load_holdout_predictions_epicseq(
            str(epic_path), _EPIC_SCORE_COLS,
        )
        for score_name, epw in epic_preds.items():
            model_preds[f"EPIC-{score_name}"] = epw
            print(f"  Loaded EPIC-{score_name}: {epw.shape[0]} genes × {epw.shape[1]} samples")
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
    model_order = [m for m in list(_BENCH_PROTEUS_MODELS.keys()) + ["XGBoost"] if m in model_preds]
    # Add EPIC-Seq score models
    for score_name in _EPIC_SCORE_COLS:
        epic_key = f"EPIC-{score_name}"
        if epic_key in model_preds:
            model_order.append(epic_key)
    model_order.append("Null")

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

    # ── NullExample: illustrative scatter / distribution baseline plots ──
    _generate_null_example(out_dir, args.overwrite)

    print("  Done")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

GROUPS = {
    "BTS-CV": run_bts_cv,
    "Benchmarking": run_benchmarking,
    "PAN-FT-CV": lambda args: run_finetune_cv(args, "PAN-FT-CV"),
    "CRPC-FT-CV": lambda args: run_finetune_cv(args, "CRPC-FT-CV"),
    "XGB-FT-CV": run_xgb_finetune_cv,
    "Patient-HO": run_patient_ho,
    "Patient-Discovery": run_patient_discovery,
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
