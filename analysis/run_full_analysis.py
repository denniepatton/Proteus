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
import shutil
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
    EXPRESSION_CEIL,
    EXPRESSION_FLOOR,
    _flatten_wide_pair,
    _augment_missing_gene_aliases,
    benjamini_hochberg,
    clean_epicseq_sample_names,
    clean_epicseq_sample_names_wcdt,
    compute_activity_odds_ratio_off_curve,
    compute_activity_odds_ratio_on_curve,
    cohort_summary_table,
    compute_cox_multivariate,
    compute_logrank_p,
    compute_cox_univariate,
    compute_gene_variance,
    compute_geneset_score,
    compute_onoff_confusion,
    compute_subset_metrics,
    ensure_dir,
    extract_pluvicto_patient_id,
    format_pval,
    gene_level_metrics,
    gene_list_available,
    generate_null_predictions,
    get_gene_list,
    GENE_SET_BREAKS,
    load_cv_predictions_proteus_full_from_paths,
    load_holdout_predictions_epicseq,
    load_holdout_predictions_proteus_full,
    load_holdout_predictions_xgboost,
    load_palette,
    load_pdx_truth_for_phs,
    load_pluvicto_clinical,
    load_pluvicto_outcomes,
    load_subtypes,
    load_truth,
    MIN_CORRELATION_N,
    mdn_activity_odds_ratio_wide,
    mdn_on_probability_wide,
    mdn_pit_values_wide,
    mdn_monte_carlo_sample,
    mdn_quantile_interval_wide,
    pam50_classify_genefu,
    parse_phs_sample,
    print_summary_table,
    sample_level_metrics,
    SCLC_PDX_STUDY,
    STANDARD_GENE_SETS,
    strip_cohort_prefix,
    VARIANCE_BINS,
    variance_group_metrics,
)
from analysis.calibration import (
    CALIBRATION_DRAW_SEED as _CALIBRATION_DRAW_SEED,
    apply_distribution_calibration_bundle as _apply_distribution_calibration_bundle,
    bundle_sample_fold_subset as _bundle_sample_fold_subset,
    constant_source_by_sample as _constant_source_by_sample,
    get_calibrated_activity_surfaces as _get_calibrated_activity_surfaces,
    get_calibrated_draws as _get_calibrated_draws,
    get_patient_distribution_calibration_state as _get_patient_distribution_calibration_state,
    write_distribution_calibration_outputs as _write_distribution_calibration_outputs,
)
from analysis.enrichment_utils import (
    aggregate_ssgsea_score_matrices as _aggregate_ssgsea_score_matrices,
    compute_ssgsea_score_matrix as _compute_ssgsea_score_matrix,
    convert_entrez_gmt as _convert_entrez_gmt,
    load_gmt as _load_gmt,
    prepare_expression_matrix_for_scores as _prepare_expression_matrix_for_scores,
)
from analysis.examples import run_examples as run_examples_workflow
from analysis.pipeline_config import SECTION_OUTPUT_DIRS
from analysis.pipeline_config import (
    BENCHMARK_COMPLETE_MODELS,
    BENCHMARK_SUBSET_MODELS,
    DISCOVERY_COHORTS as CONFIG_DISCOVERY_COHORTS,
    DISCOVERY_PROTEUS_PATHS,
    EPIC_DISPLAY_NAMES as CONFIG_EPIC_DISPLAY_NAMES,
    EPIC_SCORE_COLUMNS as CONFIG_EPIC_SCORE_COLUMNS,
    PATIENT_BENCHMARK_COHORTS,
    PATIENT_EPIC_PATHS,
    PLCA_CV_FOLD_TUMOR_PREDICTION_PATHS,
    PATIENT_PROTEUS_MODEL_PATHS,
    PATIENT_XGB_PATHS,
    SYNTHETIC_EPIC_PATH,
    SYNTHETIC_PROTEUS_MODEL_PATHS,
    SYNTHETIC_XGB_PATH,
    VALIDATION_PATIENT_CV_COHORTS,
    VALIDATION_PATIENT_HOLDOUT_COHORTS,
    VALIDATION_PATIENT_HOLDOUT_PATHS,
    VALIDATION_PDX_COHORTS,
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
    binary_status_boxplot,
    clinical_correlation_scatter,
    confusion_matrix_heatmap,
    double_heatmap,
    forest_plot,
    gene_target_scatter,
    kaplan_meier_plot,
    metric_distribution,
    metric_distribution_by_variance,
    plot_pit_histogram,
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


_MDN_PARAM_KEYS = (
    "p0_wide",
    "pi_gamma_wide",
    "pi_normal_wide",
    "gamma_beta_wide",
    "normal_mean_wide",
    "normal_stdev_wide",
)
_ACTIVE_CALL_TARGET_PPV = 0.90
_INACTIVE_CALL_TARGET_NPV = 0.90
_PRECISION_WEIGHT_EPSILON = 1e-6
_DEFAULT_MONTE_CARLO_DRAWS = 100
_DOWNSTREAM_ACTIVE_DIM_THRESHOLD = 0.5

DEFAULT_GOI_PC: Dict[str, List[str]] = {
    "AR": ["AR", "KLK3", "NKX3-1", "FOLH1", "STEAP1"],
    "NE": ["SYP", "CHGA", "ASCL1", "INSM1", "SOX2", "DLL3"],
    "Proliferation": ["MKI67", "TOP2A", "CDK1"],
}
DEFAULT_GOI_BLCA: Dict[str, List[str]] = {
    "BLCA": ["ERBB2", "FGFR3", "NECTIN4", "TACSTD2", "TOP2A"],
}
DEFAULT_GOI_SCLC: Dict[str, List[str]] = {
    "NE": ["ASCL1", "NEUROD1", "INSM1", "SYP", "DLL3"],
    "Inflammatory": ["B2M", "TAP1", "TAP2", "PSMB8", "PSMB9"],
}


_TFX_FILTER_NOTICE_KEYS: set[tuple[Any, ...]] = set()


def _section_output_path(section_name: str) -> Path:
    section_dir = SECTION_OUTPUT_DIRS.get(section_name, section_name)
    return PROJECT_ROOT / "analysis" / "results" / section_dir


def _section_results_dir(section_name: str) -> str:
    return ensure_dir(str(_section_output_path(section_name)))


def _require_existing_path(path: Path, *, description: str) -> Path:
    """Return *path* when it exists, otherwise raise a precise file error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required {description}: {path}")
    return path


def _should_skip_section(section_name: str, args: argparse.Namespace) -> bool:
    out_dir = _section_output_path(section_name)
    if out_dir.exists() and not getattr(args, "overwrite", False):
        print(f"  [skip] {section_name}: output exists ({out_dir}); pass --overwrite to regenerate")
        return True
    return False


def _is_patient_cohort(cohort: str) -> bool:
    cohort_name = str(cohort)
    return cohort_name not in {"PHS", "PC-PDX", "SCLC-PDX"} and not cohort_name.endswith("-PDX")


def _print_input_filter_banner(minimum_tfx: float) -> None:
    print(f"\nInput TFx filter: retaining patient samples with TFx >= {minimum_tfx:.3f}")


def _cohort_supports_tfx_subtypes(cohort: str, *, label_name: Optional[str] = None) -> bool:
    resolved_label = str(label_name or cohort)
    cohort_name = str(cohort)
    unsupported = {"CRPC-Pfizer", "BC-Ghana"}
    return cohort_name not in unsupported and resolved_label not in unsupported


def _note_tfx_filter_unavailable(report_context: str) -> None:
    notice_key = ("minimum_tfx_unavailable", str(report_context))
    if notice_key in _TFX_FILTER_NOTICE_KEYS:
        return
    _TFX_FILTER_NOTICE_KEYS.add(notice_key)
    print(f"  {report_context}: no TFx/subtype table available; skipping minimum TFx filtering")


def _sample_column_subset(table: pd.DataFrame, sample_names: Sequence[Any]) -> pd.DataFrame:
    column_lookup = {str(column): column for column in table.columns}
    selected_columns = [column_lookup[str(sample)] for sample in sample_names if str(sample) in column_lookup]
    return table.loc[:, selected_columns].copy()


def _print_sample_name_list(sample_names: Sequence[Any], *, prefix: str) -> None:
    """Print sample names in short wrapped chunks for CLI diagnostics."""
    names = [str(sample_name) for sample_name in sample_names]
    if not names:
        return
    chunk_size = 12
    for start in range(0, len(names), chunk_size):
        chunk = names[start:start + chunk_size]
        print(f"{prefix}{', '.join(chunk)}")


def _report_sample_drops(
    context: str,
    input_samples: Sequence[Any],
    kept_samples: Sequence[Any],
    *,
    reason_by_sample: Optional[Dict[str, str]] = None,
) -> None:
    """Print retained/dropped sample counts and explicit drop reasons."""
    ordered_input = [str(sample_name) for sample_name in input_samples]
    kept_set = {str(sample_name) for sample_name in kept_samples}
    dropped_samples = [sample_name for sample_name in ordered_input if sample_name not in kept_set]
    print(f"    {context}: retained {len(kept_set)}/{len(ordered_input)} samples")
    if not dropped_samples:
        return

    grouped_drops: Dict[str, List[str]] = {}
    for sample_name in dropped_samples:
        reason = "filtered" if reason_by_sample is None else reason_by_sample.get(sample_name, "filtered")
        grouped_drops.setdefault(reason, []).append(sample_name)

    for reason, sample_names in grouped_drops.items():
        print(f"      dropped {len(sample_names)} samples: {reason}")
        _print_sample_name_list(sample_names, prefix="        ")


def _filter_wide_to_samples(
    wide_df: Optional[pd.DataFrame],
    sample_names: Optional[Sequence[Any]],
) -> Optional[pd.DataFrame]:
    if wide_df is None:
        return None
    if sample_names is None:
        return wide_df.copy()
    return _sample_column_subset(wide_df, sample_names)


def _dedupe_names_preserve_order(names: Sequence[Any]) -> List[Any]:
    deduped: List[Any] = []
    seen_names: set[str] = set()
    for name in names:
        key = str(name)
        if key in seen_names:
            continue
        deduped.append(name)
        seen_names.add(key)
    return deduped


def _filter_wide_to_genes(
    wide_df: Optional[pd.DataFrame],
    gene_names: Optional[Sequence[Any]],
) -> Optional[pd.DataFrame]:
    if wide_df is None:
        return None
    if gene_names is None:
        return wide_df.copy()
    gene_lookup = {str(gene_name): gene_name for gene_name in wide_df.index}
    selected_genes = [
        gene_lookup[str(gene_name)]
        for gene_name in _dedupe_names_preserve_order(gene_names)
        if str(gene_name) in gene_lookup
    ]
    return wide_df.reindex(index=pd.Index(selected_genes)).copy()


def _filter_prediction_bundle_to_samples(
    bundle: Dict[str, Any],
    sample_names: Sequence[Any],
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = dict(bundle)
    sample_set = {str(sample) for sample in sample_names}
    for key, value in bundle.items():
        if key == "fold_data" and isinstance(value, dict):
            filtered[key] = {
                fold: _filter_prediction_bundle_to_samples(fold_data, sample_names)
                for fold, fold_data in value.items()
                if isinstance(fold_data, dict)
            }
        elif isinstance(value, pd.DataFrame) and any(str(column) in sample_set for column in value.columns):
            filtered[key] = _sample_column_subset(value, sample_names)
        elif key == "sample_fold" and isinstance(value, dict):
            filtered[key] = {sample: fold for sample, fold in value.items() if str(sample) in sample_set}
    return filtered


def _filter_prediction_bundle_to_genes(
    bundle: Dict[str, Any],
    gene_names: Sequence[Any],
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = dict(bundle)
    gene_set = {str(gene_name) for gene_name in gene_names}
    for key, value in bundle.items():
        if key == "fold_data" and isinstance(value, dict):
            filtered[key] = {
                fold: _filter_prediction_bundle_to_genes(fold_data, gene_names)
                for fold, fold_data in value.items()
                if isinstance(fold_data, dict)
            }
        elif isinstance(value, pd.DataFrame) and any(str(index_name) in gene_set for index_name in value.index):
            filtered[key] = _filter_wide_to_genes(value, gene_names)
    return filtered


def _concatenate_prediction_bundles_by_columns(
    *bundles: Dict[str, Any],
) -> Dict[str, Any]:
    """Concatenate aligned prediction bundle tables across sample columns."""
    if not bundles:
        return {}

    combined: Dict[str, Any] = {}
    data_keys = {
        key
        for bundle in bundles
        for key, value in bundle.items()
        if isinstance(value, pd.DataFrame)
    }
    for key in data_keys:
        tables = [bundle.get(key) for bundle in bundles]
        if not all(isinstance(table, pd.DataFrame) for table in tables):
            continue
        common_genes = tables[0].index
        for table in tables[1:]:
            common_genes = common_genes.intersection(table.index)
        if len(common_genes) == 0:
            continue
        combined[key] = pd.concat(
            [table.loc[common_genes].copy() for table in tables],
            axis=1,
        )

    # Downstream uncertainty consumers (including calibrated limma weighting)
    # need the same in-memory calibration metadata after bundle concatenation.
    # These paired bundles are assembled from slices of a common calibrated run,
    # so we intentionally carry forward the shared calibration state and merge
    # the per-sample metadata maps across the contributing bundles.
    merged_source_by_sample: Dict[str, str] = {}
    merged_sample_fold: Dict[str, int] = {}
    merged_runtime_sample_fold: Dict[str, int] = {}
    for bundle in bundles:
        source_map = bundle.get("calibration_source_by_sample")
        if isinstance(source_map, dict):
            merged_source_by_sample.update({str(sample): str(source) for sample, source in source_map.items()})
        calibration_sample_fold = bundle.get("calibration_sample_fold")
        if isinstance(calibration_sample_fold, dict):
            merged_sample_fold.update({str(sample): int(fold_id) for sample, fold_id in calibration_sample_fold.items()})
        runtime_sample_fold = bundle.get("sample_fold")
        if isinstance(runtime_sample_fold, dict):
            merged_runtime_sample_fold.update({str(sample): int(fold_id) for sample, fold_id in runtime_sample_fold.items()})

    if merged_source_by_sample:
        combined["calibration_source_by_sample"] = merged_source_by_sample
    if merged_sample_fold:
        combined["calibration_sample_fold"] = merged_sample_fold
    if merged_runtime_sample_fold:
        combined["sample_fold"] = merged_runtime_sample_fold

    if any(bool(bundle.get("distribution_calibration_applied")) for bundle in bundles):
        combined["distribution_calibration_applied"] = True
        combined["distribution_calibration_label"] = next(
            (str(bundle.get("distribution_calibration_label")) for bundle in bundles if bundle.get("distribution_calibration_label") is not None),
            "concatenated_bundle",
        )
        calibration_state = next(
            (bundle.get("distribution_calibration_state") for bundle in bundles if bundle.get("distribution_calibration_state") is not None),
            None,
        )
        if calibration_state is not None:
            combined["distribution_calibration_state"] = calibration_state
    return combined


def _load_filtered_subtypes_for_reference(
    cohort: str,
    reference_samples: Sequence[Any],
    *,
    minimum_tfx: float,
    label_name: Optional[str] = None,
    report_context: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    reference_index = pd.Index([str(sample) for sample in reference_samples])
    if len(reference_index) == 0:
        return pd.DataFrame(columns=["tfx", "subtype"])

    if not _cohort_supports_tfx_subtypes(cohort, label_name=label_name):
        if minimum_tfx > 0.0:
            _note_tfx_filter_unavailable(report_context or cohort)
        return None

    resolved_label = label_name or cohort
    subtype_path = PROJECT_ROOT / "labels" / f"{resolved_label}_TFX_Subtype.tsv"
    _require_existing_path(
        subtype_path,
        description=f"TFx subtype table for {report_context or resolved_label}",
    )

    subtypes_df = load_subtypes(resolved_label)
    source_by_name: Dict[str, Any] = {}
    for source_sample in subtypes_df.index:
        source_str = str(source_sample)
        source_by_name[source_str] = source_sample
        source_by_name[str(strip_cohort_prefix(pd.Index([source_str]), cohort)[0])] = source_sample

    source_rows: List[Any] = []
    target_rows: List[str] = []
    for sample_name in reference_index:
        source_sample = source_by_name.get(str(sample_name))
        if source_sample is None:
            source_sample = source_by_name.get(f"{cohort}_{sample_name}")
        if source_sample is not None:
            source_rows.append(source_sample)
            target_rows.append(str(sample_name))

    if not source_rows:
        return subtypes_df.iloc[0:0].copy()

    aligned = subtypes_df.loc[source_rows].copy()
    aligned.index = pd.Index(target_rows)
    if minimum_tfx > 0.0:
        if "tfx" not in aligned.columns:
            raise ValueError(f"TFx subtype table for {resolved_label} is missing required 'tfx' column")
        tfx_values = pd.to_numeric(aligned["tfx"], errors="coerce")
        keep_mask = tfx_values >= float(minimum_tfx)
        dropped = int((~keep_mask).sum())
        if dropped:
            print(f"  {report_context or cohort}: dropped {dropped} samples below minimum TFx")
        aligned = aligned.loc[keep_mask].copy()
    return aligned


def _load_mane_gene_meta_features() -> pd.DataFrame:
    """Load MANE-derived gene meta-features for OOF error diagnostics."""
    path = PROJECT_ROOT / "reference" / "MANE.GRCh38.v1.3.UniversalReference.tsv"
    _require_existing_path(path, description="MANE gene metadata table")
    meta = pd.read_csv(path, sep="\t")
    required = {"symbol", "chr_start", "chr_end", "NumExons", "mRNA_size", "transcript_length"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"MANE reference is missing required columns: {sorted(missing)}")

    out = meta[["symbol", "chr_start", "chr_end", "NumExons", "mRNA_size", "transcript_length"]].copy()
    out = out.rename(
        columns={
            "symbol": "gene",
            "NumExons": "num_exons",
            "mRNA_size": "coding_length",
            "transcript_length": "transcript_length",
        }
    )
    out["gene_length"] = (
        pd.to_numeric(out["chr_end"], errors="coerce")
        - pd.to_numeric(out["chr_start"], errors="coerce")
        + 1.0
    ).abs()
    for column in ("num_exons", "coding_length", "transcript_length", "gene_length"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.drop_duplicates(subset="gene", keep="first").set_index("gene")
    return out[["num_exons", "gene_length", "coding_length", "transcript_length"]]


def _flatten_oof_diagnostics(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    ci_lo_wide: Optional[pd.DataFrame],
    ci_hi_wide: Optional[pd.DataFrame],
    activity_odds_ratio_wide: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build a gene-sample diagnostic table from aligned OOF prediction surfaces."""
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    if ci_lo_wide is not None:
        common_genes = common_genes.intersection(ci_lo_wide.index)
        common_samples = common_samples.intersection(ci_lo_wide.columns)
    if ci_hi_wide is not None:
        common_genes = common_genes.intersection(ci_hi_wide.index)
        common_samples = common_samples.intersection(ci_hi_wide.columns)
    if activity_odds_ratio_wide is not None:
        common_genes = common_genes.intersection(activity_odds_ratio_wide.index)
        common_samples = common_samples.intersection(activity_odds_ratio_wide.columns)
    common_samples = pd.Index([sample for sample in common_samples if not str(sample).startswith("PHT")])
    if len(common_genes) == 0 or len(common_samples) == 0:
        return pd.DataFrame()

    truth = truth_wide.loc[common_genes, common_samples].astype(float)
    pred = pred_wide.loc[common_genes, common_samples].astype(float)
    residual = pred - truth
    rows = pd.DataFrame(
        {
            "gene": np.repeat(common_genes.to_numpy(dtype=object), len(common_samples)),
            "sample": np.tile(common_samples.to_numpy(dtype=object), len(common_genes)),
            "truth": truth.to_numpy(dtype=np.float64).ravel(),
            "prediction_mean": pred.to_numpy(dtype=np.float64).ravel(),
            "signed_residual": residual.to_numpy(dtype=np.float64).ravel(),
            "absolute_residual": np.abs(residual.to_numpy(dtype=np.float64)).ravel(),
        }
    )
    if ci_lo_wide is not None and ci_hi_wide is not None:
        ci_width = (
            ci_hi_wide.loc[common_genes, common_samples].astype(float)
            - ci_lo_wide.loc[common_genes, common_samples].astype(float)
        )
        rows["ci_width"] = ci_width.to_numpy(dtype=np.float64).ravel()
    if activity_odds_ratio_wide is not None:
        activity_odds_ratio = activity_odds_ratio_wide.loc[
            common_genes, common_samples
        ].astype(float).to_numpy(dtype=np.float64).ravel()
        rows["activity_odds_ratio"] = activity_odds_ratio
        rows["absolute_activity_odds_ratio"] = np.abs(activity_odds_ratio)
    return rows.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["truth", "prediction_mean", "signed_residual", "absolute_residual"]
    )


def _truth_pred_correlation_values(
    truth_values: Sequence[float],
    pred_values: Sequence[float],
) -> Tuple[int, float, float, float, float]:
    """Return n, Pearson r/p, and Spearman rho/p for finite paired values."""
    from scipy.stats import pearsonr, spearmanr

    truth = np.asarray(truth_values, dtype=np.float64)
    pred = np.asarray(pred_values, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[valid]
    pred = pred[valid]
    n_pairs = int(len(truth))
    if n_pairs < MIN_CORRELATION_N:
        return n_pairs, np.nan, np.nan, np.nan, np.nan
    if float(np.ptp(truth)) == 0.0 or float(np.ptp(pred)) == 0.0:
        return n_pairs, 0.0, np.nan, 0.0, np.nan
    pearson_stat = pearsonr(truth, pred)
    spearman_stat = spearmanr(truth, pred)
    return (
        n_pairs,
        float(pearson_stat.statistic),
        float(pearson_stat.pvalue),
        float(spearman_stat.statistic),
        float(spearman_stat.pvalue),
    )


def _generic_correlation_values(
    x_values: Sequence[float],
    y_values: Sequence[float],
    *,
    method: str = "spearman",
) -> Tuple[int, float, float]:
    """Return finite-pair count, correlation statistic, and p-value."""
    from scipy.stats import pearsonr, spearmanr

    x_arr = np.asarray(x_values, dtype=np.float64)
    y_arr = np.asarray(y_values, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    n_pairs = int(len(x_arr))
    if n_pairs < MIN_CORRELATION_N:
        return n_pairs, np.nan, np.nan
    if float(np.ptp(x_arr)) == 0.0 or float(np.ptp(y_arr)) == 0.0:
        return n_pairs, 0.0, np.nan
    method_name = method.lower()
    if method_name == "spearman":
        result = spearmanr(x_arr, y_arr)
    elif method_name == "pearson":
        result = pearsonr(x_arr, y_arr)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return n_pairs, float(result.statistic), float(result.pvalue)


def _plot_diagnostic_scatter(
    x_values: pd.Series,
    y_values: pd.Series,
    name: str,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    log_x: bool = False,
) -> None:
    """Binned diagnostic summary for OOF residuals versus uncertainty/activity surfaces."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    _ensure_dir_for_plot = os.path.dirname(name)
    if _ensure_dir_for_plot:
        os.makedirs(_ensure_dir_for_plot, exist_ok=True)
    x = pd.to_numeric(x_values, errors="coerce")
    y = pd.to_numeric(y_values, errors="coerce")
    valid = x.notna() & y.notna()
    if log_x:
        valid = valid & (x > 0.0)
    if int(valid.sum()) < MIN_CORRELATION_N:
        return
    x = x.loc[valid]
    y = y.loc[valid]
    rho = np.nan
    p_value = np.nan
    if float(np.ptp(x.values)) > 0.0 and float(np.ptp(y.values)) > 0.0:
        corr = spearmanr(x.values, y.values)
        rho = float(corr.statistic)
        p_value = float(corr.pvalue)

    x_arr = x.to_numpy(dtype=np.float64)
    y_arr = y.to_numpy(dtype=np.float64)
    if log_x:
        bin_edges = np.geomspace(float(x_arr.min()), float(x_arr.max()), num=25)
    else:
        quantiles = np.linspace(0.0, 1.0, num=25, dtype=np.float64)
        bin_edges = np.quantile(x_arr, quantiles)
        bin_edges = np.unique(bin_edges)
        if bin_edges.size < 3:
            bin_edges = np.linspace(float(x_arr.min()), float(x_arr.max()), num=25)

    if bin_edges.size < 3:
        return

    centers: List[float] = []
    medians: List[float] = []
    lo_band: List[float] = []
    hi_band: List[float] = []
    counts: List[int] = []
    for left_edge, right_edge in zip(bin_edges[:-1], bin_edges[1:]):
        if log_x:
            in_bin = (x_arr >= float(left_edge)) & (x_arr <= float(right_edge))
        else:
            in_bin = (x_arr >= float(left_edge)) & (x_arr <= float(right_edge))
        if int(in_bin.sum()) < 25:
            continue
        y_bin = y_arr[in_bin]
        if not np.isfinite(y_bin).any():
            continue
        centers.append(float(np.median(x_arr[in_bin])))
        medians.append(float(np.median(y_bin)))
        lo_band.append(float(np.quantile(y_bin, 0.25)))
        hi_band.append(float(np.quantile(y_bin, 0.75)))
        counts.append(int(in_bin.sum()))

    if not centers:
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    if int(valid.sum()) > 5000:
        sample_idx = np.linspace(0, int(valid.sum()) - 1, num=5000, dtype=int)
        x_plot = x_arr[sample_idx]
        y_plot = y_arr[sample_idx]
    else:
        x_plot = x_arr
        y_plot = y_arr
    ax.scatter(
        x_plot,
        y_plot,
        s=6,
        color="#A9A9A9",
        alpha=0.10,
        edgecolors="none",
        rasterized=True,
    )
    ax.fill_between(centers, lo_band, hi_band, color="#4C78A8", alpha=0.24, linewidth=0.0)
    ax.plot(centers, medians, color="#1F4E79", linewidth=2.0)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(
        0.03,
        0.97,
        f"Spearman rho = {rho:.3f}\n{format_pval(p_value) if np.isfinite(p_value) else 'p=NA'}\nn = {int(valid.sum()):,}\nBins = {len(centers)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostic_distribution(
    values: pd.Series,
    name: str,
    *,
    xlabel: str,
    title: str,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
) -> None:
    """Histogram diagnostic with optional AOR threshold markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir_for_plot = os.path.dirname(name)
    if _ensure_dir_for_plot:
        os.makedirs(_ensure_dir_for_plot, exist_ok=True)
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.hist(vals, bins=80, color="#4C78A8", alpha=0.72, edgecolor="white", linewidth=0.2)
    if lower_threshold is not None and np.isfinite(lower_threshold):
        ax.axvline(float(lower_threshold), color="#CC3311", linestyle="--", linewidth=1.4, label="Inactive threshold")
    if upper_threshold is not None and np.isfinite(upper_threshold):
        ax.axvline(float(upper_threshold), color="#228833", linestyle="--", linewidth=1.4, label="Active threshold")
    if lower_threshold is not None or upper_threshold is not None:
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Gene-sample pairs")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_tfx_vs_gene_rmse(
    sample_df: pd.DataFrame,
    name: str,
    *,
    title: str,
) -> None:
    """Scatter patient TFx against per-sample RMSE across genes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.dirname(name)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    plot_df = sample_df[["tfx", "gene_rmse"]].copy()
    plot_df["tfx"] = pd.to_numeric(plot_df["tfx"], errors="coerce")
    plot_df["gene_rmse"] = pd.to_numeric(plot_df["gene_rmse"], errors="coerce")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(plot_df) < 2:
        return

    rho = np.nan
    p_value = np.nan
    if len(plot_df) >= MIN_CORRELATION_N and float(np.ptp(plot_df["tfx"].values)) > 0.0 and float(np.ptp(plot_df["gene_rmse"].values)) > 0.0:
        _, rho, p_value = _generic_correlation_values(
            plot_df["tfx"].values,
            plot_df["gene_rmse"].values,
            method="spearman",
        )

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.scatter(
        plot_df["tfx"].values,
        plot_df["gene_rmse"].values,
        s=42,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
        zorder=3,
    )
    if len(plot_df) >= 2 and float(np.ptp(plot_df["tfx"].values)) > 0.0:
        slope, intercept = np.polyfit(plot_df["tfx"].values, plot_df["gene_rmse"].values, 1)
        x_line = np.linspace(float(plot_df["tfx"].min()), float(plot_df["tfx"].max()), num=100)
        ax.plot(x_line, slope * x_line + intercept, color="#1F4E79", linewidth=1.6, zorder=2)
    ax.set_xlabel("Tumor fraction (TFx)")
    ax.set_ylabel("Sample-level gene RMSE")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(color="#E5E5E5", linewidth=0.8)
    annotation = [f"n = {len(plot_df)}"]
    if np.isfinite(rho):
        annotation.append(f"Spearman rho = {rho:.3f}")
    if np.isfinite(p_value):
        annotation.append(format_pval(p_value))
    ax.text(
        0.03,
        0.97,
        "\n".join(annotation),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _save_pit_diagnostics(
    truth_wide: pd.DataFrame,
    mdn_tables: Dict[str, Optional[pd.DataFrame]],
    out_dir: str,
    *,
    overwrite: bool,
    title: str,
) -> None:
    """Write MDN PIT values, summary statistics, and histogram diagnostics."""
    if not all(mdn_tables.get(key) is not None for key in _MDN_PARAM_KEYS):
        print("  [skip] PIT diagnostics: MDN parameter surfaces unavailable")
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
    pit_values = np.asarray(pit_values, dtype=np.float64)
    pit_values = pit_values[np.isfinite(pit_values)]
    pit_values = pit_values[(pit_values >= 0.0) & (pit_values <= 1.0)]
    if pit_values.size == 0:
        print("  [skip] PIT diagnostics: no finite PIT values")
        return

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"pit": pit_values}).to_csv(
        os.path.join(out_dir, "pit_values.tsv"),
        sep="\t",
        index=False,
    )
    summary = pd.DataFrame(
        [
            {
                "n": int(pit_values.size),
                "mean": float(np.mean(pit_values)),
                "variance": float(np.var(pit_values)),
                "q05": float(np.quantile(pit_values, 0.05)),
                "q50": float(np.quantile(pit_values, 0.50)),
                "q95": float(np.quantile(pit_values, 0.95)),
            }
        ]
    )
    summary.to_csv(os.path.join(out_dir, "pit_summary.tsv"), sep="\t", index=False)
    _maybe(
        plot_pit_histogram,
        overwrite,
        pit_values,
        os.path.join(out_dir, "pit_histogram"),
        title=title,
    )


def _write_oof_validation_diagnostics(
    cv_full: Dict[str, Any],
    cohort_map: Dict[str, str],
    out_dir: str,
    *,
    overwrite: bool,
    activity_threshold_summary: Optional[Dict[str, Any]],
) -> None:
    """Write OOF residual, AOR, CI-width, PIT, and gene-meta diagnostics."""
    truth_wide = cv_full.get("truth_wide")
    pred_wide = cv_full.get("pred_wide")
    if truth_wide is None or pred_wide is None:
        print("  [skip] OOF diagnostics: truth/prediction surfaces unavailable")
        return
    ci_lo_wide = cv_full.get("ci_lo_wide")
    ci_hi_wide = cv_full.get("ci_hi_wide")
    activity_odds_ratio_wide = cv_full.get("activity_odds_ratio_wide")
    lower_threshold = (
        float(activity_threshold_summary["lower_threshold"])
        if activity_threshold_summary is not None else None
    )
    upper_threshold = (
        float(activity_threshold_summary["upper_threshold"])
        if activity_threshold_summary is not None else None
    )
    gene_meta = _load_mane_gene_meta_features()

    diagnostic_specs: List[Tuple[str, str, Tuple[str, ...]]] = [
        ("OOF_CRPC-RA", "CRPC-RA", ("CRPC-RA",)),
        ("OOF_SCLC", "SCLC", ("SCLC",)),
        ("OOF_Patient", "Patient", ("CRPC-RA", "SCLC")),
    ]
    summary_rows: List[Dict[str, Any]] = []
    for label, cohort_label, cohorts in diagnostic_specs:
        samples = [sample for sample in pred_wide.columns if cohort_map.get(str(sample)) in cohorts]
        samples = [sample for sample in samples if sample in truth_wide.columns]
        if len(samples) < 2:
            continue

        diag_dir = ensure_dir(os.path.join(out_dir, label))
        pair_df = _flatten_oof_diagnostics(
            truth_wide[samples],
            pred_wide[samples],
            ci_lo_wide[samples] if ci_lo_wide is not None else None,
            ci_hi_wide[samples] if ci_hi_wide is not None else None,
            activity_odds_ratio_wide[samples] if activity_odds_ratio_wide is not None else None,
        )
        if pair_df.empty:
            continue

        pair_summary = {
            "label": label,
            "cohort": cohort_label,
            "n_pairs": int(len(pair_df)),
            "n_genes": int(pair_df["gene"].nunique()),
            "n_samples": int(pair_df["sample"].nunique()),
            "mean_signed_residual": float(pair_df["signed_residual"].mean()),
            "median_signed_residual": float(pair_df["signed_residual"].median()),
            "mean_absolute_residual": float(pair_df["absolute_residual"].mean()),
            "median_absolute_residual": float(pair_df["absolute_residual"].median()),
            "rmse": float(np.sqrt(np.mean(np.square(pair_df["signed_residual"].values)))),
        }
        _maybe(
            _plot_diagnostic_distribution,
            overwrite,
            pair_df["signed_residual"],
            os.path.join(diag_dir, "signed_residual_distribution"),
            xlabel="Signed residual (prediction - truth)",
            title=f"{label}: signed residual distribution",
        )
        if "activity_odds_ratio" in pair_df.columns:
            _maybe(
                _plot_diagnostic_scatter,
                overwrite,
                pair_df["activity_odds_ratio"],
                pair_df["absolute_residual"],
                os.path.join(diag_dir, "aor_vs_absolute_residual"),
                xlabel="Activity odds ratio",
                ylabel="Absolute residual |prediction - truth|",
                title=f"{label}: AOR vs absolute residual",
            )
            _maybe(
                _plot_diagnostic_scatter,
                overwrite,
                pair_df["absolute_activity_odds_ratio"],
                pair_df["absolute_residual"],
                os.path.join(diag_dir, "absolute_aor_vs_absolute_residual"),
                xlabel="Absolute activity odds ratio",
                ylabel="Absolute residual |prediction - truth|",
                title=f"{label}: absolute AOR vs absolute residual",
            )
            _maybe(
                _plot_diagnostic_distribution,
                overwrite,
                pair_df["activity_odds_ratio"],
                os.path.join(diag_dir, "aor_distribution"),
                xlabel="Activity odds ratio",
                title=f"{label}: AOR distribution",
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
        if "ci_width" in pair_df.columns:
            _maybe(
                _plot_diagnostic_scatter,
                overwrite,
                pair_df["ci_width"],
                pair_df["absolute_residual"],
                os.path.join(diag_dir, "ci_width_vs_absolute_residual"),
                xlabel="90% CI width",
                ylabel="Absolute residual |prediction - truth|",
                title=f"{label}: CI width vs absolute residual",
                log_x=True,
            )
            _maybe(
                _plot_diagnostic_distribution,
                overwrite,
                pair_df["ci_width"],
                os.path.join(diag_dir, "ci_width_distribution"),
                xlabel="90% CI width",
                title=f"{label}: CI width distribution",
            )

        gene_df = pair_df.groupby("gene", sort=False).agg(
            n_pairs=("signed_residual", "size"),
            signed_residual_mean=("signed_residual", "mean"),
            rmse=("signed_residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        )
        gene_df = gene_df.join(gene_meta, how="left")
        gene_df.to_csv(os.path.join(diag_dir, "gene_rmse_meta_features.tsv"), sep="\t")
        for feature_name, x_label, log_x in [
            ("num_exons", "Number of exons (log scale)", True),
            ("gene_length", "Gene length (bp, log scale)", True),
            ("coding_length", "MANE mRNA size (log scale)", True),
        ]:
            if feature_name in gene_df.columns:
                _maybe(
                    _plot_diagnostic_scatter,
                    overwrite,
                    gene_df[feature_name],
                    gene_df["rmse"],
                    os.path.join(diag_dir, f"{feature_name}_vs_gene_rmse"),
                    xlabel=x_label,
                    ylabel="Gene-level RMSE",
                    title=f"{label}: {x_label} vs gene RMSE",
                    log_x=log_x,
                )

        sample_tfx: Dict[str, float] = {}
        for cohort_name in cohorts:
            cohort_samples = [sample_name for sample_name in samples if cohort_map.get(str(sample_name)) == cohort_name]
            if not cohort_samples:
                continue
            cohort_subtypes = _load_filtered_subtypes_for_reference(
                cohort_name,
                cohort_samples,
                minimum_tfx=0.0,
                report_context=f"Calibration {cohort_name}",
            )
            if cohort_subtypes is None or "tfx" not in cohort_subtypes.columns:
                continue
            tfx_series = pd.to_numeric(cohort_subtypes["tfx"], errors="coerce")
            for sample_name, tfx_value in tfx_series.items():
                if np.isfinite(tfx_value):
                    sample_tfx[str(sample_name)] = float(tfx_value)

        sample_df = pair_df.groupby("sample", sort=False).agg(
            n_genes=("signed_residual", "size"),
            gene_rmse=("signed_residual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        )
        sample_df["tfx"] = sample_df.index.to_series().map(sample_tfx)
        _maybe(
            _plot_tfx_vs_gene_rmse,
            overwrite,
            sample_df,
            os.path.join(diag_dir, "tfx_vs_gene_rmse"),
            title=f"{label}: TFx vs sample-level gene RMSE",
        )
        summary_rows.append(pair_summary)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "oof_diagnostic_summary.tsv"), sep="\t", index=False)

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
                out_dir,
                overwrite=overwrite,
                title="MDN PIT — OOF",
            )


def _compute_activity_odds_ratio_summary_wide(
    truth_wide: pd.DataFrame,
    activity_odds_ratio_wide: Optional[pd.DataFrame],
    *,
    target_ppv: float = _ACTIVE_CALL_TARGET_PPV,
    target_npv: float = _INACTIVE_CALL_TARGET_NPV,
    threshold: float = 2.0,
    require_gene_crosses_threshold: bool = True,
) -> Optional[Dict[str, Any]]:
    """Compute pooled lower/upper AOR threshold summaries on overlapping truth pairs."""
    if activity_odds_ratio_wide is None or truth_wide.empty:
        return None

    flattened = _flatten_activity_odds_ratio_pairs(
        truth_wide,
        activity_odds_ratio_wide,
        threshold=threshold,
        require_gene_crosses_threshold=require_gene_crosses_threshold,
    )
    if flattened is None:
        return None

    on_curve = compute_activity_odds_ratio_on_curve(
        flattened["truth_flat"],
        flattened["activity_odds_ratio_flat"],
        threshold=threshold,
        target_ppv=target_ppv,
    )
    off_curve = compute_activity_odds_ratio_off_curve(
        flattened["truth_flat"],
        flattened["activity_odds_ratio_flat"],
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
        "n_threshold_genes": int(flattened["n_threshold_genes"]),
        "n_threshold_samples": int(flattened["n_threshold_samples"]),
    }


def _flatten_activity_odds_ratio_pairs(
    truth_wide: pd.DataFrame,
    activity_odds_ratio_wide: Optional[pd.DataFrame],
    *,
    threshold: float = 2.0,
    require_gene_crosses_threshold: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return finite pooled truth/AOR pairs after the shared gene-selection filter."""
    if activity_odds_ratio_wide is None or truth_wide.empty:
        return None

    common_genes = truth_wide.index.intersection(activity_odds_ratio_wide.index)
    common_samples = truth_wide.columns.intersection(activity_odds_ratio_wide.columns)
    if len(common_genes) == 0 or len(common_samples) == 0:
        return None

    if require_gene_crosses_threshold:
        truth_common = truth_wide.loc[common_genes, common_samples].astype(float)
        crosses = (truth_common.min(axis=1) < threshold) & (truth_common.max(axis=1) >= threshold)
        common_genes = truth_common.index[crosses.fillna(False)]
        if len(common_genes) == 0:
            return None

    truth_flat = truth_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel()
    aor_flat = activity_odds_ratio_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64).ravel()
    valid = np.isfinite(truth_flat) & np.isfinite(aor_flat)
    if int(valid.sum()) < 10:
        return None
    return {
        "truth_flat": truth_flat[valid],
        "activity_odds_ratio_flat": aor_flat[valid],
        "n_threshold_genes": int(len(common_genes)),
        "n_threshold_samples": int(len(common_samples)),
    }


def _model_activity_odds_ratio_block(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    mdn_tables: Dict[str, Optional[pd.DataFrame]],
    candidate_genes: Sequence[str],
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Return aligned truth and AOR surfaces for one benchmarking model block."""
    required_keys = (
        "p0_wide",
        "pi_gamma_wide",
        "pi_normal_wide",
    )
    if not all(mdn_tables.get(key) is not None for key in required_keys):
        return None

    common_genes = [gene_name for gene_name in candidate_genes if gene_name in pred_wide.index]
    common_samples = [sample_name for sample_name in pred_wide.columns if sample_name in truth_wide.columns]
    if len(common_genes) < 5 or len(common_samples) < 3:
        return None

    mdn_genes = [
        gene_name
        for gene_name in common_genes
        if all(gene_name in mdn_tables[key].index for key in required_keys)
    ]
    mdn_samples = [
        sample_name
        for sample_name in common_samples
        if all(sample_name in mdn_tables[key].columns for key in required_keys)
    ]
    if len(mdn_genes) < 5 or len(mdn_samples) < 3:
        return None

    truth_active = truth_wide.loc[mdn_genes, mdn_samples]
    activity_odds_ratio = mdn_activity_odds_ratio_wide(
        mdn_tables["p0_wide"].loc[mdn_genes, mdn_samples],
        mdn_tables["pi_gamma_wide"].loc[mdn_genes, mdn_samples],
        mdn_tables["pi_normal_wide"].loc[mdn_genes, mdn_samples],
    )
    return truth_active, activity_odds_ratio


def _write_pooled_benchmark_threshold_artifacts(
    pooled_pairs: Dict[str, Dict[str, List[np.ndarray]]],
    out_dir: str,
    overwrite: bool,
    model_order: Sequence[str],
) -> None:
    """Write one pooled active and inactive AOR threshold sweep for Benchmarking_Patient."""
    threshold_curves: Dict[str, Dict[str, Any]] = {}
    inactive_threshold_curves: Dict[str, Dict[str, Any]] = {}
    for model_name in model_order:
        pair_blocks = pooled_pairs.get(model_name)
        if pair_blocks is None:
            continue
        truth_blocks = pair_blocks.get("truth_flat") or []
        aor_blocks = pair_blocks.get("activity_odds_ratio_flat") or []
        if not truth_blocks or not aor_blocks:
            continue

        truth_flat = np.concatenate(truth_blocks)
        aor_flat = np.concatenate(aor_blocks)
        if truth_flat.size < 10 or aor_flat.size < 10:
            continue

        threshold_curves[model_name] = compute_activity_odds_ratio_on_curve(
            truth_flat,
            aor_flat,
            threshold=_ONOFF_THRESHOLD,
            target_ppv=_ACTIVE_CALL_TARGET_PPV,
        )
        inactive_threshold_curves[model_name] = compute_activity_odds_ratio_off_curve(
            truth_flat,
            aor_flat,
            threshold=_ONOFF_THRESHOLD,
            target_npv=_INACTIVE_CALL_TARGET_NPV,
        )

    if threshold_curves:
        curve_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        for model_name in model_order:
            curve = threshold_curves.get(model_name)
            pair_blocks = pooled_pairs.get(model_name)
            if curve is None or pair_blocks is None:
                continue
            n_pairs = int(sum(len(block) for block in pair_blocks.get("truth_flat", [])))
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
                        "n_pairs": n_pairs,
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
                    "n_pairs": n_pairs,
                }
            )

        pd.DataFrame(curve_rows).to_csv(
            os.path.join(out_dir, "ActiveConfidenceCurve.tsv"),
            sep="\t",
            index=False,
        )
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "ActiveConfidenceThreshold.tsv"),
            sep="\t",
            index=False,
        )
        _maybe(
            active_confidence_threshold_plot,
            overwrite,
            threshold_curves,
            os.path.join(out_dir, "ActiveConfidenceThreshold"),
            title="Benchmarking Patient (SUBSET) — Active Threshold (AOR)",
            model_order=list(model_order),
        )

    if inactive_threshold_curves:
        curve_rows = []
        summary_rows = []
        for model_name in model_order:
            curve = inactive_threshold_curves.get(model_name)
            pair_blocks = pooled_pairs.get(model_name)
            if curve is None or pair_blocks is None:
                continue
            n_pairs = int(sum(len(block) for block in pair_blocks.get("truth_flat", [])))
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
                        "n_pairs": n_pairs,
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
                    "n_pairs": n_pairs,
                }
            )

        pd.DataFrame(curve_rows).to_csv(
            os.path.join(out_dir, "InactiveConfidenceCurve.tsv"),
            sep="\t",
            index=False,
        )
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "InactiveConfidenceThreshold.tsv"),
            sep="\t",
            index=False,
        )
        _maybe(
            inactive_confidence_threshold_plot,
            overwrite,
            inactive_threshold_curves,
            os.path.join(out_dir, "InactiveConfidenceThreshold"),
            title="Benchmarking Patient (SUBSET) — Inactive Threshold (AOR)",
            model_order=list(model_order),
        )


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
        ho_path = _require_existing_path(
            PROJECT_ROOT / rel_path,
            description=f"Validation_PDX holdout predictions for {cohort_name}",
        )

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


def _prepare_proteus_analysis_bundle(full_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return the raw prediction bundle for downstream analysis consumers."""
    return dict(full_data)


def _require_relative_input_path(rel_path: Optional[str], *, description: str) -> Path:
    """Resolve a required repo-relative input path and fail loudly if missing."""
    if rel_path is None:
        raise ValueError(f"Missing required configured path for {description}")
    return _require_existing_path(PROJECT_ROOT / rel_path, description=description)

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
    tfx_dict = None
    if subtypes_df is not None and "tfx" in subtypes_df.columns:
        tfx_dict = subtypes_df["tfx"].to_dict()

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
                tfx_dict=tfx_dict,
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
    model_path: str,
    out_dir: str,
    args: argparse.Namespace,
    *,
    activity_threshold_summary: Optional[Dict[str, Any]] = None,
    calibration_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run Patient_Validation analysis for a single cohort.

    Loads holdout predictions from ``model_path``, runs the full
    per-cohort suite (metrics + plots), and returns data for Collected.

    Returns a dict with cohort-level data for the Collected analysis,
    or None if the cohort is skipped.
    """
    truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
    _require_existing_path(truth_path, description=f"Validation_Patient truth for {cohort}")
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

    path = _require_relative_input_path(
        model_path,
        description=f"Validation_Patient predictions for {cohort}",
    )

    full_data = load_holdout_predictions_proteus_full(str(path))
    for key in ("pred_wide", "ci_lo_wide", "ci_hi_wide", "activity_odds_ratio_wide", "active_confidence_wide", *_MDN_PARAM_KEYS):
        _apply_prefix_map(full_data.get(key), pmap)
    if args.minimum_tfx > 0.0 and subtypes_df is not None:
        full_data = _filter_prediction_bundle_to_samples(full_data, subtypes_df.index)
    analysis_data = _prepare_proteus_analysis_bundle(full_data)
    if args.apply_calibration:
        if calibration_state is None:
            raise ValueError("Validation_Patient calibration was requested but no calibration state was available")
        analysis_data = _apply_distribution_calibration_bundle(
            analysis_data,
            calibration_state,
            source_by_sample=_constant_source_by_sample(analysis_data["pred_wide"].columns, cohort),
            sample_fold=None,
            label=f"Validation_Patient {cohort}",
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
        "predictive_variance_wide": analysis_data.get("predictive_variance_wide"),
        "calibration_source_by_sample": analysis_data.get("calibration_source_by_sample"),
        "calibration_sample_fold": analysis_data.get("calibration_sample_fold"),
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
    calibration_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run patient-validation analysis for one cohort using OOF CV predictions."""
    truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
    _require_existing_path(truth_path, description=f"Validation_Patient truth for {cohort}")

    truth_wide = load_truth(cohort)
    subtypes_df = _load_filtered_subtypes_for_reference(
        cohort,
        truth_wide.columns,
        minimum_tfx=args.minimum_tfx,
        report_context=f"Validation_Patient {cohort}",
    )
    if args.minimum_tfx > 0.0 and subtypes_df is not None:
        truth_wide = _filter_wide_to_samples(truth_wide, subtypes_df.index)
    cohort_samples_all = [
        s for s in cv_pred.columns
        if cohort_map_cv.get(s) == cohort
    ]
    if not cohort_samples_all:
        print(f"    [skip] No CV samples for {cohort}")
        return None

    cohort_subtypes_df = _load_filtered_subtypes_for_reference(
        cohort,
        cohort_samples_all,
        minimum_tfx=args.minimum_tfx,
        report_context=f"Validation_Patient {cohort}",
    )
    cohort_samples = list(cohort_samples_all)
    if args.minimum_tfx > 0.0 and cohort_subtypes_df is not None:
        allowed_full_samples = set(cohort_subtypes_df.index)
        cohort_samples = [sample_name for sample_name in cohort_samples_all if sample_name in allowed_full_samples]
    if not cohort_samples:
        print(f"    [skip] No CV samples remain for {cohort} after minimum TFx filtering")
        return None

    prefix_reference_samples = pd.Index(truth_wide.columns).union(pd.Index(cohort_samples))
    if cohort_subtypes_df is not None:
        prefix_reference_samples = prefix_reference_samples.union(pd.Index(cohort_subtypes_df.index.astype(str)))
    pmap = _build_prefix_map(prefix_reference_samples, cohort)

    pred_all_c = cv_pred[cohort_samples].copy()
    _apply_prefix_map(pred_all_c, pmap)
    ci_lo_all_c = (
        cv_ci_lo[[s for s in cohort_samples if s in cv_ci_lo.columns]].copy()
        if cv_ci_lo is not None else None
    )
    ci_hi_all_c = (
        cv_ci_hi[[s for s in cohort_samples if s in cv_ci_hi.columns]].copy()
        if cv_ci_hi is not None else None
    )
    _apply_prefix_map(ci_lo_all_c, pmap)
    _apply_prefix_map(ci_hi_all_c, pmap)

    full_mdn_tables: Dict[str, Optional[pd.DataFrame]] = {}
    for key in _MDN_PARAM_KEYS:
        table = cv_full.get(key)
        if table is None:
            full_mdn_tables[key] = None
            continue
        table_c = table[[s for s in cohort_samples if s in table.columns]].copy()
        _apply_prefix_map(table_c, pmap)
        full_mdn_tables[key] = table_c

    full_activity_odds_ratio_wide = None
    activity_odds_ratio_table = cv_full.get("activity_odds_ratio_wide")
    if activity_odds_ratio_table is not None:
        activity_odds_ratio_c = activity_odds_ratio_table[[s for s in cohort_samples if s in activity_odds_ratio_table.columns]].copy()
        _apply_prefix_map(activity_odds_ratio_c, pmap)
        full_activity_odds_ratio_wide = activity_odds_ratio_c
    elif full_mdn_tables.get("p0_wide") is not None and full_mdn_tables.get("pi_gamma_wide") is not None and full_mdn_tables.get("pi_normal_wide") is not None:
        full_activity_odds_ratio_wide = mdn_activity_odds_ratio_wide(
            full_mdn_tables["p0_wide"],
            full_mdn_tables["pi_gamma_wide"],
            full_mdn_tables["pi_normal_wide"],
        )

    full_active_confidence_wide = None
    if all(full_mdn_tables.get(key) is not None for key in _MDN_PARAM_KEYS):
        full_active_confidence_wide = mdn_on_probability_wide(
            full_mdn_tables["p0_wide"],
            full_mdn_tables["pi_gamma_wide"],
            full_mdn_tables["pi_normal_wide"],
            full_mdn_tables["gamma_beta_wide"],
            full_mdn_tables["normal_mean_wide"],
            full_mdn_tables["normal_stdev_wide"],
        )

    full_analysis_bundle: Dict[str, Any] = {
        "pred_wide": pred_all_c,
        "ci_lo_wide": ci_lo_all_c,
        "ci_hi_wide": ci_hi_all_c,
        "activity_odds_ratio_wide": full_activity_odds_ratio_wide,
        "active_confidence_wide": full_active_confidence_wide,
        **full_mdn_tables,
    }
    if args.apply_calibration:
        if calibration_state is None:
            raise ValueError("Validation_Patient CV calibration was requested but no calibration state was available")
        full_analysis_bundle = _apply_distribution_calibration_bundle(
            full_analysis_bundle,
            calibration_state,
            source_by_sample=_constant_source_by_sample(pred_all_c.columns, cohort),
            sample_fold=_bundle_sample_fold_subset(cv_full, pred_all_c.columns),
            label=f"Validation_Patient {cohort} CV",
        )

    common_samples = truth_wide.columns.intersection(full_analysis_bundle["pred_wide"].columns)
    if len(common_samples) == 0:
        print(f"    [skip] No overlapping truth/pred samples for {cohort}")
        return None

    truth_c = truth_wide[common_samples]
    analysis_bundle = _filter_prediction_bundle_to_samples(full_analysis_bundle, common_samples)
    pred_c = analysis_bundle["pred_wide"]
    ci_lo_c = analysis_bundle.get("ci_lo_wide")
    ci_hi_c = analysis_bundle.get("ci_hi_wide")
    mdn_tables = {key: analysis_bundle.get(key) for key in _MDN_PARAM_KEYS}
    activity_odds_ratio_wide = analysis_bundle.get("activity_odds_ratio_wide")
    active_confidence_wide = analysis_bundle.get("active_confidence_wide")
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
        crpc_ra_score_bundle = _compute_crpc_ra_shortlist_score_bundle(truth_c, pred_c)
        _run_crpc_ra_shortlist_score_scatters(
            truth_c,
            pred_c,
            cohort_out,
            subtypes_df=subtypes_df,
            score_bundle=crpc_ra_score_bundle,
            overwrite=args.overwrite,
        )
        _run_crpc_ra_pure_phenotype_heatmap(
            truth_c,
            pred_c,
            cohort_out,
            subtypes_df=subtypes_df,
            overwrite=args.overwrite,
        )
        _run_crpc_ra_pluvicto_vs_untreated_analysis(
            truth_c,
            pred_c,
            cohort_out,
            death_bundle=full_analysis_bundle,
            subtypes_df=subtypes_df,
            death_subtypes_df=cohort_subtypes_df,
            minimum_tfx=args.minimum_tfx,
            apply_calibration=args.apply_calibration,
            calibration_state=calibration_state,
            score_bundle=crpc_ra_score_bundle,
            overwrite=args.overwrite,
        )
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
        "predictive_variance_wide": analysis_bundle.get("predictive_variance_wide"),
        "calibration_source_by_sample": analysis_bundle.get("calibration_source_by_sample"),
        "calibration_sample_fold": analysis_bundle.get("calibration_sample_fold"),
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
_CRPC_RA_METS_EXCLUDED_SAMPLES: Tuple[str, ...] = (
    "18-059H1_LIVER",
    "19-045M7_R_PELVIC_LN",
    "24-016F1_SPLEEN",
)
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
_CRPC_RA_PLUVICTO_KEY_PATH: Path = PROJECT_ROOT / "reference" / "CRPC-RA-Pluvicto_key.tsv"
_CRPC_RA_PLUVICTO_TREATMENT_GENES: Tuple[str, ...] = (
    "ZNF750",
    "ADH1B",
    "KCNH6",
    "PCDHB5",
    "TOX3",
    "ELF5",
    "KCNQ2",
    "GRHL2",
    "EDN2",
    "CACNA2D1",
    "SAMD11",
    "SSTR2",
    "FSCN2",
    "LRRC31",
    "KCNG1",
    "KRT7",
    "HUNK",
    "MAP1B",
    "SMOC1",
    "RANBP3L",
)
_CRPC_RA_PLUVICTO_PAIRED_FEATURES: Tuple[str, ...] = ("FOLH1", "ARG.10", "NE.10", "CCP.31")
_CRPC_RA_PAIRED_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h")


def _load_crpc_ra_pluvicto_key() -> pd.DataFrame:
    """Load the explicit CRPC-RA death-sample to Pluvicto C1 patient map.

    This overlap is inherently cross-cohort: the CRPC-RA side contributes the
    TAN/death sample, while the CRPC-Pluvicto side contributes the baseline C1
    sample. The key file is therefore the single source of truth for matching
    those cohorts.
    """
    _require_existing_path(
        _CRPC_RA_PLUVICTO_KEY_PATH,
        description="CRPC-RA / Pluvicto overlap key",
    )
    key_df = pd.read_csv(_CRPC_RA_PLUVICTO_KEY_PATH, sep="\t").fillna("")
    pluvicto_column = None
    if "Pluvicto_ID" in key_df.columns:
        pluvicto_column = "Pluvicto_ID"
    elif "Pluvicto_ID_2" in key_df.columns:
        pluvicto_column = "Pluvicto_ID_2"

    required_columns = {"TAN_ID"}
    missing_columns = sorted(required_columns.difference(key_df.columns))
    if pluvicto_column is None:
        missing_columns.append("Pluvicto_ID|Pluvicto_ID_2")
    if missing_columns:
        raise ValueError(
            "CRPC-RA / Pluvicto overlap key is missing required columns: "
            + ", ".join(missing_columns)
        )

    key_df["TAN_ID"] = key_df["TAN_ID"].astype(str).str.strip()
    key_df[pluvicto_column] = key_df[pluvicto_column].astype(str).str.strip()
    key_df["Pluvicto_ID"] = key_df[pluvicto_column]
    key_df = key_df.loc[
        (key_df["TAN_ID"] != "") & (key_df["Pluvicto_ID"] != "")
    ].copy()
    if key_df.empty:
        raise ValueError("CRPC-RA / Pluvicto overlap key does not contain any usable TAN/Pluvicto pairs")

    duplicated_tan = key_df.loc[key_df["TAN_ID"].duplicated(), "TAN_ID"].astype(str).tolist()
    if duplicated_tan:
        raise ValueError(
            "CRPC-RA / Pluvicto overlap key has duplicate TAN_ID entries: "
            + ", ".join(duplicated_tan)
        )

    duplicated_pluvicto = key_df.loc[
        key_df["Pluvicto_ID"].duplicated(),
        "Pluvicto_ID",
    ].astype(str).tolist()
    if duplicated_pluvicto:
        raise ValueError(
            "CRPC-RA / Pluvicto overlap key has duplicate Pluvicto_ID entries: "
            + ", ".join(duplicated_pluvicto)
        )
    return key_df


def _crpc_ra_pluvicto_ctdna_overlap_key() -> pd.DataFrame:
    """Return overlap rows with ctDNA/WGS available at the CRPC-RA TAN/death timepoint."""
    key_df = _load_crpc_ra_pluvicto_key().copy()
    if "cfDNA_WGS_TAN" not in key_df.columns:
        raise ValueError("CRPC-RA / Pluvicto overlap key is missing required column: cfDNA_WGS_TAN")
    ctdna_mask = key_df["cfDNA_WGS_TAN"].astype(str).str.strip().str.upper().eq("TRUE")
    filtered = key_df.loc[ctdna_mask].copy()
    if filtered.empty:
        raise ValueError("CRPC-RA / Pluvicto overlap key does not contain any ctDNA/WGS-overlap rows")
    return filtered


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

    if int(confident.sum()) >= MIN_CORRELATION_N:
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

    if len(fit_x) >= MIN_CORRELATION_N and float(np.ptp(fit_x)) > 0.0:
        coeffs = np.polyfit(fit_x, fit_y, 1)
        x_line = np.linspace(float(fit_x.min()), float(fit_x.max()), 100)
        fit_fn = np.poly1d(coeffs)
        ax.plot(x_line, np.polyval(coeffs, x_line), "k-", lw=1, zorder=1)

        n = len(fit_x)
        if n >= MIN_CORRELATION_N:
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
    ihc_path = _require_relative_input_path(
        ihc_rel,
        description=f"IHC scores for {cohort}",
    )

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

        if len(ihc_vals) < MIN_CORRELATION_N:
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

        if int(confident.sum()) >= MIN_CORRELATION_N:
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
    _require_existing_path(
        _CRPC_RA_METS_METADATA_PATH,
        description="CRPC-RA metastasis sampling metadata",
    )

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
    _require_existing_path(
        _CRPC_RA_METS_EXPRESSION_PATH,
        description="CRPC-RA metastasis truth matrix",
    )
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
    if len(x) < MIN_CORRELATION_N:
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
    _, pearson_r, _, spearman_rho, _ = _truth_pred_correlation_values(
        truth_series.values,
        pred_series.values,
    )

    return {
        f"{prefix}n_genes": int(len(summary_df)),
        f"{prefix}mae": float(abs_error.mean()),
        f"{prefix}median_absolute_error": float(abs_error.median()),
        f"{prefix}rmse": float(np.sqrt(squared_error.mean())),
        f"{prefix}gvw_mae": _weighted_series_mean(abs_error, gene_weights),
        f"{prefix}gvw_rmse": float(np.sqrt(_weighted_series_mean(squared_error, gene_weights))),
        f"{prefix}pearson_r": pearson_r,
        f"{prefix}spearman_rho": spearman_rho,
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
        n_corr, rho_stat, rho_p = _generic_correlation_values(
            local_df[x_col].values,
            local_df[y_col].values,
            method="spearman",
        )
        if n_corr >= MIN_CORRELATION_N and np.isfinite(rho_stat):
            coeffs = np.polyfit(local_df[x_col].values.astype(np.float64), local_df[y_col].values.astype(np.float64), 1)
            x_line = np.linspace(float(local_df[x_col].min()), float(local_df[x_col].max()), 100)
            axis.plot(x_line, np.polyval(coeffs, x_line), color="#444444", lw=1.2, ls="--")
            axis.text(
                0.03,
                0.97,
                f"Spearman ρ = {rho_stat:.3f}\n{format_pval(rho_p)}",
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
        _, rho_stat, rho_p = _generic_correlation_values(
            local_df["truth_score"].values,
            local_df["pred_score"].values,
            method="spearman",
        )
        if np.isfinite(rho_stat):
            axis.text(0.03, 0.97, f"Spearman ρ = {rho_stat:.3f}\n{format_pval(rho_p)}", transform=axis.transAxes, ha="left", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))
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
    met_expr = _load_crpc_ra_met_expression()
    met_expr = met_expr.loc[~met_expr.index.duplicated(keep="first")]

    present_excluded_met_samples = [
        sample_name
        for sample_name in _CRPC_RA_METS_EXCLUDED_SAMPLES
        if sample_name in met_expr.columns
    ]
    exclusion_note_path = os.path.join(analysis_out, "excluded_met_samples.txt")
    with open(exclusion_note_path, "w", encoding="utf-8") as handle:
        handle.write("CRPC-RA metastasis composition analysis met/RNA-seq exclusions\n")
        handle.write("Configured exclusions:\n")
        for sample_name in _CRPC_RA_METS_EXCLUDED_SAMPLES:
            handle.write(f"- {sample_name}\n")
        if present_excluded_met_samples:
            handle.write("\nExcluded from this run:\n")
            for sample_name in present_excluded_met_samples:
                handle.write(f"- {sample_name}\n")
        else:
            handle.write("\nNo configured exclusions were present in the met expression matrix for this run.\n")

    if present_excluded_met_samples:
        print(
            "    CRPC-RA metastasis composition: excluding configured met/RNA-seq samples: "
            + ", ".join(present_excluded_met_samples)
        )
        met_expr = met_expr.drop(columns=present_excluded_met_samples)

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
            if len(local_df) < MIN_CORRELATION_N:
                continue
            _, rho_stat, rho_p = _generic_correlation_values(
                local_df[predictor].values,
                local_df[outcome].values,
                method="spearman",
            )
            if not np.isfinite(rho_stat):
                continue
            ci_lo, ci_hi = _crpc_ra_bootstrap_correlation_ci(local_df[predictor], local_df[outcome], method="spearman")
            correlation_rows.append({
                "outcome": outcome_label,
                "predictor": predictor_label,
                "spearman_rho": rho_stat,
                "p_value": rho_p,
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
    color_column: str = "cohort",
    legend_title: str = "Cohort",
    pred_ci_lo: Optional[pd.Series] = None,
    pred_ci_hi: Optional[pd.Series] = None,
    point_labels: Optional[pd.Series] = None,
    highlighted_samples: Optional[Sequence[str]] = None,
    default_marker: str = "o",
    highlighted_marker: str = "D",
    marker_note: Optional[str] = None,
    show_identity_line: bool = True,
    fit_line_color: str = "#CC3311",
) -> None:
    """Scatter truth-vs-predicted score values using metadata-driven coloring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shared = truth_scores.index.intersection(pred_scores.index).intersection(sample_meta.index)
    if len(shared) < MIN_CORRELATION_N:
        return

    truth_vals = pd.to_numeric(truth_scores.loc[shared], errors="coerce")
    pred_vals = pd.to_numeric(pred_scores.loc[shared], errors="coerce")
    valid = np.isfinite(truth_vals.values) & np.isfinite(pred_vals.values)
    if int(valid.sum()) < MIN_CORRELATION_N:
        return

    shared = shared[valid]
    truth_vals = truth_vals.loc[shared]
    pred_vals = pred_vals.loc[shared]
    meta = sample_meta.loc[shared]
    ci_lo_vals = pd.to_numeric(pred_ci_lo.reindex(shared), errors="coerce") if pred_ci_lo is not None else None
    ci_hi_vals = pd.to_numeric(pred_ci_hi.reindex(shared), errors="coerce") if pred_ci_hi is not None else None
    highlighted_set = {str(sample_name) for sample_name in highlighted_samples} if highlighted_samples is not None else set()
    palette = load_palette()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    seen_groups: Dict[str, str] = {}
    for sample in shared:
        group_name = str(meta.loc[sample, color_column]) if color_column in meta.columns else "Unknown"
        status = str(meta.loc[sample, "status"]) if "status" in meta.columns else "validation"
        tfx_val = pd.to_numeric(pd.Series([meta.loc[sample, "tfx"]]), errors="coerce").iloc[0] \
            if "tfx" in meta.columns else np.nan
        color = palette.get(group_name, "#999999")
        sample_key = str(sample)
        marker = highlighted_marker if sample_key in highlighted_set else ("D" if status == "holdout" else default_marker)
        size = max(40.0, 260.0 * float(tfx_val)) if np.isfinite(tfx_val) else 60.0
        if ci_lo_vals is not None and ci_hi_vals is not None:
            ci_lo_val = ci_lo_vals.loc[sample]
            ci_hi_val = ci_hi_vals.loc[sample]
            if np.isfinite(ci_lo_val) and np.isfinite(ci_hi_val):
                y_value = float(pred_vals.loc[sample])
                ax.errorbar(
                    float(truth_vals.loc[sample]),
                    y_value,
                    yerr=np.array([[max(0.0, y_value - float(ci_lo_val))], [max(0.0, float(ci_hi_val) - y_value)]]),
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.9,
                    alpha=0.35,
                    capsize=1.8,
                    zorder=1,
                )
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
        if point_labels is not None and sample in point_labels.index:
            label_text = str(point_labels.loc[sample])
            if label_text:
                ax.text(
                    float(truth_vals.loc[sample]),
                    float(pred_vals.loc[sample]),
                    label_text,
                    fontsize=6,
                    alpha=0.82,
                    ha="left",
                    va="bottom",
                    clip_on=False,
                )
        seen_groups[group_name] = color

    lo = float(min(truth_vals.min(), pred_vals.min()))
    hi = float(max(truth_vals.max(), pred_vals.max()))
    pad = 0.05 * max(hi - lo, 1e-6)
    if show_identity_line:
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="#BBBBBB", lw=1.2)

    n_corr, r_p, p_p, r_s, p_s = _truth_pred_correlation_values(
        truth_vals.values,
        pred_vals.values,
    )
    can_fit = (
        n_corr >= MIN_CORRELATION_N
        and np.isfinite(r_p)
        and float(np.ptp(truth_vals.values)) > 0.0
        and float(np.ptp(pred_vals.values)) > 0.0
    )
    if can_fit:
        coeffs = np.polyfit(truth_vals.values, pred_vals.values, 1)
        x_line = np.linspace(lo - pad, hi + pad, 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), color=fit_line_color, lw=1.5, alpha=0.9)

    axis_lo = lo - pad
    axis_hi = hi + pad
    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)
    ax.set_aspect("equal", adjustable="box")
    shared_ticks = ax.get_xticks()
    shared_ticks = shared_ticks[(shared_ticks >= axis_lo) & (shared_ticks <= axis_hi)]
    if shared_ticks.size >= 2:
        ax.set_xticks(shared_ticks)
        ax.set_yticks(shared_ticks)

    if n_corr >= MIN_CORRELATION_N and np.isfinite(r_p) and np.isfinite(r_s):
        metric_text = (
            f"Pearson r = {r_p:.3f}  ({format_pval(p_p)})\n"
            f"Spearman ρ = {r_s:.3f}  ({format_pval(p_s)})\n"
            f"n = {n_corr}"
        )
    elif n_corr >= MIN_CORRELATION_N and r_p == 0.0 and r_s == 0.0:
        metric_text = (
            "Pearson r = 0.000  (p=NA)\n"
            "Spearman ρ = 0.000  (p=NA)\n"
            f"n = {n_corr}"
        )
    else:
        metric_text = f"Pearson r = NA\nSpearman ρ = NA\nn = {n_corr}"
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", label=group_name,
            markerfacecolor=color, markeredgecolor="black", markersize=8,
        )
        for group_name, color in sorted(seen_groups.items())
    ]
    if handles:
        ax.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.02, 1.0),
                  loc="upper left", fontsize=8)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_box_aspect(1)

    fig.subplots_adjust(right=0.73)
    fig.text(
        0.76,
        0.72,
        metric_text,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    if marker_note:
        fig.text(0.76, 0.20, marker_note, ha="left", va="bottom", fontsize=8)

    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _zscore_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric[np.isfinite(numeric.values)]
    if valid.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    std = float(valid.std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        centered = valid - float(valid.mean())
        out = pd.Series(np.nan, index=values.index, dtype=float)
        out.loc[valid.index] = centered.astype(float)
        return out
    z_values = (valid - float(valid.mean())) / std
    out = pd.Series(np.nan, index=values.index, dtype=float)
    out.loc[valid.index] = z_values.astype(float)
    return out


def _compute_shortlist_scores_for_matrix(
    expr_wide: pd.DataFrame,
    sample_names: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """Compute shortlist ssGSEA scores for the selected samples."""
    shortlist_sets = _load_shortlist_sets()
    if not shortlist_sets:
        return pd.DataFrame()

    selected_samples = list(expr_wide.columns) if sample_names is None else [
        sample_name for sample_name in sample_names if sample_name in expr_wide.columns
    ]
    if not selected_samples:
        return pd.DataFrame()
    return _compute_ssgsea_score_matrix(expr_wide[selected_samples], shortlist_sets, min_size=3)


def _compute_crpc_ra_shortlist_score_bundle(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute the aligned CRPC-RA truth/pred shortlist score matrices once."""
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    return {
        "common_samples": common_samples,
        "truth_scores": _compute_shortlist_scores_for_matrix(truth_wide, common_samples),
        "pred_scores": _compute_shortlist_scores_for_matrix(pred_wide, common_samples),
    }


def _extract_expression_or_score_series(
    expr_wide: pd.DataFrame,
    score_wide: Optional[pd.DataFrame],
    feature_name: str,
    sample_names: Sequence[Any],
) -> pd.Series:
    """Return one aligned gene-expression or score series."""
    sample_index = pd.Index([str(sample_name) for sample_name in sample_names])
    if feature_name in expr_wide.index:
        series = pd.to_numeric(expr_wide.loc[feature_name].reindex(sample_index), errors="coerce")
    elif score_wide is not None and feature_name in score_wide.index:
        series = pd.to_numeric(score_wide.loc[feature_name].reindex(sample_index), errors="coerce")
    else:
        return pd.Series(dtype=float)
    series.index = sample_index
    return series.astype(float)


def _plot_crpc_ra_group_violin(
    group_df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    y_label: str,
) -> None:
    """Plot a two-group violin with jittered sample points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    if not {"group", "value"}.issubset(group_df.columns):
        return
    working = group_df.copy()
    working["group"] = working["group"].astype(str)
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    working = working.dropna(subset=["group", "value"])
    group_order = [
        group_name
        for group_name in ["Untreated", "Pluvicto-Treated"]
        if group_name in set(working["group"])
    ]
    if len(group_order) < 2:
        return

    grouped = [
        working.loc[working["group"] == group_name, "value"].astype(float).values
        for group_name in group_order
    ]
    if any(len(values) == 0 for values in grouped):
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    violin = ax.violinplot(
        grouped,
        positions=np.arange(len(group_order), dtype=float),
        widths=0.72,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    violin_colors = ["#6FA8DC", "#E69138"]
    for body, color in zip(violin["bodies"], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.35)

    rng = np.random.default_rng(42)
    for pos, values, color in zip(np.arange(len(group_order), dtype=float), grouped, violin_colors):
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(
            pos + jitter,
            values,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            s=38,
            alpha=0.90,
            zorder=3,
        )
        ax.hlines(float(np.median(values)), pos - 0.18, pos + 0.18, color="black", linewidth=2.0, zorder=4)

    p_value = float(mannwhitneyu(grouped[0], grouped[1], alternative="two-sided").pvalue)
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-6)
    y_ann = max(float(np.max(grouped[0])), float(np.max(grouped[1]))) + 0.10 * y_range
    cap_y = 0.02 * y_range
    ax.plot([0.0, 0.0, 1.0, 1.0], [y_ann - cap_y, y_ann, y_ann, y_ann - cap_y], color="black", linewidth=1.0)
    ax.text(0.5, y_ann + 0.01 * y_range, format_pval(p_value), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(y_min, max(y_max, y_ann + 0.10 * y_range))
    ax.set_xticks(np.arange(len(group_order), dtype=float))
    ax.set_xticklabels(group_order)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_crpc_ra_paired_violin(
    paired_df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    y_label: str,
) -> None:
    """Plot paired C1-versus-death violins with patient-specific markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon

    if not {"patient_id", "visit", "value"}.issubset(paired_df.columns):
        return
    working = paired_df.copy()
    working["patient_id"] = working["patient_id"].astype(str)
    working["visit"] = working["visit"].astype(str)
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    working = working.dropna(subset=["patient_id", "visit", "value"])
    visit_order = ["C1", "Death"]
    wide = (
        working.pivot_table(index="patient_id", columns="visit", values="value", aggfunc="first")
        .reindex(columns=visit_order)
        .dropna()
    )
    if wide.empty:
        return

    grouped = [wide[visit_name].astype(float).values for visit_name in visit_order]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    violin = ax.violinplot(
        grouped,
        positions=np.arange(len(visit_order), dtype=float),
        widths=0.72,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    violin_colors = ["#6FA8DC", "#C0504D"]
    for body, color in zip(violin["bodies"], violin_colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.32)

    patient_markers = {
        patient_id: _CRPC_RA_PAIRED_MARKERS[idx % len(_CRPC_RA_PAIRED_MARKERS)]
        for idx, patient_id in enumerate(wide.index.astype(str))
    }
    for patient_id, row in wide.iterrows():
        y_values = [float(row[visit_name]) for visit_name in visit_order]
        ax.plot([0.0, 1.0], y_values, color="#7A7A7A", linewidth=1.2, alpha=0.70, zorder=2)
        for x_pos, y_value in zip([0.0, 1.0], y_values):
            ax.scatter(
                x_pos,
                y_value,
                marker=patient_markers[str(patient_id)],
                color="white",
                edgecolor="black",
                linewidth=0.7,
                s=58,
                zorder=3,
            )

    for pos, values in enumerate(grouped):
        ax.hlines(float(np.median(values)), pos - 0.18, pos + 0.18, color="black", linewidth=2.0, zorder=4)

    p_value = np.nan
    if len(wide) >= 2:
        try:
            p_value = float(wilcoxon(wide["C1"].values, wide["Death"].values, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p_value = np.nan
    ax.text(
        0.03,
        0.97,
        f"n = {len(wide)}\nWilcoxon {format_pval(float(p_value)) if np.isfinite(p_value) else 'n/a'}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )
    ax.set_xticks(np.arange(len(visit_order), dtype=float))
    ax.set_xticklabels(visit_order)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_crpc_ra_pluvicto_overlap_bundle(
    *,
    minimum_tfx: float,
    apply_calibration: bool,
    calibration_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Load the Pluvicto C1 overlap bundle used by CRPC-RA paired analyses."""
    pluvicto_path = _require_relative_input_path(
        DISCOVERY_PROTEUS_PATHS.get("CRPC-Pluvicto_Full"),
        description="Validation_Patient CRPC-Pluvicto overlap predictions",
    )
    full_data = load_holdout_predictions_proteus_full(str(pluvicto_path))
    pred_wide = full_data.get("pred_wide")
    if pred_wide is None or pred_wide.empty:
        print("    [skip] CRPC-RA Pluvicto overlap: no Pluvicto prediction matrix available")
        return None

    overlap_key = _load_crpc_ra_pluvicto_key()
    baseline_samples = _select_pluvicto_baseline_samples(pred_wide.columns, include_fhl=True)
    pluvicto_patient_to_sample: Dict[str, str] = {}
    for sample_name in baseline_samples:
        patient_id = extract_pluvicto_patient_id(str(sample_name))
        if patient_id not in pluvicto_patient_to_sample:
            pluvicto_patient_to_sample[patient_id] = str(sample_name)
    baseline_sample_set = {str(sample_name) for sample_name in baseline_samples}

    patient_to_sample: Dict[str, str] = {}
    for _, row in overlap_key.iterrows():
        tan_id = str(row["TAN_ID"])
        pluvicto_overlap_id = str(row["Pluvicto_ID"])
        c1_sample = None
        if pluvicto_overlap_id in baseline_sample_set:
            c1_sample = pluvicto_overlap_id
        else:
            c1_sample = pluvicto_patient_to_sample.get(pluvicto_overlap_id)
        if c1_sample is not None:
            patient_to_sample[tan_id] = c1_sample
    if not patient_to_sample:
        print("    [skip] CRPC-RA Pluvicto overlap: no overlap C1 samples found")
        return None

    overlap_samples = list(patient_to_sample.values())
    pluvicto_subtypes = _load_filtered_subtypes_for_reference(
        "CRPC-Pluvicto",
        overlap_samples,
        minimum_tfx=minimum_tfx,
        report_context="Validation_Patient CRPC-Pluvicto overlap",
    )
    if minimum_tfx > 0.0 and pluvicto_subtypes is not None:
        overlap_samples = [sample_name for sample_name in overlap_samples if sample_name in pluvicto_subtypes.index]
        patient_to_sample = {
            patient_id: sample_name
            for patient_id, sample_name in patient_to_sample.items()
            if sample_name in overlap_samples
        }
    if not overlap_samples:
        print("    [skip] CRPC-RA Pluvicto overlap: no overlap C1 samples remain after TFx filtering")
        return None

    full_data = _filter_prediction_bundle_to_samples(full_data, overlap_samples)
    analysis_data = _prepare_proteus_analysis_bundle(full_data)
    if apply_calibration:
        if calibration_state is None:
            raise ValueError("Validation_Patient CRPC-Pluvicto overlap calibration was requested but no calibration state was available")
        analysis_data = _apply_distribution_calibration_bundle(
            analysis_data,
            calibration_state,
            source_by_sample=_constant_source_by_sample(analysis_data["pred_wide"].columns, "CRPC-Pluvicto"),
            sample_fold=None,
            label="Validation_Patient CRPC-Pluvicto overlap",
        )

    if pluvicto_subtypes is not None:
        pluvicto_subtypes = pluvicto_subtypes.reindex(analysis_data["pred_wide"].columns)
    return {
        "bundle": analysis_data,
        "pred_wide": analysis_data["pred_wide"],
        "subtypes_df": pluvicto_subtypes,
        "patient_to_sample": patient_to_sample,
    }


def _run_crpc_ra_shortlist_score_scatters(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    cohort_out: str,
    *,
    subtypes_df: Optional[pd.DataFrame],
    score_bundle: Optional[Dict[str, Any]] = None,
    overwrite: bool,
) -> None:
    if score_bundle is None:
        score_bundle = _compute_crpc_ra_shortlist_score_bundle(truth_wide, pred_wide)

    common_samples = pd.Index(score_bundle["common_samples"])
    if len(common_samples) < MIN_CORRELATION_N:
        return

    score_meta = pd.DataFrame(index=common_samples)
    if subtypes_df is not None:
        score_meta = subtypes_df.reindex(common_samples).copy()

    truth_scores = score_bundle["truth_scores"]
    raw_scores = score_bundle["pred_scores"]
    if truth_scores.empty or raw_scores.empty:
        return

    scatter_out = ensure_dir(os.path.join(cohort_out, "ShortlistScores", "scatter_plots"))
    score_names = ("ARG.10", "NE.10", "CCP.31")
    prefix_map = _build_prefix_map(pd.Index(common_samples), "CRPC-RA")
    overlap_key = _load_crpc_ra_pluvicto_key()
    overlap_pairs = [
        (tan_id, prefix_map[tan_id])
        for tan_id in overlap_key["TAN_ID"].astype(str).tolist()
        if tan_id in prefix_map and prefix_map[tan_id] in common_samples
    ]
    overlap_samples = [sample_name for _, sample_name in overlap_pairs]
    marker_note = "Diamond = pluvicto-overlap patient; circle = other CRPC-RA patients"
    for score_name in score_names:
        if score_name not in truth_scores.index or score_name not in raw_scores.index:
            continue
        display_name = _CRPC_RA_SHORTLIST_DISPLAY.get(score_name, score_name)
        _maybe(
            _crosscohort_score_scatter,
            overwrite,
            truth_scores.loc[score_name],
            raw_scores.loc[score_name],
            score_meta,
            os.path.join(scatter_out, f"{score_name}_raw"),
            title=f"CRPC-RA — {display_name} ssGSEA (Truth vs Predicted Raw)",
            x_label=f"RNA-Seq {display_name} ssGSEA Score",
            y_label=f"Predicted {display_name} ssGSEA Score",
            color_column="subtype",
            legend_title="Subtype",
            highlighted_samples=overlap_samples,
            marker_note=marker_note,
            show_identity_line=False,
            fit_line_color="#7A7A7A",
        )

        truth_z = _zscore_series(truth_scores.loc[score_name])
        pred_z = _zscore_series(raw_scores.loc[score_name])
        _maybe(
            _crosscohort_score_scatter,
            overwrite,
            truth_z,
            pred_z,
            score_meta,
            os.path.join(scatter_out, f"{score_name}_zscore"),
            title=f"CRPC-RA — {display_name} ssGSEA (Truth vs Predicted Z-score)",
            x_label=f"RNA-Seq {display_name} ssGSEA Z-score",
            y_label=f"Predicted {display_name} ssGSEA Z-score",
            color_column="subtype",
            legend_title="Subtype",
            highlighted_samples=overlap_samples,
            marker_note=marker_note,
            show_identity_line=False,
            fit_line_color="#7A7A7A",
        )


def _crpc_ra_pure_phenotype_samples(
    subtypes_df: Optional[pd.DataFrame],
    sample_names: Sequence[str],
) -> List[str]:
    """Return CRPC-RA samples with a single, non-mixed subtype label."""
    if subtypes_df is None or "subtype" not in subtypes_df.columns:
        return []

    subtype_series = subtypes_df.reindex(sample_names)["subtype"]
    subtype_text = subtype_series.astype(str).str.strip()
    pure_mask = (
        subtype_series.notna()
        & subtype_text.ne("")
        & subtype_text.str.lower().ne("nan")
        & ~subtype_text.str.contains("/", regex=False)
        & subtype_text.str.upper().ne("MIXED")
    )
    return [sample_name for sample_name in sample_names if bool(pure_mask.get(sample_name, False))]


def _run_crpc_ra_pure_phenotype_heatmap(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    cohort_out: str,
    *,
    subtypes_df: Optional[pd.DataFrame],
    overwrite: bool,
) -> None:
    """Write a CRPC-RA PC-Phenotyping heatmap restricted to pure phenotype samples."""
    resolved = STANDARD_GENE_SETS.get("PC-Phenotyping", "PC-Phenotyping")
    if not gene_list_available(resolved):
        return

    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    pure_samples = _crpc_ra_pure_phenotype_samples(subtypes_df, [str(sample_name) for sample_name in common_samples])
    if len(pure_samples) < 2:
        return

    genes = get_gene_list(resolved)
    breaks = GENE_SET_BREAKS.get("PC-Phenotyping")
    truth_sort_genes = genes[:breaks[-1]] if breaks else None
    pure_subtypes = subtypes_df.loc[pure_samples].copy() if subtypes_df is not None else None
    if pure_subtypes is not None:
        pure_subtypes.to_csv(
            os.path.join(cohort_out, "PC-Phenotyping_pure_phenotype_samples.tsv"),
            sep="\t",
        )
    _maybe(
        double_heatmap,
        overwrite,
        truth_wide[pure_samples],
        pred_wide[pure_samples],
        genes,
        os.path.join(cohort_out, "double_heatmap_PC-Phenotyping_pure_phenotype"),
        subtypes=pure_subtypes,
        breaks=breaks,
        truth_sort_genes=truth_sort_genes,
    )


def _plot_crpc_ra_pluvicto_treatment_boxplot(
    gene_name: str,
    untreated_truth: np.ndarray,
    untreated_pred: np.ndarray,
    treated_truth: np.ndarray,
    treated_pred: np.ndarray,
    out_path: str,
    *,
    p_truth: float,
    p_pred: float,
    n_untreated: int,
    n_treated: int,
) -> None:
    """Plot a four-box CRPC-RA comparison for untreated vs Pluvicto-overlap samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    positions = [0.85, 1.15, 1.85, 2.15]
    values = [untreated_truth, untreated_pred, treated_truth, treated_pred]
    colors = ["#7DB7D5", "#E39D3E", "#7DB7D5", "#E39D3E"]
    labels = ["RNA-Seq", "Predicted", "RNA-Seq", "Predicted"]

    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="#444444", linewidth=1.1),
        capprops=dict(color="#444444", linewidth=1.1),
        flierprops=dict(markersize=0),
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.65)

    rng = np.random.default_rng(42)
    for pos, vals, color in zip(positions, values, colors):
        jitter = rng.uniform(-0.045, 0.045, size=len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            s=32,
            alpha=0.90,
            zorder=3,
        )

    finite_arrays = [vals[np.isfinite(vals)] for vals in values if len(vals) > 0]
    if finite_arrays:
        data_min = min(float(np.min(vals)) for vals in finite_arrays)
        data_max = max(float(np.max(vals)) for vals in finite_arrays)
    else:
        data_min = 0.0
        data_max = 1.0
    y_range = max(data_max - data_min, 1e-6)
    y_base = data_max + 0.10 * y_range
    cap_y = 0.03 * y_range

    ax.plot([positions[0], positions[0], positions[2], positions[2]], [y_base - cap_y, y_base, y_base, y_base - cap_y], lw=1.0, color="black")
    ax.text(
        0.5 * (positions[0] + positions[2]),
        y_base + 0.02 * y_range,
        f"RNA-Seq {format_pval(float(p_truth))}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    y_second = y_base + 0.16 * y_range
    ax.plot([positions[1], positions[1], positions[3], positions[3]], [y_second - cap_y, y_second, y_second, y_second - cap_y], lw=1.0, color="black")
    ax.text(
        0.5 * (positions[1] + positions[3]),
        y_second + 0.02 * y_range,
        f"Predicted {format_pval(float(p_pred))}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.set_ylim(data_min - 0.08 * y_range, y_second + 0.14 * y_range)

    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels([
        f"Untreated\n(n={n_untreated})",
        f"Pluvicto-Treated\n(n={n_treated})",
    ])
    ax.set_ylabel("Expression (log2CPM+1)")
    ax.set_title(f"CRPC-RA Pluvicto vs Untreated — {gene_name}", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.7)
    ax.legend(
        handles=[
            Patch(facecolor="#7DB7D5", edgecolor="black", alpha=0.65, label="RNA-Seq truth"),
            Patch(facecolor="#E39D3E", edgecolor="black", alpha=0.65, label="Predicted"),
        ],
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    for pos, label in zip(positions, labels):
        ax.text(pos, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), label, ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_crpc_ra_pluvicto_vs_untreated_analysis(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    cohort_out: str,
    *,
    death_bundle: Dict[str, Any],
    subtypes_df: Optional[pd.DataFrame],
    death_subtypes_df: Optional[pd.DataFrame],
    minimum_tfx: float,
    apply_calibration: bool,
    calibration_state: Optional[Dict[str, Any]],
    score_bundle: Optional[Dict[str, Any]] = None,
    overwrite: bool,
) -> None:
    """Expand CRPC-RA Pluvicto outputs across untreated, paired, and limma views."""
    from scipy.stats import mannwhitneyu

    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    if len(common_samples) == 0:
        print("    [skip] CRPC-RA Pluvicto-vs-Untreated: no overlapping truth/prediction samples")
        return

    truth_prefix_map = _build_prefix_map(pd.Index(common_samples), "CRPC-RA")
    overlap_key = _load_crpc_ra_pluvicto_key()
    overlap_pairs = [
        (tan_id, truth_prefix_map[tan_id])
        for tan_id in overlap_key["TAN_ID"].astype(str).tolist()
        if tan_id in truth_prefix_map and truth_prefix_map[tan_id] in common_samples
    ]
    treated_samples = [sample_name for _, sample_name in overlap_pairs]
    treated_sample_set = set(treated_samples)
    untreated_samples = [sample_name for sample_name in common_samples if sample_name not in treated_sample_set]

    print(
        "    CRPC-RA Pluvicto-vs-Untreated: "
        f"{len(treated_samples)} Pluvicto-overlap vs {len(untreated_samples)} untreated samples"
    )
    if len(treated_samples) == 0 or len(untreated_samples) == 0:
        print("    [skip] CRPC-RA Pluvicto-vs-Untreated: one comparison group is empty")
        return

    analysis_out = ensure_dir(os.path.join(cohort_out, "CRPC-RA_Pluvicto-vs-Untreated"))
    true_pred_out = ensure_dir(os.path.join(analysis_out, "treated-vs-untreated_true-vs-pred"))
    pd.DataFrame(
        {
            "sample": untreated_samples + treated_samples,
            "group": (["Untreated"] * len(untreated_samples)) + (["Pluvicto-Treated"] * len(treated_samples)),
        }
    ).to_csv(os.path.join(analysis_out, "sample_groups.tsv"), sep="\t", index=False)
    rows: List[Dict[str, Any]] = []
    for gene_name in _CRPC_RA_PLUVICTO_TREATMENT_GENES:
        truth_present = gene_name in truth_wide.index
        pred_present = gene_name in pred_wide.index
        if not truth_present or not pred_present:
            rows.append(
                {
                    "gene": gene_name,
                    "status": "missing_gene",
                    "in_truth": bool(truth_present),
                    "in_prediction": bool(pred_present),
                }
            )
            continue

        untreated_truth = pd.to_numeric(truth_wide.loc[gene_name, untreated_samples], errors="coerce").dropna().to_numpy(dtype=float)
        treated_truth = pd.to_numeric(truth_wide.loc[gene_name, treated_samples], errors="coerce").dropna().to_numpy(dtype=float)
        untreated_pred = pd.to_numeric(pred_wide.loc[gene_name, untreated_samples], errors="coerce").dropna().to_numpy(dtype=float)
        treated_pred = pd.to_numeric(pred_wide.loc[gene_name, treated_samples], errors="coerce").dropna().to_numpy(dtype=float)

        if min(len(untreated_truth), len(treated_truth), len(untreated_pred), len(treated_pred)) == 0:
            rows.append(
                {
                    "gene": gene_name,
                    "status": "insufficient_values",
                    "n_untreated_truth": int(len(untreated_truth)),
                    "n_treated_truth": int(len(treated_truth)),
                    "n_untreated_pred": int(len(untreated_pred)),
                    "n_treated_pred": int(len(treated_pred)),
                }
            )
            continue

        p_truth = float(mannwhitneyu(untreated_truth, treated_truth, alternative="two-sided").pvalue)
        p_pred = float(mannwhitneyu(untreated_pred, treated_pred, alternative="two-sided").pvalue)
        rows.append(
            {
                "gene": gene_name,
                "status": "ok",
                "n_untreated_truth": int(len(untreated_truth)),
                "n_treated_truth": int(len(treated_truth)),
                "n_untreated_pred": int(len(untreated_pred)),
                "n_treated_pred": int(len(treated_pred)),
                "median_untreated_truth": float(np.median(untreated_truth)),
                "median_treated_truth": float(np.median(treated_truth)),
                "median_untreated_pred": float(np.median(untreated_pred)),
                "median_treated_pred": float(np.median(treated_pred)),
                "mwu_p_truth": p_truth,
                "mwu_p_pred": p_pred,
            }
        )
        _maybe(
            _plot_crpc_ra_pluvicto_treatment_boxplot,
            overwrite,
            gene_name,
            untreated_truth,
            untreated_pred,
            treated_truth,
            treated_pred,
            os.path.join(true_pred_out, gene_name),
            p_truth=p_truth,
            p_pred=p_pred,
            n_untreated=len(untreated_samples),
            n_treated=len(treated_samples),
        )

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(true_pred_out, "gene_boxplot_stats.tsv"), sep="\t", index=False)

    if score_bundle is None:
        score_bundle = _compute_crpc_ra_shortlist_score_bundle(truth_wide, pred_wide)

    truth_scores = score_bundle["truth_scores"]
    truth_only_out = ensure_dir(os.path.join(analysis_out, "treated-vs-untreated_rna-seq"))
    truth_feature_rows: List[Dict[str, Any]] = []
    for feature_name in _CRPC_RA_PLUVICTO_PAIRED_FEATURES:
        feature_series = _extract_expression_or_score_series(
            truth_wide,
            truth_scores,
            feature_name,
            common_samples,
        )
        if feature_series.empty:
            continue
        untreated_vals = feature_series.reindex(untreated_samples).dropna().to_numpy(dtype=float)
        treated_vals = feature_series.reindex(treated_samples).dropna().to_numpy(dtype=float)
        if min(len(untreated_vals), len(treated_vals)) == 0:
            continue
        truth_feature_rows.append(
            {
                "feature": feature_name,
                "n_untreated": int(len(untreated_vals)),
                "n_treated": int(len(treated_vals)),
                "median_untreated": float(np.median(untreated_vals)),
                "median_treated": float(np.median(treated_vals)),
                "mwu_p": float(mannwhitneyu(untreated_vals, treated_vals, alternative="two-sided").pvalue),
            }
        )
        group_rows = pd.DataFrame(
            {
                "group": (["Untreated"] * len(untreated_vals)) + (["Pluvicto-Treated"] * len(treated_vals)),
                "value": np.concatenate([untreated_vals, treated_vals]),
            }
        )
        ylabel = "RNA-Seq expression (log2CPM+1)" if feature_name == "FOLH1" else f"RNA-Seq {feature_name} ssGSEA score"
        _maybe(
            _plot_crpc_ra_group_violin,
            overwrite,
            group_rows,
            os.path.join(truth_only_out, _analysis_safe_name(feature_name)),
            title=f"CRPC-RA RNA-Seq — {feature_name}: untreated vs Pluvicto-overlap",
            y_label=ylabel,
        )
    if truth_feature_rows:
        pd.DataFrame(truth_feature_rows).to_csv(
            os.path.join(truth_only_out, "truth_rna_seq_stats.tsv"),
            sep="\t",
            index=False,
        )

    pluvicto_overlap = _load_crpc_ra_pluvicto_overlap_bundle(
        minimum_tfx=minimum_tfx,
        apply_calibration=apply_calibration,
        calibration_state=calibration_state,
    )
    if pluvicto_overlap is None:
        return

    c1_pred = pluvicto_overlap["pred_wide"]
    c1_bundle = pluvicto_overlap["bundle"]
    c1_subtypes = pluvicto_overlap.get("subtypes_df")
    c1_patient_to_sample = pluvicto_overlap["patient_to_sample"]
    death_pred_wide = death_bundle.get("pred_wide")
    if death_pred_wide is None or death_pred_wide.empty:
        print("    [skip] CRPC-RA paired C1-death: no CRPC-RA death prediction matrix available")
        return
    death_subtype_table = death_subtypes_df if death_subtypes_df is not None else subtypes_df
    predicted_death_samples = pd.Index(death_pred_wide.columns)
    if death_subtype_table is not None:
        predicted_death_samples = predicted_death_samples.union(pd.Index(death_subtype_table.index.astype(str)))
    death_prefix_map = _build_prefix_map(predicted_death_samples, "CRPC-RA")
    overlap_key = _crpc_ra_pluvicto_ctdna_overlap_key()
    paired_rows: List[Dict[str, Any]] = []
    for patient_id in overlap_key["TAN_ID"].astype(str).tolist():
        death_sample = death_prefix_map.get(patient_id)
        c1_sample = c1_patient_to_sample.get(patient_id)
        if death_sample is None or c1_sample is None:
            continue
        paired_rows.append(
            {
                "patient_id": patient_id,
                "death_sample": death_sample,
                "c1_sample": c1_sample,
                "death_tfx": float(pd.to_numeric(death_subtype_table.reindex([death_sample])["tfx"], errors="coerce").iloc[0])
                if death_subtype_table is not None and "tfx" in death_subtype_table.columns
                else np.nan,
                "c1_tfx": float(pd.to_numeric(c1_subtypes.reindex([c1_sample])["tfx"], errors="coerce").iloc[0])
                if c1_subtypes is not None and "tfx" in c1_subtypes.columns
                else np.nan,
            }
        )
    pair_df = pd.DataFrame(paired_rows)
    if pair_df.empty:
        print("    [skip] CRPC-RA paired C1-death: no matched Pluvicto/CRPC-RA overlap patients remained")
        return

    pair_df = pair_df.set_index("patient_id")
    pair_df.to_csv(os.path.join(analysis_out, "paired_patient_map.tsv"), sep="\t")

    paired_out = ensure_dir(os.path.join(analysis_out, "paired_c1-death"))
    c1_scores = _compute_shortlist_scores_for_matrix(c1_pred, pair_df["c1_sample"].tolist())
    death_scores = _compute_shortlist_scores_for_matrix(death_pred_wide, pair_df["death_sample"].tolist())
    paired_value_rows: List[Dict[str, Any]] = []
    paired_summary_rows: List[Dict[str, Any]] = []
    for feature_name in _CRPC_RA_PLUVICTO_PAIRED_FEATURES:
        c1_series = _extract_expression_or_score_series(
            c1_pred,
            c1_scores,
            feature_name,
            pair_df["c1_sample"].tolist(),
        )
        death_series = _extract_expression_or_score_series(
            death_pred_wide,
            death_scores,
            feature_name,
            pair_df["death_sample"].tolist(),
        )
        feature_rows: List[Dict[str, Any]] = []
        for patient_id, row in pair_df.iterrows():
            c1_value = float(pd.to_numeric(pd.Series([c1_series.get(str(row["c1_sample"]), np.nan)]), errors="coerce").iloc[0])
            death_value = float(pd.to_numeric(pd.Series([death_series.get(str(row["death_sample"]), np.nan)]), errors="coerce").iloc[0])
            if not np.isfinite(c1_value) or not np.isfinite(death_value):
                continue
            feature_rows.extend(
                [
                    {
                        "feature": feature_name,
                        "patient_id": patient_id,
                        "visit": "C1",
                        "sample": str(row["c1_sample"]),
                        "value": c1_value,
                    },
                    {
                        "feature": feature_name,
                        "patient_id": patient_id,
                        "visit": "Death",
                        "sample": str(row["death_sample"]),
                        "value": death_value,
                    },
                ]
            )
        if not feature_rows:
            continue
        feature_df = pd.DataFrame(feature_rows)
        paired_value_rows.extend(feature_rows)
        wide_feature = feature_df.pivot_table(index="patient_id", columns="visit", values="value", aggfunc="first")
        if {"C1", "Death"}.issubset(wide_feature.columns):
            deltas = pd.to_numeric(wide_feature["Death"], errors="coerce") - pd.to_numeric(wide_feature["C1"], errors="coerce")
            paired_summary_rows.append(
                {
                    "feature": feature_name,
                    "n_pairs": int(len(wide_feature)),
                    "median_c1": float(np.median(wide_feature["C1"].astype(float).values)),
                    "median_death": float(np.median(wide_feature["Death"].astype(float).values)),
                    "median_delta_death_minus_c1": float(np.median(deltas.values.astype(float))),
                    "mean_delta_death_minus_c1": float(np.mean(deltas.values.astype(float))),
                }
            )
        ylabel = "Predicted expression (log2CPM+1)" if feature_name == "FOLH1" else f"Predicted {feature_name} ssGSEA score"
        _maybe(
            _plot_crpc_ra_paired_violin,
            overwrite,
            feature_df,
            os.path.join(paired_out, _analysis_safe_name(feature_name)),
            title=f"CRPC-RA / Pluvicto paired C1 vs death — {feature_name}",
            y_label=ylabel,
        )
    if paired_value_rows:
        pd.DataFrame(paired_value_rows).to_csv(
            os.path.join(paired_out, "paired_feature_values.tsv"),
            sep="\t",
            index=False,
        )
    if paired_summary_rows:
        pd.DataFrame(paired_summary_rows).to_csv(
            os.path.join(paired_out, "paired_feature_summary.tsv"),
            sep="\t",
            index=False,
        )

    pharmacodynamics_out = ensure_dir(os.path.join(analysis_out, "Pluvicto_pharmacodynamics"))
    death_pair_bundle = _filter_prediction_bundle_to_samples(death_bundle, pair_df["death_sample"].tolist())
    c1_pair_bundle = _filter_prediction_bundle_to_samples(c1_bundle, pair_df["c1_sample"].tolist())
    combined_bundle = _concatenate_prediction_bundles_by_columns(c1_pair_bundle, death_pair_bundle)
    combined_pred = combined_bundle.get("pred_wide")
    if combined_pred is None or combined_pred.empty:
        print("    [skip] CRPC-RA Pluvicto pharmacodynamics: no combined paired prediction matrix available")
        return

    combined_rows: List[Dict[str, Any]] = []
    for patient_id, row in pair_df.iterrows():
        combined_rows.extend(
            [
                {
                    "sample": str(row["c1_sample"]),
                    "patient_id": patient_id,
                    "visit": "C1",
                    "subtype": "C1",
                    "tfx": row.get("c1_tfx", np.nan),
                },
                {
                    "sample": str(row["death_sample"]),
                    "patient_id": patient_id,
                    "visit": "Death",
                    "subtype": "Death",
                    "tfx": row.get("death_tfx", np.nan),
                },
            ]
        )
    combined_meta = pd.DataFrame(combined_rows).set_index("sample")
    combined_meta.to_csv(os.path.join(pharmacodynamics_out, "paired_sample_metadata.tsv"), sep="\t")
    _run_weighted_limma_suite(
        combined_pred,
        combined_meta[["subtype"]],
        pharmacodynamics_out,
        overwrite,
        bundle=combined_bundle,
        contrast="Death-C1",
    )


def _is_ne_like_subtype(cohort: str, subtype: Any) -> bool:
    """Map cohort-specific subtype labels to the requested NE-like labels."""
    subtype_str = str(subtype).strip().upper()
    if cohort in {"CRPC-RA", "CRPC-WCDT"}:
        return ("NEPC" in subtype_str) or ("AMPC" in subtype_str)
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


def _ci_center_score_matrix(
    ci_lo: Optional[pd.DataFrame],
    ci_hi: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Return the midpoint of aligned q5/q95 score interval surfaces."""
    if ci_lo is None or ci_hi is None:
        return None

    common_index = ci_lo.index.intersection(ci_hi.index)
    common_columns = ci_lo.columns.intersection(ci_hi.columns)
    if len(common_index) == 0 or len(common_columns) == 0:
        return None
    return 0.5 * (
        ci_lo.loc[common_index, common_columns].astype(float)
        + ci_hi.loc[common_index, common_columns].astype(float)
    )


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
    calibration_state: Optional[Dict[str, Any]] = None,
    run_pam50_crosscohort: bool = False,
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
        score_ci_lo_parts: List[pd.DataFrame] = []
        score_ci_hi_parts: List[pd.DataFrame] = []

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
                print(f"    [warn] Shortlist ssGSEA score intervals unavailable for {cd['cohort']}")
                continue

            print(f"    Shortlist ssGSEA MC: {cd['cohort']} ({len(common_samples)} samples)")
            if calibration_state is not None:
                mc_draws = _get_calibrated_draws(
                    cd,
                    calibration_state,
                    source_by_sample=(
                        cd.get("calibration_source_by_sample")
                        or _constant_source_by_sample(common_samples, cd["cohort"])
                    ),
                    sample_fold=cd.get("calibration_sample_fold"),
                    M=args.montecarlo_draws,
                    seed=_CALIBRATION_DRAW_SEED,
                )
                mc_draws = [draw[common_samples] for draw in mc_draws]
            else:
                mc_draws = mdn_monte_carlo_sample(
                    cd["p0_wide"][common_samples],
                    cd["pi_gamma_wide"][common_samples],
                    cd["pi_normal_wide"][common_samples],
                    cd["gamma_beta_wide"][common_samples],
                    cd["normal_mean_wide"][common_samples],
                    cd["normal_stdev_wide"][common_samples],
                    M=args.montecarlo_draws,
                    seed=_CALIBRATION_DRAW_SEED,
                )
            _, ci_lo_scores, ci_hi_scores = _aggregate_ssgsea_score_matrices(
                mc_draws, shortlist_sets, min_size=3)
            if ci_lo_scores is not None and not ci_lo_scores.empty:
                score_ci_lo_parts.append(ci_lo_scores)
            if ci_hi_scores is not None and not ci_hi_scores.empty:
                score_ci_hi_parts.append(ci_hi_scores)

        truth_scores_all = _concat_score_matrices(truth_score_parts)
        raw_scores_all = _concat_score_matrices(raw_score_parts)
        score_ci_lo_all = _concat_score_matrices(score_ci_lo_parts)
        score_ci_hi_all = _concat_score_matrices(score_ci_hi_parts)
        centered_scores_all = _ci_center_score_matrix(score_ci_lo_all, score_ci_hi_all)

        if truth_scores_all is not None:
            truth_scores_all.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_truth.tsv"), sep="\t")
        if raw_scores_all is not None:
            raw_scores_all.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_raw.tsv"), sep="\t")
        if centered_scores_all is not None:
            centered_scores_all.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_ci_centered.tsv"), sep="\t")
        if score_ci_lo_all is not None:
            score_ci_lo_all.to_csv(
                os.path.join(shortlist_out, "shortlist_score_ci_q5.tsv"), sep="\t")
        if score_ci_hi_all is not None:
            score_ci_hi_all.to_csv(
                os.path.join(shortlist_out, "shortlist_score_ci_q95.tsv"), sep="\t")

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
                    pred_ci_lo=score_ci_lo_all.loc[set_name] if score_ci_lo_all is not None and set_name in score_ci_lo_all.index else None,
                    pred_ci_hi=score_ci_hi_all.loc[set_name] if score_ci_hi_all is not None and set_name in score_ci_hi_all.index else None,
                )
                if centered_scores_all is not None and set_name in centered_scores_all.index:
                    _maybe(
                        _crosscohort_score_scatter,
                        args.overwrite,
                        truth_scores_all.loc[set_name],
                        centered_scores_all.loc[set_name],
                        collected_subtypes,
                        os.path.join(scatter_out, f"{set_name}_ci_centered"),
                        title=f"{label} — {set_name} ssGSEA (Truth vs Predicted CI-centered)",
                        y_label=f"Predicted {set_name} ssGSEA Score (CI center)",
                        pred_ci_lo=score_ci_lo_all.loc[set_name] if score_ci_lo_all is not None and set_name in score_ci_lo_all.index else None,
                        pred_ci_hi=score_ci_hi_all.loc[set_name] if score_ci_hi_all is not None and set_name in score_ci_hi_all.index else None,
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

    if run_pam50_crosscohort:
        _run_pam50_analysis(
            combined_pred,
            collected_out,
            args.overwrite,
            cohort_label=label,
            truth_wide=combined_truth,
            emit_truth_vs_pred_confusion=False,
            sample_meta=collected_subtypes,
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

# GMT files for GSEA
_GSEA_HALLMARK_GMT = PROJECT_ROOT / "reference" / "h.all.v2026.1.Hs.symbols.gmt"
_GSEA_SHORTLIST_GMT = PROJECT_ROOT / "reference" / "gsva_genesets_shortlist.gmt"
_GSEA_PRC2_GMT = PROJECT_ROOT / "reference" / "REACTOME_PRC2_METHYLATES_HISTONES_AND_DNA.v2026.1.Hs.gmt"
_GSEA_SHORTLIST_SETS = {"ARG.10", "NE.10", "CCP.31"}
_ENTREZ_SYMBOL_OVERRIDES: Dict[str, str] = {
    "25859": "PART1",
}
_PFIZER_VISIT_ORDER = ["SCN", "C3D1", "EOT"]
_PFIZER_RESPONSE_COLOR_MAP: Dict[str, str] = {
    "Progression": "#B63A3A",
    "Disease Control / Non-Progression": "#2C7A4B",
}
_PFIZER_RESPONSE_4CLASS_ORDER = ["PD", "Non-CR/PD", "SD", "PR & PSA50"]
_PFIZER_RESPONSE_4CLASS_COLOR_MAP: Dict[str, str] = {
    "PD": "#B63A3A",
    "Non-CR/PD": "#C7892F",
    "SD": "#2C7A4B",
    "PR & PSA50": "#225EA8",
}
_PFIZER_PROGRESSION_ORDER = ["PD", "non-PD"]
_PFIZER_PROGRESSION_COLOR_MAP: Dict[str, str] = {
    "PD": "#B63A3A",
    "non-PD": "#2C7A4B",
}
_PFIZER_ARM_ORDER = ["Enza-only", "EZH2i+Enza"]
_PFIZER_ARM_COLOR_MAP: Dict[str, str] = {
    "Enza-only": "#7F3C8D",
    "EZH2i+Enza": "#11A579",
}
_PFIZER_DOSE_ORDER = [
    "Enza",
    "500mg+Enza",
    "500mg+Enza with food",
    "625mg+Enza",
    "875mg+Enza",
    "1250mg+Enza",
]
_PFIZER_EFFECTIVE_DOSE_ORDER = ["<625 mg / Enza-only", ">=625 mg + Enza"]
_PFIZER_EFFECTIVE_DOSE_COLOR_MAP: Dict[str, str] = {
    "<625 mg / Enza-only": "#7A7F86",
    ">=625 mg + Enza": "#1F6AA5",
}
_PFIZER_ENZA_HISTORY_ORDER = ["Enza Experienced", "Enza Naive"]
_PFIZER_FED_STATUS_ORDER = ["Fasted", "Fed"]
_PFIZER_CLINICAL_BENEFIT_STRICT_ORDER = ["non-benefit", "indeterminate", "benefit"]
_PFIZER_CLINICAL_BENEFIT_BROAD_ORDER = ["progression", "benefit_or_nonprogression"]
_PLUVICTO_DISCOVERY_VARIANTS = {"CRPC-Pluvicto_MC", "CRPC-Pluvicto_Full"}
_PFIZER_VISIT_MARKER_MAP: Dict[str, str] = {
    "SCN": "o",
    "C3D1": "s",
    "EOT": "D",
}
_PFIZER_SHORTLIST_LABEL_MAP: Dict[str, str] = {
    "ARG.10": "AR score",
    "NE.10": "NE score",
    "CCP.31": "CCP score",
}
_PFIZER_PRC2_SCORE_NAME = "PRC2 score"
_PFIZER_BASELINE_HALLMARKS: List[str] = [
    "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_MYC_TARGETS_V2",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_HYPOXIA",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
]
_PFIZER_LONGITUDINAL_SCORE_FEATURES: List[str] = [
    "AR score",
    "NE score",
    "CCP score",
    _PFIZER_PRC2_SCORE_NAME,
    *_PFIZER_BASELINE_HALLMARKS,
]
_PFIZER_BASELINE_EXPRESSION_GENES: List[str] = [
    "EZH2",
    "EED",
    "SUZ12",
    "AR",
    "KLK3",
    "KLK2",
    "FKBP5",
    "NKX3-1",
    "TMPRSS2",
    "HOXB13",
    "ASCL1",
    "NEUROD1",
    "INSM1",
    "SYP",
    "CHGA",
    "CHGB",
    "ENO2",
    "NCAM1",
    "SOX2",
    "MYCN",
    "AURKA",
    "MKI67",
    "TOP2A",
    "BIRC5",
    "CDK1",
    "CCNB1",
    "CCNB2",
    "MCM2",
    "MCM3",
    "MCM4",
    "MCM5",
    "MCM6",
    "MCM7",
]
_PFIZER_PRESPECIFIED_BASELINE_FEATURES: List[str] = [
    "AR score",
    "NE score",
    "CCP score",
    "PRC2 target derepression score (exploratory)",
    "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_MYC_TARGETS_V2",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_HYPOXIA",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
]
_PFIZER_C3D1_CORE_FEATURES: List[str] = [
    "EZH2 Repression Target Score",
    "EZH2 activity score",
    "AR score",
    "NE score",
    "CCP score",
    "PRC2 target derepression score (exploratory)",
    "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_MYC_TARGETS_V2",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_HYPOXIA",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_INFLAMMATORY_RESPONSE",
]
_PFIZER_COMPOSITE_COMPONENTS: List[Tuple[str, float]] = [
    ("PRC2 target derepression score (exploratory)", 1.0),
    ("AR score", -1.0),
    ("CCP score", -1.0),
    ("HALLMARK_E2F_TARGETS", -1.0),
    ("HALLMARK_G2M_CHECKPOINT", -1.0),
    ("HALLMARK_MYC_TARGETS_V1", -1.0),
    ("NE score", -1.0),
    ("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION", -1.0),
    ("HALLMARK_HYPOXIA", -1.0),
]
_PFIZER_COMPOSITE_COMPONENTS_NOPRC2: List[Tuple[str, float]] = [
    component for component in _PFIZER_COMPOSITE_COMPONENTS if component[0] != "PRC2 target derepression score (exploratory)"
]
_PFIZER_AR_GENE_PANEL: List[str] = [
    "AR",
    "KLK3",
    "KLK2",
    "FKBP5",
    "NKX3-1",
    "TMPRSS2",
    "HOXB13",
    "ACPP",
    "STEAP1",
    "STEAP2",
]
_PFIZER_NE_GENE_PANEL: List[str] = [
    "ASCL1",
    "NEUROD1",
    "INSM1",
    "SYP",
    "CHGA",
    "CHGB",
    "ENO2",
    "NCAM1",
    "SOX2",
    "MYCN",
    "AURKA",
]
_PFIZER_CELL_CYCLE_GENE_PANEL: List[str] = [
    "MKI67",
    "TOP2A",
    "BIRC5",
    "CDK1",
    "CCNB1",
    "CCNB2",
    "MCM2",
    "MCM3",
    "MCM4",
    "MCM5",
    "MCM6",
    "MCM7",
    "PLK1",
    "AURKB",
]
_PFIZER_PRC2_GENE_PANEL: List[str] = [
    "EZH2",
    "EED",
    "SUZ12",
    "RBBP4",
    "RBBP7",
    "KDM6A",
    "KDM6B",
]
_PFIZER_GENE_LEVEL_PAIRED_PLOT_GENES: List[str] = [
    "KLK3",
    "FKBP5",
    "AR",
    "MKI67",
    "TOP2A",
    "BIRC5",
    "ASCL1",
    "INSM1",
    "SYP",
    "EZH2",
]
_PFIZER_EZH2_REPRESSION_TARGET_GENES: List[str] = [
    "DAB2IP",
    "MSMB",
    "SLIT2",
    "TIMP2",
    "TIMP3",
    "ID4",
    "PEBP1",
    "CDKN2A",
    "CDH1",
    "RUNX3",
]
_PFIZER_EZH2_ACTIVITY_GENES: List[str] = ["EZH2", "EED", "SUZ12"]
_PFIZER_PRIMARY_PD_FEATURE = "EZH2 Repression Target Score"
_PFIZER_EZH2_ACTIVITY_SCORE = "EZH2 activity score"
_PFIZER_SECONDARY_PD_FEATURES: List[str] = [
    _PFIZER_EZH2_ACTIVITY_SCORE,
    "AR score",
    "HALLMARK_ANDROGEN_RESPONSE",
    "CCP score",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_MYC_TARGETS_V1",
    "NE score",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_HYPOXIA",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
]
_PFIZER_EXPLORATORY_PRC2_TARGET_GENES: List[str] = [
    "ATF3",
    "BTG2",
    "CDKN1A",
    "DAB2IP",
    "EHF",
    "GATA6",
    "ICAM1",
    "IGFBP3",
    "KLF4",
    "RARRES3",
    "RUNX3",
    "SLIT2",
    "SOX17",
    "TGFBR2",
]


def _run_gsea(
    de_results: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    contrast: str = "NEPC-ARPC",
    tag: str = "",
    extra_gene_sets: Optional[Dict[str, List[str]]] = None,
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
    _require_existing_path(_GSEA_HALLMARK_GMT, description="Hallmark GMT")
    hallmark_sets = _load_gmt(_GSEA_HALLMARK_GMT, description="Hallmark GMT")
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

    # --- (2) CRPC shortlist (Entrez → Symbol) ---
    _require_existing_path(_GSEA_SHORTLIST_GMT, description="shortlist GMT")
    raw_sets = _load_gmt(_GSEA_SHORTLIST_GMT, description="shortlist GMT")
    # Filter to requested sets
    raw_sets = {k: v for k, v in raw_sets.items() if k in _GSEA_SHORTLIST_SETS}
    if raw_sets:
        converted_sets = _convert_entrez_gmt(
            raw_sets,
            mane_reference_path=PROJECT_ROOT / "reference" / "MANE.GRCh38.v1.3.UniversalReference.tsv",
            overrides=_ENTREZ_SYMBOL_OVERRIDES,
        )
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

    if extra_gene_sets:
        filtered_extra_sets = {
            set_name: list(dict.fromkeys(genes))
            for set_name, genes in extra_gene_sets.items()
            if genes
        }
        if filtered_extra_sets:
            custom_out = ensure_dir(os.path.join(gsea_out, "Custom"))
            custom_tsv = os.path.join(custom_out, f"gsea_custom_results{tag}.tsv")

            if overwrite or not os.path.exists(custom_tsv):
                res_custom = gp.prerank(
                    rnk=rank_series,
                    gene_sets=filtered_extra_sets,
                    outdir=custom_out,
                    min_size=3,
                    max_size=500,
                    permutation_num=1000,
                    seed=42,
                    no_plot=True,
                    verbose=False,
                )
                res_custom_df = res_custom.res2d
                res_custom_df.to_csv(custom_tsv, sep="\t", index=False)
                print("    Custom-signature GSEA results saved")
            else:
                res_custom_df = pd.read_csv(custom_tsv, sep="\t")

            _maybe(
                enrichment_dot_plot,
                overwrite,
                res_custom_df,
                os.path.join(custom_out, "enrichment_dotplot_custom"),
                contrast_name=contrast,
                title="GSEA: custom pharmacodynamic signatures",
            )


def _run_limma_de(
    pred_wide: pd.DataFrame,
    subtypes_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    contrast: str = "NEPC-ARPC",
    padj_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    tag: str = "",
    weights_wide: Optional[pd.DataFrame] = None,
    apply_expression_filter: bool = True,
) -> Optional[pd.DataFrame]:
    """Run subtype-only weighted limma-trend DE on one matrix via rpy2."""
    prepared = _prepare_limma_inputs(
        pred_wide,
        subtypes_df,
        contrast,
        weights_wide=weights_wide,
        apply_expression_filter=apply_expression_filter,
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
        weights=prepared["weights"],
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


_LIMMA_WEIGHT_CLIP_RANGES: Tuple[Tuple[str, float, float], ...] = (
    ("clip0p50to2p00", 0.50, 2.00),
    ("clip0p33to3p00", 0.33, 3.00),
)


def _normalize_precision_weight_matrix(
    raw_weights_wide: Optional[pd.DataFrame],
    *,
    weight_floor: Optional[float] = None,
    weight_ceiling: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Normalize a raw precision proxy into comparable limma observation weights.

    Limma uses weights on a relative scale, so we keep every weighted scheme on
    the same global footing by centering to a mean of 1 before any optional
    clipping and then centering again afterward. The first centering makes the
    requested clip window comparable across weighting formulas, while the second
    preserves the overall average leverage so weighted and unweighted fits
    remain directly comparable at the global scale.
    """
    if raw_weights_wide is None or raw_weights_wide.empty:
        return None

    weights = raw_weights_wide.apply(pd.to_numeric, errors="coerce")
    weight_values = weights.to_numpy(dtype=np.float64, copy=True)
    valid = np.isfinite(weight_values) & (weight_values > 0.0)
    if not bool(valid.any()):
        return None

    initial_mean = float(np.mean(weight_values[valid]))
    if not np.isfinite(initial_mean) or initial_mean <= 0.0:
        return None
    weight_values[valid] = weight_values[valid] / initial_mean
    if weight_floor is not None and weight_ceiling is not None:
        weight_values[valid] = np.clip(weight_values[valid], float(weight_floor), float(weight_ceiling))

    final_mean = float(np.mean(weight_values[valid]))
    if not np.isfinite(final_mean) or final_mean <= 0.0:
        return None
    weight_values[valid] = weight_values[valid] / final_mean
    return pd.DataFrame(weight_values, index=weights.index, columns=weights.columns)


def _raw_precision_weights_from_ci_width(
    ci_lo: Optional[pd.DataFrame],
    ci_hi: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Map CI90 width to an inverse-variance-style raw precision proxy.

    For a fixed central interval family, width squared is proportional to the
    corresponding variance up to a constant factor. Using ``(1 / CI90)^2`` keeps
    the weighting rule faithful to the requested inverse-variance logic while
    letting the shared normalization step control the final leverage range.
    """
    if ci_lo is None or ci_hi is None:
        return None
    common_genes = ci_lo.index.intersection(ci_hi.index)
    common_samples = ci_lo.columns.intersection(ci_hi.columns)
    if len(common_genes) == 0 or len(common_samples) == 0:
        return None
    width = ci_hi.loc[common_genes, common_samples].astype(float) - ci_lo.loc[common_genes, common_samples].astype(float)
    width_values = width.to_numpy(dtype=np.float64, copy=True)
    raw_values = np.full(width_values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(width_values) & (width_values > 0.0)
    if not bool(valid.any()):
        return None
    stabilized_width = np.maximum(width_values[valid], 1e-6)
    raw_values[valid] = np.square(1.0 / stabilized_width)
    return pd.DataFrame(raw_values, index=width.index, columns=width.columns)


def _run_weighted_limma_suite(
    expr_wide: pd.DataFrame,
    subtypes_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    *,
    bundle: Dict[str, Any],
    contrast: str = "NEPC-ARPC",
    extra_gene_sets: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Run mean-prediction limma/GSEA across the shared weight modes.

    All predicted-expression limma fits now use the same subtype-only design and
    the same observation-level precision-weight contract across cohorts. The raw
    CI90 precision proxy is computed first, then the weighted modes are either
    normalized directly or normalized, clipped, and renormalized under each
    requested leverage band so the output files are directly comparable across
    datasets.
    """
    print("    limma design: subtype-only contrast; TFx is not included as a covariate")
    print("    limma empirical Bayes: trend=TRUE, robust=TRUE")
    print(
        "    limma precision weights: calibrated uncertainty surfaces"
        if bool(bundle.get("distribution_calibration_applied"))
        else "    limma precision weights: raw uncertainty surfaces"
    )

    raw_ci90_weights = _raw_precision_weights_from_ci_width(bundle.get("ci_lo_wide"), bundle.get("ci_hi_wide"))

    de_expr = _run_limma_de(
        expr_wide,
        subtypes_df,
        out_dir,
        overwrite,
        contrast=contrast,
        tag="_mean_unweighted",
        weights_wide=None,
        apply_expression_filter=True,
    )
    if de_expr is not None:
        _run_gsea(
            de_expr,
            out_dir,
            overwrite,
            contrast=contrast,
            tag="_mean_unweighted",
            extra_gene_sets=extra_gene_sets,
        )

    print("    limma weight normalization: mean-1 (unclipped)")
    unclipped_weights = _normalize_precision_weight_matrix(raw_ci90_weights)
    if unclipped_weights is None:
        print("    [skip] limma weighted_ci90_unclipped: precision weights unavailable")
    else:
        tag = "_mean_weighted_ci90_unclipped"
        de_expr = _run_limma_de(
            expr_wide,
            subtypes_df,
            out_dir,
            overwrite,
            contrast=contrast,
            tag=tag,
            weights_wide=unclipped_weights,
            apply_expression_filter=True,
        )
        if de_expr is not None:
            _run_gsea(
                de_expr,
                out_dir,
                overwrite,
                contrast=contrast,
                tag=tag,
                extra_gene_sets=extra_gene_sets,
            )

    for range_label, weight_floor, weight_ceiling in _LIMMA_WEIGHT_CLIP_RANGES:
        print(
            f"    limma weight normalization: mean-1 -> clip [{weight_floor:.2f}, {weight_ceiling:.2f}] -> mean-1"
        )
        weights = _normalize_precision_weight_matrix(
            raw_ci90_weights,
            weight_floor=weight_floor,
            weight_ceiling=weight_ceiling,
        )
        if weights is None:
            print(f"    [skip] limma weighted_ci90 ({range_label}): precision weights unavailable")
            continue
        tag = f"_mean_weighted_ci90_{range_label}"
        de_expr = _run_limma_de(
            expr_wide,
            subtypes_df,
            out_dir,
            overwrite,
            contrast=contrast,
            tag=tag,
            weights_wide=weights,
            apply_expression_filter=True,
        )
        if de_expr is not None:
            _run_gsea(
                de_expr,
                out_dir,
                overwrite,
                contrast=contrast,
                tag=tag,
                extra_gene_sets=extra_gene_sets,
            )


def _prepare_limma_inputs(
    pred_wide: pd.DataFrame,
    subtypes_df: pd.DataFrame,
    contrast: str,
    *,
    weights_wide: Optional[pd.DataFrame] = None,
    apply_expression_filter: bool = True,
) -> Optional[Dict[str, Any]]:
    """Filter and align limma inputs and optional observation weights.

    When enabled, the shared limma expression filter drops genes with any NA in
    the aligned matrix and then retains only genes with log2(CPM+1) > 1.0 in at
    least 3 samples.
    """
    groups_in_contrast = contrast.replace(" ", "").split("-")
    if len(groups_in_contrast) != 2:
        raise ValueError(f"Contrast must be 'GroupA-GroupB', got: {contrast}")
    required_groups = set(groups_in_contrast)

    valid = subtypes_df[subtypes_df["subtype"].isin(required_groups)]
    common = [sample for sample in pred_wide.columns if sample in valid.index]

    if len(common) < 3:
        print(f"    [skip] limma DE: only {len(common)} samples in {required_groups}")
        return None

    expr = pred_wide[common]
    n_total = expr.shape[0]

    na_mask = expr.isna().any(axis=1)
    n_na = int(na_mask.sum())
    expr = expr.loc[~na_mask]

    expressed_mask = (expr > 1.0).sum(axis=1) >= 3 if apply_expression_filter else pd.Series(True, index=expr.index)
    n_low = int((~expressed_mask).sum())
    expr = expr.loc[expressed_mask]
    weights = None
    if weights_wide is not None:
        weights = weights_wide.reindex(index=expr.index, columns=common).apply(pd.to_numeric, errors="coerce")
        invalid_weights = ~np.isfinite(weights.values) | (weights.values <= 0.0)
        if bool(invalid_weights.any()):
            invalid_gene_mask = pd.Series(invalid_weights.any(axis=1), index=weights.index)
            n_invalid_weight_genes = int(invalid_gene_mask.sum())
            expr = expr.loc[~invalid_gene_mask]
            weights = weights.loc[expr.index]
            print(f"    limma DE: dropped {n_invalid_weight_genes} genes with invalid precision weights")

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
        "weights": weights,
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
    weights: Optional[pd.DataFrame] = None,
    bridge: Optional[Tuple[Any, Any, Any]] = None,
) -> pd.DataFrame:
    """Execute one limma differential-expression fit through the R bridge."""
    ro, pandas2ri, localconverter = _load_limma_bridge() if bridge is None else bridge
    converter = ro.default_converter + pandas2ri.converter
    with localconverter(converter):
        r_expr = ro.conversion.get_conversion().py2rpy(expr)
        r_groups = ro.StrVector(list(groups))
        r_weights = ro.NULL
        if weights is not None:
            r_weights = ro.conversion.get_conversion().py2rpy(weights)
        de_r = ro.globalenv["run_limma_de"](r_expr, r_groups, contrast, r_weights)
        de_results = ro.conversion.get_conversion().rpy2py(de_r)
    return pd.DataFrame(de_results)


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
    score_columns = [column for column in score_columns if not str(column).endswith("_Normal")]
    corr_cols = sorted(column for column in score_columns if column.startswith("corr_"))
    if corr_cols:
        return corr_cols
    return sorted(column for column in score_columns if column.startswith("proba_"))


def _filter_pam50_normal(pred_indexed: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Report genefu Normal calls without dropping patients from saved score tables."""
    if "predicted_subtype" not in pred_indexed.columns:
        return pred_indexed
    normal_mask = pred_indexed["predicted_subtype"].astype(str).str.casefold() == "normal"
    n_normal = int(normal_mask.sum())
    if n_normal:
        print(f"    PAM50 {context}: flagged {n_normal} Normal calls; excluding them only from subtype-based comparisons")
        _print_sample_name_list(
            pred_indexed.index[normal_mask].tolist(),
            prefix="      Normal-call samples: ",
        )
    return pred_indexed.copy()


def _plot_pam50_score_scatter(
    truth_scores: pd.DataFrame,
    pred_scores: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    sample_meta: Optional[pd.DataFrame] = None,
    color_column: str = "cohort",
    legend_title: str = "Cohort",
) -> None:
    """Plot one truth-vs-predicted PAM50 score scatter per subtype score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score_columns = [column for column in truth_scores.columns if column in pred_scores.columns]
    if not score_columns:
        return

    palette = load_palette()
    for column in score_columns:
        truth_values = pd.to_numeric(truth_scores[column], errors="coerce")
        pred_values = pd.to_numeric(pred_scores[column], errors="coerce")
        valid = truth_values.notna() & pred_values.notna()
        subtype_name = column.split("_", 1)[1] if "_" in column else column
        if int(valid.sum()) < 1:
            continue

        fig, axis = plt.subplots(figsize=(5.4, 5.4))
        valid_samples = truth_values.index[valid]
        seen_groups: Dict[str, str] = {}
        if sample_meta is not None:
            valid_samples = valid_samples.intersection(sample_meta.index)
        for sample in valid_samples:
            if sample_meta is not None:
                group_name = str(sample_meta.loc[sample, color_column]) if color_column in sample_meta.columns else "Unknown"
                status = str(sample_meta.loc[sample, "status"]) if "status" in sample_meta.columns else "validation"
                tfx_val = pd.to_numeric(pd.Series([sample_meta.loc[sample, "tfx"]]), errors="coerce").iloc[0] if "tfx" in sample_meta.columns else np.nan
                color = palette.get(group_name, "#999999")
                marker = "D" if status == "holdout" else "o"
                size = max(40.0, 260.0 * float(tfx_val)) if np.isfinite(tfx_val) else 60.0
                seen_groups[group_name] = color
            else:
                color = "#3A6EA5"
                marker = "o"
                size = 44.0
            axis.scatter(
                float(truth_values.loc[sample]),
                float(pred_values.loc[sample]),
                s=size,
                alpha=0.82,
                color=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.45,
            )
        if int(valid.sum()) >= 2:
            xy_min = float(min(truth_values.loc[valid].min(), pred_values.loc[valid].min()))
            xy_max = float(max(truth_values.loc[valid].max(), pred_values.loc[valid].max()))
            if not np.isclose(xy_min, xy_max):
                pad = 0.05 * max(xy_max - xy_min, 1e-6)
                axis.plot([xy_min - pad, xy_max + pad], [xy_min - pad, xy_max + pad], linestyle="--", color="#7A7A7A", linewidth=1.0)
                axis.set_xlim(xy_min - pad, xy_max + pad)
                axis.set_ylim(xy_min - pad, xy_max + pad)

        pearson_r = np.nan
        spearman_rho = np.nan
        if int(valid.sum()) >= MIN_CORRELATION_N:
            _, pearson_r, _, spearman_rho, _ = _truth_pred_correlation_values(
                truth_values.loc[valid].values,
                pred_values.loc[valid].values,
            )

        axis.set_title(f"{title}: {subtype_name}")
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
        if seen_groups:
            handles = [
                plt.Line2D(
                    [0], [0], marker="o", color="w", label=group_name,
                    markerfacecolor=color, markeredgecolor="black", markersize=8,
                )
                for group_name, color in sorted(seen_groups.items())
            ]
            axis.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
        axis.set_box_aspect(1)
        fig.tight_layout()
        fig.savefig(f"{out_path}_{_analysis_safe_name(subtype_name)}.pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _run_pam50_analysis(
    pred_wide: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    *,
    cohort_label: str,
    tag: str = "",
    truth_wide: Optional[pd.DataFrame] = None,
    reference_subtypes: Optional[pd.Series] = None,
    reference_label: Optional[str] = None,
    emit_truth_vs_pred_confusion: bool = True,
    sample_meta: Optional[pd.DataFrame] = None,
) -> None:
    """Run PAM50 classification and score comparisons for a cohort."""
    pred_for_scores = _prepare_expression_matrix_for_scores(
        pred_wide,
        context=f"PAM50 predicted {cohort_label}",
    )
    if pred_for_scores.empty:
        print(f"    [skip] PAM50 {cohort_label}: no scored genes available")
        return

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
    missing_classification_samples = pred_indexed.index[pred_indexed["predicted_subtype"].isna()].astype(str).tolist()
    pred_indexed = _filter_pam50_normal(pred_indexed, context=cohort_label)
    reason_by_sample = {
        sample_name: "no genefu classification"
        for sample_name in missing_classification_samples
    }
    _report_sample_drops(
        f"PAM50 {cohort_label} predicted classification",
        pred_for_scores.columns.tolist(),
        pred_indexed.index.tolist(),
        reason_by_sample=reason_by_sample,
    )
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
                if int(valid.sum()) >= MIN_CORRELATION_N:
                    truth_valid = truth_values.loc[valid]
                    pred_valid = pred_values.loc[valid]
                    _, row["pearson_r"], _, row["spearman_rho"], _ = _truth_pred_correlation_values(
                        truth_valid.values,
                        pred_valid.values,
                    )
                    row["mae"] = float((truth_valid - pred_valid).abs().mean())
                    flat_truth.append(truth_valid.to_numpy(dtype=float))
                    flat_pred.append(pred_valid.to_numpy(dtype=float))
                score_rows.append(row)

            if flat_truth and flat_pred:
                flat_truth_values = np.concatenate(flat_truth)
                flat_pred_values = np.concatenate(flat_pred)
                if flat_truth_values.size >= MIN_CORRELATION_N and flat_pred_values.size >= MIN_CORRELATION_N:
                    _, all_pearson_r, _, all_spearman_rho, _ = _truth_pred_correlation_values(
                        flat_truth_values,
                        flat_pred_values,
                    )
                    score_rows.append({
                        "score_type": score_type,
                        "score_column": "all_scores",
                        "subtype": "All",
                        "n_samples": int(flat_truth_values.size),
                        "pearson_r": all_pearson_r,
                        "spearman_rho": all_spearman_rho,
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
                sample_meta=sample_meta,
            )

        if not emit_truth_vs_pred_confusion:
            print(f"    PAM50 {cohort_label}: saved pooled truth-vs-predicted subtype score comparisons")
            return

        comparison_df = pd.DataFrame({
            "sample": matched_samples,
            "truth_subtype": truth_common["predicted_subtype"].values,
            "predicted_subtype": pred_common["predicted_subtype"].values,
        }).dropna()
        comparison_df = comparison_df.loc[
            (comparison_df["truth_subtype"].astype(str).str.casefold() != "normal")
            & (comparison_df["predicted_subtype"].astype(str).str.casefold() != "normal")
        ]
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

        class_order = [
            subtype for subtype in sorted(set(comparison_df["truth_subtype"]) | set(comparison_df["predicted_subtype"]))
            if str(subtype).casefold() != "normal"
        ]
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
    comparison_df = comparison_df.loc[
        (comparison_df["reference_subtype"].astype(str).str.casefold() != "normal")
        & (comparison_df["predicted_subtype"].astype(str).str.casefold() != "normal")
    ]
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

    class_order = [
        subtype for subtype in sorted(set(comparison_df["reference_subtype"]) | set(comparison_df["predicted_subtype"]))
        if str(subtype).casefold() != "normal"
    ]
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
    activity_odds_ratio_wide: Optional[pd.DataFrame],
    model_out: str,
    overwrite: bool,
) -> None:
    """Receptor-gene expression and AOR boxplots for BC-Ghana."""
    meta_path = _require_existing_path(
        PROJECT_ROOT / "reference" / "BC-Ghana_metadata.tsv",
        description="BC-Ghana metadata table",
    )

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

        if activity_odds_ratio_wide is None or gene not in activity_odds_ratio_wide.index:
            continue
        aor_samples = [sample for sample in common_samples if sample in activity_odds_ratio_wide.columns]
        if len(aor_samples) < 3:
            continue
        aor_vals = activity_odds_ratio_wide.loc[gene, aor_samples].astype(float)
        aor_status_vals = meta.loc[aor_samples, status_col].astype(int)
        aor_vals_neg = aor_vals[aor_status_vals == 0].values
        aor_vals_pos = aor_vals[aor_status_vals == 1].values
        if len(aor_vals_neg) >= 2 and len(aor_vals_pos) >= 2:
            _, aor_mwu_p = mannwhitneyu(aor_vals_neg, aor_vals_pos, alternative="two-sided")
        else:
            aor_mwu_p = None
        _maybe(
            binary_status_boxplot,
            overwrite,
            aor_vals_neg,
            aor_vals_pos,
            os.path.join(receptor_out, f"{receptor_name}_status_vs_{gene}_aor"),
            xlabel=f"{receptor_name} IHC status",
            ylabel=f"{gene} activity odds ratio",
            title=f"{gene} AOR by {receptor_name} status — BC-Ghana",
            pval=aor_mwu_p,
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

    _require_existing_path(_GSEA_HALLMARK_GMT, description="Hallmark GMT")

    ssgsea_out = ensure_dir(os.path.join(model_out, f"ssGSEA{tag}"))
    scores_tsv = os.path.join(ssgsea_out, f"ssgsea_hallmark_scores{tag}.tsv")

    if overwrite or not os.path.exists(scores_tsv):
        hallmark_sets = _load_gmt(_GSEA_HALLMARK_GMT, description="Hallmark GMT")
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
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Run ssGSEA on MC draws and save q5/q95 score intervals."""
    import gseapy as gp

    _require_existing_path(_GSEA_HALLMARK_GMT, description="Hallmark GMT")

    ssgsea_out = ensure_dir(os.path.join(model_out, "ssGSEA_score_intervals"))
    ci_lo_tsv = os.path.join(ssgsea_out, "ssgsea_hallmark_score_ci_q5.tsv")
    ci_hi_tsv = os.path.join(ssgsea_out, "ssgsea_hallmark_score_ci_q95.tsv")

    if not overwrite and os.path.exists(ci_lo_tsv) and os.path.exists(ci_hi_tsv):
        return (
            pd.read_csv(ci_lo_tsv, sep="\t", index_col=0),
            pd.read_csv(ci_hi_tsv, sep="\t", index_col=0),
        )

    hallmark_sets = _load_gmt(_GSEA_HALLMARK_GMT, description="Hallmark GMT")
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
        return None, None

    # Stack and compute statistics
    stacked = np.stack([s.values for s in all_scores], axis=0)  # (M, P, S)
    ci_lo = pd.DataFrame(
        np.percentile(stacked, 5.0, axis=0),
        index=all_scores[0].index, columns=all_scores[0].columns)
    ci_hi = pd.DataFrame(
        np.percentile(stacked, 95.0, axis=0),
        index=all_scores[0].index, columns=all_scores[0].columns)

    ci_lo.to_csv(ci_lo_tsv, sep="\t")
    ci_hi.to_csv(ci_hi_tsv, sep="\t")
    print("    MC ssGSEA score intervals saved")

    return ci_lo, ci_hi


def _load_shortlist_sets() -> Dict[str, List[str]]:
    """Load the configured shortlist sets used for per-sample scoring."""
    _require_existing_path(_GSEA_SHORTLIST_GMT, description="shortlist GMT")
    raw_sets = _load_gmt(_GSEA_SHORTLIST_GMT, description="shortlist GMT")
    raw_sets = {k: v for k, v in raw_sets.items() if k in _GSEA_SHORTLIST_SETS}
    if not raw_sets:
        return {}
    return _convert_entrez_gmt(
        raw_sets,
        mane_reference_path=PROJECT_ROOT / "reference" / "MANE.GRCh38.v1.3.UniversalReference.tsv",
        overrides=_ENTREZ_SYMBOL_OVERRIDES,
    )


def _load_pfizer_prc2_sets() -> Dict[str, List[str]]:
    """Load the Pfizer PRC2 Reactome gene set."""
    _require_existing_path(_GSEA_PRC2_GMT, description="Pfizer PRC2 GMT")
    return _load_gmt(_GSEA_PRC2_GMT, description="Pfizer PRC2 GMT")


def _run_pluvicto_shortlist_scores(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    tag: str = "",
) -> Optional[pd.DataFrame]:
    """Compute ARG.10 / NE.10 / CCP.31 ssGSEA score matrices for Pluvicto."""
    shortlist_sets = _load_shortlist_sets()
    if not shortlist_sets:
        raise ValueError("Shortlist GMT loaded successfully but no shortlist gene sets were available")

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


def _run_pfizer_prc2_scores(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    tag: str = "",
) -> Optional[pd.DataFrame]:
    """Compute the Pfizer PRC2 ssGSEA score matrix from the Reactome GMT."""
    prc2_sets = _load_pfizer_prc2_sets()
    if not prc2_sets:
        raise ValueError("Pfizer PRC2 GMT loaded successfully but no PRC2 gene sets were available")

    prc2_out = ensure_dir(os.path.join(model_out, f"PRC2Scores{tag}"))
    scores_tsv = os.path.join(prc2_out, f"prc2_scores{tag}.tsv")

    if overwrite or not os.path.exists(scores_tsv):
        scores = _compute_ssgsea_score_matrix(pred, prc2_sets, min_size=10)
        if scores.empty:
            print(f"    [skip] PRC2 scores{tag}: no genes present in predictions")
            return None
        scores.to_csv(scores_tsv, sep="\t")
        print("    PRC2 scores saved")
    else:
        scores = pd.read_csv(scores_tsv, sep="\t", index_col=0)

    mean_scores = scores.mean(axis=1).sort_values(ascending=False)
    print(
        "    PRC2 mean scores: "
        + ", ".join(f"{name}={mean_scores[name]:.3f}" for name in mean_scores.index)
    )
    return scores


def _run_pluvicto_shortlist_scores_mc(
    draws: List[pd.DataFrame],
    model_out: str,
    overwrite: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Aggregate q5/q95 shortlist score intervals across MC draws for Pluvicto."""
    shortlist_sets = _load_shortlist_sets()
    if not shortlist_sets:
        print("    [skip] Shortlist MC: no shortlist sets available")
        return None, None

    shortlist_out = ensure_dir(os.path.join(model_out, "ShortlistScoreIntervals"))
    ci_lo_tsv = os.path.join(shortlist_out, "shortlist_score_ci_q5.tsv")
    ci_hi_tsv = os.path.join(shortlist_out, "shortlist_score_ci_q95.tsv")

    if not overwrite and os.path.exists(ci_lo_tsv) and os.path.exists(ci_hi_tsv):
        return (
            pd.read_csv(ci_lo_tsv, sep="\t", index_col=0),
            pd.read_csv(ci_hi_tsv, sep="\t", index_col=0),
        )

    mean_scores, ci_lo, ci_hi = _aggregate_ssgsea_score_matrices(
        draws, shortlist_sets, min_size=3)
    if mean_scores is None or ci_lo is None or ci_hi is None:
        print("    [skip] Shortlist MC: no genes present in predictions")
        return None, None

    ci_lo.to_csv(ci_lo_tsv, sep="\t")
    ci_hi.to_csv(ci_hi_tsv, sep="\t")
    print("    MC shortlist score intervals saved")
    return ci_lo, ci_hi


def _analysis_safe_name(name: str) -> str:
    """Convert an arbitrary analysis label into a filesystem-safe stem."""
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)
    ).strip("_")


def _is_pluvicto_discovery_cohort(cohort: str) -> bool:
    """Return True for the split CRPC-Pluvicto discovery cohorts."""
    return str(cohort) in _PLUVICTO_DISCOVERY_VARIANTS


def _select_pluvicto_baseline_samples(
    sample_names: Sequence[Any],
    *,
    include_fhl: bool,
) -> List[str]:
    """Select the requested baseline Pluvicto samples for discovery analysis."""
    baseline_samples = [str(sample_name) for sample_name in sample_names if str(sample_name).endswith("C1")]
    if include_fhl:
        return baseline_samples
    return [
        sample_name
        for sample_name in baseline_samples
        if sample_name.startswith("FH") and not sample_name.startswith("FHL")
    ]


def _load_pluvicto_responder_subtypes(sample_names: Sequence[Any]) -> pd.DataFrame:
    """Match baseline Pluvicto prediction samples to responder labels and TFx."""
    path = PROJECT_ROOT / "reference" / "CRPC-Pluvicto_responder-classification.csv"
    responder_df = pd.read_csv(path)
    required_columns = {"Sample_ID", "Group", "TFx"}
    missing_columns = sorted(required_columns.difference(responder_df.columns))
    if missing_columns:
        raise ValueError(
            "CRPC-Pluvicto responder classification is missing required columns: "
            + ", ".join(missing_columns)
        )

    responder_df["Sample_ID"] = responder_df["Sample_ID"].astype(str).str.strip()
    responder_df["Group"] = responder_df["Group"].astype(str).str.strip()
    group_map = {
        "Responder": "Responder",
        "Non-Responder": "NonResponder",
    }
    unexpected_groups = sorted(set(responder_df["Group"]) - set(group_map))
    if unexpected_groups:
        raise ValueError(
            "Unexpected CRPC-Pluvicto responder groups: " + ", ".join(unexpected_groups)
        )

    responder_df["subtype"] = responder_df["Group"].map(group_map)
    responder_df["tfx"] = pd.to_numeric(responder_df["TFx"], errors="coerce")
    responder_df = responder_df.set_index("Sample_ID", drop=False)
    if responder_df.index.has_duplicates:
        duplicate_patient_ids = responder_df.index[responder_df.index.duplicated()].unique().astype(str).tolist()
        raise ValueError(
            "CRPC-Pluvicto responder classification has duplicate Sample_ID values: "
            + ", ".join(duplicate_patient_ids)
        )

    baseline_patient_ids = _dedupe_names_preserve_order(
        [extract_pluvicto_patient_id(str(sample_name)) for sample_name in sample_names]
    )
    available_patient_ids = set(responder_df.index.astype(str))
    matched_patient_ids = [patient_id for patient_id in baseline_patient_ids if patient_id in available_patient_ids]
    missing_patient_ids = [patient_id for patient_id in baseline_patient_ids if patient_id not in available_patient_ids]
    print(
        "    Pluvicto responder classification: "
        f"matched {len(matched_patient_ids)}/{len(baseline_patient_ids)} baseline patient IDs"
    )
    if missing_patient_ids:
        print("      patient IDs absent from reference/CRPC-Pluvicto_responder-classification.csv")
        _print_sample_name_list(missing_patient_ids, prefix="        ")

    matched_rows: List[Dict[str, Any]] = []
    for sample_name in sample_names:
        patient_id = extract_pluvicto_patient_id(str(sample_name))
        if patient_id not in responder_df.index:
            continue
        row = responder_df.loc[patient_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        matched_rows.append(
            {
                "sample_name": str(sample_name),
                "patient_id": patient_id,
                "group": str(row["Group"]),
                "subtype": str(row["subtype"]),
                "tfx": float(row["tfx"]) if pd.notna(row["tfx"]) else np.nan,
            }
        )

    if not matched_rows:
        return pd.DataFrame(columns=["patient_id", "group", "subtype", "tfx"])

    matched = pd.DataFrame(matched_rows).set_index("sample_name")
    ordered_samples = [str(sample_name) for sample_name in sample_names if str(sample_name) in matched.index]
    return matched.loc[ordered_samples]


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
    cox_mv: Optional[pd.DataFrame],
    covariates: pd.DataFrame,
    time_s: pd.Series,
    event_s: pd.Series,
    survival_out: str,
    output_stem: str,
    overwrite: bool,
    tag: str,
) -> None:
    """Write exploratory KM plots for multivariate Pluvicto survival hits.

    These curves are intentionally a visualization layer only: selection comes
    from the TFx-adjusted Cox screen, then each displayed feature is median-split
    for a readable KM summary. The Cox model remains the inferential analysis.
    """
    if cox_mv is None or cox_mv.empty:
        return

    mv_valid = cox_mv.dropna(subset=["variable", "HR", "p"])
    if mv_valid.empty:
        return

    # Prefer FDR-significant hits, but fall back to nominal multivariate hits so
    # the exploratory KM outputs are still emitted when the cohort is small.
    q_values = pd.to_numeric(mv_valid.get("q_value"), errors="coerce")
    selected = mv_valid.loc[q_values < 0.10].copy() if q_values is not None else mv_valid.iloc[0:0].copy()
    if selected.empty:
        selected = mv_valid.loc[pd.to_numeric(mv_valid["p"], errors="coerce") < 0.05].copy()
    if selected.empty:
        return

    sort_q = pd.to_numeric(selected.get("q_value"), errors="coerce")
    sort_p = pd.to_numeric(selected["p"], errors="coerce")
    selected = selected.assign(_sort_q=sort_q.fillna(np.inf), _sort_p=sort_p.fillna(np.inf))
    selected = selected.sort_values(["_sort_q", "_sort_p", "variable"]).head(6)

    for _, row in selected.iterrows():
        feature_name = str(row["variable"])
        if feature_name not in covariates.columns:
            continue

        feature_series = pd.to_numeric(covariates[feature_name], errors="coerce")
        shared_index = feature_series.index.intersection(time_s.index).intersection(event_s.index)
        feature_series = feature_series.loc[shared_index]
        time_shared = pd.to_numeric(time_s.loc[shared_index], errors="coerce")
        event_shared = pd.to_numeric(event_s.loc[shared_index], errors="coerce")
        valid_mask = feature_series.notna() & time_shared.notna() & event_shared.notna()
        if int(valid_mask.sum()) < 10:
            continue

        feature_series = feature_series.loc[valid_mask]
        time_shared = time_shared.loc[valid_mask].astype(float)
        event_shared = event_shared.loc[valid_mask].astype(int)

        split_value = float(np.median(feature_series.values.astype(float)))
        low_mask = feature_series <= split_value
        high_mask = feature_series > split_value
        if not bool(low_mask.any()) or not bool(high_mask.any()):
            continue

        logrank_p = compute_logrank_p(
            time_shared.loc[low_mask],
            event_shared.loc[low_mask],
            time_shared.loc[high_mask],
            event_shared.loc[high_mask],
        )
        out_path = os.path.join(
            survival_out,
            f"km_{output_stem}_{_analysis_safe_name(feature_name)}{tag}",
        )
        _maybe(
            kaplan_meier_plot,
            overwrite,
            time_shared.loc[low_mask].values.astype(float),
            event_shared.loc[low_mask].values.astype(int),
            time_shared.loc[high_mask].values.astype(float),
            event_shared.loc[high_mask].values.astype(int),
            out_path,
            label_lo=f"{feature_name} <= median",
            label_hi=f"{feature_name} > median",
            title=f"Kaplan-Meier — {feature_name}{tag}",
            logrank_p=logrank_p,
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

    cox_sig = cox_scores.dropna(subset=["HR"])
    if cox_sig.empty:
        print(f"    [skip] {score_label}: no finite Cox estimates")
        return

    cox_scores_mv = None
    show_rows_mv = None
    if tfx_series_v is not None:
        # Restore the pre-limma behavior the user asked to preserve: Pluvicto
        # survival keeps a TFx-adjusted Cox branch even though limma no longer
        # uses TFx as a modeled covariate.
        tfx_series = pd.to_numeric(tfx_series_v.reindex(matched_pids), errors="coerce")
        cox_scores_mv = compute_cox_multivariate(
            time_s,
            event_s,
            covariates,
            tfx_series,
            adjust_name="TFx",
        )
        cox_scores_mv.to_csv(
            os.path.join(survival_out, f"cox_{output_stem}_results_mv{tag}.tsv"),
            sep="\t",
            index=False,
        )
        _summarize_cox_ph_diagnostics(
            cox_scores_mv,
            f"{score_label} TFx-adjusted",
            ph_column="ph_min_p",
        )

    cox_sig = cox_sig.sort_values("HR")
    n_show = min(10, len(cox_sig))
    show_rows = pd.concat([cox_sig.head(n_show), cox_sig.tail(n_show)])
    show_rows = show_rows.drop_duplicates(subset=["variable"])
    if cox_scores_mv is not None:
        cox_scores_mv_valid = cox_scores_mv.dropna(subset=["variable", "HR"])
        if not cox_scores_mv_valid.empty:
            show_rows_mv = (
                cox_scores_mv_valid.set_index("variable")
                .reindex(show_rows["variable"])
                .dropna(subset=["HR"])
                .reset_index()
            )

    _maybe(
        forest_plot, overwrite,
        show_rows,
        os.path.join(survival_out, f"forest_{output_stem}{tag}"),
        hr_table_mv=show_rows_mv,
        title=f"Cox PH — {score_label}{tag}",
    )
    _plot_pluvicto_significant_multivariate_kms(
        cox_scores_mv,
        covariates,
        time_s,
        event_s,
        survival_out,
        output_stem,
        overwrite,
        tag,
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

    pluvicto_subtypes = _load_filtered_subtypes_for_reference(
        "CRPC-Pluvicto",
        samples_v,
        minimum_tfx=minimum_tfx,
        report_context="Discovery_Patient CRPC-Pluvicto survival",
    )
    tfx_series_v: Optional[pd.Series] = None
    if pluvicto_subtypes is not None and "tfx" in pluvicto_subtypes.columns:
        # Convert the per-sample subtype table back to the patient IDs used by
        # the survival models so TFx adjustment stays aligned with the chosen
        # baseline sample for each patient.
        tfx_by_pid: Dict[str, float] = {}
        for sample_name in samples_v:
            if sample_name not in pluvicto_subtypes.index:
                continue
            tfx_value = pd.to_numeric(
                pd.Series([pluvicto_subtypes.loc[sample_name, "tfx"]]),
                errors="coerce",
            ).iloc[0]
            patient_id = extract_pluvicto_patient_id(sample_name)
            tfx_by_pid[patient_id] = float(tfx_value) if np.isfinite(tfx_value) else np.nan
        tfx_series_v = pd.Series(tfx_by_pid).reindex(pids_v)

    survival_out = ensure_dir(os.path.join(model_out, f"survival{tag}"))

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

        cox_clin = compute_cox_univariate(time_c, event_c, clin_covs)
        cox_clin.to_csv(
            os.path.join(survival_out, f"cox_clinical_results{tag}.tsv"),
            sep="\t", index=False)
        _summarize_cox_ph_diagnostics(cox_clin, "clinical univariate", ph_column="ph_p")

        cox_clin_mv = None
        cox_clin_mv_valid = None
        if tfx_series_v is not None:
            tfx_clin = pd.to_numeric(tfx_series_v.reindex(clin_pids), errors="coerce")
            cox_clin_mv = compute_cox_multivariate(
                time_c,
                event_c,
                clin_covs,
                tfx_clin,
                adjust_name="TFx",
            )
            cox_clin_mv.to_csv(
                os.path.join(survival_out, f"cox_clinical_results_mv{tag}.tsv"),
                sep="\t",
                index=False,
            )
            _summarize_cox_ph_diagnostics(
                cox_clin_mv,
                "clinical TFx-adjusted",
                ph_column="ph_min_p",
            )
            cox_clin_mv_valid = cox_clin_mv.dropna(subset=["variable", "HR"])

        cox_clin_valid = cox_clin.dropna(subset=["HR"])
        if not cox_clin_valid.empty:
            _maybe(
                forest_plot, overwrite,
                cox_clin_valid,
                os.path.join(survival_out, f"forest_clinical{tag}"),
                hr_table_mv=cox_clin_mv_valid,
                title=f"Cox PH — clinical & imaging variables{tag}",
            )
        _plot_pluvicto_significant_multivariate_kms(
            cox_clin_mv,
            clin_covs,
            time_c,
            event_c,
            survival_out,
            "clinical",
            overwrite,
            tag,
        )


def _run_pluvicto_clinical_correlations(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    *,
    shortlist_scores: Optional[pd.DataFrame] = None,
    subtypes_df: Optional[pd.DataFrame] = None,
    cohort_label: str = "CRPC-Pluvicto",
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
    sample_meta = pd.DataFrame(index=pids)
    sample_meta["cohort"] = cohort_label
    sample_meta["status"] = "discovery"
    if subtypes_df is not None and "tfx" in subtypes_df.columns:
        tfx_by_pid: Dict[str, float] = {}
        for sample_name in samples:
            if sample_name not in subtypes_df.index:
                continue
            tfx_value = pd.to_numeric(pd.Series([subtypes_df.loc[sample_name, "tfx"]]), errors="coerce").iloc[0]
            tfx_by_pid[extract_pluvicto_patient_id(sample_name)] = float(tfx_value) if np.isfinite(tfx_value) else np.nan
        sample_meta["tfx"] = pd.Series(tfx_by_pid).reindex(pids)

    # --- FOLH1 vs PSMA-PET ---
    if "FOLH1" in pred.index:
        folh1_vals = pred.loc["FOLH1", samples].values.astype(float)
        folh1_series = pd.Series(folh1_vals, index=pids)
        for psma_col in ["PSMA-PET_SUVmax", "PSMA-PET_SUVmean",
                         "PSMA-PET_PSMAexpression", "PSMA-PET_Volume"]:
            if psma_col not in clinical.columns:
                continue
            clin_vals = pd.to_numeric(clinical.loc[pids, psma_col], errors="coerce")
            valid = folh1_series.notna() & clin_vals.notna()
            if valid.sum() < 5:
                continue
            log10_y = psma_col.endswith("_Volume") or psma_col.endswith("_TLG")
            _maybe(
                clinical_correlation_scatter, overwrite,
                folh1_series[valid], clin_vals[valid],
                os.path.join(corr_out, f"FOLH1_vs_{psma_col}"),
                xlabel="FOLH1 predicted expression (log₂CPM+1)",
                ylabel=f"log10({psma_col})" if log10_y else psma_col,
                title=f"FOLH1 vs {psma_col}",
                sample_meta=sample_meta.loc[valid.index],
                y_log10=log10_y,
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
        if "tfx" in sample_meta.columns:
            tfx_series = pd.to_numeric(sample_meta["tfx"], errors="coerce")
            valid_tfx = ccp31_series.notna() & tfx_series.notna()
            if int(valid_tfx.sum()) >= 5:
                _maybe(
                    clinical_correlation_scatter, overwrite,
                    tfx_series[valid_tfx], ccp31_series[valid_tfx],
                    os.path.join(corr_out, "CCP31_vs_TFx"),
                    xlabel="TFx",
                    ylabel=ccp31_xlabel,
                    title="CCP31 vs TFx",
                    sample_meta=sample_meta.loc[valid_tfx.index],
                )
            else:
                print("    [skip] CCP31 vs TFx: fewer than 5 samples with both values")

        for fdg_col in ["FDG_SUVmax", "FDG_SUVmean", "FDG_TLG", "FDG_Volume"]:
            if fdg_col not in clinical.columns:
                continue
            clin_vals = pd.to_numeric(clinical.loc[pids, fdg_col], errors="coerce")
            valid = ccp31_series.notna() & clin_vals.notna()
            if valid.sum() < 5:
                continue
            log10_y = fdg_col.endswith("_Volume") or fdg_col.endswith("_TLG")
            _maybe(
                clinical_correlation_scatter, overwrite,
                ccp31_series[valid], clin_vals[valid],
                os.path.join(corr_out, f"CCP31_vs_{fdg_col}"),
                xlabel=ccp31_xlabel,
                ylabel=f"log10({fdg_col})" if log10_y else fdg_col,
                title=f"CCP31 vs {fdg_col}",
                sample_meta=sample_meta.loc[valid.index],
                y_log10=log10_y,
            )
    else:
        print("    [skip] CCP31 gene set not available")


def _run_pluvicto_responder_deg(
    pred: pd.DataFrame,
    model_out: str,
    overwrite: bool,
    *,
    bundle: Dict[str, Any],
) -> None:
    """Run responder-vs-nonresponder limma/GSEA on baseline Pluvicto means."""
    baseline_samples = [str(sample_name) for sample_name in pred.columns]
    responder_subtypes = _load_pluvicto_responder_subtypes(pred.columns)
    matched_sample_set = set(responder_subtypes.index.astype(str))
    reason_by_sample = {
        sample_name: "preliminary patient ID absent from responder classification table"
        for sample_name in baseline_samples
        if sample_name not in matched_sample_set
    }
    retained_for_deg = [sample_name for sample_name in baseline_samples if sample_name not in reason_by_sample]
    _report_sample_drops(
        "Pluvicto responder DEG sample match",
        baseline_samples,
        retained_for_deg,
        reason_by_sample=reason_by_sample,
    )
    if responder_subtypes.empty:
        print("    [skip] Pluvicto responder DEG: no matched baseline samples")
        return

    responder_counts = responder_subtypes["group"].value_counts()
    print(
        "    Pluvicto responder DEG: "
        f"{int(responder_counts.get('Responder', 0))} responders, "
        f"{int(responder_counts.get('Non-Responder', 0))} non-responders"
    )

    limma_out = ensure_dir(os.path.join(model_out, "limma_responder"))
    _run_weighted_limma_suite(
        pred,
        responder_subtypes[["subtype"]],
        limma_out,
        overwrite,
        bundle=bundle,
        contrast="Responder-NonResponder",
    )

    surfaceome_name = STANDARD_GENE_SETS["CancerSurfaceome"]
    if not gene_list_available(surfaceome_name):
        print("    [skip] Pluvicto responder DEG CancerSurfaceome: gene list unavailable")
        return

    surfaceome_genes = get_gene_list(surfaceome_name)
    surfaceome_overlap = [
        gene_name
        for gene_name in _dedupe_names_preserve_order(surfaceome_genes)
        if gene_name in pred.index
    ]
    print(
        "    Pluvicto responder DEG CancerSurfaceome: "
        f"{len(surfaceome_overlap)}/{len(surfaceome_genes)} genes overlap predictions"
    )
    if not surfaceome_overlap:
        print("    [skip] Pluvicto responder DEG CancerSurfaceome: no overlapping genes")
        return

    surfaceome_pred = pred.loc[surfaceome_overlap].copy()
    surfaceome_bundle = _filter_prediction_bundle_to_genes(bundle, surfaceome_overlap)
    surfaceome_out = ensure_dir(os.path.join(model_out, "limma_responder_cancer_surfaceome"))
    _run_weighted_limma_suite(
        surfaceome_pred,
        responder_subtypes[["subtype"]],
        surfaceome_out,
        overwrite,
        bundle=surfaceome_bundle,
        contrast="Responder-NonResponder",
    )


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


def _pfizer_normalize_response_4class(response_value: Any) -> str:
    """Normalize Pfizer best-response labels into the requested 4-class encoding."""
    response = str(response_value).strip()
    if response == "PR &PSA50":
        return "PR & PSA50"
    return response


def _pfizer_arm_from_dose(dose_value: Any) -> str:
    """Map dose labels to the binary treatment-arm encoding used in analysis."""
    dose = str(dose_value).strip()
    return "Enza-only" if dose == "Enza" else "EZH2i+Enza"


def _pfizer_dose_mg(dose_value: Any) -> Optional[int]:
    """Extract the numeric Pfizer dose in mg when present."""
    dose = str(dose_value).strip()
    match = re.search(r"(\d+)\s*mg", dose, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1))


def _pfizer_effective_dose_group(dose_value: Any) -> str:
    """Map dose labels to the prespecified effective-dose split."""
    dose_mg = _pfizer_dose_mg(dose_value)
    if dose_mg is not None and dose_mg >= 625:
        return ">=625 mg + Enza"
    return "<625 mg / Enza-only"


def _pfizer_enza_history(label_value: Any) -> str:
    """Extract Enza-history status from the source metadata string."""
    label = str(label_value).strip()
    if label.startswith("Enza Experienced"):
        return "Enza Experienced"
    return "Enza Naive"


def _pfizer_fed_status(label_value: Any) -> str:
    """Extract fed-status from the source metadata string."""
    label = str(label_value).strip()
    if "(Fed)" in label:
        return "Fed"
    return "Fasted"


def _pfizer_progression_binary(response_4class: Any) -> str:
    """Collapse 4-class response to the primary PD-vs-non-PD grouping."""
    return "PD" if str(response_4class).strip() == "PD" else "non-PD"


def _pfizer_clinical_benefit_strict(response_4class: Any) -> str:
    """Map 4-class response to the strict clinical-benefit grouping."""
    response = str(response_4class).strip()
    if response == "PD":
        return "non-benefit"
    if response in {"SD", "PR & PSA50"}:
        return "benefit"
    return "indeterminate"


def _pfizer_clinical_benefit_broad(response_4class: Any) -> str:
    """Map 4-class response to the broad non-progression grouping."""
    response = str(response_4class).strip()
    if response == "PD":
        return "progression"
    return "benefit_or_nonprogression"


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
            "PF-1497_BID": "dose",
            "Best_Overall_Resp": "best_response",
            "Pfizer_Visit": "visit",
            "Days_from_SCN": "days_from_scn",
            "TFx": "tfx",
            "depth_WGS": "depth_wgs",
        }
    )
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["best_response"] = meta["best_response"].map(_pfizer_normalize_best_response)
    meta["response_4class"] = meta["best_response"].map(_pfizer_normalize_response_4class)
    meta["visit"] = pd.Categorical(
        meta["visit"].astype(str).str.strip(),
        categories=_PFIZER_VISIT_ORDER,
        ordered=True,
    )
    meta["visit_order"] = meta["visit"].cat.codes
    meta["days_from_scn"] = pd.to_numeric(meta["days_from_scn"], errors="coerce")
    meta["tfx"] = pd.to_numeric(meta.get("tfx"), errors="coerce")
    meta["depth_wgs"] = pd.to_numeric(meta.get("depth_wgs"), errors="coerce")
    meta["patient_id"] = meta["sample_id"].map(_pfizer_patient_id)
    meta["dose"] = meta["dose"].astype(str).str.strip()
    meta["dose_mg"] = meta["dose"].map(_pfizer_dose_mg)
    meta["arm"] = meta["dose"].map(_pfizer_arm_from_dose)
    meta["effective_dose_group"] = meta["dose"].map(_pfizer_effective_dose_group)
    meta["enza_history"] = meta["enzalutamide_group"].map(_pfizer_enza_history)
    meta["fed_status"] = meta["enzalutamide_group"].map(_pfizer_fed_status)
    meta["progression_binary"] = meta["response_4class"].map(_pfizer_progression_binary)
    meta["clinical_benefit_strict"] = meta["response_4class"].map(_pfizer_clinical_benefit_strict)
    meta["clinical_benefit_broad"] = meta["response_4class"].map(_pfizer_clinical_benefit_broad)

    meta["response_4class"] = pd.Categorical(
        meta["response_4class"],
        categories=_PFIZER_RESPONSE_4CLASS_ORDER,
        ordered=True,
    )
    meta["progression_binary"] = pd.Categorical(
        meta["progression_binary"],
        categories=_PFIZER_PROGRESSION_ORDER,
        ordered=True,
    )
    meta["arm"] = pd.Categorical(
        meta["arm"],
        categories=_PFIZER_ARM_ORDER,
        ordered=True,
    )
    meta["effective_dose_group"] = pd.Categorical(
        meta["effective_dose_group"],
        categories=_PFIZER_EFFECTIVE_DOSE_ORDER,
        ordered=True,
    )
    meta["dose"] = pd.Categorical(
        meta["dose"],
        categories=_PFIZER_DOSE_ORDER,
        ordered=True,
    )
    meta["enza_history"] = pd.Categorical(
        meta["enza_history"],
        categories=_PFIZER_ENZA_HISTORY_ORDER,
        ordered=True,
    )
    meta["fed_status"] = pd.Categorical(
        meta["fed_status"],
        categories=_PFIZER_FED_STATUS_ORDER,
        ordered=True,
    )
    meta["clinical_benefit_strict"] = pd.Categorical(
        meta["clinical_benefit_strict"],
        categories=_PFIZER_CLINICAL_BENEFIT_STRICT_ORDER,
        ordered=True,
    )
    meta["clinical_benefit_broad"] = pd.Categorical(
        meta["clinical_benefit_broad"],
        categories=_PFIZER_CLINICAL_BENEFIT_BROAD_ORDER,
        ordered=True,
    )

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
    _require_existing_path(Path(path), description=f"saved Hallmark GSEA table for {model_out}")

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

    fig.subplots_adjust(
        left=0.08,
        right=0.80 if legend_handles else 0.96,
        top=0.96,
        bottom=0.08,
        hspace=0.05,
    )
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
    from scipy.stats import mannwhitneyu

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
    visit_comparisons: List[Tuple[float, float, np.ndarray, np.ndarray]] = []
    nonempty_groups = 0
    for visit_idx, visit_name in enumerate(_PFIZER_VISIT_ORDER):
        visit_df = valid.loc[valid["visit"].astype(str) == visit_name]
        per_visit_groups: Dict[str, np.ndarray] = {}
        for response_code in response_codes:
            group_df = visit_df.loc[visit_df["response_binary"].astype(str) == response_code]
            group_vals = group_df["value"].astype(float).values
            per_visit_groups[response_code] = group_vals
            if len(group_vals) == 0:
                continue
            nonempty_groups += 1
            position = float(visit_centers[visit_idx] + response_offsets[response_code])
            box_data.append(group_vals)
            box_positions.append(position)
            box_colors.append(_PFIZER_RESPONSE_COLOR_MAP[response_labels[response_code]])
            point_groups.append((position, group_vals, _PFIZER_RESPONSE_COLOR_MAP[response_labels[response_code]]))
        if all(len(per_visit_groups.get(code, np.array([]))) > 0 for code in response_codes):
            visit_comparisons.append(
                (
                    float(visit_centers[visit_idx] + response_offsets[response_codes[0]]),
                    float(visit_centers[visit_idx] + response_offsets[response_codes[1]]),
                    per_visit_groups[response_codes[0]],
                    per_visit_groups[response_codes[1]],
                )
            )

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

    if visit_comparisons:
        y_min, y_max = ax.get_ylim()
        y_range = max(y_max - y_min, 1e-6)
        base_y = max(float(np.max(vals)) for vals in box_data if len(vals) > 0) + 0.06 * y_range
        step_y = 0.10 * y_range
        cap_y = 0.02 * y_range
        for idx, (left_x, right_x, left_vals, right_vals) in enumerate(visit_comparisons):
            p_value = float(mannwhitneyu(left_vals, right_vals, alternative="two-sided").pvalue)
            y_ann = base_y + idx * step_y
            ax.plot([left_x, left_x, right_x, right_x], [y_ann - cap_y, y_ann, y_ann, y_ann - cap_y], lw=1.0, color="black")
            ax.text(
                0.5 * (left_x + right_x),
                y_ann + 0.01 * y_range,
                format_pval(p_value),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(y_min, max(y_max, base_y + len(visit_comparisons) * step_y + 0.08 * y_range))

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
    from scipy.stats import mannwhitneyu

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

    nonempty_groups = [vals for vals in grouped if len(vals) > 0]
    if len(nonempty_groups) == 2:
        y_min, y_max = ax.get_ylim()
        y_range = max(y_max - y_min, 1e-6)
        y_top = max(float(np.max(vals)) for vals in nonempty_groups)
        y_ann = y_top + 0.08 * y_range
        cap_y = 0.02 * y_range
        p_value = float(mannwhitneyu(nonempty_groups[0], nonempty_groups[1], alternative="two-sided").pvalue)
        ax.plot([positions[0], positions[0], positions[1], positions[1]], [y_ann - cap_y, y_ann, y_ann, y_ann - cap_y], lw=1.0, color="black")
        ax.text(
            0.5 * (positions[0] + positions[1]),
            y_ann + 0.01 * y_range,
            format_pval(p_value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.set_ylim(y_min, max(y_max, y_ann + 0.10 * y_range))

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


def _pfizer_v2_bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap a simple median or mean confidence interval."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    if arr.size == 1:
        val = float(arr[0])
        return val, val

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        if statistic == "median":
            boot[idx] = float(np.median(sample))
        elif statistic == "mean":
            boot[idx] = float(np.mean(sample))
        else:
            raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _pfizer_v2_rank_biserial(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Return the Mann-Whitney rank-biserial effect size for group_b minus group_a."""
    from scipy.stats import mannwhitneyu

    arr_a = np.asarray(group_a, dtype=float)
    arr_b = np.asarray(group_b, dtype=float)
    arr_a = arr_a[np.isfinite(arr_a)]
    arr_b = arr_b[np.isfinite(arr_b)]
    if arr_a.size == 0 or arr_b.size == 0:
        return np.nan
    u_stat = float(mannwhitneyu(arr_a, arr_b, alternative="two-sided").statistic)
    return float(1.0 - (2.0 * u_stat) / (arr_a.size * arr_b.size))


def _pfizer_v2_signed_rank_biserial(deltas: Sequence[float]) -> float:
    """Return the matched-pairs rank-biserial effect size for paired deltas."""
    from scipy.stats import rankdata

    delta_arr = np.asarray(deltas, dtype=float)
    delta_arr = delta_arr[np.isfinite(delta_arr)]
    delta_arr = delta_arr[delta_arr != 0.0]
    if delta_arr.size == 0:
        return np.nan

    ranks = rankdata(np.abs(delta_arr), method="average")
    pos_sum = float(ranks[delta_arr > 0].sum())
    neg_sum = float(ranks[delta_arr < 0].sum())
    denom = pos_sum + neg_sum
    if denom <= 0.0:
        return np.nan
    return float((pos_sum - neg_sum) / denom)


def _pfizer_v2_cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Return Cohen's d for group_b minus group_a."""
    arr_a = np.asarray(group_a, dtype=float)
    arr_b = np.asarray(group_b, dtype=float)
    arr_a = arr_a[np.isfinite(arr_a)]
    arr_b = arr_b[np.isfinite(arr_b)]
    if arr_a.size < 2 or arr_b.size < 2:
        return np.nan

    var_a = float(np.var(arr_a, ddof=1))
    var_b = float(np.var(arr_b, ddof=1))
    pooled_denom = ((arr_a.size - 1) * var_a) + ((arr_b.size - 1) * var_b)
    pooled_n = arr_a.size + arr_b.size - 2
    if pooled_n <= 0 or pooled_denom <= 0.0:
        return np.nan
    pooled_sd = float(np.sqrt(pooled_denom / pooled_n))
    if pooled_sd <= 0.0:
        return np.nan
    return float((np.mean(arr_b) - np.mean(arr_a)) / pooled_sd)


def _pfizer_v2_preferred_levels(column_name: str) -> Optional[List[str]]:
    """Return preferred plotting order for key Pfizer metadata columns."""
    return {
        "response_4class": _PFIZER_RESPONSE_4CLASS_ORDER,
        "progression_binary": _PFIZER_PROGRESSION_ORDER,
        "arm": _PFIZER_ARM_ORDER,
        "effective_dose_group": _PFIZER_EFFECTIVE_DOSE_ORDER,
        "dose": _PFIZER_DOSE_ORDER,
        "enza_history": _PFIZER_ENZA_HISTORY_ORDER,
        "fed_status": _PFIZER_FED_STATUS_ORDER,
        "clinical_benefit_strict": _PFIZER_CLINICAL_BENEFIT_STRICT_ORDER,
        "clinical_benefit_broad": _PFIZER_CLINICAL_BENEFIT_BROAD_ORDER,
    }.get(column_name)


def _pfizer_v2_color_map(column_name: str, levels: Sequence[str]) -> Dict[str, str]:
    """Return a stable categorical color map for a Pfizer grouping variable."""
    base_maps: Dict[str, Dict[str, str]] = {
        "response_4class": _PFIZER_RESPONSE_4CLASS_COLOR_MAP,
        "progression_binary": _PFIZER_PROGRESSION_COLOR_MAP,
        "arm": _PFIZER_ARM_COLOR_MAP,
        "effective_dose_group": _PFIZER_EFFECTIVE_DOSE_COLOR_MAP,
    }
    preferred = dict(base_maps.get(column_name, {}))
    missing = [level for level in levels if level not in preferred]
    if not missing:
        return preferred

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    for idx, level in enumerate(missing):
        preferred[level] = matplotlib.colors.to_hex(cmap(idx % cmap.N))
    return preferred


def _pfizer_v2_response_legend_handles(
    response_levels: Optional[Sequence[str]] = None,
) -> List[Any]:
    """Return legend handles for the Pfizer four-class response color mapping."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.lines import Line2D

    levels = list(response_levels) if response_levels is not None else list(_PFIZER_RESPONSE_4CLASS_ORDER)
    color_map = _pfizer_v2_color_map("response_4class", levels)
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map.get(level, "#999999"),
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=0,
            markersize=7,
            label=level,
        )
        for level in levels
        if level in color_map
    ]


def _pfizer_v2_arm_marker_legend_handles() -> List[Any]:
    """Return legend handles for the Pfizer treatment-arm marker mapping."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.lines import Line2D

    marker_map = {"Enza-only": "o", "EZH2i+Enza": "s"}
    return [
        Line2D(
            [0],
            [0],
            marker=marker_map[level],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=0,
            markersize=7,
            label=level,
        )
        for level in _PFIZER_ARM_ORDER
        if level in marker_map
    ]


def _pfizer_v2_visit_marker_legend_handles(
    visit_levels: Optional[Sequence[str]] = None,
) -> List[Any]:
    """Return legend handles for the Pfizer visit marker mapping."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.lines import Line2D

    levels = list(visit_levels) if visit_levels is not None else list(_PFIZER_VISIT_ORDER)
    return [
        Line2D(
            [0],
            [0],
            marker=_PFIZER_VISIT_MARKER_MAP.get(level, "o"),
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            linewidth=0,
            markersize=7,
            label=level,
        )
        for level in levels
    ]


def _pfizer_v2_add_dual_legends(
    ax: Any,
    *,
    color_handles: Optional[Sequence[Any]] = None,
    color_title: Optional[str] = None,
    marker_handles: Optional[Sequence[Any]] = None,
    marker_title: Optional[str] = None,
    color_anchor: Tuple[float, float] = (1.02, 1.0),
    marker_anchor: Tuple[float, float] = (1.02, 0.62),
) -> None:
    """Attach separate color and marker legends to a Pfizer plot."""
    color_legend = None
    if color_handles:
        color_legend = ax.legend(
            handles=list(color_handles),
            title=color_title,
            fontsize=8,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=color_anchor,
            frameon=False,
        )
        ax.add_artist(color_legend)
    if marker_handles:
        ax.legend(
            handles=list(marker_handles),
            title=marker_title,
            fontsize=8,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=marker_anchor,
            frameon=False,
        )


def _pfizer_v2_patient_metadata(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse matched Pfizer metadata to one row per patient."""
    rows: List[Dict[str, Any]] = []
    ordered = sample_df.sort_values(["patient_id", "visit_order", "days_from_scn", "sample_id"])
    for patient_id, patient_rows in ordered.groupby("patient_id"):
        scn_rows = patient_rows.loc[patient_rows["visit"].astype(str) == "SCN"]
        eot_rows = patient_rows.loc[patient_rows["visit"].astype(str) == "EOT"]
        anchor = scn_rows.iloc[0] if not scn_rows.empty else patient_rows.iloc[0]
        eot_anchor = eot_rows.iloc[0] if not eot_rows.empty else patient_rows.iloc[-1]
        rows.append(
            {
                "patient_id": patient_id,
                "arm": anchor["arm"],
                "effective_dose_group": anchor["effective_dose_group"],
                "dose": anchor["dose"],
                "dose_mg": anchor["dose_mg"],
                "enza_history": anchor["enza_history"],
                "fed_status": anchor["fed_status"],
                "response_4class": anchor["response_4class"],
                "progression_binary": anchor["progression_binary"],
                "clinical_benefit_strict": anchor["clinical_benefit_strict"],
                "clinical_benefit_broad": anchor["clinical_benefit_broad"],
                "n_samples": int(len(patient_rows)),
                "visits_present": ";".join(patient_rows["visit"].astype(str).tolist()),
                "sample_SCN": scn_rows.index[0] if not scn_rows.empty else pd.NA,
                "sample_C3D1": patient_rows.loc[patient_rows["visit"].astype(str) == "C3D1"].index[0]
                if bool((patient_rows["visit"].astype(str) == "C3D1").any())
                else pd.NA,
                "sample_EOT": eot_rows.index[0] if not eot_rows.empty else pd.NA,
                "days_from_scn_EOT": pd.to_numeric(pd.Series([eot_anchor["days_from_scn"]]), errors="coerce").iloc[0],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("patient_id")


def _pfizer_v2_prepare_pam50(
    pred_subset: pd.DataFrame,
    pfizer_out: str,
    overwrite: bool,
) -> pd.DataFrame:
    """Compute PAM50 calls and primary continuous scores for Pfizer samples."""
    pred_for_scores = _prepare_expression_matrix_for_scores(
        pred_subset,
        context="CRPC-Pfizer PAM50",
    )
    if pred_for_scores.empty:
        return pd.DataFrame()

    selected = _select_pam50_result(pam50_classify_genefu(pred_for_scores))
    if selected is None:
        return pd.DataFrame()
    variant_name, variant_df = selected
    variant_out = ensure_dir(os.path.join(pfizer_out, "PAM50", _analysis_safe_name(variant_name)))
    variant_df.to_csv(
        os.path.join(variant_out, "pam50_classification_predicted.tsv"),
        sep="\t",
        index=False,
    )
    score_columns = _pam50_primary_score_columns(variant_df.columns)
    if score_columns:
        variant_df[["sample"] + score_columns].to_csv(
            os.path.join(variant_out, "pam50_primary_scores.tsv"),
            sep="\t",
            index=False,
        )
    return variant_df.drop_duplicates(subset="sample").set_index("sample")


def _pfizer_v2_build_sample_feature_table(
    pred_subset: pd.DataFrame,
    pfizer_meta: pd.DataFrame,
    pfizer_out: str,
    overwrite: bool,
    *,
    hallmark_scores: Optional[pd.DataFrame] = None,
    shortlist_scores: Optional[pd.DataFrame] = None,
    prc2_scores: Optional[pd.DataFrame] = None,
    pam50_df: Optional[pd.DataFrame] = None,
    compute_pam50: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Assemble the streamlined sample-level Pfizer feature table."""
    sample_df = pfizer_meta.copy()
    if hallmark_scores is None:
        hallmark_scores = _run_pluvicto_ssgsea(pred_subset, pfizer_out, overwrite, tag="_pfizer")
    if shortlist_scores is None:
        shortlist_scores = _run_pluvicto_shortlist_scores(pred_subset, pfizer_out, overwrite, tag="_pfizer")
    if prc2_scores is None:
        prc2_scores = _run_pfizer_prc2_scores(pred_subset, pfizer_out, overwrite, tag="_pfizer")
    if pam50_df is None:
        pam50_df = _pfizer_v2_prepare_pam50(pred_subset, pfizer_out, overwrite) if compute_pam50 else pd.DataFrame()

    available_features: List[str] = ["tfx"]
    if shortlist_scores is not None and not shortlist_scores.empty:
        for shortlist_name, display_name in _PFIZER_SHORTLIST_LABEL_MAP.items():
            if shortlist_name not in shortlist_scores.index:
                continue
            sample_df[display_name] = pd.to_numeric(
                shortlist_scores.loc[shortlist_name].reindex(sample_df.index),
                errors="coerce",
            )
            available_features.append(display_name)

    if prc2_scores is not None and not prc2_scores.empty:
        prc2_row = str(prc2_scores.index[0])
        sample_df[_PFIZER_PRC2_SCORE_NAME] = pd.to_numeric(
            prc2_scores.loc[prc2_row].reindex(sample_df.index),
            errors="coerce",
        )
        available_features.append(_PFIZER_PRC2_SCORE_NAME)

    if hallmark_scores is not None and not hallmark_scores.empty:
        for hallmark_name in _PFIZER_BASELINE_HALLMARKS:
            if hallmark_name not in hallmark_scores.index:
                continue
            sample_df[hallmark_name] = pd.to_numeric(
                hallmark_scores.loc[hallmark_name].reindex(sample_df.index),
                errors="coerce",
            )
            available_features.append(hallmark_name)

    if not pam50_df.empty:
        sample_df["PAM50 subtype"] = pam50_df["predicted_subtype"].reindex(sample_df.index)
        score_columns = _pam50_primary_score_columns(pam50_df.columns)
        for score_column in score_columns:
            renamed = f"PAM50 {score_column}"
            sample_df[renamed] = pd.to_numeric(pam50_df[score_column].reindex(sample_df.index), errors="coerce")
            available_features.append(renamed)

    sample_df = sample_df.sort_values(["patient_id", "visit_order", "days_from_scn", "sample_id"])
    available_features = list(dict.fromkeys(available_features))

    pd.DataFrame(
        {
            "feature": available_features,
            "present_in_sample_table": [feature in sample_df.columns for feature in available_features],
        }
    ).to_csv(
        os.path.join(pfizer_out, "feature_inventory.tsv"),
        sep="\t",
        index=False,
    )
    sample_df.to_csv(os.path.join(pfizer_out, "sample_level_features.tsv"), sep="\t")
    return sample_df, available_features


def _pfizer_v2_build_patient_feature_table(
    sample_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Pivot sample-level Pfizer features into one patient-level wide table."""
    patient_meta = _pfizer_v2_patient_metadata(sample_df)
    if patient_meta.empty:
        return pd.DataFrame()

    ordered = sample_df.sort_values(["patient_id", "visit_order", "days_from_scn", "sample_id"])
    rows: List[Dict[str, Any]] = []
    for patient_id, patient_rows in ordered.groupby("patient_id"):
        row = patient_meta.loc[patient_id].to_dict()
        row["patient_id"] = patient_id
        for visit_name in _PFIZER_VISIT_ORDER:
            visit_rows = patient_rows.loc[patient_rows["visit"].astype(str) == visit_name]
            if visit_rows.empty:
                continue
            visit_row = visit_rows.iloc[0]
            row[f"days_from_scn__{visit_name}"] = pd.to_numeric(
                pd.Series([visit_row["days_from_scn"]]), errors="coerce"
            ).iloc[0]
            for feature_name in feature_columns:
                if feature_name not in visit_row.index:
                    continue
                row[f"{feature_name}__{visit_name}"] = pd.to_numeric(
                    pd.Series([visit_row[feature_name]]), errors="coerce"
                ).iloc[0]

        for feature_name in feature_columns:
            scn_value = row.get(f"{feature_name}__SCN", np.nan)
            c3d1_value = row.get(f"{feature_name}__C3D1", np.nan)
            eot_value = row.get(f"{feature_name}__EOT", np.nan)
            row[f"{feature_name}__baseline_value"] = scn_value
            row[f"{feature_name}__c3d1_delta"] = (
                float(c3d1_value - scn_value)
                if np.isfinite(scn_value) and np.isfinite(c3d1_value)
                else np.nan
            )
            row[f"{feature_name}__eot_delta"] = (
                float(eot_value - scn_value)
                if np.isfinite(scn_value) and np.isfinite(eot_value)
                else np.nan
            )
            row[f"{feature_name}__eot_from_c3d1_delta"] = (
                float(eot_value - c3d1_value)
                if np.isfinite(c3d1_value) and np.isfinite(eot_value)
                else np.nan
            )
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("patient_id")


def _pfizer_v2_build_delta_frame(
    patient_df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    start_visit: str,
    end_visit: str,
    delta_column_name: str,
) -> pd.DataFrame:
    """Create a patient-level delta table between two visits."""
    rows: List[Dict[str, Any]] = []
    meta_columns = [
        "arm",
        "effective_dose_group",
        "dose",
        "dose_mg",
        "enza_history",
        "fed_status",
        "response_4class",
        "progression_binary",
        "clinical_benefit_strict",
        "clinical_benefit_broad",
        "days_from_scn_EOT",
    ]
    for patient_id, patient_row in patient_df.iterrows():
        row: Dict[str, Any] = {"patient_id": patient_id}
        for meta_column in meta_columns:
            if meta_column in patient_row.index:
                row[meta_column] = patient_row[meta_column]
        row["baseline_tfx"] = pd.to_numeric(
            pd.Series([patient_row.get(f"tfx__{start_visit}", np.nan)]),
            errors="coerce",
        ).iloc[0]
        row["end_tfx"] = pd.to_numeric(
            pd.Series([patient_row.get(f"tfx__{end_visit}", np.nan)]),
            errors="coerce",
        ).iloc[0]
        row["delta_tfx"] = (
            float(row["end_tfx"] - row["baseline_tfx"])
            if np.isfinite(row["baseline_tfx"]) and np.isfinite(row["end_tfx"])
            else np.nan
        )
        complete_any = False
        for feature_name in feature_columns:
            start_val = pd.to_numeric(pd.Series([patient_row.get(f"{feature_name}__{start_visit}", np.nan)]), errors="coerce").iloc[0]
            end_val = pd.to_numeric(pd.Series([patient_row.get(f"{feature_name}__{end_visit}", np.nan)]), errors="coerce").iloc[0]
            if np.isfinite(start_val) and np.isfinite(end_val):
                row[feature_name] = float(end_val - start_val)
                complete_any = True
            else:
                row[feature_name] = np.nan
        if complete_any:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    delta_df = pd.DataFrame(rows).set_index("patient_id")
    delta_df.attrs["delta_column_name"] = delta_column_name
    return delta_df


def _pfizer_v2_two_group_stats(
    data_df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    group_column: str,
    group_a: str,
    group_b: str,
    comparison_name: str,
) -> pd.DataFrame:
    """Summarize a two-group baseline or delta comparison across many features."""
    from scipy.stats import mannwhitneyu

    rows: List[Dict[str, Any]] = []
    for feature_name in feature_columns:
        if feature_name not in data_df.columns or group_column not in data_df.columns:
            continue
        working = data_df[[feature_name, group_column]].dropna().copy()
        if working.empty:
            continue
        vals_a = working.loc[working[group_column].astype(str) == group_a, feature_name].astype(float).values
        vals_b = working.loc[working[group_column].astype(str) == group_b, feature_name].astype(float).values
        if vals_a.size == 0 or vals_b.size == 0:
            continue

        p_value = float(mannwhitneyu(vals_a, vals_b, alternative="two-sided").pvalue)
        median_a = float(np.median(vals_a))
        median_b = float(np.median(vals_b))
        diff_samples = np.asarray([
            np.median(np.random.default_rng(42 + idx).choice(vals_b, size=vals_b.size, replace=True))
            - np.median(np.random.default_rng(1042 + idx).choice(vals_a, size=vals_a.size, replace=True))
            for idx in range(500)
        ], dtype=float)
        rows.append(
            {
                "comparison": comparison_name,
                "feature": feature_name,
                "group_a": group_a,
                "group_b": group_b,
                "n_group_a": int(vals_a.size),
                "n_group_b": int(vals_b.size),
                "median_group_a": median_a,
                "median_group_b": median_b,
                "median_difference_group_b_minus_group_a": float(median_b - median_a),
                "median_difference_ci_lo": float(np.percentile(diff_samples, 2.5)),
                "median_difference_ci_hi": float(np.percentile(diff_samples, 97.5)),
                "rank_biserial_group_b_minus_group_a": _pfizer_v2_rank_biserial(vals_a, vals_b),
                "cohens_d_group_b_minus_group_a": _pfizer_v2_cohens_d(vals_a, vals_b),
                "p_value": p_value,
            }
        )
    if not rows:
        return pd.DataFrame()
    result_df = pd.DataFrame(rows)
    result_df["fdr"] = benjamini_hochberg(result_df["p_value"])
    return result_df.sort_values(["p_value", "feature"]).reset_index(drop=True)


def _pfizer_v2_paired_delta_stats(
    delta_df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    comparison_name: str,
) -> pd.DataFrame:
    """Summarize paired delta distributions across many features."""
    from scipy.stats import wilcoxon

    rows: List[Dict[str, Any]] = []
    for feature_name in feature_columns:
        if feature_name not in delta_df.columns:
            continue
        values = pd.to_numeric(delta_df[feature_name], errors="coerce").dropna().values.astype(float)
        if values.size == 0:
            continue
        q1, q3 = np.percentile(values, [25, 75]) if values.size >= 2 else (values[0], values[0])
        try:
            p_value = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p_value = np.nan
        ci_lo, ci_hi = _pfizer_v2_bootstrap_ci(values, statistic="median")
        rows.append(
            {
                "comparison": comparison_name,
                "feature": feature_name,
                "n": int(values.size),
                "median_delta": float(np.median(values)),
                "median_delta_ci_lo": ci_lo,
                "median_delta_ci_hi": ci_hi,
                "iqr_delta_lo": float(q1),
                "iqr_delta_hi": float(q3),
                "mean_delta": float(np.mean(values)),
                "n_increasing": int(np.sum(values > 0.0)),
                "n_decreasing": int(np.sum(values < 0.0)),
                "signed_rank_biserial": _pfizer_v2_signed_rank_biserial(values),
                "p_value": p_value,
            }
        )
    if not rows:
        return pd.DataFrame()
    result_df = pd.DataFrame(rows)
    result_df["fdr"] = benjamini_hochberg(result_df["p_value"])
    return result_df.sort_values(["p_value", "feature"]).reset_index(drop=True)


def _pfizer_v2_add_composite_scores(
    delta_df: pd.DataFrame,
    *,
    out_path: str,
) -> pd.DataFrame:
    """Add the prespecified C3D1 pharmacodynamic composite score(s) to a delta table."""
    working = delta_df.copy()
    component_records: List[Dict[str, Any]] = []
    z_components: Dict[str, pd.Series] = {}
    for feature_name, weight in _PFIZER_COMPOSITE_COMPONENTS:
        if feature_name not in working.columns:
            continue
        values = pd.to_numeric(working[feature_name], errors="coerce")
        mean_val = float(values.mean(skipna=True))
        std_val = float(values.std(skipna=True, ddof=0))
        if not np.isfinite(std_val) or std_val <= 0.0:
            z_values = pd.Series(np.nan, index=values.index, dtype=float)
        else:
            z_values = (values - mean_val) / std_val
        z_components[feature_name] = z_values
        component_records.append(
            {
                "feature": feature_name,
                "weight": float(weight),
                "mean": mean_val,
                "sd": std_val,
                "included_in_primary": feature_name == "PRC2 target derepression score (exploratory)" or feature_name in dict(_PFIZER_COMPOSITE_COMPONENTS_NOPRC2),
            }
        )

    if z_components:
        primary_sum = pd.Series(0.0, index=working.index, dtype=float)
        primary_count = pd.Series(0, index=working.index, dtype=int)
        for feature_name, weight in _PFIZER_COMPOSITE_COMPONENTS:
            z_values = z_components.get(feature_name)
            if z_values is None:
                continue
            valid = z_values.notna()
            primary_sum.loc[valid] = primary_sum.loc[valid] + (weight * z_values.loc[valid])
            primary_count.loc[valid] = primary_count.loc[valid] + 1
        working["C3D1 PD Response Score"] = primary_sum.where(primary_count > 0, np.nan)

        secondary_sum = pd.Series(0.0, index=working.index, dtype=float)
        secondary_count = pd.Series(0, index=working.index, dtype=int)
        for feature_name, weight in _PFIZER_COMPOSITE_COMPONENTS_NOPRC2:
            z_values = z_components.get(feature_name)
            if z_values is None:
                continue
            valid = z_values.notna()
            secondary_sum.loc[valid] = secondary_sum.loc[valid] + (weight * z_values.loc[valid])
            secondary_count.loc[valid] = secondary_count.loc[valid] + 1
        working["C3D1 PD Response Score noPRC2"] = secondary_sum.where(secondary_count > 0, np.nan)

    pd.DataFrame(component_records).to_csv(out_path, sep="\t", index=False)
    return working


def _pfizer_v2_plot_stacked_bar(
    patient_df: pd.DataFrame,
    *,
    group_column: str,
    fill_column: str,
    out_path: str,
    title: str,
) -> None:
    """Plot a stacked bar summary for one Pfizer metadata grouping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if group_column not in patient_df.columns or fill_column not in patient_df.columns:
        return
    working = patient_df[[group_column, fill_column]].dropna().copy()
    if working.empty:
        return

    group_levels = _pfizer_v2_preferred_levels(group_column) or sorted(working[group_column].astype(str).unique().tolist())
    fill_levels = _pfizer_v2_preferred_levels(fill_column) or sorted(working[fill_column].astype(str).unique().tolist())
    tab = pd.crosstab(working[group_column].astype(str), working[fill_column].astype(str))
    tab = tab.reindex(index=[level for level in group_levels if level in tab.index], fill_value=0)
    tab = tab.reindex(columns=[level for level in fill_levels if level in tab.columns], fill_value=0)
    if tab.empty:
        return

    fill_colors = _pfizer_v2_color_map(fill_column, tab.columns.tolist())
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(tab.index)), 4.8))
    bottoms = np.zeros(len(tab.index), dtype=float)
    x_positions = np.arange(len(tab.index), dtype=float)
    for level in tab.columns:
        vals = tab[level].astype(float).values
        ax.bar(
            x_positions,
            vals,
            bottom=bottoms,
            width=0.75,
            color=fill_colors.get(level, "#999999"),
            edgecolor="black",
            linewidth=0.6,
            label=level,
        )
        bottoms = bottoms + vals
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tab.index.tolist(), rotation=25, ha="right")
    ax.set_ylabel("Patients")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(title=fill_column.replace("_", " "), fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_alluvial(
    patient_df: pd.DataFrame,
    *,
    stage_columns: Sequence[str],
    out_path: str,
    title: str,
) -> None:
    """Draw a lightweight alluvial-style flow plot across ordered metadata stages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if len(stage_columns) < 2:
        return
    working = patient_df[list(stage_columns)].dropna().copy()
    if working.empty:
        return

    stage_levels: Dict[str, List[str]] = {}
    stage_counts: Dict[str, pd.Series] = {}
    stage_positions: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for stage_column in stage_columns:
        levels = _pfizer_v2_preferred_levels(stage_column) or sorted(working[stage_column].astype(str).unique().tolist())
        counts = working[stage_column].astype(str).value_counts().reindex(levels).dropna()
        counts = counts.loc[counts > 0]
        if counts.empty:
            return
        stage_levels[stage_column] = counts.index.tolist()
        stage_counts[stage_column] = counts

        current_y = 0.0
        stage_positions[stage_column] = {}
        for level, count in counts.items():
            next_y = current_y + float(count)
            stage_positions[stage_column][level] = (current_y, next_y)
            current_y = next_y

    max_height = max(float(counts.sum()) for counts in stage_counts.values())
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x_positions = np.arange(len(stage_columns), dtype=float)
    width = 0.18
    line_scale = 6.0 / max(max_height, 1.0)

    for stage_idx, stage_column in enumerate(stage_columns):
        color_map = _pfizer_v2_color_map(stage_column, stage_levels[stage_column])
        for level in stage_levels[stage_column]:
            y0, y1 = stage_positions[stage_column][level]
            rect = Rectangle(
                (x_positions[stage_idx] - width / 2.0, y0),
                width,
                y1 - y0,
                facecolor=color_map.get(level, "#BBBBBB"),
                edgecolor="black",
                linewidth=0.7,
                alpha=0.85,
            )
            ax.add_patch(rect)
            ax.text(
                x_positions[stage_idx],
                0.5 * (y0 + y1),
                level,
                ha="center",
                va="center",
                fontsize=8,
                rotation=90,
            )

    for left_stage, right_stage in zip(stage_columns[:-1], stage_columns[1:]):
        flow_counts = pd.crosstab(working[left_stage].astype(str), working[right_stage].astype(str))
        left_offsets = {level: stage_positions[left_stage][level][0] for level in stage_levels[left_stage]}
        right_offsets = {level: stage_positions[right_stage][level][0] for level in stage_levels[right_stage]}
        right_colors = _pfizer_v2_color_map(right_stage, stage_levels[right_stage])
        left_x = float(x_positions[list(stage_columns).index(left_stage)] + width / 2.0)
        right_x = float(x_positions[list(stage_columns).index(right_stage)] - width / 2.0)
        for left_level in stage_levels[left_stage]:
            if left_level not in flow_counts.index:
                continue
            for right_level in stage_levels[right_stage]:
                if right_level not in flow_counts.columns:
                    continue
                count = float(flow_counts.loc[left_level, right_level])
                if count <= 0.0:
                    continue
                left_center = left_offsets[left_level] + 0.5 * count
                right_center = right_offsets[right_level] + 0.5 * count
                ax.plot(
                    [left_x, right_x],
                    [left_center, right_center],
                    color=right_colors.get(right_level, "#999999"),
                    alpha=0.35,
                    linewidth=max(1.0, count * line_scale),
                    solid_capstyle="round",
                    zorder=0,
                )
                left_offsets[left_level] = left_offsets[left_level] + count
                right_offsets[right_level] = right_offsets[right_level] + count

    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    ax.set_ylim(0.0, max_height)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([column.replace("_", " ") for column in stage_columns])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_eot_timing(
    patient_df: pd.DataFrame,
    *,
    out_path: str,
    title: str,
) -> None:
    """Plot end-of-treatment timing per patient, faceted by treatment arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    working = patient_df.copy()
    working["days_from_scn_EOT"] = pd.to_numeric(working["days_from_scn_EOT"], errors="coerce")
    working = working.loc[working["days_from_scn_EOT"].notna()].copy()
    if working.empty:
        return

    arm_levels = [level for level in _PFIZER_ARM_ORDER if level in set(working["arm"].astype(str))]
    if not arm_levels:
        arm_levels = sorted(working["arm"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(arm_levels), figsize=(max(7.0, 4.6 * len(arm_levels)), 5.0), sharey=True)
    if len(arm_levels) == 1:
        axes = [axes]

    response_colors = _pfizer_v2_color_map("response_4class", _PFIZER_RESPONSE_4CLASS_ORDER)
    for ax, arm_name in zip(axes, arm_levels):
        arm_df = working.loc[working["arm"].astype(str) == arm_name].copy()
        arm_df = arm_df.sort_values(["days_from_scn_EOT", "patient_id"]).reset_index()
        x_positions = np.arange(len(arm_df), dtype=float)
        colors = [response_colors.get(str(response), "#999999") for response in arm_df["response_4class"]]
        ax.scatter(
            x_positions,
            arm_df["days_from_scn_EOT"].astype(float).values,
            c=colors,
            edgecolor="black",
            linewidth=0.5,
            s=58,
            zorder=3,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(arm_df["patient_id"].tolist(), rotation=90, fontsize=8)
        ax.set_title(arm_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Patient ID")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axes[0].set_ylabel("Days from SCN at EOT")
    response_levels_present = [
        level for level in _PFIZER_RESPONSE_4CLASS_ORDER
        if level in set(working["response_4class"].astype(str))
    ]
    _pfizer_v2_add_dual_legends(
        axes[-1],
        color_handles=_pfizer_v2_response_legend_handles(response_levels_present),
        color_title="Color = response",
        marker_handles=None,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.88, 0.96])
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_annotated_heatmap(
    matrix_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    *,
    out_path: str,
    title: str,
    zscore_rows: bool = True,
) -> None:
    """Render a patient-by-feature heatmap with compact metadata annotation bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    data = matrix_df.apply(pd.to_numeric, errors="coerce")
    data = data.loc[data.notna().any(axis=1), data.notna().any(axis=0)]
    if data.empty:
        return

    annotation_working = annotation_df.reindex(data.columns).copy()
    ordered_columns = list(data.columns)
    if {"response_4class", "arm", "dose"}.issubset(annotation_working.columns):
        sort_frame = pd.DataFrame(
            {
                "response_4class": annotation_working["response_4class"].astype(str),
                "arm": annotation_working["arm"].astype(str),
                "dose": annotation_working["dose"].astype(str),
                "patient_id": ordered_columns,
            },
            index=ordered_columns,
        )
        ordered_columns = sort_frame.sort_values(["response_4class", "arm", "dose", "patient_id"]).index.tolist()
        data = data[ordered_columns]
        annotation_working = annotation_working.reindex(ordered_columns)

    if zscore_rows:
        row_means = data.mean(axis=1)
        row_sds = data.std(axis=1, ddof=0).replace(0.0, np.nan)
        data = data.sub(row_means, axis=0).div(row_sds, axis=0)

    annotation_columns = list(annotation_working.columns)
    fig = plt.figure(figsize=(max(10.0, 0.52 * data.shape[1] + 2.5), max(6.0, 0.28 * data.shape[0] + 2.4 + 0.35 * len(annotation_columns))))
    gs = gridspec.GridSpec(
        len(annotation_columns) + 1,
        1,
        height_ratios=[0.28] * len(annotation_columns) + [max(4.5, 0.32 * data.shape[0])],
        hspace=0.05,
    )

    legend_handles: List[Patch] = []
    legend_labels: set[str] = set()
    for ann_idx, ann_column in enumerate(annotation_columns):
        ax_ann = fig.add_subplot(gs[ann_idx])
        series = annotation_working[ann_column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(3, len(series) // 2):
            values = numeric.values[np.newaxis, :]
            ax_ann.imshow(values, aspect="auto", cmap="viridis", interpolation="nearest")
        else:
            levels = _pfizer_v2_preferred_levels(ann_column) or [level for level in series.astype(str).tolist() if level != "nan"]
            levels = [level for level in dict.fromkeys(levels) if level in set(series.astype(str))]
            if not levels:
                levels = sorted(series.astype(str).unique().tolist())
            color_map = _pfizer_v2_color_map(ann_column, levels)
            level_to_idx = {level: idx for idx, level in enumerate(levels)}
            arr = np.array([[level_to_idx.get(str(value), np.nan) for value in series]], dtype=float)
            cmap = ListedColormap([color_map.get(level, "#999999") for level in levels])
            cmap.set_bad("#F2F2F2")
            ax_ann.imshow(arr, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=max(len(levels) - 1, 1))
            for level in levels:
                legend_label = f"{ann_column}: {level}"
                if legend_label in legend_labels:
                    continue
                legend_handles.append(Patch(facecolor=color_map.get(level, "#999999"), edgecolor="black", label=legend_label))
                legend_labels.add(legend_label)
        ax_ann.set_yticks([0])
        ax_ann.set_yticklabels([ann_column.replace("_", " ")], fontsize=8)
        ax_ann.set_xticks([])
        for spine in ax_ann.spines.values():
            spine.set_visible(False)

    ax_main = fig.add_subplot(gs[-1])
    sns.heatmap(
        data,
        ax=ax_main,
        cmap="vlag",
        linewidths=0.15,
        linecolor="#F2F2F2",
        cbar_kws={"label": "Row z-score" if zscore_rows else "Value", "shrink": 0.7},
    )
    ax_main.set_xlabel("Patient")
    ax_main.set_ylabel("Feature")
    ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=90, fontsize=8)
    ax_main.set_yticklabels(ax_main.get_yticklabels(), rotation=0, fontsize=8)
    ax_main.set_title(title, fontsize=12, fontweight="bold")
    if legend_handles:
        ax_main.legend(
            handles=legend_handles,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
        )
    fig.subplots_adjust(
        left=0.08,
        right=0.80 if legend_handles else 0.96,
        top=0.96,
        bottom=0.08,
        hspace=0.05,
    )
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_scatter(
    data_df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    out_path: str,
    title: str,
    annotate_points: bool = True,
) -> None:
    """Plot a baseline Pfizer feature scatter with response color and arm marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = [x_column, y_column, "response_4class", "arm"]
    if any(column not in data_df.columns for column in required):
        return
    working = data_df[required].copy()
    if "patient_id" in data_df.columns:
        working["patient_id"] = data_df["patient_id"]
    working[x_column] = pd.to_numeric(working[x_column], errors="coerce")
    working[y_column] = pd.to_numeric(working[y_column], errors="coerce")
    working = working.dropna(subset=[x_column, y_column])
    if working.empty:
        return

    response_colors = _pfizer_v2_color_map("response_4class", _PFIZER_RESPONSE_4CLASS_ORDER)
    marker_map = {"Enza-only": "o", "EZH2i+Enza": "s"}
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for _, row in working.iterrows():
        color = response_colors.get(str(row["response_4class"]), "#999999")
        marker = marker_map.get(str(row["arm"]), "o")
        ax.scatter(
            float(row[x_column]),
            float(row[y_column]),
            s=70,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            marker=marker,
            alpha=0.9,
        )
        if annotate_points and "patient_id" in row.index:
            ax.text(float(row[x_column]), float(row[y_column]), str(row["patient_id"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(color="#E5E5E5", linewidth=0.8)
    response_levels_present = [
        level for level in _PFIZER_RESPONSE_4CLASS_ORDER
        if level in set(working["response_4class"].astype(str))
    ]
    arm_levels_present = [
        level for level in _PFIZER_ARM_ORDER
        if level in set(working["arm"].astype(str))
    ]
    _pfizer_v2_add_dual_legends(
        ax,
        color_handles=_pfizer_v2_response_legend_handles(response_levels_present),
        color_title="Color = response",
        marker_handles=[
            handle for handle in _pfizer_v2_arm_marker_legend_handles()
            if handle.get_label() in arm_levels_present
        ],
        marker_title="Marker = arm",
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_group_box(
    data_df: pd.DataFrame,
    *,
    feature_name: str,
    group_column: str,
    out_path: str,
    title: str,
    y_label: Optional[str] = None,
    p_value: Optional[float] = None,
) -> None:
    """Plot a grouped boxplot with four-class response-colored patient jitter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if feature_name not in data_df.columns or group_column not in data_df.columns:
        return
    working = data_df[[feature_name, group_column, "response_4class"]].copy()
    if "arm" in data_df.columns:
        working["arm"] = data_df["arm"]
    working[feature_name] = pd.to_numeric(working[feature_name], errors="coerce")
    working = working.dropna(subset=[feature_name, group_column])
    if working.empty:
        return

    group_levels = _pfizer_v2_preferred_levels(group_column) or sorted(working[group_column].astype(str).unique().tolist())
    group_levels = [level for level in group_levels if level in set(working[group_column].astype(str))]
    if len(group_levels) < 1:
        return
    group_colors = _pfizer_v2_color_map(group_column, group_levels)
    response_colors = _pfizer_v2_color_map("response_4class", _PFIZER_RESPONSE_4CLASS_ORDER)
    marker_map = {"Enza-only": "o", "EZH2i+Enza": "s"}

    grouped_values = [
        working.loc[working[group_column].astype(str) == level, feature_name].astype(float).values
        for level in group_levels
    ]
    if sum(len(vals) > 0 for vals in grouped_values) < 1:
        return

    fig, ax = plt.subplots(figsize=(max(5.2, 1.45 * len(group_levels)), 4.8))
    bp = ax.boxplot(
        grouped_values,
        positions=np.arange(len(group_levels), dtype=float),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, level in zip(bp["boxes"], group_levels):
        patch.set_facecolor(group_colors.get(level, "#BBBBBB"))
        patch.set_alpha(0.45)
        patch.set_edgecolor("black")

    rng = np.random.default_rng(42)
    for pos, level in enumerate(group_levels):
        group_rows = working.loc[working[group_column].astype(str) == level]
        if group_rows.empty:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(group_rows))
        for offset, (_, row) in zip(jitter, group_rows.iterrows()):
            ax.scatter(
                float(pos + offset),
                float(row[feature_name]),
                color=response_colors.get(str(row["response_4class"]), "#999999"),
                edgecolor="black",
                linewidth=0.5,
                marker=marker_map.get(str(row.get("arm", "Enza-only")), "o"),
                s=46,
                alpha=0.9,
                zorder=3,
            )

    if p_value is not None and np.isfinite(p_value) and len(group_levels) == 2:
        y_min, y_max = ax.get_ylim()
        y_range = max(y_max - y_min, 1e-6)
        y_ann = float(np.nanmax(working[feature_name])) + 0.10 * y_range
        cap_y = 0.02 * y_range
        ax.plot([0, 0, 1, 1], [y_ann - cap_y, y_ann, y_ann, y_ann - cap_y], color="black", linewidth=1.0)
        ax.text(0.5, y_ann + 0.01 * y_range, format_pval(float(p_value)), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(y_min, max(y_max, y_ann + 0.10 * y_range))

    ax.set_xticks(np.arange(len(group_levels), dtype=float))
    ax.set_xticklabels(group_levels, rotation=20, ha="right")
    ax.set_ylabel(y_label or feature_name)
    ax.set_title(title, fontsize=12, fontweight="bold")
    response_levels_present = [
        level for level in _PFIZER_RESPONSE_4CLASS_ORDER
        if level in set(working["response_4class"].astype(str))
    ]
    arm_levels_present = []
    if "arm" in working.columns:
        arm_levels_present = [
            level for level in _PFIZER_ARM_ORDER
            if level in set(working["arm"].astype(str))
        ]
    _pfizer_v2_add_dual_legends(
        ax,
        color_handles=_pfizer_v2_response_legend_handles(response_levels_present),
        color_title="Point color = response",
        marker_handles=[
            handle for handle in _pfizer_v2_arm_marker_legend_handles()
            if handle.get_label() in arm_levels_present
        ],
        marker_title="Point marker = arm",
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_trajectory(
    sample_df: pd.DataFrame,
    *,
    feature_name: str,
    out_path: str,
    title: str,
    visit_order: Sequence[str],
    facet_by_arm: bool = False,
) -> None:
    """Plot per-patient trajectories across ordered Pfizer visits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if feature_name not in sample_df.columns:
        return
    working = sample_df[["patient_id", "visit", "arm", "response_4class", feature_name]].copy()
    working[feature_name] = pd.to_numeric(working[feature_name], errors="coerce")
    working = working.loc[working[feature_name].notna()].copy()
    working = working.loc[working["visit"].astype(str).isin(list(visit_order))].copy()
    if working.empty:
        return

    response_colors = _pfizer_v2_color_map("response_4class", _PFIZER_RESPONSE_4CLASS_ORDER)
    arm_levels = [level for level in _PFIZER_ARM_ORDER if level in set(working["arm"].astype(str))]
    if not arm_levels:
        arm_levels = sorted(working["arm"].astype(str).unique().tolist())
    facets = arm_levels if facet_by_arm else ["all"]
    fig, axes = plt.subplots(1, len(facets), figsize=(max(6.0, 5.0 * len(facets)), 4.8), sharey=True)
    if len(facets) == 1:
        axes = [axes]

    visit_to_x = {visit: idx for idx, visit in enumerate(visit_order)}
    working["visit_rank"] = working["visit"].astype(str).map(visit_to_x)
    for ax, facet in zip(axes, facets):
        facet_df = working if facet == "all" else working.loc[working["arm"].astype(str) == facet].copy()
        for patient_id, patient_rows in facet_df.groupby("patient_id"):
            patient_rows = patient_rows.sort_values("visit_rank")
            x_vals = [visit_to_x[str(visit)] for visit in patient_rows["visit"].astype(str)]
            y_vals = patient_rows[feature_name].astype(float).tolist()
            color = response_colors.get(str(patient_rows["response_4class"].iloc[0]), "#999999")
            visit_ranks = patient_rows["visit_rank"].astype(int).tolist()
            if len(x_vals) > 1:
                for idx in range(len(x_vals) - 1):
                    if visit_ranks[idx + 1] - visit_ranks[idx] != 1:
                        continue
                    ax.plot(
                        x_vals[idx:idx + 2],
                        y_vals[idx:idx + 2],
                        color=color,
                        linewidth=1.3,
                        alpha=0.6,
                        zorder=1,
                    )
            for visit_name, x_val, y_val in zip(patient_rows["visit"].astype(str), x_vals, y_vals):
                ax.scatter(
                    x_val,
                    y_val,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    marker=_PFIZER_VISIT_MARKER_MAP.get(visit_name, "o"),
                    s=45,
                    zorder=3,
                )
        ax.set_xticks(list(visit_to_x.values()))
        ax.set_xticklabels(list(visit_order))
        ax.set_xlabel("Visit")
        if facet != "all":
            ax.set_title(facet, fontsize=10, fontweight="bold")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    axes[0].set_ylabel(feature_name)
    response_levels_present = [
        level for level in _PFIZER_RESPONSE_4CLASS_ORDER
        if level in set(working["response_4class"].astype(str))
    ]
    _pfizer_v2_add_dual_legends(
        axes[-1],
        color_handles=_pfizer_v2_response_legend_handles(response_levels_present),
        color_title="Color = response",
        marker_handles=_pfizer_v2_visit_marker_legend_handles(visit_order),
        marker_title="Marker = visit",
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 0.96])
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_waterfall(
    delta_df: pd.DataFrame,
    *,
    feature_name: str,
    out_path: str,
    title: str,
    y_label: str,
) -> None:
    """Plot a patient-ordered waterfall chart for one delta feature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if feature_name not in delta_df.columns:
        return
    working = delta_df[[feature_name, "response_4class", "arm"]].copy()
    working[feature_name] = pd.to_numeric(working[feature_name], errors="coerce")
    working = working.dropna(subset=[feature_name])
    if working.empty:
        return
    working = working.sort_values(feature_name, ascending=False)

    response_colors = _pfizer_v2_color_map("response_4class", _PFIZER_RESPONSE_4CLASS_ORDER)
    marker_map = {"Enza-only": "o", "EZH2i+Enza": "s"}
    x_positions = np.arange(len(working), dtype=float)
    bar_colors = [response_colors.get(str(response), "#999999") for response in working["response_4class"]]

    fig, ax = plt.subplots(figsize=(max(7.0, 0.48 * len(working) + 2.0), 4.8))
    ax.bar(x_positions, working[feature_name].astype(float).values, color=bar_colors, edgecolor="black", linewidth=0.5)
    for x_pos, (_, row) in zip(x_positions, working.iterrows()):
        ax.scatter(
            x_pos,
            float(row[feature_name]),
            marker=marker_map.get(str(row["arm"]), "o"),
            color="white",
            edgecolor="black",
            linewidth=0.8,
            s=42,
            zorder=4,
        )
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(working.index.tolist(), rotation=90, fontsize=8)
    ax.set_xlabel("Patient ID")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12, fontweight="bold")
    response_levels_present = [
        level for level in _PFIZER_RESPONSE_4CLASS_ORDER
        if level in set(working["response_4class"].astype(str))
    ]
    arm_levels_present = [
        level for level in _PFIZER_ARM_ORDER
        if level in set(working["arm"].astype(str))
    ]
    _pfizer_v2_add_dual_legends(
        ax,
        color_handles=_pfizer_v2_response_legend_handles(response_levels_present),
        color_title="Bar color = response",
        marker_handles=[
            handle for handle in _pfizer_v2_arm_marker_legend_handles()
            if handle.get_label() in arm_levels_present
        ],
        marker_title="Overlay marker = arm",
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_effect_forest(
    stats_df: pd.DataFrame,
    *,
    effect_column: str,
    lower_column: str,
    upper_column: str,
    out_path: str,
    title: str,
) -> None:
    """Plot a generic horizontal effect-size forest plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {"feature", effect_column, lower_column, upper_column}
    if stats_df.empty or not required.issubset(stats_df.columns):
        return
    working = stats_df[["feature", effect_column, lower_column, upper_column, "fdr", "p_value"]].copy()
    working = working.dropna(subset=[effect_column]).sort_values(effect_column)
    if working.empty:
        return

    fig_h = max(4.0, 0.34 * len(working) + 1.8)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    y_pos = np.arange(len(working), dtype=float)
    sig_mask = pd.to_numeric(working["fdr"], errors="coerce") < 0.10
    colors = ["#B63A3A" if sig else "#666666" for sig in sig_mask.fillna(False)]
    for idx, (_, row) in enumerate(working.iterrows()):
        ax.plot(
            [float(row[lower_column]), float(row[upper_column])],
            [y_pos[idx], y_pos[idx]],
            color=colors[idx],
            linewidth=1.6,
        )
        ax.scatter(float(row[effect_column]), y_pos[idx], color=colors[idx], edgecolor="black", linewidth=0.5, s=52, zorder=3)
    ax.axvline(0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(working["feature"].tolist(), fontsize=8)
    ax.set_xlabel(effect_column.replace("_", " "))
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_pam50_tile(
    sample_df: pd.DataFrame,
    *,
    out_path: str,
    title: str,
) -> None:
    """Plot a patient-by-visit PAM50 subtype tile map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if "PAM50 subtype" not in sample_df.columns:
        return
    working = sample_df[["patient_id", "visit", "PAM50 subtype"]].dropna().copy()
    if working.empty:
        return
    subtype_levels = sorted(working["PAM50 subtype"].astype(str).unique().tolist())
    patient_levels = sorted(working["patient_id"].astype(str).unique().tolist())
    value_to_idx = {level: idx for idx, level in enumerate(subtype_levels)}
    matrix = np.full((len(patient_levels), len(_PFIZER_VISIT_ORDER)), np.nan, dtype=float)
    for _, row in working.iterrows():
        patient_idx = patient_levels.index(str(row["patient_id"]))
        visit_idx = _PFIZER_VISIT_ORDER.index(str(row["visit"]))
        matrix[patient_idx, visit_idx] = value_to_idx.get(str(row["PAM50 subtype"]), np.nan)

    color_map = _pfizer_v2_color_map("PAM50 subtype", subtype_levels)
    cmap = ListedColormap([color_map[level] for level in subtype_levels])
    cmap.set_bad("#F0F0F0")
    fig, ax = plt.subplots(figsize=(6.2, max(4.5, 0.36 * len(patient_levels) + 1.8)))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=max(len(subtype_levels) - 1, 1))
    del im
    ax.set_xticks(np.arange(len(_PFIZER_VISIT_ORDER), dtype=float))
    ax.set_xticklabels(_PFIZER_VISIT_ORDER)
    ax.set_yticks(np.arange(len(patient_levels), dtype=float))
    ax.set_yticklabels(patient_levels, fontsize=8)
    ax.set_xlabel("Visit")
    ax.set_ylabel("Patient")
    ax.set_title(title, fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map[level], edgecolor="black", label=level)
        for level in subtype_levels
    ]
    ax.legend(
        handles=legend_handles,
        title="Tile color = PAM50 subtype",
        fontsize=8,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_embedding(
    embedding_df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    color_column: str,
    out_path: str,
    title: str,
    continuous_color: bool = False,
) -> None:
    """Plot a PCA or UMAP embedding with patient trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = {x_column, y_column, "patient_id", "visit", color_column}
    if not required.issubset(embedding_df.columns):
        return
    working = embedding_df.copy()
    working[x_column] = pd.to_numeric(working[x_column], errors="coerce")
    working[y_column] = pd.to_numeric(working[y_column], errors="coerce")
    working = working.dropna(subset=[x_column, y_column])
    if working.empty:
        return

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if continuous_color:
        color_values = pd.to_numeric(working[color_column], errors="coerce")
        scatter = ax.scatter(
            working[x_column].astype(float),
            working[y_column].astype(float),
            c=color_values,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.4,
            s=58,
            alpha=0.9,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.8, label=color_column)
    else:
        color_levels = sorted(working[color_column].astype(str).unique().tolist())
        color_map = _pfizer_v2_color_map(color_column, color_levels)
        for level in color_levels:
            sub = working.loc[working[color_column].astype(str) == level]
            ax.scatter(
                sub[x_column].astype(float),
                sub[y_column].astype(float),
                color=color_map.get(level, "#999999"),
                edgecolor="black",
                linewidth=0.4,
                s=58,
                alpha=0.9,
                label=level,
            )
        ax.legend(title=f"Color = {color_column.replace('_', ' ')}", fontsize=8, loc="best")

    for _, patient_rows in working.groupby("patient_id"):
        patient_rows = patient_rows.sort_values("visit")
        if len(patient_rows) < 2:
            continue
        ax.plot(
            patient_rows[x_column].astype(float),
            patient_rows[y_column].astype(float),
            color="#666666",
            linewidth=1.0,
            alpha=0.45,
            zorder=0,
        )
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(color="#E5E5E5", linewidth=0.8)
    if working["patient_id"].nunique() > 1:
        from matplotlib.lines import Line2D
        trajectory_handle = Line2D([0], [0], color="#666666", linewidth=1.0, alpha=0.45, label="Patient trajectory")
        existing_legend = ax.get_legend()
        if existing_legend is not None:
            handles, labels = ax.get_legend_handles_labels()
            handles = list(handles) + [trajectory_handle]
            labels = list(labels) + [trajectory_handle.get_label()]
            title_text = existing_legend.get_title().get_text()
            existing_legend.remove()
            ax.legend(handles, labels, title=title_text, fontsize=8, loc="best")
        else:
            ax.legend(handles=[trajectory_handle], fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_present_features(columns: Sequence[str], requested: Sequence[str]) -> List[str]:
    """Keep requested Pfizer features that are present in the current table."""
    available = set(columns)
    return [feature for feature in requested if feature in available]


def _pfizer_v2_run_metadata_overview(
    sample_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Generate the descriptive cohort-overview outputs for Pfizer."""
    module_out = ensure_dir(os.path.join(out_dir, "01_metadata_overview"))
    sample_df.to_csv(os.path.join(module_out, "matched_sample_metadata.tsv"), sep="\t")
    patient_df.to_csv(os.path.join(module_out, "patient_level_metadata.tsv"), sep="\t")

    _maybe(
        _pfizer_v2_plot_alluvial,
        overwrite,
        patient_df,
        stage_columns=["enza_history", "dose", "response_4class"],
        out_path=os.path.join(module_out, "alluvial_enza_history_to_dose_to_response"),
        title="CRPC-Pfizer: Enza history → dose → response",
    )
    _maybe(
        _pfizer_v2_plot_stacked_bar,
        overwrite,
        patient_df,
        group_column="arm",
        fill_column="response_4class",
        out_path=os.path.join(module_out, "response_by_arm"),
        title="Response by arm",
    )
    _maybe(
        _pfizer_v2_plot_stacked_bar,
        overwrite,
        patient_df,
        group_column="dose",
        fill_column="response_4class",
        out_path=os.path.join(module_out, "response_by_dose"),
        title="Response by dose",
    )
    _maybe(
        _pfizer_v2_plot_stacked_bar,
        overwrite,
        patient_df,
        group_column="enza_history",
        fill_column="response_4class",
        out_path=os.path.join(module_out, "response_by_enza_history"),
        title="Response by Enza history",
    )
    _maybe(
        _pfizer_v2_plot_eot_timing,
        overwrite,
        patient_df,
        out_path=os.path.join(module_out, "eot_timing_by_patient"),
        title="End-of-treatment timing by patient and arm",
    )


def _pfizer_v2_run_baseline_landscape(
    sample_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Generate baseline-state heatmaps, scatters, and primary boxplots."""
    module_out = ensure_dir(os.path.join(out_dir, "02_baseline_landscape"))
    baseline_df = sample_df.loc[sample_df["visit"].astype(str) == "SCN"].copy()
    if baseline_df.empty:
        return
    baseline_df = baseline_df.set_index("patient_id", drop=False)
    baseline_df.to_csv(os.path.join(module_out, "baseline_patient_table.tsv"), sep="\t")

    heatmap_features = _pfizer_v2_present_features(
        baseline_df.columns,
        ["AR score", "NE score", "CCP score", "PRC2 target derepression score (exploratory)"]
        + _PFIZER_BASELINE_HALLMARKS
        + _PFIZER_BASELINE_EXPRESSION_GENES,
    )
    annotation_df = pd.DataFrame(
        {
            "response_4class": baseline_df["response_4class"].astype(str),
            "progression_binary": baseline_df["progression_binary"].astype(str),
            "arm": baseline_df["arm"].astype(str),
            "dose": baseline_df["dose"].astype(str),
            "enza_history": baseline_df["enza_history"].astype(str),
            "PAM50 subtype": baseline_df.get("PAM50 subtype", pd.Series(index=baseline_df.index, dtype="string")).astype(str),
            "days_from_scn_EOT": patient_df.reindex(baseline_df.index)["days_from_scn_EOT"],
        },
        index=baseline_df.index,
    )
    if heatmap_features:
        heatmap_matrix = baseline_df[heatmap_features].T
        _maybe(
            _pfizer_v2_plot_annotated_heatmap,
            overwrite,
            heatmap_matrix,
            annotation_df,
            out_path=os.path.join(module_out, "baseline_feature_heatmap"),
            title="CRPC-Pfizer baseline transcriptional landscape",
        )

    scatter_specs = [
        ("AR score", "NE score", "AR score vs NE score"),
        ("AR score", "CCP score", "AR score vs CCP score"),
        ("NE score", "CCP score", "NE score vs CCP score"),
        ("AR score", "HALLMARK_ANDROGEN_RESPONSE", "AR score vs androgen response"),
        ("CCP score", "HALLMARK_E2F_TARGETS", "CCP score vs E2F targets"),
    ]
    for x_column, y_column, scatter_title in scatter_specs:
        if x_column not in baseline_df.columns or y_column not in baseline_df.columns:
            continue
        _maybe(
            _pfizer_v2_plot_scatter,
            overwrite,
            baseline_df.reset_index(drop=True),
            x_column=x_column,
            y_column=y_column,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{x_column}_vs_{y_column}")),
            title=scatter_title,
        )

    baseline_stats = _pfizer_v2_two_group_stats(
        baseline_df,
        _pfizer_v2_present_features(baseline_df.columns, ["AR score", "NE score", "CCP score"]),
        group_column="progression_binary",
        group_a="PD",
        group_b="non-PD",
        comparison_name="baseline_PD_vs_nonPD",
    )
    if not baseline_stats.empty:
        baseline_stats.to_csv(os.path.join(module_out, "baseline_primary_boxplot_stats.tsv"), sep="\t", index=False)
        for feature_name in baseline_stats["feature"].tolist():
            row = baseline_stats.loc[baseline_stats["feature"] == feature_name].iloc[0]
            _maybe(
                _pfizer_v2_plot_group_box,
                overwrite,
                baseline_df.reset_index(drop=True),
                feature_name=feature_name,
                group_column="progression_binary",
                out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_by_progression_binary")),
                title=f"{feature_name} at SCN by PD vs non-PD",
                y_label=feature_name,
                p_value=float(row["p_value"]),
            )


def _pfizer_v2_run_pam50_lineage_analysis(
    sample_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Summarize PAM50 subtype distributions and lineage-state relationships."""
    module_out = ensure_dir(os.path.join(out_dir, "03_pam50_lineage_state"))
    baseline_df = sample_df.loc[sample_df["visit"].astype(str) == "SCN"].copy()
    if baseline_df.empty or "PAM50 subtype" not in baseline_df.columns:
        return
    baseline_df = baseline_df.set_index("patient_id", drop=False)
    baseline_df.to_csv(os.path.join(module_out, "baseline_pam50_table.tsv"), sep="\t")

    for group_column, plot_name, plot_title in [
        ("response_4class", "pam50_by_response", "PAM50 subtype by response"),
        ("arm", "pam50_by_arm", "PAM50 subtype by arm"),
        ("enza_history", "pam50_by_enza_history", "PAM50 subtype by Enza history"),
    ]:
        _maybe(
            _pfizer_v2_plot_stacked_bar,
            overwrite,
            baseline_df,
            group_column=group_column,
            fill_column="PAM50 subtype",
            out_path=os.path.join(module_out, plot_name),
            title=plot_title,
        )

    for feature_name in _pfizer_v2_present_features(
        baseline_df.columns,
        ["AR score", "NE score", "CCP score", "HALLMARK_ANDROGEN_RESPONSE", "HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT"],
    ):
        _maybe(
            _pfizer_v2_plot_group_box,
            overwrite,
            baseline_df.reset_index(drop=True),
            feature_name=feature_name,
            group_column="PAM50 subtype",
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_by_pam50_subtype")),
            title=f"{feature_name} at SCN by PAM50 subtype",
            y_label=feature_name,
        )

    _maybe(
        _pfizer_v2_plot_pam50_tile,
        overwrite,
        sample_df,
        out_path=os.path.join(module_out, "pam50_trajectory_tile"),
        title="PAM50 subtype trajectory across visits",
    )


def _pfizer_v2_run_baseline_candidate_analysis(
    sample_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Run baseline PD-vs-non-PD and strict-benefit candidate biomarker summaries."""
    module_out = ensure_dir(os.path.join(out_dir, "04_baseline_candidate_biomarkers"))
    baseline_df = sample_df.loc[sample_df["visit"].astype(str) == "SCN"].copy()
    if baseline_df.empty:
        return
    baseline_df = baseline_df.set_index("patient_id", drop=False)

    optional_genes = ["EZH2", "AR", "KLK3", "FKBP5", "NKX3-1", "ASCL1", "INSM1", "SYP", "CHGA", "SOX2", "MYCN", "MKI67", "TOP2A", "BIRC5", "AURKA"]
    primary_features = _pfizer_v2_present_features(
        baseline_df.columns,
        _PFIZER_PRESPECIFIED_BASELINE_FEATURES + optional_genes,
    )
    primary_stats = _pfizer_v2_two_group_stats(
        baseline_df,
        primary_features,
        group_column="progression_binary",
        group_a="PD",
        group_b="non-PD",
        comparison_name="SCN_PD_vs_nonPD",
    )
    if not primary_stats.empty:
        primary_stats.to_csv(os.path.join(module_out, "scn_pd_vs_nonpd_stats.tsv"), sep="\t", index=False)
        _maybe(
            _pfizer_v2_plot_effect_forest,
            overwrite,
            primary_stats,
            effect_column="median_difference_group_b_minus_group_a",
            lower_column="median_difference_ci_lo",
            upper_column="median_difference_ci_hi",
            out_path=os.path.join(module_out, "scn_pd_vs_nonpd_forest"),
            title="Baseline effect sizes: PD vs non-PD",
        )
        for _, row in primary_stats.iterrows():
            _maybe(
                _pfizer_v2_plot_group_box,
                overwrite,
                baseline_df.reset_index(drop=True),
                feature_name=str(row["feature"]),
                group_column="progression_binary",
                out_path=os.path.join(module_out, _pfizer_safe_name(f"{row['feature']}_scn_pd_vs_nonpd")),
                title=f"{row['feature']} at SCN: PD vs non-PD",
                y_label=str(row["feature"]),
                p_value=float(row["p_value"]),
            )

    strict_df = baseline_df.loc[baseline_df["clinical_benefit_strict"].astype(str).isin(["non-benefit", "benefit"])].copy()
    strict_stats = _pfizer_v2_two_group_stats(
        strict_df,
        primary_features,
        group_column="clinical_benefit_strict",
        group_a="non-benefit",
        group_b="benefit",
        comparison_name="SCN_strict_benefit_vs_PD",
    )
    if not strict_stats.empty:
        strict_stats.to_csv(os.path.join(module_out, "scn_strict_benefit_vs_pd_stats.tsv"), sep="\t", index=False)
        _maybe(
            _pfizer_v2_plot_effect_forest,
            overwrite,
            strict_stats,
            effect_column="median_difference_group_b_minus_group_a",
            lower_column="median_difference_ci_lo",
            upper_column="median_difference_ci_hi",
            out_path=os.path.join(module_out, "scn_strict_benefit_vs_pd_forest"),
            title="Baseline effect sizes: strict benefit vs PD",
        )


def _pfizer_v2_run_c3d1_pharmacodynamics(
    sample_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> pd.DataFrame:
    """Summarize paired SCN→C3D1 pharmacodynamic changes for all treated patients."""
    module_out = ensure_dir(os.path.join(out_dir, "05_c3d1_pharmacodynamics"))
    feature_list = _pfizer_v2_present_features(patient_df.columns, [f"{feature}__SCN" for feature in _PFIZER_C3D1_CORE_FEATURES])
    feature_names = [feature.rsplit("__", 1)[0] for feature in feature_list]
    delta_df = _pfizer_v2_build_delta_frame(
        patient_df,
        feature_names,
        start_visit="SCN",
        end_visit="C3D1",
        delta_column_name="C3D1_minus_SCN",
    )
    if delta_df.empty:
        return delta_df
    delta_df.to_csv(os.path.join(module_out, "c3d1_minus_scn_deltas.tsv"), sep="\t")

    stats_df = _pfizer_v2_paired_delta_stats(delta_df, feature_names, comparison_name="C3D1_minus_SCN_all_patients")
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(module_out, "c3d1_minus_scn_summary.tsv"), sep="\t", index=False)
        _maybe(
            _pfizer_v2_plot_effect_forest,
            overwrite,
            stats_df,
            effect_column="median_delta",
            lower_column="median_delta_ci_lo",
            upper_column="median_delta_ci_hi",
            out_path=os.path.join(module_out, "c3d1_minus_scn_forest"),
            title="C3D1 pharmacodynamic median deltas",
        )

    for feature_name in feature_names:
        _maybe(
            _pfizer_v2_plot_trajectory,
            overwrite,
            sample_df,
            feature_name=feature_name,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_trajectory_scn_c3d1")),
            title=f"{feature_name}: SCN → C3D1",
            visit_order=["SCN", "C3D1"],
            facet_by_arm=True,
        )
        _maybe(
            _pfizer_v2_plot_group_box,
            overwrite,
            delta_df.reset_index(drop=True),
            feature_name=feature_name,
            group_column="arm",
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_delta_by_arm")),
            title=f"Δ {feature_name} (C3D1 - SCN) by arm",
            y_label=f"Δ {feature_name} (C3D1 - SCN)",
        )
        _maybe(
            _pfizer_v2_plot_waterfall,
            overwrite,
            delta_df,
            feature_name=feature_name,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_waterfall")),
            title=f"Δ {feature_name} (C3D1 - SCN)",
            y_label=f"Δ {feature_name} (C3D1 - SCN)",
        )
    return delta_df


def _pfizer_v2_run_arm_delta_analysis(
    delta_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Compare SCN→C3D1 deltas between Enza-only and EZH2i+Enza arms."""
    module_out = ensure_dir(os.path.join(out_dir, "06_arm_delta_comparison"))
    if delta_df.empty:
        return
    feature_names = _pfizer_v2_present_features(delta_df.columns, _PFIZER_C3D1_CORE_FEATURES)
    stats_df = _pfizer_v2_two_group_stats(
        delta_df,
        feature_names,
        group_column="arm",
        group_a="Enza-only",
        group_b="EZH2i+Enza",
        comparison_name="C3D1_delta_combo_vs_enza",
    )
    if stats_df.empty:
        return
    stats_df.to_csv(os.path.join(module_out, "c3d1_delta_by_arm_stats.tsv"), sep="\t", index=False)
    _maybe(
        _pfizer_v2_plot_effect_forest,
        overwrite,
        stats_df,
        effect_column="median_difference_group_b_minus_group_a",
        lower_column="median_difference_ci_lo",
        upper_column="median_difference_ci_hi",
        out_path=os.path.join(module_out, "c3d1_delta_by_arm_forest"),
        title="C3D1 delta-of-delta: EZH2i+Enza vs Enza-only",
    )
    for _, row in stats_df.iterrows():
        _maybe(
            _pfizer_v2_plot_group_box,
            overwrite,
            delta_df.reset_index(drop=True),
            feature_name=str(row["feature"]),
            group_column="arm",
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{row['feature']}_delta_by_arm")),
            title=f"Δ {row['feature']} by arm",
            y_label=f"Δ {row['feature']} (C3D1 - SCN)",
            p_value=float(row["p_value"]),
        )


def _pfizer_v2_run_prc2_target_analysis(
    sample_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Generate dedicated exploratory PRC2-derepression score outputs."""
    module_out = ensure_dir(os.path.join(out_dir, "07_prc2_target_score"))
    feature_name = "PRC2 target derepression score (exploratory)"
    if feature_name not in sample_df.columns:
        return
    _maybe(
        _pfizer_v2_plot_trajectory,
        overwrite,
        sample_df,
        feature_name=feature_name,
        out_path=os.path.join(module_out, "prc2_score_trajectory"),
        title="Exploratory PRC2 target derepression score across visits",
        visit_order=["SCN", "C3D1", "EOT"],
        facet_by_arm=True,
    )
    delta_df = _pfizer_v2_build_delta_frame(
        patient_df,
        [feature_name],
        start_visit="SCN",
        end_visit="C3D1",
        delta_column_name="C3D1_minus_SCN",
    )
    if delta_df.empty:
        return
    delta_df.to_csv(os.path.join(module_out, "prc2_c3d1_minus_scn.tsv"), sep="\t")
    arm_stats = _pfizer_v2_two_group_stats(
        delta_df,
        [feature_name],
        group_column="arm",
        group_a="Enza-only",
        group_b="EZH2i+Enza",
        comparison_name="PRC2_delta_by_arm",
    )
    response_stats = _pfizer_v2_two_group_stats(
        delta_df,
        [feature_name],
        group_column="progression_binary",
        group_a="PD",
        group_b="non-PD",
        comparison_name="PRC2_delta_by_response",
    )
    if not arm_stats.empty:
        arm_stats.to_csv(os.path.join(module_out, "prc2_delta_by_arm_stats.tsv"), sep="\t", index=False)
        _maybe(
            _pfizer_v2_plot_group_box,
            overwrite,
            delta_df.reset_index(drop=True),
            feature_name=feature_name,
            group_column="arm",
            out_path=os.path.join(module_out, "prc2_delta_by_arm"),
            title="Exploratory PRC2 score delta by arm",
            y_label=f"Δ {feature_name} (C3D1 - SCN)",
            p_value=float(arm_stats.iloc[0]["p_value"]),
        )
    if not response_stats.empty:
        response_stats.to_csv(os.path.join(module_out, "prc2_delta_by_response_stats.tsv"), sep="\t", index=False)
        _maybe(
            _pfizer_v2_plot_group_box,
            overwrite,
            delta_df.reset_index(drop=True),
            feature_name=feature_name,
            group_column="progression_binary",
            out_path=os.path.join(module_out, "prc2_delta_by_response"),
            title="Exploratory PRC2 score delta by PD vs non-PD",
            y_label=f"Δ {feature_name} (C3D1 - SCN)",
            p_value=float(response_stats.iloc[0]["p_value"]),
        )
    _maybe(
        _pfizer_v2_plot_waterfall,
        overwrite,
        delta_df,
        feature_name=feature_name,
        out_path=os.path.join(module_out, "prc2_delta_waterfall"),
        title="Exploratory PRC2 score delta waterfall",
        y_label=f"Δ {feature_name} (C3D1 - SCN)",
    )


def _pfizer_v2_run_composite_analysis(
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> pd.DataFrame:
    """Compute and plot the prespecified C3D1 pharmacodynamic response composite."""
    module_out = ensure_dir(os.path.join(out_dir, "08_c3d1_composite"))
    component_features = [feature for feature, _ in _PFIZER_COMPOSITE_COMPONENTS]
    component_features = _pfizer_v2_present_features(patient_df.columns, [f"{feature}__SCN" for feature in component_features])
    component_names = [feature.rsplit("__", 1)[0] for feature in component_features]
    delta_df = _pfizer_v2_build_delta_frame(
        patient_df,
        component_names,
        start_visit="SCN",
        end_visit="C3D1",
        delta_column_name="C3D1_minus_SCN",
    )
    if delta_df.empty:
        return delta_df
    delta_df = _pfizer_v2_add_composite_scores(
        delta_df,
        out_path=os.path.join(module_out, "composite_component_scaling.tsv"),
    )
    delta_df.to_csv(os.path.join(module_out, "c3d1_composite_input_deltas.tsv"), sep="\t")

    composite_feature = "C3D1 PD Response Score"
    fallback_feature = "C3D1 PD Response Score noPRC2"
    chosen_feature = composite_feature if composite_feature in delta_df.columns and delta_df[composite_feature].notna().any() else fallback_feature
    if chosen_feature not in delta_df.columns:
        return delta_df

    stats_df = _pfizer_v2_two_group_stats(
        delta_df,
        [chosen_feature, fallback_feature] if fallback_feature in delta_df.columns else [chosen_feature],
        group_column="progression_binary",
        group_a="PD",
        group_b="non-PD",
        comparison_name="C3D1_composite_PD_vs_nonPD",
    )
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(module_out, "composite_by_response_stats.tsv"), sep="\t", index=False)

    _maybe(
        _pfizer_v2_plot_waterfall,
        overwrite,
        delta_df,
        feature_name=chosen_feature,
        out_path=os.path.join(module_out, "composite_waterfall"),
        title=f"{chosen_feature} waterfall",
        y_label=chosen_feature,
    )
    if not stats_df.empty and chosen_feature in set(stats_df["feature"]):
        chosen_stats = stats_df.loc[stats_df["feature"] == chosen_feature].iloc[0]
        _maybe(
            _pfizer_v2_plot_group_box,
            overwrite,
            delta_df.reset_index(drop=True),
            feature_name=chosen_feature,
            group_column="progression_binary",
            out_path=os.path.join(module_out, "composite_by_progression"),
            title=f"{chosen_feature} by PD vs non-PD",
            y_label=chosen_feature,
            p_value=float(chosen_stats["p_value"]),
        )
    _maybe(
        _pfizer_v2_plot_scatter,
        overwrite,
        delta_df.reset_index(),
        x_column="days_from_scn_EOT",
        y_column=chosen_feature,
        out_path=os.path.join(module_out, "composite_vs_eot_day"),
        title=f"{chosen_feature} vs EOT timing",
    )

    component_matrix = delta_df[_pfizer_v2_present_features(delta_df.columns, component_names)].T
    if not component_matrix.empty:
        annotation_df = delta_df[["response_4class", "progression_binary", "arm", "dose"]].copy()
        _maybe(
            _pfizer_v2_plot_annotated_heatmap,
            overwrite,
            component_matrix,
            annotation_df,
            out_path=os.path.join(module_out, "component_delta_heatmap"),
            title="C3D1 composite component deltas",
        )
    return delta_df


def _pfizer_v2_run_c3d1_predictor_analysis(
    delta_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Compare early C3D1 deltas between progressors and non-progressors."""
    module_out = ensure_dir(os.path.join(out_dir, "09_c3d1_predictors_of_outcome"))
    if delta_df.empty:
        return
    feature_names = _pfizer_v2_present_features(
        delta_df.columns,
        [
            "AR score",
            "NE score",
            "CCP score",
            "HALLMARK_E2F_TARGETS",
            "HALLMARK_G2M_CHECKPOINT",
            "HALLMARK_MYC_TARGETS_V1",
            "HALLMARK_ANDROGEN_RESPONSE",
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
            "HALLMARK_HYPOXIA",
            "HALLMARK_IL6_JAK_STAT3_SIGNALING",
            "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
            "PRC2 target derepression score (exploratory)",
            "C3D1 PD Response Score",
            "C3D1 PD Response Score noPRC2",
        ],
    )
    stats_df = _pfizer_v2_two_group_stats(
        delta_df,
        feature_names,
        group_column="progression_binary",
        group_a="PD",
        group_b="non-PD",
        comparison_name="C3D1_delta_PD_vs_nonPD",
    )
    if stats_df.empty:
        return
    stats_df.to_csv(os.path.join(module_out, "c3d1_predictor_stats.tsv"), sep="\t", index=False)
    _maybe(
        _pfizer_v2_plot_effect_forest,
        overwrite,
        stats_df,
        effect_column="median_difference_group_b_minus_group_a",
        lower_column="median_difference_ci_lo",
        upper_column="median_difference_ci_hi",
        out_path=os.path.join(module_out, "c3d1_predictor_forest"),
        title="C3D1 deltas associated with later non-progression",
    )
    for _, row in stats_df.iterrows():
        feature_name = str(row["feature"])
        _maybe(
            _pfizer_v2_plot_group_box,
            overwrite,
            delta_df.reset_index(drop=True),
            feature_name=feature_name,
            group_column="progression_binary",
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_by_progression")),
            title=f"Δ {feature_name}: PD vs non-PD",
            y_label=f"Δ {feature_name} (C3D1 - SCN)",
            p_value=float(row["p_value"]),
        )
        _maybe(
            _pfizer_v2_plot_waterfall,
            overwrite,
            delta_df,
            feature_name=feature_name,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_waterfall")),
            title=f"Δ {feature_name}: patient waterfall",
            y_label=f"Δ {feature_name} (C3D1 - SCN)",
        )


def _pfizer_v2_run_eot_analysis(
    sample_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Characterize end-of-treatment resistance and escape-state biology."""
    module_out = ensure_dir(os.path.join(out_dir, "10_eot_escape_biology"))
    feature_names = _pfizer_v2_present_features(
        patient_df.columns,
        [
            "AR score__SCN",
            "NE score__SCN",
            "CCP score__SCN",
            "PRC2 target derepression score (exploratory)__SCN",
            "HALLMARK_ANDROGEN_RESPONSE__SCN",
            "HALLMARK_E2F_TARGETS__SCN",
            "HALLMARK_G2M_CHECKPOINT__SCN",
            "HALLMARK_MYC_TARGETS_V1__SCN",
            "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION__SCN",
            "HALLMARK_HYPOXIA__SCN",
            "HALLMARK_TNFA_SIGNALING_VIA_NFKB__SCN",
            "HALLMARK_IL6_JAK_STAT3_SIGNALING__SCN",
        ],
    )
    feature_names = [feature.rsplit("__", 1)[0] for feature in feature_names]
    if not feature_names:
        return

    for feature_name in feature_names:
        _maybe(
            _pfizer_v2_plot_trajectory,
            overwrite,
            sample_df,
            feature_name=feature_name,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_trajectory_scn_c3d1_eot")),
            title=f"{feature_name}: SCN → C3D1 → EOT",
            visit_order=["SCN", "C3D1", "EOT"],
            facet_by_arm=False,
        )

    c3d1_df = _pfizer_v2_build_delta_frame(patient_df, feature_names, start_visit="SCN", end_visit="C3D1", delta_column_name="C3D1_minus_SCN")
    eot_df = _pfizer_v2_build_delta_frame(patient_df, feature_names, start_visit="SCN", end_visit="EOT", delta_column_name="EOT_minus_SCN")
    rebound_df = _pfizer_v2_build_delta_frame(patient_df, feature_names, start_visit="C3D1", end_visit="EOT", delta_column_name="EOT_minus_C3D1")
    if not c3d1_df.empty:
        c3d1_df.to_csv(os.path.join(module_out, "c3d1_minus_scn.tsv"), sep="\t")
    if not eot_df.empty:
        eot_df.to_csv(os.path.join(module_out, "eot_minus_scn.tsv"), sep="\t")
    if not rebound_df.empty:
        rebound_df.to_csv(os.path.join(module_out, "eot_minus_c3d1.tsv"), sep="\t")

    if not eot_df.empty:
        heatmap_rows: Dict[str, pd.Series] = {}
        for feature_name in feature_names:
            if feature_name in c3d1_df.columns:
                heatmap_rows[f"{feature_name} | C3D1 - SCN"] = c3d1_df[feature_name]
            if feature_name in rebound_df.columns:
                heatmap_rows[f"{feature_name} | EOT - C3D1"] = rebound_df[feature_name]
            if feature_name in eot_df.columns:
                heatmap_rows[f"{feature_name} | EOT - SCN"] = eot_df[feature_name]
        heatmap_matrix = pd.DataFrame(heatmap_rows).T if heatmap_rows else pd.DataFrame()
        if not heatmap_matrix.empty:
            annotation_df = eot_df[["response_4class", "progression_binary", "arm", "dose", "days_from_scn_EOT"]]
            _maybe(
                _pfizer_v2_plot_annotated_heatmap,
                overwrite,
                heatmap_matrix,
                annotation_df,
                out_path=os.path.join(module_out, "eot_delta_heatmap"),
                title="End-of-treatment delta landscape",
            )
        for feature_name in ["CCP score", "AR score", "NE score", "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"]:
            if feature_name not in eot_df.columns:
                continue
            _maybe(
                _pfizer_v2_plot_scatter,
                overwrite,
                eot_df.reset_index(),
                x_column="days_from_scn_EOT",
                y_column=feature_name,
                out_path=os.path.join(module_out, _pfizer_safe_name(f"eot_day_vs_{feature_name}")),
                title=f"EOT day vs {feature_name} (EOT - SCN)",
            )


def _pfizer_v2_run_gene_level_analysis(
    sample_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Create focused AR, NE, proliferation, and PRC2 gene-level views."""
    module_out = ensure_dir(os.path.join(out_dir, "11_gene_level_pharmacodynamics"))
    baseline_df = sample_df.loc[sample_df["visit"].astype(str) == "SCN"].copy().set_index("patient_id", drop=False)
    focused_genes = _pfizer_v2_present_features(
        sample_df.columns,
        _PFIZER_AR_GENE_PANEL + _PFIZER_NE_GENE_PANEL + _PFIZER_CELL_CYCLE_GENE_PANEL + _PFIZER_PRC2_GENE_PANEL,
    )
    if baseline_df.empty or not focused_genes:
        return

    baseline_matrix = baseline_df[focused_genes].T
    annotation_df = pd.DataFrame(
        {
            "response_4class": baseline_df["response_4class"].astype(str),
            "progression_binary": baseline_df["progression_binary"].astype(str),
            "arm": baseline_df["arm"].astype(str),
            "dose": baseline_df["dose"].astype(str),
        },
        index=baseline_df.index,
    )
    _maybe(
        _pfizer_v2_plot_annotated_heatmap,
        overwrite,
        baseline_matrix,
        annotation_df,
        out_path=os.path.join(module_out, "baseline_gene_heatmap"),
        title="Focused gene panel at SCN",
    )

    c3d1_df = _pfizer_v2_build_delta_frame(patient_df, focused_genes, start_visit="SCN", end_visit="C3D1", delta_column_name="C3D1_minus_SCN")
    eot_df = _pfizer_v2_build_delta_frame(patient_df, focused_genes, start_visit="SCN", end_visit="EOT", delta_column_name="EOT_minus_SCN")
    if not c3d1_df.empty:
        _maybe(
            _pfizer_v2_plot_annotated_heatmap,
            overwrite,
            c3d1_df[focused_genes].T,
            c3d1_df[["response_4class", "progression_binary", "arm", "dose"]],
            out_path=os.path.join(module_out, "c3d1_delta_gene_heatmap"),
            title="Focused gene-panel deltas: C3D1 - SCN",
        )
    if not eot_df.empty:
        _maybe(
            _pfizer_v2_plot_annotated_heatmap,
            overwrite,
            eot_df[focused_genes].T,
            eot_df[["response_4class", "progression_binary", "arm", "dose", "days_from_scn_EOT"]],
            out_path=os.path.join(module_out, "eot_delta_gene_heatmap"),
            title="Focused gene-panel deltas: EOT - SCN",
        )

    for gene_name in _pfizer_v2_present_features(sample_df.columns, _PFIZER_GENE_LEVEL_PAIRED_PLOT_GENES):
        _maybe(
            _pfizer_v2_plot_trajectory,
            overwrite,
            sample_df,
            feature_name=gene_name,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{gene_name}_paired_trajectory")),
            title=f"{gene_name} across visits",
            visit_order=["SCN", "C3D1", "EOT"],
            facet_by_arm=True,
        )


def _pfizer_v2_run_unsupervised_analysis(
    sample_df: pd.DataFrame,
    pred_subset: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Create full-expression and pathway-score PCA/UMAP overviews for Pfizer."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    module_out = ensure_dir(os.path.join(out_dir, "12_unsupervised_overview"))
    expression_matrix = pred_subset.apply(pd.to_numeric, errors="coerce")
    expression_matrix = expression_matrix.loc[~expression_matrix.isna().any(axis=1)]
    if expression_matrix.shape[1] < 3 or expression_matrix.shape[0] < 10:
        return

    gene_variance = expression_matrix.var(axis=1).sort_values(ascending=False)
    variable_genes = gene_variance.head(min(500, len(gene_variance))).index.tolist()
    expr_for_embedding = expression_matrix.loc[variable_genes].T
    scaled_expr = StandardScaler().fit_transform(expr_for_embedding.values)
    pca_expr = PCA(n_components=2, random_state=42).fit_transform(scaled_expr)

    embedding_df = sample_df.loc[expr_for_embedding.index].copy()
    embedding_df["PC1"] = pca_expr[:, 0]
    embedding_df["PC2"] = pca_expr[:, 1]
    embedding_df.to_csv(os.path.join(module_out, "expression_pca_coordinates.tsv"), sep="\t")
    for color_column in ["patient_id", "visit", "arm", "response_4class", "PAM50 subtype", "AR score", "NE score", "CCP score"]:
        if color_column not in embedding_df.columns:
            continue
        _maybe(
            _pfizer_v2_plot_embedding,
            overwrite,
            embedding_df.reset_index(drop=True),
            x_column="PC1",
            y_column="PC2",
            color_column=color_column,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"expression_pca_by_{color_column}")),
            title=f"Expression PCA colored by {color_column}",
            continuous_color=color_column in {"AR score", "NE score", "CCP score"},
        )

    try:
        import umap
    except ImportError:
        print("    [skip] Pfizer UMAP: umap-learn is not installed")
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        umap_expr = reducer.fit_transform(scaled_expr)
        embedding_df["UMAP1"] = umap_expr[:, 0]
        embedding_df["UMAP2"] = umap_expr[:, 1]
        embedding_df.to_csv(os.path.join(module_out, "expression_umap_coordinates.tsv"), sep="\t")
        for color_column in ["patient_id", "visit", "arm", "response_4class", "PAM50 subtype", "AR score", "NE score", "CCP score"]:
            if color_column not in embedding_df.columns:
                continue
            _maybe(
                _pfizer_v2_plot_embedding,
                overwrite,
                embedding_df.reset_index(drop=True),
                x_column="UMAP1",
                y_column="UMAP2",
                color_column=color_column,
                out_path=os.path.join(module_out, _pfizer_safe_name(f"expression_umap_by_{color_column}")),
                title=f"Expression UMAP colored by {color_column}",
                continuous_color=color_column in {"AR score", "NE score", "CCP score"},
            )

    pathway_features = _pfizer_v2_present_features(
        sample_df.columns,
        ["AR score", "NE score", "CCP score", "PRC2 target derepression score (exploratory)"] + _PFIZER_BASELINE_HALLMARKS,
    )
    if len(pathway_features) >= 3:
        pathway_matrix = sample_df[pathway_features].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        if pathway_matrix.shape[0] >= 3:
            pathway_embed = sample_df.loc[pathway_matrix.index].copy()
            scaled_pathways = StandardScaler().fit_transform(pathway_matrix.values)
            pca_pathways = PCA(n_components=2, random_state=42).fit_transform(scaled_pathways)
            pathway_embed["PC1"] = pca_pathways[:, 0]
            pathway_embed["PC2"] = pca_pathways[:, 1]
            pathway_embed.to_csv(os.path.join(module_out, "pathway_pca_coordinates.tsv"), sep="\t")
            for color_column in ["patient_id", "visit", "arm", "response_4class", "PAM50 subtype", "AR score", "NE score", "CCP score"]:
                if color_column not in pathway_embed.columns:
                    continue
                _maybe(
                    _pfizer_v2_plot_embedding,
                    overwrite,
                    pathway_embed.reset_index(drop=True),
                    x_column="PC1",
                    y_column="PC2",
                    color_column=color_column,
                    out_path=os.path.join(module_out, _pfizer_safe_name(f"pathway_pca_by_{color_column}")),
                    title=f"Pathway-score PCA colored by {color_column}",
                    continuous_color=color_column in {"AR score", "NE score", "CCP score"},
                )


def _pfizer_v2_effective_dose_legend_handles(
    levels: Optional[Sequence[str]] = None,
) -> List[Any]:
    """Return legend handles for the Pfizer effective-dose split."""
    from matplotlib.lines import Line2D

    effective_levels = list(levels) if levels is not None else list(_PFIZER_EFFECTIVE_DOSE_ORDER)
    color_map = _pfizer_v2_color_map("effective_dose_group", effective_levels)
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_map.get(level, "#999999"),
            label=level,
            markerfacecolor=color_map.get(level, "#999999"),
            markeredgecolor="black",
            linewidth=2.0,
            markersize=7,
        )
        for level in effective_levels
    ]


def _pfizer_v2_plot_swimmer(
    sample_df: pd.DataFrame,
    *,
    out_path: str,
    title: str,
) -> None:
    """Plot the reduced Pfizer cohort as a patient swimmer with visit and TFx annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = ["patient_id", "visit", "days_from_scn", "effective_dose_group", "tfx"]
    if any(column not in sample_df.columns for column in required):
        return
    working = sample_df[required].copy()
    working["days_from_scn"] = pd.to_numeric(working["days_from_scn"], errors="coerce")
    working["tfx"] = pd.to_numeric(working["tfx"], errors="coerce")
    working = working.dropna(subset=["patient_id", "visit", "days_from_scn", "effective_dose_group"])
    if working.empty:
        return

    patient_order = (
        working.groupby("patient_id")
        .agg(
            effective_dose_group=("effective_dose_group", lambda values: str(pd.Series(values).iloc[0])),
            max_day=("days_from_scn", "max"),
        )
        .reset_index()
        .sort_values(["effective_dose_group", "max_day", "patient_id"])
    )
    patient_levels = patient_order["patient_id"].tolist()
    y_positions = {patient_id: idx for idx, patient_id in enumerate(patient_levels)}
    color_map = _pfizer_v2_color_map("effective_dose_group", _PFIZER_EFFECTIVE_DOSE_ORDER)

    fig, ax = plt.subplots(figsize=(11.0, max(5.5, 0.42 * len(patient_levels) + 1.8)))
    for patient_id, patient_rows in working.groupby("patient_id"):
        patient_rows = patient_rows.sort_values("days_from_scn")
        y_val = float(y_positions[patient_id])
        dose_group = str(patient_rows["effective_dose_group"].iloc[0])
        color = color_map.get(dose_group, "#999999")
        max_day = float(patient_rows["days_from_scn"].max())
        ax.hlines(y_val, 0.0, max_day, color=color, linewidth=2.2, alpha=0.9)
        for _, row in patient_rows.iterrows():
            visit_name = str(row["visit"])
            x_val = float(row["days_from_scn"])
            ax.scatter(
                x_val,
                y_val,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                marker=_PFIZER_VISIT_MARKER_MAP.get(visit_name, "o"),
                s=58,
                zorder=3,
            )
            if np.isfinite(float(row["tfx"])):
                ax.text(
                    x_val + 2.0,
                    y_val,
                    f"TFx={float(row['tfx']):.2f}",
                    fontsize=7,
                    va="center",
                    ha="left",
                )

    ax.set_yticks(np.arange(len(patient_levels), dtype=float))
    ax.set_yticklabels(patient_levels, fontsize=8)
    ax.set_xlabel("Days from SCN")
    ax.set_ylabel("Patient ID")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    _pfizer_v2_add_dual_legends(
        ax,
        color_handles=_pfizer_v2_effective_dose_legend_handles(
            [
                level for level in _PFIZER_EFFECTIVE_DOSE_ORDER
                if level in set(working["effective_dose_group"].astype(str))
            ]
        ),
        color_title="Color = effective-dose group",
        marker_handles=_pfizer_v2_visit_marker_legend_handles(_PFIZER_VISIT_ORDER),
        marker_title="Marker = visit",
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_plot_dose_trajectory(
    sample_df: pd.DataFrame,
    *,
    feature_name: str,
    out_path: str,
    title: str,
    visit_order: Sequence[str],
) -> None:
    """Plot paired patient trajectories colored by the effective-dose split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = ["patient_id", "visit", "effective_dose_group", feature_name]
    if any(column not in sample_df.columns for column in required):
        return
    working = sample_df[required].copy()
    working[feature_name] = pd.to_numeric(working[feature_name], errors="coerce")
    working = working.loc[working[feature_name].notna()].copy()
    working = working.loc[working["visit"].astype(str).isin(list(visit_order))].copy()
    if working.empty:
        return

    visit_to_x = {visit_name: idx for idx, visit_name in enumerate(visit_order)}
    color_map = _pfizer_v2_color_map("effective_dose_group", _PFIZER_EFFECTIVE_DOSE_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for patient_id, patient_rows in working.groupby("patient_id"):
        patient_rows = patient_rows.sort_values("visit", key=lambda values: values.astype(str).map(visit_to_x))
        if patient_rows["visit"].astype(str).nunique() < 2:
            continue
        x_vals = [visit_to_x[str(visit_name)] for visit_name in patient_rows["visit"].astype(str)]
        y_vals = patient_rows[feature_name].astype(float).tolist()
        dose_group = str(patient_rows["effective_dose_group"].iloc[0])
        color = color_map.get(dose_group, "#999999")
        ax.plot(x_vals, y_vals, color=color, linewidth=1.6, alpha=0.75, zorder=1)
        for visit_name, x_val, y_val in zip(patient_rows["visit"].astype(str), x_vals, y_vals):
            ax.scatter(
                x_val,
                y_val,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                marker=_PFIZER_VISIT_MARKER_MAP.get(visit_name, "o"),
                s=46,
                zorder=3,
            )
    ax.set_xticks(list(visit_to_x.values()))
    ax.set_xticklabels(list(visit_order))
    ax.set_xlabel("Visit")
    ax.set_ylabel(feature_name)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    _pfizer_v2_add_dual_legends(
        ax,
        color_handles=_pfizer_v2_effective_dose_legend_handles(
            [
                level for level in _PFIZER_EFFECTIVE_DOSE_ORDER
                if level in set(working["effective_dose_group"].astype(str))
            ]
        ),
        color_title="Color = effective-dose group",
        marker_handles=_pfizer_v2_visit_marker_legend_handles(visit_order),
        marker_title="Marker = visit",
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0])
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_grouped_paired_delta_stats(
    delta_df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    group_column: str,
) -> pd.DataFrame:
    """Summarize paired deltas within each level of a grouping variable."""
    grouped_results: List[pd.DataFrame] = []
    for group_level in _pfizer_v2_preferred_levels(group_column) or sorted(delta_df[group_column].astype(str).unique().tolist()):
        group_df = delta_df.loc[delta_df[group_column].astype(str) == group_level].copy()
        if group_df.empty:
            continue
        group_stats = _pfizer_v2_paired_delta_stats(
            group_df,
            feature_columns,
            comparison_name=f"{group_column}_{group_level}_paired_delta",
        )
        if group_stats.empty:
            continue
        group_stats.insert(1, group_column, group_level)
        grouped_results.append(group_stats)
    if not grouped_results:
        return pd.DataFrame()
    return pd.concat(grouped_results, ignore_index=True)


def _pfizer_v2_run_reduced_cohort_overview(
    sample_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Write reduced-cohort summaries and the primary swimmer plot."""
    module_out = ensure_dir(os.path.join(out_dir, "01_reduced_cohort_overview"))
    sample_df.to_csv(os.path.join(module_out, "matched_sample_metadata.tsv"), sep="\t")
    patient_df.to_csv(os.path.join(module_out, "patient_level_metadata.tsv"), sep="\t")

    summary_rows = [
        {"metric": "n_patients", "value": int(patient_df.index.nunique())},
        {"metric": "n_samples", "value": int(sample_df.index.nunique())},
        {
            "metric": "n_patients_with_SCN_C3D1_pair",
            "value": int(((patient_df["sample_SCN"].notna()) & (patient_df["sample_C3D1"].notna())).sum()),
        },
        {
            "metric": "n_patients_with_all_three_visits",
            "value": int(((patient_df["sample_SCN"].notna()) & (patient_df["sample_C3D1"].notna()) & (patient_df["sample_EOT"].notna())).sum()),
        },
        {
            "metric": "n_effective_dose_patients_with_SCN_C3D1_pair",
            "value": int((((patient_df["sample_SCN"].notna()) & (patient_df["sample_C3D1"].notna())) & (patient_df["effective_dose_group"].astype(str) == _PFIZER_EFFECTIVE_DOSE_ORDER[1])).sum()),
        },
        {
            "metric": "n_lower_or_none_patients_with_SCN_C3D1_pair",
            "value": int((((patient_df["sample_SCN"].notna()) & (patient_df["sample_C3D1"].notna())) & (patient_df["effective_dose_group"].astype(str) == _PFIZER_EFFECTIVE_DOSE_ORDER[0])).sum()),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(module_out, "reduced_cohort_summary.tsv"), sep="\t", index=False)
    _maybe(
        _pfizer_v2_plot_swimmer,
        overwrite,
        sample_df,
        out_path=os.path.join(module_out, "patient_swimmer"),
        title="CRPC-Pfizer reduced cohort: visits, effective dose, and TFx",
    )


def _pfizer_v2_score_feature_columns(sample_df: pd.DataFrame) -> List[str]:
    """Return the active Pfizer score features for trajectories and deltas."""
    pam50_score_columns = sorted(
        column
        for column in sample_df.columns
        if column.startswith("PAM50 corr_") or column.startswith("PAM50 proba_")
    )
    requested = list(dict.fromkeys(_PFIZER_LONGITUDINAL_SCORE_FEATURES + pam50_score_columns))
    return _pfizer_v2_present_features(sample_df.columns, requested)


def _pfizer_v2_plot_group_trajectory(
    sample_df: pd.DataFrame,
    *,
    feature_name: str,
    out_path: str,
    title: str,
    visit_order: Sequence[str],
) -> None:
    """Plot the dose-group median trajectory with interquartile ribbons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    required = [feature_name, "visit", "effective_dose_group"]
    if any(column not in sample_df.columns for column in required):
        return
    working = sample_df[required].copy()
    working[feature_name] = pd.to_numeric(working[feature_name], errors="coerce")
    working = working.dropna(subset=[feature_name, "visit", "effective_dose_group"])
    working = working.loc[working["visit"].astype(str).isin(list(visit_order))].copy()
    if working.empty:
        return

    visit_to_x = {visit_name: idx for idx, visit_name in enumerate(visit_order)}
    group_levels = [
        level for level in _PFIZER_EFFECTIVE_DOSE_ORDER
        if level in set(working["effective_dose_group"].astype(str))
    ]
    if not group_levels:
        group_levels = sorted(working["effective_dose_group"].astype(str).unique().tolist())
    color_map = _pfizer_v2_color_map("effective_dose_group", group_levels)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for group_level in group_levels:
        group_df = working.loc[working["effective_dose_group"].astype(str) == group_level].copy()
        x_vals: List[float] = []
        median_vals: List[float] = []
        q1_vals: List[float] = []
        q3_vals: List[float] = []
        for visit_name in visit_order:
            visit_vals = group_df.loc[group_df["visit"].astype(str) == visit_name, feature_name].astype(float).values
            if visit_vals.size == 0:
                continue
            x_vals.append(float(visit_to_x[visit_name]))
            median_vals.append(float(np.median(visit_vals)))
            q1_vals.append(float(np.percentile(visit_vals, 25)))
            q3_vals.append(float(np.percentile(visit_vals, 75)))
        if not x_vals:
            continue
        color = color_map.get(group_level, "#999999")
        ax.plot(x_vals, median_vals, color=color, linewidth=2.0, marker="o", markersize=6, label=group_level)
        if len(x_vals) >= 2:
            ax.fill_between(x_vals, q1_vals, q3_vals, color=color, alpha=0.18)

    ax.set_xticks(list(visit_to_x.values()))
    ax.set_xticklabels(list(visit_order))
    ax.set_xlabel("Visit")
    ax.set_ylabel(feature_name)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.legend(title="Effective-dose group", fontsize=8, title_fontsize=8, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def _pfizer_v2_run_score_trajectories(
    sample_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Write streamlined longitudinal score summaries and trajectory plots."""
    module_out = ensure_dir(os.path.join(out_dir, "02_score_trajectories"))
    score_features = _pfizer_v2_score_feature_columns(sample_df)
    if not score_features:
        return

    summary_rows: List[Dict[str, Any]] = []
    for feature_name in score_features:
        working = sample_df[[feature_name, "visit", "effective_dose_group"]].copy()
        working[feature_name] = pd.to_numeric(working[feature_name], errors="coerce")
        working = working.dropna(subset=[feature_name, "visit", "effective_dose_group"])
        if working.empty:
            continue

        for group_level in _PFIZER_EFFECTIVE_DOSE_ORDER:
            group_df = working.loc[working["effective_dose_group"].astype(str) == group_level].copy()
            if group_df.empty:
                continue
            for visit_name in _PFIZER_VISIT_ORDER:
                visit_vals = group_df.loc[group_df["visit"].astype(str) == visit_name, feature_name].astype(float).values
                if visit_vals.size == 0:
                    continue
                summary_rows.append(
                    {
                        "feature": feature_name,
                        "effective_dose_group": group_level,
                        "visit": visit_name,
                        "n_samples": int(visit_vals.size),
                        "mean": float(np.mean(visit_vals)),
                        "median": float(np.median(visit_vals)),
                        "q25": float(np.percentile(visit_vals, 25)),
                        "q75": float(np.percentile(visit_vals, 75)),
                    }
                )

        _maybe(
            _pfizer_v2_plot_group_trajectory,
            overwrite,
            sample_df,
            feature_name=feature_name,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_group_trajectory")),
            title=f"{feature_name}: dose-group trajectory",
            visit_order=_PFIZER_VISIT_ORDER,
        )
        _maybe(
            _pfizer_v2_plot_dose_trajectory,
            overwrite,
            sample_df,
            feature_name=feature_name,
            out_path=os.path.join(module_out, _pfizer_safe_name(f"{feature_name}_patient_trajectory")),
            title=f"{feature_name}: per-patient trajectory",
            visit_order=_PFIZER_VISIT_ORDER,
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(module_out, "score_trajectory_summary.tsv"),
            sep="\t",
            index=False,
        )


def _pfizer_v2_run_score_deltas(
    sample_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
) -> None:
    """Write within-group and between-group dose delta summaries for Pfizer scores."""
    module_out = ensure_dir(os.path.join(out_dir, "03_score_deltas"))
    score_features = _pfizer_v2_score_feature_columns(sample_df)
    if not score_features:
        return

    delta_specs = [
        ("SCN", "C3D1", "c3d1_minus_scn", "SCN to C3D1"),
        ("SCN", "EOT", "eot_minus_scn", "SCN to EOT"),
        ("C3D1", "EOT", "eot_minus_c3d1", "C3D1 to EOT"),
    ]
    for start_visit, end_visit, stem, title_stem in delta_specs:
        delta_df = _pfizer_v2_build_delta_frame(
            patient_df,
            score_features,
            start_visit=start_visit,
            end_visit=end_visit,
            delta_column_name=f"{end_visit}_minus_{start_visit}",
        )
        if delta_df.empty:
            continue
        delta_df.to_csv(os.path.join(module_out, f"{stem}.tsv"), sep="\t")

        grouped_stats = _pfizer_v2_grouped_paired_delta_stats(
            delta_df,
            score_features,
            group_column="effective_dose_group",
        )
        if not grouped_stats.empty:
            grouped_stats.to_csv(
                os.path.join(module_out, f"{stem}_within_group_stats.tsv"),
                sep="\t",
                index=False,
            )

        between_stats = _pfizer_v2_two_group_stats(
            delta_df,
            score_features,
            group_column="effective_dose_group",
            group_a=_PFIZER_EFFECTIVE_DOSE_ORDER[0],
            group_b=_PFIZER_EFFECTIVE_DOSE_ORDER[1],
            comparison_name=f"{stem}_effective_dose_vs_lower_or_none",
        )
        if between_stats.empty:
            continue

        between_stats.to_csv(
            os.path.join(module_out, f"{stem}_between_group_stats.tsv"),
            sep="\t",
            index=False,
        )
        _maybe(
            _pfizer_v2_plot_effect_forest,
            overwrite,
            between_stats,
            effect_column="median_difference_group_b_minus_group_a",
            lower_column="median_difference_ci_lo",
            upper_column="median_difference_ci_hi",
            out_path=os.path.join(module_out, f"{stem}_between_group_forest"),
            title=f"{title_stem}: effective-dose vs lower/none",
        )


def _pfizer_v2_run_dose_group_limma_and_gsea(
    pred_subset: pd.DataFrame,
    pfizer_meta: pd.DataFrame,
    out_dir: str,
    overwrite: bool,
    *,
    bundle: Dict[str, Any],
) -> None:
    """Run visit-specific dose-group limma/GSEA on predicted expression means."""
    module_out = ensure_dir(os.path.join(out_dir, "04_dose_group_limma_gsea"))
    extra_gene_sets = _load_pfizer_prc2_sets() or None
    contrast = "EffectiveDose-LowerOrNone"

    for visit_name, visit_label in (("C3D1", "c3d1"), ("EOT", "eot")):
        visit_meta = pfizer_meta.loc[
            (pfizer_meta["visit"].astype(str) == visit_name)
            & pfizer_meta["effective_dose_group"].notna()
        ].copy()
        if visit_meta.empty:
            continue

        visit_out = ensure_dir(os.path.join(module_out, visit_label))
        visit_meta[["patient_id", "dose", "effective_dose_group", "tfx"]].to_csv(
            os.path.join(visit_out, f"{visit_label}_effective_dose_metadata.tsv"),
            sep="\t",
        )
        subtype_df = pd.DataFrame(index=visit_meta.index)
        subtype_df["subtype"] = np.where(
            visit_meta["effective_dose_group"].astype(str) == _PFIZER_EFFECTIVE_DOSE_ORDER[1],
            "EffectiveDose",
            "LowerOrNone",
        )
        visit_bundle = _filter_prediction_bundle_to_samples(bundle, list(visit_meta.index))
        _run_weighted_limma_suite(
            pred_subset[visit_meta.index],
            subtype_df,
            visit_out,
            overwrite,
            bundle=visit_bundle,
            contrast=contrast,
            extra_gene_sets=extra_gene_sets,
        )


def _run_crpc_pfizer_analysis(
    pred: pd.DataFrame,
    activity_odds_ratio_wide: Optional[pd.DataFrame],
    analysis_bundle: Dict[str, Any],
    model_out: str,
    overwrite: bool,
) -> None:
    """Run the streamlined CRPC-Pfizer effective-dose transcriptomic analysis."""
    del activity_odds_ratio_wide
    pfizer_meta = _load_pfizer_metadata(pred.columns)
    if pfizer_meta.empty:
        print("    [skip] CRPC-Pfizer metadata: no matched samples")
        return

    print(
        f"    CRPC-Pfizer: matched {len(pfizer_meta)} samples across "
        f"{pfizer_meta['patient_id'].nunique()} patients"
    )

    pfizer_out = ensure_dir(model_out)

    pfizer_meta.to_csv(os.path.join(pfizer_out, "metadata_matched.tsv"), sep="\t")
    pred_subset = pred[pfizer_meta.index]
    pred_subset.to_csv(os.path.join(pfizer_out, "predictions_wide.tsv"), sep="\t")

    pfizer_bundle: Dict[str, Any] = {}
    for key, value in analysis_bundle.items():
        if isinstance(value, pd.DataFrame):
            sample_columns = [sample for sample in pfizer_meta.index if sample in value.columns]
            pfizer_bundle[key] = value[sample_columns] if sample_columns else value
        else:
            pfizer_bundle[key] = value

    hallmark_scores = _run_pluvicto_ssgsea(pred_subset, pfizer_out, overwrite, tag="_pfizer")
    shortlist_scores = _run_pluvicto_shortlist_scores(pred_subset, pfizer_out, overwrite, tag="_pfizer")
    prc2_scores = _run_pfizer_prc2_scores(pred_subset, pfizer_out, overwrite, tag="_pfizer")
    sample_df, feature_columns = _pfizer_v2_build_sample_feature_table(
        pred_subset,
        pfizer_meta,
        pfizer_out,
        overwrite,
        hallmark_scores=hallmark_scores,
        shortlist_scores=shortlist_scores,
        prc2_scores=prc2_scores,
        pam50_df=pd.DataFrame(),
        compute_pam50=True,
    )
    patient_df = _pfizer_v2_build_patient_feature_table(sample_df, feature_columns)
    if patient_df.empty:
        print("    [skip] CRPC-Pfizer: no patient-level feature table could be constructed")
        return
    sample_df.to_csv(os.path.join(pfizer_out, "sample_level_features.tsv"), sep="\t")
    patient_df.to_csv(os.path.join(pfizer_out, "patient_level_features.tsv"), sep="\t")
    _pfizer_v2_run_reduced_cohort_overview(sample_df, patient_df, pfizer_out, overwrite)
    _pfizer_v2_run_score_trajectories(sample_df, pfizer_out, overwrite)
    _pfizer_v2_run_score_deltas(sample_df, patient_df, pfizer_out, overwrite)
    _pfizer_v2_run_dose_group_limma_and_gsea(
        pred_subset,
        pfizer_meta,
        pfizer_out,
        overwrite,
        bundle=pfizer_bundle,
    )


def _run_discovery_cohort(
    cohort: str,
    subtype_label: Optional[str],
    out_dir: str,
    args: argparse.Namespace,
    calibration_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Run Discovery_Patient analysis for a single cohort (no truth)."""
    cohort_out = ensure_dir(os.path.join(out_dir, cohort))

    label_name = subtype_label or cohort

    rel_path = DISCOVERY_PROTEUS_PATHS.get(cohort)
    path = _require_relative_input_path(
        rel_path,
        description=f"Discovery_Patient Proteus predictions for {cohort}",
    )

    for model_name, path in [("Proteus", path)]:

        # Load full MDN parameters for uncertainty propagation
        full_data = load_holdout_predictions_proteus_full(str(path))
        if _cohort_supports_tfx_subtypes(cohort, label_name=label_name):
            subtypes_df = _load_filtered_subtypes_for_reference(
                cohort,
                full_data["pred_wide"].columns,
                minimum_tfx=args.minimum_tfx,
                label_name=label_name,
                report_context=f"Discovery_Patient {cohort}",
            )
        else:
            if args.minimum_tfx > 0.0:
                _note_tfx_filter_unavailable(f"Discovery_Patient {cohort}")
            subtypes_df = None
        if args.minimum_tfx > 0.0 and subtypes_df is not None:
            full_data = _filter_prediction_bundle_to_samples(full_data, subtypes_df.index)
        if _is_pluvicto_discovery_cohort(cohort):
            include_fhl = cohort == "CRPC-Pluvicto_Full"
            original_samples = [str(sample_name) for sample_name in full_data["pred_wide"].columns]
            keep_samples = _select_pluvicto_baseline_samples(
                full_data["pred_wide"].columns,
                include_fhl=include_fhl,
            )
            baseline_drop_reasons: Dict[str, str] = {}
            for sample_name in original_samples:
                if sample_name in keep_samples:
                    continue
                if not sample_name.endswith("C1"):
                    baseline_drop_reasons[sample_name] = "non-baseline timepoint"
                elif not include_fhl and sample_name.startswith("FHL"):
                    baseline_drop_reasons[sample_name] = "excluded from FH-only MC subset"
                else:
                    baseline_drop_reasons[sample_name] = "excluded by Pluvicto baseline selector"
            full_data = _filter_prediction_bundle_to_samples(full_data, keep_samples)
            filter_label = "all *C1" if include_fhl else "FH* !FHL*, *C1"
            print(
                f"    Pluvicto filter: {len(original_samples)} → {int(full_data['pred_wide'].shape[1])} samples "
                f"({filter_label})"
            )
            _report_sample_drops(
                "Pluvicto discovery subset",
                original_samples,
                keep_samples,
                reason_by_sample=baseline_drop_reasons,
            )

        if subtypes_df is not None:
            retained_samples = [sample for sample in full_data["pred_wide"].columns if sample in subtypes_df.index]
            subtypes_df = subtypes_df.loc[retained_samples].copy()
            if _is_pluvicto_discovery_cohort(cohort) and "subtype" in subtypes_df.columns:
                subtypes_df = subtypes_df.drop(columns=["subtype"])
                print("    Pluvicto metadata: ignoring subtype labels for discovery analysis")

        tfx_dict = subtypes_df["tfx"].to_dict() if subtypes_df is not None and "tfx" in subtypes_df.columns else None
        sub_dict = subtypes_df["subtype"].to_dict() if subtypes_df is not None and "subtype" in subtypes_df.columns else None
        analysis_data = _prepare_proteus_analysis_bundle(full_data)
        if args.apply_calibration:
            if calibration_state is None:
                raise ValueError("Discovery_Patient calibration was requested but no calibration state was available")
            analysis_data = _apply_distribution_calibration_bundle(
                analysis_data,
                calibration_state,
                source_by_sample=_constant_source_by_sample(analysis_data["pred_wide"].columns, cohort),
                sample_fold=None,
                label=f"Discovery_Patient {cohort}",
            )
        pred = analysis_data["pred_wide"]
        activity_odds_ratio_wide = analysis_data.get("activity_odds_ratio_wide")

        # Build MC draws if MDN params available
        mc_draws: Optional[List[pd.DataFrame]] = None
        has_mdn = all(full_data.get(k) is not None for k in (
            "p0_wide", "pi_gamma_wide", "pi_normal_wide",
            "gamma_beta_wide", "normal_mean_wide", "normal_stdev_wide"))
        if has_mdn:
            if args.apply_calibration:
                mc_draws = _get_calibrated_draws(
                    analysis_data,
                    calibration_state,
                    source_by_sample=_constant_source_by_sample(analysis_data["pred_wide"].columns, cohort),
                    sample_fold=None,
                    M=args.montecarlo_draws,
                    seed=_CALIBRATION_DRAW_SEED,
                )
            else:
                mc_draws = mdn_monte_carlo_sample(
                    full_data["p0_wide"], full_data["pi_gamma_wide"],
                    full_data["pi_normal_wide"], full_data["gamma_beta_wide"],
                    full_data["normal_mean_wide"], full_data["normal_stdev_wide"],
                    M=args.montecarlo_draws, seed=_CALIBRATION_DRAW_SEED)

        model_out = cohort_out if model_name == "Proteus" else ensure_dir(os.path.join(cohort_out, model_name))
        print(f"    {model_name}: {pred.shape[0]} genes × {pred.shape[1]} samples")

        if cohort == "CRPC-Pfizer":
            _run_crpc_pfizer_analysis(
                pred,
                activity_odds_ratio_wide,
                analysis_data,
                model_out,
                args.overwrite,
            )
            continue

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
        if cohort == "BC-Ghana":
            bc_ghana_meta_path = _require_existing_path(
                PROJECT_ROOT / "reference" / "BC-Ghana_metadata.tsv",
                description="BC-Ghana metadata table",
            )
            bc_ghana_meta = pd.read_csv(bc_ghana_meta_path, sep="\t").set_index("sample")
            if "subtype" in bc_ghana_meta.columns:
                pam50_reference_subtypes = bc_ghana_meta["subtype"]
                pam50_reference_label = "Metadata subtype"
        elif subtypes_df is not None and "subtype" in subtypes_df.columns:
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
        # --- Limma DE + GSEA: ARPC vs NEPC for CRPC-UW ---
        if cohort == "CRPC-UW" and sub_dict is not None and subtypes_df is not None:
            _run_weighted_limma_suite(
                pred,
                subtypes_df[["subtype"]],
                model_out,
                args.overwrite,
                bundle=analysis_data,
            )

        # --- BC-Ghana: receptor expression boxplots ---
        if cohort == "BC-Ghana":
            _run_bc_ghana_analysis(pred, activity_odds_ratio_wide, model_out, args.overwrite)

        # --- CRPC-Pluvicto: baseline-only split analyses ---
        if _is_pluvicto_discovery_cohort(cohort):
            # _raw Hallmark + shortlist scores + survival
            ssgsea_raw = _run_pluvicto_ssgsea(
                pred, model_out, args.overwrite, tag="_raw")
            shortlist_raw = _run_pluvicto_shortlist_scores(
                pred, model_out, args.overwrite, tag="_raw")
            _run_pluvicto_responder_deg(
                pred,
                model_out,
                args.overwrite,
                bundle=analysis_data,
            )
            _run_pluvicto_survival(
                pred,
                ssgsea_raw,
                shortlist_raw,
                model_out,
                args.overwrite,
                args.minimum_tfx,
                tag="_raw",
            )
            _run_pluvicto_clinical_correlations(
                pred,
                model_out,
                args.overwrite,
                shortlist_scores=shortlist_raw,
                subtypes_df=subtypes_df,
                cohort_label=cohort,
            )

            # MC draws define score intervals only; point-score analyses use prediction means.
            if mc_draws is not None:
                _run_pluvicto_ssgsea_mc(mc_draws, model_out, args.overwrite)
                _run_pluvicto_shortlist_scores_mc(mc_draws, model_out, args.overwrite)


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
    calibration_state = None
    if args.apply_calibration:
        calibration_cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
        if calibration_cv_bundle is None:
            raise ValueError("Discovery_Patient calibration requires the canonical patient CV OOF bundle")
        calibration_state = _get_patient_distribution_calibration_state(
            args.minimum_tfx,
            cv_full=calibration_cv_bundle["cv_full"],
            cohort_map=calibration_cv_bundle["cohort_map"],
        )

    # Default: iterate over all discovery cohorts
    for cohort, subtype_label in _DISCOVERY_COHORTS:
        print(f"\n  ── Cohort: {cohort} ──")
        _run_discovery_cohort(
            cohort,
            subtype_label,
            out_dir,
            args,
            calibration_state=calibration_state,
        )

    print("\n  Done")


def _load_main_plca_cv_full(minimum_tfx: float = 0.0) -> Optional[Dict[str, Any]]:
    """Load and prefix-normalize the canonical TDS CV outputs."""
    for _, rel_path in sorted(PLCA_CV_FOLD_TUMOR_PREDICTION_PATHS.items()):
        _require_existing_path(
            PROJECT_ROOT / rel_path,
            description="canonical TDS CV prediction file",
        )

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

    cv_full = load_cv_predictions_proteus_full_from_paths(
        {
            fold: PROJECT_ROOT / rel_path
            for fold, rel_path in sorted(PLCA_CV_FOLD_TUMOR_PREDICTION_PATHS.items())
        }
    )
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
                report_context=f"TDS CV {cohort_name}",
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
    """Standalone OOF diagnostics for prediction, interval, and AOR surfaces."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Calibration")
    print("=" * 70)

    if _should_skip_section("Calibration", args):
        return

    out_dir = _section_results_dir("Calibration")
    _write_run_manifest("Calibration", out_dir, args)
    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
    if cv_bundle is None:
        return

    cv_full = cv_bundle["cv_full"]
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
                    "n_threshold_genes": int(activity_threshold_summary.get("n_threshold_genes", 0)),
                    "n_threshold_samples": int(activity_threshold_summary.get("n_threshold_samples", 0)),
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
    else:
        print("  [skip] Unable to derive patient CV AOR thresholds")

    _write_oof_validation_diagnostics(
        cv_full,
        cv_bundle["cohort_map"],
        out_dir,
        overwrite=args.overwrite,
        activity_threshold_summary=activity_threshold_summary,
    )

    calibration_state = _get_patient_distribution_calibration_state(
        args.minimum_tfx,
        cv_full=cv_full,
        cohort_map=cv_bundle["cohort_map"],
    )
    _write_distribution_calibration_outputs(
        out_dir,
        calibration_state["patient_bundle"],
        calibration_state,
        overwrite=args.overwrite,
        maybe_write=_maybe,
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
        analysis_ho_preds[cohort_name] = _prepare_proteus_analysis_bundle(ho_data)
    shortlist_sets = _load_shortlist_sets()

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

        subtypes_path = _require_existing_path(
            PROJECT_ROOT / "labels" / f"{cohort_name}_TFX_Subtype.tsv",
            description=f"Validation_PDX subtype table for {cohort_name}",
        )
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

        if cohort_name == "PC-PDX" and shortlist_sets:
            common_samples = combined_truth.columns.intersection(combined_pred.columns)
            if len(common_samples) >= 3:
                score_meta = pd.DataFrame(index=common_samples)
                if subtypes_df is not None:
                    score_meta = subtypes_df.reindex(common_samples).copy()
                if "status" not in score_meta.columns:
                    score_meta["status"] = pd.Series(sample_status).reindex(common_samples)

                truth_scores = _compute_ssgsea_score_matrix(
                    combined_truth[common_samples], shortlist_sets, min_size=3,
                )
                raw_scores = _compute_ssgsea_score_matrix(
                    combined_pred[common_samples], shortlist_sets, min_size=3,
                )
                scatter_out = ensure_dir(os.path.join(cohort_out, "ShortlistScores", "scatter_plots"))

                if "CCP.31" in truth_scores.index and "CCP.31" in raw_scores.index:
                    _maybe(
                        _crosscohort_score_scatter,
                        args.overwrite,
                        truth_scores.loc["CCP.31"],
                        raw_scores.loc["CCP.31"],
                        score_meta,
                        os.path.join(scatter_out, "CCP.31_raw"),
                        title="PC-PDX — CCP.31 ssGSEA (Truth vs Predicted Raw)",
                        x_label="RNA-Seq CCP.31 ssGSEA Score",
                        y_label="Predicted CCP.31 ssGSEA Score",
                        color_column="subtype",
                        legend_title="Subtype",
                    )

                has_combined_mdn = all(combined_mdn_tables.get(key) is not None for key in _MDN_PARAM_KEYS)
                if has_combined_mdn and "CCP.31" in truth_scores.index:
                    print(f"    Shortlist ssGSEA MC: {cohort_name} ({len(common_samples)} samples)")
                    mc_draws = mdn_monte_carlo_sample(
                        combined_mdn_tables["p0_wide"][common_samples],
                        combined_mdn_tables["pi_gamma_wide"][common_samples],
                        combined_mdn_tables["pi_normal_wide"][common_samples],
                        combined_mdn_tables["gamma_beta_wide"][common_samples],
                        combined_mdn_tables["normal_mean_wide"][common_samples],
                        combined_mdn_tables["normal_stdev_wide"][common_samples],
                        M=args.montecarlo_draws,
                        seed=42,
                    )
                    _, score_ci_lo, score_ci_hi = _aggregate_ssgsea_score_matrices(
                        mc_draws, shortlist_sets, min_size=3,
                    )
                    score_interval_out = ensure_dir(os.path.join(cohort_out, "ShortlistScores"))
                    if score_ci_lo is not None:
                        score_ci_lo.to_csv(os.path.join(score_interval_out, "shortlist_score_ci_q5.tsv"), sep="\t")
                    if score_ci_hi is not None:
                        score_ci_hi.to_csv(os.path.join(score_interval_out, "shortlist_score_ci_q95.tsv"), sep="\t")
                    if (
                        score_ci_lo is not None
                        and score_ci_hi is not None
                        and "CCP.31" in raw_scores.index
                        and "CCP.31" in score_ci_lo.index
                        and "CCP.31" in score_ci_hi.index
                    ):
                        _maybe(
                            _crosscohort_score_scatter,
                            args.overwrite,
                            truth_scores.loc["CCP.31"],
                            raw_scores.loc["CCP.31"],
                            score_meta,
                            os.path.join(scatter_out, "CCP.31_raw_score_ci"),
                            title="PC-PDX — CCP.31 ssGSEA (Truth vs Predicted Raw, 90% Score CI)",
                            x_label="RNA-Seq CCP.31 ssGSEA Score",
                            y_label="Predicted CCP.31 ssGSEA Score",
                            color_column="subtype",
                            legend_title="Subtype",
                            pred_ci_lo=score_ci_lo.loc["CCP.31"],
                            pred_ci_hi=score_ci_hi.loc["CCP.31"],
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

    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)

    cv_full = None
    cv_pred = None
    cv_ci_lo = None
    cv_ci_hi = None
    cohort_map_cv = None
    activity_threshold_summary = None
    calibration_state = None
    if cv_bundle is not None:
        cv_full = cv_bundle["cv_full"]
        cohort_map_cv = cv_bundle["cohort_map"]
        cv_pred = cv_full["pred_wide"]
        cv_ci_lo = cv_full.get("ci_lo_wide")
        cv_ci_hi = cv_full.get("ci_hi_wide")
        activity_threshold_summary = _compute_patient_cv_activity_thresholds(cv_full, cohort_map_cv)
        if args.apply_calibration:
            calibration_state = _get_patient_distribution_calibration_state(
                args.minimum_tfx,
                cv_full=cv_full,
                cohort_map=cohort_map_cv,
            )
    elif args.apply_calibration:
        raise ValueError("Validation_Patient calibration requires the canonical patient CV OOF bundle")

    for cohort in VALIDATION_PATIENT_HOLDOUT_COHORTS:
        print(f"\n  ── Cohort: {cohort} (holdout) ──")
        holdout_path = VALIDATION_PATIENT_HOLDOUT_PATHS.get(cohort)
        if holdout_path is None:
            raise ValueError(f"Missing required hard-coded holdout path for Validation_Patient cohort {cohort}")
        result = _run_patient_validation_cohort(
            cohort,
            holdout_path,
            out_dir,
            args,
            activity_threshold_summary=activity_threshold_summary,
            calibration_state=calibration_state,
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
                calibration_state=calibration_state,
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
            calibration_state=calibration_state,
            run_pam50_crosscohort=True,
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


def _manifest_json_safe(value: Any) -> Any:
    """Convert common scientific/Python objects to JSON-safe values."""
    if isinstance(value, Path):
        try:
            return value.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_manifest_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Index):
        return [_manifest_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _manifest_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_manifest_json_safe(item) for item in value]
    return value


def _write_run_manifest(
    section_name: str,
    out_dir: str,
    args: argparse.Namespace,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write the section run manifest with CLI settings and optional inputs."""
    os.makedirs(out_dir, exist_ok=True)
    output_path = Path(out_dir)
    try:
        output_dir_value = output_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        output_dir_value = output_path.as_posix()
    manifest: Dict[str, Any] = {
        "section": section_name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": output_dir_value,
        "arguments": vars(args),
        "n_workers": getattr(args, "n_workers", None),
    }
    if extra:
        manifest.update(extra)
    with open(output_path / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(_manifest_json_safe(manifest), handle, indent=2, sort_keys=True)


def _file_manifest_entry(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    rel_path = path.relative_to(PROJECT_ROOT)
    return {
        "path": rel_path.as_posix(),
        "size_bytes": int(stat.st_size),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
    }


def _validate_phs_sample_names(sample_names: Sequence[str], *, model_name: str) -> None:
    invalid_samples: List[str] = []
    for sample_name in sample_names:
        sample_name = str(sample_name)
        try:
            parse_phs_sample(sample_name)
        except ValueError:
            invalid_samples.append(sample_name)
            if len(invalid_samples) >= 5:
                break
    if invalid_samples:
        raise ValueError(
            f"Benchmarking_Synthetic loaded non-PHS sample names for {model_name}: "
            f"{invalid_samples}. Check the source predictions file."
        )


def _benchmark_grid_sample_names(sample_names: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split PHS sample names into in-grid and ignored benchmarking samples."""
    in_grid: List[str] = []
    ignored: List[str] = []
    allowed_depths = set(_BENCH_DEPTH_ORDER)
    allowed_tfx = set(_BENCH_TFX_ORDER)

    for sample_name in sample_names:
        sample_name = str(sample_name)
        info = parse_phs_sample(sample_name)
        if info["coverage"] in allowed_depths and info["tfx"] in allowed_tfx:
            in_grid.append(sample_name)
        else:
            ignored.append(sample_name)

    return sorted(in_grid), sorted(ignored)


def _validate_synthetic_sample_axes(
    sample_axes: Dict[str, pd.Index],
) -> Tuple[str, List[str], Dict[str, List[str]]]:
    if not sample_axes:
        raise ValueError("Benchmarking_Synthetic did not load any sample axes.")

    reference_model = next(
        (name for name in SYNTHETIC_PROTEUS_MODEL_PATHS if name in sample_axes),
        next(iter(sample_axes)),
    )
    ignored_samples_by_model: Dict[str, List[str]] = {}

    reference_raw_samples = sorted(str(sample_name) for sample_name in sample_axes[reference_model])
    _validate_phs_sample_names(reference_raw_samples, model_name=reference_model)
    reference_samples, ignored_reference_samples = _benchmark_grid_sample_names(reference_raw_samples)
    ignored_samples_by_model[reference_model] = ignored_reference_samples
    reference_set = set(reference_samples)

    for model_name, sample_index in sample_axes.items():
        sample_names = [str(sample_name) for sample_name in sample_index]
        _validate_phs_sample_names(sample_names, model_name=model_name)
        grid_sample_names, ignored_samples = _benchmark_grid_sample_names(sample_names)
        ignored_samples_by_model[model_name] = ignored_samples
        sample_set = set(grid_sample_names)
        if sample_set == reference_set:
            continue
        missing = sorted(reference_set - sample_set)
        extra = sorted(sample_set - reference_set)
        raise ValueError(
            f"Benchmarking_Synthetic sample mismatch for {model_name} relative to {reference_model}. "
            f"Missing {len(missing)} in-grid samples and found {len(extra)} unexpected in-grid samples. "
            f"Missing examples: {missing[:5]}. Extra examples: {extra[:5]}."
        )

    return reference_model, reference_samples, ignored_samples_by_model


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


def _shared_roc_pair_arrays(
    truth_wide: pd.DataFrame,
    model_preds: Dict[str, pd.DataFrame],
    common_genes: Sequence[str],
    model_order: Sequence[str],
) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray]]:
    """Align ROC comparisons to the largest finite intersection across models."""
    comparable_models = [
        model_name
        for model_name in model_order
        if model_name in model_preds and model_name != "Naive"
    ]
    if not comparable_models:
        return [], np.array([], dtype=np.float64), {}

    common_samples = [
        sample
        for sample in truth_wide.columns
        if all(sample in model_preds[model_name].columns for model_name in comparable_models)
    ]
    aligned_genes = [
        gene_name
        for gene_name in common_genes
        if all(gene_name in model_preds[model_name].index for model_name in comparable_models)
    ]
    if len(common_samples) < 3 or len(aligned_genes) < 5:
        return [], np.array([], dtype=np.float64), {}

    truth_block = truth_wide.loc[aligned_genes, common_samples].to_numpy(dtype=np.float64)
    valid = np.isfinite(truth_block)
    pred_blocks: Dict[str, np.ndarray] = {}
    for model_name in comparable_models:
        pred_block = model_preds[model_name].loc[aligned_genes, common_samples].to_numpy(dtype=np.float64)
        pred_blocks[model_name] = pred_block
        valid &= np.isfinite(pred_block)

    if int(valid.sum()) < 10:
        return [], np.array([], dtype=np.float64), {}

    truth_flat = truth_block[valid]
    pred_flat_map = {
        model_name: pred_block[valid]
        for model_name, pred_block in pred_blocks.items()
    }
    return comparable_models, truth_flat, pred_flat_map


def _run_roc_gene_calling(
    truth_wide: pd.DataFrame,
    model_preds: Dict[str, pd.DataFrame],
    proteus_mdn: Dict[str, Dict[str, Optional[pd.DataFrame]]],
    out_dir: str,
    overwrite: bool,
    model_order: Sequence[str],
    cohort: str,
    activity_threshold_summary: Optional[Dict[str, Any]] = None,
    emit_threshold_artifacts: bool = True,
    emit_confusion_artifacts: bool = True,
) -> None:
    """Compute and plot ROC curves for on/off gene calling.

    Parameters
    ----------
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

        comparable_models, shared_truth_flat, shared_pred_flat = _shared_roc_pair_arrays(
            truth_wide,
            model_preds,
            common_genes,
            model_order,
        )
        shared_n = int(len(shared_truth_flat))

        for model_name in comparable_models:
            fpr, tpr, auc = _compute_roc_for_model(
                shared_truth_flat,
                shared_pred_flat[model_name],
            )
            roc_curves[model_name] = (fpr, tpr, auc, shared_n)

        # Shuffled-truth naive (labelled "Naive")
        if comparable_models and shared_n >= 10:
            rng_naive = np.random.default_rng(42)
            t_shuffled = rng_naive.permutation(shared_truth_flat)
            reference_model = comparable_models[0]
            fn, tn, an = _compute_roc_for_model(
                t_shuffled,
                shared_pred_flat[reference_model],
            )
            roc_curves["Naive"] = (fn, tn, an, shared_n)

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

        if emit_threshold_artifacts:
            for model_name in model_order:
                if model_name not in model_preds:
                    continue
                activity_block = _model_activity_odds_ratio_block(
                    truth_wide,
                    model_preds[model_name],
                    proteus_mdn.get(model_name, {}),
                    common_genes,
                )
                if activity_block is None:
                    continue
                truth_active, activity_odds_ratio = activity_block
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

        if emit_threshold_artifacts and threshold_curves:
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

        if emit_threshold_artifacts and inactive_threshold_curves:
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

        if emit_confusion_artifacts:
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

        if emit_confusion_artifacts and confusion_results:
            desired_confusion_order: List[str] = []
            for model_name in model_order:
                if model_name == "Naive":
                    continue
                binary_name = f"{model_name} (binary)"
                ua_name = f"{model_name} (uncertainty-informed)"
                if binary_name in confusion_results or ua_name in confusion_results:
                    if binary_name in confusion_results:
                        desired_confusion_order.append(binary_name)
                    if ua_name in confusion_results:
                        desired_confusion_order.append(ua_name)
                    continue
                if model_name in confusion_results:
                    desired_confusion_order.append(model_name)
            confusion_order = desired_confusion_order + [
                model_name for model_name in confusion_order if model_name not in desired_confusion_order
            ]
            conf_rows = []
            for m in confusion_order:
                d = confusion_results[m]
                conf_rows.append({
                    "model": m,
                    "n_total": d["n_total"],
                    "n_truth_on": d.get("n_truth_on", 0),
                    "n_truth_off": d.get("n_truth_off", 0),
                    "n_on": d["n_on"],
                    "n_off": d["n_off"],
                    "n_indet": d["n_indet"],
                    "accuracy": d.get("accuracy", float("nan")),
                    "ppv": d.get("ppv", float("nan")),
                    "npv": d.get("npv", float("nan")),
                    "sensitivity": d.get("sensitivity", float("nan")),
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

    cv_bundle = _load_main_plca_cv_full(args.minimum_tfx)
    activity_threshold_summary: Optional[Dict[str, Any]] = None
    if cv_bundle is not None:
        cv_full = cv_bundle["cv_full"]
        activity_threshold_summary = _compute_patient_cv_activity_thresholds(
            cv_full,
            cv_bundle["cohort_map"],
        )
    subset_view_models = tuple(benchmark_model_views().get("SUBSET", ()))
    pooled_threshold_pairs: Dict[str, Dict[str, List[np.ndarray]]] = {}

    # Accumulate metrics, per-gene rho, and per-sample rho across cohorts
    all_metrics_rows: List[Dict[str, Any]] = []
    all_gene_rho_rows: List[Dict[str, Any]] = []
    all_sample_rho_rows: List[Dict[str, Any]] = []

    for cohort in PATIENT_BENCHMARK_COHORTS:
        print(f"\n  ── Cohort: {cohort} ──")

        # Load truth
        truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
        _require_existing_path(truth_path, description=f"Benchmarking_Patient truth for {cohort}")
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
        proteus_mdn: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}

        # Proteus models
        proteus_paths = PATIENT_PROTEUS_MODEL_PATHS.get(cohort, {})
        for model_name, rel_path in proteus_paths.items():
            path = _require_relative_input_path(
                rel_path,
                description=f"Benchmarking_Patient {model_name} predictions for {cohort}",
            )
            full_data = load_holdout_predictions_proteus_full(str(path))
            for key in ("pred_wide", "ci_lo_wide", "ci_hi_wide", *_MDN_PARAM_KEYS):
                table = full_data.get(key)
                if table is not None:
                    _apply_prefix_map(table, pmap)
            if keep_samples is not None:
                full_data = _filter_prediction_bundle_to_samples(full_data, keep_samples)
            analysis_data = full_data
            if model_name == "Proteus":
                analysis_data = _prepare_proteus_analysis_bundle(full_data)
            pred = analysis_data["pred_wide"]
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
            proteus_mdn[model_name] = mdn_tables
            print(f"    Loaded {model_name}: {pred.shape[1]} samples")

        # XGBoost
        xgb_rel = PATIENT_XGB_PATHS.get(cohort)
        if xgb_rel:
            xgb_path = _require_relative_input_path(
                xgb_rel,
                description=f"Benchmarking_Patient XGBoost predictions for {cohort}",
            )
            xgb_pred = load_holdout_predictions_xgboost(str(xgb_path))
            _apply_prefix_map(xgb_pred, pmap)
            xgb_pred = _filter_wide_to_samples(xgb_pred, keep_samples)
            model_preds["XGBoost"] = xgb_pred
            print(f"    Loaded XGBoost: {xgb_pred.shape[1]} samples")

        # EPIC-Seq
        epic_rel = PATIENT_EPIC_PATHS.get(cohort)
        if epic_rel:
            epic_path = _require_relative_input_path(
                epic_rel,
                description=f"Benchmarking_Patient EPIC-Seq predictions for {cohort}",
            )
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
            _run_roc_gene_calling(
                truth_wide,
                view_preds,
                {m: proteus_mdn[m] for m in view_order if m in proteus_mdn},
                ensure_dir(os.path.join(out_dir, view_name)),
                args.overwrite,
                view_order,
                cohort,
                activity_threshold_summary=activity_threshold_summary,
                emit_threshold_artifacts=False,
                emit_confusion_artifacts=(view_name == "SUBSET"),
            )

        for gs_key in _ROC_GENE_SETS.values():
            if not gene_list_available(gs_key):
                continue
            gene_list = list(dict.fromkeys(get_gene_list(gs_key)))
            common_genes = [gene_name for gene_name in gene_list if gene_name in truth_wide.index]
            if len(common_genes) < 10:
                continue
            for model_name in subset_view_models:
                if model_name not in model_preds or model_name not in proteus_mdn:
                    continue
                activity_block = _model_activity_odds_ratio_block(
                    truth_wide,
                    model_preds[model_name],
                    proteus_mdn[model_name],
                    common_genes,
                )
                if activity_block is None:
                    continue
                flattened = _flatten_activity_odds_ratio_pairs(
                    activity_block[0],
                    activity_block[1],
                    threshold=_ONOFF_THRESHOLD,
                )
                if flattened is None:
                    continue
                pair_store = pooled_threshold_pairs.setdefault(
                    model_name,
                    {"truth_flat": [], "activity_odds_ratio_flat": []},
                )
                pair_store["truth_flat"].append(flattened["truth_flat"])
                pair_store["activity_odds_ratio_flat"].append(flattened["activity_odds_ratio_flat"])

    metrics_table = pd.DataFrame(all_metrics_rows)
    gene_rho_table = pd.DataFrame(all_gene_rho_rows)
    sample_rho_table = pd.DataFrame(all_sample_rho_rows)

    if metrics_table.empty:
        print("  [skip] No metrics computed")
        return

    subset_model_order = [model_name for model_name in subset_view_models if model_name in pooled_threshold_pairs]
    if subset_model_order:
        _write_pooled_benchmark_threshold_artifacts(
            pooled_threshold_pairs,
            ensure_dir(os.path.join(out_dir, "SUBSET")),
            args.overwrite,
            subset_model_order,
        )

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

    # ── Load model predictions and build truth ──────────────────────────
    model_preds: Dict[str, pd.DataFrame] = {}  # model_name -> wide (genes × samples)
    sample_axes: Dict[str, pd.Index] = {}
    manifest_inputs: List[Dict[str, Any]] = []

    for model_name, rel_path in SYNTHETIC_PROTEUS_MODEL_PATHS.items():
        path = _require_relative_input_path(
            rel_path,
            description=f"Benchmarking_Synthetic {model_name} predictions",
        )
        full_data = load_holdout_predictions_proteus_full(str(path))
        model_preds[model_name] = full_data["pred_wide"]
        pred_wide = full_data["pred_wide"]
        sample_axes[model_name] = pred_wide.columns
        manifest_inputs.append({
            "model": model_name,
            "source_type": "proteus",
            "genes": int(pred_wide.shape[0]),
            "samples": int(pred_wide.shape[1]),
            "sample_names": [str(sample_name) for sample_name in pred_wide.columns],
            **_file_manifest_entry(path),
        })
        print(f"  Loaded {model_name}: {pred_wide.shape[0]} genes × {pred_wide.shape[1]} samples [{path}]")

    # XGBoost uses a different column name ('prediction' not 'prediction_mean')
    xgb_path = _require_relative_input_path(
        SYNTHETIC_XGB_PATH,
        description="Benchmarking_Synthetic XGBoost predictions",
    )
    xgb_pred = load_holdout_predictions_xgboost(str(xgb_path))
    model_preds["XGBoost"] = xgb_pred
    sample_axes["XGBoost"] = xgb_pred.columns
    manifest_inputs.append({
        "model": "XGBoost",
        "source_type": "xgboost",
        "genes": int(xgb_pred.shape[0]),
        "samples": int(xgb_pred.shape[1]),
        "sample_names": [str(sample_name) for sample_name in xgb_pred.columns],
        **_file_manifest_entry(xgb_path),
    })
    print(f"  Loaded XGBoost: {xgb_pred.shape[0]} genes × {xgb_pred.shape[1]} samples [{xgb_path}]")

    reference_model, all_samples, ignored_samples_by_model = _validate_synthetic_sample_axes(sample_axes)
    reference_sample_set = set(all_samples)
    for model_name, ignored_samples in ignored_samples_by_model.items():
        if not ignored_samples:
            continue
        print(
            f"  {model_name}: ignoring {len(ignored_samples)} out-of-grid PHS samples "
            f"before synthetic benchmarking alignment"
        )
    for manifest_entry in manifest_inputs:
        manifest_entry["ignored_out_of_grid_samples"] = ignored_samples_by_model.get(
            str(manifest_entry.get("model")),
            [],
        )
    for model_name, pred_wide in list(model_preds.items()):
        model_preds[model_name] = pred_wide.loc[:, all_samples].copy()

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
    epic_path = _require_relative_input_path(
        SYNTHETIC_EPIC_PATH,
        description="Benchmarking_Synthetic EPIC-Seq predictions",
    )
    epic_preds: Dict[str, pd.DataFrame] = {}
    epic_preds = load_holdout_predictions_epicseq(
        str(epic_path), CONFIG_EPIC_SCORE_COLUMNS,
    )
    for score_name, epw in epic_preds.items():
        # PFE here is the raw EPIC-Seq PFE output (column "PFE"),
        # not the normalized quantity used in GEP construction.
        display = CONFIG_EPIC_DISPLAY_NAMES.get(score_name, f"EPIC_{score_name}")
        sample_axes[display] = epw.columns
        epic_sample_names = [str(sample_name) for sample_name in epw.columns]
        _validate_phs_sample_names(epic_sample_names, model_name=display)
        epic_grid_samples, epic_ignored_samples = _benchmark_grid_sample_names(epic_sample_names)
        ignored_samples_by_model[display] = epic_ignored_samples
        if epic_ignored_samples:
            print(
                f"  {display}: ignoring {len(epic_ignored_samples)} out-of-grid PHS samples "
                f"before synthetic benchmarking alignment"
            )
        epic_sample_set = set(epic_grid_samples)
        if epic_sample_set != reference_sample_set:
            missing = sorted(reference_sample_set - epic_sample_set)
            extra = sorted(epic_sample_set - reference_sample_set)
            raise ValueError(
                f"Benchmarking_Synthetic sample mismatch for {display} relative to {reference_model}. "
                f"Missing {len(missing)} in-grid samples and found {len(extra)} unexpected in-grid samples. "
                f"Missing examples: {missing[:5]}. Extra examples: {extra[:5]}."
            )
        model_preds[display] = epw.loc[:, all_samples].copy()
        manifest_inputs.append({
            "model": display,
            "source_type": "epic-seq",
            "score_name": score_name,
            "genes": int(epw.shape[0]),
            "samples": int(epw.shape[1]),
            "sample_names": [str(sample_name) for sample_name in epw.columns],
            "ignored_out_of_grid_samples": ignored_samples_by_model.get(display, []),
            **_file_manifest_entry(epic_path),
        })
        print(f"  Loaded {display}: {epw.shape[0]} genes × {epw.shape[1]} samples [{epic_path}]")

    truth_sources = [
        PROJECT_ROOT / "labels" / "PC-PDX_Log2CPMplus1.tsv",
        PROJECT_ROOT / "labels" / "SCLC-PDX_Log2CPMplus1.tsv",
    ]
    _write_run_manifest(
        "Benchmarking_Synthetic",
        out_dir,
        args,
        extra={
            "reference_model": reference_model,
            "reference_sample_count": len(all_samples),
            "reference_samples": all_samples,
            "ignored_out_of_grid_samples_by_model": ignored_samples_by_model,
            "excluded_depths": sorted(_BENCH_EXCLUDED_DEPTHS),
            "input_files": manifest_inputs + [
                {
                    "model": "truth_PC-PDX",
                    "source_type": "truth",
                    **_file_manifest_entry(truth_sources[0]),
                },
                {
                    "model": "truth_SCLC-PDX",
                    "source_type": "truth",
                    **_file_manifest_entry(truth_sources[1]),
                },
            ],
        },
    )

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

    print("  Done")


# ---------------------------------------------------------------------------
# Patient-CV truth and cohort configuration helpers
# ---------------------------------------------------------------------------

_PLCA_SPLITS_TSV = "data/TDS_sample-splits.tsv"

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
    """Load data/TDS_sample-splits.tsv → DataFrame with Cohort, Sample, Fold."""
    path = PROJECT_ROOT / _PLCA_SPLITS_TSV
    return pd.read_csv(path, sep="\t")


def _build_plca_truth(minimum_tfx: float = 0.0) -> pd.DataFrame:
    """Build canonical truth for patient-CV cohorts (CRPC-RA, SCLC, PC-PDX, SCLC-PDX).

    Loads Log2CPMplus1 labels for all four cohorts, keeps only samples present
    in the TDS sample-splits table (excludes HO PDX), concatenates on a common
    gene index.  PHT samples are excluded from truth (no independent labels).
    """
    splits = _load_plca_fold_splits()
    split_samples = set(splits["Sample"])

    parts = []
    for cohort in _PLCA_CV_COHORTS:
        truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
        _require_existing_path(truth_path, description=f"patient-CV truth for {cohort}")
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


def _preflight_minimum_tfx_filters(group_names: Sequence[str], minimum_tfx: float) -> None:
    """Emit all selected-group TFx sample-filter notices before any analysis-group banner."""
    needs_plca_truth = any(
        group_name in {
            "Examples",
            "Benchmarking_Patient",
            "Calibration",
            "Validation_PDX",
            "Validation_Patient",
            "Discovery_Patient",
        }
        for group_name in group_names
    )
    needs_patient_holdout_truth = any(
        group_name in {"Benchmarking_Patient", "Validation_Patient"}
        for group_name in group_names
    )
    needs_discovery = "Discovery_Patient" in group_names

    if not (needs_plca_truth or needs_patient_holdout_truth or needs_discovery):
        return

    if minimum_tfx <= 0.0:
        return

    if needs_plca_truth:
        splits = _load_plca_fold_splits()
        split_samples = set(splits["Sample"])
        for cohort in _PLCA_CV_COHORTS:
            if not _is_patient_cohort(cohort):
                continue
            truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
            _require_existing_path(truth_path, description=f"patient-CV truth for {cohort}")
            truth_wide = load_truth(cohort)
            reference_index = pd.Index([sample_name for sample_name in truth_wide.columns if sample_name in split_samples])
            if len(reference_index) == 0:
                continue
            _load_filtered_subtypes_for_reference(
                cohort,
                reference_index,
                minimum_tfx=minimum_tfx,
                report_context=f"patient-CV truth {cohort}",
            )

    if needs_patient_holdout_truth:
        for cohort in VALIDATION_PATIENT_HOLDOUT_COHORTS:
            truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
            _require_existing_path(truth_path, description=f"Validation_Patient truth for {cohort}")
            _load_filtered_subtypes_for_reference(
                cohort,
                load_truth(cohort).columns,
                minimum_tfx=minimum_tfx,
                report_context=cohort,
            )

    if needs_discovery:
        for cohort, label_name in CONFIG_DISCOVERY_COHORTS:
            if not _is_patient_cohort(cohort):
                continue
            resolved_label = label_name or cohort
            if not _cohort_supports_tfx_subtypes(cohort, label_name=label_name):
                if minimum_tfx > 0.0:
                    _note_tfx_filter_unavailable(cohort)
                continue
            subtype_path = _require_existing_path(
                PROJECT_ROOT / "labels" / f"{resolved_label}_TFX_Subtype.tsv",
                description=f"Discovery_Patient subtype table for {cohort}",
            )
            reference_index = pd.Index([str(sample_name) for sample_name in load_subtypes(resolved_label).index])
            _load_filtered_subtypes_for_reference(
                cohort,
                reference_index,
                minimum_tfx=minimum_tfx,
                label_name=label_name,
                report_context=cohort,
            )




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

DEFAULT_ALL_GROUPS = list(GROUPS.keys())


def main() -> None:
    group_choices = list(GROUPS.keys())
    parser = argparse.ArgumentParser(
        description="Proteus analysis framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--group", type=str, required=True,
        choices=group_choices + ["all"],
        help=(
            "Analysis group to run. Available groups: "
            + ", ".join(group_choices + ["all"])
            + "."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Regenerate all plots even if PDFs already exist.",
    )
    parser.add_argument(
        "--minimum_tfx", type=float, default=0.05,
        help="Drop patient samples with TFx below this threshold at load time (default: 0.05).",
    )
    parser.add_argument(
        "--dim_ambiguous", action="store_true",
        help="Render and exclude single-gene samples inside the shared patient-CV AOR ambiguity band as hollow markers.",
    )
    parser.add_argument(
        "--apply_calibration", action="store_true",
        help="Apply the patient-OOF distribution calibration wrapper to Validation_Patient and Discovery_Patient Proteus outputs.",
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
        help=f"Monte Carlo expression draws for 90% score intervals (default: {_DEFAULT_MONTE_CARLO_DRAWS}).",
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

    selected_groups = DEFAULT_ALL_GROUPS if args.group == "all" else [args.group]
    _print_input_filter_banner(args.minimum_tfx)
    _preflight_minimum_tfx_filters(selected_groups, args.minimum_tfx)

    t_start = time.time()

    if args.group == "all":
        for name in selected_groups:
            GROUPS[name](args)
    else:
        GROUPS[args.group](args)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
