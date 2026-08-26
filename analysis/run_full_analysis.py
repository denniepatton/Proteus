#!/usr/bin/env python3
"""
run_full_analysis.py
Proteus analysis framework — main orchestrator.
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
    exact_paired_wilcoxon_pvalue,
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
    TDS_CV_FOLD_TUMOR_PREDICTION_PATHS,
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
    add_tfx_size_legend,
    batch_gene_scatters,
    batch_sample_scatters,
    inactive_confidence_threshold_plot,
    benchmark_cohort_boxplots,
    benchmark_cohort_histograms,
    benchmark_cohort_retention_curve,
    benchmark_cohort_metrics_lines,
    benchmark_patient_onoff_confusion_plot,
    benchmark_depth_lines,
    binary_status_boxplot,
    ci_width_vs_absolute_residual_scatter,
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
from analysis.ci_driver_analysis import run_ci_driver_analysis


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
_DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD = 5.0
_CROSSCOHORT_DRAW_ORDER: Tuple[str, ...] = (
    "CRPC-RA",
    "SCLC",
    "CRPC-WCDT",
    "BLCA-RA",
)
_CROSSCOHORT_DRAW_PRIORITY: Dict[str, int] = {
    cohort_name: idx for idx, cohort_name in enumerate(_CROSSCOHORT_DRAW_ORDER)
}

DEFAULT_GOI_PC: Dict[str, List[str]] = {
    "AR": ["AR", "KLK3", "NKX3-1", "FOLH1", "STEAP1"],
    "NEPC": ["SYP", "CHGA", "ASCL1", "INSM1", "SOX2", "DLL3"],
    "Proliferation": ["MKI67", "TOP2A", "CDK1"],
}
DEFAULT_GOI_BLCA: Dict[str, List[str]] = {
    "BLCA": ["ERBB2", "FGFR3", "NECTIN4", "TACSTD2", "TOP2A"],
}
DEFAULT_GOI_SCLC: Dict[str, List[str]] = {
    "NEPC": ["ASCL1", "NEUROD1", "INSM1", "SYP", "DLL3"],
    "Inflammatory": ["B2M", "TAP1", "TAP2", "PSMB8", "PSMB9"],
}

# Validation_Patient sample-scatter highlight sets: only the specified marker
# genes per cohort, each subtype keyed to its exact palette color.
VALIDATION_GOI_CRPC: Dict[str, List[str]] = {
    "ARPC": ["AR", "KLK3", "FOLH1", "NKX3-1"],
    "NEPC": ["CHGA", "SYP", "SOX2", "ASCL1"],
}
VALIDATION_GOI_BLCA: Dict[str, List[str]] = {
    "NE": ["CHGA", "SYP"],
    "UC": ["FOXA1", "GATA3"],
    "SQ": ["EGFR", "KRT14", "KRT15"],
}
VALIDATION_GOI_SCLC: Dict[str, List[str]] = {
    "POU2F3": ["POU2F3"],
    "NEUROD1": ["NEUROD1"],
    "ASCL1": ["ASCL1"],
    "ATOH1": ["ATOH1"],
}


_AOR_THRESHOLD_NOTICE_KEYS: set[tuple[Any, ...]] = set()


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


def _crosscohort_priority(group_name: Any) -> int:
    return _CROSSCOHORT_DRAW_PRIORITY.get(str(group_name), len(_CROSSCOHORT_DRAW_PRIORITY))


def _resolve_downstream_aor_thresholds(
    *,
    apply_calibration: bool,
    activity_threshold_summary: Optional[Dict[str, Any]],
    activity_odds_ratio_available: bool,
    report_context: str,
    dim_ambiguous: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve downstream AOR ambiguity thresholds for plotting and filtering.

    When calibration is disabled, enforce a fixed ambiguity band |AOR| <= 5.0
    instead of using selected patient-CV thresholds.
    """
    if not activity_odds_ratio_available:
        return None, None

    if apply_calibration:
        if activity_threshold_summary is None:
            return None, None
        return (
            float(activity_threshold_summary["lower_threshold"]),
            float(activity_threshold_summary["upper_threshold"]),
        )

    notice_key = ("fixed_aor_threshold", str(report_context), bool(dim_ambiguous))
    if notice_key not in _AOR_THRESHOLD_NOTICE_KEYS:
        _AOR_THRESHOLD_NOTICE_KEYS.add(notice_key)
        print(
            f"  [AOR fallback] {report_context}: --apply_calibration is OFF; "
            f"using fixed AOR ambiguity thresholds lower=-{_DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD:.1f}, "
            f"upper={_DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD:.1f} (|AOR| <= {_DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD:.1f})"
        )
        if dim_ambiguous:
            print(
                "                 --dim_ambiguous will hollow/exclude points inside this fixed ambiguity band."
            )

    return (-_DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD, _DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD)


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


def _collapse_crpc_ra_subtype_label_for_display(subtype_value: Any) -> str:
    subtype_text = str(subtype_value).strip().upper()
    if subtype_text == "ARPC":
        return "ARPC"
    if subtype_text == "NEPC":
        return "NEPC"
    return "Mixed"


def _crpc_ra_display_subtypes(subtypes_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if subtypes_df is None or subtypes_df.empty or "subtype" not in subtypes_df.columns:
        return subtypes_df
    display_df = subtypes_df.copy()
    display_df["subtype"] = display_df["subtype"].map(_collapse_crpc_ra_subtype_label_for_display)
    return display_df


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


def _nice_square_ticks(axis_lo: float, axis_hi: float, *, target_ticks: int = 5) -> np.ndarray:
    """Return rounded, human-readable ticks spanning [axis_lo, axis_hi]."""
    lo = float(axis_lo)
    hi = float(axis_hi)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.array([0.0, 1.0], dtype=np.float64)
    if hi <= lo:
        hi = lo + 1.0

    span = hi - lo
    raw_step = span / max(target_ticks - 1, 1)
    step_power = float(10.0 ** np.floor(np.log10(max(raw_step, 1e-12))))
    step = step_power
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        candidate = step_power * mult
        if candidate >= raw_step:
            step = candidate
            break

    tick_lo = float(np.floor(lo / step) * step)
    tick_hi = float(np.ceil(hi / step) * step)
    ticks = np.arange(tick_lo, tick_hi + 0.5 * step, step, dtype=np.float64)

    while ticks.size > 7:
        step *= 2.0
        tick_lo = float(np.floor(lo / step) * step)
        tick_hi = float(np.ceil(hi / step) * step)
        ticks = np.arange(tick_lo, tick_hi + 0.5 * step, step, dtype=np.float64)

    return np.round(ticks, 10)


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


_DISCOVERY_AMBIGUOUS_DROP_MIN_FRACTION = 0.5


def _ambiguous_genes_to_drop(
    activity_odds_ratio_wide: Optional[pd.DataFrame],
    sample_names: Sequence[str],
    *,
    lower_threshold: Optional[float],
    upper_threshold: Optional[float],
) -> pd.Index:
    """Return genes that are AOR-ambiguous in a majority of samples.

    A (gene, sample) entry is ambiguous when its activity odds ratio falls inside
    the band ``[lower_threshold, upper_threshold]``. A gene is dropped when at
    least half of its non-missing samples are ambiguous. For the symmetric fixed
    band (lower=-5, upper=5) this is exactly a per-gene median |AOR| <= 5, and it
    generalizes consistently to the asymmetric calibrated band.
    """
    if (
        activity_odds_ratio_wide is None
        or lower_threshold is None
        or upper_threshold is None
        or not np.isfinite(lower_threshold)
        or not np.isfinite(upper_threshold)
    ):
        return pd.Index([])

    cols = [sample for sample in sample_names if sample in activity_odds_ratio_wide.columns]
    if not cols:
        return pd.Index([])

    aor = activity_odds_ratio_wide[cols].apply(pd.to_numeric, errors="coerce")
    valid = aor.notna()
    ambiguous = (aor >= float(lower_threshold)) & (aor <= float(upper_threshold)) & valid
    # Genes with no usable AOR (all-NaN) yield NaN fraction and are never dropped.
    ambiguous_fraction = ambiguous.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)
    drop_mask = ambiguous_fraction >= _DISCOVERY_AMBIGUOUS_DROP_MIN_FRACTION
    return aor.index[drop_mask.fillna(False)]


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

    Uses the full TDS truth pool for the RNA-seq mixture fit and the
    existing PC-PDX study subset for the naive-example plots.
    """
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Examples")
    print("=" * 70)

    if _should_skip_section("Examples", args):
        return

    out_dir = _section_results_dir("Examples")
    _write_run_manifest("Examples", out_dir, args)
    tds_truth = _build_tds_truth(args.minimum_tfx)
    if tds_truth.empty:
        raise ValueError("TDS truth pool is empty; cannot build Examples outputs")
    cv_bundle = _load_main_tds_cv_full(args.minimum_tfx)
    cv_full = cv_bundle["cv_full"] if cv_bundle is not None else None
    run_examples_workflow(out_dir, tds_truth, args.overwrite, cv_full)
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
    sample_scatter_axis_max: Optional[float] = None,
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
    display_subtypes_df = _crpc_ra_display_subtypes(subtypes_df) if "CRPC-RA" in label else subtypes_df

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
        axis_max=sample_scatter_axis_max,
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
        axis_max=15.0,
    )

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.1f}s)")
    return summary


# Per-cohort gene-set config: heatmap gene sets, scatter gene sets, and GOI.
_VALIDATION_COHORT_CONFIG: Dict[str, Dict[str, Any]] = {
    "CRPC-WCDT": {
        "heatmap": ["PC-Phenotyping", "Targets_CRPC"],
        "scatter": ["Targets_CRPC"],
        "goi": VALIDATION_GOI_CRPC,
    },
    "BLCA-RA": {
        "heatmap": ["Targets_BLCA"],
        "scatter": ["Targets_BLCA"],
        "goi": VALIDATION_GOI_BLCA,
    },
    "CRPC-RA": {
        "heatmap": ["PC-Phenotyping", "Targets_CRPC"],
        "scatter": ["Targets_CRPC"],
        "goi": VALIDATION_GOI_CRPC,
    },
    "SCLC": {
        "heatmap": ["Targets_SCLC"],
        "scatter": ["Targets_SCLC"],
        "goi": VALIDATION_GOI_SCLC,
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
    display_subtypes_df = _crpc_ra_display_subtypes(subtypes_df) if cohort == "CRPC-RA" else subtypes_df

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
    _save_prediction_mean_wide_tsv(
        analysis_data.get("pred_wide"),
        os.path.join(cohort_out, "compiled_predictions_mean_wide.tsv"),
    )
    pred = analysis_data["pred_wide"]
    ci_lo = analysis_data.get("ci_lo_wide")
    ci_hi = analysis_data.get("ci_hi_wide")
    activity_odds_ratio_wide = analysis_data.get("activity_odds_ratio_wide")
    active_confidence_wide = analysis_data.get("active_confidence_wide")
    activity_lower_threshold, activity_upper_threshold = _resolve_downstream_aor_thresholds(
        apply_calibration=args.apply_calibration,
        activity_threshold_summary=activity_threshold_summary,
        activity_odds_ratio_available=activity_odds_ratio_wide is not None,
        report_context=f"Validation_Patient {cohort} holdout",
        dim_ambiguous=args.dim_ambiguous,
    )
    active_threshold = (
        _DOWNSTREAM_ACTIVE_DIM_THRESHOLD
        if active_confidence_wide is not None
        else None
    )
    label = f"{cohort}"

    suite_result = _run_holdout_suite(
        pred, truth_wide, label, cohort_out,
        ci_lo=ci_lo, ci_hi=ci_hi, subtypes_df=display_subtypes_df,
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
            subtypes_df=display_subtypes_df, tfx_dict=tfx_dict,
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
        subtypes_df=display_subtypes_df,
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
        "subtypes_df": display_subtypes_df,
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
    activity_lower_threshold, activity_upper_threshold = _resolve_downstream_aor_thresholds(
        apply_calibration=args.apply_calibration,
        activity_threshold_summary=activity_threshold_summary,
        activity_odds_ratio_available=activity_odds_ratio_wide is not None,
        report_context=f"Validation_Patient {cohort} CV",
        dim_ambiguous=args.dim_ambiguous,
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
    _save_prediction_mean_wide_tsv(
        full_analysis_bundle.get("pred_wide"),
        os.path.join(cohort_out, "compiled_predictions_mean_wide.tsv"),
    )

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
        sample_scatter_axis_max=15.0,
    )

    if cohort == "CRPC-RA":
        all_crpc_ra_pred_wide = pred_all_c
        all_crpc_ra_subtypes_df = cohort_subtypes_df

        crpc_ra_pred_draws = _build_mc_draws_from_mdn_tables(
            mdn_tables, pred_c.columns, n_draws=args.montecarlo_draws,
        )
        crpc_ra_score_bundle = _compute_crpc_ra_shortlist_score_bundle(
            truth_c, pred_c, pred_draws=crpc_ra_pred_draws,
        )
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
            all_crpc_ra_pred_wide=all_crpc_ra_pred_wide,
            all_crpc_ra_subtypes_df=all_crpc_ra_subtypes_df,
            minimum_tfx=args.minimum_tfx,
            apply_calibration=args.apply_calibration,
            calibration_state=calibration_state,
            overwrite=args.overwrite,
        )
        _run_crpc_ra_metastasis_composition_analysis(
            truth_c,
            pred_c,
            cohort_out,
            subtypes_df=subtypes_df,
            pred_draws=crpc_ra_pred_draws,
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
_CRPC_RA_PAIRED_FEATURE_DISPLAY: Dict[str, str] = {
    "TPBG": "TPBG/5T4",
}
_CRPC_RA_PLUVICTO_KEY_PATH: Path = PROJECT_ROOT / "reference" / "CRPC-RA-Pluvicto_key.tsv"
_CRPC_RA_FULL_HOLDOUT_PREDICTIONS: str = "results_holdout/BioDVN/CRPC-RA-Full/predictions_tumor.tsv"
_CRPC_RA_PLUVICTO_PAIRED_PATIENT_IDS: Tuple[str, ...] = (
    "22-049",
    "22-051",
    "23-008",
    "23-009",
    "23-015",
    "23-017",
    "24-011",
    "24-017",
    "24-018",
    "24-019",
)
_CRPC_RA_PLUVICTO_RNASEQ_TREATED_PATIENT_IDS: Tuple[str, ...] = (
    "22-049",
    "22-051",
    "23-008",
    "23-009",
    "24-011",
    "24-017",
    "24-018",
    "24-009",
    "25-002",
    "25-005",
)
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
    "DLL3",
    "KLK3",
)
_CRPC_RA_PLUVICTO_PAIRED_FEATURES: Tuple[str, ...] = (
    "FOLH1",
    "DLL3",
    "KLK3",
    "STEAP1",
    "INSM1",
    "KLK2",
    "ASCL1",
    "AR",
    "ARG.10",
    "NE.10",
    "CCP.31",
)
_CRPC_RA_PLUVICTO_PAIRED_RNA_FEATURES: Tuple[str, ...] = (
    "ARG.10",
    "NE.10",
    "CCP.31",
    "FOLH1",
    "AR",
    "ASCL1",
    "DLL3",
    "KLK3",
    "HES6",
    "SCG2",
    "SERPINA3",
    "STEAP1",
    "KLK2",
    "ONECUT2",
    "TPBG",
)
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


def _crpc_ra_pluvicto_rnaseq_overlap_key() -> pd.DataFrame:
    """Return overlap rows with matched CRPC-RA RNA-seq at the TAN/death timepoint."""
    key_df = _load_crpc_ra_pluvicto_key().copy()
    if "RNA-Seq_TAN" not in key_df.columns:
        raise ValueError("CRPC-RA / Pluvicto overlap key is missing required column: RNA-Seq_TAN")
    rnaseq_mask = key_df["RNA-Seq_TAN"].astype(str).str.strip().str.upper().eq("TRUE")
    filtered = key_df.loc[rnaseq_mask].copy()
    if filtered.empty:
        raise ValueError("CRPC-RA / Pluvicto overlap key does not contain any RNA-seq-overlap rows")
    return filtered


def _canonical_crpc_ra_sample_name(sample_id: Any, prefix_map: Dict[str, str]) -> str:
    sample_str = str(sample_id).strip()
    if sample_str in prefix_map:
        return str(prefix_map[sample_str])
    if sample_str.startswith("CRPC-RA_"):
        return sample_str
    return f"CRPC-RA_{sample_str}"


def _crpc_ra_overlap_prediction_candidates(
    overlap_rows: pd.DataFrame,
    truth_prefix_map: Dict[str, str],
) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()
    for _, row in overlap_rows.iterrows():
        tan_id = str(row.get("TAN_ID", "")).strip()
        if tan_id:
            canonical_tan = _canonical_crpc_ra_sample_name(tan_id, truth_prefix_map)
            for candidate in (tan_id, canonical_tan):
                if candidate and candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)
        for column_name in ("LTBx", "MC"):
            candidate = str(row.get(column_name, "")).strip()
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    return candidates


def _rename_crpc_ra_overlap_prediction_columns(
    bundle: Dict[str, Any],
    overlap_rows: pd.DataFrame,
    truth_prefix_map: Dict[str, str],
) -> None:
    rename_map: Dict[str, str] = {}
    for _, row in overlap_rows.iterrows():
        tan_id = str(row.get("TAN_ID", "")).strip()
        if not tan_id:
            continue
        canonical_name = _canonical_crpc_ra_sample_name(tan_id, truth_prefix_map)
        rename_map[tan_id] = canonical_name
        for column_name in ("LTBx", "MC"):
            candidate = str(row.get(column_name, "")).strip()
            if candidate:
                rename_map[candidate] = canonical_name

    if not rename_map:
        return

    for key, value in bundle.items():
        if isinstance(value, pd.DataFrame):
            renamed_columns = [rename_map.get(str(column), str(column)) for column in value.columns]
            renamed = value.copy()
            renamed.columns = pd.Index(renamed_columns)
            if renamed.columns.duplicated().any():
                renamed = renamed.loc[:, ~renamed.columns.duplicated()].copy()
            bundle[key] = renamed
        elif key == "fold_data" and isinstance(value, dict):
            for fold_bundle in value.values():
                if isinstance(fold_bundle, dict):
                    _rename_crpc_ra_overlap_prediction_columns(fold_bundle, overlap_rows, truth_prefix_map)


def _resolve_crpc_ra_overlap_prediction_sample(
    overlap_row: pd.Series,
    prediction_prefix_map: Dict[str, str],
) -> Optional[str]:
    for column_name in ("TAN_ID", "LTBx", "MC"):
        candidate = str(overlap_row.get(column_name, "")).strip()
        if not candidate:
            continue
        resolved = prediction_prefix_map.get(candidate)
        if resolved is not None:
            return str(resolved)
    return None


def _load_crpc_ra_overlap_prediction_supplement(
    sample_names: Sequence[Any],
    *,
    overlap_rows: pd.DataFrame,
    minimum_tfx: float,
    apply_calibration: bool,
    calibration_state: Optional[Dict[str, Any]],
    prefix_map: Dict[str, str],
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    requested_samples = [str(sample_name) for sample_name in sample_names]
    if not requested_samples:
        return None, None

    holdout_path = _require_relative_input_path(
        _CRPC_RA_FULL_HOLDOUT_PREDICTIONS,
        description="Validation_Patient CRPC-RA-Full overlap supplement predictions",
    )
    full_data = load_holdout_predictions_proteus_full(str(holdout_path))
    for key in ("pred_wide", "ci_lo_wide", "ci_hi_wide", "activity_odds_ratio_wide", "active_confidence_wide", *_MDN_PARAM_KEYS):
        _apply_prefix_map(full_data.get(key), prefix_map)
    _rename_crpc_ra_overlap_prediction_columns(full_data, overlap_rows, prefix_map)

    full_pred_wide = full_data.get("pred_wide")
    if full_pred_wide is None or full_pred_wide.empty:
        return None, None
    available_samples = [sample_name for sample_name in requested_samples if sample_name in full_pred_wide.columns]
    if not available_samples:
        return None, None

    supplement_subtypes = _load_filtered_subtypes_for_reference(
        "CRPC-RA",
        available_samples,
        minimum_tfx=minimum_tfx,
        report_context="Validation_Patient CRPC-RA overlap supplement",
    )
    if minimum_tfx > 0.0 and supplement_subtypes is not None:
        available_samples = [sample_name for sample_name in available_samples if sample_name in supplement_subtypes.index]
    if not available_samples:
        return None, None

    filtered_data = _filter_prediction_bundle_to_samples(full_data, available_samples)
    pred_wide = filtered_data.get("pred_wide")
    if pred_wide is None or pred_wide.empty:
        return None, None

    analysis_data = _prepare_proteus_analysis_bundle(filtered_data)
    if apply_calibration:
        if calibration_state is None:
            raise ValueError("Validation_Patient CRPC-RA overlap supplement calibration was requested but no calibration state was available")
        analysis_data = _apply_distribution_calibration_bundle(
            analysis_data,
            calibration_state,
            source_by_sample=_constant_source_by_sample(analysis_data["pred_wide"].columns, "CRPC-RA"),
            sample_fold=None,
            label="Validation_Patient CRPC-RA overlap supplement",
        )
    if supplement_subtypes is not None:
        supplement_subtypes = supplement_subtypes.reindex(analysis_data["pred_wide"].columns)
    return analysis_data, supplement_subtypes


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
    from matplotlib.ticker import StrMethodFormatter
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

    if point_x and point_y:
        all_vals = np.asarray(point_x + point_y, dtype=np.float64)
        finite_vals = all_vals[np.isfinite(all_vals)]
        if finite_vals.size > 0:
            raw_min = float(np.min(finite_vals))
            raw_max = float(np.max(finite_vals))
            axis_span = raw_max - raw_min
            if axis_span <= 0.0:
                axis_span = max(1.0, abs(raw_max) * 0.1)
                raw_min -= 0.5 * axis_span
                raw_max += 0.5 * axis_span
            axis_pad = 0.10 * axis_span
            if raw_min <= 0.0:
                axis_lo = 0.0
            else:
                axis_lo = max(0.0, raw_min - axis_pad)
            axis_hi = raw_max + axis_pad
            ax.set_xlim(axis_lo, axis_hi)
            ax.set_ylim(axis_lo, axis_hi)
            shared_ticks = _nice_square_ticks(axis_lo, axis_hi)
            if shared_ticks.size >= 2:
                ax.set_xlim(float(shared_ticks[0]), float(shared_ticks[-1]))
                ax.set_ylim(float(shared_ticks[0]), float(shared_ticks[-1]))
            ax.set_xticks(shared_ticks)
            ax.set_yticks(shared_ticks)
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.set_aspect("equal", adjustable="box")

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
    fig.subplots_adjust(right=0.72)
    fig.text(
        0.74,
        0.74,
        "\n".join(note_lines),
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
    )
    marker_note = []
    if ambiguous_mask is not None and int(mask_series.sum()) > 0:
        marker_note.append("Hollow = low-confidence")
    if marker_note:
        fig.text(0.74, 0.16, " ; ".join(marker_note), ha="left", va="bottom", fontsize=8)
    if tfx_dict is not None:
        add_tfx_size_legend(
            ax,
            [tfx_dict.get(sample_name, np.nan) for sample_name in sample_names],
            bbox_to_anchor=(1.18, 0.28),
            min_size=30.0,
            scale=300.0,
            default_size=300.0,
        )
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
    display_subtypes_df = _crpc_ra_display_subtypes(subtypes_df) if cohort == "CRPC-RA" else subtypes_df

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
    rows_gated: List[Dict[str, Any]] = []
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

        # Scatter dimming preserves the existing dim_ambiguous behaviour and
        # the passed activity thresholds; correlation gating (below) always
        # uses the default |AOR| <= 5 abstention band.
        scatter_ambiguous_mask = None
        if dim_ambiguous:
            scatter_ambiguous_mask = _ambiguous_sample_mask_for_gene(
                activity_odds_ratio_wide,
                gene,
                aligned_samples,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )

        # AOR-gated selection: drop points inside the default |AOR| <= 5 band.
        gated_ambiguous_mask = (
            _ambiguous_sample_mask_for_gene(
                activity_odds_ratio_wide,
                gene,
                aligned_samples,
                lower_threshold=-_DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD,
                upper_threshold=_DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD,
            )
            if activity_odds_ratio_wide is not None
            else None
        )
        gated_confident = (
            ~gated_ambiguous_mask.to_numpy(dtype=bool)
            if gated_ambiguous_mask is not None
            else np.ones(len(aligned_samples), dtype=bool)
        )
        n_low_confidence = int(len(aligned_samples) - int(gated_confident.sum()))

        # Ungated correlations use every aligned point.
        rows.append({
            "protein": protein,
            "gene": gene,
            "n": len(ihc_arr),
            "n_low_confidence": n_low_confidence,
            "rho_truth_ihc": float(spearmanr(truth_arr, ihc_arr).statistic),
            "rho_pred_ihc": float(spearmanr(pred_arr, ihc_arr).statistic),
            "rho_truth_pred": float(spearmanr(truth_arr, pred_arr).statistic),
        })

        # AOR-gated correlations drop the low-confidence points first.
        n_gated = int(gated_confident.sum())
        if n_gated >= MIN_CORRELATION_N:
            g_truth_ihc = float(spearmanr(truth_arr[gated_confident], ihc_arr[gated_confident]).statistic)
            g_pred_ihc = float(spearmanr(pred_arr[gated_confident], ihc_arr[gated_confident]).statistic)
            g_truth_pred = float(spearmanr(truth_arr[gated_confident], pred_arr[gated_confident]).statistic)
        else:
            g_truth_ihc = np.nan
            g_pred_ihc = np.nan
            g_truth_pred = np.nan
        rows_gated.append({
            "protein": protein,
            "gene": gene,
            "n": n_gated,
            "n_dropped_low_confidence": n_low_confidence,
            "rho_truth_ihc": g_truth_ihc,
            "rho_pred_ihc": g_pred_ihc,
            "rho_truth_pred": g_truth_pred,
        })

        # Per-protein scatter plots
        _maybe(_ihc_scatter, overwrite,
             ihc_arr, truth_arr, aligned_samples, protein, gene, "Truth",
             os.path.join(ihc_out, f"truth_vs_ihc_{protein}"),
                             subtypes_df=display_subtypes_df, tfx_dict=tfx_dict, ambiguous_mask=scatter_ambiguous_mask)
        _maybe(_ihc_scatter, overwrite,
             ihc_arr, pred_arr, aligned_samples, protein, gene, "Predicted",
             os.path.join(ihc_out, f"pred_vs_ihc_{protein}"),
                             subtypes_df=display_subtypes_df, tfx_dict=tfx_dict, ambiguous_mask=scatter_ambiguous_mask)

    if not rows:
        print("    [skip] No IHC-gene matches found")
        return

    rho_df = pd.DataFrame(rows)
    rho_df.to_csv(os.path.join(ihc_out, "ihc_correlations.tsv"),
                  sep="\t", index=False)
    rho_gated_df = pd.DataFrame(rows_gated)
    rho_gated_df.to_csv(os.path.join(ihc_out, "ihc_correlations_aor_gated.tsv"),
                        sep="\t", index=False)
    print(f"    IHC correlations: {len(rows)} proteins (ungated + |AOR|<=5-gated)")

    # Summary heatmaps: ungated and AOR-gated (points inside |AOR|<=5 dropped).
    _maybe(_ihc_heatmap, overwrite, rho_df,
           os.path.join(ihc_out, "ihc_correlation_heatmap"),
           title=f"IHC Correlation — {cohort}")
    _maybe(_ihc_heatmap, overwrite, rho_gated_df,
           os.path.join(ihc_out, "ihc_correlation_heatmap_aor_gated"),
           title=f"IHC Correlation (|AOR|>5 gated) — {cohort}")


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
    pred_draws: Optional[List[pd.DataFrame]] = None,
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
    pred_shortlist_scores_median = (
        _median_shortlist_scores_from_draws(pred_draws, selected_samples, genes=shared_genes)
        if shortlist_sets else pd.DataFrame()
    )

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
                    pred_score_median = np.nan
                    if (
                        not pred_shortlist_scores_median.empty
                        and pathway_name in pred_shortlist_scores_median.index
                        and sample_name in pred_shortlist_scores_median.columns
                    ):
                        pred_score_median = pd.to_numeric(
                            pd.Series([pred_shortlist_scores_median.loc[pathway_name, sample_name]]),
                            errors="coerce",
                        ).iloc[0]
                    path_row: Dict[str, Any] = {
                        "patient_id": patient_id,
                        "sample": sample_name,
                        "pathway": pathway_name,
                        "pathway_display": display_name,
                        "pred_score": float(pred_score),
                        "pred_score_mcmedian": float(pred_score_median) if np.isfinite(pred_score_median) else np.nan,
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
    show_identity_line: bool = False,
    fit_line_color: str = "#000000",
    fixed_square_limit: Optional[float] = None,
) -> None:
    """Scatter truth-vs-predicted score values using metadata-driven coloring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    from scipy.stats import t as t_dist

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

    shared_samples = list(shared)
    shared_samples = sorted(
        shared_samples,
        key=lambda sample_name: (
            _crosscohort_priority(meta.loc[sample_name, color_column]) if color_column in meta.columns else len(_CROSSCOHORT_DRAW_PRIORITY),
            str(sample_name),
        ),
    )

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    seen_groups: Dict[str, str] = {}
    for sample in shared_samples:
        group_name = str(meta.loc[sample, color_column]) if color_column in meta.columns else "Unknown"
        tfx_val = pd.to_numeric(pd.Series([meta.loc[sample, "tfx"]]), errors="coerce").iloc[0] \
            if "tfx" in meta.columns else np.nan
        color = palette.get(group_name, "#999999")
        sample_key = str(sample)
        status_text = str(meta.loc[sample, "status"]).strip().lower() if "status" in meta.columns else ""
        if status_text in {"validation", "cv", "oof"}:
            marker = "o"
        elif status_text == "holdout":
            marker = "D"
        else:
            marker = default_marker
        is_highlighted = sample_key in highlighted_set
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
            linewidths=0.9 if is_highlighted else 0.5,
            alpha=0.78,
            zorder=4 if is_highlighted else 3,
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
    pad = 0.10 * max(hi - lo, 1e-6)
    axis_lo = lo - pad
    axis_hi = hi + pad
    force_fixed_limit = False
    if fixed_square_limit is not None:
        fixed_limit = float(fixed_square_limit)
        if np.isfinite(fixed_limit) and fixed_limit > 0.0:
            axis_lo = -fixed_limit
            axis_hi = fixed_limit
            force_fixed_limit = True

    if show_identity_line:
        ax.plot([axis_lo, axis_hi], [axis_lo, axis_hi], ls="--", color="#BBBBBB", lw=1.2)

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
        x_line = np.linspace(axis_lo, axis_hi, 100)
        fit_fn = np.poly1d(coeffs)
        y_line = fit_fn(x_line)
        ax.plot(x_line, y_line, color=fit_line_color, lw=1.5, alpha=0.9, zorder=2)

        n_fit = int(len(truth_vals))
        if n_fit > 3:
            x_vals = truth_vals.values.astype(np.float64)
            y_vals = pred_vals.values.astype(np.float64)
            residuals = y_vals - fit_fn(x_vals)
            s_err = np.sqrt(np.sum(residuals ** 2) / (n_fit - 2))
            mean_x = float(np.mean(x_vals))
            sxx = float(np.sum((x_vals - mean_x) ** 2))
            if sxx > 0.0 and np.isfinite(s_err):
                t_val = float(t_dist.ppf(0.975, n_fit - 2))
                ci = t_val * s_err * np.sqrt(1.0 / n_fit + (x_line - mean_x) ** 2 / sxx)
                ax.fill_between(
                    x_line,
                    y_line - ci,
                    y_line + ci,
                    color="#9E9E9E",
                    alpha=0.22,
                    zorder=1,
                )

    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)
    ax.set_aspect("equal", adjustable="box")
    if force_fixed_limit:
        shared_ticks = np.arange(-fixed_limit, fixed_limit + 1e-9, 1.0)
        ax.set_xlim(-fixed_limit, fixed_limit)
        ax.set_ylim(-fixed_limit, fixed_limit)
    else:
        shared_ticks = _nice_square_ticks(axis_lo, axis_hi)
        if shared_ticks.size >= 2:
            ax.set_xlim(float(shared_ticks[0]), float(shared_ticks[-1]))
            ax.set_ylim(float(shared_ticks[0]), float(shared_ticks[-1]))
    ax.set_xticks(shared_ticks)
    ax.set_yticks(shared_ticks)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))

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
        for group_name, color in sorted(
            seen_groups.items(),
            key=lambda item: (_crosscohort_priority(item[0]), str(item[0])),
        )
    ]
    group_legend = None
    if handles:
        group_legend = ax.legend(handles=handles, title=legend_title, bbox_to_anchor=(1.02, 1.0),
                                 loc="upper left", fontsize=8)
    tfx_legend = None
    if "tfx" in meta.columns:
        tfx_legend = add_tfx_size_legend(ax, meta["tfx"], bbox_to_anchor=(1.18, 0.28))
    if group_legend is not None and tfx_legend is not None:
        ax.add_artist(group_legend)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_box_aspect(1)

    fig.subplots_adjust(right=0.69)
    fig.text(
        0.72,
        0.72,
        metric_text,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    if marker_note:
        fig.text(0.72, 0.20, marker_note, ha="left", va="bottom", fontsize=8)

    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _crpc_ra_feature_y_label(feature_name: str, *, predicted: bool) -> str:
    display_name = _CRPC_RA_SHORTLIST_DISPLAY.get(
        feature_name,
        _CRPC_RA_PAIRED_FEATURE_DISPLAY.get(feature_name, feature_name),
    )
    if feature_name in _CRPC_RA_SHORTLIST_DISPLAY:
        if predicted:
            return f"Predicted {display_name} ssGSEA score"
        return f"{display_name} ssGSEA score"
    if predicted:
        return "Predicted expression (log2CPM+1)"
    return "Expression (log2CPM+1)"


def _crpc_ra_feature_display_name(feature_name: str) -> str:
    return _CRPC_RA_SHORTLIST_DISPLAY.get(
        feature_name,
        _CRPC_RA_PAIRED_FEATURE_DISPLAY.get(feature_name, feature_name),
    )


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


def _build_mc_draws_from_mdn_tables(
    mdn_tables: Dict[str, Any],
    sample_columns: Sequence[Any],
    *,
    n_draws: int,
    seed: int = _CALIBRATION_DRAW_SEED,
) -> Optional[List[pd.DataFrame]]:
    """Draw Monte Carlo expression matrices for the given samples from stored MDN params."""
    if not all(mdn_tables.get(key) is not None for key in _MDN_PARAM_KEYS):
        return None
    cols = [col for col in sample_columns if col in mdn_tables["p0_wide"].columns]
    if not cols:
        return None
    return mdn_monte_carlo_sample(
        mdn_tables["p0_wide"][cols],
        mdn_tables["pi_gamma_wide"][cols],
        mdn_tables["pi_normal_wide"][cols],
        mdn_tables["gamma_beta_wide"][cols],
        mdn_tables["normal_mean_wide"][cols],
        mdn_tables["normal_stdev_wide"][cols],
        M=n_draws,
        seed=seed,
    )


def _median_shortlist_scores_from_draws(
    draws: Optional[List[pd.DataFrame]],
    sample_names: Sequence[Any],
    *,
    genes: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """Median-of-draws shortlist ssGSEA score matrix (sets x samples).

    The median (q50) across draws is the quantile-consistent
    full-distribution-aware companion to the raw prediction-mean scores. When
    ``genes`` is supplied the draws are restricted to that gene universe so the
    ssGSEA ranking matches a gene-restricted raw computation.
    """
    if not draws:
        return pd.DataFrame()
    shortlist_sets = _load_shortlist_sets()
    if not shortlist_sets:
        return pd.DataFrame()
    wanted = {str(sample_name) for sample_name in sample_names}
    gene_index = pd.Index(genes) if genes is not None else None
    sub_draws: List[pd.DataFrame] = []
    for draw in draws:
        restricted = draw if gene_index is None else draw.loc[draw.index.intersection(gene_index)]
        restricted = restricted[[col for col in restricted.columns if str(col) in wanted]]
        if not restricted.empty:
            sub_draws.append(restricted)
    if not sub_draws:
        return pd.DataFrame()
    _, median_scores, _, _ = _aggregate_ssgsea_score_matrices(sub_draws, shortlist_sets, min_size=3)
    return median_scores if median_scores is not None else pd.DataFrame()


def _compute_crpc_ra_shortlist_score_bundle(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    *,
    pred_draws: Optional[List[pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Compute the aligned CRPC-RA truth/pred shortlist score matrices once."""
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    return {
        "common_samples": common_samples,
        "truth_scores": _compute_shortlist_scores_for_matrix(truth_wide, common_samples),
        "pred_scores": _compute_shortlist_scores_for_matrix(pred_wide, common_samples),
        "pred_scores_median": _median_shortlist_scores_from_draws(pred_draws, common_samples),
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


_CRPC_RA_BOXPLOT_STYLE: Dict[str, Any] = {
    "font_family": "Helvetica",
    "title_size": 8.0,
    "axis_title_size": 6.0,
    "axis_tick_size": 6.0,
    "stats_text_size": 5.5,
    "geom_width": 0.94,
    "jitter_w": 0.045,
    "dot_size": 22.0,
    "box_lw": 0.35,
    "box_alpha": 0.72,
    "point_alpha": 0.78,
    "axis_lw": 0.6,
    "tick_lw": 0.5,
}


_CRPC_RA_BOXPLOT_R_BRIDGE: Optional[Tuple[Any, Any, Any]] = None


def _load_crpc_ra_boxplot_r_bridge() -> Tuple[Any, Any, Any]:
    """Load and initialize the R bridge for CRPC-RA publication boxplots."""
    global _CRPC_RA_BOXPLOT_R_BRIDGE
    if _CRPC_RA_BOXPLOT_R_BRIDGE is not None:
        return _CRPC_RA_BOXPLOT_R_BRIDGE

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.r(
        """
                suppressPackageStartupMessages({
                    library(ggplot2)
                })

                render_crpc_boxplot_publication <- function(
                    df_plot,
                    y_var,
                    y_label,
                    out_pdf,
                    group_order,
                    label_vec,
                    color_vec,
                    fill_vec,
                    title_text,
                    bracket_df,
                    extra_text,
                    width_in,
                    height_in
                ) {
                    df_sub <- df_plot[!is.na(df_plot[[y_var]]), , drop = FALSE]
                    if (nrow(df_sub) < 3) {
                        p_empty <- ggplot() + theme_void()
                        grDevices::pdf(out_pdf, width = width_in, height = height_in, useDingbats = FALSE)
                        print(p_empty)
                        grDevices::dev.off()
                        return(invisible(NULL))
                    }

                    display_levels <- as.character(group_order)
                    df_sub$group_display <- factor(as.character(df_sub$group), levels = display_levels)
                    df_sub <- df_sub[!is.na(df_sub$group_display), , drop = FALSE]
                    df_sub$x_pos <- match(as.character(df_sub$group_display), display_levels)
                    if (nrow(df_sub) < 3) {
                        p_empty <- ggplot() + theme_void()
                        grDevices::pdf(out_pdf, width = width_in, height = height_in, useDingbats = FALSE)
                        print(p_empty)
                        grDevices::dev.off()
                        return(invisible(NULL))
                    }

                    display_colors <- stats::setNames(as.character(color_vec), as.character(label_vec))
                    display_fills <- stats::setNames(as.character(fill_vec), as.character(label_vec))

                    y_vals <- df_sub[[y_var]]
                    y_data_lo <- min(y_vals, na.rm = TRUE)
                    y_data_hi <- max(y_vals, na.rm = TRUE)
                    if (!all(is.finite(c(y_data_lo, y_data_hi))) || y_data_lo >= y_data_hi) {
                        y_data_lo <- -1
                        y_data_hi <- 1
                    }
                    y_span <- max(y_data_hi - y_data_lo, 0.5)

                    caption_lines <- character(0)
                    if (!is.null(extra_text) && nzchar(as.character(extra_text))) {
                        caption_lines <- c(caption_lines, as.character(extra_text))
                    }
                    if (!is.null(bracket_df) && nrow(bracket_df) > 0) {
                        bracket_df <- bracket_df[order(bracket_df$row), , drop = FALSE]
                        pval_lines <- character(0)
                        for (ii in seq_len(nrow(bracket_df))) {
                            g1 <- as.character(bracket_df$group1[ii])
                            g2 <- as.character(bracket_df$group2[ii])
                            lbl <- as.character(bracket_df$label[ii])
                            if (nzchar(g1) && nzchar(g2) && nzchar(lbl)) {
                                pval_lines <- c(pval_lines, paste0(g1, " vs ", g2, ": ", lbl))
                            }
                        }
                        if (length(pval_lines) > 0) {
                            caption_lines <- c(caption_lines, pval_lines)
                        }
                    }
                    if (length(caption_lines) > 0) {
                        # Wrap captions to the actual device width so text never clips on the right.
                        wrap_width <- max(16L, as.integer(floor((as.numeric(width_in) - 0.20) * 16.0)))
                        wrapped_lines <- unlist(lapply(caption_lines, function(line_text) {
                            strwrap(
                                as.character(line_text),
                                width = wrap_width,
                                simplify = FALSE,
                                exdent = 2
                            )[[1]]
                        }))
                        caption_text <- if (length(wrapped_lines) > 0) paste(wrapped_lines, collapse = "\n") else NULL
                    } else {
                        caption_text <- NULL
                    }
                    caption_line_count <- if (!is.null(caption_text) && nzchar(caption_text)) {
                        length(strsplit(caption_text, "\n", fixed = TRUE)[[1]])
                    } else {
                        0L
                    }
                    # Keep panel geometry unchanged and grow only the full device when caption text is long.
                    height_out <- as.numeric(height_in)
                    if (caption_line_count > 0L) {
                        height_out <- height_out + 0.10 + (as.numeric(caption_line_count) * 0.12)
                    }
                    x_side_gap <- 0.06
                    x_min <- 0.5 - x_side_gap
                    x_max <- length(display_levels) + 0.5 + x_side_gap

                    p <- ggplot(
                        df_sub,
                        aes(x = x_pos, y = .data[[y_var]], fill = group_display, color = group_display)
                    ) +
                        geom_boxplot(
                            width = 0.94,
                            outlier.shape = NA,
                            linewidth = 0.35,
                            alpha = 0.72
                        ) +
                        geom_jitter(
                            width = 0.045,
                            size = 1.35,
                            alpha = 0.78,
                            shape = 16
                        ) +
                        scale_fill_manual(values = display_fills) +
                        scale_color_manual(values = display_colors) +
                        scale_x_continuous(
                            breaks = seq_along(display_levels),
                            labels = display_levels,
                            limits = c(x_min, x_max),
                            expand = expansion(add = c(0, 0))
                        ) +
                        labs(
                            x = NULL,
                            y = y_label,
                            title = title_text,
                            caption = caption_text
                        ) +
                        theme_minimal(base_size = 7, base_family = "Helvetica") %+replace%
                        theme(
                            plot.title = element_text(size = 8, face = "bold", hjust = 0.5, margin = margin(b = 2)),
                            plot.caption = element_text(size = 5.5, face = "italic", hjust = 0, color = "black", margin = margin(t = 3)),
                            plot.caption.position = "plot",
                            axis.title.y = element_text(size = 6, face = "bold", angle = 90, margin = margin(r = 2)),
                            axis.text.x = element_text(size = 6, face = "bold", color = unname(display_colors[display_levels])),
                            axis.text.y = element_text(size = 6, face = "bold", color = "black"),
                            axis.line = element_line(color = "black", linewidth = 0.6),
                            axis.ticks = element_line(color = "black", linewidth = 0.5),
                            axis.ticks.length = unit(1.5, "pt"),
                            panel.grid.major = element_blank(),
                            panel.grid.minor = element_blank(),
                            legend.position = "none",
                            plot.margin = margin(2, 4, 2, 4),
                            aspect.ratio = 2
                        )

                    y_plot_lo <- y_data_lo - 0.06 * y_span
                    y_plot_hi <- y_data_hi + 0.06 * y_span
                    p <- p + scale_y_continuous(
                        limits = c(y_plot_lo, y_plot_hi),
                        expand = expansion(mult = c(0.02, 0.02))
                    )

                    grDevices::pdf(out_pdf, width = width_in, height = height_out, useDingbats = FALSE)
                    print(p)
                    grDevices::dev.off()
                    invisible(NULL)
                }
                """
    )
    _CRPC_RA_BOXPLOT_R_BRIDGE = (ro, pandas2ri, localconverter)
    return _CRPC_RA_BOXPLOT_R_BRIDGE


def _render_crpc_ra_boxplot_r(
    df_plot: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    y_label: str,
    group_order: Sequence[str],
    group_colors: Dict[str, str],
    brackets: Sequence[Tuple[str, str, str]],
    extra_text: Optional[str] = None,
    width_in: float = 3.2,
    height_in: float = 3.2,
) -> None:
    """Render one publication boxplot in R/ggplot using the shared bridge."""
    if df_plot.empty:
        return
    ro, pandas2ri, localconverter = _load_crpc_ra_boxplot_r_bridge()
    converter = ro.default_converter + pandas2ri.converter

    labels = [str(label) for label in group_order]
    color_values = [str(group_colors[label]) for label in labels]
    fill_values = [_lighten_hex_color(str(group_colors[label]), factor=0.55) for label in labels]
    bracket_rows: List[Dict[str, Any]] = []
    for idx, (group_1, group_2, label) in enumerate(brackets, start=1):
        bracket_rows.append(
            {
                "group1": str(group_1),
                "group2": str(group_2),
                "label": str(label),
                "row": idx,
            }
        )
    bracket_df = pd.DataFrame(bracket_rows, columns=["group1", "group2", "label", "row"])
    extra_label = "" if extra_text is None else str(extra_text)

    local_df = df_plot.copy()
    local_df["group"] = local_df["group"].astype(str)
    local_df["value"] = pd.to_numeric(local_df["value"], errors="coerce")
    local_df = local_df.dropna(subset=["group", "value"])
    if local_df.empty:
        return

    with localconverter(converter):
        ro.globalenv["render_crpc_boxplot_publication"](
            ro.conversion.get_conversion().py2rpy(local_df),
            "value",
            y_label,
            f"{out_path}.pdf",
            ro.StrVector(labels),
            ro.StrVector(labels),
            ro.StrVector(color_values),
            ro.StrVector(fill_values),
            str(title),
            ro.conversion.get_conversion().py2rpy(bracket_df),
            extra_label,
            float(width_in),
            float(height_in),
        )


def _lighten_hex_color(hex_color: str, factor: float = 0.55) -> str:
    """Lighten a hex color toward white by the requested factor."""
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"#{color}" if color else "#D3D3D3"
    try:
        red = int(color[0:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
    except ValueError:
        return f"#{color}"
    factor_clamped = min(max(float(factor), 0.0), 1.0)
    red_out = int(round(red + (255 - red) * factor_clamped))
    green_out = int(round(green + (255 - green) * factor_clamped))
    blue_out = int(round(blue + (255 - blue) * factor_clamped))
    return f"#{red_out:02X}{green_out:02X}{blue_out:02X}"


def _style_crpc_ra_boxplot_axis(
    ax: Any,
    *,
    title: str,
    y_label: str,
    x_positions: Sequence[float],
    x_labels: Sequence[str],
    x_limits: Tuple[float, float],
) -> None:
    """Apply manuscript publication styling for CRPC-RA box-plot outputs."""
    font_family = _CRPC_RA_BOXPLOT_STYLE["font_family"]
    ax.set_title(
        title,
        fontsize=_CRPC_RA_BOXPLOT_STYLE["title_size"],
        fontweight="bold",
        fontname=font_family,
        pad=2.0,
    )
    ax.set_ylabel(
        y_label,
        fontsize=_CRPC_RA_BOXPLOT_STYLE["axis_title_size"],
        fontweight="bold",
        fontname=font_family,
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(
        list(x_labels),
        fontsize=_CRPC_RA_BOXPLOT_STYLE["axis_tick_size"],
        fontweight="bold",
        fontname=font_family,
    )
    ax.tick_params(
        axis="y",
        labelsize=_CRPC_RA_BOXPLOT_STYLE["axis_tick_size"],
        width=_CRPC_RA_BOXPLOT_STYLE["tick_lw"],
        length=2.0,
    )
    ax.tick_params(
        axis="x",
        width=_CRPC_RA_BOXPLOT_STYLE["tick_lw"],
        length=2.0,
    )
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontname(font_family)
        tick_label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(_CRPC_RA_BOXPLOT_STYLE["axis_lw"])
    ax.spines["bottom"].set_linewidth(_CRPC_RA_BOXPLOT_STYLE["axis_lw"])
    ax.grid(False)
    ax.set_xlim(float(x_limits[0]), float(x_limits[1]))
    try:
        ax.set_box_aspect(1.0)
    except AttributeError:
        pass


def _plot_crpc_ra_group_violin(
    group_df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    y_label: str,
) -> None:
    """Plot a two-group publication boxplot via R/ggplot."""
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

    p_value = float(mannwhitneyu(grouped[0], grouped[1], alternative="two-sided").pvalue)
    bracket_label = "ns" if p_value >= 0.05 else format_pval(p_value)
    _render_crpc_ra_boxplot_r(
        working.loc[working["group"].isin(group_order), ["group", "value"]],
        out_path,
        title=title,
        y_label=y_label,
        group_order=group_order,
        group_colors={
            "Untreated": "#D3D3D3",
            "Pluvicto-Treated": "#2F4F4F",
        },
        brackets=[("Untreated", "Pluvicto-Treated", bracket_label)],
        width_in=1.45,
        height_in=1.65,
    )


def _paired_wilcoxon_label(values_a: np.ndarray, values_b: np.ndarray) -> str:
    if len(values_a) < 2 or len(values_b) < 2:
        return "n/a"
    p_value = exact_paired_wilcoxon_pvalue(values_a, values_b)
    return "ns" if p_value >= 0.05 else format_pval(p_value)


def _plot_crpc_ra_paired_violin(
    paired_df: pd.DataFrame,
    out_path: str,
    *,
    title: str,
    y_label: str,
) -> None:
    """Plot paired C1-versus-PT publication boxplots via R/ggplot."""
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

    c1_values = wide["C1"].astype(float).values
    pt_values = wide["Death"].astype(float).values
    plot_df = pd.DataFrame(
        {
            "group": (["C1"] * len(c1_values)) + (["PT"] * len(pt_values)),
            "value": np.concatenate([c1_values, pt_values]),
        }
    )
    bracket_label = _paired_wilcoxon_label(c1_values, pt_values)
    _render_crpc_ra_boxplot_r(
        plot_df,
        out_path,
        title=title,
        y_label=y_label,
        group_order=["C1", "PT"],
        group_colors={
            "C1": "#006D77",
            "PT": "#D31F11",
        },
        brackets=[("C1", "PT", bracket_label)],
        extra_text=f"Exact paired Wilcoxon; n = {len(wide)}",
        width_in=1.45,
        height_in=1.65,
    )


def _plot_crpc_ra_paired_with_arpc_nepc_box(
    paired_df: pd.DataFrame,
    arpc_values: pd.Series,
    nepc_values: pd.Series,
    out_path: str,
    *,
    title: str,
    y_label: str,
) -> None:
    """Plot paired C1/PT box plots plus pure-ARPC and pure-NEPC reference boxes."""
    from scipy.stats import mannwhitneyu

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
    arpc_series = pd.to_numeric(arpc_values, errors="coerce").dropna().astype(float)
    nepc_series = pd.to_numeric(nepc_values, errors="coerce").dropna().astype(float)
    if wide.empty or arpc_series.empty or nepc_series.empty:
        return

    c1_values = wide["C1"].astype(float).values
    pt_values = wide["Death"].astype(float).values
    arpc_only_values = arpc_series.values.astype(float)
    nepc_only_values = nepc_series.values.astype(float)

    def _pairwise_mwu_label(values_a: np.ndarray, values_b: np.ndarray) -> str:
        if len(values_a) < 2 or len(values_b) < 2:
            return "n/a"
        try:
            p_value = float(mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)
        except ValueError:
            return "n/a"
        if not np.isfinite(p_value):
            return "n/a"
        return "ns" if p_value >= 0.05 else format_pval(p_value)

    def _summary_line(label: str, values: np.ndarray) -> str:
        if values.size == 0:
            return f"{label}: median=NA, IQR=NA"
        quartiles = np.percentile(values, [25.0, 50.0, 75.0])
        q1, median, q3 = float(quartiles[0]), float(quartiles[1]), float(quartiles[2])
        return f"{label}: median={median:.3f}, IQR=[{q1:.3f}, {q3:.3f}]"

    p_c1_pt = _paired_wilcoxon_label(c1_values, pt_values)
    p_pt_arpc = _pairwise_mwu_label(pt_values, arpc_only_values)
    p_pt_nepc = _pairwise_mwu_label(pt_values, nepc_only_values)
    p_arpc_nepc = _pairwise_mwu_label(arpc_only_values, nepc_only_values)

    plot_df = pd.DataFrame(
        {
            "group": (
                (["C1"] * len(c1_values))
                + (["PT"] * len(pt_values))
                + (["ARPC"] * len(arpc_only_values))
                + (["NEPC"] * len(nepc_only_values))
            ),
            "value": np.concatenate([c1_values, pt_values, arpc_only_values, nepc_only_values]),
        }
    )
    palette = load_palette()
    _render_crpc_ra_boxplot_r(
        plot_df,
        out_path,
        title=title,
        y_label=y_label,
        group_order=["C1", "PT", "ARPC", "NEPC"],
        group_colors={
            "C1": "#006D77",
            "PT": "#D31F11",
            "ARPC": palette.get("ARPC", "#0077BB"),
            "NEPC": palette.get("NEPC", "#EE7733"),
        },
        brackets=[],
        extra_text=(
            f"n: pairs={len(wide)}, (PT)ARPC={len(arpc_series)}, (PT)NEPC={len(nepc_series)}\n"
            f"p: C1-PT {p_c1_pt} (exact paired Wilcoxon); other contrasts MWU\n"
            f"PT-ARPC {p_pt_arpc}; PT-NEPC {p_pt_nepc}; ARPC-NEPC {p_arpc_nepc}\n"
            f"{_summary_line('C1', c1_values)} | {_summary_line('PT', pt_values)} | "
            f"{_summary_line('ARPC', arpc_only_values)} | {_summary_line('NEPC', nepc_only_values)}"
        ),
        width_in=2.05,
        height_in=1.85,
    )


def _plot_crpc_ra_paired_with_reference_box(
    paired_df: pd.DataFrame,
    reference_values: pd.Series,
    out_path: str,
    *,
    title: str,
    y_label: str,
    reference_label: str,
    reference_annotation_label: str,
    reference_color: str,
) -> None:
    """Plot paired C1/PT plus reference publication boxplots via R/ggplot."""
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
    reference_series = pd.to_numeric(reference_values, errors="coerce").astype(float)
    reference_series.index = reference_series.index.astype(str)
    reference_series = reference_series.reindex(wide.index).dropna()
    wide = wide.reindex(reference_series.index)
    if wide.empty or reference_series.empty:
        return

    c1_values = wide["C1"].astype(float).values
    pt_values = wide["Death"].astype(float).values
    ref_values = reference_series.values.astype(float)

    # All pairwise exact Wilcoxon comparisons, ordered shortest-span first so the
    # bracket-stacking in R draws the widest span (C1 vs reference) on top.
    brackets: List[Tuple[str, str, str]] = [
        ("C1", "PT", _paired_wilcoxon_label(c1_values, pt_values)),
        ("PT", reference_label, _paired_wilcoxon_label(pt_values, ref_values)),
        ("C1", reference_label, _paired_wilcoxon_label(c1_values, ref_values)),
    ]

    plot_df = pd.DataFrame(
        {
            "group": (
                (["C1"] * len(c1_values))
                + (["PT"] * len(pt_values))
                + ([reference_label] * len(ref_values))
            ),
            "value": np.concatenate([c1_values, pt_values, ref_values]),
        }
    )
    _render_crpc_ra_boxplot_r(
        plot_df,
        out_path,
        title=title,
        y_label=y_label,
        group_order=["C1", "PT", reference_label],
        group_colors={
            "C1": "#006D77",
            "PT": "#D31F11",
            reference_label: reference_color,
        },
        brackets=brackets,
        extra_text=(
            "Exact paired Wilcoxon (nominal p)\n"
            f"n triplets = {len(wide)}; n {reference_annotation_label} = {len(reference_series)}"
        ),
        width_in=1.75,
        height_in=1.75,
    )


def _load_crpc_ra_pluvicto_overlap_bundle(
    *,
    minimum_tfx: float,
    apply_calibration: bool,
    calibration_state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Load the Pluvicto C1 overlap bundle used by CRPC-RA paired analyses."""
    pluvicto_path = _require_relative_input_path(
        DISCOVERY_PROTEUS_PATHS.get("CRPC-Pluvicto"),
        description="Validation_Patient CRPC-Pluvicto overlap predictions",
    )
    full_data = load_holdout_predictions_proteus_full(str(pluvicto_path))
    pred_wide = full_data.get("pred_wide")
    if pred_wide is None or pred_wide.empty:
        print("    [skip] CRPC-RA Pluvicto overlap: no Pluvicto prediction matrix available")
        return None

    overlap_key = _load_crpc_ra_pluvicto_key()
    baseline_samples = _select_pluvicto_baseline_samples(pred_wide.columns)
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
        allowed_overlap = set(overlap_samples)
        patient_to_sample = {
            tan_id: c1_sample
            for tan_id, c1_sample in patient_to_sample.items()
            if c1_sample in allowed_overlap
        }
    if not overlap_samples:
        print("    [skip] CRPC-RA Pluvicto overlap: no overlap C1 samples remain after key filtering")
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

    display_subtypes_df = _crpc_ra_display_subtypes(subtypes_df)
    score_meta = pd.DataFrame(index=common_samples)
    if display_subtypes_df is not None:
        score_meta = display_subtypes_df.reindex(common_samples).copy()

    truth_scores = score_bundle["truth_scores"]
    raw_scores = score_bundle["pred_scores"]
    median_scores = score_bundle.get("pred_scores_median")
    if truth_scores.empty or raw_scores.empty:
        return

    scatter_out = ensure_dir(os.path.join(cohort_out, "ShortlistScores", "scatter_plots"))
    score_names = ("ARG.10", "NE.10", "CCP.31")
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
            show_identity_line=False,
            fit_line_color="#7A7A7A",
        )

        if (
            median_scores is not None
            and not median_scores.empty
            and score_name in median_scores.index
        ):
            _maybe(
                _crosscohort_score_scatter,
                overwrite,
                truth_scores.loc[score_name],
                median_scores.loc[score_name],
                score_meta,
                os.path.join(scatter_out, f"{score_name}_mcmedian"),
                title=f"CRPC-RA — {display_name} ssGSEA (Truth vs Predicted MC Median)",
                x_label=f"RNA-Seq {display_name} ssGSEA Score",
                y_label=f"Predicted {display_name} ssGSEA Score (MC median)",
                color_column="subtype",
                legend_title="Subtype",
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
            show_identity_line=False,
            fit_line_color="#7A7A7A",
            fixed_square_limit=3.0,
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

    pure_nepc_arpc_samples: List[str] = []
    if pure_subtypes is not None and "subtype" in pure_subtypes.columns:
        nepc_arpc_mask = (
            pure_subtypes["subtype"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin({"NEPC", "ARPC"})
        )
        pure_nepc_arpc_samples = [
            str(sample_name)
            for sample_name in pure_subtypes.index[nepc_arpc_mask]
            if str(sample_name) in truth_wide.columns and str(sample_name) in pred_wide.columns
        ]

    if len(pure_nepc_arpc_samples) < 2:
        return

    pure_nepc_arpc_subtypes = (
        pure_subtypes.loc[pure_nepc_arpc_samples].copy()
        if pure_subtypes is not None
        else None
    )
    if pure_nepc_arpc_subtypes is not None:
        pure_nepc_arpc_subtypes.to_csv(
            os.path.join(cohort_out, "PC-Phenotyping_pure_nepc_arpc_samples.tsv"),
            sep="\t",
        )

    _maybe(
        double_heatmap,
        overwrite,
        truth_wide[pure_nepc_arpc_samples],
        pred_wide[pure_nepc_arpc_samples],
        genes,
        os.path.join(cohort_out, "double_heatmap_PC-Phenotyping_pure_nepc_arpc"),
        subtypes=pure_nepc_arpc_subtypes,
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
    n_untreated_truth: int,
    n_untreated_pred: int,
    n_treated_truth: int,
    n_treated_pred: int,
) -> None:
    """Plot a four-box CRPC-RA comparison via R/ggplot."""
    plot_df = pd.DataFrame(
        {
            "group": (
                ["Untreated RNA-Seq"] * int(len(untreated_truth))
                + ["Untreated Predicted"] * int(len(untreated_pred))
                + ["Pluvicto-Treated RNA-Seq"] * int(len(treated_truth))
                + ["Pluvicto-Treated Predicted"] * int(len(treated_pred))
            ),
            "value": np.concatenate([untreated_truth, untreated_pred, treated_truth, treated_pred]).astype(float),
        }
    )
    group_order = [
        "Untreated RNA-Seq",
        "Untreated Predicted",
        "Pluvicto-Treated RNA-Seq",
        "Pluvicto-Treated Predicted",
    ]
    bracket_truth = "ns" if float(p_truth) >= 0.05 else format_pval(float(p_truth))
    bracket_pred = "ns" if float(p_pred) >= 0.05 else format_pval(float(p_pred))
    _render_crpc_ra_boxplot_r(
        plot_df,
        out_path,
        title=f"CRPC-RA Pluvicto vs Untreated - {gene_name}",
        y_label="Expression (log2CPM+1)",
        group_order=group_order,
        group_colors={
            "Untreated RNA-Seq": "#007191",
            "Untreated Predicted": "#F97C00",
            "Pluvicto-Treated RNA-Seq": "#007191",
            "Pluvicto-Treated Predicted": "#F97C00",
        },
        brackets=[
            ("Untreated RNA-Seq", "Pluvicto-Treated RNA-Seq", bracket_truth),
            ("Untreated Predicted", "Pluvicto-Treated Predicted", bracket_pred),
        ],
        extra_text=(
            f"Untreated RNA n={n_untreated_truth}, Pred n={n_untreated_pred}\n"
            f"Pluvicto-Treated RNA n={n_treated_truth}, Pred n={n_treated_pred}"
        ),
        width_in=1.75,
        height_in=1.75,
    )


def _run_crpc_ra_pluvicto_vs_untreated_analysis(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    cohort_out: str,
    *,
    death_bundle: Dict[str, Any],
    subtypes_df: Optional[pd.DataFrame],
    death_subtypes_df: Optional[pd.DataFrame],
    all_crpc_ra_pred_wide: Optional[pd.DataFrame],
    all_crpc_ra_subtypes_df: Optional[pd.DataFrame],
    minimum_tfx: float,
    apply_calibration: bool,
    calibration_state: Optional[Dict[str, Any]],
    overwrite: bool,
) -> None:
    """Expand CRPC-RA Pluvicto outputs across untreated, paired, and limma views."""
    from scipy.stats import mannwhitneyu

    full_truth_wide = load_truth("CRPC-RA")
    if minimum_tfx > 0.0:
        full_truth_subtypes = _load_filtered_subtypes_for_reference(
            "CRPC-RA",
            full_truth_wide.columns,
            minimum_tfx=minimum_tfx,
            report_context="Validation_Patient CRPC-RA overlap truth",
        )
        if full_truth_subtypes is not None:
            full_truth_wide = _filter_wide_to_samples(full_truth_wide, full_truth_subtypes.index)
    full_truth_prefix_map = _build_prefix_map(full_truth_wide.columns, "CRPC-RA")
    death_subtype_table = death_subtypes_df if death_subtypes_df is not None else subtypes_df

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

    all_overlap_rows = _load_crpc_ra_pluvicto_key()
    rnaseq_target_patient_ids = list(_CRPC_RA_PLUVICTO_RNASEQ_TREATED_PATIENT_IDS)
    paired_target_patient_ids = list(_CRPC_RA_PLUVICTO_PAIRED_PATIENT_IDS)
    rnaseq_target_samples = [
        _canonical_crpc_ra_sample_name(patient_id, full_truth_prefix_map)
        for patient_id in rnaseq_target_patient_ids
    ]
    paired_target_samples = [
        _canonical_crpc_ra_sample_name(patient_id, full_truth_prefix_map)
        for patient_id in paired_target_patient_ids
    ]
    overlap_prediction_target_samples = list(pd.Index(paired_target_samples).unique())

    available_truth_samples = set(full_truth_wide.columns.astype(str))
    missing_rnaseq_truth = [sample_name for sample_name in rnaseq_target_samples if sample_name not in available_truth_samples]
    if missing_rnaseq_truth:
        print(
            "    CRPC-RA RNA-seq overlap: truth matrix is missing "
            f"{len(missing_rnaseq_truth)} key-listed samples; continuing with available RNA-seq samples: "
            + ", ".join(sorted(missing_rnaseq_truth))
        )

    overlap_truth_samples = [
        sample_name
        for sample_name in full_truth_wide.columns.astype(str).tolist()
        if sample_name not in set(rnaseq_target_samples)
    ]

    overlap_death_bundle = death_bundle
    overlap_pred_wide = overlap_death_bundle.get("pred_wide")
    if overlap_pred_wide is None or overlap_pred_wide.empty:
        print("    [skip] CRPC-RA overlap analyses: no CRPC-RA prediction matrix available")
        return
    missing_overlap_predictions = [
        sample_name for sample_name in overlap_prediction_target_samples if sample_name not in overlap_pred_wide.columns
    ]
    if missing_overlap_predictions:
        supplement_bundle, supplement_subtypes = _load_crpc_ra_overlap_prediction_supplement(
            missing_overlap_predictions,
            overlap_rows=all_overlap_rows,
            minimum_tfx=minimum_tfx,
            apply_calibration=apply_calibration,
            calibration_state=calibration_state,
            prefix_map=full_truth_prefix_map,
        )
        if supplement_bundle is not None:
            overlap_death_bundle = _concatenate_prediction_bundles_by_columns(overlap_death_bundle, supplement_bundle)
            overlap_pred_wide = overlap_death_bundle.get("pred_wide")
            supplemented_samples = list(supplement_bundle["pred_wide"].columns)
            print(
                "    CRPC-RA overlap supplement: loaded "
                f"{len(supplemented_samples)} missing predictions from CRPC-RA-Full"
            )
            if supplement_subtypes is not None:
                death_subtype_table = (
                    supplement_subtypes
                    if death_subtype_table is None
                    else death_subtype_table.combine_first(supplement_subtypes)
                )
        missing_overlap_predictions = [
            sample_name for sample_name in missing_overlap_predictions if sample_name not in overlap_pred_wide.columns
        ]
    common_prediction_samples = full_truth_wide.columns.intersection(overlap_pred_wide.columns)
    if len(common_prediction_samples) == 0:
        print("    [skip] CRPC-RA Pluvicto-vs-Untreated: no overlapping truth/prediction samples")
        return

    missing_paired_predictions = [
        sample_name for sample_name in paired_target_samples if sample_name not in overlap_pred_wide.columns
    ]
    if missing_paired_predictions:
        print(
            "    CRPC-RA paired C1-death: missing death predictions after CRPC-RA-Full supplement; "
            "continuing with available paired patients: "
            + ", ".join(sorted(missing_paired_predictions))
        )

    treated_truth_samples = [sample_name for sample_name in rnaseq_target_samples if sample_name in available_truth_samples]
    treated_pred_samples = [sample_name for sample_name in treated_truth_samples if sample_name in common_prediction_samples]
    missing_rnaseq_predictions = [
        sample_name for sample_name in treated_truth_samples if sample_name not in treated_pred_samples
    ]
    untreated_truth_samples = [sample_name for sample_name in overlap_truth_samples if sample_name in available_truth_samples]
    untreated_pred_samples = [sample_name for sample_name in common_prediction_samples if sample_name not in set(treated_pred_samples)]

    print(
        "    CRPC-RA Pluvicto-vs-Untreated: "
        f"{len(treated_truth_samples)} RNA-seq overlap vs {len(untreated_truth_samples)} untreated RNA-seq samples"
    )
    print(
        "    CRPC-RA RNA-seq overlap: "
        f"{len(treated_truth_samples)} truth samples, {len(treated_pred_samples)} with matched predictions"
    )
    if missing_rnaseq_predictions:
        print(
            "    CRPC-RA RNA-seq overlap: continuing truth-only treated-vs-untreated outputs without predictions for "
            + ", ".join(sorted(missing_rnaseq_predictions))
        )
    if len(treated_truth_samples) == 0 or len(untreated_truth_samples) == 0:
        print("    [skip] CRPC-RA Pluvicto-vs-Untreated: one comparison group is empty")
        return

    analysis_out = ensure_dir(os.path.join(cohort_out, "CRPC-RA_Pluvicto-vs-Untreated"))
    true_pred_out = ensure_dir(os.path.join(analysis_out, "treated-vs-untreated_true-vs-pred"))
    pd.DataFrame(
        {
            "sample": untreated_truth_samples + treated_truth_samples,
            "group": (["Untreated"] * len(untreated_truth_samples)) + (["Pluvicto-Treated"] * len(treated_truth_samples)),
        }
    ).to_csv(os.path.join(analysis_out, "sample_groups.tsv"), sep="\t", index=False)
    rows: List[Dict[str, Any]] = []
    for gene_name in _CRPC_RA_PLUVICTO_TREATMENT_GENES:
        truth_present = gene_name in full_truth_wide.index
        pred_present = gene_name in overlap_pred_wide.index
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

        untreated_truth = pd.to_numeric(full_truth_wide.loc[gene_name, untreated_truth_samples], errors="coerce").dropna().to_numpy(dtype=float)
        treated_truth = pd.to_numeric(full_truth_wide.loc[gene_name, treated_truth_samples], errors="coerce").dropna().to_numpy(dtype=float)
        untreated_pred = pd.to_numeric(overlap_pred_wide.loc[gene_name, untreated_pred_samples], errors="coerce").dropna().to_numpy(dtype=float)
        treated_pred = pd.to_numeric(overlap_pred_wide.loc[gene_name, treated_pred_samples], errors="coerce").dropna().to_numpy(dtype=float)

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
            n_untreated_truth=len(untreated_truth_samples),
            n_untreated_pred=len(untreated_pred_samples),
            n_treated_truth=len(treated_truth_samples),
            n_treated_pred=len(treated_pred_samples),
        )

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(true_pred_out, "gene_boxplot_stats.tsv"), sep="\t", index=False)

    truth_scores = _compute_shortlist_scores_for_matrix(
        full_truth_wide,
        untreated_truth_samples + treated_truth_samples,
    )
    truth_only_out = ensure_dir(os.path.join(analysis_out, "treated-vs-untreated_rna-seq"))
    truth_feature_rows: List[Dict[str, Any]] = []
    for feature_name in _CRPC_RA_PLUVICTO_PAIRED_FEATURES:
        feature_series = _extract_expression_or_score_series(
            full_truth_wide,
            truth_scores,
            feature_name,
            untreated_truth_samples + treated_truth_samples,
        )
        if feature_series.empty:
            continue
        untreated_vals = feature_series.reindex(untreated_truth_samples).dropna().to_numpy(dtype=float)
        treated_vals = feature_series.reindex(treated_truth_samples).dropna().to_numpy(dtype=float)
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
        ylabel = _crpc_ra_feature_y_label(feature_name, predicted=False)
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

    death_pred_wide = overlap_death_bundle.get("pred_wide")
    if death_pred_wide is None or death_pred_wide.empty:
        print("    [skip] CRPC-RA paired C1-death: no CRPC-RA death prediction matrix available")
        return
    reference_pred_wide = (
        all_crpc_ra_pred_wide
        if all_crpc_ra_pred_wide is not None and not all_crpc_ra_pred_wide.empty
        else death_pred_wide
    )
    reference_subtype_table = (
        all_crpc_ra_subtypes_df
        if all_crpc_ra_subtypes_df is not None and not all_crpc_ra_subtypes_df.empty
        else death_subtype_table
    )
    predicted_death_samples = pd.Index(death_pred_wide.columns)
    if death_subtype_table is not None:
        predicted_death_samples = predicted_death_samples.union(pd.Index(death_subtype_table.index.astype(str)))
    death_prefix_map = _build_prefix_map(predicted_death_samples, "CRPC-RA")
    paired_rows: List[Dict[str, Any]] = []
    expected_paired_patients = list(_CRPC_RA_PLUVICTO_PAIRED_PATIENT_IDS)
    for patient_id in expected_paired_patients:
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

    missing_paired_patients = [patient_id for patient_id in expected_paired_patients if patient_id not in pair_df["patient_id"].astype(str).tolist()]
    if missing_paired_patients:
        print(
            "    CRPC-RA paired C1-death: missing expected paired patients after prediction supplement; "
            "continuing with available matches: "
            + ", ".join(sorted(missing_paired_patients))
        )
    print(
        "    CRPC-RA paired C1-death: "
        f"{len(pair_df)} matched paired prediction patients"
    )

    pair_df = pair_df.set_index("patient_id")
    pair_df.to_csv(os.path.join(analysis_out, "paired_patient_map.tsv"), sep="\t")

    paired_out = ensure_dir(os.path.join(analysis_out, "paired_c1-death"))
    paired_truth_out = ensure_dir(os.path.join(analysis_out, "paired_c1-death-rna"))
    c1_scores = _compute_shortlist_scores_for_matrix(c1_pred, pair_df["c1_sample"].tolist())
    death_scores = _compute_shortlist_scores_for_matrix(death_pred_wide, pair_df["death_sample"].tolist())
    death_truth_scores = _compute_shortlist_scores_for_matrix(full_truth_wide, pair_df["death_sample"].tolist())
    pure_nepc_samples: List[str] = []
    pure_arpc_samples: List[str] = []
    if reference_subtype_table is not None and "subtype" in reference_subtype_table.columns:
        pure_nepc_samples = [
            str(sample_name)
            for sample_name in reference_subtype_table.index
            if str(sample_name) in reference_pred_wide.columns
            and str(reference_subtype_table.loc[sample_name, "subtype"]).strip().upper() == "NEPC"
        ]
        pure_arpc_samples = [
            str(sample_name)
            for sample_name in reference_subtype_table.index
            if str(sample_name) in reference_pred_wide.columns
            and str(reference_subtype_table.loc[sample_name, "subtype"]).strip().upper() == "ARPC"
        ]
    nepc_scores = _compute_shortlist_scores_for_matrix(reference_pred_wide, pure_nepc_samples) if pure_nepc_samples else pd.DataFrame()
    arpc_scores = _compute_shortlist_scores_for_matrix(reference_pred_wide, pure_arpc_samples) if pure_arpc_samples else pd.DataFrame()
    paired_value_rows: List[Dict[str, Any]] = []
    paired_summary_rows: List[Dict[str, Any]] = []
    paired_truth_value_rows: List[Dict[str, Any]] = []
    paired_truth_wide_rows: List[Dict[str, Any]] = []
    paired_truth_summary_rows: List[Dict[str, Any]] = []
    paired_feature_set = set(_CRPC_RA_PLUVICTO_PAIRED_FEATURES)
    paired_truth_feature_set = set(_CRPC_RA_PLUVICTO_PAIRED_RNA_FEATURES)
    all_paired_feature_names = _dedupe_names_preserve_order(
        [*_CRPC_RA_PLUVICTO_PAIRED_FEATURES, *_CRPC_RA_PLUVICTO_PAIRED_RNA_FEATURES]
    )
    for feature_name in all_paired_feature_names:
        display_feature_name = _crpc_ra_feature_display_name(feature_name)
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
        death_truth_series = _extract_expression_or_score_series(
            full_truth_wide,
            death_truth_scores,
            feature_name,
            pair_df["death_sample"].tolist(),
        )
        feature_rows: List[Dict[str, Any]] = []
        feature_truth_wide_rows: List[Dict[str, Any]] = []
        for patient_id, row in pair_df.iterrows():
            c1_value = float(pd.to_numeric(pd.Series([c1_series.get(str(row["c1_sample"]), np.nan)]), errors="coerce").iloc[0])
            death_value = float(pd.to_numeric(pd.Series([death_series.get(str(row["death_sample"]), np.nan)]), errors="coerce").iloc[0])
            death_truth_value = float(pd.to_numeric(pd.Series([death_truth_series.get(str(row["death_sample"]), np.nan)]), errors="coerce").iloc[0])
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
            if np.isfinite(death_truth_value):
                paired_truth_value_rows.extend(
                    [
                        {
                            "feature": feature_name,
                            "patient_id": patient_id,
                            "group": "C1 prediction",
                            "sample": str(row["c1_sample"]),
                            "value": c1_value,
                        },
                        {
                            "feature": feature_name,
                            "patient_id": patient_id,
                            "group": "PT prediction",
                            "sample": str(row["death_sample"]),
                            "value": death_value,
                        },
                        {
                            "feature": feature_name,
                            "patient_id": patient_id,
                            "group": "PT RNA",
                            "sample": str(row["death_sample"]),
                            "value": death_truth_value,
                        },
                    ]
                )
                feature_truth_wide_rows.append(
                    {
                        "feature": feature_name,
                        "patient_id": patient_id,
                        "c1_sample": str(row["c1_sample"]),
                        "pt_sample": str(row["death_sample"]),
                        "c1_prediction": c1_value,
                        "pt_prediction": death_value,
                        "pt_rna": death_truth_value,
                    }
                )
        if not feature_rows:
            continue
        feature_df = pd.DataFrame(feature_rows)
        wide_feature = feature_df.pivot_table(index="patient_id", columns="visit", values="value", aggfunc="first")
        predicted_ylabel = _crpc_ra_feature_y_label(feature_name, predicted=True)
        truth_comparison_ylabel = _crpc_ra_feature_y_label(feature_name, predicted=False)
        if feature_name in paired_feature_set:
            paired_value_rows.extend(feature_rows)
        if feature_name in paired_feature_set and {"C1", "Death"}.issubset(wide_feature.columns):
            deltas = pd.to_numeric(wide_feature["Death"], errors="coerce") - pd.to_numeric(wide_feature["C1"], errors="coerce")
            c1_values = wide_feature["C1"].astype(float).values
            death_values = wide_feature["Death"].astype(float).values
            paired_summary_rows.append(
                {
                    "feature": feature_name,
                    "n_pairs": int(len(wide_feature)),
                    "median_c1": float(np.median(wide_feature["C1"].astype(float).values)),
                    "median_death": float(np.median(wide_feature["Death"].astype(float).values)),
                    "median_delta_death_minus_c1": float(np.median(deltas.values.astype(float))),
                    "mean_delta_death_minus_c1": float(np.mean(deltas.values.astype(float))),
                    "wilcoxon_p_death_vs_c1": exact_paired_wilcoxon_pvalue(c1_values, death_values),
                }
            )
        if feature_name in paired_feature_set:
            _maybe(
                _plot_crpc_ra_paired_violin,
                overwrite,
                feature_df,
                os.path.join(paired_out, _analysis_safe_name(feature_name)),
                title=f"CRPC-RA / Pluvicto paired C1 vs PT — {display_feature_name}",
                y_label=predicted_ylabel,
            )
            if pure_nepc_samples and pure_arpc_samples:
                nepc_series = _extract_expression_or_score_series(
                    reference_pred_wide,
                    nepc_scores if not nepc_scores.empty else None,
                    feature_name,
                    pure_nepc_samples,
                )
                arpc_series = _extract_expression_or_score_series(
                    reference_pred_wide,
                    arpc_scores if not arpc_scores.empty else None,
                    feature_name,
                    pure_arpc_samples,
                )
                if "sample" in feature_df.columns:
                    c1_pt_samples = set(feature_df["sample"].astype(str).tolist())
                    arpc_series = arpc_series.loc[~arpc_series.index.astype(str).isin(c1_pt_samples)]
                if not nepc_series.empty and not arpc_series.empty:
                    _maybe(
                        _plot_crpc_ra_paired_with_arpc_nepc_box,
                        overwrite,
                        feature_df,
                        arpc_series,
                        nepc_series,
                        os.path.join(paired_out, f"{_analysis_safe_name(feature_name)}_pure_arpc_nepc"),
                        title=f"CRPC-RA / Pluvicto paired C1 vs PT + ARPC + NEPC — {display_feature_name}",
                        y_label=predicted_ylabel,
                    )
        if feature_name in paired_truth_feature_set and feature_truth_wide_rows:
            feature_truth_wide_df = pd.DataFrame(feature_truth_wide_rows)
            c1_prediction_values = feature_truth_wide_df["c1_prediction"].to_numpy(dtype=float)
            pt_prediction_values = feature_truth_wide_df["pt_prediction"].to_numpy(dtype=float)
            pt_rna_values = feature_truth_wide_df["pt_rna"].to_numpy(dtype=float)
            paired_truth_wide_rows.extend(feature_truth_wide_rows)
            paired_truth_summary_rows.append(
                {
                    "feature": feature_name,
                    "n_triplets": int(len(feature_truth_wide_df)),
                    "median_c1_prediction": float(np.median(feature_truth_wide_df["c1_prediction"].astype(float).values)),
                    "median_pt_prediction": float(np.median(feature_truth_wide_df["pt_prediction"].astype(float).values)),
                    "median_pt_rna": float(np.median(feature_truth_wide_df["pt_rna"].astype(float).values)),
                    "median_delta_pt_prediction_minus_c1": float(
                        np.median(
                            (feature_truth_wide_df["pt_prediction"] - feature_truth_wide_df["c1_prediction"]).astype(float).values
                        )
                    ),
                    "median_delta_pt_prediction_minus_rna": float(
                        np.median(
                            (feature_truth_wide_df["pt_prediction"] - feature_truth_wide_df["pt_rna"]).astype(float).values
                        )
                    ),
                    "mean_delta_pt_prediction_minus_rna": float(
                        np.mean(
                            (feature_truth_wide_df["pt_prediction"] - feature_truth_wide_df["pt_rna"]).astype(float).values
                        )
                    ),
                    "wilcoxon_p_pt_prediction_vs_c1": exact_paired_wilcoxon_pvalue(
                        c1_prediction_values, pt_prediction_values
                    ),
                    "wilcoxon_p_pt_rna_vs_pt_prediction": exact_paired_wilcoxon_pvalue(
                        pt_prediction_values, pt_rna_values
                    ),
                    "wilcoxon_p_pt_rna_vs_c1": exact_paired_wilcoxon_pvalue(
                        c1_prediction_values, pt_rna_values
                    ),
                }
            )
            _maybe(
                _plot_crpc_ra_paired_with_reference_box,
                overwrite,
                feature_df,
                feature_truth_wide_df.set_index("patient_id")["pt_rna"],
                os.path.join(paired_truth_out, _analysis_safe_name(feature_name)),
                title=f"CRPC-RA / Pluvicto paired C1 vs PT + matched PT-RNA — {display_feature_name}",
                y_label=truth_comparison_ylabel,
                reference_label="PT-RNA",
                reference_annotation_label="PT-RNA",
                reference_color="#F97C00",
            )
    if paired_value_rows:
        pd.DataFrame(paired_value_rows).to_csv(
            os.path.join(paired_out, "paired_feature_values.tsv"),
            sep="\t",
            index=False,
        )
    if paired_summary_rows:
        paired_summary_df = pd.DataFrame(paired_summary_rows)
        paired_summary_df["wilcoxon_q_death_vs_c1"] = benjamini_hochberg(
            paired_summary_df["wilcoxon_p_death_vs_c1"]
        )
        paired_summary_df.to_csv(
            os.path.join(paired_out, "paired_feature_summary.tsv"),
            sep="\t",
            index=False,
        )
    if paired_truth_value_rows:
        pd.DataFrame(paired_truth_value_rows).to_csv(
            os.path.join(paired_truth_out, "paired_feature_values.tsv"),
            sep="\t",
            index=False,
        )
    if paired_truth_wide_rows:
        pd.DataFrame(paired_truth_wide_rows).to_csv(
            os.path.join(paired_truth_out, "paired_feature_values_wide.tsv"),
            sep="\t",
            index=False,
        )
    if paired_truth_summary_rows:
        paired_truth_summary_df = pd.DataFrame(paired_truth_summary_rows)
        for contrast_name in (
            "pt_prediction_vs_c1",
            "pt_rna_vs_pt_prediction",
            "pt_rna_vs_c1",
        ):
            paired_truth_summary_df[f"wilcoxon_q_{contrast_name}"] = benjamini_hochberg(
                paired_truth_summary_df[f"wilcoxon_p_{contrast_name}"]
            )
        paired_truth_summary_df.to_csv(
            os.path.join(paired_truth_out, "paired_feature_summary.tsv"),
            sep="\t",
            index=False,
        )

    pharmacodynamics_out = ensure_dir(os.path.join(analysis_out, "Pluvicto_pharmacodynamics"))
    death_pair_bundle = _filter_prediction_bundle_to_samples(overlap_death_bundle, pair_df["death_sample"].tolist())
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
    for gene_set_label, gene_set_name in (
        ("CancerSurfaceome", "CancerSurfaceome"),
        ("DruggableGenome_Tier1", "DruggableGenome_Tier1"),
    ):
        if not gene_list_available(gene_set_name):
            print(f"    [skip] CRPC-RA Pluvicto pharmacodynamics {gene_set_label}: gene list unavailable")
            continue
        gene_set_genes = get_gene_list(gene_set_name)
        gene_set_overlap = [
            gene_name
            for gene_name in _dedupe_names_preserve_order(gene_set_genes)
            if gene_name in combined_pred.index
        ]
        print(
            f"    CRPC-RA Pluvicto pharmacodynamics {gene_set_label}: "
            f"{len(gene_set_overlap)}/{len(gene_set_genes)} genes overlap predictions"
        )
        if not gene_set_overlap:
            print(f"    [skip] CRPC-RA Pluvicto pharmacodynamics {gene_set_label}: no overlapping genes")
            continue
        gene_set_out = ensure_dir(os.path.join(pharmacodynamics_out, f"limma_{gene_set_label}"))
        _run_weighted_limma_suite(
            combined_pred.loc[gene_set_overlap].copy(),
            combined_meta[["subtype"]],
            gene_set_out,
            overwrite,
            bundle=_filter_prediction_bundle_to_genes(combined_bundle, gene_set_overlap),
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


def _zscore_score_matrix(
    score_matrix: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Column-wise z-score transform for a score matrix."""
    if score_matrix is None or score_matrix.empty:
        return None
    return score_matrix.apply(_zscore_series, axis=1)


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
    combined_lower_threshold, combined_upper_threshold = _resolve_downstream_aor_thresholds(
        apply_calibration=args.apply_calibration,
        activity_threshold_summary=activity_threshold_summary,
        activity_odds_ratio_available=combined_activity_odds_ratio is not None,
        report_context=label,
        dim_ambiguous=args.dim_ambiguous,
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
            _, _, ci_lo_scores, ci_hi_scores = _aggregate_ssgsea_score_matrices(
                mc_draws, shortlist_sets, min_size=3)
            if ci_lo_scores is not None and not ci_lo_scores.empty:
                score_ci_lo_parts.append(ci_lo_scores)
            if ci_hi_scores is not None and not ci_hi_scores.empty:
                score_ci_hi_parts.append(ci_hi_scores)

        truth_scores_all = _concat_score_matrices(truth_score_parts)
        raw_scores_all = _concat_score_matrices(raw_score_parts)
        score_ci_lo_all = _concat_score_matrices(score_ci_lo_parts)
        score_ci_hi_all = _concat_score_matrices(score_ci_hi_parts)
        truth_scores_all_z = _zscore_score_matrix(truth_scores_all)
        raw_scores_all_z = _zscore_score_matrix(raw_scores_all)

        if truth_scores_all is not None:
            truth_scores_all.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_truth.tsv"), sep="\t")
        if raw_scores_all is not None:
            raw_scores_all.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_raw.tsv"), sep="\t")
        if truth_scores_all_z is not None:
            truth_scores_all_z.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_truth_zscore.tsv"), sep="\t")
        if raw_scores_all_z is not None:
            raw_scores_all_z.to_csv(
                os.path.join(shortlist_out, "shortlist_scores_raw_zscore.tsv"), sep="\t")
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
                    show_identity_line=False,
                    fit_line_color="#000000",
                    pred_ci_lo=(
                        score_ci_lo_all.loc[set_name]
                        if args.scatter_ci and score_ci_lo_all is not None and set_name in score_ci_lo_all.index
                        else None
                    ),
                    pred_ci_hi=(
                        score_ci_hi_all.loc[set_name]
                        if args.scatter_ci and score_ci_hi_all is not None and set_name in score_ci_hi_all.index
                        else None
                    ),
                )
                if (
                    truth_scores_all_z is not None
                    and raw_scores_all_z is not None
                    and set_name in truth_scores_all_z.index
                    and set_name in raw_scores_all_z.index
                ):
                    _maybe(
                        _crosscohort_score_scatter,
                        args.overwrite,
                        truth_scores_all_z.loc[set_name],
                        raw_scores_all_z.loc[set_name],
                        collected_subtypes,
                        os.path.join(scatter_out, f"{set_name}_zscore"),
                        title=f"{label} — {set_name} ssGSEA (Truth vs Predicted Z-score)",
                        x_label=f"RNA-Seq {set_name} ssGSEA Z-score",
                        y_label=f"Predicted {set_name} ssGSEA Z-score",
                        show_identity_line=False,
                        fit_line_color="#000000",
                        fixed_square_limit=3.0,
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
            emit_score_roc=True,
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
_GSEA_SHORTLIST_SETS = {"ARG.10", "NE.10", "CCP.31"}
_ENTREZ_SYMBOL_OVERRIDES: Dict[str, str] = {
    "25859": "PART1",
}
_PLUVICTO_DISCOVERY_VARIANTS = {"CRPC-Pluvicto"}


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

        combined_out = ensure_dir(os.path.join(gsea_out, "Hallmark_Shortlist"))
        combined_df = pd.concat([res_h_df, res_s_df], axis=0, ignore_index=True)
        _maybe(
            enrichment_dot_plot,
            overwrite,
            combined_df,
            os.path.join(combined_out, "enrichment_dotplot_hallmark_shortlist"),
            contrast_name=contrast,
            n_top=len(combined_df),
            title="GSEA: Hallmark + ARG.10 / NE.10 / CCP.31",
        )

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

    # NOTE: Per current analysis scope, clipped CI90-weighted limma/GSEA modes
    # are intentionally disabled. Keep only:
    #   - _mean_unweighted
    #   - _mean_weighted_ci90_unclipped
    #
    # for range_label, weight_floor, weight_ceiling in _LIMMA_WEIGHT_CLIP_RANGES:
    #     print(
    #         f"    limma weight normalization: mean-1 -> clip [{weight_floor:.2f}, {weight_ceiling:.2f}] -> mean-1"
    #     )
    #     weights = _normalize_precision_weight_matrix(
    #         raw_ci90_weights,
    #         weight_floor=weight_floor,
    #         weight_ceiling=weight_ceiling,
    #     )
    #     if weights is None:
    #         print(f"    [skip] limma weighted_ci90 ({range_label}): precision weights unavailable")
    #         continue
    #     tag = f"_mean_weighted_ci90_{range_label}"
    #     de_expr = _run_limma_de(
    #         expr_wide,
    #         subtypes_df,
    #         out_dir,
    #         overwrite,
    #         contrast=contrast,
    #         tag=tag,
    #         weights_wide=weights,
    #         apply_expression_filter=True,
    #     )
    #     if de_expr is not None:
    #         _run_gsea(
    #             de_expr,
    #             out_dir,
    #             overwrite,
    #             contrast=contrast,
    #             tag=tag,
    #             extra_gene_sets=extra_gene_sets,
    #         )


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
# Discovery_Patient: PAM50 intrinsic-subtype classification helpers
# ---------------------------------------------------------------------------


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
    fixed_square_limit: Optional[float] = None,
) -> None:
    """Plot one truth-vs-predicted PAM50 score scatter per subtype score."""

    score_columns = [column for column in truth_scores.columns if column in pred_scores.columns]
    if not score_columns:
        return

    for column in score_columns:
        truth_values = pd.to_numeric(truth_scores[column], errors="coerce")
        pred_values = pd.to_numeric(pred_scores[column], errors="coerce")
        subtype_name = column.split("_", 1)[1] if "_" in column else column
        shared = truth_values.index.intersection(pred_values.index)
        if sample_meta is not None:
            shared = shared.intersection(sample_meta.index)
        if len(shared) < MIN_CORRELATION_N:
            continue

        _crosscohort_score_scatter(
            truth_values.loc[shared],
            pred_values.loc[shared],
            sample_meta.loc[shared] if sample_meta is not None else pd.DataFrame(index=shared),
            f"{out_path}_{_analysis_safe_name(subtype_name)}",
            title=f"{title}: {subtype_name}",
            x_label="Truth PAM50 score",
            y_label="Predicted PAM50 score",
            color_column=color_column,
            legend_title=legend_title,
            show_identity_line=False,
            fit_line_color="#000000",
            fixed_square_limit=fixed_square_limit,
        )


_PAM50_ROC_SUBTYPES: Tuple[str, ...] = ("Basal", "LumA", "LumB")


def _pam50_score_roc_by_cohort(
    truth_scores: pd.DataFrame,
    pred_scores: pd.DataFrame,
    sample_meta: pd.DataFrame,
    pam50_root: str,
    overwrite: bool,
    *,
    cohort_label: str,
    variant_name: str,
) -> None:
    """One-vs-rest PAM50 ROC per cohort.

    Truth labels come from the dominant RNA-Seq PAM50 score among the
    Basal/LumA/LumB family; the predicted PAM50 score for each subtype is the
    continuous classifier ranked against a Subtype-vs-rest binary truth.
    """
    if sample_meta is None or "cohort" not in sample_meta.columns:
        return

    subtype_to_column: Dict[str, str] = {}
    for column in truth_scores.columns:
        if column not in pred_scores.columns:
            continue
        subtype_name = column.split("_", 1)[1] if "_" in column else column
        if subtype_name in _PAM50_ROC_SUBTYPES:
            subtype_to_column[subtype_name] = column
    restricted_subtypes = [name for name in _PAM50_ROC_SUBTYPES if name in subtype_to_column]
    if len(restricted_subtypes) < 2:
        print(f"    [skip] PAM50 ROC {cohort_label}: fewer than 2 of Basal/LumA/LumB scores present")
        return
    restricted_columns = [subtype_to_column[name] for name in restricted_subtypes]

    # Dominant RNA-Seq subtype restricted to the Basal/LumA/LumB score family.
    truth_restricted = truth_scores[restricted_columns].apply(pd.to_numeric, errors="coerce")
    truth_restricted = truth_restricted.dropna(how="all")
    if truth_restricted.empty:
        print(f"    [skip] PAM50 ROC {cohort_label}: no finite truth scores")
        return
    column_to_subtype = {subtype_to_column[name]: name for name in restricted_subtypes}
    dominant_truth = truth_restricted.idxmax(axis=1).map(column_to_subtype)

    roc_out = ensure_dir(os.path.join(pam50_root, "PAM50_ROC"))
    cohort_order = [cohort for cohort in _CROSSCOHORT_DRAW_ORDER]
    input_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for subtype_name in restricted_subtypes:
        score_column = subtype_to_column[subtype_name]
        curves: Dict[str, Tuple] = {}
        for cohort in cohort_order:
            cohort_samples = sample_meta.index[sample_meta["cohort"].astype(str) == cohort]
            shared = [
                sample for sample in cohort_samples
                if sample in dominant_truth.index and sample in pred_scores.index
            ]
            labels: List[int] = []
            scores: List[float] = []
            for sample in shared:
                pred_val = pd.to_numeric(
                    pd.Series([pred_scores.loc[sample, score_column]]), errors="coerce"
                ).iloc[0]
                if not np.isfinite(pred_val):
                    continue
                is_subtype = int(str(dominant_truth.loc[sample]) == subtype_name)
                labels.append(is_subtype)
                scores.append(float(pred_val))
                input_rows.append(
                    {
                        "subtype": subtype_name,
                        "cohort": cohort,
                        "sample": str(sample),
                        "dominant_truth_subtype": str(dominant_truth.loc[sample]),
                        "is_subtype": is_subtype,
                        "predicted_score": float(pred_val),
                    }
                )
            curve = _compute_score_roc_curve(labels, scores)
            if curve is not None:
                curves[cohort] = curve
                summary_rows.append(
                    {
                        "subtype": subtype_name,
                        "cohort": cohort,
                        "auc": float(curve[2]),
                        "n_samples": int(curve[3]),
                        "n_positive": int(np.sum(labels)),
                    }
                )
            else:
                print(f"    [warn] Skipping PAM50 {subtype_name} ROC for {cohort}: insufficient class balance")
        if curves:
            _maybe(
                roc_gene_calling,
                overwrite,
                curves,
                os.path.join(roc_out, f"pam50_roc_{_analysis_safe_name(subtype_name)}"),
                title=f"PAM50 ROC — {subtype_name} vs rest ({cohort_label}, {variant_name})",
                model_order=cohort_order,
            )

    if input_rows:
        pd.DataFrame(input_rows).to_csv(
            os.path.join(roc_out, "pam50_roc_inputs.tsv"), sep="\t", index=False
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(roc_out, "pam50_roc_auc_summary.tsv"), sep="\t", index=False
        )


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
    emit_score_roc: bool = False,
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
                    "pearson_p": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "mae": np.nan,
                }
                if int(valid.sum()) >= MIN_CORRELATION_N:
                    truth_valid = truth_values.loc[valid]
                    pred_valid = pred_values.loc[valid]
                    _, row["pearson_r"], row["pearson_p"], row["spearman_rho"], row["spearman_p"] = _truth_pred_correlation_values(
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
                    _, all_pearson_r, all_pearson_p, all_spearman_rho, all_spearman_p = _truth_pred_correlation_values(
                        flat_truth_values,
                        flat_pred_values,
                    )
                    score_rows.append({
                        "score_type": score_type,
                        "score_column": "all_scores",
                        "subtype": "All",
                        "n_samples": int(flat_truth_values.size),
                        "pearson_r": all_pearson_r,
                        "pearson_p": all_pearson_p,
                        "spearman_rho": all_spearman_rho,
                        "spearman_p": all_spearman_p,
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
                os.path.join(variant_out, "pam50_score_scatter_truth_vs_pred_raw"),
                title=f"PAM50 truth vs predicted scores (raw) — {cohort_label} ({variant_name})",
                sample_meta=sample_meta,
            )
            truth_scores_z = truth_scores.apply(_zscore_series, axis=0)
            pred_scores_z = pred_scores.apply(_zscore_series, axis=0)
            _maybe(
                _plot_pam50_score_scatter,
                overwrite,
                truth_scores_z,
                pred_scores_z,
                os.path.join(variant_out, "pam50_score_scatter_truth_vs_pred_zscore"),
                title=f"PAM50 truth vs predicted scores (z-score) — {cohort_label} ({variant_name})",
                sample_meta=sample_meta,
                fixed_square_limit=3.0,
            )

            if emit_score_roc:
                _pam50_score_roc_by_cohort(
                    truth_scores,
                    pred_scores,
                    sample_meta if sample_meta is not None else pd.DataFrame(),
                    os.path.join(out_dir, f"PAM50{tag}"),
                    overwrite,
                    cohort_label=cohort_label,
                    variant_name=variant_name,
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

    if os.path.basename(os.path.normpath(model_out)) == "ssGSEA" and not tag:
        ssgsea_out = ensure_dir(os.path.join(model_out, "Hallmark"))
    else:
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
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Run ssGSEA on MC draws and save median-of-draws scores plus q5/q95 intervals."""
    import gseapy as gp

    _require_existing_path(_GSEA_HALLMARK_GMT, description="Hallmark GMT")

    ssgsea_out = ensure_dir(os.path.join(model_out, "ssGSEA_score_intervals"))
    median_tsv = os.path.join(ssgsea_out, "ssgsea_hallmark_scores_mcmedian.tsv")
    ci_lo_tsv = os.path.join(ssgsea_out, "ssgsea_hallmark_score_ci_q5.tsv")
    ci_hi_tsv = os.path.join(ssgsea_out, "ssgsea_hallmark_score_ci_q95.tsv")

    if (
        not overwrite
        and os.path.exists(median_tsv)
        and os.path.exists(ci_lo_tsv)
        and os.path.exists(ci_hi_tsv)
    ):
        return (
            pd.read_csv(median_tsv, sep="\t", index_col=0),
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
        return None, None, None

    # Stack and compute statistics
    stacked = np.stack([s.values for s in all_scores], axis=0)  # (M, P, S)
    median_scores = pd.DataFrame(
        np.percentile(stacked, 50.0, axis=0),
        index=all_scores[0].index, columns=all_scores[0].columns)
    ci_lo = pd.DataFrame(
        np.percentile(stacked, 5.0, axis=0),
        index=all_scores[0].index, columns=all_scores[0].columns)
    ci_hi = pd.DataFrame(
        np.percentile(stacked, 95.0, axis=0),
        index=all_scores[0].index, columns=all_scores[0].columns)

    median_scores.to_csv(median_tsv, sep="\t")
    ci_lo.to_csv(ci_lo_tsv, sep="\t")
    ci_hi.to_csv(ci_hi_tsv, sep="\t")
    print("    MC ssGSEA median scores + intervals saved")

    return median_scores, ci_lo, ci_hi


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


def _run_pluvicto_shortlist_scores_mc(
    draws: List[pd.DataFrame],
    model_out: str,
    overwrite: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Aggregate mean-of-draws shortlist scores plus q5/q95 intervals for Pluvicto."""
    shortlist_sets = _load_shortlist_sets()
    if not shortlist_sets:
        print("    [skip] Shortlist MC: no shortlist sets available")
        return None, None, None

    shortlist_out = ensure_dir(os.path.join(model_out, "ShortlistScoreIntervals"))
    median_tsv = os.path.join(shortlist_out, "shortlist_scores_mcmedian.tsv")
    ci_lo_tsv = os.path.join(shortlist_out, "shortlist_score_ci_q5.tsv")
    ci_hi_tsv = os.path.join(shortlist_out, "shortlist_score_ci_q95.tsv")

    if (
        not overwrite
        and os.path.exists(median_tsv)
        and os.path.exists(ci_lo_tsv)
        and os.path.exists(ci_hi_tsv)
    ):
        return (
            pd.read_csv(median_tsv, sep="\t", index_col=0),
            pd.read_csv(ci_lo_tsv, sep="\t", index_col=0),
            pd.read_csv(ci_hi_tsv, sep="\t", index_col=0),
        )

    _, median_scores, ci_lo, ci_hi = _aggregate_ssgsea_score_matrices(
        draws, shortlist_sets, min_size=3)
    if median_scores is None or ci_lo is None or ci_hi is None:
        print("    [skip] Shortlist MC: no genes present in predictions")
        return None, None, None

    median_scores.to_csv(median_tsv, sep="\t")
    ci_lo.to_csv(ci_lo_tsv, sep="\t")
    ci_hi.to_csv(ci_hi_tsv, sep="\t")
    print("    MC shortlist median scores + intervals saved")
    return median_scores, ci_lo, ci_hi


def _load_all_samples_tfx_ctdnaq() -> pd.DataFrame:
    """Load the all-sample TFx/ctDNA-Q reference table indexed by sample identifier."""
    path = _require_existing_path(
        PROJECT_ROOT / "reference" / "all_samples_TFx_ctDNAq.csv",
        description="all-sample TFx/ctDNA-Q table",
    )
    table = pd.read_csv(path)
    required_columns = {"Identifier", "Sample_ID", "Cycle", "ctdna-q"}
    missing_columns = sorted(required_columns.difference(table.columns))
    if missing_columns:
        raise ValueError(
            "all_samples_TFx_ctDNAq.csv is missing required columns: "
            + ", ".join(missing_columns)
        )
    duplicated = table.loc[table["Identifier"].astype(str).duplicated(), "Identifier"].astype(str).tolist()
    if duplicated:
        raise ValueError(
            "all_samples_TFx_ctDNAq.csv contains duplicate Identifier values: "
            + ", ".join(sorted(set(duplicated)))
        )
    table = table.copy()
    table["Identifier"] = table["Identifier"].astype(str)
    table["Sample_ID"] = table["Sample_ID"].astype(str)
    table["Cycle"] = table["Cycle"].astype(str)
    table["ctdna-q"] = pd.to_numeric(table["ctdna-q"], errors="coerce")
    return table.set_index("Identifier", drop=False)


def _pluvicto_ctdna_q_series(sample_names: Sequence[Any]) -> pd.Series:
    """Return ctDNA-Q values for Pluvicto samples, preferring exact Identifier matches."""
    ctdna_table = _load_all_samples_tfx_ctdnaq()
    values: Dict[str, float] = {}
    for sample_name_any in sample_names:
        sample_name = str(sample_name_any)
        value = np.nan
        if sample_name in ctdna_table.index:
            value = pd.to_numeric(pd.Series([ctdna_table.loc[sample_name, "ctdna-q"]]), errors="coerce").iloc[0]
        else:
            patient_id = extract_pluvicto_patient_id(sample_name)
            fallback_rows = ctdna_table.loc[
                (ctdna_table["Sample_ID"] == patient_id)
                & (ctdna_table["Cycle"].str.upper() == "C1")
            ]
            if len(fallback_rows) == 1:
                value = pd.to_numeric(fallback_rows["ctdna-q"], errors="coerce").iloc[0]
        values[sample_name] = float(value) if np.isfinite(value) else np.nan
    return pd.Series(values, dtype=float)


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
) -> List[str]:
    """Select the baseline (C1) Pluvicto samples for discovery analysis."""
    return [str(sample_name) for sample_name in sample_names if str(sample_name).endswith("C1")]


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

    _maybe_forest_plots(
        overwrite,
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
            _maybe_forest_plots(
                overwrite,
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
    hallmark_scores: Optional[pd.DataFrame] = None,
    subtypes_df: Optional[pd.DataFrame] = None,
    cohort_label: str = "CRPC-Pluvicto",
    tag: str = "",
) -> None:
    """Scatter plots: FOLH1 vs PSMA-PET, scores vs imaging/TFx/ctDNA-Q."""
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
    ctdna_q_by_sample = _pluvicto_ctdna_q_series(samples)
    ctdna_q_by_pid = {
        extract_pluvicto_patient_id(sample_name): float(ctdna_q)
        for sample_name, ctdna_q in ctdna_q_by_sample.items()
        if np.isfinite(ctdna_q)
    }
    sample_meta["ctdna_q"] = pd.Series(ctdna_q_by_pid).reindex(pids)

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

    # --- CCP31 and pathway scores vs TFx / ctDNA-Q; CCP31 vs FDG ---
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

    score_series: List[Tuple[str, str, pd.Series, str]] = []
    if ccp31_series is not None:
        score_series.append(("CCP31", "CCP31", ccp31_series, ccp31_xlabel))

    if hallmark_scores is not None:
        for hallmark_term, display_name in (
            ("HALLMARK_INFLAMMATORY_RESPONSE", "Hallmark inflammatory response"),
            ("HALLMARK_INTERFERON_GAMMA_RESPONSE", "Hallmark interferon gamma response"),
        ):
            if hallmark_term not in hallmark_scores.index:
                print(f"    [skip] {display_name}: Hallmark score not available")
                continue
            hallmark_series = pd.to_numeric(hallmark_scores.loc[hallmark_term].reindex(samples), errors="coerce")
            hallmark_series.index = pids
            score_series.append(
                (
                    hallmark_term,
                    display_name,
                    hallmark_series,
                    f"{display_name} ssGSEA NES",
                )
            )

    comparators: List[Tuple[str, str, str, pd.Series]] = []
    if "tfx" in sample_meta.columns:
        comparators.append(("TFx", "TFx", "TFx", pd.to_numeric(sample_meta["tfx"], errors="coerce")))
    comparators.append(("ctDNA_Q", "ctDNA-Q", "ctDNA-Q", pd.to_numeric(sample_meta["ctdna_q"], errors="coerce")))

    for score_key, score_display, values, ylabel in score_series:
        for comparator_key, comparator_display, xlabel, comparator_values in comparators:
            valid = values.notna() & comparator_values.notna()
            if int(valid.sum()) < 5:
                print(
                    f"    [skip] {score_display} vs {comparator_display}: fewer than 5 samples with both values"
                )
                continue
            _maybe(
                clinical_correlation_scatter,
                overwrite,
                comparator_values[valid],
                values[valid],
                os.path.join(corr_out, f"{_analysis_safe_name(score_key)}_vs_{comparator_key}"),
                xlabel=f"log10({xlabel})" if comparator_key == "ctDNA_Q" else xlabel,
                ylabel=ylabel,
                title=f"{score_display} vs {comparator_display}",
                sample_meta=sample_meta.loc[valid.index],
                x_log10=comparator_key == "ctDNA_Q",
            )

    if ccp31_series is not None:
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


def _run_discovery_cohort(
    cohort: str,
    subtype_label: Optional[str],
    out_dir: str,
    args: argparse.Namespace,
    calibration_state: Optional[Dict[str, Any]] = None,
    activity_threshold_summary: Optional[Dict[str, Any]] = None,
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
        subtypes_df = _load_filtered_subtypes_for_reference(
            cohort,
            full_data["pred_wide"].columns,
            minimum_tfx=args.minimum_tfx,
            label_name=label_name,
            report_context=f"Discovery_Patient {cohort}",
        )
        if args.minimum_tfx > 0.0 and subtypes_df is not None:
            full_data = _filter_prediction_bundle_to_samples(full_data, subtypes_df.index)
        if _is_pluvicto_discovery_cohort(cohort):
            original_samples = [str(sample_name) for sample_name in full_data["pred_wide"].columns]
            keep_samples = _select_pluvicto_baseline_samples(
                full_data["pred_wide"].columns,
            )
            baseline_drop_reasons: Dict[str, str] = {}
            for sample_name in original_samples:
                if sample_name in keep_samples:
                    continue
                if not sample_name.endswith("C1"):
                    baseline_drop_reasons[sample_name] = "non-baseline timepoint"
                else:
                    baseline_drop_reasons[sample_name] = "excluded by Pluvicto baseline selector"
            full_data = _filter_prediction_bundle_to_samples(full_data, keep_samples)
            filter_label = "all *C1"
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

        # --dim_ambiguous: drop genes that are AOR-ambiguous in a majority of
        # samples from the prediction table before any Discovery_Patient analysis
        # (limma, ssGSEA, PAM50, survival, clinical). The band
        # is the calibrated UB/LB when --apply_calibration is on, otherwise the
        # fixed |AOR| <= 5 fallback, matching the scatter dim/hollow semantics.
        ambiguous_drop_index = pd.Index([])
        if args.dim_ambiguous:
            ambiguous_lower, ambiguous_upper = _resolve_downstream_aor_thresholds(
                apply_calibration=args.apply_calibration,
                activity_threshold_summary=activity_threshold_summary,
                activity_odds_ratio_available=activity_odds_ratio_wide is not None,
                report_context=f"Discovery_Patient {cohort} gene drop",
                dim_ambiguous=args.dim_ambiguous,
            )
            ambiguous_drop_index = _ambiguous_genes_to_drop(
                activity_odds_ratio_wide,
                list(pred.columns),
                lower_threshold=ambiguous_lower,
                upper_threshold=ambiguous_upper,
            )
            if len(ambiguous_drop_index) > 0:
                pred = pred.drop(index=ambiguous_drop_index, errors="ignore")
                print(
                    f"    --dim_ambiguous: dropped {len(ambiguous_drop_index)} AOR-ambiguous genes "
                    f"(majority of samples inside band) → {pred.shape[0]} genes retained for downstream analyses"
                )
            elif activity_odds_ratio_wide is None:
                print("    --dim_ambiguous: no activity odds ratio surface available; no genes dropped")

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

        # Keep MC draws gene-aligned with the (possibly ambiguity-filtered) means
        # so MC-median ssGSEA scores the same gene universe as the raw pass.
        if len(ambiguous_drop_index) > 0 and mc_draws is not None:
            mc_draws = [draw.drop(index=ambiguous_drop_index, errors="ignore") for draw in mc_draws]

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
        if subtypes_df is not None and "subtype" in subtypes_df.columns:
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
                hallmark_scores=ssgsea_raw,
                subtypes_df=subtypes_df,
                cohort_label=cohort,
            )

            # MC draws yield both q5/q95 score intervals and the median-of-draws
            # (q50) point scores; the latter feed a second survival /
            # clinical-correlation pass so every raw ssGSEA-driven analysis has a
            # quantile-consistent full-distribution-aware counterpart.
            if mc_draws is not None:
                ssgsea_mcmedian, _, _ = _run_pluvicto_ssgsea_mc(
                    mc_draws, model_out, args.overwrite)
                shortlist_mcmedian, _, _ = _run_pluvicto_shortlist_scores_mc(
                    mc_draws, model_out, args.overwrite)
                _run_pluvicto_survival(
                    pred,
                    ssgsea_mcmedian,
                    shortlist_mcmedian,
                    model_out,
                    args.overwrite,
                    args.minimum_tfx,
                    tag="_mcmedian",
                )
                _run_pluvicto_clinical_correlations(
                    pred,
                    model_out,
                    args.overwrite,
                    shortlist_scores=shortlist_mcmedian,
                    hallmark_scores=ssgsea_mcmedian,
                    subtypes_df=subtypes_df,
                    cohort_label=cohort,
                    tag="_mcmedian",
                )


_NEPC_ROC_COHORTS: Tuple[str, ...] = ("CRPC-WCDT", "CRPC-UW", "CRPC-RA")
_NEPC_ROC_UNKNOWN_TOKENS = {"UNKNOWN", "UNKOWN", "NAN", ""}
# NE-like (NE-positive) subtypes for the detection truth. AMPC (amphicrine)
# co-expresses the AR and NE programs, so it is encoded as double-positive
# (NE-positive); DNPC (double-negative) lacks the NE program and stays NE-negative.
_NEPC_ROC_NE_POSITIVE_TOKENS = {"NEPC", "AMPC"}
_NEPC_ROC_PURE_TOKENS = {"ARPC", "NEPC"}
_NEPC_ROC_MC_SEED = 42


def _subtype_nepc_flag(subtype_text: object) -> Optional[int]:
    """Return 1 if a subtype annotation is NE-like, 0 if known but not, None if unknown.

    Cohort annotations mix comma- and slash-delimited multi-subtype strings
    (e.g. CRPC-UW ``ARPC,NEPC``, CRPC-RA ``NEPC/ARPC``).  Unknown placeholders
    are excluded so they never count as NE-negative truth.  AMPC is encoded as
    NE-positive (double-positive) and DNPC as NE-negative (double-negative).
    """
    raw = str(subtype_text).strip()
    tokens = [tok.strip().upper() for tok in re.split(r"[,/]", raw) if tok.strip()]
    known = [tok for tok in tokens if tok not in _NEPC_ROC_UNKNOWN_TOKENS]
    if not known:
        return None
    return int(any(tok in _NEPC_ROC_NE_POSITIVE_TOKENS for tok in known))


def _subtype_pure_arpc_nepc_flag(subtype_text: object) -> Optional[int]:
    """Return 1 for pure NEPC, 0 for pure ARPC, None otherwise.

    "Pure" means the annotation lists exactly one distinct known subtype and that
    subtype is ARPC or NEPC.  Mixed calls (e.g. ``ARPC,NEPC``, ``NEPC/DNPC``) and
    the double-positive/double-negative subtypes (AMPC, DNPC) are excluded.
    """
    raw = str(subtype_text).strip()
    tokens = [tok.strip().upper() for tok in re.split(r"[,/]", raw) if tok.strip()]
    known = {tok for tok in tokens if tok not in _NEPC_ROC_UNKNOWN_TOKENS}
    if len(known) != 1:
        return None
    token = next(iter(known))
    if token not in _NEPC_ROC_PURE_TOKENS:
        return None
    return int(token == "NEPC")


def _nepc_truth_series(cohort: str, *, pure: bool = False) -> pd.Series:
    """Build a per-sample NE-detection truth Series (1/0) for one cohort.

    With ``pure=False`` every known sample is labelled NE-like (1) vs not (0),
    treating AMPC as double-positive and DNPC as double-negative.  With
    ``pure=True`` only pure ARPC (0) / pure NEPC (1) samples are retained and all
    mixed / AMPC / DNPC samples are dropped.
    """
    if cohort == "CRPC-UW":
        path = _require_existing_path(
            PROJECT_ROOT / "labels" / "CRPC-UW_TFX_Subtype-Mixed.tsv",
            description="CRPC-UW mixed subtype labels",
        )
        subtype = pd.read_csv(path, sep="\t").set_index("sample")["subtype"]
    else:
        subtype = load_subtypes(cohort)["subtype"]
    flag_fn = _subtype_pure_arpc_nepc_flag if pure else _subtype_nepc_flag
    labels: Dict[str, int] = {}
    for sample, text in subtype.items():
        flag = flag_fn(text)
        if flag is not None:
            labels[str(sample)] = flag
    return pd.Series(labels, dtype="int64")


def _nepc_cohort_mdn_bundle(cohort: str, minimum_tfx: float) -> Dict[str, Any]:
    """Load the full MDN prediction bundle (point + mixture params) for a cohort."""
    if cohort == "CRPC-WCDT":
        path = _require_existing_path(
            PROJECT_ROOT / VALIDATION_PATIENT_HOLDOUT_PATHS[cohort],
            description=f"{cohort} holdout predictions",
        )
        return load_holdout_predictions_proteus_full(str(path))
    if cohort == "CRPC-UW":
        path = _require_existing_path(
            PROJECT_ROOT / DISCOVERY_PROTEUS_PATHS[cohort],
            description=f"{cohort} discovery predictions",
        )
        return load_holdout_predictions_proteus_full(str(path))
    if cohort == "CRPC-RA":
        cv_bundle = _load_main_tds_cv_full(minimum_tfx)
        if cv_bundle is None:
            raise ValueError("CRPC-RA NEPC ROC requires the canonical patient CV OOF bundle")
        cv_full = cv_bundle["cv_full"]
        cohort_map = cv_bundle["cohort_map"]
        pred_wide = cv_full["pred_wide"]
        keep = [s for s in pred_wide.columns if cohort_map.get(str(s)) == "CRPC-RA"]
        if not keep:
            raise ValueError("No CRPC-RA samples present in the canonical CV OOF bundle")
        return {
            key: (cv_full[key][keep] if cv_full.get(key) is not None else None)
            for key in ("pred_wide", *_MDN_PARAM_KEYS)
        }
    raise ValueError(f"Unsupported NEPC ROC cohort: {cohort}")


def _zscore_series(values: pd.Series) -> pd.Series:
    """Population z-score of a numeric Series; constant/empty input maps to 0."""
    numeric = pd.to_numeric(values, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=numeric.index)
    return (numeric - float(numeric.mean())) / std


def _ne_minus_ar_composite(ne_scores: pd.Series, ar_scores: pd.Series) -> pd.Series:
    """Cohort-relative z(NE) - z(AR) composite over the shared samples.

    Each score is standardized across the supplied (cohort) samples so the
    subtraction folds in both "NE-like" and "not AR-like" on a common scale.
    """
    shared = ne_scores.index.intersection(ar_scores.index)
    if len(shared) == 0:
        return pd.Series(dtype="float64")
    return _zscore_series(ne_scores.loc[shared]) - _zscore_series(ar_scores.loc[shared])


def _ne_ar_mc_medians(
    mc_draws: List[pd.DataFrame],
    ne_sets: Dict[str, List[str]],
    ar_sets: Dict[str, List[str]],
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """Single ssGSEA pass per draw -> (NE median, AR median, composite median).

    NE.10 and AR.10 are scored once per draw.  The NE.10 and AR.10 point estimates
    are the across-draw medians of their ssGSEA scores, while the composite is the
    per-draw z(NE)-z(AR) reduced by the across-draw median so every MC point
    estimate stays quantile-consistent with the full-distribution scoring.
    """
    combined = {**ne_sets, **ar_sets}
    ne_name = next(iter(ne_sets))
    ar_name = next(iter(ar_sets)) if ar_sets else None
    ne_cols: List[pd.Series] = []
    ar_cols: List[pd.Series] = []
    composite_cols: List[pd.Series] = []
    for draw in mc_draws:
        score_matrix = _compute_ssgsea_score_matrix(draw, combined, min_size=3)
        if score_matrix.empty or ne_name not in score_matrix.index:
            continue
        ne_series = pd.to_numeric(score_matrix.loc[ne_name], errors="coerce")
        ne_series.index = [str(sample) for sample in ne_series.index]
        ne_cols.append(ne_series)
        if ar_name is not None and ar_name in score_matrix.index:
            ar_series = pd.to_numeric(score_matrix.loc[ar_name], errors="coerce")
            ar_series.index = [str(sample) for sample in ar_series.index]
            ar_cols.append(ar_series)
            composite_cols.append(_ne_minus_ar_composite(ne_series, ar_series))
    ne_median = pd.concat(ne_cols, axis=1).median(axis=1) if ne_cols else None
    ar_median = pd.concat(ar_cols, axis=1).median(axis=1) if ar_cols else None
    composite_median = pd.concat(composite_cols, axis=1).median(axis=1) if composite_cols else None
    return ne_median, ar_median, composite_median


def _run_nepc_detection_roc(out_dir: str, args: argparse.Namespace) -> None:
    """NE-detection ROC across CRPC cohorts for NE.10 and the AR/NE composite.

    Two score families are plotted: the NE.10 ssGSEA score, and the cohort-relative
    composite ``z(NE.10) - z(AR.10)`` which folds in "not AR-like".  Each family is
    plotted for two point estimates (raw prediction means and MC-median draws) and
    two truth encodings: the full ``all`` truth (NE-like vs not, AMPC double-positive
    and DNPC double-negative) and a ``pure`` truth restricted to pure ARPC vs pure
    NEPC samples (mixed / AMPC / DNPC dropped).
    """
    roc_dir = ensure_dir(os.path.join(out_dir, "NEPC_Detection_ROC"))
    pdf_paths: Dict[Tuple[str, str, str], str] = {}
    for score_kind, kind_stem in (("ne10", ""), ("composite", "composite_")):
        for truth_set, truth_stem in (("all", ""), ("pure", "pure_")):
            for variant in ("raw", "mc_median"):
                stem = f"nepc_detection_roc_{kind_stem}{truth_stem}{variant}"
                pdf_paths[(score_kind, truth_set, variant)] = os.path.join(roc_dir, stem)
    if not args.overwrite and all(os.path.exists(f"{p}.pdf") for p in pdf_paths.values()):
        return

    print("\n  ── NEPC-detection ROC (NE.10 + z(NE)-z(AR)) ──")
    shortlist_sets = _load_shortlist_sets()
    if "NE.10" not in shortlist_sets:
        raise ValueError("Shortlist GMT is missing the required 'NE.10' gene set")
    if "ARG.10" not in shortlist_sets:
        raise ValueError("Shortlist GMT is missing the required 'ARG.10' gene set")
    ne_sets = {"NE.10": shortlist_sets["NE.10"]}
    ar_sets = {"AR.10": shortlist_sets["ARG.10"]}
    combined_sets = {**ne_sets, **ar_sets}

    truth_set_titles = {"all": "all subtypes", "pure": "pure ARPC vs NEPC"}
    variant_titles = {"raw": "raw scores", "mc_median": "MC median"}
    kind_titles = {"ne10": "NE.10", "composite": "z(NE.10)-z(AR.10)"}
    curves: Dict[Tuple[str, str, str], Dict[str, Tuple]] = {key: {} for key in pdf_paths}
    input_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for cohort in _NEPC_ROC_COHORTS:
        truth_by_set = {
            "all": _nepc_truth_series(cohort),
            "pure": _nepc_truth_series(cohort, pure=True),
        }
        if all(truth.empty or truth.nunique() < 2 for truth in truth_by_set.values()):
            print(f"    [skip] {cohort}: no NE-detection truth set has both classes")
            continue
        bundle = _nepc_cohort_mdn_bundle(cohort, args.minimum_tfx)
        pred_wide = bundle["pred_wide"]
        if pred_wide is None or pred_wide.empty:
            print(f"    [skip] {cohort}: no predictions available")
            continue

        raw_matrix = _compute_ssgsea_score_matrix(pred_wide, combined_sets, min_size=3)
        if raw_matrix.empty or "NE.10" not in raw_matrix.index:
            print(f"    [skip] {cohort}: NE.10 raw score unavailable")
            continue
        ne10_raw = pd.to_numeric(raw_matrix.loc["NE.10"], errors="coerce")
        ne10_raw.index = [str(sample) for sample in ne10_raw.index]
        composite_raw: Optional[pd.Series] = None
        if "AR.10" in raw_matrix.index:
            ar10_raw = pd.to_numeric(raw_matrix.loc["AR.10"], errors="coerce")
            ar10_raw.index = [str(sample) for sample in ar10_raw.index]
            composite_raw = _ne_minus_ar_composite(ne10_raw, ar10_raw)

        mc_draws = mdn_monte_carlo_sample(
            bundle["p0_wide"],
            bundle["pi_gamma_wide"],
            bundle["pi_normal_wide"],
            bundle["gamma_beta_wide"],
            bundle["normal_mean_wide"],
            bundle["normal_stdev_wide"],
            M=args.montecarlo_draws,
            seed=_NEPC_ROC_MC_SEED,
        )
        ne10_mc, _, composite_mc = _ne_ar_mc_medians(mc_draws, ne_sets, ar_sets)
        if ne10_mc is None or ne10_mc.empty:
            print(f"    [skip] {cohort}: NE.10 MC score unavailable")
            continue

        # ssGSEA / MC scores are independent of the truth encoding, so compute
        # them once per cohort and reuse them across both truth sets.
        score_by_key: Dict[Tuple[str, str], Optional[pd.Series]] = {
            ("ne10", "raw"): ne10_raw,
            ("ne10", "mc_median"): ne10_mc,
            ("composite", "raw"): composite_raw,
            ("composite", "mc_median"): composite_mc,
        }
        for truth_set, truth in truth_by_set.items():
            if truth.empty or truth.nunique() < 2:
                continue
            for (score_kind, variant), score_series in score_by_key.items():
                if score_series is None or score_series.empty:
                    continue
                shared = [str(s) for s in score_series.index if str(s) in truth.index]
                labels = [int(truth[s]) for s in shared]
                scores = [float(score_series[s]) for s in shared]
                curve = _compute_score_roc_curve(labels, scores)
                if curve is None:
                    print(f"    [skip] {cohort} ({score_kind}/{truth_set}/{variant}): insufficient labelled samples")
                    continue
                curves[(score_kind, truth_set, variant)][cohort] = curve
                summary_rows.append({
                    "cohort": cohort,
                    "score_kind": score_kind,
                    "truth_set": truth_set,
                    "variant": variant,
                    "auc": float(curve[2]),
                    "n_samples": int(curve[3]),
                    "n_nepc": int(np.sum(labels)),
                })
                for sample, label, score in zip(shared, labels, scores):
                    input_rows.append({
                        "cohort": cohort,
                        "score_kind": score_kind,
                        "truth_set": truth_set,
                        "variant": variant,
                        "sample": sample,
                        "nepc_present": int(label),
                        "score": float(score),
                    })

    if not any(curves.values()):
        print("    [skip] NEPC ROC: no cohort produced a valid curve")
        return

    for (score_kind, truth_set, variant), pdf_path in pdf_paths.items():
        cohort_curves = curves[(score_kind, truth_set, variant)]
        if not cohort_curves:
            continue
        roc_gene_calling(
            cohort_curves,
            pdf_path,
            title=f"NEPC Detection — {kind_titles[score_kind]} ({variant_titles[variant]}, {truth_set_titles[truth_set]})",
            model_order=list(_NEPC_ROC_COHORTS),
        )
    if input_rows:
        pd.DataFrame(input_rows).to_csv(
            os.path.join(roc_dir, "nepc_detection_roc_inputs.tsv"), sep="\t", index=False
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(roc_dir, "nepc_detection_roc_summary.tsv"), sep="\t", index=False
        )
    print("    NEPC-detection ROC plots saved")


_CRPC_UW_PANEL_GROUP_ORDER: Tuple[str, ...] = ("ARPC", "NEPC", "MIXED")


def _crpc_uw_phenotype_records() -> pd.DataFrame:
    """Load CRPC-UW per-sample TFx, phenotype group, and represented subtypes.

    Samples are grouped into ARPC / NEPC / MIXED, where MIXED is any sample whose
    annotation lists more than one distinct subtype (e.g. ``ARPC,NEPC``).  The
    ``subtypes`` column preserves the ordered distinct tokens so the phenotype
    row can split mixed samples into two colored triangles.
    """
    path = _require_existing_path(
        PROJECT_ROOT / "labels" / "CRPC-UW_TFX_Subtype-Mixed.tsv",
        description="CRPC-UW mixed subtype labels",
    )
    df = pd.read_csv(path, sep="\t")
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        sample = str(row["sample"])
        tfx = pd.to_numeric(row["tfx"], errors="coerce")
        tokens = [tok.strip().upper() for tok in re.split(r"[,/]", str(row["subtype"])) if tok.strip()]
        known = [tok for tok in tokens if tok not in _NEPC_ROC_UNKNOWN_TOKENS]
        if not known or pd.isna(tfx):
            continue
        unique = list(dict.fromkeys(known))
        group = "MIXED" if len(unique) > 1 else unique[0]
        records.append({
            "sample": sample,
            "tfx": float(tfx),
            "group": group,
            "subtypes": unique,
        })
    return pd.DataFrame(records)


def _plot_crpc_uw_phenotype_panel(
    records: pd.DataFrame,
    ar_scores: pd.Series,
    ne_scores: pd.Series,
    composite_scores: pd.Series,
    name: str,
    title: str,
) -> None:
    """Four-row CRPC-UW panel: TFx, AR.10/NE.10 dual bars, z(NE)-z(AR), phenotype.

    Samples are split into ARPC / NEPC / MIXED groups (left→right) and ordered
    high→low TFx within each group.  The composite row shows the cohort-relative
    z(NE.10)-z(AR.10) score (positive = NE-like, negative = AR-like).  The
    phenotype row draws a solid square per pure-subtype sample and a diagonally
    split (two-triangle) square per mixed sample, colored by the two subtypes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    palette = load_palette()
    arpc_color = palette.get("ARPC", "#0077BB")
    nepc_color = palette.get("NEPC", "#EE7733")
    mixed_color = palette.get("MIXED", "#D3D3FF")
    group_color = {"ARPC": arpc_color, "NEPC": nepc_color, "MIXED": mixed_color}
    subtype_color = {"ARPC": arpc_color, "NEPC": nepc_color}

    # Build the contiguous x layout: group blocks separated by a fixed gap,
    # high→low TFx within each block.
    positions: List[float] = []
    group_centers: Dict[str, float] = {}
    ordered_frames: List[pd.DataFrame] = []
    cursor = 0.0
    group_gap = 1.5
    for group in _CRPC_UW_PANEL_GROUP_ORDER:
        block = records[records["group"] == group].sort_values("tfx", ascending=False)
        if block.empty:
            continue
        start = cursor
        for _ in range(len(block)):
            positions.append(cursor)
            cursor += 1.0
        group_centers[group] = (start + cursor - 1.0) / 2.0
        ordered_frames.append(block)
        cursor += group_gap

    plot_df = pd.concat(ordered_frames, ignore_index=True)
    plot_df["x"] = positions
    x = plot_df["x"].to_numpy()
    ar_vals = np.array([float(ar_scores.get(s, np.nan)) for s in plot_df["sample"]])
    ne_vals = np.array([float(ne_scores.get(s, np.nan)) for s in plot_df["sample"]])
    composite_vals = np.array([float(composite_scores.get(s, np.nan)) for s in plot_df["sample"]])

    fig, (ax_tfx, ax_score, ax_comp, ax_pheno) = plt.subplots(
        4,
        1,
        figsize=(max(7.0, 0.32 * len(plot_df) + 2.5), 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 2.6, 1.8, 0.9], "hspace": 0.12},
    )

    # Row 1 — TFx bars (0–1), colored by phenotype group.
    ax_tfx.bar(
        x,
        plot_df["tfx"].to_numpy(),
        width=0.8,
        color=[group_color[g] for g in plot_df["group"]],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_tfx.set_ylim(0.0, 1.0)
    ax_tfx.set_ylabel("Tumor fraction")
    ax_tfx.set_title(title)
    ax_tfx.spines["top"].set_visible(False)
    ax_tfx.spines["right"].set_visible(False)

    # Row 2 — AR.10 / NE.10 dual bars (raw ssGSEA NES retained; a single split
    # bar would require renormalizing both scores to sum to 1 and discard the
    # native NES magnitude, so paired bars are used instead).
    bar_half = 0.2
    ax_score.bar(x - bar_half, ar_vals, width=2 * bar_half, color=arpc_color, label="AR.10", edgecolor="black", linewidth=0.3)
    ax_score.bar(x + bar_half, ne_vals, width=2 * bar_half, color=nepc_color, label="NE.10", edgecolor="black", linewidth=0.3)
    ax_score.axhline(0.0, color="#444444", linewidth=0.6)
    ax_score.set_ylabel("ssGSEA NES")
    ax_score.spines["top"].set_visible(False)
    ax_score.spines["right"].set_visible(False)
    ax_score.legend(frameon=False)

    # Row 3 — cohort-relative composite z(NE.10) - z(AR.10): positive (NE-like) is
    # drawn in the NEPC color, negative (AR-like) in the ARPC color, so "not
    # AR-like" reinforces "NE-like" on a single diverging axis.
    composite_colors = [
        nepc_color if (np.isfinite(v) and v >= 0.0) else arpc_color for v in composite_vals
    ]
    ax_comp.bar(x, composite_vals, width=0.8, color=composite_colors, edgecolor="black", linewidth=0.3)
    ax_comp.axhline(0.0, color="#444444", linewidth=0.6)
    ax_comp.set_ylabel("z(NE.10) − z(AR.10)")
    ax_comp.spines["top"].set_visible(False)
    ax_comp.spines["right"].set_visible(False)

    # Row 4 — phenotype squares: solid for pure subtypes, diagonally split for mixed.
    for xi, subtypes in zip(x, plot_df["subtypes"]):
        left, right, bottom, top = xi - 0.4, xi + 0.4, 0.1, 0.9
        if len(subtypes) <= 1:
            color = subtype_color.get(subtypes[0], group_color["MIXED"]) if subtypes else group_color["MIXED"]
            ax_pheno.add_patch(Rectangle((left, bottom), right - left, top - bottom, facecolor=color, edgecolor="black", linewidth=0.4))
        else:
            lower_color = subtype_color.get(subtypes[0], "#999999")
            upper_color = subtype_color.get(subtypes[1], "#999999")
            ax_pheno.add_patch(Polygon([(left, bottom), (right, bottom), (left, top)], closed=True, facecolor=lower_color, edgecolor="black", linewidth=0.4))
            ax_pheno.add_patch(Polygon([(right, bottom), (right, top), (left, top)], closed=True, facecolor=upper_color, edgecolor="black", linewidth=0.4))
    ax_pheno.set_ylim(0.0, 1.0)
    ax_pheno.set_yticks([])
    ax_pheno.set_ylabel("Phenotype")
    for spine in ax_pheno.spines.values():
        spine.set_visible(False)

    ax_pheno.set_xticks(x)
    ax_pheno.set_xticklabels(plot_df["sample"], rotation=90, fontsize=6)
    ax_pheno.set_xlim(x.min() - 0.8, x.max() + 0.8)

    # Group labels centered above each block.
    for group, center in group_centers.items():
        ax_tfx.text(center, 1.02, group, ha="center", va="bottom", fontsize=11, fontweight="bold", color=group_color[group], transform=ax_tfx.get_xaxis_transform())

    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _run_crpc_uw_phenotype_panel(out_dir: str, args: argparse.Namespace) -> None:
    """CRPC-UW TFx / AR.10 / NE.10 / phenotype panel (raw and MC-median variants)."""
    roc_dir = ensure_dir(os.path.join(out_dir, "NEPC_Detection_ROC"))
    raw_pdf = os.path.join(roc_dir, "crpc_uw_phenotype_panel_raw")
    mc_pdf = os.path.join(roc_dir, "crpc_uw_phenotype_panel_mc_median")
    if (
        not args.overwrite
        and os.path.exists(f"{raw_pdf}.pdf")
        and os.path.exists(f"{mc_pdf}.pdf")
    ):
        return

    print("\n  ── CRPC-UW phenotype panel (TFx / AR.10 / NE.10) ──")
    records = _crpc_uw_phenotype_records()
    records = records[records["group"].isin(_CRPC_UW_PANEL_GROUP_ORDER)].reset_index(drop=True)
    if records.empty:
        print("    [skip] CRPC-UW phenotype panel: no labelled samples")
        return

    shortlist_sets = _load_shortlist_sets()
    if "ARG.10" not in shortlist_sets or "NE.10" not in shortlist_sets:
        raise ValueError("Shortlist GMT is missing the required 'ARG.10' / 'NE.10' gene sets")
    score_sets = {"AR.10": shortlist_sets["ARG.10"], "NE.10": shortlist_sets["NE.10"]}

    bundle = _nepc_cohort_mdn_bundle("CRPC-UW", args.minimum_tfx)
    pred_wide = bundle["pred_wide"]
    if pred_wide is None or pred_wide.empty:
        print("    [skip] CRPC-UW phenotype panel: no predictions available")
        return

    available = {str(s) for s in pred_wide.columns}
    records = records[records["sample"].isin(available)].reset_index(drop=True)
    if records.empty:
        print("    [skip] CRPC-UW phenotype panel: no labelled samples have predictions")
        return

    def _scores(matrix: pd.DataFrame, term: str) -> pd.Series:
        if matrix is None or matrix.empty or term not in matrix.index:
            return pd.Series(dtype="float64")
        series = pd.to_numeric(matrix.loc[term], errors="coerce")
        series.index = [str(s) for s in series.index]
        return series

    raw_matrix = _compute_ssgsea_score_matrix(pred_wide, score_sets, min_size=3)
    mc_draws = mdn_monte_carlo_sample(
        bundle["p0_wide"],
        bundle["pi_gamma_wide"],
        bundle["pi_normal_wide"],
        bundle["gamma_beta_wide"],
        bundle["normal_mean_wide"],
        bundle["normal_stdev_wide"],
        M=args.montecarlo_draws,
        seed=_NEPC_ROC_MC_SEED,
    )
    # MC point estimates: median AR.10 / NE.10 ssGSEA for the dual bars, plus the
    # per-draw z(NE)-z(AR) median for the composite row (one ssGSEA pass per draw).
    ne_mc, ar_mc, composite_mc = _ne_ar_mc_medians(
        mc_draws, {"NE.10": score_sets["NE.10"]}, {"AR.10": score_sets["AR.10"]}
    )

    ar_raw = _scores(raw_matrix, "AR.10")
    ne_raw = _scores(raw_matrix, "NE.10")
    composite_raw = _ne_minus_ar_composite(ne_raw, ar_raw)

    def _as_series(series: Optional[pd.Series]) -> pd.Series:
        return series if series is not None else pd.Series(dtype="float64")

    variants = (
        ("raw", ar_raw, ne_raw, composite_raw, raw_pdf,
         "CRPC-UW Phenotype Panel — AR.10 / NE.10 (raw scores)"),
        ("mc_median", _as_series(ar_mc), _as_series(ne_mc), _as_series(composite_mc), mc_pdf,
         "CRPC-UW Phenotype Panel — AR.10 / NE.10 (MC median)"),
    )
    export_rows: List[Dict[str, Any]] = []
    plotted = False
    for variant, ar_scores, ne_scores, composite_scores, pdf_path, title in variants:
        if ar_scores.empty and ne_scores.empty:
            print(f"    [skip] CRPC-UW phenotype panel ({variant}): AR.10 / NE.10 scores unavailable")
            continue
        _plot_crpc_uw_phenotype_panel(records, ar_scores, ne_scores, composite_scores, pdf_path, title)
        plotted = True
        for _, row in records.iterrows():
            sample = str(row["sample"])
            export_rows.append({
                "variant": variant,
                "sample": sample,
                "group": row["group"],
                "subtypes": ",".join(row["subtypes"]),
                "tfx": float(row["tfx"]),
                "ar10_score": float(ar_scores.get(sample, np.nan)),
                "ne10_score": float(ne_scores.get(sample, np.nan)),
                "ne_minus_ar_z": float(composite_scores.get(sample, np.nan)),
            })

    if export_rows:
        pd.DataFrame(export_rows).to_csv(
            os.path.join(roc_dir, "crpc_uw_phenotype_panel_inputs.tsv"), sep="\t", index=False
        )
    if plotted:
        print("    CRPC-UW phenotype panel plots saved")


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
    activity_threshold_summary = None
    if args.apply_calibration:
        calibration_cv_bundle = _load_main_tds_cv_full(args.minimum_tfx)
        if calibration_cv_bundle is None:
            raise ValueError("Discovery_Patient calibration requires the canonical patient CV OOF bundle")
        calibration_state = _get_patient_distribution_calibration_state(
            args.minimum_tfx,
            cv_full=calibration_cv_bundle["cv_full"],
            cohort_map=calibration_cv_bundle["cohort_map"],
        )
        # Calibrated UB/LB AOR band used by --dim_ambiguous gene dropping; without
        # calibration the dropper falls back to the fixed |AOR| <= 5 band.
        if args.dim_ambiguous:
            activity_threshold_summary = _compute_patient_cv_activity_thresholds(
                calibration_cv_bundle["cv_full"],
                calibration_cv_bundle["cohort_map"],
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
            activity_threshold_summary=activity_threshold_summary,
        )

    _run_nepc_detection_roc(out_dir, args)
    _run_crpc_uw_phenotype_panel(out_dir, args)

    print("\n  Done")


def _load_main_tds_cv_full(minimum_tfx: float = 0.0) -> Optional[Dict[str, Any]]:
    """Load and prefix-normalize the canonical TDS CV outputs."""
    for _, rel_path in sorted(TDS_CV_FOLD_TUMOR_PREDICTION_PATHS.items()):
        _require_existing_path(
            PROJECT_ROOT / rel_path,
            description="canonical TDS CV prediction file",
        )

    fold_splits = _load_tds_fold_splits()
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
            for fold, rel_path in sorted(TDS_CV_FOLD_TUMOR_PREDICTION_PATHS.items())
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
    cv_bundle = _load_main_tds_cv_full(args.minimum_tfx)
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
    """Validation_PDX analysis using TDS main-model CV + true holdout PDX samples."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Validation_PDX")
    print("=" * 70)

    if _should_skip_section("Validation_PDX", args):
        return

    out_dir = _section_results_dir("Validation_PDX")
    _write_run_manifest("Validation_PDX", out_dir, args)
    n_workers = args.n_workers

    canonical_truth = _build_tds_truth(args.minimum_tfx)
    if canonical_truth.empty:
        print("  [skip] No canonical TDS truth available")
        return

    cv_bundle = _load_main_tds_cv_full(args.minimum_tfx)
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

        cohort_cfg = _TDS_CV_COHORT_CONFIG[cohort_name]
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
                    _, _, score_ci_lo, score_ci_hi = _aggregate_ssgsea_score_matrices(
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
    """Validation_Patient analysis using TDS main-model holdout and OOF CV data."""
    print("\n" + "=" * 70)
    print("  ANALYSIS GROUP: Validation_Patient")
    print("=" * 70)

    if _should_skip_section("Validation_Patient", args):
        return

    out_dir = _section_results_dir("Validation_Patient")
    _write_run_manifest("Validation_Patient", out_dir, args)
    cohort_data: List[Dict[str, Any]] = []

    cv_bundle = _load_main_tds_cv_full(args.minimum_tfx)

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


def _maybe_forest_plots(
    overwrite: bool,
    hr_table: pd.DataFrame,
    name: str,
    *,
    hr_table_mv: Optional[pd.DataFrame] = None,
    title: str,
) -> None:
    """Emit two forest-plot variants for a HR table.

    Always writes the combined univariate + multivariate/TFx-adjusted plot
    at ``{name}``. When a non-empty multivariate table is supplied, also
    writes a multivariate/TFx-adjusted-only variant at ``{name}_mvadj`` so
    readers can view the adjusted effects in isolation.
    """
    _maybe(
        forest_plot, overwrite,
        hr_table,
        name,
        hr_table_mv=hr_table_mv,
        title=title,
    )
    if hr_table_mv is not None and not hr_table_mv.empty:
        _maybe(
            forest_plot, overwrite,
            hr_table,
            f"{name}_mvadj",
            hr_table_mv=hr_table_mv,
            title=f"{title} (TFx-adjusted only)",
            mv_only=True,
        )


def _save_summary_tsv(summaries: List[Dict[str, Any]], path: str) -> None:
    """Save cohort summaries to a TSV, excluding internal DataFrames."""
    rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if not k.startswith("_")}
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print("  Summary table saved")


def _save_prediction_mean_wide_tsv(pred_wide: Optional[pd.DataFrame], path: str) -> None:
    """Save cohort predictions in wide form (genes x samples) using point means."""
    if pred_wide is None or pred_wide.empty:
        return
    pred_wide.astype(float).to_csv(path, sep="\t", index=True, index_label="gene")
    print(f"  Saved compiled point predictions: {path}")


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

        _maybe(
            benchmark_cohort_retention_curve,
            overwrite,
            gene_rho_view,
            os.path.join(view_out, "benchmark_cohort_gene_rho_retention_curve"),
            cohort_order=cohort_order,
            model_order=model_order,
            title=f"Benchmarking Patient ({view_name}) — Mean Spearman ρ by Retained Genes",
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
        shared_lower_threshold = None
        shared_upper_threshold = None
        if activity_threshold_summary is not None:
            shared_lower_threshold = float(activity_threshold_summary["lower_threshold"])
            shared_upper_threshold = float(activity_threshold_summary["upper_threshold"])

        if emit_confusion_artifacts:
            confusion_variants = [
                {
                    "suffix": "",
                    "title_suffix": "",
                    "rule": "aor_selected_thresholds",
                    "aor_lower": shared_lower_threshold,
                    "aor_upper": shared_upper_threshold,
                    "use_ci": False,
                },
                {
                    "suffix": "_AORAbs5",
                    "title_suffix": " — abs(AOR) < 5",
                    "rule": "aor_abs_lt_5",
                    "aor_lower": -5.0,
                    "aor_upper": 5.0,
                    "use_ci": False,
                },
                {
                    "suffix": "_CI90lt2",
                    "title_suffix": " — 90% CI threshold 2",
                    "rule": "ci90_threshold_2",
                    "aor_lower": None,
                    "aor_upper": None,
                    "use_ci": True,
                },
            ]

            for variant in confusion_variants:
                confusion_results: Dict[str, Dict[str, Any]] = {}
                confusion_order: List[str] = []
                variant_lower = variant["aor_lower"]
                variant_upper = variant["aor_upper"]
                use_ci_variant = bool(variant["use_ci"])

                for model_name in model_order:
                    if model_name not in model_preds or model_name == "Naive":
                        continue

                    pw = model_preds[model_name]
                    common_samples = [s for s in pw.columns if s in truth_wide.columns]
                    cg = [g for g in common_genes if g in pw.index]
                    if not common_samples or len(cg) < 5:
                        continue

                    t_base = truth_wide.loc[cg, common_samples].to_numpy(dtype=np.float64).ravel()
                    p_base = pw.loc[cg, common_samples].to_numpy(dtype=np.float64).ravel()

                    activity_odds_ratio_flat: Optional[np.ndarray] = None
                    t_aor = t_base
                    p_aor = p_base

                    ci_lo_flat: Optional[np.ndarray] = None
                    ci_hi_flat: Optional[np.ndarray] = None
                    t_ci = t_base
                    p_ci = p_base

                    mdn_tables = proteus_mdn.get(model_name, {})
                    if mdn_tables:
                        aor_required_keys = ("p0_wide", "pi_gamma_wide", "pi_normal_wide")
                        if all(mdn_tables.get(key) is not None for key in aor_required_keys):
                            cg_mdn_aor = [
                                g for g in cg
                                if all(g in mdn_tables[key].index for key in aor_required_keys)
                            ]
                            cs_mdn_aor = [
                                s for s in common_samples
                                if all(s in mdn_tables[key].columns for key in aor_required_keys)
                            ]
                            if len(cg_mdn_aor) >= 5 and len(cs_mdn_aor) >= 3:
                                activity_odds_ratio_wide = mdn_activity_odds_ratio_wide(
                                    mdn_tables["p0_wide"].loc[cg_mdn_aor, cs_mdn_aor],
                                    mdn_tables["pi_gamma_wide"].loc[cg_mdn_aor, cs_mdn_aor],
                                    mdn_tables["pi_normal_wide"].loc[cg_mdn_aor, cs_mdn_aor],
                                )
                                activity_odds_ratio_flat = activity_odds_ratio_wide.to_numpy(dtype=np.float64).ravel()
                                t_aor = truth_wide.loc[cg_mdn_aor, cs_mdn_aor].to_numpy(dtype=np.float64).ravel()
                                p_aor = pw.loc[cg_mdn_aor, cs_mdn_aor].to_numpy(dtype=np.float64).ravel()

                        ci_required_keys = ("ci_lo_wide", "ci_hi_wide")
                        if all(mdn_tables.get(key) is not None for key in ci_required_keys):
                            cg_mdn_ci = [
                                g for g in cg
                                if all(g in mdn_tables[key].index for key in ci_required_keys)
                            ]
                            cs_mdn_ci = [
                                s for s in common_samples
                                if all(s in mdn_tables[key].columns for key in ci_required_keys)
                            ]
                            if len(cg_mdn_ci) >= 5 and len(cs_mdn_ci) >= 3:
                                ci_lo_flat = mdn_tables["ci_lo_wide"].loc[cg_mdn_ci, cs_mdn_ci].to_numpy(dtype=np.float64).ravel()
                                ci_hi_flat = mdn_tables["ci_hi_wide"].loc[cg_mdn_ci, cs_mdn_ci].to_numpy(dtype=np.float64).ravel()
                                t_ci = truth_wide.loc[cg_mdn_ci, cs_mdn_ci].to_numpy(dtype=np.float64).ravel()
                                p_ci = pw.loc[cg_mdn_ci, cs_mdn_ci].to_numpy(dtype=np.float64).ravel()

                    ua_name = f"{model_name} (uncertainty-informed)"

                    if use_ci_variant and ci_lo_flat is not None and ci_hi_flat is not None:
                        binary_confusion = compute_onoff_confusion(
                            t_ci,
                            p_ci,
                            threshold=_ONOFF_THRESHOLD,
                        )
                        ua_confusion = compute_onoff_confusion(
                            t_ci,
                            p_ci,
                            ci_lo_flat=ci_lo_flat,
                            ci_hi_flat=ci_hi_flat,
                            threshold=_ONOFF_THRESHOLD,
                        )
                        if binary_confusion.get("n_total", 0) > 0:
                            confusion_results[model_name] = binary_confusion
                            confusion_order.append(model_name)
                        if ua_confusion.get("n_total", 0) > 0:
                            confusion_results[ua_name] = ua_confusion
                            confusion_order.append(ua_name)
                    elif (
                        (not use_ci_variant)
                        and activity_odds_ratio_flat is not None
                        and variant_lower is not None
                        and variant_upper is not None
                        and np.isfinite(float(variant_lower))
                        and np.isfinite(float(variant_upper))
                    ):
                        binary_confusion = compute_onoff_confusion(
                            t_aor,
                            p_aor,
                            threshold=_ONOFF_THRESHOLD,
                        )
                        ua_confusion = compute_onoff_confusion(
                            t_aor,
                            p_aor,
                            activity_odds_ratio_flat=activity_odds_ratio_flat,
                            lower_threshold=float(variant_lower),
                            upper_threshold=float(variant_upper),
                            threshold=_ONOFF_THRESHOLD,
                        )
                        if binary_confusion.get("n_total", 0) > 0:
                            confusion_results[model_name] = binary_confusion
                            confusion_order.append(model_name)
                        if ua_confusion.get("n_total", 0) > 0:
                            confusion_results[ua_name] = ua_confusion
                            confusion_order.append(ua_name)
                    else:
                        d = compute_onoff_confusion(
                            t_base,
                            p_base,
                            threshold=_ONOFF_THRESHOLD,
                        )
                        if d.get("n_total", 0) > 0:
                            confusion_results[model_name] = d
                            confusion_order.append(model_name)

                if not confusion_results:
                    continue

                desired_confusion_order: List[str] = []
                for model_name in model_order:
                    if model_name == "Naive":
                        continue
                    ua_name = f"{model_name} (uncertainty-informed)"
                    if model_name in confusion_results or ua_name in confusion_results:
                        if ua_name in confusion_results:
                            desired_confusion_order.append(ua_name)
                        if model_name in confusion_results:
                            desired_confusion_order.append(model_name)
                        continue
                confusion_order = desired_confusion_order + [
                    model_name for model_name in confusion_order if model_name not in desired_confusion_order
                ]

                conf_rows = []
                for m in confusion_order:
                    d = confusion_results[m]
                    conf_rows.append({
                        "model": m,
                        "uncertainty_rule": variant["rule"],
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
                        "lower_activity_odds_ratio_threshold": float(variant_lower) if variant_lower is not None else float("nan"),
                        "upper_activity_odds_ratio_threshold": float(variant_upper) if variant_upper is not None else float("nan"),
                        "ci_decision_threshold": _ONOFF_THRESHOLD if use_ci_variant else float("nan"),
                    })
                conf_df = pd.DataFrame(conf_rows)
                conf_df.to_csv(
                    os.path.join(out_dir, f"OnOffConfusion{variant['suffix']}_{gs_display}_{cohort}.tsv"),
                    sep="\t",
                    index=False,
                )

                _maybe(
                    onoff_confusion_summary_plot,
                    overwrite,
                    confusion_results,
                    os.path.join(out_dir, f"OnOffConfusion{variant['suffix']}_{gs_display}_{cohort}"),
                    title=f"On/Off Confusion{variant['title_suffix']} — {gs_display} ({cohort})",
                    model_order=confusion_order,
                )
                print(
                    f"    OnOffConfusion{variant['suffix']} {gs_display}: "
                    f"{len(confusion_results)} models"
                )


def _benchmark_patient_confusion_variants() -> List[Dict[str, Any]]:
    """Return the pooled Benchmarking_Patient confusion variants."""
    return [
        {
            "suffix": "_AORAbs5",
            "title_suffix": "abs(AOR) < 5",
            "rule": "aor_abs_lt_5",
            "mode": "aor",
            "aor_lower": -5.0,
            "aor_upper": 5.0,
            "ci90_max": None,
        },
        {
            "suffix": "_CI90lt2",
            "title_suffix": "CI90 < 2",
            "rule": "ci90_width_lt_2",
            "mode": "ci_width",
            "aor_lower": None,
            "aor_upper": None,
            "ci90_max": 2.0,
        },
        {
            "suffix": "_CI90lt3",
            "title_suffix": "CI90 < 3",
            "rule": "ci90_width_lt_3",
            "mode": "ci_width",
            "aor_lower": None,
            "aor_upper": None,
            "ci90_max": 3.0,
        },
        {
            "suffix": "_CI90lt4",
            "title_suffix": "CI90 < 4",
            "rule": "ci90_width_lt_4",
            "mode": "ci_width",
            "aor_lower": None,
            "aor_upper": None,
            "ci90_max": 4.0,
        },
    ]


def _benchmark_patient_confusion_rows_for_variant(
    truth_wide: pd.DataFrame,
    model_preds: Dict[str, pd.DataFrame],
    proteus_mdn: Dict[str, Dict[str, Optional[pd.DataFrame]]],
    model_order: Sequence[str],
    common_genes: Sequence[str],
    cohort: str,
    variant: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build one pooled-on-plot confusion row set for a cohort and variant."""
    rows: List[Dict[str, Any]] = []

    def _compute_ci_width_confusion(
        truth_flat: np.ndarray,
        pred_flat: np.ndarray,
        ci_width_flat: np.ndarray,
        max_width: float,
    ) -> Dict[str, Any]:
        valid = np.isfinite(truth_flat) & np.isfinite(pred_flat) & np.isfinite(ci_width_flat)
        truth = truth_flat[valid]
        pred = pred_flat[valid]
        width = ci_width_flat[valid]
        n_total = int(len(truth))
        if n_total == 0:
            return {"n_total": 0}

        truth_on = truth >= _ONOFF_THRESHOLD
        n_truth_on = int(truth_on.sum())
        n_truth_off = int((~truth_on).sum())

        confident = width < float(max_width)
        n_indet = int((~confident).sum())
        if not confident.any():
            return {
                "n_total": n_total,
                "n_truth_on": n_truth_on,
                "n_truth_off": n_truth_off,
                "n_on": 0,
                "n_off": 0,
                "n_indet": n_indet,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "accuracy": float("nan"),
                "ppv": float("nan"),
                "npv": float("nan"),
                "fpr": float("nan"),
                "fnr": float("nan"),
                "sensitivity": float("nan"),
                "coverage": 0.0,
            }

        pred_on_conf = pred[confident] >= _ONOFF_THRESHOLD
        truth_on_conf = truth_on[confident]
        tp = int((pred_on_conf & truth_on_conf).sum())
        fp = int((pred_on_conf & (~truth_on_conf)).sum())
        fn = int(((~pred_on_conf) & truth_on_conf).sum())
        tn = int(((~pred_on_conf) & (~truth_on_conf)).sum())
        n_on = int(pred_on_conf.sum())
        n_off = int((~pred_on_conf).sum())
        n_conf = int(confident.sum())
        return {
            "n_total": n_total,
            "n_truth_on": n_truth_on,
            "n_truth_off": n_truth_off,
            "n_on": n_on,
            "n_off": n_off,
            "n_indet": n_indet,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": (tp + tn) / max(n_conf, 1),
            "ppv": tp / max(tp + fp, 1),
            "npv": tn / max(tn + fn, 1),
            "fpr": fp / max(fp + tn, 1),
            "fnr": fn / max(fn + tp, 1),
            "sensitivity": tp / max(tp + fn, 1),
            "coverage": n_conf / max(n_total, 1),
        }

    for model_name in model_order:
        if model_name == "Naive" or model_name not in model_preds:
            continue

        pred_wide = model_preds[model_name]
        common_samples = [sample for sample in pred_wide.columns if sample in truth_wide.columns]
        model_genes = [gene_name for gene_name in common_genes if gene_name in pred_wide.index]
        if len(common_samples) < 3 or len(model_genes) < 5:
            continue

        truth_base = truth_wide.loc[model_genes, common_samples].to_numpy(dtype=np.float64).ravel()
        pred_base = pred_wide.loc[model_genes, common_samples].to_numpy(dtype=np.float64).ravel()

        mdn_tables = proteus_mdn.get(model_name, {})

        def _append_row(row_model: str, confusion: Dict[str, Any]) -> None:
            if confusion.get("n_total", 0) <= 0:
                return
            fpr_value = confusion.get("fpr", float("nan"))
            rows.append({
                "model": row_model,
                "cohort": cohort,
                "uncertainty_rule": str(variant["rule"]),
                "n_total": int(confusion.get("n_total", 0)),
                "n_truth_on": int(confusion.get("n_truth_on", 0)),
                "n_truth_off": int(confusion.get("n_truth_off", 0)),
                "n_on": int(confusion.get("n_on", 0)),
                "n_off": int(confusion.get("n_off", 0)),
                "n_indet": int(confusion.get("n_indet", 0)),
                "accuracy": float(confusion.get("accuracy", float("nan"))),
                "ppv": float(confusion.get("ppv", float("nan"))),
                "npv": float(confusion.get("npv", float("nan"))),
                "sensitivity": float(confusion.get("sensitivity", float("nan"))),
                "fpr": float(fpr_value),
                "specificity": 1.0 - float(fpr_value) if np.isfinite(fpr_value) else float("nan"),
                "coverage": float(confusion.get("coverage", float("nan"))),
                "lower_activity_odds_ratio_threshold": (
                    float(variant["aor_lower"]) if variant.get("aor_lower") is not None else float("nan")
                ),
                "upper_activity_odds_ratio_threshold": (
                    float(variant["aor_upper"]) if variant.get("aor_upper") is not None else float("nan")
                ),
                "ci90_max_width": float(variant["ci90_max"]) if variant.get("ci90_max") is not None else float("nan"),
            })

        if variant["mode"] == "aor":
            required_keys = ("p0_wide", "pi_gamma_wide", "pi_normal_wide")
            if all(mdn_tables.get(key) is not None for key in required_keys):
                mdn_genes = [
                    gene_name
                    for gene_name in model_genes
                    if all(gene_name in mdn_tables[key].index for key in required_keys)
                ]
                mdn_samples = [
                    sample_name
                    for sample_name in common_samples
                    if all(sample_name in mdn_tables[key].columns for key in required_keys)
                ]
                if len(mdn_genes) >= 5 and len(mdn_samples) >= 3:
                    truth_aor = truth_wide.loc[mdn_genes, mdn_samples].to_numpy(dtype=np.float64).ravel()
                    pred_aor = pred_wide.loc[mdn_genes, mdn_samples].to_numpy(dtype=np.float64).ravel()
                    activity_odds_ratio_flat = mdn_activity_odds_ratio_wide(
                        mdn_tables["p0_wide"].loc[mdn_genes, mdn_samples],
                        mdn_tables["pi_gamma_wide"].loc[mdn_genes, mdn_samples],
                        mdn_tables["pi_normal_wide"].loc[mdn_genes, mdn_samples],
                    ).to_numpy(dtype=np.float64).ravel()

                    binary_confusion = compute_onoff_confusion(
                        truth_aor,
                        pred_aor,
                        threshold=_ONOFF_THRESHOLD,
                    )
                    ua_confusion = compute_onoff_confusion(
                        truth_aor,
                        pred_aor,
                        activity_odds_ratio_flat=activity_odds_ratio_flat,
                        lower_threshold=float(variant["aor_lower"]),
                        upper_threshold=float(variant["aor_upper"]),
                        threshold=_ONOFF_THRESHOLD,
                    )
                    _append_row(model_name, binary_confusion)
                    _append_row(f"{model_name} (uncertainty-informed)", ua_confusion)
                    continue

        if variant["mode"] == "ci_width":
            ci_required_keys = ("ci_lo_wide", "ci_hi_wide")
            if all(mdn_tables.get(key) is not None for key in ci_required_keys):
                mdn_genes = [
                    gene_name
                    for gene_name in model_genes
                    if all(gene_name in mdn_tables[key].index for key in ci_required_keys)
                ]
                mdn_samples = [
                    sample_name
                    for sample_name in common_samples
                    if all(sample_name in mdn_tables[key].columns for key in ci_required_keys)
                ]
                if len(mdn_genes) >= 5 and len(mdn_samples) >= 3:
                    truth_ci = truth_wide.loc[mdn_genes, mdn_samples].to_numpy(dtype=np.float64).ravel()
                    pred_ci = pred_wide.loc[mdn_genes, mdn_samples].to_numpy(dtype=np.float64).ravel()
                    ci_lo_flat = mdn_tables["ci_lo_wide"].loc[mdn_genes, mdn_samples].to_numpy(dtype=np.float64).ravel()
                    ci_hi_flat = mdn_tables["ci_hi_wide"].loc[mdn_genes, mdn_samples].to_numpy(dtype=np.float64).ravel()
                    ci_width_flat = ci_hi_flat - ci_lo_flat
                    ci_threshold = float(variant["ci90_max"])

                    binary_confusion = compute_onoff_confusion(
                        truth_ci,
                        pred_ci,
                        threshold=_ONOFF_THRESHOLD,
                    )
                    ua_confusion = _compute_ci_width_confusion(
                        truth_ci,
                        pred_ci,
                        ci_width_flat,
                        ci_threshold,
                    )
                    _append_row(model_name, binary_confusion)
                    _append_row(f"{model_name} (uncertainty-informed)", ua_confusion)
                    continue

        deterministic_confusion = compute_onoff_confusion(
            truth_base,
            pred_base,
            threshold=_ONOFF_THRESHOLD,
        )
        _append_row(model_name, deterministic_confusion)

    return rows


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

    cv_bundle = _load_main_tds_cv_full(args.minimum_tfx)
    activity_threshold_summary: Optional[Dict[str, Any]] = None
    if cv_bundle is not None:
        cv_full = cv_bundle["cv_full"]
        activity_threshold_summary = _compute_patient_cv_activity_thresholds(
            cv_full,
            cv_bundle["cohort_map"],
        )
    subset_view_models = tuple(benchmark_model_views().get("SUBSET", ()))
    pooled_threshold_pairs: Dict[str, Dict[str, List[np.ndarray]]] = {}
    pooled_confusion_rows: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    confusion_variants = _benchmark_patient_confusion_variants()

    # Pooled uncertainty-vs-residual arrays across both patient cohorts.
    pooled_ci_width: List[np.ndarray] = []
    pooled_ci_abs_residual: List[np.ndarray] = []
    pooled_abs_aor: List[np.ndarray] = []
    pooled_aor_abs_residual: List[np.ndarray] = []
    # Gene labels for subset filtering (tracked independently per x-axis type)
    _pooled_gene_labels_ci: List[np.ndarray] = []
    _pooled_gene_labels_aor: List[np.ndarray] = []
    # Per-gene truth variance (gene → list of per-sample values across cohorts)
    _gene_truth_values: Dict[str, List[float]] = {}

    # Accumulate metrics, per-gene rho, and per-sample rho across cohorts
    all_metrics_rows: List[Dict[str, Any]] = []
    all_gene_rho_rows: List[Dict[str, Any]] = []
    all_sample_rho_rows: List[Dict[str, Any]] = []

    # Per-cohort Proteus bundles for the CIDriverAnalysis sub-analysis
    # (matched, restricted patient sets pooled across cohorts after the loop).
    ci_driver_bundles: Dict[str, Dict[str, Any]] = {}

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
                "ci_lo_wide",
                "ci_hi_wide",
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

        # Restrict truth and every model to the single sample set common to all
        # models, so each model is scored on identical patients. Without this,
        # each model intersected with truth independently (yielding divergent n)
        # and the Naive baseline inherited the full TFx-filtered truth. The
        # restriction is applied to truth, predictions, and the Proteus MDN
        # tables here so all downstream metrics/ROC/CI/AOR use the same samples.
        common_samples = [
            sample
            for sample in truth_wide.columns
            if all(sample in pw.columns for pw in model_preds.values())
        ]
        if not common_samples:
            print(f"  [skip] No samples common to all models for {cohort}")
            continue
        dropped_samples = [s for s in truth_wide.columns if s not in set(common_samples)]
        if dropped_samples:
            print(
                f"  Restricting to {len(common_samples)} samples common to all "
                f"{len(model_preds)} models (dropped {len(dropped_samples)}: "
                f"{', '.join(sorted(dropped_samples))})"
            )
        truth_wide = truth_wide[common_samples]
        for _model_name in list(model_preds.keys()):
            model_preds[_model_name] = model_preds[_model_name][common_samples]
        for _model_name, _mdn_tables in proteus_mdn.items():
            proteus_mdn[_model_name] = {
                _table_key: (
                    _table[common_samples] if _table is not None else None
                )
                for _table_key, _table in _mdn_tables.items()
            }

        # Capture the matched Proteus bundle for CIDriverAnalysis. Only stored
        # when the full MDN tables required to derive AOR/CI are present.
        _ci_mdn = proteus_mdn.get("Proteus", {})
        _ci_pred = model_preds.get("Proteus")
        if (
            _ci_pred is not None
            and subtypes_df is not None
            and all(
                _ci_mdn.get(_k) is not None
                for _k in ("ci_lo_wide", "ci_hi_wide", "p0_wide", "pi_gamma_wide", "pi_normal_wide")
            )
        ):
            ci_driver_bundles[cohort] = {
                "truth_wide": truth_wide,
                "pred_wide": _ci_pred,
                "ci_lo": _ci_mdn["ci_lo_wide"],
                "ci_hi": _ci_mdn["ci_hi_wide"],
                "p0": _ci_mdn["p0_wide"],
                "pi_gamma": _ci_mdn["pi_gamma_wide"],
                "pi_normal": _ci_mdn["pi_normal_wide"],
                "tfx": pd.to_numeric(subtypes_df["tfx"], errors="coerce").reindex(common_samples),
            }

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
                emit_confusion_artifacts=False,
            )

        # Accumulate CI-width / |residual| pairs from Proteus for this cohort.
        # Uses the canonical "Proteus" model (full CI tables required).
        _proteus_mdn_tables = proteus_mdn.get("Proteus", {})
        _proteus_pred = model_preds.get("Proteus")
        if (
            _proteus_pred is not None
            and _proteus_mdn_tables.get("ci_lo_wide") is not None
            and _proteus_mdn_tables.get("ci_hi_wide") is not None
        ):
            _ci_lo = _proteus_mdn_tables["ci_lo_wide"]
            _ci_hi = _proteus_mdn_tables["ci_hi_wide"]
            _common_genes = list(
                truth_wide.index
                .intersection(_proteus_pred.index)
                .intersection(_ci_lo.index)
                .intersection(_ci_hi.index)
            )
            _common_samples = [
                sample
                for sample in truth_wide.columns
                if (
                    sample in _proteus_pred.columns
                    and sample in _ci_lo.columns
                    and sample in _ci_hi.columns
                )
            ]
            if len(_common_genes) >= 10 and len(_common_samples) >= 2:
                _t = truth_wide.loc[_common_genes, _common_samples].to_numpy(dtype=np.float64)
                _p = _proteus_pred.loc[_common_genes, _common_samples].to_numpy(dtype=np.float64)
                _lo = _ci_lo.loc[_common_genes, _common_samples].to_numpy(dtype=np.float64)
                _hi = _ci_hi.loc[_common_genes, _common_samples].to_numpy(dtype=np.float64)
                _valid = (
                    np.isfinite(_t) & np.isfinite(_p) & np.isfinite(_lo) & np.isfinite(_hi)
                )
                # gene label broadcast: shape (n_genes, n_samples)
                _gene_labels_2d = np.array(
                    [[gene_name] * len(_common_samples) for gene_name in _common_genes],
                    dtype=object,
                )
                pooled_ci_width.append((_hi - _lo)[_valid])
                pooled_ci_abs_residual.append(np.abs(_t - _p)[_valid])
                _pooled_gene_labels_ci.append(_gene_labels_2d[_valid])
                # Accumulate per-gene truth values for variance calculation
                for gi, gene_name in enumerate(_common_genes):
                    gene_row = _t[gi, :]
                    finite_vals = gene_row[np.isfinite(gene_row)].tolist()
                    _gene_truth_values.setdefault(gene_name, []).extend(finite_vals)

        # Accumulate |AOR| / |residual| pairs from Proteus for this cohort.
        if (
            _proteus_pred is not None
            and _proteus_mdn_tables.get("p0_wide") is not None
            and _proteus_mdn_tables.get("pi_gamma_wide") is not None
            and _proteus_mdn_tables.get("pi_normal_wide") is not None
        ):
            _p0 = _proteus_mdn_tables["p0_wide"]
            _pi_gamma = _proteus_mdn_tables["pi_gamma_wide"]
            _pi_normal = _proteus_mdn_tables["pi_normal_wide"]
            _common_genes_aor = list(
                truth_wide.index
                .intersection(_proteus_pred.index)
                .intersection(_p0.index)
                .intersection(_pi_gamma.index)
                .intersection(_pi_normal.index)
            )
            _common_samples_aor = [
                sample
                for sample in truth_wide.columns
                if (
                    sample in _proteus_pred.columns
                    and sample in _p0.columns
                    and sample in _pi_gamma.columns
                    and sample in _pi_normal.columns
                )
            ]
            if len(_common_genes_aor) >= 10 and len(_common_samples_aor) >= 2:
                _t_aor = truth_wide.loc[_common_genes_aor, _common_samples_aor].to_numpy(dtype=np.float64)
                _p_aor = _proteus_pred.loc[_common_genes_aor, _common_samples_aor].to_numpy(dtype=np.float64)
                _aor = mdn_activity_odds_ratio_wide(
                    _p0.loc[_common_genes_aor, _common_samples_aor],
                    _pi_gamma.loc[_common_genes_aor, _common_samples_aor],
                    _pi_normal.loc[_common_genes_aor, _common_samples_aor],
                ).to_numpy(dtype=np.float64)
                _valid_aor = (
                    np.isfinite(_t_aor) & np.isfinite(_p_aor) & np.isfinite(_aor)
                )
                _gene_labels_aor_2d = np.array(
                    [[gene_name] * len(_common_samples_aor) for gene_name in _common_genes_aor],
                    dtype=object,
                )
                pooled_abs_aor.append(np.abs(_aor)[_valid_aor])
                pooled_aor_abs_residual.append(np.abs(_t_aor - _p_aor)[_valid_aor])
                _pooled_gene_labels_aor.append(_gene_labels_aor_2d[_valid_aor])

        for gs_display, gs_key in _ROC_GENE_SETS.items():
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

            for view_name, view_models in benchmark_model_views().items():
                view_order = [model_name for model_name in view_models if model_name in model_preds]
                if not view_order:
                    continue
                for variant in confusion_variants:
                    variant_rows = _benchmark_patient_confusion_rows_for_variant(
                        truth_wide,
                        model_preds,
                        proteus_mdn,
                        view_order,
                        common_genes,
                        cohort,
                        variant,
                    )
                    if not variant_rows:
                        continue
                    key = (view_name, gs_display, str(variant["suffix"]))
                    pooled_confusion_rows.setdefault(key, []).extend(variant_rows)

    metrics_table = pd.DataFrame(all_metrics_rows)
    gene_rho_table = pd.DataFrame(all_gene_rho_rows)
    sample_rho_table = pd.DataFrame(all_sample_rho_rows)

    if metrics_table.empty:
        print("  [skip] No metrics computed")
        return

    # ── CI-width / |AOR| vs |residual| scatter plots ──────────────────────
    if (pooled_ci_width and _pooled_gene_labels_ci) or (pooled_abs_aor and _pooled_gene_labels_aor):

        # Per-gene variance from pooled truth values (used for top-10% filter)
        _gene_variance: Dict[str, float] = {}
        for gene_name, vals in _gene_truth_values.items():
            arr = np.asarray(vals, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            if len(finite) >= 3:
                _gene_variance[gene_name] = float(np.var(finite, ddof=1))

        # Top 10% most variable genes by truth variance
        if _gene_variance:
            _var_series = pd.Series(_gene_variance)
            _top10_threshold = float(_var_series.quantile(0.90))
            _top10_genes: set = set(_var_series[_var_series >= _top10_threshold].index.tolist())
        else:
            _top10_genes = set()

        def _scatter_subset_masks(all_gene_labels: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
            masks: Dict[str, Optional[np.ndarray]] = {"WholeGenome": None}
            for _gs_display, _gs_key in _ROC_GENE_SETS.items():
                if gene_list_available(_gs_key):
                    _gs_genes = set(dict.fromkeys(get_gene_list(_gs_key)))
                    masks[_gs_display] = np.isin(all_gene_labels, list(_gs_genes))
            if _top10_genes:
                masks["Top10PctMostVariable"] = np.isin(all_gene_labels, list(_top10_genes))
            return masks

        if pooled_ci_width and _pooled_gene_labels_ci:
            _all_ci_width = np.concatenate(pooled_ci_width)
            _all_ci_abs_residual = np.concatenate(pooled_ci_abs_residual)
            _all_ci_gene_labels = np.concatenate(_pooled_gene_labels_ci).astype(str)
            _ci_scatter_subsets = _scatter_subset_masks(_all_ci_gene_labels)

            for _subset_label, _mask in _ci_scatter_subsets.items():
                if _mask is not None:
                    _xv = _all_ci_width[_mask]
                    _yv = _all_ci_abs_residual[_mask]
                else:
                    _xv = _all_ci_width
                    _yv = _all_ci_abs_residual
                if len(_xv) < 100:
                    print(f"  [skip] CI-width scatter {_subset_label}: too few points ({len(_xv)})")
                    continue
                _scatter_stem = os.path.join(
                    out_dir, f"CIWidth_vs_AbsResidual_{_subset_label}"
                )
                _maybe(
                    ci_width_vs_absolute_residual_scatter,
                    args.overwrite,
                    _xv,
                    _yv,
                    _scatter_stem,
                    title=f"CI Width vs |Residual| — {_subset_label.replace('PctMostVariable', '% Most Variable')}  (CRPC-WCDT + BLCA-RA, Proteus)",
                    x_label="90% CI width",
                    fit_label="Linear fit",
                    x_max=8.0,
                    y_max=8.0,
                )
                print(f"  CIWidth scatter {_subset_label}: {len(_xv):,} pairs")

        if pooled_abs_aor and _pooled_gene_labels_aor:
            _all_abs_aor = np.concatenate(pooled_abs_aor)
            _all_aor_abs_residual = np.concatenate(pooled_aor_abs_residual)
            _all_aor_gene_labels = np.concatenate(_pooled_gene_labels_aor).astype(str)
            _aor_scatter_subsets = _scatter_subset_masks(_all_aor_gene_labels)

            for _subset_label, _mask in _aor_scatter_subsets.items():
                if _mask is not None:
                    _xv = _all_abs_aor[_mask]
                    _yv = _all_aor_abs_residual[_mask]
                else:
                    _xv = _all_abs_aor
                    _yv = _all_aor_abs_residual
                if len(_xv) < 100:
                    print(f"  [skip] AbsAOR scatter {_subset_label}: too few points ({len(_xv)})")
                    continue
                _scatter_stem = os.path.join(
                    out_dir, f"AbsAOR_vs_AbsResidual_{_subset_label}"
                )
                _maybe(
                    ci_width_vs_absolute_residual_scatter,
                    args.overwrite,
                    _xv,
                    _yv,
                    _scatter_stem,
                    title=f"|AOR| vs |Residual| — {_subset_label.replace('PctMostVariable', '% Most Variable')}  (CRPC-WCDT + BLCA-RA, Proteus)",
                    x_label="|Activity odds ratio|",
                    fit_label="Linear fit",
                    x_max=10.0,
                    y_max=8.0,
                )
                print(f"  AbsAOR scatter {_subset_label}: {len(_xv):,} pairs")

    subset_model_order = [model_name for model_name in subset_view_models if model_name in pooled_threshold_pairs]
    if subset_model_order:
        _write_pooled_benchmark_threshold_artifacts(
            pooled_threshold_pairs,
            ensure_dir(os.path.join(out_dir, "SUBSET")),
            args.overwrite,
            subset_model_order,
        )

    for view_name, view_models in benchmark_model_views().items():
        view_model_order = [model_name for model_name in view_models if model_name in set(metrics_table["model"])]
        if not view_model_order:
            continue
        view_out = ensure_dir(os.path.join(out_dir, view_name))
        for gs_display in _ROC_GENE_SETS.keys():
            for variant in confusion_variants:
                variant_suffix = str(variant["suffix"])
                key = (view_name, gs_display, variant_suffix)
                rows = pooled_confusion_rows.get(key, [])
                if not rows:
                    continue
                conf_df = pd.DataFrame(rows)
                if conf_df.empty:
                    continue

                stem = f"OnOffConfusion{variant_suffix}_{gs_display}_PooledCRPC-WCDT-BLCA-RA"
                conf_df.to_csv(
                    os.path.join(view_out, f"{stem}.tsv"),
                    sep="\t",
                    index=False,
                )

                present_models = set(conf_df["model"].astype(str).tolist())
                plot_model_order: List[str] = []
                if "Proteus (uncertainty-informed)" in present_models:
                    plot_model_order.append("Proteus (uncertainty-informed)")
                if "Proteus" in present_models:
                    plot_model_order.append("Proteus")
                if "XGBoost" in present_models:
                    plot_model_order.append("XGBoost")
                for epic_model in ("EPIC_GEP", "EPIC_NDR", "EPIC_PFE"):
                    if epic_model in present_models:
                        plot_model_order.append(epic_model)
                        break

                _maybe(
                    benchmark_patient_onoff_confusion_plot,
                    args.overwrite,
                    conf_df,
                    os.path.join(view_out, stem),
                    title=(
                        f"On/Off Confusion - {gs_display} ({variant['title_suffix']})"
                    ),
                    model_order=plot_model_order,
                    cohort_order=PATIENT_BENCHMARK_COHORTS,
                )
                print(
                    f"    Pooled OnOffConfusion{variant_suffix} {view_name} {gs_display}: "
                    f"{len(conf_df)} rows"
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

    # CIDriverAnalysis: pooled, matched-patient drivers of confidence vs error.
    if ci_driver_bundles:
        run_ci_driver_analysis(
            ci_driver_bundles,
            ensure_dir(os.path.join(out_dir, "CIDriverAnalysis")),
            overwrite=args.overwrite,
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

_TDS_SPLITS_TSV = "data/TDS_sample-splits.tsv"

# Cohorts that have ground-truth expression labels and appear in TDS training.
_TDS_CV_COHORTS = ["CRPC-RA", "SCLC", "PC-PDX", "SCLC-PDX"]

# Per-cohort analysis config for TDS-derived validation splits
_TDS_CV_COHORT_CONFIG: Dict[str, Dict[str, Any]] = {
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


def _load_tds_fold_splits() -> pd.DataFrame:
    """Load data/TDS_sample-splits.tsv → DataFrame with Cohort, Sample, Fold."""
    path = PROJECT_ROOT / _TDS_SPLITS_TSV
    return pd.read_csv(path, sep="\t")


def _build_tds_truth(minimum_tfx: float = 0.0) -> pd.DataFrame:
    """Build canonical truth for patient-CV cohorts (CRPC-RA, SCLC, PC-PDX, SCLC-PDX).

    Loads Log2CPMplus1 labels for all four cohorts, keeps only samples present
    in the TDS sample-splits table (excludes HO PDX), concatenates on a common
    gene index.  PHT samples are excluded from truth (no independent labels).
    """
    splits = _load_tds_fold_splits()
    split_samples = set(splits["Sample"])

    parts = []
    for cohort in _TDS_CV_COHORTS:
        truth_path = PROJECT_ROOT / "labels" / f"{cohort}_Log2CPMplus1.tsv"
        _require_existing_path(truth_path, description=f"patient-CV truth for {cohort}")
        tw = load_truth(cohort)
        keep = [s for s in tw.columns if s in split_samples]
        if minimum_tfx > 0.0 and _is_patient_cohort(cohort):
            filtered_subtypes = _load_filtered_subtypes_for_reference(
                cohort,
                pd.Index(keep),
                minimum_tfx=minimum_tfx,
                report_context=f"TDS truth {cohort}",
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
    needs_tds_truth = any(
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

    if not (needs_tds_truth or needs_patient_holdout_truth or needs_discovery):
        return

    if minimum_tfx <= 0.0:
        return

    if needs_tds_truth:
        splits = _load_tds_fold_splits()
        split_samples = set(splits["Sample"])
        for cohort in _TDS_CV_COHORTS:
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
        help=(
            "Render and exclude single-gene samples inside the AOR ambiguity band as hollow markers "
            "(fixed to |AOR| <= 5.0 when --apply_calibration is OFF). In Discovery_Patient this also "
            "drops genes that are AOR-ambiguous in a majority of samples from the prediction table "
            "before all downstream analyses (limma, ssGSEA, PAM50, survival/clinical)."
        ),
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
