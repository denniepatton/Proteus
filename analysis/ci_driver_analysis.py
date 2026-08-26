"""CIDriverAnalysis — what drives Proteus confidence relative to actual error.

This sub-analysis of ``Benchmarking_Patient`` pools the matched, held-out
patient data (CRPC-WCDT + BLCA-RA) and asks two questions:

1. Does Proteus abstention/error come from *sample* (technical) effects or
   *gene* (biological) effects? Answered with a crossed random-effects
   variance decomposition plus sample-, gene-, and gene×sample diagnostics.
2. What makes a gene intrinsically hard? Answered by residualizing technical
   effects (a gene-level random effect) and relating that difficulty to
   gene-level covariates.

Design notes
------------
* The crossed mixed model ``outcome ~ fixed + (1|sample) + (1|gene)`` is fit
  with a dependency-free EM variance-components estimator
  (:func:`_fit_crossed_variance_components`); there is no ``statsmodels``/
  ``lme4`` requirement. The estimator alternates BLUP updates for the two
  crossed grouping factors with a GLS update of the fixed effects and an EM
  update of the three variance components (posterior-variance corrected).
* "Training dynamic range" per gene is the per-gene training-truth variance,
  computed identically to Proteus training: unclipped variance of
  ``flat/tumor_exp`` over the TDS training pool with PHT admixtures excluded
  (:func:`_compute_training_gene_stats`).
* Gene-level depth is the TSS ``mean-region-depth`` Triton feature; promoter
  GC% and a CpG-island flag are derived from the stored TSS nucleotides over a
  ±500 bp core window.
* Low-confidence / abstained is defined as ``|AOR| <= 5`` — the fixed
  downstream ambiguity band used elsewhere in Benchmarking_Patient.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from analysis.analysis_utils import mdn_activity_odds_ratio_wide
from proteus_utils.compile_sample_data import GB_FEATURES, TSS_FEATURES

__all__ = ["run_ci_driver_analysis"]


# Fixed downstream ambiguity band: |AOR| <= 5 marks a low-confidence/abstained
# call (matches _DOWNSTREAM_FIXED_AOR_ABS_THRESHOLD in run_full_analysis.py).
_ABSTAIN_AOR_THRESHOLD: float = 5.0

# On/off expression boundary in log2(CPM+1) (matches the analysis on/off
# threshold used elsewhere). Signed distance = truth_expression - threshold.
_ON_OFF_THRESHOLD: float = 2.0

# Number of most-abstained genes to render in the abstention heatmap.
_DROPOUT_TOP_GENES: int = 800

# TSS depth feature index within the 11 Triton TSS features; gene-body depth
# index within the identical GB feature set.
_DEPTH_FEATURE_INDEX: int = TSS_FEATURES.index("mean-region-depth")
_GB_DEPTH_FEATURE_INDEX: int = GB_FEATURES.index("mean-region-depth")

# Promoter core window for GC/CpG (±500 bp about the 2000 bp window centre).
_PROMOTER_CORE_START: int = 500
_PROMOTER_CORE_END: int = 1500

# Gardiner-Garden & Frommer CpG-island criteria over the core window.
_CPG_GC_MIN: float = 0.50
_CPG_OBS_EXP_MIN: float = 0.60

_MANE_REFERENCE_REL: str = "reference/MANE.GRCh38.v1.3.UniversalReference.tsv"

# Variance-decomposition component labels and stable colours.
_VARIANCE_COMPONENTS: List[str] = [
    "sample identity",
    "gene identity",
    "technical covariates",
    "truth/boundary covariates",
    "unexplained",
]
_COMPONENT_COLORS: Dict[str, str] = {
    "sample identity": "#0077BB",
    "gene identity": "#009988",
    "technical covariates": "#EE7733",
    "truth/boundary covariates": "#CC3311",
    "unexplained": "#BBBBBB",
}

# Legend labels annotating exactly which model terms enter each source group.
_COMPONENT_LEGEND: Dict[str, str] = {
    "sample identity": "sample identity — (1 | sample) random effect",
    "gene identity": "gene identity — (1 | gene) random effect",
    "technical covariates": "technical covariates — cohort, TFx, gene depth",
    "truth/boundary covariates": "truth/boundary covariates — truth expression, training DR",
    "unexplained": "unexplained — residual variance",
}

# Outcome display names for the variance decomposition.
_OUTCOME_LABELS: Dict[str, str] = {
    "ci_width": "90% CI width",
    "neg_abs_aor": "-|AOR|",
    "abs_residual": "|residual|",
    "abstain_flag": "abstain flag (|AOR|<=5)",
}


# =====================================================================
# Gene-level metadata (MANE universal reference)
# =====================================================================

def _load_gene_metadata(project_root: Path, genes: List[str]) -> pd.DataFrame:
    """Return per-gene length, transcript length, and exon count for *genes*.

    Reads the MANE universal reference. Genes absent from MANE receive NaN
    metadata (kept so the row survives; the elastic net imputes/drops NaNs).
    """
    path = project_root / _MANE_REFERENCE_REL
    if not path.exists():
        raise FileNotFoundError(
            f"CIDriverAnalysis requires gene metadata at {path}"
        )
    meta = pd.read_csv(path, sep="\t")
    # The symbol column is duplicated ("symbol" and "Gene_symbol"); prefer the
    # canonical HGNC symbol column when present.
    symbol_col = "Gene_symbol" if "Gene_symbol" in meta.columns else "symbol"
    meta = meta.dropna(subset=[symbol_col]).copy()
    meta[symbol_col] = meta[symbol_col].astype(str)
    meta = meta.drop_duplicates(subset=[symbol_col], keep="first")
    meta = meta.set_index(symbol_col)

    gene_length = (
        pd.to_numeric(meta["chr_end"], errors="coerce")
        - pd.to_numeric(meta["chr_start"], errors="coerce")
    ).abs()
    frame = pd.DataFrame(index=pd.Index(genes, name="gene"))
    frame["gene_length"] = gene_length.reindex(genes).to_numpy()
    frame["transcript_length"] = pd.to_numeric(
        meta["transcript_length"], errors="coerce"
    ).reindex(genes).to_numpy()
    frame["num_exons"] = pd.to_numeric(
        meta["NumExons"], errors="coerce"
    ).reindex(genes).to_numpy()
    return frame


# =====================================================================
# Training-truth per-gene statistics (identical to Proteus training)
# =====================================================================

def _zarr_store_is_valid(zpath: str) -> bool:
    """Return True when *zpath* is a fully-initialized Zarr v2 group.

    ``os.path.isdir`` alone accepts partially-written stores (e.g. a run
    interrupted on scratch) that lack group metadata; requiring the ``.zgroup``
    marker rejects those so callers can fail loudly instead of crashing inside
    ``zarr.open``.
    """
    return os.path.isdir(zpath) and os.path.exists(os.path.join(zpath, ".zgroup"))


def _collect_training_zarr_paths(project_root: Path, zarr_base: str) -> List[str]:
    """Enumerate and validate every non-PHT TDS training Zarr store.

    Requires all expected stores to be present and valid so the per-gene
    variance pool matches training exactly. Missing or partially-written stores
    raise a clear, actionable error listing which ones to regenerate.
    """
    import csv

    from proteus_utils.data_helpers import zarr_path_for_sample

    training_tsv = project_root / "data" / "TDS_sample-splits.tsv"
    if not training_tsv.exists():
        raise FileNotFoundError(
            f"CIDriverAnalysis requires the training splits TSV at {training_tsv}"
        )

    valid_paths: List[str] = []
    missing: List[str] = []
    with open(training_tsv, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cohort = row["Cohort"].strip()
            if cohort == "PHT":
                continue
            sample = row["Sample"].strip()
            zpath = zarr_path_for_sample(sample, cohort, zarr_base)
            if _zarr_store_is_valid(zpath):
                valid_paths.append(zpath)
            else:
                missing.append(f"{cohort}/{sample}")

    if missing:
        raise FileNotFoundError(
            "CIDriverAnalysis needs every non-PHT TDS training Zarr store to "
            "compute per-gene dynamic range identically to training, but "
            f"{len(missing)} store(s) are missing or partially written under "
            f"{zarr_base}. Regenerate them and retry. Missing: "
            f"{', '.join(missing[:15])}"
            + (" ..." if len(missing) > 15 else "")
        )
    if not valid_paths:
        raise FileNotFoundError(
            "CIDriverAnalysis found no non-PHT TDS training Zarr stores under "
            f"{zarr_base}; the training pool is required to compute per-gene "
            "dynamic range."
        )
    return valid_paths


def _compute_training_gene_stats(
    project_root: Path,
    zarr_base: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-gene training variance ("dynamic range") and mean expression.

    Mirrors Proteus training exactly: unclipped variance and mean of
    ``flat/tumor_exp`` over the TDS training pool with PHT admixtures excluded
    (PHT samples duplicate their parent-PDX labels and are non-independent).

    Unlike ``collect_zarr_paths`` (which silently skips missing stores), every
    expected non-PHT training store must be present and valid so the pool is
    identical to training; missing/partial stores raise a clear error.
    """
    from proteus_utils.proteus_utils import (
        compute_gene_mean_std_from_zarr,
        compute_gene_variances_unclipped,
    )

    train_zarr_paths = _collect_training_zarr_paths(project_root, zarr_base)
    train_variance = compute_gene_variances_unclipped(
        train_zarr_paths, label_key="flat/tumor_exp"
    )
    train_mean, _train_std = compute_gene_mean_std_from_zarr(
        train_zarr_paths, label_key="flat/tumor_exp"
    )
    return train_variance, train_mean


# =====================================================================
# Held-out Zarr covariates (per-sample depth, sequence GC/CpG, feature var)
# =====================================================================

def _promoter_gc_and_cpg(nucleotides: np.ndarray) -> Tuple[float, float]:
    """Return (GC fraction, CpG-island flag) for one gene's TSS nucleotides.

    ``nucleotides`` is (2000, 4) one-hot in A, C, G, T order. GC and the
    observed/expected CpG ratio are computed over the ±500 bp core window
    using the Gardiner-Garden & Frommer definition.
    """
    core = nucleotides[_PROMOTER_CORE_START:_PROMOTER_CORE_END, :]
    counts = core.sum(axis=0).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan"), float("nan")
    a_n, c_n, g_n, t_n = counts
    gc_fraction = (c_n + g_n) / total

    # Observed CpG dinucleotides: C at position i followed by G at i+1.
    c_track = core[:, 1].astype(np.float64)
    g_track = core[:, 2].astype(np.float64)
    observed_cpg = float(np.sum(c_track[:-1] * g_track[1:]))
    expected_cpg = (c_n * g_n) / total if total > 0 else 0.0
    obs_exp = observed_cpg / expected_cpg if expected_cpg > 0 else 0.0

    is_island = float(gc_fraction >= _CPG_GC_MIN and obs_exp >= _CPG_OBS_EXP_MIN)
    return float(gc_fraction), is_island


def _read_cohort_zarr_covariates(
    cohort: str,
    samples: List[str],
    zarr_base: str,
) -> Dict[str, object]:
    """Read per-(gene, sample) depth and per-gene sequence/feature covariates.

    Returns a dict with:
        gene_depth_wide : DataFrame (genes x samples) TSS mean-region-depth
        gene_gbdepth_wide : DataFrame (genes x samples) GB mean-region-depth
        sample_median_depth : Series (samples) median gene depth
        gc_promoter : Series (genes) promoter GC fraction
        cpg_island : Series (genes) CpG-island flag
        promoter_feature_variance : Series (genes) mean across-sample var of TSS features
        genebody_feature_variance : Series (genes) mean across-sample var of GB features
    """
    from proteus_utils.data_helpers import zarr_path_for_sample

    depth_columns: Dict[str, pd.Series] = {}
    gbdepth_columns: Dict[str, pd.Series] = {}
    tss_feature_stack: Dict[str, List[np.ndarray]] = {}
    gb_feature_stack: Dict[str, List[np.ndarray]] = {}
    gc_series: Dict[str, float] = {}
    cpg_series: Dict[str, float] = {}
    sample_median_depth: Dict[str, float] = {}

    for sample in samples:
        zpath = zarr_path_for_sample(sample, cohort, zarr_base)
        if not _zarr_store_is_valid(zpath):
            raise FileNotFoundError(
                f"CIDriverAnalysis requires the held-out Zarr store for "
                f"{cohort} sample '{sample}' at {zpath} (missing or partially "
                f"written); regenerate it and retry."
            )
        store = zarr.open(zpath, mode="r")
        gene_names = np.array([str(g) for g in store["gene_names"][:]])
        tss_features = np.asarray(store["flat/TSS_features"][:], dtype=np.float64)
        gb_features = np.asarray(store["flat/GB_features"][:], dtype=np.float64)
        depth_vec = tss_features[:, _DEPTH_FEATURE_INDEX]
        gbdepth_vec = gb_features[:, _GB_DEPTH_FEATURE_INDEX]

        depth_columns[sample] = pd.Series(depth_vec, index=gene_names)
        gbdepth_columns[sample] = pd.Series(gbdepth_vec, index=gene_names)
        finite_depth = depth_vec[np.isfinite(depth_vec)]
        sample_median_depth[sample] = (
            float(np.median(finite_depth)) if finite_depth.size else float("nan")
        )

        for gi, gene in enumerate(gene_names):
            tss_feature_stack.setdefault(gene, []).append(tss_features[gi])
            gb_feature_stack.setdefault(gene, []).append(gb_features[gi])

        # Sequence is reference-genome derived and identical across samples;
        # read nucleotides once per gene from the first sample that carries it.
        if not gc_series:
            nucleotides = np.asarray(store["flat/TSS_nucleotides"][:], dtype=np.uint8)
            for gi, gene in enumerate(gene_names):
                gc_val, cpg_val = _promoter_gc_and_cpg(nucleotides[gi])
                gc_series[gene] = gc_val
                cpg_series[gene] = cpg_val

    gene_depth_wide = pd.DataFrame(depth_columns)
    gene_gbdepth_wide = pd.DataFrame(gbdepth_columns)

    def _mean_feature_variance(stack: Dict[str, List[np.ndarray]]) -> pd.Series:
        values: Dict[str, float] = {}
        for gene, rows in stack.items():
            arr = np.asarray(rows, dtype=np.float64)  # (n_samples, 11)
            if arr.shape[0] < 2:
                values[gene] = float("nan")
                continue
            per_feature_var = np.nanvar(arr, axis=0, ddof=1)
            values[gene] = float(np.nanmean(per_feature_var))
        return pd.Series(values)

    return {
        "gene_depth_wide": gene_depth_wide,
        "gene_gbdepth_wide": gene_gbdepth_wide,
        "sample_median_depth": pd.Series(sample_median_depth),
        "gc_promoter": pd.Series(gc_series),
        "cpg_island": pd.Series(cpg_series),
        "promoter_feature_variance": _mean_feature_variance(tss_feature_stack),
        "genebody_feature_variance": _mean_feature_variance(gb_feature_stack),
    }


# =====================================================================
# Long-format (gene, sample) assembly
# =====================================================================

def _assemble_long_table(
    bundles: Dict[str, Dict[str, object]],
    zarr_base: str,
    project_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the pooled per-(gene, sample) table plus gene and sample covariate
    frames across both patient cohorts.

    Returns
    -------
    long_df : per-(gene, sample) outcomes + covariates.
    gene_cov : per-gene covariates (indexed by gene).
    sample_cov : per-sample covariates (indexed by "cohort::sample").
    """
    train_variance, train_mean = _compute_training_gene_stats(project_root, zarr_base)

    long_rows: List[pd.DataFrame] = []
    sample_records: List[Dict[str, object]] = []
    zarr_cov_by_cohort: Dict[str, Dict[str, object]] = {}

    for cohort, bundle in bundles.items():
        truth_wide: pd.DataFrame = bundle["truth_wide"]
        pred_wide: pd.DataFrame = bundle["pred_wide"]
        ci_lo: pd.DataFrame = bundle["ci_lo"]
        ci_hi: pd.DataFrame = bundle["ci_hi"]
        p0: pd.DataFrame = bundle["p0"]
        pi_gamma: pd.DataFrame = bundle["pi_gamma"]
        pi_normal: pd.DataFrame = bundle["pi_normal"]
        tfx: pd.Series = bundle["tfx"]

        samples = list(truth_wide.columns)
        zarr_cov = _read_cohort_zarr_covariates(cohort, samples, zarr_base)
        zarr_cov_by_cohort[cohort] = zarr_cov
        gene_depth_wide: pd.DataFrame = zarr_cov["gene_depth_wide"]
        gene_gbdepth_wide: pd.DataFrame = zarr_cov["gene_gbdepth_wide"]
        sample_median_depth: pd.Series = zarr_cov["sample_median_depth"]

        aor = mdn_activity_odds_ratio_wide(p0, pi_gamma, pi_normal)

        genes = (
            truth_wide.index
            .intersection(pred_wide.index)
            .intersection(ci_lo.index)
            .intersection(ci_hi.index)
            .intersection(aor.index)
        )
        genes = [g for g in genes if g in gene_depth_wide.index]

        ci_width = (ci_hi.loc[genes, samples] - ci_lo.loc[genes, samples])
        t = truth_wide.loc[genes, samples]
        p = pred_wide.loc[genes, samples]
        a = aor.loc[genes, samples]
        d = gene_depth_wide.loc[genes, samples]
        gb = gene_gbdepth_wide.reindex(index=genes, columns=samples)

        for sample in samples:
            col_t = t[sample].to_numpy(dtype=np.float64)
            col_p = p[sample].to_numpy(dtype=np.float64)
            col_ci = ci_width[sample].to_numpy(dtype=np.float64)
            col_a = a[sample].to_numpy(dtype=np.float64)
            col_d = d[sample].to_numpy(dtype=np.float64)
            col_gb = gb[sample].to_numpy(dtype=np.float64)
            valid = np.isfinite(col_t) & np.isfinite(col_p) & np.isfinite(col_a)
            if not valid.any():
                continue
            abs_res = np.abs(col_t - col_p)
            tfx_val = float(tfx.get(sample, np.nan))
            frame = pd.DataFrame({
                "gene": np.asarray(genes)[valid],
                "sample": f"{cohort}::{sample}",
                "cohort": cohort,
                "truth_expression": col_t[valid],
                "prediction": col_p[valid],
                "ci_width": col_ci[valid],
                "abs_residual": abs_res[valid],
                "neg_abs_aor": -np.abs(col_a[valid]),
                "abstain_flag": (np.abs(col_a[valid]) <= _ABSTAIN_AOR_THRESHOLD).astype(float),
                "gene_depth": col_d[valid],
                "gb_depth": col_gb[valid],
                "boundary_distance": col_t[valid] - _ON_OFF_THRESHOLD,
                "effective_tumor_coverage": tfx_val * col_gb[valid],
                "tfx": tfx_val,
            })
            long_rows.append(frame)

        for sample in samples:
            sample_records.append({
                "sample": f"{cohort}::{sample}",
                "cohort": cohort,
                "tfx": float(tfx.get(sample, np.nan)),
                "sample_depth": float(sample_median_depth.get(sample, np.nan)),
            })

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df["training_dynamic_range"] = long_df["gene"].map(train_variance)
    long_df["training_mean_expression"] = long_df["gene"].map(train_mean)

    # Validation dynamic range: per-gene variance of held-out truth WITHIN each
    # cohort (BLCA/CRPC have distinct baseline epigenetic profiles). Keyed by
    # (gene, cohort); the pooled per-gene value is also stored for annotations.
    val_dr_by_gene_cohort = (
        long_df.groupby(["gene", "cohort"])["truth_expression"].var(ddof=1)
        .rename("validation_dynamic_range")
        .reset_index()
    )
    long_df = long_df.merge(val_dr_by_gene_cohort, on=["gene", "cohort"], how="left")

    sample_cov = pd.DataFrame(sample_records).set_index("sample")

    # Per-gene covariate frame (pooled across cohorts).
    genes_all = sorted(long_df["gene"].unique())
    gene_meta = _load_gene_metadata(project_root, genes_all)
    gene_cov = gene_meta.copy()
    gene_cov["training_dynamic_range"] = pd.Series(train_variance).reindex(genes_all).to_numpy()
    gene_cov["training_mean_expression"] = pd.Series(train_mean).reindex(genes_all).to_numpy()

    # Pool the zarr-derived per-gene covariates across cohorts (mean where a
    # gene appears in both). Sequence covariates are identical across cohorts.
    def _pool(series_key: str) -> pd.Series:
        frames = [
            zarr_cov_by_cohort[c][series_key] for c in zarr_cov_by_cohort
        ]
        pooled = pd.concat(frames, axis=1).mean(axis=1)
        return pooled.reindex(genes_all)

    gene_cov["gc_promoter"] = _pool("gc_promoter").to_numpy()
    gene_cov["cpg_island"] = _pool("cpg_island").to_numpy()
    gene_cov["promoter_feature_variance"] = _pool("promoter_feature_variance").to_numpy()
    gene_cov["genebody_feature_variance"] = _pool("genebody_feature_variance").to_numpy()
    # Local fragment depth / effective coverage: mean gene depth across samples.
    mean_gene_depth = (
        long_df.groupby("gene")["gene_depth"].mean().reindex(genes_all)
    )
    gene_cov["mean_gene_depth"] = mean_gene_depth.to_numpy()
    # Local gene-body coverage: mean gene-body depth across samples.
    mean_gb_depth = (
        long_df.groupby("gene")["gb_depth"].mean().reindex(genes_all)
    )
    gene_cov["mean_gb_depth"] = mean_gb_depth.to_numpy()

    return long_df, gene_cov, sample_cov


# =====================================================================
# Crossed random-effects variance components (dependency-free EM)
# =====================================================================

def _fit_crossed_variance_components(
    y: np.ndarray,
    X: np.ndarray,
    sample_codes: np.ndarray,
    gene_codes: np.ndarray,
    n_samples: int,
    n_genes: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """Fit ``y ~ X*beta + (1|sample) + (1|gene)`` by EM variance components.

    Crossed grouping factors are handled by alternating BLUP (shrinkage) updates
    for the sample and gene effects, a GLS/OLS update of the fixed effects on
    the RE-adjusted response, and an EM update of the three variance components
    (with posterior-variance corrections). Deterministic; no RNG.

    Returns beta, sample/gene BLUPs, and the three variance components.
    """
    n = y.shape[0]
    counts_s = np.bincount(sample_codes, minlength=n_samples).astype(np.float64)
    counts_g = np.bincount(gene_codes, minlength=n_genes).astype(np.float64)
    counts_s_safe = np.maximum(counts_s, 1.0)
    counts_g_safe = np.maximum(counts_g, 1.0)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u_s = np.zeros(n_samples)
    u_g = np.zeros(n_genes)
    resid = y - X @ beta
    var_e = float(np.var(resid)) or 1.0
    var_s = var_e
    var_g = var_e

    for _ in range(max_iter):
        # Sample BLUPs given beta and gene effects.
        partial_s = y - X @ beta - u_g[gene_codes]
        sum_s = np.bincount(sample_codes, weights=partial_s, minlength=n_samples)
        shrink_s = var_s / (var_s + var_e / counts_s_safe)
        u_s = shrink_s * (sum_s / counts_s_safe)

        # Gene BLUPs given beta and sample effects.
        partial_g = y - X @ beta - u_s[sample_codes]
        sum_g = np.bincount(gene_codes, weights=partial_g, minlength=n_genes)
        shrink_g = var_g / (var_g + var_e / counts_g_safe)
        u_g = shrink_g * (sum_g / counts_g_safe)

        # Fixed effects on RE-adjusted response.
        y_adj = y - u_s[sample_codes] - u_g[gene_codes]
        beta_new, *_ = np.linalg.lstsq(X, y_adj, rcond=None)

        resid = y - X @ beta_new - u_s[sample_codes] - u_g[gene_codes]
        var_e_new = float(np.mean(resid ** 2))
        # EM variance updates: E[u^2] = BLUP^2 + posterior variance.
        post_var_s = var_e * shrink_s / counts_s_safe
        post_var_g = var_e * shrink_g / counts_g_safe
        var_s_new = float(np.mean(u_s ** 2 + post_var_s))
        var_g_new = float(np.mean(u_g ** 2 + post_var_g))

        delta = (
            abs(var_e_new - var_e)
            + abs(var_s_new - var_s)
            + abs(var_g_new - var_g)
        )
        beta, var_e, var_s, var_g = beta_new, var_e_new, var_s_new, var_g_new
        if delta < tol:
            break

    return {
        "beta": beta,
        "u_s": u_s,
        "u_g": u_g,
        "var_e": var_e,
        "var_s": var_s,
        "var_g": var_g,
    }


def _build_fixed_design(
    long_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Return the standardized fixed design matrix and the technical / boundary
    column index blocks.

    Columns: [intercept, cohort(BLCA-RA=1), tfx, gene_depth,
              truth_expression, training_dynamic_range].
    Technical block = cohort, tfx, gene_depth.
    Truth/boundary block = truth_expression, training_dynamic_range.
    """
    def _z(values: np.ndarray) -> np.ndarray:
        v = values.astype(np.float64)
        finite = np.isfinite(v)
        mean = float(np.nanmean(v[finite])) if finite.any() else 0.0
        std = float(np.nanstd(v[finite])) if finite.any() else 1.0
        std = std if std > 0 else 1.0
        out = np.where(finite, (v - mean) / std, 0.0)
        return out

    cohorts = long_df["cohort"].astype("category")
    cohort_dummy = (cohorts.cat.codes.to_numpy() > 0).astype(np.float64)

    intercept = np.ones(len(long_df))
    tfx = _z(long_df["tfx"].to_numpy())
    depth = _z(long_df["gene_depth"].to_numpy())
    truth = _z(long_df["truth_expression"].to_numpy())
    dyn_range = _z(long_df["training_dynamic_range"].to_numpy())

    X = np.column_stack([intercept, cohort_dummy, tfx, depth, truth, dyn_range])
    technical_cols = [1, 2, 3]
    boundary_cols = [4, 5]
    return X, technical_cols, boundary_cols


def _variance_decomposition_for_outcome(
    long_df: pd.DataFrame,
    outcome: str,
    sample_codes: np.ndarray,
    gene_codes: np.ndarray,
    n_samples: int,
    n_genes: int,
) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Fit the mixed model for one outcome and return normalized variance
    component shares plus the raw fit (for gene random effects)."""
    y = long_df[outcome].to_numpy(dtype=np.float64)
    finite = np.isfinite(y)
    y = y[finite]
    X, technical_cols, boundary_cols = _build_fixed_design(long_df.loc[finite])
    s_codes = sample_codes[finite]
    g_codes = gene_codes[finite]

    fit = _fit_crossed_variance_components(
        y, X, s_codes, g_codes, n_samples, n_genes
    )
    beta = fit["beta"]
    technical_contrib = X[:, technical_cols] @ beta[technical_cols]
    boundary_contrib = X[:, boundary_cols] @ beta[boundary_cols]

    # The technical and truth/boundary predictors are collinear (e.g. TFx,
    # depth, truth expression, dynamic range covary), so var(technical) and
    # var(boundary) individually double-count the shared 2·Cov(·,·) cross-term
    # and do not sum to the total fixed-effect variance. Split the total
    # fixed-effect variance between the two blocks with a Shapley (LMG)
    # decomposition — the average over both sequential (Type-I) orderings — which
    # is order-independent and sums exactly to var(technical + boundary):
    #   φ_A = ½·var(A) + ½·(var(A+B) − var(B))
    var_technical = float(np.var(technical_contrib))
    var_boundary = float(np.var(boundary_contrib))
    var_fixed_total = float(np.var(technical_contrib + boundary_contrib))
    shapley_technical = 0.5 * var_technical + 0.5 * (var_fixed_total - var_boundary)
    shapley_boundary = 0.5 * var_boundary + 0.5 * (var_fixed_total - var_technical)

    components = {
        "sample identity": max(fit["var_s"], 0.0),
        "gene identity": max(fit["var_g"], 0.0),
        "technical covariates": max(shapley_technical, 0.0),
        "truth/boundary covariates": max(shapley_boundary, 0.0),
        "unexplained": max(fit["var_e"], 0.0),
    }
    total = sum(components.values())
    if total <= 0:
        shares = {k: 0.0 for k in components}
    else:
        shares = {k: v / total for k, v in components.items()}
    return shares, fit


# =====================================================================
# Plots
# =====================================================================

def _plot_variance_decomposition(
    shares_by_outcome: Dict[str, Dict[str, float]],
    out_path: str,
) -> None:
    """Stacked bar of % variance explained per outcome, with % labels."""
    outcomes = list(shares_by_outcome.keys())
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(outcomes))
    bottom = np.zeros(len(outcomes))
    for component in _VARIANCE_COMPONENTS:
        heights = np.array([
            100.0 * shares_by_outcome[o].get(component, 0.0) for o in outcomes
        ])
        ax.bar(
            x, heights, bottom=bottom, width=0.62,
            color=_COMPONENT_COLORS[component], label=_COMPONENT_LEGEND[component],
            edgecolor="white", linewidth=0.6,
        )
        # Label each segment large enough to read (>= 3%).
        for xi, (height, base) in enumerate(zip(heights, bottom)):
            if height >= 3.0:
                txt_color = "white" if component != "unexplained" else "#333333"
                ax.text(
                    x[xi], base + height / 2.0, f"{height:.1f}%",
                    ha="center", va="center", fontsize=8.5, color=txt_color,
                    fontweight="bold",
                )
        bottom += heights
    ax.set_xticks(x)
    ax.set_xticklabels([_OUTCOME_LABELS.get(o, o) for o in outcomes], fontsize=10)
    ax.set_ylabel("% variance explained", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "CIDriverAnalysis — variance decomposition of confidence/error",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(
        fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, title="Variance source",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_diagnostics(
    sample_stats: pd.DataFrame,
    out_path: str,
) -> None:
    """Sample-level scatter: x=TFx, size=depth, color=cohort, 4 y-axis views."""
    y_specs = [
        ("median_ci_width", "median 90% CI width"),
        ("fraction_abstained", "fraction genes abstained"),
        ("median_abs_residual", "median |residual|"),
        ("sample_r2", "sample R² across genes"),
    ]
    cohorts = sorted(sample_stats["cohort"].unique())
    palette = {c: color for c, color in zip(cohorts, ["#0077BB", "#EE7733", "#009988", "#CC3311"])}
    depth = sample_stats["sample_depth"].to_numpy(dtype=np.float64)
    finite_depth = depth[np.isfinite(depth)]
    d_min = float(np.min(finite_depth)) if finite_depth.size else 0.0
    d_max = float(np.max(finite_depth)) if finite_depth.size else 1.0
    d_range = (d_max - d_min) or 1.0

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for ax, (col, ylabel) in zip(axes, y_specs):
        for cohort in cohorts:
            sub = sample_stats[sample_stats["cohort"] == cohort]
            sizes = 40 + 220 * (
                (sub["sample_depth"].to_numpy(dtype=np.float64) - d_min) / d_range
            )
            sizes = np.where(np.isfinite(sizes), sizes, 60)
            ax.scatter(
                sub["tfx"], sub[col], s=sizes, alpha=0.75,
                color=palette[cohort], edgecolor="black", linewidth=0.4,
                label=cohort,
            )
        ax.set_xlabel("Tumor fraction (TFx)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.15)
    axes[0].legend(title="Cohort", fontsize=8)
    fig.suptitle("CIDriverAnalysis — sample-level diagnostics (point size ∝ sample depth)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_diagnostics(
    gene_stats: pd.DataFrame,
    gene_cohort_stats: pd.DataFrame,
    out_path: str,
    *,
    skill_val_quantile: float = 0.75,
) -> None:
    """Gene reliability map with an absolute-error and a variance-normalized view.

    Left: x=fraction samples abstained, y=median |residual| (absolute error),
    color=training dynamic range, size=training mean expression. Right: the same
    x versus gene-wise skill score (R²) among validation-dynamic genes (top
    quartile of validation DR within cohort), where R² is stable; color=
    validation dynamic range. Skill > 0 means the model beats the gene's mean.
    """
    x = gene_stats["fraction_abstained"].to_numpy(dtype=np.float64)
    y = gene_stats["median_abs_residual"].to_numpy(dtype=np.float64)
    color = gene_stats["training_dynamic_range"].to_numpy(dtype=np.float64)
    mean_expr = gene_stats["training_mean_expression"].to_numpy(dtype=np.float64)

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    color = color[finite]
    mean_expr = mean_expr[finite]
    m_finite = mean_expr[np.isfinite(mean_expr)]
    m_min = float(np.min(m_finite)) if m_finite.size else 0.0
    m_max = float(np.max(m_finite)) if m_finite.size else 1.0
    m_range = (m_max - m_min) or 1.0
    sizes = 15 + 120 * ((mean_expr - m_min) / m_range)
    sizes = np.where(np.isfinite(sizes), sizes, 20)

    color_finite = color[np.isfinite(color)]
    vmax = float(np.nanpercentile(color_finite, 95)) if color_finite.size else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    scatter = ax.scatter(
        x, y, c=color, s=sizes, cmap="viridis", vmin=0, vmax=vmax,
        alpha=0.7, edgecolor="none",
    )
    ax.axvline(float(np.nanmedian(x)), color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(float(np.nanmedian(y)), color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("Fraction of samples abstained (|AOR|<=5)")
    ax.set_ylabel("Median |residual| (absolute error)")
    ax.set_title("Absolute error")
    ax.text(0.02, 0.98, "low conf / low err\n(conservative)", transform=ax.transAxes,
            va="top", ha="left", fontsize=7, color="#555555")
    ax.text(0.98, 0.02, "high conf / high err\n(dangerous)", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=7, color="#555555")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Training dynamic range (per-gene variance)")

    # Right panel: variance-normalized skill on validation-dynamic genes.
    ax2 = axes[1]
    gcs = gene_cohort_stats.copy()
    gcs["val_rank"] = gcs.groupby("cohort")["validation_dynamic_range"].rank(pct=True)
    dyn = gcs[gcs["val_rank"] >= skill_val_quantile]
    sx = dyn["fraction_abstained"].to_numpy(dtype=np.float64)
    sy = dyn["skill_score"].to_numpy(dtype=np.float64)
    sc = dyn["validation_dynamic_range"].to_numpy(dtype=np.float64)
    sfinite = np.isfinite(sx) & np.isfinite(sy)
    sc_finite = sc[np.isfinite(sc)]
    svmax = float(np.nanpercentile(sc_finite, 95)) if sc_finite.size else 1.0
    sc2 = ax2.scatter(
        sx[sfinite], sy[sfinite], c=sc[sfinite], s=18, cmap="cividis",
        vmin=0, vmax=svmax, alpha=0.7, edgecolor="none",
    )
    ax2.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlabel("Fraction of samples abstained (|AOR|<=5)")
    ax2.set_ylabel("Gene-wise skill score (R²)")
    ax2.set_title("Variance-normalized skill (validation-dynamic genes)")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label("Validation dynamic range")

    fig.suptitle("CIDriverAnalysis — gene reliability map", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_dropout_heatmap(
    dropout: pd.DataFrame,
    sample_stats: pd.DataFrame,
    gene_stats: pd.DataFrame,
    out_path: str,
    *,
    top_genes: int = _DROPOUT_TOP_GENES,
) -> None:
    """Gene x sample abstention heatmap for the most-abstained genes.

    Rows = the ``top_genes`` genes with the highest abstention rate, ordered by
    that rate; columns = samples grouped by cohort then descending TFx. Left
    strips annotate each gene's training and (pooled) validation dynamic range;
    top strips annotate sample TFx, depth, and cohort.
    """
    gene_order = list(
        gene_stats.sort_values("fraction_abstained", ascending=False).index
    )
    gene_order = [g for g in gene_order if g in dropout.index][:int(top_genes)]

    # Columns grouped by cohort, then descending TFx within cohort.
    sample_meta = sample_stats.loc[[s for s in sample_stats.index if s in dropout.columns]]
    sample_order = list(
        sample_meta.sort_values(["cohort", "tfx"], ascending=[True, False]).index
    )
    matrix = dropout.loc[gene_order, sample_order].to_numpy(dtype=np.float64)

    n_samples = len(sample_order)
    fig = plt.figure(figsize=(max(6, 0.34 * n_samples + 3.5), 9))
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[0.5, 0.5, 0.5, 12],
        width_ratios=[0.45, 0.45, 12],
        hspace=0.05, wspace=0.06,
    )

    cohort_codes = sample_stats.loc[sample_order, "cohort"].astype("category").cat.codes
    ax_tfx = fig.add_subplot(gs[0, 2])
    ax_depth = fig.add_subplot(gs[1, 2])
    ax_cohort = fig.add_subplot(gs[2, 2])
    ax_main = fig.add_subplot(gs[3, 2])
    ax_train_dr = fig.add_subplot(gs[3, 0])
    ax_val_dr = fig.add_subplot(gs[3, 1])

    ax_tfx.imshow(
        sample_stats.loc[sample_order, "tfx"].to_numpy()[None, :],
        aspect="auto", cmap="magma",
    )
    ax_tfx.set_yticks([0]); ax_tfx.set_yticklabels(["TFx"], fontsize=7); ax_tfx.set_xticks([])
    ax_depth.imshow(
        sample_stats.loc[sample_order, "sample_depth"].to_numpy()[None, :],
        aspect="auto", cmap="cividis",
    )
    ax_depth.set_yticks([0]); ax_depth.set_yticklabels(["depth"], fontsize=7); ax_depth.set_xticks([])
    ax_cohort.imshow(
        cohort_codes.to_numpy()[None, :], aspect="auto", cmap="Set2",
    )
    ax_cohort.set_yticks([0]); ax_cohort.set_yticklabels(["cohort"], fontsize=7); ax_cohort.set_xticks([])

    ax_main.imshow(matrix, aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    ax_main.set_xlabel("samples (grouped by cohort, then descending TFx)")
    ax_main.set_ylabel(f"top {len(gene_order)} most-abstained genes")
    ax_main.set_yticks([])
    ax_main.set_xticks(np.arange(n_samples))
    ax_main.set_xticklabels(
        [s.split("::", 1)[-1] for s in sample_order], rotation=90, fontsize=5,
    )

    def _dr_strip(ax, column: str, label: str):
        values = np.log1p(
            gene_stats.loc[gene_order, column].to_numpy(dtype=np.float64)
        )
        finite = values[np.isfinite(values)]
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
        ax.imshow(values[:, None], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks([0]); ax.set_xticklabels([label], rotation=90, fontsize=7)
        ax.set_yticks([])

    _dr_strip(ax_train_dr, "training_dynamic_range", "train DR")
    _dr_strip(ax_val_dr, "validation_dynamic_range", "val DR")

    fig.suptitle(
        f"CIDriverAnalysis — top-{len(gene_order)} gene × sample abstention (1=abstained)"
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_difficulty(
    gene_re: pd.DataFrame,
    out_path: str,
) -> pd.DataFrame:
    """Residualized gene difficulty: uncertainty RE vs error RE, label top-10
    high-high genes. Returns the ranked high-high table."""
    x = gene_re["uncertainty_effect"].to_numpy(dtype=np.float64)
    y = gene_re["error_effect"].to_numpy(dtype=np.float64)
    x_med = float(np.nanmedian(x))
    y_med = float(np.nanmedian(y))

    high_high = gene_re[
        (gene_re["uncertainty_effect"] > x_med)
        & (gene_re["error_effect"] > y_med)
    ].copy()
    high_high["difficulty_score"] = (
        (high_high["uncertainty_effect"] - x_med)
        + (high_high["error_effect"] - y_med)
    )
    top10 = high_high.sort_values("difficulty_score", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=14, alpha=0.5, color="#0077BB", edgecolor="none")
    ax.axvline(x_med, color="grey", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(y_med, color="grey", ls="--", lw=0.8, alpha=0.7)
    for gene, row in top10.iterrows():
        ax.annotate(
            str(gene), (row["uncertainty_effect"], row["error_effect"]),
            fontsize=7, color="#CC3311",
            xytext=(3, 3), textcoords="offset points",
        )
    ax.set_xlabel("Residualized gene uncertainty effect (CI-width RE)")
    ax.set_ylabel("Residualized gene error effect (|residual| RE)")
    ax.set_title("CIDriverAnalysis — intrinsically hard genes")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return top10


def _plot_feature_importance(
    gene_re: pd.DataFrame,
    gene_cov: pd.DataFrame,
    out_path: str,
) -> pd.DataFrame:
    """Elastic-net standardized coefficients predicting gene RE (uncertainty
    and error) from gene-level covariates. Returns the coefficient table."""
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    predictors = [
        "training_dynamic_range",
        "training_mean_expression",
        "gene_length",
        "transcript_length",
        "num_exons",
        "gc_promoter",
        "cpg_island",
        "mean_gene_depth",
        "promoter_feature_variance",
        "genebody_feature_variance",
    ]
    targets = [("uncertainty_effect", "uncertainty RE"), ("error_effect", "error RE")]

    design = gene_cov.reindex(gene_re.index)
    coef_records: List[Dict[str, object]] = []
    r2_by_target: Dict[str, float] = {}

    for target_col, _label in targets:
        y = gene_re[target_col].to_numpy(dtype=np.float64)
        X_raw = design[predictors].to_numpy(dtype=np.float64)
        row_ok = np.isfinite(y) & np.all(np.isfinite(X_raw), axis=1)
        if row_ok.sum() < 20:
            for pred in predictors:
                coef_records.append({"target": target_col, "predictor": pred, "coef": np.nan})
            r2_by_target[target_col] = float("nan")
            continue
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_raw[row_ok])
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5, random_state=0, max_iter=10000,
        )
        model.fit(X_std, y[row_ok])
        r2_by_target[target_col] = float(model.score(X_std, y[row_ok]))
        for pred, coef in zip(predictors, model.coef_):
            coef_records.append({"target": target_col, "predictor": pred, "coef": float(coef)})

    coef_df = pd.DataFrame(coef_records)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(predictors))
    width = 0.4
    for offset, (target_col, label) in zip((-width / 2, width / 2), targets):
        sub = coef_df[coef_df["target"] == target_col].set_index("predictor").reindex(predictors)
        ax.barh(
            y_pos + offset, sub["coef"].to_numpy(dtype=np.float64), height=width,
            label=f"{label} (R²={r2_by_target[target_col]:.2f})",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors, fontsize=8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Elastic-net standardized coefficient")
    ax.set_title("CIDriverAnalysis — technical vs biological drivers of gene difficulty")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return coef_df


def _plot_dynamic_range_diagnostic(
    gene_stats: pd.DataFrame,
    gene_cohort_stats: pd.DataFrame,
    out_path: str,
    *,
    skill_val_quantile: float = 0.75,
) -> None:
    """Binned diagnostic: training dynamic range vs held-out error / CI / |AOR|,
    plus variance-normalized skill among validation-dynamic genes.

    The first three panels use absolute quantities (which rise with dynamic
    range by construction). The skill panel (gene-wise R² on validation-dynamic
    genes) is the variance-normalized companion: if skill also rises with
    training DR, higher-variance genes are genuinely better predicted, not just
    numerically larger-error.
    """
    dr = gene_stats["training_dynamic_range"].to_numpy(dtype=np.float64)
    finite = np.isfinite(dr)
    if finite.sum() < 10:
        return
    ranks = pd.qcut(gene_stats.loc[finite, "training_dynamic_range"], q=10, duplicates="drop")
    binned = gene_stats.loc[finite].groupby(ranks, observed=True)
    centers = binned["training_dynamic_range"].median().to_numpy()

    specs = [
        ("median_abs_residual", "median |residual|", "#CC3311"),
        ("median_ci_width", "median 90% CI width", "#0077BB"),
        ("median_abs_aor", "median |AOR|", "#009988"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for ax, (col, ylabel, color) in zip(axes[:3], specs):
        if col not in gene_stats.columns:
            continue
        ax.plot(centers, binned[col].median().to_numpy(), "o-", color=color)
        ax.set_xlabel("Training dynamic range (per-gene variance)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.15)

    # Skill panel among validation-dynamic genes.
    ax_skill = axes[3]
    gcs = gene_cohort_stats.copy()
    gcs["val_rank"] = gcs.groupby("cohort")["validation_dynamic_range"].rank(pct=True)
    dyn = gcs[gcs["val_rank"] >= skill_val_quantile]
    dyn = dyn[np.isfinite(dyn["training_dynamic_range"]) & np.isfinite(dyn["skill_score"])]
    if len(dyn) >= 20:
        skill_ranks = pd.qcut(dyn["training_dynamic_range"], q=10, duplicates="drop")
        skill_binned = dyn.groupby(skill_ranks, observed=True)
        ax_skill.plot(
            skill_binned["training_dynamic_range"].median().to_numpy(),
            skill_binned["skill_score"].median().to_numpy(),
            "o-", color="#333333",
        )
        ax_skill.axhline(0.0, color="grey", ls="--", lw=0.8)
    ax_skill.set_xlabel("Training dynamic range (per-gene variance)")
    ax_skill.set_ylabel("median skill score (R²), val-dynamic genes")
    ax_skill.grid(True, alpha=0.15)

    fig.suptitle("CIDriverAnalysis — dynamic-range diagnostic (decile bins)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Per-sample / per-gene diagnostic aggregation
# =====================================================================

def _aggregate_sample_stats(long_df: pd.DataFrame, sample_cov: pd.DataFrame) -> pd.DataFrame:
    grouped = long_df.groupby("sample")
    stats = pd.DataFrame({
        "median_ci_width": grouped["ci_width"].median(),
        "median_abs_residual": grouped["abs_residual"].median(),
        "fraction_abstained": grouped["abstain_flag"].mean(),
    })
    # Per-sample fit across genes: R^2 = squared Pearson r(truth, pred).
    sample_r = _grouped_pearson(long_df, "sample", "truth_expression", "prediction")
    stats["sample_r2"] = (sample_r ** 2).reindex(stats.index)
    stats = stats.join(sample_cov, how="left")
    return stats


def _grouped_pearson(
    df: pd.DataFrame, key: str, x_col: str, y_col: str,
) -> pd.Series:
    """Vectorised per-group Pearson r(x, y) via grouped moment sums."""
    work = df[[key, x_col, y_col]].copy()
    work["_xy"] = work[x_col] * work[y_col]
    work["_xx"] = work[x_col] * work[x_col]
    work["_yy"] = work[y_col] * work[y_col]
    grouped = work.groupby(key)
    mean_x = grouped[x_col].mean()
    mean_y = grouped[y_col].mean()
    cov = grouped["_xy"].mean() - mean_x * mean_y
    var_x = grouped["_xx"].mean() - mean_x * mean_x
    var_y = grouped["_yy"].mean() - mean_y * mean_y
    denom = np.sqrt(var_x * var_y)
    return (cov / denom.where(denom > 0, np.nan))


def _aggregate_gene_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = long_df.groupby("gene")
    stats = pd.DataFrame({
        "median_ci_width": grouped["ci_width"].median(),
        "median_abs_residual": grouped["abs_residual"].median(),
        "median_abs_aor": grouped["neg_abs_aor"].median().mul(-1.0),
        "fraction_abstained": grouped["abstain_flag"].mean(),
        "training_dynamic_range": grouped["training_dynamic_range"].first(),
        "training_mean_expression": grouped["training_mean_expression"].first(),
        # Pooled validation variance across all held-out samples (annotation).
        "validation_dynamic_range": grouped["truth_expression"].var(ddof=1),
        "validation_mean_expression": grouped["truth_expression"].mean(),
        "median_boundary_distance": grouped["boundary_distance"].median(),
        "n_obs": grouped["abstain_flag"].size(),
    })
    return stats


def _aggregate_gene_cohort_stats(
    long_df: pd.DataFrame, gene_cov: pd.DataFrame,
) -> pd.DataFrame:
    """Per (gene, cohort) skill/error/confidence metrics + covariates.

    ``skill_score = 1 - MSE_model / MSE_const`` where ``MSE_const`` is the error
    of predicting the gene's held-out mean (a gene-wise R²). It is unstable for
    genes with near-zero validation variance, so validation-dynamic filtering is
    applied by downstream consumers.
    """
    work = long_df.assign(_res_sq=long_df["abs_residual"].to_numpy() ** 2)
    grouped = work.groupby(["gene", "cohort"])
    mse_model = grouped["_res_sq"].mean()
    mse_const = grouped["truth_expression"].var(ddof=0)  # mean-baseline MSE
    # Correlation across samples within (gene, cohort): a scale-free companion
    # to |residual| that credits genes for tracking their own variation.
    work2 = work.assign(
        _tp=work["truth_expression"] * work["prediction"],
        _tt=work["truth_expression"] * work["truth_expression"],
        _pp=work["prediction"] * work["prediction"],
    )
    g2 = work2.groupby(["gene", "cohort"])
    mean_t = g2["truth_expression"].mean()
    mean_p = g2["prediction"].mean()
    cov_tp = g2["_tp"].mean() - mean_t * mean_p
    var_t = g2["_tt"].mean() - mean_t * mean_t
    var_p = g2["_pp"].mean() - mean_p * mean_p
    denom = np.sqrt(var_t * var_p)
    pearson = cov_tp / denom.where(denom > 0, np.nan)
    stats = pd.DataFrame({
        "median_abs_residual": grouped["abs_residual"].median(),
        "mse_model": mse_model,
        "mse_const": mse_const,
        "median_ci_width": grouped["ci_width"].median(),
        "median_abs_aor": grouped["neg_abs_aor"].median().mul(-1.0),
        "fraction_abstained": grouped["abstain_flag"].mean(),
        "pearson_across_samples": pearson,
        "training_dynamic_range": grouped["training_dynamic_range"].first(),
        "training_mean_expression": grouped["training_mean_expression"].first(),
        "validation_dynamic_range": grouped["validation_dynamic_range"].first(),
        "validation_mean_expression": grouped["truth_expression"].mean(),
        "mean_effective_coverage": grouped["effective_tumor_coverage"].mean(),
        "median_boundary_distance": grouped["boundary_distance"].median(),
        "mean_gb_depth": grouped["gb_depth"].mean(),
        "n_obs": grouped["abstain_flag"].size(),
    })
    safe_const = stats["mse_const"].where(stats["mse_const"] > 0, np.nan)
    stats["rmse"] = np.sqrt(stats["mse_model"])
    stats["normalized_rmse"] = np.sqrt(stats["mse_model"] / safe_const)
    stats["skill_score"] = 1.0 - stats["mse_model"] / safe_const
    cov_cols = [
        c for c in ("gc_promoter", "cpg_island", "gene_length",
                    "transcript_length", "num_exons")
        if c in gene_cov.columns
    ]
    stats = stats.reset_index()
    stats = stats.merge(
        gene_cov[cov_cols], left_on="gene", right_index=True, how="left",
    )
    return stats


# =====================================================================
# New diagnostics (v2)
# =====================================================================

def _plot_why_tough_genes(
    gene_stats: pd.DataFrame,
    gene_cov: pd.DataFrame,
    out_dir: str,
    *,
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    """Contrast the most-abstained genes against the rest across covariates.

    "Tough" genes are the top ``top_fraction`` by abstention rate. For each
    covariate the standardized mean difference (Cohen's d, tough − rest) is
    reported; positive d means the covariate is higher in tough genes. Writes a
    contrast bar plot, a contrast TSV, and the tough-gene covariate table.
    Returns the contrast table.
    """
    feature_cols = [
        ("training_dynamic_range", "training dynamic range"),
        ("validation_dynamic_range", "validation dynamic range"),
        ("training_mean_expression", "training mean expression"),
        ("validation_mean_expression", "validation mean expression"),
        ("median_boundary_distance", "distance to on/off boundary"),
        ("median_abs_residual", "median |residual|"),
        ("gc_promoter", "promoter GC"),
        ("cpg_island", "CpG island"),
        ("gene_length", "gene length"),
        ("transcript_length", "transcript length"),
        ("num_exons", "num exons"),
        ("mean_gb_depth", "gene-body depth"),
    ]
    frame = gene_stats.join(
        gene_cov[[c for c, _ in feature_cols if c in gene_cov.columns
                  and c not in gene_stats.columns]],
        how="left",
    )
    frac = frame["fraction_abstained"].to_numpy(dtype=np.float64)
    cutoff = float(np.nanquantile(frac, 1.0 - top_fraction))
    is_tough = frame["fraction_abstained"] >= cutoff

    records: List[Dict[str, object]] = []
    for col, label in feature_cols:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        tough_vals = values[is_tough].dropna().to_numpy()
        rest_vals = values[~is_tough].dropna().to_numpy()
        if tough_vals.size < 5 or rest_vals.size < 5:
            continue
        pooled_sd = np.sqrt(
            (np.var(tough_vals, ddof=1) + np.var(rest_vals, ddof=1)) / 2.0
        )
        cohen_d = (
            float((tough_vals.mean() - rest_vals.mean()) / pooled_sd)
            if pooled_sd > 0 else 0.0
        )
        records.append({
            "covariate": label,
            "mean_tough": float(tough_vals.mean()),
            "mean_rest": float(rest_vals.mean()),
            "cohen_d": cohen_d,
        })

    contrast = pd.DataFrame(records).sort_values("cohen_d")
    contrast.to_csv(os.path.join(out_dir, "why_tough_genes_contrast.tsv"),
                    sep="\t", index=False)
    tough_table = frame[is_tough].sort_values("fraction_abstained", ascending=False)
    tough_table.to_csv(os.path.join(out_dir, "why_tough_genes.tsv"), sep="\t")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#CC3311" if d > 0 else "#0077BB" for d in contrast["cohen_d"]]
    ax.barh(contrast["covariate"], contrast["cohen_d"], color=colors)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Standardized mean difference (Cohen's d), tough − rest")
    ax.set_title(
        f"CIDriverAnalysis — what makes the top {int(top_fraction*100)}% "
        f"most-abstained genes hard",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "why_tough_genes.pdf"), bbox_inches="tight")
    plt.close(fig)
    return contrast


def _quintile_labels(values: pd.Series, q: int = 5) -> pd.Series:
    """Return integer quantile-bin codes (0..q-1) for a numeric series."""
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        binned = pd.qcut(numeric, q, labels=False, duplicates="drop")
    except ValueError:
        binned = pd.Series(np.nan, index=values.index)
    return binned


def _plot_dr_2d_heatmap(
    gene_cohort_stats: pd.DataFrame,
    out_path: str,
) -> None:
    """2D binned heatmap: training-DR quintile (x) × validation-DR quintile (y).

    Left panel colour = median |residual|; right panel colour = median
    normalized RMSE. Cell text = mean fraction abstained. Unit = (gene, cohort).
    """
    df = gene_cohort_stats.copy()
    df["train_q"] = _quintile_labels(df["training_dynamic_range"])
    df["val_q"] = _quintile_labels(df["validation_dynamic_range"])
    df = df.dropna(subset=["train_q", "val_q"])
    df["train_q"] = df["train_q"].astype(int)
    df["val_q"] = df["val_q"].astype(int)

    def _grid(value_col: str, aggfunc: str) -> np.ndarray:
        pivot = df.pivot_table(
            index="val_q", columns="train_q", values=value_col, aggfunc=aggfunc,
        )
        pivot = pivot.reindex(index=range(5), columns=range(5))
        return pivot.to_numpy(dtype=np.float64)

    resid_grid = _grid("median_abs_residual", "median")
    nrmse_grid = _grid("normalized_rmse", "median")
    skill_grid = _grid("skill_score", "median")
    frac_grid = _grid("fraction_abstained", "mean")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (grid, title, cmap) in zip(
        axes,
        [(resid_grid, "median |residual| (absolute error)", "magma_r"),
         (nrmse_grid, "median normalized RMSE", "viridis_r"),
         (skill_grid, "median skill score (gene-wise R²)", "viridis")],
    ):
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
        for yi in range(5):
            for xi in range(5):
                frac = frac_grid[yi, xi]
                if np.isfinite(frac):
                    ax.text(xi, yi, f"{frac:.2f}", ha="center", va="center",
                            fontsize=8, color="#111111")
        ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{i+1}" for i in range(5)])
        ax.set_yticks(range(5)); ax.set_yticklabels([f"Q{i+1}" for i in range(5)])
        ax.set_xlabel("training dynamic range quintile")
        ax.set_ylabel("validation dynamic range quintile")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        "CIDriverAnalysis — error/skill by training × validation dynamic range "
        "(cell text = fraction abstained)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_validation_dynamic_performance(
    gene_cohort_stats: pd.DataFrame,
    out_path: str,
    *,
    top_quantile: float = 0.75,
) -> None:
    """Among validation-dynamic genes, relate training DR to skill/confidence.

    Genes in the top quartile of validation dynamic range (within cohort, where
    the skill baseline is stable) are binned by training-DR quintile. Panels
    show median gene-wise skill score, median CI width, and fraction abstained.
    """
    df = gene_cohort_stats.copy()
    df["val_rank"] = df.groupby("cohort")["validation_dynamic_range"].rank(pct=True)
    dynamic = df[df["val_rank"] >= top_quantile].copy()
    if len(dynamic) < 25:
        return
    dynamic["train_q"] = _quintile_labels(dynamic["training_dynamic_range"])
    dynamic = dynamic.dropna(subset=["train_q"])
    dynamic["train_q"] = dynamic["train_q"].astype(int)
    grouped = dynamic.groupby("train_q")
    centers = grouped["training_dynamic_range"].median()

    specs = [
        ("skill_score", "median skill score (1 − MSE/MSE₀)", "#CC3311"),
        ("median_ci_width", "median 90% CI width", "#0077BB"),
        ("fraction_abstained", "fraction abstained", "#EE7733"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (col, ylabel, color) in zip(axes, specs):
        agg = grouped[col].median()
        ax.plot(centers.to_numpy(), agg.to_numpy(), "o-", color=color)
        ax.set_xlabel("training dynamic range (per-gene variance)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.15)
    if "skill_score" in specs[0]:
        axes[0].axhline(0.0, color="grey", ls="--", lw=0.8)
    fig.suptitle(
        f"CIDriverAnalysis — performance among validation-dynamic genes "
        f"(top {int((1-top_quantile)*100)}% validation DR)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fit_interaction_model(
    gene_cohort_stats: pd.DataFrame,
    out_dir: str,
) -> pd.DataFrame:
    """Standardized OLS of error/confidence on training/validation DR + interaction.

    Model: outcome ~ validation_DR + training_DR + training_DR×validation_DR +
    mean expression + |boundary distance| + gene-body coverage + gene length +
    transcript length + exons + GC + CpG + cohort. All continuous terms are
    z-scored, so coefficients are comparable. Tests whether, holding validation
    dynamism fixed, higher training DR lowers normalized error / abstention.
    Returns the coefficient table.
    """
    df = gene_cohort_stats.copy()
    df["abs_boundary_distance"] = df["median_boundary_distance"].abs()
    continuous = [
        ("validation_dynamic_range", "validation DR"),
        ("training_dynamic_range", "training DR"),
        ("validation_mean_expression", "mean expression"),
        ("abs_boundary_distance", "|boundary distance|"),
        ("mean_gb_depth", "gene-body coverage"),
        ("gene_length", "gene length"),
        ("transcript_length", "transcript length"),
        ("num_exons", "num exons"),
        ("gc_promoter", "promoter GC"),
        ("cpg_island", "CpG island"),
    ]

    def _z(col: str) -> np.ndarray:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        mean = np.nanmean(v)
        std = np.nanstd(v)
        std = std if std > 0 else 1.0
        return (v - mean) / std

    z_by_col = {col: _z(col) for col, _ in continuous}
    interaction = z_by_col["training_dynamic_range"] * z_by_col["validation_dynamic_range"]
    cohort_dummy = (df["cohort"].astype("category").cat.codes.to_numpy() > 0).astype(np.float64)

    term_labels = [label for _, label in continuous] + ["training × validation", "cohort"]
    design_cols = [z_by_col[col] for col, _ in continuous] + [interaction, cohort_dummy]
    X = np.column_stack([np.ones(len(df))] + design_cols)

    outcomes = [
        ("normalized_rmse", "normalized error"),
        ("fraction_abstained", "fraction abstained"),
    ]
    coef_records: List[Dict[str, object]] = []
    r2_by_outcome: Dict[str, float] = {}
    for col, label in outcomes:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        y_mean = np.nanmean(y)
        y_std = np.nanstd(y)
        y_std = y_std if y_std > 0 else 1.0
        yz = (y - y_mean) / y_std
        row_ok = np.all(np.isfinite(X), axis=1) & np.isfinite(yz)
        if int(row_ok.sum()) < 30:
            for term in term_labels:
                coef_records.append({"outcome": label, "term": term, "coef": np.nan})
            r2_by_outcome[label] = float("nan")
            continue
        beta, *_ = np.linalg.lstsq(X[row_ok], yz[row_ok], rcond=None)
        fitted = X[row_ok] @ beta
        ss_res = float(np.sum((yz[row_ok] - fitted) ** 2))
        ss_tot = float(np.sum((yz[row_ok] - yz[row_ok].mean()) ** 2))
        r2_by_outcome[label] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        for term, coef in zip(term_labels, beta[1:]):
            coef_records.append({"outcome": label, "term": term, "coef": float(coef)})

    coef_df = pd.DataFrame(coef_records)
    coef_df.to_csv(os.path.join(out_dir, "interaction_model.tsv"), sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    y_pos = np.arange(len(term_labels))
    width = 0.4
    for offset, (col, label) in zip((-width / 2, width / 2), outcomes):
        sub = coef_df[coef_df["outcome"] == label].set_index("term").reindex(term_labels)
        ax.barh(y_pos + offset, sub["coef"].to_numpy(dtype=np.float64), height=width,
                label=f"{label} (R²={r2_by_outcome[label]:.2f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(term_labels, fontsize=9)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Standardized OLS coefficient")
    ax.set_title(
        "CIDriverAnalysis — drivers of error/confidence "
        "(training × validation dynamic range)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "interaction_model.pdf"), bbox_inches="tight")
    plt.close(fig)
    return coef_df


def _plot_coverage_boundary_diagnostics(
    long_df: pd.DataFrame,
    coverage_path: str,
    boundary_path: str,
) -> None:
    """Coverage and on/off-boundary abstention diagnostics (per gene×sample).

    Coverage plot: fraction abstained and median |AOR| across effective tumor
    coverage (TFx × gene-body depth) deciles. Boundary plot: the same versus
    signed distance from the on/off boundary, stratified by coverage tertile —
    abstention should peak near the boundary and at low coverage.
    """
    d = long_df.copy()
    d["abs_aor"] = -d["neg_abs_aor"]

    # --- Effective coverage diagnostic ---
    cov = pd.to_numeric(d["effective_tumor_coverage"], errors="coerce")
    d_cov = d[np.isfinite(cov)].copy()
    d_cov["cov_bin"] = pd.qcut(
        d_cov["effective_tumor_coverage"], 10, labels=False, duplicates="drop",
    )
    grp = d_cov.groupby("cov_bin")
    centers = grp["effective_tumor_coverage"].median()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(centers.to_numpy(), grp["abstain_flag"].mean().to_numpy(), "o-", color="#CC3311")
    axes[0].set_ylabel("fraction abstained")
    axes[1].plot(centers.to_numpy(), grp["abs_aor"].median().to_numpy(), "o-", color="#009988")
    axes[1].set_ylabel("median |AOR|")
    for ax in axes:
        ax.set_xlabel("effective tumor coverage (TFx × gene-body depth)")
        ax.grid(True, alpha=0.15)
    fig.suptitle("CIDriverAnalysis — abstention vs effective tumor coverage", fontsize=12)
    fig.tight_layout()
    fig.savefig(coverage_path, bbox_inches="tight")
    plt.close(fig)

    # --- Boundary-distance diagnostic, stratified by coverage tertile ---
    d_b = d[np.isfinite(d["boundary_distance"]) & np.isfinite(cov)].copy()
    d_b["cov_tertile"] = pd.qcut(
        d_b["effective_tumor_coverage"], 3, labels=["low cov", "med cov", "high cov"],
        duplicates="drop",
    )
    edges = np.linspace(-2.0, 12.0, 15)
    d_b["bd_bin"] = pd.cut(d_b["boundary_distance"], bins=edges)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    tertile_colors = {"low cov": "#CC3311", "med cov": "#EE7733", "high cov": "#0077BB"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for tertile in [t for t in ["low cov", "med cov", "high cov"]
                    if t in d_b["cov_tertile"].cat.categories]:
        sub = d_b[d_b["cov_tertile"] == tertile]
        grouped = sub.groupby("bd_bin", observed=False)
        frac = grouped["abstain_flag"].mean().to_numpy()
        aor = grouped["abs_aor"].median().to_numpy()
        color = tertile_colors[tertile]
        axes[0].plot(bin_centers, frac, "o-", color=color, label=tertile, markersize=4)
        axes[1].plot(bin_centers, aor, "o-", color=color, label=tertile, markersize=4)
    axes[0].set_ylabel("fraction abstained")
    axes[1].set_ylabel("median |AOR|")
    for ax in axes:
        ax.axvline(0.0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("signed distance from on/off boundary (truth − 2.0)")
        ax.grid(True, alpha=0.15)
        ax.legend(title="effective coverage", fontsize=8)
    fig.suptitle("CIDriverAnalysis — abstention vs on/off boundary distance", fontsize=12)
    fig.tight_layout()
    fig.savefig(boundary_path, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Entry point
# =====================================================================

def run_ci_driver_analysis(
    bundles: Dict[str, Dict[str, object]],
    out_dir: str,
    *,
    zarr_base: Optional[str] = None,
    project_root: Optional[Path] = None,
    overwrite: bool = False,
) -> None:
    """Run the CIDriverAnalysis sub-analysis and write outputs to *out_dir*.

    Parameters
    ----------
    bundles : mapping cohort -> dict with keys ``truth_wide``, ``pred_wide``,
        ``ci_lo``, ``ci_hi``, ``p0``, ``pi_gamma``, ``pi_normal`` (all genes x
        samples, restricted to the matched patient set) and ``tfx`` (Series).
    out_dir : output directory (``.../Benchmarking_Patient/CIDriverAnalysis``).
    zarr_base : resolved Zarr base; defaults to ``resolve_zarr_base()``.
    project_root : repository root; defaults to two levels above this file.
    """
    if not bundles:
        print("  [skip] CIDriverAnalysis: no patient bundles available")
        return

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    if zarr_base is None:
        from config.runtime_paths import resolve_zarr_base
        zarr_base = resolve_zarr_base()

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  ── CIDriverAnalysis ({', '.join(bundles)}) ──")

    long_df, gene_cov, sample_cov = _assemble_long_table(bundles, zarr_base, project_root)
    print(
        f"    Pooled long table: {len(long_df)} (gene, sample) rows | "
        f"{long_df['gene'].nunique()} genes × {long_df['sample'].nunique()} samples"
    )

    # Encode crossed grouping factors once for all outcomes.
    sample_cat = long_df["sample"].astype("category")
    gene_cat = long_df["gene"].astype("category")
    sample_codes = sample_cat.cat.codes.to_numpy()
    gene_codes = gene_cat.cat.codes.to_numpy()
    n_samples = len(sample_cat.cat.categories)
    n_genes = len(gene_cat.cat.categories)

    # --- Group 1: variance decomposition + gene random effects ---
    # The abstain flag is excluded from the decomposition (kept as CI/AOR/res).
    outcomes = ["ci_width", "neg_abs_aor", "abs_residual"]
    shares_by_outcome: Dict[str, Dict[str, float]] = {}
    fits: Dict[str, Dict[str, object]] = {}
    for outcome in outcomes:
        shares, fit = _variance_decomposition_for_outcome(
            long_df, outcome, sample_codes, gene_codes, n_samples, n_genes,
        )
        shares_by_outcome[outcome] = shares
        fits[outcome] = fit

    _plot_variance_decomposition(
        shares_by_outcome, os.path.join(out_dir, "variance_decomposition.pdf"),
    )
    pd.DataFrame(shares_by_outcome).T.to_csv(
        os.path.join(out_dir, "variance_decomposition.tsv"), sep="\t",
    )

    # Gene random effects for the uncertainty (CI width) and error (|residual|)
    # models, mapped back to gene names.
    gene_categories = list(gene_cat.cat.categories)
    gene_re = pd.DataFrame(
        {
            "uncertainty_effect": fits["ci_width"]["u_g"],
            "error_effect": fits["abs_residual"]["u_g"],
        },
        index=pd.Index(gene_categories, name="gene"),
    )

    # --- Diagnostics: sample and gene aggregates ---
    sample_stats = _aggregate_sample_stats(long_df, sample_cov)
    gene_stats = _aggregate_gene_stats(long_df)
    # Per (gene, cohort) skill/error metrics (variance-normalized companions to
    # |residual|); reused by the gene, dynamic-range, and interaction plots.
    gene_cohort_stats = _aggregate_gene_cohort_stats(long_df, gene_cov)
    sample_stats.to_csv(os.path.join(out_dir, "sample_diagnostics.tsv"), sep="\t")
    gene_stats.to_csv(os.path.join(out_dir, "gene_diagnostics.tsv"), sep="\t")
    gene_cohort_stats.to_csv(
        os.path.join(out_dir, "gene_cohort_diagnostics.tsv"), sep="\t", index=False,
    )

    _plot_sample_diagnostics(sample_stats, os.path.join(out_dir, "sample_diagnostics.pdf"))
    _plot_gene_diagnostics(
        gene_stats, gene_cohort_stats, os.path.join(out_dir, "gene_diagnostics.pdf"),
    )

    # Gene × sample abstention heatmap.
    dropout = long_df.pivot_table(
        index="gene", columns="sample", values="abstain_flag", aggfunc="mean",
    )
    _plot_dropout_heatmap(
        dropout, sample_stats, gene_stats,
        os.path.join(out_dir, "dropout_heatmap.pdf"),
    )

    # --- Group 2: gene difficulty + drivers ---
    top10 = _plot_gene_difficulty(
        gene_re, os.path.join(out_dir, "gene_difficulty_ranking.pdf"),
    )
    top10.to_csv(os.path.join(out_dir, "gene_difficulty_top10.tsv"), sep="\t")
    gene_re.join(gene_cov, how="left").to_csv(
        os.path.join(out_dir, "gene_random_effects.tsv"), sep="\t",
    )

    coef_df = _plot_feature_importance(
        gene_re, gene_cov, os.path.join(out_dir, "feature_importance.pdf"),
    )
    coef_df.to_csv(os.path.join(out_dir, "feature_importance.tsv"), sep="\t", index=False)

    # Dynamic-range diagnostic (absolute error/CI/AOR + variance-normalized skill).
    _plot_dynamic_range_diagnostic(
        gene_stats, gene_cohort_stats,
        os.path.join(out_dir, "dynamic_range_diagnostic.pdf"),
    )

    # --- v2 diagnostics ---
    # Why are the most-abstained genes hard? (covariate contrast)
    _plot_why_tough_genes(gene_stats, gene_cov, out_dir)

    # Training × validation dynamic range: 2D error grid.
    _plot_dr_2d_heatmap(
        gene_cohort_stats, os.path.join(out_dir, "dynamic_range_2d_heatmap.pdf"),
    )
    # Skill/confidence vs training DR among validation-dynamic genes.
    _plot_validation_dynamic_performance(
        gene_cohort_stats,
        os.path.join(out_dir, "validation_dynamic_performance.pdf"),
    )
    # Interaction model of error/confidence drivers.
    _fit_interaction_model(gene_cohort_stats, out_dir)

    # Effective-coverage and on/off-boundary abstention diagnostics.
    _plot_coverage_boundary_diagnostics(
        long_df,
        os.path.join(out_dir, "effective_coverage_diagnostic.pdf"),
        os.path.join(out_dir, "boundary_distance_diagnostic.pdf"),
    )

    print("    CIDriverAnalysis complete")
