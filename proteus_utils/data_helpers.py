"""Shared data-path and sample-mapping helpers used by Proteus.py and proteus_utils.

This module centralises three patterns that were previously copy-pasted
across multiple call sites:

1. **Zarr path construction** — stripping the cohort prefix from TSV
   sample names to build bare-name Zarr file paths.
2. **PHT→source resolution** — regex-matching PHT sample names to their
   parent PDX and mapping through ``source_groups`` to a source label.
3. **Source-group mapping** — inverting ``source_groups`` and building
   sample→source / zarr_path→source dictionaries.
"""

from __future__ import annotations

import csv
import math
import os
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

# Pre-compiled PHT regex: PHT-Fold-<N>_<PDX_NAME>-BP<N>
_PHT_RE = re.compile(r'^PHT-Fold-\d+_(.+)-BP\d+')


# Fixed PLCAv2 selection weights.  These are intentionally hard-coded so
# the weighted objective stays stable across trainer, evaluation, and tuning.
PLCA_WEIGHTED_SOURCE_WEIGHTS: Dict[str, float] = {
    'CRPC-RA': 0.30,
    'SCLC': 0.30,
    'Prostate-Synthetic': 0.20,
    'SCLC-Synthetic': 0.20,
}

# Fixed PLCAv4 cohort-objective constants requested for the fast sweep.
# These are intentionally not exposed as new experiment knobs.
COHORT_OBJECTIVE_ALPHA: float = 2.0
COHORT_OBJECTIVE_ETA: float = 1.05

# PLCAv4-2 continuous objective constants.  The CRPS reference reflects the
# observed regime where good fold/source GVW-CRPS is approximately <= 1.0.
PLCA_V4_2_CRPS_REFERENCE: float = 1.0
PLCA_V4_2_CRPS_LOG_WEIGHT: float = 0.30
PLCA_V4_2_CRPS_EXCESS_WEIGHT: float = 1.00
PLCA_V4_2_CRPS_SOFTNESS: float = 0.03

# PLCA soft-objective constants. This objective operates directly on the
# four source-bucket GVW-CRPS and GVW-r values using an unnormalized
# soft maximum (tau * log-sum-exp).
PLCA_SOFT_OBJECTIVE_TAU: float = 0.10
PLCA_SOFT_OBJECTIVE_ONE_MINUS_R_WEIGHT: float = 2.0
PLCA_SOFT_OBJECTIVE_OFFSET: float = 2.0


# ---------------------------------------------------------------------------
# Zarr path primitives
# ---------------------------------------------------------------------------

def zarr_name_for_sample(sample: str, cohort: str) -> str:
    """Strip the cohort prefix from a TSV sample name.

    On disk, Zarr stores use the bare name (e.g. ``00-147.zarr``), but
    the TSV carries the full name (e.g. ``CRPC-RA_00-147``).
    """
    if sample.startswith(f"{cohort}_"):
        return sample[len(cohort) + 1:]
    return sample


def zarr_path_for_sample(sample: str, cohort: str, zarr_base: str) -> str:
    """Build the on-disk Zarr path for a TSV sample."""
    bare = zarr_name_for_sample(sample, cohort)
    return os.path.join(zarr_base, cohort, 'zarr_files', f"{bare}.zarr")


# ---------------------------------------------------------------------------
# PHT resolution
# ---------------------------------------------------------------------------

def resolve_pht_source(
    name: str,
    pdx_sample_to_cohort: Dict[str, str],
    cohort_to_source: Dict[str, str],
) -> str:
    """Resolve a PHT sample name to its source-group label.

    Args:
        name: Either the TSV sample name (e.g. ``PHT-Fold-1_00-147-BP3``)
            or the bare Zarr basename (same string without ``.zarr``).
        pdx_sample_to_cohort: Map from bare PDX name → cohort.
        cohort_to_source: Map from cohort → source label.

    Returns:
        Source label string.

    Raises:
        ValueError: If the name cannot be parsed or the parent PDX
            is not found.
    """
    m = _PHT_RE.match(name)
    if not m:
        raise ValueError(
            f"Could not parse PDX name from PHT sample: '{name}'. "
            f"Expected format: PHT-Fold-X_<PDX_NAME>-BP..."
        )
    pdx_name = m.group(1)
    pdx_coh = pdx_sample_to_cohort.get(pdx_name)
    if pdx_coh is None:
        raise ValueError(
            f"PHT sample '{name}' has PDX name '{pdx_name}' "
            f"that does not match any known PDX sample. "
            f"Known: {sorted(pdx_sample_to_cohort.keys())}"
        )
    return cohort_to_source[pdx_coh]


# ---------------------------------------------------------------------------
# Source-group mapping from a splits TSV
# ---------------------------------------------------------------------------

def invert_source_groups(source_groups: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert ``{source: [cohort, ...]}`` → ``{cohort: source}``."""
    cohort_to_source: Dict[str, str] = {}
    for src, cohorts in source_groups.items():
        for coh in cohorts:
            cohort_to_source[coh] = src
    return cohort_to_source


def collect_pdx_names(
    training_set_tsv: str,
    cohort_to_source: Dict[str, str],
) -> Dict[str, str]:
    """Read the splits TSV and return ``{bare_pdx_name: cohort}`` for all
    cohorts that appear in *cohort_to_source*."""
    pdx_sample_to_cohort: Dict[str, str] = {}
    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            coh = row['Cohort'].strip()
            samp = row['Sample'].strip()
            bare = zarr_name_for_sample(samp, coh)
            if coh in cohort_to_source:
                pdx_sample_to_cohort[bare] = coh
    return pdx_sample_to_cohort


def collect_zarr_paths(
    training_set_tsv: str,
    zarr_base: str,
    exclude_fold: Optional[int] = None,
    exclude_cohorts: Optional[Set[str]] = None,
) -> List[str]:
    """Collect existing Zarr paths from a splits TSV.

    Args:
        training_set_tsv: TSV with Cohort/Sample/Fold columns.
        zarr_base: Root directory for Zarr stores.
        exclude_fold: Skip samples whose Fold equals this value.
        exclude_cohorts: Skip samples whose Cohort is in this set.
    """
    paths: List[str] = []
    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if exclude_fold is not None and int(row['Fold']) == exclude_fold:
                continue
            cohort = row['Cohort'].strip()
            if exclude_cohorts and cohort in exclude_cohorts:
                continue
            sample = row['Sample'].strip()
            zpath = zarr_path_for_sample(sample, cohort, zarr_base)
            if os.path.isdir(zpath):
                paths.append(zpath)
    return paths


def get_plca_selection_group_weights(
    source_groups: Optional[Dict[str, List[str]]],
) -> Optional[Dict[str, float]]:
    """Return the fixed PLCAv2 source weights when the source groups match.

    The PLCAv2 weighted objective is only defined for the four-bucket
    patient/synthetic grouping.  Other source-group layouts keep the usual
    balanced metrics but do not get a weighted selection metric.
    """
    if source_groups is None:
        return None
    if set(source_groups) != set(PLCA_WEIGHTED_SOURCE_WEIGHTS):
        return None
    return dict(PLCA_WEIGHTED_SOURCE_WEIGHTS)


def validate_plca_selection_groups(source_groups: Dict[str, List[str]]) -> None:
    """Fail loudly if a weighted PLCAv2 run does not use the fixed 4 buckets."""
    expected = set(PLCA_WEIGHTED_SOURCE_WEIGHTS)
    actual = set(source_groups)
    if actual != expected:
        raise ValueError(
            "Weighted PLCA selection requires source_groups with exactly "
            f"these keys: {sorted(expected)}. Got: {sorted(actual)}"
        )


def weighted_group_mean(
    group_values: Dict[str, float],
    group_weights: Dict[str, float],
) -> float:
    """Compute a weighted mean over named groups.

    Returns NaN when any required group is missing or non-finite so the
    caller can fail loudly instead of silently renormalizing the objective.
    """
    missing = [
        name for name in group_weights
        if name not in group_values or not math.isfinite(float(group_values[name]))
    ]
    if missing:
        return float('nan')

    total_weight = sum(float(w) for w in group_weights.values())
    if total_weight <= 0.0:
        return float('nan')

    weighted_sum = sum(
        float(group_values[name]) * float(weight)
        for name, weight in group_weights.items()
    )
    return float(weighted_sum / total_weight)


def compute_cohort_objective(
    cohort_r_balanced: float,
    cohort_crps_balanced: float,
    best_observed_cohort_crps_balanced: float,
) -> Tuple[float, float, float]:
    """Return the gated PLCAv4 cohort objective, gate, and CRPS penalty.

    Objective:
        cohort_r_balanced - alpha * max(0, cohort_crps_balanced / crps_gate - 1)

    where:
        crps_gate = min(1.0, eta * best_observed_cohort_crps_balanced)
    """
    r_balanced = float(cohort_r_balanced)
    crps_balanced = float(cohort_crps_balanced)
    best_crps = float(best_observed_cohort_crps_balanced)
    if not (
        math.isfinite(r_balanced)
        and math.isfinite(crps_balanced)
        and math.isfinite(best_crps)
    ):
        return float('nan'), float('nan'), float('nan')

    crps_gate = min(1.0, COHORT_OBJECTIVE_ETA * best_crps)
    if crps_gate <= 0.0:
        return float('nan'), float('nan'), float('nan')

    crps_penalty = max(0.0, crps_balanced / crps_gate - 1.0)
    objective = r_balanced - COHORT_OBJECTIVE_ALPHA * crps_penalty
    return float(objective), float(crps_gate), float(crps_penalty)


def _stable_softplus(value: float) -> float:
    if value > 40.0:
        return float(value)
    if value < -40.0:
        return float(math.exp(value))
    return float(math.log1p(math.exp(value)))


def _stable_logsumexp(values: List[float]) -> float:
    if not values:
        return float('nan')
    max_value = max(values)
    if not math.isfinite(max_value):
        return float('nan')
    return float(
        max_value + math.log(sum(math.exp(value - max_value) for value in values))
    )


def _soft_logsumexp_max(
    group_values: Dict[str, float],
    required_names: Optional[List[str]] = None,
    tau: float = PLCA_SOFT_OBJECTIVE_TAU,
) -> float:
    if tau <= 0.0 or not math.isfinite(tau):
        return float('nan')

    names = required_names if required_names is not None else sorted(group_values)
    if not names:
        return float('nan')

    scaled_values: List[float] = []
    for name in names:
        raw_value = group_values.get(name)
        if raw_value is None or not math.isfinite(float(raw_value)):
            return float('nan')
        scaled_values.append(float(raw_value) / tau)

    return float(tau * _stable_logsumexp(scaled_values))


def compute_plcav4_2_objective(
    cohort_r_balanced: float,
    cohort_crps_balanced: float,
) -> Tuple[float, float, float]:
    """Return the continuous PLCAv4-2 robust cohort objective.

    ``cohort_r_balanced`` is the fixed-source weighted GVW-r minus half the
    source standard deviation.  ``cohort_crps_balanced`` is the fixed-source
    weighted GVW-CRPS plus half the source standard deviation.  The objective
    rewards robust r, rewards lower robust CRPS continuously via log-ratio,
    and applies a smooth centered excess penalty around CRPS=1.0.
    """
    robust_r = float(cohort_r_balanced)
    robust_crps = float(cohort_crps_balanced)
    if not (math.isfinite(robust_r) and math.isfinite(robust_crps)):
        return float('nan'), float('nan'), float('nan')
    if robust_crps <= 0.0:
        return float('nan'), float('nan'), float('nan')

    crps_ratio = robust_crps / PLCA_V4_2_CRPS_REFERENCE
    crps_log_ratio = math.log(crps_ratio)
    excess_scaled = (
        robust_crps - PLCA_V4_2_CRPS_REFERENCE
    ) / PLCA_V4_2_CRPS_SOFTNESS
    centered_smooth_excess = PLCA_V4_2_CRPS_SOFTNESS * (
        _stable_softplus(excess_scaled) - math.log(2.0)
    )
    objective = (
        robust_r
        - PLCA_V4_2_CRPS_LOG_WEIGHT * crps_log_ratio
        - PLCA_V4_2_CRPS_EXCESS_WEIGHT * centered_smooth_excess
    )
    return float(objective), float(crps_log_ratio), float(centered_smooth_excess)


def compute_soft_objective(
    cohort_r_values: Dict[str, float],
    cohort_crps_values: Dict[str, float],
    required_names: Optional[List[str]] = None,
) -> Tuple[float, float, float]:
    """Return the source-level soft objective.

    Objective:
        S_tau(GVW-CRPS_i) + 2 * S_tau(1 - GVW-r_i) - 2

    where ``S_tau`` is the unnormalized soft maximum ``tau * log-sum-exp``
    across the source buckets.
    """
    names = required_names if required_names is not None else sorted(cohort_crps_values)
    if not names or set(names) != set(cohort_r_values) or set(names) != set(cohort_crps_values):
        return float('nan'), float('nan'), float('nan')

    soft_crps = _soft_logsumexp_max(cohort_crps_values, names)
    one_minus_r = {
        name: 1.0 - float(cohort_r_values[name])
        for name in names
    }
    soft_one_minus_r = _soft_logsumexp_max(one_minus_r, names)
    if not (math.isfinite(soft_crps) and math.isfinite(soft_one_minus_r)):
        return float('nan'), float('nan'), float('nan')

    objective = (
        soft_crps
        + PLCA_SOFT_OBJECTIVE_ONE_MINUS_R_WEIGHT * soft_one_minus_r
        - PLCA_SOFT_OBJECTIVE_OFFSET
    )
    return float(objective), float(soft_crps), float(soft_one_minus_r)


# ---------------------------------------------------------------------------
# Source-map construction from (key, sample_name, cohort) triples
# ---------------------------------------------------------------------------

def assign_source(
    cohort: str,
    sample_name: str,
    cohort_to_source: Dict[str, str],
    pdx_sample_to_cohort: Dict[str, str],
) -> Optional[str]:
    """Return the source label for *sample_name* in *cohort*, or ``None``
    if the cohort is not part of any source group.

    PHT samples are resolved through their parent PDX.
    """
    if cohort in cohort_to_source:
        return cohort_to_source[cohort]
    if cohort == 'PHT':
        return resolve_pht_source(sample_name, pdx_sample_to_cohort,
                                  cohort_to_source)
    return None


def build_source_map_from_tsv(
    training_set_tsv: str,
    source_groups: Dict[str, List[str]],
    zarr_base: str,
    exclude_fold: Optional[int] = None,
) -> Tuple[
    Dict[str, str],             # sample_to_source
    Dict[str, List[str]],       # source_variance_paths (non-PHT only)
]:
    """Read the splits TSV and build sample->source + per-source zarr paths.

    Returns:
        sample_to_source: ``{TSV_sample_name: source_label}``.
        source_variance_paths: ``{source: [zarr_path, ...]}`` containing
            only non-PHT samples (PHT duplicates their parent PDX labels
            and must be excluded from variance computation).
    """
    cohort_to_source = invert_source_groups(source_groups)
    pdx_names = collect_pdx_names(training_set_tsv, cohort_to_source)

    sample_to_source: Dict[str, str] = {}
    source_variance_paths: Dict[str, List[str]] = {s: [] for s in source_groups}

    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if exclude_fold is not None and int(row['Fold']) == exclude_fold:
                continue
            cohort = row['Cohort'].strip()
            sample = row['Sample'].strip()
            source = assign_source(cohort, sample, cohort_to_source,
                                   pdx_names)
            if source is None:
                continue
            sample_to_source[sample] = source
            zpath = zarr_path_for_sample(sample, cohort, zarr_base)
            if os.path.isdir(zpath) and cohort != 'PHT':
                source_variance_paths[source].append(zpath)

    return sample_to_source, source_variance_paths


# ---------------------------------------------------------------------------
# Per-source variance computation
# ---------------------------------------------------------------------------

def compute_source_variance(
    source_groups: Dict[str, List[str]],
    source_variance_paths: Dict[str, List[str]],
    compute_fn: Callable[[List[str], str], Dict[str, float]],
    label_key: str = 'flat/tumor_exp',
) -> Dict[str, Dict[str, float]]:
    """Compute per-source gene variance maps and log summary statistics.

    Args:
        source_groups: ``{source: [cohort, ...]}`` -- only used for
            iteration order.
        source_variance_paths: ``{source: [zarr_path, ...]}`` with
            non-PHT paths only.
        compute_fn: Typically ``compute_gene_variances_unclipped``.
        label_key: Zarr array key for the label data.
    """
    print('Computing per-source gene variance for GVW evaluation...')
    result: Dict[str, Dict[str, float]] = {}
    for src_name in sorted(source_groups):
        src_paths = source_variance_paths[src_name]
        if src_paths:
            result[src_name] = compute_fn(src_paths, label_key)
            vals = list(result[src_name].values())
            print(
                f'  Source "{src_name}": {len(vals)} genes, '
                f'variance range [{min(vals):.3f}, {max(vals):.3f}], '
                f'{len(src_paths)} samples'
            )
        else:
            print(f'  Source "{src_name}": no zarr paths found -- skipping')
    return result
