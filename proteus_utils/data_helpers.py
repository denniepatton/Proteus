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
import os
import re
from typing import Callable, Dict, List, Optional, Set, Tuple

# Pre-compiled PHT regex: PHT-Fold-<N>_<PDX_NAME>-BP<N>
_PHT_RE = re.compile(r'^PHT-Fold-\d+_(.+)-BP\d+')


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
