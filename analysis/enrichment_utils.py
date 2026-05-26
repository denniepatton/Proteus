"""Shared enrichment and ssGSEA helpers for analysis workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.analysis_utils import _augment_missing_gene_aliases


def require_existing_path(path: Path, *, description: str) -> Path:
    """Return *path* when it exists, otherwise raise a precise file error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required {description}: {path}")
    return path


def load_gmt(path: Path, *, description: str) -> Dict[str, List[str]]:
    """Parse a GMT file into a mapping from set name to gene symbols."""
    require_existing_path(path, description=description)

    genesets: Dict[str, List[str]] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genesets[parts[0]] = [gene_name for gene_name in parts[2:] if gene_name]
    return genesets


def prepare_expression_matrix_for_scores(
    expr_df: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Apply targeted aliasing and drop NaN-bearing rows before pathway scoring."""
    del context

    prepared = _augment_missing_gene_aliases(expr_df)
    if prepared.empty:
        return prepared

    invalid_rows = prepared.isna().any(axis=1)
    if bool(invalid_rows.any()):
        prepared = prepared.loc[~invalid_rows]
    return prepared


def build_entrez_to_symbol(
    mane_reference_path: Path,
    *,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build an Entrez ID to gene symbol map from the MANE reference table."""
    require_existing_path(mane_reference_path, description="MANE universal reference table")

    mapping: Dict[str, str] = {}
    with open(mane_reference_path) as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            gene_id_raw = parts[0]
            symbol = parts[3] or (parts[14] if len(parts) > 14 else "")
            entrez = gene_id_raw.replace("GeneID:", "")
            if entrez and symbol:
                mapping[entrez] = symbol
    if overrides:
        mapping.update(overrides)
    return mapping


def convert_entrez_gmt(
    genesets: Dict[str, List[str]],
    *,
    mane_reference_path: Path,
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """Convert Entrez-ID gene sets to gene-symbol gene sets via MANE."""
    entrez_to_symbol = build_entrez_to_symbol(
        mane_reference_path,
        overrides=overrides,
    )
    converted: Dict[str, List[str]] = {}
    for name, entrez_ids in genesets.items():
        symbols: List[str] = []
        unmapped: List[str] = []
        for entrez_id in entrez_ids:
            symbol = entrez_to_symbol.get(entrez_id)
            if symbol:
                symbols.append(symbol)
            else:
                unmapped.append(entrez_id)
        if unmapped:
            suffix = "..." if len(unmapped) > 5 else ""
            print(
                f"    GSEA: {name}: {len(unmapped)} unmapped Entrez IDs: "
                f"{unmapped[:5]}{suffix}"
            )
        converted[name] = symbols
    return converted


def compute_ssgsea_score_matrix(
    expr_df: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    *,
    min_size: int,
    max_size: int = 500,
) -> pd.DataFrame:
    """Compute a per-sample ssGSEA score matrix for the supplied gene sets."""
    import gseapy as gp

    if expr_df.empty or not gene_sets:
        return pd.DataFrame()

    prepared_expr = prepare_expression_matrix_for_scores(
        expr_df,
        context="ssGSEA score matrix",
    )
    if prepared_expr.empty:
        return pd.DataFrame()

    result = gp.ssgsea(
        data=prepared_expr,
        gene_sets=gene_sets,
        outdir=None,
        min_size=min_size,
        max_size=max_size,
        no_plot=True,
        verbose=False,
    )
    if getattr(result, "res2d", None) is None or result.res2d.empty:
        return pd.DataFrame()
    return result.res2d.pivot(index="Term", columns="Name", values="NES").apply(
        pd.to_numeric,
        errors="coerce",
    )


def aggregate_ssgsea_score_matrices(
    draws: List[pd.DataFrame],
    gene_sets: Dict[str, List[str]],
    *,
    min_size: int,
    max_size: int = 500,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Aggregate ssGSEA score matrices across Monte Carlo draws."""
    all_scores: List[pd.DataFrame] = []
    for draw in draws:
        draw_scores = compute_ssgsea_score_matrix(
            draw,
            gene_sets,
            min_size=min_size,
            max_size=max_size,
        )
        if not draw_scores.empty:
            all_scores.append(draw_scores)

    if not all_scores:
        return None, None, None

    template = all_scores[0]
    stacked = np.stack(
        [
            score.reindex(index=template.index, columns=template.columns).values
            for score in all_scores
        ],
        axis=0,
    )
    mean_scores = pd.DataFrame(
        np.mean(stacked, axis=0),
        index=template.index,
        columns=template.columns,
    )
    ci_lo = pd.DataFrame(
        np.percentile(stacked, 5.0, axis=0),
        index=template.index,
        columns=template.columns,
    )
    ci_hi = pd.DataFrame(
        np.percentile(stacked, 95.0, axis=0),
        index=template.index,
        columns=template.columns,
    )
    return mean_scores, ci_lo, ci_hi


__all__ = [
    "aggregate_ssgsea_score_matrices",
    "build_entrez_to_symbol",
    "compute_ssgsea_score_matrix",
    "convert_entrez_gmt",
    "load_gmt",
    "prepare_expression_matrix_for_scores",
    "require_existing_path",
]