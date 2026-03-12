"""
plot_utils.py
Proteus analysis framework — all plotting functions.

Every public function saves a PDF (+ closes the figure) and returns None
unless documented otherwise.  Style defaults: Type-42 fonts, font.size 12,
palette from reference/palette.tsv.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from .analysis_utils import (
    _pearson_r,
    _spearman_rho,
    compute_gene_variance,
    format_pval,
    load_palette,
    pearson_with_ci,
    spearman_with_ci,
    VARIANCE_BINS,
)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.dpi"] = 150

_PALETTE: Optional[Dict[str, str]] = None


def _palette() -> Dict[str, str]:
    global _PALETTE
    if _PALETTE is None:
        _PALETTE = load_palette()
    return _PALETTE


def _color(subtype: str) -> str:
    return _palette().get(subtype, "#999999")


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. (removed — density_scatter)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. (removed — scatter_by_gene_set)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Variance-group boxplot
# ---------------------------------------------------------------------------

def variance_group_boxplot(
    gene_metrics: pd.DataFrame,
    var_map: Dict[str, float],
    name: str,
    *,
    metric: str = "pearson_r",
    title: Optional[str] = None,
) -> None:
    """Box + strip plot of per-gene metric split by variance group."""
    _ensure_dir(name)
    df = gene_metrics.copy()
    df["variance"] = df.index.map(lambda g: var_map.get(g, 0.0))

    def _bin(v: float) -> str:
        for grp, (lo, hi) in VARIANCE_BINS.items():
            if lo <= v < hi:
                return grp
        return "high"

    df["group"] = df["variance"].apply(_bin)
    order = ["low", "medium", "high"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        x="group", y=metric, data=df, order=order, hue="group",
        ax=ax, palette="Set2", legend=False, inner="quart", cut=0,
    )

    counts = df["group"].value_counts()
    labels = [f"{g}\n(n={counts.get(g, 0):,})" for g in order]
    tick_positions = list(range(len(order)))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Variance Group")
    ax.set_title(title or f"Per-gene {metric} by variance group")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Sample-level scatter (single sample: genes as dots)
# ---------------------------------------------------------------------------

def sample_scatter(
    truth_col: pd.Series,
    pred_col: pd.Series,
    sample_name: str,
    name: str,
    *,
    ci_lo: Optional[pd.Series] = None,
    ci_hi: Optional[pd.Series] = None,
    goi: Optional[Dict[str, List[str]]] = None,
    tfx: Optional[float] = None,
    depth: Optional[float] = None,
) -> None:
    """Scatter for one sample: genes as dots with hexbin density background.

    GOI genes are highlighted and colored by subtype (from goi dict keys).
    ``goi`` maps subtype name → list of gene names.
    """
    _ensure_dir(name)

    common = truth_col.index.intersection(pred_col.index)
    t = truth_col.loc[common].astype(np.float64)
    p = pred_col.loc[common].astype(np.float64)
    mask = np.isfinite(t) & np.isfinite(p)
    t, p = t[mask], p[mask]

    if len(t) < 3:
        return

    r_val, p_val = pearsonr(t.values, p.values)
    rho_val, _ = spearmanr(t.values, p.values)
    rmse = float(np.sqrt(np.mean((t.values - p.values) ** 2)))
    mae = float(np.mean(np.abs(t.values - p.values)))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Hexbin density background (matches legacy scatter_plot)
    axis_max = max(t.max(), p.max()) * 1.03
    hb = ax.hexbin(
        t.values, p.values, gridsize=100, cmap="viridis_r", bins="log",
        extent=(0, axis_max, 0, axis_max),
    )
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, label=r"$\log_{10}(N+1)$")

    # GOI highlights colored by subtype with CI error bars
    pal = _palette()
    if goi:
        for subtype, genes_in_subtype in goi.items():
            present = [g for g in genes_in_subtype if g in t.index]
            if not present:
                continue
            gt = t.loc[present]
            gp = p.loc[present]

            # CI error bars (drawn once per subtype group)
            if ci_lo is not None and ci_hi is not None:
                ci_l = ci_lo.reindex(present).astype(float)
                ci_h = ci_hi.reindex(present).astype(float)
                yerr_lo = (gp - ci_l).clip(lower=0).values
                yerr_hi = (ci_h - gp).clip(lower=0).values
                yerr = np.vstack([yerr_lo, yerr_hi])
                ax.errorbar(
                    gt.values, gp.values, yerr=yerr,
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=2,
                )

            color = pal.get(subtype, "#999999")
            ax.scatter(
                gt.values, gp.values,
                s=300, edgecolors="black", linewidths=0.5, color=color, zorder=3,
            )
            for g in present:
                ax.annotate(
                    g, (gt[g], gp[g]),
                    xytext=(5, 5), textcoords="offset points", fontsize=12,
                    color=color,
                    path_effects=[PathEffects.withStroke(linewidth=0.75, foreground="black")],
                    zorder=4,
                )

    ax.plot([0, axis_max], [0, axis_max], "k--", alpha=0.5, lw=1)
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"RNA-Seq log$_2$(CPM+1)")
    ax.set_ylabel(r"Predicted log$_2$(CPM+1)")

    info_lines = [sample_name]
    if tfx is not None:
        info_lines[0] += f"  TFx={tfx:.2f}"
    if depth is not None:
        info_lines[0] += f"  depth={depth:.1f}x"
    info_lines.append(f"n={len(t):,}  r={r_val:.4f}  ρ={rho_val:.4f}  {format_pval(p_val)}")
    info_lines.append(f"RMSE={rmse:.3f}  MAE={mae:.3f}")
    ax.set_title("\n".join(info_lines), fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Gene target scatter (samples as dots, colored by subtype)
# ---------------------------------------------------------------------------

def gene_target_scatter(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    gene: str,
    name: str,
    *,
    subtypes_df: Optional[pd.DataFrame] = None,
    tfx_dict: Optional[Dict[str, float]] = None,
) -> None:
    """Scatter: samples as dots for one gene, truth vs pred.

    Colored by subtype, sized by TFx.  Fit line + 95% CI band.
    """
    _ensure_dir(name)
    if gene not in truth_wide.index or gene not in pred_wide.index:
        return

    common = truth_wide.columns.intersection(pred_wide.columns)
    t = truth_wide.loc[gene, common].astype(np.float64)
    p = pred_wide.loc[gene, common].astype(np.float64)
    mask = np.isfinite(t) & np.isfinite(p)
    t, p = t[mask], p[mask]

    if len(t) < 3:
        return

    r_val, p_val = pearsonr(t.values, p.values)
    rmse = float(np.sqrt(np.mean((t.values - p.values) ** 2)))

    fig, ax = plt.subplots(figsize=(6, 6))

    # Fit line
    coeffs = np.polyfit(t.values, p.values, 1)
    fit_fn = np.poly1d(coeffs)
    x_line = np.linspace(t.min(), t.max(), 100)
    ax.plot(x_line, fit_fn(x_line), "k-", lw=1, zorder=1)

    # CI band
    n = len(t)
    if n > 3:
        from scipy.stats import t as t_dist
        x_vals = t.values
        residuals = p.values - fit_fn(x_vals)
        s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        mean_x = np.mean(x_vals)
        Sxx = np.sum((x_vals - mean_x) ** 2)
        t_val = t_dist.ppf(0.975, n - 2)

        y_line = fit_fn(x_line)
        ci = t_val * s_err * np.sqrt(1 / n + (x_line - mean_x) ** 2 / Sxx)
        ax.fill_between(x_line, y_line - ci, y_line + ci, color="gray", alpha=0.25, zorder=0)

    # Plot points
    pal = _palette()
    if subtypes_df is not None:
        seen = set()
        for sample in t.index:
            sz = 200
            if tfx_dict is not None:
                sz = max(30, tfx_dict.get(sample, 1.0) * 300)
            color = "#999999"
            marker = "o"
            if sample in subtypes_df.index:
                sub = subtypes_df.loc[sample, "subtype"] if "subtype" in subtypes_df.columns else "Unknown"
                color = pal.get(sub, "#999999")
                seen.add(sub)
                if "status" in subtypes_df.columns:
                    status = subtypes_df.loc[sample, "status"]
                    marker = "^" if status == "holdout" else "o"
            ax.scatter(
                t[sample], p[sample], s=sz, c=color, marker=marker,
                edgecolors="black", linewidths=0.5, alpha=0.75, zorder=2,
            )
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=pal.get(s, "#999999"), markersize=8, label=s)
            for s in sorted(seen)
        ]
        if handles:
            ax.legend(handles=handles, title="Subtype", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        sizes = [200] * len(t)
        if tfx_dict is not None:
            sizes = [max(30, tfx_dict.get(s, 1.0) * 300) for s in t.index]
        ax.scatter(
            t.values, p.values, s=sizes,
            edgecolors="black", linewidths=0.5, alpha=0.75, zorder=2,
        )

    ax.set_xlabel(r"RNA-Seq log$_2$(CPM+1)")
    ax.set_ylabel(r"Predicted log$_2$(CPM+1)")
    ax.set_title(gene)
    ax.text(
        0.05, 0.95,
        f"Pearson r = {r_val:.3f}\n{format_pval(p_val)}\nRMSE = {rmse:.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
    )
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Double heatmap (true vs pred side-by-side)
# ---------------------------------------------------------------------------

def double_heatmap(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    genes: List[str],
    name: str,
    *,
    subtypes: Optional[pd.DataFrame] = None,
    breaks: Optional[List[int]] = None,
    gsea: bool = False,
) -> None:
    """Side-by-side heatmaps: truth (left) and predictions (right)."""
    _ensure_dir(name)
    tw = truth_wide.apply(pd.to_numeric, errors="coerce")
    pw = pred_wide.apply(pd.to_numeric, errors="coerce")

    common_samples = tw.columns.intersection(pw.columns)
    valid_genes = [g for g in genes if g in tw.index or g in pw.index]
    if not valid_genes or len(common_samples) == 0:
        return

    # Ensure rows exist
    for g in valid_genes:
        if g not in tw.index:
            tw.loc[g] = np.nan
        if g not in pw.index:
            pw.loc[g] = np.nan

    # Sort samples by subtype then descending TFX (matches legacy plot_double_heatmap)
    sorted_samples = sorted(common_samples)
    if subtypes is not None:
        shared = subtypes.index.intersection(common_samples)
        if len(shared) > 0:
            sub_dict = subtypes["subtype"].to_dict() if "subtype" in subtypes.columns else {}
            tfx_dict = subtypes["tfx"].to_dict() if "tfx" in subtypes.columns else {}
            sorted_samples = sorted(
                shared,
                key=lambda s: (sub_dict.get(s, ""), -tfx_dict.get(s, 0)),
            )

    tw = tw.loc[valid_genes, sorted_samples]
    pw = pw.loc[valid_genes, sorted_samples]

    cmap = plt.get_cmap("inferno")
    cmap.set_bad("lightgray")

    if gsea:
        vmin, vmax = -1.0, 1.0
    else:
        all_vals = pd.concat([tw.stack(), pw.stack()]).dropna()
        vmin = float(all_vals.min()) if len(all_vals) else 0
        vmax = float(all_vals.max()) if len(all_vals) else 1

    n_genes = len(valid_genes)
    n_samp = len(sorted_samples)
    cell = 0.25
    fig_w = max(10, n_samp * cell * 2 + 3)
    fig_h = max(6, n_genes * cell + 2.5)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.02},
        constrained_layout=True,
    )

    sns.heatmap(tw, cmap=cmap, linewidths=0.1, linecolor="black",
                ax=ax1, vmin=vmin, vmax=vmax, cbar=False, square=False)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("True", fontsize=10)

    sns.heatmap(pw, cmap=cmap, linewidths=0.1, linecolor="black",
                ax=ax2, vmin=vmin, vmax=vmax, cbar=True, yticklabels=False, square=False,
                cbar_kws={"shrink": 0.6, "pad": 0.04})
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Predicted", fontsize=10)

    if ax2.collections:
        cb = ax2.collections[0].colorbar
        if cb:
            cb.ax.tick_params(labelsize=7)
            cb.set_label("GSVA Score" if gsea else r"log$_2$(CPM+1)", size=8)

    if breaks:
        for b in breaks:
            if 0 < b < n_genes:
                ax1.axhline(b, color="white", linewidth=2)
                ax2.axhline(b, color="white", linewidth=2)

    for ax in (ax1, ax2):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=max(4, min(10, 200 // n_samp)))

    ax1.set_yticks(np.arange(n_genes) + 0.5)
    ax1.set_yticklabels(valid_genes, rotation=0, fontsize=max(4, min(10, 200 // n_genes)))

    fig.savefig(f"{name}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Single heatmap with TFx bar
# ---------------------------------------------------------------------------

def single_heatmap(
    df: pd.DataFrame,
    genes: List[str],
    name: str,
    *,
    tfx_dict: Optional[Dict[str, float]] = None,
    sub_dict: Optional[Dict[str, str]] = None,
    gsea: bool = False,
) -> None:
    """Single heatmap with optional TFx bar on top."""
    _ensure_dir(name)
    df = df.apply(pd.to_numeric, errors="coerce")
    valid_genes = [g for g in genes if g in df.index]
    if not valid_genes:
        return
    df = df.loc[valid_genes].dropna(axis=1, how="all")
    if df.empty:
        return

    # Sort samples
    if sub_dict is not None:
        common = [c for c in df.columns if c in sub_dict]
        df = df[sorted(common, key=lambda s: (sub_dict.get(s, ""), -(tfx_dict or {}).get(s, 0)))]
    elif tfx_dict is not None:
        df = df[sorted(df.columns, key=lambda s: tfx_dict.get(s, 0), reverse=True)]

    n_genes, n_samp = len(valid_genes), len(df.columns)
    has_tfx_bar = tfx_dict is not None
    fig_w = max(10, n_samp * 0.22 + 2)
    fig_h = max(5, n_genes * 0.4 + (1.5 if has_tfx_bar else 0.5))

    cmap = plt.get_cmap("inferno")
    cmap.set_bad("lightgray")

    fig = plt.figure(figsize=(fig_w, fig_h))
    if has_tfx_bar:
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, max(10, n_genes)])
        ax_bar = fig.add_subplot(gs[0])
        ax_hm = fig.add_subplot(gs[1])
    else:
        ax_hm = fig.add_subplot(111)

    vmin = -1.0 if gsea else 0
    vmax = 1.0 if gsea else None
    sns.heatmap(
        df, cmap=cmap, ax=ax_hm, linewidths=0.1, linecolor="black",
        cbar=True, vmin=vmin, vmax=vmax,
        cbar_kws={"shrink": 0.6, "label": "GSVA Score" if gsea else r"log$_2$(CPM+1)"},
    )
    ax_hm.set_aspect("equal")
    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=90, fontsize=max(4, min(10, 200 // n_samp)))
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0, fontsize=max(4, min(10, 200 // n_genes)))

    if has_tfx_bar:
        pal = _palette()
        for i, sample in enumerate(df.columns):
            tfx = (tfx_dict or {}).get(sample, 0)
            color = pal.get((sub_dict or {}).get(sample, ""), "darkgray")
            ax_bar.bar(i, tfx, color=color, width=1.0, edgecolor="black", linewidth=0.1)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_xlim(-0.5, n_samp - 0.5)
        ax_bar.set_ylabel("TFx")
        ax_bar.set_xticks([])

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Expression profile KDEs
# ---------------------------------------------------------------------------

def sample_expression_kde(
    df_wide: pd.DataFrame,
    name: str,
    *,
    subtypes_df: Optional[pd.DataFrame] = None,
) -> None:
    """KDE plot of per-sample expression distributions, colored by subtype."""
    _ensure_dir(name)
    pal = _palette()
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in df_wide.columns:
        vals = df_wide[col].dropna()
        if len(vals) > 10:
            color = "#999999"
            label = col
            if subtypes_df is not None and col in subtypes_df.index:
                sub = subtypes_df.loc[col, "subtype"] if "subtype" in subtypes_df.columns else "Unknown"
                color = pal.get(sub, "#999999")
            sns.kdeplot(vals, ax=ax, color=color, linewidth=0.6, label=label)
    ax.set_xlabel(r"log$_2$(CPM+1)")
    ax.set_ylabel("Density")
    ax.set_title("Sample Expression Profiles")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={"size": 5})
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Metric distribution histogram
# ---------------------------------------------------------------------------

def metric_distribution(
    values: pd.Series,
    metric_name: str,
    name: str,
    *,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    """Histogram + KDE of a per-gene or per-sample metric."""
    _ensure_dir(name)
    vals = values.dropna()
    if len(vals) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(vals, kde=True, stat="density", bins=40, ax=ax)
    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.axvline(vals.mean(), color="blue", ls="--", lw=1, label=f"mean={vals.mean():.3f}")
    ax.axvline(vals.median(), color="red", ls="--", lw=1, label=f"median={vals.median():.3f}")
    ax.legend(fontsize=9)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_title(f"Distribution of {metric_name} (n={len(vals):,})")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9b. Metric distribution colored by variance group
# ---------------------------------------------------------------------------

def metric_distribution_by_variance(
    gene_metrics: pd.DataFrame,
    var_map: Dict[str, float],
    metric_name: str,
    name: str,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    """Three overlaid distributions (low/med/high variance) that add to 1.

    Each group's KDE is scaled by its proportion of total genes so the
    three distributions visually sum to the total distribution.
    """
    _ensure_dir(name)
    df = gene_metrics[[metric_name]].copy()
    df["variance"] = df.index.map(lambda g: var_map.get(g, 0.0))

    groups = {
        "low (var<1)": df[(df["variance"] >= 0) & (df["variance"] < 1)][metric_name].dropna(),
        "medium (1≤var<3)": df[(df["variance"] >= 1) & (df["variance"] < 3)][metric_name].dropna(),
        "high (var≥3)": df[df["variance"] >= 3][metric_name].dropna(),
    }

    total_n = sum(len(v) for v in groups.values())
    if total_n < 3:
        return

    colors = {"low (var<1)": "#33BBEE", "medium (1≤var<3)": "#EE7733", "high (var≥3)": "#EE3377"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, vals in groups.items():
        if len(vals) < 3:
            continue
        weight = len(vals) / total_n
        sns.kdeplot(
            vals, ax=ax, color=colors[label], linewidth=1.5, label=label,
            fill=True, alpha=0.25, common_norm=False,
        )
        # Re-scale the last fill and line by the group weight.
        # sns.kdeplot draws a PolyCollection (fill) and a Line2D.
        # We scale the y-data of both by the group proportion.
        for child in ax.get_children():
            # Scale the most recently added PolyCollection
            if hasattr(child, "get_paths") and isinstance(child, mpl.collections.PolyCollection):
                for path in child.get_paths():
                    verts = path.vertices
                    verts[:, 1] *= weight
                break
        # Scale the most recently added Line2D (the KDE line)
        for line in reversed(ax.lines):
            ydata = line.get_ydata()
            if len(ydata) > 2:
                line.set_ydata(ydata * weight)
                break

    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_ylabel("Density (scaled to total)")
    ax.legend(fontsize=9)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_title(title or f"Distribution of {metric_name} by variance group (n={total_n:,})")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10. (removed — plot_roc)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. Calibration plot
# ---------------------------------------------------------------------------

def plot_calibration(
    pred_mean: pd.Series,
    pred_q5: pd.Series,
    pred_q95: pd.Series,
    truth: pd.Series,
    name: str,
) -> None:
    """Calibration: fraction of truths falling inside predicted 90% CI vs expected 90%."""
    _ensure_dir(name)
    common = pred_mean.index.intersection(truth.index)
    pm = pred_mean.loc[common].astype(float)
    q5 = pred_q5.loc[common].astype(float)
    q95 = pred_q95.loc[common].astype(float)
    t = truth.loc[common].astype(float)
    mask = np.isfinite(pm) & np.isfinite(q5) & np.isfinite(q95) & np.isfinite(t)
    q5, q95, t = q5[mask], q95[mask], t[mask]

    if len(t) < 10:
        return

    inside = ((t >= q5) & (t <= q95)).sum()
    coverage = inside / len(t)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["Observed", "Expected"], [coverage, 0.90], color=["#0077BB", "#BBBBBB"], edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_title(f"90% CI Calibration\nn={len(t):,}, coverage={coverage:.3f}")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 12. Samples summary bar (depth × subtype)
# ---------------------------------------------------------------------------

def samples_summary_bar(
    meta_df: pd.DataFrame,
    name: str,
    *,
    tfx_col: str = "tfx",
    depth_col: str = "depth",
    subtype_col: str = "subtype",
) -> None:
    """Bar chart: one bar per sample, height=depth, colored by subtype, TFx shading."""
    _ensure_dir(name)
    if depth_col not in meta_df.columns:
        return

    df = meta_df.sort_values(depth_col, ascending=False).copy()
    pal = _palette()

    width = max(8, len(df) * 0.3 + 2)
    fig, ax = plt.subplots(figsize=(width, 4))

    for i, (sample, row) in enumerate(df.iterrows()):
        depth = row.get(depth_col, 0)
        tfx = row.get(tfx_col, 1.0)
        sub = row.get(subtype_col, "Unknown")
        color = pal.get(str(sub), "#999999")

        if tfx < 1.0:
            ax.bar(i, depth * tfx, color=color, width=1.0, edgecolor="black", linewidth=0.1)
            ax.bar(i, depth * (1 - tfx), bottom=depth * tfx, color="gray", width=1.0, edgecolor="black", linewidth=0.1)
        else:
            ax.bar(i, depth, color=color, width=1.0, edgecolor="black", linewidth=0.1)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=90, fontsize=max(4, min(8, 150 // len(df))))
    ax.set_ylabel("Mean Depth")
    ax.set_xlim(-0.5, len(df) - 0.5)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 13. Benchmark grouped bar
# ---------------------------------------------------------------------------

def benchmark_bar(
    metrics_dict: Dict[str, Dict[str, float]],
    name: str,
    *,
    metrics_to_plot: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """Grouped bar chart: methods × metrics.

    ``metrics_dict`` maps method_name → {metric_name: value, ...}.
    """
    _ensure_dir(name)
    methods = list(metrics_dict.keys())
    if metrics_to_plot is None:
        metrics_to_plot = sorted({k for d in metrics_dict.values() for k in d})

    n_methods = len(methods)
    n_metrics = len(metrics_to_plot)
    x = np.arange(n_metrics)
    width = 0.8 / n_methods
    colors = ["#0077BB", "#EE7733", "#33BBEE", "#EE3377", "#009988", "#CC3311"]

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 5))
    for i, method in enumerate(methods):
        vals = [metrics_dict[method].get(m, 0) for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width, label=method, color=colors[i % len(colors)], edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ") for m in metrics_to_plot], rotation=30, ha="right")
    ax.legend(title="Method")
    ax.set_title(title or "Benchmark Comparison")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 14. Thresholding boxes (accuracy by TFx/depth bins)
# ---------------------------------------------------------------------------

def thresholding_boxes(
    sample_metrics: pd.DataFrame,
    meta_df: pd.DataFrame,
    name: str,
    *,
    metric: str = "pearson_r",
    group_col: str = "tfx",
    n_bins: int = 10,
) -> None:
    """Box plots of per-sample metric binned by TFx or depth."""
    _ensure_dir(name)
    common = sample_metrics.index.intersection(meta_df.index)
    if len(common) < 3:
        return

    df = sample_metrics.loc[common, [metric]].copy()
    df[group_col] = meta_df.loc[common, group_col].astype(float)

    if df[group_col].nunique() < n_bins:
        bin_col = group_col
    else:
        bins = np.linspace(df[group_col].min(), df[group_col].max(), n_bins + 1)
        bin_col = f"{group_col}_bin"
        df[bin_col] = pd.cut(df[group_col], bins=bins, include_lowest=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=bin_col, y=metric, data=df, hue=bin_col, ax=ax, legend=False)
    sns.stripplot(x=bin_col, y=metric, data=df, ax=ax, color="black", alpha=0.4, size=3)
    ax.set_title(f"{metric} by {group_col}")
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=90)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 15. Parallelised sample scatter generation
# ---------------------------------------------------------------------------

def _render_sample_scatter(args: Tuple) -> str:
    """Worker function for parallel sample scatter generation."""
    (truth_col, pred_col, sample_name, out_path,
     ci_lo, ci_hi, goi, tfx, depth) = args
    sample_scatter(
        truth_col, pred_col, sample_name, out_path,
        ci_lo=ci_lo, ci_hi=ci_hi, goi=goi, tfx=tfx, depth=depth,
    )
    return sample_name


def batch_sample_scatters(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    out_dir: str,
    *,
    ci_lo_wide: Optional[pd.DataFrame] = None,
    ci_hi_wide: Optional[pd.DataFrame] = None,
    goi: Optional[Dict[str, List[str]]] = None,
    subtypes_df: Optional[pd.DataFrame] = None,
    n_workers: Optional[int] = None,
) -> None:
    """Generate per-sample scatter plots in parallel.

    ``goi`` maps subtype name → list of gene names (passed through to sample_scatter).
    """
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    os.makedirs(out_dir, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))

    args_list = []
    for sample in common_samples:
        tfx = None
        depth = None
        if subtypes_df is not None and sample in subtypes_df.index:
            row = subtypes_df.loc[sample]
            tfx = row.get("tfx")
            depth = row.get("depth")

        ci_lo = ci_lo_wide[sample] if ci_lo_wide is not None and sample in ci_lo_wide.columns else None
        ci_hi = ci_hi_wide[sample] if ci_hi_wide is not None and sample in ci_hi_wide.columns else None

        args_list.append((
            truth_wide[sample], pred_wide[sample], sample,
            os.path.join(out_dir, sample),
            ci_lo, ci_hi, goi, tfx, depth,
        ))

    if n_workers > 1 and len(args_list) > 4:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_render_sample_scatter, a) for a in args_list]
            for fut in futures:
                fut.result()  # propagate exceptions
    else:
        for a in args_list:
            _render_sample_scatter(a)


# ---------------------------------------------------------------------------
# 16. Parallelised gene target scatter generation
# ---------------------------------------------------------------------------

def _render_gene_scatter(args: Tuple) -> str:
    """Worker for parallel gene target scatters."""
    truth_wide, pred_wide, gene, out_path, subtypes_df, tfx_dict = args
    gene_target_scatter(
        truth_wide, pred_wide, gene, out_path,
        subtypes_df=subtypes_df, tfx_dict=tfx_dict,
    )
    return gene


def batch_gene_scatters(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    genes: List[str],
    out_dir: str,
    *,
    subtypes_df: Optional[pd.DataFrame] = None,
    tfx_dict: Optional[Dict[str, float]] = None,
    n_workers: Optional[int] = None,
) -> None:
    """Generate per-gene target scatter plots in parallel."""
    os.makedirs(out_dir, exist_ok=True)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))

    args_list = [
        (truth_wide, pred_wide, gene, os.path.join(out_dir, gene), subtypes_df, tfx_dict)
        for gene in genes
        if gene in truth_wide.index and gene in pred_wide.index
    ]

    if n_workers > 1 and len(args_list) > 4:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_render_gene_scatter, a) for a in args_list]
            for fut in futures:
                fut.result()
    else:
        for a in args_list:
            _render_gene_scatter(a)
