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
    format_pval,
    load_palette,
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


def _darken(hex_color: str, factor: float = 0.6) -> str:
    """Return a darker shade of *hex_color* by scaling RGB toward black."""
    c = mpl.colors.to_rgb(hex_color)
    return mpl.colors.to_hex(tuple(ch * factor for ch in c))


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Sample-level scatter (single sample: genes as dots)
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
    is_holdout: bool = False,
) -> None:
    """Scatter for one sample: genes as dots with hexbin density background.

    GOI genes are highlighted and colored by subtype (from goi dict keys).
    ``goi`` maps subtype name → list of gene names.

    If ``is_holdout`` is True, GOI dots are drawn as filled diamonds
    instead of filled circles.  Axes are fixed to [0, 15].
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

    # Fixed axis range
    axis_max = 15.0
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
            if is_holdout:
                # Filled diamond for holdout samples
                ax.scatter(
                    gt.values, gp.values,
                    s=300, edgecolors="black", linewidths=0.5, color=color,
                    marker="D", zorder=3,
                )
            else:
                # Filled circle for validation samples
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

    status_tag = " [HO]" if is_holdout else ""
    info_lines = [f"{sample_name}{status_tag}"]
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
# Gene target scatter (samples as dots, colored by subtype)
# ---------------------------------------------------------------------------

def gene_target_scatter(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    gene: str,
    name: str,
    *,
    subtypes_df: Optional[pd.DataFrame] = None,
    tfx_dict: Optional[Dict[str, float]] = None,
    ci_lo_wide: Optional[pd.DataFrame] = None,
    ci_hi_wide: Optional[pd.DataFrame] = None,
    bg_truth_wide: Optional[pd.DataFrame] = None,
    bg_pred_wide: Optional[pd.DataFrame] = None,
    dim_threshold_genes: bool = True,
) -> None:
    """Scatter: samples as dots for one gene, truth vs pred.

    Colored by subtype, sized by TFx.  Fit line + 95% CI band.
    Vertical error bars show per-sample model CIs when available.
    When ``bg_truth_wide`` / ``bg_pred_wide`` are provided, HD-L
    background predictions are overlaid as gray diamonds.

    When ``dim_threshold_genes`` is True and CIs are available, samples
    whose 90% CI crosses the on/off threshold (2.0) are drawn at 50%
    alpha and excluded from fit-line / metric calculations.
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

    # Per-sample CI series for this gene (if available)
    has_ci = (
        ci_lo_wide is not None
        and ci_hi_wide is not None
        and gene in ci_lo_wide.index
        and gene in ci_hi_wide.index
    )
    ci_lo_gene: Optional[pd.Series] = None
    ci_hi_gene: Optional[pd.Series] = None
    if has_ci:
        ci_lo_gene = ci_lo_wide.loc[gene, common].reindex(t.index).astype(np.float64)
        ci_hi_gene = ci_hi_wide.loc[gene, common].reindex(t.index).astype(np.float64)

    # Identify ambiguous samples: 90% CI crosses the on/off threshold (2.0)
    _DIM_THRESHOLD = 2.0
    is_ambiguous = pd.Series(False, index=t.index)
    if dim_threshold_genes and ci_lo_gene is not None and ci_hi_gene is not None:
        is_ambiguous = (ci_lo_gene < _DIM_THRESHOLD) & (ci_hi_gene > _DIM_THRESHOLD)
        is_ambiguous = is_ambiguous.fillna(False)

    # Metrics computed on confident (non-ambiguous) subset only
    confident_mask = ~is_ambiguous
    t_conf = t[confident_mask]
    p_conf = p[confident_mask]
    if len(t_conf) >= 3:
        r_val, p_val = pearsonr(t_conf.values, p_conf.values)
        rmse = float(np.sqrt(np.mean((t_conf.values - p_conf.values) ** 2)))
    else:
        r_val, p_val = pearsonr(t.values, p.values)
        rmse = float(np.sqrt(np.mean((t.values - p.values) ** 2)))
        t_conf, p_conf = t, p  # fall back for fit line

    # Dynamic axis range (includes CI bounds so bars are fully visible).
    # Calibrated CIs are already clamped to [0, 19.93] by the calibrator.
    all_vals = np.concatenate([t.values, p.values])
    if has_ci:
        lo_ci_vals = ci_lo_wide.loc[gene, common].reindex(t.index).dropna().values
        hi_ci_vals = ci_hi_wide.loc[gene, common].reindex(t.index).dropna().values
        all_vals = np.concatenate([all_vals, lo_ci_vals, hi_ci_vals])

    # Background overlay data
    bg_t: Optional[pd.Series] = None
    bg_p: Optional[pd.Series] = None
    if (bg_truth_wide is not None and bg_pred_wide is not None
            and gene in bg_truth_wide.index and gene in bg_pred_wide.index):
        bg_common = bg_truth_wide.columns.intersection(bg_pred_wide.columns)
        _bt = bg_truth_wide.loc[gene, bg_common].astype(np.float64)
        _bp = bg_pred_wide.loc[gene, bg_common].astype(np.float64)
        bg_mask = np.isfinite(_bt) & np.isfinite(_bp)
        if bg_mask.sum() >= 1:
            bg_t, bg_p = _bt[bg_mask], _bp[bg_mask]
            all_vals = np.concatenate([all_vals, bg_t.values, bg_p.values])
    data_min = max(float(np.nanmin(all_vals)) - 0.5, 0.0)
    data_max = float(np.nanmax(all_vals)) + 0.5
    axis_lo = np.floor(data_min)
    axis_hi = np.ceil(data_max)
    # Ensure a minimum span of 2
    if axis_hi - axis_lo < 2:
        mid = (axis_lo + axis_hi) / 2
        axis_lo = mid - 1
        axis_hi = mid + 1

    fig, ax = plt.subplots(figsize=(6, 6))

    # Fit line (computed on confident subset)
    coeffs = np.polyfit(t_conf.values, p_conf.values, 1)
    fit_fn = np.poly1d(coeffs)
    x_line = np.linspace(axis_lo, axis_hi, 100)
    ax.plot(x_line, np.clip(fit_fn(x_line), axis_lo, axis_hi), "k-", lw=1, zorder=1)

    # CI band (on confident subset)
    n = len(t_conf)
    if n > 3:
        from scipy.stats import t as t_dist
        x_vals = t_conf.values
        residuals = p_conf.values - fit_fn(x_vals)
        s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        mean_x = np.mean(x_vals)
        Sxx = np.sum((x_vals - mean_x) ** 2)
        t_val = t_dist.ppf(0.975, n - 2)

        y_line = fit_fn(x_line)
        ci = t_val * s_err * np.sqrt(1 / n + (x_line - mean_x) ** 2 / Sxx)
        ax.fill_between(x_line, np.clip(y_line - ci, axis_lo, axis_hi),
                         np.clip(y_line + ci, axis_lo, axis_hi),
                         color="gray", alpha=0.25, zorder=0)

    # Plot points
    pal = _palette()
    if subtypes_df is not None:
        seen = set()
        for sample in t.index:
            sz = 200
            if tfx_dict is not None:
                sz = max(30, tfx_dict.get(sample, 1.0) * 300)
            color = "#999999"
            is_holdout = False
            if sample in subtypes_df.index:
                sub = subtypes_df.loc[sample, "subtype"] if "subtype" in subtypes_df.columns else "Unknown"
                color = pal.get(sub, "#999999")
                seen.add(sub)
                if "status" in subtypes_df.columns:
                    is_holdout = subtypes_df.loc[sample, "status"] == "holdout"

            alpha = 0.5 if is_ambiguous[sample] else 0.75

            # Error bar (CI) — drawn before marker so marker sits on top
            if ci_lo_gene is not None and ci_hi_gene is not None:
                lo = ci_lo_gene[sample]
                hi = ci_hi_gene[sample]
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar(
                        t[sample], p[sample],
                        yerr=[[max(p[sample] - lo, 0)], [max(hi - p[sample], 0)]],
                        fmt="none", ecolor="black", elinewidth=0.8,
                        capsize=2, zorder=2, alpha=alpha,
                    )

            if is_holdout:
                ax.scatter(
                    t[sample], p[sample], s=sz, c=color, marker="D",
                    edgecolors="black", linewidths=0.5, alpha=alpha, zorder=3,
                )
            else:
                ax.scatter(
                    t[sample], p[sample], s=sz, c=color, marker="o",
                    edgecolors="black", linewidths=0.5, alpha=alpha, zorder=3,
                )
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=pal.get(s, "#999999"), markersize=8, label=s)
            for s in sorted(seen)
        ]
        if handles:
            ax.legend(handles=handles, title="Subtype", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        # Error bars (CI) for the non-subtype branch
        if ci_lo_gene is not None and ci_hi_gene is not None:
            yerr_lo = (p - ci_lo_gene).clip(lower=0).values
            yerr_hi = (ci_hi_gene - p).clip(lower=0).values
            eb_alpha = np.where(is_ambiguous.values, 0.5, 0.75)
            for i in range(len(t)):
                ax.errorbar(
                    t.values[i], p.values[i],
                    yerr=[[yerr_lo[i]], [yerr_hi[i]]],
                    fmt="none", ecolor="black", elinewidth=0.8,
                    capsize=2, zorder=2, alpha=float(eb_alpha[i]),
                )
        sizes = [200] * len(t)
        if tfx_dict is not None:
            sizes = [max(30, tfx_dict.get(s, 1.0) * 300) for s in t.index]
        # Draw confident and ambiguous points with different alpha
        conf_idx = ~is_ambiguous
        amb_idx = is_ambiguous
        if conf_idx.any():
            ax.scatter(
                t[conf_idx].values, p[conf_idx].values,
                s=[sizes[i] for i, v in enumerate(conf_idx) if v],
                edgecolors="black", linewidths=0.5, alpha=0.75, zorder=3,
            )
        if amb_idx.any():
            ax.scatter(
                t[amb_idx].values, p[amb_idx].values,
                s=[sizes[i] for i, v in enumerate(amb_idx) if v],
                edgecolors="black", linewidths=0.5, alpha=0.5, zorder=3,
            )

    # HD-L background overlay (gray diamonds, behind tumor legend)
    if bg_t is not None and bg_p is not None:
        ax.scatter(
            bg_t.values, bg_p.values, s=80, c="#AAAAAA", marker="D",
            edgecolors="black", linewidths=0.5, alpha=0.6, zorder=2, label="HD-L",
        )

    ax.set_xlabel(r"RNA-Seq log$_2$(CPM+1)")
    ax.set_ylabel(r"Predicted log$_2$(CPM+1)")
    ax.set_xlim(axis_lo, axis_hi)
    ax.set_ylim(axis_lo, axis_hi)
    tick_step = max(1, int(np.round((axis_hi - axis_lo) / 5)))
    ticks = np.arange(axis_lo, axis_hi + tick_step, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_box_aspect(1)
    ax.set_title(gene)

    # Annotation: indicate when metrics exclude ambiguous points
    n_amb = int(is_ambiguous.sum())
    metric_note = f"Pearson r = {r_val:.3f}\n{format_pval(p_val)}\nRMSE = {rmse:.3f}"
    if n_amb > 0:
        metric_note += f"\n({n_amb} ambiguous dimmed)"
    ax.text(
        0.05, 0.95, metric_note,
        transform=ax.transAxes, va="top", fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Double heatmap (true vs pred side-by-side)
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
# Single heatmap with TFx bar
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
# Expression profile KDEs
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
# Metric distribution histogram
# ---------------------------------------------------------------------------

def metric_distribution(
    values: pd.Series,
    metric_name: str,
    name: str,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    square: bool = False,
    log_y: bool = False,
    draw_stat_lines: bool = True,
) -> None:
    """Histogram + KDE of a per-gene or per-sample metric.

    If ``square`` is True, figure uses equal aspect and a dashed gray
    vertical line is drawn at x=0.
    If ``log_y`` is True, the y-axis uses a log scale.
    If ``draw_stat_lines`` is False, mean/median vertical lines are
    suppressed (text annotations in the legend are still shown).
    """
    _ensure_dir(name)
    vals = values.dropna()
    if len(vals) < 3:
        return

    size = (6, 6) if square else (7, 4)
    fig, ax = plt.subplots(figsize=size)
    sns.histplot(vals, kde=True, stat="density", bins=40, ax=ax)
    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_ylabel("Density")
    if draw_stat_lines:
        ax.axvline(vals.mean(), color="blue", ls="--", lw=1, label=f"mean={vals.mean():.3f}")
        ax.axvline(vals.median(), color="red", ls="--", lw=1, label=f"median={vals.median():.3f}")
    else:
        # Text-only legend entries (no drawn lines)
        ax.plot([], [], ' ', label=f"mean={vals.mean():.3f}")
        ax.plot([], [], ' ', label=f"median={vals.median():.3f}")
    if square:
        ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    if xlim:
        ax.set_xlim(xlim)
    if log_y:
        ax.set_yscale("log")
    if square:
        ax.set_box_aspect(1)
    ax.set_title(f"Distribution of {metric_name} (n={len(vals):,})")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metric distribution colored by variance group
# ---------------------------------------------------------------------------

def metric_distribution_by_variance(
    gene_metrics: pd.DataFrame,
    var_map: Dict[str, float],
    metric_name: str,
    name: str,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    square: bool = False,
) -> None:
    """Three overlaid distributions (low/med/high variance) that add to 1.

    Each group's KDE is scaled by its proportion of total genes so the
    three distributions visually sum to the total distribution.  A
    dashed vertical line marks each group's median, and the legend
    includes ``n=`` counts.
    """
    from scipy.stats import gaussian_kde

    _ensure_dir(name)
    df = gene_metrics[[metric_name]].copy()
    df["variance"] = df.index.map(lambda g: var_map.get(g, 0.0))

    groups = {
        "low (var<1)": df[(df["variance"] >= 0) & (df["variance"] < 1)][metric_name].dropna(),
        "medium (1\u2264var<3)": df[(df["variance"] >= 1) & (df["variance"] < 3)][metric_name].dropna(),
        "high (var\u22653)": df[df["variance"] >= 3][metric_name].dropna(),
    }

    total_n = sum(len(v) for v in groups.values())
    if total_n < 3:
        return

    colors = {"low (var<1)": "#33BBEE", "medium (1\u2264var<3)": "#EE7733", "high (var\u22653)": "#EE3377"}

    size = (6, 6) if square else (7, 4)
    fig, ax = plt.subplots(figsize=size)

    # Determine x-range for KDE evaluation
    if xlim is not None:
        x_eval = np.linspace(xlim[0], xlim[1], 512)
    else:
        all_vals = pd.concat(groups.values())
        x_eval = np.linspace(float(all_vals.min()) - 0.5,
                             float(all_vals.max()) + 0.5, 512)

    for label, vals in groups.items():
        if len(vals) < 3:
            continue
        weight = len(vals) / total_n
        color = colors[label]

        # Compute KDE manually so we can scale exactly by group proportion
        kde = gaussian_kde(vals.values)
        y_eval = kde(x_eval) * weight

        legend_label = f"{label}  (n={len(vals):,})"
        ax.plot(x_eval, y_eval, color=color, linewidth=1.5, label=legend_label)
        ax.fill_between(x_eval, y_eval, alpha=0.25, color=color)

        # Dotted vertical line at group median
        med = float(vals.median())
        y_at_med = float(kde(np.array([med]))[0]) * weight
        ax.axvline(med, color=color, ls=":", lw=1.5, alpha=0.85)
        ax.text(med, y_at_med * 1.05, f"med={med:.3f}", color=color,
                fontsize=7, ha="center", va="bottom",
                path_effects=[PathEffects.withStroke(linewidth=1.5, foreground="white")])

    ax.set_xlabel(metric_name.replace("_", " ").title())
    ax.set_ylabel("Density (scaled to total)")
    if square:
        ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.legend(fontsize=8)
    if xlim:
        ax.set_xlim(xlim)
    if square:
        ax.set_box_aspect(1)
    ax.set_title(title or f"Distribution of {metric_name} by variance group (n={total_n:,})")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration coverage comparison
# ---------------------------------------------------------------------------

def plot_calibration_coverage(
    base_curve: Dict,
    conf_curve: Dict,
    name: str,
    *,
    title: Optional[str] = None,
) -> None:
    """Plot empirical coverage vs nominal level for base and CQR-calibrated CIs.

    Accepts two curve formats:
      - old style: ``{level: coverage}`` dict
      - new style: ``{"nominal": [...], "empirical": [...], ...}`` dict
        (as returned by ``calibration.evaluate_calibration``).

    A diagonal identity line indicates perfect calibration.
    """
    _ensure_dir(name)

    # Normalise both curves to (levels, values) lists
    def _unpack(curve):
        if "nominal" in curve:
            return list(curve["nominal"]), list(curve["empirical"])
        levels = sorted(curve.keys())
        return levels, [curve[l] for l in levels]

    base_levels, base_vals = _unpack(base_curve)
    conf_levels, conf_vals = _unpack(conf_curve)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.plot(base_levels, base_vals, "o-", color="#0077BB", label="Base model", markersize=6)
    ax.plot(conf_levels, conf_vals, "s-", color="#EE3377", label="CQR calibrated", markersize=6)
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.05)
    ax.set_box_aspect(1)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title or "Calibration Coverage")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_regime_coverage_heatmap(
    regime_coverage: Dict[str, float],
    name: str,
    *,
    n_var_bins: int = 4,
    n_zero_bins: int = 4,
    target_coverage: float = 0.90,
    title: Optional[str] = None,
) -> None:
    """Heatmap of empirical coverage by (variance quartile × zero-rate quartile).

    ``regime_coverage`` maps regime labels like "(2, 3)" to empirical
    coverage floats.  The heatmap shows how well calibration performs
    in each gene regime, with the target coverage level annotated.
    """
    _ensure_dir(name)

    grid = np.full((n_var_bins, n_zero_bins), np.nan)
    for label, cov in regime_coverage.items():
        # Parse "(var_q, zero_q)"
        parts = label.strip("()").split(",")
        if len(parts) != 2:
            continue
        vq, zq = int(parts[0].strip()), int(parts[1].strip())
        if 0 <= vq < n_var_bins and 0 <= zq < n_zero_bins:
            grid[vq, zq] = cov

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.80, vmax=1.0,
                   aspect="auto", origin="lower")

    # Annotate cells
    for i in range(n_var_bins):
        for j in range(n_zero_bins):
            val = grid[i, j]
            if np.isfinite(val):
                color = "white" if val < 0.88 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(n_zero_bins))
    ax.set_xticklabels([f"Q{q+1}" for q in range(n_zero_bins)])
    ax.set_yticks(range(n_var_bins))
    ax.set_yticklabels([f"Q{q+1}" for q in range(n_var_bins)])
    ax.set_xlabel("Zero-Rate Quartile")
    ax.set_ylabel("Variance Quartile")

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Empirical Coverage")

    # Target reference line on colorbar
    cb.ax.axhline(y=target_coverage, color="black", linewidth=1.5, linestyle="--")

    ax.set_title(title or f"Regime Coverage (target={target_coverage:.0%})")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_reliability_scatter(
    predicted_r: pd.Series,
    true_r: pd.Series,
    name: str,
    *,
    title: Optional[str] = None,
) -> None:
    """Scatter plot: predicted gene reliability (r̂) vs true holdout Pearson r.

    Includes identity line, Spearman ρ annotation, and marginal density.
    """
    _ensure_dir(name)

    common = predicted_r.index.intersection(true_r.index)
    pred = predicted_r.loc[common].dropna()
    true = true_r.reindex(pred.index).dropna()
    pred = pred.reindex(true.index)

    if len(pred) < 5:
        return

    x = pred.values
    y = true.values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(x, y, s=6, alpha=0.3, color="#4477AA", edgecolors="none")
    ax.plot([-1, 1], [-1, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Predicted Gene Reliability (r̂)")
    ax.set_ylabel("True Holdout Pearson r")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_box_aspect(1)
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}\nn = {len(pred)}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_title(title or "Gene Reliability: Predicted vs True")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_retention_curve(
    retention: Dict[str, Any],
    name: str,
    *,
    baseline_mean_r: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    """Top-K gene retention curve: mean true Pearson r vs retention fraction.

    If the reliability model works, retaining only the top-predicted genes
    should yield substantially higher actual r than retaining all genes.

    ``retention`` is the dict from ``reliability.retention_curve`` with
    keys ``fractions``, ``mean_true_r``, ``n_genes``.
    """
    _ensure_dir(name)

    fractions = retention.get("fractions", [])
    mean_r = retention.get("mean_true_r", [])
    n_genes = retention.get("n_genes", [])

    if not fractions or all(np.isnan(v) for v in mean_r):
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Main curve
    ax.plot(
        [f * 100 for f in fractions], mean_r,
        "o-", color="#EE3377", linewidth=2, markersize=7, label="Reliability-ranked"
    )

    # Annotate gene counts
    for f, r_val, n in zip(fractions, mean_r, n_genes):
        if np.isfinite(r_val):
            ax.annotate(f"n={n}", (f * 100, r_val), fontsize=7,
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", color="#666666")

    # Baseline: mean r using all genes
    if baseline_mean_r is not None:
        ax.axhline(y=baseline_mean_r, color="#999999", linestyle="--", linewidth=1,
                   label=f"All genes (r={baseline_mean_r:.3f})")

    ax.set_xlabel("Genes Retained (%)")
    ax.set_ylabel("Mean True Pearson r")
    ax.set_xlim(5, 105)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title or "Gene Retention Curve")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_regime_qhat_heatmap(
    q_hats: Dict[str, float],
    name: str,
    *,
    n_var_bins: int = 4,
    n_zero_bins: int = 4,
    title: Optional[str] = None,
) -> None:
    """Heatmap of per-regime q̂ values (the normalized conformal quantile).

    Visualises how much calibration correction each gene regime requires.
    Higher q̂ means the model's base intervals were more under-covering
    in that regime.
    """
    _ensure_dir(name)

    grid = np.full((n_var_bins, n_zero_bins), np.nan)
    for label, qval in q_hats.items():
        parts = label.strip("()").split(",")
        if len(parts) != 2:
            continue
        vq, zq = int(parts[0].strip()), int(parts[1].strip())
        if 0 <= vq < n_var_bins and 0 <= zq < n_zero_bins:
            grid[vq, zq] = qval

    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(0.5, np.nanmax(grid) * 1.1) if np.any(np.isfinite(grid)) else 1.0
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=vmax,
                   aspect="auto", origin="lower")

    for i in range(n_var_bins):
        for j in range(n_zero_bins):
            val = grid[i, j]
            if np.isfinite(val):
                color = "white" if val > vmax * 0.7 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(n_zero_bins))
    ax.set_xticklabels([f"Q{q+1}" for q in range(n_zero_bins)])
    ax.set_yticks(range(n_var_bins))
    ax.set_yticklabels([f"Q{q+1}" for q in range(n_var_bins)])
    ax.set_xlabel("Zero-Rate Quartile")
    ax.set_ylabel("Variance Quartile")

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Normalized q̂")

    ax.set_title(title or "Per-Regime Conformal q̂")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_reliability_weights(
    feature_columns: List[str],
    weights: np.ndarray,
    name: str,
    *,
    title: Optional[str] = None,
) -> None:
    """Horizontal bar chart of ridge regression feature weights.

    Shows which prediction features contribute most to the gene-level
    reliability estimate.
    """
    _ensure_dir(name)

    # Human-readable display names for reliability model features.
    _DISPLAY_NAMES: Dict[str, str] = {
        "f1_mean_width": "Mean CI Width",
        "f2_median_width": "Median CI Width",
        "f3_mean_entropy": "Mixture Entropy (δ/γ/n)",
        "f4_mean_hurdle_amb": "Hurdle Ambiguity (πδ·(1−πδ))",
        "f5_mean_cont_amb": "Continuous Ambiguity (πγ·πn)",
        "f6_mean_pmax": "Mean Max Component π",
        "f7_median_pmax": "Median Max Component π",
        "f8_mean_gap": "Component Gap (πmax − π2nd)",
        "f9_mean_u_normal": "Normal Variance (πn·σ²n)",
        "f10_mean_u_gamma": "Gamma Variance (πγ·k/β²)",
        "f11_mean_d_mu": "Fold-to-Fold SD(μ)",
        "f12_mean_d_width": "Fold-to-Fold SD(CI Width)",
        "f13_mean_d_delta": "Fold-to-Fold SD(πδ)",
        "f14_pred_sd": "Prediction SD (across samples)",
        "f15_gene_zero_rate": "Gene Zero-Rate",
        "f16_gene_variance": "Gene Expression Variance",
    }

    n = len(feature_columns)
    if n == 0:
        return

    # Sort by absolute weight for visual clarity
    order = np.argsort(np.abs(weights))[::-1]
    cols = [feature_columns[i] for i in order]
    vals = weights[order]
    display_cols = [_DISPLAY_NAMES.get(c, c) for c in cols]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * n + 1)))
    colors = ["#EE3377" if v < 0 else "#0077BB" for v in vals]
    ax.barh(range(n), vals, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(display_cols, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ridge Weight")
    ax.set_title(title or "Reliability Model — Feature Weights")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_error_vs_width(
    width_base: pd.Series,
    width_conf: pd.Series,
    gene_metrics: pd.DataFrame,
    name: str,
    *,
    title: Optional[str] = None,
) -> None:
    """Scatter: mean CI width per gene vs gene-level metric (Spearman rho).

    Two series: base and conformal-adjusted widths.  ``gene_metrics`` must
    have a ``spearman_rho`` column indexed by gene name.
    """
    _ensure_dir(name)
    common = width_base.index.intersection(gene_metrics.index)
    if len(common) < 10:
        return

    wb = width_base.loc[common]
    wc = width_conf.loc[common]
    rho = gene_metrics.loc[common, "spearman_rho"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(wb.values, rho.values, s=8, alpha=0.3, color="#0077BB", label="Base width")
    ax.scatter(wc.values, rho.values, s=8, alpha=0.3, color="#EE3377", label="Conformal width")
    ax.set_xlabel("Mean CI width (per gene)")
    ax.set_ylabel("Gene Spearman ρ")
    ax.legend(fontsize=9)
    ax.set_title(title or "Interval Width vs Gene Spearman ρ")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parallelised sample scatter generation
# ---------------------------------------------------------------------------

def _render_sample_scatter(args: Tuple) -> str:
    """Worker function for parallel sample scatter generation."""
    (truth_col, pred_col, sample_name, out_path,
     ci_lo, ci_hi, goi, tfx, depth, is_holdout) = args
    sample_scatter(
        truth_col, pred_col, sample_name, out_path,
        ci_lo=ci_lo, ci_hi=ci_hi, goi=goi, tfx=tfx, depth=depth,
        is_holdout=is_holdout,
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
    If ``subtypes_df`` has a ``status`` column, samples with status == "holdout"
    will be drawn with hollow markers.
    """
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    os.makedirs(out_dir, exist_ok=True)

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))

    args_list = []
    for sample in common_samples:
        tfx = None
        depth = None
        is_holdout = False
        if subtypes_df is not None and sample in subtypes_df.index:
            row = subtypes_df.loc[sample]
            tfx = row.get("tfx")
            depth = row.get("depth")
            is_holdout = row.get("status") == "holdout"

        ci_lo = ci_lo_wide[sample] if ci_lo_wide is not None and sample in ci_lo_wide.columns else None
        ci_hi = ci_hi_wide[sample] if ci_hi_wide is not None and sample in ci_hi_wide.columns else None

        args_list.append((
            truth_wide[sample], pred_wide[sample], sample,
            os.path.join(out_dir, sample),
            ci_lo, ci_hi, goi, tfx, depth, is_holdout,
        ))

    if n_workers > 1 and len(args_list) > 4:
        print(f"    Rendering {len(args_list)} sample scatters "
              f"across {n_workers} workers (multi-process) ...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_render_sample_scatter, a) for a in args_list]
            for fut in futures:
                fut.result()  # propagate exceptions
    else:
        for a in args_list:
            _render_sample_scatter(a)


# ---------------------------------------------------------------------------
# Parallelised gene target scatter generation
# ---------------------------------------------------------------------------

def _render_gene_scatter(args: Tuple) -> str:
    """Worker for parallel gene target scatters."""
    truth_wide, pred_wide, gene, out_path, subtypes_df, tfx_dict, ci_lo_wide, ci_hi_wide, bg_truth_wide, bg_pred_wide, dim_threshold_genes = args
    gene_target_scatter(
        truth_wide, pred_wide, gene, out_path,
        subtypes_df=subtypes_df, tfx_dict=tfx_dict,
        ci_lo_wide=ci_lo_wide, ci_hi_wide=ci_hi_wide,
        bg_truth_wide=bg_truth_wide, bg_pred_wide=bg_pred_wide,
        dim_threshold_genes=dim_threshold_genes,
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
    ci_lo_wide: Optional[pd.DataFrame] = None,
    ci_hi_wide: Optional[pd.DataFrame] = None,
    bg_truth_wide: Optional[pd.DataFrame] = None,
    bg_pred_wide: Optional[pd.DataFrame] = None,
    dim_threshold_genes: bool = True,
    n_workers: Optional[int] = None,
) -> None:
    """Generate per-gene target scatter plots in parallel."""
    os.makedirs(out_dir, exist_ok=True)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))

    args_list = [
        (truth_wide, pred_wide, gene, os.path.join(out_dir, gene),
         subtypes_df, tfx_dict, ci_lo_wide, ci_hi_wide,
         bg_truth_wide, bg_pred_wide, dim_threshold_genes)
        for gene in genes
        if gene in truth_wide.index and gene in pred_wide.index
    ]

    if n_workers > 1 and len(args_list) > 4:
        print(f"    Rendering {len(args_list)} gene scatters "
              f"across {n_workers} workers (multi-process) ...")
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_render_gene_scatter, a) for a in args_list]
            for fut in futures:
                fut.result()
    else:
        for a in args_list:
            _render_gene_scatter(a)


# ---------------------------------------------------------------------------
# Benchmarking grid plots
# ---------------------------------------------------------------------------

# Palette-based model colours for all benchmark figures.
# Falls back to grey cycle if a model is absent from the palette.
_BENCH_FALLBACK_COLORS = [
    "#0077BB", "#EE7733", "#33BBEE", "#EE3377", "#009988", "#CC3311",
    "#332288", "#DDCC77", "#882255", "#44AA99", "#88CCEE", "#AA4499",
]


def _bench_color_map(model_names: Sequence[str]) -> Dict[str, str]:
    pal = _palette()
    cmap: Dict[str, str] = {}
    fb_idx = 0
    for m in model_names:
        if m in pal:
            cmap[m] = pal[m]
        else:
            cmap[m] = _BENCH_FALLBACK_COLORS[fb_idx % len(_BENCH_FALLBACK_COLORS)]
            fb_idx += 1
    return cmap


def benchmark_grid_boxplots(
    metrics_table: pd.DataFrame,
    name: str,
    *,
    metric_col: str,
    ylabel: str,
    title: str,
    tfx_order: Sequence[str],
    depth_order: Sequence[str],
    model_order: Sequence[str],
    naive_model: str = "Naive",
) -> None:
    """TFx x Depth grid of box plots, one box per model/score.

    ``metrics_table`` must have columns: tfx, depth, model, and ``metric_col``.
    A dashed gray horizontal line is drawn at the Naive model's mean value
    (or single value) in each panel for reference.
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)
    n_rows = len(tfx_order)
    n_cols = len(depth_order)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols + 1, 3.2 * n_rows + 1),
        sharex=True, sharey=True, squeeze=False,
    )

    for ri, tfx in enumerate(tfx_order):
        for ci, depth in enumerate(depth_order):
            ax = axes[ri, ci]
            sub = metrics_table[
                (metrics_table["tfx"] == tfx) & (metrics_table["depth"] == depth)
            ]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
            else:
                # Naive reference line — dashed gray spanning full x-extent
                naive_vals = sub.loc[sub["model"] == naive_model, metric_col].dropna().values
                if len(naive_vals) > 0:
                    naive_ref = float(np.mean(naive_vals))
                    ax.axhline(naive_ref, color="#BBBBBB", ls="--", lw=1.2, zorder=0)

                plot_data = []
                labels = []
                colors = []
                for model in model_order:
                    vals = sub.loc[sub["model"] == model, metric_col].dropna().values
                    if len(vals) > 0:
                        plot_data.append(vals)
                        labels.append(model)
                        colors.append(cmap.get(model, "#BBBBBB"))
                if plot_data:
                    # If every model has exactly 1 value, use dot plot
                    all_scalar = all(len(v) == 1 for v in plot_data)
                    if all_scalar:
                        for i, (vals, c) in enumerate(zip(plot_data, colors), start=1):
                            ax.plot(i, vals[0], "o", color=c, markersize=24,
                                    markeredgecolor=_darken(c), markeredgewidth=1.5,
                                    zorder=3)
                        ax.set_xticks(range(1, len(labels) + 1))
                        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
                    else:
                        bp = ax.boxplot(
                            plot_data, patch_artist=True, widths=0.6,
                            medianprops=dict(color="black", linewidth=1.2),
                            flierprops=dict(markersize=2),
                        )
                        for patch, c in zip(bp["boxes"], colors):
                            patch.set_facecolor(c)
                            patch.set_alpha(0.7)
                        # Increase outlier dot size
                        for flier in bp.get("fliers", []):
                            flier.set_markersize(4)
                        ax.set_xticks(range(1, len(labels) + 1))
                        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)

            if ri == 0:
                ax.set_title(depth, fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"TFx {tfx}\n{ylabel}", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def benchmark_grid_histograms(
    gene_rho_table: pd.DataFrame,
    name: str,
    *,
    tfx_order: Sequence[str],
    depth_order: Sequence[str],
    model_order: Sequence[str],
    title: Optional[str] = None,
) -> None:
    """TFx x Depth grid of overlaid Spearman rho histograms across models.

    ``gene_rho_table`` has columns: tfx, depth, model, spearman_rho, n_rho_ge070.
    Each row is one gene.
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)
    n_rows = len(tfx_order)
    n_cols = len(depth_order)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols + 1, 4 * n_rows + 1),
        sharex=True, sharey=False, squeeze=False,
    )

    for ri, tfx in enumerate(tfx_order):
        for ci, depth in enumerate(depth_order):
            ax = axes[ri, ci]
            sub = gene_rho_table[
                (gene_rho_table["tfx"] == tfx) & (gene_rho_table["depth"] == depth)
            ]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
            else:
                legend_entries = []
                for model in model_order:
                    vals = sub.loc[sub["model"] == model, "spearman_rho"].dropna().values
                    if len(vals) < 3:
                        continue
                    n_good = int((vals >= 0.70).sum())
                    color = cmap.get(model, "#BBBBBB")
                    sns.kdeplot(
                        vals, ax=ax, color=color, linewidth=1.5,
                        fill=False, clip=(-1, 1),
                    )
                    legend_entries.append(f"{model} (n≥0.7={n_good})")
                if legend_entries:
                    ax.legend(legend_entries, fontsize=5, loc="upper left",
                              framealpha=0.6)
            ax.set_xlim(-1, 1)
            if ri == 0:
                ax.set_title(depth, fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"TFx {tfx}\nDensity", fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel("Spearman ρ", fontsize=8)

    fig.suptitle(title or "Per-Gene Spearman ρ Distribution", fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def benchmark_ecdna_lines(
    ecdna_table: pd.DataFrame,
    name: str,
    *,
    model_order: Sequence[str],
) -> None:
    """Line plots of 6 metrics vs effective ctDNA (TFx * depth).

    Layout: 2 columns x 3 rows.  A single shared legend is placed at the
    bottom of the figure.
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)

    metric_defs = [
        ("gvw_rho", "GVW Spearman \u03c1"),
        ("mean_top10pct_gene_rho", "Mean \u03c1 (Top 10% Var.)"),
        ("median_gene_rho", "Median Gene Spearman \u03c1"),
        ("mean_gene_rho", "Mean Gene Spearman \u03c1"),
        ("sample_rho_mean", "Mean Sample Spearman \u03c1"),
        ("rmse", "Mean Gene RMSE"),
    ]

    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), squeeze=False)

    legend_handles = []
    for idx, (col, label) in enumerate(metric_defs):
        ri, ci = divmod(idx, n_cols)
        ax = axes[ri, ci]
        for model in model_order:
            sub = ecdna_table[ecdna_table["model"] == model].sort_values("ecdna")
            if sub.empty or col not in sub.columns or sub[col].isna().all():
                continue
            color = cmap.get(model, "#BBBBBB")
            line, = ax.plot(sub["ecdna"], sub[col], "o-",
                            color=color, markersize=16, linewidth=1.4,
                            markeredgecolor=_darken(color),
                            markeredgewidth=1.2)
            # Collect legend entries from first panel only
            if idx == 0:
                legend_handles.append((line, model))
        ax.set_xlabel("Effective ctDNA (TFx \u00d7 depth)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xscale("log")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Shared legend at bottom
    if legend_handles:
        handles, labels = zip(*legend_handles)
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 6), fontsize=8,
                   framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Benchmark Metrics vs Effective ctDNA",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def benchmark_depth_lines(
    metrics_table: pd.DataFrame,
    name: str,
    *,
    depth: str,
    model_order: Sequence[str],
) -> None:
    """Line plots of 6 metrics vs TFx for a single sequencing depth.

    Parameters
    ----------
    metrics_table : pd.DataFrame
        Full benchmark metrics table with columns ``tfx``, ``depth``,
        ``model``, and metric columns.
    depth : str
        Depth filter value (e.g. "10x" or "5x").
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)

    # Parse TFx strings to numeric
    _tfx_num = {"TF0.50": 0.50, "TF0.30": 0.30, "TF0.10": 0.10,
                "TF0.05": 0.05, "TF0.03": 0.03, "TF0.01": 0.01}

    sub = metrics_table[metrics_table["depth"] == depth].copy()
    if sub.empty:
        return

    sub["tfx_numeric"] = sub["tfx"].map(_tfx_num)
    sub = sub.dropna(subset=["tfx_numeric"])

    metric_defs = [
        ("gvw_rho", "GVW Spearman \u03c1"),
        ("mean_top10pct_gene_rho", "Mean \u03c1 (Top 10% Var.)"),
        ("median_gene_rho", "Median Gene Spearman \u03c1"),
        ("mean_gene_rho", "Mean Gene Spearman \u03c1"),
        ("sample_rho_mean", "Mean Sample Spearman \u03c1"),
        ("rmse", "Mean Gene RMSE"),
    ]

    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), squeeze=False)

    legend_handles = []
    for idx, (col, label) in enumerate(metric_defs):
        ri, ci = divmod(idx, n_cols)
        ax = axes[ri, ci]
        for model in model_order:
            msub = sub[sub["model"] == model].sort_values("tfx_numeric")
            if msub.empty or col not in msub.columns or msub[col].isna().all():
                continue
            color = cmap.get(model, "#BBBBBB")
            line, = ax.plot(msub["tfx_numeric"], msub[col], "o-",
                            color=color, markersize=10, linewidth=1.4,
                            markeredgecolor=_darken(color),
                            markeredgewidth=1.2)
            if idx == 0:
                legend_handles.append((line, model))
        ax.set_xlabel("Tumor Fraction (TFx)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xscale("log")
        # Absolute-value tick labels at every observed TFx
        seen_tfx = sorted(sub["tfx_numeric"].unique())
        if seen_tfx:
            ax.set_xticks(seen_tfx)
            ax.set_xticklabels([f"{v:g}" for v in seen_tfx])
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    if legend_handles:
        handles, labels = zip(*legend_handles)
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 6), fontsize=8,
                   framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Benchmark Metrics vs TFx — {depth} depth",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Benchmarking cohort-panel plots (Benchmarking_PAN-FT)
# ---------------------------------------------------------------------------

def benchmark_cohort_boxplots(
    metrics_table: pd.DataFrame,
    name: str,
    *,
    metric_col: str,
    ylabel: str,
    title: str,
    cohort_order: Sequence[str],
    model_order: Sequence[str],
    naive_model: str = "Naive",
) -> None:
    """Single-panel cohort-paired box/dot plots, one group per model.

    ``metrics_table`` must have columns: cohort, model, and ``metric_col``.
    Both cohorts are shown side-by-side within each model group.  For
    scalar data (one value per model per cohort), different marker symbols
    distinguish cohorts (square / triangle).  A dashed gray line marks the
    Naive model's grand mean for reference.
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)
    n_cohorts = len(cohort_order)

    # Filter to models present in the data
    present_models = [m for m in model_order
                      if m in set(metrics_table["model"])]
    n_models = len(present_models)
    if n_models == 0:
        return

    # Determine if data is scalar (1 value per model×cohort) or distributional
    all_scalar = True
    for model in present_models:
        for cohort in cohort_order:
            vals = metrics_table.loc[
                (metrics_table["model"] == model) &
                (metrics_table["cohort"] == cohort), metric_col
            ].dropna().values
            if len(vals) > 1:
                all_scalar = False
                break
        if not all_scalar:
            break

    fig_width = max(5, 0.7 * n_models + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Naive reference line
    naive_vals = metrics_table.loc[
        metrics_table["model"] == naive_model, metric_col
    ].dropna().values
    if len(naive_vals) > 0:
        ax.axhline(float(np.mean(naive_vals)), color="#BBBBBB",
                   ls="--", lw=1.2, zorder=0)

    if all_scalar:
        # Dot plot: circle markers for all cohorts; second cohort gets
        # different marker to distinguish.
        from matplotlib.patches import Patch
        group_width = 0.3
        offsets = np.linspace(-group_width / 2, group_width / 2, n_cohorts)
        cohort_alpha = {c: 0.55 + 0.15 * i for i, c in enumerate(cohort_order)}
        _cohort_markers = ["o", "D", "s", "^", "v"]  # circle, diamond, square, ...

        for ci_idx, cohort in enumerate(cohort_order):
            marker = _cohort_markers[ci_idx % len(_cohort_markers)]
            for mi, model in enumerate(present_models):
                vals = metrics_table.loc[
                    (metrics_table["model"] == model) &
                    (metrics_table["cohort"] == cohort), metric_col
                ].dropna().values
                if len(vals) == 0:
                    continue
                color = cmap.get(model, "#BBBBBB")
                x = mi + 1 + offsets[ci_idx]
                y = float(vals[0])
                alpha = cohort_alpha.get(cohort, 0.7)
                ax.plot(x, y, marker=marker, markersize=10,
                        color=color, markeredgecolor=_darken(color),
                        markeredgewidth=1.2, alpha=alpha, zorder=3,
                        linestyle="none")

        # Legend: proxy patches
        legend_patches = []
        for ci_idx, cohort in enumerate(cohort_order):
            marker = _cohort_markers[ci_idx % len(_cohort_markers)]
            p = ax.plot([], [], marker=marker, markersize=8,
                        color="#888888", alpha=cohort_alpha.get(cohort, 0.7),
                        markeredgecolor="#555555", linestyle="none",
                        label=cohort)[0]
            legend_patches.append(p)
        ax.legend(handles=legend_patches, fontsize=8, loc="best",
                  framealpha=0.8)

        ax.set_xticks(range(1, n_models + 1))
        ax.set_xticklabels(present_models, rotation=55, ha="right", fontsize=7)
        # Auto-scale y
        all_vals = metrics_table[metric_col].dropna().values
        if len(all_vals):
            ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
            margin = max(0.05 * (ymax - ymin), 0.02)
            ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_xlim(0.4, n_models + 0.6)
        ax.set_aspect("auto")

    else:
        # Paired boxplots: for each model, n_cohorts side-by-side boxes
        box_width = 0.25
        group_gap = 0.05
        total_group = n_cohorts * box_width + (n_cohorts - 1) * group_gap
        all_positions = []
        all_data = []
        all_colors = []
        cohort_color_alpha = {c: 0.55 + 0.15 * i for i, c in enumerate(cohort_order)}
        tick_positions = []
        tick_labels = []

        for mi, model in enumerate(present_models):
            group_center = mi * (total_group + 0.3)
            tick_positions.append(group_center)
            tick_labels.append(model)
            for ci_idx, cohort in enumerate(cohort_order):
                vals = metrics_table.loc[
                    (metrics_table["model"] == model) &
                    (metrics_table["cohort"] == cohort), metric_col
                ].dropna().values
                if len(vals) == 0:
                    vals = np.array([np.nan])
                pos = (group_center
                       - total_group / 2
                       + ci_idx * (box_width + group_gap)
                       + box_width / 2)
                all_positions.append(pos)
                all_data.append(vals)
                all_colors.append(cmap.get(model, "#BBBBBB"))

        if all_data:
            bp = ax.boxplot(
                all_data, positions=all_positions, widths=box_width,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.2),
                flierprops=dict(markersize=2),
            )
            for pi, (patch, c) in enumerate(zip(bp["boxes"], all_colors)):
                ci_idx = pi % n_cohorts
                alpha = cohort_color_alpha.get(cohort_order[ci_idx], 0.7)
                patch.set_facecolor(c)
                patch.set_alpha(alpha)
                # Hatch second cohort for visual distinction
                if ci_idx > 0:
                    patch.set_hatch("//")
            for flier in bp.get("fliers", []):
                flier.set_markersize(4)

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=55, ha="right", fontsize=7)

            # Legend for cohorts (using proxy patches)
            from matplotlib.patches import Patch
            legend_patches = []
            for ci_idx, cohort in enumerate(cohort_order):
                p = Patch(facecolor="#888888",
                          alpha=cohort_color_alpha.get(cohort, 0.7),
                          hatch="//" if ci_idx > 0 else "",
                          label=cohort)
                legend_patches.append(p)
            ax.legend(handles=legend_patches, fontsize=8, loc="best",
                      framealpha=0.8)

    ax.set_ylabel(ylabel, fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def benchmark_cohort_histograms(
    gene_rho_table: pd.DataFrame,
    name: str,
    *,
    cohort_order: Sequence[str],
    model_order: Sequence[str],
    title: Optional[str] = None,
) -> None:
    """Side-by-side cohort panels of overlaid per-gene Spearman rho KDEs.

    ``gene_rho_table`` has columns: cohort, model, spearman_rho.
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)
    n_panels = len(cohort_order)

    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels + 1, 5),
        sharex=True, sharey=False, squeeze=False,
    )

    for ci, cohort in enumerate(cohort_order):
        ax = axes[0, ci]
        sub = gene_rho_table[gene_rho_table["cohort"] == cohort]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
        else:
            legend_entries = []
            for model in model_order:
                vals = sub.loc[sub["model"] == model, "spearman_rho"].dropna().values
                if len(vals) < 3:
                    continue
                n_good = int((vals >= 0.70).sum())
                color = cmap.get(model, "#BBBBBB")
                sns.kdeplot(vals, ax=ax, color=color, linewidth=1.5,
                            fill=False, clip=(-1, 1))
                legend_entries.append(f"{model} (n≥0.7={n_good})")
            if legend_entries:
                ax.legend(legend_entries, fontsize=5, loc="upper left",
                          framealpha=0.6)
        ax.set_xlim(-1, 1)
        ax.set_title(cohort, fontsize=10, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("Density", fontsize=9)
        ax.set_xlabel("Spearman ρ", fontsize=8)

    fig.suptitle(title or "Per-Gene Spearman ρ Distribution",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def benchmark_cohort_metrics_lines(
    metrics_table: pd.DataFrame,
    name: str,
    *,
    cohort_order: Sequence[str],
    model_order: Sequence[str],
) -> None:
    """Multi-metric panel (2×3) showing scalar metrics per model per cohort.

    Similar to ``benchmark_ecdna_lines`` but the x-axis is the cohort
    (categorical) rather than eCDNA.  One dot per (model, cohort).
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)

    metric_defs = [
        ("gvw_rho", "GVW Spearman ρ"),
        ("mean_top10pct_gene_rho", "Mean ρ (Top 10% Var.)"),
        ("median_gene_rho", "Median Gene Spearman ρ"),
        ("mean_gene_rho", "Mean Gene Spearman ρ"),
        ("sample_rho_mean", "Mean Sample Spearman ρ"),
        ("rmse", "Mean Gene RMSE"),
    ]

    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), squeeze=False)

    legend_handles = []
    x_positions = list(range(len(cohort_order)))
    for idx, (col, label) in enumerate(metric_defs):
        ri, ci = divmod(idx, n_cols)
        ax = axes[ri, ci]
        for model in model_order:
            sub = metrics_table[metrics_table["model"] == model]
            if sub.empty or col not in sub.columns or sub[col].isna().all():
                continue
            color = cmap.get(model, "#BBBBBB")
            y_vals = []
            for cohort in cohort_order:
                v = sub.loc[sub["cohort"] == cohort, col].values
                y_vals.append(float(v[0]) if len(v) > 0 else np.nan)
            line, = ax.plot(x_positions, y_vals, "o-",
                            color=color, markersize=16, linewidth=1.4,
                            markeredgecolor=_darken(color),
                            markeredgewidth=1.2)
            if idx == 0:
                legend_handles.append((line, model))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(cohort_order, fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    if legend_handles:
        handles, labels = zip(*legend_handles)
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 6), fontsize=8,
                   framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Benchmark Metrics by Cohort",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Volcano plot (limma DE results)
# ---------------------------------------------------------------------------

def volcano_plot(
    de_results: pd.DataFrame,
    name: str,
    padj_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    contrast_name: str = "NEPC-ARPC",
    n_label: int = 20,
) -> None:
    """Volcano plot from limma topTable results.

    Args:
        de_results: DataFrame with logFC, adj.P.Val columns (gene names as index).
        name: Output file stem (saves {name}.pdf).
        padj_thresh: Adjusted p-value significance threshold.
        lfc_thresh: Absolute log2 fold-change threshold.
        contrast_name: Contrast string (group2-group1) for axis label.
        n_label: Number of top genes to label.
    """
    _ensure_dir(os.path.dirname(name))
    pal = _palette()

    # Parse contrast for colors: "NEPC-ARPC" → up_group=NEPC, down_group=ARPC
    parts = contrast_name.split("-", 1)
    up_group = parts[0]          # positive logFC direction
    down_group = parts[1] if len(parts) > 1 else "Down"
    up_color = pal.get(up_group, "#EE7733")
    down_color = pal.get(down_group, "#0077BB")
    ns_color = "#BBBBBB"

    logfc = de_results["logFC"].values
    neg_log_p = -np.log10(de_results["adj.P.Val"].clip(lower=1e-300).values)
    genes = de_results.index.astype(str).values

    sig_up = (logfc > lfc_thresh) & (de_results["adj.P.Val"].values < padj_thresh)
    sig_down = (logfc < -lfc_thresh) & (de_results["adj.P.Val"].values < padj_thresh)
    ns = ~(sig_up | sig_down)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(logfc[ns], neg_log_p[ns], c=ns_color, s=6, alpha=0.4,
               linewidths=0, rasterized=True, label="NS")
    ax.scatter(logfc[sig_down], neg_log_p[sig_down], c=down_color, s=14,
               alpha=0.7, linewidths=0, label=f"{down_group}-up ({sig_down.sum()})")
    ax.scatter(logfc[sig_up], neg_log_p[sig_up], c=up_color, s=14,
               alpha=0.7, linewidths=0, label=f"{up_group}-up ({sig_up.sum()})")

    # Threshold lines
    ax.axhline(-np.log10(padj_thresh), ls="--", lw=0.7, color="grey", alpha=0.6)
    ax.axvline(lfc_thresh, ls="--", lw=0.7, color="grey", alpha=0.6)
    ax.axvline(-lfc_thresh, ls="--", lw=0.7, color="grey", alpha=0.6)

    # Label top genes by adjusted p-value among significant
    sig_mask = sig_up | sig_down
    if sig_mask.any():
        sig_idx = np.where(sig_mask)[0]
        top_idx = sig_idx[np.argsort(de_results["adj.P.Val"].values[sig_idx])[:n_label]]
        for i in top_idx:
            ax.annotate(
                genes[i], (logfc[i], neg_log_p[i]),
                fontsize=6, alpha=0.85,
                textcoords="offset points", xytext=(4, 4),
                path_effects=[
                    PathEffects.withStroke(linewidth=2, foreground="white"),
                ],
            )

    ax.set_xlabel(f"log$_2$ FC ({contrast_name})", fontsize=12)
    ax.set_ylabel(r"$-\log_{10}$(adj. p-value)", fontsize=12)
    ax.set_title(f"Differential Expression: {contrast_name}", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Enrichment dot plot (GSEA results)
# ---------------------------------------------------------------------------

def enrichment_dot_plot(
    gsea_df: pd.DataFrame,
    name: str,
    contrast_name: str = "NEPC-ARPC",
    fdr_thresh: float = 0.25,
    n_top: int = 30,
    title: Optional[str] = None,
) -> None:
    """Dot plot of GSEA pathway enrichment results.

    Args:
        gsea_df: DataFrame with columns: Term, NES, FDR q-val (or fdr),
            and optionally FWER p-val.  One row per gene set.
        name: Output file stem (saves {name}.pdf).
        contrast_name: Contrast string for color assignment (e.g. "NEPC-ARPC").
        fdr_thresh: FDR threshold for significance annotation line.
        n_top: Max pathways to show (by FDR).
        title: Plot title (default: "GSEA: {contrast_name}").
    """
    _ensure_dir(os.path.dirname(name))
    pal = _palette()

    # Standardise column names (gseapy uses different names across versions)
    df = gsea_df.copy()

    # Find the first matching column for each required field
    _term_candidates = ["Term", "term", "Name"]
    _nes_candidates = ["NES", "nes"]
    _fdr_candidates = ["FDR q-val", "fdr", "FDR"]

    def _pick(cols: Sequence[str], candidates: list) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        return None

    orig_cols = list(df.columns)
    term_col = _pick(orig_cols, _term_candidates)
    nes_col = _pick(orig_cols, _nes_candidates)
    fdr_col = _pick(orig_cols, _fdr_candidates)

    if not all([term_col, nes_col, fdr_col]):
        warnings.warn(f"enrichment_dot_plot: missing required columns in {orig_cols}")
        return

    df = df.rename(columns={term_col: "term", nes_col: "nes", fdr_col: "fdr"})
    # Drop any remaining duplicate columns from the rename
    df = df.loc[:, ~df.columns.duplicated()]

    df["fdr"] = pd.to_numeric(df["fdr"], errors="coerce")
    df["nes"] = pd.to_numeric(df["nes"], errors="coerce")
    df = df.dropna(subset=["fdr", "nes"])
    if df.empty:
        return

    # Sort by FDR, take top n
    df = df.sort_values("fdr").head(n_top).copy()

    # Clean pathway names: strip HALLMARK_ prefix, replace _ with space
    df["label"] = (
        df["term"]
        .str.replace("^HALLMARK_", "", regex=True)
        .str.replace("_", " ")
    )
    # Sort by NES for visual grouping
    df = df.sort_values("nes")

    # Colors based on direction
    parts = contrast_name.split("-", 1)
    up_group = parts[0]
    down_group = parts[1] if len(parts) > 1 else "Down"
    up_color = pal.get(up_group, "#EE7733")
    down_color = pal.get(down_group, "#0077BB")

    colors = [up_color if nes > 0 else down_color for nes in df["nes"].values]

    # Dot size: -log10(FDR), clamped
    neg_log_fdr = -np.log10(df["fdr"].clip(lower=1e-10).values)
    # Scale to reasonable point sizes
    size_min, size_max = 30, 250
    if neg_log_fdr.max() > neg_log_fdr.min():
        sizes = size_min + (size_max - size_min) * (
            (neg_log_fdr - neg_log_fdr.min()) / (neg_log_fdr.max() - neg_log_fdr.min())
        )
    else:
        sizes = np.full_like(neg_log_fdr, (size_min + size_max) / 2)

    fig_height = max(4, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(7, fig_height))

    y_pos = np.arange(len(df))
    ax.scatter(df["nes"].values, y_pos, c=colors, s=sizes,
               alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].values, fontsize=8)
    ax.axvline(0, color="grey", ls="-", lw=0.6, alpha=0.5)

    # FDR threshold annotation
    sig_count = int((df["fdr"] < fdr_thresh).sum())
    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=11)
    ax.set_title(title or f"GSEA: {contrast_name}", fontsize=13, fontweight="bold")

    # Legend for direction
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=up_color,
               markersize=8, label=f"{up_group}-enriched"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=down_color,
               markersize=8, label=f"{down_group}-enriched"),
    ]
    # Size legend (small / large)
    for fdr_val, lbl in [(0.05, "FDR=0.05"), (0.001, "FDR=0.001")]:
        sz = size_min + (size_max - size_min) * (
            (-np.log10(fdr_val) - neg_log_fdr.min())
            / max(neg_log_fdr.max() - neg_log_fdr.min(), 1e-6)
        )
        sz = np.clip(sz, size_min, size_max)
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=np.sqrt(sz) * 0.6, label=lbl),
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
              framealpha=0.8, handletextpad=0.5)

    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Binary status boxplot (e.g. receptor status 0/1 vs gene expression)
# =====================================================================

def binary_status_boxplot(
    values_0: np.ndarray,
    values_1: np.ndarray,
    name: str,
    *,
    xlabel: str = "Status",
    ylabel: str = "Predicted Expression (log₂CPM+1)",
    title: str = "",
    label_0: str = "Negative (0)",
    label_1: str = "Positive (1)",
    pval: Optional[float] = None,
) -> None:
    """Paired boxplot with strip overlay for binary status comparison.

    Draws two boxes (status 0 and 1) with individual data points
    overlaid, and annotates with a p-value (typically Mann-Whitney U).
    """
    _ensure_dir(name)
    pal = load_palette()

    fig, ax = plt.subplots(figsize=(4, 5))

    data = [np.asarray(values_0, dtype=float),
            np.asarray(values_1, dtype=float)]
    positions = [0, 1]
    colors = ["#BBDDEE", "#EE9955"]

    bp = ax.boxplot(
        data, positions=positions, widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(markersize=0),
        showfliers=False,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")

    # Strip overlay
    for pos, vals, c in zip(positions, data, colors):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(pos + jitter, vals, color=c, edgecolor="black",
                   linewidth=0.6, s=40, zorder=4, alpha=0.85)

    ax.set_xticks(positions)
    ax.set_xticklabels([label_0, label_1], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    # P-value annotation
    if pval is not None:
        y_top = max(np.max(data[0]) if len(data[0]) else 0,
                    np.max(data[1]) if len(data[1]) else 0)
        y_ann = y_top + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.plot([0, 0, 1, 1], [y_ann - 0.02, y_ann, y_ann, y_ann - 0.02],
                lw=1.0, color="black")
        ax.text(0.5, y_ann + 0.01, format_pval(pval),
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Confusion matrix heatmap
# =====================================================================

def confusion_matrix_heatmap(
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    name: str,
    *,
    class_order: Optional[Sequence[str]] = None,
    title: str = "PAM50 Classification",
) -> None:
    """Plot a confusion matrix as an annotated heatmap.

    Rows = true, columns = predicted.  Numbers are absolute counts
    with row-normalised color shading.
    """
    _ensure_dir(name)

    labels_true = list(true_labels)
    labels_pred = list(pred_labels)
    if class_order is None:
        class_order = sorted(set(labels_true + labels_pred))
    n = len(class_order)
    label_to_idx = {l: i for i, l in enumerate(class_order)}

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels_true, labels_pred):
        ti = label_to_idx.get(t)
        pi = label_to_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1

    # Row-normalised for colour
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums > 0,
                        out=np.zeros_like(cm, dtype=float))

    fig, ax = plt.subplots(figsize=(max(4, n + 1), max(4, n + 1)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    # Annotate with counts
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(class_order, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_order, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Forest plot (hazard ratios with confidence intervals)
# =====================================================================

def forest_plot(
    hr_table: pd.DataFrame,
    name: str,
    *,
    hr_table_mv: Optional[pd.DataFrame] = None,
    title: str = "Forest Plot",
    hr_col: str = "HR",
    lo_col: str = "HR_lo",
    hi_col: str = "HR_hi",
    pval_col: str = "p",
    label_col: str = "variable",
    sig_threshold: float = 0.05,
) -> None:
    """Horizontal forest plot of hazard ratios with 95% CI.

    ``hr_table`` must have columns for the HR point estimate,
    lower/upper CI bounds, p-value, and variable label.

    When *hr_table_mv* is provided (multivariate results, e.g.
    TFx-adjusted), both univariate and multivariate HRs are shown:

    - **Univariate**: black (significant) / gray (non-significant)
    - **Multivariate**: maroon (significant) / faded maroon (non-sig)
    - **Significant** markers are diamonds at full saturation
    - **Non-significant** markers are circles at reduced saturation
    """
    _ensure_dir(name)

    df = hr_table.copy()
    df = df.sort_values(hr_col, ascending=True).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return

    has_mv = hr_table_mv is not None and not hr_table_mv.empty

    fig_height = max(4, 0.4 * n + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_pos = np.arange(n)
    hrs = df[hr_col].values.astype(float)
    lo = df[lo_col].values.astype(float)
    hi = df[hi_col].values.astype(float)
    pvals = df[pval_col].values.astype(float)

    # Build mv lookup by variable name
    mv_map: Dict[str, Dict] = {}
    if has_mv:
        for _, row in hr_table_mv.iterrows():
            mv_map[str(row[label_col])] = {
                "HR": float(row[hr_col]), "lo": float(row[lo_col]),
                "hi": float(row[hi_col]), "p": float(row[pval_col]),
            }

    # Vertical offset for uni vs mv when both present
    offset = 0.15 if has_mv else 0.0

    # --- Univariate ---
    for i in range(n):
        sig = pvals[i] < sig_threshold
        if has_mv:
            color = "#000000" if sig else "#AAAAAA"
            alpha = 1.0 if sig else 0.45
        else:
            color = "#CC3311" if sig else "#555555"
            alpha = 1.0
        y = y_pos[i] + offset
        ax.plot([lo[i], hi[i]], [y, y],
                color=color, lw=1.5, solid_capstyle="round", alpha=alpha)
        marker = "D" if sig else "o"
        ax.plot(hrs[i], y, marker, color=color, markersize=7,
                zorder=4, alpha=alpha)

    # --- Multivariate ---
    if has_mv:
        _MAROON = "#800000"
        _MAROON_FADED = "#CC9999"
        for i in range(n):
            var_name = str(df[label_col].iloc[i])
            mv = mv_map.get(var_name)
            if mv is None or np.isnan(mv["HR"]):
                continue
            sig = mv["p"] < sig_threshold
            color = _MAROON if sig else _MAROON_FADED
            alpha = 1.0 if sig else 0.45
            y = y_pos[i] - offset
            ax.plot([mv["lo"], mv["hi"]], [y, y],
                    color=color, lw=1.5, solid_capstyle="round", alpha=alpha)
            marker = "D" if sig else "o"
            ax.plot(mv["HR"], y, marker, color=color, markersize=7,
                    zorder=4, alpha=alpha)

    # Reference line at HR=1
    ax.axvline(1.0, color="grey", ls="--", lw=0.8, alpha=0.7)

    # Labels and p-values on right margin
    ax.set_yticks(y_pos)
    labels = df[label_col].values
    cleaned = [str(l).replace("HALLMARK_", "").replace("_", " ") for l in labels]
    ax.set_yticklabels(cleaned, fontsize=8)

    # P-value text on right side (univariate)
    all_hi = list(hi)
    if has_mv:
        all_hi.extend(v["hi"] for v in mv_map.values() if np.isfinite(v["hi"]))
    x_max = float(np.nanmax(all_hi)) * 1.3 if all_hi else 2.0
    for i in range(n):
        pv_str = format_pval(pvals[i])
        var_name = str(df[label_col].iloc[i])
        mv = mv_map.get(var_name)
        if has_mv and mv is not None and np.isfinite(mv["p"]):
            pv_str += f"  |  {format_pval(mv['p'])}"
        ax.text(x_max, y_pos[i], pv_str,
                va="center", fontsize=7, color="#555555")

    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.15)

    # Legend when multivariate is present
    if has_mv:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="#000000", marker="D", markersize=6,
                   lw=1.5, label="Univariate (sig.)"),
            Line2D([0], [0], color="#AAAAAA", marker="o", markersize=6,
                   lw=1.5, alpha=0.45, label="Univariate (n.s.)"),
            Line2D([0], [0], color=_MAROON, marker="D", markersize=6,
                   lw=1.5, label="Multivariate / TFx-adj. (sig.)"),
            Line2D([0], [0], color=_MAROON_FADED, marker="o", markersize=6,
                   lw=1.5, alpha=0.45, label="Multivariate / TFx-adj. (n.s.)"),
        ]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right",
                  framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Kaplan-Meier survival curve
# =====================================================================

def kaplan_meier_plot(
    time_lo: np.ndarray,
    event_lo: np.ndarray,
    time_hi: np.ndarray,
    event_hi: np.ndarray,
    name: str,
    *,
    label_lo: str = "Low",
    label_hi: str = "High",
    title: str = "Kaplan-Meier",
    logrank_p: Optional[float] = None,
) -> None:
    """Kaplan-Meier curves for two groups (low vs high).

    Uses the lifelines ``KaplanMeierFitter`` to compute and plot
    survival curves with at-risk counts and log-rank p-value.
    """
    from lifelines import KaplanMeierFitter

    _ensure_dir(name)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#0077BB", "#CC3311"]

    for arr_t, arr_e, label, color in [
        (time_lo, event_lo, label_lo, colors[0]),
        (time_hi, event_hi, label_hi, colors[1]),
    ]:
        kmf = KaplanMeierFitter()
        kmf.fit(arr_t, event_observed=arr_e, label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, lw=2)

    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    if logrank_p is not None:
        ax.text(0.98, 0.02, format_pval(logrank_p),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="white", alpha=0.8))

    ax.legend(fontsize=10, loc="best", framealpha=0.8)
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Clinical correlation scatter (gene expression vs clinical variable)
# =====================================================================

def clinical_correlation_scatter(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    name: str,
    *,
    xlabel: str = "Clinical Variable",
    ylabel: str = "Predicted Expression",
    title: str = "",
) -> None:
    """Scatter with regression line and correlation annotation.

    Displays both Pearson and Spearman correlation coefficients and
    a least-squares regression line with 95% confidence band.
    """
    _ensure_dir(name)

    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, color="#0077BB", edgecolor="white", s=50,
               alpha=0.8, linewidths=0.5)

    # Regression line
    from scipy.stats import linregress as _lr
    slope, intercept, _, _, _ = _lr(x, y)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#CC3311",
            lw=1.8, ls="-", alpha=0.8)

    # Correlation annotations
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    annot = (f"Pearson r = {r_p:.3f}  ({format_pval(p_p)})\n"
             f"Spearman ρ = {r_s:.3f}  ({format_pval(p_s)})")
    ax.text(0.03, 0.97, annot, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# ROC — on/off gene calling
# ---------------------------------------------------------------------------

def roc_gene_calling(
    roc_curves: Dict[str, Tuple],
    name: str,
    *,
    title: str = "ROC — Gene On/Off Calling",
    model_order: Sequence[str] = (),
) -> None:
    """Plot ROC curves for binary on/off gene expression classification.

    Parameters
    ----------
    roc_curves : dict
        ``{model_name: (fpr, tpr, auc, n)}`` where fpr/tpr are arrays,
        auc is a float, and n is the number of gene-sample pairs tested.
    name : str
        Output path (without .pdf extension).
    title : str
        Plot title.
    model_order : sequence[str]
        Preferred ordering; models not in this list appear at the end.
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Diagonal reference
    ax.plot([0, 1], [0, 1], ls="--", color="#BBBBBB", lw=1.2, zorder=0)

    # Sort models: preferential order, then alphabetical remainder
    ordered = [m for m in model_order if m in roc_curves]
    remaining = sorted(set(roc_curves) - set(ordered))
    ordered.extend(remaining)

    for model in ordered:
        entry = roc_curves[model]
        fpr, tpr, auc_val = entry[0], entry[1], entry[2]
        n_tested = entry[3] if len(entry) > 3 else None

        # CI-filtered lines inherit their parent model's color
        is_ci = "CI-filtered" in model
        if is_ci:
            parent = model.replace(" (CI-filtered)", "")
            color = cmap.get(parent, cmap.get(model, "#888888"))
        else:
            color = cmap.get(model, "#888888")

        lw = 2.5 if "Proteus" in model else 1.8
        ls = "--" if is_ci else "-"

        # Legend label: AUC + n
        lbl = f"{model} (AUC={auc_val:.3f}"
        if n_tested is not None:
            lbl += f", n={n_tested:,}"
        lbl += ")"

        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=lbl, zorder=2)

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Risk coverage curve (coverage vs PPV / FPR for gene on/off calling)
# ---------------------------------------------------------------------------


def risk_coverage_plot(
    coverage_data: Dict[str, Dict[str, np.ndarray]],
    name: str,
    *,
    title: str = "Risk Coverage — Gene On/Off Calling",
    model_order: Sequence[str] = (),
    proteus_models: Sequence[str] = (),
) -> None:
    """Combined coverage-vs-PPV and coverage-vs-FPR plot.

    Parameters
    ----------
    coverage_data : dict
        ``{model_name: {"coverage": array, "ppv": array, "fpr": array}}``.
        Proteus models have many points (smooth sweep); comparators have
        a single point at coverage=1.0.
    name : str
        Output path (without .pdf extension).
    title : str
        Plot title.
    model_order : sequence[str]
        Preferred plotting order.
    proteus_models : sequence[str]
        Model names that should be drawn as smooth lines (others as points).
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)

    fig, (ax_ppv, ax_fpr) = plt.subplots(1, 2, figsize=(14, 6))

    ordered = [m for m in model_order if m in coverage_data]
    remaining = sorted(set(coverage_data) - set(ordered))
    ordered.extend(remaining)

    for model in ordered:
        d = coverage_data[model]
        cov, ppv, fpr = d["coverage"], d["ppv"], d["fpr"]
        color = cmap.get(model, "#888888")
        is_proteus = model in proteus_models

        if is_proteus and len(cov) > 1:
            lw = 2.5
            ax_ppv.plot(cov, ppv, color=color, lw=lw, label=model, zorder=2)
            ax_fpr.plot(cov, fpr, color=color, lw=lw, label=model, zorder=2)
        else:
            # Single-point comparator at 100% coverage
            ms = 10
            ax_ppv.scatter(cov, ppv, color=color, s=ms**2,
                           edgecolors="black", linewidths=0.5,
                           marker="o", label=model, zorder=3)
            ax_fpr.scatter(cov, fpr, color=color, s=ms**2,
                           edgecolors="black", linewidths=0.5,
                           marker="o", label=model, zorder=3)

    for ax, ylabel, ax_title in [
        (ax_ppv, "PPV (Precision)", f"{title} — PPV"),
        (ax_fpr, "FPR", f"{title} — FPR"),
    ]:
        ax.set_xlabel("Coverage (fraction retained)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim([-0.02, 1.05])
        ax.set_ylim([-0.02, 1.05])
        ax.set_title(ax_title, fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# On/off confusion summary table (3-way: ON / OFF / INDETERMINATE)
# ---------------------------------------------------------------------------

def onoff_confusion_summary_plot(
    confusion_data: Dict[str, Dict[str, Any]],
    name: str,
    *,
    title: str = "On/Off Gene Calling — Confusion Summary",
    model_order: Sequence[str] = (),
) -> None:
    """Render a comparison table of on/off/indeterminate confusion metrics.

    The figure has two parts stacked vertically:

    1. **Stacked bar chart** — fraction ON / OFF / INDET per model.
    2. **Metrics table** — Accuracy, PPV, NPV, FPR, Coverage for each model.

    Parameters
    ----------
    confusion_data : dict
        ``{model_name: confusion_dict}`` where each value is the dict
        returned by ``compute_onoff_confusion()``.
    name : str
        Output path stem (without .pdf).
    title : str
        Figure title.
    model_order : sequence[str]
        Preferred left-to-right ordering.
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)

    ordered = [m for m in model_order if m in confusion_data]
    remaining = sorted(set(confusion_data) - set(ordered))
    ordered.extend(remaining)

    if not ordered:
        return

    # Collect data arrays
    models = []
    frac_on, frac_off, frac_indet = [], [], []
    acc, ppv, npv, fpr_arr, cov = [], [], [], [], []

    for m in ordered:
        d = confusion_data[m]
        n = d.get("n_total", 0)
        if n == 0:
            continue
        models.append(m)
        frac_on.append(d.get("n_on", 0) / n)
        frac_off.append(d.get("n_off", 0) / n)
        frac_indet.append(d.get("n_indet", 0) / n)
        acc.append(d.get("accuracy", float("nan")))
        ppv.append(d.get("ppv", float("nan")))
        npv.append(d.get("npv", float("nan")))
        fpr_arr.append(d.get("fpr", float("nan")))
        cov.append(d.get("coverage", float("nan")))

    if not models:
        return

    n_models = len(models)
    fig, (ax_bar, ax_tbl) = plt.subplots(
        2, 1, figsize=(max(8, n_models * 1.2), 7),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ── Stacked bar chart ──
    x = np.arange(n_models)
    bar_w = 0.6
    colors_on = "#2ca02c"    # green
    colors_off = "#1f77b4"   # blue
    colors_ind = "#ff7f0e"   # orange

    ax_bar.bar(x, frac_on, bar_w, label="ON (confident)", color=colors_on)
    ax_bar.bar(x, frac_off, bar_w, bottom=frac_on, label="OFF (confident)",
               color=colors_off)
    bottoms = [a + b for a, b in zip(frac_on, frac_off)]
    ax_bar.bar(x, frac_indet, bar_w, bottom=bottoms,
               label="Indeterminate", color=colors_ind)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax_bar.set_ylabel("Fraction of gene–sample pairs", fontsize=10)
    ax_bar.set_ylim([0, 1.05])
    ax_bar.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_bar.set_title(title, fontsize=11, fontweight="bold")

    # ── Metrics table ──
    ax_tbl.axis("off")
    col_labels = ["Model", "Accuracy", "PPV", "NPV", "FPR", "Coverage"]
    cell_text = []
    for i, m in enumerate(models):
        row = [
            m,
            f"{acc[i]:.3f}" if np.isfinite(acc[i]) else "—",
            f"{ppv[i]:.3f}" if np.isfinite(ppv[i]) else "—",
            f"{npv[i]:.3f}" if np.isfinite(npv[i]) else "—",
            f"{fpr_arr[i]:.3f}" if np.isfinite(fpr_arr[i]) else "—",
            f"{cov[i]:.1%}" if np.isfinite(cov[i]) else "—",
        ]
        cell_text.append(row)

    table = ax_tbl.table(
        cellText=cell_text, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#d9d9d9")
        table[0, j].set_text_props(fontweight="bold")

    # Color model column cells
    for i, m in enumerate(models):
        c = cmap.get(m, "#ffffff")
        table[i + 1, 0].set_facecolor(c + "40")  # 25% alpha hex

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    plt.close(fig)

