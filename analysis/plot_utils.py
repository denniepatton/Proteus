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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kstest, pearsonr, spearmanr

from .analysis_utils import (
    format_pval,
    load_palette,
    MIN_CORRELATION_N,
)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.dpi"] = 150

_PALETTE: Optional[Dict[str, str]] = None

_ONOFF_COLOR = "#E68613"
_ONOFF_OFF_COLOR = "#3B76AF"
_WITHHELD_COLOR = "#D9D9D9"
_SENSITIVITY_COLOR = "#2E8B57"
_SPECIFICITY_COLOR = "#3F3F3F"
_RESPONDER_COLOR = "#007191"
_NON_RESPONDER_COLOR = "#F97C00"
_PT_COLOR = "#D31F11"
_MIXED_TARGET_SCATTER_COLOR = "#999999"
_CROSSCOHORT_DRAW_ORDER: Tuple[str, ...] = (
    "CRPC-RA",
    "SCLC",
    "CRPC-WCDT",
    "BLCA-RA",
)
_CROSSCOHORT_DRAW_PRIORITY: Dict[str, int] = {
    cohort_name: idx for idx, cohort_name in enumerate(_CROSSCOHORT_DRAW_ORDER)
}


def _install_off_axis_legend_policy() -> None:
    """Default all Matplotlib legends to outside the plotting axes."""
    if getattr(Axes.legend, "_proteus_off_axis", False):
        return

    original_axes_legend = Axes.legend
    original_figure_legend = Figure.legend

    def axes_legend(self: Axes, *args: Any, **kwargs: Any) -> Legend:
        if "bbox_to_anchor" not in kwargs:
            kwargs["bbox_to_anchor"] = (1.04, 1.0)
            kwargs["loc"] = "upper left"
        kwargs.setdefault("borderaxespad", 0.0)
        return original_axes_legend(self, *args, **kwargs)

    def figure_legend(self: Figure, *args: Any, **kwargs: Any) -> Legend:
        if "bbox_to_anchor" not in kwargs:
            kwargs["bbox_to_anchor"] = (1.02, 0.5)
            kwargs["loc"] = "center left"
        kwargs.setdefault("borderaxespad", 0.0)
        return original_figure_legend(self, *args, **kwargs)

    axes_legend._proteus_off_axis = True  # type: ignore[attr-defined]
    figure_legend._proteus_off_axis = True  # type: ignore[attr-defined]
    Axes.legend = axes_legend  # type: ignore[method-assign]
    Figure.legend = figure_legend  # type: ignore[method-assign]


_install_off_axis_legend_policy()


def _palette() -> Dict[str, str]:
    global _PALETTE
    if _PALETTE is None:
        _PALETTE = load_palette()
    return _PALETTE


def _color(subtype: str) -> str:
    palette = _palette()
    subtype_text = str(subtype).strip()
    # Prefer an exact palette key so context-specific subtypes (e.g. the BLCA
    # "NE" entry) resolve to their own color rather than a short-name alias.
    if subtype_text in palette:
        return palette[subtype_text]
    alias_map = {
        "AR": "ARPC",
        "NE": "NEPC",
        "BLCA": "BLCA-RA",
    }
    alias = alias_map.get(subtype_text.upper())
    if alias is not None and alias in palette:
        return palette[alias]
    return palette.get(subtype_text, "#999999")


def _contrast_group_display_name(group_name: str) -> str:
    mapping = {
        "NonResponder": "Non-Responder",
        "Non-Responder": "Non-Responder",
        "Death": "PT",
    }
    return mapping.get(group_name, group_name)


def _contrast_group_color(group_name: str, fallback: str) -> str:
    mapping = {
        "Responder": _RESPONDER_COLOR,
        "NonResponder": _NON_RESPONDER_COLOR,
        "Non-Responder": _NON_RESPONDER_COLOR,
        "C1": _RESPONDER_COLOR,
        "Death": _PT_COLOR,
        "PT": _PT_COLOR,
    }
    return mapping.get(group_name, _palette().get(group_name, fallback))


def _collapse_crpc_target_scatter_subtype(subtype: str) -> Tuple[str, str]:
    subtype_text = str(subtype).strip().upper()
    if subtype_text == "ARPC":
        return "ARPC", _palette().get("ARPC", "#0077BB")
    if subtype_text == "NEPC":
        return "NEPC", _palette().get("NEPC", "#EE7733")
    return "Mixed", _MIXED_TARGET_SCATTER_COLOR


def _crosscohort_priority(group_name: Any) -> int:
    return _CROSSCOHORT_DRAW_PRIORITY.get(str(group_name), len(_CROSSCOHORT_DRAW_PRIORITY))


def _darken(hex_color: str, factor: float = 0.6) -> str:
    """Return a darker shade of *hex_color* by scaling RGB toward black."""
    c = mpl.colors.to_rgb(hex_color)
    return mpl.colors.to_hex(tuple(ch * factor for ch in c))


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _format_correlation_line(label: str, value: float, p_value: float) -> str:
    if not np.isfinite(value):
        return f"{label}=NA"
    p_text = format_pval(float(p_value)) if np.isfinite(p_value) else "p=NA"
    return f"{label}={value:.3f} ({p_text})"


def _set_explicit_ticks(
    ax: plt.Axes,
    *,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
) -> None:
    if x_ticks is not None:
        ax.set_xticks(list(x_ticks))
    if y_ticks is not None:
        ax.set_yticks(list(y_ticks))


def _nice_step_for_five_ticks(span: float) -> float:
    if not np.isfinite(span) or span <= 0.0:
        return 1.0
    target = span / 4.0
    steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 4.0, 5.0, 10.0, 20.0]
    for step in steps:
        if step >= target:
            return step
    return steps[-1]


def _set_five_y_ticks(ax: plt.Axes, *, lower: float = 0.0) -> None:
    y_hi = float(ax.get_ylim()[1])
    step = _nice_step_for_five_ticks(max(y_hi - lower, 1e-9))
    upper = lower + 4.0 * step
    ax.set_ylim(lower, upper)
    ax.set_yticks(np.linspace(lower, upper, 5))


def _compute_correlation_stats(
    x_vals: Sequence[float] | np.ndarray | pd.Series,
    y_vals: Sequence[float] | np.ndarray | pd.Series,
) -> Dict[str, float]:
    """Return Pearson/Spearman correlation statistics for aligned finite pairs."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < MIN_CORRELATION_N or float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "n": float(len(x)),
        }
    pearson_stat = pearsonr(x, y)
    spearman_stat = spearmanr(x, y)
    return {
        "pearson_r": float(pearson_stat.statistic),
        "pearson_p": float(pearson_stat.pvalue),
        "spearman_rho": float(spearman_stat.statistic),
        "spearman_p": float(spearman_stat.pvalue),
        "n": float(len(x)),
    }


def _tfx_marker_size(
    tfx_value: float,
    *,
    min_size: float = 40.0,
    scale: float = 260.0,
    default_size: float = 60.0,
) -> float:
    if np.isfinite(tfx_value):
        return max(min_size, scale * float(tfx_value))
    return default_size


def add_tfx_size_legend(
    ax: plt.Axes,
    tfx_values: Sequence[float] | np.ndarray | pd.Series,
    *,
    title: str = "TFx",
    bbox_to_anchor: Tuple[float, float] = (1.18, 0.28),
    loc: str = "center left",
    fontsize: float = 8,
    min_size: float = 40.0,
    scale: float = 260.0,
    default_size: float = 60.0,
) -> Optional[Legend]:
    numeric = pd.to_numeric(pd.Series(tfx_values), errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).any():
        return None

    legend_values = [0.1, 0.5, 1.0]

    handles = [
        ax.scatter(
            [],
            [],
            s=_tfx_marker_size(
                value,
                min_size=min_size,
                scale=scale,
                default_size=default_size,
            ),
            facecolors="#BDBDBD",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.78,
        )
        for value in legend_values
    ]
    labels = [f"{value:.1f}" for value in legend_values]
    return ax.legend(
        handles,
        labels,
        title=title,
        bbox_to_anchor=bbox_to_anchor,
        loc=loc,
        fontsize=fontsize,
        labelspacing=1.2,
        borderpad=0.7,
        handletextpad=1.0,
    )


def sample_scatter(
    truth_col: pd.Series,
    pred_col: pd.Series,
    sample_name: str,
    name: str,
    *,
    goi: Optional[Dict[str, List[str]]] = None,
    tfx: Optional[float] = None,
    depth: Optional[float] = None,
    is_holdout: bool = False,
    point_style: str = "scatter",
    point_color: str = "#7A7A7A",
    point_size: float = 7.0,
    axis_max: Optional[float] = None,
    diagonal_color: str = "black",
    diagonal_lw: float = 1.0,
    diagonal_alpha: float = 0.6,
    diagonal_zorder: float = 1.0,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
) -> None:
    """Scatter one sample's truth and predicted expression across genes."""
    _ensure_dir(name)
    truth = pd.to_numeric(pd.Series(truth_col), errors="coerce")
    pred = pd.to_numeric(pd.Series(pred_col), errors="coerce").reindex(truth.index)
    valid = truth.notna() & pred.notna()
    truth = truth.loc[valid]
    pred = pred.loc[valid]
    if len(truth) < MIN_CORRELATION_N:
        return

    corr_stats = _compute_correlation_stats(truth.values, pred.values)
    residual = pred.values - truth.values
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    if axis_max is None:
        max_value = float(np.nanmax([truth.max(), pred.max(), 1.0]))
        axis_max = max(15.0, np.ceil(max_value))

    fig, ax = plt.subplots(figsize=(6, 6))
    base_scatter = None
    if len(truth) >= 200:
        x_vals = truth.values.astype(np.float64, copy=False)
        y_vals = pred.values.astype(np.float64, copy=False)
        bins = min(150, max(60, int(np.sqrt(len(x_vals)))))
        hist, x_edges, y_edges = np.histogram2d(
            x_vals,
            y_vals,
            bins=bins,
            range=[[0.0, float(axis_max)], [0.0, float(axis_max)]],
        )
        x_idx = np.clip(np.searchsorted(x_edges, x_vals, side="right") - 1, 0, bins - 1)
        y_idx = np.clip(np.searchsorted(y_edges, y_vals, side="right") - 1, 0, bins - 1)
        counts = hist[x_idx, y_idx]
        density = np.log10(counts + 1.0)
        draw_order = np.argsort(density)
        base_scatter = ax.scatter(
            x_vals[draw_order],
            y_vals[draw_order],
            c=density[draw_order],
            cmap="viridis_r",
            s=point_size,
            alpha=0.9,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
    elif point_style == "hexbin":
        hb = ax.hexbin(
            truth.values,
            pred.values,
            gridsize=65,
            extent=(0, axis_max, 0, axis_max),
            cmap="Greys",
            mincnt=1,
        )
        base_scatter = hb
    else:
        base_scatter = ax.scatter(
            truth.values,
            pred.values,
            s=point_size,
            color=point_color,
            alpha=0.35,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

    if len(truth) >= 200 and base_scatter is not None:
        cbar = fig.colorbar(base_scatter, ax=ax, pad=0.02)
        cbar.set_label(r"$\log_{10}(N + 1)$")

    if goi:
        for subtype, genes in goi.items():
            present = [gene for gene in genes if gene in truth.index]
            if not present:
                continue
            gt = truth.loc[present]
            gp = pred.loc[present]
            color = _color(subtype)
            marker = "D" if is_holdout else "o"
            ax.scatter(
                gt.values,
                gp.values,
                s=300,
                edgecolors="black",
                linewidths=0.5,
                color=color,
                marker=marker,
                zorder=4,
            )
            for gene in present:
                ax.annotate(
                    gene,
                    (gt[gene], gp[gene]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=12,
                    color=color,
                    path_effects=[PathEffects.withStroke(linewidth=0.75, foreground="black")],
                    zorder=4,
                )

    ax.plot(
        [0, axis_max],
        [0, axis_max],
        ls="--",
        color=diagonal_color,
        alpha=diagonal_alpha,
        lw=diagonal_lw,
        zorder=diagonal_zorder,
    )
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    _set_explicit_ticks(ax, x_ticks=x_ticks, y_ticks=y_ticks)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"RNA-Seq log$_2$(CPM+1)")
    ax.set_ylabel(r"Predicted log$_2$(CPM+1)")

    status_tag = " [HO]" if is_holdout else ""
    info_lines = [f"{sample_name}{status_tag}"]
    if tfx is not None:
        info_lines[0] += f"  TFx={tfx:.2f}"
    if depth is not None:
        info_lines[0] += f"  depth={depth:.1f}x"
    info_lines.append(_format_correlation_line("Pearson r", corr_stats["pearson_r"], corr_stats["pearson_p"]))
    info_lines.append(_format_correlation_line("Spearman rho", corr_stats["spearman_rho"], corr_stats["spearman_p"]))
    info_lines.append(f"n={len(truth):,}  RMSE={rmse:.3f}  MAE={mae:.3f}")
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
    activity_odds_ratio_wide: Optional[pd.DataFrame] = None,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    active_confidence_wide: Optional[pd.DataFrame] = None,
    active_threshold: Optional[float] = None,
    bg_truth_wide: Optional[pd.DataFrame] = None,
    bg_pred_wide: Optional[pd.DataFrame] = None,
    dim_threshold_genes: bool = False,
) -> None:
    """Scatter: samples as dots for one gene, truth vs pred.

    Colored by subtype, sized by TFx.  Fit line + 95% CI band.
    When ``bg_truth_wide`` / ``bg_pred_wide`` are provided, HD-L
    background predictions are overlaid as gray diamonds.

    When ``dim_threshold_genes`` is True and AOR thresholds are available,
    samples inside the ambiguous band are drawn as hollow markers and
    excluded from fit-line / metric calculations.
    """
    _ensure_dir(name)
    if gene not in truth_wide.index or gene not in pred_wide.index:
        return

    common = truth_wide.columns.intersection(pred_wide.columns)
    t = truth_wide.loc[gene, common].astype(np.float64)
    p = pred_wide.loc[gene, common].astype(np.float64)
    mask = np.isfinite(t) & np.isfinite(p)
    t, p = t[mask], p[mask]

    if len(t) < MIN_CORRELATION_N:
        return

    activity_odds_ratio_gene: Optional[pd.Series] = None
    if activity_odds_ratio_wide is not None and gene in activity_odds_ratio_wide.index:
        activity_odds_ratio_gene = pd.to_numeric(
            activity_odds_ratio_wide.loc[gene, common].reindex(t.index),
            errors="coerce",
        )

    active_confidence_gene: Optional[pd.Series] = None
    if active_confidence_wide is not None and gene in active_confidence_wide.index:
        active_confidence_gene = pd.to_numeric(
            active_confidence_wide.loc[gene, common].reindex(t.index),
            errors="coerce",
        )

    is_low_confidence = pd.Series(False, index=t.index)
    if (
        dim_threshold_genes
        and activity_odds_ratio_gene is not None
        and lower_threshold is not None
        and upper_threshold is not None
        and np.isfinite(lower_threshold)
        and np.isfinite(upper_threshold)
    ):
        is_low_confidence = (
            (activity_odds_ratio_gene >= float(lower_threshold))
            & (activity_odds_ratio_gene <= float(upper_threshold))
        ).fillna(False)
    elif (
        dim_threshold_genes
        and active_confidence_gene is not None
        and active_threshold is not None
        and np.isfinite(active_threshold)
    ):
        confidence_threshold = float(np.clip(active_threshold, 0.5, 1.0))
        is_low_confidence = (
            (active_confidence_gene > (1.0 - confidence_threshold))
            & (active_confidence_gene < confidence_threshold)
        ).fillna(False)

    # Metrics computed on confident (non-low-confidence) subset only
    confident_mask = ~is_low_confidence
    t_conf = t[confident_mask]
    p_conf = p[confident_mask]
    def _has_spread(vals: pd.Series) -> bool:
        arr = vals.to_numpy(dtype=np.float64)
        return arr.size >= 2 and np.isfinite(arr).all() and float(np.ptp(arr)) > 0.0

    if dim_threshold_genes and bool(is_low_confidence.any()):
        metric_t = t_conf
        metric_p = p_conf
    else:
        metric_t = t_conf if len(t_conf) >= MIN_CORRELATION_N else t
        metric_p = p_conf if len(p_conf) >= MIN_CORRELATION_N else p
    corr_stats = _compute_correlation_stats(metric_t.values, metric_p.values)
    r_val = corr_stats["pearson_r"]
    p_val = corr_stats["pearson_p"]
    rho_val = corr_stats["spearman_rho"]
    rho_p_val = corr_stats["spearman_p"]
    rmse = (
        float(np.sqrt(np.mean((metric_t.values - metric_p.values) ** 2)))
        if len(metric_t) > 0 else np.nan
    )

    if dim_threshold_genes and bool(is_low_confidence.any()):
        fit_t = t_conf
        fit_p = p_conf
    else:
        fit_t = t_conf if len(t_conf) >= MIN_CORRELATION_N else t
        fit_p = p_conf if len(p_conf) >= MIN_CORRELATION_N else p
    can_fit = (
        _has_spread(fit_t)
        and _has_spread(fit_p)
        and len(fit_t) >= MIN_CORRELATION_N
        and np.isfinite(fit_p.to_numpy(dtype=np.float64)).all()
    )

    # Dynamic axis range from displayed values only.
    all_vals = np.concatenate([t.values, p.values])

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
    fit_fn = None
    x_line = np.linspace(axis_lo, axis_hi, 100)
    if can_fit:
        coeffs = np.polyfit(fit_t.values, fit_p.values, 1)
        fit_fn = np.poly1d(coeffs)
        ax.plot(x_line, np.clip(fit_fn(x_line), axis_lo, axis_hi), "k-", lw=1, zorder=1)

    # CI band (on confident subset)
    n = len(fit_t)
    if n > 3 and fit_fn is not None:
        from scipy.stats import t as t_dist
        x_vals = fit_t.values
        residuals = fit_p.values - fit_fn(x_vals)
        s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        mean_x = np.mean(x_vals)
        Sxx = np.sum((x_vals - mean_x) ** 2)
        if Sxx > 0.0 and np.isfinite(s_err):
            t_val = t_dist.ppf(0.975, n - 2)
            y_line = fit_fn(x_line)
            ci = t_val * s_err * np.sqrt(1 / n + (x_line - mean_x) ** 2 / Sxx)
            ax.fill_between(x_line, np.clip(y_line - ci, axis_lo, axis_hi),
                             np.clip(y_line + ci, axis_lo, axis_hi),
                             color="gray", alpha=0.25, zorder=0)

    # Plot points
    pal = _palette()
    seen = set()
    seen_colors: Dict[str, str] = {}
    default_color = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["#4477AA"])[0]
    sample_order = list(t.index)
    if subtypes_df is not None and "subtype" in subtypes_df.columns:
        sample_order = sorted(
            sample_order,
            key=lambda sample_name: (
                _crosscohort_priority(subtypes_df.loc[sample_name, "subtype"]) if sample_name in subtypes_df.index else len(_CROSSCOHORT_DRAW_PRIORITY),
                str(sample_name),
            ),
        )

    for sample in sample_order:
        sz = 200
        if tfx_dict is not None:
            sz = _tfx_marker_size(
                tfx_dict.get(sample, np.nan),
                min_size=30.0,
                scale=300.0,
                default_size=200.0,
            )
        color = default_color
        marker = "o"
        subtype = "Unknown"
        if subtypes_df is not None and sample in subtypes_df.index:
            subtype = subtypes_df.loc[sample, "subtype"] if "subtype" in subtypes_df.columns else "Unknown"
            if "gene_tumor_scatters_Targets_CRPC" in name:
                subtype, color = _collapse_crpc_target_scatter_subtype(str(subtype))
            else:
                color = pal.get(subtype, "#999999")
            seen.add(subtype)
            seen_colors[subtype] = color
            if "status" in subtypes_df.columns and subtypes_df.loc[sample, "status"] == "holdout":
                marker = "D"
        elif "gene_tumor_scatters_Targets_CRPC" in name:
            subtype = "Mixed"
            color = _MIXED_TARGET_SCATTER_COLOR
            seen.add(subtype)
            seen_colors[subtype] = color

        if bool(is_low_confidence.loc[sample]):
            ax.scatter(
                t[sample],
                p[sample],
                s=sz,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=2.0,
                alpha=1.0,
                zorder=4,
            )
        else:
            ax.scatter(
                t[sample],
                p[sample],
                s=sz,
                color=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.75,
                zorder=3,
            )
    subtype_legend = None
    if subtypes_df is not None:
        ordered_seen = sorted(
            seen,
            key=lambda subtype_name: (_crosscohort_priority(subtype_name), str(subtype_name)),
        )
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=seen_colors.get(s, pal.get(s, "#999999")), markersize=8, label=s)
            for s in ordered_seen
        ]
        if handles:
            subtype_legend = ax.legend(handles=handles, title="Subtype", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    tfx_legend = None
    if tfx_dict is not None:
        tfx_legend = add_tfx_size_legend(
            ax,
            [tfx_dict.get(sample, np.nan) for sample in t.index],
            bbox_to_anchor=(1.18, 0.28),
            min_size=30.0,
            scale=300.0,
            default_size=200.0,
        )
    if subtype_legend is not None and tfx_legend is not None:
        ax.add_artist(subtype_legend)

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
    n_low_conf = int(is_low_confidence.sum())
    rmse_text = f"RMSE = {rmse:.3f}" if np.isfinite(rmse) else "RMSE = NA"
    metric_lines = [
        _format_correlation_line("Spearman ρ", rho_val, rho_p_val),
        _format_correlation_line("Pearson r", r_val, p_val),
        f"n = {int(corr_stats['n'])}",
        rmse_text,
    ]
    if n_low_conf > 0:
        metric_lines.append(f"{n_low_conf} low-confidence excluded from fit/stats")
    metric_note = "\n".join(metric_lines)
    ax.text(
        0.05, 0.95, metric_note,
        transform=ax.transAxes, va="top", fontsize=9,
    )
    if n_low_conf > 0:
        note = []
        if n_low_conf > 0:
            note.append("Hollow = low-confidence")
        ax.text(
            0.95,
            0.05,
            " ; ".join(note),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
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
    truth_sort_genes: Optional[List[str]] = None,
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

    # Optional truth-based ordering for curated panels such as PC-Phenotyping.
    sorted_samples = sorted(common_samples)
    if truth_sort_genes:
        truth_order_genes = [g for g in truth_sort_genes if g in tw.index]
        if truth_order_genes:
            truth_order_scores = tw.loc[truth_order_genes, common_samples].mean(axis=0, skipna=True)
            sorted_samples = sorted(
                common_samples,
                key=lambda sample: (
                    float(truth_order_scores.get(sample, np.inf))
                    if np.isfinite(truth_order_scores.get(sample, np.nan)) else np.inf,
                    str(sample),
                ),
            )
    elif subtypes is not None:
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
            cb.set_label("ssGSEA Score" if gsea else r"log$_2$(CPM+1)", size=8)

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
        cbar_kws={"shrink": 0.6, "label": "ssGSEA Score" if gsea else r"log$_2$(CPM+1)"},
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
    ylim: Optional[Tuple[float, float]] = None,
    square: bool = False,
    log_y: bool = False,
    draw_stat_lines: bool = True,
    hist_color: Optional[str] = None,
    kde_color: Optional[str] = None,
    hist_alpha: float = 0.6,
    kde_linewidth: float = 1.5,
    reference_line_color: str = "gray",
    reference_line_lw: float = 0.8,
    reference_line_alpha: float = 0.6,
    reference_line_zorder: float = 5.0,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
    enforce_five_y_ticks: bool = False,
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
    if hist_color is None and kde_color is None and np.isclose(kde_linewidth, 1.5):
        sns.histplot(vals, kde=True, stat="density", bins=40, ax=ax)
    else:
        sns.histplot(
            vals,
            kde=False,
            stat="density",
            bins=40,
            ax=ax,
            color=hist_color,
            alpha=hist_alpha,
            edgecolor=None,
        )
        sns.kdeplot(
            vals,
            ax=ax,
            color=(kde_color or hist_color or "#4C72B0"),
            lw=kde_linewidth,
        )
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
        ax.axvline(
            0,
            color=reference_line_color,
            ls="--",
            lw=reference_line_lw,
            alpha=reference_line_alpha,
            zorder=reference_line_zorder,
        )
    ax.legend(fontsize=9)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if log_y:
        ax.set_yscale("log")
    if square:
        ax.set_box_aspect(1)
    _set_explicit_ticks(ax, x_ticks=x_ticks, y_ticks=y_ticks)
    if enforce_five_y_ticks and y_ticks is None and not log_y:
        _set_five_y_ticks(ax)
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


def plot_pit_histogram(
    pit_values: Sequence[float] | np.ndarray | pd.Series,
    name: str,
    *,
    title: Optional[str] = None,
    n_bins: int = 20,
) -> None:
    """Plot a PIT histogram against the uniform reference density."""
    _ensure_dir(name)

    pit = np.asarray(pit_values, dtype=np.float64)
    pit = pit[np.isfinite(pit)]
    pit = pit[(pit >= 0.0) & (pit <= 1.0)]
    if len(pit) < 10:
        return

    ks_result = kstest(pit, "uniform")

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ax.hist(
        pit,
        bins=bins,
        density=True,
        color="#4477AA",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.axhline(1.0, color="#CC3311", linestyle="--", linewidth=1.4, alpha=0.85)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Density")
    ax.set_title(title or "PIT Histogram")
    ax.grid(True, alpha=0.15, axis="y")
    ax.text(
        0.98,
        0.97,
        "\n".join([
            f"n = {len(pit):,}",
            f"mean = {float(np.mean(pit)):.3f}",
            f"var = {float(np.var(pit)):.3f}",
            f"KS = {float(ks_result.statistic):.3f}  ({format_pval(float(ks_result.pvalue))})",
        ]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_prediction_calibration_curve(
    raw_curve: pd.DataFrame,
    calibrated_curve: pd.DataFrame,
    name: str,
    *,
    title: Optional[str] = None,
) -> None:
    """Plot binned mean truth vs mean prediction before/after point calibration."""
    _ensure_dir(name)
    if raw_curve.empty and calibrated_curve.empty:
        return

    frames = [df for df in (raw_curve, calibrated_curve) if df is not None and not df.empty]
    mean_pred = np.concatenate([df["mean_pred"].to_numpy(dtype=np.float64) for df in frames])
    mean_truth = np.concatenate([df["mean_truth"].to_numpy(dtype=np.float64) for df in frames])
    lo = float(min(np.nanmin(mean_pred), np.nanmin(mean_truth)))
    hi = float(max(np.nanmax(mean_pred), np.nanmax(mean_truth)))
    pad = max(0.05 * (hi - lo), 0.1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5, label="Ideal")
    if raw_curve is not None and not raw_curve.empty:
        ax.plot(
            raw_curve["mean_pred"],
            raw_curve["mean_truth"],
            "o-",
            color="#0077BB",
            label="Raw",
            markersize=5,
        )
    if calibrated_curve is not None and not calibrated_curve.empty:
        ax.plot(
            calibrated_curve["mean_pred"],
            calibrated_curve["mean_truth"],
            "s-",
            color="#EE7733",
            label="Calibrated",
            markersize=5,
        )
    ax.set_xlabel("Binned mean prediction")
    ax.set_ylabel("Binned mean truth")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc="best")
    ax.set_title(title or "Point Prediction Calibration")

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_shrinkage_scatter(
    raw_values: pd.Series,
    shrunk_values: pd.Series,
    name: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Scatter raw vs shrunk calibration coefficients with an identity line."""
    _ensure_dir(name)
    raw = pd.to_numeric(raw_values, errors="coerce").to_numpy(dtype=np.float64)
    shrunk = pd.to_numeric(shrunk_values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(raw) & np.isfinite(shrunk)
    if int(valid.sum()) < MIN_CORRELATION_N:
        return

    raw = raw[valid]
    shrunk = shrunk[valid]
    corr_stats = _compute_correlation_stats(raw, shrunk)
    lo = float(min(np.nanmin(raw), np.nanmin(shrunk)))
    hi = float(max(np.nanmax(raw), np.nanmax(shrunk)))
    pad = max(0.05 * (hi - lo), 0.05)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(raw, shrunk, s=9, alpha=0.18, c="#0077BB", edgecolors="none", rasterized=True)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.2)
    ax.set_title(title)
    ax.text(
        0.05,
        0.95,
        "\n".join([
            _format_correlation_line(
                "Spearman ρ",
                corr_stats["spearman_rho"],
                corr_stats["spearman_p"],
            ),
            f"n = {int(corr_stats['n'])}",
        ]),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

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

    corr_stats = _compute_correlation_stats(x, y)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(x, y, s=6, alpha=0.3, color="#4477AA", edgecolors="none")
    ax.plot([-1, 1], [-1, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Predicted Gene Reliability (r̂)")
    ax.set_ylabel("True Holdout Pearson r")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_box_aspect(1)
    ax.text(
        0.05,
        0.95,
        "\n".join([
            _format_correlation_line(
                "Spearman ρ",
                corr_stats["spearman_rho"],
                corr_stats["spearman_p"],
            ),
            _format_correlation_line(
                "Pearson r",
                corr_stats["pearson_r"],
                corr_stats["pearson_p"],
            ),
            f"n = {int(corr_stats['n'])}",
        ]),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
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


def plot_reliability_weights(
    feature_columns: List[str],
    weights: np.ndarray,
    name: str,
    *,
    title: Optional[str] = None,
) -> None:
    """Horizontal bar chart of ridge regression feature weights.

    Shows which prediction features contribute most to the gene-level
    reliability model.
    """
    _ensure_dir(name)
    if len(feature_columns) == 0 or len(weights) == 0:
        return

    weight_series = pd.Series(weights, index=feature_columns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if weight_series.empty:
        return
    weight_series = weight_series.reindex(weight_series.abs().sort_values(ascending=True).index)

    fig_height = max(4.0, 0.35 * len(weight_series) + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    colors = np.where(weight_series.values >= 0.0, "#4477AA", "#CC6677")
    ax.barh(weight_series.index.astype(str), weight_series.values, color=colors, alpha=0.85)
    ax.axvline(0.0, color="#555555", linewidth=0.9)
    ax.set_xlabel("Standardized coefficient")
    ax.set_ylabel("Feature")
    ax.set_title(title or "Reliability Model Feature Weights")
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parallelised sample scatter generation
# ---------------------------------------------------------------------------

def _render_sample_scatter(args: Tuple) -> str:
    """Worker function for parallel sample scatter generation."""
    (truth_col, pred_col, sample_name, out_path,
     goi, tfx, depth, is_holdout, axis_max) = args
    sample_scatter(
        truth_col, pred_col, sample_name, out_path,
        goi=goi, tfx=tfx, depth=depth,
        is_holdout=is_holdout, axis_max=axis_max,
    )
    return sample_name


def batch_sample_scatters(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    out_dir: str,
    *,
    goi: Optional[Dict[str, List[str]]] = None,
    subtypes_df: Optional[pd.DataFrame] = None,
    n_workers: Optional[int] = None,
    axis_max: Optional[float] = None,
) -> None:
    """Generate per-sample scatter plots in parallel.

    ``goi`` maps subtype name → list of gene names (passed through to sample_scatter).
    If ``subtypes_df`` has a ``status`` column, samples with status == "holdout"
    will be drawn with hollow markers. ``axis_max`` forces a fixed square axis
    limit when provided.
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

        args_list.append((
            truth_wide[sample], pred_wide[sample], sample,
            os.path.join(out_dir, sample),
            goi, tfx, depth, is_holdout, axis_max,
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
    truth_wide, pred_wide, gene, out_path, subtypes_df, tfx_dict, ci_lo_wide, ci_hi_wide, activity_odds_ratio_wide, lower_threshold, upper_threshold, active_confidence_wide, active_threshold, bg_truth_wide, bg_pred_wide, dim_threshold_genes = args
    gene_target_scatter(
        truth_wide, pred_wide, gene, out_path,
        subtypes_df=subtypes_df, tfx_dict=tfx_dict,
        ci_lo_wide=ci_lo_wide, ci_hi_wide=ci_hi_wide,
        activity_odds_ratio_wide=activity_odds_ratio_wide,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        active_confidence_wide=active_confidence_wide,
        active_threshold=active_threshold,
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
    activity_odds_ratio_wide: Optional[pd.DataFrame] = None,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    active_confidence_wide: Optional[pd.DataFrame] = None,
    active_threshold: Optional[float] = None,
    bg_truth_wide: Optional[pd.DataFrame] = None,
    bg_pred_wide: Optional[pd.DataFrame] = None,
    dim_threshold_genes: bool = False,
    n_workers: Optional[int] = None,
) -> None:
    """Generate per-gene target scatter plots in parallel."""
    os.makedirs(out_dir, exist_ok=True)
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1))

    args_list = [
        (truth_wide, pred_wide, gene, os.path.join(out_dir, gene),
         subtypes_df, tfx_dict, ci_lo_wide, ci_hi_wide,
         activity_odds_ratio_wide, lower_threshold, upper_threshold,
         active_confidence_wide, active_threshold,
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

_BENCH_EXPLICIT_COLORS = {
    "Naive": "#8F8F8F",
    "Proteus": "#1F78B4",
}


def _bench_parent_model_name(model_name: str) -> str:
    for suffix in (
        " (adjusted)",
        " (uncertainty-aware)",
        " (uncertainty-aware, 90% CI)",
        " (uncertainty-aware, 50% CI)",
        " (uncertainty-informed)",
    ):
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


def _bench_display_label(model_name: str) -> str:
    return (
        model_name
        .replace(" (adjusted)", "\nAdjusted")
        .replace(" (uncertainty-aware, 90% CI)", "\nUncertainty-aware\n90% CI")
        .replace(" (uncertainty-aware, 50% CI)", "\nUncertainty-aware\n50% CI")
        .replace(" (uncertainty-aware)", "\nUncertainty-aware")
    )


def _bench_color_map(model_names: Sequence[str]) -> Dict[str, str]:
    pal = _palette()
    cmap: Dict[str, str] = {}
    fb_idx = 0
    for m in model_names:
        parent = _bench_parent_model_name(m)
        if parent in _BENCH_EXPLICIT_COLORS:
            cmap[m] = _BENCH_EXPLICIT_COLORS[parent]
        elif parent in pal:
            cmap[m] = pal[parent]
        else:
            cmap[m] = _BENCH_FALLBACK_COLORS[fb_idx % len(_BENCH_FALLBACK_COLORS)]
            fb_idx += 1
    return cmap


def benchmark_depth_lines(
    metrics_table: pd.DataFrame,
    name: str,
    *,
    depth: str,
    model_order: Sequence[str],
    metric_defs: Optional[Sequence[Tuple[str, str]]] = None,
    title: Optional[str] = None,
) -> None:
    """Line plots of one or more metrics vs TFx for a single sequencing depth.

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

    if metric_defs is None:
        metric_defs = [
            ("gvw_rho", "GVW Spearman \u03c1"),
            ("mean_top10pct_gene_rho", "Mean \u03c1 (Top 10% Var.)"),
            ("median_gene_rho", "Median Gene Spearman \u03c1"),
            ("n_genes_rho_ge070", "Genes with \u03c1\u22650.70"),
            ("sample_rho_mean", "Mean Sample Spearman \u03c1"),
            ("rmse", "Mean Gene RMSE"),
        ]

    if len(metric_defs) == 1:
        n_rows, n_cols = 1, 1
        fig_size = (6.5, 5.5)
    elif len(metric_defs) <= 2:
        n_rows, n_cols = 1, len(metric_defs)
        fig_size = (6 * n_cols, 5.5)
    else:
        n_cols = 2
        n_rows = int(np.ceil(len(metric_defs) / n_cols))
        fig_size = (12, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), squeeze=False)
    if fig_size != (12, 12):
        fig.set_size_inches(*fig_size)

    legend_handles = []
    preferred_draw_order = ["Naive", "EPIC", "XGBoost", "Proteus"]
    model_order_index = {model_name: idx for idx, model_name in enumerate(model_order)}

    def _draw_priority(model_name: str) -> tuple[int, int]:
        lower_name = model_name.lower()
        if lower_name == "naive":
            return (0, model_order_index.get(model_name, len(model_order_index)))
        if "epic" in lower_name:
            return (1, model_order_index.get(model_name, len(model_order_index)))
        if "xgboost" in lower_name:
            return (2, model_order_index.get(model_name, len(model_order_index)))
        if "proteus" in lower_name:
            return (3, model_order_index.get(model_name, len(model_order_index)))
        return (len(preferred_draw_order), model_order_index.get(model_name, len(model_order_index)))

    ordered_models = sorted(model_order, key=_draw_priority)
    for idx, (col, label) in enumerate(metric_defs):
        ri, ci = divmod(idx, n_cols)
        ax = axes[ri, ci]
        for model in ordered_models:
            msub = sub[sub["model"] == model].sort_values("tfx_numeric")
            if msub.empty or col not in msub.columns or msub[col].isna().all():
                continue
            color = cmap.get(model, "#BBBBBB")
            line, = ax.plot(msub["tfx_numeric"], msub[col], "o-",
                            color=color, markersize=8, linewidth=1.4,
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

    for idx in range(len(metric_defs), n_rows * n_cols):
        ri, ci = divmod(idx, n_cols)
        axes[ri, ci].axis("off")

    if legend_handles:
        handles, labels = zip(*legend_handles)
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 6), fontsize=8,
                   framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title or f"Benchmark Metrics vs TFx — {depth} depth",
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
                ax.plot(x, y, marker=marker, markersize=14,
                        color=color, markeredgecolor=_darken(color),
                    markeredgewidth=1.4, alpha=alpha, zorder=3,
                        linestyle="none")

        # Legend: proxy patches
        legend_patches = []
        for ci_idx, cohort in enumerate(cohort_order):
            marker = _cohort_markers[ci_idx % len(_cohort_markers)]
            p = ax.plot([], [], marker=marker, markersize=10,
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

            rng = np.random.default_rng(42)
            for pos, vals, color in zip(all_positions, all_data, all_colors):
                numeric_vals = np.asarray(vals, dtype=float)
                numeric_vals = numeric_vals[np.isfinite(numeric_vals)]
                if numeric_vals.size == 0:
                    continue
                jitter = rng.uniform(-0.08, 0.08, size=numeric_vals.size)
                ax.scatter(
                    float(pos) + jitter,
                    numeric_vals,
                    color=color,
                    edgecolor="black",
                    linewidth=0.4,
                    s=28,
                    alpha=0.85,
                    zorder=3,
                )

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


def benchmark_cohort_retention_curve(
    gene_rho_table: pd.DataFrame,
    name: str,
    *,
    cohort_order: Sequence[str],
    model_order: Sequence[str],
    title: Optional[str] = None,
) -> None:
    """Retention curve of mean per-gene Spearman rho vs retained genes.

    Each curve is ranked by per-gene Spearman rho within a cohort/model pair,
    then plotted as the cumulative mean of the retained top-k genes.
    Model identity is encoded by color; cohort identity is encoded by marker
    shape and line style.
    """
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)
    cohort_style_map = {
        "BLCA-RA": {"marker": "o", "linestyle": "-"},
        "CRPC-WCDT": {"marker": "s", "linestyle": ":"},
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plotted_any = False
    max_retained = 0

    for model in model_order:
        model_color = cmap.get(model, "#BBBBBB")
        for cohort in cohort_order:
            sub = gene_rho_table[
                (gene_rho_table["cohort"] == cohort)
                & (gene_rho_table["model"] == model)
            ]
            rho_vals = pd.to_numeric(sub["spearman_rho"], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if rho_vals.size == 0:
                continue
            rho_vals = np.sort(rho_vals)[::-1]
            retained = np.arange(1, rho_vals.size + 1, dtype=np.int32)
            mean_rho = np.cumsum(rho_vals) / retained
            max_retained = max(max_retained, int(retained[-1]))

            style = cohort_style_map.get(cohort, {"marker": "o", "linestyle": "-"})
            markevery = max(1, int(rho_vals.size // 40))
            ax.plot(
                retained,
                mean_rho,
                color=model_color,
                linewidth=1.6,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=3.5,
                markevery=markevery,
                alpha=0.95,
                zorder=2,
            )
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    model_handles = [
        Line2D([0], [0], color=cmap.get(model, "#BBBBBB"), linewidth=2.0, label=model)
        for model in model_order
    ]
    cohort_handles = [
        Line2D(
            [0], [0],
            color="#444444",
            linestyle=cohort_style_map.get(cohort, {"linestyle": "-"})["linestyle"],
            marker=cohort_style_map.get(cohort, {"marker": "o"})["marker"],
            markersize=6,
            linewidth=1.6,
            label=cohort,
        )
        for cohort in cohort_order
    ]

    model_legend = ax.legend(
        handles=model_handles,
        title="Model",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=cohort_handles,
        title="Cohort",
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.46),
    )

    ax.set_xlabel("Retained Genes")
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_xlim(1, max_retained)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(title or "Gene Retention Curve")
    ax.grid(True, alpha=0.2)

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
        n_label: Ignored; volcano plots no longer label genes directly.
    """
    _ensure_dir(os.path.dirname(name))
    pal = _palette()

    # Parse contrast for colors: "NEPC-ARPC" → up_group=NEPC, down_group=ARPC
    parts = contrast_name.split("-", 1)
    up_group = parts[0]          # positive logFC direction
    down_group = parts[1] if len(parts) > 1 else "Down"
    up_label = _contrast_group_display_name(up_group)
    down_label = _contrast_group_display_name(down_group)
    display_contrast = f"{up_label}-{down_label}"
    up_color = _contrast_group_color(up_group, "#EE7733")
    down_color = _contrast_group_color(down_group, "#0077BB")
    ns_color = "#BBBBBB"

    logfc = de_results["logFC"].values
    neg_log_p = -np.log10(de_results["adj.P.Val"].clip(lower=1e-300).values)
    sig_up = (logfc > lfc_thresh) & (de_results["adj.P.Val"].values < padj_thresh)
    sig_down = (logfc < -lfc_thresh) & (de_results["adj.P.Val"].values < padj_thresh)
    ns = ~(sig_up | sig_down)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(logfc[ns], neg_log_p[ns], c=ns_color, s=24, alpha=0.4,
               linewidths=0, rasterized=True, label="NS")
    ax.scatter(logfc[sig_down], neg_log_p[sig_down], c=down_color, s=84,
               alpha=0.7, linewidths=0, label=f"{down_label}-up ({sig_down.sum()})")
    ax.scatter(logfc[sig_up], neg_log_p[sig_up], c=up_color, s=84,
               alpha=0.7, linewidths=0, label=f"{up_label}-up ({sig_up.sum()})")

    # Threshold lines
    ax.axhline(-np.log10(padj_thresh), ls="--", lw=0.7, color="grey", alpha=0.6)
    ax.axvline(lfc_thresh, ls="--", lw=0.7, color="grey", alpha=0.6)
    ax.axvline(-lfc_thresh, ls="--", lw=0.7, color="grey", alpha=0.6)

    ax.set_xlabel(f"log$_2$ FC ({display_contrast})", fontsize=12)
    ax.set_ylabel(r"$-\log_{10}$(adj. p-value)", fontsize=12)
    ax.set_title(f"Differential Expression: {display_contrast}", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, framealpha=0.8)
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
    up_label = _contrast_group_display_name(up_group)
    down_label = _contrast_group_display_name(down_group)
    display_contrast = f"{up_label}-{down_label}"
    up_color = _contrast_group_color(up_group, "#EE7733")
    down_color = _contrast_group_color(down_group, "#0077BB")

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

    max_label_chars = int(df["label"].astype(str).map(len).max()) if not df.empty else 0
    left_margin_in = min(4.0, max(1.6, 0.06 * max_label_chars + 0.35))
    right_margin_in = 1.45
    axis_width_in = 1.0
    top_margin_in = 0.55
    bottom_margin_in = 0.35
    axis_height_in = max(1.0, 0.22 * len(df))

    fig_width = left_margin_in + axis_width_in + right_margin_in
    fig_height = top_margin_in + axis_height_in + bottom_margin_in
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_position([
        left_margin_in / fig_width,
        bottom_margin_in / fig_height,
        axis_width_in / fig_width,
        axis_height_in / fig_height,
    ])

    # Fill encodes significance so the dot SIZE (a within-plot relative
    # -log10(FDR) rescaling, not an absolute scale) is not misread as
    # significance: significant pathways (FDR < threshold) are filled, others are
    # drawn hollow (open outline in the direction color).
    significant = df["fdr"].values < fdr_thresh
    sig_count = int(significant.sum())
    face_colors = [color if is_sig else "none" for color, is_sig in zip(colors, significant)]
    edge_colors = ["white" if is_sig else color for color, is_sig in zip(colors, significant)]
    line_widths = [0.5 if is_sig else 1.1 for is_sig in significant]

    y_pos = np.arange(len(df))
    ax.scatter(df["nes"].values, y_pos, facecolors=face_colors, s=sizes,
               alpha=0.85, edgecolors=edge_colors, linewidths=line_widths)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"].values, fontsize=8)
    ax.axvline(0, color="grey", ls="-", lw=0.6, alpha=0.5)

    # FDR threshold annotation
    x_values = pd.to_numeric(df["nes"], errors="coerce").to_numpy(dtype=np.float64)
    finite_x = x_values[np.isfinite(x_values)]
    if finite_x.size > 0:
        x_min = float(np.min(finite_x))
        x_max = float(np.max(finite_x))
        x_span = x_max - x_min
        if x_span <= 0.0:
            x_span = max(1.0, abs(x_min) * 0.1)
            x_min -= 0.5 * x_span
            x_max += 0.5 * x_span
        max_marker_radius_points = float(np.sqrt(float(np.max(sizes)) / np.pi))
        marker_pad_data = (max_marker_radius_points / 72.0) * (x_span / axis_width_in)
        base_pad_data = 0.03 * x_span
        x_pad = marker_pad_data + base_pad_data
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.margins(x=0.0)

    ax.set_xlabel("Normalized Enrichment Score (NES)", fontsize=11)
    ax.set_title(title or f"GSEA: {display_contrast}", fontsize=13, fontweight="bold")

    # Legend for direction
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=up_color,
             markersize=8, label=f"{up_label}-enriched"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=down_color,
             markersize=8, label=f"{down_label}-enriched"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markeredgecolor="white", markersize=8,
               label=f"FDR < {fdr_thresh:g} (n={sig_count})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="#888888", markersize=8, label="n.s."),
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
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
              framealpha=0.8, handletextpad=0.5)

    ax.grid(True, axis="x", alpha=0.2)
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
    mv_only: bool = False,
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

    When *mv_only* is True, only the multivariate/TFx-adjusted HRs are
    drawn (the univariate series is omitted). ``hr_table`` still supplies
    the variable ordering and labels; a multivariate table is required.
    """
    _ensure_dir(name)

    df = hr_table.copy()
    has_mv = hr_table_mv is not None and not hr_table_mv.empty
    if mv_only and not has_mv:
        return
    if mv_only:
        # TFx-adjusted-only plots order rows by the adjusted (multivariate) HR
        # rather than inheriting the univariate ordering used by the combined
        # plot. Variables absent from the multivariate table cannot be drawn.
        mv_hr_lookup = hr_table_mv.set_index(label_col)[hr_col]
        df = df[df[label_col].astype(str).isin(mv_hr_lookup.index.astype(str))].copy()
        df["_mv_hr_sort"] = df[label_col].map(mv_hr_lookup)
        df = (
            df.sort_values("_mv_hr_sort", ascending=True)
            .drop(columns="_mv_hr_sort")
            .reset_index(drop=True)
        )
    else:
        df = df.sort_values(hr_col, ascending=True).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return

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

    # Vertical offset for uni vs mv when both present. In mv-only mode the
    # multivariate markers are centred on each row.
    offset = 0.0 if mv_only else (0.15 if has_mv else 0.0)

    # --- Univariate ---
    if not mv_only:
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

    # P-value text on right side
    if mv_only:
        all_hi = [v["hi"] for v in mv_map.values() if np.isfinite(v["hi"])]
    else:
        all_hi = list(hi)
        if has_mv:
            all_hi.extend(v["hi"] for v in mv_map.values() if np.isfinite(v["hi"]))
    x_max = float(np.nanmax(all_hi)) * 1.3 if all_hi else 2.0
    for i in range(n):
        var_name = str(df[label_col].iloc[i])
        mv = mv_map.get(var_name)
        if mv_only:
            if mv is None or not np.isfinite(mv["p"]):
                continue
            pv_str = format_pval(mv["p"])
        else:
            pv_str = format_pval(pvals[i])
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
        if mv_only:
            legend_elements = [
                Line2D([0], [0], color=_MAROON, marker="D", markersize=6,
                       lw=1.5, label="Multivariate / TFx-adj. (sig.)"),
                Line2D([0], [0], color=_MAROON_FADED, marker="o", markersize=6,
                       lw=1.5, alpha=0.45, label="Multivariate / TFx-adj. (n.s.)"),
            ]
        else:
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
    survival curves with confidence bands and an optional log-rank p-value.
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
    # Force a 1:1 visual aspect for the plotting box (not the figure) so the two
    # data axes get equal on-screen length regardless of their differing units.
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Clinical correlation scatter (gene expression vs clinical variable)
# =====================================================================

def clinical_correlation_scatter(
    x_vals: Sequence[float] | np.ndarray | pd.Series,
    y_vals: Sequence[float] | np.ndarray | pd.Series,
    name: str,
    *,
    xlabel: str = "Clinical Variable",
    ylabel: str = "Predicted Expression",
    title: str = "",
    sample_meta: Optional[pd.DataFrame] = None,
    color_column: str = "cohort",
    legend_title: str = "Cohort",
    x_log10: bool = False,
    y_log10: bool = False,
) -> None:
    """Scatter with regression line and correlation annotation.

    When sample metadata are provided, points follow the shared scatter style:
    cohort coloring, TFx-sized markers, and a square plotting panel. Optional
    log10 transforms are applied directly to the variables before fitting.
    """
    _ensure_dir(name)

    x_series = pd.to_numeric(pd.Series(x_vals), errors="coerce")
    y_series = pd.to_numeric(pd.Series(y_vals), errors="coerce")
    if isinstance(x_vals, pd.Series):
        x_series.index = x_vals.index
    if isinstance(y_vals, pd.Series):
        y_series.index = y_vals.index

    shared = x_series.index.intersection(y_series.index)
    meta = None
    if sample_meta is not None:
        shared = shared.intersection(sample_meta.index)
        meta = sample_meta.loc[shared].copy()

    if len(shared) < MIN_CORRELATION_N:
        return

    x_series = x_series.loc[shared]
    y_series = y_series.loc[shared]

    mask = np.isfinite(x_series.values) & np.isfinite(y_series.values)
    if x_log10:
        mask &= x_series.values > 0.0
    if y_log10:
        mask &= y_series.values > 0.0
    if int(mask.sum()) < MIN_CORRELATION_N:
        return

    shared = shared[mask]
    x_series = x_series.loc[shared]
    y_series = y_series.loc[shared]
    if meta is not None:
        meta = meta.loc[shared]

    if x_log10:
        x_series = pd.Series(np.log10(x_series.values), index=shared)
    if y_log10:
        y_series = pd.Series(np.log10(y_series.values), index=shared)

    corr_stats = _compute_correlation_stats(x_series.values, y_series.values)
    if not np.isfinite(corr_stats["pearson_r"]) or not np.isfinite(corr_stats["spearman_rho"]):
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    seen_groups: Dict[str, str] = {}
    if meta is not None and color_column in meta.columns:
        palette = _palette()
        for sample_name in shared:
            group_name = str(meta.loc[sample_name, color_column])
            tfx_val = (
                pd.to_numeric(pd.Series([meta.loc[sample_name, "tfx"]]), errors="coerce").iloc[0]
                if "tfx" in meta.columns
                else np.nan
            )
            marker = (
                "D"
                if "status" in meta.columns and str(meta.loc[sample_name, "status"]) == "holdout"
                else "o"
            )
            color = palette.get(group_name, "#999999")
            size = _tfx_marker_size(tfx_val)
            ax.scatter(
                float(x_series.loc[sample_name]),
                float(y_series.loc[sample_name]),
                s=size,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.78,
            )
            seen_groups[group_name] = color
    else:
        ax.scatter(
            x_series.values,
            y_series.values,
            color="#0077BB",
            edgecolor="black",
            s=60,
            alpha=0.8,
            linewidths=0.5,
        )

    # Regression line
    from scipy.stats import linregress as _lr
    slope, intercept, _, _, _ = _lr(x_series.values, y_series.values)
    x_line = np.linspace(float(np.min(x_series.values)), float(np.max(x_series.values)), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#CC3311",
            lw=1.8, ls="-", alpha=0.8)

    # Correlation annotations
    annot = (
        f"Pearson r = {corr_stats['pearson_r']:.3f}  ({format_pval(corr_stats['pearson_p'])})\n"
        f"Spearman ρ = {corr_stats['spearman_rho']:.3f}  ({format_pval(corr_stats['spearman_p'])})\n"
        f"n = {int(corr_stats['n'])}"
    )
    ax.text(0.03, 0.97, annot, transform=ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    group_legend = None
    if seen_groups:
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", label=group_name,
                markerfacecolor=color, markeredgecolor="black", markersize=8,
            )
            for group_name, color in sorted(seen_groups.items())
        ]
        group_legend = ax.legend(
            handles=handles,
            title=legend_title,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            fontsize=8,
        )
    tfx_legend = None
    if meta is not None and "tfx" in meta.columns:
        tfx_legend = add_tfx_size_legend(ax, meta["tfx"], bbox_to_anchor=(1.18, 0.28))
    if group_legend is not None and tfx_legend is not None:
        ax.add_artist(group_legend)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_box_aspect(1)

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
    tick_values = np.linspace(0.0, 1.0, 5)

    fig, ax = plt.subplots(figsize=(7.8, 7.8))

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
        color = cmap.get(model, "#888888")
        lw = 2.5 if "Proteus" in model else 1.8

        # Legend label: AUC + n
        lbl = f"{model} (AUC={auc_val:.3f}"
        if n_tested is not None:
            lbl += f", n={n_tested:,}"
        lbl += ")"

        ax.plot(fpr, tpr, color=color, lw=lw,
                label=lbl, zorder=2)

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    _set_explicit_ticks(ax, x_ticks=tick_values, y_ticks=tick_values)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.set_box_aspect(1)

    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
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
        cov = np.asarray(d["coverage"], dtype=np.float64)
        ppv = np.asarray(d["ppv"], dtype=np.float64)
        fpr = np.asarray(d["fpr"], dtype=np.float64)
        order_idx = np.argsort(cov)
        cov = cov[order_idx]
        ppv = ppv[order_idx]
        fpr = fpr[order_idx]
        color = cmap.get(model, "#888888")
        is_proteus = model in proteus_models
        ppv_auc = d.get("ppv_auc", float("nan"))
        fpr_auc = d.get("fpr_auc", float("nan"))

        ppv_label = model
        fpr_label = model
        if is_proteus and np.isfinite(ppv_auc):
            ppv_label = f"{model} (AUC={ppv_auc:.3f})"
        if is_proteus and np.isfinite(fpr_auc):
            fpr_label = f"{model} (AUC={fpr_auc:.3f})"

        if is_proteus and len(cov) > 1:
            lw = 2.5
            ax_ppv.plot(cov, ppv, color=color, lw=lw, label=ppv_label, zorder=2)
            ax_fpr.plot(cov, fpr, color=color, lw=lw, label=fpr_label, zorder=2)
        else:
            # Single-point comparator at 100% coverage
            ms = 10
            ax_ppv.scatter(cov, ppv, color=color, s=ms**2,
                           edgecolors="black", linewidths=0.5,
                           marker="o", label=ppv_label, zorder=3)
            ax_fpr.scatter(cov, fpr, color=color, s=ms**2,
                           edgecolors="black", linewidths=0.5,
                           marker="o", label=fpr_label, zorder=3)

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


def _confidence_threshold_plot(
    threshold_data: Dict[str, Dict[str, Any]],
    name: str,
    *,
    title: str,
    model_order: Sequence[str] = (),
    primary_key: str,
    primary_label: str,
    secondary_key: str,
    secondary_label: str,
    target_key: str,
    selected_primary_key: str,
    selected_secondary_key: str,
    threshold_xlabel: str,
    coverage_ylabel: str,
) -> None:
    """Plot threshold sweeps for one selective-calling metric family."""
    _ensure_dir(name)
    cmap = _bench_color_map(model_order)

    fig, (ax_perf, ax_cov) = plt.subplots(1, 2, figsize=(14, 6))

    ordered = [m for m in model_order if m in threshold_data]
    ordered.extend(sorted(set(threshold_data) - set(ordered)))

    finite_thresholds: List[np.ndarray] = []

    for model in ordered:
        d = threshold_data[model]
        thresholds = np.asarray(d.get("threshold", []), dtype=np.float64)
        if thresholds.size == 0:
            continue
        finite_thresholds.append(thresholds[np.isfinite(thresholds)])
        primary_values = np.asarray(d.get(primary_key, []), dtype=np.float64)
        secondary_values = np.asarray(d.get(secondary_key, []), dtype=np.float64)
        coverage = np.asarray(d.get("coverage", []), dtype=np.float64)
        color = cmap.get(model, "#888888")

        ax_perf.plot(thresholds, primary_values, color=color, lw=2.5, label=f"{model} {primary_label}")
        ax_perf.plot(thresholds, secondary_values, color=color, lw=1.6, linestyle="--", label=f"{model} {secondary_label}")
        ax_cov.plot(thresholds, coverage, color=color, lw=2.5, label=model)

        selected_threshold = d.get("selected_threshold")
        if selected_threshold is not None and np.isfinite(selected_threshold):
            selected_primary = d.get(selected_primary_key, float("nan"))
            selected_secondary = d.get(selected_secondary_key, float("nan"))
            selected_coverage = d.get("selected_coverage", float("nan"))
            ax_perf.scatter([selected_threshold], [selected_primary], color=color, s=80, edgecolors="black", linewidths=0.5, zorder=4)
            ax_perf.scatter([selected_threshold], [selected_secondary], color=color, s=70, edgecolors="black", linewidths=0.5, marker="s", zorder=4)
            ax_cov.scatter([selected_threshold], [selected_coverage], color=color, s=80, edgecolors="black", linewidths=0.5, zorder=4)

    target_vals = [d.get(target_key) for d in threshold_data.values() if np.isfinite(d.get(target_key, np.nan))]
    if target_vals:
        ax_perf.axhline(float(target_vals[0]), color="#444444", linestyle=":", linewidth=1.5)

    if finite_thresholds:
        concatenated_thresholds = np.concatenate([vals for vals in finite_thresholds if vals.size > 0])
        threshold_min = float(np.nanmin(concatenated_thresholds))
        threshold_max = float(np.nanmax(concatenated_thresholds))
        if np.isclose(threshold_min, threshold_max):
            threshold_pad = max(0.1, abs(threshold_min) * 0.05)
        else:
            threshold_pad = max(0.05, (threshold_max - threshold_min) * 0.05)
        x_limits = (threshold_min - threshold_pad, threshold_max + threshold_pad)
    else:
        x_limits = (-1.0, 1.0)

    ax_perf.set_xlabel(threshold_xlabel, fontsize=11)
    ax_perf.set_ylabel(f"{primary_label} / {secondary_label}", fontsize=11)
    ax_perf.set_xlim(x_limits)
    ax_perf.set_ylim([-0.02, 1.05])
    ax_perf.set_title(f"{title} — {primary_label} / {secondary_label}", fontsize=11, fontweight="bold")
    ax_perf.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_perf.grid(True, alpha=0.2)

    ax_cov.set_xlabel(threshold_xlabel, fontsize=11)
    ax_cov.set_ylabel(coverage_ylabel, fontsize=11)
    ax_cov.set_xlim(x_limits)
    ax_cov.set_ylim([-0.02, 1.05])
    ax_cov.set_title(f"{title} — Coverage", fontsize=11, fontweight="bold")
    ax_cov.legend(loc="best", fontsize=8, framealpha=0.9)
    ax_cov.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def active_confidence_threshold_plot(
    threshold_data: Dict[str, Dict[str, Any]],
    name: str,
    *,
    title: str = "Active Confidence Threshold Sweep",
    model_order: Sequence[str] = (),
) -> None:
    """Plot AOR upper-threshold sweeps for ON calling."""
    _confidence_threshold_plot(
        threshold_data,
        name,
        title=title,
        model_order=model_order,
        primary_key="ppv",
        primary_label="PPV",
        secondary_key="fdr",
        secondary_label="FDR",
        target_key="target_ppv",
        selected_primary_key="selected_ppv",
        selected_secondary_key="selected_fdr",
        threshold_xlabel="Upper activity odds ratio threshold",
        coverage_ylabel="Coverage (fraction called ON)",
    )


def inactive_confidence_threshold_plot(
    threshold_data: Dict[str, Dict[str, Any]],
    name: str,
    *,
    title: str = "Inactive Confidence Threshold Sweep",
    model_order: Sequence[str] = (),
) -> None:
    """Plot AOR lower-threshold sweeps for OFF calling."""
    _confidence_threshold_plot(
        threshold_data,
        name,
        title=title,
        model_order=model_order,
        primary_key="npv",
        primary_label="NPV",
        secondary_key="for",
        secondary_label="FOR",
        target_key="target_npv",
        selected_primary_key="selected_npv",
        selected_secondary_key="selected_for",
        threshold_xlabel="Lower activity odds ratio threshold",
        coverage_ylabel="Coverage (fraction called OFF)",
    )


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
    """Render selective-prediction composition and quality metrics.

    The figure has two side-by-side panels:

     1. **Paired horizontal stacked bars** — model call partition and the
         corresponding truth ON/OFF composition.
     2. **Dot plot** — PPV, NPV, sensitivity, specificity, and coverage
         for each model.

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
        Preferred ordering for the model rows.
    """
    _ensure_dir(name)
    tie_break_order = {model: idx for idx, model in enumerate(model_order)}
    rows: List[Dict[str, Any]] = []
    for model_name, summary in confusion_data.items():
        n_total = summary.get("n_total", 0)
        if n_total == 0:
            continue
        fnr_value = summary.get("fnr", float("nan"))
        if np.isfinite(fnr_value):
            sensitivity_value = 1.0 - float(fnr_value)
        else:
            sensitivity_value = summary.get("sensitivity", float("nan"))
        rows.append(
            {
                "model": model_name,
                "display_label": _bench_display_label(model_name),
                "n_total": n_total,
                "frac_on": summary.get("n_on", 0) / n_total,
                "frac_off": summary.get("n_off", 0) / n_total,
                "frac_withheld": summary.get("n_indet", 0) / n_total,
                "frac_truth_on": summary.get("n_truth_on", 0) / n_total,
                "frac_truth_off": summary.get("n_truth_off", 0) / n_total,
                "ppv": summary.get("ppv", float("nan")),
                "npv": summary.get("npv", float("nan")),
                "sensitivity": sensitivity_value,
                "fpr": summary.get("fpr", float("nan")),
                "specificity": (1.0 - summary.get("fpr", float("nan")))
                if np.isfinite(summary.get("fpr", float("nan"))) else float("nan"),
                "coverage": summary.get("coverage", float("nan")),
                "tie_break": tie_break_order.get(model_name, len(tie_break_order)),
            }
        )

    if not rows:
        return

    rows.sort(
        key=lambda row: (
            row["tie_break"],
            row["display_label"],
        )
    )

    n_models = len(rows)
    fig, ax_comp = plt.subplots(
        ncols=1,
        figsize=(9.0, max(3.8, 0.58 * n_models + 1.9)),
    )

    y_truth = np.array([0.0], dtype=np.float64)
    y_calls = np.arange(1, n_models + 1, dtype=np.float64)
    bar_height = 0.52
    withheld_edge = "#8E8E8E"

    frac_on = np.array([row["frac_on"] for row in rows], dtype=np.float64)
    frac_off = np.array([row["frac_off"] for row in rows], dtype=np.float64)
    frac_withheld = np.array([row["frac_withheld"] for row in rows], dtype=np.float64)

    total_truth_on = float(sum(row["frac_truth_on"] * row["n_total"] for row in rows))
    total_truth_off = float(sum(row["frac_truth_off"] * row["n_total"] for row in rows))
    truth_den = max(total_truth_on + total_truth_off, 1e-12)
    frac_truth_on = total_truth_on / truth_den
    frac_truth_off = total_truth_off / truth_den

    # One shared Truth row (top, fully saturated) followed by per-model calls.
    ax_comp.barh(
        y_truth,
        [frac_truth_on],
        height=bar_height,
        color=_ONOFF_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Truth ON",
    )
    ax_comp.barh(
        y_truth,
        [frac_truth_off],
        left=[frac_truth_on],
        height=bar_height,
        color=_ONOFF_OFF_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Truth OFF",
    )

    ax_comp.barh(
        y_calls,
        frac_on,
        height=bar_height,
        color=_ONOFF_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Calls ON (confident)",
    )
    ax_comp.barh(
        y_calls,
        frac_off,
        left=frac_on,
        height=bar_height,
        color=_ONOFF_OFF_COLOR,
        edgecolor="white",
        linewidth=0.8,
        label="Calls OFF (confident)",
    )
    ax_comp.barh(
        y_calls,
        frac_withheld,
        left=frac_on + frac_off,
        height=bar_height,
        color=_WITHHELD_COLOR,
        edgecolor=withheld_edge,
        linewidth=0.8,
        hatch="//",
        alpha=0.95,
        label="Withheld",
    )

    def _segment_label(
        ax: plt.Axes,
        x_left: float,
        width: float,
        y_pos: float,
        label: str,
        fill_key: str,
        always_show: bool = False,
    ) -> None:
        if not always_show and width < 0.1:
            return
        if fill_key == "withheld":
            text_color = "#4F4F4F" if width >= 0.18 else "black"
            stroke_color = "white"
        else:
            text_color = "white" if width >= 0.18 else "black"
            stroke_color = "black" if text_color == "white" else "white"
        text = ax.text(
            x_left + width / 2.0,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color=text_color,
            fontweight="bold",
            linespacing=0.95,
        )
        text.set_path_effects([PathEffects.withStroke(linewidth=1.4, foreground=stroke_color, alpha=0.35)])

    _segment_label(ax_comp, 0.0, frac_truth_on, float(y_truth[0]), f"ON\n{frac_truth_on:.0%}", "on", always_show=True)
    _segment_label(ax_comp, frac_truth_on, frac_truth_off, float(y_truth[0]), f"OFF\n{frac_truth_off:.0%}", "off", always_show=True)

    for idx, row in enumerate(rows):
        _segment_label(ax_comp, 0.0, row["frac_on"], y_calls[idx], f"ON\n{row['frac_on']:.0%}", "on")
        _segment_label(ax_comp, row["frac_on"], row["frac_off"], y_calls[idx], f"OFF\n{row['frac_off']:.0%}", "off")
        if row["frac_withheld"] > 0.0:
            _segment_label(
                ax_comp,
                row["frac_on"] + row["frac_off"],
                row["frac_withheld"],
                y_calls[idx],
                f"Withheld\n{row['frac_withheld']:.0%}",
                "withheld",
            )

    ax_comp.set_yticks(np.concatenate([y_truth, y_calls]))
    ax_comp.set_yticklabels(["Truth", *[row["display_label"] for row in rows]], fontsize=10)
    ax_comp.invert_yaxis()
    ax_comp.set_xlim(0.0, 1.0)
    _set_explicit_ticks(ax_comp, x_ticks=np.linspace(0.0, 1.0, 5))
    ax_comp.set_xlabel("Fraction of gene-sample pairs", fontsize=11)
    ax_comp.set_title("Truth and call composition", fontsize=11, fontweight="bold")
    ax_comp.grid(True, axis="x", alpha=0.2)

    legend_handles = [
        mpl.patches.Patch(facecolor=_ONOFF_COLOR, edgecolor="white", label="ON"),
        mpl.patches.Patch(facecolor=_ONOFF_OFF_COLOR, edgecolor="white", label="OFF"),
        mpl.patches.Patch(facecolor=_WITHHELD_COLOR, edgecolor=withheld_edge, hatch="//", label="Withheld"),
    ]
    ax_comp.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8.5,
        frameon=False,
    )

    def _fmt_metric(value: Any) -> str:
        return f"{float(value):.3f}" if np.isfinite(value) else "NA"

    metrics_caption_lines = ["Calls metrics (confident calls):"]
    for row in rows:
        metrics_caption_lines.append(
            (
                f"{row['display_label']}: "
                f"PPV={_fmt_metric(row['ppv'])}, "
                f"NPV={_fmt_metric(row['npv'])}, "
                f"Sensitivity={_fmt_metric(row['sensitivity'])}, "
                f"Specificity={_fmt_metric(row['specificity'])}, "
                f"Coverage={_fmt_metric(row['coverage'])}"
            )
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    bottom_pad = min(0.62, 0.20 + 0.04 * len(rows))
    fig.subplots_adjust(bottom=bottom_pad, right=0.83, top=0.92)
    fig.text(
        0.01,
        0.02,
        "\n".join(metrics_caption_lines),
        ha="left",
        va="bottom",
        fontsize=8,
    )
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def benchmark_patient_onoff_confusion_plot(
    confusion_table: pd.DataFrame,
    name: str,
    *,
    title: str,
    model_order: Sequence[str],
    cohort_order: Sequence[str] = ("CRPC-WCDT", "BLCA-RA"),
) -> None:
    """Benchmarking_Patient ON/OFF confusion plot with pooled cohorts.

    Layout:
      - Left: one Truth row + model rows, each with two cohort bars
        (CRPC-WCDT solid, BLCA-RA hatched).
      - Right: per model/cohort thin horizontal metric bars for
        PPV/NPV/Sensitivity/Specificity/Coverage.

    The input table must contain columns:
      model, cohort, n_total, n_on, n_off, n_indet,
      n_truth_on, n_truth_off, ppv, npv, sensitivity, specificity, coverage.
    """
    _ensure_dir(name)
    required = {
        "model",
        "cohort",
        "n_total",
        "n_on",
        "n_off",
        "n_indet",
        "n_truth_on",
        "n_truth_off",
        "ppv",
        "npv",
        "sensitivity",
        "specificity",
        "coverage",
    }
    if confusion_table.empty or not required.issubset(confusion_table.columns):
        return

    table = confusion_table.copy()
    table["cohort"] = table["cohort"].astype(str)
    table["model"] = table["model"].astype(str)
    table["n_total"] = pd.to_numeric(table["n_total"], errors="coerce")
    table = table[table["n_total"] > 0]
    if table.empty:
        return

    def _display_model_label(model_name: str) -> str:
        if model_name == "XGBoost":
            return "XGBoost (Triton)"
        if model_name.startswith("EPIC_"):
            return "EPIC-Seq"
        return model_name.replace("_", " ")

    present_models = set(table["model"].tolist())
    ordered_models = [model_name for model_name in model_order if model_name in present_models]

    row_keys = ["RNA-Seq (Ground Truth)", *ordered_models]
    row_labels = ["RNA-Seq (Ground Truth)", *[_display_model_label(model_name) for model_name in ordered_models]]
    row_y = np.arange(len(row_keys), dtype=np.float64)

    cohort_offsets = {
        str(cohort_order[0]): -0.16,
        str(cohort_order[1]): 0.16,
    }
    call_bar_height = 0.28

    metric_specs = [
        ("PPV", "ppv", "#008080"),
        ("NPV", "npv", "#800000"),
        ("Specificity", "specificity", "#000000"),
        ("Sensitivity", "sensitivity", "#6E6E6E"),
    ]

    cmap = _bench_color_map(ordered_models)
    for model_name in ordered_models:
        if model_name.endswith(" (uncertainty-informed)"):
            cmap[model_name] = "#069494"

    fig_height = 2.0 * max(3.4, 0.42 * len(row_keys) + 1.1)
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(17.0, fig_height),
        gridspec_kw={"width_ratios": [1.0, 2.0]},
    )

    def _lighten(hex_color: str, factor: float = 0.45) -> str:
        rgb = np.array(mpl.colors.to_rgb(hex_color), dtype=np.float64)
        mixed = rgb + (1.0 - rgb) * float(np.clip(factor, 0.0, 1.0))
        return mpl.colors.to_hex(np.clip(mixed, 0.0, 1.0))

    def _segment_label(ax: plt.Axes, left: float, width: float, y_pos: float, label: str) -> None:
        if width < 0.08:
            return
        text_color = "white" if width >= 0.18 else "black"
        stroke_color = "black" if text_color == "white" else "white"
        text = ax.text(
            left + width / 2.0,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=text_color,
            fontweight="bold",
        )
        text.set_path_effects([
            PathEffects.withStroke(linewidth=1.3, foreground=stroke_color, alpha=0.35)
        ])

    for row_idx, row_key in enumerate(row_keys):
        for cohort_name in cohort_order:
            cohort_name = str(cohort_name)
            y_center = row_y[row_idx] + cohort_offsets.get(cohort_name, 0.0)

            if row_key == "RNA-Seq (Ground Truth)":
                cohort_truth = table[table["cohort"] == cohort_name]
                if cohort_truth.empty:
                    continue
                n_truth_on = float(pd.to_numeric(cohort_truth["n_truth_on"], errors="coerce").sum())
                n_truth_off = float(pd.to_numeric(cohort_truth["n_truth_off"], errors="coerce").sum())
                truth_total = n_truth_on + n_truth_off
                if truth_total <= 0.0:
                    continue
                frac_on = n_truth_on / truth_total
                frac_off = n_truth_off / truth_total
                frac_withheld = 0.0
                base_color = "#444444"
            else:
                row_df = table[(table["model"] == row_key) & (table["cohort"] == cohort_name)]
                if row_df.empty:
                    continue
                row = row_df.iloc[0]
                n_total = float(row["n_total"])
                if n_total <= 0.0:
                    continue
                frac_on = float(row["n_on"]) / n_total
                frac_off = float(row["n_off"]) / n_total
                frac_withheld = float(row["n_indet"]) / n_total
                base_color = cmap.get(row_key, "#777777")

            off_color = _lighten(base_color, factor=0.45)
            cohort_hatch = "" if cohort_name == "CRPC-WCDT" else "///"

            ax_left.barh(
                y_center,
                frac_on,
                height=call_bar_height,
                color=base_color,
                edgecolor="white",
                linewidth=0.8,
                hatch=cohort_hatch,
                zorder=2,
            )
            ax_left.barh(
                y_center,
                frac_off,
                left=frac_on,
                height=call_bar_height,
                color=off_color,
                edgecolor="white",
                linewidth=0.8,
                hatch=cohort_hatch,
                zorder=2,
            )
            if frac_withheld > 0.0:
                withh_hatch = "xx" if cohort_name == "CRPC-WCDT" else "///xx"
                ax_left.barh(
                    y_center,
                    frac_withheld,
                    left=frac_on + frac_off,
                    height=call_bar_height,
                    color=_WITHHELD_COLOR,
                    edgecolor="#8E8E8E",
                    linewidth=0.8,
                    hatch=withh_hatch,
                    zorder=2,
                )

            _segment_label(ax_left, 0.0, frac_on, y_center, f"{frac_on:.0%}")
            _segment_label(ax_left, frac_on, frac_off, y_center, f"{frac_off:.0%}")
            if frac_withheld > 0.0:
                _segment_label(ax_left, frac_on + frac_off, frac_withheld, y_center, f"{frac_withheld:.0%}")

    plot_models = [model_name for model_name in ordered_models if model_name != "RNA-Seq (Ground Truth)"]
    n_models = len(plot_models)
    n_cohorts = len(cohort_order)
    bars_per_metric = max(1, n_models * n_cohorts)
    group_width = 0.86
    bar_width = group_width / bars_per_metric

    for metric_index, (metric_label, metric_key, _) in enumerate(metric_specs):
        x_center = float(metric_index)
        left_edge = x_center - group_width / 2.0
        slot_index = 0
        for model_name in plot_models:
            for cohort_name in cohort_order:
                metric_row = table[(table["model"] == model_name) & (table["cohort"] == str(cohort_name))]
                if metric_row.empty:
                    slot_index += 1
                    continue
                metric_value = pd.to_numeric(pd.Series([metric_row.iloc[0][metric_key]]), errors="coerce").iloc[0]
                x_position = left_edge + slot_index * bar_width + bar_width * 0.5
                slot_index += 1
                if not np.isfinite(metric_value):
                    continue
                model_color = cmap.get(model_name, "#777777")
                cohort_hatch = "" if str(cohort_name) == str(cohort_order[0]) else "///"
                ax_right.bar(
                    x_position,
                    float(metric_value),
                    width=bar_width * 0.92,
                    color=model_color,
                    edgecolor="white",
                    linewidth=0.7,
                    hatch=cohort_hatch,
                    alpha=0.95,
                    zorder=3,
                )

    ax_left.set_xlim(0.0, 1.0)
    _set_explicit_ticks(ax_left, x_ticks=np.linspace(0.0, 1.0, 5))
    ax_left.set_xlabel("Fraction of gene-sample pairs", fontsize=10)
    ax_left.set_title("Truth/Calls composition", fontsize=11, fontweight="bold")
    ax_left.grid(True, axis="x", alpha=0.18)

    ax_left.set_yticks(row_y)
    ax_left.set_yticklabels(row_labels, fontsize=9)
    ax_left.invert_yaxis()
    ax_left.set_box_aspect(1.0)

    ax_right.set_xlim(-0.6, len(metric_specs) - 0.4)
    ax_right.set_ylim(0.0, 1.0)
    ax_right.set_xticks(np.arange(len(metric_specs), dtype=float))
    ax_right.set_xticklabels([label for label, _, _ in metric_specs], fontsize=9)
    _set_explicit_ticks(ax_right, y_ticks=np.linspace(0.0, 1.0, 5))
    ax_right.set_xlabel("Metric", fontsize=10)
    ax_right.set_ylabel("Metric value", fontsize=10)
    ax_right.set_title("Per-metric model/cohort bars", fontsize=11, fontweight="bold")
    ax_right.grid(True, axis="y", alpha=0.18)
    ax_right.set_box_aspect(0.5)

    cohort_handles = [
        mpl.patches.Patch(facecolor="#666666", edgecolor="white", label="CRPC-WCDT (solid)"),
        mpl.patches.Patch(facecolor="#666666", edgecolor="white", hatch="///", label="BLCA-RA (hatched)"),
    ]
    model_handles = [
        mpl.patches.Patch(facecolor=cmap.get(model_name, "#777777"), edgecolor="white", label=_display_model_label(model_name))
        for model_name in plot_models
    ]
    call_handles = [
        mpl.patches.Patch(facecolor="#666666", edgecolor="white", label="ON (base model color)"),
        mpl.patches.Patch(facecolor="#BBBBBB", edgecolor="white", label="OFF (lightened model color)"),
        mpl.patches.Patch(facecolor=_WITHHELD_COLOR, edgecolor="#8E8E8E", hatch="xx", label="Withheld"),
    ]

    fig.legend(
        handles=model_handles + cohort_handles + call_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=7.6,
        frameon=False,
        ncol=3,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.0,
    )

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.0, 0.14, 1.0, 0.95])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def ci_width_vs_absolute_residual_scatter(
    ci_width_flat: np.ndarray,
    abs_residual_flat: np.ndarray,
    name: str,
    *,
    title: str = "CI Width vs |Residual|",
    x_label: str = "90% CI width",
    fit_label: str = "Linear fit",
    density_gamma: float = 2.2,
    x_max: float = 8.0,
    y_max: float = 8.0,
) -> None:
    """Density-colored scatter of 90% CI width (x) vs absolute residual (y).

    Includes a linear fit with 95% confidence band and Pearson/Spearman
    annotations. Axes are hard-clipped to [0, x_max] × [0, y_max]; points
    outside that range are excluded from the density estimate and regression.

    Parameters
    ----------
    ci_width_flat:
        CI width per gene-sample pair (hi − lo), 1-D.
    abs_residual_flat:
        Absolute residual |truth − pred| per gene-sample pair, 1-D.
    name:
        Output path prefix (`.pdf` appended automatically).
    title:
        Figure suptitle.
    x_max, y_max:
        Upper limits for both display and fitting (values beyond are excluded).
    """
    _ensure_dir(name)

    x = np.asarray(ci_width_flat, dtype=np.float64)
    y = np.asarray(abs_residual_flat, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0) & (x <= float(x_max)) & (y <= float(y_max))
    x = x[valid]
    y = y[valid]
    n = len(x)
    if n < MIN_CORRELATION_N:
        return

    corr_stats = _compute_correlation_stats(x, y)

    # Linear fit via least squares with 95% CI band
    x_fit = np.linspace(0.0, float(x_max), 300)
    slope: float = float("nan")
    intercept: float = float("nan")
    y_fit_lo: Optional[np.ndarray] = None
    y_fit_hi: Optional[np.ndarray] = None
    y_fit: Optional[np.ndarray] = None
    try:
        from scipy.stats import t as t_dist
        x_mean = float(np.mean(x))
        x_sum_sq = float(np.sum((x - x_mean) ** 2))
        if x_sum_sq > 0.0:
            slope = float(np.sum((x - x_mean) * y) / x_sum_sq)
            intercept = float(np.mean(y) - slope * x_mean)
            y_fit = slope * x_fit + intercept
            residuals = y - (slope * x + intercept)
            sigma2 = float(np.sum(residuals ** 2) / max(n - 2, 1))
            se_fit = np.sqrt(sigma2 * (1.0 / n + (x_fit - x_mean) ** 2 / x_sum_sq))
            t_crit = float(t_dist.ppf(0.975, df=max(n - 2, 1)))
            y_fit_lo = y_fit - t_crit * se_fit
            y_fit_hi = y_fit + t_crit * se_fit
    except Exception:
        pass

    # Density coloring: 2-D histogram → per-point log density
    bins = min(150, max(50, int(np.sqrt(n))))
    hist, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[0.0, float(x_max)], [0.0, float(y_max)]],
    )
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, bins - 1)
    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, bins - 1)
    counts = hist[x_idx, y_idx]
    density = np.log10(counts + 1.0)
    draw_order = np.argsort(density)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    vmax = float(np.nanmax(density)) if len(density) else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    density_norm = mpl.colors.PowerNorm(gamma=max(float(density_gamma), 1.0), vmin=0.0, vmax=vmax)
    sc = ax.scatter(
        x[draw_order],
        y[draw_order],
        c=density[draw_order],
        cmap="Greys",
        norm=density_norm,
        s=4,
        alpha=0.85,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(N + 1)$", fontsize=9)

    if y_fit is not None:
        ax.plot(x_fit, y_fit, color="#CC3311", lw=1.6, zorder=4, label=fit_label)
    if y_fit_lo is not None and y_fit_hi is not None:
        ax.fill_between(x_fit, y_fit_lo, y_fit_hi, color="#CC3311", alpha=0.15, zorder=3)

    ax.set_xlim(0.0, float(x_max))
    ax.set_ylim(0.0, float(y_max))
    x_tick_step = 2.0 if x_max >= 6.0 else 1.0
    y_tick_step = 2.0 if y_max >= 6.0 else 1.0
    _set_explicit_ticks(
        ax,
        x_ticks=np.arange(0.0, float(x_max) + x_tick_step * 0.5, x_tick_step),
        y_ticks=np.arange(0.0, float(y_max) + y_tick_step * 0.5, y_tick_step),
    )
    ax.set_box_aspect(1)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("|Residual|  (|truth − pred|)", fontsize=11)

    stat_lines = [
        _format_correlation_line("Pearson r", corr_stats["pearson_r"], corr_stats["pearson_p"]),
        _format_correlation_line("Spearman ρ", corr_stats["spearman_rho"], corr_stats["spearman_p"]),
        f"n={n:,}",
    ]
    if np.isfinite(slope):
        stat_lines.append(f"slope={slope:.3f}, intercept={intercept:.3f}")
    fig.text(
        0.80,
        0.23,
        "\n".join(stat_lines),
        ha="left",
        va="bottom",
        fontsize=8.5,
        family="monospace",
    )

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0.0, 0.0, 0.77, 1.0])
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

