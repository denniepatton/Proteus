"""Fold-level evaluation: inference, metrics, TSV/JSON output, and plots.

Extracted from Proteus.py ``_save_fold_predictions`` to keep the main
orchestrator lean.  The single public entry point is
:func:`save_fold_predictions`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from proteus_utils.metrics import (
    compute_regression_metrics,
    coverage,
    group_pearson_r_named,
    group_spearman_r2,
    gvw_r,
    scatter_stats,
)


# ---------------------------------------------------------------------------
# Internal data container produced by the inference loop
# ---------------------------------------------------------------------------

@dataclass
class _HeadAccumulator:
    """Per-head (tumor / background) prediction accumulators."""
    pred_mean: List[float] = field(default_factory=list)
    q5: List[float] = field(default_factory=list)
    q95: List[float] = field(default_factory=list)
    truth: List[float] = field(default_factory=list)
    pi_delta: List[float] = field(default_factory=list)
    pi_gamma: List[float] = field(default_factory=list)
    pi_normal: List[float] = field(default_factory=list)
    gamma_beta: List[float] = field(default_factory=list)
    normal_mu: List[float] = field(default_factory=list)
    normal_sigma: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1: Run inference and accumulate predictions + streaming CRPS
# ---------------------------------------------------------------------------

def _run_inference(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    full_tumor_variance: Optional[Dict[str, float]],
    full_background_variance: Optional[Dict[str, float]],
    sample_to_cohort: Optional[Dict[str, str]],
    sample_to_source: Optional[Dict[str, str]],
    source_tumor_variance: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[
    _HeadAccumulator, _HeadAccumulator,
    List[str], List[str],
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    Dict[str, List[torch.Tensor]],
    Dict[str, List[torch.Tensor]],
]:
    """Run model eval and return accumulated predictions + CRPS sums."""

    from models.model_utils import mixture_quantile, mixture_crps

    tumor = _HeadAccumulator()
    background = _HeadAccumulator()
    all_samples: List[str] = []
    all_genes: List[str] = []

    tumor_crps_sum = torch.tensor(0.0)
    tumor_crps_denom = torch.tensor(0.0)
    bg_crps_sum = torch.tensor(0.0)
    bg_crps_denom = torch.tensor(0.0)

    cohort_crps: Dict[str, List[torch.Tensor]] = {}
    source_crps: Dict[str, List[torch.Tensor]] = {}

    model.eval()
    with torch.no_grad():
        for features, labels in val_loader:
            if 'sample_name' not in features or 'gene_name' not in features:
                raise KeyError(
                    "Validation batches are missing 'sample_name'/'gene_name'. "
                    "Pass include_metadata=True when creating the validation DataLoader."
                )

            feat_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in features.items()}
            lab_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in labels.items()}

            predictions, _ = model(feat_dev, return_internal_loss=False)

            # ---- Tumor head ----
            t_mdn = predictions['tumor_mdn']
            t_q = mixture_quantile(t_mdn, quantiles=(0.05, 0.95))
            t_pi = t_mdn['log_pi'].exp().cpu().numpy()

            tumor.pred_mean.extend(predictions['tumor_exp'].cpu().numpy().flatten())
            tumor.q5.extend(t_q[:, 0].cpu().numpy().flatten())
            tumor.q95.extend(t_q[:, 1].cpu().numpy().flatten())
            tumor.truth.extend(lab_dev['tumor_exp'].cpu().numpy().flatten())
            tumor.pi_delta.extend(t_pi[:, 0].flatten())
            tumor.pi_gamma.extend(t_pi[:, 1].flatten())
            tumor.pi_normal.extend(t_pi[:, 2].flatten())
            tumor.gamma_beta.extend(t_mdn['gamma_beta'].cpu().numpy().flatten())
            tumor.normal_mu.extend(t_mdn['normal_mu'].cpu().numpy().flatten())
            tumor.normal_sigma.extend(t_mdn['normal_sigma'].cpu().numpy().flatten())

            # ---- Background head ----
            b_mdn = predictions['background_mdn']
            b_q = mixture_quantile(b_mdn, quantiles=(0.05, 0.95))
            b_pi = b_mdn['log_pi'].exp().cpu().numpy()

            background.pred_mean.extend(predictions['background_exp'].cpu().numpy().flatten())
            background.q5.extend(b_q[:, 0].cpu().numpy().flatten())
            background.q95.extend(b_q[:, 1].cpu().numpy().flatten())
            background.truth.extend(lab_dev['background_exp'].cpu().numpy().flatten())
            background.pi_delta.extend(b_pi[:, 0].flatten())
            background.pi_gamma.extend(b_pi[:, 1].flatten())
            background.pi_normal.extend(b_pi[:, 2].flatten())
            background.gamma_beta.extend(b_mdn['gamma_beta'].cpu().numpy().flatten())
            background.normal_mu.extend(b_mdn['normal_mu'].cpu().numpy().flatten())
            background.normal_sigma.extend(b_mdn['normal_sigma'].cpu().numpy().flatten())

            # ---- Variance weights for GVW-CRPS ----
            gnames = features['gene_name']
            w_tumor = _variance_weights(gnames, full_tumor_variance, device)
            w_bg = _variance_weights(gnames, full_background_variance, device)

            crps_s, crps_d = mixture_crps(t_mdn, lab_dev['tumor_exp'], sample_weight=w_tumor)
            tumor_crps_sum += crps_s.cpu()
            tumor_crps_denom += crps_d.cpu()

            bg_cs, bg_cd = mixture_crps(b_mdn, lab_dev['background_exp'], sample_weight=w_bg)
            bg_crps_sum += bg_cs.cpu()
            bg_crps_denom += bg_cd.cpu()

            # ---- Per-cohort GVW-CRPS (tumor) ----
            _accumulate_group_crps(
                cohort_crps, features, 'sample_name', sample_to_cohort,
                t_mdn, lab_dev['tumor_exp'], w_tumor, device,
            )

            # ---- Per-source GVW-CRPS (tumor, per-source variance weights) ----
            if sample_to_source and source_tumor_variance:
                batch_samples = features['sample_name']
                if not isinstance(batch_samples, (list, tuple)):
                    batch_samples = [batch_samples]
                src_label = sample_to_source.get(str(batch_samples[0]))
                if src_label is not None and src_label in source_tumor_variance:
                    src_var = source_tumor_variance[src_label]
                    w_src = _variance_weights(gnames, src_var, device)
                    src_cs, src_cd = mixture_crps(
                        t_mdn, lab_dev['tumor_exp'], sample_weight=w_src,
                    )
                    if src_label not in source_crps:
                        source_crps[src_label] = [torch.tensor(0.0), torch.tensor(0.0)]
                    source_crps[src_label][0] += src_cs.cpu()
                    source_crps[src_label][1] += src_cd.cpu()

            # ---- Metadata ----
            sample_names = features['sample_name']
            gene_names = features['gene_name']
            if isinstance(sample_names, (list, tuple)):
                all_samples.extend([str(s) for s in sample_names])
            else:
                all_samples.append(str(sample_names))
            if isinstance(gene_names, (list, tuple)):
                all_genes.extend([str(g) for g in gene_names])
            else:
                all_genes.append(str(gene_names))

    return (
        tumor, background,
        all_samples, all_genes,
        tumor_crps_sum, tumor_crps_denom,
        bg_crps_sum, bg_crps_denom,
        cohort_crps, source_crps,
    )


def _variance_weights(
    gene_names: Any,
    var_map: Optional[Dict[str, float]],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build a (batch, 1) variance-weight tensor from a gene→variance map."""
    if var_map is None:
        return None
    if isinstance(gene_names, (list, tuple)):
        return torch.tensor(
            [float(var_map.get(str(g), 1.0)) for g in gene_names],
            dtype=torch.float32, device=device,
        ).unsqueeze(-1)
    return torch.tensor(
        [[float(var_map.get(str(gene_names), 1.0))]],
        dtype=torch.float32, device=device,
    )


def _accumulate_group_crps(
    accum: Dict[str, List[torch.Tensor]],
    features: Dict[str, Any],
    sample_key: str,
    group_map: Optional[Dict[str, str]],
    mdn_params: Dict[str, torch.Tensor],
    truth: torch.Tensor,
    weights: Optional[torch.Tensor],
    device: torch.device,
) -> None:
    """Accumulate per-group (cohort or source) GVW-CRPS in-place."""
    from models.model_utils import mixture_crps

    if not group_map or sample_key not in features:
        return
    batch_samples = features[sample_key]
    if not isinstance(batch_samples, (list, tuple)):
        batch_samples = [batch_samples]
    group_labels = [group_map.get(str(s), 'unknown') for s in batch_samples]
    for grp in set(group_labels):
        idx = torch.tensor(
            [i for i, g in enumerate(group_labels) if g == grp],
            dtype=torch.long, device=device,
        )
        g_mdn = {k: v[idx] for k, v in mdn_params.items()}
        g_w = weights[idx] if weights is not None else None
        cs, cd = mixture_crps(g_mdn, truth[idx], sample_weight=g_w)
        if grp not in accum:
            accum[grp] = [torch.tensor(0.0), torch.tensor(0.0)]
        accum[grp][0] += cs.cpu()
        accum[grp][1] += cd.cpu()


# ---------------------------------------------------------------------------
# Stage 2: Build prediction DataFrames
# ---------------------------------------------------------------------------

def _build_prediction_df(
    acc: _HeadAccumulator,
    all_samples: List[str],
    all_genes: List[str],
) -> pd.DataFrame:
    """Build a prediction DataFrame from an accumulator."""
    return pd.DataFrame(
        {
            'truth': np.asarray(acc.truth, dtype=np.float64),
            'prediction_mean': np.asarray(acc.pred_mean, dtype=np.float64),
            'ci_q5': np.asarray(acc.q5, dtype=np.float64),
            'ci_q95': np.asarray(acc.q95, dtype=np.float64),
            'pi_delta': acc.pi_delta,
            'pi_gamma': acc.pi_gamma,
            'gamma_beta': acc.gamma_beta,
            'pi_normal': acc.pi_normal,
            'normal_mean': acc.normal_mu,
            'normal_stdev': acc.normal_sigma,
            'sample': all_samples,
        },
        index=all_genes,
    )


# ---------------------------------------------------------------------------
# Stage 3: Compute metrics (fold / cohort / source / balanced)
# ---------------------------------------------------------------------------

def _crps_ratio(s: torch.Tensor, d: torch.Tensor) -> float:
    return float((s / d).item()) if float(d.item()) > 0 else float('nan')


def _compute_fold_metrics(
    df: pd.DataFrame,
    crps_sum: torch.Tensor,
    crps_denom: torch.Tensor,
) -> Dict[str, Any]:
    """Compute aggregate metrics for one head (tumor or background)."""
    truth = df['truth'].values
    pred = df['prediction_mean'].values
    m = compute_regression_metrics(truth, pred)
    m['gvw_crps'] = _crps_ratio(crps_sum, crps_denom)

    df_tmp = df.copy()
    df_tmp['gene'] = df_tmp.index.astype(str)
    gene_r2 = group_spearman_r2(df_tmp, group_col='gene', pred_col='prediction_mean')
    sample_r2 = group_spearman_r2(df, group_col='sample', pred_col='prediction_mean')

    m['gene_r2_mean'] = float(np.nanmean(gene_r2)) if gene_r2.size else float('nan')
    m['gene_r2_median'] = float(np.nanmedian(gene_r2)) if gene_r2.size else float('nan')
    m['gene_r2_n'] = int(gene_r2.size)
    m['sample_r2_mean'] = float(np.nanmean(sample_r2)) if sample_r2.size else float('nan')
    m['sample_r2_median'] = float(np.nanmedian(sample_r2)) if sample_r2.size else float('nan')
    m['sample_r2_n'] = int(sample_r2.size)

    q5 = df['ci_q5'].values
    q95 = df['ci_q95'].values
    m['quantile_coverage'] = coverage(truth, q5, q95)
    return m


def _compute_group_metrics(
    df_tumor: pd.DataFrame,
    df_background: pd.DataFrame,
    all_genes: List[str],
    group_arr: np.ndarray,
    crps_accum: Dict[str, List[torch.Tensor]],
    var_maps: Optional[Dict[str, Dict[str, float]]],
    exclude_prefix: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-group (cohort or source) metrics.

    Args:
        var_maps: If provided, maps group→(gene→variance) for external
            GVW-r weighting.  If *None*, GVW-r uses within-group variance.
        exclude_prefix: Skip groups whose name starts with this prefix
            (used for PHT exclusion in cohort mode).
    """
    truth_t = df_tumor['truth'].values
    pred_t = df_tumor['prediction_mean'].values
    truth_b = df_background['truth'].values
    pred_b = df_background['prediction_mean'].values

    per_group: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(group_arr)):
        if exclude_prefix and str(name).startswith(exclude_prefix):
            continue
        if name == 'unknown':
            continue
        mask = group_arr == name
        if mask.sum() == 0:
            continue

        g_tumor = compute_regression_metrics(truth_t[mask], pred_t[mask])
        g_bg = compute_regression_metrics(truth_b[mask], pred_b[mask])

        g_genes = [g for g, m in zip(all_genes, mask) if m]
        ext_var = var_maps.get(name) if var_maps else None
        g_tumor['gvw_r'] = gvw_r(truth_t[mask], pred_t[mask], g_genes, external_var=ext_var)
        g_bg['gvw_r'] = gvw_r(truth_b[mask], pred_b[mask], g_genes)

        if name in crps_accum:
            s, d = crps_accum[name]
            g_tumor['gvw_crps'] = _crps_ratio(s, d)

        per_group[name] = {'tumor': g_tumor, 'background': g_bg}
    return per_group


def _compute_balanced_metrics(
    per_group: Dict[str, Dict[str, Any]],
    exclude_prefix: Optional[str] = None,
) -> Dict[str, float]:
    """Average GVW-CRPS and GVW-r across groups with equal weight."""
    items = per_group.items()
    if exclude_prefix:
        items = [(k, v) for k, v in items if not k.startswith(exclude_prefix)]

    gvw_crps_vals = [v['tumor'].get('gvw_crps', float('nan')) for _, v in items]
    gvw_r_vals = [v['tumor'].get('gvw_r', float('nan')) for _, v in items]
    finite_crps = [x for x in gvw_crps_vals if np.isfinite(x)]
    finite_r = [x for x in gvw_r_vals if np.isfinite(x)]

    return {
        'balanced_tumor_gvw_crps': float(np.mean(finite_crps)) if finite_crps else float('nan'),
        'balanced_tumor_gvw_crps_std': float(np.std(finite_crps)) if len(finite_crps) > 1 else float('nan'),
        'balanced_tumor_gvw_r': float(np.mean(finite_r)) if finite_r else float('nan'),
        'balanced_tumor_gvw_r_std': float(np.std(finite_r)) if len(finite_r) > 1 else float('nan'),
        'min_tumor_gvw_r': float(np.min(finite_r)) if finite_r else float('nan'),
    }


# ---------------------------------------------------------------------------
# Stage 4: Fold-level plots
# ---------------------------------------------------------------------------

def _generate_fold_plots(
    fold_dir: str,
    fold_num: int,
    df_tumor: pd.DataFrame,
    df_background: pd.DataFrame,
    full_tumor_variance: Optional[Dict[str, float]],
    full_background_variance: Optional[Dict[str, float]],
) -> None:
    """Generate scatter and gene-r histogram plots for a fold."""
    from models.model_utils import Y_MAX

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = os.path.join(fold_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for head, df, var_map in [
        ('tumor', df_tumor, full_tumor_variance),
        ('background', df_background, full_background_variance),
    ]:
        truth = df['truth'].values
        pred = df['prediction_mean'].values
        _plot_true_vs_pred(plt, truth, pred, head, 'mean', fold_num, plot_dir, Y_MAX)
        _plot_gene_r_hist(plt, df, 'prediction_mean', var_map, head, 'mean', fold_num, plot_dir)


def _plot_true_vs_pred(
    plt: Any, truth: np.ndarray, pred: np.ndarray,
    head_label: str, est_label: str, fold_num: int,
    plot_dir: str, y_max: float,
) -> None:
    mask = np.isfinite(truth) & np.isfinite(pred)
    x, y_pts = truth[mask], pred[mask]
    if x.size == 0:
        return
    stats = scatter_stats(truth, pred)
    rng = np.random.RandomState(0)
    if x.size > 20000:
        idx = rng.choice(x.size, size=20000, replace=False)
        x, y_pts = x[idx], y_pts[idx]
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(x, y_pts, s=3, alpha=0.25, linewidths=0)
    plt.plot([0.0, y_max], [0.0, y_max], color='black', linewidth=1)
    plt.xlim(0.0, y_max)
    plt.ylim(0.0, y_max)
    plt.xlabel('truth')
    plt.ylabel(f'prediction ({est_label})')
    plt.title(f'Fold {fold_num} {head_label}: true vs pred ({est_label})')
    if stats:
        ann = '\n'.join(f'{k} = {v:.4f}' for k, v in stats.items())
        plt.text(
            0.03, 0.97, ann, transform=plt.gca().transAxes, fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'),
            family='monospace',
        )
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{head_label}_true_vs_pred-{est_label}.png'), dpi=150)
    plt.close(fig)


def _plot_gene_r_hist(
    plt: Any, df: pd.DataFrame, pred_col: str,
    var_map: Optional[Dict[str, float]],
    head_label: str, est_label: str, fold_num: int,
    plot_dir: str,
) -> None:
    df_tmp = df.copy()
    df_tmp['gene'] = df_tmp.index.astype(str)
    gene_r, gene_names_r = group_pearson_r_named(df_tmp, 'gene', pred_col)
    if gene_r.size == 0:
        return
    gene_var = np.array(
        [float(var_map.get(g, 0.0)) for g in gene_names_r], dtype=np.float64,
    ) if var_map else np.zeros(len(gene_names_r), dtype=np.float64)
    low_mask = gene_var < 1.0
    mid_mask = (gene_var >= 1.0) & (gene_var < 3.0)
    high_mask = gene_var >= 3.0
    n_total = gene_r.size

    groups = [
        (low_mask, 'Low [0, 1)', '#3b82f6'),
        (mid_mask, 'Med [1, 3)', '#f59e0b'),
        (high_mask, 'High [3, \u221e)', '#ef4444'),
    ]

    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(-1.0, 1.0, 61)
    for gmask, glabel, gcolor in groups:
        r_sub = gene_r[gmask]
        if r_sub.size == 0:
            continue
        pct = 100.0 * r_sub.size / n_total
        ax.hist(
            r_sub, bins=bins, alpha=0.55, color=gcolor, density=True,
            label=f'{glabel} ({pct:.0f}%)', edgecolor='white', linewidth=0.3,
        )
    ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel('gene-level Pearson r')
    ax.set_ylabel('density')
    ax.set_title(f'Fold {fold_num} {head_label}: gene Pearson r ({est_label})')
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f'{head_label}-{est_label}_gene_r_hist.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def save_fold_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    fold_dir: str,
    fold_num: int,
    device: torch.device,
    full_tumor_variance: Optional[Dict[str, float]] = None,
    full_background_variance: Optional[Dict[str, float]] = None,
    sample_to_cohort: Optional[Dict[str, str]] = None,
    sample_to_source: Optional[Dict[str, str]] = None,
    source_tumor_variance: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Save MDN predictions, distribution params, quantile CIs, and metrics.

    For each observation the output TSV contains::

        truth, prediction_mean, ci_q5, ci_q95,
        pi_delta, pi_gamma, gamma_beta, pi_normal, normal_mean,
        normal_stdev, sample.

    ci_q5 / ci_q95 are CDF-based quantiles clamped to [0, Y_MAX].
    """

    # ---- 1. Inference ----
    (
        tumor, background,
        all_samples, all_genes,
        tumor_crps_sum, tumor_crps_denom,
        bg_crps_sum, bg_crps_denom,
        cohort_crps, source_crps,
    ) = _run_inference(
        model, val_loader, device,
        full_tumor_variance, full_background_variance,
        sample_to_cohort, sample_to_source, source_tumor_variance,
    )

    # ---- 2. Build DataFrames ----
    df_tumor = _build_prediction_df(tumor, all_samples, all_genes)
    df_background = _build_prediction_df(background, all_samples, all_genes)

    # ---- 3. Save TSVs ----
    df_tumor.to_csv(
        os.path.join(fold_dir, f'validation-fold_{fold_num}_predictions_tumor.tsv'),
        sep='\t',
    )
    df_background.to_csv(
        os.path.join(fold_dir, f'validation-fold_{fold_num}_predictions_background.tsv'),
        sep='\t',
    )

    # ---- 4. Fold-level metrics ----
    t_metrics = _compute_fold_metrics(df_tumor, tumor_crps_sum, tumor_crps_denom)
    b_metrics = _compute_fold_metrics(df_background, bg_crps_sum, bg_crps_denom)

    metrics: Dict[str, Any] = {
        'fold': int(fold_num),
        'tumor': t_metrics,
        'background': b_metrics,
    }

    # ---- 5. Per-cohort metrics ----
    per_cohort: Dict[str, Dict[str, Any]] = {}
    if sample_to_cohort:
        samples_arr = np.asarray(all_samples, dtype=object)
        cohort_arr = np.array(
            [sample_to_cohort.get(str(s), 'unknown') for s in samples_arr],
            dtype=object,
        )
        per_cohort = _compute_group_metrics(
            df_tumor, df_background, all_genes, cohort_arr,
            cohort_crps, var_maps=None,
        )
        if per_cohort:
            metrics['per_cohort'] = per_cohort

    # ---- 6. Per-source metrics ----
    per_source: Dict[str, Dict[str, Any]] = {}
    if sample_to_source and source_tumor_variance:
        samples_arr = np.asarray(all_samples, dtype=object)
        source_arr = np.array(
            [sample_to_source.get(str(s), 'unknown') for s in samples_arr],
            dtype=object,
        )
        per_source = _compute_group_metrics(
            df_tumor, df_background, all_genes, source_arr,
            source_crps, var_maps=source_tumor_variance,
        )
        if per_source:
            metrics['per_source'] = per_source

    # ---- 7. Balanced metrics ----
    if per_source:
        metrics.update(_compute_balanced_metrics(per_source))
    elif per_cohort:
        metrics.update(_compute_balanced_metrics(per_cohort, exclude_prefix='PH'))

    # ---- 8. Save metrics JSON ----
    with open(os.path.join(fold_dir, f'validation-fold_{fold_num}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ---- 9. Console summary ----
    print(
        f'Fold {fold_num} tumor metrics: '
        f'gvw_crps={t_metrics.get("gvw_crps", float("nan")):.4f} '
        f'gene_r2(mean/med)={t_metrics.get("gene_r2_mean", float("nan")):.4f}/'
        f'{t_metrics.get("gene_r2_median", float("nan")):.4f} '
        f'sample_r2(mean/med)={t_metrics.get("sample_r2_mean", float("nan")):.4f}/'
        f'{t_metrics.get("sample_r2_median", float("nan")):.4f}'
    )
    if per_source:
        for src_name, src_data in sorted(per_source.items()):
            crps_str = f'{src_data["tumor"].get("gvw_crps", float("nan")):.4f}'
            r_str = f'{src_data["tumor"].get("gvw_r", float("nan")):.4f}'
            print(f'  source {src_name}: gvw_crps={crps_str} gvw_r={r_str}')
        balanced_crps_str = f'{metrics.get("balanced_tumor_gvw_crps", float("nan")):.4f}'
        balanced_r_str = f'{metrics.get("balanced_tumor_gvw_r", float("nan")):.4f}'
        print(f'  balanced_gvw_crps={balanced_crps_str} balanced_gvw_r={balanced_r_str}')
    elif per_cohort:
        balanced_crps_str = f'{metrics.get("balanced_tumor_gvw_crps", float("nan")):.4f}'
        balanced_r_str = f'{metrics.get("balanced_tumor_gvw_r", float("nan")):.4f}'
        print(f'  balanced_gvw_crps={balanced_crps_str} balanced_gvw_r={balanced_r_str}')

    # ---- 10. Plots ----
    try:
        _generate_fold_plots(
            fold_dir, fold_num,
            df_tumor, df_background,
            full_tumor_variance, full_background_variance,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        print(f'  Warning: failed to generate evaluation plots for fold {fold_num}: {e}')
