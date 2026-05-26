from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from .analysis_utils import (
    PC_PDX_STUDY,
    _gvw_r_from_wide,
    _gvw_rho_from_wide,
    compute_gene_variance,
    ensure_dir,
    generate_null_predictions,
)
from .plot_utils import metric_distribution, sample_scatter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MIXTURE_PDF_NAME = "MDN_Mixture_Fit.pdf"
_EPS = 1e-6
_LIGHT_GRAY = "#CFCFCF"
_GRAY = "#7A7A7A"
_TEAL = "#0F8B8D"
_BLACK = "#000000"
_SCATTER_TICKS = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]
_SPEARMAN_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]
_SPEARMAN_DENSITY_Y_TICKS = [0.0, 0.5, 1.0, 1.5, 2.0]


def _nice_step_for_five_ticks(span: float) -> float:
    if not np.isfinite(span) or span <= 0.0:
        return 1.0
    target = span / 4.0
    steps = [0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 4.0, 5.0, 10.0, 20.0]
    for step in steps:
        if step >= target:
            return step
    return steps[-1]


def _five_ticks(lower: float, upper: float) -> np.ndarray:
    return np.linspace(float(lower), float(upper), 5)


def _set_axis_ticks(
    ax: plt.Axes,
    *,
    x_ticks: Optional[Sequence[float]] = None,
    y_ticks: Optional[Sequence[float]] = None,
) -> None:
    if x_ticks is not None:
        ax.set_xticks(list(x_ticks))
    if y_ticks is not None:
        ax.set_yticks(list(y_ticks))


def _set_five_y_ticks(ax: plt.Axes, *, lower: float = 0.0) -> None:
    y_hi = float(ax.get_ylim()[1])
    step = _nice_step_for_five_ticks(max(y_hi - lower, 1e-9))
    upper = lower + 4.0 * step
    ax.set_ylim(lower, upper)
    ax.set_yticks(_five_ticks(lower, upper))


def _nice_upper_bound(max_value: float) -> float:
    step = _nice_step_for_five_ticks(max(float(max_value), 1e-9))
    return 4.0 * step


def gamma_pdf(x: np.ndarray, k: float, beta: float) -> np.ndarray:
    return np.where(x > 0.0, gamma_dist.pdf(x, a=k, loc=0.0, scale=1.0 / beta), 0.0)


def gamma_cdf(x: np.ndarray, k: float, beta: float) -> np.ndarray:
    return np.where(x >= 0.0, gamma_dist.cdf(x, a=k, loc=0.0, scale=1.0 / beta), 0.0)


def continuous_pdf(
    x: np.ndarray,
    pi_gam: float,
    pi_norm: float,
    k: float,
    beta: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    return pi_gam * gamma_pdf(x, k, beta) + pi_norm * norm.pdf(x, loc=mu, scale=sigma)


def continuous_cdf(
    x: np.ndarray,
    pi_gam: float,
    pi_norm: float,
    k: float,
    beta: float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    return pi_gam * gamma_cdf(x, k, beta) + pi_norm * norm.cdf(x, loc=mu, scale=sigma)


def _softplus(x: float) -> float:
    if x > 20.0:
        return x
    return np.log1p(np.exp(x))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    z = np.exp(x)
    return z / (1.0 + z)


def unpack_params(par: np.ndarray, pi_delta: float) -> tuple[float, float, float, float, float, float]:
    raw_w, raw_k, raw_beta, mu_raw, raw_sigma = par
    w = _sigmoid(float(raw_w))
    pi_cont = 1.0 - pi_delta
    pi_gam = pi_cont * w
    pi_norm = pi_cont * (1.0 - w)
    k = _softplus(float(raw_k)) + _EPS
    beta = _softplus(float(raw_beta)) + _EPS
    mu = max(float(mu_raw), 0.0)
    sigma = _softplus(float(raw_sigma)) + _EPS
    return pi_gam, pi_norm, k, beta, mu, sigma


def neg_log_likelihood(
    par: np.ndarray,
    bin_centers: np.ndarray,
    bin_heights: np.ndarray,
    pi_delta: float,
) -> float:
    pi_gam, pi_norm, k, beta, mu, sigma = unpack_params(par, pi_delta)
    pdf_vals = continuous_pdf(bin_centers, pi_gam, pi_norm, k, beta, mu, sigma)
    pdf_vals = np.clip(pdf_vals, 1e-30, None)
    return float(-np.sum(bin_heights * np.log(pdf_vals)))


def compute_gof(
    data: np.ndarray,
    pi_delta: float,
    pi_gam: float,
    pi_norm: float,
    k: float,
    beta: float,
    mu: float,
    sigma: float,
    n_bins: int = 200,
) -> Dict[str, float]:
    data_pos = data[data > 0]
    counts, edges = np.histogram(data_pos, bins=n_bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    dx = edges[1] - edges[0]
    p_emp = counts * dx
    p_emp = p_emp / p_emp.sum()

    p_fit = continuous_pdf(centers, pi_gam, pi_norm, k, beta, mu, sigma) * dx
    p_fit = np.clip(p_fit, 1e-30, None)
    p_fit = p_fit / p_fit.sum()

    kl = float(np.sum(p_emp * np.log(p_emp / p_fit + 1e-30)))
    tv = float(0.5 * np.sum(np.abs(p_emp - p_fit)))

    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n

    fitted_cdf = (
        pi_delta * (sorted_data >= 0.0).astype(float)
        + continuous_cdf(sorted_data, pi_gam, pi_norm, k, beta, mu, sigma)
    )

    ks_stat = float(np.max(np.abs(ecdf - fitted_cdf)))
    wasserstein = float(np.mean(np.abs(ecdf - fitted_cdf)))

    y_max = sorted_data[-1] * 1.1
    y_grid = np.linspace(0.0, y_max, 500)
    dz = y_max / (len(y_grid) - 1)

    n_sub = min(n, 5000)
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(n, size=n_sub, replace=False)
    sub_data = data[sub_idx]

    cdf_mix_grid = (
        pi_delta * (y_grid >= 0.0).astype(float)
        + continuous_cdf(y_grid, pi_gam, pi_norm, k, beta, mu, sigma)
    )

    crps_vals = np.empty(n_sub, dtype=np.float64)
    for i, y_obs in enumerate(sub_data):
        step = (y_grid >= y_obs).astype(float)
        crps_vals[i] = np.sum((cdf_mix_grid - step) ** 2) * dz

    return {
        "KL_divergence": kl,
        "total_variation": tv,
        "KS_statistic": ks_stat,
        "wasserstein_1": wasserstein,
        "crps_mean": float(crps_vals.mean()),
    }


def _fit_mixture(raw_data: np.ndarray) -> Dict[str, float]:
    n_total = len(raw_data)
    n_zero = int(np.sum(raw_data < 1e-10))
    pi_delta = n_zero / n_total

    data_pos = raw_data[raw_data > 1e-10]
    counts, edges = np.histogram(data_pos, bins=200, density=True)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    dx = edges[1] - edges[0]
    bin_heights = counts * dx
    bin_heights = bin_heights / bin_heights.sum()

    bounds_de = [
        (-5.0, 5.0),
        (-2.0, 4.0),
        (-2.0, 5.0),
        (0.0, 18.0),
        (-2.0, 5.0),
    ]

    de_result = differential_evolution(
        neg_log_likelihood,
        bounds=bounds_de,
        args=(bin_centers, bin_heights, pi_delta),
        seed=42,
        maxiter=1000,
        tol=1e-10,
        polish=False,
        disp=False,
    )
    res = minimize(
        neg_log_likelihood,
        x0=de_result.x,
        args=(bin_centers, bin_heights, pi_delta),
        method="L-BFGS-B",
        options={"maxiter": 2000, "disp": False},
    )

    pi_gam, pi_norm, k, beta, mu, sigma = unpack_params(res.x, pi_delta)
    return {
        "pi_delta": pi_delta,
        "pi_gam": pi_gam,
        "pi_norm": pi_norm,
        "k": k,
        "beta": beta,
        "mu": mu,
        "sigma": sigma,
        "nll": float(res.fun),
        "converged": bool(res.success),
    }


def fit_expression_mixture(
    truth_wide: pd.DataFrame,
    output_dir: str,
    overwrite: bool,
) -> None:
    naive_dir = ensure_dir(str(Path(output_dir)))
    pdf_path = Path(naive_dir) / _MIXTURE_PDF_NAME
    if pdf_path.exists() and not overwrite:
        return

    raw_data = truth_wide.values.astype(np.float64).flatten()
    raw_data = raw_data[np.isfinite(raw_data)]
    raw_data = raw_data[raw_data >= 0.0]
    if raw_data.size == 0:
        raise ValueError("PLCA truth pool is empty; cannot fit RNA-seq mixture")

    fit = _fit_mixture(raw_data)
    gof = compute_gof(
        raw_data,
        fit["pi_delta"],
        fit["pi_gam"],
        fit["pi_norm"],
        fit["k"],
        fit["beta"],
        fit["mu"],
        fit["sigma"],
    )

    data_pos = raw_data[raw_data > 1e-10]
    x_max = _nice_upper_bound(raw_data.max() * 1.05)
    xvals = np.linspace(1e-6, x_max, 500)

    pdf_gam_comp = fit["pi_gam"] * gamma_pdf(xvals, fit["k"], fit["beta"])
    pdf_norm_comp = fit["pi_norm"] * norm.pdf(xvals, loc=fit["mu"], scale=fit["sigma"])
    pdf_continuous = pdf_gam_comp + pdf_norm_comp

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    ax = axes[0]
    hist_counts, hist_edges = np.histogram(data_pos, bins=100, density=True)
    hist_density = hist_counts * (1.0 - fit["pi_delta"])
    hist_centers = 0.5 * (hist_edges[1:] + hist_edges[:-1])
    hist_widths = hist_edges[1:] - hist_edges[:-1]
    ax.bar(
        hist_centers,
        hist_density,
        width=hist_widths,
        alpha=0.4,
        color=_LIGHT_GRAY,
        label="Empirical density (x > 0, scaled by 1-pi_delta)",
    )
    ax.axvline(0.0, color="red", lw=2, ls="-", alpha=0.7)
    ax.annotate(
        f"delta(0)\npi={fit['pi_delta']:.3f}",
        xy=(0, pdf_continuous.max() * 0.85),
        xytext=(0.8, pdf_continuous.max() * 0.9),
        fontsize=13,
        color="red",
        arrowprops={"arrowstyle": "->", "color": "red"},
    )
    ax.plot(xvals, pdf_continuous, "k-", lw=2.5, label="Fitted continuous mixture")
    ax.plot(
        xvals,
        pdf_gam_comp,
        "--",
        color="orange",
        lw=3.0,
        label=f"Gamma(k={fit['k']:.2f}, beta={fit['beta']:.2f})  pi={fit['pi_gam']:.3f}",
    )
    ax.plot(
        xvals,
        pdf_norm_comp,
        "--",
        color="green",
        lw=3.0,
        label=f"N(mu={fit['mu']:.2f}, sigma={fit['sigma']:.2f})  pi={fit['pi_norm']:.3f}",
    )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("log2(CPM + 1)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Mixture Fit: delta(0) + Gamma + Normal", fontsize=15)
    ax.legend(fontsize=12)
    ax.set_xlim(0, x_max)
    _set_axis_ticks(
        ax,
        x_ticks=_five_ticks(0.0, x_max),
        y_ticks=_five_ticks(0.0, 1.0),
    )
    ax.tick_params(labelsize=12)
    ax.text(
        0.98,
        0.98,
        "\n".join([
            f"KL={gof['KL_divergence']:.4f}",
            f"TV={gof['total_variation']:.4f}",
            f"KS={gof['KS_statistic']:.4f}",
            f"CRPS={gof['crps_mean']:.4f}",
        ]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    ax2 = axes[1]
    sorted_data = np.sort(raw_data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    fitted_cdf = (
        fit["pi_delta"] * (sorted_data >= 0.0).astype(float)
        + continuous_cdf(sorted_data, fit["pi_gam"], fit["pi_norm"], fit["k"], fit["beta"], fit["mu"], fit["sigma"])
    )
    ax2.plot(sorted_data, ecdf, color=_LIGHT_GRAY, lw=3.0, label="Empirical CDF")
    ax2.plot(sorted_data, fitted_cdf, "k--", lw=3.0, label="Fitted CDF", zorder=5)
    ax2.set_xlabel("log2(CPM + 1)", fontsize=14)
    ax2.set_ylabel("CDF", fontsize=14)
    ax2.set_title("CDF Comparison", fontsize=15)
    ax2.legend(fontsize=12)
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0.0, 1.0)
    _set_axis_ticks(
        ax2,
        x_ticks=_five_ticks(0.0, x_max),
        y_ticks=_five_ticks(0.0, 1.0),
    )
    ax2.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(pdf_path, dpi=150)
    plt.close(fig)


def _gene_spearman_rho(truth_arr: np.ndarray, pred_arr: np.ndarray) -> np.ndarray:
    truth_rank = pd.DataFrame(truth_arr).rank(axis=1, method="average", na_option="keep").values
    pred_rank = pd.DataFrame(pred_arr).rank(axis=1, method="average", na_option="keep").values

    return _gene_pearson_like_correlation(truth_rank, pred_rank)


def _gene_pearson_like_correlation(truth_arr: np.ndarray, pred_arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(truth_arr) & np.isfinite(pred_arr)
    n_valid = finite.sum(axis=1)
    result = np.full(truth_arr.shape[0], np.nan)
    ok = n_valid >= 5
    if not ok.any():
        return result

    tc = np.where(finite, truth_arr, 0.0)
    pc = np.where(finite, pred_arr, 0.0)
    nv = n_valid[ok].astype(np.float64)
    st = tc[ok].sum(axis=1)
    sp = pc[ok].sum(axis=1)
    stt = (tc[ok] ** 2).sum(axis=1)
    spp = (pc[ok] ** 2).sum(axis=1)
    stp = (tc[ok] * pc[ok]).sum(axis=1)
    num = nv * stp - st * sp
    with np.errstate(invalid="ignore"):
        den = np.sqrt((nv * stt - st ** 2) * (nv * spp - sp ** 2))
    truth_ss = nv * stt - st ** 2
    pred_ss = nv * spp - sp ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        result[ok] = np.where(den > 0, num / den, np.nan)
    result[ok] = np.where(truth_ss > 1e-12, result[ok], np.nan)
    result[ok] = np.where((truth_ss > 1e-12) & (pred_ss <= 1e-12), 0.0, result[ok])
    return result


def _pca_sample_coordinates(
    truth_wide: pd.DataFrame,
    *,
    n_components: int = 2,
) -> pd.DataFrame:
    sample_names = truth_wide.columns.astype(str).tolist()
    if len(sample_names) < 2:
        return pd.DataFrame(columns=[f"PC{i + 1}" for i in range(n_components)])

    sample_matrix = truth_wide.T.values.astype(np.float64)
    feature_means = np.nanmean(sample_matrix, axis=0, keepdims=True)
    centered = sample_matrix - feature_means
    centered[~np.isfinite(centered)] = 0.0

    max_components = min(n_components, centered.shape[0], centered.shape[1])
    if max_components < 1:
        return pd.DataFrame(columns=[f"PC{i + 1}" for i in range(n_components)])

    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :max_components] * s[:max_components]
    if scores.shape[0] < 2:
        return pd.DataFrame(columns=[f"PC{i + 1}" for i in range(n_components)])

    if max_components < n_components:
        padding = np.zeros((scores.shape[0], n_components - max_components), dtype=np.float64)
        scores = np.hstack((scores, padding))
    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(scores[:, :n_components], index=sample_names, columns=columns)


def _pca_nearest_neighbor_map(pca_coords: pd.DataFrame) -> Dict[str, str]:
    if pca_coords.empty or len(pca_coords.index) < 2:
        return {}

    coord_arr = pca_coords.values.astype(np.float64)
    deltas = coord_arr[:, None, :] - coord_arr[None, :, :]
    dist_sq = np.sum(deltas ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)

    nearest_indices = np.argmin(dist_sq, axis=1)
    nearest_map: Dict[str, str] = {}
    sample_names = pca_coords.index.astype(str).tolist()
    for idx, neighbor_idx in enumerate(nearest_indices):
        if not np.isfinite(dist_sq[idx, neighbor_idx]):
            continue
        nearest_map[sample_names[idx]] = sample_names[neighbor_idx]
    return nearest_map


def _unique_pairs_from_neighbor_map(neighbor_map: Dict[str, str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sample_name, neighbor_name in neighbor_map.items():
        key = tuple(sorted((sample_name, neighbor_name)))
        if sample_name == neighbor_name or key in seen:
            continue
        seen.add(key)
        pairs.append((sample_name, neighbor_name))
    return pairs


def _plot_pca_nearest_neighbors(
    pca_coords: pd.DataFrame,
    pca_pairs: list[tuple[str, str]],
    out_path: Path,
) -> None:
    if pca_coords.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    for sample_name_a, sample_name_b in pca_pairs:
        ax.plot(
            [pca_coords.loc[sample_name_a, "PC1"], pca_coords.loc[sample_name_b, "PC1"]],
            [pca_coords.loc[sample_name_a, "PC2"], pca_coords.loc[sample_name_b, "PC2"]],
            color=_TEAL,
            alpha=0.35,
            lw=1.5,
            zorder=1,
        )

    ax.scatter(
        pca_coords["PC1"],
        pca_coords["PC2"],
        color=_TEAL,
        s=52.0,
        alpha=0.9,
        linewidths=0,
        zorder=2,
    )
    if "LuCaP_136" in pca_coords.index:
        ax.scatter(
            [pca_coords.loc["LuCaP_136", "PC1"]],
            [pca_coords.loc["LuCaP_136", "PC2"]],
            color=_BLACK,
            s=68.0,
            linewidths=0,
            zorder=3,
        )

    for sample_name, row in pca_coords.iterrows():
        ax.annotate(
            sample_name,
            (float(row["PC1"]), float(row["PC2"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=_BLACK,
            zorder=4,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC-PDX PCA Nearest-Neighbor Pairings")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_naive_example(out_dir: str, overwrite: bool) -> None:
    naive_dir = ensure_dir(str(Path(out_dir)))

    pc_truth = pd.read_csv(
        PROJECT_ROOT / "labels" / "PC-PDX_Log2CPMplus1.tsv",
        sep="\t",
        index_col=0,
    )
    keep = [s for s in PC_PDX_STUDY if s in pc_truth.columns]
    if len(keep) < 5:
        raise ValueError("Naive example requires at least five PC-PDX study samples")

    truth_wide = pc_truth[keep]
    truth_arr = truth_wide.values.astype(np.float64)
    n_genes, n_samples = truth_arr.shape
    pca_coords = _pca_sample_coordinates(truth_wide)
    pca_neighbor_map = _pca_nearest_neighbor_map(pca_coords)
    pca_pairs = _unique_pairs_from_neighbor_map(pca_neighbor_map)

    pca_pairs_path = Path(naive_dir) / "pca_nearest_neighbor_pairs.tsv"
    pd.DataFrame(
        [
            {
                "sample": sample_name,
                "nearest_neighbor": neighbor_name,
                "sample_pc1": float(pca_coords.loc[sample_name, "PC1"]),
                "sample_pc2": float(pca_coords.loc[sample_name, "PC2"]),
                "neighbor_pc1": float(pca_coords.loc[neighbor_name, "PC1"]),
                "neighbor_pc2": float(pca_coords.loc[neighbor_name, "PC2"]),
                "pca_distance": float(
                    np.sqrt(
                        (pca_coords.loc[sample_name, "PC1"] - pca_coords.loc[neighbor_name, "PC1"]) ** 2
                        + (pca_coords.loc[sample_name, "PC2"] - pca_coords.loc[neighbor_name, "PC2"]) ** 2
                    )
                ),
            }
            for sample_name, neighbor_name in pca_pairs
        ]
    ).to_csv(pca_pairs_path, sep="\t", index=False)

    pca_plot_path = Path(naive_dir) / "pca_nearest_neighbor_embedding.pdf"
    if overwrite or not pca_plot_path.exists():
        _plot_pca_nearest_neighbors(pca_coords, pca_pairs, pca_plot_path)

    if "LuCaP_136" in truth_wide.columns and "LuCaP_136CR" in truth_wide.columns:
        out_path = Path(naive_dir) / "replicate_scatter_136_vs_136CR"
        if overwrite or not Path(f"{out_path}.pdf").exists():
            sample_scatter(
                truth_wide["LuCaP_136CR"],
                truth_wide["LuCaP_136"],
                "LuCaP_136 vs 136CR",
                str(out_path),
                goi=None,
                point_style="scatter",
                point_color=_TEAL,
                point_size=16.0,
                axis_max=15.0,
                diagonal_color=_BLACK,
                diagonal_lw=2.0,
                diagonal_alpha=1.0,
                diagonal_zorder=6.0,
                x_ticks=_SCATTER_TICKS,
                y_ticks=_SCATTER_TICKS,
            )

    if "LuCaP_136" in truth_wide.columns and "LuCaP_136" in pca_neighbor_map:
        nearest_sample = pca_neighbor_map["LuCaP_136"]
        out_path = Path(naive_dir) / "pca_nearest_neighbor_scatter_136_vs_nearest"
        if overwrite or not Path(f"{out_path}.pdf").exists():
            sample_scatter(
                truth_wide["LuCaP_136"],
                truth_wide[nearest_sample],
                f"LuCaP_136 vs PCA-Nearest ({nearest_sample})",
                str(out_path),
                goi=None,
                point_style="scatter",
                point_color=_TEAL,
                point_size=16.0,
                axis_max=15.0,
                diagonal_color=_BLACK,
                diagonal_lw=2.0,
                diagonal_alpha=1.0,
                diagonal_zorder=6.0,
                x_ticks=_SCATTER_TICKS,
                y_ticks=_SCATTER_TICKS,
            )

    if "LuCaP_136" in truth_wide.columns:
        rng_scatter = np.random.default_rng(42)
        idx_136 = truth_wide.columns.get_loc("LuCaP_136")
        other_indices = [j for j in range(n_samples) if j != idx_136]
        chosen = rng_scatter.choice(other_indices, size=n_genes)
        naive_vals = truth_arr[np.arange(n_genes), chosen]
        naive_series = pd.Series(naive_vals, index=truth_wide.index, name="random_other")
        out_path = Path(naive_dir) / "naive_scatter_136_vs_random_other"
        if overwrite or not Path(f"{out_path}.pdf").exists():
            sample_scatter(
                truth_wide["LuCaP_136"],
                naive_series,
                "LuCaP_136 vs Random Other",
                str(out_path),
                goi=None,
                point_style="scatter",
                point_color=_GRAY,
                point_size=16.0,
                axis_max=15.0,
                diagonal_color=_BLACK,
                diagonal_lw=2.0,
                diagonal_alpha=1.0,
                diagonal_zorder=6.0,
                x_ticks=_SCATTER_TICKS,
                y_ticks=_SCATTER_TICKS,
            )

    cr_pairs = [
        ("LuCaP_35", "LuCaP_35CR"),
        ("LuCaP_70", "LuCaP_70CR"),
        ("LuCaP_96", "LuCaP_96CR"),
        ("LuCaP_136", "LuCaP_136CR"),
        ("LuCaP_167", "LuCaP_167CR"),
        ("LuCaP_176", "LuCaP_176CR"),
    ]
    valid_pairs = [(a, b) for a, b in cr_pairs if a in truth_wide.columns and b in truth_wide.columns]
    lineage_r_series: Optional[pd.Series] = None
    if len(valid_pairs) >= 2:
        non_cr_arr = truth_wide[[a for a, _ in valid_pairs]].values.astype(np.float64)
        cr_arr = truth_wide[[b for _, b in valid_pairs]].values.astype(np.float64)
        lineage_r = _gene_spearman_rho(non_cr_arr, cr_arr)
        lineage_r_series = pd.Series(lineage_r[np.isfinite(lineage_r)], name="spearman_rho")
        out_path = Path(naive_dir) / "lineage_derivative_spearman_rho_distribution"
        if overwrite or not Path(f"{out_path}.pdf").exists():
            metric_distribution(
                lineage_r_series,
            "Per-Gene Spearman rho (CR / non-CR Pairs)",
                str(out_path),
                xlim=(-1.0, 1.0),
                square=True,
                draw_stat_lines=False,
                hist_color=_TEAL,
                kde_color=_TEAL,
                reference_line_color=_BLACK,
                reference_line_lw=1.6,
                reference_line_alpha=1.0,
                reference_line_zorder=6.0,
                x_ticks=_SPEARMAN_TICKS,
                y_ticks=_SPEARMAN_DENSITY_Y_TICKS,
                ylim=(0.0, 2.0),
            )

    pca_neighbor_r_series: Optional[pd.Series] = None
    if len(pca_pairs) >= 2:
        pca_anchor_arr = truth_wide[[a for a, _ in pca_pairs]].values.astype(np.float64)
        pca_neighbor_arr = truth_wide[[b for _, b in pca_pairs]].values.astype(np.float64)
        pca_neighbor_r = _gene_spearman_rho(pca_anchor_arr, pca_neighbor_arr)
        pca_neighbor_r_series = pd.Series(
            pca_neighbor_r[np.isfinite(pca_neighbor_r)],
            name="spearman_rho",
        )
        out_path = Path(naive_dir) / "pca_nearest_neighbor_spearman_rho_distribution"
        if overwrite or not Path(f"{out_path}.pdf").exists():
            metric_distribution(
                pca_neighbor_r_series,
                "Per-Gene Spearman rho (PCA Nearest Neighbors)",
                str(out_path),
                xlim=(-1.0, 1.0),
                square=True,
                draw_stat_lines=False,
                hist_color=_TEAL,
                kde_color=_TEAL,
                reference_line_color=_BLACK,
                reference_line_lw=1.6,
                reference_line_alpha=1.0,
                reference_line_zorder=6.0,
                x_ticks=_SPEARMAN_TICKS,
                y_ticks=_SPEARMAN_DENSITY_Y_TICKS,
                ylim=(0.0, 2.0),
            )

    naive_pred = generate_null_predictions(truth_wide, seed=42)
    naive_r = _gene_spearman_rho(truth_arr, naive_pred.values.astype(np.float64))
    naive_r_series = pd.Series(naive_r[np.isfinite(naive_r)], name="spearman_rho")
    out_path = Path(naive_dir) / "naive_gene_spearman_rho_distribution"
    if overwrite or not Path(f"{out_path}.pdf").exists():
        metric_distribution(
            naive_r_series,
            "Per-Gene Spearman rho (Naive)",
            str(out_path),
            xlim=(-1.0, 1.0),
            square=True,
            draw_stat_lines=False,
            hist_color=_GRAY,
            kde_color=_GRAY,
            reference_line_color=_BLACK,
            reference_line_lw=1.6,
            reference_line_alpha=1.0,
            reference_line_zorder=6.0,
            x_ticks=_SPEARMAN_TICKS,
            y_ticks=_SPEARMAN_DENSITY_Y_TICKS,
            ylim=(0.0, 2.0),
        )

    var_map = compute_gene_variance(truth_wide)
    gene_names = truth_wide.index.astype(str).values
    w_gene = np.array([var_map.get(g, 0.0) for g in gene_names], dtype=np.float64)

    gvw_r_naive = _gvw_r_from_wide(truth_arr, naive_pred.values.astype(np.float64), w_gene)
    gvw_rho_naive = _gvw_rho_from_wide(truth_arr, naive_pred.values.astype(np.float64), w_gene)

    rows = []
    if lineage_r_series is not None:
        rows.append({
            "method": "Lineage_Derivative",
            "GVW_r": _gvw_r_from_wide(non_cr_arr, cr_arr, w_gene),
            "GVW_rho": _gvw_rho_from_wide(non_cr_arr, cr_arr, w_gene),
            "median_gene_rho": float(lineage_r_series.median()),
            "n_genes": len(lineage_r_series),
            "n_samples": len(valid_pairs),
        })
    if pca_neighbor_r_series is not None:
        rows.append({
            "method": "PCA_Nearest_Neighbor",
            "GVW_r": _gvw_r_from_wide(pca_anchor_arr, pca_neighbor_arr, w_gene),
            "GVW_rho": _gvw_rho_from_wide(pca_anchor_arr, pca_neighbor_arr, w_gene),
            "median_gene_rho": float(pca_neighbor_r_series.median()),
            "n_genes": len(pca_neighbor_r_series),
            "n_samples": len(pca_pairs),
        })
    rows.append({
        "method": "Naive_Shuffle",
        "GVW_r": gvw_r_naive,
        "GVW_rho": gvw_rho_naive,
        "median_gene_rho": float(naive_r_series.median()),
        "n_genes": len(naive_r_series),
        "n_samples": n_samples,
    })
    pd.DataFrame(rows).to_csv(
        Path(naive_dir) / "naive_example_gvw_summary.tsv",
        sep="\t",
        index=False,
    )


def run_examples(output_dir: str, plca_truth_wide: pd.DataFrame, overwrite: bool) -> None:
    ensure_dir(output_dir)
    generate_naive_example(output_dir, overwrite)
    fit_expression_mixture(plca_truth_wide, output_dir, overwrite)