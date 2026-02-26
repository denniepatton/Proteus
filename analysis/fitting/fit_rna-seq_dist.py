"""Fit a 3-component hurdle mixture to empirical RNA-seq data.

Mixture (mixed discrete–continuous):
    pi_delta * δ(0)              — true point mass at zero (hurdle)
  + pi_gam  * Gamma(k, β)       — decay near zero (shape k free, not fixed to 1)
  + pi_norm * Normal(mu, σ²)    — Gaussian bulk

pi_delta is fixed to the empirical zero fraction; the optimizer fits
pi_gam, pi_norm (constrained to sum to 1 - pi_delta), k, beta, mu, sigma.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, gamma as gamma_dist


# ---------------------------------------------------------------------------
# PDF / CDF helpers  (vectorized)
# ---------------------------------------------------------------------------

def gamma_pdf(x: np.ndarray, k: float, beta: float) -> np.ndarray:
    """Gamma(shape=k, rate=beta) PDF for x >= 0.

    scipy parameterizes as gamma(a=k, loc=0, scale=1/beta).
    """
    return np.where(x > 0.0, gamma_dist.pdf(x, a=k, loc=0.0, scale=1.0 / beta), 0.0)


def gamma_cdf(x: np.ndarray, k: float, beta: float) -> np.ndarray:
    """Gamma(shape=k, rate=beta) CDF."""
    return np.where(x >= 0.0, gamma_dist.cdf(x, a=k, loc=0.0, scale=1.0 / beta), 0.0)


def continuous_pdf(
    x: np.ndarray,
    pi_gam: float, pi_norm: float,
    k: float, beta: float,
    mu: float, sigma: float,
) -> np.ndarray:
    """PDF of the continuous part (Gamma + Normal)."""
    return pi_gam * gamma_pdf(x, k, beta) + pi_norm * norm.pdf(x, loc=mu, scale=sigma)


def continuous_cdf(
    x: np.ndarray,
    pi_gam: float, pi_norm: float,
    k: float, beta: float,
    mu: float, sigma: float,
) -> np.ndarray:
    """CDF of the continuous part (Gamma + Normal)."""
    return pi_gam * gamma_cdf(x, k, beta) + pi_norm * norm.cdf(x, loc=mu, scale=sigma)


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------
# Raw parameter vector (all unconstrained):
#   [raw_w,        -- sigmoid → w; pi_gam = (1-pi_delta)*w, pi_norm = (1-pi_delta)*(1-w)
#    raw_k,        -- softplus → k > 0   (Gamma shape)
#    raw_beta,     -- softplus → beta > 0 (Gamma rate)
#    mu,           -- clamped ≥ 0
#    raw_sigma]    -- softplus → sigma > 0

_EPS = 1e-6


def _softplus(x: float) -> float:
    if x > 20.0:
        return x
    return np.log1p(np.exp(x))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def unpack_params(par, pi_delta: float):
    """Unpack raw parameters → interpretable mixture parameters.

    Returns:
        pi_gam, pi_norm, k, beta, mu, sigma
    """
    raw_w, raw_k, raw_beta, mu_raw, raw_sigma = par
    w = _sigmoid(raw_w)
    pi_cont = 1.0 - pi_delta
    pi_gam = pi_cont * w
    pi_norm = pi_cont * (1.0 - w)
    k = _softplus(raw_k) + _EPS        # Gamma shape > 0
    beta = _softplus(raw_beta) + _EPS   # Gamma rate > 0
    mu = max(mu_raw, 0.0)
    sigma = _softplus(raw_sigma) + _EPS
    return pi_gam, pi_norm, k, beta, mu, sigma


# ---------------------------------------------------------------------------
# Negative log-likelihood  (continuous part only)
# ---------------------------------------------------------------------------

def neg_log_likelihood(par, bin_centers, bin_heights, pi_delta):
    """NLL of the Gamma+Normal mixture against the positive-data histogram."""
    pi_gam, pi_norm, k, beta, mu, sigma = unpack_params(par, pi_delta)

    pdf_vals = continuous_pdf(bin_centers, pi_gam, pi_norm, k, beta, mu, sigma)
    pdf_vals = np.clip(pdf_vals, 1e-30, None)
    nll = -np.sum(bin_heights * np.log(pdf_vals))
    return nll


# ---------------------------------------------------------------------------
# Goodness-of-fit metrics
# ---------------------------------------------------------------------------

def compute_gof(
    data: np.ndarray,
    pi_delta: float,
    pi_gam: float, pi_norm: float,
    k: float, beta: float,
    mu: float, sigma: float,
    n_bins: int = 200,
):
    """Goodness-of-fit on the full mixed distribution."""
    # --- Histogram-based KL & TV (continuous part, x > 0) ---
    data_pos = data[data > 0]
    counts, edges = np.histogram(data_pos, bins=n_bins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    dx = edges[1] - edges[0]
    p_emp = counts * dx
    p_emp = p_emp / p_emp.sum()

    p_fit = continuous_pdf(centers, pi_gam, pi_norm, k, beta, mu, sigma) * dx
    p_fit = np.clip(p_fit, 1e-30, None)
    p_fit = p_fit / p_fit.sum()

    kl = np.sum(p_emp * np.log(p_emp / p_fit + 1e-30))
    tv = 0.5 * np.sum(np.abs(p_emp - p_fit))

    # --- CDF-based metrics on full data ---
    sorted_data = np.sort(data)
    n = len(sorted_data)
    ecdf = np.arange(1, n + 1) / n

    fitted_cdf = (
        pi_delta * (sorted_data >= 0.0).astype(float)
        + continuous_cdf(sorted_data, pi_gam, pi_norm, k, beta, mu, sigma)
    )

    ks_stat = np.max(np.abs(ecdf - fitted_cdf))
    wasserstein = np.mean(np.abs(ecdf - fitted_cdf))

    # --- Grid-based CRPS ---
    y_max = sorted_data[-1] * 1.1
    n_grid = 500
    y_grid = np.linspace(0.0, y_max, n_grid)
    dz = y_max / (n_grid - 1)

    n_sub = min(n, 5000)
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(n, size=n_sub, replace=False)
    sub_data = data[sub_idx]

    cdf_mix_grid = (
        pi_delta * (y_grid >= 0.0).astype(float)
        + continuous_cdf(y_grid, pi_gam, pi_norm, k, beta, mu, sigma)
    )

    crps_vals = np.empty(n_sub)
    for i, y_obs in enumerate(sub_data):
        step = (y_grid >= y_obs).astype(float)
        crps_vals[i] = np.sum((cdf_mix_grid - step) ** 2) * dz
    crps_mean = crps_vals.mean()

    return {
        'KL_divergence': kl,
        'total_variation': tv,
        'KS_statistic': ks_stat,
        'wasserstein_1': wasserstein,
        'crps_mean': crps_mean,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1) Load data
    df = pd.read_csv("../../../RNA-Seq/PC-PDX_Log2CPMplus1.tsv", sep="\t", index_col=0)
    raw_data = df.values.flatten()
    raw_data = raw_data[~np.isnan(raw_data)]
    raw_data = raw_data[raw_data >= 0]
    print(f"Loaded {len(raw_data)} values, min={raw_data.min():.4f}, max={raw_data.max():.4f}")

    # 2) Point-mass weight = empirical zero fraction (FIXED, not fitted)
    n_total = len(raw_data)
    n_zero = np.sum(raw_data < 1e-10)
    pi_delta = n_zero / n_total
    print(f"Exact zeros: {n_zero} / {n_total}  →  pi_delta = {pi_delta:.6f}")

    # 3) Build histogram of x > 0 values for fitting the continuous components
    data_pos = raw_data[raw_data > 1e-10]
    n_fit_bins = 200
    counts, edges = np.histogram(data_pos, bins=n_fit_bins, density=True)
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    dx = edges[1] - edges[0]
    bin_heights = counts * dx
    bin_heights = bin_heights / bin_heights.sum()

    # 4) Differential evolution → L-BFGS-B polish
    #    Raw param vector: [raw_w, raw_k, raw_beta, mu, raw_sigma]
    bounds_de = [
        (-5.0, 5.0),   # raw_w       → sigmoid: split between gamma & normal
        (-2.0, 4.0),   # raw_k       → softplus: Gamma shape k ∈ ~(0.1, 55)
        (-2.0, 5.0),   # raw_beta    → softplus: Gamma rate β ∈ ~(0.1, 150)
        (0.0, 18.0),   # mu          → Normal location
        (-2.0, 5.0),   # raw_sigma   → softplus: Normal σ ∈ ~(0.1, 150)
    ]

    print("\n--- Phase 1: Differential evolution (global search) ---")
    de_result = differential_evolution(
        neg_log_likelihood,
        bounds=bounds_de,
        args=(bin_centers, bin_heights, pi_delta),
        seed=42,
        maxiter=1000,
        tol=1e-10,
        polish=False,
        disp=True,
    )
    print(f"DE best NLL = {de_result.fun:.6f}")

    print("\n--- Phase 2: L-BFGS-B polish ---")
    res = minimize(
        neg_log_likelihood,
        x0=de_result.x,
        args=(bin_centers, bin_heights, pi_delta),
        method='L-BFGS-B',
        options={'maxiter': 2000, 'disp': True},
    )
    print(f"Final NLL = {res.fun:.6f},  converged = {res.success}")

    # 5) Extract final parameters
    pi_gam, pi_norm, k, beta, mu, sigma = unpack_params(res.x, pi_delta)

    print("\n" + "=" * 60)
    print("FITTED MIXTURE PARAMETERS  (δ(0) + Gamma(k,β) + Normal)")
    print("=" * 60)
    print(f"  pi_delta  = {pi_delta:.6f}   (FIXED — empirical zero fraction)")
    print(f"  pi_gam    = {pi_gam:.6f}   (Gamma decay)")
    print(f"  pi_norm   = {pi_norm:.6f}   (Normal bulk)")
    print(f"  sum(pi)   = {pi_delta + pi_gam + pi_norm:.6f}")
    print()
    print(f"  Delta component:  point mass at 0")
    print()
    print(f"  Gamma component:")
    print(f"    k (shape)   = {k:.6f}")
    print(f"    β (rate)    = {beta:.6f}")
    print(f"    mean = k/β  = {k/beta:.6f}")
    print(f"    var  = k/β² = {k/(beta**2):.6f}")
    print(f"    mode         = {max((k - 1) / beta, 0.0):.6f}  (0 if k < 1)")
    print()
    print(f"  Normal component:")
    print(f"    mu (loc)    = {mu:.6f}")
    print(f"    sigma       = {sigma:.6f}")
    print("=" * 60)

    # 6) Goodness-of-fit
    print("\n--- Goodness of fit ---")
    gof = compute_gof(raw_data, pi_delta, pi_gam, pi_norm, k, beta, mu, sigma)
    for name, val in gof.items():
        print(f"  {name:20s} = {val:.6f}")

    # Mixture mean: E[Y] = pi_delta*0 + pi_gam*(k/beta) + pi_norm*mu
    mix_mean = pi_gam * (k / beta) + pi_norm * mu
    print(f"\n  Mixture mean E[Y]  = {mix_mean:.6f}")
    print(f"  Empirical mean     = {raw_data.mean():.6f}")

    # 7) Plot
    x_max = raw_data.max() * 1.05
    xvals = np.linspace(1e-6, x_max, 500)

    pdf_gam_comp = pi_gam * gamma_pdf(xvals, k, beta)
    pdf_norm_comp = pi_norm * norm.pdf(xvals, loc=mu, scale=sigma)
    pdf_continuous = pdf_gam_comp + pdf_norm_comp

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})

    # --- Top panel: PDF overlay ---
    ax = axes[0]
    hist_counts, hist_edges = np.histogram(data_pos, bins=100, density=True)
    hist_density = hist_counts * (1.0 - pi_delta)
    hist_centers = 0.5 * (hist_edges[1:] + hist_edges[:-1])
    hist_widths = hist_edges[1:] - hist_edges[:-1]
    ax.bar(hist_centers, hist_density, width=hist_widths, alpha=0.4,
           color='steelblue', label=f'Empirical density (x > 0, scaled by 1−π_δ)')

    # Point mass annotation
    ax.axvline(0.0, color='red', lw=2, ls='-', alpha=0.7)
    ax.annotate(f'δ(0)\nπ={pi_delta:.3f}',
                xy=(0, pdf_continuous.max() * 0.85),
                xytext=(0.8, pdf_continuous.max() * 0.9),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.plot(xvals, pdf_continuous, 'k-', lw=2.5, label='Fitted continuous mixture')
    ax.plot(xvals, pdf_gam_comp, '--', color='orange', lw=1.5,
            label=f'Γ(k={k:.2f}, β={beta:.2f})  π={pi_gam:.3f}')
    ax.plot(xvals, pdf_norm_comp, '--', color='green', lw=1.5,
            label=f'N(μ={mu:.2f}, σ={sigma:.2f})  π={pi_norm:.3f}')

    ax.set_xlabel('log₂(CPM + 1)')
    ax.set_ylabel('Density')
    ax.set_title('Mixture Fit: δ(0) + Gamma(k,β) + Normal')
    ax.legend(fontsize=9)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)

    # --- Bottom panel: CDF comparison ---
    ax2 = axes[1]
    sorted_data = np.sort(raw_data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    fitted_cdf = (
        pi_delta * (sorted_data >= 0.0).astype(float)
        + continuous_cdf(sorted_data, pi_gam, pi_norm, k, beta, mu, sigma)
    )

    ax2.plot(sorted_data, ecdf, 'steelblue', lw=1.5, label='Empirical CDF')
    ax2.plot(sorted_data, fitted_cdf, 'k--', lw=1.5, label='Fitted CDF')
    ax2.set_xlabel('log₂(CPM + 1)')
    ax2.set_ylabel('CDF')
    ax2.legend()
    ax2.set_xlim(0, x_max)

    plt.tight_layout()
    plt.savefig("MDN_Mixture_Fit.pdf", dpi=150)
    print(f"\nPlot saved to MDN_Mixture_Fit.pdf")


if __name__ == "__main__":
    main()
