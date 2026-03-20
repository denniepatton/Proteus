"""
calibration.py — Conformalized Quantile Regression (CQR) with gene-regime pooling.

Implements K-fold conformalized quantile calibration with normalized
non-conformity scores, pooled within broad gene regimes.  This is the
standard post-hoc interval calibration strategy for preserving
heteroscedastic structure when K-fold models already exist.

Background & Rationale
----------------------
Conformal prediction provides finite-sample, distribution-free coverage
guarantees.  Standard (split) conformal adds a single scalar *q̂* to widen
intervals uniformly, which destroys the model's adaptive width structure.

**Conformalized Quantile Regression (CQR)** (Romano, Patterson & Candès,
NeurIPS 2019) overcomes this by defining non-conformity scores relative to
quantile predictions:

    A_i = max(L_i − y_i,  y_i − U_i,  0)

and computing *q̂* as the ⌈(1−α)(n+1)⌉/n empirical quantile of {A_i}.
At test time, intervals become [L − q̂, U + q̂], which preserves
the model's heteroscedastic width structure.

**Normalized scores** go further: dividing A_i by a local scale
S_ig = U_ig − L_ig makes the correction *multiplicative* in the
model's own uncertainty, so wider base intervals receive proportionally
larger adjustments:

    Â_ig = max(L_ig − y_ig,  y_ig − U_ig,  0) / (S_ig + ε)

At test time:

    [L_ig − q̂_r · S_ig,  U_ig + q̂_r · S_ig]

**Gene-regime pooling** addresses the fact that per-gene calibration
with only ~40 OOF samples is unstable (effective quantile rounds to max),
while global calibration ignores systematic differences across gene
expression regimes.  We partition genes into regimes defined by
(variance quartile × zero-rate quartile) = 16 bins, computed from the
development truth set, and pool non-conformity scores within each regime.

References
----------
- Romano, Patterson & Candès (2019). "Conformalized Quantile Regression."
  NeurIPS 2019. https://arxiv.org/abs/1905.03222
- Barber, Candès, Ramdas & Tibshirani (2021). "Predictive Inference with
  the Jackknife+." Annals of Statistics 49(1):486–507.

Public API
----------
- ``compute_gene_regimes``    — assign genes to (var_q, zero_q) regime bins
- ``compute_regime_qhats``    — CQR normalized scores → per-regime q̂
- ``calibrate_intervals``     — apply regime-based q̂ to new intervals
- ``evaluate_calibration``    — empirical coverage at multiple nominal levels
- ``RegimeCalibrator``        — stateful container bundling the above
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .analysis_utils import empirical_coverage


# ── Constants ──────────────────────────────────────────────────────────────

# Expression bounds on the log₂(CPM+1) scale.
EXPRESSION_FLOOR = 0.0
EXPRESSION_CEIL = 19.93  # log₂(10⁶ + 1)

# Regularisation term in the denominator of normalized non-conformity scores.
# Prevents division by zero for genes with near-zero interval width.
_EPS = 1e-6


# ── Gene Regimes ──────────────────────────────────────────────────────────


def compute_gene_regimes(
    truth_wide: pd.DataFrame,
    *,
    n_var_bins: int = 4,
    n_zero_bins: int = 4,
) -> pd.DataFrame:
    """Assign every gene to a regime bin based on variance and zero-rate.

    Regimes are defined **once** from development-set gene properties so
    that the same assignment can be applied to unseen data.

    Parameters
    ----------
    truth_wide : genes × samples truth expression matrix (development set).
    n_var_bins : number of variance quartile bins (default 4).
    n_zero_bins : number of zero-rate quartile bins (default 4).

    Returns
    -------
    DataFrame indexed by gene with columns:
        ``variance``   — gene variance across development samples.
        ``zero_rate``  — fraction of development samples where expression = 0.
        ``var_q``      — variance quartile label (0 … n_var_bins−1).
        ``zero_q``     — zero-rate quartile label (0 … n_zero_bins−1).
        ``regime``     — string "(var_q, zero_q)" for grouping convenience.
    """
    gene_var = truth_wide.var(axis=1, ddof=1)
    gene_zero_rate = (truth_wide == 0.0).mean(axis=1)

    var_q = pd.qcut(gene_var.rank(method="first"), n_var_bins,
                     labels=range(n_var_bins)).astype(int)
    zero_q = pd.qcut(gene_zero_rate.rank(method="first"), n_zero_bins,
                      labels=range(n_zero_bins)).astype(int)

    regime_df = pd.DataFrame({
        "variance": gene_var,
        "zero_rate": gene_zero_rate,
        "var_q": var_q,
        "zero_q": zero_q,
    }, index=truth_wide.index)
    regime_df["regime"] = (
        "(" + regime_df["var_q"].astype(str) + ", "
        + regime_df["zero_q"].astype(str) + ")"
    )
    return regime_df


# ── Normalized CQR Scores ────────────────────────────────────────────────


def _normalized_scores(
    truth: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized CQR non-conformity scores and local scales.

    For each observation (i, g):
        S_ig = U_ig − L_ig                         (local scale)
        A_ig = max(L_ig − y_ig, y_ig − U_ig, 0)   (raw CQR score)
             / (S_ig + ε)                           (normalized)

    Parameters
    ----------
    truth, ci_lo, ci_hi : aligned arrays of any shape.

    Returns
    -------
    (scores, scales) — arrays of the same shape as inputs.
    """
    scales = ci_hi - ci_lo  # S_ig
    raw = np.maximum(np.maximum(ci_lo - truth, truth - ci_hi), 0.0)
    scores = raw / (scales + _EPS)
    return scores, scales


def compute_regime_qhats(
    truth_wide: pd.DataFrame,
    ci_lo_wide: pd.DataFrame,
    ci_hi_wide: pd.DataFrame,
    regime_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute per-regime conformal quantile q̂_r from OOF data.

    For every regime *r*, collects all normalized non-conformity scores
    from genes in that regime, then takes the ⌈(1−α)(n+1)⌉/n empirical
    quantile (the standard finite-sample-corrected conformal quantile).

    Parameters
    ----------
    truth_wide, ci_lo_wide, ci_hi_wide :
        genes × samples wide DataFrames of OOF truth and base quantile
        predictions (L = q₂.₅%, U = q₉₇.₅%).
    regime_df :
        output of ``compute_gene_regimes``.  Must cover all genes in the
        prediction matrices.
    alpha : miscoverage budget (default 0.05 → target 95% coverage).

    Returns
    -------
    Dict mapping regime label → q̂ (float).  Regimes with fewer than 10
    valid observations get q̂ = 0.0 (no correction).
    """
    common_genes = (truth_wide.index
                    .intersection(ci_lo_wide.index)
                    .intersection(ci_hi_wide.index)
                    .intersection(regime_df.index))
    common_samples = (truth_wide.columns
                      .intersection(ci_lo_wide.columns)
                      .intersection(ci_hi_wide.columns))

    t = truth_wide.loc[common_genes, common_samples].values
    lo = ci_lo_wide.loc[common_genes, common_samples].values
    hi = ci_hi_wide.loc[common_genes, common_samples].values
    regimes = regime_df.loc[common_genes, "regime"].values

    scores, _ = _normalized_scores(t, lo, hi)

    unique_regimes = np.unique(regimes)
    q_hats: Dict[str, float] = {}

    for r in unique_regimes:
        mask = regimes == r  # boolean mask over genes
        regime_scores = scores[mask].flatten()
        valid = regime_scores[np.isfinite(regime_scores)]
        n = len(valid)
        if n < 10:
            q_hats[r] = 0.0
            continue
        quantile_level = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
        q_hats[r] = float(np.quantile(valid, quantile_level))

    return q_hats


# ── Apply Calibration ────────────────────────────────────────────────────


def calibrate_intervals(
    ci_lo_wide: pd.DataFrame,
    ci_hi_wide: pd.DataFrame,
    regime_df: pd.DataFrame,
    q_hats: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply regime-based CQR calibration to interval predictions.

    For each gene *g* in regime *r*, the calibrated interval is:
        [L_ig − q̂_r · S_ig,  U_ig + q̂_r · S_ig]
    where S_ig = U_ig − L_ig is the base interval width.

    Intervals are clamped to the valid expression range
    [EXPRESSION_FLOOR, EXPRESSION_CEIL] and prevented from inverting.

    Parameters
    ----------
    ci_lo_wide, ci_hi_wide : genes × samples base interval bounds.
    regime_df : gene regime assignments (from ``compute_gene_regimes``).
    q_hats : per-regime q̂ values (from ``compute_regime_qhats``).

    Returns
    -------
    (calibrated_lo, calibrated_hi) — same shape as inputs.
    """
    common_genes = ci_lo_wide.index.intersection(regime_df.index)
    lo = ci_lo_wide.loc[common_genes]
    hi = ci_hi_wide.loc[common_genes]

    # Build per-gene q̂ vector from regime assignments
    gene_regimes = regime_df.loc[common_genes, "regime"]
    gene_q = gene_regimes.map(q_hats).fillna(0.0).astype(np.float64)

    # Local scale S_ig = U_ig − L_ig  (genes × samples)
    S = hi.values - lo.values  # ndarray

    # q̂_r broadcast to (genes, 1) then multiplied by S  →  (genes, samples)
    q_arr = gene_q.values[:, np.newaxis]
    adjustment = q_arr * S

    lo_adj = pd.DataFrame(
        lo.values - adjustment,
        index=lo.index, columns=lo.columns,
    )
    hi_adj = pd.DataFrame(
        hi.values + adjustment,
        index=hi.index, columns=hi.columns,
    )

    # Prevent interval inversion: collapse to midpoint
    inverted = lo_adj.values > hi_adj.values
    if inverted.any():
        mid = (lo.values + hi.values) / 2
        lo_arr = lo_adj.values
        hi_arr = hi_adj.values
        lo_arr[inverted] = mid[inverted]
        hi_arr[inverted] = mid[inverted]
        lo_adj = pd.DataFrame(lo_arr, index=lo.index, columns=lo.columns)
        hi_adj = pd.DataFrame(hi_arr, index=hi.index, columns=hi.columns)

    # Clamp to valid expression range
    lo_adj = lo_adj.clip(lower=EXPRESSION_FLOOR)
    hi_adj = hi_adj.clip(upper=EXPRESSION_CEIL)

    # Restore any genes not in regime_df (unchanged)
    missing = ci_lo_wide.index.difference(common_genes)
    if len(missing):
        lo_adj = pd.concat([lo_adj, ci_lo_wide.loc[missing]])
        hi_adj = pd.concat([hi_adj, ci_hi_wide.loc[missing]])
        lo_adj = lo_adj.loc[ci_lo_wide.index]
        hi_adj = hi_adj.loc[ci_hi_wide.index]

    return lo_adj, hi_adj


# ── Evaluation ───────────────────────────────────────────────────────────


def evaluate_calibration(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
    ci_lo_wide: pd.DataFrame,
    ci_hi_wide: pd.DataFrame,
    regime_df: pd.DataFrame,
    q_hats: Dict[str, float],
    *,
    nominal_levels: Sequence[float] = (0.50, 0.80, 0.90, 0.95),
) -> Dict[str, Any]:
    """Comprehensive calibration evaluation on held-out data.

    For each nominal level, the interval is scaled symmetrically around the
    point prediction μ such that its width corresponds to the nominal
    fraction of the full (calibrated) 95% interval.

    Computes:
      - overall empirical coverage at each nominal level
      - mean and median interval width (at 95%)
      - per-regime coverage (at 95%)
      - raw arrays for plotting

    Parameters
    ----------
    truth_wide, pred_wide, ci_lo_wide, ci_hi_wide :
        genes × samples DataFrames for the evaluation split.
    regime_df : gene regime assignments.
    q_hats : per-regime q̂ values.
    nominal_levels : target coverage levels to evaluate.

    Returns
    -------
    Dict with keys:
        ``nominal``       — list of nominal levels
        ``empirical``     — list of corresponding empirical coverages
        ``mean_width``    — mean calibrated interval width
        ``median_width``  — median calibrated interval width
        ``regime_coverage`` — dict {regime: empirical_coverage_at_95%}
        ``n_observations``  — total number of valid observation pairs
    """
    # Apply calibration
    cal_lo, cal_hi = calibrate_intervals(ci_lo_wide, ci_hi_wide, regime_df, q_hats)

    # Align everything
    common_genes = (truth_wide.index.intersection(pred_wide.index)
                    .intersection(cal_lo.index).intersection(cal_hi.index))
    common_samples = (truth_wide.columns.intersection(pred_wide.columns)
                      .intersection(cal_lo.columns).intersection(cal_hi.columns))

    t = truth_wide.loc[common_genes, common_samples].values
    mu = pred_wide.loc[common_genes, common_samples].values
    lo = cal_lo.loc[common_genes, common_samples].values
    hi = cal_hi.loc[common_genes, common_samples].values

    mask = np.isfinite(t) & np.isfinite(mu) & np.isfinite(lo) & np.isfinite(hi)
    t_f, mu_f = t[mask], mu[mask]
    lo_f, hi_f = lo[mask], hi[mask]

    if len(t_f) < 10:
        return {
            "nominal": list(nominal_levels),
            "empirical": [np.nan] * len(nominal_levels),
            "mean_width": np.nan,
            "median_width": np.nan,
            "regime_coverage": {},
            "n_observations": 0,
        }

    # ── Overall coverage at multiple nominal levels ──
    max_nominal = max(nominal_levels)
    empiricals = []
    for level in nominal_levels:
        scale = level / max_nominal
        adj_lo = mu_f - scale * (mu_f - lo_f)
        adj_hi = mu_f + scale * (hi_f - mu_f)
        coverage = float(((t_f >= adj_lo) & (t_f <= adj_hi)).sum() / len(t_f))
        empiricals.append(coverage)

    # ── Interval widths at full (95%) level ──
    widths = hi_f - lo_f
    mean_width = float(np.mean(widths))
    median_width = float(np.median(widths))

    # ── Per-regime coverage at 95% ──
    regimes_arr = regime_df.loc[common_genes, "regime"].values
    regime_coverage: Dict[str, float] = {}
    for r in np.unique(regimes_arr):
        gene_mask = regimes_arr == r
        # Build a 2-D mask: genes in regime × all samples → flatten
        r_mask = np.zeros_like(mask, dtype=bool)
        r_mask[gene_mask, :] = mask[gene_mask, :]
        r_t = t[r_mask]
        r_lo = lo[r_mask]
        r_hi = hi[r_mask]
        if len(r_t) > 0:
            regime_coverage[r] = float(((r_t >= r_lo) & (r_t <= r_hi)).sum() / len(r_t))

    return {
        "nominal": list(nominal_levels),
        "empirical": empiricals,
        "mean_width": mean_width,
        "median_width": median_width,
        "regime_coverage": regime_coverage,
        "n_observations": int(len(t_f)),
    }


# ── Stateful Container ──────────────────────────────────────────────────


@dataclass
class RegimeCalibrator:
    """Stateful calibration object that bundles regime definitions and q̂ values.

    Typical usage (BTS-CV example)::

        # 1. Fit on OOF development data
        regime_df = compute_gene_regimes(dev_truth)
        calibrator = RegimeCalibrator.fit(
            dev_truth, dev_ci_lo, dev_ci_hi, regime_df, alpha=0.05
        )

        # 2. Apply to new data
        cal_lo, cal_hi = calibrator.apply(new_ci_lo, new_ci_hi)

        # 3. Evaluate (if truth available)
        eval_dict = calibrator.evaluate(
            eval_truth, eval_pred, eval_ci_lo, eval_ci_hi
        )
    """
    regime_df: pd.DataFrame
    q_hats: Dict[str, float]
    alpha: float = 0.05
    n_regimes: int = 0
    n_calibration_obs: int = 0

    @classmethod
    def fit(
        cls,
        truth_wide: pd.DataFrame,
        ci_lo_wide: pd.DataFrame,
        ci_hi_wide: pd.DataFrame,
        regime_df: pd.DataFrame,
        *,
        alpha: float = 0.05,
    ) -> "RegimeCalibrator":
        """Fit q̂ values from OOF data.

        Parameters
        ----------
        truth_wide, ci_lo_wide, ci_hi_wide :
            OOF (out-of-fold) truth and base quantile predictions.
        regime_df : gene regime assignments from ``compute_gene_regimes``.
        alpha : miscoverage budget.

        Returns
        -------
        Fitted ``RegimeCalibrator`` instance.
        """
        q_hats = compute_regime_qhats(
            truth_wide, ci_lo_wide, ci_hi_wide, regime_df, alpha=alpha,
        )
        # Count total calibration observations
        common_genes = (truth_wide.index.intersection(ci_lo_wide.index)
                        .intersection(ci_hi_wide.index)
                        .intersection(regime_df.index))
        common_samples = (truth_wide.columns.intersection(ci_lo_wide.columns)
                          .intersection(ci_hi_wide.columns))
        n_obs = len(common_genes) * len(common_samples)
        return cls(
            regime_df=regime_df,
            q_hats=q_hats,
            alpha=alpha,
            n_regimes=len(q_hats),
            n_calibration_obs=n_obs,
        )

    def apply(
        self,
        ci_lo_wide: pd.DataFrame,
        ci_hi_wide: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply calibration to new interval predictions.

        Returns (calibrated_lo, calibrated_hi).
        """
        return calibrate_intervals(
            ci_lo_wide, ci_hi_wide, self.regime_df, self.q_hats,
        )

    def evaluate(
        self,
        truth_wide: pd.DataFrame,
        pred_wide: pd.DataFrame,
        ci_lo_wide: pd.DataFrame,
        ci_hi_wide: pd.DataFrame,
        *,
        nominal_levels: Sequence[float] = (0.50, 0.80, 0.90, 0.95),
    ) -> Dict[str, Any]:
        """Full evaluation on held-out data (delegates to ``evaluate_calibration``)."""
        return evaluate_calibration(
            truth_wide, pred_wide, ci_lo_wide, ci_hi_wide,
            self.regime_df, self.q_hats,
            nominal_levels=nominal_levels,
        )

    def summary(self) -> str:
        """Human-readable summary of the calibration state."""
        q_vals = list(self.q_hats.values())
        lines = [
            f"RegimeCalibrator (α={self.alpha})",
            f"  {self.n_regimes} regimes, {self.n_calibration_obs} calibration obs",
        ]
        if q_vals:
            lines.append(
                f"  q̂ range: [{min(q_vals):.4f}, {max(q_vals):.4f}]  "
                f"median={float(np.median(q_vals)):.4f}  "
                f"mean={float(np.mean(q_vals)):.4f}"
            )
        return "\n".join(lines)
