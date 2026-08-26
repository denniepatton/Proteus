"""
reliability.py — Post-hoc gene-level reliability model from OOF predictions.

Builds a ridge regression that maps gene-level prediction features (computed
from frozen OOF outputs) to Fisher-z-transformed per-gene Pearson *r* — a
**meta-prediction** of how reliably the model predicts each gene.

Background & Rationale
----------------------
The question is not "is this interval calibrated?" but rather:

    For this gene, on this new cohort, how likely is Pearson *r* to be good?

This is a meta-prediction problem.  Raw CI width alone is too crude.  The
most defensible post-hoc approach is to learn the mapping from OOF behaviour
using features that the frozen model already produces.

Method
------
1. Using the 5-fold OOF predictions on development data, for each **gene g
   within each OOF fold k** compute:

   - **Target**: Fisher z-transform of Pearson *r* for gene g across samples
     in fold k:

         T_gk = z(r_gk)  where  z(r) = ½ · ln((1+r)/(1−r))

   - **Features** (all computed from OOF predictions for gene g in fold k):

     Core uncertainty:
       f1  mean_width:       mean_i(U_ig − L_ig)
       f2  median_width:     median_i(U_ig − L_ig)
       f3  mean_entropy:     mean_i(H_ig)  — mixture entropy over 3 components
       f4  mean_hurdle_amb:  mean_i(p_δ · (1 − p_δ))
       f5  mean_cont_amb:    mean_i(π_γ · π_n)

     Confidence / decisiveness:
       f6  mean_pmax:        mean_i(max(π_δ, π_γ, π_n))
       f7  median_pmax:      median_i(max(π_δ, π_γ, π_n))
       f8  mean_gap:         mean_i(p_max − p_second)

     Component uncertainty:
       f9  mean_u_normal:    mean_i(π_n · σ_n²)
       f10 mean_u_gamma:     mean_i(π_γ · k/β²)  — Gamma variance weighted
                             by mixing weight, k = K_GAMMA = 0.75

     Epistemic (fold-to-fold, requires all-fold inference):
       f11 mean_d_mu:        mean_i(SD_m(μ^(m)_ig))
       f12 mean_d_width:     mean_i(SD_m(w^(m)_ig))
       f13 mean_d_delta:     mean_i(SD_m(p_δ^(m)_ig))

     Predictive behaviour:
       f14 pred_sd:          SD_i(μ_ig) across samples

     Data properties (from development truth, static per gene):
       f15 gene_zero_rate:   Pr(y_g = 0)
       f16 gene_variance:    Var(y_g)

2. Fit a **ridge regression** (linear model with L₂ penalty):

       T_gk ~ f₁ + ⋯ + f₁₆

   Training data has one row per (gene, fold), giving ~5 × N_genes rows.

3. On new data without truth:

   - compute f₁…f₁₆ from predictions only (f₁₅, f₁₆ from development truth)
   - predict Fisher-z reliability
   - convert back:  r̂_g = tanh(T̂_g)

Evaluation (on holdout)
-----------------------
- Spearman correlation between predicted r̂_g and true holdout r_g
- RMSE of r̂_g vs true r_g (in both r-space and z-space)
- Top-K retention curve: sort genes by predicted r̂_g, take top 10%, 20%,
  …, 100%, compute actual mean Pearson r in each retained set.

References
----------
- Fisher, R. A. (1921). "On the 'probable error' of a coefficient of
  correlation deduced from a small sample." Metron, 1:3–32.
- Hoerl & Kennard (1970). "Ridge Regression: Biased Estimation for
  Nonorthogonal Problems." Technometrics, 12(1):55–67.

Public API
----------
- ``compute_gene_features``    — feature matrix for one set of predictions
- ``ReliabilityModel``         — fit / predict / evaluate container
- ``retention_curve``          — top-K enrichment analysis
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from proteus_constants import K_GAMMA
from .analysis_utils import MIN_CORRELATION_N


# ── Fisher z-transform ───────────────────────────────────────────────────


def _fisher_z(r: float) -> float:
    """Fisher z-transform: z = ½ ln((1+r)/(1−r)) = arctanh(r).

    Clamps |r| < 0.9999 to avoid ±∞.
    """
    r = np.clip(r, -0.9999, 0.9999)
    return float(np.arctanh(r))


def _inv_fisher_z(z: float) -> float:
    """Inverse Fisher z (= tanh): r = tanh(z)."""
    return float(np.tanh(z))


# ── Constants ────────────────────────────────────────────────────────────

# Minimum value for log to avoid log(0)
_LOG_EPS: float = 1e-12


# ── Gene-Level Feature Computation ───────────────────────────────────────


def compute_gene_features(
    pred_wide: pd.DataFrame,
    ci_lo_wide: pd.DataFrame,
    ci_hi_wide: pd.DataFrame,
    dev_truth_wide: pd.DataFrame,
    *,
    p0_wide: Optional[pd.DataFrame] = None,
    pi_gamma_wide: Optional[pd.DataFrame] = None,
    pi_normal_wide: Optional[pd.DataFrame] = None,
    gamma_beta_wide: Optional[pd.DataFrame] = None,
    normal_stdev_wide: Optional[pd.DataFrame] = None,
    fold_preds: Optional[Dict[int, pd.DataFrame]] = None,
    fold_widths: Optional[Dict[int, pd.DataFrame]] = None,
    fold_p0s: Optional[Dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Compute per-gene feature vectors from one set of predictions.

    All wide inputs are genes × samples DataFrames.  ``dev_truth_wide``
    provides development-set properties (zero-rate, variance) that are
    static features of each gene, not of the predictions.

    Parameters
    ----------
    pred_wide : point predictions (μ_ig), genes × samples.
    ci_lo_wide : lower quantile predictions (L_ig).
    ci_hi_wide : upper quantile predictions (U_ig).
    dev_truth_wide : development truth expression (for zero rate / variance).
    p0_wide : pi_delta — hurdle (zero-component) mixing weight.
    pi_gamma_wide : pi_gamma — Gamma component mixing weight.
    pi_normal_wide : pi_normal — Normal component mixing weight.
    gamma_beta_wide : Gamma rate parameter β.
    normal_stdev_wide : Normal component standard deviation σ_n.
    fold_preds : {fold: genes × samples pred} for epistemic fold-to-fold
        features.  Requires all-fold inference (each fold model predicting
        all samples).  None if unavailable.
    fold_widths : {fold: genes × samples CI width} for epistemic features.
    fold_p0s : {fold: genes × samples pi_delta} for epistemic features.

    Returns
    -------
    DataFrame indexed by gene with 16 feature columns (f1…f16).
    Features that cannot be computed (missing MDN columns or fold data)
    are set to NaN.
    """
    # Genes with all-NaN slices (no valid samples in a fold) correctly
    # produce NaN features, filtered during ridge fitting.  Suppress the
    # expected numpy warnings for nanmean/nanmedian on empty slices.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "Mean of empty slice", RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore", "All-NaN slice encountered", RuntimeWarning,
        )
        return _compute_gene_features_impl(
            pred_wide, ci_lo_wide, ci_hi_wide, dev_truth_wide,
            p0_wide=p0_wide, pi_gamma_wide=pi_gamma_wide,
            pi_normal_wide=pi_normal_wide,
            gamma_beta_wide=gamma_beta_wide,
            normal_stdev_wide=normal_stdev_wide,
            fold_preds=fold_preds, fold_widths=fold_widths,
            fold_p0s=fold_p0s,
        )


def _compute_gene_features_impl(
    pred_wide: pd.DataFrame,
    ci_lo_wide: pd.DataFrame,
    ci_hi_wide: pd.DataFrame,
    dev_truth_wide: pd.DataFrame,
    *,
    p0_wide: Optional[pd.DataFrame] = None,
    pi_gamma_wide: Optional[pd.DataFrame] = None,
    pi_normal_wide: Optional[pd.DataFrame] = None,
    gamma_beta_wide: Optional[pd.DataFrame] = None,
    normal_stdev_wide: Optional[pd.DataFrame] = None,
    fold_preds: Optional[Dict[int, pd.DataFrame]] = None,
    fold_widths: Optional[Dict[int, pd.DataFrame]] = None,
    fold_p0s: Optional[Dict[int, pd.DataFrame]] = None,
) -> pd.DataFrame:
    genes = pred_wide.index
    samples = pred_wide.columns
    n_genes = len(genes)
    features: Dict[str, np.ndarray] = {}

    # ── Helper: align and extract as float64 array ──
    def _arr(df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
        if df is None:
            return None
        return df.reindex(index=genes, columns=samples).values.astype(np.float64)

    mu = _arr(pred_wide)
    lo = _arr(ci_lo_wide)
    hi = _arr(ci_hi_wide)
    p_delta = _arr(p0_wide)
    p_gamma = _arr(pi_gamma_wide)
    p_normal = _arr(pi_normal_wide)
    beta = _arr(gamma_beta_wide)
    sigma_n = _arr(normal_stdev_wide)

    # ── MDN component arrays available? ──
    has_components = all(a is not None for a in (p_delta, p_gamma, p_normal))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Core uncertainty (f1–f5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # f1: mean CI width
    widths = hi - lo  # (n_genes, n_samples)
    features["f1_mean_width"] = np.nanmean(widths, axis=1)

    # f2: median CI width
    features["f2_median_width"] = np.nanmedian(widths, axis=1)

    # f3: mean mixture entropy H = -Σ p_c ln(p_c)
    if has_components:
        # Stack (3, n_genes, n_samples), compute per-component -p*ln(p)
        probs = np.stack([p_delta, p_gamma, p_normal], axis=0)
        log_probs = np.log(np.clip(probs, _LOG_EPS, None))
        H = -np.sum(probs * log_probs, axis=0)  # (n_genes, n_samples)
        features["f3_mean_entropy"] = np.nanmean(H, axis=1)
    else:
        features["f3_mean_entropy"] = np.full(n_genes, np.nan)

    # f4: mean hurdle ambiguity — mean_i(p_δ · (1 − p_δ))
    if p_delta is not None:
        features["f4_mean_hurdle_amb"] = np.nanmean(p_delta * (1.0 - p_delta), axis=1)
    else:
        features["f4_mean_hurdle_amb"] = np.full(n_genes, np.nan)

    # f5: mean continuous ambiguity — mean_i(π_γ · π_n)
    if p_gamma is not None and p_normal is not None:
        features["f5_mean_cont_amb"] = np.nanmean(p_gamma * p_normal, axis=1)
    else:
        features["f5_mean_cont_amb"] = np.full(n_genes, np.nan)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Confidence / decisiveness (f6–f8)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    if has_components:
        probs = np.stack([p_delta, p_gamma, p_normal], axis=0)  # (3, G, S)
        p_max = np.max(probs, axis=0)                           # (G, S)

        # f6: mean p_max
        features["f6_mean_pmax"] = np.nanmean(p_max, axis=1)

        # f7: median p_max
        features["f7_median_pmax"] = np.nanmedian(p_max, axis=1)

        # f8: mean gap (p_max − p_second)
        sorted_probs = np.sort(probs, axis=0)  # ascending along component axis
        p_second = sorted_probs[-2]            # second-largest
        gap = p_max - p_second
        features["f8_mean_gap"] = np.nanmean(gap, axis=1)
    else:
        features["f6_mean_pmax"] = np.full(n_genes, np.nan)
        features["f7_median_pmax"] = np.full(n_genes, np.nan)
        features["f8_mean_gap"] = np.full(n_genes, np.nan)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Component uncertainty (f9–f10)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # f9: mean normal uncertainty — mean_i(π_n · σ_n²)
    if p_normal is not None and sigma_n is not None:
        features["f9_mean_u_normal"] = np.nanmean(p_normal * sigma_n ** 2, axis=1)
    else:
        features["f9_mean_u_normal"] = np.full(n_genes, np.nan)

    # f10: mean gamma uncertainty — mean_i(π_γ · k/β²)
    #      Var(Gamma(k, rate=β)) = k/β²
    if p_gamma is not None and beta is not None:
        gamma_var = K_GAMMA / (beta ** 2 + _LOG_EPS)
        features["f10_mean_u_gamma"] = np.nanmean(p_gamma * gamma_var, axis=1)
    else:
        features["f10_mean_u_gamma"] = np.full(n_genes, np.nan)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Epistemic / fold-to-fold features (f11–f13)
    # Require all-fold inference: each fold model predicts ALL samples.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _fold_to_fold_sd(
        fold_dict: Optional[Dict[int, pd.DataFrame]],
    ) -> np.ndarray:
        """Mean across samples of SD across folds for each gene."""
        if fold_dict is None or len(fold_dict) < 2:
            return np.full(n_genes, np.nan)
        fold_keys = sorted(fold_dict.keys())
        stacked = np.stack([
            fold_dict[k].reindex(index=genes, columns=samples).values
            for k in fold_keys
        ], axis=0)  # (n_folds, n_genes, n_samples)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Degrees of freedom <= 0",
                                    category=RuntimeWarning)
            sd = np.nanstd(stacked, axis=0, ddof=1)  # (G, S)
        return np.nanmean(sd, axis=1)

    # f11: mean fold-to-fold SD of prediction_mean
    features["f11_mean_d_mu"] = _fold_to_fold_sd(fold_preds)

    # f12: mean fold-to-fold SD of CI width
    features["f12_mean_d_width"] = _fold_to_fold_sd(fold_widths)

    # f13: mean fold-to-fold SD of pi_delta
    features["f13_mean_d_delta"] = _fold_to_fold_sd(fold_p0s)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Predictive behaviour (f14)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # f14: SD of point predictions across samples
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0",
                                category=RuntimeWarning)
        features["f14_pred_sd"] = np.nanstd(mu, axis=1, ddof=1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Data properties (f15–f16) — from development truth, static per gene
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    dev = dev_truth_wide.reindex(index=genes)

    # f15: gene zero rate
    features["f15_gene_zero_rate"] = (dev == 0.0).mean(axis=1).values

    # f16: gene variance
    features["f16_gene_variance"] = dev.var(axis=1, ddof=1).values

    return pd.DataFrame(features, index=genes)


def _compute_gene_pearson_r(
    truth_wide: pd.DataFrame,
    pred_wide: pd.DataFrame,
) -> pd.Series:
    """Per-gene Pearson r across samples. Returns Series indexed by gene.

    Genes with fewer than ``MIN_CORRELATION_N`` valid samples or zero
    truth variance are excluded. Genes with variable truth but constant
    predictions receive 0.0.
    """
    common_genes = truth_wide.index.intersection(pred_wide.index)
    common_samples = truth_wide.columns.intersection(pred_wide.columns)
    tw = truth_wide.loc[common_genes, common_samples]
    pw = pred_wide.loc[common_genes, common_samples]

    r_values = {}
    for gene in common_genes:
        t = tw.loc[gene].values.astype(np.float64)
        p = pw.loc[gene].values.astype(np.float64)
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() < MIN_CORRELATION_N:
            r_values[gene] = np.nan
            continue
        tv = t[mask]
        pv = p[mask]
        if float(np.ptp(tv)) <= 0.0:
            r_values[gene] = np.nan
            continue
        if float(np.ptp(pv)) <= 0.0:
            r_values[gene] = 0.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_values[gene] = float(pearsonr(tv, pv)[0])

    return pd.Series(r_values, name="pearson_r")


# ── Ridge Regression (self-contained, no sklearn dependency) ─────────────


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Fit ridge regression: y = X @ w + b.

    Features are z-score standardized (per column) before fitting so the L₂
    penalty acts uniformly across features that live on very different scales
    (interval widths, entropies, variances). The closed-form solution is
    computed in standardized space::

        w_std = (Z^T Z + α I)^{-1} Z^T (y − ȳ),   Z = (X − X̄) / σ_X

    and then rescaled back into raw-feature space so that
    ``_predict_ridge`` can operate directly on unstandardized features:

        w = w_std / σ_X
        b = ȳ − X̄ · w

    Constant columns (σ_X = 0) are assigned σ = 1 so they contribute nothing
    after centering rather than producing a divide-by-zero.

    Parameters
    ----------
    X : (n, p) feature matrix.
    y : (n,) target vector.
    alpha : L₂ penalty strength (applied in standardized feature space).

    Returns
    -------
    (weights, intercept) in raw-feature space: weights is (p,), intercept scalar.
    """
    n, p = X.shape
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std = np.where(X_std == 0.0, 1.0, X_std)
    y_mean = y.mean()

    Z = (X - X_mean) / X_std
    yc = y - y_mean

    # Closed-form ridge in standardized space: w_std = (Z^T Z + alpha I)^{-1} Z^T yc
    A = Z.T @ Z + alpha * np.eye(p)
    b_vec = Z.T @ yc
    weights_std = np.linalg.solve(A, b_vec)

    # Fold the standardization back into raw-feature-space weights/intercept.
    weights = weights_std / X_std
    intercept = y_mean - X_mean @ weights
    return weights, float(intercept)


def _predict_ridge(
    X: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Predict with fitted ridge model."""
    return X @ weights + intercept


# ── Reliability Model ────────────────────────────────────────────────────


@dataclass
class ReliabilityModel:
    """Post-hoc gene-level reliability model.

    Predicts Fisher-z-transformed Pearson *r* for each gene from
    prediction-derived features, then converts to predicted Pearson r.

    Typical usage::

        # Fit on per-fold OOF development data
        model = ReliabilityModel.fit(
            fold_data=cv_full["fold_data"],
            dev_truth_wide=canonical_truth,
            ridge_alpha=1.0,
        )

        # Predict on new data
        new_features = compute_gene_features(new_pred, ...)
        predicted_r = model.predict(new_features)

        # Evaluate on holdout
        eval_dict = model.evaluate(predicted_r, true_holdout_r)
    """
    weights: np.ndarray
    intercept: float
    ridge_alpha: float
    feature_columns: List[str]
    n_train_rows: int = 0
    n_train_genes: int = 0
    train_r2: float = np.nan

    @classmethod
    def fit(
        cls,
        fold_data: Dict[int, Dict[str, pd.DataFrame]],
        dev_truth_wide: pd.DataFrame,
        *,
        ridge_alpha: float = 1.0,
        min_samples_per_fold: int = 3,
    ) -> "ReliabilityModel":
        """Fit the reliability model from per-fold OOF features.

        For each gene g in each OOF fold k, computes gene-level features
        from fold k's held-out samples and the per-fold Pearson r target:

            T_gk = Fisher-z( r(truth_gk, pred_gk) )

        This gives ~5 × N_genes training rows for the ridge regression.
        Features f15/f16 (gene zero-rate, variance) come from the full
        development truth and are shared across folds.

        Parameters
        ----------
        fold_data : {fold_int: {"pred_wide": ..., "truth_wide": ...,
            "ci_lo_wide": ..., "ci_hi_wide": ..., "p0_wide": ...,
            "pi_gamma_wide": ..., "pi_normal_wide": ...,
            "gamma_beta_wide": ..., "normal_stdev_wide": ...}}
            Per-fold wide DataFrames from ``load_cv_predictions_proteus_full``.
        dev_truth_wide : Full development truth (genes × all samples)
            for computing static gene-level features.
        ridge_alpha : L₂ penalty strength.
        min_samples_per_fold : Minimum samples in a fold to compute r.

        Returns
        -------
        Fitted ``ReliabilityModel`` instance.
        """
        feature_rows: List[pd.DataFrame] = []
        target_rows: List[pd.Series] = []
        gene_set: set = set()

        for fold_k, fd in sorted(fold_data.items()):
            pred_k = fd["pred_wide"]
            truth_k = fd.get("truth_wide")
            ci_lo_k = fd.get("ci_lo_wide")
            ci_hi_k = fd.get("ci_hi_wide")

            if truth_k is None or ci_lo_k is None or ci_hi_k is None:
                continue
            if pred_k.shape[1] < min_samples_per_fold:
                continue

            # Compute features for this fold's samples
            feat_k = compute_gene_features(
                pred_k, ci_lo_k, ci_hi_k, dev_truth_wide,
                p0_wide=fd.get("p0_wide"),
                pi_gamma_wide=fd.get("pi_gamma_wide"),
                pi_normal_wide=fd.get("pi_normal_wide"),
                gamma_beta_wide=fd.get("gamma_beta_wide"),
                normal_stdev_wide=fd.get("normal_stdev_wide"),
            )

            # Per-fold Pearson r target
            r_k = _compute_gene_pearson_r(truth_k, pred_k)
            z_k = r_k.apply(_fisher_z)

            # Use fold index as suffix to make index unique across folds
            feat_k.index = [f"{g}__fold{fold_k}" for g in feat_k.index]
            z_k.index = feat_k.index
            gene_set.update(fd["pred_wide"].index)

            feature_rows.append(feat_k)
            target_rows.append(z_k)

        if not feature_rows:
            raise ValueError("No valid folds for reliability model fitting")

        all_features = pd.concat(feature_rows)
        all_targets = pd.concat(target_rows)

        # Drop features that are all-NaN
        valid_cols = [c for c in all_features.columns if all_features[c].notna().any()]
        all_features = all_features[valid_cols]

        # Combined mask: all features finite AND target finite
        mask = all_features.notna().all(axis=1) & all_targets.notna()
        all_features = all_features.loc[mask]
        all_targets = all_targets.loc[mask]

        if len(all_features) < 10:
            raise ValueError(
                f"Too few valid (gene, fold) rows for reliability model: "
                f"{len(all_features)}"
            )

        X = all_features.values.astype(np.float64)
        y = all_targets.values.astype(np.float64)

        weights, intercept = _fit_ridge(X, y, alpha=ridge_alpha)

        # Training R²
        y_hat = _predict_ridge(X, weights, intercept)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return cls(
            weights=weights,
            intercept=intercept,
            ridge_alpha=ridge_alpha,
            feature_columns=valid_cols,
            n_train_rows=len(all_features),
            n_train_genes=len(gene_set),
            train_r2=r2,
        )

    def predict(self, feature_df: pd.DataFrame) -> pd.Series:
        """Predict gene-level reliability (Pearson r) for new data.

        Genes with missing features get NaN predictions.

        Parameters
        ----------
        feature_df : DataFrame from ``compute_gene_features``.

        Returns
        -------
        Series of predicted Pearson r̂_g, indexed by gene.
        """
        feat = feature_df[self.feature_columns]
        result = pd.Series(np.nan, index=feat.index, name="predicted_r")
        valid_mask = feat.notna().all(axis=1)
        if valid_mask.sum() == 0:
            return result

        X = feat.loc[valid_mask].values.astype(np.float64)
        z_hat = _predict_ridge(X, self.weights, self.intercept)
        r_hat = np.tanh(z_hat)  # inverse Fisher z
        result.loc[valid_mask] = r_hat
        return result

    def evaluate(
        self,
        predicted_r: pd.Series,
        true_r: pd.Series,
    ) -> Dict[str, Any]:
        """Evaluate reliability predictions against true holdout gene r.

        Returns dict with:
            spearman_rho — rank correlation of predicted vs true r
            rmse_r       — RMSE in Pearson r space
            rmse_z       — RMSE in Fisher-z space
            n_genes      — number of genes evaluated
            retention    — top-K retention curve dict
        """
        common = predicted_r.index.intersection(true_r.index)
        pred = predicted_r.loc[common]
        true = true_r.loc[common]
        mask = pred.notna() & true.notna()
        pred, true = pred[mask], true[mask]

        if len(pred) < 5:
            return {
                "spearman_rho": np.nan,
                "rmse_r": np.nan,
                "rmse_z": np.nan,
                "n_genes": 0,
                "retention": {},
            }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = float(spearmanr(pred.values, true.values)[0])

        rmse_r = float(np.sqrt(np.mean((pred.values - true.values) ** 2)))
        pred_z = np.vectorize(_fisher_z)(pred.values)
        true_z = np.vectorize(_fisher_z)(true.values)
        rmse_z = float(np.sqrt(np.mean((pred_z - true_z) ** 2)))

        ret = retention_curve(pred, true)

        return {
            "spearman_rho": rho,
            "rmse_r": rmse_r,
            "rmse_z": rmse_z,
            "n_genes": len(pred),
            "retention": ret,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ReliabilityModel (ridge α={self.ridge_alpha})",
            f"  Trained on {self.n_train_rows} (gene, fold) rows "
            f"({self.n_train_genes} unique genes), R²={self.train_r2:.4f}",
            f"  Features ({len(self.feature_columns)}): {self.feature_columns}",
        ]
        if self.weights is not None:
            for col, w in zip(self.feature_columns, self.weights):
                lines.append(f"    {col:30s}  w={w:+.5f}")
            lines.append(f"    {'intercept':30s}  b={self.intercept:+.5f}")
        return "\n".join(lines)


# ── Top-K Retention Curve ────────────────────────────────────────────────


def retention_curve(
    predicted_r: pd.Series,
    true_r: pd.Series,
    *,
    fractions: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50,
                                   0.60, 0.70, 0.80, 0.90, 1.00),
) -> Dict[str, Any]:
    """Top-K enrichment / retention curve.

    Sort genes by predicted reliability (descending), then for each
    retention fraction, compute the actual mean Pearson r in the
    retained set.

    This is the primary diagnostic for the reliability model: if it
    works, retaining only the top-predicted genes should yield
    substantially higher actual r than retaining all.

    Parameters
    ----------
    predicted_r : predicted gene-level reliability, indexed by gene.
    true_r : actual gene-level Pearson r from holdout, indexed by gene.
    fractions : retention fractions (0.10 = top 10%).

    Returns
    -------
    Dict with:
        ``fractions``  — list of retention fractions
        ``mean_true_r`` — list of mean true r at each retention fraction
        ``n_genes``    — list of gene counts at each retention fraction
    """
    common = predicted_r.index.intersection(true_r.index)
    pred = predicted_r.loc[common].dropna()
    true = true_r.loc[common]
    mask = pred.notna() & true.notna()
    pred, true = pred[mask], true[mask]

    # Sort by predicted reliability (descending)
    sort_idx = pred.sort_values(ascending=False).index
    true_sorted = true.loc[sort_idx].values
    n = len(true_sorted)

    if n < 5:
        return {
            "fractions": list(fractions),
            "mean_true_r": [np.nan] * len(fractions),
            "n_genes": [0] * len(fractions),
        }

    means = []
    counts = []
    for f in fractions:
        k = max(1, int(np.ceil(n * f)))
        top_k = true_sorted[:k]
        means.append(float(np.nanmean(top_k)))
        counts.append(k)

    return {
        "fractions": list(fractions),
        "mean_true_r": means,
        "n_genes": counts,
    }
