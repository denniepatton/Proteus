"""Patient-OOF distribution calibration helpers for the analysis pipeline."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.analysis_utils import (
	EXPRESSION_CEIL,
	EXPRESSION_FLOOR,
	VARIANCE_BINS,
	compute_gene_variance,
	ensure_dir,
	mdn_on_probability_wide,
)
from analysis.pipeline_config import VALIDATION_PATIENT_CV_COHORTS
from analysis.plot_utils import plot_prediction_calibration_curve

__all__ = (
	"CALIBRATION_DRAW_SEED",
	"apply_distribution_calibration_bundle",
	"bundle_sample_fold_subset",
	"constant_source_by_sample",
	"get_calibrated_activity_surfaces",
	"get_calibrated_draws",
	"get_patient_distribution_calibration_state",
	"write_distribution_calibration_outputs",
)


_MDN_PARAM_KEYS = (
	"p0_wide",
	"pi_gamma_wide",
	"pi_normal_wide",
	"gamma_beta_wide",
	"normal_mean_wide",
	"normal_stdev_wide",
)
_CALIBRATION_ACTIVITY_THRESHOLD = 2.0
_CALIBRATION_GAMMA_SHAPE = 0.75
_CALIBRATION_INTERVAL_ALPHA = 0.10
_CALIBRATION_TAIL_ALPHA = _CALIBRATION_INTERVAL_ALPHA * 0.5
_CALIBRATION_AOR_EPSILON = 1e-6
_CALIBRATION_MAP_GRID = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
_CALIBRATION_INTEGRATION_PROBS = np.linspace(
	0.5 / 17.0,
	1.0 - 0.5 / 17.0,
	17,
	dtype=np.float64,
)
_CALIBRATION_POOLING_PATIENTS = 6.0
_CALIBRATION_RANDOM_SEED = 0
CALIBRATION_DRAW_SEED = 42
_PATIENT_DISTRIBUTION_CALIBRATION_CACHE: Dict[Tuple[float], Dict[str, Any]] = {}


def _calibration_cache_key(minimum_tfx: float) -> Tuple[float]:
	return (round(float(minimum_tfx), 6),)


def _variance_bin_label(variance_value: float) -> str:
	if np.isfinite(float(variance_value)):
		for label, (lower_bound, upper_bound) in VARIANCE_BINS.items():
			if float(variance_value) >= float(lower_bound) and float(variance_value) < float(upper_bound):
				return str(label)
	return "unknown"


def _sample_equal_weights(sample_labels: Sequence[Any]) -> np.ndarray:
	sample_index = pd.Index([str(label) for label in sample_labels])
	if sample_index.empty:
		return np.asarray([], dtype=np.float64)
	counts = sample_index.value_counts()
	weights = np.asarray([1.0 / float(counts[str(label)]) for label in sample_index], dtype=np.float64)
	total_weight = float(weights.sum())
	if total_weight > 0.0:
		weights = weights / total_weight
	return weights


def _weighted_ecdf_on_grid(
	values: np.ndarray,
	weights: np.ndarray,
	grid: np.ndarray,
) -> np.ndarray:
	sorted_values = np.asarray(values, dtype=np.float64)
	sorted_weights = np.asarray(weights, dtype=np.float64)
	valid = np.isfinite(sorted_values) & np.isfinite(sorted_weights) & (sorted_weights > 0.0)
	if not bool(valid.any()):
		return grid.copy()
	order = np.argsort(sorted_values[valid], kind="mergesort")
	x = np.clip(sorted_values[valid][order], 0.0, 1.0)
	w = sorted_weights[valid][order]
	cumulative = np.cumsum(w)
	total = float(cumulative[-1])
	if total <= 0.0:
		return grid.copy()
	cumulative = cumulative / total
	counts = np.searchsorted(x, grid, side="right")
	result = np.zeros_like(grid, dtype=np.float64)
	nonzero = counts > 0
	result[nonzero] = cumulative[counts[nonzero] - 1]
	result[0] = 0.0
	result[-1] = 1.0
	return np.maximum.accumulate(np.clip(result, 0.0, 1.0))


def _weighted_quantile(
	values: np.ndarray,
	weights: np.ndarray,
	probability: float,
) -> float:
	values_arr = np.asarray(values, dtype=np.float64)
	weights_arr = np.asarray(weights, dtype=np.float64)
	valid = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0.0)
	if not bool(valid.any()):
		return float("nan")
	order = np.argsort(values_arr[valid], kind="mergesort")
	x = values_arr[valid][order]
	w = weights_arr[valid][order]
	cumulative = np.cumsum(w)
	total = float(cumulative[-1])
	if total <= 0.0:
		return float("nan")
	threshold = float(np.clip(probability, 0.0, 1.0)) * total
	index = int(np.searchsorted(cumulative, threshold, side="left"))
	index = min(index, len(x) - 1)
	return float(x[index])


def _eval_calibration_map(
	probabilities: np.ndarray,
	map_y: np.ndarray,
	*,
	grid: np.ndarray = _CALIBRATION_MAP_GRID,
) -> np.ndarray:
	probs = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
	mapped = np.interp(probs, grid, np.asarray(map_y, dtype=np.float64), left=0.0, right=1.0)
	return np.clip(mapped, 0.0, 1.0)


def _invert_calibration_map(
	probabilities: np.ndarray,
	map_y: np.ndarray,
	*,
	grid: np.ndarray = _CALIBRATION_MAP_GRID,
) -> np.ndarray:
	probs = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
	x_grid = np.asarray(grid, dtype=np.float64)
	y_grid = np.maximum.accumulate(np.asarray(map_y, dtype=np.float64))
	unique_y, first_indices = np.unique(y_grid, return_index=True)
	unique_x = x_grid[first_indices]
	if unique_y.size == 0:
		return probs.copy()
	if unique_y[0] > 0.0:
		unique_y = np.insert(unique_y, 0, 0.0)
		unique_x = np.insert(unique_x, 0, 0.0)
	if unique_y[-1] < 1.0:
		unique_y = np.append(unique_y, 1.0)
		unique_x = np.append(unique_x, 1.0)
	inverted = np.asarray(
		np.interp(probs, unique_y, unique_x, left=unique_x[0], right=unique_x[-1]),
		dtype=np.float64,
	)
	return np.clip(inverted, 0.0, 1.0)


def _normalize_mdn_probabilities(
	p0_values: np.ndarray,
	pi_gamma_values: np.ndarray,
	pi_normal_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	p0_arr = np.asarray(p0_values, dtype=np.float64)
	pg_arr = np.asarray(pi_gamma_values, dtype=np.float64)
	pn_arr = np.asarray(pi_normal_values, dtype=np.float64)
	prob_sum = p0_arr + pg_arr + pn_arr
	safe_sum = np.where(np.isfinite(prob_sum) & (prob_sum > 0.0), prob_sum, np.nan)
	return (
		p0_arr / safe_sum,
		pg_arr / safe_sum,
		pn_arr / safe_sum,
		safe_sum,
	)


def _mdn_cdf_array(
	values: np.ndarray,
	p0_values: np.ndarray,
	pi_gamma_values: np.ndarray,
	pi_normal_values: np.ndarray,
	gamma_beta_values: np.ndarray,
	normal_mean_values: np.ndarray,
	normal_stdev_values: np.ndarray,
) -> np.ndarray:
	from scipy.special import erf, gammainc

	p0_arr, pg_arr, pn_arr, safe_sum = _normalize_mdn_probabilities(
		p0_values,
		pi_gamma_values,
		pi_normal_values,
	)
	y = np.asarray(values, dtype=np.float64)
	beta = np.clip(np.asarray(gamma_beta_values, dtype=np.float64), 1e-12, None)
	mu = np.asarray(normal_mean_values, dtype=np.float64)
	sigma = np.clip(np.asarray(normal_stdev_values, dtype=np.float64), 1e-12, None)

	gamma_cdf = gammainc(_CALIBRATION_GAMMA_SHAPE, beta * np.maximum(y, 0.0))
	normal_cdf = 0.5 * (1.0 + erf((y - mu) / (sigma * np.sqrt(2.0))))
	cdf_zero = (y >= 0.0).astype(np.float64)
	cdf = p0_arr * cdf_zero + pg_arr * gamma_cdf + pn_arr * normal_cdf
	invalid = ~np.isfinite(safe_sum)
	cdf = np.where(invalid, np.nan, cdf)
	return np.clip(cdf, 0.0, 1.0)


def _mdn_randomized_pit_flat(
	truth_values: np.ndarray,
	p0_values: np.ndarray,
	pi_gamma_values: np.ndarray,
	pi_normal_values: np.ndarray,
	gamma_beta_values: np.ndarray,
	normal_mean_values: np.ndarray,
	normal_stdev_values: np.ndarray,
	*,
	random_seed: int,
) -> np.ndarray:
	from scipy.special import erf, gammainc

	truth_arr = np.asarray(truth_values, dtype=np.float64)
	p0_arr, pg_arr, pn_arr, safe_sum = _normalize_mdn_probabilities(
		p0_values,
		pi_gamma_values,
		pi_normal_values,
	)
	beta = np.clip(np.asarray(gamma_beta_values, dtype=np.float64), 1e-12, None)
	mu = np.asarray(normal_mean_values, dtype=np.float64)
	sigma = np.clip(np.asarray(normal_stdev_values, dtype=np.float64), 1e-12, None)

	valid = (
		np.isfinite(truth_arr)
		& np.isfinite(p0_arr)
		& np.isfinite(pg_arr)
		& np.isfinite(pn_arr)
		& np.isfinite(beta)
		& np.isfinite(mu)
		& np.isfinite(sigma)
		& np.isfinite(safe_sum)
	)
	pit = np.full(truth_arr.shape, np.nan, dtype=np.float64)
	if not bool(valid.any()):
		return pit

	truth_valid = truth_arr[valid]
	gamma_cdf = gammainc(_CALIBRATION_GAMMA_SHAPE, beta[valid] * np.maximum(truth_valid, 0.0))
	normal_cdf = 0.5 * (1.0 + erf((truth_valid - mu[valid]) / (sigma[valid] * np.sqrt(2.0))))
	cdf_right = (
		p0_arr[valid] * (truth_valid >= 0.0).astype(np.float64)
		+ pg_arr[valid] * gamma_cdf
		+ pn_arr[valid] * normal_cdf
	)
	pit_valid = np.clip(cdf_right, 0.0, 1.0)
	zero_mask = np.isclose(truth_valid, 0.0, atol=1e-12)
	if bool(zero_mask.any()):
		cdf_left = pn_arr[valid][zero_mask] * normal_cdf[zero_mask]
		rng = np.random.default_rng(random_seed)
		random_u = rng.uniform(size=int(zero_mask.sum()))
		pit_valid[zero_mask] = cdf_left + random_u * (cdf_right[zero_mask] - cdf_left)
	pit[valid] = np.clip(pit_valid, 0.0, 1.0)
	return pit


def _mdn_randomized_pit_wide(
	truth_wide: pd.DataFrame,
	p0_wide: pd.DataFrame,
	pi_gamma_wide: pd.DataFrame,
	pi_normal_wide: pd.DataFrame,
	gamma_beta_wide: pd.DataFrame,
	normal_mean_wide: pd.DataFrame,
	normal_stdev_wide: pd.DataFrame,
	*,
	random_seed: int,
) -> pd.DataFrame:
	truth_arr = np.asarray(truth_wide.to_numpy(dtype=np.float64), dtype=np.float64)
	pit_flat = _mdn_randomized_pit_flat(
		truth_arr.reshape(-1),
		p0_wide.to_numpy(dtype=np.float64).reshape(-1),
		pi_gamma_wide.to_numpy(dtype=np.float64).reshape(-1),
		pi_normal_wide.to_numpy(dtype=np.float64).reshape(-1),
		gamma_beta_wide.to_numpy(dtype=np.float64).reshape(-1),
		normal_mean_wide.to_numpy(dtype=np.float64).reshape(-1),
		normal_stdev_wide.to_numpy(dtype=np.float64).reshape(-1),
		random_seed=random_seed,
	)
	pit_arr = pit_flat.reshape(truth_arr.shape)
	return pd.DataFrame(pit_arr, index=truth_wide.index, columns=truth_wide.columns)


def _mdn_quantile_matrix(
	probabilities: np.ndarray,
	p0_values: np.ndarray,
	pi_gamma_values: np.ndarray,
	pi_normal_values: np.ndarray,
	gamma_beta_values: np.ndarray,
	normal_mean_values: np.ndarray,
	normal_stdev_values: np.ndarray,
	*,
	n_steps: int = 50,
) -> np.ndarray:
	probs = np.asarray(probabilities, dtype=np.float64)
	p0_arr = np.asarray(p0_values, dtype=np.float64).reshape(-1)
	pg_arr = np.asarray(pi_gamma_values, dtype=np.float64).reshape(-1)
	pn_arr = np.asarray(pi_normal_values, dtype=np.float64).reshape(-1)
	beta_arr = np.clip(np.asarray(gamma_beta_values, dtype=np.float64).reshape(-1), 1e-12, None)
	mu_arr = np.asarray(normal_mean_values, dtype=np.float64).reshape(-1)
	sigma_arr = np.clip(np.asarray(normal_stdev_values, dtype=np.float64).reshape(-1), 1e-12, None)

	if p0_arr.size == 0:
		return np.empty((0, 0), dtype=np.float64)

	if probs.ndim == 1:
		probs = np.broadcast_to(probs.reshape(1, -1), (p0_arr.size, probs.size))
	elif probs.ndim == 2 and probs.shape[0] != p0_arr.size:
		raise ValueError("probabilities must be shape (n_points,) or (n_points, n_quantiles)")

	p0_norm, pg_norm, pn_norm, safe_sum = _normalize_mdn_probabilities(p0_arr, pg_arr, pn_arr)
	hi_init = np.maximum(
		(_CALIBRATION_GAMMA_SHAPE + 4.0 * np.sqrt(_CALIBRATION_GAMMA_SHAPE)) / beta_arr,
		mu_arr + 5.0 * sigma_arr,
	)
	hi_init = np.clip(hi_init, 1.0, EXPRESSION_CEIL)

	lo = np.full(probs.shape, EXPRESSION_FLOOR, dtype=np.float64)
	hi = np.broadcast_to(hi_init.reshape(-1, 1), probs.shape).copy()

	invalid_row = ~np.isfinite(safe_sum)
	if bool(invalid_row.any()):
		lo[invalid_row] = np.nan
		hi[invalid_row] = np.nan

	for _ in range(n_steps):
		mid = 0.5 * (lo + hi)
		cdf_mid = _mdn_cdf_array(
			mid,
			p0_norm.reshape(-1, 1),
			pg_norm.reshape(-1, 1),
			pn_norm.reshape(-1, 1),
			beta_arr.reshape(-1, 1),
			mu_arr.reshape(-1, 1),
			sigma_arr.reshape(-1, 1),
		)
		lo = np.where(cdf_mid < probs, mid, lo)
		hi = np.where(cdf_mid >= probs, mid, hi)

	quantiles = 0.5 * (lo + hi)
	return np.clip(quantiles, EXPRESSION_FLOOR, EXPRESSION_CEIL)


def _conformal_quantile_level(n_patients: int, alpha: float) -> float:
	if n_patients <= 0:
		return 1.0
	return float(min(1.0, np.ceil((n_patients + 1) * (1.0 - alpha)) / n_patients))


def _calibration_stratum_key(source: str, activity_state: str, variance_bin: str) -> str:
	return f"{source}__{activity_state}__{variance_bin}"


def _aligned_mdn_bundle_matrices(
	bundle: Dict[str, Any],
	*,
	sample_names: Optional[Sequence[Any]] = None,
	require_truth: bool = False,
) -> Optional[Dict[str, pd.DataFrame]]:
	if bundle.get("pred_wide") is None or not all(bundle.get(key) is not None for key in _MDN_PARAM_KEYS):
		return None

	genes = pd.Index(bundle["pred_wide"].index)
	samples = pd.Index(bundle["pred_wide"].columns)
	for key in _MDN_PARAM_KEYS:
		genes = genes.intersection(bundle[key].index)
		samples = samples.intersection(bundle[key].columns)
	if require_truth:
		truth_wide = bundle.get("truth_wide")
		if truth_wide is None:
			raise ValueError("truth_wide is required for calibration diagnostics")
		genes = genes.intersection(truth_wide.index)
		samples = samples.intersection(truth_wide.columns)
	if sample_names is not None:
		allowed = {str(sample_name) for sample_name in sample_names}
		samples = pd.Index([sample_name for sample_name in samples if str(sample_name) in allowed])
	if len(genes) == 0 or len(samples) == 0:
		return None

	aligned: Dict[str, pd.DataFrame] = {
		"pred_wide": bundle["pred_wide"].loc[genes, samples].astype(float),
	}
	if require_truth:
		aligned["truth_wide"] = bundle["truth_wide"].loc[genes, samples].astype(float)
	for key in ("ci_lo_wide", "ci_hi_wide"):
		table = bundle.get(key)
		if table is None:
			continue
		ci_genes = genes.intersection(table.index)
		ci_samples = samples.intersection(table.columns)
		if len(ci_genes) == len(genes) and len(ci_samples) == len(samples):
			aligned[key] = table.loc[genes, samples].astype(float)
	for key in _MDN_PARAM_KEYS:
		aligned[key] = bundle[key].loc[genes, samples].astype(float)

	active_conf = bundle.get("active_confidence_wide")
	if active_conf is not None:
		active_genes = genes.intersection(active_conf.index)
		active_samples = samples.intersection(active_conf.columns)
		if len(active_genes) == len(genes) and len(active_samples) == len(samples):
			aligned["active_confidence_wide"] = active_conf.loc[genes, samples].astype(float)
	return aligned


def _extract_patient_distribution_calibration_bundle(
	cv_full: Dict[str, Any],
	cohort_map: Dict[str, str],
) -> Dict[str, Any]:
	keep_samples = [
		sample_name
		for sample_name in cv_full["pred_wide"].columns
		if cohort_map.get(str(sample_name)) in VALIDATION_PATIENT_CV_COHORTS
	]
	aligned = _aligned_mdn_bundle_matrices(
		cv_full,
		sample_names=keep_samples,
		require_truth=True,
	)
	if aligned is None:
		raise ValueError("Patient OOF MDN surfaces are required for distribution calibration")

	if "active_confidence_wide" not in aligned:
		aligned["active_confidence_wide"] = mdn_on_probability_wide(
			aligned["p0_wide"],
			aligned["pi_gamma_wide"],
			aligned["pi_normal_wide"],
			aligned["gamma_beta_wide"],
			aligned["normal_mean_wide"],
			aligned["normal_stdev_wide"],
			threshold=_CALIBRATION_ACTIVITY_THRESHOLD,
		)

	gene_variance = compute_gene_variance(aligned["truth_wide"])
	gene_variance_bin = {
		str(gene_name): _variance_bin_label(gene_variance.get(gene_name, np.nan))
		for gene_name in aligned["truth_wide"].index
	}

	sample_fold = {
		str(sample_name): int(cv_full.get("sample_fold", {}).get(str(sample_name)))
		for sample_name in aligned["truth_wide"].columns
		if str(sample_name) in cv_full.get("sample_fold", {})
	}
	source_by_sample = {
		str(sample_name): str(cohort_map.get(str(sample_name), "pooled"))
		for sample_name in aligned["truth_wide"].columns
	}

	return {
		**aligned,
		"gene_variance_bin": gene_variance_bin,
		"sample_fold": sample_fold,
		"source_by_sample": source_by_sample,
	}


def _fit_patient_distribution_calibrator(
	patient_bundle: Dict[str, Any],
	*,
	holdout_folds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
	held_out = {int(fold_id) for fold_id in holdout_folds} if holdout_folds is not None else set()
	fit_samples = [
		sample_name
		for sample_name in patient_bundle["truth_wide"].columns
		if str(sample_name) not in patient_bundle["sample_fold"]
		or int(patient_bundle["sample_fold"][str(sample_name)]) not in held_out
	]
	if not fit_samples:
		raise ValueError("No patient OOF samples remain after excluding calibration holdout folds")

	truth_wide = patient_bundle["truth_wide"].loc[:, fit_samples]
	pred_wide = patient_bundle["pred_wide"].loc[:, fit_samples]
	active_conf_wide = patient_bundle["active_confidence_wide"].loc[:, fit_samples]
	p0_wide = patient_bundle["p0_wide"].loc[:, fit_samples]
	pi_gamma_wide = patient_bundle["pi_gamma_wide"].loc[:, fit_samples]
	pi_normal_wide = patient_bundle["pi_normal_wide"].loc[:, fit_samples]
	gamma_beta_wide = patient_bundle["gamma_beta_wide"].loc[:, fit_samples]
	normal_mean_wide = patient_bundle["normal_mean_wide"].loc[:, fit_samples]
	normal_stdev_wide = patient_bundle["normal_stdev_wide"].loc[:, fit_samples]

	truth_arr = truth_wide.to_numpy(dtype=np.float64)
	pred_arr = pred_wide.to_numpy(dtype=np.float64)
	active_conf_arr = active_conf_wide.to_numpy(dtype=np.float64)
	p0_arr = p0_wide.to_numpy(dtype=np.float64)
	pg_arr = pi_gamma_wide.to_numpy(dtype=np.float64)
	pn_arr = pi_normal_wide.to_numpy(dtype=np.float64)
	beta_arr = gamma_beta_wide.to_numpy(dtype=np.float64)
	mu_arr = normal_mean_wide.to_numpy(dtype=np.float64)
	sigma_arr = normal_stdev_wide.to_numpy(dtype=np.float64)

	pit_raw_flat = _mdn_randomized_pit_flat(
		truth_arr.reshape(-1),
		p0_arr.reshape(-1),
		pg_arr.reshape(-1),
		pn_arr.reshape(-1),
		beta_arr.reshape(-1),
		mu_arr.reshape(-1),
		sigma_arr.reshape(-1),
		random_seed=_CALIBRATION_RANDOM_SEED,
	)

	n_genes, n_samples = truth_arr.shape
	sample_names = np.asarray([str(sample_name) for sample_name in fit_samples], dtype=object)
	variance_bins = np.asarray(
		[patient_bundle["gene_variance_bin"].get(str(gene_name), "unknown") for gene_name in truth_wide.index],
		dtype=object,
	)
	activity_state = np.where(
		np.isfinite(active_conf_arr),
		active_conf_arr >= 0.5,
		pred_arr >= _CALIBRATION_ACTIVITY_THRESHOLD,
	)

	valid = (
		np.isfinite(truth_arr)
		& np.isfinite(pred_arr)
		& np.isfinite(p0_arr)
		& np.isfinite(pg_arr)
		& np.isfinite(pn_arr)
		& np.isfinite(beta_arr)
		& np.isfinite(mu_arr)
		& np.isfinite(sigma_arr)
		& np.isfinite(pit_raw_flat.reshape(n_genes, n_samples))
	)
	valid_flat = valid.reshape(-1)
	if not bool(valid_flat.any()):
		raise ValueError("No finite patient OOF gene-sample pairs were available for distribution calibration")

	truth_flat = truth_arr.reshape(-1)[valid_flat]
	p0_flat = p0_arr.reshape(-1)[valid_flat]
	pg_flat = pg_arr.reshape(-1)[valid_flat]
	pn_flat = pn_arr.reshape(-1)[valid_flat]
	beta_flat = beta_arr.reshape(-1)[valid_flat]
	mu_flat = mu_arr.reshape(-1)[valid_flat]
	sigma_flat = sigma_arr.reshape(-1)[valid_flat]
	pit_raw = pit_raw_flat[valid_flat]
	sample_flat = np.tile(sample_names, n_genes)[valid_flat]
	source_flat = np.tile(
		np.asarray([patient_bundle["source_by_sample"].get(str(sample_name), "pooled") for sample_name in fit_samples], dtype=object),
		n_genes,
	)[valid_flat]
	variance_flat = np.repeat(variance_bins, n_samples)[valid_flat]
	activity_flat = np.where(activity_state.reshape(-1)[valid_flat], "active", "inactive")

	pooled_weights = _sample_equal_weights(sample_flat)
	pooled_map_y = _weighted_ecdf_on_grid(pit_raw, pooled_weights, _CALIBRATION_MAP_GRID)
	pooled_probs = _invert_calibration_map(
		np.asarray([_CALIBRATION_TAIL_ALPHA, 1.0 - _CALIBRATION_TAIL_ALPHA], dtype=np.float64),
		pooled_map_y,
	)
	pooled_quantiles = _mdn_quantile_matrix(
		pooled_probs,
		p0_flat,
		pg_flat,
		pn_flat,
		beta_flat,
		mu_flat,
		sigma_flat,
	)
	pooled_patient_count = len(pd.Index(sample_flat).unique())
	pooled_qhat_low = _weighted_quantile(
		np.maximum(pooled_quantiles[:, 0] - truth_flat, 0.0),
		pooled_weights,
		_conformal_quantile_level(pooled_patient_count, _CALIBRATION_TAIL_ALPHA),
	)
	pooled_qhat_high = _weighted_quantile(
		np.maximum(truth_flat - pooled_quantiles[:, 1], 0.0),
		pooled_weights,
		_conformal_quantile_level(pooled_patient_count, _CALIBRATION_TAIL_ALPHA),
	)

	pooled_entry = {
		"label": "pooled",
		"n_pairs": int(len(truth_flat)),
		"n_patients": int(pooled_patient_count),
		"shrinkage_weight": 0.0,
		"map_y": pooled_map_y,
		"qhat_low": float(pooled_qhat_low),
		"qhat_high": float(pooled_qhat_high),
		"source": "pooled",
		"activity_state": "pooled",
		"variance_bin": "pooled",
	}

	strata: Dict[str, Dict[str, Any]] = {}
	summary_rows: List[Dict[str, Any]] = []
	variance_levels = list(VARIANCE_BINS.keys()) + ["unknown"]
	for source_name in sorted({str(source_name) for source_name in source_flat}):
		for activity_name in ("inactive", "active"):
			for variance_name in variance_levels:
				group_mask = (
					(source_flat == source_name)
					& (activity_flat == activity_name)
					& (variance_flat == variance_name)
				)
				if not bool(group_mask.any()):
					continue

				group_samples = sample_flat[group_mask]
				group_weights = _sample_equal_weights(group_samples)
				group_patient_count = len(pd.Index(group_samples).unique())
				shrinkage_weight = float(
					group_patient_count / (group_patient_count + _CALIBRATION_POOLING_PATIENTS)
				)
				group_map_raw = _weighted_ecdf_on_grid(
					pit_raw[group_mask],
					group_weights,
					_CALIBRATION_MAP_GRID,
				)
				shrunk_map = (
					shrinkage_weight * group_map_raw
					+ (1.0 - shrinkage_weight) * pooled_map_y
				)
				group_probs = _invert_calibration_map(
					np.asarray([_CALIBRATION_TAIL_ALPHA, 1.0 - _CALIBRATION_TAIL_ALPHA], dtype=np.float64),
					shrunk_map,
				)
				group_quantiles = _mdn_quantile_matrix(
					group_probs,
					p0_flat[group_mask],
					pg_flat[group_mask],
					pn_flat[group_mask],
					beta_flat[group_mask],
					mu_flat[group_mask],
					sigma_flat[group_mask],
				)
				group_qhat_low_raw = _weighted_quantile(
					np.maximum(group_quantiles[:, 0] - truth_flat[group_mask], 0.0),
					group_weights,
					_conformal_quantile_level(group_patient_count, _CALIBRATION_TAIL_ALPHA),
				)
				group_qhat_high_raw = _weighted_quantile(
					np.maximum(truth_flat[group_mask] - group_quantiles[:, 1], 0.0),
					group_weights,
					_conformal_quantile_level(group_patient_count, _CALIBRATION_TAIL_ALPHA),
				)
				entry_key = _calibration_stratum_key(source_name, activity_name, variance_name)
				strata[entry_key] = {
					"label": entry_key,
					"n_pairs": int(group_mask.sum()),
					"n_patients": int(group_patient_count),
					"shrinkage_weight": shrinkage_weight,
					"map_y": np.maximum.accumulate(np.clip(shrunk_map, 0.0, 1.0)),
					"qhat_low": float(
						shrinkage_weight * group_qhat_low_raw
						+ (1.0 - shrinkage_weight) * pooled_qhat_low
					),
					"qhat_high": float(
						shrinkage_weight * group_qhat_high_raw
						+ (1.0 - shrinkage_weight) * pooled_qhat_high
					),
					"source": source_name,
					"activity_state": activity_name,
					"variance_bin": variance_name,
				}
				summary_rows.append(
					{
						"stratum": entry_key,
						"source": source_name,
						"activity_state": activity_name,
						"variance_bin": variance_name,
						"n_pairs": int(group_mask.sum()),
						"n_patients": int(group_patient_count),
						"shrinkage_weight": float(shrinkage_weight),
						"qhat_low": float(strata[entry_key]["qhat_low"]),
						"qhat_high": float(strata[entry_key]["qhat_high"]),
					}
				)

	return {
		"pooled": pooled_entry,
		"strata": strata,
		"summary_rows": summary_rows,
		"tail_alpha": float(_CALIBRATION_TAIL_ALPHA),
		"activity_threshold": float(_CALIBRATION_ACTIVITY_THRESHOLD),
		"grid": _CALIBRATION_MAP_GRID.copy(),
	}


def get_patient_distribution_calibration_state(
	minimum_tfx: float,
	*,
	cv_full: Optional[Dict[str, Any]] = None,
	cohort_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
	"""Return cached patient-OOF calibration state, building it on cache miss."""
	cache_key = _calibration_cache_key(minimum_tfx)
	cached = _PATIENT_DISTRIBUTION_CALIBRATION_CACHE.get(cache_key)
	if cached is not None:
		return cached
	if cv_full is None or cohort_map is None:
		raise ValueError("cv_full and cohort_map are required to build patient distribution calibration state")

	patient_bundle = _extract_patient_distribution_calibration_bundle(cv_full, cohort_map)
	fold_ids = sorted({int(fold_id) for fold_id in patient_bundle["sample_fold"].values()})
	state = {
		"patient_bundle": patient_bundle,
		"gene_variance_bin": patient_bundle["gene_variance_bin"],
		"full_calibrator": _fit_patient_distribution_calibrator(patient_bundle),
		"fold_calibrators": {
			int(fold_id): _fit_patient_distribution_calibrator(
				patient_bundle,
				holdout_folds=(int(fold_id),),
			)
			for fold_id in fold_ids
		},
	}
	_PATIENT_DISTRIBUTION_CALIBRATION_CACHE[cache_key] = state
	return state


def constant_source_by_sample(
	sample_names: Sequence[Any],
	source_name: str,
) -> Dict[str, str]:
	return {str(sample_name): str(source_name) for sample_name in sample_names}


def bundle_sample_fold_subset(
	bundle: Dict[str, Any],
	sample_names: Sequence[Any],
) -> Dict[str, int]:
	sample_set = {str(sample_name) for sample_name in sample_names}
	return {
		str(sample_name): int(fold_id)
		for sample_name, fold_id in bundle.get("sample_fold", {}).items()
		if str(sample_name) in sample_set
	}


def _select_distribution_calibration_entry(
	calibrator: Dict[str, Any],
	*,
	source: str,
	activity_state: str,
	variance_bin: str,
) -> Dict[str, Any]:
	entry_key = _calibration_stratum_key(source, activity_state, variance_bin)
	return calibrator["strata"].get(entry_key, calibrator["pooled"])


def _sample_specific_distribution_calibrator(
	calibration_state: Dict[str, Any],
	*,
	sample_name: str,
	sample_fold: Optional[Dict[str, int]],
) -> Dict[str, Any]:
	if sample_fold is not None and str(sample_name) in sample_fold:
		fold_id = int(sample_fold[str(sample_name)])
		if fold_id in calibration_state["fold_calibrators"]:
			return calibration_state["fold_calibrators"][fold_id]
	return calibration_state["full_calibrator"]


def _iter_distribution_calibration_groups(
	genes: pd.Index,
	samples: pd.Index,
	pred_arr: np.ndarray,
	active_conf_arr: Optional[np.ndarray],
	calibration_state: Dict[str, Any],
	*,
	source_by_sample: Dict[str, str],
	sample_fold: Optional[Dict[str, int]],
) -> List[Tuple[int, np.ndarray, Dict[str, Any]]]:
	variance_bins = np.asarray(
		[calibration_state["gene_variance_bin"].get(str(gene_name), "unknown") for gene_name in genes],
		dtype=object,
	)
	variance_levels = list(dict.fromkeys(list(VARIANCE_BINS.keys()) + ["unknown"]))
	groups: List[Tuple[int, np.ndarray, Dict[str, Any]]] = []
	for sample_idx, sample_name in enumerate(samples):
		source_name = str(source_by_sample.get(str(sample_name), "pooled"))
		calibrator = _sample_specific_distribution_calibrator(
			calibration_state,
			sample_name=str(sample_name),
			sample_fold=sample_fold,
		)
		if active_conf_arr is not None:
			activity_mask = np.where(
				np.isfinite(active_conf_arr[:, sample_idx]),
				active_conf_arr[:, sample_idx] >= 0.5,
				pred_arr[:, sample_idx] >= _CALIBRATION_ACTIVITY_THRESHOLD,
			)
		else:
			activity_mask = pred_arr[:, sample_idx] >= _CALIBRATION_ACTIVITY_THRESHOLD
		for variance_name in variance_levels:
			variance_idx = np.flatnonzero(variance_bins == variance_name)
			if variance_idx.size == 0:
				continue
			variance_activity = activity_mask[variance_idx]
			for activity_name, state_mask in (
				("inactive", ~variance_activity),
				("active", variance_activity),
			):
				gene_indices = variance_idx[state_mask]
				if gene_indices.size == 0:
					continue
				entry = _select_distribution_calibration_entry(
					calibrator,
					source=source_name,
					activity_state=activity_name,
					variance_bin=variance_name,
				)
				groups.append((sample_idx, gene_indices, entry))
	return groups


def _compute_distribution_calibration_surfaces(
	bundle: Dict[str, Any],
	calibration_state: Dict[str, Any],
	*,
	source_by_sample: Dict[str, str],
	sample_fold: Optional[Dict[str, int]],
) -> Dict[str, pd.DataFrame]:
	aligned = _aligned_mdn_bundle_matrices(bundle, require_truth=False)
	if aligned is None:
		raise ValueError("Distribution calibration requires full Proteus MDN surfaces")

	active_conf_wide = aligned.get("active_confidence_wide")
	if active_conf_wide is None:
		active_conf_wide = mdn_on_probability_wide(
			aligned["p0_wide"],
			aligned["pi_gamma_wide"],
			aligned["pi_normal_wide"],
			aligned["gamma_beta_wide"],
			aligned["normal_mean_wide"],
			aligned["normal_stdev_wide"],
			threshold=_CALIBRATION_ACTIVITY_THRESHOLD,
		)

	genes = aligned["pred_wide"].index
	samples = aligned["pred_wide"].columns
	pred_arr = aligned["pred_wide"].to_numpy(dtype=np.float64)
	active_conf_arr = active_conf_wide.loc[genes, samples].to_numpy(dtype=np.float64)
	p0_arr = aligned["p0_wide"].to_numpy(dtype=np.float64)
	pg_arr = aligned["pi_gamma_wide"].to_numpy(dtype=np.float64)
	pn_arr = aligned["pi_normal_wide"].to_numpy(dtype=np.float64)
	beta_arr = aligned["gamma_beta_wide"].to_numpy(dtype=np.float64)
	mu_arr = aligned["normal_mean_wide"].to_numpy(dtype=np.float64)
	sigma_arr = aligned["normal_stdev_wide"].to_numpy(dtype=np.float64)

	pred_cal = np.full(pred_arr.shape, np.nan, dtype=np.float64)
	variance_cal = np.full(pred_arr.shape, np.nan, dtype=np.float64)
	ci_lo_cal = np.full(pred_arr.shape, np.nan, dtype=np.float64)
	ci_hi_cal = np.full(pred_arr.shape, np.nan, dtype=np.float64)

	target_probs = np.concatenate(
		[
			_CALIBRATION_INTEGRATION_PROBS,
			np.asarray([_CALIBRATION_TAIL_ALPHA, 1.0 - _CALIBRATION_TAIL_ALPHA], dtype=np.float64),
		]
	)
	n_moment_probs = len(_CALIBRATION_INTEGRATION_PROBS)
	for sample_idx, gene_indices, entry in _iter_distribution_calibration_groups(
		genes,
		samples,
		pred_arr,
		active_conf_arr,
		calibration_state,
		source_by_sample=source_by_sample,
		sample_fold=sample_fold,
	):
		raw_probs = _invert_calibration_map(target_probs, entry["map_y"])
		quantiles = _mdn_quantile_matrix(
			raw_probs,
			p0_arr[gene_indices, sample_idx],
			pg_arr[gene_indices, sample_idx],
			pn_arr[gene_indices, sample_idx],
			beta_arr[gene_indices, sample_idx],
			mu_arr[gene_indices, sample_idx],
			sigma_arr[gene_indices, sample_idx],
		)
		moment_quantiles = quantiles[:, :n_moment_probs]
		pred_mean = np.mean(moment_quantiles, axis=1)
		pred_var = np.maximum(np.mean(moment_quantiles ** 2, axis=1) - pred_mean ** 2, 1e-12)
		ci_lo = np.clip(quantiles[:, n_moment_probs] - float(entry["qhat_low"]), EXPRESSION_FLOOR, EXPRESSION_CEIL)
		ci_hi = np.clip(quantiles[:, n_moment_probs + 1] + float(entry["qhat_high"]), EXPRESSION_FLOOR, EXPRESSION_CEIL)
		ci_hi = np.maximum(ci_hi, ci_lo)

		pred_cal[gene_indices, sample_idx] = pred_mean
		variance_cal[gene_indices, sample_idx] = pred_var
		ci_lo_cal[gene_indices, sample_idx] = ci_lo
		ci_hi_cal[gene_indices, sample_idx] = ci_hi

	return {
		"pred_wide": pd.DataFrame(pred_cal, index=genes, columns=samples),
		"ci_lo_wide": pd.DataFrame(ci_lo_cal, index=genes, columns=samples),
		"ci_hi_wide": pd.DataFrame(ci_hi_cal, index=genes, columns=samples),
		"predictive_variance_wide": pd.DataFrame(variance_cal, index=genes, columns=samples),
	}


def apply_distribution_calibration_bundle(
	bundle: Dict[str, Any],
	calibration_state: Dict[str, Any],
	*,
	source_by_sample: Dict[str, str],
	sample_fold: Optional[Dict[str, int]],
	label: str,
) -> Dict[str, Any]:
	"""Return a bundle with calibrated mean, interval, and predictive-variance surfaces."""
	calibrated_bundle = dict(bundle)
	surfaces = _compute_distribution_calibration_surfaces(
		bundle,
		calibration_state,
		source_by_sample=source_by_sample,
		sample_fold=sample_fold,
	)
	calibrated_bundle["raw_pred_wide"] = bundle.get("pred_wide")
	calibrated_bundle["raw_ci_lo_wide"] = bundle.get("ci_lo_wide")
	calibrated_bundle["raw_ci_hi_wide"] = bundle.get("ci_hi_wide")
	calibrated_bundle.update(surfaces)
	calibrated_bundle["distribution_calibration_applied"] = True
	calibrated_bundle["distribution_calibration_label"] = str(label)
	# Keep the in-memory calibrator attached to calibrated bundles so downstream
	# consumers can reuse the same monotone PIT map instead of re-deriving or
	# silently falling back to raw confidence surfaces.
	calibrated_bundle["distribution_calibration_state"] = calibration_state
	calibrated_bundle["calibration_source_by_sample"] = {
		str(sample_name): str(source_name)
		for sample_name, source_name in source_by_sample.items()
	}
	calibrated_bundle["calibration_sample_fold"] = {
		str(sample_name): int(fold_id)
		for sample_name, fold_id in (sample_fold or {}).items()
		if pd.notna(fold_id)
	}
	return calibrated_bundle


def get_calibrated_activity_surfaces(
	bundle: Dict[str, Any],
	calibration_state: Optional[Dict[str, Any]] = None,
	*,
	source_by_sample: Optional[Dict[str, str]] = None,
	sample_fold: Optional[Dict[str, int]] = None,
) -> Dict[str, pd.DataFrame]:
	"""Return calibrated threshold-active probability and its odds-ratio analogue.

	Distribution calibration in this pipeline is fitted on the thresholded event
	``Y >= 2`` through a monotone PIT/CDF map. Once calibration is applied, the
	raw MDN mixture-weight AOR is no longer the right certainty surface for
	downstream precision weighting because that raw log-odds ignores the fitted
	calibration map entirely. This helper therefore derives a calibrated analogue
	from the calibrated threshold-active probability itself and caches the result
	on the bundle for any downstream consumer that needs a confidence surface.
	"""
	cached_active = bundle.get("calibrated_active_confidence_wide")
	cached_aor = bundle.get("calibrated_activity_odds_ratio_wide")
	if isinstance(cached_active, pd.DataFrame) and isinstance(cached_aor, pd.DataFrame):
		return {
			"active_confidence_wide": cached_active,
			"activity_odds_ratio_wide": cached_aor,
		}

	resolved_state = calibration_state or bundle.get("distribution_calibration_state")
	if resolved_state is None:
		raise ValueError("Calibrated activity surfaces require the fitted distribution calibration state")

	resolved_source_by_sample = source_by_sample or bundle.get("calibration_source_by_sample")
	if not isinstance(resolved_source_by_sample, dict) or not resolved_source_by_sample:
		raise ValueError("Calibrated activity surfaces require calibration source assignments by sample")

	resolved_sample_fold = sample_fold
	if resolved_sample_fold is None and isinstance(bundle.get("calibration_sample_fold"), dict):
		resolved_sample_fold = bundle.get("calibration_sample_fold")

	aligned = _aligned_mdn_bundle_matrices(bundle, require_truth=False)
	if aligned is None:
		raise ValueError("Calibrated activity surfaces require full Proteus MDN outputs")

	active_conf_wide = aligned.get("active_confidence_wide")
	if active_conf_wide is None:
		active_conf_wide = mdn_on_probability_wide(
			aligned["p0_wide"],
			aligned["pi_gamma_wide"],
			aligned["pi_normal_wide"],
			aligned["gamma_beta_wide"],
			aligned["normal_mean_wide"],
			aligned["normal_stdev_wide"],
			threshold=_CALIBRATION_ACTIVITY_THRESHOLD,
		)

	genes = aligned["pred_wide"].index
	samples = aligned["pred_wide"].columns
	pred_arr = aligned["pred_wide"].to_numpy(dtype=np.float64)
	active_conf_arr = active_conf_wide.loc[genes, samples].to_numpy(dtype=np.float64)
	calibrated_active = np.full(active_conf_arr.shape, np.nan, dtype=np.float64)
	for sample_idx, gene_indices, entry in _iter_distribution_calibration_groups(
		genes,
		samples,
		pred_arr,
		active_conf_arr,
		resolved_state,
		source_by_sample={str(sample_name): str(source_name) for sample_name, source_name in resolved_source_by_sample.items()},
		sample_fold=resolved_sample_fold,
	):
		# The fitted map acts on the raw CDF / PIT scale, so we convert the raw
		# threshold-active probability back to the corresponding CDF value,
		# calibrate that probability mass, and then return to the active-event
		# scale. This preserves the exact threshold semantics used during fitting.
		raw_cdf = 1.0 - active_conf_arr[gene_indices, sample_idx]
		calibrated_cdf = _eval_calibration_map(raw_cdf, entry["map_y"])
		calibrated_active[gene_indices, sample_idx] = np.clip(1.0 - calibrated_cdf, 0.0, 1.0)

	active_conf_df = pd.DataFrame(calibrated_active, index=genes, columns=samples)
	safe_active = np.clip(calibrated_active, _CALIBRATION_AOR_EPSILON, 1.0 - _CALIBRATION_AOR_EPSILON)
	activity_odds_ratio_df = pd.DataFrame(
		np.log2(safe_active / (1.0 - safe_active)),
		index=genes,
		columns=samples,
	)
	bundle["calibrated_active_confidence_wide"] = active_conf_df
	bundle["calibrated_activity_odds_ratio_wide"] = activity_odds_ratio_df
	return {
		"active_confidence_wide": active_conf_df,
		"activity_odds_ratio_wide": activity_odds_ratio_df,
	}


def get_calibrated_draws(
	bundle: Dict[str, Any],
	calibration_state: Dict[str, Any],
	*,
	source_by_sample: Dict[str, str],
	sample_fold: Optional[Dict[str, int]],
	M: int,
	seed: int,
) -> List[pd.DataFrame]:
	"""Sample calibrated draws from the recalibrated patient-OOF distribution wrapper."""
	cached_draws = bundle.get("calibrated_draws")
	if cached_draws is not None and int(bundle.get("calibrated_draws_M", -1)) == int(M):
		return cached_draws

	aligned = _aligned_mdn_bundle_matrices(bundle, require_truth=False)
	if aligned is None:
		raise ValueError("Distribution calibration draws require full Proteus MDN surfaces")

	active_conf_wide = aligned.get("active_confidence_wide")
	if active_conf_wide is None:
		active_conf_wide = mdn_on_probability_wide(
			aligned["p0_wide"],
			aligned["pi_gamma_wide"],
			aligned["pi_normal_wide"],
			aligned["gamma_beta_wide"],
			aligned["normal_mean_wide"],
			aligned["normal_stdev_wide"],
			threshold=_CALIBRATION_ACTIVITY_THRESHOLD,
		)

	genes = aligned["pred_wide"].index
	samples = aligned["pred_wide"].columns
	pred_arr = aligned["pred_wide"].to_numpy(dtype=np.float64)
	active_conf_arr = active_conf_wide.loc[genes, samples].to_numpy(dtype=np.float64)
	p0_arr = aligned["p0_wide"].to_numpy(dtype=np.float64)
	pg_arr = aligned["pi_gamma_wide"].to_numpy(dtype=np.float64)
	pn_arr = aligned["pi_normal_wide"].to_numpy(dtype=np.float64)
	beta_arr = aligned["gamma_beta_wide"].to_numpy(dtype=np.float64)
	mu_arr = aligned["normal_mean_wide"].to_numpy(dtype=np.float64)
	sigma_arr = aligned["normal_stdev_wide"].to_numpy(dtype=np.float64)

	draw_arrays = [np.full(pred_arr.shape, np.nan, dtype=np.float64) for _ in range(int(M))]
	rng = np.random.default_rng(seed)
	for sample_idx, gene_indices, entry in _iter_distribution_calibration_groups(
		genes,
		samples,
		pred_arr,
		active_conf_arr,
		calibration_state,
		source_by_sample=source_by_sample,
		sample_fold=sample_fold,
	):
		calibrated_probs = rng.random((gene_indices.size, int(M)))
		raw_probs = _invert_calibration_map(calibrated_probs, entry["map_y"])
		quantiles = _mdn_quantile_matrix(
			raw_probs,
			p0_arr[gene_indices, sample_idx],
			pg_arr[gene_indices, sample_idx],
			pn_arr[gene_indices, sample_idx],
			beta_arr[gene_indices, sample_idx],
			mu_arr[gene_indices, sample_idx],
			sigma_arr[gene_indices, sample_idx],
		)
		quantiles = np.clip(quantiles, EXPRESSION_FLOOR, EXPRESSION_CEIL)
		for draw_idx in range(int(M)):
			draw_arrays[draw_idx][gene_indices, sample_idx] = quantiles[:, draw_idx]

	draws = [pd.DataFrame(draw_arr, index=genes, columns=samples) for draw_arr in draw_arrays]
	bundle["calibrated_draws"] = draws
	bundle["calibrated_draws_M"] = int(M)
	return draws


def _weighted_point_calibration_curve(
	truth_wide: pd.DataFrame,
	pred_wide: pd.DataFrame,
	*,
	n_bins: int = 20,
) -> pd.DataFrame:
	common_genes = truth_wide.index.intersection(pred_wide.index)
	common_samples = truth_wide.columns.intersection(pred_wide.columns)
	if len(common_genes) == 0 or len(common_samples) == 0:
		return pd.DataFrame(columns=["bin", "mean_pred", "mean_truth", "n_pairs"])

	truth_arr = truth_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64)
	pred_arr = pred_wide.loc[common_genes, common_samples].to_numpy(dtype=np.float64)
	valid = np.isfinite(truth_arr) & np.isfinite(pred_arr)
	if int(valid.sum()) < 50:
		return pd.DataFrame(columns=["bin", "mean_pred", "mean_truth", "n_pairs"])

	truth_flat = truth_arr.reshape(-1)[valid.reshape(-1)]
	pred_flat = pred_arr.reshape(-1)[valid.reshape(-1)]
	sample_flat = np.tile(np.asarray([str(sample_name) for sample_name in common_samples], dtype=object), len(common_genes))[valid.reshape(-1)]
	weights = _sample_equal_weights(sample_flat)

	bin_edges = np.unique(np.quantile(pred_flat, np.linspace(0.0, 1.0, n_bins + 1)))
	if len(bin_edges) < 3:
		return pd.DataFrame(columns=["bin", "mean_pred", "mean_truth", "n_pairs"])

	bin_ids = np.digitize(pred_flat, bin_edges[1:-1], right=False)
	rows: List[Dict[str, Any]] = []
	for bin_id in range(len(bin_edges) - 1):
		mask = bin_ids == bin_id
		if not bool(mask.any()):
			continue
		bin_weights = weights[mask]
		weight_sum = float(bin_weights.sum())
		if weight_sum <= 0.0:
			continue
		normalized = bin_weights / weight_sum
		rows.append(
			{
				"bin": int(bin_id),
				"mean_pred": float(np.sum(normalized * pred_flat[mask])),
				"mean_truth": float(np.sum(normalized * truth_flat[mask])),
				"n_pairs": int(mask.sum()),
			}
		)
	return pd.DataFrame(rows)


def _pit_summary_rows(
	raw_pit_wide: pd.DataFrame,
	calibrated_pit_wide: pd.DataFrame,
	source_by_sample: Dict[str, str],
) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	labels: List[Tuple[str, List[str]]] = [
		("Patient", list(raw_pit_wide.columns)),
	]
	for source_name in sorted({str(source_name) for source_name in source_by_sample.values()}):
		labels.append(
			(
				str(source_name),
				[sample_name for sample_name in raw_pit_wide.columns if source_by_sample.get(str(sample_name)) == source_name],
			)
		)

	for label_name, sample_names in labels:
		if not sample_names:
			continue
		raw_arr = raw_pit_wide.loc[:, sample_names].to_numpy(dtype=np.float64)
		cal_arr = calibrated_pit_wide.loc[:, sample_names].to_numpy(dtype=np.float64)
		valid = np.isfinite(raw_arr) & np.isfinite(cal_arr)
		if not bool(valid.any()):
			continue
		raw_flat = raw_arr.reshape(-1)[valid.reshape(-1)]
		cal_flat = cal_arr.reshape(-1)[valid.reshape(-1)]
		sample_flat = np.tile(np.asarray([str(sample_name) for sample_name in sample_names], dtype=object), raw_arr.shape[0])[valid.reshape(-1)]
		weights = _sample_equal_weights(sample_flat)
		raw_mean = float(np.sum(weights * raw_flat))
		cal_mean = float(np.sum(weights * cal_flat))
		rows.append(
			{
				"label": label_name,
				"n_pairs": int(valid.sum()),
				"n_samples": int(len(sample_names)),
				"raw_mean": raw_mean,
				"raw_variance": float(np.sum(weights * (raw_flat - raw_mean) ** 2)),
				"raw_q05": _weighted_quantile(raw_flat, weights, 0.05),
				"raw_q50": _weighted_quantile(raw_flat, weights, 0.50),
				"raw_q95": _weighted_quantile(raw_flat, weights, 0.95),
				"calibrated_mean": cal_mean,
				"calibrated_variance": float(np.sum(weights * (cal_flat - cal_mean) ** 2)),
				"calibrated_q05": _weighted_quantile(cal_flat, weights, 0.05),
				"calibrated_q50": _weighted_quantile(cal_flat, weights, 0.50),
				"calibrated_q95": _weighted_quantile(cal_flat, weights, 0.95),
			}
		)
	return pd.DataFrame(rows)


def _interval_coverage_summary_rows(
	truth_wide: pd.DataFrame,
	raw_ci_lo_wide: pd.DataFrame,
	raw_ci_hi_wide: pd.DataFrame,
	calibrated_ci_lo_wide: pd.DataFrame,
	calibrated_ci_hi_wide: pd.DataFrame,
	source_by_sample: Dict[str, str],
) -> pd.DataFrame:
	rows: List[Dict[str, Any]] = []
	labels: List[Tuple[str, List[str]]] = [("Patient", list(truth_wide.columns))]
	for source_name in sorted({str(source_name) for source_name in source_by_sample.values()}):
		labels.append(
			(
				str(source_name),
				[sample_name for sample_name in truth_wide.columns if source_by_sample.get(str(sample_name)) == source_name],
			)
		)

	for label_name, sample_names in labels:
		if not sample_names:
			continue
		truth_arr = truth_wide.loc[:, sample_names].to_numpy(dtype=np.float64)
		raw_lo = raw_ci_lo_wide.loc[:, sample_names].to_numpy(dtype=np.float64)
		raw_hi = raw_ci_hi_wide.loc[:, sample_names].to_numpy(dtype=np.float64)
		cal_lo = calibrated_ci_lo_wide.loc[:, sample_names].to_numpy(dtype=np.float64)
		cal_hi = calibrated_ci_hi_wide.loc[:, sample_names].to_numpy(dtype=np.float64)
		valid = (
			np.isfinite(truth_arr)
			& np.isfinite(raw_lo)
			& np.isfinite(raw_hi)
			& np.isfinite(cal_lo)
			& np.isfinite(cal_hi)
		)
		if not bool(valid.any()):
			continue
		sample_flat = np.tile(np.asarray([str(sample_name) for sample_name in sample_names], dtype=object), truth_arr.shape[0])[valid.reshape(-1)]
		weights = _sample_equal_weights(sample_flat)
		raw_cover = ((truth_arr >= raw_lo) & (truth_arr <= raw_hi)).reshape(-1)[valid.reshape(-1)].astype(np.float64)
		cal_cover = ((truth_arr >= cal_lo) & (truth_arr <= cal_hi)).reshape(-1)[valid.reshape(-1)].astype(np.float64)
		raw_width = (raw_hi - raw_lo).reshape(-1)[valid.reshape(-1)]
		cal_width = (cal_hi - cal_lo).reshape(-1)[valid.reshape(-1)]
		rows.append(
			{
				"label": label_name,
				"n_pairs": int(valid.sum()),
				"n_samples": int(len(sample_names)),
				"raw_coverage": float(np.sum(weights * raw_cover)),
				"calibrated_coverage": float(np.sum(weights * cal_cover)),
				"raw_mean_width": float(np.sum(weights * raw_width)),
				"calibrated_mean_width": float(np.sum(weights * cal_width)),
			}
		)
	return pd.DataFrame(rows)


def _plot_pit_comparison(
	raw_values: np.ndarray,
	raw_weights: np.ndarray,
	calibrated_values: np.ndarray,
	calibrated_weights: np.ndarray,
	name: str,
	*,
	title: str,
) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	plot_dir = os.path.dirname(name)
	if plot_dir:
		os.makedirs(plot_dir, exist_ok=True)
	bins = np.linspace(0.0, 1.0, 21)
	fig, ax = plt.subplots(figsize=(6.4, 4.8))
	ax.hist(
		raw_values,
		bins=bins,
		weights=raw_weights,
		density=True,
		color="#4C78A8",
		alpha=0.45,
		label="Raw PIT",
		edgecolor="white",
		linewidth=0.2,
	)
	ax.hist(
		calibrated_values,
		bins=bins,
		weights=calibrated_weights,
		density=True,
		color="#F58518",
		alpha=0.45,
		label="Calibrated PIT",
		edgecolor="white",
		linewidth=0.2,
	)
	ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
	ax.set_xlabel("PIT")
	ax.set_ylabel("Weighted density")
	ax.set_title(title)
	ax.legend(fontsize=8, loc="best")
	fig.tight_layout()
	fig.savefig(f"{name}.pdf", bbox_inches="tight")
	plt.close(fig)


def _plot_interval_coverage_comparison(
	summary_df: pd.DataFrame,
	name: str,
	*,
	title: str,
) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	if summary_df.empty:
		return
	plot_dir = os.path.dirname(name)
	if plot_dir:
		os.makedirs(plot_dir, exist_ok=True)
	labels = summary_df["label"].astype(str).tolist()
	x = np.arange(len(labels), dtype=np.float64)
	width = 0.35

	fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
	axes[0].bar(x - width / 2.0, summary_df["raw_coverage"], width=width, color="#4C78A8", label="Raw")
	axes[0].bar(x + width / 2.0, summary_df["calibrated_coverage"], width=width, color="#F58518", label="Calibrated")
	axes[0].axhline(1.0 - _CALIBRATION_INTERVAL_ALPHA, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
	axes[0].set_xticks(x)
	axes[0].set_xticklabels(labels, rotation=20, ha="right")
	axes[0].set_ylim(0.0, 1.0)
	axes[0].set_ylabel("Weighted 90% coverage")
	axes[0].legend(fontsize=8, loc="best")

	axes[1].bar(x - width / 2.0, summary_df["raw_mean_width"], width=width, color="#4C78A8")
	axes[1].bar(x + width / 2.0, summary_df["calibrated_mean_width"], width=width, color="#F58518")
	axes[1].set_xticks(x)
	axes[1].set_xticklabels(labels, rotation=20, ha="right")
	axes[1].set_ylabel("Weighted mean interval width")
	fig.suptitle(title)
	fig.tight_layout()
	fig.savefig(f"{name}.pdf", bbox_inches="tight")
	plt.close(fig)


def write_distribution_calibration_outputs(
	out_dir: str,
	patient_bundle: Dict[str, Any],
	calibration_state: Dict[str, Any],
	*,
	overwrite: bool,
	maybe_write: Callable[..., Any],
) -> None:
	"""Write calibration summaries and sanity-check plots under the Calibration section."""
	patient_out = ensure_dir(os.path.join(out_dir, "PatientDistributionCalibration"))
	full_calibrator = calibration_state["full_calibrator"]
	summary_rows = pd.DataFrame(full_calibrator["summary_rows"])
	summary_rows.to_csv(os.path.join(patient_out, "stratum_summary.tsv"), sep="\t", index=False)

	calibrated_bundle = apply_distribution_calibration_bundle(
		patient_bundle,
		calibration_state,
		source_by_sample=patient_bundle["source_by_sample"],
		sample_fold=patient_bundle["sample_fold"],
		label="Patient CV OOF cross-fit",
	)
	raw_pit_wide = _mdn_randomized_pit_wide(
		patient_bundle["truth_wide"],
		patient_bundle["p0_wide"],
		patient_bundle["pi_gamma_wide"],
		patient_bundle["pi_normal_wide"],
		patient_bundle["gamma_beta_wide"],
		patient_bundle["normal_mean_wide"],
		patient_bundle["normal_stdev_wide"],
		random_seed=_CALIBRATION_RANDOM_SEED,
	)
	calibrated_pit_wide = raw_pit_wide.copy()
	pit_arr = raw_pit_wide.to_numpy(dtype=np.float64)
	pred_arr = patient_bundle["pred_wide"].to_numpy(dtype=np.float64)
	active_conf_arr = patient_bundle["active_confidence_wide"].to_numpy(dtype=np.float64)
	calibrated_arr = calibrated_pit_wide.to_numpy(dtype=np.float64)
	for sample_idx, gene_indices, entry in _iter_distribution_calibration_groups(
		patient_bundle["truth_wide"].index,
		patient_bundle["truth_wide"].columns,
		pred_arr,
		active_conf_arr,
		calibration_state,
		source_by_sample=patient_bundle["source_by_sample"],
		sample_fold=patient_bundle["sample_fold"],
	):
		raw_values = pit_arr[gene_indices, sample_idx]
		finite = np.isfinite(raw_values)
		if not bool(finite.any()):
			continue
		calibrated_arr[gene_indices[finite], sample_idx] = _eval_calibration_map(raw_values[finite], entry["map_y"])
	calibrated_pit_wide = pd.DataFrame(
		calibrated_arr,
		index=patient_bundle["truth_wide"].index,
		columns=patient_bundle["truth_wide"].columns,
	)

	pit_summary = _pit_summary_rows(
		raw_pit_wide,
		calibrated_pit_wide,
		patient_bundle["source_by_sample"],
	)
	pit_summary.to_csv(os.path.join(patient_out, "pit_summary_crossfit.tsv"), sep="\t", index=False)

	point_curve_raw = _weighted_point_calibration_curve(
		patient_bundle["truth_wide"],
		patient_bundle["pred_wide"],
	)
	point_curve_calibrated = _weighted_point_calibration_curve(
		patient_bundle["truth_wide"],
		calibrated_bundle["pred_wide"],
	)
	point_curve_raw.to_csv(os.path.join(patient_out, "point_calibration_curve_raw.tsv"), sep="\t", index=False)
	point_curve_calibrated.to_csv(os.path.join(patient_out, "point_calibration_curve_calibrated_crossfit.tsv"), sep="\t", index=False)
	maybe_write(
		plot_prediction_calibration_curve,
		overwrite,
		point_curve_raw,
		point_curve_calibrated,
		os.path.join(patient_out, "point_calibration_curve_crossfit"),
		title="Patient OOF Point Calibration (cross-fit)",
	)

	interval_summary = _interval_coverage_summary_rows(
		patient_bundle["truth_wide"],
		patient_bundle["ci_lo_wide"],
		patient_bundle["ci_hi_wide"],
		calibrated_bundle["ci_lo_wide"],
		calibrated_bundle["ci_hi_wide"],
		patient_bundle["source_by_sample"],
	)
	interval_summary.to_csv(os.path.join(patient_out, "interval_coverage_summary_crossfit.tsv"), sep="\t", index=False)
	maybe_write(
		_plot_interval_coverage_comparison,
		overwrite,
		interval_summary,
		os.path.join(patient_out, "interval_coverage_comparison_crossfit"),
		title="Patient OOF 90% Interval Coverage (cross-fit)",
	)

	raw_valid = np.isfinite(raw_pit_wide.to_numpy(dtype=np.float64))
	raw_values = raw_pit_wide.to_numpy(dtype=np.float64)[raw_valid]
	sample_flat = np.tile(
		np.asarray([str(sample_name) for sample_name in raw_pit_wide.columns], dtype=object),
		raw_pit_wide.shape[0],
	)[raw_valid.reshape(-1)]
	raw_weights = _sample_equal_weights(sample_flat)
	calibrated_values = calibrated_pit_wide.to_numpy(dtype=np.float64)[raw_valid]
	calibrated_weights = raw_weights.copy()
	maybe_write(
		_plot_pit_comparison,
		overwrite,
		raw_values,
		raw_weights,
		calibrated_values,
		calibrated_weights,
		os.path.join(patient_out, "pit_histogram_crossfit"),
		title="Patient OOF PIT (raw vs calibrated cross-fit)",
	)
