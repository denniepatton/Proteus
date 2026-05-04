"""Optuna-based HP tuning for the XGBoost baseline.

Runs TPE-guided search over XGBoost HPs using 5-fold CV.  The default
baseline objective is GVW-MSE, while PLCA runs can switch to the
Proteus-aligned weighted source-balanced GVW-MAE surrogate.

Usage
-----
    PYTHONPATH=analysis python -m XGBoost.tune [OPTIONS]

See ``--help`` for options.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import xgboost as xgb

from proteus_utils.data_helpers import (
    PLCA_WEIGHTED_SOURCE_WEIGHTS,
    build_source_map_from_tsv,
    compute_source_variance,
    weighted_group_mean,
    zarr_name_for_sample,
)

from .data_utils import (
    PROJECT_ROOT,
    collect_zarr_paths,
    compute_gene_variances,
    default_splits_tsv,
    default_zarr_base,
    gene_weight_vector,
    load_dataset,
)
from .train import EARLY_STOPPING_ROUNDS, run_cv

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_N_TRIALS = 200
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT / 'analysis' / 'XGBoost' / 'optuna_xgb.db'}"
DEFAULT_STUDY_NAME = "xgboost_baseline"
DEFAULT_RESULTS_DIR = str(
    PROJECT_ROOT / "analysis" / "XGBoost" / "tuning_results"
)
FOLDS = [1, 2, 3, 4, 5]
PRUNER_STARTUP_TRIALS = 2
PROGRESS_LOG_EVERY_ROUNDS = 100
MIN_THREADS_PER_TRIAL = 4
MAX_PARALLEL_TRIALS = 4


def _available_cpu_count() -> int:
    """Return the CPU budget available to this process on HPC or locally."""
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            cpu_count = int(slurm_cpus)
        except ValueError:
            cpu_count = 0
        if cpu_count > 0:
            return cpu_count

    try:
        affinity_count = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity_count = 0
    except OSError:
        affinity_count = 0
    if affinity_count > 0:
        return affinity_count

    return max(1, os.cpu_count() or 1)


def _resolve_parallelism(cpu_count: int) -> tuple[int, int]:
    """Split available CPUs across a small number of concurrent Optuna trials.

    For this 25-feature XGBoost workload, single-trial scaling saturates well
    before typical CPU allocations, so throughput is usually better with a few
    concurrent trials instead of one trial monopolizing every core.
    """
    safe_cpu_count = max(1, int(cpu_count))
    n_jobs = min(
        MAX_PARALLEL_TRIALS,
        max(1, safe_cpu_count // MIN_THREADS_PER_TRIAL),
    )
    nthread = max(1, safe_cpu_count // n_jobs)
    return n_jobs, nthread


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------

PLCA_SOURCE_GROUPS: Dict[str, List[str]] = {
    "CRPC-RA": ["CRPC-RA"],
    "SCLC": ["SCLC"],
    "Prostate-Synthetic": ["PC-PDX"],
    "SCLC-Synthetic": ["SCLC-PDX"],
}


def _is_plca_tsv(tsv_path: str) -> bool:
    return Path(tsv_path).name == "PLCA_sample-splits.tsv"


def _resolve_metric_name(metric_name: str, tsv_path: str) -> str:
    if metric_name != "auto":
        return metric_name
    if _is_plca_tsv(tsv_path):
        return "weighted_source_gvw_mae"
    return "gvw_mse"


def _resolve_search_profile(search_profile: str, tsv_path: str) -> str:
    if search_profile != "auto":
        return search_profile
    if _is_plca_tsv(tsv_path):
        return "plca"
    return "baseline"


def _booster_metric_params(metric_name: str) -> Dict[str, Any]:
    """Return fixed booster objective params for the selected tuning metric."""
    if metric_name == "weighted_source_gvw_mae":
        return {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
        }
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }


def _suggest_params(
    trial: optuna.Trial,
    search_profile: str,
) -> Dict[str, Any]:
    if search_profile == "plca":
        return {
            "max_depth": trial.suggest_int("max_depth", 10, 14),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.03, 0.06, log=True,
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 10.0, 40.0, log=True,
            ),
            "subsample": trial.suggest_float("subsample", 0.85, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.65, 0.82,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-7, 1e-2, log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-6, 1e-2, log=True,
            ),
            "gamma": trial.suggest_float("gamma", 1e-5, 1e-3, log=True),
        }

    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }


def _weighted_mae(
    pred: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> float:
    pred64 = np.asarray(pred, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    abs_err = np.abs(pred64 - truth64)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(weights * abs_err) / denom)


def _augment_sample_source_aliases(
    sample_to_source: Dict[str, str],
) -> Dict[str, str]:
    """Add unambiguous bare-name aliases for TSV sample IDs.

    XGBoost validation rows carry the Zarr ``sample_name`` attribute, which
    may be the bare on-disk sample name rather than the full TSV sample ID.
    """
    alias_sources: Dict[str, set[str]] = {}
    for sample_name, source_name in sample_to_source.items():
        bare_name = sample_name
        for cohort_name in ("CRPC-RA", "SCLC", "PC-PDX", "SCLC-PDX", "PHT"):
            candidate = zarr_name_for_sample(sample_name, cohort_name)
            if candidate != sample_name:
                bare_name = candidate
                break
        alias_sources.setdefault(bare_name, set()).add(source_name)

    augmented = dict(sample_to_source)
    for bare_name, source_names in alias_sources.items():
        if len(source_names) == 1:
            augmented.setdefault(bare_name, next(iter(source_names)))
    return augmented


def _score_weighted_source_gvw_mae(
    pred: np.ndarray,
    truth: np.ndarray,
    genes: np.ndarray,
    samples: np.ndarray,
    sample_to_source: Dict[str, str],
    source_variance: Dict[str, Dict[str, float]],
) -> tuple[float, Dict[str, float]]:
    prepared = _prepare_weighted_source_eval_data(
        genes,
        samples,
        sample_to_source,
        source_variance,
    )
    return _score_weighted_source_gvw_mae_prepared(pred, truth, prepared)


def _prepare_weighted_source_eval_data(
    genes: np.ndarray,
    samples: np.ndarray,
    sample_to_source: Dict[str, str],
    source_variance: Dict[str, Dict[str, float]],
) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    """Precompute row indices and source-specific gene weights for fast eval."""
    row_sources = np.fromiter(
        (str(sample_to_source.get(str(sample), "")) for sample in samples),
        dtype=object,
        count=len(samples),
    )

    prepared: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_source: Dict[str, float] = {}
    for source_name in PLCA_WEIGHTED_SOURCE_WEIGHTS:
        idx = np.flatnonzero(row_sources == source_name)
        if idx.size == 0:
            raise RuntimeError(
                f'No validation rows found for PLCA source "{source_name}"',
            )

        source_weight_map = source_variance.get(source_name)
        if source_weight_map is None:
            raise RuntimeError(
                f'Missing source variance map for PLCA source "{source_name}"',
            )

        w_src = np.fromiter(
            (float(source_weight_map.get(str(gene), 1.0)) for gene in genes[idx]),
            dtype=np.float64,
            count=idx.size,
        )
        prepared[source_name] = (idx, w_src)

    return prepared


def _score_weighted_source_gvw_mae_prepared(
    pred: np.ndarray,
    truth: np.ndarray,
    prepared: Dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, Dict[str, float]]:
    per_source: Dict[str, float] = {}
    for source_name, (idx, w_src) in prepared.items():
        per_source[source_name] = _weighted_mae(pred[idx], truth[idx], w_src)

    return weighted_group_mean(per_source, PLCA_WEIGHTED_SOURCE_WEIGHTS), per_source


def _weighted_source_gvw_mae_eval_fn(
    prepared: Dict[str, tuple[np.ndarray, np.ndarray]],
    truth: np.ndarray,
    profiler: Optional["_MetricProfiler"] = None,
):
    """Build an XGBoost custom eval callable for weighted_source_gvw_mae."""

    truth64 = np.asarray(truth, dtype=np.float64)

    def _eval(pred: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
        t0 = time.perf_counter()
        score, _ = _score_weighted_source_gvw_mae_prepared(
            pred,
            truth64,
            prepared,
        )
        if profiler is not None:
            profiler.record(time.perf_counter() - t0)
        return "weighted_source_gvw_mae", float(score)

    return _eval


def _xgb_custom_metric_kwargs(metric_fn) -> Dict[str, Any]:
    """Return the supported train() kwarg for a custom evaluation metric."""
    train_params = inspect.signature(xgb.train).parameters
    if "custom_metric" in train_params:
        return {"custom_metric": metric_fn}
    if "feval" in train_params:
        return {"feval": metric_fn}
    return {}


class _MetricProfiler:
    """Accumulate timing for Python-side custom metric evaluation."""

    def __init__(self, label: str):
        self.label = label
        self.calls = 0
        self.total_seconds = 0.0
        self.last_seconds = 0.0

    def record(self, duration_seconds: float) -> None:
        self.calls += 1
        self.total_seconds += float(duration_seconds)
        self.last_seconds = float(duration_seconds)

    def summary(self, total_elapsed_seconds: float) -> str:
        if self.calls <= 0:
            return f"{self.label}=0 calls"
        mean_ms = 1000.0 * self.total_seconds / self.calls
        share_pct = 100.0 * self.total_seconds / max(total_elapsed_seconds, 1e-9)
        return (
            f"{self.label}={self.total_seconds:.1f}s over {self.calls} calls "
            f"({mean_ms:.1f} ms/call, {share_pct:.1f}% wall)"
        )


class _FoldProgressCallback(xgb.callback.TrainingCallback):
    """Periodic progress logging from inside xgb.train()."""

    def __init__(
        self,
        trial_number: int,
        fold_num: int,
        fold_step: int,
        max_rounds: int,
        metric_profiler: Optional[_MetricProfiler],
        trial: Optional[optuna.Trial],
        completed_fold_scores: List[float],
        log_every_rounds: int = PROGRESS_LOG_EVERY_ROUNDS,
    ):
        self.trial_number = int(trial_number)
        self.fold_num = int(fold_num)
        self.fold_step = int(fold_step)
        self.max_rounds = int(max_rounds)
        self.metric_profiler = metric_profiler
        self.trial = trial
        self.completed_fold_scores = completed_fold_scores
        self.log_every_rounds = int(log_every_rounds)
        self.start_time = time.perf_counter()
        self.progress_points_per_fold = self.max_rounds // self.log_every_rounds + 2

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        round_num = int(epoch) + 1
        if round_num != 1 and round_num % self.log_every_rounds != 0:
            return False

        elapsed_seconds = time.perf_counter() - self.start_time
        val_metrics = evals_log.get("val", {})
        metric_parts: List[str] = []
        for metric_name, history in val_metrics.items():
            if history:
                metric_parts.append(f"{metric_name}={float(history[-1]):.6f}")

        profiler_part = ""
        if self.metric_profiler is not None:
            profiler_part = f", {self.metric_profiler.summary(elapsed_seconds)}"

        metric_summary = ""
        if metric_parts:
            metric_summary = ", " + ", ".join(metric_parts)

        current_metric_value: Optional[float] = None
        if val_metrics:
            first_metric_history = next(iter(val_metrics.values()))
            if first_metric_history:
                current_metric_value = float(first_metric_history[-1])

        print(
            f"      T{self.trial_number} F{self.fold_num} iter "
            f"{round_num}/{self.max_rounds}: elapsed={elapsed_seconds:.1f}s"
            f"{metric_summary}{profiler_part}",
            flush=True,
        )

        if self.trial is not None and current_metric_value is not None:
            partial_cv_scores = list(self.completed_fold_scores)
            partial_cv_scores.append(current_metric_value)
            partial_mean = float(np.mean(partial_cv_scores))
            checkpoint_index = 0 if round_num == 1 else round_num // self.log_every_rounds
            global_step = (
                (self.fold_step - 1) * self.progress_points_per_fold
                + checkpoint_index
                + 1
            )
            self.trial.report(partial_mean, step=global_step)
            if self.trial.should_prune():
                print(
                    f"      T{self.trial_number} F{self.fold_num} pruned at iter "
                    f"{round_num}/{self.max_rounds}: partial_cv={partial_mean:.6f}",
                    flush=True,
                )
                raise optuna.TrialPruned()
        return False


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_objective(
    tsv_path: str,
    zarr_base: str,
    seed: int,
    nthread: int,
    n_estimators: int,
    metric_name: str,
    search_profile: str,
):
    """Return an Optuna objective closure.

    The objective trains all 5 folds with early stopping and returns the
    mean validation metric for the selected search profile.
    """
    sample_to_source: Optional[Dict[str, str]] = None
    source_variance: Optional[Dict[str, Dict[str, float]]] = None
    if metric_name == "weighted_source_gvw_mae":
        sample_to_source, source_variance_paths = build_source_map_from_tsv(
            tsv_path,
            PLCA_SOURCE_GROUPS,
            zarr_base,
        )
        sample_to_source = _augment_sample_source_aliases(sample_to_source)
        source_variance = compute_source_variance(
            PLCA_SOURCE_GROUPS,
            source_variance_paths,
            lambda paths, label_key: compute_gene_variances(
                paths,
                label_key=label_key,
            )[0],
        )

    # Pre-load data once so each trial doesn't re-read zarr
    fold_data: Dict[int, Dict[str, Any]] = {}
    for fold_num in FOLDS:
        train_zarr = collect_zarr_paths(tsv_path, zarr_base, exclude_fold=fold_num)
        val_zarr = collect_zarr_paths(tsv_path, zarr_base, fold=fold_num)
        _, clipped = compute_gene_variances(train_zarr)

        X_train, y_train, genes_train, _ = load_dataset(train_zarr)
        X_val, y_val, genes_val, samples_val = load_dataset(val_zarr)
        w_train = gene_weight_vector(genes_train, clipped, default=0.1)
        w_val = gene_weight_vector(genes_val, clipped, default=0.1)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
        prepared_source_eval = None
        if metric_name == "weighted_source_gvw_mae":
            if sample_to_source is None or source_variance is None:
                raise RuntimeError(
                    "weighted_source_gvw_mae requires PLCA source metadata",
                )
            prepared_source_eval = _prepare_weighted_source_eval_data(
                genes_val,
                samples_val,
                sample_to_source,
                source_variance,
            )
        fold_data[fold_num] = {
            "dtrain": dtrain,
            "dval": dval,
            "w_val": w_val,
            "y_val": y_val,
            "genes_val": genes_val,
            "samples_val": samples_val,
            "prepared_source_eval": prepared_source_eval,
        }
    print(f"Data pre-loaded for {len(FOLDS)} folds.")
    print(
        f"  Train DMatrix: {fold_data[1]['dtrain'].num_row():,} rows x "
        f"{fold_data[1]['dtrain'].num_col():,} features"
    )
    print(
        f"  Val DMatrix:   {fold_data[1]['dval'].num_row():,} rows x "
        f"{fold_data[1]['dval'].num_col():,} features"
    )
    print(f"  Search profile: {search_profile}")
    print(f"  Objective metric: {metric_name}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params: Dict[str, Any] = {
            "tree_method": "hist",
            "seed": seed,
            "nthread": nthread,
            "verbosity": 0,
            **_booster_metric_params(metric_name),
            **_suggest_params(trial, search_profile),
        }
        print(
            f"  Trial {trial.number:>3d} params: "
            f"depth={params['max_depth']} lr={params['learning_rate']:.5f} "
            f"min_child={params['min_child_weight']:.4f} "
            f"sub={params['subsample']:.3f} col={params['colsample_bytree']:.3f} "
            f"alpha={params['reg_alpha']:.3e} lambda={params['reg_lambda']:.3e} "
            f"gamma={params['gamma']:.3e}",
            flush=True,
        )

        fold_scores: List[float] = []
        fold_iters: List[int] = []
        fold_source_scores: Dict[int, Dict[str, float]] = {}
        for fold_step, fold_num in enumerate(FOLDS, start=1):
            t_fold = time.perf_counter()
            fd = fold_data[fold_num]
            metric_profiler: Optional[_MetricProfiler] = None
            print(
                f"    T{trial.number} F{fold_num}: starting",
                flush=True,
            )
            train_kwargs: Dict[str, Any] = {
                "params": dict(params),
                "dtrain": fd["dtrain"],
                "num_boost_round": n_estimators,
                "evals": [(fd["dval"], "val")],
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "verbose_eval": False,
                "callbacks": [
                    _FoldProgressCallback(
                        trial_number=trial.number,
                        fold_num=fold_num,
                        fold_step=fold_step,
                        max_rounds=n_estimators,
                        metric_profiler=None,
                        trial=trial,
                        completed_fold_scores=fold_scores,
                    )
                ],
            }
            if metric_name == "weighted_source_gvw_mae":
                if sample_to_source is None or source_variance is None:
                    raise RuntimeError(
                        "weighted_source_gvw_mae requires PLCA source metadata",
                    )
                metric_profiler = _MetricProfiler("python_weighted_source_eval")
                train_kwargs["callbacks"] = [
                    _FoldProgressCallback(
                        trial_number=trial.number,
                        fold_num=fold_num,
                        fold_step=fold_step,
                        max_rounds=n_estimators,
                        metric_profiler=metric_profiler,
                        trial=trial,
                        completed_fold_scores=fold_scores,
                    )
                ]
                custom_metric_kwargs = _xgb_custom_metric_kwargs(
                    _weighted_source_gvw_mae_eval_fn(
                        fd["prepared_source_eval"],
                        fd["y_val"],
                        metric_profiler,
                    )
                )
                if custom_metric_kwargs:
                    train_kwargs["params"]["disable_default_eval_metric"] = 1
                    train_kwargs["params"].pop("eval_metric", None)
                    train_kwargs.update(custom_metric_kwargs)

            model = xgb.train(**train_kwargs)
            pred = model.predict(fd["dval"])

            if metric_name == "weighted_source_gvw_mae":
                if fd["prepared_source_eval"] is None:
                    raise RuntimeError(
                        "weighted_source_gvw_mae requires prepared PLCA eval data",
                    )
                fold_score, per_source = _score_weighted_source_gvw_mae_prepared(
                    pred,
                    fd["y_val"],
                    fd["prepared_source_eval"],
                )
                fold_source_scores[fold_num] = per_source
            else:
                err = (pred - fd["y_val"].astype(np.float64)) ** 2
                fold_score = float(np.sum(fd["w_val"] * err) / np.sum(fd["w_val"]))

            fold_scores.append(fold_score)
            fold_iters.append(int(model.best_iteration))
            partial_mean = float(np.mean(fold_scores))
            profiler_summary = ""
            if metric_profiler is not None:
                profiler_summary = ", " + metric_profiler.summary(
                    time.perf_counter() - t_fold
                )
            print(
                f"    T{trial.number} F{fold_num}: "
                f"{model.best_iteration} iters, "
                f"score={fold_score:.6f}, "
                f"{time.perf_counter() - t_fold:.1f}s"
                f"{profiler_summary}",
                flush=True,
            )
            fold_step_offset = (fold_step * (n_estimators // PROGRESS_LOG_EVERY_ROUNDS + 2))
            trial.report(partial_mean, step=fold_step_offset)
            if trial.should_prune():
                trial.set_user_attr("fold_scores_partial", list(fold_scores))
                trial.set_user_attr("fold_iters_partial", list(fold_iters))
                if fold_source_scores:
                    trial.set_user_attr(
                        "fold_source_scores_partial",
                        fold_source_scores,
                    )
                print(
                    f"    T{trial.number} pruned after F{fold_num}: "
                    f"partial_{metric_name}={partial_mean:.6f}",
                    flush=True,
                )
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        elapsed = time.perf_counter() - t0
        mean_iters = int(np.mean(fold_iters))
        print(
            f"  Trial {trial.number:>3d} | "
            f"{metric_name}: {mean_score:.6f} | "
            f"iters: {fold_iters} (mean {mean_iters}) | "
            f"{elapsed:.1f}s"
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_iters", fold_iters)
        if fold_source_scores:
            trial.set_user_attr("fold_source_scores", fold_source_scores)
        return mean_score

    return objective


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Optuna HP tuning for XGBoost baseline",
    )
    parser.add_argument("--n_trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--storage", type=str, default=DEFAULT_STORAGE)
    parser.add_argument("--study_name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_estimators", type=int, default=1000,
        help="Max boosting rounds per trial (early stopping decides actual count)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["auto", "gvw_mse", "weighted_source_gvw_mae"],
        default="auto",
        help="Trial objective metric. 'auto' uses the Proteus-aligned PLCA metric for PLCA splits.",
    )
    parser.add_argument(
        "--search_profile",
        type=str,
        choices=["auto", "baseline", "plca"],
        default="auto",
        help="Hyperparameter search profile. 'auto' switches to the narrowed PLCA ranges for PLCA splits.",
    )
    parser.add_argument("--tsv", type=str, default=None)
    parser.add_argument("--zarr_base", type=str, default=None)
    parser.add_argument(
        "--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Directory for best_params.json and optional best_cv_eval outputs",
    )
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Skip full evaluation of best trial after tuning",
    )
    args = parser.parse_args(argv)

    tsv_path = args.tsv or default_splits_tsv()
    zarr_base = args.zarr_base or default_zarr_base()
    metric_name = _resolve_metric_name(args.metric, tsv_path)
    search_profile = _resolve_search_profile(args.search_profile, tsv_path)

    # ---- Create / load study ----
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=PRUNER_STARTUP_TRIALS,
        n_warmup_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # ---- Run trials ----
    cpu_count = _available_cpu_count()
    n_jobs, nthread = _resolve_parallelism(cpu_count)
    print(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials")
    print(f"  Storage: {args.storage}")
    print(f"  Data: {tsv_path}")
    print(f"  Zarr: {zarr_base}")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Max rounds per trial: {args.n_estimators}")
    print(f"  Objective metric: {metric_name}")
    print(f"  Search profile: {search_profile}")
    print(
        f"  CPUs available: {cpu_count}, auto parallelism={n_jobs} concurrent "
        f"trial(s) x {nthread} thread(s)"
    )
    print(
        f"  Trial pruning: median pruner after {PRUNER_STARTUP_TRIALS} full trials "
        "with fold-level intermediate reports"
    )

    objective = _build_objective(
        tsv_path,
        zarr_base,
        args.seed,
        nthread,
        args.n_estimators,
        metric_name,
        search_profile,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # ---- Report ----
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"  Value ({metric_name}): {study.best_trial.value:.6f}")
    if metric_name == "gvw_mse":
        print(f"  RMSE ≈ {math.sqrt(study.best_trial.value):.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # Save study summary
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "objective_metric": metric_name,
        "search_profile": search_profile,
        "best_params": {
            **study.best_trial.params,
            **_booster_metric_params(metric_name),
        },
        "n_trials": len(study.trials),
        "n_estimators_max": args.n_estimators,
    }
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Full evaluation of best config ----
    if not args.skip_final_eval:
        print("\nRunning full evaluation of best config...")
        best_params = {
            **_booster_metric_params(metric_name),
            **study.best_trial.params,
            "tree_method": "hist",
            "seed": args.seed,
            "nthread": nthread,
            "verbosity": 0,
        }
        eval_dir = os.path.join(out_dir, "best_cv_eval")
        cv_summary = run_cv(
            xgb_params=best_params,
            n_estimators=args.n_estimators,
            out_dir=eval_dir,
            tsv_path=tsv_path,
            zarr_base=zarr_base,
        )
        with open(os.path.join(out_dir, "best_cv_summary.json"), "w") as f:
            json.dump(cv_summary, f, indent=2)


if __name__ == "__main__":
    main()
