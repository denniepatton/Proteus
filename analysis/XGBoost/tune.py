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


def _suggest_params(
    trial: optuna.Trial,
    search_profile: str,
) -> Dict[str, Any]:
    if search_profile == "plca":
        return {
            "max_depth": trial.suggest_int("max_depth", 8, 14),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.03, 0.08, log=True,
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 5.0, 40.0, log=True,
            ),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 0.8,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 1e-2, log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 1e-2, log=True,
            ),
            "gamma": trial.suggest_float("gamma", 1e-8, 1e-3, log=True),
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
    abs_err = np.abs(pred.astype(np.float64) - truth.astype(np.float64))
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
    per_source: Dict[str, float] = {}
    for source_name in PLCA_WEIGHTED_SOURCE_WEIGHTS:
        mask = np.array(
            [sample_to_source.get(str(sample)) == source_name for sample in samples],
            dtype=bool,
        )
        if int(mask.sum()) == 0:
            return float("nan"), {}

        source_weight_map = source_variance.get(source_name)
        if source_weight_map is None:
            return float("nan"), {}

        w_src = np.array(
            [float(source_weight_map.get(str(gene), 1.0)) for gene in genes[mask]],
            dtype=np.float64,
        )
        per_source[source_name] = _weighted_mae(pred[mask], truth[mask], w_src)

    return weighted_group_mean(per_source, PLCA_WEIGHTED_SOURCE_WEIGHTS), per_source


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
        fold_data[fold_num] = {
            "dtrain": dtrain,
            "dval": dval,
            "w_val": w_val,
            "y_val": y_val,
            "genes_val": genes_val,
            "samples_val": samples_val,
        }
    print(f"Data pre-loaded for {len(FOLDS)} folds.")
    print(f"  Train DMatrix: {fold_data[1]['dtrain'].num_row():,} rows")
    print(f"  Val DMatrix:   {fold_data[1]['dval'].num_row():,} rows")
    print(f"  Search profile: {search_profile}")
    print(f"  Objective metric: {metric_name}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "seed": seed,
            "nthread": nthread,
            "verbosity": 0,
            **_suggest_params(trial, search_profile),
        }

        fold_scores: List[float] = []
        fold_iters: List[int] = []
        fold_source_scores: Dict[int, Dict[str, float]] = {}
        for fold_num in FOLDS:
            t_fold = time.perf_counter()
            fd = fold_data[fold_num]
            model = xgb.train(
                params,
                fd["dtrain"],
                num_boost_round=n_estimators,
                evals=[(fd["dval"], "val")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
            pred = model.predict(fd["dval"])

            if metric_name == "weighted_source_gvw_mae":
                if sample_to_source is None or source_variance is None:
                    raise RuntimeError(
                        "weighted_source_gvw_mae requires PLCA source metadata",
                    )
                fold_score, per_source = _score_weighted_source_gvw_mae(
                    pred,
                    fd["y_val"],
                    fd["genes_val"],
                    fd["samples_val"],
                    sample_to_source,
                    source_variance,
                )
                fold_source_scores[fold_num] = per_source
            else:
                err = (pred - fd["y_val"].astype(np.float64)) ** 2
                fold_score = float(np.sum(fd["w_val"] * err) / np.sum(fd["w_val"]))

            fold_scores.append(fold_score)
            fold_iters.append(int(model.best_iteration))
            print(
                f"    T{trial.number} F{fold_num}: "
                f"{model.best_iteration} iters, "
                f"score={fold_score:.6f}, "
                f"{time.perf_counter() - t_fold:.1f}s",
                flush=True,
            )

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
        "--nthread", type=int, default=None,
        help="Threads per XGBoost model (default: use all available CPUs)",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Parallel Optuna trials (default: 1). Safe because XGBoost releases the GIL.",
    )
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
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
    )

    # ---- Run trials ----
    cpu_count = os.cpu_count() or 1
    if args.nthread is None or args.nthread <= 0:
        nthread = cpu_count
    else:
        nthread = min(args.nthread, cpu_count)
    n_jobs = max(1, args.n_jobs)
    print(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials")
    print(f"  Storage: {args.storage}")
    print(f"  Data: {tsv_path}")
    print(f"  Zarr: {zarr_base}")
    print(f"  Results dir: {args.results_dir}")
    print(f"  Max rounds per trial: {args.n_estimators}")
    print(f"  Objective metric: {metric_name}")
    print(f"  Search profile: {search_profile}")
    print(f"  CPUs available: {cpu_count}, nthread={nthread}, n_jobs={n_jobs} "
          f"(total threads: {nthread * n_jobs})")

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
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
        "n_estimators_max": args.n_estimators,
    }
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Full evaluation of best config ----
    if not args.skip_final_eval:
        print("\nRunning full evaluation of best config...")
        best_params = {
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
