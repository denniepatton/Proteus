"""Optuna-based HP tuning for the XGBoost baseline.

Runs TPE-guided search over standard XGBoost HPs using 5-fold CV with
GVW-MSE (weighted squared error) as the objective.  After the study
completes, the best config is evaluated with full metrics and saved.

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
FOLDS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_objective(
    tsv_path: str,
    zarr_base: str,
    seed: int,
    nthread: int,
):
    """Return an Optuna objective closure.

    The objective trains all 5 folds with early stopping and returns the
    mean validation GVW-MSE (= weighted MSE, since XGBoost with
    ``sample_weight`` minimises exactly that).
    """
    # Pre-load data once so each trial doesn't re-read zarr
    fold_data: Dict[int, Dict[str, Any]] = {}
    for fold_num in FOLDS:
        train_zarr = collect_zarr_paths(tsv_path, zarr_base, exclude_fold=fold_num)
        val_zarr = collect_zarr_paths(tsv_path, zarr_base, fold=fold_num)
        _, clipped = compute_gene_variances(train_zarr)

        X_train, y_train, genes_train, _ = load_dataset(train_zarr)
        X_val, y_val, genes_val, _ = load_dataset(val_zarr)
        w_train = gene_weight_vector(genes_train, clipped, default=0.1)
        w_val = gene_weight_vector(genes_val, clipped, default=0.1)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)
        fold_data[fold_num] = {
            "dtrain": dtrain,
            "dval": dval,
            "w_val": w_val,
            "y_val": y_val,
        }
    print(f"Data pre-loaded for {len(FOLDS)} folds.")
    print(f"  Train DMatrix: {fold_data[1]['dtrain'].num_row():,} rows")
    print(f"  Val DMatrix:   {fold_data[1]['dval'].num_row():,} rows")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()
        params: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "seed": seed,
            "nthread": nthread,
            "verbosity": 0,
            # Tuned HPs
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
        n_estimators = 1000  # upper bound; early stopping decides

        fold_scores: List[float] = []
        fold_iters: List[int] = []
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
            # Weighted MSE on validation
            pred = model.predict(fd["dval"])
            err = (pred - fd["y_val"].astype(np.float64)) ** 2
            wmse = float(np.sum(fd["w_val"] * err) / np.sum(fd["w_val"]))
            fold_scores.append(wmse)
            fold_iters.append(int(model.best_iteration))
            print(
                f"    T{trial.number} F{fold_num}: "
                f"{model.best_iteration} iters, "
                f"{time.perf_counter() - t_fold:.1f}s",
                flush=True,
            )

        mean_wmse = float(np.mean(fold_scores))
        elapsed = time.perf_counter() - t0
        mean_iters = int(np.mean(fold_iters))
        print(
            f"  Trial {trial.number:>3d} | "
            f"GVW-MSE: {mean_wmse:.6f} | "
            f"iters: {fold_iters} (mean {mean_iters}) | "
            f"{elapsed:.1f}s"
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_iters", fold_iters)
        return mean_wmse

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
        "--nthread", type=int, default=16,
        help="Threads per XGBoost model (default: 16; avoid using all cores on NUMA nodes)",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Parallel Optuna trials (default: 1). Safe because XGBoost releases the GIL.",
    )
    parser.add_argument("--tsv", type=str, default=None)
    parser.add_argument("--zarr_base", type=str, default=None)
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Skip full evaluation of best trial after tuning",
    )
    args = parser.parse_args(argv)

    tsv_path = args.tsv or default_splits_tsv()
    zarr_base = args.zarr_base or default_zarr_base()

    # ---- Create / load study ----
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
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
    nthread = min(args.nthread, cpu_count)
    n_jobs = max(1, args.n_jobs)
    print(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials")
    print(f"  Storage: {args.storage}")
    print(f"  Data: {tsv_path}")
    print(f"  Zarr: {zarr_base}")
    print(f"  CPUs available: {cpu_count}, nthread={nthread}, n_jobs={n_jobs} "
          f"(total threads: {nthread * n_jobs})")

    objective = _build_objective(tsv_path, zarr_base, args.seed, nthread)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # ---- Report ----
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"  Value (mean GVW-MSE): {study.best_trial.value:.6f}")
    print(f"  RMSE ≈ {math.sqrt(study.best_trial.value):.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # Save study summary
    out_dir = str(PROJECT_ROOT / "analysis" / "XGBoost" / "tuning_results")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
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
            n_estimators=1000,
            out_dir=eval_dir,
            tsv_path=tsv_path,
            zarr_base=zarr_base,
        )
        with open(os.path.join(out_dir, "best_cv_summary.json"), "w") as f:
            json.dump(cv_summary, f, indent=2)


if __name__ == "__main__":
    main()
