"""Optuna HP tuning for XGBoost fine-tuning on PAN-FT cohorts.

Warm-starts from the BTS base model and searches over fine-tuning HPs
(learning_rate, n_rounds, subsample) using 5-fold CV on PAN-FT splits.
PHT samples are train-only (excluded from validation metrics), mirroring
the Proteus fine-tuning convention.

Usage
-----
    PYTHONPATH=analysis python -m XGBoost.tune_finetune [OPTIONS]

See ``--help`` for options.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    default_panft_splits_tsv,
    default_zarr_base,
    gene_weight_vector,
    load_dataset,
)
from .train import EARLY_STOPPING_ROUNDS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = str(
    PROJECT_ROOT / "analysis" / "XGBoost" / "trained_models" / "final"
)
DEFAULT_N_TRIALS = 100
DEFAULT_STORAGE = f"sqlite:///{PROJECT_ROOT / 'analysis' / 'XGBoost' / 'optuna_xgb_finetune.db'}"
DEFAULT_STUDY_NAME = "xgboost_finetune"
FOLDS = [1, 2, 3, 4, 5]
TRAIN_ONLY_COHORTS = ["PHT"]


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_objective(
    tsv_path: str,
    zarr_base: str,
    base_model_path: str,
    nthread: int,
):
    """Return an Optuna objective closure for fine-tuning HP search.

    Each trial warm-starts from the base model, continues boosting on
    the training folds of PAN-FT, and evaluates on the held-out fold.
    PHT samples are excluded from validation (train-only cohort).
    """
    # Pre-load fold data once
    fold_data: Dict[int, Dict[str, Any]] = {}
    for fold_num in FOLDS:
        # Training: all folds except the held-out fold (PHT always included)
        train_zarr = collect_zarr_paths(
            tsv_path, zarr_base, exclude_fold=fold_num,
        )
        # Validation: only the held-out fold, excluding PHT
        val_zarr = collect_zarr_paths(
            tsv_path, zarr_base, fold=fold_num,
            exclude_cohorts=TRAIN_ONLY_COHORTS,
        )
        # Gene-variance weights from training data
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

    print(f"Data pre-loaded for {len(FOLDS)} folds "
          f"(PHT train-only, excluded from val).")
    print(f"  Train DMatrix: {fold_data[1]['dtrain'].num_row():,} rows")
    print(f"  Val DMatrix:   {fold_data[1]['dval'].num_row():,} rows")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.perf_counter()

        # Fine-tuning HPs
        lr = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
        n_rounds = trial.suggest_int("n_rounds", 50, 500, step=50)
        subsample = trial.suggest_float("subsample", 0.7, 1.0)

        params: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "nthread": nthread,
            "verbosity": 0,
            "learning_rate": lr,
            "subsample": subsample,
        }

        fold_scores: List[float] = []
        fold_iters: List[int] = []
        for fold_num in FOLDS:
            t_fold = time.perf_counter()
            fd = fold_data[fold_num]
            model = xgb.train(
                params,
                fd["dtrain"],
                num_boost_round=n_rounds,
                evals=[(fd["dval"], "val")],
                early_stopping_rounds=min(EARLY_STOPPING_ROUNDS, n_rounds),
                xgb_model=base_model_path,
                verbose_eval=False,
            )
            # Weighted MSE on validation (excluding PHT)
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
        print(
            f"  Trial {trial.number:>3d} | "
            f"GVW-MSE: {mean_wmse:.6f} | "
            f"lr={lr:.4f} n_rounds={n_rounds} sub={subsample:.2f} | "
            f"iters: {fold_iters} | "
            f"{elapsed:.1f}s"
        )
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_iters", fold_iters)
        return mean_wmse

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Optuna HP tuning for XGBoost fine-tuning on PAN-FT",
    )
    parser.add_argument("--n_trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--storage", type=str, default=DEFAULT_STORAGE)
    parser.add_argument("--study_name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument(
        "--base_model_dir", type=str, default=DEFAULT_BASE_MODEL,
        help="Directory containing the pre-trained base model.ubj",
    )
    parser.add_argument(
        "--nthread", type=int, default=16,
        help="Threads per XGBoost model",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Parallel Optuna trials",
    )
    parser.add_argument("--tsv", type=str, default=None)
    parser.add_argument("--zarr_base", type=str, default=None)
    args = parser.parse_args(argv)

    tsv_path = args.tsv or default_panft_splits_tsv()
    zarr_base = args.zarr_base or default_zarr_base()
    base_model_path = os.path.join(args.base_model_dir, "model.ubj")

    if not os.path.isfile(base_model_path):
        raise FileNotFoundError(
            f"Base model not found: {base_model_path}\n"
            "Train the BTS base model first with train_final.py."
        )

    # ---- Create / load study ----
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
    )

    cpu_count = os.cpu_count() or 1
    nthread = min(args.nthread, cpu_count)
    n_jobs = max(1, args.n_jobs)
    print(f"Starting Optuna fine-tune study '{args.study_name}' "
          f"with {args.n_trials} trials")
    print(f"  Storage:    {args.storage}")
    print(f"  Splits:     {tsv_path}")
    print(f"  Base model: {base_model_path}")
    print(f"  CPUs: {cpu_count}, nthread={nthread}, n_jobs={n_jobs}")
    print()

    objective = _build_objective(tsv_path, zarr_base, base_model_path, nthread)
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

    # Save best params
    out_dir = str(PROJECT_ROOT / "analysis" / "XGBoost" / "finetune_tuning_results")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "n_trials": len(study.trials),
        "base_model_dir": args.base_model_dir,
    }
    out_path = os.path.join(out_dir, "best_finetune_params.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved best params to {out_path}")


if __name__ == "__main__":
    main()
