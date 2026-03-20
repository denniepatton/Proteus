"""Fine-tune XGBoost via warm-start on PAN-FT cohorts.

Two modes:
  cv         – 5-fold CV producing per-fold predictions (for analysis).
  full-train – Single model on all PAN-FT data (for deployment / inference).

Both modes warm-start from the BTS base model.  PHT samples are
train-only (excluded from validation/metric computation).

Usage
-----
    # 5-fold CV evaluation
    PYTHONPATH=analysis python -m XGBoost.train_finetune --mode cv

    # Full-train final model
    PYTHONPATH=analysis python -m XGBoost.train_finetune --mode full-train

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
from .train import EARLY_STOPPING_ROUNDS, evaluate_fold, _aggregate_cv_metrics

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = str(
    PROJECT_ROOT / "analysis" / "XGBoost" / "trained_models" / "final"
)
DEFAULT_CV_OUT = str(
    PROJECT_ROOT / "results_training" / "xgboost_fine-tune_cv"
)
DEFAULT_FT_MODEL_OUT = str(
    PROJECT_ROOT / "analysis" / "XGBoost" / "trained_models" / "fine-tuned"
)
DEFAULT_PARAMS_JSON = str(
    PROJECT_ROOT / "analysis" / "XGBoost" / "finetune_tuning_results"
    / "best_finetune_params.json"
)
TRAIN_ONLY_COHORTS = ["PHT"]

# Cemented fine-tuning HPs — updated after Optuna study completes.
# These serve as fallback defaults so runs are reproducible without
# re-reading the Optuna DB.
CEMENTED_FT_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.01,
    "n_rounds": 200,
    "subsample": 0.9,
}


# ---------------------------------------------------------------------------
# 5-fold CV with warm-start fine-tuning
# ---------------------------------------------------------------------------

def run_finetune_cv(
    ft_params: Dict[str, Any],
    base_model_path: str,
    out_dir: str,
    tsv_path: Optional[str] = None,
    zarr_base: Optional[str] = None,
    nthread: int = 16,
    folds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run 5-fold fine-tuning CV, save predictions + metrics.

    For each fold:
      1. Load base model.
      2. Continue boosting on PAN-FT training folds.
      3. Evaluate on held-out fold (PHT excluded from val).
      4. Save predictions in the same format as train.py for analysis
         pipeline compatibility.

    Output structure mirrors train.py:
        out_dir/
            CV-fold_1/ validation-fold_1_predictions_tumor.tsv
                       validation-fold_1_metrics.json
                       model.ubj
            ...
            cv_objective.json
    """
    tsv_path = tsv_path or default_panft_splits_tsv()
    zarr_base = zarr_base or default_zarr_base()
    os.makedirs(out_dir, exist_ok=True)

    n_rounds = ft_params.get("n_rounds", 200)
    lr = ft_params.get("learning_rate", 0.01)
    subsample = ft_params.get("subsample", 0.9)

    xgb_params: Dict[str, Any] = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "nthread": nthread,
        "verbosity": 0,
        "learning_rate": lr,
        "subsample": subsample,
    }

    folds = folds or [1, 2, 3, 4, 5]
    all_metrics: List[Dict[str, Any]] = []

    for fold_num in folds:
        print(f"\n{'='*60}")
        print(f"  Fold {fold_num}")
        print(f"{'='*60}")

        # Train: all folds except held-out (PHT always in training)
        train_zarr = collect_zarr_paths(
            tsv_path, zarr_base, exclude_fold=fold_num,
        )
        # Val: held-out fold, excluding PHT
        val_zarr = collect_zarr_paths(
            tsv_path, zarr_base, fold=fold_num,
            exclude_cohorts=TRAIN_ONLY_COHORTS,
        )
        # All PAN-FT zarr for full-study variance weights
        all_zarr = collect_zarr_paths(tsv_path, zarr_base)

        _, clipped_weights = compute_gene_variances(train_zarr)
        raw_weights_full, _ = compute_gene_variances(all_zarr)

        X_train, y_train, genes_train, samples_train = load_dataset(train_zarr)
        X_val, y_val, genes_val, samples_val = load_dataset(val_zarr)

        w_train = gene_weight_vector(genes_train, clipped_weights, default=0.1)
        w_val = gene_weight_vector(genes_val, clipped_weights, default=0.1)

        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

        n_train_samples = len(set(samples_train.tolist()))
        n_val_samples = len(set(samples_val.tolist()))
        print(f"  Train: {n_train_samples} samples, {len(y_train):,} rows")
        print(f"  Val:   {n_val_samples} samples, {len(y_val):,} rows "
              f"(PHT excluded)")
        print(f"  Fine-tuning: {n_rounds} rounds, lr={lr}, "
              f"subsample={subsample}")

        # Warm-start from base model
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=min(EARLY_STOPPING_ROUNDS, n_rounds),
            xgb_model=base_model_path,
            verbose_eval=False,
        )

        best_iter = int(model.best_iteration)
        print(f"  Best iteration: {best_iter}")

        # Evaluate using the existing evaluate_fold infrastructure
        fold_info = {
            "fold_num": fold_num,
            "n_train": len(y_train),
            "n_val": len(y_val),
            "best_iteration": best_iter,
            "X_val": X_val,
            "y_val": y_val,
            "genes_val": genes_val,
            "samples_val": samples_val,
            "raw_weights_full": raw_weights_full,
        }
        metrics, df_pred = evaluate_fold(model, fold_info)

        # Save
        fold_dir = os.path.join(out_dir, f"CV-fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir,
                  f"validation-fold_{fold_num}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        df_pred.to_csv(
            os.path.join(fold_dir,
                         f"validation-fold_{fold_num}_predictions_tumor.tsv"),
            sep="\t",
        )
        model.save_model(os.path.join(fold_dir, "model.ubj"))

        all_metrics.append(metrics)
        t = metrics["tumor"]
        print(
            f"  RMSE: {t['rmse']:.4f}  "
            f"R²: {t['r2']:.4f}  "
            f"GVW-R²: {t['gvw_mean_r2']:.4f}  "
            f"gene_r2: {t['gene_r2_mean']:.4f}  "
            f"sample_r2: {t['sample_r2_mean']:.4f}"
        )

    # Aggregate
    summary = _aggregate_cv_metrics(all_metrics)
    summary["fine_tune_params"] = {
        "learning_rate": lr,
        "n_rounds": n_rounds,
        "subsample": subsample,
        "base_model": base_model_path,
    }
    with open(os.path.join(out_dir, "cv_objective.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  Fine-Tune CV Summary")
    print(f"{'='*60}")
    print(f"  Mean RMSE:      {summary['tumor_rmse_mean']:.4f} "
          f"± {summary['tumor_rmse_std']:.4f}")
    print(f"  Mean R²:        {summary['tumor_r2_mean']:.4f} "
          f"± {summary['tumor_r2_std']:.4f}")
    print(f"  Mean GVW-R²:    {summary['tumor_gvw_mean_r2_mean']:.4f} "
          f"± {summary['tumor_gvw_mean_r2_std']:.4f}")
    print(f"  Mean gene_r2:   {summary['tumor_gene_r2_mean_mean']:.4f}")
    print(f"  Mean sample_r2: {summary['tumor_sample_r2_mean_mean']:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Full-train: all PAN-FT data → single deployable model
# ---------------------------------------------------------------------------

def run_finetune_full_train(
    ft_params: Dict[str, Any],
    base_model_path: str,
    out_dir: str,
    tsv_path: Optional[str] = None,
    zarr_base: Optional[str] = None,
    nthread: int = 16,
    val_frac: float = 0.10,
    seed: int = 42,
) -> None:
    """Fine-tune a single model on ALL PAN-FT data for deployment.

    Uses a deterministic 10% gene-level row split for early stopping
    (same strategy as train_final.py).
    """
    tsv_path = tsv_path or default_panft_splits_tsv()
    zarr_base = zarr_base or default_zarr_base()
    os.makedirs(out_dir, exist_ok=True)

    n_rounds = ft_params.get("n_rounds", 200)
    lr = ft_params.get("learning_rate", 0.01)
    subsample = ft_params.get("subsample", 0.9)

    xgb_params: Dict[str, Any] = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "nthread": nthread,
        "verbosity": 0,
        "learning_rate": lr,
        "subsample": subsample,
    }

    # Load all PAN-FT data
    all_zarr = collect_zarr_paths(tsv_path, zarr_base)
    print(f"Loading {len(all_zarr)} zarr stores (all PAN-FT samples)")

    X, y, gene_names, sample_ids = load_dataset(all_zarr)
    n_total = len(y)

    _, clipped_weights = compute_gene_variances(all_zarr)
    w = gene_weight_vector(gene_names, clipped_weights, default=0.1)

    # Deterministic gene-level row split for early stopping
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], weight=w[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx], weight=w[val_idx])

    n_samples = len(set(sample_ids.tolist()))
    print(f"Total: {n_samples} samples, {n_total:,} rows")
    print(f"  Train rows: {len(train_idx):,} ({100*(1-val_frac):.0f}%)")
    print(f"  Val rows:   {len(val_idx):,} ({100*val_frac:.0f}%) "
          f"— for early stopping only")
    print(f"  Fine-tuning: up to {n_rounds} rounds, lr={lr}, "
          f"subsample={subsample}")

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=min(EARLY_STOPPING_ROUNDS, n_rounds),
        xgb_model=base_model_path,
        verbose_eval=50,
    )

    best_iter = int(model.best_iteration)
    print(f"\nBest iteration: {best_iter}")
    model.save_model(os.path.join(out_dir, "model.ubj"))

    meta = {
        "mode": "fine-tune-full-train",
        "base_model": base_model_path,
        "fine_tune_params": {
            "learning_rate": lr,
            "n_rounds": n_rounds,
            "subsample": subsample,
        },
        "best_iteration": best_iter,
        "val_frac": val_frac,
        "seed": seed,
        "n_samples": n_samples,
        "n_rows_total": n_total,
        "n_rows_train": len(train_idx),
        "n_rows_val": len(val_idx),
        "n_features": int(X.shape[1]),
    }
    with open(os.path.join(out_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFine-tuned model saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_ft_params(params_json: Optional[str]) -> Dict[str, Any]:
    """Load fine-tuning HPs from JSON or fall back to cemented defaults."""
    if params_json is not None and os.path.isfile(params_json):
        with open(params_json, "r") as f:
            summary = json.load(f)
        params = summary["best_params"]
        print(f"Loaded fine-tune HPs from {params_json}")
        return params
    print("Using cemented fine-tune HPs (defaults)")
    return dict(CEMENTED_FT_PARAMS)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="XGBoost fine-tuning on PAN-FT (CV or full-train)",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["cv", "full-train"],
        help="'cv' for 5-fold CV evaluation, 'full-train' for final model.",
    )
    parser.add_argument(
        "--base_model_dir", type=str, default=DEFAULT_BASE_MODEL,
        help="Directory containing base model.ubj",
    )
    parser.add_argument(
        "--params_json", type=str, default=DEFAULT_PARAMS_JSON,
        help="Path to best_finetune_params.json (or use cemented defaults)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (defaults depend on mode)",
    )
    parser.add_argument(
        "--nthread", type=int, default=16,
        help="Threads per XGBoost model",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.10,
        help="Row fraction for early-stopping val in full-train mode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsv", type=str, default=None)
    parser.add_argument("--zarr_base", type=str, default=None)
    parser.add_argument(
        "--folds", type=int, nargs="+", default=None,
        help="Specific folds for CV mode (default: all 5)",
    )
    # Allow direct HP overrides (take precedence over params_json)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--n_rounds", type=int, default=None)
    parser.add_argument("--subsample", type=float, default=None)

    args = parser.parse_args(argv)

    base_model_path = os.path.join(args.base_model_dir, "model.ubj")
    if not os.path.isfile(base_model_path):
        raise FileNotFoundError(
            f"Base model not found: {base_model_path}\n"
            "Train the BTS base model first with train_final.py."
        )

    nthread = min(args.nthread, os.cpu_count() or 1)

    # Resolve fine-tuning HPs
    ft_params = _load_ft_params(args.params_json)
    if args.learning_rate is not None:
        ft_params["learning_rate"] = args.learning_rate
    if args.n_rounds is not None:
        ft_params["n_rounds"] = args.n_rounds
    if args.subsample is not None:
        ft_params["subsample"] = args.subsample

    print(f"Fine-tune HPs: {ft_params}")
    print(f"Base model:    {base_model_path}")
    print()

    if args.mode == "cv":
        out_dir = args.out_dir or DEFAULT_CV_OUT
        run_finetune_cv(
            ft_params=ft_params,
            base_model_path=base_model_path,
            out_dir=out_dir,
            tsv_path=args.tsv,
            zarr_base=args.zarr_base,
            nthread=nthread,
            folds=args.folds,
        )
    elif args.mode == "full-train":
        out_dir = args.out_dir or DEFAULT_FT_MODEL_OUT
        run_finetune_full_train(
            ft_params=ft_params,
            base_model_path=base_model_path,
            out_dir=out_dir,
            tsv_path=args.tsv,
            zarr_base=args.zarr_base,
            nthread=nthread,
            val_frac=args.val_frac,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
