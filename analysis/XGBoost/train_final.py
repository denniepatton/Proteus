"""Train a final XGBoost model on ALL BTS data with best HPs.

All samples contribute to both training and validation.  A deterministic
10% gene-level row split (shuffled by seed) provides the early-stopping
signal.  Gene-variance weights are computed from 100% of the data
*before* the split so both train and val use identical weights.

Usage
-----
    PYTHONPATH=analysis python -m XGBoost.train_final [OPTIONS]

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
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
from .train import EARLY_STOPPING_ROUNDS

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "analysis" / "XGBoost" / "trained_models" / "final")
DEFAULT_PARAMS_JSON = str(
    PROJECT_ROOT / "analysis" / "XGBoost" / "tuning_results" / "best_params.json"
)

# Cemented HPs from Optuna study (200 trials, best = trial 186).
# These serve as the hard-coded fallback so the final model can be
# reproduced without re-reading the Optuna DB or best_params.json.
CEMENTED_PARAMS: Dict[str, Any] = {
    "max_depth": 9,
    "learning_rate": 0.140353323491014,
    "min_child_weight": 11.225050344590567,
    "subsample": 0.9478298350326214,
    "colsample_bytree": 0.9250965537025436,
    "reg_alpha": 2.6042291082394298e-05,
    "reg_lambda": 3.5824108347711833e-07,
    "gamma": 0.0030623006705801535,
}


def train_final_model(
    xgb_params: Dict[str, Any],
    n_estimators: int,
    out_dir: str,
    tsv_path: Optional[str] = None,
    zarr_base: Optional[str] = None,
    val_frac: float = 0.10,
    seed: int = 42,
) -> None:
    """Train a single XGBoost regressor on all BTS data and save.

    A deterministic gene-level row split (val_frac of rows) is used for
    early stopping.  Gene-variance weights are computed from **all** data
    before the split.
    """
    tsv_path = tsv_path or default_splits_tsv()
    zarr_base = zarr_base or default_zarr_base()
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load all data (no fold exclusion) ----
    all_zarr = collect_zarr_paths(tsv_path, zarr_base)
    print(f"Loading {len(all_zarr)} zarr stores (all folds)")

    X, y, gene_names, sample_ids = load_dataset(all_zarr)
    n_total = len(y)

    # Variance weights from 100% of data (before split)
    _, clipped_weights = compute_gene_variances(all_zarr)
    w = gene_weight_vector(gene_names, clipped_weights, default=0.1)

    # ---- Deterministic gene-level row split ----
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], weight=w[train_idx])
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx], weight=w[val_idx])

    print(f"Total rows: {n_total:,}")
    print(f"  Train rows: {len(train_idx):,} ({100*(1-val_frac):.0f}%)")
    print(f"  Val rows:   {len(val_idx):,} ({100*val_frac:.0f}%) — for early stopping only")

    # ---- Train with early stopping on val; keep best checkpoint ----
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        **xgb_params,
    }
    print(f"Training model (up to {n_estimators} rounds, "
          f"early_stopping_rounds={EARLY_STOPPING_ROUNDS})...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )

    best_iter = int(model.best_iteration)
    print(f"\nBest iteration: {best_iter} "
          f"(early stopping patience={EARLY_STOPPING_ROUNDS})")
    model.save_model(os.path.join(out_dir, "model.ubj"))

    # ---- Save training metadata ----
    meta = {
        "xgb_params": {k: v for k, v in xgb_params.items() if k != "nthread"},
        "n_estimators_max": n_estimators,
        "best_iteration": best_iter,
        "val_frac": val_frac,
        "seed": seed,
        "n_samples": len(set(sample_ids.tolist())),
        "n_rows_total": n_total,
        "n_rows_train": len(train_idx),
        "n_rows_val": len(val_idx),
        "n_features": int(X.shape[1]),
    }
    with open(os.path.join(out_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFinal model saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train final XGBoost on all BTS data with best HPs",
    )
    parser.add_argument(
        "--params_json", type=str, default=None,
        help="Path to best_params.json (default: use cemented HPs)",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=10000,
        help="Max boosting rounds (early stopping decides actual count)",
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.10,
        help="Fraction of gene-level rows for early-stopping validation (default: 0.10)",
    )
    parser.add_argument(
        "--nthread", type=int, default=None,
        help="Threads for XGBoost (default: use all available CPUs)",
    )
    parser.add_argument("--tsv", type=str, default=None)
    parser.add_argument("--zarr_base", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # ---- Resolve HPs: JSON override or cemented defaults ----
    if args.params_json is not None:
        with open(args.params_json, "r") as f:
            summary = json.load(f)
        best_params = summary["best_params"]
        print(f"Loaded HPs from {args.params_json}")
    else:
        best_params = CEMENTED_PARAMS
        print("Using cemented HPs (Optuna trial 186)")

    cpu_count = os.cpu_count() or 1
    if args.nthread is None or args.nthread <= 0:
        nthread = cpu_count
    else:
        nthread = min(args.nthread, cpu_count)
    xgb_params: Dict[str, Any] = {
        **best_params,
        "tree_method": "hist",
        "seed": args.seed,
        "nthread": nthread,
        "verbosity": 0,
    }

    n_est = args.n_estimators

    train_final_model(
        xgb_params=xgb_params,
        n_estimators=n_est,
        out_dir=args.out_dir,
        tsv_path=args.tsv,
        zarr_base=args.zarr_base,
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
