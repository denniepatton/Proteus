"""Warm-start fine-tuning for XGBoost baseline.

Loads a pre-trained XGBoost model and continues boosting on new data.
XGBoost natively supports warm-start via the ``xgb_model`` parameter
in ``xgb.train()``.

Usage
-----
    python -m analysis.XGBoost.finetune \
        --base_model_dir analysis/XGBoost/trained_models/final \
        --use_set CRPC-WCDT \
        --out_dir analysis/XGBoost/finetuned_models/CRPC-WCDT

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xgboost as xgb

from .data_utils import (
    PROJECT_ROOT,
    compute_gene_variances,
    default_zarr_base,
    gene_weight_vector,
    load_dataset,
)
from .inference import _discover_zarr_paths

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_FINETUNE_ROUNDS = 200
DEFAULT_LEARNING_RATE = 0.01  # conservative for fine-tuning


def finetune(
    base_model_dir: str,
    zarr_paths: List[str],
    out_dir: str,
    n_rounds: int = DEFAULT_FINETUNE_ROUNDS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> None:
    """Fine-tune model on new data via warm-start.

    Loads model from ``base_model_dir``, continues boosting on data from
    ``zarr_paths``, and saves updated model to ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load data ----
    X, y, gene_names, sample_ids = load_dataset(zarr_paths)
    _, clipped_weights = compute_gene_variances(zarr_paths)
    w = gene_weight_vector(gene_names, clipped_weights, default=0.1)
    dtrain = xgb.DMatrix(X, label=y, weight=w)

    # ---- Shared training params (lowered lr for fine-tuning) ----
    ft_params: Dict[str, Any] = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "learning_rate": learning_rate,
        "nthread": 0,
        "verbosity": 0,
    }

    # ---- Fine-tune model ----
    print(f"Fine-tuning model ({n_rounds} rounds, lr={learning_rate})")
    base_model_path = os.path.join(base_model_dir, "model.ubj")
    model = xgb.train(
        ft_params,
        dtrain,
        num_boost_round=n_rounds,
        xgb_model=base_model_path,
        verbose_eval=False,
    )
    model.save_model(os.path.join(out_dir, "model.ubj"))

    # ---- Save fine-tuning metadata ----
    meta = {
        "base_model_dir": str(base_model_dir),
        "n_finetune_rounds": n_rounds,
        "learning_rate": learning_rate,
        "n_samples": len(set(sample_ids.tolist())),
        "n_genes_total": len(y),
    }
    with open(os.path.join(out_dir, "finetune_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Fine-tuned model saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune XGBoost baseline on new data (warm-start)",
    )
    parser.add_argument(
        "--base_model_dir", type=str, required=True,
        help="Directory containing pre-trained model.ubj",
    )
    parser.add_argument(
        "--use_set", type=str, required=True,
        help="Dataset name for fine-tuning data",
    )
    parser.add_argument("--zarr_base", type=str, default=None)
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for fine-tuned models",
    )
    parser.add_argument(
        "--n_rounds", type=int, default=DEFAULT_FINETUNE_ROUNDS,
        help=f"Number of additional boosting rounds (default: {DEFAULT_FINETUNE_ROUNDS})",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate for fine-tuning (default: {DEFAULT_LEARNING_RATE})",
    )
    args = parser.parse_args(argv)

    zarr_base = args.zarr_base or default_zarr_base()
    zarr_paths = _discover_zarr_paths(args.use_set, zarr_base)
    print(f"Found {len(zarr_paths)} zarr stores for fine-tuning on {args.use_set}")

    out_dir = args.out_dir or str(
        PROJECT_ROOT / "analysis" / "XGBoost" / "finetuned_models" / args.use_set
    )

    finetune(
        base_model_dir=args.base_model_dir,
        zarr_paths=zarr_paths,
        out_dir=out_dir,
        n_rounds=args.n_rounds,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
