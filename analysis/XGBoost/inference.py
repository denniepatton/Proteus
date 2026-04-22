"""Run inference on arbitrary use_sets with a trained XGBoost model.

Loads a saved model and produces per-sample prediction TSVs.

Usage
-----
    PYTHONPATH=analysis python -m XGBoost.inference \
        --model_dir analysis/XGBoost/results_cv/CV-fold_1 \
        --use_set CRPC-WCDT

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .data_utils import (
    PROJECT_ROOT,
    default_zarr_base,
    load_sample_arrays,
)

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def predict_sample(
    zarr_path: str,
    model: xgb.Booster,
) -> pd.DataFrame:
    """Predict expression for all genes in a single sample zarr.

    Returns a DataFrame indexed by gene name with columns:
        sample, prediction
    """
    arrs = load_sample_arrays(zarr_path)
    X = np.concatenate(
        [arrs["TSS_features"], arrs["GB_features"], arrs["ref_scalars"]],
        axis=1,
    ).astype(np.float32)
    gene_names = arrs["gene_names"]
    sample_name = arrs["sample_name"]

    dmat = xgb.DMatrix(X)
    pred = model.predict(dmat)

    return pd.DataFrame(
        {
            "sample": sample_name,
            "prediction": pred.astype(np.float64),
        },
        index=gene_names.astype(str),
    )


def run_inference(
    model_dir: str,
    zarr_paths: List[str],
    out_dir: str,
    nthread: Optional[int] = None,
) -> str:
    """Run inference over a list of zarr samples.

    Loads model from ``model_dir`` and writes a single concatenated
    predictions TSV to ``out_dir/predictions_tumor.tsv``.

    Returns the path to the output file.
    """
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "model.ubj"))
    if nthread is not None:
        model.set_param({"nthread": nthread})

    dfs: List[pd.DataFrame] = []
    for zp in zarr_paths:
        sample_name = Path(zp).stem
        print(f"  Inferring: {sample_name}")
        df = predict_sample(zp, model)
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No valid zarr samples found for inference.")

    df_all = pd.concat(dfs, axis=0)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions_tumor.tsv")
    df_all.to_csv(out_path, sep="\t")
    print(f"Saved {len(df_all)} predictions to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Zarr discovery for use_set
# ---------------------------------------------------------------------------

def _discover_zarr_paths(use_set: str, zarr_base: str) -> List[str]:
    """Find all zarr stores for a use_set under zarr_base."""
    zarr_dir = os.path.join(zarr_base, use_set, "zarr_files")
    if not os.path.isdir(zarr_dir):
        raise FileNotFoundError(
            f"Zarr directory not found: {zarr_dir}\n"
            f"Run Proteus --process_data --use_set {use_set} first."
        )
    paths = sorted(
        os.path.join(zarr_dir, d)
        for d in os.listdir(zarr_dir)
        if d.endswith(".zarr") and os.path.isdir(os.path.join(zarr_dir, d))
    )
    if not paths:
        raise FileNotFoundError(f"No .zarr stores found in {zarr_dir}")
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="XGBoost baseline inference on a use_set",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory containing model.ubj",
    )
    parser.add_argument(
        "--use_set", type=str, required=True,
        help="Dataset name (e.g. CRPC-WCDT). Zarr stores discovered under zarr_base/<use_set>/zarr_files/",
    )
    parser.add_argument("--zarr_base", type=str, default=None)
    parser.add_argument(
        "--nthread", type=int, default=None,
        help="Threads for XGBoost inference (default: use all available CPUs)",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: results_holdout/XGBoost/<use_set>/)",
    )
    args = parser.parse_args(argv)

    zarr_base = args.zarr_base or default_zarr_base()
    zarr_paths = _discover_zarr_paths(args.use_set, zarr_base)
    print(f"Found {len(zarr_paths)} zarr stores for {args.use_set}")

    cpu_count = os.cpu_count() or 1
    if args.nthread is None or args.nthread <= 0:
        nthread = cpu_count
    else:
        nthread = min(args.nthread, cpu_count)
    print(f"Using nthread={nthread}")

    out_dir = args.out_dir or str(
        PROJECT_ROOT / "results_holdout" / "XGBoost" / args.use_set
    )

    run_inference(args.model_dir, zarr_paths, out_dir, nthread=nthread)


if __name__ == "__main__":
    main()
