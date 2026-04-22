"""
interpretation.py
v3.0.0
Author: Alexander Netzley, anetzley@fredhutch.org
Updates: Robert Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

Command-line wrapper for model interpretation analysis on trained BioDVN
models.  Supports kernel visualisation, Captum-based attribution analysis,
signal–sequence interaction, ReductionBlock activations, latent-space
UMAP, and zero-component (hurdle) attribution.
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path
from typing import List

import torch

# ---------------------------------------------------------------------------
# Resolve project root so imports work regardless of CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from interpretation.interpretation_utils import (
    attribution_analysis,
    get_data,
    kernel_visualization,
    load_model,
)
from proteus_utils.compile_sample_data import (
    DEFAULT_ZARR_BASE,
    get_zarr_output_dir,
)

def _resolve_zarr_paths_from_tsv(
    splits_tsv: str, zarr_base: str, label: str,
) -> List[str]:
    """Read a sample-splits TSV and return existing .zarr store paths."""
    paths: List[str] = []
    seen: set = set()
    with open(splits_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cohort = row['Cohort'].strip()
            sample = row['Sample'].strip()
            sample_name = sample
            if sample.startswith(f'{cohort}_'):
                sample_name = sample[len(cohort) + 1:]
            zarr_path = os.path.join(
                zarr_base, cohort, 'zarr_files', f'{sample_name}.zarr',
            )
            if zarr_path not in seen and os.path.isdir(zarr_path):
                paths.append(zarr_path)
                seen.add(zarr_path)
    if not paths:
        raise FileNotFoundError(
            f'No zarr stores found for {label} in {zarr_base}.  '
            f'Run --process_data first or check PROTEUS_ZARR_BASE.'
        )
    return paths


def _resolve_zarr_paths(use_set: str, zarr_base: str) -> List[str]:
    """Resolve a dataset name to a list of existing .zarr store paths.

    Supports two modes:
    - **Composite dataset** (e.g. BTS, PAN-FT): if a matching
      ``data/{use_set}_sample-splits.tsv`` exists, reads it to gather
      zarr paths from all cohorts listed in the TSV.
    - **Single cohort** (e.g. PC-PDX, CRPC-RA): globs ``*.zarr`` from
      the cohort's zarr directory.
    """
    # Check for a sample-splits TSV (covers BTS, PAN-FT, and any future
    # composite datasets).
    splits_tsv = _PROJECT_ROOT / 'data' / f'{use_set}_sample-splits.tsv'
    if splits_tsv.exists():
        return _resolve_zarr_paths_from_tsv(str(splits_tsv), zarr_base, use_set)

    # Single-cohort: glob the dataset's zarr directory
    zarr_dir = get_zarr_output_dir(use_set, zarr_base)
    paths = sorted(glob.glob(os.path.join(zarr_dir, '*.zarr')))
    if not paths:
        raise FileNotFoundError(
            f'No zarr stores found in {zarr_dir}.  '
            f'Run --process_data --use_set {use_set} first.'
        )
    return paths


# ---------------------------------------------------------------------------
# Interpretation entry point
# ---------------------------------------------------------------------------

def interpretation_analysis(
    model: torch.nn.Module,
    data: dict,
    save_filepath: str,
    background_size: int = 50_000,
    test_size: int = 5_000,
    kernel_viz: bool = True,
    attribution: bool = True,
    method: str = 'integrated_gradients',
    n_steps: int = 50,
    batch_size: int = 32,
    run_interaction: bool = True,
    run_reduction: bool = True,
    run_umap: bool = True,
) -> None:
    """Run interpretation analyses on a trained BioDVN model.

    Args:
        model: Trained BioDVN in eval mode on the target device.
        data: Dict of numpy arrays from :func:`load_interpretation_data` or
            :func:`get_data`.
        save_filepath: Root directory for output figures.
        background_size: Genes for the attribution baseline set.
        test_size: Genes for the attribution test set.
        kernel_viz: Whether to run kernel/motif visualisation.
        attribution: Whether to run Captum attribution analysis.
        method: ``'integrated_gradients'`` or ``'gradient_shap'``.
        n_steps: Interpolation steps (IG) or samples (GradientShap).
        batch_size: Internal batch size for attribution computation.
        run_interaction: Whether to run signal–sequence interaction analysis.
        run_reduction: Whether to run ReductionBlock activation analysis.
        run_umap: Whether to run latent-space UMAP visualization.
    """
    print('Performing interpretation analysis...')

    if kernel_viz:
        print('Performing kernel visualization...')
        kernel_visualization(model, data, save_filepath)

    if attribution:
        print('Performing attribution analysis...')
        attribution_analysis(
            model, data,
            background_size=background_size,
            test_size=test_size,
            method=method,
            n_steps=n_steps,
            batch_size=batch_size,
            save_filepath=save_filepath,
            run_interaction=run_interaction,
            run_reduction=run_reduction,
            run_umap=run_umap,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Interpretation analysis for trained BioDVN models',
    )
    parser.add_argument(
        '--model_dir', type=str, required=True,
        help='Directory containing final_model.pt and model_config.json',
    )
    # Data source: either --use_set or --zarr_paths (at least one required)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--use_set', type=str, default=None,
        help=('Dataset name to auto-resolve zarr paths '
              '(e.g. BTS, PC-PDX, PHT, SCLC-PDX)'),
    )
    data_group.add_argument(
        '--zarr_paths', type=str, nargs='+', default=None,
        help='Explicit paths to sample .zarr stores',
    )
    parser.add_argument(
        '--zarr_base', type=str, default=None,
        help=('Base directory for zarr stores (default: PROTEUS_ZARR_BASE '
              'or paths.yaml). Only used with --use_set.'),
    )
    parser.add_argument(
        '--kernel_viz', action='store_true', default=True,
        help='Run kernel/motif visualization (default: True)',
    )
    parser.add_argument(
        '--no_kernel_viz', dest='kernel_viz', action='store_false',
        help='Skip kernel/motif visualization',
    )
    parser.add_argument(
        '--attribution', action='store_true', default=True,
        help='Run attribution analysis (default: True)',
    )
    parser.add_argument(
        '--no_attribution', dest='attribution', action='store_false',
        help='Skip attribution analysis',
    )
    parser.add_argument(
        '--method', type=str, default='integrated_gradients',
        choices=['integrated_gradients', 'gradient_shap'],
        help='Attribution method (default: integrated_gradients)',
    )
    parser.add_argument(
        '--background_size', type=int, default=50_000,
        help='Background set size for attribution (default: 50000)',
    )
    parser.add_argument(
        '--test_size', type=int, default=5_000,
        help='Test set size for attribution (default: 5000)',
    )
    parser.add_argument(
        '--n_steps', type=int, default=50,
        help='Interpolation steps (IG) or samples (GradientShap) (default: 50)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for attribution computation (default: 32)',
    )
    parser.add_argument(
        '--max_genes', type=int, default=200_000,
        help='Max total genes to load (distributed across stores). '
             'Default 200k ≈ 20 GB RAM. Sufficient for all analyses.',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for figures (default: <model_dir>/interpretation/)',
    )
    # ── New analysis toggles ──
    parser.add_argument(
        '--no_interaction', dest='run_interaction', action='store_false',
        help='Skip signal–sequence interaction analysis',
    )
    parser.add_argument(
        '--no_reduction', dest='run_reduction', action='store_false',
        help='Skip ReductionBlock activation analysis',
    )
    parser.add_argument(
        '--no_umap', dest='run_umap', action='store_false',
        help='Skip latent-space UMAP visualization',
    )
    parser.set_defaults(
        run_interaction=True, run_reduction=True,
        run_umap=True,
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Hardware report
    # ------------------------------------------------------------------
    n_cpus = os.cpu_count() or 1
    torch.set_num_threads(n_cpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Hardware: device={device}, CPUs={n_cpus}, '
          f'PyTorch threads={torch.get_num_threads()}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}, '
              f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('  ⚠ Running on CPU — attribution analysis will be slow. '
              'Use a GPU node for 10-50x speedup.')

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    model_path = model_dir / 'final_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'No checkpoint found at {model_path}')

    model = load_model(str(model_path), device=device)
    print(f'Model loaded from {model_path}')

    # ------------------------------------------------------------------
    # Resolve zarr paths
    # ------------------------------------------------------------------
    if args.use_set is not None:
        zarr_base = args.zarr_base or DEFAULT_ZARR_BASE
        zarr_paths = _resolve_zarr_paths(args.use_set, zarr_base)
        print(f'Resolved --use_set {args.use_set!r} → {len(zarr_paths)} zarr stores')
    else:
        zarr_paths = args.zarr_paths

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    output_dir = args.output_dir or str(model_dir / 'interpretation')
    os.makedirs(output_dir, exist_ok=True)

    data = get_data(
        source=zarr_paths,
        max_genes=args.max_genes,
    )
    print(f'Data loaded: {data["TSS_signals"].shape[0]} genes')

    # ------------------------------------------------------------------
    # Run analyses
    # ------------------------------------------------------------------
    interpretation_analysis(
        model, data,
        save_filepath=output_dir,
        background_size=args.background_size,
        test_size=args.test_size,
        kernel_viz=args.kernel_viz,
        attribution=args.attribution,
        method=args.method,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        run_interaction=args.run_interaction,
        run_reduction=args.run_reduction,
        run_umap=args.run_umap,
    )

    print('Interpretation analysis complete.')


if __name__ == '__main__':
    main()
