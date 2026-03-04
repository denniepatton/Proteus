"""
interpretation.py
v2.0.0
Author: Alexander Netzley, anetzley@fredhutch.org
Updates: Robert Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

Command-line wrapper for model interpretation analysis on trained BioDVN
models.  Supports kernel visualisation and Captum-based attribution analysis.
"""

import argparse
import os
import sys
from pathlib import Path

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
    batch_size: int = 256,
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
        help='Directory containing best_model.pt and model_config.json',
    )
    parser.add_argument(
        '--zarr_paths', type=str, nargs='+', required=True,
        help='One or more paths to sample .zarr stores',
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
        '--batch_size', type=int, default=256,
        help='Batch size for attribution computation (default: 256)',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for figures (default: <model_dir>/interpretation/)',
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    model_path = model_dir / 'best_model.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'No checkpoint found at {model_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(str(model_path), device=device)
    print(f'Model loaded from {model_path}  (device={device})')

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    output_dir = args.output_dir or str(model_dir / 'interpretation')
    os.makedirs(output_dir, exist_ok=True)

    npz_cache = os.path.join(output_dir, 'cached_data.npz')
    data = get_data(
        zarr_paths=args.zarr_paths,
        save_filepath=npz_cache,
        regenerate_data=False,
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
    )

    print('Interpretation analysis complete.')


if __name__ == '__main__':
    main()
