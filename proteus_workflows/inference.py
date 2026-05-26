"""Inference workflows extracted from the CLI entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

from config.runtime_paths import normalize_path
from models import BioDVNConfig, channel_mode_for_model, get_model_instance
from proteus_utils import create_inference_loader, get_zarr_output_dir


DEFAULT_SHIPPED_MODEL_DIRS: Dict[str, str] = {
    'BioDVN': 'trained_models/proteus_plca-model/full_train',
    'BioDVN_NoSeq': 'trained_models/proteus_plca-model_no-seq/full_train',
    'BioDVN_OnlySeq': 'trained_models/proteus_plca-model_only-seq/full_train',
}


def run_model(
    model_path: str,
    sample_files: List[str],
    out_dir: str,
    model_name: str = 'BioDVN',
):
    """Run a saved model on a list of sample Zarr stores."""
    from models.model_utils import mixture_quantile

    print('Running model for inference...')
    os.makedirs(out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required (GPU-only). No CPU fallback is supported.')
    device = torch.device('cuda')

    model_dir = str(Path(model_path).parent)
    cfg_path = os.path.join(model_dir, 'model_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as handle:
            cfg_raw = json.load(handle)
        allowed = set(BioDVNConfig.__dataclass_fields__.keys())
        cfg = {key: value for key, value in cfg_raw.items() if key in allowed}
        cfg['channel_mode'] = channel_mode_for_model(model_name)
        model_config = BioDVNConfig(**cfg)
        print(f'Loaded model config from: {cfg_path}')
    else:
        model_config = BioDVNConfig(channel_mode=channel_mode_for_model(model_name))
        print('WARNING: No model_config.json found; using BioDVNConfig defaults for inference.')

    model_class = get_model_instance(model_name)
    model = model_class(model_config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    pred_cols = [
        'prediction_mean',
        'ci_q5',
        'ci_q95',
        'pi_delta',
        'pi_gamma',
        'gamma_beta',
        'pi_normal',
        'normal_mean',
        'normal_stdev',
    ]
    tumor_rows: Dict[str, list] = {key: [] for key in pred_cols}
    background_rows: Dict[str, list] = {key: [] for key in pred_cols}
    all_genes: List[str] = []
    all_tumor_samples: List[str] = []
    all_background_samples: List[str] = []

    n_samples = len(sample_files)
    for idx, sample_file in enumerate(sample_files, 1):
        sample = Path(sample_file).stem.replace('.zarr', '')
        print(f'[{idx}/{n_samples}] Running inference on {sample}...')

        loader, gene_names = create_inference_loader(sample_file)
        with torch.no_grad():
            for batch in loader:
                batch_device = {key: value.to(device) for key, value in batch.items()}
                predictions, _ = model(batch_device, return_internal_loss=False)

                for rows, head in ((tumor_rows, 'tumor'), (background_rows, 'background')):
                    mdn = predictions[f'{head}_mdn']
                    q_vals = mixture_quantile(mdn, quantiles=(0.05, 0.95))
                    pi = mdn['log_pi'].exp().cpu().numpy()

                    rows['prediction_mean'].extend(predictions[f'{head}_exp'].cpu().numpy().flatten())
                    rows['ci_q5'].extend(q_vals[:, 0].cpu().numpy().flatten())
                    rows['ci_q95'].extend(q_vals[:, 1].cpu().numpy().flatten())
                    rows['pi_delta'].extend(pi[:, 0].flatten())
                    rows['pi_gamma'].extend(pi[:, 1].flatten())
                    rows['gamma_beta'].extend(mdn['gamma_beta'].cpu().numpy().flatten())
                    rows['pi_normal'].extend(pi[:, 2].flatten())
                    rows['normal_mean'].extend(mdn['normal_mu'].cpu().numpy().flatten())
                    rows['normal_stdev'].extend(mdn['normal_sigma'].cpu().numpy().flatten())

        n_genes = len(gene_names)
        all_genes.extend([str(gene) for gene in gene_names])
        all_tumor_samples.extend([sample] * n_genes)
        all_background_samples.extend([sample] * n_genes)
        print(f'[{idx}/{n_samples}] Done  ({n_genes} genes).')

    tumor_rows['sample'] = all_tumor_samples
    background_rows['sample'] = all_background_samples

    df_tumor = pd.DataFrame(tumor_rows, index=all_genes)
    df_background = pd.DataFrame(background_rows, index=all_genes)

    df_tumor.to_csv(os.path.join(out_dir, 'predictions_tumor.tsv'), sep='\t')
    df_background.to_csv(os.path.join(out_dir, 'predictions_background.tsv'), sep='\t')

    print('Finished running model on all samples.')


def run_saved_model_on_dataset(
    model_name: str,
    dataset_name: str,
    zarr_base: str,
    experiment_output_dir: Optional[str] = None,
) -> None:
    """Run either a supplied experiment checkpoint or a shipped checkpoint on one dataset."""
    if experiment_output_dir is not None:
        model_dir = os.path.join(experiment_output_dir, 'full_train')
    else:
        default_model_dir = DEFAULT_SHIPPED_MODEL_DIRS.get(model_name)
        if default_model_dir is None:
            raise ValueError(
                f"No default shipped checkpoint is configured for model_name='{model_name}'. "
                f'Available: {sorted(DEFAULT_SHIPPED_MODEL_DIRS)}. Provide --experiment to select a model explicitly.'
            )
        model_dir = normalize_path(default_model_dir)

    model_path = os.path.join(model_dir, 'final_model.pt')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f'Final full-train model not found at: {model_path}\n'
            'Run --run_mode full-train first, or provide --experiment to point to a different model directory.'
        )

    zarr_dir = get_zarr_output_dir(dataset_name, zarr_base)
    if not os.path.isdir(zarr_dir) or not any(name.endswith('.zarr') for name in os.listdir(zarr_dir)):
        raise FileNotFoundError(
            f"No Zarr stores found for '{dataset_name}' in: {zarr_dir}\n"
            'Run --process_data for that dataset, or point --zarr_output_dir to the correct base.'
        )

    sample_files = sorted(
        os.path.join(zarr_dir, name)
        for name in os.listdir(zarr_dir)
        if name.endswith('.zarr')
    )
    output_base = normalize_path(os.path.join('results_holdout', model_name, dataset_name))
    run_model(model_path, sample_files, output_base, model_name=model_name)
