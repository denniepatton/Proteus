"""Proteus.py

PyTorch implementation of the Proteus pipeline for predicting gene expression from cfDNA.

Primary model inputs (per gene):
    - TSS_signals: (n_genes, 2000, 9) float32
    - TSS_nucleotides: (n_genes, 2000, 4) uint8/float32 one-hot nucleotides
    - TSS_features: (n_genes, 11) float32
    - GB_features: (n_genes, 11) float32
    - ref_scalars: (n_genes, 3) float32

Usage:
    # Generate Zarr data from raw Triton files
    python Proteus.py --process_data --use_set PC-PDX
    
    # Train with cross-validation (requires experiment YAML with training_set TSV)
    python Proteus.py --run_mode train --experiment config/experiments/biodvn_train.yaml
    
    # Train specific fold (for HPC parallelization)
    python Proteus.py --run_mode train --experiment config/experiments/biodvn_train.yaml --test_fold 1
    
    # Full training (all data, no CV holdout)
    python Proteus.py --run_mode full-train --experiment config/experiments/biodvn_train.yaml
    
    # Optuna HP tuning
    python Proteus.py --run_mode tune --experiment config/experiments/biodvn_tune.yaml --n_trials 50
    
    # Run inference on holdout samples (uses trained_models/ by default)
    python Proteus.py --run_mode run --model_name BioDVN --use_set Crompton
    python Proteus.py --run_mode run --model_name BioDVN_NoSeq --use_set Crompton
"""

# Standard library imports
import os
import sys
import csv
import json
import random
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dataclasses import asdict, replace
from typing import Optional, Dict, List, Any, Callable

# Third party imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# cuDNN benchmark mode: let the auto-tuner profile convolution algorithms
# for each tensor shape once, then reuse the fastest.  Combined with
# _seed_everything() (which deterministically seeds weight init, dropout,
# and data ordering), this gives practically reproducible training while
# keeping full GPU throughput.
torch.backends.cudnn.benchmark = True

# Local imports
from models import BioDVN, BioDVNConfig, get_model_instance, channel_mode_for_model
from proteus_utils.trainer import TrainingConfig, Trainer
from proteus_utils import (
    get_samples, get_zarr_output_dir, process_dataset,
    process_sample_to_zarr,
    create_data_loader, create_inference_loader,
    build_loaders_from_splits,
    DEFAULT_ZARR_BASE
)
from proteus_utils.metrics import (
    compute_regression_metrics,
    pearson_r, spearman_r, spearman_r2, rankdata,
    group_spearman_r2, group_pearson_r_named,
    coverage, gvw_r, scatter_stats,
)
from proteus_utils.evaluation import save_fold_predictions
from proteus_utils.data_helpers import (
    zarr_name_for_sample, zarr_path_for_sample,
    resolve_pht_source, invert_source_groups, collect_pdx_names,
    collect_zarr_paths, build_source_map_from_tsv, compute_source_variance,
    get_plca_selection_group_weights, validate_plca_selection_groups,
)
from config.dataset_config import load_dataset_config, list_datasets

# Resolve project-relative paths robustly (do not depend on current working directory)
PROJECT_ROOT = Path(__file__).resolve().parent


def _seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducible model init, dropout, and training.

    Must be called before model construction so that
    ``nn.init.xavier_uniform_`` (and any other weight-init that draws from
    torch's global RNG) is deterministic.  Also seeds Python's ``random``
    and NumPy for any stochastic data-loading helpers that run in the main
    process.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # seeds both CPU and CUDA RNGs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_workers() -> int:
    """Number of DataLoader workers, respecting Slurm/cgroup CPU limits."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        # macOS lacks sched_getaffinity
        return max(1, os.cpu_count() or 1)


def ensure_datasets_exist(
    dataset_names: List[str],
    zarr_base: str,
    raw_data_base: str,
    labels_dir: str,
    ref_path: str,
    n_workers: int = 0
) -> Dict[str, bool]:
    """
    Check that all configured datasets have Zarr stores, regenerate if missing.
    
    Zarr stores live on fast-access temporary storage that may be cleared
    periodically (e.g., every 30 days).
    
    Args:
        dataset_names: List of dataset names to check
        zarr_base: Base directory for Zarr stores
        raw_data_base: Base directory for raw Triton outputs
        labels_dir: Directory containing label files
        ref_path: Path to reference file
        n_workers: Number of parallel workers (0=auto)
        
    Returns:
        Dictionary mapping dataset names to whether they were regenerated
    """
    from functools import partial

    regenerated: Dict[str, bool] = {}

    def _is_present_zarr_store(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        try:
            return any(os.scandir(path))
        except FileNotFoundError:
            return False

    for dataset_name in dataset_names:
        # Load dataset config from YAML
        try:
            dataset_config = load_dataset_config(dataset_name)
        except FileNotFoundError:
            print(f"[Auto-regenerate] Warning: No config for '{dataset_name}', skipping")
            continue

        # Auto-detect n_workers if 0
        actual_workers = n_workers if n_workers > 0 else cpu_count()

        zarr_dir = get_zarr_output_dir(dataset_name, zarr_base)

        # Build absolute paths from config
        tss_dir = os.path.join(raw_data_base, dataset_config.triton_tss)
        gb_dir = os.path.join(raw_data_base, dataset_config.triton_gb)

        # Directory missing or empty -> process whole dataset
        zarr_files: List[str] = []
        if os.path.exists(zarr_dir):
            zarr_files = [f for f in os.listdir(zarr_dir) if f.endswith('.zarr')]

        if (not os.path.exists(zarr_dir)) or (not zarr_files):
            reason = "directory missing" if not os.path.exists(zarr_dir) else "no zarr files"
            print(f"\n[Auto-regenerate] Dataset '{dataset_name}' {reason}, processing...")
            process_dataset(
                dataset=dataset_name,
                tss_dir=tss_dir,
                gb_dir=gb_dir,
                ref_path=ref_path,
                labels_dir=labels_dir,
                tfx_file=dataset_config.tfx_file,
                zarr_base_dir=zarr_base,
                tumor_exp_file=dataset_config.tumor_exp_file,
                background_exp_file=dataset_config.background_exp_file,
                overwrite=False,
                n_workers=actual_workers
            )
            regenerated[dataset_name] = True
            continue

        # Discover expected samples from TSS folder structure
        discovered_samples = get_samples(tss_dir, dataset_name)

        # Restrict to samples in the TFX file (if TFX labels exist)
        if dataset_config.tfx_file is not None:
            tfx_file_path = os.path.join(labels_dir, dataset_config.tfx_file)
            try:
                tfx_df = pd.read_csv(tfx_file_path, sep='\t', index_col='sample')
                tfx_index = [str(i) for i in tfx_df.index.tolist()]

                def _matches_tfx(sample: str) -> bool:
                    if sample in tfx_df.index:
                        return True
                    for idx in tfx_index:
                        if sample in idx or idx in sample:
                            return True
                    return False

                expected_samples = [s for s in discovered_samples if _matches_tfx(s)]
            except (FileNotFoundError, ValueError, UnicodeDecodeError) as e:
                raise ValueError(f"Failed to read TFX file for '{dataset_name}': {e}")
        else:
            # No TFX file — all discovered samples are expected
            expected_samples = discovered_samples

        # Existence-only check (fast): only regenerate missing/empty stores.
        missing_samples: List[str] = []
        for sample in expected_samples:
            store_path = os.path.join(zarr_dir, f'{sample}.zarr')
            if not _is_present_zarr_store(store_path):
                missing_samples.append(sample)

        samples_to_regen = sorted(set(missing_samples))
        if not samples_to_regen:
            regenerated[dataset_name] = False
            continue

        print(
            f"[Auto-regenerate] Regenerating {len(samples_to_regen)} sample(s) "
            f"(missing={len(missing_samples)})"
        )

        # Build label paths for regeneration
        tfx_path = os.path.join(labels_dir, dataset_config.tfx_file) if dataset_config.tfx_file else None
        tumor_exp_path = os.path.join(labels_dir, dataset_config.tumor_exp_file) if dataset_config.tumor_exp_file else None
        background_exp_path = os.path.join(labels_dir, dataset_config.background_exp_file) if dataset_config.background_exp_file else None

        worker_fn = partial(
            process_sample_to_zarr,
            tss_dir=tss_dir,
            gb_dir=gb_dir,
            ref_path=ref_path,
            tfx_path=tfx_path,
            tumor_exp_path=tumor_exp_path,
            background_exp_path=background_exp_path,
            out_dir=zarr_dir,
            overwrite=True,
        )

        if actual_workers <= 1:
            for sample in samples_to_regen:
                worker_fn(sample)
        else:
            with Pool(actual_workers) as pool:
                pool.map(worker_fn, samples_to_regen)
        regenerated[dataset_name] = True
    
    # Summary
    regen_count = sum(regenerated.values())
    if regen_count > 0:
        print(f"\n[Auto-regenerate] Regenerated {regen_count} dataset(s)")
    
    return regenerated


def generate_data(
    dataset: str,
    tss_dir: str,
    gb_dir: str,
    labels_dir: str,
    ref_path: str,
    tfx_file: Optional[str],
    zarr_base_dir: Optional[str] = None,
    tumor_exp_file: Optional[str] = None,
    background_exp_file: Optional[str] = None,
    overwrite: bool = False,
):
    """
    Generate Zarr stores from Triton outputs using the sample-level folder structure.

    Zarr stores are saved to:
        {zarr_base_dir}/{dataset}/zarr_files/SAMPLE.zarr
    
    Args:
        dataset: Dataset name (e.g., 'PC-PDX')
        tss_dir: Absolute path to TSS results directory
        gb_dir: Absolute path to GB results directory
        labels_dir: Directory containing label files
        ref_path: Path to reference file
        tfx_file: Name of TFX file in labels_dir (None for unlabeled datasets)
        zarr_base_dir: Base directory for Zarr output (default: /hpc/temp/...)
        tumor_exp_file: Name of tumor expression file
        background_exp_file: Name of background expression file
        overwrite: Whether to overwrite existing Zarr stores
    """
    print(f"Generating Zarr stores for dataset: {dataset}")
    
    # Auto-detect n_workers
    n_workers = cpu_count()
    print(f"  Workers: {n_workers}")
    
    created_paths = process_dataset(
        dataset=dataset,
        tss_dir=tss_dir,
        gb_dir=gb_dir,
        ref_path=ref_path,
        labels_dir=labels_dir,
        tfx_file=tfx_file,
        zarr_base_dir=zarr_base_dir,
        tumor_exp_file=tumor_exp_file,
        background_exp_file=background_exp_file,
        overwrite=overwrite,
        n_workers=n_workers
    )
    
    if created_paths:
        print(f"\nSuccessfully created {len(created_paths)} Zarr stores.")
    else:
        print("\nNo Zarr stores created. Check input paths and data.")
    
    return created_paths


def train_model_cv(
    model_name: str,
    training_set_tsv: str,
    zarr_base: str,
    test_fold: Optional[int] = None,
    out_dir: Optional[str] = None,
    model_config: Optional[BioDVNConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    seed: int = 42,
    subsample_frac: Optional[float] = None,
    epoch_end_callback: Optional[Callable[..., None]] = None,
    external_stop_check: Optional[Callable[..., bool]] = None,
    train_only_cohorts: Optional[List[str]] = None,
    balance_cohorts: Optional[List[str]] = None,
    source_groups: Optional[Dict[str, List[str]]] = None,
    target_n_per_source: Optional[int] = None,
    steps_per_epoch: int = 4000,
    batch_mode: str = 'multi_source',
    fixed_schedule_oof: bool = False,
):
    """Train a model using cross-validation with deterministic fold assignment.

    Args:
        model_name: Name of the model class (e.g. 'BioDVN', 'BioDVN_NoSeq').
        training_set_tsv: Path to TSV file with Cohort/Sample/Fold columns.
        zarr_base: Base directory for Zarr stores ({zarr_base}/{cohort}/{sample}.zarr).
        test_fold: If set, only train this fold (for HPC parallelization).
        out_dir: Output directory for checkpoints/predictions.
        model_config: Fully-specified BioDVNConfig.
        training_config: Fully-specified TrainingConfig.
        seed: Random seed for reproducibility.
        subsample_frac: Optional fraction (0.0-1.0) of genes for fast sanity checks.
        fixed_schedule_oof: If True, train each fold on a fixed schedule and
            always emit OOF predictions from ``final_model.pt`` at the terminal
            epoch. The held-out fold may still be monitored during training for
            diagnostics and loss curves, but it is never used for checkpoint
            selection or early stopping.

    Raises:
        FileNotFoundError: If training_set_tsv or Zarr stores don't exist.
        ValueError: If TSV format is invalid or no samples in folds.
    """

    if fixed_schedule_oof and (
        epoch_end_callback is not None or external_stop_check is not None
    ):
        raise ValueError(
            'fixed_schedule_oof=True is incompatible with per-epoch validation '
            'callbacks because leak-proof OOF generation must not consume the '
            'held-out fold during training.'
        )

    # Determine folds from TSV
    if not os.path.exists(training_set_tsv):
        raise FileNotFoundError(f"Training set TSV not found: {training_set_tsv}")

    folds_in_tsv: set[int] = set()
    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            folds_in_tsv.add(int(row['Fold']))
    n_folds = len(folds_in_tsv)
    if n_folds == 0:
        raise ValueError(f"No folds found in {training_set_tsv}")
    fold_list = sorted(folds_in_tsv)

    print(f'Training model with {n_folds} folds from {training_set_tsv}...')
    if out_dir is None:
        out_dir = f'results_training/{model_name}/'
    os.makedirs(out_dir, exist_ok=True)

    if model_config is None:
        model_config = BioDVNConfig()
    if training_config is None:
        training_config = TrainingConfig(checkpoint_dir=out_dir)

    # Device setup (GPU-only)
    device = torch.device('cuda')

    from proteus_utils.proteus_utils import (
        compute_gene_weights_from_zarr,
        compute_gene_variances_unclipped,
        compute_gene_mean_std_from_zarr,
    )

    # Build sample-name → cohort mapping for per-cohort validation metrics.
    # Keys use full TSV sample names (with cohort prefix when present)
    # to match the prefixed names emitted by the validation dataset.
    sample_to_cohort: Dict[str, str] = {}
    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cohort = row['Cohort'].strip()
            sample = row['Sample'].strip()
            sample_to_cohort[sample] = cohort

    selection_group_weights = get_plca_selection_group_weights(source_groups)
    if training_config.selection_metric == 'min_weighted_source_crps':
        if source_groups is None:
            raise ValueError(
                "selection_metric='min_weighted_source_crps' requires source_groups"
            )
        validate_plca_selection_groups(source_groups)

    # When source_groups is set, build sample→source mapping and compute
    # per-source gene variance for GVW-CRPS / GVW-r evaluation.
    sample_to_source: Optional[Dict[str, str]] = None
    source_tumor_variance: Optional[Dict[str, Dict[str, float]]] = None
    if source_groups:
        sample_to_source, _source_variance_paths = build_source_map_from_tsv(
            training_set_tsv, source_groups, zarr_base,
        )

    for fold_num in fold_list:
        if test_fold is not None and fold_num != test_fold:
            continue

        # Deterministic seeding *before* model construction so that
        # nn.init.xavier_uniform_ draws are reproducible.  Each fold
        # gets a distinct but deterministic seed.
        _seed_everything(seed + fold_num)

        print(f'\nStarting fold {fold_num}/{n_folds}')
        fold_dir = os.path.join(out_dir, f'CV-fold_{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)

        fold_training_config = replace(training_config, checkpoint_dir=fold_dir)

        # Compute per-gene variance weights from the K-1 training folds
        # (excluding the held-out validation fold).  Clipped variance weights
        # scale the CRPS loss so high-variance genes contribute more.
        print(f'Computing gene variance weights from training folds (excluding fold {fold_num}, excluding PHT)...')
        train_zarr_paths = collect_zarr_paths(
            training_set_tsv, zarr_base, exclude_fold=fold_num,
            exclude_cohorts={'PHT'},
        )
        train_fold_variance_raw = compute_gene_variances_unclipped(train_zarr_paths, label_key='flat/tumor_exp')
        gene_weights = compute_gene_weights_from_zarr(
            train_zarr_paths,
            label_key='flat/tumor_exp',
        )
        train_gene_mean_total, train_gene_std_total = compute_gene_mean_std_from_zarr(
            train_zarr_paths,
            label_key='flat/tumor_exp',
        )
        _raw_vals = list(train_fold_variance_raw.values())
        print(f'  {len(gene_weights)} genes, n={len(train_zarr_paths)} samples — '
              f'raw variance [{min(_raw_vals):.4f}, {max(_raw_vals):.4f}], '
              f'clipped to [0.100, 10.000]')

        train_background_variance: Optional[Dict[str, float]] = compute_gene_variances_unclipped(
            collect_zarr_paths(training_set_tsv, zarr_base, exclude_fold=fold_num),
            label_key='flat/background_exp',
        ) or None

        fold_group_tumor_variance: Optional[Dict[str, Dict[str, float]]] = None
        if source_groups:
            _, fold_source_variance_paths = build_source_map_from_tsv(
                training_set_tsv,
                source_groups,
                zarr_base,
                exclude_fold=fold_num,
            )
            fold_group_tumor_variance = compute_source_variance(
                source_groups,
                fold_source_variance_paths,
                compute_gene_variances_unclipped,
            )

        # Build data loaders from TSV
        print('Building data loaders...')
        if batch_mode == 'multi_source':
            from proteus_utils.proteus_utils import build_multisource_loaders_from_splits
            if source_groups is None:
                raise ValueError(
                    "batch_mode='multi_source' requires source_groups")
            train_loader, val_loader, train_steps, val_steps = build_multisource_loaders_from_splits(
                training_set_tsv=training_set_tsv,
                zarr_base=zarr_base,
                val_fold=fold_num,
                source_groups=source_groups,
                gene_variance=train_fold_variance_raw,
                n_eff_per_source=target_n_per_source or 24,
                steps_per_epoch=steps_per_epoch,
                num_workers=_loader_workers(),
                seed=seed,
                train_only_cohorts=train_only_cohorts,
                gene_weights=gene_weights,
                subsample_frac=subsample_frac,
            )
        else:
            train_loader, val_loader, train_steps, val_steps = build_loaders_from_splits(
                training_set_tsv=training_set_tsv,
                zarr_base=zarr_base,
                val_fold=fold_num,
                num_workers=_loader_workers(),
                seed=seed,
                subsample_frac=subsample_frac,
                gene_weights=gene_weights,
                train_only_cohorts=train_only_cohorts,
                balance_cohorts=balance_cohorts,
                source_groups=source_groups,
                target_n_per_source=target_n_per_source,
            )
        print(f'Training batches: {train_steps}, Validation batches: {val_steps}')
        
        # Create model
        ModelClass = get_model_instance(model_name)
        model = ModelClass(model_config)
        
        # Move model to device BEFORE passing to Trainer
        model = model.to(device)

        # Persist run configuration for reproducibility / analysis.
        try:
            with open(os.path.join(fold_dir, 'model_config.json'), 'w') as f:
                json.dump(asdict(model_config), f, indent=2)
            with open(os.path.join(fold_dir, 'training_config.json'), 'w') as f:
                json.dump(asdict(fold_training_config), f, indent=2)
        except TypeError as e:
            raise TypeError(f'Failed to JSON-serialize configs for fold {fold_num}: {e}')
        
        # Train
        trainer = Trainer(model, fold_training_config, model_config, device)
        trainer.set_srl_stats(train_gene_mean_total, train_gene_std_total)
        trainer.val_gene_weight_map = train_fold_variance_raw
        trainer.val_background_gene_weight_map = train_background_variance
        # Per-source gene variance for epoch-level per-source GVW-CRPS/GVW-r.
        trainer.val_source_gene_weight_map = fold_group_tumor_variance
        # Enable epoch-level per-group CRPS monitoring.  When source_groups
        # is active, group by source (4 groups); otherwise by cohort.
        trainer.val_sample_group_map = sample_to_source or sample_to_cohort or None
        trainer.val_selection_group_weights = selection_group_weights
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=fold_training_config.epochs,
            epoch_end_callback=epoch_end_callback,
            external_stop_check=external_stop_check,
            monitor_only_validation=fixed_schedule_oof,
        )

        # Save training history for plotting / audit.
        try:
            with open(os.path.join(fold_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
        except TypeError as e:
            raise TypeError(f'Failed to JSON-serialize training history for fold {fold_num}: {e}')

        if fixed_schedule_oof:
            final_checkpoint_path = os.path.join(fold_dir, 'final_model.pt')
            ema_backup = None
            if trainer._ema is not None:
                ema_backup = trainer._ema.apply(trainer.model)
            trainer._save_checkpoint(
                fold_training_config.epochs,
                is_best=False,
                filename='final_model.pt',
            )
            if ema_backup is not None:
                trainer._ema.restore(trainer.model, ema_backup)
            trainer.load_checkpoint(final_checkpoint_path)
            print(f'Saved fixed-schedule CV checkpoint to: {final_checkpoint_path}')
        else:
            best_checkpoint_path = os.path.join(fold_dir, 'best_model.pt')
            if os.path.exists(best_checkpoint_path):
                trainer.load_checkpoint(best_checkpoint_path)

        # Prefer the trainer's model reference (may be compiled).
        model_for_eval = trainer.model
        model_for_save = getattr(model_for_eval, '_orig_mod', model_for_eval)
        
        # Save final weights
        torch.save(model_for_save.state_dict(), os.path.join(fold_dir, f'validation-fold_{fold_num}_weights.pt'))
        
        # Generate and save predictions
        save_fold_predictions(
            model_for_eval,
            val_loader,
            fold_dir,
            fold_num,
            device,
            full_tumor_variance=train_fold_variance_raw,
            full_background_variance=train_background_variance,
            sample_to_cohort=sample_to_cohort if sample_to_cohort else None,
            sample_to_source=sample_to_source,
            source_tumor_variance=fold_group_tumor_variance,
            selection_group_weights=selection_group_weights,
        )

        # Per-fold plots
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plot_dir = os.path.join(fold_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)

            # Loss curves
            fig = plt.figure(figsize=(7, 4))
            epochs_axis = np.arange(1, len(history.get('train_loss', [])) + 1)
            val_epochs = np.asarray(history.get('val_epochs', []), dtype=np.int32)

            def _plot_validation_series(series_name: str, label: str) -> None:
                series = np.asarray(history.get(series_name, []), dtype=np.float64)
                if val_epochs.size == 0 or series.size == 0:
                    return
                n_points = min(val_epochs.size, series.size)
                x_vals = val_epochs[:n_points]
                y_vals = series[:n_points]
                finite = np.isfinite(y_vals)
                if not np.any(finite):
                    return
                plt.plot(x_vals[finite], y_vals[finite], marker='o', linewidth=1.5, markersize=3.5, label=label)

            if len(epochs_axis) > 0:
                plt.plot(epochs_axis, history.get('train_loss', []), label='train_loss')
                _plot_validation_series('val_loss', 'val_loss')
                _plot_validation_series('val_tumor_gvw_crps', 'val_tumor_gvw_crps')
                _plot_validation_series('val_weighted_tumor_gvw_crps', 'val_weighted_tumor_gvw_crps')
                plt.xlabel('epoch')
                plt.ylabel('metric')
                plt.title('Training / validation curves')
                plt.legend(frameon=False)
                plt.tight_layout()
                fig.savefig(os.path.join(plot_dir, 'loss_curves.png'), dpi=150)
            plt.close(fig)
        except (ImportError, OSError, ValueError) as e:
            print(f'  Warning: failed to generate fold plots for fold {fold_num}: {e}')
        
        # Tear down DataLoader workers and free GPU memory before the next
        # fold.  With persistent_workers=True the worker processes survive
        # until the DataLoader is GC'd; explicit deletion ensures they stop
        # promptly and don't double I/O pressure on scratch.
        del train_loader, val_loader
        del trainer, model, model_for_eval, model_for_save
        torch.cuda.empty_cache()

        print(f'Finished fold {fold_num}/{n_folds}.')
    
    if test_fold is None:
        print('Finished complete training across all folds.')
        _summarize_cv_results(out_dir)


def _summarize_cv_results(out_dir: str) -> None:
    """Aggregate per-fold prediction TSVs into a deterministic CV metrics summary."""
    fold_metrics: List[Dict[str, float]] = []
    for entry in sorted(os.listdir(out_dir)):
        if not entry.startswith('CV-fold_'):
            continue
        fold_dir = os.path.join(out_dir, entry)
        if not os.path.isdir(fold_dir):
            continue
        # fold number is suffix after underscore
        try:
            fold_num = int(entry.split('_')[-1])
        except ValueError:
            fold_num = -1

        metrics_path = os.path.join(fold_dir, f'validation-fold_{fold_num}_metrics.json')
        fold_metrics_json: Optional[Dict[str, Any]] = None
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    fold_metrics_json = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f'  Warning: could not load {metrics_path}: {e}')
                fold_metrics_json = None

        # Prefer metrics.json (includes GVW-CRPS + R² summaries). Fall back to TSV recomputation.
        if fold_metrics_json is not None:
            t_metrics = dict(fold_metrics_json.get('tumor', {}))
            b_metrics = dict(fold_metrics_json.get('background', {}))
        else:
            tumor_path = os.path.join(fold_dir, f'validation-fold_{fold_num}_predictions_tumor.tsv')
            bg_path = os.path.join(fold_dir, f'validation-fold_{fold_num}_predictions_background.tsv')
            if not (os.path.exists(tumor_path) and os.path.exists(bg_path)):
                continue
            df_t = pd.read_csv(tumor_path, sep='\t', index_col=0)
            df_b = pd.read_csv(bg_path, sep='\t', index_col=0)
            t_metrics = compute_regression_metrics(df_t['truth'].to_numpy(), df_t['prediction_mean'].to_numpy())
            b_metrics = compute_regression_metrics(df_b['truth'].to_numpy(), df_b['prediction_mean'].to_numpy())

        weighted_tumor_gvw_crps = (
            float(fold_metrics_json.get('weighted_tumor_gvw_crps', float('nan')))
            if fold_metrics_json is not None else float('nan')
        )
        score = weighted_tumor_gvw_crps
        if not np.isfinite(score):
            score = float(t_metrics.get('gvw_crps', float('nan')))

        row = {
            'fold': int(fold_num),
            # Primary selection metric
            'score': score,
            'weighted_tumor_gvw_crps': weighted_tumor_gvw_crps,
            'tumor_gvw_crps': float(t_metrics.get('gvw_crps', float('nan'))),

            # Tumor regression summary
            'tumor_n': int(t_metrics.get('n', 0)),
            'tumor_mae': float(t_metrics.get('mae', float('nan'))),
            'tumor_rmse': float(t_metrics.get('rmse', float('nan'))),
            'tumor_r2': float(t_metrics.get('r2', float('nan'))),
            'tumor_gene_r2_mean': float(t_metrics.get('gene_r2_mean', float('nan'))),
            'tumor_gene_r2_median': float(t_metrics.get('gene_r2_median', float('nan'))),
            'tumor_sample_r2_mean': float(t_metrics.get('sample_r2_mean', float('nan'))),
            'tumor_sample_r2_median': float(t_metrics.get('sample_r2_median', float('nan'))),

            # Background regression summary
            'background_n': int(b_metrics.get('n', 0)),
            'background_mae': float(b_metrics.get('mae', float('nan'))),
            'background_rmse': float(b_metrics.get('rmse', float('nan'))),
            'background_r2': float(b_metrics.get('r2', float('nan'))),
            'background_gvw_crps': float(b_metrics.get('gvw_crps', float('nan'))),
        }
        fold_metrics.append(row)

    if not fold_metrics:
        return

    summary_df = pd.DataFrame(fold_metrics).sort_values('fold')
    summary_df.to_csv(os.path.join(out_dir, 'cv_metrics_summary.tsv'), sep='\t', index=False)

    # Also write a small JSON with mean/std across folds for quick comparison.
    numeric_cols = [c for c in summary_df.columns if c not in ('fold',)]
    primary_metric_name = 'weighted_tumor_gvw_crps'
    if not np.isfinite(pd.to_numeric(summary_df[primary_metric_name], errors='coerce').to_numpy(dtype=np.float64)).any():
        primary_metric_name = 'tumor_gvw_crps'

    agg: Dict[str, Any] = {
        'primary_metric': {'name': primary_metric_name, 'direction': 'minimize'},
    }
    for c in numeric_cols:
        vals = pd.to_numeric(summary_df[c], errors='coerce').to_numpy(dtype=np.float64)
        agg[c] = {
            'mean': float(np.nanmean(vals)),
            'std': float(np.nanstd(vals)),
        }

    # Best fold by primary metric
    if primary_metric_name in summary_df.columns:
        s = pd.to_numeric(summary_df[primary_metric_name], errors='coerce')
        if np.isfinite(s.to_numpy(dtype=np.float64)).any():
            best_idx = int(np.nanargmin(s.to_numpy(dtype=np.float64)))
            best_row = summary_df.iloc[best_idx].to_dict()
            agg['best'] = {
                'fold': int(best_row.get('fold', -1)),
                primary_metric_name: float(best_row.get(primary_metric_name, float('nan'))),
            }
    with open(os.path.join(out_dir, 'cv_metrics_summary.json'), 'w') as f:
        json.dump(agg, f, indent=2)


def full_train_from_splits(
    model_name: str = 'BioDVN',
    training_set_tsv: str = '',
    zarr_base: str = '',
    out_dir: str = 'final_models/',
    model_config: Optional[BioDVNConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    seed: int = 42,
    subsample_frac: Optional[float] = None,
    train_only_cohorts: Optional[List[str]] = None,
    balance_cohorts: Optional[List[str]] = None,
    source_groups: Optional[Dict[str, List[str]]] = None,
    target_n_per_source: Optional[int] = None,
    steps_per_epoch: int = 4000,
    batch_mode: str = 'multi_source',
):
    """Train a single model on ALL samples in the training_set TSV (no fold holdout).

    This is for producing a final production model after CV tuning has selected
    hyperparameters.  All samples are used for training.

    Full-train now defaults to a fixed schedule: the epoch count and HPs come
    from CV, the model trains on 100% of the data, and the terminal EMA weights
    are saved as the canonical ``final_model.pt`` artifact.  We intentionally do
    not checkpoint-select against a same-sample 10% monitoring split by default,
    because that holdout is not a true unseen-sample validation set.

    A clearly marked internal branch remains below for a future patient-only
    10% monitoring holdout if CV later shows that full-train still needs
    checkpoint selection.  If that path is re-enabled, the monitored branch can
    keep using ``best_model.pt`` without changing the fixed-schedule default.

    When ``train_only_cohorts`` is set, those cohorts (e.g. PHT admixtures)
    are used for training only.  The optional monitoring holdout loader draws
    exclusively from patient cohorts so that checkpoint selection, if later
    re-enabled, reflects clinical performance.

    Args:
        model_name: Name of the model class (e.g. 'BioDVN', 'BioDVN_NoSeq').
        training_set_tsv: Path to TSV file with Cohort/Sample/Fold columns.
        zarr_base: Base directory for Zarr stores.
        out_dir: Output directory for final model.
        model_config: Fully-specified BioDVNConfig.
        training_config: Fully-specified TrainingConfig.
        seed: Random seed for reproducibility.
        subsample_frac: Optional fraction (0.0-1.0) of genes for fast sanity checks.
        train_only_cohorts: Cohorts used only for training (excluded from val).
        balance_cohorts: Cohorts to balance via per-epoch undersampling.
    """
    import tempfile

    print('Full training mode (all samples)...')
    os.makedirs(out_dir, exist_ok=True)

    if model_config is None:
        model_config = BioDVNConfig()
    if training_config is None:
        training_config = TrainingConfig(checkpoint_dir=out_dir)
    training_config = replace(training_config, checkpoint_dir=out_dir)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required (GPU-only). No CPU fallback is supported.')
    device = torch.device('cuda')

    # Read all samples from TSV (ignore fold assignments)
    if not os.path.exists(training_set_tsv):
        raise FileNotFoundError(f"Training set TSV not found: {training_set_tsv}")

    _train_only = set(train_only_cohorts or [])

    all_paths: List[str] = []
    patient_paths: List[str] = []          # excludes train_only_cohorts
    zarr_cohort_map: Dict[str, str] = {}   # zarr_path → cohort (for train loader)
    zarr_sample_name_map: Dict[str, str] = {}  # zarr_path → full TSV sample name
    sample_cohort_map: Dict[str, str] = {} # TSV sample name → cohort (for Trainer)
    missing_samples: List[str] = []

    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cohort = row['Cohort'].strip()
            sample = row['Sample'].strip()

            zpath = zarr_path_for_sample(sample, cohort, zarr_base)
            if not os.path.isdir(zpath):
                bare = zarr_name_for_sample(sample, cohort)
                missing_samples.append(
                    f"  {cohort}/zarr_files/{bare}.zarr  (TSV sample={sample})"
                )
                continue
            all_paths.append(zpath)
            zarr_cohort_map[zpath] = cohort
            zarr_sample_name_map[zpath] = sample
            sample_cohort_map[sample] = cohort
            if cohort not in _train_only:
                patient_paths.append(zpath)

    if missing_samples:
        raise FileNotFoundError(
            f"Missing {len(missing_samples)} Zarr store(s) under {zarr_base}:\n"
            + "\n".join(missing_samples)
        )

    print(f'Total samples: {len(all_paths)} ({len(patient_paths)} patient, '
          f'{len(all_paths) - len(patient_paths)} train-only)')

    # Write index files: one for all samples (training), one for
    # patient-only samples (validation early-stopping signal).
    temp_dir = tempfile.mkdtemp(prefix='proteus_full_train_')
    all_index = os.path.join(temp_dir, 'full_all.json')
    with open(all_index, 'w') as f:
        json.dump(all_paths, f)

    if patient_paths and _train_only:
        patient_index = os.path.join(temp_dir, 'full_patient.json')
        with open(patient_index, 'w') as f:
            json.dump(patient_paths, f)
    else:
        patient_index = all_index

    # Log subsampling info if active
    if subsample_frac is not None and subsample_frac < 1.0:
        print(f"  [Subsample] Using ~{subsample_frac*100:.0f}% of genes for fast sanity check")

    # Compute per-gene variance weights from ALL samples (before split).
    # Clipped variance weights for the loss function.
    print('Computing gene variance weights from all non-PHT tumor samples...')
    from proteus_utils.proteus_utils import (
        compute_gene_weights_from_zarr,
        compute_gene_variances_unclipped,
        compute_gene_mean_std_from_zarr,
    )
    tumor_weight_paths = [p for p in all_paths if zarr_cohort_map[p] != 'PHT']
    train_fold_variance_raw = compute_gene_variances_unclipped(tumor_weight_paths, label_key='flat/tumor_exp')
    train_gene_mean_total, train_gene_std_total = compute_gene_mean_std_from_zarr(
        tumor_weight_paths,
        label_key='flat/tumor_exp',
    )
    gene_weights = compute_gene_weights_from_zarr(
        tumor_weight_paths,
        label_key='flat/tumor_exp',
    )
    _raw_vals = list(train_fold_variance_raw.values())
    print(f'  {len(gene_weights)} genes — '
          f'raw variance [{min(_raw_vals):.4f}, {max(_raw_vals):.4f}], '
          f'clipped to [0.100, 10.000]')

    selection_group_weights = get_plca_selection_group_weights(source_groups)
    if training_config.selection_metric == 'min_weighted_source_crps':
        if source_groups is None:
            raise ValueError(
                "selection_metric='min_weighted_source_crps' requires source_groups"
            )
        validate_plca_selection_groups(source_groups)

    # -- Source groups handling (for multi-source batch mode / weighted metrics) --
    sample_to_source: Optional[Dict[str, str]] = None
    source_tumor_variance: Optional[Dict[str, Dict[str, float]]] = None
    _sample_paths_by_source: Optional[Dict[str, List[str]]] = None
    if source_groups:
        from proteus_utils.data_helpers import assign_source
        _cohort_to_source = invert_source_groups(source_groups)
        _pdx_names = collect_pdx_names(training_set_tsv, _cohort_to_source)

        sample_to_source = {}
        _sample_paths_by_source = {src: [] for src in source_groups}
        _source_variance_paths: Dict[str, List[str]] = {src: [] for src in source_groups}

        for zarr_path in all_paths:
            cohort = zarr_cohort_map[zarr_path]
            sample = zarr_sample_name_map[zarr_path]
            source = assign_source(cohort, sample, _cohort_to_source,
                                   _pdx_names)
            if source is None:
                continue
            sample_to_source[sample] = source
            _sample_paths_by_source[source].append(zarr_path)
            if cohort != 'PHT':
                _source_variance_paths[source].append(zarr_path)

        source_tumor_variance = compute_source_variance(
            source_groups, _source_variance_paths,
            compute_gene_variances_unclipped,
        )

    # Full-train defaults to the CV-chosen fixed epoch budget.  If CV later
    # suggests that the final model still benefits from monitored checkpoint
    # selection, flip this internal switch and keep the patient-only holdout
    # path below.  We keep it local instead of exposing a new CLI/config knob
    # so the production training contract stays explicit and stable.
    use_checkpoint_holdout = False
    val_frac = 0.10
    val_loader: Optional[DataLoader] = None
    val_steps = 0

    if batch_mode == 'multi_source':
        # Multi-source training loader: gene-conditioned batches balanced
        # across source groups.  This branch is retained for direct comparison
        # runs, but PLCAv2 should prefer the chunk path above it.
        from proteus_utils.proteus_utils import (
            ProteusMultiSourceBatchDataset, ProteusHoldoutValDataset,
            _collate_identity,
        )
        if _sample_paths_by_source is None:
            raise ValueError(
                "batch_mode='multi_source' requires source_groups")

        # Build training dataset (preloads all samples).
        train_dataset = ProteusMultiSourceBatchDataset(
            sample_paths_by_source=_sample_paths_by_source,
            gene_variance=train_fold_variance_raw,
            n_eff_per_source=target_n_per_source or 24,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
            labeled=True,
            sample_name_map=zarr_sample_name_map,
            include_metadata=True,
            subsample_frac=subsample_frac,
        )

        # Build holdout validation dataset (shares preloaded arrays).
        base_generator = torch.Generator()
        base_generator.manual_seed(seed)

        def _seed_worker(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % (2**32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        _nw = _loader_workers()
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=_nw,
            collate_fn=_collate_identity,
            persistent_workers=_nw > 0,
            prefetch_factor=4 if _nw > 0 else None,
            worker_init_fn=_seed_worker if _nw > 0 else None,
            generator=base_generator,
            drop_last=False,
            pin_memory=True,
        )
        train_steps = len(train_dataset)

        if use_checkpoint_holdout:
            # This same-sample random holdout path is intentionally disabled by
            # default.  If we re-enable checkpoint monitoring here after CV,
            # keep the patient/source-weighted validation metrics as the sole
            # selection signal and continue saving a monitored best_model.pt.
            n_samples = len(train_dataset._all_sample_paths)
            n_union = len(train_dataset._union_genes)
            rng_mask = np.random.RandomState(seed + 12345)
            val_mask = np.zeros((n_samples, n_union), dtype=np.bool_)
            _path_to_src: Dict[str, str] = {}
            for src, paths in _sample_paths_by_source.items():
                for p in paths:
                    _path_to_src[p] = src
            for s_idx, zpath in enumerate(train_dataset._all_sample_paths):
                src = _path_to_src.get(zpath)
                if src is None:
                    continue
                src_mask = train_dataset._source_gene_mask[src]
                valid_gene_indices = np.where(src_mask)[0]
                n_valid = len(valid_gene_indices)
                n_holdout = max(1, int(n_valid * val_frac))
                chosen = rng_mask.choice(valid_gene_indices, size=n_holdout, replace=False)
                val_mask[s_idx, chosen] = True

            total_pairs = int(val_mask.sum())
            total_valid = sum(
                int(train_dataset._source_gene_mask[_path_to_src[p]].sum())
                for p in train_dataset._all_sample_paths
                if p in _path_to_src
            )
            print(f'  [HoldoutVal] {total_pairs} val pairs / {total_valid} total '
                  f'({100.0 * total_pairs / total_valid:.1f}%)')

            train_dataset.val_mask = val_mask
            val_dataset = ProteusHoldoutValDataset(
                train_dataset=train_dataset,
                val_mask=val_mask,
                batch_size=32,
                seed=seed,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=_nw,
                collate_fn=_collate_identity,
                persistent_workers=_nw > 0,
                prefetch_factor=4 if _nw > 0 else None,
                worker_init_fn=_seed_worker if _nw > 0 else None,
                generator=base_generator,
                drop_last=False,
                pin_memory=True,
            )
            val_steps = len(val_dataset)
    else:
        # Standard chunk-based training loader with actual epoch shuffling and
        # bounded cross-sample mixing in the worker-local microbatch buffer.
        # This is the preferred PLCAv2 full-train path.
        train_loader, train_steps = create_data_loader(
            index_file=all_index,
            num_workers=_loader_workers(),
            seed=seed,
            subsample_frac=subsample_frac,
            gene_weights=gene_weights,
            val_frac=val_frac if use_checkpoint_holdout else None,
            val_split=False,
            cohort_map=zarr_cohort_map if balance_cohorts else None,
            balance_cohorts=balance_cohorts,
        )

        if use_checkpoint_holdout:
            val_loader, val_steps = create_data_loader(
                index_file=patient_index,
                num_workers=_loader_workers(),
                seed=seed,
                include_metadata=True,
                subsample_frac=subsample_frac,
                gene_weights=gene_weights,
                val_frac=val_frac,
                val_split=True,
                sample_name_map=zarr_sample_name_map,
            )

    print(f'Training batches: {train_steps}, Validation batches: {val_steps}')

    # Seed before model construction for reproducible weight init.
    _seed_everything(seed)

    # Create model
    ModelClass = get_model_instance(model_name)
    model = ModelClass(model_config)

    # Save config
    with open(os.path.join(out_dir, 'model_config.json'), 'w') as f:
        json.dump(asdict(model_config), f, indent=2)
    with open(os.path.join(out_dir, 'training_config.json'), 'w') as f:
        json.dump(asdict(training_config), f, indent=2)

    # Train
    trainer = Trainer(model, training_config, model_config, device)
    trainer.set_srl_stats(train_gene_mean_total, train_gene_std_total)
    if val_loader is not None:
        trainer.val_gene_weight_map = train_fold_variance_raw
        trainer.val_source_gene_weight_map = source_tumor_variance
        trainer.val_selection_group_weights = selection_group_weights
        if sample_to_source:
            trainer.val_sample_group_map = sample_to_source
        elif sample_cohort_map and _train_only:
            patient_cohort_map = {
                k: v for k, v in sample_cohort_map.items() if v not in _train_only
            }
            trainer.val_sample_group_map = patient_cohort_map

    history = trainer.fit(train_loader, val_loader, epochs=training_config.epochs)
    with open(os.path.join(out_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plot_dir = os.path.join(out_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        fig = plt.figure(figsize=(7, 4))
        epochs_axis = np.arange(1, len(history.get('train_loss', [])) + 1)
        if len(epochs_axis) > 0:
            plt.plot(epochs_axis, history.get('train_loss', []), label='train_loss')
            if history.get('val_loss'):
                plt.plot(epochs_axis, history.get('val_loss', []), label='val_loss')
            if history.get('val_tumor_gvw_crps'):
                plt.plot(epochs_axis, history.get('val_tumor_gvw_crps', []), label='val_tumor_gvw_crps')
            plt.xlabel('epoch')
            plt.ylabel('metric')
            plt.title('Training / validation curves')
            plt.legend(frameon=False)
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, 'loss_curves.png'), dpi=150)
        plt.close(fig)
    except (ImportError, OSError, ValueError) as e:
        print(f'  Warning: failed to generate full-train plots: {e}')

    final_path = os.path.join(out_dir, 'final_model.pt')
    if val_loader is None:
        ema_backup = None
        if trainer._ema is not None:
            ema_backup = trainer._ema.apply(trainer.model)
        trainer._save_checkpoint(
            training_config.epochs,
            is_best=False,
            filename='final_model.pt',
        )
        if ema_backup is not None:
            trainer._ema.restore(trainer.model, ema_backup)
        print(f'Saved fixed-schedule full-train checkpoint to: {final_path}')
    else:
        best_path = os.path.join(out_dir, 'best_model.pt')
        print(f'Best monitored full-train checkpoint available at: {best_path}')


def run_model(
    model_path: str,
    sample_files: List[str],
    out_dir: str,
    model_name: str = 'BioDVN',
):
    """Run the model on samples for inference.

    Outputs long-format TSVs (one row per gene × sample) with full MDN
    distribution parameters and quantile CIs — matching the
    validation-fold prediction schema (minus the ``truth`` column).

    Args:
        model_path: Path to saved model weights.
        sample_files: List of paths to sample Zarr stores.
        out_dir: Output directory.
        model_name: Model architecture name (e.g. 'BioDVN', 'BioDVN_NoSeq',
            'BioDVN_OnlySeq').  Must match the architecture that produced
            the checkpoint at *model_path*.
    """
    from models.model_utils import mixture_quantile

    print('Running model for inference...')
    os.makedirs(out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required (GPU-only). No CPU fallback is supported.')
    device = torch.device('cuda')

    # Load model config if present (for reproducible inference)
    model_dir = str(Path(model_path).parent)
    cfg_path = os.path.join(model_dir, 'model_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            cfg_raw = json.load(f)
        allowed = set(BioDVNConfig.__dataclass_fields__.keys())
        cfg = {k: v for k, v in cfg_raw.items() if k in allowed}
        # Ensure channel_mode matches the model variant name.
        cfg['channel_mode'] = channel_mode_for_model(model_name)
        model_config = BioDVNConfig(**cfg)
        print(f'Loaded model config from: {cfg_path}')
    else:
        model_config = BioDVNConfig(
            channel_mode=channel_mode_for_model(model_name),
        )
        print('WARNING: No model_config.json found; using BioDVNConfig defaults for inference.')

    ModelClass = get_model_instance(model_name)
    model = ModelClass(model_config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Long-format accumulators (shared across all samples).
    _pred_cols = [
        'prediction_mean',
        'ci_q5', 'ci_q95',
        'pi_delta', 'pi_gamma', 'gamma_beta',
        'pi_normal', 'normal_mean', 'normal_stdev',
    ]
    tumor_rows: Dict[str, list] = {k: [] for k in _pred_cols}
    bg_rows: Dict[str, list] = {k: [] for k in _pred_cols}
    all_genes: List[str] = []
    all_tumor_samples: List[str] = []
    all_bg_samples: List[str] = []

    n_samples = len(sample_files)
    for idx, sample_file in enumerate(sample_files, 1):
        sample = Path(sample_file).stem.replace('.zarr', '')
        print(f'[{idx}/{n_samples}] Running inference on {sample}...')

        loader, gene_names = create_inference_loader(sample_file)

        with torch.no_grad():
            for batch in loader:
                batch_device = {k: v.to(device) for k, v in batch.items()}
                predictions, _ = model(batch_device, return_internal_loss=False)

                for rows, head in [(tumor_rows, 'tumor'), (bg_rows, 'background')]:
                    mdn = predictions[f'{head}_mdn']
                    q_vals = mixture_quantile(mdn, quantiles=(0.05, 0.95))  # (batch, 2)
                    pi = mdn['log_pi'].exp().cpu().numpy()

                    rows['prediction_mean'].extend(
                        predictions[f'{head}_exp'].cpu().numpy().flatten())
                    rows['ci_q5'].extend(q_vals[:, 0].cpu().numpy().flatten())
                    rows['ci_q95'].extend(q_vals[:, 1].cpu().numpy().flatten())
                    rows['pi_delta'].extend(pi[:, 0].flatten())
                    rows['pi_gamma'].extend(pi[:, 1].flatten())
                    rows['gamma_beta'].extend(
                        mdn['gamma_beta'].cpu().numpy().flatten())
                    rows['pi_normal'].extend(pi[:, 2].flatten())
                    rows['normal_mean'].extend(
                        mdn['normal_mu'].cpu().numpy().flatten())
                    rows['normal_stdev'].extend(
                        mdn['normal_sigma'].cpu().numpy().flatten())

        n_genes = len(gene_names)
        all_genes.extend([str(g) for g in gene_names])
        all_tumor_samples.extend([sample] * n_genes)
        all_bg_samples.extend([sample] * n_genes)

        print(f'[{idx}/{n_samples}] Done  ({n_genes} genes).')

    # Build and save long-format TSVs (one row per gene × sample).
    tumor_rows['sample'] = all_tumor_samples
    bg_rows['sample'] = all_bg_samples

    df_tumor = pd.DataFrame(tumor_rows, index=all_genes)
    df_bg = pd.DataFrame(bg_rows, index=all_genes)

    df_tumor.to_csv(os.path.join(out_dir, 'predictions_tumor.tsv'), sep='\t')
    df_bg.to_csv(os.path.join(out_dir, 'predictions_background.tsv'), sep='\t')

    print('Finished running model on all samples.')


def _build_configs_from_spec(experiment_spec, args, model_name: str = 'BioDVN') -> tuple:
    """Build BioDVNConfig and TrainingConfig from an experiment spec.

    Returns (model_cfg, train_cfg).
    """
    model_cfg = BioDVNConfig(
        channel_mode=channel_mode_for_model(model_name),
        encoder_start=int(experiment_spec.scale.encoder_start.fixed),
        latent_size=int(experiment_spec.scale.latent_size.fixed),
        decoder_start=int(experiment_spec.scale.decoder_start.fixed),
        mdn_hidden=int(experiment_spec.scale.mdn_hidden.fixed),
        mdn_final=int(experiment_spec.scale.mdn_final.fixed),
        signal_filters=int(experiment_spec.scale.signal_filters.fixed),
        signal_ksize=int(experiment_spec.scale.signal_ksize.fixed),
        interaction_ksize=int(experiment_spec.scale.interaction_ksize.fixed),
        initial_reduction_channels=int(experiment_spec.scale.initial_reduction_channels.fixed),
        n_reductions=int(experiment_spec.scale.n_reductions.fixed),
        funnel_reduction_factor=int(experiment_spec.scale.funnel_reduction_factor.fixed),
        reduction_dilation=int(experiment_spec.scale.reduction_dilation.fixed),
        beta_background=float(experiment_spec.structural.beta_background.fixed),
        beta_kl=float(experiment_spec.structural.beta_kl.fixed),
        beta_contrastive=float(experiment_spec.structural.beta_contrastive.fixed),
        beta_reconstruction=float(experiment_spec.structural.beta_reconstruction.fixed),
        beta_tfx=float(experiment_spec.structural.beta_tfx.fixed),
        beta_ccc=float(experiment_spec.structural.beta_ccc.fixed),
        beta_srl_max=float(experiment_spec.structural.beta_srl_max.fixed),
        srl_delta=float(experiment_spec.structural.srl_delta.fixed),
        srl_start=int(experiment_spec.structural.srl_start.fixed),
        srl_ramp=int(experiment_spec.structural.srl_ramp.fixed),
        beta_pht_consistency=float(experiment_spec.structural.beta_pht_consistency.fixed),
        beta_pht_uncertainty=float(experiment_spec.structural.beta_pht_uncertainty.fixed),
        cl_margin=float(experiment_spec.structural.cl_margin.fixed),
        internal_weight=float(experiment_spec.structural.internal_weight.fixed),
        dropout_rate=float(experiment_spec.scale.dropout_rate.fixed),
        weight_decay=float(experiment_spec.training.weight_decay.fixed),
        max_lr=float(experiment_spec.training.max_lr.fixed),
        max_grad_norm=float(experiment_spec.training.max_grad_norm.fixed),
    )

    train_cfg = TrainingConfig(
        checkpoint_dir='',
        max_lr=float(experiment_spec.training.max_lr.fixed),
        min_lr_ratio=float(experiment_spec.training.min_lr_ratio.fixed),
        warmup_start_lr=float(experiment_spec.training.warmup_start_lr.fixed),
        weight_decay=float(experiment_spec.training.weight_decay.fixed),
        max_grad_norm=float(experiment_spec.training.max_grad_norm.fixed),
        warmup_frac=float(experiment_spec.training.warmup_frac.fixed),
        schedule_frac=float(experiment_spec.training.schedule_frac.fixed),
        epochs=int(experiment_spec.training.epochs.fixed),
        patience=int(experiment_spec.training.patience.fixed),
        use_amp=bool(experiment_spec.training.use_amp.fixed),
        ema_decay=float(experiment_spec.training.ema_decay.fixed),
        grad_accum_steps=int(experiment_spec.training.grad_accum_steps.fixed),
        selection_metric=str(experiment_spec.data.selection_metric),
        runtime_device_banner=bool(args.runtime_device_banner),
        base_weights=experiment_spec.transfer.base_weights,
    )

    return model_cfg, train_cfg


def main():
    """Main entry point for Proteus PyTorch pipeline."""
    parser = argparse.ArgumentParser(description="Run Proteus (PyTorch) - Gene expression prediction from cfDNA", formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="""
    Examples:
    # Process raw data to Zarr stores
    python Proteus.py --process_data --use_set PC-PDX
    
    # Train with cross-validation (requires experiment YAML)
    python Proteus.py --run_mode train --experiment config/experiments/biodvn_train.yaml
    
    # Train specific fold (for HPC parallelization)
    python Proteus.py --run_mode train --experiment config/experiments/biodvn_train.yaml --test_fold 1

    # Full training (all data, no CV holdout)
    python Proteus.py --run_mode full-train --experiment config/experiments/biodvn_train.yaml

    # Optuna CV tuning
    python Proteus.py --run_mode tune --experiment config/experiments/biodvn_tune.yaml --n_trials 50
    
    # Run inference on holdout samples (uses trained_models/ by default)
    python Proteus.py --run_mode run --model_name BioDVN --use_set Crompton
    python Proteus.py --run_mode run --model_name BioDVN_NoSeq --use_set Crompton
    """
    )
    
    # Data processing arguments
    parser.add_argument("--process_data", action='store_true', help="Process raw Triton data into Zarr stores")
    parser.add_argument("--verify_data", action='store_true', help="Fast presence check for Zarr stores (existence-only)")
    
    # Dataset configuration
    parser.add_argument("--use_set", type=str, default='PC-PDX', help="Dataset name (e.g., PC-PDX, UW, SCLC). Use 'all' to process every configured dataset.")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Base directory for raw Triton outputs (default: /fh/fast/ha_g/user/rpatton)")
    parser.add_argument("--zarr_output_dir", type=str, default=None, help="Zarr output directory (default: /hpc/temp/ha_g/user/rpatton/ProteusZarrs)")
    parser.add_argument("--scalars_reference_path", type=str, default="reference/MANE.GRCh38.v1.3.UniversalReference.tsv", help="Path to reference TSV file")
    
    # Training arguments
    parser.add_argument("--model_name", type=str, default="BioDVN", help="Model architecture name")
    parser.add_argument("--run_mode", type=str, default='train', choices=['train', 'full-train', 'tune', 'tune-rescue', 'tune-ask', 'tune-tell', 'tune-fold', 'tune-proxy-ask', 'tune-proxy-tell', 'tune-proxy-finalize', 'run'], help="train=CV training, full-train=final model, tune=Optuna CV tuning, tune-rescue=backfill completed Optuna trials from saved fold artifacts, tune-ask/tune-tell/tune-fold=coordinated fold-parallel tuning, tune-proxy-ask/tune-proxy-tell/tune-proxy-finalize=proxy-fold racing, run=inference")
    parser.add_argument("--test_fold", type=int, default=None, help="Specific fold to train (for HPC parallelization)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    # Experiment / tuning
    parser.add_argument("--experiment", type=str, default=None, help="Path to experiment YAML (required for train/full-train/tune modes)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials for --run_mode tune")
    parser.add_argument("--n_configs", type=int, default=4, help="Number of parallel config groups for --run_mode tune-ask")
    parser.add_argument("--trial_number", type=int, default=None, help="Trial number for --run_mode tune-fold")
    parser.add_argument("--trial_numbers", type=str, default=None, help="Comma-separated trial numbers for --run_mode tune-tell (default: read from coordinated_active.json)")
    parser.add_argument("--optuna_storage", type=str, default="sqlite:///results_training/optuna.db", help="Optuna storage URL (default: sqlite:///results_training/optuna.db)")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name override (default: experiment.name)")
    parser.add_argument("--sampler_seed", type=int, default=42, help="Optuna sampler seed")

    # Manual warm-start only (transfer learning is HP-controlled in experiment YAML)
    parser.add_argument("--base_weights", type=str, default=None, help="Optional: path to a .pt state_dict to load before training (manual warm-start).")
    parser.add_argument("--runtime_device_banner", action='store_true', help="Print device and memory info at runtime")
    
    # Sanity check / fast iteration
    parser.add_argument("--subsample", type=float, default=None, 
                        help="Subsample fraction (0.0-1.0) of genes for fast sanity checks. "
                             "E.g., --subsample 0.1 uses ~10%% of genes per epoch.")
    # General options
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing data files")
    args = parser.parse_args()
    
    # If an experiment is provided, it becomes the source of truth for training.
    experiment_spec = None
    if args.experiment is not None:
        from config.experiment_spec import load_experiment_spec
        experiment_spec = load_experiment_spec(args.experiment)

    # For data processing and inference, use_set is still used
    active_use_set = args.use_set
    
    # Helper function to load dataset config.
    def get_dataset_config(dataset_name: str):
        """Load dataset config from YAML or return None if not found."""
        try:
            return load_dataset_config(dataset_name)
        except FileNotFoundError:
            return None
    
    # Determine Zarr output directory
    zarr_base = args.zarr_output_dir or DEFAULT_ZARR_BASE

    # Resolve project-relative inputs regardless of CWD
    labels_dir = str(PROJECT_ROOT / 'labels')
    default_ref_path = str(PROJECT_ROOT / 'reference' / 'MANE.GRCh38.v1.3.UniversalReference.tsv')
    
    # Verify data mode
    if args.verify_data:
        zarr_dir = get_zarr_output_dir(args.use_set, zarr_base)
        if not (os.path.exists(zarr_dir) and any(str(f).endswith('.zarr') for f in os.listdir(zarr_dir))):
            print(f"No Zarr stores found in {zarr_dir}")
            return

        stores = [f for f in os.listdir(zarr_dir) if f.endswith('.zarr')]
        print(f"Found {len(stores)} Zarr stores in {zarr_dir}")
        preview = sorted(stores)[:20]
        if preview:
            print("First stores:")
            for s in preview:
                print(f"  {s}")
        return
    
    # Process data into Zarr stores
    if args.process_data:
        # Resolve dataset list: 'all' expands to every configured YAML.
        if args.use_set.lower() == 'all':
            datasets_to_process = list_datasets()
            if not datasets_to_process:
                print("No dataset configs found in config/datasets/")
                sys.exit(1)
            print(f"Processing all configured datasets: {datasets_to_process}")
        else:
            datasets_to_process = [args.use_set]

        raw_data_dir = args.raw_data_dir or "/fh/fast/ha_g/user/rpatton"

        for dataset_name in datasets_to_process:
            dataset_config = get_dataset_config(dataset_name)
            if dataset_config is None:
                available = list_datasets()
                print(f"Dataset '{dataset_name}' is not configured.")
                print(f"Available datasets: {available}")
                print(f"Add a YAML config file at: config/datasets/{dataset_name}.yaml")
                sys.exit(1)

            # Build absolute paths from config-specified relative paths
            tss_dir = os.path.join(raw_data_dir, dataset_config.triton_tss)
            gb_dir = os.path.join(raw_data_dir, dataset_config.triton_gb)

            print(f'Processing {dataset_name}...')
            print(f'  Config loaded from: config/datasets/{dataset_name}.yaml')
            generate_data(
                dataset=dataset_name,
                tss_dir=tss_dir,
                gb_dir=gb_dir,
                labels_dir=labels_dir,
                ref_path=args.scalars_reference_path or default_ref_path,
                tfx_file=dataset_config.tfx_file,
                zarr_base_dir=zarr_base,
                tumor_exp_file=dataset_config.tumor_exp_file,
                background_exp_file=dataset_config.background_exp_file,
                overwrite=args.overwrite,
            )

        print('Finished processing data.')
        return
    
    # ==========================================================================
    # AUTO-CHECK: Ensure required datasets exist before training/tuning.
    # Skipped in inference mode — inference should only run on existing data.
    # ==========================================================================
    if args.run_mode != 'run':
        raw_data_dir = args.raw_data_dir or "/fh/fast/ha_g/user/rpatton"

        datasets_to_check: List[str] = []
        if get_dataset_config(active_use_set) is not None:
            datasets_to_check.append(active_use_set)

        if datasets_to_check:
            ensure_datasets_exist(
                dataset_names=datasets_to_check,
                zarr_base=zarr_base,
                raw_data_base=raw_data_dir,
                labels_dir=labels_dir,
                ref_path=args.scalars_reference_path or default_ref_path,
                n_workers=0  # 0 = auto-detect (use all available CPUs)
            )

    # NOTE: transfer learning is controlled via base_weights in experiment YAML.
    # CLI --base_weights is a manual warm-start convenience only.
    
    # Training modes
    if args.run_mode == 'train':
        if experiment_spec is None:
            raise ValueError(
                "--run_mode train requires --experiment <path/to/experiment.yaml>\n"
                "Training now requires an experiment YAML that defines the training_set TSV."
            )

        # Resolve fixed parameters from experiment spec
        model_cfg, train_cfg = _build_configs_from_spec(
            experiment_spec, args, model_name=args.model_name,
        )

        out_dir = experiment_spec.output_dir
        test_fold = args.test_fold  # CLI override for HPC parallelization

        if test_fold is not None:
            print(f'Running fold {test_fold}...')
        else:
            print('Running all folds from training_set TSV...')

        train_model_cv(
            model_name=args.model_name,
            training_set_tsv=experiment_spec.data.training_set,
            zarr_base=experiment_spec.data.zarr_base,
            test_fold=test_fold,
            out_dir=out_dir,
            model_config=model_cfg,
            training_config=train_cfg,
            seed=experiment_spec.data.seed,
            subsample_frac=args.subsample,
            train_only_cohorts=experiment_spec.data.train_only_cohorts,
            balance_cohorts=experiment_spec.data.balance_cohorts,
            source_groups=experiment_spec.data.source_groups,
            target_n_per_source=experiment_spec.data.target_n_per_source,
            steps_per_epoch=experiment_spec.data.steps_per_epoch,
            batch_mode=experiment_spec.data.batch_mode,
            fixed_schedule_oof=True,
        )
    
    elif args.run_mode == 'full-train':
        if experiment_spec is None:
            raise ValueError(
                "--run_mode full-train requires --experiment <path/to/experiment.yaml>\n"
                "Full training now requires an experiment YAML that defines the training_set TSV."
            )

        # For full-train, we train on ALL samples in the TSV (no fold held out)
        # This produces a single model using all available data.
        model_cfg, train_cfg = _build_configs_from_spec(
            experiment_spec, args, model_name=args.model_name,
        )

        out_dir = os.path.join(experiment_spec.output_dir, 'full_train')

        full_train_from_splits(
            model_name=args.model_name,
            training_set_tsv=experiment_spec.data.training_set,
            zarr_base=experiment_spec.data.zarr_base,
            out_dir=out_dir,
            model_config=model_cfg,
            training_config=train_cfg,
            seed=experiment_spec.data.seed,
            subsample_frac=args.subsample,
            train_only_cohorts=experiment_spec.data.train_only_cohorts,
            balance_cohorts=experiment_spec.data.balance_cohorts,
            source_groups=experiment_spec.data.source_groups,
            target_n_per_source=experiment_spec.data.target_n_per_source,
            steps_per_epoch=experiment_spec.data.steps_per_epoch,
            batch_mode=experiment_spec.data.batch_mode,
        )

    elif args.run_mode == 'tune':
        if args.experiment is None:
            raise ValueError("--run_mode tune requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import run_optuna_cv_tuning, run_optuna_fold_tuning

        # If --test_fold is provided, run fold-parallel tuning:
        # launch 5 jobs (one per fold) pointing at the same study/storage.
        # Each fold independently calls study.optimize() — 5 folds = 5 different trials.
        if args.test_fold is not None:
            run_optuna_fold_tuning(
                experiment_path=str(args.experiment),
                fold_num=int(args.test_fold),
                storage=str(args.optuna_storage),
                n_trials=(int(args.n_trials) if args.n_trials else None),
                study_name=(None if args.study_name is None else str(args.study_name)),
                sampler_seed=int(args.sampler_seed),
            )
        else:
            run_optuna_cv_tuning(
                experiment_path=str(args.experiment),
                n_trials=int(args.n_trials),
                storage=str(args.optuna_storage),
                study_name=(None if args.study_name is None else str(args.study_name)),
                sampler_seed=int(args.sampler_seed),
            )

    elif args.run_mode == 'tune-rescue':
        if args.experiment is None:
            raise ValueError("--run_mode tune-rescue requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import rescue_failed_fold_trials

        trial_nums = None
        if args.trial_numbers is not None:
            trial_nums = [int(x.strip()) for x in args.trial_numbers.split(",")]

        rescued_trials = rescue_failed_fold_trials(
            experiment_path=str(args.experiment),
            storage=str(args.optuna_storage),
            study_name=(None if args.study_name is None else str(args.study_name)),
            trial_numbers=trial_nums,
            sampler_seed=int(args.sampler_seed),
        )
        print(f"RESCUED_TRIAL_NUMBERS={','.join(str(t) for t in rescued_trials)}", flush=True)

    elif args.run_mode == 'tune-ask':
        if args.experiment is None:
            raise ValueError("--run_mode tune-ask requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import coordinated_ask

        trial_numbers = coordinated_ask(
            experiment_path=str(args.experiment),
            n_configs=int(args.n_configs),
            storage=str(args.optuna_storage),
            study_name=(None if args.study_name is None else str(args.study_name)),
            sampler_seed=int(args.sampler_seed),
        )
        # Print trial numbers for SLURM script to parse
        print(f"TRIAL_NUMBERS={','.join(str(t) for t in trial_numbers)}", flush=True)

    elif args.run_mode == 'tune-tell':
        if args.experiment is None:
            raise ValueError("--run_mode tune-tell requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import coordinated_tell

        trial_nums = None
        if args.trial_numbers is not None:
            trial_nums = [int(x.strip()) for x in args.trial_numbers.split(",")]

        coordinated_tell(
            experiment_path=str(args.experiment),
            trial_numbers=trial_nums,
            storage=str(args.optuna_storage),
            study_name=(None if args.study_name is None else str(args.study_name)),
            sampler_seed=int(args.sampler_seed),
        )

    elif args.run_mode == 'tune-proxy-ask':
        if args.experiment is None:
            raise ValueError("--run_mode tune-proxy-ask requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import coordinated_proxy_ask

        trial_numbers = coordinated_proxy_ask(
            experiment_path=str(args.experiment),
            n_configs=int(args.n_configs),
            storage=str(args.optuna_storage),
            study_name=(None if args.study_name is None else str(args.study_name)),
            sampler_seed=int(args.sampler_seed),
        )
        print(f"TRIAL_NUMBERS={','.join(str(t) for t in trial_numbers)}", flush=True)

    elif args.run_mode == 'tune-proxy-tell':
        if args.experiment is None:
            raise ValueError("--run_mode tune-proxy-tell requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import coordinated_proxy_tell

        trial_nums = None
        if args.trial_numbers is not None:
            trial_nums = [int(x.strip()) for x in args.trial_numbers.split(",")]

        reported_trials = coordinated_proxy_tell(
            experiment_path=str(args.experiment),
            trial_numbers=trial_nums,
            storage=str(args.optuna_storage),
            study_name=(None if args.study_name is None else str(args.study_name)),
            sampler_seed=int(args.sampler_seed),
        )
        print(f"REPORTED_TRIAL_NUMBERS={','.join(str(t) for t in reported_trials)}", flush=True)

    elif args.run_mode == 'tune-proxy-finalize':
        if args.experiment is None:
            raise ValueError("--run_mode tune-proxy-finalize requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import coordinated_proxy_finalize

        trial_nums = None
        if args.trial_numbers is not None:
            trial_nums = [int(x.strip()) for x in args.trial_numbers.split(",")]

        finalized_trials = coordinated_proxy_finalize(
            experiment_path=str(args.experiment),
            trial_numbers=trial_nums,
        )
        print(f"FINALIZED_TRIAL_NUMBERS={','.join(str(t) for t in finalized_trials)}", flush=True)

    elif args.run_mode == 'tune-fold':
        if args.experiment is None:
            raise ValueError("--run_mode tune-fold requires --experiment <path/to/experiment.yaml>")
        if args.trial_number is None:
            raise ValueError("--run_mode tune-fold requires --trial_number <N>")
        if args.test_fold is None:
            raise ValueError("--run_mode tune-fold requires --test_fold <K>")

        from tuning.optuna_tuner import run_coordinated_fold

        run_coordinated_fold(
            experiment_path=str(args.experiment),
            trial_number=int(args.trial_number),
            fold_num=int(args.test_fold),
        )
    
    elif args.run_mode == 'run':
        # Resolve model weights path.  When an experiment YAML is provided,
        # use its output_dir (mirrors full-train).  Otherwise, derive from
        # the standard trained_models/ naming convention.
        if experiment_spec is not None:
            model_dir = os.path.join(experiment_spec.output_dir, 'full_train')
        else:
            # Convention: trained_models/proteus_<variant>/full_train/
            _MODEL_DIR_MAP = {
                'BioDVN':        'proteus_base-model',
                'BioDVN_NoSeq':  'proteus_base-model_no-seq',
                'BioDVN_OnlySeq':'proteus_base-model_only-seq',
                'BioDVN_PAN-FT': 'proteus_PAN-FT',
            }
            dir_name = _MODEL_DIR_MAP.get(args.model_name)
            if dir_name is None:
                raise ValueError(
                    f"No known trained-model directory for model_name='{args.model_name}'. "
                    f"Available: {list(_MODEL_DIR_MAP.keys())}. "
                    f"Provide --experiment <yaml> to specify the model location explicitly."
                )
            model_dir = os.path.join('trained_models', dir_name, 'full_train')

        model_path = os.path.join(model_dir, 'final_model.pt')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Final full-train model not found at: {model_path}\n"
                f"Run --run_mode full-train first, or provide --experiment <yaml> "
                f"to point to the correct model directory."
            )

        # Inference runs on the dataset specified by --use_set
        patient_set = args.use_set
        zarr_dir = get_zarr_output_dir(patient_set, zarr_base)

        if not (os.path.exists(zarr_dir) and any(f.endswith('.zarr') for f in os.listdir(zarr_dir))):
            raise FileNotFoundError(
                f"No Zarr stores found for '{patient_set}' in: {zarr_dir}\n"
                "Run --process_data for that dataset (or point --zarr_output_dir to the correct base)."
            )

        sample_files = [os.path.join(zarr_dir, f) for f in os.listdir(zarr_dir) if f.endswith('.zarr')]
        output_base = f'results_holdout/{args.model_name}/{patient_set}/'
        run_model(model_path, sample_files, output_base, model_name=args.model_name)
    
    else:
        print('Invalid run mode. Exiting.')
        sys.exit(1)


if __name__ == "__main__":
    main()
