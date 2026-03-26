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
from models import BioDVN, BioDVNConfig, get_model_instance
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
from config.config_loader import ConfigLoader

# Loader sizing invariants
FIXED_GENES_PER_BATCH = 32
FIXED_CHUNK_SIZE = 256  # Must match Zarr store chunking in compile_sample_data.py

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
    config_loader: 'ConfigLoader',
    dataset_names: List[str],
    zarr_base: str,
    raw_data_base: str,
    labels_dir: str,
    ref_path: str,
    n_workers: int = 0
) -> Dict[str, bool]:
    """
    Check that all configured datasets have Zarr stores, regenerate if missing.
    
    This is useful because Zarr stores are kept in fast-access temporary storage
    that may be cleared periodically (e.g., every 30 days).
    
    Args:
        config_loader: ConfigLoader instance for loading dataset configs
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
            dataset_config = config_loader.load_dataset(dataset_name)
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


def _collect_zarr_paths(
    training_set_tsv: str,
    zarr_base: str,
    exclude_fold: Optional[int] = None,
    exclude_cohorts: Optional[set] = None,
) -> List[str]:
    """Collect Zarr paths from a training-set TSV.

    Args:
        training_set_tsv: TSV with Cohort/Sample/Fold columns.
        zarr_base: Root directory for Zarr stores.
        exclude_fold: If set, skip samples whose Fold == this value.
            If None, include ALL samples (no fold filter).
        exclude_cohorts: If set, skip samples whose Cohort is in this set.

    Path construction strips the cohort prefix from TSV sample names for
    Zarr file access only (files on disk use bare names), mirroring
    build_loaders_from_splits.
    """
    paths: List[str] = []
    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if exclude_fold is not None and int(row['Fold']) == exclude_fold:
                continue
            cohort = row['Cohort'].strip()
            if exclude_cohorts and cohort in exclude_cohorts:
                continue
            sample = row['Sample'].strip()
            zarr_name = sample
            if sample.startswith(f"{cohort}_"):
                zarr_name = sample[len(cohort) + 1:]
            zarr_path = os.path.join(zarr_base, cohort, 'zarr_files', f"{zarr_name}.zarr")
            if os.path.isdir(zarr_path):
                paths.append(zarr_path)
    return paths


def train_model_cv(
    model_name: str,
    training_set_tsv: str,
    zarr_base: str,
    test_fold: Optional[int] = None,
    out_dir: Optional[str] = None,
    model_config: Optional[BioDVNConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    genes_per_batch: int = 32,
    chunk_size: int = 256,
    shuffle_genes_in_batch: bool = True,
    seed: int = 42,
    subsample_frac: Optional[float] = None,
    epoch_end_callback: Optional[Callable[..., None]] = None,
    external_stop_check: Optional[Callable[..., bool]] = None,
    train_only_cohorts: Optional[List[str]] = None,
    balance_cohorts: Optional[List[str]] = None,
    source_groups: Optional[Dict[str, List[str]]] = None,
    target_n_per_source: Optional[int] = None,
):
    """Train a model using cross-validation with deterministic fold assignment.

    Args:
        model_name: Name of the model class.
        training_set_tsv: Path to TSV file with Cohort/Sample/Fold columns.
        zarr_base: Base directory for Zarr stores ({zarr_base}/{cohort}/{sample}.zarr).
        test_fold: If set, only train this fold (for HPC parallelization).
        out_dir: Output directory for checkpoints/predictions.
        model_config: Fully-specified BioDVNConfig.
        training_config: Fully-specified TrainingConfig.
        genes_per_batch: Genes per batch (32 invariant).
        chunk_size: Genes per chunk (256 invariant).
        shuffle_genes_in_batch: Whether to shuffle genes within batch.
        seed: Random seed for reproducibility.
        subsample_frac: Optional fraction (0.0-1.0) of genes for fast sanity checks.

    Raises:
        FileNotFoundError: If training_set_tsv or Zarr stores don't exist.
        ValueError: If TSV format is invalid or no samples in folds.
    """

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

    # Compute full-dataset (all folds) gene variance once for GVW evaluation.
    # This gives validation metrics a stable, unbiased variance estimate.
    # PHT samples are excluded: their truth labels duplicate their parent PDX
    # samples, so including them would inflate sample counts and bias variance.
    from proteus_utils.proteus_utils import compute_gene_weights_from_zarr, compute_gene_variances_unclipped
    print('Computing full-dataset gene variance for GVW evaluation (excluding PHT)...')
    all_zarr_paths = _collect_zarr_paths(training_set_tsv, zarr_base, exclude_cohorts={'PHT'})
    full_tumor_variance = compute_gene_variances_unclipped(
        all_zarr_paths, label_key='flat/tumor_exp',
    )
    if not full_tumor_variance:
        raise ValueError(
            f"No finite tumor expression values found across {len(all_zarr_paths)} "
            f"samples in {training_set_tsv} — cannot compute GVW variance weights"
        )
    print(f'  Full-dataset tumor variance: {len(full_tumor_variance)} genes, '
          f'n={len(all_zarr_paths)} samples, '
          f'range [{min(full_tumor_variance.values()):.3f}, {max(full_tumor_variance.values()):.3f}]')

    # Background variance: PHT samples are the only cohort with real
    # background expression labels (from HD-L).  Non-PHT zarr stores have
    # all-NaN flat/background_exp, which the variance computation skips.
    # We use ALL samples here (not the PHT-excluded set) so PHT's HD-L
    # background labels contribute.  Background variance is used only for
    # validation GVW metrics, not for training gene weights.
    all_zarr_paths_with_pht = _collect_zarr_paths(training_set_tsv, zarr_base)
    full_background_variance: Optional[Dict[str, float]] = compute_gene_variances_unclipped(
        all_zarr_paths_with_pht, label_key='flat/background_exp',
    ) or None
    if full_background_variance is not None:
        print(f'  Full-dataset background variance: {len(full_background_variance)} genes, '
              f'range [{min(full_background_variance.values()):.3f}, {max(full_background_variance.values()):.3f}]')
    else:
        print('  No background expression labels found — background GVW weighting disabled')

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

    # When source_groups is set, build sample→source mapping and compute
    # per-source gene variance for GVW-CRPS / GVW-r evaluation.
    # PHT samples are assigned to their parent PDX source using the same
    # regex logic as build_loaders_from_splits.
    sample_to_source: Optional[Dict[str, str]] = None
    source_tumor_variance: Optional[Dict[str, Dict[str, float]]] = None
    if source_groups:
        import re
        # Invert source_groups: cohort → source name.
        _cohort_to_source: Dict[str, str] = {}
        for src_name, cohorts in source_groups.items():
            for coh in cohorts:
                _cohort_to_source[coh] = src_name

        # Collect known PDX sample names per cohort (for matching PHT names).
        _pdx_sample_to_cohort: Dict[str, str] = {}
        with open(training_set_tsv, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                coh = row['Cohort'].strip()
                samp = row['Sample'].strip()
                if samp.startswith(f"{coh}_"):
                    samp = samp[len(coh) + 1:]
                if coh in _cohort_to_source:
                    _pdx_sample_to_cohort[samp] = coh

        _pht_re = re.compile(r'^PHT-Fold-\d+_(.+)-BP\d+')
        sample_to_source = {}
        _source_zarr_paths: Dict[str, List[str]] = {src: [] for src in source_groups}

        with open(training_set_tsv, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                cohort = row['Cohort'].strip()
                sample = row['Sample'].strip()

                # Zarr files on disk use bare names (without cohort prefix).
                zarr_name = sample
                if sample.startswith(f"{cohort}_"):
                    zarr_name = sample[len(cohort) + 1:]

                if cohort in _cohort_to_source:
                    source = _cohort_to_source[cohort]
                elif cohort == 'PHT':
                    m = _pht_re.match(sample)
                    if m:
                        pdx_name = m.group(1)
                        pdx_coh = _pdx_sample_to_cohort.get(pdx_name)
                        if pdx_coh is not None:
                            source = _cohort_to_source[pdx_coh]
                        else:
                            raise ValueError(
                                f"PHT sample '{sample}' has PDX name '{pdx_name}' "
                                f"that does not match any known PDX sample. "
                                f"Known: {sorted(_pdx_sample_to_cohort.keys())}"
                            )
                    else:
                        raise ValueError(
                            f"Could not parse PDX name from PHT sample: '{sample}'. "
                            f"Expected format: PHT-Fold-X_<PDX_NAME>-BP..."
                        )
                else:
                    continue  # cohort not in any source group

                sample_to_source[sample] = source
                zarr_path = os.path.join(
                    zarr_base, cohort, 'zarr_files', f"{zarr_name}.zarr",
                )
                if os.path.isdir(zarr_path):
                    # Only non-PHT samples contribute to per-source variance.
                    # PHT truth labels duplicate their parent PDX, so
                    # including them would double-count and bias variance.
                    if cohort != 'PHT':
                        _source_zarr_paths[source].append(zarr_path)

        # Compute per-source unclipped tumor variance (all 5 folds).
        print('Computing per-source gene variance for GVW evaluation...')
        source_tumor_variance = {}
        for src_name in sorted(source_groups):
            src_paths = _source_zarr_paths[src_name]
            if src_paths:
                source_tumor_variance[src_name] = compute_gene_variances_unclipped(
                    src_paths, label_key='flat/tumor_exp',
                )
                vals = list(source_tumor_variance[src_name].values())
                print(
                    f'  Source "{src_name}": {len(vals)} genes, '
                    f'variance range [{min(vals):.3f}, {max(vals):.3f}], '
                    f'{len(src_paths)} samples'
                )
            else:
                print(f'  Source "{src_name}": no zarr paths found — skipping')

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
        train_zarr_paths = _collect_zarr_paths(
            training_set_tsv, zarr_base, exclude_fold=fold_num,
            exclude_cohorts={'PHT'},
        )
        train_fold_variance_raw = compute_gene_variances_unclipped(train_zarr_paths, label_key='flat/tumor_exp')
        gene_weights = compute_gene_weights_from_zarr(
            train_zarr_paths,
            label_key='flat/tumor_exp',
        )
        _raw_vals = list(train_fold_variance_raw.values())
        print(f'  {len(gene_weights)} genes, n={len(train_zarr_paths)} samples — '
              f'raw variance [{min(_raw_vals):.4f}, {max(_raw_vals):.4f}], '
              f'clipped to [0.100, 10.000]')

        # Build data loaders from TSV
        print('Building data loaders...')
        train_loader, val_loader, train_steps, val_steps = build_loaders_from_splits(
            training_set_tsv=training_set_tsv,
            zarr_base=zarr_base,
            val_fold=fold_num,
            num_workers=_loader_workers(),
            loader_mode='chunk',
            chunk_size=chunk_size,
            genes_per_batch=genes_per_batch,
            shuffle_genes_in_batch=shuffle_genes_in_batch,
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
        trainer.val_gene_weight_map = full_tumor_variance
        trainer.val_background_gene_weight_map = full_background_variance
        # Per-source gene variance for epoch-level per-source GVW-CRPS/GVW-r.
        trainer.val_source_gene_weight_map = source_tumor_variance
        # Enable epoch-level per-group CRPS monitoring.  When source_groups
        # is active, group by source (4 groups); otherwise by cohort.
        trainer.val_sample_cohort_map = sample_to_source or sample_to_cohort or None
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=fold_training_config.epochs,
            epoch_end_callback=epoch_end_callback,
            external_stop_check=external_stop_check,
        )

        # Save training history for plotting / audit.
        try:
            with open(os.path.join(fold_dir, 'training_history.json'), 'w') as f:
                json.dump(history, f, indent=2)
        except TypeError as e:
            raise TypeError(f'Failed to JSON-serialize training history for fold {fold_num}: {e}')

        # Prefer the trainer's model reference (may be compiled).
        model_for_eval = trainer.model
        model_for_save = getattr(model_for_eval, '_orig_mod', model_for_eval)
        
        # Save final weights
        torch.save(model_for_save.state_dict(), os.path.join(fold_dir, f'validation-fold_{fold_num}_weights.pt'))
        
        # Generate and save predictions
        _save_fold_predictions(
            model_for_eval,
            val_loader,
            fold_dir,
            fold_num,
            device,
            full_tumor_variance=full_tumor_variance,
            full_background_variance=full_background_variance,
            sample_to_cohort=sample_to_cohort if sample_to_cohort else None,
            sample_to_source=sample_to_source,
            source_tumor_variance=source_tumor_variance,
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

        row = {
            'fold': int(fold_num),
            # Primary selection metric
            'score': float(t_metrics.get('gvw_crps', float('nan'))),
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
    agg: Dict[str, Any] = {
        'primary_metric': {'name': 'tumor_gvw_crps', 'direction': 'minimize'},
    }
    for c in numeric_cols:
        vals = pd.to_numeric(summary_df[c], errors='coerce').to_numpy(dtype=np.float64)
        agg[c] = {
            'mean': float(np.nanmean(vals)),
            'std': float(np.nanstd(vals)),
        }

    # Best fold by primary metric
    if 'tumor_gvw_crps' in summary_df.columns:
        s = pd.to_numeric(summary_df['tumor_gvw_crps'], errors='coerce')
        if np.isfinite(s.to_numpy(dtype=np.float64)).any():
            best_idx = int(np.nanargmin(s.to_numpy(dtype=np.float64)))
            best_row = summary_df.iloc[best_idx].to_dict()
            agg['best'] = {
                'fold': int(best_row.get('fold', -1)),
                'tumor_gvw_crps': float(best_row.get('tumor_gvw_crps', float('nan'))),
            }
    with open(os.path.join(out_dir, 'cv_metrics_summary.json'), 'w') as f:
        json.dump(agg, f, indent=2)


def _save_fold_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    fold_dir: str,
    fold_num: int,
    device: torch.device,
    full_tumor_variance: Optional[Dict[str, float]] = None,
    full_background_variance: Optional[Dict[str, float]] = None,
    sample_to_cohort: Optional[Dict[str, str]] = None,
    sample_to_source: Optional[Dict[str, str]] = None,
    source_tumor_variance: Optional[Dict[str, Dict[str, float]]] = None,
):
    """Save MDN predictions, full distribution params, and quantile CIs for a validation fold.

    For each observation the output TSV contains:
        truth, prediction_mean, ci_q5, ci_q95,
        pi_delta, pi_gamma, gamma_beta, pi_normal, normal_mean,
        normal_stdev, sample.

    ci_q5 / ci_q95 are CDF-based quantiles F^{-1}(0.05) / F^{-1}(0.95),
    clamped to [0, Y_MAX].
    """
    from models.model_utils import (
        mixture_quantile, mixture_crps, Y_MAX,
    )

    model.eval()

    # ---- Accumulators (tumor) ----
    all_t_pred_mean = []
    all_t_q5 = []
    all_t_q95 = []
    all_t_true = []
    all_t_pi_delta = []
    all_t_pi_gamma = []
    all_t_pi_normal = []
    all_t_gamma_beta = []
    all_t_normal_mu = []
    all_t_normal_sigma = []

    # ---- Accumulators (background) ----
    all_b_pred_mean = []
    all_b_q5 = []
    all_b_q95 = []
    all_b_true = []
    all_b_pi_delta = []
    all_b_pi_gamma = []
    all_b_pi_normal = []
    all_b_gamma_beta = []
    all_b_normal_mu = []
    all_b_normal_sigma = []

    # Metadata
    all_samples = []
    all_genes = []

    # CRPS accumulators
    tumor_crps_sum = torch.tensor(0.0)
    tumor_crps_denom = torch.tensor(0.0)
    bg_crps_sum = torch.tensor(0.0)
    bg_crps_denom = torch.tensor(0.0)

    # Per-cohort CRPS accumulators (for balanced fine-tuning objective)
    _cohort_crps: Dict[str, List[torch.Tensor]] = {}  # cohort -> [t_sum, t_den]

    # Per-source CRPS accumulators (when source_groups is active).
    # Each source uses its own per-source variance as CRPS weights.
    _source_crps: Dict[str, List[torch.Tensor]] = {}  # source -> [t_sum, t_den]

    with torch.no_grad():
        for features, labels in val_loader:
            if 'sample_name' not in features or 'gene_name' not in features:
                raise KeyError(
                    "Validation batches are missing 'sample_name'/'gene_name'. "
                    "This typically means the validation DataLoader was created without include_metadata=True."
                )

            features_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in features.items()}
            labels_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in labels.items()}

            predictions, _ = model(features_device, return_internal_loss=False)

            # ---- Tumor head ----
            t_mdn = predictions['tumor_mdn']
            t_q = mixture_quantile(t_mdn, quantiles=(0.05, 0.95))  # (batch, 2)
            t_pi = t_mdn['log_pi'].exp().cpu().numpy()

            all_t_pred_mean.extend(predictions['tumor_exp'].cpu().numpy().flatten())
            all_t_q5.extend(t_q[:, 0].cpu().numpy().flatten())
            all_t_q95.extend(t_q[:, 1].cpu().numpy().flatten())
            all_t_true.extend(labels_device['tumor_exp'].cpu().numpy().flatten())
            all_t_pi_delta.extend(t_pi[:, 0].flatten())
            all_t_pi_gamma.extend(t_pi[:, 1].flatten())
            all_t_pi_normal.extend(t_pi[:, 2].flatten())
            all_t_gamma_beta.extend(t_mdn['gamma_beta'].cpu().numpy().flatten())
            all_t_normal_mu.extend(t_mdn['normal_mu'].cpu().numpy().flatten())
            all_t_normal_sigma.extend(t_mdn['normal_sigma'].cpu().numpy().flatten())

            # ---- Background head ----
            b_mdn = predictions['background_mdn']
            b_q = mixture_quantile(b_mdn, quantiles=(0.05, 0.95))  # (batch, 2)
            b_pi = b_mdn['log_pi'].exp().cpu().numpy()

            all_b_pred_mean.extend(predictions['background_exp'].cpu().numpy().flatten())
            all_b_q5.extend(b_q[:, 0].cpu().numpy().flatten())
            all_b_q95.extend(b_q[:, 1].cpu().numpy().flatten())
            all_b_true.extend(labels_device['background_exp'].cpu().numpy().flatten())
            all_b_pi_delta.extend(b_pi[:, 0].flatten())
            all_b_pi_gamma.extend(b_pi[:, 1].flatten())
            all_b_pi_normal.extend(b_pi[:, 2].flatten())
            all_b_gamma_beta.extend(b_mdn['gamma_beta'].cpu().numpy().flatten())
            all_b_normal_mu.extend(b_mdn['normal_mu'].cpu().numpy().flatten())
            all_b_normal_sigma.extend(b_mdn['normal_sigma'].cpu().numpy().flatten())

            # ---- GVW-CRPS (tumor) ----
            w_tumor = None
            if full_tumor_variance is not None and 'gene_name' in features:
                gnames = features['gene_name']
                if isinstance(gnames, (list, tuple)):
                    w_tumor = torch.tensor(
                        [float(full_tumor_variance.get(str(g), 1.0)) for g in gnames],
                        dtype=torch.float32, device=device,
                    ).unsqueeze(-1)
                else:
                    w_tumor = torch.tensor(
                        [[float(full_tumor_variance.get(str(gnames), 1.0))]],
                        dtype=torch.float32, device=device,
                    )
            crps_s, crps_d = mixture_crps(t_mdn, labels_device['tumor_exp'], sample_weight=w_tumor)
            tumor_crps_sum += crps_s.cpu()
            tumor_crps_denom += crps_d.cpu()

            # ---- GVW-CRPS (background) ----
            w_bg = None
            if full_background_variance is not None and 'gene_name' in features:
                gnames = features['gene_name']
                if isinstance(gnames, (list, tuple)):
                    w_bg = torch.tensor(
                        [float(full_background_variance.get(str(g), 1.0)) for g in gnames],
                        dtype=torch.float32, device=device,
                    ).unsqueeze(-1)
                else:
                    w_bg = torch.tensor(
                        [[float(full_background_variance.get(str(gnames), 1.0))]],
                        dtype=torch.float32, device=device,
                    )
            bg_crps_s, bg_crps_d = mixture_crps(b_mdn, labels_device['background_exp'], sample_weight=w_bg)
            bg_crps_sum += bg_crps_s.cpu()
            bg_crps_denom += bg_crps_d.cpu()

            # ---- Per-cohort GVW-CRPS (tumor) ----
            if sample_to_cohort and 'sample_name' in features:
                batch_samples = features['sample_name']
                if not isinstance(batch_samples, (list, tuple)):
                    batch_samples = [batch_samples]
                cohort_labels = [sample_to_cohort.get(str(s), 'unknown')
                                 for s in batch_samples]
                for coh in set(cohort_labels):
                    idx = torch.tensor(
                        [i for i, c in enumerate(cohort_labels) if c == coh],
                        dtype=torch.long, device=device,
                    )
                    c_mdn = {k: v[idx] for k, v in t_mdn.items()}
                    c_w = w_tumor[idx] if w_tumor is not None else None
                    cs, cd = mixture_crps(
                        c_mdn, labels_device['tumor_exp'][idx],
                        sample_weight=c_w,
                    )
                    if coh not in _cohort_crps:
                        _cohort_crps[coh] = [torch.tensor(0.0),
                                              torch.tensor(0.0)]
                    _cohort_crps[coh][0] += cs.cpu()
                    _cohort_crps[coh][1] += cd.cpu()

            # ---- Per-source GVW-CRPS (tumor, per-source variance weights) ----
            if sample_to_source and source_tumor_variance and 'sample_name' in features:
                batch_samples = features['sample_name']
                if not isinstance(batch_samples, (list, tuple)):
                    batch_samples = [batch_samples]
                src_label = sample_to_source.get(str(batch_samples[0]))
                if src_label is not None and src_label in source_tumor_variance:
                    src_var = source_tumor_variance[src_label]
                    gnames = features['gene_name']
                    if isinstance(gnames, (list, tuple)):
                        w_src = torch.tensor(
                            [float(src_var.get(str(g), 1.0)) for g in gnames],
                            dtype=torch.float32, device=device,
                        ).unsqueeze(-1)
                    else:
                        w_src = torch.tensor(
                            [[float(src_var.get(str(gnames), 1.0))]],
                            dtype=torch.float32, device=device,
                        )
                    src_cs, src_cd = mixture_crps(
                        t_mdn, labels_device['tumor_exp'],
                        sample_weight=w_src,
                    )
                    if src_label not in _source_crps:
                        _source_crps[src_label] = [
                            torch.tensor(0.0), torch.tensor(0.0),
                        ]
                    _source_crps[src_label][0] += src_cs.cpu()
                    _source_crps[src_label][1] += src_cd.cpu()

            # Metadata
            sample_names = features['sample_name']
            gene_names = features['gene_name']
            if isinstance(sample_names, (list, tuple)):
                all_samples.extend([str(s) for s in sample_names])
            else:
                all_samples.append(str(sample_names))
            if isinstance(gene_names, (list, tuple)):
                all_genes.extend([str(g) for g in gene_names])
            else:
                all_genes.append(str(gene_names))

    # ---- Convert to arrays ----
    truth_t = np.asarray(all_t_true, dtype=np.float64)
    pred_mean_t = np.asarray(all_t_pred_mean, dtype=np.float64)
    q5_t = np.asarray(all_t_q5, dtype=np.float64)
    q95_t = np.asarray(all_t_q95, dtype=np.float64)

    truth_b = np.asarray(all_b_true, dtype=np.float64)
    pred_mean_b = np.asarray(all_b_pred_mean, dtype=np.float64)
    q5_b = np.asarray(all_b_q5, dtype=np.float64)
    q95_b = np.asarray(all_b_q95, dtype=np.float64)

    # ---- Build and save TSVs ----
    df_tumor = pd.DataFrame({
        'truth': truth_t,
        'prediction_mean': pred_mean_t,
        'ci_q5': q5_t,
        'ci_q95': q95_t,
        'pi_delta': all_t_pi_delta,
        'pi_gamma': all_t_pi_gamma,
        'gamma_beta': all_t_gamma_beta,
        'pi_normal': all_t_pi_normal,
        'normal_mean': all_t_normal_mu,
        'normal_stdev': all_t_normal_sigma,
        'sample': all_samples,
    }, index=all_genes)

    df_background = pd.DataFrame({
        'truth': truth_b,
        'prediction_mean': pred_mean_b,
        'ci_q5': q5_b,
        'ci_q95': q95_b,
        'pi_delta': all_b_pi_delta,
        'pi_gamma': all_b_pi_gamma,
        'gamma_beta': all_b_gamma_beta,
        'pi_normal': all_b_pi_normal,
        'normal_mean': all_b_normal_mu,
        'normal_stdev': all_b_normal_sigma,
        'sample': all_samples,
    }, index=all_genes)

    df_tumor.to_csv(os.path.join(fold_dir, f'validation-fold_{fold_num}_predictions_tumor.tsv'), sep='\t')
    df_background.to_csv(os.path.join(fold_dir, f'validation-fold_{fold_num}_predictions_background.tsv'), sep='\t')

    # ---- Fold-level metrics ----
    t_metrics = compute_regression_metrics(truth_t, pred_mean_t)
    b_metrics = compute_regression_metrics(truth_b, pred_mean_b)

    # GVW-CRPS
    t_metrics['gvw_crps'] = float((tumor_crps_sum / tumor_crps_denom).item()) if float(tumor_crps_denom.item()) > 0 else float('nan')
    b_metrics['gvw_crps'] = float((bg_crps_sum / bg_crps_denom).item()) if float(bg_crps_denom.item()) > 0 else float('nan')

    # Tumor gene/sample summaries
    df_t_tmp = df_tumor.copy()
    df_t_tmp['gene'] = df_t_tmp.index.astype(str)
    gene_r2_t = group_spearman_r2(df_t_tmp, group_col='gene', pred_col='prediction_mean')
    sample_r2_t = group_spearman_r2(df_tumor, group_col='sample', pred_col='prediction_mean')
    t_metrics['gene_r2_mean'] = float(np.nanmean(gene_r2_t)) if gene_r2_t.size else float('nan')
    t_metrics['gene_r2_median'] = float(np.nanmedian(gene_r2_t)) if gene_r2_t.size else float('nan')
    t_metrics['gene_r2_n'] = int(gene_r2_t.size)
    t_metrics['sample_r2_mean'] = float(np.nanmean(sample_r2_t)) if sample_r2_t.size else float('nan')
    t_metrics['sample_r2_median'] = float(np.nanmedian(sample_r2_t)) if sample_r2_t.size else float('nan')
    t_metrics['sample_r2_n'] = int(sample_r2_t.size)

    t_metrics['quantile_coverage'] = coverage(truth_t, q5_t, q95_t)
    b_metrics['quantile_coverage'] = coverage(truth_b, q5_b, q95_b)

    # ---- Per-cohort metrics (for balanced tuning objectives) ----
    per_cohort: Dict[str, Dict[str, Any]] = {}
    if sample_to_cohort:
        samples_arr = np.asarray(all_samples, dtype=object)
        cohort_arr = np.array(
            [sample_to_cohort.get(str(s), 'unknown') for s in samples_arr],
            dtype=object,
        )
        for cohort_name in sorted(set(cohort_arr)):
            mask_c = cohort_arr == cohort_name
            if mask_c.sum() == 0:
                continue
            c_tumor = compute_regression_metrics(
                truth_t[mask_c], pred_mean_t[mask_c],
            )
            c_bg = compute_regression_metrics(
                truth_b[mask_c], pred_mean_b[mask_c],
            )

            # Per-cohort GVW-r: per-gene Pearson r, variance-weighted
            # average using per-cohort truth variance (computed from
            # the genes present in this cohort's validation data).
            c_genes = [g for g, m in zip(all_genes, mask_c) if m]
            c_truth_t = truth_t[mask_c]
            c_pred_t = pred_mean_t[mask_c]
            c_truth_b = truth_b[mask_c]
            c_pred_b = pred_mean_b[mask_c]

            c_tumor['gvw_r'] = gvw_r(c_truth_t, c_pred_t, c_genes)
            c_bg['gvw_r'] = gvw_r(c_truth_b, c_pred_b, c_genes)

            # Per-cohort GVW-CRPS from streaming accumulators
            if cohort_name in _cohort_crps:
                s, d = _cohort_crps[cohort_name]
                c_tumor['gvw_crps'] = (
                    float((s / d).item()) if float(d.item()) > 0
                    else float('nan')
                )
            per_cohort[cohort_name] = {
                'tumor': c_tumor,
                'background': c_bg,
            }

    # ---- Per-source metrics (when source_groups active) ----
    per_source: Dict[str, Dict[str, Any]] = {}
    if sample_to_source and source_tumor_variance:
        samples_arr = np.asarray(all_samples, dtype=object)
        source_arr = np.array(
            [sample_to_source.get(str(s), 'unknown') for s in samples_arr],
            dtype=object,
        )
        for source_name in sorted(set(source_arr)):
            if source_name == 'unknown':
                continue
            mask_s = source_arr == source_name
            if mask_s.sum() == 0:
                continue
            s_tumor = compute_regression_metrics(
                truth_t[mask_s], pred_mean_t[mask_s],
            )
            s_bg = compute_regression_metrics(
                truth_b[mask_s], pred_mean_b[mask_s],
            )

            s_genes = [g for g, m in zip(all_genes, mask_s) if m]
            s_truth_t = truth_t[mask_s]
            s_pred_t = pred_mean_t[mask_s]
            s_truth_b = truth_b[mask_s]
            s_pred_b = pred_mean_b[mask_s]

            # Per-source GVW-r: variance weights from all 5 folds for
            # this source (precomputed in source_tumor_variance).
            src_var = source_tumor_variance.get(source_name)
            s_tumor['gvw_r'] = gvw_r(s_truth_t, s_pred_t, s_genes, external_var=src_var)
            s_bg['gvw_r'] = gvw_r(s_truth_b, s_pred_b, s_genes)

            # Per-source GVW-CRPS from streaming accumulators
            # (accumulated with per-source variance weights).
            if source_name in _source_crps:
                s_sum, s_den = _source_crps[source_name]
                s_tumor['gvw_crps'] = (
                    float((s_sum / s_den).item()) if float(s_den.item()) > 0
                    else float('nan')
                )
            per_source[source_name] = {
                'tumor': s_tumor,
                'background': s_bg,
            }

    metrics = {
        'fold': int(fold_num),
        'tumor': t_metrics,
        'background': b_metrics,
    }
    if per_cohort:
        metrics['per_cohort'] = per_cohort
    if per_source:
        metrics['per_source'] = per_source

    # Balanced metrics: when source mode is active, average across sources
    # (all sources count, including those containing PHT samples).
    # Otherwise, fall back to cohort-level averaging with PH* exclusion.
    if per_source:
        source_gvw_crps = [
            v['tumor'].get('gvw_crps', float('nan'))
            for v in per_source.values()
        ]
        source_gvw_r = [
            v['tumor'].get('gvw_r', float('nan'))
            for v in per_source.values()
        ]
        finite_crps = [x for x in source_gvw_crps if np.isfinite(x)]
        finite_r = [x for x in source_gvw_r if np.isfinite(x)]
        metrics['balanced_tumor_gvw_crps'] = (
            float(np.mean(finite_crps)) if finite_crps else float('nan')
        )
        metrics['balanced_tumor_gvw_crps_std'] = (
            float(np.std(finite_crps)) if len(finite_crps) > 1 else float('nan')
        )
        metrics['balanced_tumor_gvw_r'] = (
            float(np.mean(finite_r)) if finite_r else float('nan')
        )
        metrics['balanced_tumor_gvw_r_std'] = (
            float(np.std(finite_r)) if len(finite_r) > 1 else float('nan')
        )
    elif per_cohort:
        patient_gvw_crps = [
            v['tumor'].get('gvw_crps', float('nan'))
            for k, v in per_cohort.items()
            if not k.startswith('PH')
        ]
        patient_gvw_r = [
            v['tumor'].get('gvw_r', float('nan'))
            for k, v in per_cohort.items()
            if not k.startswith('PH')
        ]
        finite_crps = [x for x in patient_gvw_crps if np.isfinite(x)]
        finite_r = [x for x in patient_gvw_r if np.isfinite(x)]
        metrics['balanced_tumor_gvw_crps'] = (
            float(np.mean(finite_crps)) if finite_crps else float('nan')
        )
        metrics['balanced_tumor_gvw_crps_std'] = (
            float(np.std(finite_crps)) if len(finite_crps) > 1 else float('nan')
        )
        metrics['balanced_tumor_gvw_r'] = (
            float(np.mean(finite_r)) if finite_r else float('nan')
        )
        metrics['balanced_tumor_gvw_r_std'] = (
            float(np.std(finite_r)) if len(finite_r) > 1 else float('nan')
        )

    with open(os.path.join(fold_dir, f'validation-fold_{fold_num}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(
        f'Fold {fold_num} tumor metrics: '
        f'gvw_crps={t_metrics.get("gvw_crps", float("nan")):.4f} '
        f'gene_r2(mean/med)={t_metrics.get("gene_r2_mean", float("nan")):.4f}/'
        f'{t_metrics.get("gene_r2_median", float("nan")):.4f} '
        f'sample_r2(mean/med)={t_metrics.get("sample_r2_mean", float("nan")):.4f}/'
        f'{t_metrics.get("sample_r2_median", float("nan")):.4f}'
    )
    if per_source:
        for src_name, src_data in sorted(per_source.items()):
            crps_str = f'{src_data["tumor"].get("gvw_crps", float("nan")):.4f}'
            r_str = f'{src_data["tumor"].get("gvw_r", float("nan")):.4f}'
            print(f'  source {src_name}: gvw_crps={crps_str} gvw_r={r_str}')
        balanced_crps_str = f'{metrics.get("balanced_tumor_gvw_crps", float("nan")):.4f}'
        balanced_r_str = f'{metrics.get("balanced_tumor_gvw_r", float("nan")):.4f}'
        print(f'  balanced_gvw_crps={balanced_crps_str} balanced_gvw_r={balanced_r_str}')
    elif per_cohort:
        balanced_crps_str = f'{metrics.get("balanced_tumor_gvw_crps", float("nan")):.4f}'
        balanced_r_str = f'{metrics.get("balanced_tumor_gvw_r", float("nan")):.4f}'
        print(f'  balanced_gvw_crps={balanced_crps_str} balanced_gvw_r={balanced_r_str}')

    # ---- Plots (per fold) ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plot_dir = os.path.join(fold_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Helper: true vs predicted scatter plot.
        def _plot_true_vs_pred(truth, pred, head_label, est_label, fname):
            mask = np.isfinite(truth) & np.isfinite(pred)
            x, y_pts = truth[mask], pred[mask]
            if x.size == 0:
                return
            stats = scatter_stats(truth, pred)
            rng = np.random.RandomState(0)
            if x.size > 20000:
                idx = rng.choice(x.size, size=20000, replace=False)
                x, y_pts = x[idx], y_pts[idx]
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(x, y_pts, s=3, alpha=0.25, linewidths=0)
            plt.plot([0.0, Y_MAX], [0.0, Y_MAX], color='black', linewidth=1)
            plt.xlim(0.0, Y_MAX)
            plt.ylim(0.0, Y_MAX)
            plt.xlabel('truth')
            plt.ylabel(f'prediction ({est_label})')
            plt.title(f'Fold {fold_num} {head_label}: true vs pred ({est_label})')
            if stats:
                ann = '\n'.join(f'{k} = {v:.4f}' for k, v in stats.items())
                plt.text(
                    0.03, 0.97, ann, transform=plt.gca().transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'),
                    family='monospace',
                )
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, fname), dpi=150)
            plt.close(fig)

        # Helper: gene-level Pearson r density histogram by variance group.
        def _plot_gene_r_hist(df, pred_col, var_map, head_label, est_label, fname):
            df_tmp = df.copy()
            df_tmp['gene'] = df_tmp.index.astype(str)
            gene_r, gene_names_r = group_pearson_r_named(df_tmp, 'gene', pred_col)
            if gene_r.size == 0:
                return
            # Assign genes to biologically-meaningful variance groups.
            gene_var = np.array(
                [float(var_map.get(g, 0.0)) for g in gene_names_r], dtype=np.float64,
            ) if var_map else np.zeros(len(gene_names_r), dtype=np.float64)
            low_mask = gene_var < 1.0
            mid_mask = (gene_var >= 1.0) & (gene_var < 3.0)
            high_mask = gene_var >= 3.0
            n_total = gene_r.size

            groups = [
                (low_mask, 'Low [0, 1)', '#3b82f6'),
                (mid_mask, 'Med [1, 3)', '#f59e0b'),
                (high_mask, 'High [3, \u221e)', '#ef4444'),
            ]

            fig, ax = plt.subplots(figsize=(5, 5))
            bins = np.linspace(-1.0, 1.0, 61)
            for gmask, glabel, gcolor in groups:
                r_sub = gene_r[gmask]
                if r_sub.size == 0:
                    continue
                pct = 100.0 * r_sub.size / n_total
                ax.hist(
                    r_sub, bins=bins, alpha=0.55, color=gcolor, density=True,
                    label=f'{glabel} ({pct:.0f}%)', edgecolor='white', linewidth=0.3,
                )
            ax.axvline(0.0, color='black', linestyle='--', linewidth=0.8)
            ax.set_xlim(-1.0, 1.0)
            ax.set_xlabel('gene-level Pearson r')
            ax.set_ylabel('density')
            ax.set_title(f'Fold {fold_num} {head_label}: gene Pearson r ({est_label})')
            ax.legend(loc='upper left', fontsize=8, frameon=True)
            ax.set_box_aspect(1)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, fname), dpi=150)
            plt.close(fig)

        # Generate all plots for both heads.
        for head, df, truth, pmean, var_map in [
            ('tumor', df_tumor, truth_t, pred_mean_t, full_tumor_variance),
            ('background', df_background, truth_b, pred_mean_b, full_background_variance),
        ]:
            _plot_true_vs_pred(truth, pmean, head, 'mean',
                               f'{head}_true_vs_pred-mean.png')
            _plot_gene_r_hist(df, 'prediction_mean', var_map, head, 'mean',
                              f'{head}-mean_gene_r_hist.png')

    except (ImportError, OSError, ValueError, RuntimeError) as e:
        print(f'  Warning: failed to generate evaluation plots for fold {fold_num}: {e}')


def full_train_from_splits(
    model_name: str = 'BioDVN',
    training_set_tsv: str = '',
    zarr_base: str = '',
    out_dir: str = 'final_models/',
    model_config: Optional[BioDVNConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    genes_per_batch: int = 32,
    chunk_size: int = 256,
    shuffle_genes_in_batch: bool = True,
    seed: int = 42,
    subsample_frac: Optional[float] = None,
    train_only_cohorts: Optional[List[str]] = None,
    balance_cohorts: Optional[List[str]] = None,
):
    """Train a single model on ALL samples in the training_set TSV (no fold holdout).

    This is for producing a final production model after CV tuning has selected
    hyperparameters.  All samples are used for training.  A deterministic
    gene-level chunk split (10% of 256-gene chunks) provides the early-stopping
    validation signal — every sample and every gene contributes to both train
    and val.  Gene-variance weights are computed from 100% of the data before
    the split.

    When ``train_only_cohorts`` is set, those cohorts (e.g. PHT admixtures)
    are used for training only.  The validation loader draws exclusively from
    patient cohorts so that early-stopping reflects clinical performance.
    When multiple patient cohorts exist, the Trainer computes a balanced
    (equal-weight) per-cohort CRPS for model selection.

    Args:
        model_name: Name of the model class (e.g. 'BioDVN', 'BioDVN_NoSeq').
        training_set_tsv: Path to TSV file with Cohort/Sample/Fold columns.
        zarr_base: Base directory for Zarr stores.
        out_dir: Output directory for final model.
        model_config: Fully-specified BioDVNConfig.
        training_config: Fully-specified TrainingConfig.
        genes_per_batch: Genes per batch (32 invariant).
        chunk_size: Genes per chunk (256 invariant).
        shuffle_genes_in_batch: Whether to shuffle genes within batch.
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

            # Zarr files on disk use bare names (without cohort prefix).
            zarr_name = sample
            if sample.startswith(f"{cohort}_"):
                zarr_name = sample[len(cohort) + 1:]

            zarr_path = os.path.join(zarr_base, cohort, 'zarr_files', f"{zarr_name}.zarr")
            if not os.path.isdir(zarr_path):
                missing_samples.append(
                    f"  {cohort}/zarr_files/{zarr_name}.zarr  (TSV sample={sample})"
                )
                continue
            all_paths.append(zarr_path)
            zarr_cohort_map[zarr_path] = cohort
            zarr_sample_name_map[zarr_path] = sample
            sample_cohort_map[sample] = cohort
            if cohort not in _train_only:
                patient_paths.append(zarr_path)

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
    print('Computing gene variance weights from all samples...')
    from proteus_utils.proteus_utils import compute_gene_weights_from_zarr, compute_gene_variances_unclipped
    train_fold_variance_raw = compute_gene_variances_unclipped(all_paths, label_key='flat/tumor_exp')
    gene_weights = compute_gene_weights_from_zarr(
        all_paths,
        label_key='flat/tumor_exp',
    )
    _raw_vals = list(train_fold_variance_raw.values())
    print(f'  {len(gene_weights)} genes — '
          f'raw variance [{min(_raw_vals):.4f}, {max(_raw_vals):.4f}], '
          f'clipped to [0.100, 10.000]')

    # Create loaders.
    # Train: ALL samples with gene-level 10% chunk split; cohort-balanced
    # sampling so minority cohorts are not under-represented per epoch.
    # Val: PATIENT-ONLY samples with the same chunk split, metadata enabled
    # so the Trainer can compute per-cohort balanced CRPS for early stopping.
    val_frac = 0.10
    train_loader, train_steps = create_data_loader(
        index_file=all_index,
        num_workers=_loader_workers(),
        shuffle=False,
        labeled=True,
        loader_mode='chunk',
        chunk_size=chunk_size,
        genes_per_batch=genes_per_batch,
        shuffle_genes_in_batch=shuffle_genes_in_batch,
        seed=seed,
        drop_last=False,
        subsample_frac=subsample_frac,
        gene_weights=gene_weights,
        val_frac=val_frac,
        val_split=False,
        cohort_map=zarr_cohort_map if balance_cohorts else None,
        balance_cohorts=balance_cohorts,
    )

    val_loader, val_steps = create_data_loader(
        index_file=patient_index,
        num_workers=_loader_workers(),
        shuffle=False,
        labeled=True,
        loader_mode='chunk',
        chunk_size=chunk_size,
        genes_per_batch=genes_per_batch,
        shuffle_genes_in_batch=False,
        seed=seed,
        include_metadata=True,
        drop_last=False,
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
    # Provide sample→cohort mapping so the Trainer can compute per-cohort
    # balanced CRPS for early stopping (patient cohorts only).
    if sample_cohort_map and _train_only:
        patient_cohort_map = {
            k: v for k, v in sample_cohort_map.items() if v not in _train_only
        }
        trainer.val_sample_cohort_map = patient_cohort_map
    history = trainer.fit(train_loader, val_loader, epochs=training_config.epochs)

    # Save final model
    model_for_save = getattr(trainer.model, '_orig_mod', trainer.model)
    final_path = os.path.join(out_dir, 'final_model.pt')
    torch.save(model_for_save.state_dict(), final_path)
    print(f'Saved final model to: {final_path}')


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
        model_config = BioDVNConfig(**cfg)
        print(f'Loaded model config from: {cfg_path}')
    else:
        model_config = BioDVNConfig()
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


def _build_configs_from_spec(experiment_spec, args) -> tuple:
    """Build BioDVNConfig and TrainingConfig from an experiment spec.

    Returns (model_cfg, train_cfg).
    """
    model_cfg = BioDVNConfig(
        encoder_start=int(experiment_spec.scale.encoder_start.fixed),
        latent_size=int(experiment_spec.scale.latent_size.fixed),
        decoder_start=int(experiment_spec.scale.decoder_start.fixed),
        mdn_hidden=int(experiment_spec.scale.mdn_hidden.fixed),
        mdn_final=int(experiment_spec.scale.mdn_final.fixed),
        signal_filters=int(experiment_spec.scale.signal_filters.fixed),
        signal_encoder_width=int(experiment_spec.scale.signal_encoder_width.fixed),
        signal_ksize=int(experiment_spec.scale.signal_ksize.fixed),
        interaction_ksize=int(experiment_spec.scale.interaction_ksize.fixed),
        initial_reduction_channels=int(experiment_spec.scale.initial_reduction_channels.fixed),
        n_reductions=int(experiment_spec.scale.n_reductions.fixed),
        reduction_dilation=int(experiment_spec.scale.reduction_dilation.fixed),
        beta_background=float(experiment_spec.structural.beta_background.fixed),
        beta_tcvae=float(experiment_spec.structural.beta_tcvae.fixed),
        beta_contrastive=float(experiment_spec.structural.beta_contrastive.fixed),
        beta_reconstruction=float(experiment_spec.structural.beta_reconstruction.fixed),
        beta_grl=float(experiment_spec.structural.beta_grl.fixed),
        beta_point_mse=float(experiment_spec.structural.beta_point_mse.fixed),
        cl_margin=float(experiment_spec.structural.cl_margin.fixed),
        internal_weight=float(experiment_spec.structural.internal_weight.fixed),
        dropout_rate=float(experiment_spec.scale.dropout_rate.fixed),
        weight_decay=float(experiment_spec.training.weight_decay.fixed),
        max_lr=float(experiment_spec.training.max_lr.fixed),
        max_grad_norm=float(experiment_spec.training.max_grad_norm.fixed),
    )

    # Resolve transfer-learning parameters.  freeze_mode and
    # backbone_lr_multiplier are ParamSpec; extract their .fixed values.
    _freeze_mode_raw = experiment_spec.transfer.freeze_mode.fixed
    _freeze_mode = str(_freeze_mode_raw) if _freeze_mode_raw is not None else None
    _backbone_lr_mult = float(experiment_spec.transfer.backbone_lr_multiplier.fixed)

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
        runtime_device_banner=bool(args.runtime_device_banner),
        base_weights=experiment_spec.transfer.base_weights,
        freeze_level=experiment_spec.transfer.freeze_level,
        freeze_mode=_freeze_mode,
        backbone_lr_multiplier=_backbone_lr_mult,
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
    parser.add_argument("--run_mode", type=str, default='train', choices=['train', 'full-train', 'tune', 'run'], help="train=CV training, full-train=final model, tune=Optuna CV tuning, run=inference")
    parser.add_argument("--test_fold", type=int, default=None, help="Specific fold to train (for HPC parallelization)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")

    # Experiment / tuning
    parser.add_argument("--experiment", type=str, default=None, help="Path to experiment YAML (required for train/full-train/tune modes)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials for --run_mode tune")
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
    
    # Initialize ConfigLoader for YAML-based dataset configurations
    config_loader = ConfigLoader()
    
    # Helper function to load dataset config.
    def get_dataset_config(dataset_name: str):
        """Load dataset config from YAML or return None if not found."""
        try:
            return config_loader.load_dataset(dataset_name)
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
            datasets_to_process = sorted(f.stem for f in config_loader.datasets_dir.glob('*.yaml'))
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
                available = [f.stem for f in config_loader.datasets_dir.glob('*.yaml')]
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
                config_loader=config_loader,
                dataset_names=datasets_to_check,
                zarr_base=zarr_base,
                raw_data_base=raw_data_dir,
                labels_dir=labels_dir,
                ref_path=args.scalars_reference_path or default_ref_path,
                n_workers=0  # 0 = auto-detect (use all available CPUs)
            )

    # NOTE: transfer learning (including freeze_level) is HP-controlled in experiment YAML.
    # CLI --base_weights is a manual warm-start convenience only.
    
    # Training modes
    if args.run_mode == 'train':
        if experiment_spec is None:
            raise ValueError(
                "--run_mode train requires --experiment <path/to/experiment.yaml>\n"
                "Training now requires an experiment YAML that defines the training_set TSV."
            )

        # Resolve fixed parameters from experiment spec
        model_cfg, train_cfg = _build_configs_from_spec(experiment_spec, args)

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
            genes_per_batch=FIXED_GENES_PER_BATCH,
            chunk_size=FIXED_CHUNK_SIZE,
            shuffle_genes_in_batch=True,
            seed=experiment_spec.data.seed,
            subsample_frac=args.subsample,
            train_only_cohorts=experiment_spec.data.train_only_cohorts,
            balance_cohorts=experiment_spec.data.balance_cohorts,
        )
    
    elif args.run_mode == 'full-train':
        if experiment_spec is None:
            raise ValueError(
                "--run_mode full-train requires --experiment <path/to/experiment.yaml>\n"
                "Full training now requires an experiment YAML that defines the training_set TSV."
            )

        # For full-train, we train on ALL samples in the TSV (no fold held out)
        # This produces a single model using all available data.
        model_cfg, train_cfg = _build_configs_from_spec(experiment_spec, args)

        out_dir = os.path.join(experiment_spec.output_dir, 'full_train')

        full_train_from_splits(
            model_name=args.model_name,
            training_set_tsv=experiment_spec.data.training_set,
            zarr_base=experiment_spec.data.zarr_base,
            out_dir=out_dir,
            model_config=model_cfg,
            training_config=train_cfg,
            genes_per_batch=FIXED_GENES_PER_BATCH,
            chunk_size=FIXED_CHUNK_SIZE,
            shuffle_genes_in_batch=True,
            seed=experiment_spec.data.seed,
            subsample_frac=args.subsample,
            train_only_cohorts=experiment_spec.data.train_only_cohorts,
            balance_cohorts=experiment_spec.data.balance_cohorts,
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

        model_path = os.path.join(model_dir, 'best_model.pt')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
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
