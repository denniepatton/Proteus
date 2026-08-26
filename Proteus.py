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
    python Proteus.py --run_mode train --experiment config/experiments/tds_cv.yaml
    
    # Train specific fold (for HPC parallelization)
    python Proteus.py --run_mode train --experiment config/experiments/tds_cv.yaml --test_fold 1
    
    # Full training (all data, no CV holdout)
    python Proteus.py --run_mode full-train --experiment config/experiments/tds_final-training.yaml
    
    # Optuna HP tuning
    python Proteus.py --run_mode tune --experiment config/experiments/TDS_tuning/biodvn_tune-tds-v2_finalize.yaml --n_trials 50
    
    # Run inference on a dataset using a shipped checkpoint
    python Proteus.py --run_mode run --model_name BioDVN --use_set PHS
    python Proteus.py --run_mode run --model_name BioDVN_NoSeq --use_set Crompton
"""

# Standard library imports
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch

# cuDNN autotuning selects the fastest convolution kernels for the pipeline's
# fixed input shapes, improving training/inference throughput. It makes kernel
# selection nondeterministic, so runs are not guaranteed to be bit-for-bit
# reproducible even with a fixed seed; seeding still controls data ordering,
# weight initialization, and stochastic augmentation.
torch.backends.cudnn.benchmark = True

# Local imports
from config import (
    resolve_labels_dir,
    resolve_raw_data_dir,
    resolve_reference_path,
    resolve_zarr_base,
)
from config.dataset_config import load_dataset_config, list_datasets
from config.experiment_spec import load_experiment_spec
from proteus_workflows.data import (
    ensure_datasets_exist,
    generate_data,
    verify_dataset_stores,
)
from proteus_workflows.train import (
    apply_cli_training_overrides,
    build_training_configs_from_spec,
    full_train_from_splits,
    train_model_cv,
)
from proteus_workflows.inference import run_saved_model_on_dataset


DEFAULT_OPTUNA_STORAGE = 'sqlite:///results_training/optuna.db'


def _configure_streaming_stdout() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)


def _build_configs_from_spec(
    experiment_spec,
    *,
    model_name: str = 'BioDVN',
    verbose: bool = False,
    runtime_device_banner: bool = False,
    base_weights_override: Optional[str] = None,
) -> tuple:
    model_cfg, train_cfg = build_training_configs_from_spec(
        experiment_spec,
        model_name=model_name,
        runtime_device_banner=runtime_device_banner,
        base_weights_override=base_weights_override,
    )
    return model_cfg, apply_cli_training_overrides(
        train_cfg,
        verbose=verbose,
        runtime_device_banner=runtime_device_banner,
    )


def _load_optional_dataset_config(dataset_name: str):
    try:
        return load_dataset_config(dataset_name)
    except FileNotFoundError:
        return None


def _resolve_optuna_study_name(
    experiment_spec,
    study_name_override: Optional[str],
) -> Optional[str]:
    if study_name_override is not None:
        return str(study_name_override)
    if experiment_spec is None:
        return None
    return str(experiment_spec.name)


def _resolve_optuna_storage(
    experiment_spec,
    storage_override: str,
) -> str:
    if experiment_spec is None:
        return str(storage_override)
    if str(storage_override) != DEFAULT_OPTUNA_STORAGE:
        return str(storage_override)

    tuning_phase = getattr(getattr(experiment_spec, 'tuning', None), 'phase', None)
    if tuning_phase is None:
        return str(storage_override)

    phase_name = str(tuning_phase).strip()
    if not phase_name:
        return str(storage_override)

    return f'sqlite:///results_training/optuna_{phase_name}.db'


def main():
    """Main entry point for Proteus PyTorch pipeline."""
    _configure_streaming_stdout()

    parser = argparse.ArgumentParser(
        description='Run Proteus - cfDNA to gene-expression data prep, training, tuning, and inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Process raw data to Zarr stores
    python Proteus.py --process_data --use_set PC-PDX
    
    # Verify compiled Zarr stores
    python Proteus.py --verify_data --use_set PC-PDX
    
    # Train with cross-validation from an experiment YAML
    python Proteus.py --run_mode train --experiment config/experiments/tds_cv.yaml

    # Full training with an optional fine-tune warm start
    python Proteus.py --run_mode full-train --experiment config/experiments/tds_final-training.yaml --base_weights trained_models/proteus_tds-model/full_train/final_model.pt

    # Optuna CV tuning
    python Proteus.py --run_mode tune --experiment config/experiments/TDS_tuning/biodvn_tune-tds-v2_finalize.yaml --n_trials 50
    
    # Run inference on a dataset using a shipped checkpoint
    python Proteus.py --run_mode run --model_name BioDVN --use_set PHS
    """
    )
    
    # Data processing arguments
    parser.add_argument('--process_data', action='store_true', help='Process raw Triton outputs into Zarr stores')
    parser.add_argument('--verify_data', action='store_true', help='Validate Zarr schema and array integrity for one dataset')
    
    # Dataset configuration
    parser.add_argument('--use_set', type=str, default=None, help="Dataset name for --process_data, --verify_data, or --run_mode run. Use 'all' with --process_data to process every configured dataset.")
    parser.add_argument('--raw_data_dir', type=str, default=None, help='Override the raw Triton base directory. Defaults to PROTEUS_RAW_DATA_DIR, config/paths.yaml, or data/raw_triton.')
    parser.add_argument('--zarr_output_dir', type=str, default=None, help='Override the Zarr base directory. Defaults to PROTEUS_ZARR_BASE, config/paths.yaml, or data/zarr.')
    parser.add_argument('--scalars_reference_path', type=str, default=None, help='Override the reference TSV path. Defaults to config/paths.yaml or reference/MANE.GRCh38.v1.3.UniversalReference.tsv.')
    
    # Training arguments
    parser.add_argument('--model_name', type=str, default='BioDVN', help='Model architecture name')
    parser.add_argument('--run_mode', type=str, default='train', choices=['train', 'full-train', 'tune', 'tune-rescue', 'tune-ask', 'tune-tell', 'tune-fold', 'tune-proxy-ask', 'tune-proxy-tell', 'tune-proxy-finalize', 'run'], help='train=CV training, full-train=final model, tune=Optuna CV tuning, tune-rescue=backfill completed Optuna trials from saved fold artifacts, tune-ask/tune-tell/tune-fold=coordinated fold-parallel tuning, tune-proxy-ask/tune-proxy-tell/tune-proxy-finalize=proxy-fold racing, run=inference')
    parser.add_argument('--test_fold', type=int, default=None, help='Specific fold to train or tune (for HPC parallelization)')

    # Experiment / tuning
    parser.add_argument('--experiment', type=str, default=None, help='Path to experiment YAML (required for train/full-train/tune modes)')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials for --run_mode tune')
    parser.add_argument('--n_configs', type=int, default=4, help='Number of parallel config groups for --run_mode tune-ask')
    parser.add_argument('--trial_number', type=int, default=None, help='Trial number for --run_mode tune-fold')
    parser.add_argument('--trial_numbers', type=str, default=None, help='Comma-separated trial numbers for --run_mode tune-tell (default: read from coordinated_active.json)')
    parser.add_argument('--optuna_storage', type=str, default=DEFAULT_OPTUNA_STORAGE, help='Optuna storage URL')
    parser.add_argument('--study_name', type=str, default=None, help='Optuna study name override (default: experiment.name)')
    parser.add_argument('--sampler_seed', type=int, default=42, help='Optuna sampler seed')

    # Manual warm-start only (transfer learning is HP-controlled in experiment YAML)
    parser.add_argument('--base_weights', type=str, default=None, help='Optional warm-start checkpoint for train/full-train. Overrides transfer.base_weights from the experiment YAML.')
    parser.add_argument('--runtime_device_banner', action='store_true', help='Print device and memory info at runtime')
    parser.add_argument('--verbose', action='store_true', help='Enable progress bars, batch timing, and more frequent training logs')
    
    # Sanity check / fast iteration
    parser.add_argument('--subsample', type=float, default=None, help='Subsample fraction (0.0-1.0) of genes for fast sanity checks, e.g. --subsample 0.1 for ~10%% of genes per epoch.')
    # General options
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing data files')
    args = parser.parse_args()
    experiment_spec = load_experiment_spec(args.experiment) if args.experiment is not None else None
    resolved_optuna_storage = _resolve_optuna_storage(experiment_spec, args.optuna_storage)
    resolved_study_name = _resolve_optuna_study_name(experiment_spec, args.study_name)
    resolved_sampler_seed = int(args.sampler_seed)
    zarr_base = resolve_zarr_base(args.zarr_output_dir)
    labels_dir = resolve_labels_dir()
    reference_path = resolve_reference_path(args.scalars_reference_path)

    def _require_dataset_name() -> str:
        if args.use_set is None:
            raise ValueError('--use_set is required for --process_data, --verify_data, and --run_mode run')
        return args.use_set

    if args.verify_data:
        dataset_name = _require_dataset_name()
        verification_df = verify_dataset_stores(dataset_name, zarr_base)
        if 'valid' in verification_df.columns and not bool(verification_df['valid'].all()):
            invalid_files = verification_df.loc[~verification_df['valid'], 'file'].tolist()
            print('\nInvalid Zarr stores:')
            for file_name in invalid_files[:20]:
                print(f'  {file_name}')
            if len(invalid_files) > 20:
                print(f'  ... and {len(invalid_files) - 20} more')
            sys.exit(1)
        return

    if args.process_data:
        dataset_name = _require_dataset_name()
        if dataset_name.lower() == 'all':
            datasets_to_process = list_datasets()
            if not datasets_to_process:
                print('No dataset configs found in config/datasets/')
                sys.exit(1)
            print(f'Processing all configured datasets: {datasets_to_process}')
        else:
            datasets_to_process = [dataset_name]

        raw_data_dir = resolve_raw_data_dir(args.raw_data_dir)
        for current_dataset in datasets_to_process:
            dataset_config = _load_optional_dataset_config(current_dataset)
            if dataset_config is None:
                available = list_datasets()
                print(f"Dataset '{current_dataset}' is not configured.")
                print(f'Available datasets: {available}')
                print(f'Add a YAML config file at: config/datasets/{current_dataset}.yaml')
                sys.exit(1)

            tss_dir = str(Path(raw_data_dir) / dataset_config.triton_tss)
            gb_dir = str(Path(raw_data_dir) / dataset_config.triton_gb)

            print(f'Processing {current_dataset}...')
            print(f'  Config loaded from: config/datasets/{current_dataset}.yaml')
            generate_data(
                dataset=current_dataset,
                tss_dir=tss_dir,
                gb_dir=gb_dir,
                labels_dir=labels_dir,
                ref_path=reference_path,
                tfx_file=dataset_config.tfx_file,
                zarr_base_dir=zarr_base,
                tumor_exp_file=dataset_config.tumor_exp_file,
                background_exp_file=dataset_config.background_exp_file,
                overwrite=args.overwrite,
            )

        print('Finished processing data.')
        return

    if args.run_mode != 'run' and args.use_set:
        dataset_config = _load_optional_dataset_config(args.use_set)
        if dataset_config is not None:
            ensure_datasets_exist(
                dataset_names=[args.use_set],
                zarr_base=zarr_base,
                raw_data_base=resolve_raw_data_dir(args.raw_data_dir),
                labels_dir=labels_dir,
                ref_path=reference_path,
                n_workers=0,
            )
    
    # Training modes
    if args.run_mode == 'train':
        if experiment_spec is None:
            raise ValueError(
                "--run_mode train requires --experiment <path/to/experiment.yaml>\n"
                "Training now requires an experiment YAML that defines the training_set TSV."
            )

        # Resolve fixed parameters from experiment spec
        model_cfg, train_cfg = _build_configs_from_spec(
            experiment_spec,
            model_name=args.model_name,
            verbose=bool(args.verbose),
            runtime_device_banner=bool(args.runtime_device_banner),
            base_weights_override=args.base_weights,
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
            source_groups=experiment_spec.data.source_groups,
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
            experiment_spec,
            model_name=args.model_name,
            verbose=bool(args.verbose),
            runtime_device_banner=bool(args.runtime_device_banner),
            base_weights_override=args.base_weights,
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
            source_groups=experiment_spec.data.source_groups,
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
                storage=str(resolved_optuna_storage),
                n_trials=(int(args.n_trials) if args.n_trials else None),
                study_name=resolved_study_name,
                sampler_seed=int(resolved_sampler_seed),
            )
        else:
            run_optuna_cv_tuning(
                experiment_path=str(args.experiment),
                n_trials=int(args.n_trials),
                storage=str(resolved_optuna_storage),
                study_name=resolved_study_name,
                sampler_seed=int(resolved_sampler_seed),
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
            storage=str(resolved_optuna_storage),
            study_name=resolved_study_name,
            trial_numbers=trial_nums,
            sampler_seed=int(resolved_sampler_seed),
        )
        print(f"RESCUED_TRIAL_NUMBERS={','.join(str(t) for t in rescued_trials)}", flush=True)

    elif args.run_mode == 'tune-ask':
        if args.experiment is None:
            raise ValueError("--run_mode tune-ask requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import coordinated_ask

        trial_numbers = coordinated_ask(
            experiment_path=str(args.experiment),
            n_configs=int(args.n_configs),
            storage=str(resolved_optuna_storage),
            study_name=resolved_study_name,
            sampler_seed=int(resolved_sampler_seed),
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
            storage=str(resolved_optuna_storage),
            study_name=resolved_study_name,
            sampler_seed=int(resolved_sampler_seed),
        )

    elif args.run_mode == 'tune-proxy-ask':
        if args.experiment is None:
            raise ValueError("--run_mode tune-proxy-ask requires --experiment <path/to/experiment.yaml>")

        from tuning.optuna_tuner import coordinated_proxy_ask

        trial_numbers = coordinated_proxy_ask(
            experiment_path=str(args.experiment),
            n_configs=int(args.n_configs),
            storage=str(resolved_optuna_storage),
            study_name=resolved_study_name,
            sampler_seed=int(resolved_sampler_seed),
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
            storage=str(resolved_optuna_storage),
            study_name=resolved_study_name,
            sampler_seed=int(resolved_sampler_seed),
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
            storage=str(resolved_optuna_storage),
            study_name=resolved_study_name,
            sampler_seed=int(resolved_sampler_seed),
            verbose=bool(args.verbose),
            runtime_device_banner=bool(args.runtime_device_banner),
        )
    
    elif args.run_mode == 'run':
        run_saved_model_on_dataset(
            model_name=args.model_name,
            dataset_name=_require_dataset_name(),
            zarr_base=zarr_base,
            experiment_output_dir=(experiment_spec.output_dir if experiment_spec is not None else None),
        )
    
    else:
        print('Invalid run mode. Exiting.')
        sys.exit(1)


if __name__ == "__main__":
    main()
