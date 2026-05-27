"""Training workflows extracted from the CLI entrypoint."""

from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config.runtime_paths import normalize_path
from models import BioDVNConfig, channel_mode_for_model, get_model_instance
from proteus_utils import (
    ProteusHoldoutValDataset,
    ProteusMultiSourceBatchDataset,
    Trainer,
    TrainingConfig,
    build_loaders_from_splits,
    build_multisource_loaders_from_splits,
    compute_regression_metrics,
    create_single_item_loader,
    create_data_loader,
    get_source_weighted_selection_group_weights,
    invert_source_groups,
    save_fold_predictions,
    validate_source_weighted_selection_groups,
)
from proteus_utils.data_helpers import (
    build_source_map_from_tsv,
    collect_pdx_names,
    collect_zarr_paths,
    compute_source_variance,
)


def _read_split_rows(training_set_tsv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(training_set_tsv, 'r', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            rows.append({str(key): str(value) for key, value in row.items()})
    return rows


def _split_folds_and_sample_map(split_rows: List[Dict[str, str]]) -> Tuple[List[int], Dict[str, str]]:
    fold_list = sorted({int(row['Fold']) for row in split_rows})
    sample_to_cohort = {
        row['Sample'].strip(): row['Cohort'].strip()
        for row in split_rows
    }
    return fold_list, sample_to_cohort


def _write_json_file(path: str, payload: Any, *, context: str) -> None:
    try:
        with open(path, 'w') as handle:
            json.dump(payload, handle, indent=2)
    except TypeError as exc:
        raise TypeError(f'Failed to JSON-serialize {context}: {exc}')


def _write_training_metadata(
    output_dir: str,
    *,
    model_config: BioDVNConfig,
    training_config: TrainingConfig,
    context: str,
) -> None:
    _write_json_file(
        os.path.join(output_dir, 'model_config.json'),
        asdict(model_config),
        context=f'model config for {context}',
    )
    _write_json_file(
        os.path.join(output_dir, 'training_config.json'),
        asdict(training_config),
        context=f'training config for {context}',
    )


def _write_training_history(
    output_dir: str,
    history: Dict[str, Any],
    *,
    context: str,
) -> None:
    _write_json_file(
        os.path.join(output_dir, 'training_history.json'),
        history,
        context=f'training history for {context}',
    )


def _write_temp_index_files(prefix: str, named_index_paths: Dict[str, List[str]]) -> Dict[str, str]:
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    index_paths: Dict[str, str] = {}
    for name, paths in named_index_paths.items():
        index_path = os.path.join(temp_dir, f'{name}.json')
        with open(index_path, 'w') as handle:
            json.dump(paths, handle)
        index_paths[name] = index_path
    return index_paths


def _filter_split_rows(
    split_rows: List[Dict[str, str]],
    *,
    exclude_fold: Optional[int] = None,
    exclude_cohorts: Optional[set[str]] = None,
) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    excluded = exclude_cohorts or set()
    for row in split_rows:
        if exclude_fold is not None and int(row['Fold']) == exclude_fold:
            continue
        if row['Cohort'].strip() in excluded:
            continue
        filtered.append(row)
    return filtered


def _expected_zarr_path_preview(
    split_rows: List[Dict[str, str]],
    zarr_base: str,
    *,
    limit: int = 8,
) -> str:
    from proteus_utils.data_helpers import zarr_path_for_sample

    if not split_rows:
        return '  (no eligible TSV rows)'

    lines: List[str] = []
    for row in split_rows[:limit]:
        cohort = row['Cohort'].strip()
        sample = row['Sample'].strip()
        lines.append(f'  {zarr_path_for_sample(sample, cohort, zarr_base)}')

    remaining = len(split_rows) - limit
    if remaining > 0:
        lines.append(f'  ... and {remaining} more expected path(s)')

    return '\n'.join(lines)


def _require_existing_variance_pool(
    zarr_paths: List[str],
    split_rows: List[Dict[str, str]],
    zarr_base: str,
    *,
    context: str,
) -> None:
    if zarr_paths:
        return

    raise FileNotFoundError(
        f'No Zarr stores found for {context} under {zarr_base}. '
        f'Eligible TSV rows: {len(split_rows)}.\n'
        f'First expected paths:\n{_expected_zarr_path_preview(split_rows, zarr_base)}'
    )


def _log_raw_variance_summary(
    raw_variance: Dict[str, float],
    *,
    gene_count: int,
    sample_count: int,
    context: str,
    label_key: str,
) -> None:
    raw_values = list(raw_variance.values())
    if not raw_values:
        raise ValueError(
            f'Variance computation for {context} produced no finite genes from '
            f'{sample_count} Zarr store(s) using label_key {label_key!r}.'
        )

    print(
        f'  {gene_count} genes, n={sample_count} samples — '
        f'raw variance [{min(raw_values):.4f}, {max(raw_values):.4f}], '
        'clipped to [0.100, 10.000]'
    )


def _collect_full_train_paths(
    split_rows: List[Dict[str, str]],
    zarr_base: str,
    train_only_cohorts: Optional[List[str]],
) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    from proteus_utils.data_helpers import zarr_name_for_sample, zarr_path_for_sample

    train_only = set(train_only_cohorts or [])
    all_paths: List[str] = []
    patient_paths: List[str] = []
    zarr_cohort_map: Dict[str, str] = {}
    zarr_sample_name_map: Dict[str, str] = {}
    sample_cohort_map: Dict[str, str] = {}
    missing_samples: List[str] = []

    for row in split_rows:
        cohort = row['Cohort'].strip()
        sample = row['Sample'].strip()

        zpath = zarr_path_for_sample(sample, cohort, zarr_base)
        if not os.path.isdir(zpath):
            bare = zarr_name_for_sample(sample, cohort)
            missing_samples.append(f'  {cohort}/zarr_files/{bare}.zarr  (TSV sample={sample})')
            continue

        all_paths.append(zpath)
        zarr_cohort_map[zpath] = cohort
        zarr_sample_name_map[zpath] = sample
        sample_cohort_map[sample] = cohort
        if cohort not in train_only:
            patient_paths.append(zpath)

    if missing_samples:
        raise FileNotFoundError(
            f'Missing {len(missing_samples)} Zarr store(s) under {zarr_base}:\n' + '\n'.join(missing_samples)
        )

    return all_paths, patient_paths, zarr_cohort_map, zarr_sample_name_map, sample_cohort_map


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _multisource_train_loader_kwargs() -> Dict[str, int | bool]:
    # The multi-source training dataset is already fully preloaded into RAM.
    # Running it through forked persistent workers adds process startup and
    # prefetch overhead without providing I/O overlap, and can stall before the
    # first batch on the current 100+ GB resident panel.
    return {
        'num_workers': 0,
        'persistent_workers': False,
        'prefetch_factor': 2,
    }


def apply_cli_training_overrides(
    training_config: TrainingConfig,
    *,
    verbose: bool = False,
    runtime_device_banner: bool = False,
) -> TrainingConfig:
    updates: Dict[str, object] = {}

    if verbose:
        updates.update({
            'progress_bar': True,
            'progress_backend': 'tqdm',
            'track_batch_timing': True,
            'log_interval': min(int(training_config.log_interval), 10),
        })

    if verbose or runtime_device_banner:
        updates['runtime_device_banner'] = True

    if not updates:
        return training_config

    return replace(training_config, **updates)


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
    source_groups: Optional[Dict[str, List[str]]] = None,
    steps_per_epoch: int = 4000,
    batch_mode: str = 'multi_source',
    fixed_schedule_oof: bool = False,
):
    """Train a model using cross-validation with deterministic fold assignment."""
    from proteus_utils.proteus_utils import (
        compute_gene_mean_std_from_zarr,
        compute_gene_variances_unclipped,
        compute_gene_weights_from_zarr,
    )

    if fixed_schedule_oof and (
        epoch_end_callback is not None or external_stop_check is not None
    ):
        raise ValueError(
            'fixed_schedule_oof=True is incompatible with per-epoch validation '
            'callbacks because leak-proof OOF generation must not consume the '
            'held-out fold during training.'
        )

    if not os.path.exists(training_set_tsv):
        raise FileNotFoundError(f'Training set TSV not found: {training_set_tsv}')

    split_rows = _read_split_rows(training_set_tsv)
    fold_list, sample_to_cohort = _split_folds_and_sample_map(split_rows)
    n_folds = len(fold_list)
    if n_folds == 0:
        raise ValueError(f'No folds found in {training_set_tsv}')

    print(f'Training model with {n_folds} folds from {training_set_tsv}...')
    if out_dir is None:
        out_dir = f'results_training/{model_name}/'
    os.makedirs(out_dir, exist_ok=True)

    if model_config is None:
        model_config = BioDVNConfig()
    if training_config is None:
        training_config = TrainingConfig(checkpoint_dir=out_dir)

    device = torch.device('cuda')

    selection_group_weights = get_source_weighted_selection_group_weights(source_groups)
    if training_config.selection_metric == 'min_cohort_soft_objective':
        if source_groups is None:
            raise ValueError("selection_metric='min_cohort_soft_objective' requires source_groups")
        validate_source_weighted_selection_groups(source_groups)

    sample_to_source: Optional[Dict[str, str]] = None
    if source_groups:
        sample_to_source, _source_variance_paths = build_source_map_from_tsv(
            training_set_tsv, source_groups, zarr_base,
        )

    for fold_num in fold_list:
        if test_fold is not None and fold_num != test_fold:
            continue

        _seed_everything(seed + fold_num)

        print(f'\nStarting fold {fold_num}/{n_folds}')
        fold_dir = os.path.join(out_dir, f'CV-fold_{fold_num}')
        os.makedirs(fold_dir, exist_ok=True)

        fold_training_config = replace(training_config, checkpoint_dir=fold_dir)

        print(f'Computing gene variance weights from training folds (excluding fold {fold_num}, excluding PHT)...')
        variance_rows = _filter_split_rows(
            split_rows,
            exclude_fold=fold_num,
            exclude_cohorts={'PHT'},
        )
        train_zarr_paths = collect_zarr_paths(
            training_set_tsv,
            zarr_base,
            exclude_fold=fold_num,
            exclude_cohorts={'PHT'},
        )
        _require_existing_variance_pool(
            train_zarr_paths,
            variance_rows,
            zarr_base,
            context=f'fold {fold_num} tumor-variance computation',
        )
        train_fold_variance_raw = compute_gene_variances_unclipped(
            train_zarr_paths,
            label_key='flat/tumor_exp',
        )
        gene_weights = compute_gene_weights_from_zarr(
            train_zarr_paths,
            label_key='flat/tumor_exp',
        )
        train_gene_mean_total, train_gene_std_total = compute_gene_mean_std_from_zarr(
            train_zarr_paths,
            label_key='flat/tumor_exp',
        )
        _log_raw_variance_summary(
            train_fold_variance_raw,
            gene_count=len(gene_weights),
            sample_count=len(train_zarr_paths),
            context=f'fold {fold_num} tumor-variance computation',
            label_key='flat/tumor_exp',
        )

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

        print('Building data loaders...')
        if batch_mode == 'multi_source':
            if source_groups is None:
                raise ValueError("batch_mode='multi_source' requires source_groups")
            train_loader_kwargs = _multisource_train_loader_kwargs()
            print(
                '  [MultiSource] training loader uses single-process iteration '
                'because samples are already preloaded in RAM'
            )
            train_loader, val_loader, train_steps, val_steps = build_multisource_loaders_from_splits(
                training_set_tsv=training_set_tsv,
                zarr_base=zarr_base,
                val_fold=fold_num,
                source_groups=source_groups,
                gene_variance=train_fold_variance_raw,
                steps_per_epoch=steps_per_epoch,
                num_workers=_loader_workers(),
                train_num_workers=int(train_loader_kwargs['num_workers']),
                train_persistent_workers=bool(train_loader_kwargs['persistent_workers']),
                train_prefetch_factor=int(train_loader_kwargs['prefetch_factor']),
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
            )
        print(f'Training batches: {train_steps}, Validation batches: {val_steps}')

        model_class = get_model_instance(model_name)
        model = model_class(model_config).to(device)

        _write_training_metadata(
            fold_dir,
            model_config=model_config,
            training_config=fold_training_config,
            context=f'fold {fold_num}',
        )

        trainer = Trainer(model, fold_training_config, model_config, device)
        trainer.set_srl_stats(train_gene_mean_total, train_gene_std_total)
        trainer.val_gene_weight_map = train_fold_variance_raw
        trainer.val_background_gene_weight_map = train_background_variance
        trainer.val_source_gene_weight_map = fold_group_tumor_variance
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

        _write_training_metadata(
            fold_dir,
            model_config=model_config,
            training_config=fold_training_config,
            context=f'fold {fold_num}',
        )
        _write_training_history(
            fold_dir,
            history,
            context=f'fold {fold_num}',
        )

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

        model_for_eval = trainer.model
        model_for_save = getattr(model_for_eval, '_orig_mod', model_for_eval)
        torch.save(
            model_for_save.state_dict(),
            os.path.join(fold_dir, f'validation-fold_{fold_num}_weights.pt'),
        )

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

        _write_loss_curve_plot(
            plot_root=fold_dir,
            history=history,
            weighted_key='val_weighted_tumor_gvw_crps',
        )

        del train_loader, val_loader
        del trainer, model, model_for_eval, model_for_save
        torch.cuda.empty_cache()

        print(f'Finished fold {fold_num}/{n_folds}.')

    if test_fold is None:
        print('Finished complete training across all folds.')
        _summarize_cv_results(out_dir)


def _write_loss_curve_plot(
    plot_root: str,
    history: Dict[str, Any],
    weighted_key: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plot_dir = os.path.join(plot_root, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

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
            plt.plot(
                x_vals[finite],
                y_vals[finite],
                marker='o',
                linewidth=1.5,
                markersize=3.5,
                label=label,
            )

        if len(epochs_axis) > 0:
            plt.plot(epochs_axis, history.get('train_loss', []), label='train_loss')
            _plot_validation_series('val_loss', 'val_loss')
            _plot_validation_series('val_tumor_gvw_crps', 'val_tumor_gvw_crps')
            _plot_validation_series(weighted_key, weighted_key)
            plt.xlabel('epoch')
            plt.ylabel('metric')
            plt.title('Training / validation curves')
            plt.legend(frameon=False)
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, 'loss_curves.png'), dpi=150)
        plt.close(fig)
    except (ImportError, OSError, ValueError) as exc:
        print(f'  Warning: failed to generate loss curves under {plot_root}: {exc}')


def _summarize_cv_results(out_dir: str) -> None:
    fold_metrics: List[Dict[str, float]] = []
    for entry in sorted(os.listdir(out_dir)):
        if not entry.startswith('CV-fold_'):
            continue
        fold_dir = os.path.join(out_dir, entry)
        if not os.path.isdir(fold_dir):
            continue
        try:
            fold_num = int(entry.split('_')[-1])
        except ValueError:
            fold_num = -1

        metrics_path = os.path.join(fold_dir, f'validation-fold_{fold_num}_metrics.json')
        fold_metrics_json: Optional[Dict[str, Any]] = None
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as handle:
                    fold_metrics_json = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                print(f'  Warning: could not load {metrics_path}: {exc}')
                fold_metrics_json = None

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

        fold_metrics.append({
            'fold': int(fold_num),
            'score': score,
            'weighted_tumor_gvw_crps': weighted_tumor_gvw_crps,
            'tumor_gvw_crps': float(t_metrics.get('gvw_crps', float('nan'))),
            'tumor_n': int(t_metrics.get('n', 0)),
            'tumor_mae': float(t_metrics.get('mae', float('nan'))),
            'tumor_rmse': float(t_metrics.get('rmse', float('nan'))),
            'tumor_r2': float(t_metrics.get('r2', float('nan'))),
            'tumor_gene_r2_mean': float(t_metrics.get('gene_r2_mean', float('nan'))),
            'tumor_gene_r2_median': float(t_metrics.get('gene_r2_median', float('nan'))),
            'tumor_sample_r2_mean': float(t_metrics.get('sample_r2_mean', float('nan'))),
            'tumor_sample_r2_median': float(t_metrics.get('sample_r2_median', float('nan'))),
            'background_n': int(b_metrics.get('n', 0)),
            'background_mae': float(b_metrics.get('mae', float('nan'))),
            'background_rmse': float(b_metrics.get('rmse', float('nan'))),
            'background_r2': float(b_metrics.get('r2', float('nan'))),
            'background_gvw_crps': float(b_metrics.get('gvw_crps', float('nan'))),
        })

    if not fold_metrics:
        return

    summary_df = pd.DataFrame(fold_metrics).sort_values('fold')
    summary_df.to_csv(os.path.join(out_dir, 'cv_metrics_summary.tsv'), sep='\t', index=False)

    numeric_cols = [column for column in summary_df.columns if column != 'fold']
    primary_metric_name = 'weighted_tumor_gvw_crps'
    if not np.isfinite(pd.to_numeric(summary_df[primary_metric_name], errors='coerce').to_numpy(dtype=np.float64)).any():
        primary_metric_name = 'tumor_gvw_crps'

    aggregate: Dict[str, Any] = {
        'primary_metric': {'name': primary_metric_name, 'direction': 'minimize'},
    }
    for column in numeric_cols:
        values = pd.to_numeric(summary_df[column], errors='coerce').to_numpy(dtype=np.float64)
        aggregate[column] = {
            'mean': float(np.nanmean(values)),
            'std': float(np.nanstd(values)),
        }

    if primary_metric_name in summary_df.columns:
        series = pd.to_numeric(summary_df[primary_metric_name], errors='coerce')
        if np.isfinite(series.to_numpy(dtype=np.float64)).any():
            best_idx = int(np.nanargmin(series.to_numpy(dtype=np.float64)))
            best_row = summary_df.iloc[best_idx].to_dict()
            aggregate['best'] = {
                'fold': int(best_row.get('fold', -1)),
                primary_metric_name: float(best_row.get(primary_metric_name, float('nan'))),
            }

    with open(os.path.join(out_dir, 'cv_metrics_summary.json'), 'w') as handle:
        json.dump(aggregate, handle, indent=2)


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
    source_groups: Optional[Dict[str, List[str]]] = None,
    steps_per_epoch: int = 4000,
    batch_mode: str = 'multi_source',
):
    """Train a single model on all samples in the training set."""
    from proteus_utils.proteus_utils import (
        compute_gene_mean_std_from_zarr,
        compute_gene_variances_unclipped,
        compute_gene_weights_from_zarr,
    )
    from proteus_utils.data_helpers import assign_source

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

    if not os.path.exists(training_set_tsv):
        raise FileNotFoundError(f'Training set TSV not found: {training_set_tsv}')

    split_rows = _read_split_rows(training_set_tsv)
    train_only = set(train_only_cohorts or [])
    all_paths, patient_paths, zarr_cohort_map, zarr_sample_name_map, sample_cohort_map = (
        _collect_full_train_paths(split_rows, zarr_base, train_only_cohorts)
    )

    print(
        f'Total samples: {len(all_paths)} ({len(patient_paths)} patient, '
        f'{len(all_paths) - len(patient_paths)} train-only)'
    )

    index_paths = _write_temp_index_files(
        'proteus_full_train_',
        {
            'full_all': all_paths,
            'full_patient': patient_paths,
        },
    )
    all_index = index_paths['full_all']
    patient_index = index_paths['full_patient'] if patient_paths and train_only else all_index

    if subsample_frac is not None and subsample_frac < 1.0:
        print(f'  [Subsample] Using ~{subsample_frac * 100:.0f}% of genes for fast sanity check')

    print('Computing gene variance weights from all non-PHT tumor samples...')
    tumor_weight_rows = _filter_split_rows(split_rows, exclude_cohorts={'PHT'})
    tumor_weight_paths = [path for path in all_paths if zarr_cohort_map[path] != 'PHT']
    _require_existing_variance_pool(
        tumor_weight_paths,
        tumor_weight_rows,
        zarr_base,
        context='full-train tumor-variance computation',
    )
    train_fold_variance_raw = compute_gene_variances_unclipped(tumor_weight_paths, label_key='flat/tumor_exp')
    train_gene_mean_total, train_gene_std_total = compute_gene_mean_std_from_zarr(
        tumor_weight_paths,
        label_key='flat/tumor_exp',
    )
    gene_weights = compute_gene_weights_from_zarr(
        tumor_weight_paths,
        label_key='flat/tumor_exp',
    )
    _log_raw_variance_summary(
        train_fold_variance_raw,
        gene_count=len(gene_weights),
        sample_count=len(tumor_weight_paths),
        context='full-train tumor-variance computation',
        label_key='flat/tumor_exp',
    )

    selection_group_weights = get_source_weighted_selection_group_weights(source_groups)
    if training_config.selection_metric == 'min_cohort_soft_objective':
        if source_groups is None:
            raise ValueError("selection_metric='min_cohort_soft_objective' requires source_groups")
        validate_source_weighted_selection_groups(source_groups)

    sample_to_source: Optional[Dict[str, str]] = None
    source_tumor_variance: Optional[Dict[str, Dict[str, float]]] = None
    sample_paths_by_source: Optional[Dict[str, List[str]]] = None
    if source_groups:
        cohort_to_source = invert_source_groups(source_groups)
        pdx_names = collect_pdx_names(training_set_tsv, cohort_to_source)

        sample_to_source = {}
        sample_paths_by_source = {source_name: [] for source_name in source_groups}
        source_variance_paths: Dict[str, List[str]] = {source_name: [] for source_name in source_groups}

        for zarr_path in all_paths:
            cohort = zarr_cohort_map[zarr_path]
            sample = zarr_sample_name_map[zarr_path]
            source = assign_source(cohort, sample, cohort_to_source, pdx_names)
            if source is None:
                continue
            sample_to_source[sample] = source
            sample_paths_by_source[source].append(zarr_path)
            if cohort != 'PHT':
                source_variance_paths[source].append(zarr_path)

        source_tumor_variance = compute_source_variance(
            source_groups,
            source_variance_paths,
            compute_gene_variances_unclipped,
        )

    val_loader: Optional[DataLoader] = None
    val_steps = 0
    use_checkpoint_holdout = False
    val_frac = 0.10

    if batch_mode == 'multi_source':
        if sample_paths_by_source is None:
            raise ValueError("batch_mode='multi_source' requires source_groups")

        train_loader_kwargs = _multisource_train_loader_kwargs()
        print(
            '  [MultiSource] training loader uses single-process iteration '
            'because samples are already preloaded in RAM'
        )

        train_dataset = ProteusMultiSourceBatchDataset(
            sample_paths_by_source=sample_paths_by_source,
            gene_variance=train_fold_variance_raw,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
            labeled=True,
            sample_name_map=zarr_sample_name_map,
            include_metadata=True,
            subsample_frac=subsample_frac,
        )

        train_loader = create_single_item_loader(
            train_dataset,
            num_workers=int(train_loader_kwargs['num_workers']),
            seed=seed,
            persistent_workers=bool(train_loader_kwargs['persistent_workers']),
            prefetch_factor=int(train_loader_kwargs['prefetch_factor']),
        )
        train_steps = len(train_dataset)

        if use_checkpoint_holdout:
            loader_workers = _loader_workers()
            n_samples = len(train_dataset._all_sample_paths)
            n_union = len(train_dataset._union_genes)
            rng_mask = np.random.RandomState(seed + 12345)
            val_mask = np.zeros((n_samples, n_union), dtype=np.bool_)
            path_to_source: Dict[str, str] = {}
            for source_name, paths in sample_paths_by_source.items():
                for path in paths:
                    path_to_source[path] = source_name
            for sample_index, zpath in enumerate(train_dataset._all_sample_paths):
                source_name = path_to_source.get(zpath)
                if source_name is None:
                    continue
                source_mask = train_dataset._source_gene_mask[source_name]
                valid_gene_indices = np.where(source_mask)[0]
                n_valid = len(valid_gene_indices)
                n_holdout = max(1, int(n_valid * val_frac))
                chosen = rng_mask.choice(valid_gene_indices, size=n_holdout, replace=False)
                val_mask[sample_index, chosen] = True

            total_pairs = int(val_mask.sum())
            total_valid = sum(
                int(train_dataset._source_gene_mask[path_to_source[path]].sum())
                for path in train_dataset._all_sample_paths
                if path in path_to_source
            )
            print(f'  [HoldoutVal] {total_pairs} val pairs / {total_valid} total ({100.0 * total_pairs / total_valid:.1f}%)')

            train_dataset.val_mask = val_mask
            val_dataset = ProteusHoldoutValDataset(
                train_dataset=train_dataset,
                val_mask=val_mask,
                batch_size=32,
                seed=seed,
            )
            val_loader = create_single_item_loader(
                val_dataset,
                num_workers=loader_workers,
                seed=seed,
                persistent_workers=True,
                prefetch_factor=4,
            )
            val_steps = len(val_dataset)
    else:
        train_loader, train_steps = create_data_loader(
            index_file=all_index,
            num_workers=_loader_workers(),
            seed=seed,
            subsample_frac=subsample_frac,
            gene_weights=gene_weights,
            val_frac=val_frac if use_checkpoint_holdout else None,
            val_split=False,
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

    _seed_everything(seed)

    model_class = get_model_instance(model_name)
    model = model_class(model_config)

    _write_training_metadata(
        out_dir,
        model_config=model_config,
        training_config=training_config,
        context='full-train run',
    )

    trainer = Trainer(model, training_config, model_config, device)
    trainer.set_srl_stats(train_gene_mean_total, train_gene_std_total)
    if val_loader is not None:
        trainer.val_gene_weight_map = train_fold_variance_raw
        trainer.val_source_gene_weight_map = source_tumor_variance
        trainer.val_selection_group_weights = selection_group_weights
        if sample_to_source:
            trainer.val_sample_group_map = sample_to_source
        elif sample_cohort_map and train_only:
            patient_cohort_map = {
                sample_name: cohort_name
                for sample_name, cohort_name in sample_cohort_map.items()
                if cohort_name not in train_only
            }
            trainer.val_sample_group_map = patient_cohort_map

    history = trainer.fit(train_loader, val_loader, epochs=training_config.epochs)
    _write_training_metadata(
        out_dir,
        model_config=model_config,
        training_config=training_config,
        context='full-train run',
    )
    _write_training_history(
        out_dir,
        history,
        context='full-train run',
    )

    _write_loss_curve_plot(
        plot_root=out_dir,
        history=history,
        weighted_key='val_tumor_gvw_crps',
    )

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


def build_training_configs_from_spec(
    experiment_spec: Any,
    *,
    model_name: str = 'BioDVN',
    runtime_device_banner: bool = False,
    base_weights_override: Optional[str] = None,
) -> Tuple[BioDVNConfig, TrainingConfig]:
    """Build concrete model/training configs from an experiment spec."""
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
        beta_srl_max=float(experiment_spec.structural.beta_srl_max.fixed),
        beta_dynamic_pair_max=float(experiment_spec.structural.beta_dynamic_pair_max.fixed),
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

    base_weights = base_weights_override or experiment_spec.transfer.base_weights
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
        runtime_device_banner=bool(runtime_device_banner),
        base_weights=normalize_path(base_weights) if base_weights else None,
    )
    return model_cfg, train_cfg
