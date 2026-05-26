"""Dataset preparation and verification workflows."""

from __future__ import annotations

import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

import pandas as pd

from config.dataset_config import load_dataset_config
from proteus_utils import (
    get_samples,
    get_zarr_output_dir,
    process_dataset,
    process_sample_to_zarr,
    verify_dataset,
)


def ensure_datasets_exist(
    dataset_names: List[str],
    zarr_base: str,
    raw_data_base: str,
    labels_dir: str,
    ref_path: str,
    n_workers: int = 0,
) -> Dict[str, bool]:
    """Check configured datasets for missing Zarr stores and regenerate them."""
    regenerated: Dict[str, bool] = {}

    def _is_present_zarr_store(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        try:
            return any(os.scandir(path))
        except FileNotFoundError:
            return False

    for dataset_name in dataset_names:
        try:
            dataset_config = load_dataset_config(dataset_name)
        except FileNotFoundError:
            print(f"[Auto-regenerate] Warning: No config for '{dataset_name}', skipping")
            continue

        actual_workers = n_workers if n_workers > 0 else cpu_count()
        zarr_dir = get_zarr_output_dir(dataset_name, zarr_base)
        tss_dir = os.path.join(raw_data_base, dataset_config.triton_tss)
        gb_dir = os.path.join(raw_data_base, dataset_config.triton_gb)

        zarr_files: List[str] = []
        if os.path.exists(zarr_dir):
            zarr_files = [name for name in os.listdir(zarr_dir) if name.endswith('.zarr')]

        if (not os.path.exists(zarr_dir)) or (not zarr_files):
            reason = 'directory missing' if not os.path.exists(zarr_dir) else 'no zarr files'
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
                n_workers=actual_workers,
            )
            regenerated[dataset_name] = True
            continue

        discovered_samples = get_samples(tss_dir, dataset_name)

        if dataset_config.tfx_file is not None:
            tfx_file_path = os.path.join(labels_dir, dataset_config.tfx_file)
            try:
                tfx_df = pd.read_csv(tfx_file_path, sep='\t', index_col='sample')
                tfx_index = [str(index_value) for index_value in tfx_df.index.tolist()]

                def _matches_tfx(sample: str) -> bool:
                    if sample in tfx_df.index:
                        return True
                    for index_value in tfx_index:
                        if sample in index_value or index_value in sample:
                            return True
                    return False

                expected_samples = [sample for sample in discovered_samples if _matches_tfx(sample)]
            except (FileNotFoundError, ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Failed to read TFX file for '{dataset_name}': {exc}")
        else:
            expected_samples = discovered_samples

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
    """Generate Zarr stores from Triton outputs using the sample-level layout."""
    print(f'Generating Zarr stores for dataset: {dataset}')
    n_workers = cpu_count()
    print(f'  Workers: {n_workers}')

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
        n_workers=n_workers,
    )

    if created_paths:
        print(f'\nSuccessfully created {len(created_paths)} Zarr stores.')
    else:
        print('\nNo Zarr stores created. Check input paths and data.')

    return created_paths


def verify_dataset_stores(dataset_name: str, zarr_base: str) -> pd.DataFrame:
    """Run the full Zarr schema/content validator for one configured dataset."""
    zarr_dir = get_zarr_output_dir(dataset_name, zarr_base)
    if not os.path.isdir(zarr_dir):
        raise FileNotFoundError(f'No Zarr directory found for {dataset_name}: {zarr_dir}')
    if not any(name.endswith('.zarr') for name in os.listdir(zarr_dir)):
        raise FileNotFoundError(f'No Zarr stores found for {dataset_name}: {zarr_dir}')
    return verify_dataset(zarr_dir)
