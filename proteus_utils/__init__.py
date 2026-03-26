# PyTorch proteus utilities package
# Data loading and preprocessing utilities

from .logging_config import get_logger, setup_logging, set_verbosity

from .compile_sample_data import (
    get_samples,
    get_zarr_output_dir,
    process_sample_to_zarr,
    process_dataset,
    verify_zarr_store,
    verify_dataset,
    DEFAULT_ZARR_BASE,
    SHAPES,
    DTYPES,
    TSS_FEATURES,
    GB_FEATURES,
    SIGNAL_CHANNELS,
    NUCLEOTIDE_CHANNELS,
    LOG_TRANSFORM_FEATURES,
    LOG_ONLY_FEATURES,
    MAX_FEATURE_NAN_FRACTION,
)

from .proteus_utils import (
    ProteusInferenceDataset,
    collate_inference,
    create_data_loader,
    create_inference_loader,
    build_loaders_from_splits,
    extract_gene_depths,
)

from .trainer import TrainingConfig, Trainer

from .metrics import (
    compute_regression_metrics,
    pearson_r,
    spearman_r,
    spearman_r2,
    rankdata,
    group_spearman_r2,
    group_pearson_r_named,
    coverage,
    gvw_r,
    scatter_stats,
)

__all__ = [
    # logging
    'get_logger', 'setup_logging', 'set_verbosity',
    # compile_sample_data
    'get_samples', 'get_zarr_output_dir',
    'process_sample_to_zarr', 'process_dataset',
    'verify_zarr_store', 'verify_dataset',
    'DEFAULT_ZARR_BASE', 'SHAPES', 'DTYPES',
    'TSS_FEATURES', 'GB_FEATURES', 'SIGNAL_CHANNELS', 'NUCLEOTIDE_CHANNELS',
    'LOG_TRANSFORM_FEATURES', 'LOG_ONLY_FEATURES',
    'MAX_FEATURE_NAN_FRACTION',
    # proteus_utils
    'ProteusInferenceDataset',
    'collate_inference',
    'create_data_loader', 'create_inference_loader',
    'build_loaders_from_splits', 'extract_gene_depths',
    # trainer
    'TrainingConfig', 'Trainer',
    # metrics
    'compute_regression_metrics', 'pearson_r', 'spearman_r', 'spearman_r2',
    'rankdata', 'group_spearman_r2', 'group_pearson_r_named',
    'coverage', 'gvw_r', 'scatter_stats',
]