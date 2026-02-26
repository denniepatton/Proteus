# PyTorch proteus utilities package
# Data loading and preprocessing utilities

from .logging_config import get_logger, setup_logging, set_verbosity

from .compile_sample_data import (
    get_samples,
    get_samples_from_tfx,
    get_zarr_output_dir,
    process_sample_to_zarr,
    process_dataset,
    create_sample_splits,
    create_fold_index,
    verify_zarr_store,
    verify_dataset,
    normalize_cohort_name,
    COHORT_NAME_MAP,
    DEFAULT_ZARR_BASE,
    SHAPES,
    DTYPES,
    TSS_FEATURES,
    GB_FEATURES,
    SIGNAL_CHANNELS,
    NUCLEOTIDE_CHANNELS,
    LOG_TRANSFORM_FEATURES,
    LOG_ONLY_FEATURES,
    MAX_SIGNAL_NAN_FRACTION,
    MAX_FEATURE_NAN_FRACTION,
)

from .proteus_utils import (
    soft_class_label,
    ProteusSampleDataset,
    ProteusDataset,
    ProteusInferenceDataset,
    collate_proteus_samples,
    collate_proteus,
    collate_inference,
    create_data_loader,
    create_inference_loader,
    get_fold_indices,
    build_train_val_loaders,
    build_loaders_from_splits,
    extract_gene_depths,
)

from .debug_utils import print_input_diagnostics
from .trainer import TrainingConfig, Trainer

__all__ = [
    # logging
    'get_logger', 'setup_logging', 'set_verbosity',
    # compile_sample_data
    'get_samples', 'get_samples_from_tfx', 'get_zarr_output_dir',
    'process_sample_to_zarr', 'process_dataset',
    'create_sample_splits', 'create_fold_index',
    'verify_zarr_store', 'verify_dataset',
    'normalize_cohort_name', 'COHORT_NAME_MAP',
    'DEFAULT_ZARR_BASE', 'SHAPES', 'DTYPES',
    'TSS_FEATURES', 'GB_FEATURES', 'SIGNAL_CHANNELS', 'NUCLEOTIDE_CHANNELS',
    'LOG_TRANSFORM_FEATURES', 'LOG_ONLY_FEATURES',
    'MAX_SIGNAL_NAN_FRACTION', 'MAX_FEATURE_NAN_FRACTION',
    # proteus_utils
    'soft_class_label', 'ProteusSampleDataset', 'ProteusDataset', 'ProteusInferenceDataset',
    'collate_proteus_samples', 'collate_proteus', 'collate_inference',
    'create_data_loader', 'create_inference_loader',
    'get_fold_indices', 'build_train_val_loaders', 'build_loaders_from_splits', 'extract_gene_depths',
    # debug
    'print_input_diagnostics',
    # trainer
    'TrainingConfig', 'Trainer',
]