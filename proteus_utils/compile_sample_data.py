"""compile_sample_data.py (PyTorch version)

This script contains functions for compiling sample data from raw Triton outputs
and reference files into optimized Zarr stores for PyTorch training.

Pipeline: Raw Triton files → Zarr stores → Fold indices → PyTorch DataLoader

Raw Triton output structure (per sample folder):
    SAMPLE_TritonSignalProfiles.npz - 12 signal channels x 2000 bp (TSS only)
    SAMPLE_TritonFeatures.tsv - 19 columns of region-level features

Zarr layout (gene-level hierarchy for fast random access):
    sample.zarr/
      ├── gene_names: array of gene IDs for indexing
      ├── genes/
      │   └── GENE_NAME/
      │       ├── TSS_signals (2000, 8) float32 - biological signals
      │       ├── TSS_nucleotides (2000, 4) uint8 - one-hot nucleotides
      │       ├── TSS_features (17,) float32 - transformed features
      │       ├── GB_features (11,) float32 - gene body features
      │       ├── ref_scalars (3,) float32 - reference gene info
      │       └── .zattrs: tfx, tumor_exp, background_exp
      └── flat/  (alternative flat array access for batch loading)
          └── TSS_signals (n_genes, 2000, 8), etc.

Feature transforms (applied at Zarr conversion time):
    - mean-region-depth: log1p only (preserves absolute depth info)
    - fl-kurtosis, pn-mean-spacing, pn-mean-amplitude, central-depth: log + z-score
    - All others: z-score only
    
NaN handling:
    - Sample-mean imputation for features after transforms
    - Channel-mean imputation for signals
    - Genes with >50% NaN in signals or features are dropped
"""

# Standard library imports
import os
import json
import shutil
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
from multiprocessing import Pool

# Third party imports
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc, VLenUTF8

# =============================================================================
# Configuration
# =============================================================================

def _get_default_zarr_base() -> str:
    """
    Get default Zarr base directory from config or fallback to hardcoded value.
    
    Order of precedence:
    1. PROTEUS_ZARR_BASE environment variable
    2. paths.yaml configuration file
    3. Hardcoded fallback
    """
    # Check environment variable first
    env_zarr_base = os.environ.get('PROTEUS_ZARR_BASE')
    if env_zarr_base:
        return env_zarr_base
    
    # Try to load from config
    try:
        from config.config_loader import ConfigLoader
        loader = ConfigLoader()
        paths_config = loader.load_paths()
        return paths_config.get_path('zarr_base_dir')
    except Exception:
        pass
    
    # Fallback to hardcoded default
    return "/hpc/temp/ha_g/user/rpatton/ProteusZarrs"


def _get_default_raw_data_dir() -> str:
    """
    Get default raw data directory from config or fallback to hardcoded value.
    
    Order of precedence:
    1. PROTEUS_RAW_DATA_DIR environment variable
    2. paths.yaml configuration file
    3. Hardcoded fallback
    """
    # Check environment variable first
    env_raw_data = os.environ.get('PROTEUS_RAW_DATA_DIR')
    if env_raw_data:
        return env_raw_data
    
    # Try to load from config
    try:
        from config.config_loader import ConfigLoader
        loader = ConfigLoader()
        paths_config = loader.load_paths()
        return paths_config.get_path('raw_data_dir')
    except Exception:
        pass
    
    # Fallback to hardcoded default
    return "/fh/fast/ha_g/user/rpatton"


# Default Zarr output directory (loaded from config or environment)
DEFAULT_ZARR_BASE = _get_default_zarr_base()

# Compression settings for Zarr
COMPRESSOR = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# Signal channels (8 biological signals)
SIGNAL_CHANNELS = [
    'depth',                    # 0: GC-corrected coverage
    'frag_end_coverage',        # 1: Fragment end density
    'frag_end_asymmetry',       # 2: (5' - 3') / (5' + 3') orientation
    'pn_signal',                # 3: Phased-nucleosome signal
    'fl_subnucleosomal_ratio',  # 4: log2(< 147 bp / >= 147 bp)
    'fl_entropy',               # 5: Shannon entropy
    'fl_gini_simpson',          # 6: Diversity index
    'pn_peaks',                 # 7: Local maxima/minima
]

# Nucleotide one-hot channels (stored separately as uint8)
NUCLEOTIDE_CHANNELS = ['A', 'C', 'G', 'T']

# TSS feature columns (17 features)
TSS_FEATURES = [
    'fl-mean',
    'fl-stdev',
    'fl-skew',
    'fl-kurtosis',
    'fl-subnucleosomal-ratio',
    'fl-entropy',
    'fl-gini-simpson',
    'pn-compaction-score',
    'pn-mean-spacing',
    'pn-mean-amplitude',
    'mean-region-depth',
    'central-depth',
    'central-entropy',
    'central-gini-simpson',
    'window-depth',
    'window-entropy',
    'window-gini-simpson',
]

# GB features (11 features - no central-*/window-* which are TSS-window specific)
GB_FEATURES = [
    'fl-mean',
    'fl-stdev',
    'fl-skew',
    'fl-kurtosis',
    'fl-subnucleosomal-ratio',
    'fl-entropy',
    'fl-gini-simpson',
    'pn-compaction-score',
    'pn-mean-spacing',
    'pn-mean-amplitude',
    'mean-region-depth',
]

# Feature transform categories
LOG_ONLY_FEATURES = ['mean-region-depth']  # log1p only, NO z-score
LOG_TRANSFORM_FEATURES = ['fl-kurtosis', 'pn-mean-spacing', 'pn-mean-amplitude', 'central-depth']

# NaN thresholds for gene exclusion
MAX_SIGNAL_NAN_FRACTION = 0.5
MAX_FEATURE_NAN_FRACTION = 0.5

# Cohort name normalization: map legacy/alternative names to canonical names
# PC-TAN and CRPC-RA refer to the same prostate cancer cohort -> use CRPC-RA
# BC-TAN and BLCA-RA refer to the same bladder cancer cohort -> use BLCA-RA
COHORT_NAME_MAP = {
    'PC-TAN': 'CRPC-RA',
    'BC-TAN': 'BLCA-RA',
}

def normalize_cohort_name(name: str) -> str:
    """Normalize cohort name to canonical form.
    
    PC-TAN, CRPC-RA -> CRPC-RA
    BC-TAN, BLCA-RA -> BLCA-RA
    """
    return COHORT_NAME_MAP.get(name, name)

# Data shapes for Zarr stores
SHAPES = {
    'TSS_signals': (2000, 8),
    'TSS_nucleotides': (2000, 4),
    'TSS_features': (17,),
    'GB_features': (11,),
    'ref_scalars': (3,),
    'TFX_label': (),
    'tumor_exp': (),
    'background_exp': (),
}

DTYPES = {
    'TSS_signals': np.float32,
    'TSS_nucleotides': np.uint8,
    'TSS_features': np.float32,
    'GB_features': np.float32,
    'ref_scalars': np.float32,
    'TFX_label': np.float32,
    'tumor_exp': np.float32,
    'background_exp': np.float32,
}


# =============================================================================
# Sample Discovery (per-sample folder structure)
# =============================================================================

def get_samples(triton_results_dir: str, dataset: str = None) -> List[str]:
    """
    Retrieves sample names from a results directory.
    
    TritonV2 stores results in sample-level folders:
        results/SAMPLE/SAMPLE_TritonSignalProfiles.npz
        
    Args:
        triton_results_dir: Path to Triton results directory
        dataset: Optional dataset name (used for TAN naming convention)
        
    Returns:
        List of sample names
    """
    if not os.path.exists(triton_results_dir):
        print(f'Warning: Directory not found: {triton_results_dir}')
        return []
    
    samples = []
    for item in os.listdir(triton_results_dir):
        item_path = os.path.join(triton_results_dir, item)
        
        # Skip non-directories and hidden files
        if not os.path.isdir(item_path) or item.startswith('.'):
            continue
        
        # Check if this folder contains Triton outputs
        # Look for the signal profiles file (TSS) or features file (GB)
        signal_file = os.path.join(item_path, f'{item}_TritonSignalProfiles.npz')
        features_file = os.path.join(item_path, f'{item}_TritonFeatures.tsv')
        
        if os.path.exists(signal_file) or os.path.exists(features_file):
            samples.append(item)
    
    # Handle TAN naming convention with canonical cohort name
    if dataset and 'TAN' in dataset:
        canonical_cohort = normalize_cohort_name(dataset)
        samples = [f'{canonical_cohort}_{sample[:6]}' for sample in samples]
    
    return sorted(samples)


def get_samples_from_tfx(tfx_path: Union[str, int, float]) -> List[str]:
    """Get sample names from TFX labels file."""
    if isinstance(tfx_path, (int, float)):
        return []
    if not os.path.exists(tfx_path):
        return []
    df = pd.read_csv(tfx_path, sep='\t')
    # First column should be sample names
    return df.iloc[:, 0].tolist()


# =============================================================================
# Data Processing Functions
# =============================================================================

def _scale_signal(input_array: np.ndarray) -> np.ndarray:
    """Scale an input array by dividing by its mean (avoid division by zero/Inf)."""
    mean_val = np.nanmean(input_array)
    if mean_val > 1e-10 and not np.isnan(mean_val):  # Use small threshold, not just != 0
        result = input_array / mean_val
        # Clip any Inf values that might result from extreme inputs
        result = np.clip(result, -1e6, 1e6)
        return result
    return input_array


def _load_signal_profiles(
    sample_dir: str,
    sample: str,
    excluded_genes_tracker: Optional[Dict[str, int]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load signal profiles from TritonV2 output.
    
    Separates biological signals (8 channels) from nucleotide OHE (4 channels).
    Applies channel-mean imputation for NaN values in signals.
    
    Args:
        sample_dir: Path to sample's Triton output folder
        sample: Sample name
        excluded_genes_tracker: Optional dict to track excluded genes by reason
        
    Returns:
        Tuple of:
            - signals_dict: Gene name -> (2000, 8) float32 biological signals
            - nucleotides_dict: Gene name -> (2000, 4) uint8 one-hot nucleotides
    """
    import warnings
    signal_file = os.path.join(sample_dir, f'{sample}_TritonSignalProfiles.npz')
    
    if not os.path.exists(signal_file):
        return {}, {}
    
    data = np.load(signal_file, allow_pickle=True)
    signals_dict = {}
    nucleotides_dict = {}
    
    for gene in data.files:
        raw_data = data[gene]
        
        # Skip empty arrays
        if raw_data.size == 0:
            continue
        
        # Handle shape - expect (2000, 12) or (12, 2000)
        if raw_data.shape == (12, 2000):
            raw_data = raw_data.T
        elif raw_data.shape == (2000, 12):
            pass
        else:
            # Try to handle legacy shapes
            if raw_data.shape == (11, 2000):
                raw_data = raw_data.T
                # Pad to 12 channels if legacy 11-channel format
                raw_data = np.pad(raw_data, ((0, 0), (0, 1)), constant_values=0)
            elif raw_data.shape == (2000, 11):
                raw_data = np.pad(raw_data, ((0, 0), (0, 1)), constant_values=0)
            else:
                print(f'Unexpected signal shape for {gene}: {raw_data.shape}')
                continue
        
        # Split into signals (0:8) and nucleotides (8:12)
        signals = raw_data[:, :8].astype(np.float32)
        
        # Handle nucleotide conversion with suppressed warnings (invalid values -> 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            nuc_raw = raw_data[:, 8:12]
            # Replace any NaN/Inf before casting to uint8
            nuc_raw = np.nan_to_num(nuc_raw, nan=0.0, posinf=0.0, neginf=0.0)
            nucleotides = nuc_raw.astype(np.uint8)
        
        # Check NaN fraction in signals
        nan_fraction = np.isnan(signals).mean()
        if nan_fraction > MAX_SIGNAL_NAN_FRACTION:
            # Skip this gene if too many NaNs - silently track
            if excluded_genes_tracker is not None:
                excluded_genes_tracker['high_nan_signals'] = excluded_genes_tracker.get('high_nan_signals', 0) + 1
            continue
        
        # Apply channel-mean imputation for remaining NaNs (suppress empty slice warning)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for ch in range(signals.shape[1]):
                ch_data = signals[:, ch]
                nan_mask = np.isnan(ch_data)
                if nan_mask.any():
                    ch_mean = np.nanmean(ch_data)
                    if np.isnan(ch_mean):
                        ch_mean = 0.0
                    ch_data[nan_mask] = ch_mean
                    signals[:, ch] = ch_data
        
        # Normalize certain signal channels by their mean (already flank-normalized)
        # Channel 0 (depth): normalize by mean
        signals[:, 0] = _scale_signal(signals[:, 0])
        # Channel 1 (frag end coverage): normalize by mean
        signals[:, 1] = _scale_signal(signals[:, 1])
        # Channel 3 (PN signal): normalize by mean  
        signals[:, 3] = _scale_signal(signals[:, 3])
        # Other channels (2, 4, 5, 6, 7) are already normalized or bounded
        
        # FINAL CHECK: Replace any remaining NaN/Inf with 0
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        
        signals_dict[gene] = signals
        nucleotides_dict[gene] = nucleotides
    
    return signals_dict, nucleotides_dict


def _load_features(
    sample_dir: str,
    sample: str,
    feature_list: List[str]
) -> Dict[str, np.ndarray]:
    """
    Load Triton features from TSV file.
    
    Args:
        sample_dir: Path to sample's Triton output folder
        sample: Sample name
        feature_list: List of feature column names to extract
        
    Returns:
        Dictionary mapping gene names to feature arrays (raw, untransformed)
    """
    features_file = os.path.join(sample_dir, f'{sample}_TritonFeatures.tsv')
    
    if not os.path.exists(features_file):
        return {}
    
    df = pd.read_csv(features_file, sep='\t')
    
    # Keep only the specified feature columns
    feature_cols = [c for c in feature_list if c in df.columns]
    
    if len(feature_cols) != len(feature_list):
        missing = set(feature_list) - set(feature_cols)
        print(f'Warning: Missing feature columns: {missing}')
    
    out_dict = {}
    
    for _, row in df.iterrows():
        gene = row['site']
        
        # Extract features in order specified by feature_list
        features = np.array([row.get(col, np.nan) for col in feature_list], dtype=np.float32)
        
        out_dict[gene] = features
    
    return out_dict


def _transform_features(
    features_dict: Dict[str, np.ndarray],
    feature_list: List[str]
) -> Dict[str, np.ndarray]:
    """
    Apply feature-specific transforms and z-score normalization.
    
    Transform strategy:
    - LOG_ONLY_FEATURES (mean-region-depth): log1p only, NO z-score
    - LOG_TRANSFORM_FEATURES: log1p + z-score
    - All others: z-score only
    
    NaN handling: sample-mean imputation after transforms.
    
    Args:
        features_dict: Dictionary mapping gene names to raw feature arrays
        feature_list: List of feature names in order
        
    Returns:
        Transformed features dictionary
    """
    if not features_dict:
        return features_dict
    
    # Stack all features into matrix
    genes = list(features_dict.keys())
    features_matrix = np.stack([features_dict[g] for g in genes])
    n_features = features_matrix.shape[1]
    
    # Build index lookup
    feature_indices = {feat: i for i, feat in enumerate(feature_list)}
    
    # Phase 1: Apply log1p to appropriate features
    for feat in LOG_ONLY_FEATURES + LOG_TRANSFORM_FEATURES:
        if feat in feature_indices:
            idx = feature_indices[feat]
            col = features_matrix[:, idx].copy()
            # log1p handles zeros gracefully: log1p(0) = 0
            # For negative or zero values, set to NaN first to avoid warnings
            valid_mask = col > 0
            result = np.full_like(col, np.nan)
            result[valid_mask] = np.log1p(col[valid_mask])
            features_matrix[:, idx] = result
    
    # Phase 2: Apply z-score to non-LOG_ONLY features
    for col_idx in range(n_features):
        feat_name = feature_list[col_idx]
        
        # Skip LOG_ONLY features (they keep raw log1p values)
        if feat_name in LOG_ONLY_FEATURES:
            continue
        
        col_data = features_matrix[:, col_idx]
        col_mean = np.nanmean(col_data)
        col_std = np.nanstd(col_data)
        
        if col_std > 0:
            features_matrix[:, col_idx] = (col_data - col_mean) / col_std
        else:
            # Constant column - set to 0
            features_matrix[:, col_idx] = 0.0
    
    # Phase 3: Apply sample-mean imputation for remaining NaNs
    for col_idx in range(n_features):
        col_data = features_matrix[:, col_idx]
        nan_mask = np.isnan(col_data)
        if nan_mask.any():
            col_mean = np.nanmean(col_data)
            if np.isnan(col_mean):
                col_mean = 0.0
            col_data[nan_mask] = col_mean
            features_matrix[:, col_idx] = col_data
    
    # FINAL CHECK: Replace ANY remaining NaN/Inf with 0
    features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Rebuild dictionary
    return {gene: features_matrix[i].astype(np.float32) for i, gene in enumerate(genes)}


def _process_label(filepath: str, sample: str) -> Dict[str, float]:
    """Process a label file and return data as a dictionary."""
    if filepath is None:
        return {}
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    if sample not in df.columns:
        return {}
    df = df[[sample]]
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    return df[sample].to_dict()


def _process_ref(filepath: str, cols: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Process a reference file and return data as a dictionary.
    
    Returns:
        Dictionary mapping gene symbols to (3,) float arrays.
    """
    if cols is None:
        cols = ['NumExons', 'transcript_length', 'mRNA_size']
    
    df = pd.read_csv(filepath, sep='\t')
    df[cols] = df[cols].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df[cols] = df[cols].apply(lambda x: np.log(x))
    
    col_std = df[cols].std()
    col_mean = df[cols].mean()
    df[cols] = (df[cols] - col_mean) / col_std

    return {row['symbol']: np.array(row[cols].values, dtype=np.float32)
            for _, row in df.iterrows()}


# =============================================================================
# Zarr Store Creation
# =============================================================================

def get_zarr_output_dir(dataset: str, base_dir: Optional[str] = None) -> str:
    """
    Get the Zarr output directory for a dataset.
    
    Args:
        dataset: Dataset name (will be normalized to canonical form)
        base_dir: Base directory for Zarr stores (default: DEFAULT_ZARR_BASE)
        
    Returns:
        Path to dataset's Zarr directory
    """
    if base_dir is None:
        base_dir = DEFAULT_ZARR_BASE
    # Use canonical cohort name for output directory
    canonical_name = normalize_cohort_name(dataset)
    return os.path.join(base_dir, canonical_name, 'zarr_files')


def process_sample_to_zarr(
    sample: str,
    tss_dir: str,
    gb_dir: str,
    ref_path: str,
    tfx_path: Union[str, float],
    tumor_exp_path: Optional[str],
    background_exp_path: Optional[str],
    out_dir: str,
    overwrite: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, int]]]:
    """
    Process a sample and write data to a Zarr store (flat per-sample arrays).
    
    Creates a sample-level Zarr store (flat per-sample arrays):
    
        sample.zarr/
            ├── gene_names (n_genes,)  (string)
            └── flat/
                    ├── TSS_signals (n_genes, 2000, 8) float32
                    ├── TSS_nucleotides (n_genes, 2000, 4) uint8
                    ├── TSS_features (n_genes, 17) float32
                    ├── GB_features (n_genes, 11) float32
                    ├── ref_scalars (n_genes, 3) float32
                    ├── TFX_label (n_genes,) float32
                    ├── tumor_exp (n_genes,) float32 (NaN allowed)
                    ├── background_exp (n_genes,) float32 (NaN allowed)
                    └── gene_weight (n_genes,) float32 (default=1.0)

        Chunking policy:
            - Chunks NEVER split within-gene dimensions (2000bp axis, channels, or feature vectors).
            - Chunking (if any) is only along the gene axis.
    
    Returns:
        Tuple of (zarr_path or None, skip_reason or None, excluded_genes or None)
        - If successful: (path, None, excluded_genes_dict)
        - If skipped: (None, reason_string, None)
    """
    zarr_path = os.path.join(out_dir, f'{sample}.zarr')
    
    if os.path.exists(zarr_path) and not overwrite:
        # Verify existing store is valid (not empty)
        try:
            existing = zarr.open(zarr_path, mode='r')
            if 'flat' in existing and 'gene_names' in existing:
                flat = existing['flat']
                if 'TSS_signals' in flat and flat['TSS_signals'].shape[0] > 0:
                    return (zarr_path, None, None)  # Valid existing store
        except (KeyError, OSError, zarr.errors.ContainsGroupError) as e:
            # Invalid store, will regenerate
            print(f'  Note: Regenerating invalid store for {sample}: {e}')
    
    # Construct sample directories
    tss_sample_dir = os.path.join(tss_dir, sample)
    gb_sample_dir = os.path.join(gb_dir, sample)
    
    # Verify directories exist
    if not os.path.exists(tss_sample_dir):
        return (None, f'TSS directory not found', None)
    
    # Get TFX value - samples not in TFX file are SKIPPED
    if isinstance(tfx_path, (int, float)):
        tfx = float(tfx_path)
    else:
        try:
            tfx_df = pd.read_csv(tfx_path, sep='\t', index_col='sample')
            if sample in tfx_df.index:
                tfx = float(tfx_df.at[sample, 'tfx'])
            else:
                # Try matching by prefix
                matching = [idx for idx in tfx_df.index if sample in idx or idx in sample]
                if matching:
                    tfx = float(tfx_df.at[matching[0], 'tfx'])
                else:
                    return (None, 'Not in TFX file', None)
        except Exception as e:
            return (None, f'TFX read error: {str(e)[:50]}', None)

    print(f'Processing sample {sample}...')

    # Initialize gene exclusion tracker (collects exclusions from signal loading AND validation)
    excluded_genes = {
        'no_signals': 0, 
        'no_features': 0, 
        'high_nan': 0, 
        'no_ref': 0,
        'high_nan_signals': 0,  # Tracked during signal loading
    }

    # Load all data - pass exclusion tracker to capture genes dropped during loading
    tss_signals, tss_nucleotides = _load_signal_profiles(tss_sample_dir, sample, excluded_genes)
    
    # Load features with appropriate feature lists
    tss_features_raw = _load_features(tss_sample_dir, sample, TSS_FEATURES)
    gb_features_raw = _load_features(gb_sample_dir, sample, GB_FEATURES) if os.path.exists(gb_sample_dir) else {}
    
    # Load reference and labels
    ref_data = _process_ref(ref_path)
    tumor_exp_data = _process_label(tumor_exp_path, sample) if tumor_exp_path else {}
    background_exp_data = _process_label(background_exp_path, sample) if background_exp_path else {}
    
    # Apply feature-specific transforms
    tss_features = _transform_features(tss_features_raw, TSS_FEATURES)
    gb_features = _transform_features(gb_features_raw, GB_FEATURES)

    # Verify we have signal data (TSS is required)
    if not tss_signals:
        return (None, 'No TSS signal data', excluded_genes)

    # Determine valid genes (must have TSS signals, nucleotides, and features)
    valid_genes = []
    
    for gene in tss_signals:
        # Check signals shape
        if tss_signals[gene].shape != (2000, 8):
            excluded_genes['no_signals'] += 1
            continue
        # Check nucleotides exist
        if gene not in tss_nucleotides or tss_nucleotides[gene].shape != (2000, 4):
            excluded_genes['no_signals'] += 1
            continue
        # Check TSS features exist
        if gene not in tss_features:
            excluded_genes['no_features'] += 1
            continue
        # Check feature NaN fraction (after transforms/imputation, should be low)
        tss_feat = tss_features[gene]
        nan_frac = np.isnan(tss_feat).mean()
        if nan_frac > MAX_FEATURE_NAN_FRACTION:
            excluded_genes['high_nan'] += 1
            continue
        # Check reference exists
        if gene not in ref_data:
            excluded_genes['no_ref'] += 1
            continue
        
        valid_genes.append(gene)
    
    valid_genes.sort()
    n_genes = len(valid_genes)
    
    # Log exclusion summary
    total_excluded = sum(excluded_genes.values())
    if total_excluded > 0:
        print(f'  Excluded {total_excluded} genes: {excluded_genes}')
    
    if n_genes == 0:
        return (None, f'No valid genes (excluded: {excluded_genes})', excluded_genes)

    # Create Zarr store with explicit DirectoryStore for reliable persistence
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    
    # ==========================================================================
    # BUILD ALL ARRAYS FIRST (in memory), then write to Zarr once
    # This is more reliable than creating empty arrays and filling them
    # ==========================================================================
    
    # Pre-allocate numpy arrays
    all_tss_signals = np.zeros((n_genes, 2000, 8), dtype=np.float32)
    all_tss_nucleotides = np.zeros((n_genes, 2000, 4), dtype=np.uint8)
    all_tss_features = np.zeros((n_genes, 17), dtype=np.float32)
    all_gb_features = np.zeros((n_genes, 11), dtype=np.float32)
    all_ref_scalars = np.zeros((n_genes, 3), dtype=np.float32)
    all_tfx = np.full((n_genes,), tfx, dtype=np.float32)
    all_tumor_exp = np.full((n_genes,), np.nan, dtype=np.float32)
    all_background_exp = np.full((n_genes,), np.nan, dtype=np.float32)
    
    # Compute GB mean once if needed (for imputation)
    gb_mean = None
    if gb_features:
        gb_vals = [gb_features[g] for g in gb_features if g in valid_genes]
        if gb_vals:
            gb_mean = np.nanmean(gb_vals, axis=0)
            gb_mean = np.nan_to_num(gb_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    
    # Fill arrays
    for i, gene in enumerate(valid_genes):
        # TSS signals and nucleotides (required)
        all_tss_signals[i] = tss_signals[gene]
        all_tss_nucleotides[i] = tss_nucleotides[gene]
        all_tss_features[i] = tss_features[gene]
        
        # GB data (optional, use mean imputation)
        if gene in gb_features:
            all_gb_features[i] = gb_features[gene]
        elif gb_mean is not None:
            all_gb_features[i] = gb_mean
        
        # Reference scalars (required)
        all_ref_scalars[i] = ref_data[gene]
        
        # Expression labels (NaN is intentional for missing)
        if gene in tumor_exp_data:
            all_tumor_exp[i] = tumor_exp_data[gene]
        if gene in background_exp_data:
            all_background_exp[i] = background_exp_data[gene]
    
    # ==========================================================================
    # FINAL CLEANUP: Ensure NO NaN/Inf in training arrays (labels excluded)
    # ==========================================================================
    all_tss_signals = np.nan_to_num(all_tss_signals, nan=0.0, posinf=0.0, neginf=0.0)
    all_tss_features = np.nan_to_num(all_tss_features, nan=0.0, posinf=0.0, neginf=0.0)
    all_gb_features = np.nan_to_num(all_gb_features, nan=0.0, posinf=0.0, neginf=0.0)
    all_ref_scalars = np.nan_to_num(all_ref_scalars, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ==========================================================================
    # WRITE TO ZARR (flat-only)
    # ==========================================================================
    root = zarr.open_group(zarr_path, mode='w')
    
    # Sample-level metadata
    root.attrs['sample_name'] = sample
    root.attrs['n_genes'] = n_genes
    root.attrs['tfx'] = tfx
    # Intentionally do not record schema/version tags in attrs.
    
    # Gene name index (for stable alignment across arrays)
    root.create_dataset(
        'gene_names',
        data=np.array(valid_genes, dtype=object),
        dtype=object,
        object_codec=VLenUTF8(),
        compressor=COMPRESSOR,
    )

    # Flat arrays (primary access pattern: read whole sample, then shuffle genes in-memory)
    flat = root.create_group('flat')

    # Chunking: never split within-gene dimensions. Only chunk along genes.
    # For whole-sample reads, chunk size is not critical; keep it moderate.
    chunk_genes_large = int(min(256, n_genes))
    chunk_genes_small = int(min(1024, n_genes))

    flat.create_dataset(
        'TSS_signals',
        data=all_tss_signals,
        chunks=(chunk_genes_large, 2000, 8),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'TSS_nucleotides',
        data=all_tss_nucleotides,
        chunks=(chunk_genes_large, 2000, 4),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'TSS_features',
        data=all_tss_features,
        chunks=(chunk_genes_small, 17),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'GB_features',
        data=all_gb_features,
        chunks=(chunk_genes_small, 11),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'ref_scalars',
        data=all_ref_scalars,
        chunks=(chunk_genes_small, 3),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'TFX_label',
        data=all_tfx,
        chunks=(chunk_genes_small,),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'tumor_exp',
        data=all_tumor_exp,
        chunks=(chunk_genes_small,),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'background_exp',
        data=all_background_exp,
        chunks=(chunk_genes_small,),
        compressor=COMPRESSOR,
    )

    # Gene/sample weights (for per-gene weighting in loss; default=1.0)
    flat.create_dataset(
        'gene_weight',
        data=np.ones((n_genes,), dtype=np.float32),
        chunks=(chunk_genes_small,),
        compressor=COMPRESSOR,
    )
    
    # Consolidate metadata for faster subsequent opens
    zarr.consolidate_metadata(zarr_path)
    
    # Verify write succeeded - check that store can be reopened and has expected data
    try:
        verify_store = zarr.open(zarr_path, mode='r')
        actual_genes = len(verify_store['gene_names']) if 'gene_names' in verify_store else 0
        if actual_genes != n_genes:
            shutil.rmtree(zarr_path)
            return (None, f'Write verification failed: expected {n_genes} genes, found {actual_genes}', excluded_genes)
        
        # Check minimum file size to detect incomplete writes
        import subprocess
        result = subprocess.run(['du', '-sb', zarr_path], capture_output=True, text=True)
        if result.returncode == 0:
            size_bytes = int(result.stdout.split()[0])
            min_expected_size = n_genes * 1000  # Conservative estimate: ~1KB per gene minimum
            if size_bytes < min_expected_size:
                shutil.rmtree(zarr_path)
                return (None, f'Store too small: {size_bytes} bytes for {n_genes} genes', excluded_genes)
    except Exception as e:
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        return (None, f'Write verification error: {e}', excluded_genes)
    
    print(f'  -> {sample}: {n_genes} genes')
    return (zarr_path, None, excluded_genes)


def process_dataset(
    dataset: str,
    tss_dir: str,
    gb_dir: str,
    ref_path: str,
    labels_dir: str,
    tfx_file: str,
    zarr_base_dir: Optional[str] = None,
    tumor_exp_file: Optional[str] = None,
    background_exp_file: Optional[str] = None,
    overwrite: bool = False,
    n_workers: int = 1
) -> List[str]:
    """
    Process all samples in a dataset to Zarr stores.
    
    Args:
        dataset: Dataset name (e.g., 'PC-PDX')
        tss_dir: Absolute path to TSS results directory
        gb_dir: Absolute path to GB results directory
        ref_path: Path to reference file
        labels_dir: Directory containing label files
        tfx_file: Name of TFX file in labels_dir (required)
        zarr_base_dir: Base directory for Zarr output (default: DEFAULT_ZARR_BASE)
        tumor_exp_file: Name of tumor expression file in labels_dir
        background_exp_file: Name of background expression file in labels_dir
        overwrite: Whether to overwrite existing Zarr stores
        n_workers: Number of parallel workers
        
    Returns:
        List of created Zarr store paths
    """
    out_dir = get_zarr_output_dir(dataset, zarr_base_dir)

    # Discover samples BEFORE creating output directory.
    # This prevents confusing "empty zarr output dirs" when inputs are missing/mispointed.
    samples = get_samples(tss_dir, dataset)
    print(f'Found {len(samples)} samples in {tss_dir}')

    if not samples:
        print(f'No samples found for dataset {dataset}.')
        print(f'  TSS dir checked: {tss_dir}')
        print(f'  Ensure this directory exists and contains per-sample folders.')
        return []

    # Only create output directory once we know we have work to do.
    os.makedirs(out_dir, exist_ok=True)

    # Build label paths
    tfx_path = os.path.join(labels_dir, tfx_file)
    tumor_exp_path = os.path.join(labels_dir, tumor_exp_file) if tumor_exp_file else None
    background_exp_path = os.path.join(labels_dir, background_exp_file) if background_exp_file else None
    
    # Print actual paths being used
    print(f'  TSS dir: {tss_dir}')
    print(f'  GB dir: {gb_dir}')
    print(f'  TFX: {tfx_path}')
    print(f'  Tumor exp: {tumor_exp_path}')
    print(f'  Background exp: {background_exp_path}')
    print(f'  Output: {out_dir}')
    
    # Process samples - now returns (path, skip_reason, excluded_genes) tuples
    created_paths = []
    skipped_with_reasons = {}  # sample -> reason
    gene_exclusion_stats = {}  # sample -> excluded_genes dict
    
    if n_workers <= 1:
        # Sequential processing
        for sample in samples:
            result = process_sample_to_zarr(
                sample=sample,
                tss_dir=tss_dir,
                gb_dir=gb_dir,
                ref_path=ref_path,
                tfx_path=tfx_path,
                tumor_exp_path=tumor_exp_path,
                background_exp_path=background_exp_path,
                out_dir=out_dir,
                overwrite=overwrite
            )
            path, skip_reason, excluded_genes = result
            if path:
                created_paths.append(path)
                if excluded_genes and sum(excluded_genes.values()) > 0:
                    gene_exclusion_stats[sample] = excluded_genes
            elif skip_reason:
                skipped_with_reasons[sample] = skip_reason
                if excluded_genes:
                    gene_exclusion_stats[sample] = excluded_genes
    else:
        # Parallel processing using functools.partial
        worker_fn = partial(
            process_sample_to_zarr,
            tss_dir=tss_dir,
            gb_dir=gb_dir,
            ref_path=ref_path,
            tfx_path=tfx_path,
            tumor_exp_path=tumor_exp_path,
            background_exp_path=background_exp_path,
            out_dir=out_dir,
            overwrite=overwrite
        )
        
        with Pool(n_workers) as pool:
            results = pool.map(worker_fn, samples)
        
        # Process results - each is (path, skip_reason, excluded_genes)
        for sample, result in zip(samples, results):
            path, skip_reason, excluded_genes = result
            if path:
                created_paths.append(path)
                if excluded_genes and sum(excluded_genes.values()) > 0:
                    gene_exclusion_stats[sample] = excluded_genes
            elif skip_reason:
                skipped_with_reasons[sample] = skip_reason
                if excluded_genes:
                    gene_exclusion_stats[sample] = excluded_genes
    
    # Compute aggregate gene exclusion statistics
    aggregate_exclusions = {
        'no_signals': 0, 
        'no_features': 0, 
        'high_nan': 0, 
        'no_ref': 0,
        'high_nan_signals': 0,  # Genes dropped during signal loading due to NaN
    }
    for sample_stats in gene_exclusion_stats.values():
        for reason, count in sample_stats.items():
            if reason in aggregate_exclusions:
                aggregate_exclusions[reason] += count
            else:
                # Handle any unexpected keys gracefully
                aggregate_exclusions[reason] = aggregate_exclusions.get(reason, 0) + count
    
    # Save processing manifest with actual paths, skip reasons, and gene exclusion stats
    from datetime import datetime
    canonical_dataset = normalize_cohort_name(dataset)
    manifest = {
        'dataset': canonical_dataset,
        'original_dataset_name': dataset if dataset != canonical_dataset else None,
        'timestamp': datetime.now().isoformat(),
        'paths': {
            'tss_dir': tss_dir,
            'gb_dir': gb_dir,
            'tfx_file': str(tfx_path),
            'tumor_exp_file': tumor_exp_path,
            'background_exp_file': background_exp_path,
            'zarr_output_dir': out_dir,
        },
        'counts': {
            'total_discovered': len(samples),
            'successfully_processed': len(created_paths),
            'skipped': len(skipped_with_reasons),
        },
        'skipped_samples': skipped_with_reasons,
        'gene_exclusions': {
            'per_sample': gene_exclusion_stats,
            'aggregate': aggregate_exclusions,
            'total_excluded': sum(aggregate_exclusions.values()),
        },
    }
    manifest_path = os.path.join(out_dir, 'processing_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f'\\nProcessed {len(created_paths)} / {len(samples)} samples for {dataset}')
    if skipped_with_reasons:
        print(f'  Skipped {len(skipped_with_reasons)} samples:')
        # Group by reason
        reasons = {}
        for sample, reason in skipped_with_reasons.items():
            reasons.setdefault(reason, []).append(sample)
        for reason, samps in reasons.items():
            print(f'    - {reason}: {len(samps)} samples')
    
    # Print gene exclusion summary
    total_excluded = sum(aggregate_exclusions.values())
    if total_excluded > 0:
        print(f'  Gene exclusions (aggregate across all samples):')
        for reason, count in aggregate_exclusions.items():
            if count > 0:
                print(f'    - {reason}: {count:,} genes')
        print(f'    Total: {total_excluded:,} genes excluded')
    
    print(f'  Manifest: {manifest_path}')
    
    return created_paths


# =============================================================================
# Fold Index Creation
# =============================================================================

def create_sample_splits(
    dataset: str,
    zarr_dir: str,
    output_path: str,
    n_folds: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create sample splits file for cross-validation from existing Zarr stores.
    
    Args:
        dataset: Name of the dataset/cohort
        zarr_dir: Directory containing sample Zarr stores
        output_path: Path to save the splits TSV file
        n_folds: Number of CV folds
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with Sample, Fold, Cohort columns
    """
    # Find all Zarr stores
    zarr_files = [f for f in os.listdir(zarr_dir) if f.endswith('.zarr')]
    samples = [f.replace('.zarr', '') for f in zarr_files]
    
    if not samples:
        raise ValueError(f"No Zarr stores found in {zarr_dir}")
    
    # Shuffle and assign folds
    np.random.seed(seed)
    shuffled_samples = samples.copy()
    np.random.shuffle(shuffled_samples)
    
    folds = [(i % n_folds) + 1 for i in range(len(shuffled_samples))]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sample': shuffled_samples,
        'Fold': folds,
        'Cohort': dataset
    })
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f'Created sample splits with {len(samples)} samples across {n_folds} folds')
    print(f'Saved to: {output_path}')
    
    return df


def create_fold_index(
    dataset: str,
    zarr_dir: str,
    output_dir: str,
    splits_file: Optional[str] = None,
    n_folds: int = 5
) -> Dict[int, List[str]]:
    """
    Create fold index files that map sample Zarr paths for each fold.

    Each fold index is a JSON list of absolute `*.zarr` paths (one per sample).
    During training, genes are shuffled in-memory across the batch of samples,
    so we do NOT store per-gene indices in the fold files.
    
    Args:
        dataset: Name of the dataset
        zarr_dir: Directory containing sample Zarr stores
        output_dir: Output directory for index files
        splits_file: Path to sample splits file (optional, will create if missing)
        n_folds: Number of CV folds (used if creating splits)
        
    Returns:
        Dictionary mapping fold numbers to lists of sample zarr paths
    """
    # Handle splits file
    if splits_file is None:
        splits_file = os.path.join(zarr_dir, f'{dataset}_sample-splits.tsv')
    
    if not os.path.exists(splits_file):
        print(f'Sample splits file not found: {splits_file}')
        print('Creating sample splits from existing Zarr stores...')
        create_sample_splits(dataset, zarr_dir, splits_file, n_folds=n_folds)
    
    splits_df = pd.read_csv(splits_file, sep='\t')
    
    # Make zarr_dir absolute
    if not os.path.isabs(zarr_dir):
        zarr_dir = os.path.abspath(zarr_dir)
    
    # Group by fold
    fold_indices: Dict[int, List[str]] = {}
    for _, row in splits_df.iterrows():
        sample = row['Sample']
        fold = int(row['Fold'])
        
        zarr_path = os.path.join(zarr_dir, f'{sample}.zarr')
        
        if not os.path.exists(zarr_path):
            print(f'Warning: Zarr store not found for {sample}. Skipping.')
            continue

        # Light validation: ensure it's openable and has flat arrays
        try:
            store = zarr.open(zarr_path, mode='r')
            if 'flat' not in store:
                print(f'Warning: Missing flat/ group in {zarr_path}. Skipping.')
                continue
            if 'gene_names' not in store:
                print(f'Warning: Missing gene_names in {zarr_path}. Skipping.')
                continue
        except Exception as e:
            print(f'Error reading {zarr_path}: {e}. Skipping.')
            continue

        fold_indices.setdefault(fold, []).append(zarr_path)
    
    if not fold_indices:
        raise ValueError("No valid samples found. Check Zarr stores and splits file.")
    
    # Save index files
    os.makedirs(output_dir, exist_ok=True)
    
    for fold, indices in fold_indices.items():
        np.random.seed(42 + fold)
        np.random.shuffle(indices)
        
        index_file = os.path.join(output_dir, f'fold_{fold}_index.json')
        with open(index_file, 'w') as f:
            json.dump(indices, f)
        
        print(f'Fold {fold}: {len(indices)} samples')
    
    metadata = {
        'dataset': dataset,
        'n_folds': len(fold_indices),
        'fold_sizes': {str(fold): len(indices) for fold, indices in fold_indices.items()},
        'total_samples': sum(len(indices) for indices in fold_indices.values()),
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return fold_indices


# =============================================================================
# Verification Functions
# =============================================================================

def verify_zarr_store(zarr_path: str, check_nan_inf: bool = True) -> Dict:
    """
    Verify a Zarr store and return summary statistics.

    Enforces the single supported layout (flat arrays under flat/).
    Checks for NaN/Inf in training arrays (expression labels may have NaN).
    
    Args:
        zarr_path: Path to Zarr store
        check_nan_inf: Whether to check for NaN/Inf in training arrays
        
    Returns:
        Dictionary with validation results
    """
    try:
        store = zarr.open(zarr_path, mode='r')
        
        n_genes = store.attrs['n_genes']
        sample_name = store.attrs['sample_name']

        if 'flat' not in store:
            raise ValueError(f"Missing required group 'flat' in {zarr_path}")
        data_grp = store['flat']
        
        # Required arrays (in data group)
        required_arrays = ['TSS_signals', 'TSS_nucleotides', 'TSS_features',
                         'GB_features', 'ref_scalars',
                         'TFX_label', 'tumor_exp', 'background_exp']

        required_arrays.append('gene_weight')
        missing = [r for r in required_arrays if r not in data_grp]
        
        # Also need gene_names at root
        if 'gene_names' not in store:
            missing.append('gene_names')
        
        shapes = {}
        for key in required_arrays:
            if key in data_grp:
                shapes[key] = data_grp[key].shape
        
        # Expression labels - NaN is ALLOWED (flags missing data)
        tumor_nan = np.isnan(data_grp['tumor_exp'][:]).sum() if 'tumor_exp' in data_grp else n_genes
        background_nan = np.isnan(data_grp['background_exp'][:]).sum() if 'background_exp' in data_grp else n_genes
        
        # Check training arrays for NaN/Inf (NOT allowed)
        nan_inf_issues = {}
        if check_nan_inf:
            training_arrays = [
                'TSS_signals', 'TSS_nucleotides', 'TSS_features',
                'GB_features', 'ref_scalars', 'TFX_label'
            ]
            if 'gene_weight' in required_arrays:
                training_arrays.append('gene_weight')
            for arr_name in training_arrays:
                if arr_name in data_grp:
                    arr = data_grp[arr_name][:]
                    nan_count = int(np.isnan(arr).sum())
                    inf_count = int(np.isinf(arr).sum())
                    if nan_count > 0 or inf_count > 0:
                        nan_inf_issues[arr_name] = {'nan': nan_count, 'inf': inf_count}
        
        # Check file size (to detect empty stores)
        import subprocess
        try:
            result = subprocess.run(['du', '-sh', zarr_path], capture_output=True, text=True)
            size_str = result.stdout.split()[0] if result.stdout else 'unknown'
        except:
            size_str = 'unknown'
        
        return {
            'valid': len(missing) == 0 and len(nan_inf_issues) == 0,
            'sample_name': sample_name,
            'n_genes': n_genes,
            'tfx': store.attrs.get('tfx', 'N/A'),
            'disk_size': size_str,
            'missing_arrays': missing,
            'shapes': shapes,
            'tumor_nan_count': int(tumor_nan),
            'background_nan_count': int(background_nan),
            'labeled': tumor_nan < n_genes,
            'nan_inf_issues': nan_inf_issues  # Empty dict = clean
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }


def verify_dataset(zarr_dir: str) -> pd.DataFrame:
    """
    Verify all Zarr stores in a directory.
    
    Reports:
    - Schema compliance
    - NaN/Inf in training arrays (NOT allowed)
    - NaN in expression labels (allowed, used as flag)
    """
    results = []
    zarr_files = [f for f in os.listdir(zarr_dir) if f.endswith('.zarr')]
    
    nan_inf_samples = []
    
    for zarr_file in zarr_files:
        zarr_path = os.path.join(zarr_dir, zarr_file)
        result = verify_zarr_store(zarr_path, check_nan_inf=True)
        result['file'] = zarr_file
        results.append(result)
        
        if result.get('nan_inf_issues'):
            nan_inf_samples.append((zarr_file, result['nan_inf_issues']))
    
    df = pd.DataFrame(results)
    
    print(f"\nDataset verification summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Valid samples: {df['valid'].sum()}")
    print(f"  Invalid samples: {(~df['valid']).sum()}")
    if 'n_genes' in df.columns:
        print(f"  Total genes: {df['n_genes'].sum():,}")
        print(f"  Avg genes/sample: {df['n_genes'].mean():.0f}")
    if 'labeled' in df.columns:
        print(f"  Labeled samples: {df['labeled'].sum()}")
    
    # Report NaN/Inf issues
    if nan_inf_samples:
        print(f"\n  *** WARNING: {len(nan_inf_samples)} samples have NaN/Inf in training arrays ***")
        for sample, issues in nan_inf_samples[:5]:  # Show first 5
            print(f"    {sample}: {issues}")
        if len(nan_inf_samples) > 5:
            print(f"    ... and {len(nan_inf_samples) - 5} more")
    else:
        print(f"\n  ✓ All training arrays are clean (no NaN/Inf)")
    
    return df
