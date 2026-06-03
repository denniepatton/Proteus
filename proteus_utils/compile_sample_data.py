"""compile_sample_data.py (PyTorch version)

This script contains functions for compiling sample data from raw Triton outputs
and reference files into optimized Zarr stores for PyTorch training.

Pipeline: Raw Triton files → Zarr stores → Fold indices → PyTorch DataLoader

Raw Triton output structure (per sample folder):
    SAMPLE_TritonSignalProfiles.npz - 12 signal channels x 2000 bp (TSS only)
    SAMPLE_TritonFeatures.tsv - 19 columns of region-level features

Zarr layout (flat arrays for batch loading):
    sample.zarr/
      ├── gene_names: array of gene IDs for indexing
      ├── attrs: {sample_name, n_genes, tfx}
      └── flat/
          ├── TSS_signals      (n_genes, 2000, 9)   float32
          ├── TSS_nucleotides  (n_genes, 2000, 4)   uint8
          ├── TSS_features     (n_genes, 11)        float32
          ├── GB_features      (n_genes, 11)        float32
          ├── ref_scalars      (n_genes, 3)         float32
          ├── TFX_label        (n_genes,)           float32
          ├── tumor_exp        (n_genes,)           float32
          ├── background_exp   (n_genes,)           float32
          └── gene_weight      (n_genes,)           float32

Feature transforms (applied at Zarr conversion time):
    - mean-region-depth: log1p only (preserves absolute depth info)
    - fl-kurtosis, pn-mean-spacing, pn-mean-amplitude: log + z-score
    - All others: z-score only
    
NaN handling:
    - Sample-mean imputation for features after transforms
    - Signal NaN positions are imputed to 0 and recorded in a 9th binary mask channel
    - Genes with >50% NaN in features are dropped (no signal NaN threshold)
"""

# Standard library imports
import os
import json
import logging
import shutil
from functools import partial
from pathlib import Path
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
    
    # Try to load from paths.yaml
    try:
        import yaml as _yaml
        _paths_file = Path(__file__).resolve().parent.parent / 'config' / 'paths.yaml'
        with open(_paths_file) as _f:
            _paths = _yaml.safe_load(_f)
        return _paths['base']['zarr_base_dir']
    except (FileNotFoundError, KeyError, OSError) as e:
        logging.debug('Config-based zarr_base_dir lookup failed: %s', e)
    
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
    
    # Try to load from paths.yaml
    try:
        import yaml as _yaml
        _paths_file = Path(__file__).resolve().parent.parent / 'config' / 'paths.yaml'
        with open(_paths_file) as _f:
            _paths = _yaml.safe_load(_f)
        return _paths['base']['raw_data_dir']
    except (FileNotFoundError, KeyError, OSError) as e:
        logging.debug('Config-based raw_data_dir lookup failed: %s', e)
    
    # Fallback to hardcoded default
    return "/fh/fast/ha_g/user/rpatton"


# Default Zarr output directory (loaded from config or environment)
DEFAULT_ZARR_BASE = _get_default_zarr_base()

# Compression settings for Zarr
COMPRESSOR = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

# Signal channels (8 biological signals + 1 NaN mask)
SIGNAL_CHANNELS = [
    'depth',                    # 0: GC-corrected coverage
    'frag_end_coverage',        # 1: Fragment end density
    'frag_end_asymmetry',       # 2: (5' - 3') / (5' + 3') orientation
    'pn_signal',                # 3: Phased-nucleosome signal
    'fl_subnucleosomal_ratio',  # 4: log2(< 147 bp / >= 147 bp)
    'fl_entropy',               # 5: Shannon entropy
    'fl_gini_simpson',          # 6: Diversity index
    'pn_peaks',                 # 7: Local maxima/minima
    'nan_mask',                 # 8: 1 where any signal channel had NaN, 0 otherwise
]

# Nucleotide one-hot channels (stored separately as uint8)
NUCLEOTIDE_CHANNELS = ['A', 'C', 'G', 'T']

# TSS feature columns (11 shared Triton features — matches GB)
# NOTE: central-*/window-* (6 TSS-window-specific features) have been dropped.
# They are always NaN in GB mode, and unused by the model.
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
]

# GB features (11 features — identical feature set to TSS)
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
LOG_TRANSFORM_FEATURES = ['fl-kurtosis', 'pn-mean-spacing', 'pn-mean-amplitude']

# NaN thresholds for gene exclusion (signal NaN is encoded, not thresholded)
MAX_FEATURE_NAN_FRACTION = 0.5

# Signal clipping bounds for mean-normalized channels.
# After division by gene-level mean, values beyond these bounds are extreme
# spikes (>10× the mean) — clipped to prevent float16 overflow under AMP.
SIGNAL_CLIP_MIN = -10.0
SIGNAL_CLIP_MAX = 10.0

# Data shapes for Zarr stores
SHAPES = {
    'TSS_signals': (2000, 9),
    'TSS_nucleotides': (2000, 4),
    'TSS_features': (len(TSS_FEATURES),),
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
    
    return sorted(samples)


# =============================================================================
# Data Processing Functions
# =============================================================================

def _scale_signal(input_array: np.ndarray) -> np.ndarray:
    """Scale an input array by dividing by its mean (avoid division by zero/Inf).

    After division the expected baseline is ~1.0.  Values beyond the clipping
    bounds (SIGNAL_CLIP_MIN/MAX) are extreme spikes (>10× the gene-level mean)
    and are clipped to prevent float16 overflow under AMP and to suppress noise.
    """
    mean_val = np.nanmean(input_array)
    if mean_val > 1e-10 and not np.isnan(mean_val):  # Use small threshold, not just != 0
        result = input_array / mean_val
        result = np.clip(result, SIGNAL_CLIP_MIN, SIGNAL_CLIP_MAX)
        return result
    return input_array


def _load_signal_profiles(
    sample_dir: str,
    sample: str,
    excluded_genes_tracker: Optional[Dict[str, int]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load signal profiles from TritonV2 output.
    
    Separates biological signals (8 channels) from nucleotide OHE (4 channels),
    then appends a 9th binary NaN mask channel (1 = any signal was NaN at that
    position, 0 = all signals observed).  NaN values are imputed to 0.
    
    Args:
        sample_dir: Path to sample's Triton output folder
        sample: Sample name
        excluded_genes_tracker: Optional dict to track excluded genes by reason
        
    Returns:
        Tuple of:
            - signals_dict: Gene name -> (2000, 9) float32 signals + NaN mask
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
        
        # Handle shape — expect (2000, 12); accept transposed (12, 2000).
        if raw_data.shape == (12, 2000):
            raw_data = raw_data.T
        elif raw_data.shape == (2000, 12):
            pass
        else:
            raise ValueError(
                f'Unexpected signal shape for gene {gene}: {raw_data.shape}. '
                f'Expected (2000, 12) or (12, 2000).'
            )
        
        # Split into signals (0:8) and nucleotides (8:12)
        signals = raw_data[:, :8].astype(np.float32)
        
        # Handle nucleotide conversion with suppressed warnings (invalid values -> 0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            nuc_raw = raw_data[:, 8:12]
            # Replace any NaN/Inf before casting to uint8
            nuc_raw = np.nan_to_num(nuc_raw, nan=0.0, posinf=0.0, neginf=0.0)
            nucleotides = nuc_raw.astype(np.uint8)
        
        # Build NaN mask BEFORE any imputation: 1 where any channel had NaN
        nan_mask = np.isnan(signals).any(axis=1).astype(np.float32)  # (2000,)
        
        # Mean-normalize channels 0, 1, 3 before imputation (nanmean ignores NaN)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            # Channel 0 (depth): normalize by mean
            signals[:, 0] = _scale_signal(signals[:, 0])
            # Channel 1 (frag end coverage): normalize by mean
            signals[:, 1] = _scale_signal(signals[:, 1])
            # Channel 3 (PN signal): normalize by mean
            signals[:, 3] = _scale_signal(signals[:, 3])
        
        # Impute all NaN/Inf to 0 (not channel-mean)
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Append NaN mask as 9th channel
        signals = np.concatenate([signals, nan_mask[:, np.newaxis]], axis=1)  # (2000, 9)
        
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
    """Process a label file and return data as a dictionary.

    Uses exact column match first; if that fails, falls back to
    substring matching (e.g. bare sample name ``22-002`` matching
    prefixed column ``CRPC-RA_22-002``).  The fuzzy match must be
    unique (exactly one column) to succeed.
    """
    if filepath is None:
        return {}
    df = pd.read_csv(filepath, sep='\t', index_col=0)
    if sample not in df.columns:
        # Fuzzy match: bare sample name vs prefixed columns (or vice-versa)
        matching = [c for c in df.columns if sample in c or c in sample]
        if len(matching) == 1:
            sample = matching[0]
        else:
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
    return os.path.join(base_dir, dataset, 'zarr_files')


def _resolve_tfx(
    tfx_path: Union[str, float, None],
    sample: str,
) -> Tuple[float, Optional[str]]:
    """Resolve TFX value from None, numeric, or file path.

    Returns (tfx_value, error_reason).  On success error_reason is None.
    On failure tfx_value is NaN and error_reason describes the problem.
    """
    if tfx_path is None:
        return float('nan'), None
    if isinstance(tfx_path, (int, float)):
        return float(tfx_path), None
    try:
        tfx_df = pd.read_csv(tfx_path, sep='\t', index_col='sample')
        if sample in tfx_df.index:
            return float(tfx_df.at[sample, 'tfx']), None
        # Try matching by prefix
        matching = [idx for idx in tfx_df.index if sample in idx or idx in sample]
        if matching:
            return float(tfx_df.at[matching[0], 'tfx']), None
        return float('nan'), 'Not in TFX file'
    except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as e:
        return float('nan'), f'TFX read error: {e}'


def _validate_genes(
    tss_signals: Dict[str, np.ndarray],
    tss_nucleotides: Dict[str, np.ndarray],
    tss_features: Dict[str, np.ndarray],
    ref_data: Dict[str, np.ndarray],
) -> Tuple[List[str], Dict[str, int]]:
    """Determine valid genes and count exclusions by reason.

    Returns (sorted valid_genes list, excluded_counts dict).
    """
    excluded_counts: Dict[str, int] = {
        'no_signals': 0,
        'no_features': 0,
        'high_nan': 0,
        'no_ref': 0,
    }
    valid_genes: List[str] = []

    for gene in tss_signals:
        if tss_signals[gene].shape != (2000, 9):
            excluded_counts['no_signals'] += 1
            continue
        if gene not in tss_nucleotides or tss_nucleotides[gene].shape != (2000, 4):
            excluded_counts['no_signals'] += 1
            continue
        if gene not in tss_features:
            excluded_counts['no_features'] += 1
            continue
        nan_frac = np.isnan(tss_features[gene]).mean()
        if nan_frac > MAX_FEATURE_NAN_FRACTION:
            excluded_counts['high_nan'] += 1
            continue
        if gene not in ref_data:
            excluded_counts['no_ref'] += 1
            continue
        valid_genes.append(gene)

    valid_genes.sort()
    return valid_genes, excluded_counts


def _build_flat_arrays(
    valid_genes: List[str],
    tss_signals: Dict[str, np.ndarray],
    tss_nucleotides: Dict[str, np.ndarray],
    tss_features: Dict[str, np.ndarray],
    gb_features: Dict[str, np.ndarray],
    ref_data: Dict[str, np.ndarray],
    tumor_exp_data: Dict[str, float],
    background_exp_data: Dict[str, float],
    tfx: float,
) -> Dict[str, np.ndarray]:
    """Build all flat numpy arrays for Zarr storage.

    Returns a dict keyed by Zarr dataset name.
    """
    n_genes = len(valid_genes)

    all_tss_signals = np.zeros((n_genes, 2000, 9), dtype=np.float32)
    all_tss_nucleotides = np.zeros((n_genes, 2000, 4), dtype=np.uint8)
    all_tss_features = np.zeros((n_genes, len(TSS_FEATURES)), dtype=np.float32)
    all_gb_features = np.zeros((n_genes, 11), dtype=np.float32)
    all_ref_scalars = np.zeros((n_genes, 3), dtype=np.float32)
    all_tfx = np.full((n_genes,), tfx, dtype=np.float32)
    all_tumor_exp = np.full((n_genes,), np.nan, dtype=np.float32)
    all_background_exp = np.full((n_genes,), np.nan, dtype=np.float32)

    # Compute GB mean once for imputation
    gb_mean = None
    if gb_features:
        gb_vals = [gb_features[g] for g in gb_features if g in valid_genes]
        if gb_vals:
            gb_mean = np.nanmean(gb_vals, axis=0)
            gb_mean = np.nan_to_num(gb_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    for i, gene in enumerate(valid_genes):
        all_tss_signals[i] = tss_signals[gene]
        all_tss_nucleotides[i] = tss_nucleotides[gene]
        all_tss_features[i] = tss_features[gene]

        if gene in gb_features:
            all_gb_features[i] = gb_features[gene]
        elif gb_mean is not None:
            all_gb_features[i] = gb_mean

        all_ref_scalars[i] = ref_data[gene]

        if gene in tumor_exp_data:
            all_tumor_exp[i] = tumor_exp_data[gene]
        if gene in background_exp_data:
            all_background_exp[i] = background_exp_data[gene]

    # Final NaN/Inf cleanup for training arrays (labels excluded)
    all_tss_signals = np.nan_to_num(all_tss_signals, nan=0.0, posinf=0.0, neginf=0.0)
    all_tss_features = np.nan_to_num(all_tss_features, nan=0.0, posinf=0.0, neginf=0.0)
    all_gb_features = np.nan_to_num(all_gb_features, nan=0.0, posinf=0.0, neginf=0.0)
    all_ref_scalars = np.nan_to_num(all_ref_scalars, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'TSS_signals': all_tss_signals,
        'TSS_nucleotides': all_tss_nucleotides,
        'TSS_features': all_tss_features,
        'GB_features': all_gb_features,
        'ref_scalars': all_ref_scalars,
        'TFX_label': all_tfx,
        'tumor_exp': all_tumor_exp,
        'background_exp': all_background_exp,
    }


def _write_zarr_store(
    zarr_path: str,
    sample: str,
    valid_genes: List[str],
    arrays: Dict[str, np.ndarray],
    tfx: float,
    n_genes: int,
) -> None:
    """Write all arrays to a Zarr store. Raises on error."""
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    root = zarr.open_group(zarr_path, mode='w')

    root.attrs['sample_name'] = sample
    root.attrs['n_genes'] = n_genes
    root.attrs['tfx'] = tfx

    root.create_dataset(
        'gene_names',
        data=np.array(valid_genes, dtype=object),
        dtype=object,
        object_codec=VLenUTF8(),
        compressor=COMPRESSOR,
    )

    flat = root.create_group('flat')

    chunk_genes_large = int(min(256, n_genes))
    chunk_genes_small = int(min(1024, n_genes))

    flat.create_dataset(
        'TSS_signals',
        data=arrays['TSS_signals'],
        chunks=(chunk_genes_large, 2000, 9),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'TSS_nucleotides',
        data=arrays['TSS_nucleotides'],
        chunks=(chunk_genes_large, 2000, 4),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'TSS_features',
        data=arrays['TSS_features'],
        chunks=(chunk_genes_small, len(TSS_FEATURES)),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'GB_features',
        data=arrays['GB_features'],
        chunks=(chunk_genes_small, 11),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'ref_scalars',
        data=arrays['ref_scalars'],
        chunks=(chunk_genes_small, 3),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'TFX_label',
        data=arrays['TFX_label'],
        chunks=(chunk_genes_small,),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'tumor_exp',
        data=arrays['tumor_exp'],
        chunks=(chunk_genes_small,),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'background_exp',
        data=arrays['background_exp'],
        chunks=(chunk_genes_small,),
        compressor=COMPRESSOR,
    )
    flat.create_dataset(
        'gene_weight',
        data=np.ones((n_genes,), dtype=np.float32),
        chunks=(chunk_genes_small,),
        compressor=COMPRESSOR,
    )

    zarr.consolidate_metadata(zarr_path)


def _verify_zarr_write(zarr_path: str, n_genes: int) -> Optional[str]:
    """Verify a Zarr store was written correctly.

    Returns an error message string on failure, or None on success.
    """
    import subprocess
    try:
        verify_store = zarr.open(zarr_path, mode='r')
        actual_genes = len(verify_store['gene_names']) if 'gene_names' in verify_store else 0
        if actual_genes != n_genes:
            shutil.rmtree(zarr_path)
            return f'Write verification failed: expected {n_genes} genes, found {actual_genes}'

        result = subprocess.run(['du', '-sb', zarr_path], capture_output=True, text=True)
        if result.returncode == 0:
            size_bytes = int(result.stdout.split()[0])
            min_expected_size = n_genes * 1000
            if size_bytes < min_expected_size:
                shutil.rmtree(zarr_path)
                return f'Store too small: {size_bytes} bytes for {n_genes} genes'
    except (OSError, subprocess.SubprocessError, ValueError, zarr.errors.PathNotFoundError) as e:
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        return f'Write verification error: {e}'
    return None


def process_sample_to_zarr(
    sample: str,
    tss_dir: str,
    gb_dir: str,
    ref_path: str,
    tfx_path: Union[str, float, None],
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
                    ├── TSS_signals (n_genes, 2000, 9) float32
                    ├── TSS_nucleotides (n_genes, 2000, 4) uint8
                    ├── TSS_features (n_genes, 11) float32
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

    # --- Early return: valid existing store ---
    if os.path.exists(zarr_path) and not overwrite:
        try:
            existing = zarr.open(zarr_path, mode='r')
            if 'flat' in existing and 'gene_names' in existing:
                flat = existing['flat']
                if 'TSS_signals' in flat and flat['TSS_signals'].shape[0] > 0:
                    return (zarr_path, None, None)
        except (KeyError, OSError, zarr.errors.ContainsGroupError, zarr.errors.PathNotFoundError) as e:
            print(f'  Note: Regenerating invalid store for {sample}: {e}')

    # --- Verify TSS directory ---
    tss_sample_dir = os.path.join(tss_dir, sample)
    gb_sample_dir = os.path.join(gb_dir, sample)
    if not os.path.exists(tss_sample_dir):
        return (None, 'TSS directory not found', None)

    # --- Resolve TFX ---
    tfx, tfx_err = _resolve_tfx(tfx_path, sample)
    if tfx_err is not None:
        return (None, tfx_err, None)

    print(f'Processing sample {sample}...')

    # --- Load raw data ---
    excluded_genes: Dict[str, int] = {
        'no_signals': 0,
        'no_features': 0,
        'high_nan': 0,
        'no_ref': 0,
    }
    tss_signals, tss_nucleotides = _load_signal_profiles(tss_sample_dir, sample, excluded_genes)
    tss_features_raw = _load_features(tss_sample_dir, sample, TSS_FEATURES)
    gb_features_raw = _load_features(gb_sample_dir, sample, GB_FEATURES) if os.path.exists(gb_sample_dir) else {}
    ref_data = _process_ref(ref_path)
    tumor_exp_data = _process_label(tumor_exp_path, sample) if tumor_exp_path else {}
    background_exp_data = _process_label(background_exp_path, sample) if background_exp_path else {}

    if tumor_exp_path and not tumor_exp_data:
        print(f'  WARNING: tumor_exp file provided but sample "{sample}" not matched '
              f'in {os.path.basename(tumor_exp_path)}')
    if background_exp_path and not background_exp_data:
        print(f'  WARNING: background_exp file provided but sample "{sample}" not matched '
              f'in {os.path.basename(background_exp_path)}')

    # --- Transform features ---
    tss_features = _transform_features(tss_features_raw, TSS_FEATURES)
    gb_features = _transform_features(gb_features_raw, GB_FEATURES)

    if not tss_signals:
        return (None, 'No TSS signal data', excluded_genes)

    # --- Validate genes ---
    valid_genes, validation_excluded = _validate_genes(
        tss_signals, tss_nucleotides, tss_features, ref_data,
    )
    for key in validation_excluded:
        excluded_genes[key] = excluded_genes.get(key, 0) + validation_excluded[key]

    n_genes = len(valid_genes)
    total_excluded = sum(excluded_genes.values())
    if total_excluded > 0:
        print(f'  Excluded {total_excluded} genes: {excluded_genes}')
    if n_genes == 0:
        return (None, f'No valid genes (excluded: {excluded_genes})', excluded_genes)

    # --- Build arrays and write Zarr ---
    arrays = _build_flat_arrays(
        valid_genes, tss_signals, tss_nucleotides, tss_features,
        gb_features, ref_data, tumor_exp_data, background_exp_data, tfx,
    )
    os.makedirs(out_dir, exist_ok=True)
    _write_zarr_store(zarr_path, sample, valid_genes, arrays, tfx, n_genes)

    # --- Verify ---
    verify_err = _verify_zarr_write(zarr_path, n_genes)
    if verify_err is not None:
        return (None, verify_err, excluded_genes)

    print(f'  -> {sample}: {n_genes} genes')
    return (zarr_path, None, excluded_genes)


def process_dataset(
    dataset: str,
    tss_dir: str,
    gb_dir: str,
    ref_path: str,
    labels_dir: str,
    tfx_file: Optional[str],
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
        tfx_file: Name of TFX file in labels_dir (None for unlabeled datasets)
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
    tfx_path = os.path.join(labels_dir, tfx_file) if tfx_file else None
    tumor_exp_path = os.path.join(labels_dir, tumor_exp_file) if tumor_exp_file else None
    background_exp_path = os.path.join(labels_dir, background_exp_file) if background_exp_file else None
    
    # Print actual paths being used
    print(f'  TSS dir: {tss_dir}')
    print(f'  GB dir: {gb_dir}')
    print(f'  TFX: {tfx_path}')
    print(f'  Tumor exp: {tumor_exp_path}')
    print(f'  Background exp: {background_exp_path}')
    print(f'  Output: {out_dir}')

    # Pre-flight: warn if a label file is provided but no samples match its columns.
    # This catches naming mismatches (e.g. bare "22-002" vs prefixed "CRPC-RA_22-002")
    # before spending time on per-sample Zarr writes.
    for label_tag, label_path in [('tumor_exp', tumor_exp_path),
                                  ('background_exp', background_exp_path)]:
        if label_path is None:
            continue
        try:
            label_cols = set(pd.read_csv(label_path, sep='\t', index_col=0, nrows=0).columns)
        except (FileNotFoundError, pd.errors.ParserError) as e:
            logging.warning('Could not read label file %s for pre-flight check: %s', label_path, e)
            continue
        exact_hits = sum(1 for s in samples if s in label_cols)
        fuzzy_hits = sum(
            1 for s in samples
            if s not in label_cols and
            len([c for c in label_cols if s in c or c in s]) == 1
        )
        if exact_hits == 0 and fuzzy_hits == 0:
            raise ValueError(
                f'Label file {os.path.basename(label_path)} ({label_tag}) has '
                f'{len(label_cols)} columns but NONE match any of the '
                f'{len(samples)} discovered samples.  First 3 samples: '
                f'{samples[:3]}, first 3 columns: {sorted(label_cols)[:3]}.  '
                f'This is almost certainly a naming mismatch.'
            )
        if exact_hits == 0 and fuzzy_hits > 0:
            print(f'  Note: {label_tag} labels matched {fuzzy_hits}/{len(samples)} '
                  f'samples via fuzzy (substring) matching.')
    
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
    manifest = {
        'dataset': dataset,
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
    
    print(f'\nProcessed {len(created_paths)} / {len(samples)} samples for {dataset}')
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
        except (OSError, subprocess.SubprocessError, IndexError):
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
    except (OSError, KeyError, ValueError, zarr.errors.PathNotFoundError) as e:
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
