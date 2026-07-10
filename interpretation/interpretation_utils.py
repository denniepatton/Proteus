"""
interpretation_utils.py
v3.0.0
Author: Alexander Netzley, anetzley@fredhutch.org
Updates: Robert Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

Interpretability utilities for the PyTorch BioDVN model.

Provides:
    - Model & data loading (PyTorch checkpoints, Zarr stores)
    - Kernel visualization (grouped Conv1d signals, Conv2d nucleotides,
      ReductionBlock cross-channel filters)
    - PWM / motif analysis from learned nucleotide convolutional filters
        - Attribution analysis via SHAP / gradient methods
    - Polarity-impact and summary attribution plots
    - Signal–sequence interaction analysis (fragment-end ↔ nucleotide)
    - ReductionBlock activation analysis (cross-channel interaction)
    - Latent-space visualization (UMAP of z_tumor)
        - Attribution summaries across complementary gene subsets
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import zlib

import logomaker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Deferred optional imports — only loaded when the relevant functions are
# called, so the rest of the module works without them installed.
# ---------------------------------------------------------------------------
_captum_available: Optional[bool] = None
_shap_available: Optional[bool] = None
_umap_available: Optional[bool] = None


def _ensure_captum() -> None:
    """Import Captum on first use; raise a clear error if unavailable."""
    global _captum_available
    if _captum_available is True:
        return
    try:
        import captum  # noqa: F401
        _captum_available = True
    except ImportError as exc:
        _captum_available = False
        raise ImportError(
            "Captum is required for attribution analysis.  "
            "Install with:  pip install captum"
        ) from exc


def _ensure_umap() -> None:
    """Import umap-learn on first use; raise a clear error if unavailable."""
    global _umap_available
    if _umap_available is True:
        return
    try:
        import umap  # noqa: F401
        _umap_available = True
    except ImportError as exc:
        _umap_available = False
        raise ImportError(
            "umap-learn is required for latent-space visualization.  "
            "Install with:  pip install umap-learn"
        ) from exc


def _ensure_shap() -> None:
    """Import SHAP on first use; raise a clear error if unavailable."""
    global _shap_available
    if _shap_available is True:
        return
    try:
        import shap  # noqa: F401
        _shap_available = True
    except ImportError as exc:
        _shap_available = False
        raise ImportError(
            "shap is required for SHAP attribution analysis.  "
            "Install with:  pip install shap"
        ) from exc


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

import sys as _sys
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from models.models import (  # noqa: E402
    BioDVN,
    BioDVNConfig,
    channel_mode_for_model,
    get_model_instance,
)
from models.model_utils import mixture_mean, mixture_quantile  # noqa: E402
from proteus_utils.metrics import pearson_r  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#                             CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# All 9 signal channels as stored in Zarr and consumed by the model.
SIGNAL_DISPLAY_NAMES: List[str] = [
    'GC-Corrected Depth',          # 0
    'Fragment End Coverage',        # 1
    'Fragment End Asymmetry',       # 2
    'Phased-Nucleosome Signal',     # 3
    'Fragment Short:Long Ratio',    # 4
    'Fragment Length Entropy',      # 5
    'Fragment Length Diversity',    # 6
    'Peak Calls',                   # 7
    'NaN Mask',                     # 8 — binary bookkeeping channel
]

# The 8 biological signal channels (dropping the NaN mask).
BIOLOGICAL_SIGNAL_NAMES: List[str] = SIGNAL_DISPLAY_NAMES[:8]

NUCLEOTIDE_NAMES: List[str] = ['A', 'C', 'G', 'T']

# Combined signal (bio only) + nucleotide labels for heatmaps.
SIGNAL_AND_SEQ_DISPLAY_NAMES: List[str] = BIOLOGICAL_SIGNAL_NAMES + ['Reference Sequence']

TSS_FEATURE_NAMES: List[str] = [
    'fl-mean', 'fl-stdev', 'fl-skew', 'fl-kurtosis',
    'fl-subnucleosomal-ratio', 'fl-entropy', 'fl-gini-simpson',
    'pn-compaction-score', 'pn-mean-spacing', 'pn-mean-amplitude',
    'mean-region-depth',
]

GB_FEATURE_NAMES: List[str] = TSS_FEATURE_NAMES  # identical 11-feature set

REF_SCALAR_NAMES: List[str] = ['num_exons', 'transcript_length', 'mRNA_length']

# Model input groups in Captum tuple order:
#   0: TSS_signals      (batch, 2000, 9)
#   1: TSS_nucleotides  (batch, 2000, 4)
#   2: TSS_features     (batch, 11)
#   3: GB_features      (batch, 11)
#   4: ref_scalars      (batch, 3)
INPUT_GROUP_NAMES: List[str] = [
    'TSS_signals', 'TSS_nucleotides', 'TSS_features', 'GB_features', 'ref_scalars',
]

# Custom diverging colormap (blue → black → red) for polarity plots.
_blue_rgb = (0x00 / 255, 0x8a / 255, 0xfb / 255)
_red_rgb  = (0xff / 255, 0x00 / 255, 0x51 / 255)
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    'blue_black_red',
    [_blue_rgb, (0, 0, 0), _red_rgb],
    N=100,
)

# Beeswarm diverging colormap.
# Keep the original smooth blue → fuchsia-violet → red ramp. We control the
# effective saturation point through the normalization rather than by
# compressing the midpoint stops, which keeps the colorbar visually continuous.
SHAP_CMAP = LinearSegmentedColormap.from_list(
    'shap_blue_violet_red',
    [
        (0.00, '#1E88E5'),  # blue
        (0.50, '#D500F9'),  # fuchsia-violet midpoint
        (1.00, '#FF0D57'),  # red / hot-pink
    ],
    N=256,
)

TARGET_DYNAMIC_GENE_NAMES: List[str] = [
    'AR',
    'ASCL1',
    'KLK3',
    'NEUROD1',
    'FOLH1',
]


# ═══════════════════════════════════════════════════════════════════════════
#                         MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_model(
    weights_path: str,
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> BioDVN:
    """Load a trained BioDVN model from a ``.pt`` state-dict checkpoint.

    Args:
        weights_path: Path to the ``.pt`` file containing the model
            ``state_dict`` (as saved by ``torch.save(model.state_dict(), …)``).
        config_path: Optional path to ``model_config.json``.  If *None* the
            function looks for ``model_config.json`` in the same directory as
            *weights_path*.  Interpretation requires that config file so the
            saved architecture can be reconstructed exactly.
        device: Target device.  Defaults to CUDA if available, else CPU.

    Returns:
        The model in **eval** mode on *device*.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve config
    if config_path is None:
        candidate = os.path.join(os.path.dirname(weights_path), 'model_config.json')
        if os.path.isfile(candidate):
            config_path = candidate

    if config_path is None:
        raise FileNotFoundError(
            'Interpretation requires model_config.json next to the checkpoint so '
            'the current BioDVN architecture can be reconstructed exactly. '
            f'Missing config for checkpoint: {weights_path}'
        )

    with open(config_path, 'r') as f:
        cfg_raw = json.load(f)
    allowed = set(BioDVNConfig.__dataclass_fields__.keys())
    cfg = {k: v for k, v in cfg_raw.items() if k in allowed}
    model_config = BioDVNConfig(**cfg)

    channel_mode = model_config.channel_mode
    channel_mode_to_model_name = {
        channel_mode_for_model('BioDVN'): 'BioDVN',
        channel_mode_for_model('BioDVN_NoSeq'): 'BioDVN_NoSeq',
        channel_mode_for_model('BioDVN_OnlySeq'): 'BioDVN_OnlySeq',
    }
    model_name = channel_mode_to_model_name.get(channel_mode)
    if model_name is None:
        raise ValueError(
            f'Unsupported channel_mode {channel_mode!r} in {config_path}. '
            'Interpretation currently supports BioDVN all/no_seq/only_seq.'
        )

    ModelClass = get_model_instance(model_name)
    model = ModelClass(model_config).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    # Checkpoints saved by Trainer include a nested dict; plain state-dicts
    # (from Proteus.py full-train) are loaded directly.
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f'Checkpoint/config mismatch for interpretation model load. '
            f'weights={weights_path}, config={config_path}. '
            'This usually means the BioDVN architecture changed or the wrong '
            'checkpoint/config pair was supplied.'
        ) from exc
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#                          DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

# Approximate bytes per gene loaded into RAM:
#   TSS_signals(2000×9) + TSS_nucleotides(2000×4) + TSS_features(11)
#   + GB_features(11) + ref_scalars(3) + TFX_label(1)
#   + tumor_exp(1) + background_exp(1)
# = 26,028 float32 values × 4 bytes ≈ 104 KB/gene
_BYTES_PER_GENE = 26_028 * 4


def load_interpretation_data(
    source: Union[str, List[str]],
    max_genes: int = 200_000,
) -> Dict[str, np.ndarray]:
    """Load data from Zarr stores into a dict of NumPy arrays.

    Accepted *source* formats:
        - A path to a fold-index JSON (list of Zarr sample paths).
        - A direct list of ``*.zarr`` paths.
        - A single ``*.zarr`` path (string ending with ``.zarr``).

    The *max_genes* budget is distributed evenly across stores so that
    every sample contributes proportionally without exceeding memory.
    Within each store, genes are selected by a deterministic random subset
    rather than taking the first rows only.
    With 130 BTS stores and the default of 200,000 genes, each store
    contributes ~1,538 genes (≈20 GB RAM total).

    Returns:
        Dictionary with keys::

            TSS_signals      (N, 2000, 9)   float32
            TSS_nucleotides  (N, 2000, 4)   float32
            TSS_features     (N, 11)        float32
            GB_features      (N, 11)        float32
            ref_scalars      (N, 3)         float32
            TFX_label        (N,)           float32
            tumor_exp        (N,)           float32
            background_exp   (N,)           float32
            gene_names       list[str]       length N
            sample_names     list[str]       length N
            cohort_names     list[str]       length N

        where *N* is total genes across all loaded samples.
    """
    # Resolve source into list of Zarr paths
    if isinstance(source, str):
        if source.endswith('.zarr'):
            zarr_paths = [source]
        else:
            with open(source, 'r') as f:
                zarr_paths = json.load(f)
            if isinstance(zarr_paths[0], (list, tuple)):
                zarr_paths = list({p[0] for p in zarr_paths})
    else:
        zarr_paths = list(source)

    n_stores = len(zarr_paths)
    genes_per_store = max(1, max_genes // n_stores)
    est_gb = (genes_per_store * n_stores * _BYTES_PER_GENE) / 1e9
    print(f'  Gene budget: {max_genes:,} total → {genes_per_store:,}/store '
          f'× {n_stores} stores (est. {est_gb:.1f} GB RAM)')

    all_arrays: Dict[str, list] = {
        'TSS_signals': [], 'TSS_nucleotides': [], 'TSS_features': [],
        'GB_features': [], 'ref_scalars': [], 'TFX_label': [],
        'tumor_exp': [], 'background_exp': [],
    }
    all_gene_names: List[str] = []
    all_sample_names: List[str] = []
    all_cohort_names: List[str] = []
    total_loaded = 0

    for idx, zpath in enumerate(zarr_paths, 1):
        if n_stores > 1 and idx % 20 == 0:
            print(f'  Loading zarr store {idx}/{n_stores} '
                  f'({total_loaded:,} genes so far)...')
        store = zarr.open(zpath, mode='r')
        flat = store['flat'] if 'flat' in store else store

        store_gene_count = int(flat['TSS_signals'].shape[0])
        n_genes = min(store_gene_count, genes_per_store)
        store_seed = zlib.crc32(str(zpath).encode('utf-8')) & 0xFFFFFFFF
        store_rng = np.random.default_rng(store_seed)
        gene_indices = np.sort(
            store_rng.choice(store_gene_count, size=n_genes, replace=False)
        )

        all_arrays['TSS_signals'].append(flat['TSS_signals'][gene_indices].astype(np.float32))
        all_arrays['TSS_nucleotides'].append(flat['TSS_nucleotides'][gene_indices].astype(np.float32))
        tss_feat = flat['TSS_features'][gene_indices].astype(np.float32)
        if tss_feat.ndim != 2 or int(tss_feat.shape[1]) != 11:
            raise ValueError(
                f"Expected TSS_features shape (n_genes, 11) in {zpath}, got {tuple(tss_feat.shape)}"
            )
        all_arrays['TSS_features'].append(tss_feat)
        all_arrays['GB_features'].append(flat['GB_features'][gene_indices].astype(np.float32))
        all_arrays['ref_scalars'].append(flat['ref_scalars'][gene_indices].astype(np.float32))
        all_arrays['TFX_label'].append(flat['TFX_label'][gene_indices].astype(np.float32))

        if 'tumor_exp' in flat:
            all_arrays['tumor_exp'].append(flat['tumor_exp'][gene_indices].astype(np.float32))
        else:
            all_arrays['tumor_exp'].append(np.full(n_genes, np.nan, dtype=np.float32))
        if 'background_exp' in flat:
            all_arrays['background_exp'].append(flat['background_exp'][gene_indices].astype(np.float32))
        else:
            all_arrays['background_exp'].append(np.full(n_genes, np.nan, dtype=np.float32))

        # Gene names
        if 'gene_names' in store:
            gene_names_arr = store['gene_names'][gene_indices]
            names = [str(g) for g in gene_names_arr]
        else:
            names = [f'gene_{i}' for i in range(n_genes)]
        all_gene_names.extend(names)

        # Sample name
        sample_name = store.attrs.get('sample_name', Path(zpath).stem.replace('.zarr', ''))
        zarr_path = Path(zpath)
        cohort_name = zarr_path.parent.parent.name if zarr_path.parent.name == 'zarr_files' else 'Unknown'
        all_sample_names.extend([str(sample_name)] * n_genes)
        all_cohort_names.extend([str(cohort_name)] * n_genes)
        total_loaded += n_genes

    data = {k: np.concatenate(v, axis=0) for k, v in all_arrays.items()}
    data['gene_names'] = all_gene_names
    data['sample_names'] = all_sample_names
    data['cohort_names'] = all_cohort_names
    actual_gb = sum(v.nbytes for v in data.values() if isinstance(v, np.ndarray)) / 1e9
    print(f'  Loaded {total_loaded:,} genes from {n_stores} stores ({actual_gb:.1f} GB)')
    return data


def subsample(
    data: Dict[str, np.ndarray],
    sample_size: int = 20_000,
    seed: int = 200,
    restriction: str = 'all_genes',
    min_depth: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Return a random subsample of *data*.

    Args:
        data: Dict from :func:`load_interpretation_data`.
        sample_size: Number of genes to sample.
        seed: Random seed for reproducibility.
        restriction: One of ``'all_genes'``, ``'active_genes'``,
            ``'inactive_genes'``.  Active/inactive split at
            ``log2(CPM+1) >= 2``.
        min_depth: Minimum ``mean-region-depth`` (unlogged).  Genes below
            this threshold are excluded before sampling.  The depth feature
            is at index 10 of ``TSS_features`` and stored as ``log1p(depth)``.

    Returns:
        Subsampled dict with the same keys as *data*.
    """
    rng = np.random.default_rng(seed)

    # mean-region-depth is the last (index 10) TSS feature, stored as log1p
    depth_col = data['TSS_features'][:, 10]
    log_min = np.log1p(min_depth) if min_depth > 0 else -np.inf

    tumor_exp = data['tumor_exp']

    if restriction == 'active_genes':
        valid = np.where((depth_col > log_min) & (tumor_exp >= 2.0))[0]
    elif restriction == 'inactive_genes':
        valid = np.where((depth_col > log_min) & (tumor_exp < 2.0))[0]
    else:
        valid = np.where(depth_col > log_min)[0]

    if sample_size > len(valid):
        print(f'Warning: sample_size ({sample_size}) > valid genes ({len(valid)}). '
              f'Using all {len(valid)} valid genes.')
        sample_size = len(valid)

    indices = rng.choice(valid, sample_size, replace=False)

    out: Dict[str, np.ndarray] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = v[indices]
        elif isinstance(v, list):
            out[k] = [v[i] for i in indices]
        else:
            out[k] = v
    return out


def subsample_disjoint_pair(
    data: Dict[str, np.ndarray],
    background_size: int,
    test_size: int,
    *,
    seed: int = 200,
    restriction: str = 'all_genes',
    min_depth: float = 10.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return disjoint background/test subsamples from a shared valid pool."""
    rng = np.random.default_rng(seed)

    depth_col = data['TSS_features'][:, 10]
    log_min = np.log1p(min_depth) if min_depth > 0 else -np.inf
    tumor_exp = data['tumor_exp']

    if restriction == 'active_genes':
        valid = np.where((depth_col > log_min) & (tumor_exp >= 2.0))[0]
    elif restriction == 'inactive_genes':
        valid = np.where((depth_col > log_min) & (tumor_exp < 2.0))[0]
    else:
        valid = np.where(depth_col > log_min)[0]

    if background_size + test_size > len(valid):
        print(
            f'Warning: requested disjoint background/test sizes '
            f'({background_size}+{test_size}) exceed valid genes ({len(valid)}). '
            'Capping to the available pool.'
        )

    n_background = min(int(background_size), len(valid))
    remaining = len(valid) - n_background
    n_test = min(int(test_size), remaining)

    permuted = rng.permutation(valid)
    background_indices = np.sort(permuted[:n_background])
    test_indices = np.sort(permuted[n_background:n_background + n_test])
    return _slice_data_dict(data, background_indices), _slice_data_dict(data, test_indices)


def save_npz(data: Dict[str, np.ndarray], filepath: str) -> None:
    """Persist interpretation data to a compressed ``.npz`` file."""
    import time as _time
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    for k in ('gene_names', 'sample_names', 'cohort_names'):
        if k in data and isinstance(data[k], list):
            arrays[k] = np.array(data[k], dtype=object)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    n_genes = next((v.shape[0] for v in arrays.values() if hasattr(v, 'shape') and v.ndim >= 1), 0)
    print(f'  Saving compressed cache ({n_genes:,} genes) to {filepath} ...')
    t0 = _time.time()
    np.savez_compressed(filepath, **arrays)
    fsize = os.path.getsize(filepath) / 1e9
    print(f'  Cache saved ({fsize:.1f} GB, {_time.time() - t0:.0f}s)')


def load_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Load a ``.npz`` file saved by :func:`save_npz`."""
    import time as _time
    fsize = os.path.getsize(filepath) / 1e9
    print(f'  Reading cache ({fsize:.1f} GB) from {filepath} ...')
    t0 = _time.time()
    npz = np.load(filepath, allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for key in npz:
        arr = npz[key]
        if arr.dtype == object:
            out[key] = list(arr)
        else:
            out[key] = arr
    print(f'  Cache loaded ({_time.time() - t0:.0f}s)')
    return out


def get_data(
    source: Union[str, List[str]],
    max_genes: int = 200_000,
) -> Dict[str, np.ndarray]:
    """Load interpretation data from Zarr stores.

    Args:
        source: Zarr path(s) or a JSON index file.
        max_genes: Maximum total genes to load across all stores.
            Distributed evenly so every sample contributes.
    """
    if isinstance(source, list):
        print(f'Loading data from {len(source)} zarr stores...')
    else:
        print(f'Loading data from source: {source}')
    return load_interpretation_data(source, max_genes=max_genes)



# ═══════════════════════════════════════════════════════════════════════════
#                     DATA ↔ TENSOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _data_to_tensors(
    data: Dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Convert a numpy data dict to the 5-tensor tuple used by attribution engines.

    Returns:
        ``(TSS_signals, TSS_nucleotides, TSS_features, GB_features, ref_scalars)``
    """
    return (
        torch.from_numpy(data['TSS_signals']).to(device),
        torch.from_numpy(data['TSS_nucleotides']).to(device),
        torch.from_numpy(data['TSS_features']).to(device),
        torch.from_numpy(data['GB_features']).to(device),
        torch.from_numpy(data['ref_scalars']).to(device),
    )


def _tensors_to_feature_dict(tensors: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
    """Build the feature dict that BioDVN.forward expects from an attribution tuple."""
    return {
        'TSS_signals': tensors[0],
        'TSS_nucleotides': tensors[1],
        'TSS_features': tensors[2],
        'GB_features': tensors[3],
        'ref_scalars': tensors[4],
    }


# ═══════════════════════════════════════════════════════════════════════════
#         ATTRIBUTION MODEL WRAPPER  (dict inputs → scalar output)
# ═══════════════════════════════════════════════════════════════════════════

class _TumorMeanWrapper(nn.Module):
    """Thin wrapper around :class:`BioDVN` for attribution workflows.

    Attribution libraries used here expect models that accept positional
    tensor arguments and return a single scalar output per sample.
    This wrapper:

    1. Accepts the 5 model input tensors as positional args.
    2. Packs them into the feature dict that ``BioDVN.forward`` expects.
    3. Returns ``tumor_exp`` (the mixture-mean point estimate).

    Works with standard BioDVN, BioDVN_NoSeq, and BioDVN_OnlySeq.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        tss_signals: torch.Tensor,
        tss_nucleotides: torch.Tensor,
        tss_features: torch.Tensor,
        gb_features: torch.Tensor,
        ref_scalars: torch.Tensor,
    ) -> torch.Tensor:
        features = {
            'TSS_signals': tss_signals,
            'TSS_nucleotides': tss_nucleotides,
            'TSS_features': tss_features,
            'GB_features': gb_features,
            'ref_scalars': ref_scalars,
        }
        predictions, _ = self.model(features, return_internal_loss=False)
        return predictions['tumor_exp'].squeeze(-1)  # (batch,)


class _TumorMeanColumnWrapper(_TumorMeanWrapper):
    """Return tumor predictions with an explicit singleton output dimension."""

    def forward(
        self,
        tss_signals: torch.Tensor,
        tss_nucleotides: torch.Tensor,
        tss_features: torch.Tensor,
        gb_features: torch.Tensor,
        ref_scalars: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(
            tss_signals,
            tss_nucleotides,
            tss_features,
            gb_features,
            ref_scalars,
        ).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#                      PLOT HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _save_or_show(fig: plt.Figure, folder_path: Optional[str], filename: str) -> None:
    """Save figure to *folder_path/filename* or show interactively."""
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
        # Default to PDF for publication-quality vector output
        filename = os.path.splitext(filename)[0] + '.pdf'
        fig.savefig(os.path.join(folder_path, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#                    KERNEL VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def _get_conv_weights(module: nn.Module) -> np.ndarray:
    """Extract weight tensor from a Conv1d or Conv2d as a numpy array."""
    return module.weight.detach().cpu().numpy()


def plot_signal_conv1d_filters(model: BioDVN, folder_path: Optional[str] = None) -> None:
    """Plot the per-channel Conv1d first-layer signal filters.

    The ``SignalProcessingBlock`` uses a single grouped Conv1d where groups
    equals the number of signal channels.  Each group's filters are
    contiguous in the weight tensor: ``w[ch*sf : (ch+1)*sf, 0, :]``.
    """
    sp = model.signal_processor
    if sp.conv1d_signals is None:
        print('  [skip] Signal Conv1d filters unavailable for this channel mode')
        return
    n_channels = sp.n_signal_channels
    signal_filters = sp.signal_filters
    ksize = sp.signal_ksize

    # Extract weight from the Conv1d inside the Sequential
    conv1d = sp.conv1d_signals[0]
    w_all = _get_conv_weights(conv1d)  # (n_channels * signal_filters, 1, ksize)

    # Slice per channel (grouped conv layout)
    channel_weights = []
    for ch in range(n_channels):
        w_ch = w_all[ch * signal_filters : (ch + 1) * signal_filters, 0, :]
        channel_weights.append(w_ch)  # (signal_filters, ksize)

    vmin = w_all.min()
    vmax = w_all.max()

    fig, axs = plt.subplots(n_channels, signal_filters,
                            figsize=(2 * signal_filters, 2 * n_channels),
                            constrained_layout=True)
    if signal_filters == 1:
        axs = axs[:, np.newaxis]
    if n_channels == 1:
        axs = axs[np.newaxis, :]

    for ch_idx in range(n_channels):
        for f_idx in range(signal_filters):
            ax = axs[ch_idx, f_idx]
            kernel = channel_weights[ch_idx][f_idx]  # (ksize,)
            ax.bar(range(ksize), kernel, color='steelblue')
            ax.set_ylim(vmin * 1.1, vmax * 1.1)
            if f_idx == 0:
                label = SIGNAL_DISPLAY_NAMES[ch_idx] if ch_idx < len(SIGNAL_DISPLAY_NAMES) else f'Ch{ch_idx}'
                ax.set_ylabel(label, fontsize=7, rotation=45, ha='right')
            else:
                ax.set_yticks([])
            if ch_idx == 0:
                ax.set_title(f'F{f_idx + 1}', fontsize=8)
            ax.set_xticks([])

    fig.suptitle(
        f'Signal Conv1d Filters ({signal_filters} per channel, kernel={ksize})',
        fontsize=14,
    )
    _save_or_show(fig, folder_path, 'signal_conv1d_filters.png')


def plot_nucleotide_conv2d_filters(model: BioDVN, folder_path: Optional[str] = None) -> None:
    """Plot the Conv2d nucleotide filters from the SignalProcessingBlock.

    The nucleotide convolution operates on a ``(2000, 4)`` one-hot input with
    shape ``(1, 4, ksize)`` kernels, producing *signal_filters* output channels.
    """
    sp = model.signal_processor
    if sp.conv2d_nt is None:
        print('  [skip] Nucleotide Conv2d filters unavailable for this channel mode')
        return
    # conv2d_nt is a Sequential; the Conv2d layer is the first element
    conv2d = sp.conv2d_nt[0]
    w = _get_conv_weights(conv2d)  # (out_ch, in_ch=1, H=4, W=ksize)
    num_filters = w.shape[0]
    ksize = w.shape[3]

    cols = min(8, num_filters)
    rows = int(np.ceil(num_filters / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows),
                            constrained_layout=True)
    if rows == 1 and cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array(axs).reshape(1, -1)

    im = None
    for idx in range(num_filters):
        r, c = divmod(idx, cols)
        ax = axs[r, c]
        kernel = w[idx, 0, :, :]  # (4, ksize)
        im = ax.imshow(kernel, cmap='viridis', aspect='auto')
        ax.set_yticks(range(4))
        ax.set_yticklabels(NUCLEOTIDE_NAMES, fontsize=8)
        ax.set_title(f'Filter {idx + 1}', fontsize=9)
        ax.set_xticks([])

    # Hide empty subplots
    for idx in range(num_filters, rows * cols):
        r, c = divmod(idx, cols)
        axs[r, c].axis('off')

    if im is not None:
        fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.03, label='Weight')
    fig.suptitle(
        f'Nucleotide Conv2d Filters ({num_filters} filters, kernel=4×{ksize})',
        fontsize=14,
    )
    _save_or_show(fig, folder_path, 'nucleotide_conv2d_filters.png')


def plot_reduction_filters(model: BioDVN, folder_path: Optional[str] = None) -> None:
    """Plot Conv2d cross-channel interaction kernels from the ReductionBlock.

    The single ReductionBlock has ``n_reductions`` Conv2d layers stored in
    ``model.reduction.convs`` (a ModuleList).  Each Conv2d weight has shape
    ``(out_ch, 1, interaction_ksize, in_channels)`` — visualized as a
    ``(in_channels × interaction_ksize)`` heatmap per output channel.
    """
    reduction = model.reduction

    for iteration_idx, conv in enumerate(reduction.convs):
        w = _get_conv_weights(conv)  # (out_ch, 1, ksize_h, ksize_w)
        out_ch = w.shape[0]
        ksize_h = w.shape[2]  # interaction_ksize
        ksize_w = w.shape[3]  # in_channels

        cols = min(8, out_ch)
        rows = int(np.ceil(out_ch / cols))

        fig, axs = plt.subplots(rows, cols,
                                figsize=(2 * cols, max(3, ksize_w * 0.05) * rows),
                                constrained_layout=True)
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1:
            axs = axs[np.newaxis, :]
        elif cols == 1:
            axs = axs[:, np.newaxis]

        im = None
        for idx in range(out_ch):
            r, c = divmod(idx, cols)
            ax = axs[r, c]
            # (1, ksize_h, ksize_w) → squeeze to (ksize_h, ksize_w)
            kernel = w[idx, 0, :, :]
            im = ax.imshow(kernel, cmap='viridis', aspect='auto', interpolation='none')
            ax.set_title(f'F{idx + 1}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(out_ch, rows * cols):
            r, c = divmod(idx, cols)
            axs[r, c].axis('off')

        fig.suptitle(
            f'ReductionBlock iteration {iteration_idx} Conv2d filters '
            f'({out_ch} filters, kernel=({ksize_h},{ksize_w}), '
            f'{ksize_w} input channels)',
            fontsize=11,
        )
        _save_or_show(fig, folder_path, f'reduction_iter{iteration_idx}_conv2d_filters.png')




# ═══════════════════════════════════════════════════════════════════════════
#                    PWM / MOTIF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_PWM(
    model: BioDVN,
    data: Dict[str, np.ndarray],
    n_samples: int = 10,
    save_filepath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute position-weight matrices from the learned nucleotide Conv2d filters.

    For each Conv2d filter, nucleotide windows are scored using a vectorized
    convolution; windows with positive activations contribute to the PWM.

    Args:
        model: Trained BioDVN model.
        data: Dict from :func:`load_interpretation_data`.
        n_samples: Number of data samples to use for PWM computation.
        save_filepath: If provided, save a heatmap of the raw PWM values.

    Returns:
        ``(pwm_filters, background_freq)``  where *pwm_filters* has shape
        ``(ksize, 4, num_filters)`` and *background_freq* is ``(4,)``.
    """
    sp = model.signal_processor
    if sp.conv2d_nt is None:
        raise ValueError(
            'PWM computation requires the nucleotide Conv2d branch, but the '
            f'loaded model channel_mode is {model.config.channel_mode!r}.'
        )
    # conv2d_nt is a Sequential; Conv2d is the first element
    conv2d = sp.conv2d_nt[0]
    w = _get_conv_weights(conv2d)  # (num_filters, 1, 4, ksize)
    num_filters = w.shape[0]
    ksize = w.shape[3]

    nucleotides_all = data['TSS_nucleotides']  # (N, 2000, 4)
    n_use = min(n_samples, nucleotides_all.shape[0])

    # Background nucleotide frequencies
    bg_freq = nucleotides_all[:n_use].mean(axis=(0, 1))  # (4,)

    # Vectorized PWM computation using PyTorch conv2d
    device = next(model.parameters()).device
    w_tensor = torch.from_numpy(w).to(device)  # (num_filters, 1, 4, ksize)

    pwm_accum = np.zeros((ksize, 4, num_filters))
    total_counts = np.zeros(num_filters)

    for s_idx in range(n_use):
        nuc = nucleotides_all[s_idx]  # (2000, 4)
        # Reshape to match Conv2d input: (1, 1, 4, 2000)
        nuc_tensor = torch.from_numpy(nuc.T[np.newaxis, np.newaxis]).float().to(device)
        with torch.no_grad():
            # (1, num_filters, 1, 2000-ksize+1) — raw conv activations
            activations = F.conv2d(nuc_tensor, w_tensor)
            act_np = activations.squeeze(2).squeeze(0).cpu().numpy()  # (num_filters, L)

        L = act_np.shape[1]
        for f_idx in range(num_filters):
            pos_mask = act_np[f_idx] > 0
            n_positive = int(pos_mask.sum())
            if n_positive == 0:
                continue
            # Gather positive windows: shape (n_positive, ksize, 4)
            pos_indices = np.where(pos_mask)[0]
            windows = np.zeros((n_positive, ksize, 4), dtype=np.float32)
            for k, p in enumerate(pos_indices):
                windows[k] = nuc[p:p + ksize, :]
            pwm_accum[:, :, f_idx] += windows.sum(axis=0)
            total_counts[f_idx] += n_positive

    # Normalize
    for f_idx in range(num_filters):
        if total_counts[f_idx] > 0:
            pwm_accum[:, :, f_idx] /= total_counts[f_idx]

    if save_filepath is not None:
        cols = min(8, num_filters)
        rows = int(np.ceil(num_filters / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows),
                                constrained_layout=True)
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1:
            axs = np.array(axs).reshape(1, -1)
        for idx in range(num_filters):
            r, c = divmod(idx, cols)
            ax = axs[r, c]
            ax.imshow(pwm_accum[:, :, idx].T, cmap='viridis', aspect='auto')
            ax.set_yticks(range(4))
            ax.set_yticklabels(NUCLEOTIDE_NAMES)
            ax.set_title(f'Filter {idx + 1}', fontsize=9)
        for idx in range(num_filters, rows * cols):
            r, c = divmod(idx, cols)
            axs[r, c].axis('off')
        fig.suptitle(f'Nucleotide PWM ({num_filters} filters)', fontsize=14)
        _save_or_show(fig, save_filepath, 'nucleotide_motifs_PWM.png')

    return pwm_accum, bg_freq


def export_PWM_to_MEME(
    pwm_filters: np.ndarray,
    background_freq: np.ndarray,
    save_filepath: Optional[str] = None,
) -> str:
    """Export PWMs to a MEME-format motif file.

    Args:
        pwm_filters: ``(ksize, 4, num_filters)`` from :func:`compute_PWM`.
        background_freq: ``(4,)`` array of [A, C, G, T] frequencies.
        save_filepath: Directory to write ``pwm_motif.meme`` into.

    Returns:
        The MEME-formatted string.
    """
    meme = "MEME version 4\n\n"
    meme += "ALPHABET= ACGT\n\n"
    meme += "strands: + -\n\n"
    meme += "Background letter frequencies\n"
    for base, freq in zip(NUCLEOTIDE_NAMES, background_freq):
        meme += f"{base} {freq:.4f} "
    meme += "\n\n"

    for i in range(pwm_filters.shape[2]):
        pwm = pwm_filters[:, :, i]  # (ksize, 4)
        meme += f"MOTIF kernel_{i + 1}\n"
        meme += f"letter-probability matrix: alength= 4 w= {pwm.shape[0]}\n"
        for row in range(pwm.shape[0]):
            meme += " ".join(f"{pwm[row, col]:.6f}" for col in range(4)) + "\n"
        meme += "\n"

    if save_filepath is not None:
        out_path = os.path.join(save_filepath, 'pwm_motif.meme')
        os.makedirs(save_filepath, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(meme)

    return meme


def plot_nucleotide_motifs(
    pwm_filters: np.ndarray,
    save_filepath: Optional[str] = None,
) -> None:
    """Plot nucleotide motifs as sequence logos (information content & probability).

    Args:
        pwm_filters: ``(ksize, 4, num_filters)`` from :func:`compute_PWM`.
        save_filepath: Directory to save figures to.
    """
    num_filters = pwm_filters.shape[2]
    cols = 2
    rows = int(np.ceil(num_filters / cols))

    for mode, transform, ylabel, fname in [
        ('information', 'information', 'bits', 'nucleotide_motifs_PWM_IC.png'),
        ('probability', None, 'probability', 'nucleotide_motifs_PWM_Prob.png'),
    ]:
        fig, axs = plt.subplots(rows, cols, figsize=(5, 2 * rows),
                                constrained_layout=True)
        if rows == 1:
            axs = np.array(axs).reshape(1, -1)

        for idx in range(num_filters):
            r, c = divmod(idx, cols)
            ax = axs[r, c]
            df = pd.DataFrame(pwm_filters[:, :, idx], columns=NUCLEOTIDE_NAMES)
            if transform == 'information':
                df = logomaker.transform_matrix(df, from_type='probability',
                                                to_type='information')
            logomaker.Logo(df, ax=ax)
            ax.set_ylim(0, 2 if mode == 'information' else 1)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            else:
                ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(f'kernel {idx + 1}', fontsize=9)

        for idx in range(num_filters, rows * cols):
            r, c = divmod(idx, cols)
            axs[r, c].axis('off')

        fig.suptitle(f'PWMs in {mode.capitalize()}', fontsize=14)
        _save_or_show(fig, save_filepath, fname)


def plot_motifs(
    model: BioDVN,
    data: Dict[str, np.ndarray],
    save_filepath: Optional[str] = None,
) -> None:
    """Wrapper: compute PWMs, export MEME, and plot motif logos."""
    if model.signal_processor.conv2d_nt is None:
        print('  [skip] Motif analysis unavailable for this channel mode')
        return
    pwm_filters, bg_freq = compute_PWM(model, data, save_filepath=save_filepath)
    export_PWM_to_MEME(pwm_filters, bg_freq, save_filepath)
    plot_nucleotide_motifs(pwm_filters, save_filepath)


def kernel_visualization(
    model: BioDVN,
    data: Dict[str, np.ndarray],
    save_filepath: Optional[str] = None,
) -> None:
    """Run all kernel visualization analyses.

    Generates:
        - Signal Conv1d filter bar plots
        - Nucleotide Conv2d filter heatmaps
        - Reduction block filter heatmaps
        - PWM motif logos + MEME export
    """
    folder_path = None
    if save_filepath is not None:
        folder_path = os.path.join(save_filepath, 'kernel_visualizations')
        os.makedirs(folder_path, exist_ok=True)

    print('  Signal Conv1d filters...')
    plot_signal_conv1d_filters(model, folder_path)
    print('  Nucleotide Conv2d filters...')
    plot_nucleotide_conv2d_filters(model, folder_path)
    print('  Reduction block filters...')
    plot_reduction_filters(model, folder_path)
    print('  PWM motif logos & MEME export...')
    plot_motifs(model, data, folder_path)
    print('  Kernel visualization complete.')



# ═══════════════════════════════════════════════════════════════════════════
#                  ATTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _coerce_attribution_batch(batch_attr: Any) -> List[np.ndarray]:
    """Normalise attribution outputs from SHAP/Captum into a list of arrays."""
    if isinstance(batch_attr, torch.Tensor):
        return [batch_attr.detach().cpu().numpy()]
    if isinstance(batch_attr, np.ndarray):
        return [np.asarray(batch_attr)]
    if isinstance(batch_attr, (list, tuple)):
        if batch_attr and isinstance(batch_attr[0], (list, tuple)):
            if len(batch_attr) != 1:
                raise ValueError(
                    'Expected a single-output attribution result, got '
                    f'{len(batch_attr)} outputs.'
                )
            batch_attr = batch_attr[0]
        coerced: List[np.ndarray] = []
        for item in batch_attr:
            if isinstance(item, torch.Tensor):
                coerced.append(item.detach().cpu().numpy())
            else:
                coerced.append(np.asarray(item))
        return coerced
    raise TypeError(f'Unsupported attribution batch type: {type(batch_attr)!r}')


def _canonicalize_attribution_method(method: str) -> str:
    """Map user-facing method names onto internal attribution backends."""
    if method in ('shap', 'deep_shap'):
        return 'shap'
    return method

def _run_attribution(
    model: BioDVN,
    test_tensors: Tuple[torch.Tensor, ...],
    baseline_tensors: Tuple[torch.Tensor, ...],
    method: str = 'shap',
    n_steps: int = 50,
    batch_size: int = 32,
    seed: int = 200,
    wrapper_cls: type = _TumorMeanWrapper,
) -> List[np.ndarray]:
    """Run attribution and return per-input-group numpy arrays.

    Args:
        model: Trained BioDVN in eval mode.
        test_tensors: 5-tuple of test input tensors.
        baseline_tensors: 5-tuple of baseline (background) tensors.
        method: ``'shap'``/``'deep_shap'``, ``'integrated_gradients'``, or
            ``'gradient_shap'``.
        n_steps: Number of interpolation steps (IntegratedGradients) or
            samples (SHAP expected gradients / GradientShap).
        batch_size: Number of test genes per outer loop iteration.
            Each batch is internally expanded by *n_steps* interpolation
            points, so VRAM usage scales as ``batch_size × n_steps``.
        seed: Random seed for stochastic attribution engines.
        wrapper_cls: Wrapper class to use (default: _TumorMeanWrapper).
            Must accept a model and return a scalar per sample.

    Returns:
        List of 5 numpy arrays (one per input group) with the same shape as
        the corresponding test tensors.
    """
    method = _canonicalize_attribution_method(method)

    device = next(model.parameters()).device
    if method == 'shap':
        _ensure_shap()
        import shap
        wrapper = _TumorMeanColumnWrapper(model).to(device)
    else:
        _ensure_captum()
        from captum.attr import IntegratedGradients, GradientShap
        wrapper = wrapper_cls(model).to(device)
    wrapper.eval()

    if method == 'shap':
        background_pool_size = min(128, baseline_tensors[0].shape[0])
        baseline_summary = [
            tensor[:background_pool_size].to(device)
            for tensor in baseline_tensors
        ]
        attr_method = shap.GradientExplainer(wrapper, baseline_summary)
        print(
            f'  SHAP expected-gradients background pool capped at {background_pool_size:,} genes '
            'for GPU stability'
        )
    elif method == 'integrated_gradients':
        attr_method = IntegratedGradients(wrapper)
        baseline_summary = tuple(
            tensor.mean(dim=0, keepdim=True).to(device)
            for tensor in baseline_tensors
        )
    elif method == 'gradient_shap':
        attr_method = GradientShap(wrapper)
        baseline_pool_size = min(256, baseline_tensors[0].shape[0])
        baseline_summary = tuple(
            tensor[:baseline_pool_size].to(device)
            for tensor in baseline_tensors
        )
        print(
            f'  GradientShap baseline pool capped at {baseline_pool_size:,} genes '
            'for GPU stability'
        )
    else:
        raise ValueError(f"Unknown attribution method: {method!r}")

    n_test = test_tensors[0].shape[0]
    attr_accum: Optional[List[np.ndarray]] = None

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        batch_inputs = tuple(t[start:end].to(device) for t in test_tensors)

        if method == 'shap':
            batch_attr = attr_method.shap_values(
                list(batch_inputs),
                nsamples=n_steps,
                rseed=seed + start,
            )
        elif method == 'integrated_gradients':
            baselines = tuple(
                baseline.expand_as(batch_input)
                for baseline, batch_input in zip(baseline_summary, batch_inputs)
            )
            batch_attr = attr_method.attribute(
                batch_inputs, baselines=baselines, n_steps=n_steps,
                internal_batch_size=batch_size,
            )
        else:  # gradient_shap
            batch_attr = attr_method.attribute(
                batch_inputs, baselines=baseline_summary, n_samples=n_steps,
            )

        batch_np = _coerce_attribution_batch(batch_attr)
        expected_shapes = [tuple(batch_input.shape) for batch_input in batch_inputs]
        if len(batch_np) != len(expected_shapes):
            raise ValueError(
                'Attribution engine returned the wrong number of input groups: '
                f'{len(batch_np)} vs {len(expected_shapes)}'
            )
        normalised_batch: List[np.ndarray] = []
        for arr, expected_shape in zip(batch_np, expected_shapes):
            if arr.shape == expected_shape + (1,):
                arr = np.squeeze(arr, axis=-1)
            if tuple(arr.shape) != expected_shape:
                raise ValueError(
                    'Attribution shape mismatch for input group: '
                    f'got {tuple(arr.shape)}, expected {expected_shape}'
                )
            normalised_batch.append(arr)
        batch_np = normalised_batch

        if attr_accum is None:
            attr_accum = batch_np
        else:
            attr_accum = [
                np.concatenate([prev, cur], axis=0)
                for prev, cur in zip(attr_accum, batch_np)
            ]

        print(f'\r  Attribution: {end:,}/{n_test:,} genes ({100*end/n_test:.0f}%)', end='', flush=True)

    print()  # newline after progress
    if attr_accum is None:
        raise RuntimeError('No test samples processed.')
    return attr_accum


def _observed_sequence_contribution(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return the signed nucleotide attribution for the observed base at each position."""
    return (attributions[1] * data['TSS_nucleotides']).sum(axis=-1)


def _signed_signal_sequence_interaction(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return signed signal × observed-sequence interaction for all bio channels."""
    sig_attr = attributions[0][:, :, :8]
    observed_seq_attr = _observed_sequence_contribution(attributions, data)
    return sig_attr * observed_seq_attr[:, :, np.newaxis]


def _bp_window_to_indices(start_bp: int, end_bp: int, seq_len: int) -> Tuple[int, int]:
    """Convert a TSS-relative bp window to half-open array indices."""
    start_idx = max(0, min(seq_len, int(start_bp + 1000)))
    end_idx = max(start_idx + 1, min(seq_len, int(end_bp + 1000)))
    return start_idx, end_idx


def _build_weighted_pwm(
    nuc_data: np.ndarray,
    weights: np.ndarray,
    center_pos: int,
    window_size: int,
) -> Optional[np.ndarray]:
    """Build a weighted PWM around a center position."""
    half_window = window_size // 2
    lo = max(0, center_pos - half_window)
    hi = min(nuc_data.shape[1], center_pos + half_window + 1)
    positive_weights = np.clip(weights.astype(np.float64), 0.0, None)
    weight_sum = positive_weights.sum()
    if weight_sum <= 1e-12:
        return None
    norm_weights = positive_weights / weight_sum
    return np.einsum('n,nwc->wc', norm_weights, nuc_data[:, lo:hi, :])


def _mean_group_attribution_per_gene(attributions: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Return per-gene mean absolute attribution for each input group."""
    return {
        'Signals': np.abs(attributions[0][:, :, :8]).mean(axis=(1, 2)),
        'Sequence': np.abs(attributions[1]).mean(axis=(1, 2)),
        'TSS Scalars': np.abs(attributions[2]).mean(axis=1),
        'GB Scalars': np.abs(attributions[3]).mean(axis=1),
        'Ref Scalars': np.abs(attributions[4]).mean(axis=1),
    }


def _signed_group_contributions(attributions: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Return signed per-gene attribution totals for each input group."""
    return {
        'Signals': attributions[0][:, :, :8].sum(axis=(1, 2)),
        'Sequence': attributions[1].sum(axis=(1, 2)),
        'TSS Scalars': attributions[2].sum(axis=1),
        'GB Scalars': attributions[3].sum(axis=1),
        'Ref Scalars': attributions[4].sum(axis=1),
    }


def _signed_channel_contributions(attributions: List[np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """Return signed per-gene contributions for signal channels and sequence."""
    signal_vals = attributions[0][:, :, :8].sum(axis=1)
    sequence_vals = attributions[1].sum(axis=(1, 2))[:, np.newaxis]
    return (
        BIOLOGICAL_SIGNAL_NAMES + ['Reference Sequence'],
        np.concatenate([signal_vals, sequence_vals], axis=1),
    )


def _signed_scalar_contributions(attributions: List[np.ndarray]) -> Tuple[List[str], np.ndarray]:
    """Return signed per-gene contributions for all scalar features."""
    names = (
        [f'TSS: {name}' for name in TSS_FEATURE_NAMES]
        + [f'GB: {name}' for name in GB_FEATURE_NAMES]
        + [f'Ref: {name}' for name in REF_SCALAR_NAMES]
    )
    values = np.concatenate([attributions[2], attributions[3], attributions[4]], axis=1)
    return names, values


def _weighted_mean(array: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the weighted mean over the leading axis."""
    weights = np.asarray(weights, dtype=np.float64)
    if array.shape[0] != weights.shape[0]:
        raise ValueError(
            'Array and weight lengths must match on axis 0: '
            f'{array.shape[0]} vs {weights.shape[0]}'
        )
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError('weights must sum to a positive value')
    reshape = (weights.shape[0],) + (1,) * (array.ndim - 1)
    return (array * weights.reshape(reshape)).sum(axis=0) / weight_sum


def _expression_state_label(value: float) -> str:
    """Map an expression value to a coarse off/low/mid/high label."""
    if not np.isfinite(value):
        return 'nan'
    if value < 1.0:
        return 'off'
    if value < 3.0:
        return 'low'
    if value < 6.0:
        return 'mid'
    return 'high'


def _top_spaced_positions(profile: np.ndarray, top_k: int, spacing: int) -> List[int]:
    """Select top interaction positions while suppressing nearby duplicates."""
    work = np.asarray(profile, dtype=np.float32).copy()
    positions: List[int] = []
    radius = max(int(spacing), 1)
    for _ in range(top_k):
        pos = int(np.argmax(work))
        if not np.isfinite(work[pos]) or work[pos] <= 0:
            break
        positions.append(pos)
        lo = max(0, pos - radius)
        hi = min(work.shape[0], pos + radius + 1)
        work[lo:hi] = -np.inf
    return positions


def _slice_data_dict(data: Dict[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    """Slice an interpretation data dictionary by row indices."""
    sliced: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            sliced[key] = value[indices].copy()
        elif isinstance(value, list):
            sliced[key] = [value[int(i)] for i in indices]
        else:
            sliced[key] = value
    return sliced


def _predict_tumor_expression(
    model: nn.Module,
    data: Dict[str, Any],
    batch_size: int = 64,
) -> np.ndarray:
    """Run batched forward passes and return tumor mean predictions."""
    device = next(model.parameters()).device
    n_total = int(data['TSS_signals'].shape[0])
    preds: List[np.ndarray] = []
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        features = {
            'TSS_signals': torch.from_numpy(data['TSS_signals'][start:end]).to(device),
            'TSS_nucleotides': torch.from_numpy(data['TSS_nucleotides'][start:end]).to(device),
            'TSS_features': torch.from_numpy(data['TSS_features'][start:end]).to(device),
            'GB_features': torch.from_numpy(data['GB_features'][start:end]).to(device),
            'ref_scalars': torch.from_numpy(data['ref_scalars'][start:end]).to(device),
        }
        with torch.no_grad():
            batch_preds, _ = model(features, return_internal_loss=False)
        preds.append(batch_preds['tumor_exp'].detach().cpu().numpy().squeeze(-1))
    return np.concatenate(preds, axis=0)


# ═══════════════════════════════════════════════════════════════════════════
#                  ATTRIBUTION PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_signal_attributions(
    attributions: List[np.ndarray],
    folder_path: Optional[str] = None,
) -> None:
    """Plot mean absolute attributions across TSS signal channels.

    Creates a (channels × 2000) heatmap showing the spatial pattern of
    attribution for each of the 8 biological signal channels (NaN mask
    excluded), plus a summed nucleotide channel.
    """
    sig_attr = attributions[0]    # (N, 2000, 9)
    nuc_attr = attributions[1]    # (N, 2000, 4)

    # Drop NaN mask channel (index 8)
    sig_attr_bio = np.abs(sig_attr[:, :, :8])     # (N, 2000, 8)

    # Sum nucleotide attributions to a single channel
    nuc_summed = np.abs(nuc_attr).sum(axis=-1, keepdims=True)   # (N, 2000, 1)
    combined = np.concatenate([sig_attr_bio, nuc_summed], axis=-1)  # (N, 2000, 9)

    # Mean across samples, transpose to (9, 2000)
    means = combined.mean(axis=0).T

    labels = SIGNAL_AND_SEQ_DISPLAY_NAMES  # 8 bio channels + 'Reference Sequence'

    fig, axs = plt.subplots(len(labels), 1,
                            figsize=(12, 0.8 * len(labels)),
                            constrained_layout=True)
    for j, (ax, label) in enumerate(zip(axs, labels)):
        ax.imshow(means[j:j + 1], cmap='viridis', aspect='auto',
                  interpolation='nearest')
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=10, ha='right', fontsize=7)
        if j < len(labels) - 1:
            ax.set_xticks([])
        else:
            ticks = np.linspace(0, means.shape[1] - 1, 9)
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [f'{int(x)}' for x in np.linspace(-1000, 1000, 9)])

    fig.suptitle('Mean |Attribution| for TSS Signals + Nucleotides', fontsize=13)
    _save_or_show(fig, folder_path, 'TSS_signal_attributions.png')


def plot_signal_polarity(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
) -> None:
    """Plot polarity-impact scores for the 8 biological signal channels.

    Polarity captures whether higher signal values drive the prediction
    *up* (red) or *down* (blue).  Magnitude is the median absolute
    attribution.  The NaN mask channel is excluded.
    """
    sig_attr = attributions[0][:, :, :8]   # (N, 2000, 8) — bio only
    sig_vals = data['TSS_signals'][:, :, :8]  # (N, 2000, 8) — bio only

    n_channels = sig_attr.shape[2]
    seq_len = sig_attr.shape[1]
    impact_polarity = np.zeros((n_channels, seq_len))

    def _plot_polarity_heatmaps(
        impact_matrix: np.ndarray,
        title_suffix: str,
        file_suffix: str,
    ) -> None:
        fig, axs = plt.subplots(n_channels, 1,
                                figsize=(12, 0.7 * n_channels),
                                constrained_layout=True)
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        for ch in range(n_channels):
            im = axs[ch].imshow(
                impact_matrix[ch:ch + 1], cmap=CUSTOM_CMAP,
                aspect='auto', interpolation='nearest', norm=norm,
            )
            axs[ch].set_yticks([])
            axs[ch].set_ylabel(
                BIOLOGICAL_SIGNAL_NAMES[ch], rotation=0,
                labelpad=10, ha='right', fontsize=7,
            )
            if ch < n_channels - 1:
                axs[ch].set_xticks([])
            else:
                ticks = np.linspace(0, seq_len - 1, 9)
                axs[ch].set_xticks(ticks)
                axs[ch].set_xticklabels(
                    [f'{int(x)}' for x in np.linspace(-1000, 1000, 9)])

        fig.suptitle(
            f'Attribution Polarity-Impact (Signals{title_suffix})',
            fontsize=13,
        )
        fig.colorbar(im, ax=axs.ravel().tolist(), label='Direction & impact')
        _save_or_show(
            fig,
            folder_path,
            f'TSS_signal_polarity_impact{file_suffix}.png',
        )

        subset_indices = [0, 3, 6, 1]  # depth, nucleosome, diversity, coverage
        n_sub = len(subset_indices)
        fig2, axs2 = plt.subplots(n_sub, 1,
                                  figsize=(12, 0.7 * n_sub),
                                  constrained_layout=True)
        for row, ch in enumerate(subset_indices):
            im2 = axs2[row].imshow(
                impact_matrix[ch:ch + 1], cmap=CUSTOM_CMAP,
                aspect='auto', interpolation='nearest', norm=norm,
            )
            axs2[row].set_yticks([])
            axs2[row].set_ylabel(
                BIOLOGICAL_SIGNAL_NAMES[ch], rotation=0,
                labelpad=10, ha='right', fontsize=7,
            )
            if row < n_sub - 1:
                axs2[row].set_xticks([])
            else:
                ticks = np.linspace(0, seq_len - 1, 9)
                axs2[row].set_xticks(ticks)
                axs2[row].set_xticklabels(
                    [f'{int(x)}' for x in np.linspace(-1000, 1000, 9)])

        fig2.suptitle(
            f'Attribution Polarity-Impact (Key Signals{title_suffix})',
            fontsize=13,
        )
        fig2.colorbar(im2, ax=axs2.ravel().tolist(), label='Direction & impact')
        _save_or_show(
            fig2,
            folder_path,
            f'TSS_signal_polarity_impact_subset{file_suffix}.png',
        )

    for ch in range(n_channels):
        attr_ch = sig_attr[:, :, ch]   # (N, 2000)
        vals_ch = sig_vals[:, :, ch]   # (N, 2000)

        above = vals_ch > 1.0
        pos_above = (attr_ch > 0) & above
        neg_below = (attr_ch < 0) & ~above
        pos_below = (attr_ch > 0) & ~above
        neg_above = (attr_ch < 0) & above

        polarity = (
            pos_above.sum(axis=0) + neg_below.sum(axis=0)
            - pos_below.sum(axis=0) - neg_above.sum(axis=0)
        ) / attr_ch.shape[0]

        impact = polarity * np.median(np.abs(attr_ch), axis=0)

        # Normalise to [-1, +1]
        pos_vals = impact[impact > 0]
        neg_vals = impact[impact < 0]
        if pos_vals.size > 0:
            impact[impact > 0] /= np.max(pos_vals)
        if neg_vals.size > 0:
            impact[impact < 0] /= np.abs(np.min(neg_vals))

        impact_polarity[ch] = impact

    smoothed_impact_polarity = gaussian_filter1d(
        impact_polarity,
        sigma=4.0,
        axis=1,
        mode='reflect',
    )
    smoothed_impact_polarity = np.clip(smoothed_impact_polarity, -1.0, 1.0)

    _plot_polarity_heatmaps(impact_polarity, ' - Raw', '')
    # Smooth only the final visualization score so raw polarity remains intact.
    _plot_polarity_heatmaps(
        smoothed_impact_polarity,
        ' - Gaussian Smoothed',
        '_smoothed',
    )


def plot_scalar_summary(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
) -> None:
    """Plot beeswarm-style summary plots for scalar feature attributions.

    Each dot represents one gene.  X-axis is the attribution value in
    Log₂(CPM+1); colour is the z-scored feature value (blue = negative,
    black = zero, red = positive).  Y-jitter is density-based so each
    row resembles a violin distribution.
    """
    from scipy.stats import gaussian_kde

    groups = [
        ('TSS_features',  attributions[2], data['TSS_features'],  TSS_FEATURE_NAMES),
        ('GB_features',   attributions[3], data['GB_features'],   GB_FEATURE_NAMES),
        ('ref_scalars',   attributions[4], data['ref_scalars'],   REF_SCALAR_NAMES),
    ]

    # For TSS and GB beeswarms: use a shared feature order derived from
    # GB attribution magnitudes, grouped as depth → fl-* → pn-*.
    # Within each group, features are sorted ascending by GB mean|attr|
    # so the most important feature in each group sits at the top.
    gb_mean_abs = np.abs(attributions[3]).mean(axis=0)  # (11,)
    _depth_idx = [10]
    _fl_idx = list(range(0, 7))
    _pn_idx = list(range(7, 10))
    # Sort within each group by ascending GB importance (top = highest)
    _fl_sorted = sorted(_fl_idx, key=lambda i: gb_mean_abs[i])
    _pn_sorted = sorted(_pn_idx, key=lambda i: gb_mean_abs[i])
    # Bottom-to-top: pn (least→most), fl (least→most), depth
    shared_11_order = np.array(_pn_sorted + _fl_sorted + _depth_idx)

    # Keep attribution_* beeswarm outputs on a consistent interpretation scale.
    shared_xlim = (-3.0, 3.0)

    for group_name, attr_vals, feat_vals, feat_names in groups:
        n_features = attr_vals.shape[1]
        mean_abs = np.abs(attr_vals).mean(axis=0)

        # TSS/GB: use shared grouped order; other groups: default sort
        if n_features == 11 and group_name in ('TSS_features', 'GB_features'):
            order = shared_11_order
        else:
            order = np.argsort(mean_abs)  # ascending

        # Z-scale mean-region-depth (index 10) which is *not* pre-scaled.
        # It is stored as log1p(depth); standardise to zero-mean, unit-var.
        feat_vals = feat_vals.copy()
        if feat_vals.shape[1] > 10:
            col = feat_vals[:, 10]
            mu, sigma = col.mean(), col.std()
            if sigma > 1e-8:
                feat_vals[:, 10] = (col - mu) / sigma

        # Symmetric colour limits centred at 0 for z-scored features.
        # Cap the effective colour range at +/-1.5 so the beeswarm keeps a
        # smooth continuum but reaches saturated blue/red before the extreme
        # tails near +/-2.
        all_feat = feat_vals[:, order]
        vlim = min(
            max(
                abs(np.nanpercentile(all_feat, 1)),
                abs(np.nanpercentile(all_feat, 99)),
                1e-6,
            ),
            1.5,
        )
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

        fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * n_features)),
                               constrained_layout=True)
        rng = np.random.default_rng(42)
        sc = None  # for colorbar

        for row_idx, feat_idx in enumerate(order):
            vals = attr_vals[:, feat_idx]
            feat = feat_vals[:, feat_idx]

            # Density-based y-jitter (violin shape)
            try:
                kde = gaussian_kde(vals)
                density = kde(vals)
                density = density / density.max()  # normalise to [0, 1]
            except (np.linalg.LinAlgError, ValueError):
                density = np.ones_like(vals)
            jitter = rng.uniform(-1, 1, size=vals.shape) * density * 0.35
            y = np.full_like(vals, row_idx) + jitter

            # Random shuffle so no systematic occlusion bias
            perm = rng.permutation(len(vals))

            sc = ax.scatter(vals[perm], y[perm],
                            c=feat[perm], cmap=SHAP_CMAP, norm=norm,
                            s=8, alpha=0.5, edgecolors='none',
                            rasterized=True)

        ax.set_yticks(range(n_features))
        ax.set_yticklabels([feat_names[i] for i in order], fontsize=8)
        ax.set_xlabel('Attribution — impact on predicted Log₂(CPM+1)', fontsize=9)
        ax.set_title(f'Attribution Beeswarm — {group_name}', fontsize=12)
        ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.set_xlim(shared_xlim)
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=30)
            cbar.solids.set_alpha(1.0)
            cbar.set_label('Feature value (z-scored)', fontsize=8)
        _save_or_show(fig, folder_path, f'attribution_beeswarm_{group_name}.png')


def plot_scalar_bars(
    attributions: List[np.ndarray],
    folder_path: Optional[str] = None,
) -> None:
    """Plot per-feature importance bars and a group-level summary.

    Produces two figures:

    1. **scalar_feature_importance.png** — Per-feature horizontal bar chart
       with all model inputs on the *same x-axis scale*:
       8 biological signal channels, reference sequence, 11 TSS scalars,
       11 GB scalars, and 3 reference scalars.

    2. **group_importance_summary.png** — Five bars showing the *total*
       mean |attribution| for each input group: Signals, Sequence,
       TSS Scalars, GB Scalars, Ref Scalars.
    """
    sig_attr = attributions[0]  # (N, 2000, 9)
    nuc_attr = attributions[1]  # (N, 2000, 4)

    # Per-channel signal importance: mean |attribution| over samples & positions
    sig_bio = np.abs(sig_attr[:, :, :8]).mean(axis=(0, 1))   # (8,)
    nuc_sum = np.abs(nuc_attr).sum(axis=-1).mean(axis=(0,)).mean()  # scalar

    # Scalar groups
    tss_mean = np.abs(attributions[2]).mean(axis=0)  # (11,)
    gb_mean  = np.abs(attributions[3]).mean(axis=0)  # (11,)
    ref_mean = np.abs(attributions[4]).mean(axis=0)  # (3,)

    # ── Figure 1: all features on one scale ──
    names_all = (
        BIOLOGICAL_SIGNAL_NAMES
        + ['Reference Sequence']
        + [f'TSS: {n}' for n in TSS_FEATURE_NAMES]
        + [f'GB: {n}' for n in GB_FEATURE_NAMES]
        + [f'Ref: {n}' for n in REF_SCALAR_NAMES]
    )
    vals_all = np.concatenate([sig_bio, [nuc_sum], tss_mean, gb_mean, ref_mean])
    order = np.argsort(vals_all)

    # Colour by group
    group_idx = (
        [0] * 8     # signals
        + [1] * 1   # sequence
        + [2] * len(TSS_FEATURE_NAMES)
        + [3] * len(GB_FEATURE_NAMES)
        + [4] * len(REF_SCALAR_NAMES)
    )
    group_colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    group_labels  = ['Signals', 'Sequence', 'TSS Scalars', 'GB Scalars', 'Ref Scalars']
    bar_colours = [group_colours[group_idx[i]] for i in order]

    n_bars = len(vals_all)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.3 * n_bars)),
                           constrained_layout=True)
    ax.barh(range(n_bars), vals_all[order], color=bar_colours)
    ax.set_yticks(range(n_bars))
    ax.set_yticklabels([names_all[i] for i in order], fontsize=7)
    ax.set_xlabel('Mean |Attribution| — Log₂(CPM+1)', fontsize=9)
    ax.set_title('Feature Importance (all model inputs)', fontsize=12)

    # Legend for groups
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=l)
               for c, l in zip(group_colours, group_labels)]
    ax.legend(handles=handles, fontsize=7, loc='lower right')
    _save_or_show(fig, folder_path, 'scalar_feature_importance.png')

    # ── Figure 2: group-level summary (5 bars) ──
    group_totals = [
        sig_bio.sum(),
        nuc_sum,
        tss_mean.sum(),
        gb_mean.sum(),
        ref_mean.sum(),
    ]
    g_order = np.argsort(group_totals)
    fig2, ax2 = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax2.barh(range(5), [group_totals[i] for i in g_order],
             color=[group_colours[i] for i in g_order])
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([group_labels[i] for i in g_order], fontsize=9)
    ax2.set_xlabel('Total Mean |Attribution| — Log₂(CPM+1)', fontsize=9)
    ax2.set_title('Input Group Importance', fontsize=12)
    _save_or_show(fig2, folder_path, 'group_importance_summary.png')


def local_prediction_decision_analysis(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    test_predictions: np.ndarray,
    reference_output: float,
    folder_path: Optional[str] = None,
    *,
    method: str,
) -> None:
    """Summarize representative local prediction decisions for individual rows."""
    local_folder = None
    if folder_path is not None:
        local_folder = os.path.join(folder_path, 'local_prediction_decisions')
        os.makedirs(local_folder, exist_ok=True)

    truth = np.asarray(data['tumor_exp'], dtype=np.float32)
    preds = np.asarray(test_predictions, dtype=np.float32)
    residuals = preds - truth
    finite = np.isfinite(truth) & np.isfinite(preds)
    if not finite.any():
        return

    case_specs = [
        ('off_correct', finite & (preds < 1.0) & (truth < 1.0), 'min_abs_resid'),
        ('low_correct', finite & (preds >= 1.0) & (preds < 3.0) & (truth >= 1.0) & (truth < 3.0), 'min_abs_resid'),
        ('mid_correct', finite & (preds >= 3.0) & (preds < 6.0) & (truth >= 3.0) & (truth < 6.0), 'min_abs_resid'),
        ('high_correct', finite & (preds >= 6.0) & (truth >= 6.0), 'min_abs_resid'),
        ('over_predicted', finite & (residuals > 0), 'max_resid'),
        ('under_predicted', finite & (residuals < 0), 'min_resid'),
        ('off_to_on', finite & (preds >= 3.0) & (truth < 1.0), 'max_abs_resid'),
        ('on_to_off', finite & (preds < 1.0) & (truth >= 3.0), 'max_abs_resid'),
    ]

    used_indices: set[int] = set()
    selected_rows: List[Dict[str, Any]] = []

    def _choose_index(indices: np.ndarray, strategy: str) -> Optional[int]:
        if indices.size == 0:
            return None
        if strategy == 'min_abs_resid':
            ranked = indices[np.argsort(np.abs(residuals[indices]))]
        elif strategy == 'max_resid':
            ranked = indices[np.argsort(-residuals[indices])]
        elif strategy == 'min_resid':
            ranked = indices[np.argsort(residuals[indices])]
        elif strategy == 'max_abs_resid':
            ranked = indices[np.argsort(-np.abs(residuals[indices]))]
        else:
            raise ValueError(f'Unknown case selection strategy: {strategy}')
        for idx in ranked:
            idx_int = int(idx)
            if idx_int not in used_indices:
                return idx_int
        return int(ranked[0])

    for case_name, mask, strategy in case_specs:
        chosen = _choose_index(np.flatnonzero(mask), strategy)
        if chosen is None:
            continue
        used_indices.add(chosen)
        selected_rows.append({
            'case_name': case_name,
            'row_index': chosen,
            'sample_name': str(data['sample_names'][chosen]),
            'gene_name': str(data['gene_names'][chosen]),
            'cohort_name': str(data['cohort_names'][chosen]),
            'truth': float(truth[chosen]),
            'prediction': float(preds[chosen]),
            'residual': float(residuals[chosen]),
            'pred_state': _expression_state_label(float(preds[chosen])),
            'truth_state': _expression_state_label(float(truth[chosen])),
        })

    if not selected_rows:
        return

    selected_df = pd.DataFrame(selected_rows)
    if local_folder is not None:
        selected_df.to_csv(
            os.path.join(local_folder, 'selected_cases.tsv'),
            sep='\t',
            index=False,
        )

    group_contribs = _signed_group_contributions(attributions)
    channel_names, channel_contribs = _signed_channel_contributions(attributions)
    scalar_names, scalar_contribs = _signed_scalar_contributions(attributions)
    group_names = list(group_contribs.keys())

    overview_rows: List[np.ndarray] = []
    overview_labels: List[str] = []

    for row in selected_rows:
        idx = int(row['row_index'])
        group_vals = np.array([group_contribs[name][idx] for name in group_names], dtype=np.float32)
        overview_rows.append(group_vals)
        overview_labels.append(row['case_name'])

        channel_vals = channel_contribs[idx]
        scalar_vals = scalar_contribs[idx]
        top_scalar_idx = np.argsort(np.abs(scalar_vals))[-12:]
        top_scalar_idx = top_scalar_idx[np.argsort(np.abs(scalar_vals[top_scalar_idx]))]
        signal_map = attributions[0][idx, :, :8].T
        vlim = float(np.nanpercentile(np.abs(signal_map), 99))
        if not np.isfinite(vlim) or vlim <= 0:
            vlim = 1e-6

        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2])

        ax_group = fig.add_subplot(gs[0, 0])
        ax_group.barh(group_names, group_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax_group.axvline(0, color='grey', linewidth=0.7, linestyle='--')
        ax_group.set_xlabel('Signed contribution to prediction')
        ax_group.set_title('Input-group contributions', fontsize=11)

        ax_channel = fig.add_subplot(gs[0, 1])
        ax_channel.barh(channel_names, channel_vals, color='#4c72b0')
        ax_channel.axvline(0, color='grey', linewidth=0.7, linestyle='--')
        ax_channel.set_xlabel('Signed contribution to prediction')
        ax_channel.set_title('Signal / sequence contributions', fontsize=11)

        ax_scalar = fig.add_subplot(gs[1, 0])
        ax_scalar.barh(
            [scalar_names[i] for i in top_scalar_idx],
            scalar_vals[top_scalar_idx],
            color='#55a868',
        )
        ax_scalar.axvline(0, color='grey', linewidth=0.7, linestyle='--')
        ax_scalar.set_xlabel('Signed contribution to prediction')
        ax_scalar.set_title('Top scalar feature contributions', fontsize=11)

        ax_signal = fig.add_subplot(gs[1, 1])
        im = ax_signal.imshow(
            signal_map,
            cmap='RdBu_r',
            aspect='auto',
            interpolation='nearest',
            norm=TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim),
        )
        ax_signal.set_yticks(range(len(BIOLOGICAL_SIGNAL_NAMES)))
        ax_signal.set_yticklabels(BIOLOGICAL_SIGNAL_NAMES, fontsize=8)
        ticks = np.linspace(0, signal_map.shape[1] - 1, 9)
        ax_signal.set_xticks(ticks)
        ax_signal.set_xticklabels([f'{int(x)}' for x in np.linspace(-1000, 1000, 9)])
        ax_signal.set_title('Single-gene TSS signal attribution map', fontsize=11)
        fig.colorbar(im, ax=ax_signal, pad=0.02, aspect=25)

        total_attr = float(group_vals.sum())
        fig.suptitle(
            f"{row['case_name']} | {row['cohort_name']} / {row['sample_name']} / {row['gene_name']}\n"
            f"truth={row['truth']:.2f}, pred={row['prediction']:.2f}, residual={row['residual']:.2f}, "
            f"background_mean={reference_output:.2f}, total_attr={total_attr:.2f}, method={method}",
            fontsize=12,
        )
        _save_or_show(fig, local_folder, f"local_decision_{row['case_name']}.png")

    overview = np.vstack(overview_rows)
    vmax = float(np.nanmax(np.abs(overview)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-6
    fig_overview, ax_overview = plt.subplots(
        figsize=(8, max(4, 0.7 * len(overview_labels))),
        constrained_layout=True,
    )
    im = ax_overview.imshow(
        overview,
        cmap='RdBu_r',
        aspect='auto',
        interpolation='nearest',
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    ax_overview.set_xticks(range(len(group_names)))
    ax_overview.set_xticklabels(group_names, rotation=30, ha='right')
    ax_overview.set_yticks(range(len(overview_labels)))
    ax_overview.set_yticklabels(overview_labels)
    ax_overview.set_title('Representative local decision cases — group contributions', fontsize=12)
    fig_overview.colorbar(im, ax=ax_overview, pad=0.02, aspect=30)
    _save_or_show(fig_overview, local_folder, 'local_decision_case_overview.png')


def _non_pht_row_mask(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return a mask excluding synthetic PHT rows from dynamic-gene analyses."""
    sample_names = np.asarray(data.get('sample_names', []), dtype=object)
    cohort_names = np.asarray(data.get('cohort_names', []), dtype=object)
    return np.array([
        not (
            str(sample).upper().startswith('PHT')
            or str(cohort).upper().startswith('PHT')
        )
        for sample, cohort in zip(sample_names, cohort_names)
    ], dtype=bool)


def _build_dynamic_gene_frame(
    data: Dict[str, np.ndarray],
    tumor_pred: np.ndarray,
    *,
    min_depth: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build row-level and gene-level truth-centered dynamic summaries."""
    depth_col = np.asarray(data['TSS_features'][:, 10], dtype=np.float32)
    truth = np.asarray(data['tumor_exp'], dtype=np.float32)
    pred = np.asarray(tumor_pred, dtype=np.float32)
    log_min = np.log1p(min_depth) if min_depth > 0 else -np.inf
    valid_mask = (depth_col > log_min) & np.isfinite(truth) & np.isfinite(pred)
    analysis_mask = valid_mask & _non_pht_row_mask(data)
    if not analysis_mask.any():
        return pd.DataFrame(), pd.DataFrame()

    row_df = pd.DataFrame({
        'row_index': np.flatnonzero(analysis_mask),
        'gene_name': np.asarray(data['gene_names'], dtype=object)[analysis_mask],
        'sample_name': np.asarray(data['sample_names'], dtype=object)[analysis_mask],
        'cohort_name': np.asarray(data['cohort_names'], dtype=object)[analysis_mask],
        'truth': truth[analysis_mask],
        'pred': pred[analysis_mask],
    })
    if row_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows: List[Dict[str, Any]] = []
    for gene_name, sub in row_df.groupby('gene_name', sort=False):
        truth_vals = sub['truth'].to_numpy(dtype=np.float64)
        pred_vals = sub['pred'].to_numpy(dtype=np.float64)
        n_stats = int(len(sub))
        truth_mean = float(np.mean(truth_vals))
        truth_std = float(np.std(truth_vals, ddof=0))
        truth_variance = float(np.var(truth_vals, ddof=0))
        truth_median = float(np.median(truth_vals))
        pred_median = float(np.median(pred_vals))
        recoverability = float('nan')
        if n_stats >= 3:
            recoverability = pearson_r(
                truth_vals.astype(np.float32),
                pred_vals.astype(np.float32),
            )
        if not np.isfinite(recoverability):
            recoverability = 0.0
        recoverability = max(recoverability, 0.0)
        summary_rows.append({
            'gene_name': str(gene_name),
            'n_stats': n_stats,
            'truth_mean': truth_mean,
            'truth_std': truth_std,
            'truth_variance': truth_variance,
            'truth_median': truth_median,
            'pred_median': pred_median,
            'recoverability': recoverability,
        })

    gene_summary = pd.DataFrame(summary_rows)
    if gene_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    gene_summary['weight_raw'] = (
        gene_summary['truth_variance'] * gene_summary['recoverability']
    )
    total_weight = float(gene_summary['weight_raw'].sum())
    if total_weight > 0:
        gene_summary['weight_norm'] = gene_summary['weight_raw'] / total_weight
    else:
        gene_summary['weight_norm'] = 0.0

    row_df = row_df.merge(gene_summary, on='gene_name', how='left', validate='many_to_one')
    std_vals = row_df['truth_std'].to_numpy(dtype=np.float64)
    centered_truth = row_df['truth'].to_numpy(dtype=np.float64) - row_df['truth_mean'].to_numpy(dtype=np.float64)
    centered_pred = row_df['pred'].to_numpy(dtype=np.float64) - row_df['truth_mean'].to_numpy(dtype=np.float64)
    z_truth = np.full(len(row_df), np.nan, dtype=np.float64)
    z_pred = np.full(len(row_df), np.nan, dtype=np.float64)
    stable = std_vals > 1e-6
    z_truth[stable] = centered_truth[stable] / std_vals[stable]
    z_pred[stable] = centered_pred[stable] / std_vals[stable]
    row_df['truth_centered'] = centered_truth
    row_df['pred_centered'] = centered_pred
    row_df['truth_z'] = z_truth
    row_df['pred_z'] = z_pred
    row_df['z_residual'] = z_pred - z_truth
    row_df['abs_truth_z'] = np.abs(z_truth)
    row_df['abs_pred_z'] = np.abs(z_pred)
    row_df['sign_agree'] = (z_truth * z_pred) > 0
    return row_df, gene_summary.sort_values('weight_raw', ascending=False).reset_index(drop=True)


def _select_gene_relative_row_sets(
    gene_rows: pd.DataFrame,
    *,
    background_per_gene: int,
    test_per_direction: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Choose same-gene baseline, up, and down rows from truth-based z scores."""
    work = gene_rows[np.isfinite(gene_rows['truth_z']) & np.isfinite(gene_rows['pred_z'])].copy()
    if work.empty:
        return work, work, work

    work['abs_z_residual'] = np.abs(work['z_residual'])
    work['baseline_score'] = (
        work['abs_truth_z']
        + work['abs_pred_z']
        + 0.5 * work['abs_z_residual']
    )

    if len(work) <= 1:
        return work.copy(), work.iloc[0:0].copy(), work.iloc[0:0].copy()

    if len(work) >= 6:
        min_background_rows = 3
    elif len(work) >= 3:
        min_background_rows = 2
    else:
        min_background_rows = 1
    max_test_rows = max(len(work) - min_background_rows, 1)
    target_test_rows = min(2 * max(int(test_per_direction), 1), max_test_rows)
    background_count = min(background_per_gene, len(work) - target_test_rows)
    background_count = max(background_count, min_background_rows)
    background_count = min(background_count, len(work) - 1)

    bg_rows = work.nsmallest(
        background_count,
        ['baseline_score', 'abs_truth_z', 'abs_pred_z'],
    ).copy()
    remaining = work.loc[~work['row_index'].isin(bg_rows['row_index'])].copy()
    if remaining.empty:
        return bg_rows, remaining, remaining

    def _pick_direction_rows(direction: str) -> pd.DataFrame:
        same_sign = remaining['sign_agree'].to_numpy(dtype=bool)
        if direction == 'up':
            tier_masks = [
                same_sign & (remaining['truth_z'] >= 0.5) & (remaining['pred_z'] >= 0.5),
                same_sign & ((remaining['truth_z'] > 0.0) | (remaining['pred_z'] > 0.0)),
                (remaining['truth_z'] > 0.0) | (remaining['pred_z'] > 0.0),
            ]
            sort_cols = ['pred_z', 'truth_z', 'abs_z_residual']
            ascending = [False, False, True]
        else:
            tier_masks = [
                same_sign & (remaining['truth_z'] <= -0.5) & (remaining['pred_z'] <= -0.5),
                same_sign & ((remaining['truth_z'] < 0.0) | (remaining['pred_z'] < 0.0)),
                (remaining['truth_z'] < 0.0) | (remaining['pred_z'] < 0.0),
            ]
            sort_cols = ['pred_z', 'truth_z', 'abs_z_residual']
            ascending = [True, True, True]

        ranked_parts: List[pd.DataFrame] = []
        used_row_indices: set[int] = set()
        for tier_mask in tier_masks:
            tier_rows = remaining.loc[
                tier_mask & ~remaining['row_index'].isin(list(used_row_indices))
            ].copy()
            if tier_rows.empty:
                continue
            tier_rows = tier_rows.sort_values(sort_cols, ascending=ascending)
            ranked_parts.append(tier_rows)
            used_row_indices.update(tier_rows['row_index'].astype(int).tolist())

        if not ranked_parts:
            return remaining.iloc[0:0].copy()
        ranked = pd.concat(ranked_parts, axis=0, ignore_index=True)
        return ranked.head(test_per_direction).copy()

    up_rows = _pick_direction_rows('up')
    down_rows = _pick_direction_rows('down')

    return bg_rows.reset_index(drop=True), up_rows.reset_index(drop=True), down_rows.reset_index(drop=True)


def _run_same_gene_relative_attribution(
    model: BioDVN,
    data: Dict[str, np.ndarray],
    gene_rows: pd.DataFrame,
    *,
    method: str,
    n_steps: int,
    batch_size: int,
    background_per_gene: int,
    test_per_direction: int,
) -> Optional[Dict[str, Any]]:
    """Run same-gene relative attribution around a truth-centered baseline."""
    bg_rows, up_rows, down_rows = _select_gene_relative_row_sets(
        gene_rows,
        background_per_gene=background_per_gene,
        test_per_direction=test_per_direction,
    )
    if bg_rows.empty:
        return None

    test_parts: List[pd.DataFrame] = []
    if not up_rows.empty:
        up_tagged = up_rows.copy()
        up_tagged['direction'] = 'up'
        test_parts.append(up_tagged)
    if not down_rows.empty:
        down_tagged = down_rows.copy()
        down_tagged['direction'] = 'down'
        test_parts.append(down_tagged)
    if not test_parts:
        return None

    test_rows = pd.concat(test_parts, axis=0, ignore_index=True)
    gene_name = str(gene_rows['gene_name'].iloc[0])
    bg_indices = bg_rows['row_index'].to_numpy(dtype=np.int64)
    test_indices = test_rows['row_index'].to_numpy(dtype=np.int64)
    bg_data = _slice_data_dict(data, bg_indices)
    test_data = _slice_data_dict(data, test_indices)
    bg_tensors = _data_to_tensors(bg_data, torch.device('cpu'))
    test_tensors = _data_to_tensors(test_data, torch.device('cpu'))
    seed = zlib.crc32(gene_name.encode('utf-8')) & 0xFFFFFFFF
    attributions = _run_attribution(
        model,
        test_tensors,
        bg_tensors,
        method=method,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
    )
    del bg_tensors, test_tensors, bg_data, test_data
    torch.cuda.empty_cache()
    return {
        'background_rows': bg_rows,
        'test_rows': test_rows,
        'attributions': attributions,
    }


def _plot_relative_direction_bars(
    value_map: Dict[str, np.ndarray],
    labels: List[str],
    folder_path: Optional[str],
    title: str,
    filename: str,
) -> None:
    """Plot side-by-side up/down bar summaries with a shared symmetric scale."""
    available = [direction for direction in ('up', 'down') if direction in value_map]
    if not available:
        return
    fig, axes = plt.subplots(
        1,
        len(available),
        figsize=(6 * len(available), max(4, 0.3 * len(labels) + 1.5)),
        constrained_layout=True,
        squeeze=False,
    )
    vmax = max(float(np.max(np.abs(value_map[direction]))) for direction in available)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-6
    for ax, direction in zip(axes[0], available):
        values = value_map[direction]
        order = np.argsort(values)
        ax.barh(
            range(len(labels)),
            values[order],
            color='#c44e52' if direction == 'up' else '#4c72b0',
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([labels[i] for i in order], fontsize=8)
        ax.axvline(0, color='grey', linewidth=0.7, linestyle='--')
        ax.set_xlim(-vmax, vmax)
        ax.set_title(direction.title(), fontsize=11)
    fig.suptitle(title, fontsize=12)
    _save_or_show(fig, folder_path, filename)


def _plot_dynamic_gene_weight_overview(
    gene_summary: pd.DataFrame,
    folder_path: Optional[str],
) -> None:
    """Visualize which genes drive the dynamic-centered analysis weighting."""
    if gene_summary.empty:
        return

    top_summary = gene_summary.head(20).iloc[::-1].copy()
    fig_bar, ax_bar = plt.subplots(
        figsize=(8, max(5, 0.32 * len(top_summary))),
        constrained_layout=True,
    )
    ax_bar.barh(
        top_summary['gene_name'],
        top_summary['weight_raw'],
        color='#4c72b0',
    )
    ax_bar.set_xlabel('Variance x recoverability weight', fontsize=9)
    ax_bar.set_title('Top dynamic genes by centered-analysis weight', fontsize=12)
    _save_or_show(fig_bar, folder_path, 'dynamic_gene_weight_top20.png')

    fig_scatter, ax_scatter = plt.subplots(
        figsize=(6.5, 5.0),
        constrained_layout=True,
    )
    sizes = np.clip(
        24.0 + 240.0 * gene_summary['weight_norm'].to_numpy(dtype=np.float32),
        24.0,
        220.0,
    )
    ax_scatter.scatter(
        gene_summary['recoverability'],
        gene_summary['truth_variance'],
        s=sizes,
        alpha=0.6,
        c='#c44e52',
        edgecolors='none',
    )
    ax_scatter.set_xlabel('Within-gene recoverability (Pearson r)', fontsize=9)
    ax_scatter.set_ylabel('Within-gene truth variance', fontsize=9)
    ax_scatter.set_title('Dynamic gene weighting landscape', fontsize=12)
    _save_or_show(fig_scatter, folder_path, 'dynamic_gene_weight_scatter.png')


def _write_marker_gene_relative_report(
    gene_name: str,
    gene_rows: pd.DataFrame,
    result: Dict[str, Any],
    folder_path: Optional[str],
    *,
    method: str,
) -> None:
    """Write same-gene relative attribution outputs for a requested marker gene."""
    marker_folder = None
    if folder_path is not None:
        marker_folder = os.path.join(folder_path, 'same_gene_markers', gene_name)
        os.makedirs(marker_folder, exist_ok=True)

    all_rows = gene_rows.sort_values('truth_z').copy()
    if marker_folder is not None:
        all_rows.to_csv(
            os.path.join(marker_folder, 'gene_rows.tsv'),
            sep='\t',
            index=False,
        )

    bg_rows = result['background_rows'].copy()
    bg_rows['selection'] = 'baseline'
    test_rows = result['test_rows'].copy()
    test_rows['selection'] = test_rows['direction']
    selected_rows = pd.concat([bg_rows, test_rows], axis=0, ignore_index=True)
    if marker_folder is not None:
        selected_rows.to_csv(
            os.path.join(marker_folder, 'selected_rows.tsv'),
            sep='\t',
            index=False,
        )

    attributions = result['attributions']
    direction_values: Dict[str, List[np.ndarray]] = {}
    for direction in ('up', 'down'):
        mask = test_rows['direction'] == direction
        if not mask.any():
            continue
        idx = np.flatnonzero(mask.to_numpy(dtype=bool))
        direction_values[direction] = [attr[idx] for attr in attributions]

    if not direction_values:
        return

    group_names = ['Signals', 'Sequence', 'TSS Scalars', 'GB Scalars', 'Ref Scalars']
    group_map: Dict[str, np.ndarray] = {}
    channel_labels = BIOLOGICAL_SIGNAL_NAMES + ['Reference Sequence']
    channel_map: Dict[str, np.ndarray] = {}
    scalar_names = (
        [f'TSS: {name}' for name in TSS_FEATURE_NAMES]
        + [f'GB: {name}' for name in GB_FEATURE_NAMES]
        + [f'Ref: {name}' for name in REF_SCALAR_NAMES]
    )
    scalar_map: Dict[str, np.ndarray] = {}
    signal_profiles: Dict[str, np.ndarray] = {}

    for direction, direction_attrs in direction_values.items():
        signed_groups = _signed_group_contributions(direction_attrs)
        group_map[direction] = np.array(
            [float(np.mean(signed_groups[name])) for name in group_names],
            dtype=np.float32,
        )
        _, signed_channels = _signed_channel_contributions(direction_attrs)
        channel_map[direction] = signed_channels.mean(axis=0)
        _, signed_scalars = _signed_scalar_contributions(direction_attrs)
        scalar_map[direction] = signed_scalars.mean(axis=0)
        signal_profiles[direction] = direction_attrs[0][:, :, :8].mean(axis=0).T

    _plot_relative_direction_bars(
        group_map,
        group_names,
        marker_folder,
        f'{gene_name} relative contributions by input group',
        'relative_group_contributions.png',
    )
    _plot_relative_direction_bars(
        channel_map,
        channel_labels,
        marker_folder,
        f'{gene_name} relative contributions by signal / sequence channel',
        'relative_signal_channel_contributions.png',
    )

    scalar_max = np.zeros(len(scalar_names), dtype=np.float32)
    for direction in scalar_map:
        scalar_max = np.maximum(scalar_max, np.abs(scalar_map[direction]))
    top_scalar_idx = np.argsort(scalar_max)[-15:]
    top_scalar_idx = top_scalar_idx[np.argsort(scalar_max[top_scalar_idx])]
    scalar_plot_map = {
        direction: values[top_scalar_idx]
        for direction, values in scalar_map.items()
    }
    _plot_relative_direction_bars(
        scalar_plot_map,
        [scalar_names[i] for i in top_scalar_idx],
        marker_folder,
        f'{gene_name} relative scalar contributions',
        'relative_scalar_contributions.png',
    )

    profile_vmax = max(float(np.nanpercentile(np.abs(signal_profiles[direction]), 99)) for direction in signal_profiles)
    if not np.isfinite(profile_vmax) or profile_vmax <= 0:
        profile_vmax = 1e-6
    fig_profiles, axes_profiles = plt.subplots(
        len(signal_profiles),
        1,
        figsize=(12, 2.2 * len(signal_profiles)),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, direction in zip(axes_profiles[:, 0], [d for d in ('up', 'down') if d in signal_profiles]):
        im = ax.imshow(
            signal_profiles[direction],
            cmap='RdBu_r',
            aspect='auto',
            interpolation='nearest',
            norm=TwoSlopeNorm(vmin=-profile_vmax, vcenter=0.0, vmax=profile_vmax),
        )
        ax.set_yticks(range(len(BIOLOGICAL_SIGNAL_NAMES)))
        ax.set_yticklabels(BIOLOGICAL_SIGNAL_NAMES, fontsize=8)
        ticks = np.linspace(0, signal_profiles[direction].shape[1] - 1, 9)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{int(x)}' for x in np.linspace(-1000, 1000, 9)])
        ax.set_title(f'{gene_name} {direction} relative to same-gene baseline', fontsize=11)
    fig_profiles.suptitle(
        f'{gene_name} same-gene relative signal profiles (method={method})',
        fontsize=12,
    )
    fig_profiles.colorbar(im, ax=axes_profiles.ravel().tolist(), pad=0.02, aspect=35)
    _save_or_show(fig_profiles, marker_folder, 'relative_signal_profiles.png')

    fig_scatter, ax_scatter = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)
    ax_scatter.scatter(
        all_rows['truth_z'],
        all_rows['pred_z'],
        s=16,
        alpha=0.7,
        c='#4c72b0',
        edgecolors='none',
    )
    ax_scatter.axvline(0, color='grey', linewidth=0.7, linestyle='--')
    ax_scatter.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax_scatter.set_xlabel('Truth z-score (truth mean/std)')
    ax_scatter.set_ylabel('Prediction z-score (truth mean/std)')
    ax_scatter.set_title(f'{gene_name} same-gene dynamic scatter', fontsize=12)
    _save_or_show(fig_scatter, marker_folder, 'truth_vs_prediction_zscore.png')


def _plot_centered_relative_attribution_suite(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    selected_meta: pd.DataFrame,
    folder_path: Optional[str],
    *,
    method: str,
    reference_output: float,
) -> None:
    """Emit the main attribution plot suite for centered same-gene relative rows."""
    suite_folder = None
    if folder_path is not None:
        suite_folder = os.path.join(folder_path, 'relative_attribution')
        os.makedirs(suite_folder, exist_ok=True)

    plot_signal_attributions(attributions, suite_folder)
    plot_signal_polarity(attributions, data, suite_folder)
    plot_scalar_summary(attributions, data, suite_folder)
    plot_scalar_bars(attributions, suite_folder)

    local_prediction_decision_analysis(
        attributions,
        data,
        selected_meta['prediction'].to_numpy(dtype=np.float32),
        reference_output,
        suite_folder,
        method=method,
    )
    tfx_input_focus_analysis(attributions, data, suite_folder)
    depth_input_focus_analysis(attributions, data, suite_folder)

    focus_df = selected_meta.copy()
    group_scores = _mean_group_attribution_per_gene(attributions)
    total_mean_abs_attr = np.zeros(len(focus_df), dtype=np.float32)
    for group_name, group_values in group_scores.items():
        values = np.asarray(group_values, dtype=np.float32)
        focus_df[f'{group_name}_mean_abs_attr'] = values
        total_mean_abs_attr += values
    focus_df['total_mean_abs_attr'] = total_mean_abs_attr

    if suite_folder is not None:
        focus_df.to_csv(
            os.path.join(suite_folder, 'centered_relative_rows.tsv'),
            sep='\t',
            index=False,
        )

    _continuum_input_focus_analysis(
        sample_df=focus_df,
        continuum_col='truth_z',
        continuum_label='Truth z-score (gene-centered)',
        folder_path=suite_folder,
        output_stem='truth_z_input_focus',
        n_bins=8,
        title_prefix='Centered Truth-Z',
    )
    _continuum_input_focus_analysis(
        sample_df=focus_df,
        continuum_col='pred_z',
        continuum_label='Prediction z-score (gene-centered)',
        folder_path=suite_folder,
        output_stem='prediction_z_input_focus',
        n_bins=8,
        title_prefix='Centered Prediction-Z',
    )
    _continuum_input_focus_analysis(
        sample_df=focus_df,
        continuum_col='z_residual',
        continuum_label='Prediction minus truth z-score',
        folder_path=suite_folder,
        output_stem='z_residual_input_focus',
        n_bins=8,
        title_prefix='Centered Residual',
    )

    fig_scatter, ax_scatter = plt.subplots(
        figsize=(6.0, 5.0),
        constrained_layout=True,
    )
    point_sizes = np.clip(
        18.0 + 420.0 * focus_df['row_weight'].to_numpy(dtype=np.float32),
        18.0,
        90.0,
    )
    point_colors = np.where(
        focus_df['direction'].to_numpy(dtype=object) == 'up',
        '#c44e52',
        '#4c72b0',
    )
    ax_scatter.scatter(
        focus_df['truth_z'],
        focus_df['pred_z'],
        s=point_sizes,
        c=point_colors,
        alpha=0.65,
        edgecolors='none',
    )
    ax_scatter.axvline(0.0, color='grey', linewidth=0.7, linestyle='--')
    ax_scatter.axhline(0.0, color='grey', linewidth=0.7, linestyle='--')
    ax_scatter.set_xlabel('Truth z-score (gene-centered)', fontsize=9)
    ax_scatter.set_ylabel('Prediction z-score (gene-centered)', fontsize=9)
    ax_scatter.set_title('Centered relative rows: truth vs prediction', fontsize=12)
    _save_or_show(fig_scatter, suite_folder, 'truth_vs_prediction_zscore.png')


def dynamic_gene_baseline_analysis(
    model: BioDVN,
    data: Dict[str, np.ndarray],
    tumor_pred: np.ndarray,
    folder_path: Optional[str] = None,
    *,
    method: str,
    n_steps: int,
    batch_size: int,
    min_depth: float = 10.0,
) -> None:
    """Analyze within-gene dynamics relative to truth-based same-gene baselines."""
    method = _canonicalize_attribution_method(method)

    dynamic_folder = None
    if folder_path is not None:
        dynamic_folder = os.path.join(folder_path, 'dynamic_gene_centered')
        os.makedirs(dynamic_folder, exist_ok=True)

    row_df, gene_summary = _build_dynamic_gene_frame(
        data,
        tumor_pred,
        min_depth=min_depth,
    )
    if row_df.empty or gene_summary.empty:
        return

    min_samples_per_gene = 6
    max_dynamic_genes = 64
    background_per_gene = 8
    test_per_direction = 2
    eligible_summary = gene_summary[
        (gene_summary['n_stats'] >= min_samples_per_gene)
        & (gene_summary['truth_std'] > 1e-6)
        & (gene_summary['weight_raw'] > 0)
    ].copy()
    if eligible_summary.empty:
        print('  [skip] No dynamic genes passed the non-PHT truth-centered eligibility filters')
        return
    eligible_summary = eligible_summary.sort_values('weight_raw', ascending=False).reset_index(drop=True)
    weight_sum = float(eligible_summary['weight_raw'].sum())
    if weight_sum > 0:
        eligible_summary['weight_norm'] = eligible_summary['weight_raw'] / weight_sum
    else:
        eligible_summary['weight_norm'] = 0.0

    if dynamic_folder is not None:
        eligible_summary.to_csv(
            os.path.join(dynamic_folder, 'dynamic_gene_summary.tsv'),
            sep='\t',
            index=False,
        )
        row_df.to_csv(
            os.path.join(dynamic_folder, 'dynamic_gene_rows.tsv'),
            sep='\t',
            index=False,
        )
    print(
        '  Dynamic centered candidate genes: '
        f'{len(eligible_summary):,} '
        f'(top {min(max_dynamic_genes, len(eligible_summary)):,} used for relative attribution)'
    )
    _plot_dynamic_gene_weight_overview(eligible_summary, dynamic_folder)

    aggregate_gene_names = eligible_summary['gene_name'].head(max_dynamic_genes).tolist()
    weight_map = {
        str(row['gene_name']): float(row['weight_norm'])
        for _, row in eligible_summary.iterrows()
    }
    variance_map = {
        str(row['gene_name']): float(row['truth_variance'])
        for _, row in eligible_summary.iterrows()
    }
    recoverability_map = {
        str(row['gene_name']): float(row['recoverability'])
        for _, row in eligible_summary.iterrows()
    }

    direction_attr_parts: Dict[str, List[List[np.ndarray]]] = {
        'up': [[] for _ in range(5)],
        'down': [[] for _ in range(5)],
    }
    direction_weights: Dict[str, List[float]] = {'up': [], 'down': []}
    selected_rows: List[Dict[str, Any]] = []
    combined_attr_parts: List[List[np.ndarray]] = [[] for _ in range(5)]
    selected_row_indices: List[int] = []
    baseline_predictions: List[float] = []
    selection_status_rows: List[Dict[str, Any]] = []

    for gene_name in aggregate_gene_names:
        gene_rows = row_df[row_df['gene_name'] == gene_name].copy()
        result = _run_same_gene_relative_attribution(
            model,
            data,
            gene_rows,
            method=method,
            n_steps=n_steps,
            batch_size=batch_size,
            background_per_gene=background_per_gene,
            test_per_direction=test_per_direction,
        )
        if result is None:
            selection_status_rows.append({
                'gene_name': str(gene_name),
                'status': 'insufficient_relative_rows',
                'n_gene_rows': int(len(gene_rows)),
                'truth_variance': variance_map[gene_name],
                'recoverability': recoverability_map[gene_name],
                'weight_norm': weight_map[gene_name],
            })
            continue

        bg_rows = result['background_rows']
        test_rows = result['test_rows']
        if test_rows.empty:
            selection_status_rows.append({
                'gene_name': str(gene_name),
                'status': 'empty_test_rows',
                'n_gene_rows': int(len(gene_rows)),
                'n_background_rows': int(len(bg_rows)),
                'truth_variance': variance_map[gene_name],
                'recoverability': recoverability_map[gene_name],
                'weight_norm': weight_map[gene_name],
            })
            continue

        baseline_predictions.extend(bg_rows['pred'].astype(float).tolist())
        selection_status_rows.append({
            'gene_name': str(gene_name),
            'status': 'ok',
            'n_gene_rows': int(len(gene_rows)),
            'n_background_rows': int(len(bg_rows)),
            'n_test_rows': int(len(test_rows)),
            'n_up_rows': int((test_rows['direction'] == 'up').sum()),
            'n_down_rows': int((test_rows['direction'] == 'down').sum()),
            'truth_variance': variance_map[gene_name],
            'recoverability': recoverability_map[gene_name],
            'weight_norm': weight_map[gene_name],
        })

        row_weight = weight_map[gene_name] / float(len(test_rows))
        attributions = result['attributions']
        for local_idx, test_row in enumerate(test_rows.itertuples(index=False)):
            direction = str(test_row.direction)
            for attr_idx in range(5):
                attr_slice = attributions[attr_idx][local_idx:local_idx + 1].copy()
                direction_attr_parts[direction][attr_idx].append(
                    attr_slice
                )
                combined_attr_parts[attr_idx].append(attr_slice)
            direction_weights[direction].append(row_weight)
            selected_row_indices.append(int(test_row.row_index))
            selected_rows.append({
                'gene_name': str(gene_name),
                'row_index': int(test_row.row_index),
                'sample_name': str(test_row.sample_name),
                'cohort_name': str(test_row.cohort_name),
                'truth': float(test_row.truth),
                'prediction': float(test_row.pred),
                'truth_mean': float(test_row.truth_mean),
                'truth_std': float(test_row.truth_std),
                'truth_variance': variance_map[gene_name],
                'recoverability': recoverability_map[gene_name],
                'truth_z': float(test_row.truth_z),
                'pred_z': float(test_row.pred_z),
                'z_residual': float(test_row.z_residual),
                'direction': direction,
                'row_weight': row_weight,
            })

        del attributions, result
        torch.cuda.empty_cache()

    if dynamic_folder is not None and selection_status_rows:
        pd.DataFrame(selection_status_rows).to_csv(
            os.path.join(dynamic_folder, 'dynamic_gene_selection_status.tsv'),
            sep='\t',
            index=False,
        )

    if selected_rows:
        ok_gene_count = sum(1 for row in selection_status_rows if row.get('status') == 'ok')
        print(
            '  Dynamic centered relative rows: '
            f'{len(selected_rows):,} across {ok_gene_count:,} genes'
        )
        selected_df = pd.DataFrame(selected_rows)
        if dynamic_folder is not None:
            selected_df.to_csv(
                os.path.join(dynamic_folder, 'dynamic_gene_selected_rows.tsv'),
                sep='\t',
                index=False,
            )

        selected_data = _slice_data_dict(
            data,
            np.asarray(selected_row_indices, dtype=np.int64),
        )
        combined_attributions = [
            np.concatenate(parts, axis=0) for parts in combined_attr_parts
        ]
        reference_output = float(np.mean(baseline_predictions)) if baseline_predictions else float(selected_df['prediction'].mean())
        _plot_centered_relative_attribution_suite(
            combined_attributions,
            selected_data,
            selected_df,
            dynamic_folder,
            method=method,
            reference_output=reference_output,
        )

        direction_stacks: Dict[str, Tuple[List[np.ndarray], np.ndarray]] = {}
        for direction in ('up', 'down'):
            if not direction_weights[direction]:
                continue
            stacked_attrs = [np.concatenate(parts, axis=0) for parts in direction_attr_parts[direction]]
            weights = np.asarray(direction_weights[direction], dtype=np.float64)
            direction_stacks[direction] = (stacked_attrs, weights)

        group_names = ['Signals', 'Sequence', 'TSS Scalars', 'GB Scalars', 'Ref Scalars']
        channel_labels = BIOLOGICAL_SIGNAL_NAMES + ['Reference Sequence']
        scalar_names = (
            [f'TSS: {name}' for name in TSS_FEATURE_NAMES]
            + [f'GB: {name}' for name in GB_FEATURE_NAMES]
            + [f'Ref: {name}' for name in REF_SCALAR_NAMES]
        )

        group_values: Dict[str, np.ndarray] = {}
        channel_values: Dict[str, np.ndarray] = {}
        scalar_values: Dict[str, np.ndarray] = {}
        signal_profiles: Dict[str, np.ndarray] = {}

        for direction, (stacked_attrs, weights) in direction_stacks.items():
            signed_groups = _signed_group_contributions(stacked_attrs)
            group_values[direction] = np.array(
                [np.average(signed_groups[name], weights=weights) for name in group_names],
                dtype=np.float32,
            )
            _, signed_channels = _signed_channel_contributions(stacked_attrs)
            channel_values[direction] = _weighted_mean(signed_channels, weights)
            _, signed_scalars = _signed_scalar_contributions(stacked_attrs)
            scalar_values[direction] = _weighted_mean(signed_scalars, weights)
            signal_profiles[direction] = _weighted_mean(stacked_attrs[0][:, :, :8], weights).T

        _plot_relative_direction_bars(
            group_values,
            group_names,
            dynamic_folder,
            'Variance x recoverability weighted centered contributions by input group',
            'dynamic_group_contributions.png',
        )
        _plot_relative_direction_bars(
            channel_values,
            channel_labels,
            dynamic_folder,
            'Variance x recoverability weighted centered contributions by signal / sequence channel',
            'dynamic_signal_channel_contributions.png',
        )

        scalar_max = np.zeros(len(scalar_names), dtype=np.float32)
        for direction in scalar_values:
            scalar_max = np.maximum(scalar_max, np.abs(scalar_values[direction]))
        top_scalar_idx = np.argsort(scalar_max)[-15:]
        top_scalar_idx = top_scalar_idx[np.argsort(scalar_max[top_scalar_idx])]
        scalar_plot_values = {
            direction: values[top_scalar_idx]
            for direction, values in scalar_values.items()
        }
        _plot_relative_direction_bars(
            scalar_plot_values,
            [scalar_names[i] for i in top_scalar_idx],
            dynamic_folder,
            'Variance x recoverability weighted centered scalar contributions',
            'dynamic_scalar_contributions.png',
        )

        available_profiles = [direction for direction in ('up', 'down') if direction in signal_profiles]
        profile_vmax = max(float(np.nanpercentile(np.abs(signal_profiles[direction]), 99)) for direction in available_profiles)
        if not np.isfinite(profile_vmax) or profile_vmax <= 0:
            profile_vmax = 1e-6
        fig_profiles, axes_profiles = plt.subplots(
            len(available_profiles),
            1,
            figsize=(12, 2.2 * len(available_profiles)),
            constrained_layout=True,
            squeeze=False,
        )
        for ax, direction in zip(axes_profiles[:, 0], available_profiles):
            im = ax.imshow(
                signal_profiles[direction],
                cmap='RdBu_r',
                aspect='auto',
                interpolation='nearest',
                norm=TwoSlopeNorm(vmin=-profile_vmax, vcenter=0.0, vmax=profile_vmax),
            )
            ax.set_yticks(range(len(BIOLOGICAL_SIGNAL_NAMES)))
            ax.set_yticklabels(BIOLOGICAL_SIGNAL_NAMES, fontsize=8)
            ticks = np.linspace(0, signal_profiles[direction].shape[1] - 1, 9)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f'{int(x)}' for x in np.linspace(-1000, 1000, 9)])
            ax.set_title(f'{direction.title()} relative to same-gene truth baseline', fontsize=11)
        fig_profiles.suptitle(
            f'Dynamic gene centered signal profiles (variance x recoverability weighted, method={method})',
            fontsize=12,
        )
        fig_profiles.colorbar(im, ax=axes_profiles.ravel().tolist(), pad=0.02, aspect=35)
        _save_or_show(fig_profiles, dynamic_folder, 'dynamic_signal_profiles.png')
    else:
        print(
            '  [skip] No same-gene centered relative rows were selected; '
            'see dynamic_gene_selection_status.tsv for per-gene diagnostics'
        )

    marker_summary_rows: List[Dict[str, Any]] = []
    for gene_name in TARGET_DYNAMIC_GENE_NAMES:
        gene_rows = row_df[row_df['gene_name'] == gene_name].copy()
        if gene_rows.empty:
            marker_summary_rows.append({'gene_name': gene_name, 'status': 'missing'})
            continue
        result = _run_same_gene_relative_attribution(
            model,
            data,
            gene_rows,
            method=method,
            n_steps=n_steps,
            batch_size=batch_size,
            background_per_gene=max(10, background_per_gene),
            test_per_direction=max(3, test_per_direction),
        )
        if result is None:
            marker_summary_rows.append({'gene_name': gene_name, 'status': 'insufficient_relative_rows'})
            continue
        _write_marker_gene_relative_report(
            gene_name,
            gene_rows,
            result,
            dynamic_folder,
            method=method,
        )
        gene_info = gene_rows.iloc[0]
        marker_summary_rows.append({
            'gene_name': gene_name,
            'status': 'ok',
            'n_stats': int(gene_info['n_stats']),
            'truth_mean': float(gene_info['truth_mean']),
            'truth_std': float(gene_info['truth_std']),
            'truth_variance': float(gene_info['truth_variance']),
            'recoverability': float(gene_info['recoverability']),
            'weight_raw': float(gene_info['weight_raw']),
        })
        del result
        torch.cuda.empty_cache()

    if dynamic_folder is not None and marker_summary_rows:
        pd.DataFrame(marker_summary_rows).to_csv(
            os.path.join(dynamic_folder, 'same_gene_marker_summary.tsv'),
            sep='\t',
            index=False,
        )
    marker_ok_count = sum(1 for row in marker_summary_rows if row.get('status') == 'ok')
    print(
        '  Same-gene marker analyses completed: '
        f'{marker_ok_count:,}/{len(marker_summary_rows):,}'
    )


# ═══════════════════════════════════════════════════════════════════════════
#            SIGNAL–SEQUENCE INTERACTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_signal_sequence_interaction(
    attributions: List[np.ndarray],
    data: Optional[Dict[str, np.ndarray]] = None,
    folder_path: Optional[str] = None,
) -> None:
    """Plot position-resolved signal × observed-sequence interaction scores."""
    if data is None:
        sig_attr = attributions[0][:, :, :8]
        nuc_summed = np.abs(attributions[1]).sum(axis=-1)
        mean_interaction = (np.abs(sig_attr) * nuc_summed[:, :, np.newaxis]).mean(axis=0).T
    else:
        signed_interaction = _signed_signal_sequence_interaction(attributions, data)
        mean_interaction = np.abs(signed_interaction).mean(axis=0).T
    positions = np.linspace(-1000, 1000, mean_interaction.shape[1])
    promoter_lo, promoter_hi = -150, 0
    promoter_start, promoter_end = _bp_window_to_indices(promoter_lo, promoter_hi, mean_interaction.shape[1])
    promoter_strength = mean_interaction[:, promoter_start:promoter_end].mean(axis=1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        constrained_layout=True,
        gridspec_kw={'height_ratios': [3.5, 1.2]},
    )
    im = axes[0].imshow(mean_interaction, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[0].set_yticks(range(len(BIOLOGICAL_SIGNAL_NAMES)))
    axes[0].set_yticklabels(BIOLOGICAL_SIGNAL_NAMES, fontsize=8)
    axes[0].set_xticks(np.linspace(0, mean_interaction.shape[1] - 1, 9))
    axes[0].set_xticklabels([f'{int(x)}' for x in np.linspace(-1000, 1000, 9)])
    axes[0].axvline(promoter_start, color='white', linewidth=0.9, linestyle='--')
    axes[0].axvline(promoter_end - 1, color='white', linewidth=0.9, linestyle='--')
    axes[0].set_title('Mean |signal × sequence interaction|', fontsize=11)
    axes[0].set_ylabel('Signal channel', fontsize=9)
    fig.colorbar(im, ax=axes[0], label='Mean |interaction|')

    order = np.argsort(promoter_strength)
    axes[1].barh(
        range(len(order)),
        promoter_strength[order],
        color='#2C7FB8',
        alpha=0.9,
    )
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels([BIOLOGICAL_SIGNAL_NAMES[i] for i in order], fontsize=8)
    axes[1].set_xlabel('Promoter-window mean |interaction| (-150 to 0 bp)', fontsize=9)
    axes[1].set_title('Promoter-focused interaction ranking', fontsize=10)

    fig.suptitle('Signal–Sequence Interaction Scores', fontsize=13)
    _save_or_show(fig, folder_path, 'signal_sequence_interaction.png')


def plot_signed_signal_sequence_interaction(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
) -> None:
    """Plot signed signal × observed-sequence interaction profiles."""
    signed_interaction = _signed_signal_sequence_interaction(attributions, data)
    mean_signed = signed_interaction.mean(axis=0).T
    promoter_start, promoter_end = _bp_window_to_indices(-150, 0, mean_signed.shape[1])

    vmax = np.nanpercentile(np.abs(mean_signed), 99)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    im = ax.imshow(mean_signed, cmap=CUSTOM_CMAP, aspect='auto', interpolation='nearest', norm=norm)
    ax.set_yticks(range(len(BIOLOGICAL_SIGNAL_NAMES)))
    ax.set_yticklabels(BIOLOGICAL_SIGNAL_NAMES, fontsize=8)
    ax.set_xticks(np.linspace(0, mean_signed.shape[1] - 1, 9))
    ax.set_xticklabels([f'{int(x)}' for x in np.linspace(-1000, 1000, 9)])
    ax.axvline(promoter_start, color='white', linewidth=0.9, linestyle='--')
    ax.axvline(promoter_end - 1, color='white', linewidth=0.9, linestyle='--')
    ax.set_xlabel('Position relative to TSS (bp)', fontsize=10)
    ax.set_ylabel('Signal channel', fontsize=9)
    ax.set_title('Signed signal–sequence interaction across all signal channels', fontsize=11)
    fig.colorbar(im, ax=ax, label='Signed interaction')

    fig.suptitle('Signed Signal–Sequence Interaction Scores', fontsize=13)
    _save_or_show(fig, folder_path, 'signal_sequence_interaction_signed.png')


def counterfactual_signal_sequence_synergy(
    model: nn.Module,
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
    top_k: int = 4,
    n_genes: int = 256,
    window_size: int = 21,
    batch_size: int = 64,
    max_plot_rows: int = 12,
) -> None:
    """Quantify joint sequence and signal effects with local ablations."""
    signed_interaction = _signed_signal_sequence_interaction(attributions, data)
    bg_freq = data['TSS_nucleotides'].mean(axis=(0, 1)).astype(np.float32)

    half_window = window_size // 2
    summary_rows: List[Dict[str, Any]] = []

    for signal_channel, channel_name in enumerate(BIOLOGICAL_SIGNAL_NAMES):
        profile = np.abs(signed_interaction[:, :, signal_channel]).mean(axis=0)
        positions = _top_spaced_positions(profile, top_k=top_k, spacing=window_size)

        for pos in positions:
            gene_scores = np.abs(signed_interaction[:, pos, signal_channel])
            if gene_scores.size == 0:
                continue
            order = np.argsort(gene_scores)[::-1]
            use_idx = order[: min(n_genes, order.size)]
            subset = _slice_data_dict(data, use_idx)

            lo = max(0, pos - half_window)
            hi = min(subset['TSS_signals'].shape[1], pos + half_window + 1)

            seq_only = _slice_data_dict(subset, np.arange(subset['TSS_signals'].shape[0]))
            sig_only = _slice_data_dict(subset, np.arange(subset['TSS_signals'].shape[0]))
            both = _slice_data_dict(subset, np.arange(subset['TSS_signals'].shape[0]))

            seq_only['TSS_nucleotides'][:, lo:hi, :] = bg_freq.reshape(1, 1, 4)
            sig_only['TSS_signals'][:, lo:hi, signal_channel] = 0.0
            both['TSS_nucleotides'][:, lo:hi, :] = bg_freq.reshape(1, 1, 4)
            both['TSS_signals'][:, lo:hi, signal_channel] = 0.0

            original_pred = _predict_tumor_expression(model, subset, batch_size=batch_size)
            seq_pred = _predict_tumor_expression(model, seq_only, batch_size=batch_size)
            sig_pred = _predict_tumor_expression(model, sig_only, batch_size=batch_size)
            both_pred = _predict_tumor_expression(model, both, batch_size=batch_size)

            cooperative_gain = seq_pred + sig_pred - both_pred - original_pred
            signal_window_mean = subset['TSS_signals'][:, lo:hi, signal_channel].mean(axis=1)
            positive_mask = signal_window_mean > 0.0
            negative_mask = signal_window_mean < 0.0

            summary_rows.append({
                'channel': channel_name,
                'position_index': int(pos),
                'position_bp': int(pos - 1000),
                'n_genes': int(cooperative_gain.shape[0]),
                'n_positive_signal': int(positive_mask.sum()),
                'n_negative_signal': int(negative_mask.sum()),
                'mean_abs_interaction': float(profile[pos]),
                'mean_signed_interaction': float(signed_interaction[:, pos, signal_channel].mean()),
                'cooperative_gain_mean': float(np.mean(cooperative_gain)),
                'cooperative_gain_median': float(np.median(cooperative_gain)),
                'cooperative_gain_positive_signal_mean': (
                    float(np.mean(cooperative_gain[positive_mask]))
                    if bool(positive_mask.any()) else np.nan
                ),
                'cooperative_gain_negative_signal_mean': (
                    float(np.mean(cooperative_gain[negative_mask]))
                    if bool(negative_mask.any()) else np.nan
                ),
            })

    if not summary_rows:
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df['abs_cooperative_gain_mean'] = summary_df['cooperative_gain_mean'].abs()
    summary_df = summary_df.sort_values(
        ['abs_cooperative_gain_mean', 'mean_abs_interaction'],
        ascending=[False, False],
    ).reset_index(drop=True)
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
        summary_df.to_csv(
            os.path.join(folder_path, 'signal_sequence_counterfactual_synergy.tsv'),
            sep='\t',
            index=False,
        )

    plot_df = summary_df.head(min(max_plot_rows, len(summary_df))).copy()
    labels = [
        f"{row.channel} {int(row.position_bp):+d} bp"
        for row in plot_df.itertuples(index=False)
    ]
    cmap = plt.get_cmap('tab10')
    channel_color = {
        name: cmap(idx % 10)
        for idx, name in enumerate(BIOLOGICAL_SIGNAL_NAMES)
    }
    bar_colors = [channel_color.get(name, '#555555') for name in plot_df['channel']]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, max(5.5, 0.42 * len(labels) + 2.5)),
        constrained_layout=True,
        gridspec_kw={'height_ratios': [2.2, 1.6]},
    )

    axes[0].axvline(0.0, color='black', linewidth=0.8)
    axes[0].barh(
        range(len(labels)),
        plot_df['cooperative_gain_mean'].to_numpy(dtype=np.float32),
        color=bar_colors,
        alpha=0.9,
    )
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlabel('Mean cooperative gain after joint ablation', fontsize=9)
    axes[0].set_title('Top signal–sequence synergy hits', fontsize=10)

    axes[1].axvline(0.0, color='black', linewidth=0.8)
    y_pos = np.arange(len(labels))
    pos_vals = plot_df['cooperative_gain_positive_signal_mean'].to_numpy(dtype=np.float32)
    neg_vals = plot_df['cooperative_gain_negative_signal_mean'].to_numpy(dtype=np.float32)
    for row_idx, (pos_val, neg_val) in enumerate(zip(pos_vals, neg_vals)):
        if np.isfinite(pos_val) and np.isfinite(neg_val):
            axes[1].plot([neg_val, pos_val], [row_idx, row_idx], color='#999999', linewidth=1.0, alpha=0.9)
    axes[1].scatter(pos_vals, y_pos, color='#CC3311', s=18, label='Positive signal window')
    axes[1].scatter(neg_vals, y_pos, color='#0077BB', s=18, label='Negative signal window')
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlabel('Direction-specific cooperative gain', fontsize=9)
    axes[1].set_title('Conditional gain by signal sign', fontsize=10)
    axes[1].legend(fontsize=8, loc='best')

    fig.suptitle('Counterfactual Signal–Sequence Synergy', fontsize=13)
    _save_or_show(fig, folder_path, 'signal_sequence_counterfactual_synergy.png')


def extract_interaction_motifs(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
    top_k: int = 20,
    window_size: int = 20,
) -> None:
    """Extract and plot motifs at top signal–sequence interaction hotspots.

    Identifies positions with the highest combined fragment-end ×
    nucleotide interaction score, then builds attribution-weighted PWMs
    from the nucleotide windows surrounding those positions.  The top 6
    motifs are displayed as sequence logos.
    """
    sig_attr = attributions[0]                                # (N, 2000, 9)
    nuc_attr = attributions[1]                                # (N, 2000, 4)
    nuc_data = data['TSS_nucleotides']                        # (N, 2000, 4)

    # Combine both fragment-end channels
    frag_end_attr = np.abs(sig_attr[:, :, 1:3]).mean(axis=-1)  # (N, 2000)
    nuc_summed = np.abs(nuc_attr).sum(axis=-1)                 # (N, 2000)
    interaction = frag_end_attr * nuc_summed                   # (N, 2000)
    mean_interaction = interaction.mean(axis=0)                # (2000,)

    # Find top-k peaks with minimum spacing of window_size
    half_w = window_size // 2
    peaks: List[int] = []
    profile = mean_interaction.copy()
    for _ in range(top_k):
        search_start = half_w
        search_end = len(profile) - half_w
        if search_start >= search_end:
            break
        pos = int(np.argmax(profile[search_start:search_end]) + search_start)
        if profile[pos] <= 0:
            break
        peaks.append(pos)
        lo = max(0, pos - window_size)
        hi = min(len(profile), pos + window_size)
        profile[lo:hi] = 0
    peaks.sort()

    # Build attribution-weighted PWMs at each peak position
    all_pwms: List[Tuple[int, np.ndarray]] = []
    for pos in peaks:
        lo = max(0, pos - half_w)
        hi = min(2000, pos + half_w)
        windows = nuc_data[:, lo:hi, :]         # (N, W, 4)
        weights = interaction[:, pos]            # (N,)
        weight_sum = weights.sum()
        if weight_sum < 1e-12:
            continue
        weights = weights / weight_sum
        pwm = np.einsum('n,nwc->wc', weights, windows)  # (W, 4)
        all_pwms.append((pos, pwm))

    n_show = min(6, len(all_pwms))
    if n_show == 0:
        return

    fig, axes = plt.subplots(n_show, 1, figsize=(8, 1.8 * n_show),
                            constrained_layout=True)
    if n_show == 1:
        axes = [axes]

    for ax, (pos, pwm) in zip(axes, all_pwms[:n_show]):
        row_sums = pwm.sum(axis=1, keepdims=True)
        pwm_norm = np.where(row_sums > 0, pwm / row_sums, 0.25)
        # Information content per position
        ic = np.log2(4) + np.where(
            pwm_norm > 0,
            pwm_norm * np.log2(pwm_norm + 1e-12),
            0.0,
        ).sum(axis=1, keepdims=True)
        ic_df = pd.DataFrame(pwm_norm * ic, columns=['A', 'C', 'G', 'T'])
        logomaker.Logo(ic_df, ax=ax, shade_below=0.5, fade_below=0.5)
        bp_offset = pos - 1000
        ax.set_title(f'Position {bp_offset:+d} bp', fontsize=9)
        ax.set_ylabel('bits', fontsize=8)

    fig.suptitle('Interaction-Conditioned Motifs (Fragment End × Sequence)',
                 fontsize=12)
    _save_or_show(fig, folder_path, 'interaction_conditioned_motifs.png')


def highlight_directional_promoter_motifs(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
    window_start_bp: int = -150,
    window_end_bp: int = 0,
    motif_window_size: int = 21,
) -> None:
    """Highlight the strongest directional sequence motifs in the core promoter."""
    signed_interaction = _signed_signal_sequence_interaction(attributions, data)
    nuc_data = data['TSS_nucleotides']
    start_idx, end_idx = _bp_window_to_indices(window_start_bp, window_end_bp, signed_interaction.shape[1])
    region = signed_interaction[:, start_idx:end_idx, :]
    mean_region = region.mean(axis=0)

    positive_flat = int(np.argmax(mean_region))
    negative_flat = int(np.argmin(mean_region))
    n_positions, n_channels = mean_region.shape
    candidates = [
        ('positive', positive_flat, float(mean_region.reshape(-1)[positive_flat])),
        ('negative', negative_flat, float(mean_region.reshape(-1)[negative_flat])),
    ]

    motif_rows: List[Dict[str, Any]] = []
    motifs_to_plot: List[Tuple[str, int, int, float, np.ndarray]] = []
    for direction, flat_index, score in candidates:
        pos_offset, channel_idx = divmod(flat_index, n_channels)
        pos_idx = start_idx + pos_offset
        channel_name = BIOLOGICAL_SIGNAL_NAMES[channel_idx]
        signed_weights = signed_interaction[:, pos_idx, channel_idx]
        if direction == 'positive':
            pwm = _build_weighted_pwm(nuc_data, signed_weights, pos_idx, motif_window_size)
        else:
            pwm = _build_weighted_pwm(nuc_data, -signed_weights, pos_idx, motif_window_size)
        if pwm is None:
            continue
        motifs_to_plot.append((direction, pos_idx, channel_idx, score, pwm))
        motif_rows.append({
            'direction': direction,
            'channel': channel_name,
            'position_index': int(pos_idx),
            'position_bp': int(pos_idx - 1000),
            'mean_signed_interaction': float(score),
        })

    if not motifs_to_plot:
        return

    summary_df = pd.DataFrame(motif_rows)
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
        summary_df.to_csv(
            os.path.join(folder_path, 'promoter_directional_motif_highlights.tsv'),
            sep='\t',
            index=False,
        )

    fig, axes = plt.subplots(
        len(motifs_to_plot),
        1,
        figsize=(8, 2.1 * len(motifs_to_plot)),
        constrained_layout=True,
    )
    if len(motifs_to_plot) == 1:
        axes = [axes]

    for ax, (direction, pos_idx, channel_idx, score, pwm) in zip(axes, motifs_to_plot):
        row_sums = pwm.sum(axis=1, keepdims=True)
        pwm_norm = np.where(row_sums > 0, pwm / row_sums, 0.25)
        ic = np.log2(4) + np.where(
            pwm_norm > 0,
            pwm_norm * np.log2(pwm_norm + 1e-12),
            0.0,
        ).sum(axis=1, keepdims=True)
        ic_df = pd.DataFrame(pwm_norm * ic, columns=['A', 'C', 'G', 'T'])
        logomaker.Logo(ic_df, ax=ax, shade_below=0.5, fade_below=0.5)
        ax.set_ylabel('bits', fontsize=8)
        ax.set_title(
            f"{direction.title()} motif: {BIOLOGICAL_SIGNAL_NAMES[channel_idx]} at {pos_idx - 1000:+d} bp "
            f"(mean signed interaction {score:.3g})",
            fontsize=9,
        )

    fig.suptitle('Directional promoter motifs (-150 to 0 bp)', fontsize=12)
    _save_or_show(fig, folder_path, 'promoter_directional_motif_highlights.png')


def tfx_input_focus_analysis(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
    n_bins: int = 8,
) -> None:
    """Summarize how input-group focus shifts across the TFX continuum."""
    group_scores = _mean_group_attribution_per_gene(attributions)
    sample_names = np.asarray(
        data.get('sample_names', [f'gene_{idx}' for idx in range(len(data['TFX_label']))]),
        dtype=object,
    )
    cohort_names = np.asarray(
        data.get('cohort_names', ['Unknown'] * len(sample_names)),
        dtype=object,
    )
    tfx_vals = np.asarray(data['TFX_label'], dtype=np.float32)

    sample_df = pd.DataFrame({
        'cohort_name': cohort_names,
        'sample_name': sample_names,
        'tfx': tfx_vals,
    })
    agg_spec: Dict[str, str] = {'tfx': 'mean'}
    for group_name, scores in group_scores.items():
        col = f'{group_name}_mean_abs_attr'
        sample_df[col] = np.asarray(scores, dtype=np.float32)
        agg_spec[col] = 'mean'

    sample_df = sample_df.groupby(
        ['cohort_name', 'sample_name'],
        as_index=False,
        sort=True,
    ).agg(agg_spec)

    if sample_df.empty:
        return
    _continuum_input_focus_analysis(
        sample_df=sample_df,
        continuum_col='tfx',
        continuum_label='TFX',
        folder_path=folder_path,
        output_stem='tfx_input_focus',
        n_bins=n_bins,
        x_min=0.0,
        x_max=1.0,
        title_prefix='TFX-Conditioned',
    )


def _continuum_input_focus_analysis(
    sample_df: pd.DataFrame,
    continuum_col: str,
    continuum_label: str,
    folder_path: Optional[str],
    output_stem: str,
    n_bins: int,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    title_prefix: Optional[str] = None,
) -> None:
    """Summarize how attribution focus shifts across a scalar continuum."""
    sample_df = sample_df.copy()
    continuum_vals = np.asarray(sample_df[continuum_col], dtype=np.float32)
    finite_mask = np.isfinite(continuum_vals)
    if not bool(finite_mask.any()):
        return
    sample_df = sample_df.loc[finite_mask].copy()
    continuum_vals = np.asarray(sample_df[continuum_col], dtype=np.float32)

    group_names = [
        'Signals', 'Sequence', 'TSS Scalars', 'GB Scalars', 'Ref Scalars'
    ]
    total_scores = np.zeros(len(sample_df), dtype=np.float32)
    for group_name in group_names:
        total_scores += np.asarray(sample_df[f'{group_name}_mean_abs_attr'], dtype=np.float32)
    total_scores = np.maximum(total_scores, 1e-12)
    sample_df['total_mean_abs_attr'] = total_scores
    for group_name in group_names:
        sample_df[f'{group_name}_share'] = sample_df[f'{group_name}_mean_abs_attr'] / sample_df['total_mean_abs_attr']

    if x_min is None:
        x_min = float(np.nanmin(continuum_vals))
    if x_max is None:
        x_max = float(np.nanmax(continuum_vals))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return
    if x_max <= x_min:
        x_max = x_min + 1e-6

    sample_df['_continuum_value'] = np.clip(continuum_vals, x_min, x_max)
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    sample_df['_continuum_bin'] = pd.cut(
        sample_df['_continuum_value'],
        bins=bin_edges,
        include_lowest=True,
        labels=False,
    )

    bin_rows: List[Dict[str, float]] = []
    for bin_idx in range(n_bins):
        bin_mask = sample_df['_continuum_bin'] == bin_idx
        if not bool(bin_mask.any()):
            continue
        subset = sample_df.loc[bin_mask]
        center = float((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2.0)
        row = {
            'continuum_bin': int(bin_idx),
            'continuum_center': center,
            'n_samples': int(len(subset)),
            'continuum_mean': float(subset[continuum_col].mean()),
        }
        for group_name in group_names:
            row[f'{group_name}_mean_abs_attr'] = float(subset[f'{group_name}_mean_abs_attr'].mean())
            row[f'{group_name}_share'] = float(subset[f'{group_name}_share'].mean())
        bin_rows.append(row)

    if not bin_rows:
        return

    bin_df = pd.DataFrame(bin_rows)
    def _safe_slope_and_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute slope/correlation without NumPy divide warnings on constants."""
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        finite = np.isfinite(x_arr) & np.isfinite(y_arr)
        if int(finite.sum()) < 2:
            return 0.0, 0.0

        x_valid = x_arr[finite]
        y_valid = y_arr[finite]
        x_centered = x_valid - float(np.mean(x_valid))
        y_centered = y_valid - float(np.mean(y_valid))
        x_ss = float(np.dot(x_centered, x_centered))
        y_ss = float(np.dot(y_centered, y_centered))
        if x_ss <= 1e-12 or y_ss <= 1e-12:
            return 0.0, 0.0

        xy = float(np.dot(x_centered, y_centered))
        slope = xy / x_ss
        corr = xy / float(np.sqrt(x_ss * y_ss))
        corr = float(np.clip(corr, -1.0, 1.0))
        return float(slope), corr

    slope_rows: List[Dict[str, float]] = []
    for group_name in group_names:
        share_col = f'{group_name}_share'
        abs_col = f'{group_name}_mean_abs_attr'
        share_slope, share_corr = _safe_slope_and_corr(
            sample_df['_continuum_value'].to_numpy(dtype=np.float64),
            sample_df[share_col].to_numpy(dtype=np.float64),
        )
        abs_slope, abs_corr = _safe_slope_and_corr(
            sample_df['_continuum_value'].to_numpy(dtype=np.float64),
            sample_df[abs_col].to_numpy(dtype=np.float64),
        )
        slope_rows.append({
            'group': group_name,
            f'share_slope_vs_{continuum_col}': share_slope,
            f'share_corr_vs_{continuum_col}': share_corr,
            f'abs_slope_vs_{continuum_col}': abs_slope,
            f'abs_corr_vs_{continuum_col}': abs_corr,
        })
    slope_df = pd.DataFrame(slope_rows)

    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
        sample_df.to_csv(os.path.join(folder_path, f'{output_stem}_samples.tsv'), sep='\t', index=False)
        bin_df.to_csv(os.path.join(folder_path, f'{output_stem}_bins.tsv'), sep='\t', index=False)
        slope_df.to_csv(os.path.join(folder_path, f'{output_stem}_summary.tsv'), sep='\t', index=False)

    colors = {
        'Signals': '#1f77b4',
        'Sequence': '#ff7f0e',
        'TSS Scalars': '#2ca02c',
        'GB Scalars': '#d62728',
        'Ref Scalars': '#9467bd',
    }
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True, sharex=True)
    for group_name in group_names:
        axes[0].plot(
            bin_df['continuum_center'],
            bin_df[f'{group_name}_share'],
            marker='o',
            linewidth=1.8,
            label=group_name,
            color=colors[group_name],
        )
        axes[1].plot(
            bin_df['continuum_center'],
            bin_df[f'{group_name}_mean_abs_attr'],
            marker='o',
            linewidth=1.8,
            label=group_name,
            color=colors[group_name],
        )
    axes[0].set_ylabel('Mean attribution share', fontsize=9)
    axes[0].set_title(f'Relative input focus across the {continuum_label} continuum', fontsize=11)
    axes[0].legend(fontsize=8, ncol=3, loc='best')
    axes[1].set_ylabel('Mean |attribution| per entity', fontsize=9)
    axes[1].set_xlabel(continuum_label, fontsize=10)
    axes[1].set_title(f'Absolute input reliance across the {continuum_label} continuum', fontsize=11)
    axes[1].set_xlim(x_min, x_max)
    suptitle_prefix = title_prefix if title_prefix is not None else f'{continuum_label}-Conditioned'
    fig.suptitle(f'{suptitle_prefix} Input Focus', fontsize=13)
    _save_or_show(fig, folder_path, f'{output_stem}.png')


def depth_input_focus_analysis(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
    n_bins: int = 8,
) -> None:
    """Summarize how input-group focus shifts across per-gene mean depth."""
    group_scores = _mean_group_attribution_per_gene(attributions)
    depth_log1p = np.asarray(data['TSS_features'][:, 10], dtype=np.float32)
    depth_raw = np.expm1(depth_log1p)

    gene_df = pd.DataFrame({
        'gene_index': np.arange(depth_raw.shape[0], dtype=np.int32),
        'mean_region_depth': depth_raw,
    })
    for group_name, scores in group_scores.items():
        gene_df[f'{group_name}_mean_abs_attr'] = np.asarray(scores, dtype=np.float32)

    _continuum_input_focus_analysis(
        sample_df=gene_df,
        continuum_col='mean_region_depth',
        continuum_label='Mean Region Depth',
        folder_path=folder_path,
        output_stem='depth_input_focus',
        n_bins=n_bins,
        x_min=0.0,
        x_max=float(np.nanpercentile(depth_raw[np.isfinite(depth_raw)], 99.0)) if bool(np.isfinite(depth_raw).any()) else None,
        title_prefix='Depth-Conditioned',
    )


# ═══════════════════════════════════════════════════════════════════════════
#              REDUCTION-BLOCK ACTIVATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def reduction_activation_analysis(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
    n_samples: int = 2000,
    seed: int = 200,
) -> None:
    """Analyze ReductionBlock filter activations.

    Registers forward hooks on ``model.reduction.convs`` to capture the
    raw Conv2d output at each reduction iteration, then:

    1. Plots mean |activation| ranked by filter for each layer.
    2. Plots spatial activation profiles for the top-k filters.
    """
    device = next(model.parameters()).device
    sub = subsample(data, sample_size=n_samples, seed=seed,
                    restriction='all_genes')
    tensors = _data_to_tensors(sub, device)
    features = _tensors_to_feature_dict(tensors)

    # Register forward hooks on each reduction conv
    activation_accum: Dict[int, List[np.ndarray]] = {}
    hooks: List[torch.utils.hooks.RemovableHook] = []

    for i, conv in enumerate(model.reduction.convs):
        activation_accum[i] = []
        def _make_hook(idx: int) -> Callable:
            def _hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
                # Conv2d output: (batch, out_ch, L', 1) → squeeze + permute
                activation_accum[idx].append(
                    out.detach().squeeze(-1).permute(0, 2, 1).cpu().numpy()
                )
            return _hook
        hooks.append(conv.register_forward_hook(_make_hook(i)))

    # Batched forward pass to avoid GPU OOM
    fwd_batch = 32
    n_total = tensors[0].shape[0]
    for start in range(0, n_total, fwd_batch):
        end = min(start + fwd_batch, n_total)
        batch_feat = {k: v[start:end] for k, v in features.items()}
        with torch.no_grad():
            model(batch_feat, return_internal_loss=False)

    for h in hooks:
        h.remove()

    del tensors, features
    torch.cuda.empty_cache()

    activations: Dict[int, np.ndarray] = {
        i: np.concatenate(parts, axis=0) for i, parts in activation_accum.items()
    }

    # ── Per-layer filter magnitude bar chart ──
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers,
                             figsize=(5 * n_layers, 5), squeeze=False,
                             constrained_layout=True)
    axes = axes[0]
    for i in range(n_layers):
        act = activations[i]                          # (N, L, C)
        mean_per_filter = np.abs(act).mean(axis=(0, 1))  # (C,)
        order = np.argsort(mean_per_filter)[::-1]
        axes[i].bar(range(len(mean_per_filter)),
                    mean_per_filter[order], color='steelblue')
        axes[i].set_xlabel('Filter (ranked)', fontsize=9)
        axes[i].set_ylabel('Mean |Activation|', fontsize=9)
        axes[i].set_title(f'Reduction Conv {i + 1}', fontsize=11)
    fig.suptitle('ReductionBlock Filter Activation Magnitudes', fontsize=13)
    _save_or_show(fig, folder_path, 'reduction_activation_magnitudes.png')

    # ── Spatial profiles for top-k filters per layer ──
    top_k = min(8, min(a.shape[-1] for a in activations.values()))
    for i in range(n_layers):
        act = activations[i]                                  # (N, L, C)
        mean_spatial = np.abs(act).mean(axis=0)               # (L, C)
        mean_per_filter = mean_spatial.mean(axis=0)           # (C,)
        top_indices = np.argsort(mean_per_filter)[::-1][:top_k]

        fig_sp, axes_sp = plt.subplots(
            top_k, 1, figsize=(10, 1.2 * top_k),
            constrained_layout=True,
        )
        if top_k == 1:
            axes_sp = [axes_sp]

        for j, (ax, fidx) in enumerate(zip(axes_sp, top_indices)):
            profile = mean_spatial[:, fidx]
            ax.fill_between(range(len(profile)), profile,
                            alpha=0.6, color='steelblue')
            ax.set_ylabel(f'F{fidx}', fontsize=8, rotation=0,
                          labelpad=20, ha='right')
            ax.set_xlim(0, len(profile))
            if j < top_k - 1:
                ax.set_xticklabels([])
        axes_sp[-1].set_xlabel('Spatial position', fontsize=9)
        fig_sp.suptitle(
            f'Reduction Conv {i + 1} — Top {top_k} Filter Spatial Profiles',
            fontsize=12,
        )
        _save_or_show(fig_sp, folder_path,
                      f'reduction_conv{i + 1}_spatial_profiles.png')

    # ── Importance-weighted composite spatial profile (1 figure, 3 layers) ──
    fig_comp, axes_comp = plt.subplots(
        n_layers, 1, figsize=(10, 3.0 * n_layers),
        constrained_layout=True,
    )
    if n_layers == 1:
        axes_comp = [axes_comp]
    for i, ax in enumerate(axes_comp):
        act = activations[i]                                 # (N, L, C)
        mean_spatial = np.abs(act).mean(axis=0)              # (L, C)
        importance = mean_spatial.mean(axis=0)                # (C,)
        weighted_profile = np.average(mean_spatial, axis=1,
                                      weights=importance)    # (L,)
        ax.fill_between(range(len(weighted_profile)),
                        weighted_profile, alpha=0.6,
                        color='steelblue')
        ax.plot(weighted_profile, color='steelblue', linewidth=1.0)
        ax.set_xlim(0, len(weighted_profile))
        ax.set_ylabel('Weighted Mean\n|Activation|', fontsize=9)
        ax.set_title(f'Reduction Conv {i + 1}  '
                     f'({act.shape[-1]} filters, spatial {act.shape[1]})',
                     fontsize=11)
        if i < n_layers - 1:
            ax.set_xticklabels([])
    axes_comp[-1].set_xlabel('Spatial position (gene-body bins)', fontsize=10)
    fig_comp.suptitle(
        'ReductionBlock — Importance-Weighted Spatial Profiles', fontsize=13,
    )
    _save_or_show(fig_comp, folder_path,
                  'reduction_weighted_spatial_summary.png')


# ═══════════════════════════════════════════════════════════════════════════
#                LATENT-SPACE UMAP VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def _build_sample_metadata() -> pd.DataFrame:
    """Build a lookup table mapping ``(cohort, sample_name)`` to subtype and TFx.

    Scans all ``data/*_sample-splits.tsv`` files so cohort/sample metadata is
    available for PLCA, BTS, and other composite datasets rather than being
    hard-wired to BTS only.
    """
    import csv

    sample_keys: set[Tuple[str, str]] = set()
    for splits_path in sorted((_PROJECT_ROOT / 'data').glob('*_sample-splits.tsv')):
        with open(splits_path, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                cohort = row['Cohort'].strip()
                sample = row['Sample'].strip()
                sample_keys.add((cohort, sample))
                prefix = f'{cohort}_'
                if sample.startswith(prefix):
                    sample_keys.add((cohort, sample[len(prefix):]))

    if not sample_keys:
        return pd.DataFrame(columns=['subtype', 'tfx'])

    # Read per-cohort label files
    labels_dir = _PROJECT_ROOT / 'labels'
    label_rows: List[Dict[str, Any]] = []
    for cohort in sorted({cohort for cohort, _ in sample_keys}):
        label_path = labels_dir / f'{cohort}_TFX_Subtype.tsv'
        if not label_path.exists():
            continue
        df = pd.read_csv(label_path, sep='\t')
        rename_map = {str(col).strip().lower(): col for col in df.columns}
        sample_col = rename_map.get('sample')
        subtype_col = rename_map.get('subtype')
        tfx_col = rename_map.get('tfx')
        if sample_col is None:
            continue
        for _, row in df.iterrows():
            sample = str(row[sample_col]).strip()
            subtype = str(row[subtype_col]).strip() if subtype_col is not None else 'Unknown'
            tfx = pd.to_numeric(row[tfx_col], errors='coerce') if tfx_col is not None else np.nan
            label_rows.append({
                'cohort': cohort,
                'sample_name': sample,
                'subtype': subtype,
                'tfx': float(tfx) if pd.notna(tfx) else np.nan,
            })
            prefix = f'{cohort}_'
            if sample.startswith(prefix):
                label_rows.append({
                    'cohort': cohort,
                    'sample_name': sample[len(prefix):],
                    'subtype': subtype,
                    'tfx': float(tfx) if pd.notna(tfx) else np.nan,
                })

    if label_rows:
        meta = pd.DataFrame(label_rows).drop_duplicates(subset=['cohort', 'sample_name'])
        meta = meta.set_index(['cohort', 'sample_name'])[['subtype', 'tfx']]
    else:
        meta = pd.DataFrame(columns=['subtype', 'tfx'])
        meta.index = pd.MultiIndex.from_tuples([], names=['cohort', 'sample_name'])

    missing_rows: List[Dict[str, Any]] = []
    for cohort, sample in sorted(sample_keys):
        if (cohort, sample) in meta.index:
            continue
        cohort_index = meta.loc[cohort] if cohort in meta.index.get_level_values(0) else None
        if cohort_index is not None and not cohort_index.empty:
            matches = [idx for idx in cohort_index.index if str(idx).startswith(sample)]
            if matches:
                matched = cohort_index.loc[matches[0]]
                missing_rows.append({
                    'cohort': cohort,
                    'sample_name': sample,
                    'subtype': matched['subtype'],
                    'tfx': matched['tfx'],
                })
                continue
        missing_rows.append({
            'cohort': cohort,
            'sample_name': sample,
            'subtype': 'Unknown',
            'tfx': np.nan,
        })

    if missing_rows:
        missing_df = pd.DataFrame(missing_rows).drop_duplicates(subset=['cohort', 'sample_name'])
        missing_df = missing_df.set_index(['cohort', 'sample_name'])[['subtype', 'tfx']]
        meta = pd.concat([meta, missing_df])

    return meta.sort_index()

def latent_space_umap(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
    n_samples: int = 10_000,
    seed: int = 200,
) -> None:
    """Generate UMAP plots of the tumor latent space.

    Extracts ``z_tumor_mean`` via a forward hook on the tumor encoder, then
    projects it to 2D with UMAP.

    Produces:

    1. Predicted expression (continuous).
    2. Ground-truth expression (from labels).
    3. Cancer subtype (ARPC / NEPC / ASCL1 / etc.).
    4. Tumor fraction (TFx, continuous).
    5. Prediction residuals (signed error).
    6. Absolute error.
    7. Cohort identity (PC-PDX vs SCLC-PDX vs HD-L vs PHT).
    8. Sample identity (top-15 by count, rest as "Other").
    9. CI90 width (90% predictive interval, the default reported CI).
    10. Activity odds ratio (raw, signed log2 mixture-weight contrast).
    11. Activity odds ratio (absolute, |AOR|).
    """
    _ensure_umap()
    import umap
    from collections import Counter

    device = next(model.parameters()).device
    sub = subsample(data, sample_size=n_samples, seed=seed,
                    restriction='all_genes')
    tensors = _data_to_tensors(sub, device)
    features = _tensors_to_feature_dict(tensors)

    z_tumor_mean_parts: List[np.ndarray] = []
    tumor_exp_parts: List[np.ndarray] = []
    tumor_ci90_width_parts: List[np.ndarray] = []
    tumor_pi_normal_parts: List[np.ndarray] = []

    fwd_batch = 32
    n_total = tensors[0].shape[0]
    for start in range(0, n_total, fwd_batch):
        end = min(start + fwd_batch, n_total)
        batch_feat = {k: v[start:end] for k, v in features.items()}

        z_tumor_chunk: Optional[np.ndarray] = None

        def _tumor_hook(module: nn.Module, inp: Any,
                        out: Tuple[torch.Tensor, torch.Tensor]) -> None:
            nonlocal z_tumor_chunk
            z_tumor_chunk = out[0].detach().cpu().numpy()

        h_t = model.tumor_encoder.register_forward_hook(_tumor_hook)
        with torch.no_grad():
            preds, _ = model(batch_feat, return_internal_loss=False)
        h_t.remove()

        z_tumor_mean_parts.append(z_tumor_chunk)
        tumor_exp_parts.append(preds['tumor_exp'].cpu().numpy().squeeze(-1))

        # CI90 width and AOR mixture weight are derived from the tumor MDN so the
        # latent colourings match the downstream analysis surfaces exactly.
        tumor_mdn = preds['tumor_mdn']
        ci_bounds = mixture_quantile(tumor_mdn, quantiles=(0.05, 0.95)).cpu().numpy()
        tumor_ci90_width_parts.append(ci_bounds[:, 1] - ci_bounds[:, 0])
        pi_normal = tumor_mdn['log_pi'].exp()[:, 2].cpu().numpy()
        tumor_pi_normal_parts.append(pi_normal)

    del tensors, features
    torch.cuda.empty_cache()

    z_tumor_mean = np.concatenate(z_tumor_mean_parts, axis=0)
    tumor_pred = np.concatenate(tumor_exp_parts, axis=0)
    tumor_ci90_width = np.concatenate(tumor_ci90_width_parts, axis=0)
    tumor_pi_normal = np.concatenate(tumor_pi_normal_parts, axis=0)

    # Activity odds ratio (raw, signed) via the exact analysis formula:
    # log2((pi_normal + eps) / (1 - pi_normal + eps)).  eps mirrors
    # analysis/analysis_utils.py::_AOR_EPSILON (1e-6).
    _aor_epsilon = 1e-6
    _normal_mass = np.clip(tumor_pi_normal.astype(np.float64), 0.0, 1.0)
    _off_mass = np.clip(1.0 - _normal_mass, 0.0, 1.0)
    tumor_aor = np.log2((_normal_mass + _aor_epsilon) / (_off_mass + _aor_epsilon))
    tumor_aor_abs = np.abs(tumor_aor)

    tumor_truth = sub['tumor_exp']
    sample_names = np.array(sub['sample_names'])
    cohort_names = np.array(sub.get('cohort_names', ['Unknown'] * len(sample_names)))

    # Build sample-level metadata lookup
    meta = _build_sample_metadata()

    cohorts = cohort_names
    subtypes = np.array([
        meta.loc[(cohort, sample), 'subtype']
        if (cohort, sample) in meta.index else 'Unknown'
        for cohort, sample in zip(cohort_names, sample_names)
    ])
    tfx_values = np.array([
        float(meta.loc[(cohort, sample), 'tfx'])
        if (cohort, sample) in meta.index else np.nan
        for cohort, sample in zip(cohort_names, sample_names)
    ], dtype=np.float32)

    # ── Run UMAP + plots for the tumor branch ──
    for branch, z_latent, pred_arr, truth_arr in [
        ('tumor', z_tumor_mean, tumor_pred, tumor_truth),
    ]:
        branch_label = branch.capitalize()

        reducer = umap.UMAP(
            n_components=2, random_state=seed, n_jobs=1,
            n_neighbors=30, min_dist=0.3, metric='cosine',
        )
        with warnings.catch_warnings():
            # Seeded UMAP is intentionally deterministic and single-threaded.
            warnings.filterwarnings(
                'ignore',
                message=(
                    'n_jobs value 1 overridden to 1 by setting random_state. '
                    'Use no seed for parallelism.'
                ),
                category=UserWarning,
                module='umap\\.umap_',
            )
            embedding = reducer.fit_transform(z_latent)

        residuals = pred_arr - truth_arr
        abs_error = np.abs(residuals)

        # ── Helper for continuous-colour UMAPs ──
        def _plot_continuous(values: np.ndarray, cmap: str, label: str,
                             title: str, fname: str, *,
                             vcenter: Optional[float] = None,
                             _emb=embedding) -> None:
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
            ax.set_aspect('equal')
            if vcenter is not None:
                vmin, vmax = np.nanpercentile(values, [2, 98])
                norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
            else:
                norm = None
            sc = ax.scatter(
                _emb[:, 0], _emb[:, 1],
                c=values, cmap=cmap, norm=norm,
                s=2, alpha=0.5, rasterized=True,
            )
            ax.set_xlabel('UMAP 1', fontsize=10)
            ax.set_ylabel('UMAP 2', fontsize=10)
            ax.set_title(title, fontsize=12)
            fig.colorbar(sc, ax=ax, label=label)
            _save_or_show(fig, folder_path, fname)

        # ── Helper for categorical UMAPs ──
        def _plot_categorical(labels: np.ndarray, title: str, fname: str,
                              *, max_legend: int = 20,
                              palette: Optional[str] = None,
                              _emb=embedding) -> None:
            fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
            ax.set_aspect('equal')
            unique = sorted(set(labels))
            n_cats = len(unique)
            cmap_name = palette or ('tab10' if n_cats <= 10 else 'tab20')
            cmap_c = plt.cm.get_cmap(cmap_name, max(n_cats, 1))

            if n_cats <= max_legend:
                for i, cat in enumerate(unique):
                    mask = labels == cat
                    ax.scatter(_emb[mask, 0], _emb[mask, 1],
                               c=[cmap_c(i)], s=2, alpha=0.4,
                               label=cat, rasterized=True)
            else:
                counts = Counter(labels.tolist())
                top_cats = [c for c, _ in counts.most_common(max_legend - 1)]
                cmap_c = plt.cm.get_cmap(cmap_name, max_legend)
                for i, cat in enumerate(top_cats):
                    mask = labels == cat
                    ax.scatter(_emb[mask, 0], _emb[mask, 1],
                               c=[cmap_c(i)], s=2, alpha=0.4,
                               label=cat, rasterized=True)
                other_mask = ~np.isin(labels, top_cats)
                if other_mask.any():
                    ax.scatter(_emb[other_mask, 0], _emb[other_mask, 1],
                               c='grey', s=1, alpha=0.2,
                               label=f'Other ({other_mask.sum():,})',
                               rasterized=True)
            ax.legend(fontsize=6, markerscale=4, ncol=1,
                      loc='center left', bbox_to_anchor=(1.02, 0.5))
            ax.set_xlabel('UMAP 1', fontsize=10)
            ax.set_ylabel('UMAP 2', fontsize=10)
            ax.set_title(title, fontsize=12)
            _save_or_show(fig, folder_path, fname)

        # 1. Predicted expression (continuous)
        _plot_continuous(
            pred_arr, 'viridis', 'Predicted Log\u2082(CPM+1)',
            f'{branch_label} Latent Space \u2014 Predicted Expression',
            f'umap_{branch}_predicted.png',
        )

        # 2. Ground-truth expression (continuous)
        _plot_continuous(
            truth_arr, 'viridis', 'True Log\u2082(CPM+1)',
            f'{branch_label} Latent Space \u2014 True Expression',
            f'umap_{branch}_truth.png',
        )

        # 3. Cancer subtype
        _plot_categorical(
            subtypes,
            f'{branch_label} Latent Space \u2014 Cancer Subtype',
            f'umap_{branch}_cancer_subtype.png',
        )

        # 4. Tumor fraction (TFx)
        _plot_continuous(
            tfx_values, 'magma', 'Tumor Fraction',
            f'{branch_label} Latent Space \u2014 TFx',
            f'umap_{branch}_tfx.png',
        )

        # 5. Prediction residuals (signed, centred at 0)
        _plot_continuous(
            residuals, 'RdBu_r', 'Residual (pred \u2212 truth)',
            f'{branch_label} Latent Space \u2014 Prediction Residuals',
            f'umap_{branch}_residuals.png', vcenter=0.0,
        )

        # 6. Absolute error
        _plot_continuous(
            abs_error, 'hot_r', '|Prediction Error|',
            f'{branch_label} Latent Space \u2014 Absolute Error',
            f'umap_{branch}_abs_error.png',
        )

        # 7. Cohort identity
        _plot_categorical(
            cohorts,
            f'{branch_label} Latent Space \u2014 Cohort',
            f'umap_{branch}_cohort.png',
        )

        # 8. Sample identity (top 15 + Other)
        _plot_categorical(
            sample_names,
            f'{branch_label} Latent Space \u2014 Sample Identity',
            f'umap_{branch}_samples.png',
            max_legend=16,
        )

        # 9. CI90 width (90% predictive interval — the default reported CI)
        _plot_continuous(
            tumor_ci90_width, 'cividis', 'CI90 Width (q95 \u2212 q05)',
            f'{branch_label} Latent Space \u2014 CI90 Width',
            f'umap_{branch}_ci90_width.png',
        )

        # 10. Activity odds ratio (raw, signed; centred at 0)
        _plot_continuous(
            tumor_aor, 'RdBu_r', 'Activity Odds Ratio (log\u2082)',
            f'{branch_label} Latent Space \u2014 Activity Odds Ratio (raw)',
            f'umap_{branch}_aor_raw.png', vcenter=0.0,
        )

        # 11. Activity odds ratio (absolute, |AOR|)
        _plot_continuous(
            tumor_aor_abs, 'hot_r', '|Activity Odds Ratio| (log\u2082)',
            f'{branch_label} Latent Space \u2014 Activity Odds Ratio (absolute)',
            f'umap_{branch}_aor_abs.png',
        )


# ═══════════════════════════════════════════════════════════════════════════
#                     ATTRIBUTION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def attribution_analysis(
    model: BioDVN,
    data: Dict[str, np.ndarray],
    background_size: int = 4_096,
    test_size: int = 1_536,
    method: str = 'shap',
    n_steps: int = 32,
    batch_size: int = 16,
    save_filepath: Optional[str] = None,
    seed: int = 200,
    run_interaction: bool = True,
    run_reduction: bool = True,
    run_umap: bool = True,
) -> None:
    """Run the default all-gene interpretation workflow.

    Runs one tumor-mean attribution analysis with disjoint background and test
    draws from the full loaded gene pool. After the attribution pass,
    optionally runs:
    - Dynamic-gene centered baseline analysis.
    - ReductionBlock activation analysis.
    - Latent-space UMAP visualization.

    Args:
        model: Trained BioDVN in eval mode.
        data: Full dataset dict from :func:`load_interpretation_data`.
        background_size: Number of genes for the attribution baseline.
        test_size: Number of genes for the test set.
        method: ``'shap'``/``'deep_shap'``, ``'integrated_gradients'``, or
            ``'gradient_shap'``.
        n_steps: Interpolation steps (IG), samples (SHAP expected gradients /
            GradientShap).
        batch_size: Internal batch size for attribution computation.
        save_filepath: Root directory for output figures.
        seed: Random seed for subsampling.
        run_interaction: If True, run signal–sequence interaction analysis.
        run_reduction: If True, run ReductionBlock activation analysis.
        run_umap: If True, run latent-space UMAP visualization.
    """
    tumor_pred = _predict_tumor_expression(
        model,
        data,
        batch_size=max(64, batch_size),
    )
    print('\n=== Attribution analysis: attribution ===')

    folder_path = None
    if save_filepath is not None:
        folder_path = os.path.join(save_filepath, 'attribution')
        os.makedirs(folder_path, exist_ok=True)

    bg_data, test_data = subsample_disjoint_pair(
        data,
        background_size=background_size,
        test_size=test_size,
        seed=seed,
        restriction='all_genes',
    )

    bg_tensors = _data_to_tensors(bg_data, torch.device('cpu'))
    test_tensors = _data_to_tensors(test_data, torch.device('cpu'))

    print(f'  Background: {bg_tensors[0].shape[0]} genes')
    print(f'  Test: {test_tensors[0].shape[0]} genes')
    print('  Background/test gene sets are disjoint')
    canonical_method = _canonicalize_attribution_method(method)
    if canonical_method == 'shap':
        print('  Method: shap (expected gradients)')
    else:
        print(f'  Method: {canonical_method} (n_steps={n_steps})')

    bg_predictions = _predict_tumor_expression(
        model,
        bg_data,
        batch_size=max(64, batch_size),
    )
    test_predictions = _predict_tumor_expression(
        model,
        test_data,
        batch_size=max(64, batch_size),
    )
    finite_bg = np.isfinite(bg_predictions)
    reference_output = float(bg_predictions[finite_bg].mean()) if finite_bg.any() else float('nan')
    print(f'  Background mean prediction: {reference_output:.3f}')

    attributions = _run_attribution(
        model,
        test_tensors,
        bg_tensors,
        method=method,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
    )

    del bg_tensors, test_tensors
    torch.cuda.empty_cache()

    print('  Plotting signal attributions...')
    plot_signal_attributions(attributions, folder_path)

    print('  Plotting polarity-impact...')
    plot_signal_polarity(attributions, test_data, folder_path)

    print('  Plotting scalar summaries...')
    plot_scalar_summary(attributions, test_data, folder_path)
    plot_scalar_bars(attributions, folder_path)

    print('  Explaining representative local decisions...')
    local_prediction_decision_analysis(
        attributions,
        test_data,
        test_predictions,
        reference_output,
        folder_path,
        method=canonical_method,
    )

    print('  Testing TFX-conditioned input focus...')
    tfx_input_focus_analysis(attributions, test_data, folder_path)

    print('  Testing depth-conditioned input focus...')
    depth_input_focus_analysis(attributions, test_data, folder_path)

    if run_interaction and model.signal_processor.conv1d_signals is not None and model.signal_processor.conv2d_nt is not None:
        print('  Plotting signal–sequence interaction...')
        plot_signal_sequence_interaction(attributions, test_data, folder_path)
        print('  Plotting signed signal–sequence interaction...')
        plot_signed_signal_sequence_interaction(attributions, test_data, folder_path)
        print('  Extracting interaction-conditioned motifs...')
        extract_interaction_motifs(attributions, test_data, folder_path)
        print('  Highlighting directional promoter motifs...')
        highlight_directional_promoter_motifs(attributions, test_data, folder_path)
        print('  Running counterfactual signal–sequence synergy analysis...')
        counterfactual_signal_sequence_synergy(
            model,
            attributions,
            test_data,
            folder_path,
            batch_size=max(32, batch_size),
        )
    elif run_interaction:
        print('  [skip] Signal–sequence interaction analysis requires both signal and sequence branches')

    del attributions
    print('  Done: attribution')

    # ── Analyses that run once ──

    analysis_root = save_filepath

    print('\n=== Dynamic gene centered analysis ===')
    dynamic_gene_baseline_analysis(
        model,
        data,
        tumor_pred,
        analysis_root,
        method=method,
        n_steps=n_steps,
        batch_size=max(8, batch_size),
    )

    if run_reduction:
        print('\n=== ReductionBlock activation analysis ===')
        red_folder = None
        if analysis_root is not None:
            red_folder = os.path.join(analysis_root, 'reduction_activations')
            os.makedirs(red_folder, exist_ok=True)
        reduction_activation_analysis(model, data, red_folder, seed=seed)

    if run_umap:
        print('\n=== Latent-space UMAP ===')
        umap_folder = None
        if analysis_root is not None:
            umap_folder = os.path.join(analysis_root, 'latent_umap')
            os.makedirs(umap_folder, exist_ok=True)
        latent_space_umap(model, data, umap_folder, seed=seed)
