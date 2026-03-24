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
    - Attribution analysis via Captum (IntegratedGradients, GradientShap)
    - Polarity-impact and summary attribution plots
    - Signal–sequence interaction analysis (fragment-end ↔ nucleotide)
    - ReductionBlock activation analysis (cross-channel interaction)
    - Latent-space visualization (UMAP of z_tumor)
    - Mixture-component attribution (zero-class probability)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import logomaker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---------------------------------------------------------------------------
# Deferred optional imports — only loaded when the relevant functions are
# called, so the rest of the module works without them installed.
# ---------------------------------------------------------------------------
_captum_available: Optional[bool] = None
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


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

import sys as _sys
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from models.models import BioDVN, BioDVNConfig  # noqa: E402
from models.model_utils import mixture_mean  # noqa: E402


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

# Classic SHAP-style diverging colormap (blue → violet → red).
# Matches the shap.plots.beeswarm palette.
SHAP_CMAP = LinearSegmentedColormap.from_list(
    'shap_blue_violet_red',
    [
        (0.0, '#1E88E5'),   # blue
        (0.5, '#D500F9'),   # violet / purple
        (1.0, '#FF0D57'),   # red / hot-pink
    ],
    N=256,
)


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
            *weights_path*.  Falls back to ``BioDVNConfig()`` defaults when no
            config file is found.
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

    if config_path is not None:
        with open(config_path, 'r') as f:
            cfg_raw = json.load(f)
        allowed = set(BioDVNConfig.__dataclass_fields__.keys())
        cfg = {k: v for k, v in cfg_raw.items() if k in allowed}
        model_config = BioDVNConfig(**cfg)
    else:
        model_config = BioDVNConfig()

    model = BioDVN(model_config).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    # Checkpoints saved by Trainer include a nested dict; plain state-dicts
    # (from Proteus.py full-train) are loaded directly.
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#                          DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

# Approximate bytes per gene loaded into RAM:
#   TSS_signals(2000×9) + TSS_nucleotides(2000×4) + TSS_features(11)
#   + GB_features(11) + ref_scalars(3) + TFX_label(1) + tumor_exp(1)
# = 26,027 float32 values × 4 bytes ≈ 104 KB/gene
_BYTES_PER_GENE = 26_027 * 4


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
            gene_names       list[str]       length N
            sample_names     list[str]       length N

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
        'GB_features': [], 'ref_scalars': [], 'TFX_label': [], 'tumor_exp': [],
    }
    all_gene_names: List[str] = []
    all_sample_names: List[str] = []
    total_loaded = 0

    for idx, zpath in enumerate(zarr_paths, 1):
        if n_stores > 1 and idx % 20 == 0:
            print(f'  Loading zarr store {idx}/{n_stores} '
                  f'({total_loaded:,} genes so far)...')
        store = zarr.open(zpath, mode='r')
        flat = store['flat'] if 'flat' in store else store

        n_genes = min(int(flat['TSS_signals'].shape[0]), genes_per_store)

        all_arrays['TSS_signals'].append(flat['TSS_signals'][:n_genes].astype(np.float32))
        all_arrays['TSS_nucleotides'].append(flat['TSS_nucleotides'][:n_genes].astype(np.float32))
        # Backward-compatible: old Zarr stores have 17 TSS features; use first 11.
        tss_feat = flat['TSS_features'][:n_genes].astype(np.float32)
        all_arrays['TSS_features'].append(tss_feat[:, :11])
        all_arrays['GB_features'].append(flat['GB_features'][:n_genes].astype(np.float32))
        all_arrays['ref_scalars'].append(flat['ref_scalars'][:n_genes].astype(np.float32))
        all_arrays['TFX_label'].append(flat['TFX_label'][:n_genes].astype(np.float32))

        if 'tumor_exp' in flat:
            all_arrays['tumor_exp'].append(flat['tumor_exp'][:n_genes].astype(np.float32))
        else:
            all_arrays['tumor_exp'].append(np.full(n_genes, np.nan, dtype=np.float32))

        # Gene names
        if 'gene_names' in store:
            names = [str(g) for g in store['gene_names'][:n_genes]]
        else:
            names = [f'gene_{i}' for i in range(n_genes)]
        all_gene_names.extend(names)

        # Sample name
        sample_name = store.attrs.get('sample_name', Path(zpath).stem.replace('.zarr', ''))
        all_sample_names.extend([str(sample_name)] * n_genes)
        total_loaded += n_genes

    data = {k: np.concatenate(v, axis=0) for k, v in all_arrays.items()}
    data['gene_names'] = all_gene_names
    data['sample_names'] = all_sample_names
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


def save_npz(data: Dict[str, np.ndarray], filepath: str) -> None:
    """Persist interpretation data to a compressed ``.npz`` file."""
    import time as _time
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    for k in ('gene_names', 'sample_names'):
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
    """Convert a numpy data dict to the 5-tensor tuple expected by Captum.

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
    """Build the feature dict that BioDVN.forward expects from a Captum tuple."""
    return {
        'TSS_signals': tensors[0],
        'TSS_nucleotides': tensors[1],
        'TSS_features': tensors[2],
        'GB_features': tensors[3],
        'ref_scalars': tensors[4],
    }


# ═══════════════════════════════════════════════════════════════════════════
#         CAPTUM MODEL WRAPPER  (dict inputs → scalar output)
# ═══════════════════════════════════════════════════════════════════════════

class _TumorMeanWrapper(nn.Module):
    """Thin wrapper around :class:`BioDVN` for Captum compatibility.

    Captum attribution methods require models that accept positional
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


class _ZeroComponentWrapper(nn.Module):
    """Wrapper that returns the zero-class mixture weight (log-probability).

    Instead of returning the point estimate, this returns the model's
    predicted probability of zero expression — specifically ``exp(log_pi[:, 0])``,
    the hurdle/zero-component weight from the tumor MDN.
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
        # log_pi[:, 0] is the zero-component log-probability
        return predictions['tumor_mdn']['log_pi'][:, 0]  # (batch,)


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

        dilation = reduction._skip_offsets[iteration_idx] * 2 // max(ksize_h - 1, 1) if ksize_h > 1 else 1
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
#                  ATTRIBUTION ANALYSIS (CAPTUM)
# ═══════════════════════════════════════════════════════════════════════════

def _run_attribution(
    model: BioDVN,
    test_tensors: Tuple[torch.Tensor, ...],
    baseline_tensors: Tuple[torch.Tensor, ...],
    method: str = 'integrated_gradients',
    n_steps: int = 50,
    batch_size: int = 32,
    wrapper_cls: type = _TumorMeanWrapper,
) -> List[np.ndarray]:
    """Run Captum attribution and return per-input-group numpy arrays.

    Args:
        model: Trained BioDVN in eval mode.
        test_tensors: 5-tuple of test input tensors.
        baseline_tensors: 5-tuple of baseline (background) tensors.
        method: ``'integrated_gradients'`` or ``'gradient_shap'``.
        n_steps: Number of interpolation steps (IntegratedGradients) or
            samples (GradientShap).
        batch_size: Number of test genes per outer loop iteration.
            Each batch is internally expanded by *n_steps* interpolation
            points, so VRAM usage scales as ``batch_size × n_steps``.
        wrapper_cls: Wrapper class to use (default: _TumorMeanWrapper).
            Must accept a model and return a scalar per sample.

    Returns:
        List of 5 numpy arrays (one per input group) with the same shape as
        the corresponding test tensors.
    """
    _ensure_captum()
    from captum.attr import IntegratedGradients, GradientShap

    wrapper = wrapper_cls(model)
    wrapper.eval()

    if method == 'integrated_gradients':
        attr_method = IntegratedGradients(wrapper)
    elif method == 'gradient_shap':
        attr_method = GradientShap(wrapper)
    else:
        raise ValueError(f"Unknown attribution method: {method!r}")

    n_test = test_tensors[0].shape[0]
    attr_accum: Optional[List[np.ndarray]] = None

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        batch_inputs = tuple(t[start:end] for t in test_tensors)

        if method == 'integrated_gradients':
            baselines = tuple(
                t.mean(dim=0, keepdim=True).expand_as(b)
                for t, b in zip(baseline_tensors, batch_inputs)
            )
            batch_attr = attr_method.attribute(
                batch_inputs, baselines=baselines, n_steps=n_steps,
                internal_batch_size=batch_size,
            )
        else:  # gradient_shap
            batch_attr = attr_method.attribute(
                batch_inputs, baselines=baseline_tensors, n_samples=n_steps,
            )

        batch_np = [a.detach().cpu().numpy() for a in batch_attr]

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

    fig, axs = plt.subplots(n_channels, 1,
                            figsize=(12, 0.7 * n_channels),
                            constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for ch in range(n_channels):
        im = axs[ch].imshow(
            impact_polarity[ch:ch + 1], cmap=CUSTOM_CMAP,
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

    fig.suptitle('Attribution Polarity-Impact (Signals)', fontsize=13)
    fig.colorbar(im, ax=axs.ravel().tolist(), label='Direction & impact')
    _save_or_show(fig, folder_path, 'TSS_signal_polarity_impact.png')

    # ── Subset version: curated signal order ──
    subset_indices = [0, 3, 6, 1]  # depth, nucleosome, diversity, coverage
    n_sub = len(subset_indices)
    fig2, axs2 = plt.subplots(n_sub, 1,
                               figsize=(12, 0.7 * n_sub),
                               constrained_layout=True)
    for row, ch in enumerate(subset_indices):
        im2 = axs2[row].imshow(
            impact_polarity[ch:ch + 1], cmap=CUSTOM_CMAP,
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

    fig2.suptitle('Attribution Polarity-Impact (Key Signals)', fontsize=13)
    fig2.colorbar(im2, ax=axs2.ravel().tolist(), label='Direction & impact')
    _save_or_show(fig2, folder_path, 'TSS_signal_polarity_impact_subset.png')


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

    # Shared x-axis limits for TSS and GB beeswarms (use the wider GB range)
    gb_attr = attributions[3]
    tss_attr = attributions[2]
    _xmax = max(np.abs(gb_attr).max(), np.abs(tss_attr).max()) * 1.05
    shared_xlim = (-_xmax, _xmax)

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

        # Symmetric colour limits centred at 0 for z-scored features
        all_feat = feat_vals[:, order]
        vlim = max(abs(np.nanpercentile(all_feat, 1)),
                   abs(np.nanpercentile(all_feat, 99)),
                   1e-6)
        norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

        fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * n_features)),
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
                            s=2, alpha=0.5, edgecolors='none',
                            rasterized=True)

        ax.set_yticks(range(n_features))
        ax.set_yticklabels([feat_names[i] for i in order], fontsize=8)
        ax.set_xlabel('Attribution — impact on predicted Log₂(CPM+1)', fontsize=9)
        ax.set_title(f'Attribution Beeswarm — {group_name}', fontsize=12)
        ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
        if group_name in ('TSS_features', 'GB_features'):
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


# ═══════════════════════════════════════════════════════════════════════════
#            SIGNAL–SEQUENCE INTERACTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_signal_sequence_interaction(
    attributions: List[np.ndarray],
    folder_path: Optional[str] = None,
) -> None:
    """Plot position-resolved fragment-end × nucleotide interaction scores.

    Computes the element-wise product ``|attr_frag| × |attr_nuc_sum|`` at
    each of the 2000 TSS positions.  Positions with high interaction
    indicate genomic locations where the model jointly relies on both
    fragment-end patterns (channels 1–2) and nucleotide context.
    """
    sig_attr = attributions[0]    # (N, 2000, 9)
    nuc_attr = attributions[1]    # (N, 2000, 4)

    # Fragment-end channels: 1 = coverage, 2 = asymmetry
    frag_end_attr = np.abs(sig_attr[:, :, 1:3])           # (N, 2000, 2)
    nuc_summed = np.abs(nuc_attr).sum(axis=-1)             # (N, 2000)
    interaction = frag_end_attr * nuc_summed[:, :, np.newaxis]  # (N, 2000, 2)
    mean_interaction = interaction.mean(axis=0)            # (2000, 2)

    channel_names = ['Frag End Coverage × Sequence', 'Frag End Asymmetry × Sequence']
    positions = np.linspace(-1000, 1000, 2000)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), constrained_layout=True)
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        ax.fill_between(positions, mean_interaction[:, i],
                        alpha=0.6, color='steelblue')
        ax.set_ylabel('Interaction', fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.set_xlim(-1000, 1000)
        if i < 1:
            ax.set_xticklabels([])
    axes[-1].set_xlabel('Position relative to TSS (bp)', fontsize=10)

    fig.suptitle('Signal–Sequence Interaction Scores', fontsize=13)
    _save_or_show(fig, folder_path, 'signal_sequence_interaction.png')


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
    """Build a lookup table mapping sample names to cohort, subtype, and TFx.

    Reads ``data/BTS_sample-splits.tsv`` for cohort membership and each
    cohort's ``labels/{cohort}_TFX_Subtype.tsv`` for subtype and TFx.

    Returns:
        DataFrame indexed by sample name with columns
        ``['cohort', 'subtype', 'tfx']``.
    """
    import csv

    splits_path = _PROJECT_ROOT / 'data' / 'BTS_sample-splits.tsv'
    if not splits_path.exists():
        return pd.DataFrame(columns=['cohort', 'subtype', 'tfx'])

    # Read cohort assignments
    with open(splits_path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = [(r['Sample'].strip(), r['Cohort'].strip()) for r in reader]
    sample_cohort = {s: c for s, c in rows}

    # Read per-cohort label files
    labels_dir = _PROJECT_ROOT / 'labels'
    label_dfs: List[pd.DataFrame] = []
    for cohort in sorted(set(sample_cohort.values())):
        label_path = labels_dir / f'{cohort}_TFX_Subtype.tsv'
        if not label_path.exists():
            continue
        df = pd.read_csv(label_path, sep='\t')
        df['cohort'] = cohort
        label_dfs.append(df)

    if not label_dfs:
        # Fall back to cohort-only
        return pd.DataFrame({
            'cohort': sample_cohort,
            'subtype': 'Unknown',
            'tfx': np.nan,
        })

    meta = pd.concat(label_dfs, ignore_index=True)
    meta = meta.rename(columns={'sample': 'sample_name'})
    meta = meta.set_index('sample_name')[['cohort', 'subtype', 'tfx']]
    meta['tfx'] = pd.to_numeric(meta['tfx'], errors='coerce')

    # Add any BTS samples missing from label files (cohort only)
    for sample, cohort in sample_cohort.items():
        if sample not in meta.index:
            # HD-L samples differ: BTS has "BP1124576" but label file has
            # "BP1124576_S78".  Try prefix match.
            matched = [idx for idx in meta.index if idx.startswith(sample)]
            if matched:
                first = matched[0]
                meta.loc[sample] = meta.loc[first]
            else:
                meta.loc[sample] = {'cohort': cohort, 'subtype': 'Unknown',
                                    'tfx': np.nan}

    return meta

def latent_space_umap(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
    n_samples: int = 10_000,
    seed: int = 200,
) -> None:
    """Generate UMAP plots of the tumor latent space.

    Extracts ``z_tumor_mean`` via a forward hook on the tumor VAE encoder,
    then projects to 2D with UMAP.  Produces:

    1. Predicted tumor expression (continuous).
    2. Ground-truth tumor expression (from labels).
    3. Cancer subtype (ARPC / NEPC / ASCL1 / etc.).
    4. Tumor fraction (TFx, continuous).
    5. Prediction residuals (signed error).
    6. Absolute error.
    7. Cohort identity (PC-PDX vs SCLC-PDX vs HD-L vs PHT).
    8. Sample identity (top-15 by count, rest as "Other").
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

    fwd_batch = 32
    n_total = tensors[0].shape[0]
    for start in range(0, n_total, fwd_batch):
        end = min(start + fwd_batch, n_total)
        batch_feat = {k: v[start:end] for k, v in features.items()}

        z_chunk: Optional[np.ndarray] = None

        def _tumor_hook(module: nn.Module, inp: Any,
                        out: Tuple[torch.Tensor, torch.Tensor]) -> None:
            nonlocal z_chunk
            z_chunk = out[0].detach().cpu().numpy()

        h = model.tumor_encoder.register_forward_hook(_tumor_hook)
        with torch.no_grad():
            preds, _ = model(batch_feat, return_internal_loss=False)
        h.remove()

        z_tumor_mean_parts.append(z_chunk)
        tumor_exp_parts.append(preds['tumor_exp'].cpu().numpy().squeeze(-1))

    del tensors, features
    torch.cuda.empty_cache()

    z_tumor_mean = np.concatenate(z_tumor_mean_parts, axis=0)
    tumor_pred = np.concatenate(tumor_exp_parts, axis=0)
    tumor_truth = sub['tumor_exp']
    sample_names = np.array(sub['sample_names'])

    # Derive error metrics
    residuals = tumor_pred - tumor_truth          # signed error
    abs_error = np.abs(residuals)

    # Build sample-level metadata lookup
    meta = _build_sample_metadata()

    # Map per-gene metadata from sample names
    cohorts = np.array([
        meta.loc[s, 'cohort'] if s in meta.index else 'Unknown'
        for s in sample_names
    ])
    subtypes = np.array([
        meta.loc[s, 'subtype'] if s in meta.index else 'Unknown'
        for s in sample_names
    ])
    tfx_values = np.array([
        float(meta.loc[s, 'tfx']) if s in meta.index else np.nan
        for s in sample_names
    ], dtype=np.float32)

    # UMAP projection
    reducer = umap.UMAP(
        n_components=2, random_state=seed,
        n_neighbors=30, min_dist=0.3, metric='cosine',
    )
    embedding = reducer.fit_transform(z_tumor_mean)

    # ── Helper for continuous-colour UMAPs ──
    def _plot_continuous(values: np.ndarray, cmap: str, label: str,
                         title: str, fname: str, *,
                         vcenter: Optional[float] = None) -> None:
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_aspect('equal')
        if vcenter is not None:
            vmin, vmax = np.nanpercentile(values, [2, 98])
            norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        else:
            norm = None
        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1],
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
                          palette: Optional[str] = None) -> None:
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        ax.set_aspect('equal')
        unique = sorted(set(labels))
        n_cats = len(unique)
        cmap_name = palette or ('tab10' if n_cats <= 10 else 'tab20')
        cmap_c = plt.cm.get_cmap(cmap_name, max(n_cats, 1))

        if n_cats <= max_legend:
            for i, cat in enumerate(unique):
                mask = labels == cat
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=[cmap_c(i)], s=2, alpha=0.4,
                           label=cat, rasterized=True)
        else:
            counts = Counter(labels.tolist())
            top_cats = [c for c, _ in counts.most_common(max_legend - 1)]
            cmap_c = plt.cm.get_cmap(cmap_name, max_legend)
            for i, cat in enumerate(top_cats):
                mask = labels == cat
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=[cmap_c(i)], s=2, alpha=0.4,
                           label=cat, rasterized=True)
            other_mask = ~np.isin(labels, top_cats)
            if other_mask.any():
                ax.scatter(embedding[other_mask, 0], embedding[other_mask, 1],
                           c='grey', s=1, alpha=0.2,
                           label=f'Other ({other_mask.sum():,})',
                           rasterized=True)
        ax.legend(fontsize=6, markerscale=4, ncol=1,
                  loc='center left', bbox_to_anchor=(1.02, 0.5))
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.set_title(title, fontsize=12)
        _save_or_show(fig, folder_path, fname)

    # 1. Predicted tumor expression (continuous)
    _plot_continuous(
        tumor_pred, 'viridis', 'Predicted Log\u2082(CPM+1)',
        'Tumor Latent Space — Predicted Expression',
        'umap_tumor_predicted.png',
    )

    # 2. Ground-truth tumor expression (continuous)
    _plot_continuous(
        tumor_truth, 'viridis', 'True Log\u2082(CPM+1)',
        'Tumor Latent Space — True Expression',
        'umap_tumor_truth.png',
    )

    # 3. Cancer subtype
    _plot_categorical(
        subtypes,
        'Tumor Latent Space — Cancer Subtype',
        'umap_cancer_subtype.png',
    )

    # 4. Tumor fraction (TFx)
    _plot_continuous(
        tfx_values, 'magma', 'Tumor Fraction',
        'Tumor Latent Space — TFx',
        'umap_tfx.png',
    )

    # 5. Prediction residuals (signed, centred at 0)
    _plot_continuous(
        residuals, 'RdBu_r', 'Residual (pred \u2212 truth)',
        'Tumor Latent Space — Prediction Residuals',
        'umap_residuals.png', vcenter=0.0,
    )

    # 6. Absolute error
    _plot_continuous(
        abs_error, 'hot_r', '|Prediction Error|',
        'Tumor Latent Space — Absolute Error',
        'umap_abs_error.png',
    )

    # 7. Cohort identity
    _plot_categorical(
        cohorts,
        'Tumor Latent Space — Cohort',
        'umap_cohort.png',
    )

    # 8. Sample identity (top 15 + Other)
    _plot_categorical(
        sample_names,
        'Tumor Latent Space — Sample Identity',
        'umap_tumor_samples.png',
        max_legend=16,
    )


# ═══════════════════════════════════════════════════════════════════════════
#           MIXTURE-COMPONENT (ZERO-CLASS) ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════

def zero_component_attribution(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    background_size: int = 50_000,
    test_size: int = 5_000,
    method: str = 'integrated_gradients',
    n_steps: int = 50,
    batch_size: int = 32,
    folder_path: Optional[str] = None,
    seed: int = 200,
) -> None:
    """Attribution targeting the zero-component probability.

    Uses :class:`_ZeroComponentWrapper` so that Captum attributes the
    log-probability of the hurdle (zero expression) component instead of
    the point estimate.  Analysis focuses on inactive genes where the
    zero component dominates.
    """
    device = next(model.parameters()).device

    bg_data = subsample(data, sample_size=background_size,
                        seed=seed, restriction='all_genes')
    test_data = subsample(data, sample_size=test_size,
                          seed=seed + 1, restriction='inactive_genes')

    bg_tensors = _data_to_tensors(bg_data, device)
    test_tensors = _data_to_tensors(test_data, device)

    print(f'  Zero-component attribution: {test_tensors[0].shape[0]} test genes')

    attributions = _run_attribution(
        model, test_tensors, bg_tensors,
        method=method, n_steps=n_steps, batch_size=batch_size,
        wrapper_cls=_ZeroComponentWrapper,
    )

    del bg_tensors, test_tensors
    torch.cuda.empty_cache()

    zero_folder: Optional[str] = None
    if folder_path is not None:
        zero_folder = os.path.join(folder_path, 'zero_component')
        os.makedirs(zero_folder, exist_ok=True)

    print('  Plotting zero-component signal attributions...')
    plot_signal_attributions(attributions, zero_folder)
    print('  Plotting zero-component polarity-impact...')
    plot_signal_polarity(attributions, test_data, zero_folder)
    print('  Plotting zero-component scalar summaries...')
    plot_scalar_summary(attributions, test_data, zero_folder)
    plot_scalar_bars(attributions, zero_folder)
    print('  Done: zero-component attribution')


# ═══════════════════════════════════════════════════════════════════════════
#                     ATTRIBUTION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def attribution_analysis(
    model: BioDVN,
    data: Dict[str, np.ndarray],
    background_size: int = 50_000,
    test_size: int = 5_000,
    method: str = 'integrated_gradients',
    n_steps: int = 50,
    batch_size: int = 32,
    save_filepath: Optional[str] = None,
    seed: int = 200,
    run_interaction: bool = True,
    run_reduction: bool = True,
    run_umap: bool = True,
) -> None:
    """Run attribution analysis across gene subsets.

    For each of ``'all_genes'``, ``'active_genes'``, ``'inactive_genes'``:

    1. Sub-sample background and test sets.
    2. Compute Captum attributions.
    3. Generate polarity-impact, beeswarm, and bar-chart plots.
    4. *(Optional)* Signal–sequence interaction + motif extraction.

    After the per-subset loop, optionally runs:
    - ReductionBlock activation analysis.
    - Latent-space UMAP visualization.

    Args:
        model: Trained BioDVN in eval mode.
        data: Full dataset dict from :func:`load_interpretation_data`.
        background_size: Number of genes for the attribution baseline.
        test_size: Number of genes for the test set.
        method: ``'integrated_gradients'`` or ``'gradient_shap'``.
        n_steps: Interpolation steps (IG) or samples (GradientShap).
        batch_size: Internal batch size for attribution computation.
        save_filepath: Root directory for output figures.
        seed: Random seed for subsampling.
        run_interaction: If True, run signal–sequence interaction analysis.
        run_reduction: If True, run ReductionBlock activation analysis.
        run_umap: If True, run latent-space UMAP visualization.
    """
    device = next(model.parameters()).device

    print('\n=== Attribution analysis: all_genes ===')

    folder_path = None
    if save_filepath is not None:
        folder_path = os.path.join(save_filepath, 'attribution_analysis')
        os.makedirs(folder_path, exist_ok=True)

    bg_data = subsample(data, sample_size=background_size,
                        seed=seed, restriction='all_genes')
    test_data = subsample(data, sample_size=test_size,
                          seed=seed + 1, restriction='all_genes')

    bg_tensors = _data_to_tensors(bg_data, device)
    test_tensors = _data_to_tensors(test_data, device)

    print(f'  Background: {bg_tensors[0].shape[0]} genes')
    print(f'  Test: {test_tensors[0].shape[0]} genes')
    print(f'  Method: {method} (n_steps={n_steps})')

    attributions = _run_attribution(
        model, test_tensors, bg_tensors,
        method=method, n_steps=n_steps, batch_size=batch_size,
    )

    # Free baseline tensors from GPU
    del bg_tensors, test_tensors
    torch.cuda.empty_cache()

    print('  Plotting signal attributions...')
    plot_signal_attributions(attributions, folder_path)

    print('  Plotting polarity-impact...')
    plot_signal_polarity(attributions, test_data, folder_path)

    print('  Plotting scalar summaries...')
    plot_scalar_summary(attributions, test_data, folder_path)
    plot_scalar_bars(attributions, folder_path)

    if run_interaction:
        print('  Plotting signal–sequence interaction...')
        plot_signal_sequence_interaction(attributions, folder_path)
        print('  Extracting interaction-conditioned motifs...')
        extract_interaction_motifs(attributions, test_data, folder_path)

    del attributions
    print('  Done: all_genes')

    # ── Analyses that run once ──

    analysis_root = save_filepath

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
