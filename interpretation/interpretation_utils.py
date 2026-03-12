"""
interpretation_utils.py
v2.0.0
Author: Alexander Netzley, anetzley@fredhutch.org
Updates: Robert Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

Interpretability utilities for the PyTorch BioDVN model.

Provides:
    - Model & data loading (PyTorch checkpoints, Zarr stores)
    - Kernel visualization (Conv1d signals, Conv2d nucleotides, reduction filters)
    - PWM / motif analysis from learned nucleotide convolutional filters
    - Attribution analysis via Captum (IntegratedGradients, GradientShap)
    - Polarity-impact and summary attribution plots
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import logomaker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---------------------------------------------------------------------------
# Deferred Captum import — only loaded when attribution functions are called,
# so the rest of the module works without Captum installed.
# ---------------------------------------------------------------------------
_captum_available: Optional[bool] = None


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

SIGNAL_DISPLAY_NAMES: List[str] = [
    'GC-Corrected Depth',          # 0
    'Fragment End Coverage',        # 1
    'Fragment End Asymmetry',       # 2
    'Phased-Nucleosome Signal',     # 3
    'Fragment Short:Long Ratio',    # 4
    'Fragment Length Entropy',      # 5
    'Fragment Length Diversity',    # 6
    'Peak Calls',                   # 7
]

NUCLEOTIDE_NAMES: List[str] = ['A', 'C', 'G', 'T']

# Combined signal + nucleotide labels for 9-row heatmaps.
SIGNAL_AND_SEQ_DISPLAY_NAMES: List[str] = SIGNAL_DISPLAY_NAMES + ['Reference Sequence']

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
    state = torch.load(weights_path, map_location=device)
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

def load_interpretation_data(
    source: Union[str, List[str]],
    max_genes_per_sample: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load data from Zarr stores into a dict of NumPy arrays.

    Accepted *source* formats:
        - A path to a fold-index JSON (list of Zarr sample paths).
        - A direct list of ``*.zarr`` paths.
        - A single ``*.zarr`` path (string ending with ``.zarr``).

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

    all_arrays: Dict[str, list] = {
        'TSS_signals': [], 'TSS_nucleotides': [], 'TSS_features': [],
        'GB_features': [], 'ref_scalars': [], 'TFX_label': [], 'tumor_exp': [],
    }
    all_gene_names: List[str] = []
    all_sample_names: List[str] = []

    for zpath in zarr_paths:
        store = zarr.open(zpath, mode='r')
        flat = store['flat'] if 'flat' in store else store

        n_genes = int(flat['TSS_signals'].shape[0])
        if max_genes_per_sample is not None and n_genes > max_genes_per_sample:
            n_genes = max_genes_per_sample

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

    data = {k: np.concatenate(v, axis=0) for k, v in all_arrays.items()}
    data['gene_names'] = all_gene_names
    data['sample_names'] = all_sample_names
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
    rng = np.random.RandomState(seed)

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
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    for k in ('gene_names', 'sample_names'):
        if k in data and isinstance(data[k], list):
            arrays[k] = np.array(data[k], dtype=object)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    np.savez_compressed(filepath, **arrays)


def load_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Load a ``.npz`` file saved by :func:`save_npz`."""
    npz = np.load(filepath, allow_pickle=True)
    out: Dict[str, np.ndarray] = {}
    for key in npz:
        arr = npz[key]
        if arr.dtype == object:
            out[key] = list(arr)
        else:
            out[key] = arr
    return out


def get_data(
    source: Union[str, List[str]],
    regenerate_data: bool = False,
    max_genes_per_sample: Optional[int] = None,
    save_filepath: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """High-level data loader with optional caching.

    If *save_filepath* points to an existing ``.npz`` and *regenerate_data*
    is ``False``, the cached file is loaded.  Otherwise data is loaded fresh
    from Zarr and (optionally) saved to *save_filepath*.
    """
    if save_filepath and os.path.isfile(save_filepath) and not regenerate_data:
        print(f'Loading cached data: {save_filepath}')
        return load_npz(save_filepath)

    print(f'Loading data from source: {source}')
    data = load_interpretation_data(source, max_genes_per_sample=max_genes_per_sample)

    if save_filepath:
        save_npz(data, save_filepath)
        print(f'Saved interpretation data: {save_filepath}')

    return data



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
    """

    def __init__(self, model: BioDVN) -> None:
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


# ═══════════════════════════════════════════════════════════════════════════
#                      PLOT HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _save_or_show(fig: plt.Figure, folder_path: Optional[str], filename: str) -> None:
    """Save figure to *folder_path/filename* or show interactively."""
    if folder_path is not None:
        os.makedirs(folder_path, exist_ok=True)
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
    """Plot the 9 per-channel Conv1d first-layer signal filters.

    Each signal channel has its own ``Conv1d(1, signal_filters, ksize)`` inside
    the :class:`SignalProcessingBlock`.  This visualizes the learned kernels.
    """
    sp = model.signal_processor
    n_channels = len(sp.signal_convs)
    signal_filters = model.config.signal_filters
    ksize = model.config.signal_ksize

    fig, axs = plt.subplots(n_channels, signal_filters,
                            figsize=(2 * signal_filters, 2 * n_channels))
    if signal_filters == 1:
        axs = axs[:, np.newaxis]
    if n_channels == 1:
        axs = axs[np.newaxis, :]

    all_weights = []
    for ch_idx in range(n_channels):
        w = _get_conv_weights(sp.signal_convs[ch_idx])  # (out, 1, ksize)
        all_weights.append(w)

    vmin = min(w.min() for w in all_weights)
    vmax = max(w.max() for w in all_weights)

    for ch_idx in range(n_channels):
        w = all_weights[ch_idx]
        for f_idx in range(signal_filters):
            ax = axs[ch_idx, f_idx]
            kernel = w[f_idx, 0, :]  # (ksize,)
            ax.bar(range(ksize), kernel, color='steelblue')
            ax.set_ylim(vmin * 1.1, vmax * 1.1)
            if f_idx == 0:
                ax.set_ylabel(SIGNAL_DISPLAY_NAMES[ch_idx], fontsize=7,
                              rotation=45, ha='right')
            else:
                ax.set_yticks([])
            if ch_idx == 0:
                ax.set_title(f'F{f_idx + 1}', fontsize=8)
            ax.set_xticks([])

    fig.suptitle(
        f'Signal Conv1d Filters ({signal_filters} per channel, kernel={ksize})',
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig, folder_path, 'signal_conv1d_filters.png')


def plot_nucleotide_conv2d_filters(model: BioDVN, folder_path: Optional[str] = None) -> None:
    """Plot the Conv2d nucleotide filters from the SignalProcessingBlock.

    The nucleotide convolution operates on a ``(2000, 4)`` one-hot input with
    shape ``(1, 4, ksize)`` kernels, producing *signal_filters* output channels.
    """
    sp = model.signal_processor
    w = _get_conv_weights(sp.nuc_conv)  # (out_ch, in_ch=1, H=4, W=ksize)
    num_filters = w.shape[0]
    ksize = w.shape[3]

    cols = min(8, num_filters)
    rows = int(np.ceil(num_filters / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
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
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    _save_or_show(fig, folder_path, 'nucleotide_conv2d_filters.png')


def plot_reduction_filters(model: BioDVN, folder_path: Optional[str] = None) -> None:
    """Plot Conv1d interaction kernels from both ReductionBlocks (small & large).

    Only the first-layer interaction convolution of each reduction path is
    plotted as a (in_channels × kernel_size) heatmap per output channel.
    """
    for label, block in [('small', model.reduction_small),
                         ('large', model.reduction_large)]:
        first_conv = None
        for child in block.blocks[0].modules():
            if isinstance(child, nn.Conv1d):
                first_conv = child
                break

        if first_conv is None:
            continue

        w = _get_conv_weights(first_conv)  # (out_ch, in_ch, ksize)
        out_ch, in_ch, ksize = w.shape
        cols = min(8, out_ch)
        rows = int(np.ceil(out_ch / cols))

        fig, axs = plt.subplots(rows, cols,
                                figsize=(2 * cols, max(3, in_ch * 0.15) * rows))
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
            im = ax.imshow(w[idx], cmap='viridis', aspect='auto', interpolation='none')
            ax.set_title(f'F{idx + 1}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(out_ch, rows * cols):
            r, c = divmod(idx, cols)
            axs[r, c].axis('off')

        dilation = getattr(first_conv, 'dilation', (1,))[0]
        fig.suptitle(
            f'Reduction-{label} Conv1d filters '
            f'({out_ch} filters, kernel={ksize}, dilation={dilation})',
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        _save_or_show(fig, folder_path, f'reduction_{label}_conv1d_filters.png')




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

    For each Conv2d filter, sliding windows of nucleotide one-hot data are
    scored; windows with positive activations are collected to build a PWM.

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
    w = _get_conv_weights(sp.nuc_conv)  # (num_filters, 1, 4, ksize)
    num_filters = w.shape[0]
    ksize = w.shape[3]

    nucleotides_all = data['TSS_nucleotides']  # (N, 2000, 4)
    n_use = min(n_samples, nucleotides_all.shape[0])

    # Background nucleotide frequencies
    bg_freq = nucleotides_all[:n_use].mean(axis=(0, 1))  # (4,)

    pwm_accum = np.zeros((ksize, 4, num_filters, n_use))

    for s_idx in range(n_use):
        nuc = nucleotides_all[s_idx]  # (2000, 4)

        for f_idx in range(num_filters):
            kernel = w[f_idx, 0, :, :]  # (4, ksize)
            activated = np.zeros((ksize, 4))
            n_positive = 0
            for pos in range(nuc.shape[0] - ksize + 1):
                window = nuc[pos:pos + ksize, :].T  # (4, ksize)
                activation = np.sum(window * kernel)
                if activation > 0:
                    activated += window.T  # (ksize, 4)
                    n_positive += 1
            if n_positive > 0:
                activated /= n_positive
            pwm_accum[:, :, f_idx, s_idx] = activated

    pwm_filters = pwm_accum.mean(axis=3)  # (ksize, 4, num_filters)

    if save_filepath is not None:
        cols = min(8, num_filters)
        rows = int(np.ceil(num_filters / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1:
            axs = np.array(axs).reshape(1, -1)
        for idx in range(num_filters):
            r, c = divmod(idx, cols)
            ax = axs[r, c]
            ax.imshow(pwm_filters[:, :, idx].T, cmap='viridis', aspect='auto')
            ax.set_yticks(range(4))
            ax.set_yticklabels(NUCLEOTIDE_NAMES)
            ax.set_title(f'Filter {idx + 1}', fontsize=9)
        for idx in range(num_filters, rows * cols):
            r, c = divmod(idx, cols)
            axs[r, c].axis('off')
        fig.suptitle(f'Nucleotide PWM ({num_filters} filters)', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        _save_or_show(fig, save_filepath, 'nucleotide_motifs_PWM.png')

    return pwm_filters, bg_freq


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

    plot_signal_conv1d_filters(model, folder_path)
    plot_nucleotide_conv2d_filters(model, folder_path)
    plot_reduction_filters(model, folder_path)
    plot_motifs(model, data, folder_path)



# ═══════════════════════════════════════════════════════════════════════════
#                  ATTRIBUTION ANALYSIS (CAPTUM)
# ═══════════════════════════════════════════════════════════════════════════

def _run_attribution(
    model: BioDVN,
    test_tensors: Tuple[torch.Tensor, ...],
    baseline_tensors: Tuple[torch.Tensor, ...],
    method: str = 'integrated_gradients',
    n_steps: int = 50,
    batch_size: int = 256,
) -> List[np.ndarray]:
    """Run Captum attribution and return per-input-group numpy arrays.

    Args:
        model: Trained BioDVN in eval mode.
        test_tensors: 5-tuple of test input tensors.
        baseline_tensors: 5-tuple of baseline (background) tensors.
        method: ``'integrated_gradients'`` or ``'gradient_shap'``.
        n_steps: Number of interpolation steps (IntegratedGradients) or
            samples (GradientShap).
        batch_size: Internal batch size for attribution computation.

    Returns:
        List of 5 numpy arrays (one per input group) with the same shape as
        the corresponding test tensors.
    """
    _ensure_captum()
    from captum.attr import IntegratedGradients, GradientShap

    wrapper = _TumorMeanWrapper(model)
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

        print(f'  Attribution batch {start}–{end} / {n_test}')

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
    attribution for each of the 9 signal channels, plus a summed nucleotide
    channel (total of 10 rows).
    """
    sig_attr = attributions[0]    # (N, 2000, 9)
    nuc_attr = attributions[1]    # (N, 2000, 4)

    # Sum nucleotide attributions to a single channel
    nuc_summed = np.abs(nuc_attr).sum(axis=-1, keepdims=True)   # (N, 2000, 1)
    combined = np.concatenate([np.abs(sig_attr), nuc_summed], axis=-1)  # (N, 2000, 10)

    # Mean across samples, transpose to (10, 2000)
    means = combined.mean(axis=0).T

    labels = SIGNAL_AND_SEQ_DISPLAY_NAMES

    fig, axs = plt.subplots(len(labels), 1,
                            figsize=(12, 0.8 * len(labels)),
                            gridspec_kw={'hspace': 0.1})
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
    """Plot polarity-impact scores for signals.

    Polarity captures whether higher signal values drive the prediction
    *up* (red) or *down* (blue).  Magnitude is the median absolute
    attribution.  The result is a per-position value in ``[-1, +1]``.
    """
    sig_attr = attributions[0]   # (N, 2000, 9)
    sig_vals = data['TSS_signals']  # (N, 2000, 9)

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
                            gridspec_kw={'hspace': 0.1})
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for ch in range(n_channels):
        im = axs[ch].imshow(
            impact_polarity[ch:ch + 1], cmap=CUSTOM_CMAP,
            aspect='auto', interpolation='nearest', norm=norm,
        )
        axs[ch].set_yticks([])
        axs[ch].set_ylabel(
            SIGNAL_DISPLAY_NAMES[ch], rotation=0,
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


def plot_scalar_summary(
    attributions: List[np.ndarray],
    data: Dict[str, np.ndarray],
    folder_path: Optional[str] = None,
) -> None:
    """Plot beeswarm-style summary plots for scalar feature attributions.

    Each dot represents one gene.  X-axis is the attribution value; color
    is the feature value (blue=low, red=high).
    """
    groups = [
        ('TSS_features',  attributions[2], data['TSS_features'],  TSS_FEATURE_NAMES),
        ('GB_features',   attributions[3], data['GB_features'],   GB_FEATURE_NAMES),
        ('ref_scalars',   attributions[4], data['ref_scalars'],   REF_SCALAR_NAMES),
    ]

    for group_name, attr_vals, feat_vals, feat_names in groups:
        n_features = attr_vals.shape[1]
        mean_abs = np.abs(attr_vals).mean(axis=0)
        order = np.argsort(mean_abs)  # ascending

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * n_features)))
        rng = np.random.default_rng(42)

        for row_idx, feat_idx in enumerate(order):
            vals = attr_vals[:, feat_idx]
            feat = feat_vals[:, feat_idx]
            y = np.full_like(vals, row_idx) + rng.uniform(-0.3, 0.3, size=vals.shape)
            fmin, fmax = feat.min(), feat.max()
            if fmax > fmin:
                colours = (feat - fmin) / (fmax - fmin)
            else:
                colours = np.full_like(feat, 0.5)
            ax.scatter(vals, y, c=colours, cmap=CUSTOM_CMAP,
                       s=3, alpha=0.5, rasterized=True)

        ax.set_yticks(range(n_features))
        ax.set_yticklabels([feat_names[i] for i in order], fontsize=8)
        ax.set_xlabel('Attribution (impact on mixture mean)', fontsize=9)
        ax.set_title(f'Attribution Beeswarm — {group_name}', fontsize=12)
        ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
        plt.tight_layout()
        _save_or_show(fig, folder_path, f'attribution_beeswarm_{group_name}.png')


def plot_scalar_bars(
    attributions: List[np.ndarray],
    folder_path: Optional[str] = None,
) -> None:
    """Plot grouped bar charts of mean |attribution| for all scalar features."""
    groups = [
        ('TSS Features', attributions[2], TSS_FEATURE_NAMES),
        ('GB Features',  attributions[3], GB_FEATURE_NAMES),
        ('Ref Scalars',  attributions[4], REF_SCALAR_NAMES),
    ]

    fig, axs = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5))
    for ax, (title, attr_vals, names) in zip(axs, groups):
        mean_abs = np.abs(attr_vals).mean(axis=0)
        order = np.argsort(mean_abs)
        ax.barh(range(len(names)), mean_abs[order], color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([names[i] for i in order], fontsize=8)
        ax.set_xlabel('Mean |Attribution|', fontsize=9)
        ax.set_title(title, fontsize=11)

    fig.suptitle('Scalar Feature Importance (Mean |Attribution|)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig, folder_path, 'scalar_feature_importance.png')


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
    batch_size: int = 256,
    save_filepath: Optional[str] = None,
    seed: int = 200,
) -> None:
    """Run attribution analysis across gene subsets.

    For each of ``'all_genes'``, ``'active_genes'``, ``'inactive_genes'``:

    1. Sub-sample background and test sets.
    2. Compute Captum attributions.
    3. Generate polarity-impact, beeswarm, and bar-chart plots.

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
    """
    device = next(model.parameters()).device

    for analysis in ['all_genes', 'active_genes', 'inactive_genes']:
        print(f'\n=== Attribution analysis: {analysis} ===')

        folder_path = None
        if save_filepath is not None:
            folder_path = os.path.join(save_filepath, 'attribution_analysis', analysis)
            os.makedirs(folder_path, exist_ok=True)

        bg_data = subsample(data, sample_size=background_size,
                            seed=seed, restriction=analysis)
        test_data = subsample(data, sample_size=test_size,
                              seed=seed + 1, restriction=analysis)

        bg_tensors = _data_to_tensors(bg_data, device)
        test_tensors = _data_to_tensors(test_data, device)

        print(f'  Background: {bg_tensors[0].shape[0]} genes')
        print(f'  Test: {test_tensors[0].shape[0]} genes')
        print(f'  Method: {method} (n_steps={n_steps})')

        attributions = _run_attribution(
            model, test_tensors, bg_tensors,
            method=method, n_steps=n_steps, batch_size=batch_size,
        )

        print('  Plotting signal attributions...')
        plot_signal_attributions(attributions, folder_path)

        print('  Plotting polarity-impact...')
        plot_signal_polarity(attributions, test_data, folder_path)

        print('  Plotting scalar summaries...')
        plot_scalar_summary(attributions, test_data, folder_path)
        plot_scalar_bars(attributions, folder_path)

        print(f'  Done: {analysis}')
