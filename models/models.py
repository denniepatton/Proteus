"""models.py (PyTorch)

PyTorch models for predicting gene expression from cfDNA.

Model inputs (per-gene):
    - TSS signals (biological channels)
    - TSS nucleotides (one-hot)
    - Scalar feature blocks (11 shared Triton features per region)
    - Reference scalars
                - Tumor fraction (TFX) — training label only

Labels:
    - tumor/background expression (regression)
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import (
    Sampling, MergeLatents, TrainingChoice,
    InternalLossComputer,
    MixtureDensityHead, mixture_mean,
    TFXPredictor,
    get_norm_layer, get_norm_layer_conv1d, get_norm_layer_2d,
)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Total number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, model_name: str = "Model") -> None:
    """Print a summary of the model including parameter counts.
    
    Args:
        model: PyTorch model
        model_name: Name to display in the summary
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params
    
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Model size (MB):      {total_params * 4 / 1024 / 1024:.2f}")
    print(f"{'='*60}\n")


def get_model_instance(model_name: str = 'BioDVN') -> type:
    """Get the model class for *model_name*.

    Variant names (``BioDVN_NoSeq``, ``BioDVN_OnlySeq``) are mapped to the
    unified ``BioDVN`` class — the caller should set
    ``BioDVNConfig.channel_mode`` to ``'no_seq'`` or ``'only_seq'``
    accordingly.

    Returns:
        Model **class** (not instantiated).
    """
    if model_name not in _MODEL_NAMES:
        raise ValueError(
            f"No model found with name: {model_name}. "
            f"Available: {sorted(_MODEL_NAMES)}"
        )
    return BioDVN


# Maps model_name → channel_mode for BioDVNConfig.
_MODEL_CHANNEL_MODE: Dict[str, str] = {
    'BioDVN': 'all',
    'BioDVN_NoSeq': 'no_seq',
    'BioDVN_OnlySeq': 'only_seq',
    'BioDVN_PAN-FT': 'all',
}

_MODEL_NAMES = set(_MODEL_CHANNEL_MODE)


def channel_mode_for_model(model_name: str) -> str:
    """Return the ``channel_mode`` that corresponds to *model_name*."""
    if model_name not in _MODEL_CHANNEL_MODE:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Available: {sorted(_MODEL_CHANNEL_MODE)}"
        )
    return _MODEL_CHANNEL_MODE[model_name]


@dataclass
class BioDVNConfig:
    """Configuration for BioDVN model architecture and training."""
    # Architecture hyperparameters
    channel_mode: str = 'all'  # 'all', 'no_seq', or 'only_seq'
    signal_filters: int = 16
    signal_ksize: int = 3
    interaction_ksize: int = 5
    initial_reduction_channels: int = 64
    n_reductions: int = 3
    funnel_reduction_factor: int = 4
    reduction_dilation: int = 3
    encoder_start: int = 256
    latent_size: int = 128
    decoder_start: int = 64
    dropout_rate: float = 0.25
    
    # MDN output head hyperparameters
    mdn_hidden: int = 64       # Hidden dim inside MixtureDensityHead
    mdn_final: int = 32        # Final hidden dim before 6-output projection
    
    # Loss function hyperparameters
    beta_kl: float = 1.0
    beta_contrastive: float = 1.0
    beta_reconstruction: float = 3.0
    beta_background: float = 1.0
    internal_weight: float = 0.5
    cl_margin: float = 1.0
    beta_tfx: float = 0.5
    beta_srl_max: float = 2.0
    beta_dynamic_pair_max: float = 0.0
    srl_delta: float = 5.0
    srl_start: int = 0
    srl_ramp: int = 50
    beta_pht_consistency: float = 1.0
    beta_pht_uncertainty: float = 1.0
    
    # Training hyperparameters
    weight_decay: float = 0.0
    max_lr: float = 1.5e-4  # Peak learning rate

    # Gradient clipping
    max_grad_norm: float = 10.0


class SignalProcessingBlock(nn.Module):
    """
    Signal processing block for initial feature extraction from TSS data.

    Depending on *channel_mode*, builds:
      ``'all'``      — grouped Conv1D (9 bio signals) + Conv2D (4 nucleotides)
      ``'no_seq'``   — grouped Conv1D only (9 bio signals)
      ``'only_seq'`` — Conv2D only (4 nucleotides)
    """

    _VALID_MODES = ('all', 'no_seq', 'only_seq')

    def __init__(
        self,
        channel_mode: str = 'all',
        n_signal_channels: int = 9,
        signal_filters: int = 8,
        signal_ksize: int = 6,
    ):
        super().__init__()
        if channel_mode not in self._VALID_MODES:
            raise ValueError(
                f"channel_mode must be one of {self._VALID_MODES}, "
                f"got '{channel_mode}'"
            )
        self.channel_mode = channel_mode
        self.n_signal_channels = int(n_signal_channels)
        self.signal_filters = signal_filters
        self.signal_ksize = signal_ksize

        # Grouped Conv1D for biological signal channels.
        if channel_mode in ('all', 'no_seq'):
            out_channels_1d = self.n_signal_channels * signal_filters
            self.conv1d_signals = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.n_signal_channels,
                    out_channels=out_channels_1d,
                    kernel_size=signal_ksize,
                    padding=0,
                    groups=self.n_signal_channels,
                ),
                get_norm_layer_conv1d(out_channels_1d),
                nn.ReLU(),
            )
        else:
            self.conv1d_signals = None

        # Conv2D for nucleotide channels.
        if channel_mode in ('all', 'only_seq'):
            self.conv2d_nt = nn.Sequential(
                nn.Conv2d(1, signal_filters, kernel_size=(4, signal_ksize), padding=0),
                get_norm_layer_2d(signal_filters),
                nn.ReLU(),
            )
        else:
            self.conv2d_nt = None

    @property
    def out_channels(self) -> int:
        """Number of output channels after concatenation."""
        n = 0
        if self.conv1d_signals is not None:
            n += self.n_signal_channels * self.signal_filters
        if self.conv2d_nt is not None:
            n += self.signal_filters
        return n

    def forward(
        self,
        x_signals: Optional[torch.Tensor],
        x_nucleotides: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        if self.conv1d_signals is not None and x_signals is not None:
            parts.append(self.conv1d_signals(x_signals.permute(0, 2, 1)))

        if self.conv2d_nt is not None and x_nucleotides is not None:
            x_nt_2d = x_nucleotides.permute(0, 2, 1).unsqueeze(1)
            x_nt_conv = self.conv2d_nt(x_nt_2d).squeeze(2)
            parts.append(x_nt_conv)

        if not parts:
            raise ValueError("SignalProcessingBlock received no valid input")

        x = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return x.permute(0, 2, 1)  # (batch, length, channels)


class ReductionBlock(nn.Module):
    """
    Higher-order refinement and time(spatial)-reduction block.
    
    Applies 2D convolutions to capture cross-channel relationships,
    interleaved with 1D average pooling for spatial reduction.
    Each iteration has a residual connection following standard ResNet
    design: both branches are normalized before summation, and a shared
    activation follows the addition.

    Residual path (standard post-activation residual):
        main:  Conv2D → GroupNorm          (no activation)
        skip:  Linear → LayerNorm          (no activation)
        merge: main + skip → LayerNorm → ReLU → AvgPool

    Channel schedule: initial_reduction_channels // (2 ** i) per iteration.
    """
    
    def __init__(
        self,
        in_channels: int,
        initial_reduction_channels: int = 32,
        n_reductions: int = 3,
        interaction_ksize: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.skip_norms = nn.ModuleList()
        self.post_add_norms = nn.ModuleList()
        self.pools = nn.ModuleList()
        self._residual_crop_offsets: List[int] = []  # center-crop offsets for residual alignment
        
        current_channels = in_channels
        for i in range(n_reductions):
            out_channels = max(initial_reduction_channels // (2 ** i), 1)
            
            # Main path: Conv2D (no activation — deferred to post-add)
            self.convs.append(
                nn.Conv2d(1, out_channels, kernel_size=(interaction_ksize, current_channels),
                         dilation=(dilation, 1), padding=0)
            )
            self.conv_norms.append(get_norm_layer_2d(out_channels))
            
            # Skip path: Linear projection + LayerNorm
            self.skip_projections.append(nn.Linear(current_channels, out_channels))
            self.skip_norms.append(get_norm_layer(out_channels))
            
            # Post-addition LayerNorm (normalizes the sum, prevents magnitude
            # drift under AMP float16 and across iterations).
            self.post_add_norms.append(get_norm_layer(out_channels))
            
            # Center-crop offset so the residual aligns with the conv output
            field_reduction = dilation * (interaction_ksize - 1)
            self._residual_crop_offsets.append(field_reduction // 2)
            
            self.pools.append(nn.AvgPool1d(kernel_size=2))
            current_channels = out_channels
        
        self.out_channels = current_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, length, channels)
            
        Returns:
            Reduced tensor of shape (batch, reduced_length, out_channels)
        """
        for conv, conv_norm, pool, skip_proj, skip_norm, post_norm, offset in zip(
            self.convs, self.conv_norms, self.pools,
            self.skip_projections, self.skip_norms,
            self.post_add_norms, self._residual_crop_offsets,
        ):
            # Skip path: project channels + normalize
            residual = skip_norm(skip_proj(x))  # (batch, length, out_channels)
            
            # Main path: Conv2D + normalize (no activation yet)
            x = x.unsqueeze(1)                  # (batch, 1, length, channels)
            x = conv_norm(conv(x))              # (batch, out_channels, length', 1)
            x = x.squeeze(-1).permute(0, 2, 1)  # (batch, length', out_channels)
            
            # Center-crop residual to match conv output length
            L_out = x.shape[1]
            residual = residual[:, offset : offset + L_out, :]
            
            # Residual addition → LayerNorm → ReLU
            x = F.relu(post_norm(x + residual))
            
            # Pool: (batch, length', out_channels) → (batch, length'//2, out_channels)
            x = x.permute(0, 2, 1)
            x = pool(x)
            x = x.permute(0, 2, 1)
        
        return x


class ReducedSequenceSignalEncoder(nn.Module):
    """Flatten the reduced promoter map through a configurable dense funnel.

    The reduction block preserves the full reduced promoter axis. This module
    flattens that reduced map, compresses it through a staged dense funnel,
    and emits the fixed 64-d signal summary used by the dual encoders.

    Each funnel starts with a 1024-d hidden layer. ``funnel_reduction_factor=4``
    yields ``1024 -> 256 -> 64``, while ``funnel_reduction_factor=2`` yields
    ``1024 -> 512 -> 256 -> 128 -> 64``.
    """

    def __init__(
        self,
        input_channels: int,
        input_length: int,
        output_dim: int = 64,
        dropout_rate: float = 0.25,
        funnel_reduction_factor: int = 4,
    ):
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f'input_channels must be > 0, got {input_channels}')
        if input_length <= 0:
            raise ValueError(f'input_length must be > 0, got {input_length}')
        if funnel_reduction_factor not in (2, 4):
            raise ValueError(
                'funnel_reduction_factor must be 2 or 4, '
                f'got {funnel_reduction_factor}'
            )

        self.input_channels = int(input_channels)
        self.input_length = int(input_length)
        self.output_dim = int(output_dim)
        self.funnel_reduction_factor = int(funnel_reduction_factor)
        self.flattened_dim = self.input_length * self.input_channels
        self.funnel_start_dim = 1024

        readout_layers: List[nn.Module] = [
            nn.Linear(self.flattened_dim, self.funnel_start_dim),
            get_norm_layer(self.funnel_start_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]

        current_dim = self.funnel_start_dim
        while current_dim // self.funnel_reduction_factor > self.output_dim:
            next_dim = current_dim // self.funnel_reduction_factor
            readout_layers.extend([
                nn.Linear(current_dim, next_dim),
                get_norm_layer(next_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            current_dim = next_dim

        readout_layers.append(nn.Linear(current_dim, self.output_dim))
        self.readout = nn.Sequential(*readout_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected x to have shape (batch, length, channels), got {tuple(x.shape)}')
        if int(x.shape[1]) != self.input_length or int(x.shape[2]) != self.input_channels:
            raise ValueError(
                'ReducedSequenceSignalEncoder received '
                f'length/channels {(int(x.shape[1]), int(x.shape[2]))}, '
                f'expected {(self.input_length, self.input_channels)}'
            )
        return self.readout(torch.flatten(x, start_dim=1))


class Encoder(nn.Module):
    """VAE encoder network (funnel-shaped MLP)."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        start_dim: int = 256,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        
        layers = []
        current_dim = start_dim
        
        # First layer from input to start_dim
        layers.append(nn.Linear(input_dim, current_dim))
        layers.append(get_norm_layer(current_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Halving layers until we reach latent_dim
        while current_dim > latent_dim:
            next_dim = current_dim // 2
            if next_dim < latent_dim:
                break
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(get_norm_layer(next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = next_dim
        
        self.layers = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(current_dim, latent_dim)
        self.fc_log_var = nn.Linear(current_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            z_mean: Mean of latent distribution (batch, latent_dim)
            z_log_var: Log variance of latent distribution (batch, latent_dim)
        """
        h = self.layers(x)
        # Force float32 before the mean/logvar projections.  z_mean is
        # unbounded and feeds Sampling, contrastive_loss, KL, and the MDN
        # heads — all of which assume finite input.
        #
        # IMPORTANT: h.float() alone is NOT sufficient under an enclosing
        # torch.amp.autocast context; nn.Linear is on the autocast allow list.
        # Disable autocast for the two projection layers.
        h = h.float()
        with torch.amp.autocast('cuda', enabled=False):
            z_mean = self.fc_mean(h)
            z_log_var = self.fc_log_var(h)
        # Clamp log-variance to prevent numerical overflow/underflow.
        # exp(0.5 * z_log_var) in Sampling must stay within float32;
        # exp(-z_log_var) in total_correlation must also stay finite.
        # Range [-20, 20] keeps all downstream exp() calls safe and
        # matches the clamping strategy used in MixtureDensityHead.
        z_log_var = z_log_var.clamp(-20.0, 20.0)
        return z_mean, z_log_var


class Decoder(nn.Module):
    """Decoder/predictor network (adaptive MLP that can expand or contract)."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        start_dim: int,
        dropout_rate: float = 0.25,
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        layers = []
        current_dim = start_dim
        
        # First layer
        layers.append(nn.Linear(input_dim, current_dim))
        layers.append(get_norm_layer(current_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        if start_dim > output_dim:
            # Contracting (funnel)
            while current_dim > output_dim:
                next_dim = current_dim // 2
                if next_dim < output_dim:
                    break
                layers.append(nn.Linear(current_dim, next_dim))
                layers.append(get_norm_layer(next_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                current_dim = next_dim
        else:
            # Expanding (inverted funnel)
            while current_dim < output_dim:
                next_dim = current_dim * 2
                if next_dim > output_dim:
                    break
                layers.append(nn.Linear(current_dim, next_dim))
                layers.append(get_norm_layer(next_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                current_dim = next_dim
        
        # Final layer (no batch norm or dropout)
        layers.append(nn.Linear(current_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation)
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BioDVN(nn.Module):
    """
    Biologically-informed Disentangling Variational Network (BioDVN).
    
    This model uses CNNs to process cfDNA TSS signal data before combining with
    scalar features in a VAE to disentangle latent representations of tumor and
    background components.
    
    Architecture:
    1. Signal Processing Block: Conv1D per fragmentomic channel + Conv2D for nucleotides
    2. Reduction Block: Conv2D cross-channel interaction with residual skips + AvgPool
    3. Signal Encoding: Full reduced-sequence funnel to 64 dims
    4. Dual Encoder Block: Parallel VAE encoders for tumor and background
    5. Latent Merging: Learned internally predicted TFx-conditional combination of latent spaces
    6. Reconstruction: Decoder for scalar reconstruction (internal loss)
    7. Prediction Block: MDN mixture-density regression heads
    
    Args:
        config: BioDVNConfig with architecture and training hyperparameters
    """
    
    def __init__(self, config: Optional[BioDVNConfig] = None):
        super().__init__()
        
        if config is None:
            config = BioDVNConfig()
        self.config = config
        self._use_sequence_only_objective = config.channel_mode == 'only_seq'
        
        # Signal processing block — channel_mode controls which
        # convolutions are built (bio signals, nucleotides, or both).
        n_signal_channels = 9
        self.signal_processor = SignalProcessingBlock(
            channel_mode=config.channel_mode,
            n_signal_channels=n_signal_channels,
            signal_filters=config.signal_filters,
            signal_ksize=config.signal_ksize,
        )
        
        signal_channels = self.signal_processor.out_channels
        
        # Spatial reduction block: Conv2D cross-channel interaction with
        # residual skip connections, followed by AvgPool.  Skip connections
        # preserve fine-grained features through coarser layers, naturally
        # integrating local and broader spatial scales in a single path.
        self.reduction = ReductionBlock(
            in_channels=signal_channels,
            initial_reduction_channels=config.initial_reduction_channels,
            n_reductions=config.n_reductions,
            interaction_ksize=config.interaction_ksize,
            dilation=config.reduction_dilation,
        )
        
        self.scalar_input_dim = 0

        # only_seq excludes all scalar covariates so the ablation is a true
        # sequence-only path.
        if config.channel_mode != 'only_seq':
            self.scalar_input_dim += 11  # TSS scalar features
            self.scalar_input_dim += 11  # GB features
            self.scalar_input_dim += 3   # Reference scalars

        # Signal encoding keeps the full reduced sequence and flattens it
        # through a configurable dense funnel to a 64-d summary.
        self._tss_input_length = 2000
        self.signal_encoded_dim = 64
        reduction_input_length = self._tss_input_length - int(config.signal_ksize) + 1
        if reduction_input_length <= 0:
            raise ValueError(
                f'signal_ksize={int(config.signal_ksize)} is too large for '
                f'TSS length {self._tss_input_length}'
            )
        reduction_out_channels = self.reduction.out_channels
        reduction_output_length = self._compute_reduction_output_length(
            reduction_input_length,
            interaction_ksize=config.interaction_ksize,
            dilation=config.reduction_dilation,
            n_reductions=config.n_reductions,
        )
        if reduction_output_length <= 0:
            raise ValueError(
                'Reduction configuration collapses the TSS sequence: '
                f'input_length={reduction_input_length}, '
                f'interaction_ksize={int(config.interaction_ksize)}, '
                f'dilation={int(config.reduction_dilation)}, '
                f'n_reductions={int(config.n_reductions)}'
            )
        self.signal_encoder = ReducedSequenceSignalEncoder(
            input_channels=reduction_out_channels,
            input_length=reduction_output_length,
            output_dim=self.signal_encoded_dim,
            dropout_rate=config.dropout_rate,
            funnel_reduction_factor=config.funnel_reduction_factor,
        )

        self.encoder_input_dim = int(self.signal_encoded_dim + self.scalar_input_dim)
        
        # Dual encoders
        self.tumor_encoder = Encoder(
            input_dim=self.encoder_input_dim,
            latent_dim=config.latent_size,
            start_dim=config.encoder_start,
            dropout_rate=config.dropout_rate,
        )
        self.background_encoder = Encoder(
            input_dim=self.encoder_input_dim,
            latent_dim=config.latent_size,
            start_dim=config.encoder_start,
            dropout_rate=config.dropout_rate,
        )
        
        # Sampling and merging
        self.sampling = Sampling()
        self.merge_latents = MergeLatents()
        self.training_choice = TrainingChoice()
        
        self.tfx_predictor = TFXPredictor(latent_size=config.latent_size)

        # Weight prediction for merging: TFX (predicted or raw) + latent stats.
        # Small nonlinear MLP gives capacity to learn non-trivial
        # TFX-to-weight mappings beyond a linear decision boundary.
        merge_input_dim = 1 + 4 * config.latent_size  # TFX + 2*(mean + log_var)
        self.weight_predictor = nn.Sequential(
            nn.Linear(merge_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )
        
        # Decoder for reconstruction
        self.decoder = Decoder(
            input_dim=config.latent_size,
            output_dim=self.encoder_input_dim,
            start_dim=config.decoder_start,
            dropout_rate=0.0,
        )
        
        # Mixture Density Network output heads.
        # Input is the reparameterized latent sample z (same representation
        # the decoder sees), so the MDN's full input bandwidth is devoted
        # to the biological signal rather than raw encoder uncertainty.
        mdn_input_dim = config.latent_size
        self.tumor_mdn_head = MixtureDensityHead(
            input_dim=mdn_input_dim,
            hidden_dim=config.mdn_hidden,
            final_dim=config.mdn_final,
        )
        self.background_mdn_head = MixtureDensityHead(
            input_dim=mdn_input_dim,
            hidden_dim=config.mdn_hidden,
            final_dim=config.mdn_final,
        )
        
        # Internal loss computer
        self.internal_loss_fn = InternalLossComputer(
            beta_contrastive=config.beta_contrastive,
            beta_kl=config.beta_kl,
            beta_reconstruction=config.beta_reconstruction,
            beta_tfx=config.beta_tfx,
            cl_margin=config.cl_margin,
            internal_weight=config.internal_weight,
        )
        
        # Apply TensorFlow-compatible weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights to match TensorFlow/Keras defaults for parity with v1.
        
        TensorFlow Dense uses Glorot uniform (Xavier uniform) by default.
        TensorFlow Conv uses Glorot uniform by default.
        LayerNorm/GroupNorm defaults (weight=1, bias=0) are already correct.
        """
        if isinstance(module, nn.Linear):
            # Glorot/Xavier uniform initialization (TensorFlow default for Dense)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            # Glorot/Xavier uniform for convolutions (TensorFlow default)
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # PyTorch defaults (weight=1, bias=0) are already correct.
            pass

    @staticmethod
    def _compute_reduction_output_length(
        length: int,
        interaction_ksize: int,
        dilation: int,
        n_reductions: int,
    ) -> int:
        """Compute the output length of ReductionBlock.forward for a given input length."""
        out = int(length)
        for _ in range(int(n_reductions)):
            out = out - int(dilation) * (int(interaction_ksize) - 1)
            out = out // 2  # AvgPool1d(kernel_size=2, stride=2)
        return out

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        return_internal_loss: bool = True,
        tfx: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Tuple[torch.Tensor, Dict]]]:
        """
        Forward pass through BioDVN.
        
        Args:
            features: Dictionary with keys:
                - TSS_signals: (batch, 2000, 9) biological signals + NaN mask
                - TSS_nucleotides: (batch, 2000, 4) one-hot nucleotides
                - TSS_features: (batch, 11+) TSS region features (first 11 used)
                - GB_features: (batch, 11) gene body features
                - ref_scalars: (batch, 3) reference gene info
            return_internal_loss: Whether to compute and return internal loss
            tfx: (batch, 1) ground-truth tumor fraction.  Required during
                training for auxiliary TFx supervision; ignored at inference
                because TFx is predicted internally from the latents.
            
        Returns:
            predictions: Dictionary with keys:
                - tumor_mdn: dict of mixture params from MixtureDensityHead
                - background_mdn: dict of mixture params from MixtureDensityHead
                - tumor_exp: (batch, 1) mixture mean point estimate
                - background_exp: (batch, 1) mixture mean point estimate
            internal_loss: Tuple of (loss_tensor, metrics_dict) if return_internal_loss=True
        """
        # Extract features based on channel_mode.
        x_signals: Optional[torch.Tensor] = None
        x_nucleotides: Optional[torch.Tensor] = None

        if self.config.channel_mode in ('all', 'no_seq'):
            x_signals = features['TSS_signals']
            if int(x_signals.shape[-1]) < 9:
                raise ValueError(
                    f"Expected at least 9 biological signal channels, "
                    f"got {int(x_signals.shape[-1])}"
                )
            x_signals = x_signals[:, :, :9]

        if self.config.channel_mode in ('all', 'only_seq'):
            x_nucleotides = features['TSS_nucleotides']

        # Signal processing
        x = self.signal_processor(x_signals, x_nucleotides)
        
        # Spatial reduction (skip connections integrate multi-scale features)
        x = self.reduction(x)
        
        scalar_inputs: List[torch.Tensor] = []
        if self.config.channel_mode != 'only_seq':
            scalar_inputs.append(features['TSS_features'][:, :11])
            scalar_inputs.append(features['GB_features'])
            scalar_inputs.append(features['ref_scalars'])

        # Signal encoding operates directly on the reduced (length, channel)
        # map so the full promoter axis is preserved through the dense funnel.
        x_encoded = self.signal_encoder(x)
        
        # Combine with scalar features.
        # TFX is never included as an input.  A supervised internal TFx
        # prediction conditions the merge weights during training/inference.
        encoder_inputs: List[torch.Tensor] = [x_encoded]
        if scalar_inputs:
            encoder_inputs.extend(scalar_inputs)
        y = torch.cat(encoder_inputs, dim=1) if len(encoder_inputs) > 1 else encoder_inputs[0]
        
        # Dual encoding
        z_tumor_mean, z_tumor_log_var = self.tumor_encoder(y)
        z_background_mean, z_background_log_var = self.background_encoder(y)
        
        # Training uses sampled latents; inference uses the means so
        # repeated forward passes are deterministic.
        z_tumor_sample = self.sampling(z_tumor_mean, z_tumor_log_var)
        z_background_sample = self.sampling(z_background_mean, z_background_log_var)
        z_tumor = self.training_choice(z_tumor_sample, z_tumor_mean)
        z_background = self.training_choice(z_background_sample, z_background_mean)
        
        predicted_tfx = z_tumor_mean.new_zeros((z_tumor_mean.shape[0], 1))
        tfx_loss = None
        y_reconstructed = None

        if not self._use_sequence_only_objective:
            predicted_tfx = self.tfx_predictor(z_tumor_mean, z_background_mean)

            if self.training:
                if tfx is None:
                    raise ValueError(
                        "tfx (ground-truth tumor fraction) must be provided "
                        "during training for auxiliary TFx supervision."
                    )
                predicted_tfx_flat = predicted_tfx.float().view(-1)
                tfx_flat = tfx.float().view(-1)
                finite_tfx = torch.isfinite(tfx_flat)
                if finite_tfx.any():
                    tfx_loss = F.mse_loss(
                        predicted_tfx_flat[finite_tfx],
                        tfx_flat[finite_tfx],
                    )
                else:
                    tfx_loss = predicted_tfx_flat.new_tensor(0.0)

            merge_input = torch.cat([
                predicted_tfx, z_tumor_mean, z_tumor_log_var,
                z_background_mean, z_background_log_var,
            ], dim=1)
            weights = self.weight_predictor(merge_input)

            z_merged_mean, z_merged_log_var = self.merge_latents(
                weights, z_tumor_mean, z_tumor_log_var, z_background_mean, z_background_log_var
            )
            z_merged_sample = self.sampling(z_merged_mean, z_merged_log_var)
            z_merged = self.training_choice(z_merged_sample, z_merged_mean)

            # Sequence-only can only learn a gene-conditioned cohort mean, so
            # sample-level TFx supervision and merged-latent reconstruction are
            # intentionally disabled for that ablation.
            y_reconstructed = self.decoder(z_merged)
        
        # MDN output heads follow the same training/inference choice as the
        # decoder path: samples during training, means during inference.
        tumor_mdn = self.tumor_mdn_head(z_tumor)
        background_mdn = self.background_mdn_head(z_background)
        
        # Point estimates (mixture mean) for backward-compatible metrics / analysis
        tumor_exp = mixture_mean(tumor_mdn)
        background_exp = mixture_mean(background_mdn)
        
        predictions = {
            'tumor_mdn': tumor_mdn,
            'background_mdn': background_mdn,
            'tumor_exp': tumor_exp,
            'background_exp': background_exp,
            'predicted_tfx': predicted_tfx,
        }
        
        # Internal loss
        if return_internal_loss and self.training and not self._use_sequence_only_objective:
            internal_loss, internal_metrics = self.internal_loss_fn(
                z_tumor, z_background,
                z_tumor_mean, z_tumor_log_var,
                z_background_mean, z_background_log_var,
                y, y_reconstructed,
                tfx_loss=tfx_loss,
            )
            return predictions, (internal_loss, internal_metrics)
        
        return predictions, None
