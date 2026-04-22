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
    TFXDiscriminator,
    get_norm_layer, get_norm_layer_conv1d, get_norm_layer_2d,
    Y_MAX, K_GAMMA,
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
    signal_ksize: int = 6
    interaction_ksize: int = 3
    initial_reduction_channels: int = 64
    n_reductions: int = 3
    reduction_dilation: int = 3
    encoder_start: int = 128
    latent_size: int = 64
    decoder_start: int = 64
    dropout_rate: float = 0.20
    
    # MDN output head hyperparameters
    mdn_hidden: int = 64       # Hidden dim inside MixtureDensityHead
    mdn_final: int = 32        # Final hidden dim before 6-output projection
    
    # Loss function hyperparameters
    beta_tcvae: float = 2.0    # β-TCVAE TC weight: loss = KL + (beta_tcvae − 1) × TC
    beta_contrastive: float = 1.0
    beta_reconstruction: float = 0.50
    beta_background: float = 2.0
    internal_weight: float = 1.0
    cl_margin: float = 1.0
    beta_grl: float = 0.15
    beta_point_mse: float = 0.30  # Auxiliary MSE on mixture_mean; 0 = off
    
    # Training hyperparameters
    weight_decay: float = 1e-5
    max_lr: float = 0.0015  # Peak learning rate

    # Gradient clipping
    max_grad_norm: float = 18.0


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
        signal_filters: int = 16,
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
        initial_reduction_channels: int = 64,
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


class SpatialSignalEncoder(nn.Module):
    """Compact position-aware encoder over the reduced TSS representation.

    The old encoder globally averaged the reduced spatial axis, making the
    final signal summary translation-invariant over the whole 2 kb window.
    This encoder keeps a small number of ordered spatial bins, then learns a
    compact projection from those coarse positional summaries.
    """

    def __init__(
        self,
        input_channels: int,
        output_dim: int = 64,
        spatial_bins: int = 16,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if spatial_bins <= 0:
            raise ValueError(f'spatial_bins must be > 0, got {spatial_bins}')
        self.spatial_bins = int(spatial_bins)
        self.output_dim = int(output_dim)
        hidden_dim = max(self.output_dim * 2, int(input_channels) * 4)

        self.spatial_pool = nn.AdaptiveAvgPool1d(self.spatial_bins)
        self.projection = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(int(input_channels) * self.spatial_bins, hidden_dim),
            get_norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, self.output_dim),
            get_norm_layer(self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_pool(x)
        return self.projection(x)


class Encoder(nn.Module):
    """VAE encoder network (funnel-shaped MLP)."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        start_dim: int = 128,
        dropout_rate: float = 0.15,
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
        # Force float32 before the mean/logvar projections.  Under AMP the
        # upstream MLP runs in float16; if activations drift large, the
        # Linear output can overflow float16 (max ~65504).  z_mean is
        # unbounded and feeds Sampling, contrastive_loss, KL, and the MDN
        # heads — all of which assume finite input.
        #
        # IMPORTANT: h.float() alone is NOT sufficient — under an enclosing
        # torch.amp.autocast context, nn.Linear is on the "allow" list and
        # silently recasts the float32 input back to float16.  We must
        # explicitly disable autocast for the two projection layers.
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
        dropout_rate: float = 0.0,
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
    3. Signal Encoding: Adaptive coarse spatial binning + compact MLP projection
    4. Dual Encoder Block: Parallel VAE encoders for tumor and background
    5. Latent Merging: Learned TFx-conditional combination of latent spaces
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
        
        # Signal encoding: preserve coarse spatial structure after reduction.
        # We adaptively pool the reduced representation into a small number of
        # ordered bins, then project those bin summaries with a compact MLP.
        # This keeps position information at a coarse scale without reviving
        # the very large flattened encoder used historically.
        self._tss_input_length = 2000
        self.signal_encoded_dim = 64  # Signal encoder output dim
        self.signal_spatial_bins = 16
        reduction_out_channels = self.reduction.out_channels
        self.signal_encoder = SpatialSignalEncoder(
            input_channels=reduction_out_channels,
            output_dim=self.signal_encoded_dim,
            spatial_bins=self.signal_spatial_bins,
            dropout_rate=config.dropout_rate,
        )
        self.scalar_input_dim = 0

        # only_seq excludes sample-derived scalar features (TSS/GB)
        # to ensure a true sequence-only ablation.
        if config.channel_mode != 'only_seq':
            self.scalar_input_dim += 11  # TSS scalar features
            self.scalar_input_dim += 11  # GB features

        # Reference scalars always included.
        self.scalar_input_dim += 3

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
        
        # TFX discriminator with gradient reversal for TFX-invariant latents.
        # The discriminator predicts TFX from the latents; the predicted TFX
        # feeds the merge weight predictor.  The gradient reversal layer
        # pushes the encoders toward TFX-invariant representations.
        self.tfx_discriminator = TFXDiscriminator(
            latent_size=config.latent_size, lam=1.0,
        )

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
            beta_tcvae=config.beta_tcvae,
            beta_reconstruction=config.beta_reconstruction,
            beta_grl=config.beta_grl,
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
                training for the GRL discriminator loss; ignored at inference
                (the discriminator predicts TFX from the latents).
            
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

        ref_scalars = features['ref_scalars']
        
        # Signal processing
        x = self.signal_processor(x_signals, x_nucleotides)
        
        # Spatial reduction (skip connections integrate multi-scale features)
        x = self.reduction(x)
        
        # Signal encoding: permute to (batch, channels, length), keep a small
        # number of ordered spatial bins, then project to the fixed-width
        # signal summary used by the downstream encoders.
        x = x.permute(0, 2, 1)
        x_encoded = self.signal_encoder(x)
        
        # Combine with scalar features.
        # TFX is never included — the encoder learns TFX-invariant
        # representations.  TFX enters only through the merge weight
        # predictor (via GRL discriminator prediction).
        encoder_inputs: List[torch.Tensor] = [x_encoded]
        if self.config.channel_mode != 'only_seq':
            # Defensive slice: clip TSS features to 11 to guard against
            # any older stores with extra columns.
            encoder_inputs.append(features['TSS_features'][:, :11])
            encoder_inputs.append(features['GB_features'])
        encoder_inputs.append(ref_scalars)
        y = torch.cat(encoder_inputs, dim=1)
        
        # Dual encoding
        z_tumor_mean, z_tumor_log_var = self.tumor_encoder(y)
        z_background_mean, z_background_log_var = self.background_encoder(y)
        
        # Sampling
        z_tumor = self.sampling(z_tumor_mean, z_tumor_log_var)
        z_background = self.sampling(z_background_mean, z_background_log_var)
        
        # Latent merging for reconstruction.
        # The TFX discriminator predicts TFX from the latent means (with
        # gradient reversal pushing encoders toward TFX-invariant
        # representations).  The predicted TFX conditions the merge weights.
        predicted_tfx = self.tfx_discriminator(z_tumor_mean, z_background_mean)

        # Discriminator loss: MSE between predicted and actual TFX.
        # Only computed during training (requires the ground-truth label).
        grl_loss = None
        if self.training:
            if tfx is None:
                raise ValueError(
                    "tfx (ground-truth tumor fraction) must be provided "
                    "during training for the GRL discriminator loss."
                )
            grl_loss = F.mse_loss(predicted_tfx.float(), tfx.float())

        tfx_for_merge = predicted_tfx

        merge_input = torch.cat([
            tfx_for_merge, z_tumor_mean, z_tumor_log_var,
            z_background_mean, z_background_log_var,
        ], dim=1)
        weights = self.weight_predictor(merge_input)
        
        z_merged_mean, z_merged_log_var = self.merge_latents(
            weights, z_tumor_mean, z_tumor_log_var, z_background_mean, z_background_log_var
        )
        z_merged = self.sampling(z_merged_mean, z_merged_log_var)
        
        # Reconstruction
        y_reconstructed = self.decoder(z_merged)
        
        # MDN output heads: feed the reparameterized latent sample —
        # the same representation the decoder path receives.
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
        }
        
        # Internal loss
        if return_internal_loss and self.training:
            internal_loss, internal_metrics = self.internal_loss_fn(
                z_tumor, z_background,
                z_tumor_mean, z_tumor_log_var,
                z_background_mean, z_background_log_var,
                y, y_reconstructed,
                grl_loss=grl_loss,
            )
            return predictions, (internal_loss, internal_metrics)
        
        return predictions, None

    def freeze_signal_processing(self):
        """Freeze signal processing layers for transfer learning."""
        for param in self.signal_processor.parameters():
            param.requires_grad = False
        for param in self.reduction.parameters():
            param.requires_grad = False
        for param in self.signal_encoder.parameters():
            param.requires_grad = False
    
    def freeze_encoders(self):
        """Freeze encoder layers for transfer learning."""
        self.freeze_signal_processing()
        for param in self.tumor_encoder.parameters():
            param.requires_grad = False
        for param in self.background_encoder.parameters():
            param.requires_grad = False
    
    def get_freeze_level_params(self, level: str) -> List[str]:
        """
        Get parameter names for different freeze levels.
        
        Levels:
        - 'Medium-Heavy': Signal processing + first encoding layer
        - 'Heavy': Complete signal processing
        - 'Extra-Heavy': Signal processing + first encoder layer
        - 'Extreme': Everything through encoders
        """
        frozen = []
        
        if level in ['Medium-Heavy', 'Heavy', 'Extra-Heavy', 'Extreme']:
            frozen.extend([n for n, _ in self.signal_processor.named_parameters()])
            frozen.extend([n for n, _ in self.reduction.named_parameters()])
        
        if level in ['Heavy', 'Extra-Heavy', 'Extreme']:
            frozen.extend([n for n, _ in self.signal_encoder.named_parameters()])
        
        if level in ['Extra-Heavy', 'Extreme']:
            # First layers of encoders
            for name, _ in self.tumor_encoder.named_parameters():
                if 'layers.0' in name or 'layers.1' in name or 'layers.2' in name or 'layers.3' in name:
                    frozen.append(f'tumor_encoder.{name}')
            for name, _ in self.background_encoder.named_parameters():
                if 'layers.0' in name or 'layers.1' in name or 'layers.2' in name or 'layers.3' in name:
                    frozen.append(f'background_encoder.{name}')
        
        if level == 'Extreme':
            frozen.extend([f'tumor_encoder.{n}' for n, _ in self.tumor_encoder.named_parameters()])
            frozen.extend([f'background_encoder.{n}' for n, _ in self.background_encoder.named_parameters()])
        
        return frozen

    # ------------------------------------------------------------------
    # Freeze mode dispatch (fine-tuning)
    # ------------------------------------------------------------------

    _FREEZE_MODES = (
        'full_unfreeze',
        'backbone_only',
        'backbone_plus_encoders',
        'decoder_plus_heads',
        'heads_only',
    )

    def apply_freeze_mode(self, mode: str) -> None:
        """Apply a named freeze configuration for fine-tuning.

        Normalization layers (GroupNorm, LayerNorm) keep their affine
        parameters trainable in all modes so that running statistics can
        adapt to the new data distribution (norm adaptation).

        Modes (from least to most constrained):
            full_unfreeze          All parameters trainable.
            backbone_only          Freeze signal_processor + reduction +
                                   signal_encoder.
            backbone_plus_encoders Freeze backbone + both VAE encoders.
            decoder_plus_heads     Freeze everything except decoder +
                                   MDN heads.
            heads_only             Only MDN output heads are trainable.
        """
        if mode not in self._FREEZE_MODES:
            raise ValueError(
                f"Unknown freeze_mode '{mode}'. "
                f"Valid modes: {self._FREEZE_MODES}"
            )

        if mode == 'full_unfreeze':
            return  # Nothing to freeze

        # Define which top-level modules to freeze for each mode.
        backbone_modules = [self.signal_processor, self.reduction,
                            self.signal_encoder]
        encoder_modules = [self.tumor_encoder, self.background_encoder]
        latent_modules = [self.tfx_discriminator, self.weight_predictor,
                          self.merge_latents]
        decoder_modules = [self.decoder]
        head_modules = [self.tumor_mdn_head, self.background_mdn_head]

        freeze_map = {
            'backbone_only': backbone_modules,
            'backbone_plus_encoders': backbone_modules + encoder_modules,
            'decoder_plus_heads': (backbone_modules + encoder_modules
                                   + latent_modules),
            'heads_only': (backbone_modules + encoder_modules
                           + latent_modules + decoder_modules),
        }

        modules_to_freeze = freeze_map[mode]
        norm_types = (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d,
                      nn.BatchNorm2d)

        for mod in modules_to_freeze:
            for name, param in mod.named_parameters():
                # Keep normalization affine params trainable for distribution
                # adaptation, freeze everything else.
                is_norm = False
                for sub_name, sub_mod in mod.named_modules():
                    if isinstance(sub_mod, norm_types):
                        for pname, _ in sub_mod.named_parameters():
                            full = f"{sub_name}.{pname}" if sub_name else pname
                            if name == full:
                                is_norm = True
                                break
                    if is_norm:
                        break
                if not is_norm:
                    param.requires_grad = False
