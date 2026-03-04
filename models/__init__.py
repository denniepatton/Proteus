# PyTorch models package
# Exposes key classes and functions for model building and training

from .model_utils import (
    # Custom layers
    Sampling,
    MergeLatents,
    TrainingChoice,
    # Loss components
    kl_divergence,
    contrastive_loss,
    reconstruction_loss,
    InternalLossComputer,
    # Gradient reversal for TFX-invariant latents
    GradientReversalLayer,
    TFXDiscriminator,
    # Combined loss
    BioDVNMDNLoss,
    # Metrics
    MaskedR2Metric,
    MetricsTracker,
)

from .models import (
    BioDVN,
    BioDVNConfig,
    get_model_instance,
    SignalProcessingBlock,
    ReductionBlock,
)

__all__ = [
    # Layers
    'Sampling', 'MergeLatents', 'TrainingChoice',
    # Loss functions
    'kl_divergence', 'contrastive_loss', 'reconstruction_loss',
    'InternalLossComputer', 'BioDVNMDNLoss',
    # Metrics
    'MaskedR2Metric', 'MetricsTracker',
    # Models
    'BioDVN', 'BioDVNConfig', 'get_model_instance',
    'SignalProcessingBlock', 'ReductionBlock',
]