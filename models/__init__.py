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
    # Auxiliary TFX prediction
    TFXPredictor,
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
    channel_mode_for_model,
    SignalProcessingBlock,
    ReductionBlock,
)

__all__ = [
    # Layers
    'Sampling', 'MergeLatents', 'TrainingChoice',
    # Loss functions
    'kl_divergence', 'contrastive_loss', 'reconstruction_loss',
    'InternalLossComputer', 'BioDVNMDNLoss',
    # Auxiliary TFX prediction
    'TFXPredictor',
    # Metrics
    'MaskedR2Metric', 'MetricsTracker',
    # Models
    'BioDVN', 'BioDVNConfig', 'get_model_instance',
    'channel_mode_for_model',
    'SignalProcessingBlock', 'ReductionBlock',
]