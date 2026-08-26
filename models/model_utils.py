"""
model_utils.py (PyTorch)
Author: Robert D Patton, rpatton@fredhutch.org

Custom layers, loss functions, and metrics for BioDVN.
Includes MDN heads, analytical CRPS loss, β-TCVAE decomposition,
auxiliary TFX prediction, and masked regression losses.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import K_GAMMA, Y_MAX

# Mixture prior used to seed the MDN logits so the Normal component starts
# as the dominant active-expression branch instead of relying on symmetric
# zero-bias initialization.
MDN_MIXTURE_PRIORS: Tuple[float, float, float] = (0.10, 0.30, 0.60)

# Smooth bounded supports for the MDN component parameters. NORMAL_MU spans the
# expression range the TDS production models were trained against.
NORMAL_MU_MIN: float = 2.5
NORMAL_MU_MAX: float = 19.0
NORMAL_SIGMA_MIN: float = 0.10
NORMAL_SIGMA_MAX: float = 5.0
GAMMA_MEAN_MIN: float = 0.01
GAMMA_MEAN_MAX: float = 2.5
MDN_PARAM_EPS: float = 1e-6
SRL_STAT_EPS: float = 1e-6

# Empirically derived from training-truth summaries so the MDN starts in a
# realistic expression regime before fold-specific adaptation.
GAMMA_MEAN_INIT: float = 1.5
NORMAL_MU_INIT: float = 5.5
NORMAL_SIGMA_INIT: float = 1.5


def _inverse_sigmoid_bounded_init(target: float, lower: float, upper: float) -> float:
    """Return the raw bias whose sigmoid-bounded map yields *target*."""
    if not lower < target < upper:
        raise ValueError(
            f"Initialization target {target} must lie strictly within ({lower}, {upper})."
        )
    normalized = (target - lower) / (upper - lower)
    return math.log(normalized / (1.0 - normalized))


# ---------------------------------------------------------------------------
# Gauss-Laguerre quadrature constants for the analytical CRPS.
# Used for Gamma expectation terms whose closed forms route through
# incomplete-gamma functions with unstable CUDA backward behavior for k < 1.
# Precomputed once at import time; cached per device to avoid repeated
# CPU -> CUDA copies.
# ---------------------------------------------------------------------------
_QUAD_N: int = 32  # 32-node quadrature is exact for poly degree 63; ample for our smooth integrand

_gl_nodes_np, _gl_weights_np = np.polynomial.laguerre.laggauss(_QUAD_N)
_GL_NODES_CPU = torch.from_numpy(_gl_nodes_np.copy()).float()     # (Q,)
_GL_WEIGHTS_CPU = torch.from_numpy(_gl_weights_np.copy()).float() # (Q,)
# tᵢ^(k-1) where k = K_GAMMA = 0.75, so exponent = -0.25.
# Precomputed because k is a global constant.
_GL_NODE_POWERS_CPU = _GL_NODES_CPU.pow(K_GAMMA - 1.0)           # (Q,)

# B(k, 0.5) = Γ(k)Γ(0.5) / Γ(k + 0.5)  — appears in the Gamma
# self-interaction term  Bᵍᵍ = 2 / (β · B(k, 0.5)).
_BETA_K_HALF: float = math.exp(
    math.lgamma(K_GAMMA) + math.lgamma(0.5) - math.lgamma(K_GAMMA + 0.5)
)

# 1/Γ(k) — normalization factor for the Gamma density in quadrature.
_INV_GAMMA_K: float = 1.0 / math.exp(math.lgamma(K_GAMMA))

# Per-device cache: (nodes, weights, node_powers) tensor triples.
_quad_cache: Dict[torch.device, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_quad_tensors(
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (nodes, weights, node_powers) on *device*, caching the result."""
    if device not in _quad_cache:
        _quad_cache[device] = (
            _GL_NODES_CPU.to(device),
            _GL_WEIGHTS_CPU.to(device),
            _GL_NODE_POWERS_CPU.to(device),
        )
    return _quad_cache[device]


# ==============================================================================
# Normalization Factory
# ==============================================================================

def get_norm_layer(num_features: int) -> nn.Module:
    """Return a LayerNorm for 1D (Linear / MLP) blocks.

    Args:
        num_features: Number of features for normalization.
    """
    return nn.LayerNorm(num_features, eps=1e-5)


def get_norm_layer_conv1d(num_features: int) -> nn.Module:
    """Return a LayerNorm-equivalent for Conv1D blocks.

    Conv1d output shape is (N, C, L).  nn.LayerNorm(C) does not work here
    because it expects the *last* dimension to match C, but the last dim is L.
    GroupNorm(1, C) normalises over (C, L) per sample — semantically
    equivalent to LayerNorm for spatial tensors.

    Args:
        num_features: Number of channels for normalization.
    """
    return nn.GroupNorm(1, num_features, eps=1e-5)


def get_norm_layer_2d(num_features: int) -> nn.Module:
    """Return a LayerNorm-equivalent for Conv2D blocks.

    GroupNorm(1, C) normalises over (C, H, W) per sample — semantically
    equivalent to LayerNorm for 4-D conv tensors (B, C, H, W).

    Args:
        num_features: Number of channels for normalization.
    """
    return nn.GroupNorm(1, num_features, eps=1e-5)



# ==============================================================================
# Custom Layers
# ==============================================================================

class Sampling(nn.Module):
    """
    VAE reparameterization trick: sample from N(z_mean, exp(z_log_var/2))
    """
    def forward(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon


class MergeLatents(nn.Module):
    """
    Merges tumor and background latent spaces conditional on weights.
    Assumes Gaussian latent distributions and computes combined mean/variance.
    """
    def forward(
        self,
        weights: torch.Tensor,
        z_tumor_mean: torch.Tensor,
        z_tumor_log_var: torch.Tensor,
        z_background_mean: torch.Tensor,
        z_background_log_var: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Force float32: exp(z_log_var) overflows float16 when z_log_var > ~11.
        weights = weights.float()
        z_tumor_mean = z_tumor_mean.float()
        z_tumor_log_var = z_tumor_log_var.float()
        z_background_mean = z_background_mean.float()
        z_background_log_var = z_background_log_var.float()

        # weights shape: (batch, 2) where weights[:, 0] = tumor, weights[:, 1] = background
        w_tumor = weights[:, 0:1]
        w_background = weights[:, 1:2]
        
        # Merge latent means
        z_merged_mean = w_tumor * z_tumor_mean + w_background * z_background_mean
        
        # Convert log variances to variances
        z_tumor_var = torch.exp(z_tumor_log_var)
        z_background_var = torch.exp(z_background_log_var)
        
        # Merge latent variances via the law of total variance:
        #   Var(Z) = E[Var(Z|C)] + Var(E[Z|C])
        # Expanded:  w_t·σ²_t + w_b·σ²_b + w_t·μ²_t + w_b·μ²_b − μ²_merged
        #
        # Floating-point arithmetic can make this expression slightly
        # negative (the subtracted squared-mean can overshoot).  We must
        # clamp *before* log — adding a small epsilon does NOT prevent
        # log(negative) when the negative value exceeds the epsilon.
        z_merged_var = (
            w_tumor * z_tumor_var +
            w_background * z_background_var +
            w_tumor * torch.square(z_tumor_mean) +
            w_background * torch.square(z_background_mean) -
            torch.square(z_merged_mean)
        )
        
        # Convert merged variances back to log variances
        z_merged_log_var = torch.log(z_merged_var.clamp(min=1e-8))
        
        return z_merged_mean, z_merged_log_var


class TrainingChoice(nn.Module):
    """
    Returns sampled latents during training, means during inference.
    """
    def forward(
        self,
        latent_sampling: torch.Tensor,
        latent_means: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            return latent_sampling
        return latent_means


# ==============================================================================
# Loss Function Components (for internal losses)
# ==============================================================================

def kl_divergence(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
    """
    KL divergence between N(z_mean, exp(z_log_var)) and N(0, 1).
    """
    # Force float32: exp(z_log_var) overflows float16 when z_log_var > ~11.
    z_mean = z_mean.float()
    z_log_var = z_log_var.float()
    return -0.5 * torch.mean(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))


def contrastive_loss(z_tumor: torch.Tensor, z_background: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """
    Contrastive loss to push tumor and background latent spaces apart.
    """
    z_tumor = z_tumor.float()
    z_background = z_background.float()
    distance = torch.mean(torch.square(z_tumor - z_background), dim=-1)
    return torch.mean(F.relu(margin - distance))


def reconstruction_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Simple MSE reconstruction loss.
    """
    # Force float32: squared differences of raw feature vectors can overflow
    # float16 (max ~65504) when any component has magnitude > ~256.
    return F.mse_loss(y_pred.float(), y_true.float())


# ==============================================================================
# Auxiliary TFX prediction
# ==============================================================================


class TFXPredictor(nn.Module):
    """Small MLP that predicts TFX from tumor + background latent means.

    TFX is supervised during training but is never supplied as an input at
    inference.  Gradients flow normally so the latent space can preserve tumor
    fraction information needed for mixture weighting and uncertainty.
    """

    def __init__(self, latent_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_tumor_mean: torch.Tensor, z_bg_mean: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_tumor_mean, z_bg_mean], dim=1)
        return self.net(h)


class InternalLossComputer:
    """
    Computes internal losses (KL, contrastive, reconstruction) for the VAE.
    """
    def __init__(
        self,
        beta_contrastive: float = 1.0,
        beta_kl: float = 1.0,
        beta_reconstruction: float = 1.25,
        beta_tfx: float = 2.0,
        cl_margin: float = 1.0,
        internal_weight: float = 0.5,
    ):
        self.beta_contrastive = beta_contrastive
        self.beta_kl = beta_kl
        self.beta_reconstruction = beta_reconstruction
        self.beta_tfx = beta_tfx
        self.cl_margin = cl_margin
        self.internal_weight = internal_weight
    
    def _compute_kl_loss(
        self,
        z_tumor_mean: torch.Tensor,
        z_tumor_log_var: torch.Tensor,
        z_background_mean: torch.Tensor,
        z_background_log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute KL regularization loss for the two latent branches."""
        l_kl_tumor = kl_divergence(z_tumor_mean, z_tumor_log_var)
        l_kl_background = kl_divergence(z_background_mean, z_background_log_var)
        kl_loss = l_kl_tumor + l_kl_background
        metrics = {
            'kl_loss': kl_loss.detach(),
        }
        return self.beta_kl * kl_loss, metrics
    
    def __call__(
        self,
        z_tumor: torch.Tensor,
        z_background: torch.Tensor,
        z_tumor_mean: torch.Tensor,
        z_tumor_log_var: torch.Tensor,
        z_background_mean: torch.Tensor,
        z_background_log_var: torch.Tensor,
        y: torch.Tensor,
        y_reconstructed: torch.Tensor,
        tfx_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: Scaled internal loss
            metrics: Dictionary of individual loss components for logging
        """
        # Force float32 for all internal loss computation.  Under AMP the
        # encoder/decoder outputs are float16; exp(z_log_var) overflows at
        # z_log_var > ~11 and MSE overflows for feature magnitudes > ~256.
        z_tumor = z_tumor.float()
        z_background = z_background.float()
        z_tumor_mean = z_tumor_mean.float()
        z_tumor_log_var = z_tumor_log_var.float()
        z_background_mean = z_background_mean.float()
        z_background_log_var = z_background_log_var.float()
        y = y.float()
        y_reconstructed = y_reconstructed.float()

        l_contrastive = contrastive_loss(z_tumor, z_background, self.cl_margin)
        l_reconstruction = reconstruction_loss(y, y_reconstructed)
        
        # Compute KL regularization for both latent branches.
        vae_loss, vae_metrics = self._compute_kl_loss(
            z_tumor_mean, z_tumor_log_var,
            z_background_mean, z_background_log_var,
        )
        
        total_loss = (
            self.beta_contrastive * l_contrastive +
            vae_loss +
            self.beta_reconstruction * l_reconstruction
        )

        if tfx_loss is not None:
            tfx_loss = tfx_loss.float()
            total_loss = total_loss + self.beta_tfx * tfx_loss
        
        scaled_loss = self.internal_weight * total_loss
        
        metrics = {
            # Keep metrics as tensors to avoid Dynamo graph breaks when the
            # model forward is compiled (Tensor.item() forces a graph break).
            # Detach to ensure these are used only for logging.
            'internal_loss': scaled_loss.detach(),
            'contrastive_loss': l_contrastive.detach(),
            'recon_loss': l_reconstruction.detach(),
            **vae_metrics,
        }
        if tfx_loss is not None:
            metrics['tfx_mse_loss'] = tfx_loss.detach()
        
        return scaled_loss, metrics


# ==============================================================================
# Metrics
# ==============================================================================

class MaskedR2Metric:
    """
    R² metric that ignores NaN values. Accumulates across batches.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum_y = 0.0
        self.sum_y_sq = 0.0
        self.sum_res = 0.0
        self.count = 0.0
    
    def update(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ):
        mask = ~torch.isnan(y_true)
        mask_f = mask.float()
        
        y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))
        y_pred_masked = torch.where(mask, y_pred, torch.zeros_like(y_pred))
        
        if sample_weight is not None:
            effective_weights = sample_weight * mask_f
        else:
            effective_weights = mask_f
        
        self.sum_res += (effective_weights * torch.square(y_true_masked - y_pred_masked)).sum().item()
        self.sum_y += (effective_weights * y_true_masked).sum().item()
        self.sum_y_sq += (effective_weights * torch.square(y_true_masked)).sum().item()
        self.count += effective_weights.sum().item()
    
    def compute(self) -> float:
        if self.count < 1e-8:
            return 0.0
        ss_tot = self.sum_y_sq - (self.sum_y ** 2) / (self.count + 1e-8)
        return 1.0 - (self.sum_res / (ss_tot + 1e-8))


class MetricsTracker:
    """
    Tracks multiple metrics across training/validation.
    """
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def reset(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, name: str, value: float | torch.Tensor, count: int = 1):
        if isinstance(value, torch.Tensor):
            detached_value = value.detach()
            if detached_value.numel() != 1:
                raise ValueError(
                    f"MetricsTracker.update expects scalar tensors, got shape {tuple(detached_value.shape)} for '{name}'."
                )
            detached_value = detached_value.reshape(())
            if name not in self.metrics:
                self.metrics[name] = detached_value.new_tensor(0.0)
                self.counts[name] = 0
            elif not isinstance(self.metrics[name], torch.Tensor):
                self.metrics[name] = detached_value.new_tensor(float(self.metrics[name]))
            self.metrics[name] = self.metrics[name] + detached_value * count
        else:
            scalar_value = float(value)
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            elif isinstance(self.metrics[name], torch.Tensor):
                self.metrics[name] = self.metrics[name] + self.metrics[name].new_tensor(scalar_value * count)
                self.counts[name] += count
                return
            else:
                self.metrics[name] += scalar_value * count
        self.counts[name] += count
    
    def compute(self, *, as_python: bool = True) -> dict:
        result = {}
        for name, total in self.metrics.items():
            avg = total / max(self.counts[name], 1)
            if as_python:
                if isinstance(avg, torch.Tensor):
                    result[name] = float(avg.detach().cpu().item())
                else:
                    result[name] = float(avg)
            else:
                result[name] = avg
        return result


# ==============================================================================
# Mixture Density Network (MDN) Components
# ==============================================================================

class MixtureDensityHead(nn.Module):
    """Output head that produces parameters for a 3-component mixture density.

    Components:
        1. Zero (hurdle) — true point mass at 0 (zero-expression genes)
        2. Gamma — Gamma(k=K_GAMMA, rate=beta), shape fixed, mean predicted
        3. Normal — N(mu, sigma^2), both parameters smoothly bounded

    Notes:
        - Targets are in RNA-Seq log2(CPM+1) space and are expected to be in [0, y_max].
        - Point-estimate predictions are clamped to [0, y_max] to respect the label domain.

    The head outputs:
        - log_pi: (batch, 3) log-softmax mixture weights [pi_zero, pi_gamma, pi_normal]
        - gamma_beta: (batch, 1) Gamma rate parameter derived from bounded mean
        - normal_mu: (batch, 1) Normal location in [NORMAL_MU_MIN, NORMAL_MU_MAX]
        - normal_sigma: (batch, 1) Normal scale in [NORMAL_SIGMA_MIN, NORMAL_SIGMA_MAX]

    Args:
        input_dim: Dimension of the input feature vector.
        hidden_dim: Hidden layer width inside the head (default 64).
        final_dim: Width of the last hidden layer before the 6-output
            projection (default 16).  Intermediate layers halve from
            *hidden_dim* down to *final_dim*.  Example with
            hidden_dim=64, final_dim=16::

                input_dim → 64 → ReLU → 32 → ReLU → 16 → ReLU → heads
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        final_dim: int = 16,
    ):
        super().__init__()

        # Build halving-by-two reduction cone: hidden_dim → … → final_dim.
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        dim = hidden_dim
        while dim // 2 >= final_dim:
            next_dim = dim // 2
            layers.append(nn.Linear(dim, next_dim))
            layers.append(nn.ReLU())
            dim = next_dim
        self.shared = nn.Sequential(*layers)

        self.linear_pi = nn.Linear(dim, 3)
        self.linear_gamma_mean = nn.Linear(dim, 1)
        self.linear_mu = nn.Linear(dim, 1)
        self.linear_sigma = nn.Linear(dim, 1)

        # Xavier init + explicit bias initialization on every Linear so the
        # head starts in a biologically plausible regime while preserving the
        # requested prior over mixture usage.
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in (
            self.linear_pi,
            self.linear_gamma_mean,
            self.linear_mu,
            self.linear_sigma,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        with torch.no_grad():
            prior_tensor = torch.tensor(
                MDN_MIXTURE_PRIORS,
                dtype=self.linear_pi.bias.dtype,
                device=self.linear_pi.bias.device,
            )
            self.linear_pi.bias.copy_(torch.log(prior_tensor))
            self.linear_gamma_mean.bias.fill_(
                _inverse_sigmoid_bounded_init(
                    GAMMA_MEAN_INIT,
                    GAMMA_MEAN_MIN,
                    GAMMA_MEAN_MAX,
                )
            )
            self.linear_mu.bias.fill_(
                _inverse_sigmoid_bounded_init(
                    NORMAL_MU_INIT,
                    NORMAL_MU_MIN,
                    NORMAL_MU_MAX,
                )
            )
            self.linear_sigma.bias.fill_(
                _inverse_sigmoid_bounded_init(
                    NORMAL_SIGMA_INIT,
                    NORMAL_SIGMA_MIN,
                    NORMAL_SIGMA_MAX,
                )
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, input_dim) feature tensor.

        Returns:
            Dictionary with keys:
                log_pi: (batch, 3) log-softmax of mixture weights
                gamma_beta: (batch, 1) Gamma rate (>0)
                normal_mu: (batch, 1) bounded to the active-expression range
                normal_sigma: (batch, 1) bounded away from zero and blow-up
        """
        # Force float32 before parameterization.  The smooth sigmoid maps
        # below keep every parameter inside a validated support while
        # preserving gradient flow across the full raw range.
        # Keep the MDN head in float32 under AMP. The encoder dominates runtime,
        # so the throughput cost is negligible, while this removes half-precision
        # matmuls from the path that has produced most of the overflow churn.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            h = self.shared(x.float())
            logits = self.linear_pi(h)
            raw_gamma_mean = self.linear_gamma_mean(h)
            raw_mu = self.linear_mu(h)
            raw_sigma = self.linear_sigma(h)
            log_pi = F.log_softmax(logits, dim=-1)  # (batch, 3)
            gamma_mean = GAMMA_MEAN_MIN + (GAMMA_MEAN_MAX - GAMMA_MEAN_MIN) * torch.sigmoid(raw_gamma_mean)
            gamma_beta = K_GAMMA / (gamma_mean + MDN_PARAM_EPS)
            normal_mu = NORMAL_MU_MIN + (NORMAL_MU_MAX - NORMAL_MU_MIN) * torch.sigmoid(raw_mu)
            normal_sigma = NORMAL_SIGMA_MIN + (NORMAL_SIGMA_MAX - NORMAL_SIGMA_MIN) * torch.sigmoid(raw_sigma)

        return {
            'log_pi': log_pi,
            'gamma_beta': gamma_beta,
            'normal_mu': normal_mu,
            'normal_sigma': normal_sigma,
        }


def mixture_mean(
    mdn_params: dict,
) -> torch.Tensor:
    """Compute the mixture mean (point estimate) from MDN parameters.

    E[Y] = pi_zero * 0 + pi_gamma * (K_GAMMA / beta) + pi_normal * mu

    Args:
        mdn_params: dict from MixtureDensityHead.forward().

    Returns:
        mean: (batch, 1) point estimate.
    """
    pi = mdn_params['log_pi'].float().exp()  # (batch, 3)
    pi_gamma = pi[:, 1:2]
    pi_normal = pi[:, 2:3]

    beta = mdn_params['gamma_beta'].float().clamp(min=1e-6)
    mu = mdn_params['normal_mu'].float()
    # E[Gamma(k, beta)] = k / beta.  Clamp to Y_MAX to respect label domain.
    gamma_mean = (K_GAMMA / beta).clamp(max=Y_MAX)
    mean = pi_gamma * gamma_mean + pi_normal * mu
    return mean.clamp(0.0, Y_MAX)  # (batch, 1)


def mixture_variance(
    mdn_params: dict,
) -> torch.Tensor:
    """Compute predictive variance of the 3-component MDN mixture."""
    pi = mdn_params['log_pi'].float().exp()
    pi_gamma = pi[:, 1:2]
    pi_normal = pi[:, 2:3]

    beta = mdn_params['gamma_beta'].float().clamp(min=1e-6)
    mu = mdn_params['normal_mu'].float()
    sigma = mdn_params['normal_sigma'].float().clamp(min=1e-6)

    gamma_mean = (K_GAMMA / beta).clamp(max=Y_MAX)
    gamma_second = (K_GAMMA * (K_GAMMA + 1.0) / beta.pow(2)).clamp(max=Y_MAX ** 2)
    normal_second = sigma.pow(2) + mu.pow(2)
    second_moment = pi_gamma * gamma_second + pi_normal * normal_second
    mean = pi_gamma * gamma_mean + pi_normal * mu
    return (second_moment - mean.pow(2)).clamp(min=0.0)


def _mixture_cdf_at_points(
    y: torch.Tensor,
    mdn_params: dict,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Evaluate the mixture CDF at per-sample query points.

    Unlike a grid-based CDF that broadcasts a shared 1-D grid across the
    batch, this evaluates the CDF at a *different* y value for each sample —
    exactly what vectorized bisection needs.

    Args:
        y: (batch,) or (batch, 1) per-sample query points.
        mdn_params: dict from MixtureDensityHead.forward().
        eps: small constant for numerical stability.

    Returns:
        cdf: (batch,) CDF values.
    """
    y = y.float().view(-1, 1)  # (batch, 1)

    pi = mdn_params['log_pi'].float().exp()  # (batch, 3)
    pi_zero = pi[:, 0:1]
    pi_gamma = pi[:, 1:2]
    pi_normal = pi[:, 2:3]

    cdf_zero = (y >= 0.0).to(dtype=y.dtype)  # (batch, 1)

    beta = mdn_params['gamma_beta'].float().clamp(min=eps)  # (batch, 1)
    y_pos = y.clamp(min=0.0)
    cdf_gamma = torch.special.gammainc(
        torch.tensor(K_GAMMA, device=y.device, dtype=y.dtype),
        beta * y_pos,
    )  # (batch, 1)

    mu = mdn_params['normal_mu'].float()  # (batch, 1)
    sigma = mdn_params['normal_sigma'].float().clamp(min=eps)  # (batch, 1)
    cdf_normal = 0.5 * (1.0 + torch.erf((y - mu) / (sigma * math.sqrt(2.0))))

    cdf = pi_zero * cdf_zero + pi_gamma * cdf_gamma + pi_normal * cdf_normal
    return cdf.squeeze(-1)  # (batch,)


def mixture_quantile(
    mdn_params: dict,
    quantiles: Tuple[float, ...] = (0.05, 0.95),
    n_steps: int = 50,
) -> torch.Tensor:
    """Compute quantiles of the mixture distribution via vectorized bisection.

    For each requested quantile q, finds the value x where CDF(x) = q
    using bisection on the monotone mixture CDF.  All computation is done
    on CPU with detached parameters.

    Args:
        mdn_params: dict from MixtureDensityHead.forward().
        quantiles: tuple of quantile levels (e.g., (0.05, 0.95) for 90% CI).
        n_steps: number of bisection iterations (50 gives precision < 1e-12).

    Returns:
        result: (batch, len(quantiles)) tensor of quantile values, clamped to [0, Y_MAX].
    """
    device = mdn_params['log_pi'].device
    mdn_cpu = {k: v.detach().float().cpu() for k, v in mdn_params.items()}

    beta = mdn_cpu['gamma_beta'].clamp(min=1e-6)        # (batch, 1)
    mu = mdn_cpu['normal_mu']                            # (batch, 1)
    sigma = mdn_cpu['normal_sigma'].clamp(min=1e-6)      # (batch, 1)
    batch_size = mu.shape[0]

    # Upper bound per sample: envelope of gamma and normal tails.
    gamma_upper = (K_GAMMA + 4.0 * math.sqrt(K_GAMMA)) / beta
    normal_upper = mu + 5.0 * sigma
    hi_init = torch.maximum(gamma_upper, normal_upper).clamp(
        min=1.0, max=Y_MAX,
    ).view(-1)  # (batch,)

    results = []
    for q in quantiles:
        lo = torch.zeros(batch_size)
        hi = hi_init.clone()

        for _ in range(n_steps):
            mid = (lo + hi) * 0.5
            cdf_val = _mixture_cdf_at_points(mid, mdn_cpu)  # (batch,)
            lo = torch.where(cdf_val < q, mid, lo)
            hi = torch.where(cdf_val >= q, mid, hi)

        result = ((lo + hi) * 0.5).clamp(0.0, Y_MAX)
        results.append(result.unsqueeze(-1))  # (batch, 1)

    return torch.cat(results, dim=-1).to(device)  # (batch, len(quantiles))


# ==============================================================================
# Analytical CRPS primitives
# ==============================================================================
#
# For a mixture  F = w₀·δ₀ + wᵍ·Gamma(k, β) + wⁿ·N(μ,σ²)  the CRPS
# decomposes as (Gneiting & Raftery 2007, §5.2):
#
#   CRPS(F, y) = Σᵢ wᵢ Aᵢ(y)  −  ½ Σᵢⱼ wᵢwⱼ Bᵢⱼ
#
# where  Aᵢ(y) = E|Xᵢ − y|  and  Bᵢⱼ = E|Xᵢ − X′ⱼ|.
#
# This gives three A terms and six unique B entries in the symmetric 3×3
# matrix.  Aᵍ(y) and Bᵍₙ use 1-D Gauss-Laguerre quadrature; the remaining
# non-zero terms are closed form.
# ==============================================================================


def _std_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """Φ(z) — standard Normal CDF."""
    return 0.5 * (1.0 + torch.erf(z * _SQRT_HALF))


def _std_normal_pdf(z: torch.Tensor) -> torch.Tensor:
    """φ(z) — standard Normal PDF."""
    return torch.exp(-0.5 * z * z) * _INV_SQRT_2PI


# Precomputed constants used by the Normal helpers.
_SQRT_HALF: float = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI: float = 1.0 / math.sqrt(2.0 * math.pi)
_TWO_OVER_SQRT_PI: float = 2.0 / math.sqrt(math.pi)


def _normal_abs_dev(
    y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
) -> torch.Tensor:
    """Aₙ(y) = E|X − y|  for  X ~ N(μ, σ²).

    Closed form:
        z = (y − μ)/σ
        Aₙ = σ · [ z(2Φ(z) − 1) + 2φ(z) ]
    """
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * _std_normal_cdf(z) - 1.0) + 2.0 * _std_normal_pdf(z))


def _gamma_abs_dev(
    y: torch.Tensor, beta: torch.Tensor,
) -> torch.Tensor:
    """Aᵍ(y) = E|X − y|  for  X ~ Gamma(k, rate=β),  k = K_GAMMA.

    Closed form using the regularized lower incomplete gamma P(a, x):
        Aᵍ = y(2P(k, βy) − 1) + (k/β)(1 − 2P(k+1, βy))

    Derivation:
        E|X − y| = y·F(y) − ∫₀ʸ x·f(x)dx + E[X] − ∫₀ʸ x·f(x)dx − y(1−F(y))
                 = y(2F(y) − 1) − 2(k/β)P(k+1, βy) + k/β
    where F(y) = P(k, βy) and ∫₀ʸ x·f(x)dx = (k/β)P(k+1, βy).

    Numerical note:
        The backward of ``gammainc(a, x)`` is singular at x = 0 when
        0 < a < 1.  Clamping βy away from zero keeps the closed-form CRPS
        path finite while leaving zero-expression forward values effectively
        at the Gamma mean.
    """
    by = (beta * y).clamp(min=1e-7)
    k_t = y.new_tensor(K_GAMMA)
    k1_t = y.new_tensor(K_GAMMA + 1.0)
    P_k = torch.special.gammainc(k_t, by)
    P_k1 = torch.special.gammainc(k1_t, by)
    return y * (2.0 * P_k - 1.0) + (K_GAMMA / beta) * (1.0 - 2.0 * P_k1)


def _gamma_self_interaction(beta: torch.Tensor) -> torch.Tensor:
    """Bᵍᵍ = E|X − X′|  for  X, X′ iid ~ Gamma(k, β).

    Closed form:  Bᵍᵍ = 2 / (β · B(k, ½)).

    B(k, ½) = Γ(k)Γ(½)/Γ(k+½) is a constant (precomputed as _BETA_K_HALF).
    """
    return 2.0 / (beta * _BETA_K_HALF)


def _cross_term_gamma_normal(
    beta: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
) -> torch.Tensor:
    """B_{gn} = E|Xᵍ − Xₙ|  via Gauss-Laguerre quadrature.

    Strategy (law of iterated expectation):
        B_{gn} = E_{Xᵍ}[ h(Xᵍ) ]
    where
        h(c) = E_{Xₙ}|c − Xₙ| = σ[z(2Φ(z)−1) + 2φ(z)],   z = (c−μ)/σ
    is the Normal absolute-deviation function (closed form).

    The outer expectation over Xᵍ ~ Gamma(k, β) is turned into a
    standard Gauss-Laguerre integral via the substitution t = βx:

        E_{Xᵍ}[h(Xᵍ)] = (1/Γ(k)) ∫₀^∞ h(t/β) · t^{k−1} · e^{−t} dt
                        ≈ (1/Γ(k)) Σᵢ wᵢ · h(tᵢ/β) · tᵢ^{k−1}

    Nodes tᵢ and weights wᵢ are standard Gauss-Laguerre (weight = e^{−t});
    tᵢ^{k−1} is precomputed since k is a global constant.
    """
    nodes, weights, node_powers = _get_quad_tensors(beta.device)
    # nodes: (Q,), weights: (Q,), node_powers: (Q,)  —  Q = _QUAD_N

    # Gamma sample points mapped through the change-of-variable:
    #   xᵍ = tᵢ / β   →   shape (batch, Q)
    x_g = nodes.unsqueeze(0) / beta  # (batch, Q)

    # Evaluate h(xᵍ) — the Normal absolute-deviation at each quad node.
    z = (x_g - mu) / sigma  # (batch, Q)
    h_vals = sigma * (z * (2.0 * _std_normal_cdf(z) - 1.0)
                       + 2.0 * _std_normal_pdf(z))  # (batch, Q)

    # Weighted sum:  (1/Γ(k)) Σᵢ wᵢ · h(tᵢ/β) · tᵢ^{k−1}
    integrand = h_vals * node_powers.unsqueeze(0) * weights.unsqueeze(0)
    return integrand.sum(dim=-1, keepdim=True) * _INV_GAMMA_K  # (batch, 1)


# ==============================================================================
# Public CRPS entry point
# ==============================================================================

def mixture_crps(
    mdn_params: dict,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytical CRPS for the 3-component MDN mixture.

    Mixture:  F = w₀·δ₀ + wᵍ·Gamma(k, β) + wⁿ·N(μ, σ²).

    Uses the expectation-form decomposition (Gneiting & Raftery 2007):

        CRPS(F, y) = Σᵢ wᵢ Aᵢ(y)  −  ½ Σᵢⱼ wᵢwⱼ Bᵢⱼ

    where  Aᵢ(y) = E|Xᵢ − y|  and  Bᵢⱼ = E|Xᵢ − X′ⱼ|.

    Term summary (k = K_GAMMA = 0.75):
        A₀(y)  = |y|                            (point mass — trivial)
        Aᵍ(y)  = E|Xᵍ−y|                       (1-D Gauss-Laguerre)
        Aₙ(y)  = σ[z(2Φ(z)−1) + 2φ(z)]         (closed form)
        B₀₀    = 0                               (both at zero)
        B₀ᵍ    = k/β                             (Gamma mean)
        B₀ₙ    = E|Xₙ| = Aₙ(0)                  (closed form)
        Bᵍᵍ    = 2/(β·B(k,½))                    (closed form)
        Bₙₙ    = 2σ/√π                           (closed form)
        Bᵍₙ    = E_{Xᵍ}[h(Xᵍ)]                  (1-D Gauss-Laguerre)

    All terms are differentiable.  The two quadrature terms (Aᵍ and Bᵍₙ)
    use a 32-node Gauss-Laguerre rule whose nodes/weights are precomputed
    constants, so autodiff flows cleanly through ordinary tensor operations.

    Args:
        mdn_params: dict from ``MixtureDensityHead.forward()``.
        target: (batch, 1) ground-truth expression values.
        sample_weight: optional (batch, 1) or (batch,) per-gene weights.

    Returns:
        (crps_sum, denom): scalar tensors on the same device as *target*.
        ``crps_sum / denom`` gives the weighted-mean CRPS for the batch
        when ``denom > 0``.  If no target entries are finite, returns
        ``(0, 0)`` so callers can skip the missing slice without injecting
        a fake pseudo-count into pooled metrics.
    """
    device = target.device
    valid = torch.isfinite(target).view(-1)
    if not valid.any():
        return target.new_tensor(0.0), target.new_tensor(0.0)

    y = target.view(-1, 1)[valid].float()

    # Keep invalid-label rows out of the CRPS graph entirely.  Masking after
    # the distribution math can still leave undefined backward paths alive.
    mdn_params = {
        k: v.to(device)[valid] if isinstance(v, torch.Tensor) else v
        for k, v in mdn_params.items()
    }

    # ------------------------------------------------------------------
    # Extract mixture parameters
    # ------------------------------------------------------------------
    pi = mdn_params['log_pi'].float().exp()                       # (batch, 3)
    w0 = pi[:, 0:1]                                               # hurdle
    wg = pi[:, 1:2]                                               # gamma
    wn = pi[:, 2:3]                                               # normal
    beta = mdn_params['gamma_beta'].float().clamp(min=1e-6)       # (batch, 1)
    mu = mdn_params['normal_mu'].float()                          # (batch, 1)
    sigma = mdn_params['normal_sigma'].float().clamp(min=1e-6)    # (batch, 1)

    # ------------------------------------------------------------------
    # A-terms:  Aᵢ(y) = E|Xᵢ − y|
    # ------------------------------------------------------------------
    a0 = y.abs()                              # |y|;  y ≥ 0 in log₂(CPM+1)
    ag = _gamma_abs_dev(y, beta)              # closed form via gammainc
    an = _normal_abs_dev(y, mu, sigma)        # closed form via erf

    a_sum = w0 * a0 + wg * ag + wn * an      # (batch, 1)

    # ------------------------------------------------------------------
    # B-terms:  Bᵢⱼ = E|Xᵢ − X′ⱼ|   (symmetric, so 6 unique entries)
    # ------------------------------------------------------------------
    # b00 = 0                                → omitted (contributes nothing)
    b0g = K_GAMMA / beta                       # E[Gamma] = k/β
    b0n = _normal_abs_dev(                     # E|N(μ,σ²)| = Aₙ(y=0)
        torch.zeros_like(y), mu, sigma,
    )
    bgg = _gamma_self_interaction(beta)        # 2/(β·B(k,½))
    bnn = _TWO_OVER_SQRT_PI * sigma            # 2σ/√π
    bgn = _cross_term_gamma_normal(            # 32-pt Gauss-Laguerre
        beta, mu, sigma,
    )

    # Full symmetric matrix sum:  Σᵢⱼ wᵢwⱼBᵢⱼ
    # (b00 = 0 so the w0*w0 term vanishes)
    b_sum = (wg * wg * bgg + wn * wn * bnn
             + 2.0 * w0 * wg * b0g
             + 2.0 * w0 * wn * b0n
             + 2.0 * wg * wn * bgn)           # (batch, 1)

    crps = a_sum - 0.5 * b_sum                 # (valid_batch, 1)

    # ------------------------------------------------------------------
    # Sample-weight handling
    # ------------------------------------------------------------------
    if sample_weight is not None:
        w = sample_weight
        if w.dim() == 1:
            w = w.unsqueeze(-1)
        w = w.to(device=device, dtype=torch.float32).view(-1, 1)[valid]
        # Re-normalise weights so batch mean weight = 1.0.
        w_mean = w.mean()
        w = w / w_mean.clamp(min=1e-8)

        crps = crps * w
        denom = w.sum()
    else:
        denom = y.new_tensor(float(y.shape[0]))

    return crps.sum(), denom.clamp(min=1)


# ==============================================================================
# Loss classes
# ==============================================================================

class MixtureDensityCRPSLoss(nn.Module):
    """Masked, gene-weighted analytical CRPS loss for the 3-component mixture.

    The external objective keeps CRPS as the distributional likelihood-aligned
    term, then adds SRL, same-gene/source dynamic structure, and
    PHT structure losses when metadata are available from the gene-conditioned
    loader.
    """

    def __init__(
        self,
        beta_background: float = 1.0,
        beta_srl_max: float = 1.5,
        beta_dynamic_pair_max: float = 0.0,
        srl_delta: float = 1.0,
        srl_start: int = 5,
        srl_ramp: int = 10,
        beta_pht_consistency: float = 0.5,
        beta_pht_uncertainty: float = 1.5,
    ):
        super().__init__()
        self.beta_background = beta_background
        self.beta_srl_max = beta_srl_max
        self.beta_dynamic_pair_max = beta_dynamic_pair_max
        self.srl_delta = srl_delta
        self.srl_start = srl_start
        self.srl_ramp = srl_ramp
        self.beta_pht_consistency = beta_pht_consistency
        self.beta_pht_uncertainty = beta_pht_uncertainty
        self._current_epoch = 0
        self._srl_gene_mean_map: Optional[Dict[str, float]] = None
        self._srl_gene_std_map: Optional[Dict[str, float]] = None

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def set_srl_stats(
        self,
        gene_mean_map: Dict[str, float],
        gene_std_map: Dict[str, float],
    ) -> None:
        self._srl_gene_mean_map = dict(gene_mean_map)
        self._srl_gene_std_map = dict(gene_std_map)

    def forward(
        self,
        predictions: dict,
        targets: dict,
        sample_weight: Optional[torch.Tensor] = None,
        features: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            predictions: dict with 'tumor_mdn' and 'background_mdn'.
            targets: dict with 'tumor_exp' (batch,1) and 'background_exp' (batch,1).
            sample_weight: optional (batch,1) or (batch,) per-gene weights.
            features: optional metadata from the gene-conditioned loader.

        Returns:
            total_loss: scalar
            loss_dict: per-head CRPS values for logging
        """
        tumor_crps = self._head_crps(
            predictions['tumor_mdn'], targets['tumor_exp'], sample_weight
        )
        background_crps = self._head_crps(
            predictions['background_mdn'], targets['background_exp'], sample_weight
        )

        total_loss = tumor_crps + self.beta_background * background_crps

        loss_dict = {
            'tumor_crps': tumor_crps.detach(),
            'background_crps': background_crps.detach(),
            'total_external_loss': total_loss.detach(),
        }

        srl_weight = self._current_srl_weight()
        if srl_weight > 0.0 and features is not None:
            srl_loss = self._standardized_residual_loss(
                predictions['tumor_exp'],
                targets['tumor_exp'],
                features,
                sample_weight,
            )
            total_loss = total_loss + srl_weight * srl_loss
            loss_dict['standardized_residual_loss'] = srl_loss.detach()
            loss_dict['standardized_residual_weight'] = srl_weight
            loss_dict['total_external_loss'] = total_loss.detach()

        dynamic_pair_weight = self._current_dynamic_pair_weight()
        if dynamic_pair_weight > 0.0 and features is not None:
            dynamic_pair_loss = self._dynamic_pair_loss(
                predictions['tumor_exp'],
                targets['tumor_exp'],
                features,
                sample_weight,
            )
            total_loss = total_loss + dynamic_pair_weight * dynamic_pair_loss
            loss_dict['dynamic_pair_loss'] = dynamic_pair_loss.detach()
            loss_dict['dynamic_pair_weight'] = dynamic_pair_weight
            loss_dict['total_external_loss'] = total_loss.detach()

        if features is not None and (
            self._current_pht_consistency_weight() > 0.0 or self._current_pht_uncertainty_weight() > 0.0
        ):
            pht_consistency, pht_uncertainty = self._pht_losses(
                predictions, targets, features,
            )
            total_loss = (
                total_loss
                + self._current_pht_consistency_weight() * pht_consistency
                + self._current_pht_uncertainty_weight() * pht_uncertainty
            )
            loss_dict['pht_consistency_loss'] = pht_consistency.detach()
            loss_dict['pht_uncertainty_loss'] = pht_uncertainty.detach()
            loss_dict['pht_consistency_weight'] = self._current_pht_consistency_weight()
            loss_dict['pht_uncertainty_weight'] = self._current_pht_uncertainty_weight()
            loss_dict['total_external_loss'] = total_loss.detach()
        return total_loss, loss_dict

    def _head_crps(
        self,
        mdn_params: dict,
        target: torch.Tensor,
        gene_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CRPS loss for one output head."""
        crps_sum, denom = mixture_crps(
            mdn_params, target, sample_weight=gene_weight,
        )
        return torch.where(
            denom > 0,
            crps_sum / denom.clamp(min=1e-8),
            crps_sum.new_tensor(0.0),
        )

    def _current_ramped_weight(self, beta_max: float) -> float:
        if beta_max <= 0.0:
            return 0.0

        epoch = int(self._current_epoch)
        if epoch < self.srl_start:
            return 0.0
        if self.srl_ramp <= 0:
            return float(beta_max)

        progress = min(max((epoch - self.srl_start) / float(self.srl_ramp), 0.0), 1.0)
        cosine_warmup = 0.5 * (1.0 - math.cos(math.pi * progress))
        return float(beta_max) * cosine_warmup

    def _current_srl_weight(self) -> float:
        return self._current_ramped_weight(self.beta_srl_max)

    def _current_dynamic_pair_weight(self) -> float:
        return self._current_ramped_weight(self.beta_dynamic_pair_max)

    def _current_pht_consistency_weight(self) -> float:
        return self._current_ramped_weight(self.beta_pht_consistency)

    def _current_pht_uncertainty_weight(self) -> float:
        return self._current_ramped_weight(self.beta_pht_uncertainty)

    @staticmethod
    def _item_weights(
        values: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if sample_weight is None:
            return torch.ones_like(values)

        weight = sample_weight
        if weight.dim() == 1:
            weight = weight.unsqueeze(-1)
        weight = weight.to(device=values.device, dtype=torch.float32).view(-1)
        if weight.shape[0] != values.shape[0]:
            raise ValueError(
                f"sample_weight length {weight.shape[0]} does not match batch size {values.shape[0]}"
            )
        return weight

    @staticmethod
    def _normalized_item_weights(
        values: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = MixtureDensityCRPSLoss._item_weights(values, sample_weight)

        masked_weight = torch.where(valid_mask, weight, torch.zeros_like(weight))
        n_valid = valid_mask.to(dtype=weight.dtype).sum().clamp(min=1.0)
        mean_weight = masked_weight.sum() / n_valid
        return weight / mean_weight.clamp(min=1e-8)

    def _standardized_targets(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        features: dict,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]]:
        if self._srl_gene_mean_map is None or self._srl_gene_std_map is None:
            raise RuntimeError(
                "Standardized residual loss requested but SRL gene mean/std maps have not been configured."
            )

        pred = pred.float().view(-1)
        target = target.float().view(-1)
        n = pred.shape[0]
        gene_names = MixtureDensityCRPSLoss._metadata_list(features, 'gene_name', n)
        if not gene_names:
            return None

        gene_mean = pred.new_tensor([
            self._srl_gene_mean_map.get(g, float('nan')) for g in gene_names
        ])
        gene_std = pred.new_tensor([
            self._srl_gene_std_map.get(g, float('nan')) for g in gene_names
        ])
        valid = (
            torch.isfinite(pred)
            & torch.isfinite(target)
            & torch.isfinite(gene_mean)
            & torch.isfinite(gene_std)
        )
        if not valid.any():
            return None

        denom = gene_std.clamp(min=SRL_STAT_EPS)
        z_pred = (pred - gene_mean) / denom
        z_target = (target - gene_mean) / denom
        return pred, target, valid, z_pred, z_target, gene_names

    def _standardized_residual_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        features: dict,
        sample_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        standardized = self._standardized_targets(pred, target, features)
        if standardized is None:
            return pred.float().view(-1).new_tensor(0.0)

        pred, _, valid, z_pred, z_target, _ = standardized
        huber = F.huber_loss(
            z_pred[valid],
            z_target[valid],
            reduction='none',
            delta=float(self.srl_delta),
        )

        normalized_weight = self._normalized_item_weights(pred, sample_weight, valid)
        valid_weight = normalized_weight[valid]
        weighted_huber = huber * valid_weight
        denom_weight = valid_weight.sum()
        return weighted_huber.sum() / denom_weight.clamp(min=1e-8)

    def _dynamic_pair_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        features: dict,
        sample_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        standardized = self._standardized_targets(pred, target, features)
        if standardized is None:
            return pred.float().view(-1).new_tensor(0.0)

        pred, _, valid, z_pred, z_target, gene_names = standardized
        source_names = MixtureDensityCRPSLoss._metadata_list(
            features, 'source_name', pred.shape[0],
        )
        if not source_names:
            return pred.new_tensor(0.0)

        item_weight = self._item_weights(pred, sample_weight)
        group_to_indices: Dict[Tuple[str, str], list[int]] = {}
        for idx in torch.nonzero(valid, as_tuple=False).view(-1).tolist():
            group_to_indices.setdefault(
                (gene_names[idx], source_names[idx]),
                [],
            ).append(idx)

        group_losses = []
        for member_indices in group_to_indices.values():
            if len(member_indices) < 2:
                continue

            group_idx = torch.tensor(member_indices, device=pred.device, dtype=torch.long)
            pair_i, pair_j = torch.triu_indices(
                group_idx.numel(), group_idx.numel(), offset=1, device=pred.device,
            )
            if pair_i.numel() == 0:
                continue

            group_z_pred = z_pred[group_idx]
            group_z_target = z_target[group_idx]
            group_weight = item_weight[group_idx].clamp(min=0.0)

            true_delta = group_z_target[pair_i] - group_z_target[pair_j]
            pair_resid = (
                (group_z_pred[pair_i] - group_z_pred[pair_j]) - true_delta
            )
            pair_weight = (
                torch.sqrt(torch.clamp(group_weight[pair_i] * group_weight[pair_j], min=0.0))
                * true_delta.abs().detach().clamp(max=2.0)
            )
            denom_weight = pair_weight.sum()

            pair_huber = F.huber_loss(
                pair_resid,
                torch.zeros_like(pair_resid),
                reduction='none',
                delta=float(self.srl_delta),
            )
            group_loss = (pair_weight * pair_huber).sum() / denom_weight.clamp(min=1e-8)
            group_losses.append(
                torch.where(
                    denom_weight > 0,
                    group_loss,
                    group_loss.new_tensor(float('nan')),
                )
            )

        if not group_losses:
            return pred.new_tensor(0.0)
        group_loss_tensor = torch.stack(group_losses)
        finite_group_losses = group_loss_tensor[torch.isfinite(group_loss_tensor)]
        if finite_group_losses.numel() == 0:
            return pred.new_tensor(0.0)
        return finite_group_losses.mean()

    @staticmethod
    def _metadata_list(features: dict, key: str, n: int) -> Optional[list]:
        values = features.get(key)
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            return [str(v) for v in values.detach().cpu().view(-1).tolist()]
        if isinstance(values, (list, tuple)):
            return [str(v) for v in values]
        return [str(values)] * n

    @staticmethod
    def _group_ids(values: list, device: torch.device) -> Optional[torch.Tensor]:
        if not values:
            return None
        group_ids = torch.empty(len(values), dtype=torch.long, device=device)
        id_map = {}
        next_id = 0
        for idx, value in enumerate(values):
            group_id = id_map.get(value)
            if group_id is None:
                group_id = next_id
                id_map[value] = group_id
                next_id += 1
            group_ids[idx] = group_id
        return group_ids

    @staticmethod
    def _group_variance_loss(values: torch.Tensor, group_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.view(-1).bool()
        if not valid.any():
            return values.new_tensor(0.0)

        valid_groups = group_ids[valid]
        valid_values = values[valid]
        group_count = int(valid_groups.max().detach().cpu().item()) + 1
        counts = values.new_zeros(group_count).scatter_add(
            0,
            valid_groups,
            values.new_ones((valid_groups.numel(),)),
        )
        sums = values.new_zeros(group_count).scatter_add(0, valid_groups, valid_values)
        means = sums / counts.clamp(min=1.0)

        centered = valid_values - means[valid_groups]
        sq_sums = values.new_zeros(group_count).scatter_add(0, valid_groups, centered.pow(2))

        eligible = counts >= 2.0
        eligible_count = eligible.to(dtype=values.dtype).sum()
        eligible_variance = torch.where(
            eligible,
            sq_sums / counts.clamp(min=1.0),
            torch.zeros_like(sq_sums),
        )
        return torch.where(
            eligible_count > 0,
            eligible_variance.sum() / eligible_count.clamp(min=1.0),
            values.new_tensor(0.0),
        )

    @staticmethod
    def _pht_losses(
        predictions: dict,
        targets: dict,
        features: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tumor_pred = predictions['tumor_exp'].float().view(-1)
        bg_pred = predictions['background_exp'].float().view(-1)
        n = tumor_pred.shape[0]
        device = tumor_pred.device

        is_pht = features.get('is_pht')
        if is_pht is None:
            zero = tumor_pred.new_tensor(0.0)
            return zero, zero
        if isinstance(is_pht, torch.Tensor):
            pht_mask = is_pht.to(device=device).view(-1).bool()
        else:
            pht_mask = torch.tensor([bool(v) for v in is_pht], device=device)
        if pht_mask.numel() != n or not pht_mask.any():
            zero = tumor_pred.new_tensor(0.0)
            return zero, zero

        genes = MixtureDensityCRPSLoss._metadata_list(features, 'gene_name', n)
        parents = MixtureDensityCRPSLoss._metadata_list(features, 'pht_parent_id', n)
        donors = MixtureDensityCRPSLoss._metadata_list(features, 'pht_donor_id', n)
        if not genes or not parents or not donors:
            zero = tumor_pred.new_tensor(0.0)
            return zero, zero

        parent_group_ids = MixtureDensityCRPSLoss._group_ids(
            list(zip(genes, parents)), device,
        )
        donor_group_ids = MixtureDensityCRPSLoss._group_ids(
            list(zip(genes, donors)), device,
        )
        if parent_group_ids is None or donor_group_ids is None:
            zero = tumor_pred.new_tensor(0.0)
            return zero, zero

        finite_tumor = pht_mask & torch.isfinite(targets['tumor_exp'].float().view(-1))
        finite_bg = pht_mask & torch.isfinite(targets['background_exp'].float().view(-1))
        tumor_consistency = MixtureDensityCRPSLoss._group_variance_loss(
            tumor_pred, parent_group_ids, finite_tumor,
        )
        background_consistency = MixtureDensityCRPSLoss._group_variance_loss(
            bg_pred, donor_group_ids, finite_bg,
        )
        consistency = tumor_consistency + background_consistency

        tfx = targets['TFX_label'].float().view(-1).to(device=device)
        uncertainty = mixture_variance(predictions['tumor_mdn']).view(-1)
        valid_tfx = pht_mask & torch.isfinite(tfx)
        if valid_tfx.sum() < 2:
            uncertainty_loss = tumor_pred.new_tensor(0.0)
        else:
            valid_group_ids = parent_group_ids[valid_tfx]
            valid_tfx_values = tfx[valid_tfx]
            valid_uncertainty = uncertainty[valid_tfx]
            m = int(valid_tfx_values.shape[0])

            group_eq = valid_group_ids.unsqueeze(1) == valid_group_ids.unsqueeze(0)
            upper = torch.triu(
                torch.ones((m, m), dtype=torch.bool, device=device),
                diagonal=1,
            )
            delta_tfx = valid_tfx_values.unsqueeze(1) - valid_tfx_values.unsqueeze(0)
            delta_uncertainty = (
                valid_uncertainty.unsqueeze(1) - valid_uncertainty.unsqueeze(0)
            )
            valid_pairs = group_eq & upper & (delta_tfx.abs() >= 0.05)
            if valid_pairs.any():
                pair_losses = F.relu(torch.sign(delta_tfx) * delta_uncertainty)
                uncertainty_loss = pair_losses[valid_pairs].mean()
            else:
                uncertainty_loss = tumor_pred.new_tensor(0.0)

        return consistency, uncertainty_loss


class BioDVNMDNLoss:
    """Combined external loss for BioDVN with MDN output heads.

    Total loss = analytical CRPS (tumor + background) + SRL/dynamic/PHT structure losses.
    The internal loss (KL + contrastive + reconstruction) is added by the
    training loop, not by this class.
    """

    def __init__(
        self,
        beta_background: float = 1.0,
        beta_srl_max: float = 1.5,
        beta_dynamic_pair_max: float = 0.0,
        srl_delta: float = 1.0,
        srl_start: int = 5,
        srl_ramp: int = 10,
        beta_pht_consistency: float = 0.5,
        beta_pht_uncertainty: float = 1.5,
    ):
        self.mdn_crps = MixtureDensityCRPSLoss(
            beta_background=beta_background,
            beta_srl_max=beta_srl_max,
            beta_dynamic_pair_max=beta_dynamic_pair_max,
            srl_delta=srl_delta,
            srl_start=srl_start,
            srl_ramp=srl_ramp,
            beta_pht_consistency=beta_pht_consistency,
            beta_pht_uncertainty=beta_pht_uncertainty,
        )

    def set_epoch(self, epoch: int) -> None:
        self.mdn_crps.set_epoch(epoch)

    def set_srl_stats(
        self,
        gene_mean_map: Dict[str, float],
        gene_std_map: Dict[str, float],
    ) -> None:
        self.mdn_crps.set_srl_stats(gene_mean_map, gene_std_map)

    def __call__(
        self,
        predictions: dict,
        targets: dict,
        sample_weight: Optional[torch.Tensor] = None,
        features: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        return self.mdn_crps(predictions, targets, sample_weight, features)

