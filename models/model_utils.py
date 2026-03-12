"""
model_utils.py (PyTorch)
Author: Robert D Patton, rpatton@fredhutch.org

Custom layers, loss functions, and metrics for BioDVN.
Includes MDN heads, analytical CRPS loss, β-TCVAE decomposition,
TFX gradient reversal, and masked regression losses.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hard upper bound on expression values in log2(CPM+1) space.
# Derived from the data ceiling; this will never be exceeded.
Y_MAX: float = 19.93

# Fixed Gamma shape parameter for the MDN mixture component.
# Determined by empirical fitting of RNA-seq expression distributions.
K_GAMMA: float = 0.80

# Upper bound on log(normal_sigma) in the MDN.  Caps the Normal
# component's scale at exp(LOG_SIGMA_CEIL) ≈ 5.0, which still allows
# a 90% CI spanning ±8.2 in log₂(CPM+1) space but prevents the
# over-dispersed calibration basin observed in phase 3.0.
LOG_SIGMA_CEIL: float = math.log(5.0)  # ≈ 1.609


# ---------------------------------------------------------------------------
# Gauss-Laguerre quadrature constants for the analytical CRPS.
# Used only for the gamma-normal cross-term integral (the single term in
# the mixture CRPS that lacks a closed form).  Everything else is exact.
# Precomputed once at import time; cached per device to avoid repeated
# CPU -> CUDA copies.
# ---------------------------------------------------------------------------
_QUAD_N: int = 32  # 32-node quadrature is exact for poly degree 63; ample for our smooth integrand

_gl_nodes_np, _gl_weights_np = np.polynomial.laguerre.laggauss(_QUAD_N)
_GL_NODES_CPU = torch.from_numpy(_gl_nodes_np.copy()).float()     # (Q,)
_GL_WEIGHTS_CPU = torch.from_numpy(_gl_weights_np.copy()).float() # (Q,)
# tᵢ^(k-1) where k = K_GAMMA = 0.80, so exponent = -0.20.
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
# Gradient Reversal Layer (for TFX-invariant latent learning)
# ==============================================================================


class _GradientReversalFn(torch.autograd.Function):
    """Autograd function: identity in forward, negated & scaled in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lam: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer (Ganin et al., 2016).

    Forward pass is the identity.  Backward pass multiplies the gradient
    by ``-lambda``, so the upstream encoder learns to produce representations
    that *fool* the downstream discriminator.

    Args:
        lam: Reversal strength multiplier (default 1.0).
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lam)


class TFXDiscriminator(nn.Module):
    """Small MLP that predicts TFX from concatenated tumor + background latents.

    Used with :class:`GradientReversalLayer` so the encoder is pushed to
    produce TFX-invariant latent representations while the discriminator
    itself still learns to predict TFX accurately (its forward path is
    normal; only the gradients flowing back through the GRL are negated).

    Architecture: ``concat(z_tumor_mean, z_bg_mean)`` → GRL → 2·L → 16 → 1

    Args:
        latent_size: Dimensionality of each latent vector (default 32).
        lam: GRL reversal strength (default 1.0).
    """

    def __init__(self, latent_size: int = 32, lam: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lam=lam)
        self.net = nn.Sequential(
            nn.Linear(2 * latent_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # TFX is in [0, 1]
        )

    def forward(self, z_tumor_mean: torch.Tensor, z_bg_mean: torch.Tensor) -> torch.Tensor:
        """Predict TFX from latents (with gradient reversal).

        Returns:
            predicted_tfx: (batch, 1)
        """
        h = torch.cat([z_tumor_mean, z_bg_mean], dim=1)
        h = self.grl(h)
        return self.net(h)


def _log_density_gaussian(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Log density of Gaussian distribution.
    
    Args:
        x: Sample point (batch, latent_dim)
        mu: Mean (batch, latent_dim)
        logvar: Log variance (batch, latent_dim)
    
    Returns:
        Log probability (batch, latent_dim)
    """
    # Clamp to prevent exp(-logvar) overflow: float32 max ~ 3.4e38 = exp(88).
    # Without this, very negative logvar produces exp(-logvar) = inf, and
    # (x - mu)^2 ≈ 0 (when z ≈ mu due to sampling underflow) gives 0*inf = NaN.
    logvar = logvar.clamp(-86.0, 86.0)
    norm = -0.5 * (math.log(2 * math.pi) + logvar)
    log_density = norm - 0.5 * ((x - mu).pow(2) * torch.exp(-logvar))
    return log_density


def total_correlation(z: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
    """
    Estimate total correlation (TC) using minibatch-weighted sampling.
    
    TC measures statistical dependence between latent dimensions.
    Based on β-TCVAE (Chen et al., 2018): Isolating Sources of Disentanglement in VAEs.
    
    TC = KL(q(z) || prod_j q(z_j)) where q(z_j) are marginal distributions
    
    Args:
        z: Sampled latent codes (batch, latent_dim)
        z_mean: Encoder mean (batch, latent_dim)
        z_log_var: Encoder log variance (batch, latent_dim)
    
    Returns:
        Estimated total correlation (scalar)
    """
    batch_size, latent_dim = z.shape
    
    # Compute log q(z) using minibatch weighted sampling
    # For each sample z_i, compute log q(z_i) ≈ log(1/N * sum_j q(z_i | x_j))
    # Expand dims: z (batch, 1, latent_dim), mu/logvar (1, batch, latent_dim)
    z_expand = z.unsqueeze(1)
    mu_expand = z_mean.unsqueeze(0)
    logvar_expand = z_log_var.unsqueeze(0)
    
    # log q(z_i | x_j) for all pairs, shape: (batch, batch, latent_dim)
    log_qz_ij = _log_density_gaussian(z_expand, mu_expand, logvar_expand)
    
    # log q(z) = logsumexp over j, minus log(N) for each dimension
    # Then sum over latent dimensions for joint log q(z)
    # Shape: (batch, latent_dim)
    log_qz_j = torch.logsumexp(log_qz_ij, dim=1) - math.log(batch_size)
    
    # log prod_j q(z_j) - treat dimensions as independent marginals
    log_prod_qzj = log_qz_j.sum(dim=1)
    
    # log q(z) joint - proper joint density using summed per-dim log densities
    log_qz_joint = torch.logsumexp(log_qz_ij.sum(dim=2), dim=1) - math.log(batch_size)
    
    tc = (log_qz_joint - log_prod_qzj).mean()
    # Clamp to prevent pathologically large TC from destabilising training.
    # Healthy TC is O(1–10); values above ~50 indicate near-degenerate
    # latent correlation and produce gradients that overwhelm grad clipping.
    return tc.clamp(-100.0, 100.0)


class InternalLossComputer:
    """
    Computes internal losses (KL, contrastive, reconstruction) for the VAE.
    Uses β-TCVAE objective: decomposes KL and upweights total correlation.
    """
    def __init__(
        self,
        beta_contrastive: float = 0.5,
        beta_tcvae: float = 3.0,
        beta_reconstruction: float = 1.5,
        beta_grl: float = 1.0,
        cl_margin: float = 0.5,
        internal_weight: float = 2.5,
    ):
        self.beta_contrastive = beta_contrastive
        self.beta_tcvae = beta_tcvae
        self.beta_reconstruction = beta_reconstruction
        self.beta_grl = beta_grl
        self.cl_margin = cl_margin
        self.internal_weight = internal_weight
    
    def _compute_vae_loss(
        self,
        z_tumor: torch.Tensor,
        z_tumor_mean: torch.Tensor,
        z_tumor_log_var: torch.Tensor,
        z_background: torch.Tensor,
        z_background_mean: torch.Tensor,
        z_background_log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute β-TCVAE regularization loss.
        
        Decomposes KL and upweights total correlation:
        loss = KL + (beta_tcvae − 1) × TC
        """
        # Standard KL (weight 1.0 for MI and dim-wise KL terms)
        l_kl_tumor = kl_divergence(z_tumor_mean, z_tumor_log_var)
        l_kl_background = kl_divergence(z_background_mean, z_background_log_var)
        
        # Total correlation (weight beta_tcvae)
        tc_tumor = total_correlation(z_tumor, z_tumor_mean, z_tumor_log_var)
        tc_background = total_correlation(z_background, z_background_mean, z_background_log_var)
        
        # Final loss: KL + (beta - 1) * TC (so TC is weighted by beta total)
        vae_loss = (l_kl_tumor + l_kl_background) + (self.beta_tcvae - 1.0) * (tc_tumor + tc_background)
        
        metrics = {
            'kl_loss': (l_kl_tumor + l_kl_background).detach(),
            'tc_loss': (tc_tumor + tc_background).detach(),
        }
        
        return vae_loss, metrics
    
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
        grl_loss: Optional[torch.Tensor] = None,
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
        
        # Compute VAE loss (β-TCVAE)
        vae_loss, vae_metrics = self._compute_vae_loss(
            z_tumor, z_tumor_mean, z_tumor_log_var,
            z_background, z_background_mean, z_background_log_var,
        )
        
        total_loss = (
            self.beta_contrastive * l_contrastive +
            vae_loss +
            self.beta_reconstruction * l_reconstruction
        )

        # Adversarial TFX-invariance loss (gradient reversal discriminator).
        if grl_loss is not None:
            grl_loss = grl_loss.float()
            total_loss = total_loss + self.beta_grl * grl_loss
        
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
        if grl_loss is not None:
            metrics['grl_loss'] = grl_loss.detach()
        
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
    
    def update(self, name: str, value: float, count: int = 1):
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0
        self.metrics[name] += value * count
        self.counts[name] += count
    
    def compute(self) -> dict:
        return {
            name: self.metrics[name] / max(self.counts[name], 1)
            for name in self.metrics
        }


# ==============================================================================
# Mixture Density Network (MDN) Components
# ==============================================================================

class MixtureDensityHead(nn.Module):
    """Output head that produces parameters for a 3-component mixture density.

    Components:
        1. Zero (hurdle) — true point mass at 0 (zero-expression genes)
        2. Gamma — Gamma(k=K_GAMMA, rate=beta), shape fixed, rate predicted
        3. Normal — N(mu, sigma^2), unconstrained Gaussian bulk

    Notes:
        - Targets are in RNA-Seq log2(CPM+1) space and are expected to be in [0, y_max].
        - Point-estimate predictions are clamped to [0, y_max] to respect the label domain.

    The head outputs:
        - log_pi: (batch, 3) log-softmax mixture weights [pi_zero, pi_gamma, pi_normal]
        - gamma_beta: (batch, 1) Gamma rate parameter beta > 0
        - normal_mu: (batch, 1) Normal location, clamped to [0, y_max]
        - normal_sigma: (batch, 1) Normal scale sigma in (0, exp(LOG_SIGMA_CEIL)]

    Args:
        input_dim: Dimension of the input feature vector.
        hidden_dim: Hidden layer width inside the head (default 64).
        final_dim: Width of the last hidden layer before the 6-output
            projection (default 16).  Intermediate layers halve from
            *hidden_dim* down to *final_dim*.  Example with
            hidden_dim=64, final_dim=16::

                input_dim → 64 → ReLU → 32 → ReLU → 16 → ReLU → 6
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

        # 3 logits + 1 log(gamma_beta) + 1 normal_mu + 1 log(normal_sigma) = 6 outputs
        self.proj = nn.Linear(dim, 6)

        # Xavier init + zero-bias on every Linear in the shared trunk and on
        # the projection, so that the initial mixture is roughly uniform and
        # component params are moderate.
        for module in self.shared:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, input_dim) feature tensor.

        Returns:
            Dictionary with keys:
                log_pi: (batch, 3) log-softmax of mixture weights
                gamma_beta: (batch, 1) Gamma rate (>0)
                normal_mu: (batch, 1) clamped to [0, y_max]
                normal_sigma: (batch, 1) (>0)
        """
        h = self.shared(x)
        out = self.proj(h)  # (batch, 6)

        logits = out[:, :3]          # mixture weight logits
        log_gamma_beta = out[:, 3:4]
        normal_mu_raw = out[:, 4:5]
        log_normal_sigma = out[:, 5:6]

        # Log/exp parameterization with clamping for numerical stability.
        gamma_beta = torch.exp(log_gamma_beta.clamp(-20.0, 20.0))
        normal_mu = normal_mu_raw.clamp(0.0, Y_MAX)
        normal_sigma = torch.exp(log_normal_sigma.clamp(-20.0, LOG_SIGMA_CEIL))

        log_pi = F.log_softmax(logits, dim=-1)  # (batch, 3)

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
# This gives 6 A/B terms (one A per component, six B entries in the
# symmetric 3×3 matrix).  Five are fully closed-form; only the
# gamma-normal cross term B_{gn} requires 1-D quadrature.
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
        The backward of ``gammainc(a, x)`` involves ``x^(a-1)``.  For
        k = 0.80 < 1 this exponent is −0.20, giving ``0^(−0.2) = ∞``
        when ``βy = 0`` (i.e. unexpressed genes with y = 0).  The chain
        rule then produces ``0 × ∞ = NaN``.  Clamping ``βy`` to a tiny
        positive epsilon avoids the singularity: the clamp gates the
        gradient to zero for y = 0 genes, and the forward value is
        negligibly biased (see inline comment).
    """
    # Clamp βy away from 0 to avoid the gammainc backward singularity
    # when k < 1.  At y = 0 the true value is A_g = k/β (the Gamma mean);
    # with eps = 1e-7 the formula gives k/β · (1 − 2·P(k+1, 1e-7)) ≈ k/β
    # (bias < 1e-5 relative), and the clamp zeros the gradient pathway
    # through P_k for y = 0 entries, which is the correct analytical
    # derivative (d(A_g)/d(β) = −k/β² when y = 0).
    by = (beta * y).clamp(min=1e-7)
    k_t = torch.tensor(K_GAMMA, device=y.device, dtype=y.dtype)
    k1_t = torch.tensor(K_GAMMA + 1.0, device=y.device, dtype=y.dtype)
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

    Term summary (k = K_GAMMA = 0.80):
        A₀(y)  = |y|                            (point mass — trivial)
        Aᵍ(y)  = y(2P(k,βy)−1) + (k/β)(1−2P(k+1,βy))   (closed form)
        Aₙ(y)  = σ[z(2Φ(z)−1) + 2φ(z)]         (closed form)
        B₀₀    = 0                               (both at zero)
        B₀ᵍ    = k/β                             (Gamma mean)
        B₀ₙ    = E|Xₙ| = Aₙ(0)                  (closed form)
        Bᵍᵍ    = 2/(β·B(k,½))                    (closed form)
        Bₙₙ    = 2σ/√π                           (closed form)
        Bᵍₙ    = E_{Xᵍ}[h(Xᵍ)]                  (1-D Gauss-Laguerre)

    All terms are differentiable.  The single quadrature term (Bᵍₙ) uses
    a 32-node Gauss-Laguerre rule whose nodes/weights are precomputed
    constants, so autodiff flows cleanly through the smooth integrand.

    Args:
        mdn_params: dict from ``MixtureDensityHead.forward()``.
        target: (batch, 1) ground-truth expression values.
        sample_weight: optional (batch, 1) or (batch,) per-gene weights.

    Returns:
        (crps_sum, denom): scalar tensors on the same device as *target*.
        ``crps_sum / denom`` gives the weighted-mean CRPS for the batch.
    """
    device = target.device
    mask = torch.isfinite(target)
    y = torch.where(mask, target, torch.zeros_like(target)).float()  # (batch, 1)

    # Ensure mdn_params are on the same device as target.
    mdn_params = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in mdn_params.items()}

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

    crps = a_sum - 0.5 * b_sum                 # (batch, 1)
    crps = torch.where(mask, crps, torch.zeros_like(crps))

    # ------------------------------------------------------------------
    # Sample-weight handling (identical to previous implementation)
    # ------------------------------------------------------------------
    if sample_weight is not None:
        w = sample_weight
        if w.dim() == 1:
            w = w.unsqueeze(-1)
        w = w.to(device=device, dtype=torch.float32)
        # Re-normalise weights so batch mean weight = 1.0.
        w_valid_sum = torch.where(mask, w, torch.zeros_like(w)).sum()
        n_valid_w = mask.to(dtype=w.dtype).sum().clamp(min=1)
        w_mean = w_valid_sum / n_valid_w
        w = w / w_mean.clamp(min=1e-8)

        crps = crps * w
        denom = torch.where(mask, w, torch.zeros_like(w)).sum()
    else:
        denom = mask.to(dtype=crps.dtype).sum()

    return crps.sum(), denom.clamp(min=1)


# ==============================================================================
# Loss classes
# ==============================================================================

class MixtureDensityCRPSLoss(nn.Module):
    """Masked, gene-weighted analytical CRPS loss for the 3-component mixture.

    Loss per head = weighted mean CRPS over valid observations.
    Optional auxiliary MSE on the mixture-mean point estimate.

    Args:
        beta_background: Weight on the background expression CRPS (default 1.0).
        beta_point_mse:  Weight on auxiliary MSE(mixture_mean, truth) (default 0.0 = off).
    """

    def __init__(self, beta_background: float = 1.0, beta_point_mse: float = 0.0):
        super().__init__()
        self.beta_background = beta_background
        self.beta_point_mse = beta_point_mse

    def forward(
        self,
        predictions: dict,
        targets: dict,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            predictions: dict with 'tumor_mdn' and 'background_mdn'.
            targets: dict with 'tumor_exp' (batch,1) and 'background_exp' (batch,1).
            sample_weight: optional (batch,1) or (batch,) per-gene weights.

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
            'tumor_crps': tumor_crps.item(),
            'background_crps': background_crps.item(),
            'total_external_loss': total_loss.item(),
        }

        # Auxiliary MSE on mixture-mean point estimates.
        # This gives direct backprop gradients through the mixture weights
        # and component parameters, pushing E[Y] toward truth at the
        # extremes where CRPS alone tends to under-/over-predict.
        if self.beta_point_mse > 0.0:
            tumor_mse = self._head_point_mse(
                predictions['tumor_exp'], targets['tumor_exp'], sample_weight
            )
            bg_mse = self._head_point_mse(
                predictions['background_exp'], targets['background_exp'], sample_weight
            )
            point_mse = tumor_mse + self.beta_background * bg_mse
            total_loss = total_loss + self.beta_point_mse * point_mse
            loss_dict['tumor_point_mse'] = tumor_mse.item()
            loss_dict['background_point_mse'] = bg_mse.item()
            loss_dict['total_external_loss'] = total_loss.item()

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
        return crps_sum / denom

    @staticmethod
    def _head_point_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        gene_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Weighted MSE between mixture-mean point estimate and truth."""
        pred = pred.float().view(-1)
        target = target.float().view(-1)
        mask = torch.isfinite(target)
        if not mask.any():
            return pred.new_tensor(0.0)
        p = pred[mask]
        t = target[mask]
        sq_err = (p - t).pow(2)
        if gene_weight is not None:
            w = gene_weight.float().view(-1)
            w = w[mask] if w.shape[0] == mask.shape[0] else w[:sq_err.shape[0]]
            # Renormalise so mean weight = 1 (same convention as CRPS).
            w = w / (w.mean() + 1e-12)
            sq_err = sq_err * w
        return sq_err.mean()


class BioDVNMDNLoss:
    """Combined external loss for BioDVN with MDN output heads.

    Total loss = Analytical CRPS (tumor + background) + optional point MSE.
    The internal loss (KL + contrastive + reconstruction) is added by the
    training loop, not by this class.
    """

    def __init__(
        self,
        beta_background: float = 1.0,
        beta_point_mse: float = 0.0,
    ):
        self.mdn_crps = MixtureDensityCRPSLoss(
            beta_background=beta_background,
            beta_point_mse=beta_point_mse,
        )

    def __call__(
        self,
        predictions: dict,
        targets: dict,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        return self.mdn_crps(predictions, targets, sample_weight)

