"""Parameter suggestion, config resolution, and fixed-param replay.

This module converts an ``ExperimentSpec`` and an Optuna trial (or a
fixed parameter dict) into ``BioDVNConfig`` + ``TrainingConfig`` pairs
ready for training.  It also provides geometry guards that prune
physically impossible hyperparameter combinations before any GPU time
is spent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

import optuna

from config.experiment_spec import ExperimentSpec, ExperimentSpecError, ParamSpec
from models import BioDVNConfig
from proteus_utils.trainer import TrainingConfig

__all__ = [
    "suggest_from_param",
    "resolve_trial_configs",
    "FixedParamTrial",
    "config_hash",
]


# ---------------------------------------------------------------------------
# Single-parameter suggestion
# ---------------------------------------------------------------------------

def suggest_from_param(trial: optuna.Trial, name: str, spec: ParamSpec) -> Any:
    """Suggest a value from Optuna or return the fixed value."""
    if spec.fixed is not None or spec.suggest is None:
        return spec.fixed

    s = dict(spec.suggest)
    stype = str(s.get("type", "")).strip().lower()

    if stype == "categorical":
        choices = s.get("choices", None)
        if not isinstance(choices, list) or not choices:
            raise ExperimentSpecError(f"Param '{name}': categorical requires non-empty list 'choices'")
        return trial.suggest_categorical(name, choices)

    if stype == "int":
        low = s.get("low", None)
        high = s.get("high", None)
        step = s.get("step", 1)
        if low is None or high is None:
            raise ExperimentSpecError(f"Param '{name}': int requires low/high")
        return trial.suggest_int(name, int(low), int(high), step=int(step))

    if stype == "float":
        low = s.get("low", None)
        high = s.get("high", None)
        log = bool(s.get("log", False))
        step = s.get("step", None)
        if low is None or high is None:
            raise ExperimentSpecError(f"Param '{name}': float requires low/high")
        if step is not None:
            # step and log are mutually exclusive in Optuna
            return trial.suggest_float(name, float(low), float(high), step=float(step))
        return trial.suggest_float(name, float(low), float(high), log=log)

    raise ExperimentSpecError(
        f"Param '{name}': unsupported suggest.type='{stype}'. Expected categorical|int|float"
    )


# ---------------------------------------------------------------------------
# Full config resolution
# ---------------------------------------------------------------------------

def resolve_trial_configs(
    exp: ExperimentSpec, trial: optuna.Trial
) -> Tuple[BioDVNConfig, TrainingConfig, Dict[str, Any]]:
    """Resolve experiment spec parameters for a single Optuna trial.

    Each parameter is either fixed (plain value in YAML) or searched
    (has a ``suggest`` block).  There is no group-level gating — every
    parameter that carries a ``suggest`` block will be suggested by
    Optuna, regardless of which YAML section it lives in.
    """

    # Helper aliases for brevity — prefixed names keep Optuna's
    # parameter namespace tidy.
    def s(name: str, spec: ParamSpec) -> Any:
        """Structural group parameter."""
        return suggest_from_param(trial, f"structural.{name}", spec)

    def t(name: str, spec: ParamSpec) -> Any:
        """Training group parameter."""
        return suggest_from_param(trial, f"training.{name}", spec)

    def sc(name: str, spec: ParamSpec) -> Any:
        """Scale group parameter."""
        return suggest_from_param(trial, f"scale.{name}", spec)

    # =====================================================================
    # Structural group
    # =====================================================================
    beta_background = s("beta_background", exp.structural.beta_background)
    beta_kl = s("beta_kl", exp.structural.beta_kl)
    beta_contrastive = s("beta_contrastive", exp.structural.beta_contrastive)
    beta_reconstruction = s("beta_reconstruction", exp.structural.beta_reconstruction)
    cl_margin = s("cl_margin", exp.structural.cl_margin)
    internal_weight = s("internal_weight", exp.structural.internal_weight)
    beta_tfx = s("beta_tfx", exp.structural.beta_tfx)
    beta_srl_max = s("beta_srl_max", exp.structural.beta_srl_max)
    beta_dynamic_pair_max = s(
        "beta_dynamic_pair_max", exp.structural.beta_dynamic_pair_max,
    )
    srl_delta = s("srl_delta", exp.structural.srl_delta)
    srl_start = s("srl_start", exp.structural.srl_start)
    srl_ramp = s("srl_ramp", exp.structural.srl_ramp)
    beta_pht_consistency = s("beta_pht_consistency", exp.structural.beta_pht_consistency)
    beta_pht_uncertainty = s("beta_pht_uncertainty", exp.structural.beta_pht_uncertainty)

    # =====================================================================
    # Training group
    # =====================================================================
    max_lr = t("max_lr", exp.training.max_lr)
    min_lr_ratio = t("min_lr_ratio", exp.training.min_lr_ratio)
    warmup_start_lr = t("warmup_start_lr", exp.training.warmup_start_lr)
    weight_decay = t("weight_decay", exp.training.weight_decay)
    max_grad_norm = t("max_grad_norm", exp.training.max_grad_norm)
    warmup_frac = t("warmup_frac", exp.training.warmup_frac)
    schedule_frac = t("schedule_frac", exp.training.schedule_frac)
    epochs = t("epochs", exp.training.epochs)
    patience = t("patience", exp.training.patience)
    use_amp = t("use_amp", exp.training.use_amp)
    ema_decay = t("ema_decay", exp.training.ema_decay)
    grad_accum_steps = t("grad_accum_steps", exp.training.grad_accum_steps)

    # =====================================================================
    # Scale group
    # =====================================================================
    encoder_start = sc("encoder_start", exp.scale.encoder_start)
    latent_size = sc("latent_size", exp.scale.latent_size)
    decoder_start = sc("decoder_start", exp.scale.decoder_start)
    mdn_hidden = sc("mdn_hidden", exp.scale.mdn_hidden)
    mdn_final = sc("mdn_final", exp.scale.mdn_final)
    signal_filters = sc("signal_filters", exp.scale.signal_filters)
    signal_ksize = sc("signal_ksize", exp.scale.signal_ksize)
    interaction_ksize = sc("interaction_ksize", exp.scale.interaction_ksize)
    initial_reduction_channels = sc("initial_reduction_channels", exp.scale.initial_reduction_channels)
    n_reductions = sc("n_reductions", exp.scale.n_reductions)
    funnel_reduction_factor = sc("funnel_reduction_factor", exp.scale.funnel_reduction_factor)
    reduction_dilation = sc("reduction_dilation", exp.scale.reduction_dilation)
    dropout_rate = sc("dropout_rate", exp.scale.dropout_rate)

    # Geometry guard: prune combos that produce non-positive sequence
    # lengths in the reduction blocks before constructing the model.
    # Mirrors BioDVN._compute_reduction_output_length logic.
    _tss_len = 2000
    _base_len = _tss_len - int(signal_ksize) + 1
    if _base_len <= 0:
        raise optuna.TrialPruned(
            f"signal_ksize={signal_ksize} produces non-positive base length"
        )
    _ksize = int(interaction_ksize)
    _dil = int(reduction_dilation)
    _out = _base_len
    for _ in range(int(n_reductions)):
        _out = _out - _dil * (_ksize - 1)
        _out = _out // 2
    if _out <= 0:
        raise optuna.TrialPruned(
            f"Reduction produces non-positive seq length {_out} "
            f"(n_reductions={int(n_reductions)}, ksize={_ksize}, dilation={_dil})"
        )

    # =====================================================================
    # Build configs
    # =====================================================================
    model_cfg = BioDVNConfig(
        # Scale
        encoder_start=int(encoder_start),
        latent_size=int(latent_size),
        decoder_start=int(decoder_start),
        mdn_hidden=int(mdn_hidden),
        mdn_final=int(mdn_final),
        signal_filters=int(signal_filters),
        signal_ksize=int(signal_ksize),
        interaction_ksize=int(interaction_ksize),
        initial_reduction_channels=int(initial_reduction_channels),
        n_reductions=int(n_reductions),
        funnel_reduction_factor=int(funnel_reduction_factor),
        reduction_dilation=int(reduction_dilation),
        # Structural
        beta_background=float(beta_background),
        beta_kl=float(beta_kl),
        beta_contrastive=float(beta_contrastive),
        beta_reconstruction=float(beta_reconstruction),
        cl_margin=float(cl_margin),
        internal_weight=float(internal_weight),
        beta_tfx=float(beta_tfx),
        beta_srl_max=float(beta_srl_max),
        beta_dynamic_pair_max=float(beta_dynamic_pair_max),
        srl_delta=float(srl_delta),
        srl_start=int(srl_start),
        srl_ramp=int(srl_ramp),
        beta_pht_consistency=float(beta_pht_consistency),
        beta_pht_uncertainty=float(beta_pht_uncertainty),
        # Training HPs stored in model config for downstream reconstruction.
        dropout_rate=float(dropout_rate),
        weight_decay=float(weight_decay),
        max_lr=float(max_lr),
        max_grad_norm=float(max_grad_norm),
    )

    train_cfg = TrainingConfig(
        checkpoint_dir="",  # Populated per trial
        max_lr=float(max_lr),
        min_lr_ratio=float(min_lr_ratio),
        warmup_start_lr=float(warmup_start_lr),
        weight_decay=float(weight_decay),
        max_grad_norm=float(max_grad_norm),
        warmup_frac=float(warmup_frac),
        schedule_frac=float(schedule_frac),
        epochs=int(epochs),
        patience=int(patience),
        use_amp=bool(use_amp),
        ema_decay=float(ema_decay),
        grad_accum_steps=int(grad_accum_steps),
        selection_metric=str(exp.data.selection_metric),
        base_weights=exp.transfer.base_weights,
    )

    # =====================================================================
    # Trial config snapshot (for logging / worker reconstruction)
    # =====================================================================
    trial_cfg = {
        "structural": {
            "beta_background": float(beta_background),
            "beta_kl": float(beta_kl),
            "beta_contrastive": float(beta_contrastive),
            "beta_reconstruction": float(beta_reconstruction),
            "cl_margin": float(cl_margin),
            "internal_weight": float(internal_weight),
            "beta_tfx": float(beta_tfx),
            "beta_srl_max": float(beta_srl_max),
            "beta_dynamic_pair_max": float(beta_dynamic_pair_max),
            "srl_delta": float(srl_delta),
            "srl_start": int(srl_start),
            "srl_ramp": int(srl_ramp),
            "beta_pht_consistency": float(beta_pht_consistency),
            "beta_pht_uncertainty": float(beta_pht_uncertainty),
        },
        "training": {
            "max_lr": float(max_lr),
            "min_lr_ratio": float(min_lr_ratio),
            "warmup_start_lr": float(warmup_start_lr),
            "weight_decay": float(weight_decay),
            "max_grad_norm": float(max_grad_norm),
            "warmup_frac": float(warmup_frac),
            "schedule_frac": float(schedule_frac),
            "epochs": int(epochs),
            "patience": int(patience),
            "use_amp": bool(use_amp),
            "ema_decay": float(ema_decay),
            "grad_accum_steps": int(grad_accum_steps),
        },
        "scale": {
            "encoder_start": int(encoder_start),
            "latent_size": int(latent_size),
            "decoder_start": int(decoder_start),
            "mdn_hidden": int(mdn_hidden),
            "mdn_final": int(mdn_final),
            "signal_filters": int(signal_filters),
            "signal_ksize": int(signal_ksize),
            "interaction_ksize": int(interaction_ksize),
            "initial_reduction_channels": int(initial_reduction_channels),
            "n_reductions": int(n_reductions),
            "funnel_reduction_factor": int(funnel_reduction_factor),
            "reduction_dilation": int(reduction_dilation),
            "dropout_rate": float(dropout_rate),
        },
        "transfer": {
            "base_weights": exp.transfer.base_weights,
        },
    }

    return model_cfg, train_cfg, trial_cfg


# ---------------------------------------------------------------------------
# Config hashing & fixed-param replay
# ---------------------------------------------------------------------------

def config_hash(params: Dict[str, Any]) -> str:
    """Deterministic 16-char hex hash of a parameter dict."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class FixedParamTrial:
    """Minimal mock of ``optuna.Trial`` that replays fixed parameters.

    Used by the promotion system to feed a recorded parameter dict
    through ``resolve_trial_configs`` without going through the Optuna
    suggest machinery.
    """

    def __init__(self, params: Dict[str, Any]):
        self._params = dict(params)
        self.number = -1  # sentinel for log messages

    # The three suggest_* signatures mirror what suggest_from_param calls.
    def suggest_float(self, name: str, low: float = 0.0, high: float = 1.0,
                      *, step: Optional[float] = None,
                      log: bool = False) -> float:
        return float(self._params[name])

    def suggest_int(self, name: str, low: int = 0, high: int = 1,
                    *, step: int = 1, log: bool = False) -> int:
        return int(self._params[name])

    def suggest_categorical(self, name: str, choices: list) -> Any:
        return self._params[name]


def configs_from_trial_json(
    trial_cfg: Dict[str, Any], exp: ExperimentSpec,
) -> Tuple[BioDVNConfig, TrainingConfig]:
    """Reconstruct model and training configs from a saved trial_config.json.

    Replaces the duplicated config-building logic that was previously
    inlined in ``run_coordinated_fold``.
    """
    s = trial_cfg["structural"]
    t = trial_cfg["training"]
    sc = trial_cfg["scale"]
    tr = trial_cfg.get("transfer", {})

    model_cfg = BioDVNConfig(
        encoder_start=int(sc["encoder_start"]),
        latent_size=int(sc["latent_size"]),
        decoder_start=int(sc["decoder_start"]),
        mdn_hidden=int(sc["mdn_hidden"]),
        mdn_final=int(sc["mdn_final"]),
        signal_filters=int(sc["signal_filters"]),
        signal_ksize=int(sc["signal_ksize"]),
        interaction_ksize=int(sc["interaction_ksize"]),
        initial_reduction_channels=int(sc["initial_reduction_channels"]),
        n_reductions=int(sc["n_reductions"]),
        funnel_reduction_factor=int(sc.get("funnel_reduction_factor", 4)),
        reduction_dilation=int(sc["reduction_dilation"]),
        beta_background=float(s["beta_background"]),
        beta_kl=float(s["beta_kl"]),
        beta_contrastive=float(s["beta_contrastive"]),
        beta_reconstruction=float(s["beta_reconstruction"]),
        cl_margin=float(s["cl_margin"]),
        internal_weight=float(s["internal_weight"]),
        beta_tfx=float(s["beta_tfx"]),
        beta_srl_max=float(s.get("beta_srl_max", 2.0)),
        beta_dynamic_pair_max=float(s.get("beta_dynamic_pair_max", 0.0)),
        srl_delta=float(s.get("srl_delta", 5.0)),
        srl_start=int(s.get("srl_start", 0)),
        srl_ramp=int(s.get("srl_ramp", 50)),
        beta_pht_consistency=float(s["beta_pht_consistency"]),
        beta_pht_uncertainty=float(s["beta_pht_uncertainty"]),
        dropout_rate=float(sc["dropout_rate"]),
        weight_decay=float(t["weight_decay"]),
        max_lr=float(t["max_lr"]),
        max_grad_norm=float(t["max_grad_norm"]),
    )

    train_cfg = TrainingConfig(
        checkpoint_dir="",
        max_lr=float(t["max_lr"]),
        min_lr_ratio=float(t["min_lr_ratio"]),
        warmup_start_lr=float(t["warmup_start_lr"]),
        weight_decay=float(t["weight_decay"]),
        max_grad_norm=float(t["max_grad_norm"]),
        warmup_frac=float(t["warmup_frac"]),
        schedule_frac=float(t["schedule_frac"]),
        epochs=int(t["epochs"]),
        patience=int(t["patience"]),
        use_amp=bool(t["use_amp"]),
        ema_decay=float(t["ema_decay"]),
        grad_accum_steps=int(t.get("grad_accum_steps", 4)),
        selection_metric=str(exp.data.selection_metric),
        base_weights=tr.get("base_weights") or exp.transfer.base_weights,
    )

    return model_cfg, train_cfg
