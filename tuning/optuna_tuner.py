"""Optuna-based CV tuner for Proteus/BioDVN.

This module integrates with the three-group experiment_spec.py format:
- Structural HPs: architecture, loss function, gate regularisation
- Training HPs:   optimizer, schedule, regularisation
- Scale HPs:      width, depth, capacity

Each fold runs independently as a separate Optuna trial via
study.optimize(), with constant_liar for coordination.  Phase
transition logic narrows ranges from a previous study.

Usage (parallel by fold, one job per fold):
    python Proteus.py --run_mode tune --experiment config/experiments/biodvn_phase1_training.yaml --test_fold 1
    python Proteus.py --run_mode tune --experiment config/experiments/biodvn_phase1_training.yaml --test_fold 2
    ...

Usage (sequential, all folds per trial):
    python Proteus.py --run_mode tune --experiment config/experiments/biodvn_phase1_training.yaml
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sys
import time
import traceback
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import optuna

# Suppress Optuna experimental feature warnings (multivariate, constant_liar are stable in practice)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from config.experiment_spec import (
    ExperimentSpec,
    ExperimentSpecError,
    ParamSpec,
    load_experiment_spec,
)
from models import BioDVNConfig
from proteus_utils.trainer import TrainingConfig


# =============================================================================
# Hyperband / Phased Search Configuration
# =============================================================================


def _make_storage(url: str) -> Any:
    """Wrap a storage URL for better concurrent-write handling.

    SQLite can only handle one writer at a time.  With many parallel
    workers the default 5-second busy timeout is too short and causes
    ``database is locked`` errors.  This helper creates an
    ``RDBStorage`` with a 300-second timeout so workers wait rather
    than crash.  Non-SQLite URLs are returned unchanged.
    """
    if url.startswith("sqlite"):
        return optuna.storages.RDBStorage(
            url=url,
            engine_kwargs={
                "connect_args": {"timeout": 300},
                "pool_pre_ping": True,
            },
        )
    return url

HYPERBAND_MIN_RESOURCE = 3
HYPERBAND_REDUCTION_FACTOR = 3


def _min_patience_from_spec(spec: ParamSpec, default: int = 3) -> int:
    """Extract the minimum patience value from a ParamSpec.

    Used to set pruner warmup floors so Optuna never prunes a trial
    before it has run for at least ``patience`` epochs.
    """
    if spec.fixed is not None:
        return int(spec.fixed)
    if spec.suggest is not None:
        s = dict(spec.suggest)
        stype = str(s.get("type", "")).strip().lower()
        if stype == "int" and s.get("low") is not None:
            return int(s["low"])
        if stype == "categorical" and isinstance(s.get("choices"), list) and s["choices"]:
            return int(min(s["choices"]))
    return default


def _infer_hyperband_max_resource(exp: ExperimentSpec) -> int:
    """Infer max_resource for Hyperband from experiment.training.epochs."""
    epochs_spec = exp.training.epochs
    if epochs_spec.fixed is not None:
        return int(epochs_spec.fixed)  # type: ignore[arg-type]
    if epochs_spec.suggest is not None:
        s = dict(epochs_spec.suggest)
        stype = str(s.get('type', '')).strip().lower()
        if stype == 'int' and s.get('high', None) is not None:
            return int(s['high'])
        if stype == 'categorical' and isinstance(s.get('choices'), list) and s['choices']:
            return int(max(s['choices']))
    return 30


def _folds_from_tsv(training_set_tsv: str) -> List[int]:
    import csv
    folds: set[int] = set()
    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            folds.add(int(row['Fold']))
    return sorted(folds)


def _write_json_atomic(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _suggest_from_param(trial: optuna.Trial, name: str, spec: ParamSpec) -> Any:
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


# =============================================================================
# Duplicate-trial guard
# =============================================================================

_DEDUP_MAX_RETRIES = 100  # safety cap to avoid infinite loops


def _is_duplicate_trial(study: optuna.Study, trial: optuna.Trial) -> bool:
    """Return True if an earlier trial in the study has identical params."""
    for t in study.trials:
        if t.number < trial.number and t.params == trial.params:
            return True
    return False


# =============================================================================
# Config Resolution
# =============================================================================

def _resolve_trial_configs(
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
        return _suggest_from_param(trial, f"structural.{name}", spec)

    def t(name: str, spec: ParamSpec) -> Any:
        """Training group parameter."""
        return _suggest_from_param(trial, f"training.{name}", spec)

    def sc(name: str, spec: ParamSpec) -> Any:
        """Scale group parameter."""
        return _suggest_from_param(trial, f"scale.{name}", spec)

    # =====================================================================
    # Structural group
    # =====================================================================
    beta_background = s("beta_background", exp.structural.beta_background)
    beta_tcvae = s("beta_tcvae", exp.structural.beta_tcvae)
    beta_contrastive = s("beta_contrastive", exp.structural.beta_contrastive)
    beta_reconstruction = s("beta_reconstruction", exp.structural.beta_reconstruction)
    cl_margin = s("cl_margin", exp.structural.cl_margin)
    internal_weight = s("internal_weight", exp.structural.internal_weight)
    beta_grl = s("beta_grl", exp.structural.beta_grl)
    beta_point_mse = s("beta_point_mse", exp.structural.beta_point_mse)

    # β-TCVAE guard: the TC loss term is weighted by (beta_tcvae − 1).  When
    # beta_tcvae < 1 this coefficient is *negative*, which rewards correlated
    # latents instead of penalising them — creating a runaway feedback loop
    # (TC escalates to 200+ in a few hundred steps, then NaN).  Prune these
    # trials before wasting GPU time.
    if float(beta_tcvae) < 1.0:
        raise optuna.TrialPruned(
            f"beta_tcvae={float(beta_tcvae):.4f} < 1.0 produces "
            f"a negative TC coefficient ({float(beta_tcvae) - 1:.4f}), which "
            f"rewards latent correlation and leads to divergence."
        )

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
    # Transfer learning group
    # =====================================================================
    freeze_mode = _suggest_from_param(
        trial, "transfer.freeze_mode", exp.transfer.freeze_mode,
    )
    backbone_lr_multiplier = _suggest_from_param(
        trial, "transfer.backbone_lr_multiplier",
        exp.transfer.backbone_lr_multiplier,
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
        reduction_dilation=int(reduction_dilation),
        # Structural
        beta_background=float(beta_background),
        beta_tcvae=float(beta_tcvae),
        beta_contrastive=float(beta_contrastive),
        beta_reconstruction=float(beta_reconstruction),
        cl_margin=float(cl_margin),
        internal_weight=float(internal_weight),
        beta_grl=float(beta_grl),
        beta_point_mse=float(beta_point_mse),
        # Training HPs stored in model config (legacy coupling)
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
        base_weights=exp.transfer.base_weights,
        freeze_level=exp.transfer.freeze_level,
        freeze_mode=str(freeze_mode) if freeze_mode is not None else None,
        backbone_lr_multiplier=(
            float(backbone_lr_multiplier)
            if backbone_lr_multiplier is not None else 1.0
        ),
    )

    # =====================================================================
    # Trial config snapshot (for logging / worker reconstruction)
    # =====================================================================
    trial_cfg = {
        "structural": {
            "beta_background": float(beta_background),
            "beta_tcvae": float(beta_tcvae),
            "beta_contrastive": float(beta_contrastive),
            "beta_reconstruction": float(beta_reconstruction),
            "cl_margin": float(cl_margin),
            "internal_weight": float(internal_weight),
            "beta_grl": float(beta_grl),
            "beta_point_mse": float(beta_point_mse),
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
            "reduction_dilation": int(reduction_dilation),
            "dropout_rate": float(dropout_rate),
        },
        "transfer": {
            "base_weights": exp.transfer.base_weights,
            "freeze_level": exp.transfer.freeze_level,
            "freeze_mode": str(freeze_mode) if freeze_mode is not None else None,
            "backbone_lr_multiplier": (
                float(backbone_lr_multiplier)
                if backbone_lr_multiplier is not None else 1.0
            ),
        },
    }

    return model_cfg, train_cfg, trial_cfg


def _collect_fold_metrics(run_dir: str) -> List[Dict[str, Any]]:
    """Collect metrics from all CV fold directories."""
    out: List[Dict[str, Any]] = []
    for entry in sorted(os.listdir(run_dir)):
        if not entry.startswith("CV-fold_"):
            continue
        fold_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(fold_dir):
            continue
        try:
            fold_num = int(entry.split("_")[-1])
        except ValueError:
            continue
        metrics_path = os.path.join(fold_dir, f"validation-fold_{fold_num}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                out.append(json.load(f))
    return out


def _nanmean(xs: List[float]) -> float:
    vals = [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(sum(float(x) for x in vals) / len(vals))


def _nanstd(xs: List[float]) -> float:
    vals = [float(x) for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return float(math.sqrt(var))


def compute_cv_objective(run_dir: str) -> Dict[str, Any]:
    """Compute CV objective from fold metrics."""
    rows = _collect_fold_metrics(run_dir)
    if not rows:
        raise RuntimeError(f"No fold metrics found under: {run_dir}")

    tumor_mae = [float(r.get("tumor", {}).get("mae")) for r in rows if r.get("tumor", {}).get("mae") is not None]
    tumor_r2 = [float(r.get("tumor", {}).get("r2")) for r in rows if r.get("tumor", {}).get("r2") is not None]
    tumor_gvw_crps = [
        float(r.get("tumor", {}).get("gvw_crps"))
        for r in rows
        if r.get("tumor", {}).get("gvw_crps") is not None
    ]
    bg_mae = [float(r.get("background", {}).get("mae")) for r in rows if r.get("background", {}).get("mae") is not None]
    bg_r2 = [float(r.get("background", {}).get("r2")) for r in rows if r.get("background", {}).get("r2") is not None]
    bg_gvw_crps = [
        float(r.get("background", {}).get("gvw_crps"))
        for r in rows
        if r.get("background", {}).get("gvw_crps") is not None
    ]

    obj = {
        "folds_with_metrics": len(rows),
        "tumor_mae_mean": _nanmean(tumor_mae),
        "tumor_mae_std": _nanstd(tumor_mae),
        "tumor_r2_mean": _nanmean(tumor_r2),
        "tumor_r2_std": _nanstd(tumor_r2),
        "tumor_gvw_crps_mean": _nanmean(tumor_gvw_crps),
        "tumor_gvw_crps_std": _nanstd(tumor_gvw_crps),
        "background_mae_mean": _nanmean(bg_mae),
        "background_mae_std": _nanstd(bg_mae),
        "background_r2_mean": _nanmean(bg_r2),
        "background_r2_std": _nanstd(bg_r2),
        "background_gvw_crps_mean": _nanmean(bg_gvw_crps),
        "background_gvw_crps_std": _nanstd(bg_gvw_crps),
        # Combined objective: average of tumor and background MAE
        "val_loss": _nanmean([_nanmean(tumor_mae), _nanmean(bg_mae)]),
    }

    # Balanced fine-tuning objective: per-fold balanced_tumor_gvw_crps
    # (equal-weight average of per-source GVW-CRPS when source_groups is
    # active, or per-cohort GVW-CRPS otherwise; computed in
    # _save_fold_predictions).
    balanced_crps = [
        float(r["balanced_tumor_gvw_crps"])
        for r in rows
        if r.get("balanced_tumor_gvw_crps") is not None
    ]
    if balanced_crps:
        obj["balanced_tumor_gvw_crps_mean"] = _nanmean(balanced_crps)
        obj["balanced_tumor_gvw_crps_std"] = _nanstd(balanced_crps)

    # Balanced GVW-r: per-fold balanced_tumor_gvw_r
    # (equal-weight average of per-source GVW-r when source_groups is
    # active, or per-cohort GVW-r otherwise).
    balanced_gvw_r = [
        float(r["balanced_tumor_gvw_r"])
        for r in rows
        if r.get("balanced_tumor_gvw_r") is not None
    ]
    if balanced_gvw_r:
        obj["balanced_tumor_gvw_r_mean"] = _nanmean(balanced_gvw_r)
        obj["balanced_tumor_gvw_r_std"] = _nanstd(balanced_gvw_r)

    # Per-fold balanced_tumor_gvw_crps_std (stdev across cohorts within each fold)
    balanced_crps_std = [
        float(r["balanced_tumor_gvw_crps_std"])
        for r in rows
        if r.get("balanced_tumor_gvw_crps_std") is not None
    ]
    if balanced_crps_std:
        obj["balanced_tumor_gvw_crps_std_mean"] = _nanmean(balanced_crps_std)

    # Per-fold balanced_tumor_gvw_r_std (stdev across cohorts within each fold)
    balanced_gvw_r_std = [
        float(r["balanced_tumor_gvw_r_std"])
        for r in rows
        if r.get("balanced_tumor_gvw_r_std") is not None
    ]
    if balanced_gvw_r_std:
        obj["balanced_tumor_gvw_r_std_mean"] = _nanmean(balanced_gvw_r_std)

    # Composite lower-bound / upper-bound objectives
    # GVW-r lower bound: mean − std (maximize to raise worst-source floor)
    _r_m = obj.get("balanced_tumor_gvw_r_mean")
    _r_s = obj.get("balanced_tumor_gvw_r_std")
    if (_r_m is not None and _r_s is not None
            and math.isfinite(float(_r_m)) and math.isfinite(float(_r_s))):
        obj["balanced_tumor_gvw_r_lower_bound"] = float(_r_m) - float(_r_s)
    # GVW-CRPS upper bound: mean + std (minimize to lower worst-source ceiling)
    _c_m = obj.get("balanced_tumor_gvw_crps_mean")
    _c_s = obj.get("balanced_tumor_gvw_crps_std")
    if (_c_m is not None and _c_s is not None
            and math.isfinite(float(_c_m)) and math.isfinite(float(_c_s))):
        obj["balanced_tumor_gvw_crps_upper_bound"] = float(_c_m) + float(_c_s)

    # Min-source GVW-r: per-fold minimum across sources, averaged over folds.
    min_r_per_fold = [
        float(r["min_tumor_gvw_r"])
        for r in rows
        if r.get("min_tumor_gvw_r") is not None
           and math.isfinite(float(r["min_tumor_gvw_r"]))
    ]
    if min_r_per_fold:
        obj["min_tumor_gvw_r"] = _nanmean(min_r_per_fold)

    out_path = os.path.join(run_dir, "cv_objective.json")
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)

    return obj


def _objective_value(obj: Dict[str, Any], metric: str, direction: str) -> float:
    """Extract objective value from computed metrics."""
    if metric not in obj:
        raise ExperimentSpecError(
            f"Objective metric '{metric}' not found. Available: {sorted(obj.keys())}"
        )
    val = obj[metric]
    if val is None or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        return float("inf") if direction == "minimize" else float("-inf")
    return float(val)


# =============================================================================
# Phase Transition Logic
# =============================================================================

def narrow_search_space_from_study(
    prev_study_name: str,
    storage: str,
    top_fraction: float = 0.2,
    enqueue_top_k: int = 3,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Analyse a completed study and produce narrowed search-space hints.

    Returns
    -------
    narrowed_distributions : dict
        Mapping from param name -> dict with narrowed range info.
        For float/int params: {'low': p10, 'high': p90} from the top
        ``top_fraction`` of completed trials.
        For categorical params: list of values that appeared in the top slice.
    top_params : list[dict]
        Parameter dicts of the top-K trials, suitable for
        ``study.enqueue_trial(params)``.
    """
    study = optuna.load_study(study_name=prev_study_name, storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"No completed trials in study '{prev_study_name}'")

    # Sort by objective (ascending for minimize, descending for maximize)
    is_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    completed.sort(key=lambda t: t.value if t.value is not None else float("inf"),
                   reverse=not is_minimize)

    n_top = max(1, int(len(completed) * top_fraction))
    top_trials = completed[:n_top]
    top_params_list = [dict(t.params) for t in top_trials[:enqueue_top_k]]

    narrowed: Dict[str, Any] = {}
    all_param_names: set[str] = set()
    for t in top_trials:
        all_param_names.update(t.params.keys())

    for pname in sorted(all_param_names):
        vals = [t.params[pname] for t in top_trials if pname in t.params]
        if not vals:
            continue

        dist = study.trials[0].distributions.get(pname, None) if study.trials else None

        if isinstance(vals[0], (int, float)) and not isinstance(vals[0], bool):
            numeric = sorted(float(v) for v in vals)
            n = len(numeric)
            p10_idx = max(0, int(n * 0.1))
            p90_idx = min(n - 1, int(n * 0.9))
            narrowed[pname] = {
                "type": "int" if all(isinstance(v, int) for v in vals) else "float",
                "low": numeric[p10_idx],
                "high": numeric[p90_idx],
                "log": getattr(dist, "log", False) if dist else False,
            }
        else:
            narrowed[pname] = {
                "type": "categorical",
                "choices": sorted(set(str(v) for v in vals)),
            }

    return narrowed, top_params_list


# =============================================================================
# Sequential CV Tuning (all folds per trial, single process)
# =============================================================================

def run_optuna_cv_tuning(
    experiment_path: str,
    n_trials: int,
    storage: str,
    study_name: Optional[str] = None,
    sampler_seed: int = 42,
    pruner_type: Optional[Literal["hyperband", "median", "none"]] = None,
) -> None:
    """Run Optuna hyperparameter tuning with CV (all folds per trial).

    This is the sequential mode: each trial trains all folds. For HPC
    parallelism use ``run_optuna_fold_tuning`` with ``--test_fold``.

    Args:
        experiment_path: Path to experiment YAML.
        n_trials: Number of Optuna trials.
        storage: Optuna storage URL.
        study_name: Study name override.
        sampler_seed: Random seed for sampler.
        pruner_type: Pruning strategy ('hyperband', 'median', 'none').
            If None, reads from experiment YAML tuning.pruner_type
            (default 'hyperband').
    """
    exp = load_experiment_spec(experiment_path)
    study_name = str(study_name) if study_name is not None else exp.name
    n_trials_effective = n_trials or exp.tuning.n_trials_per_fold
    subsample_frac = exp.data.subsample_frac  # from experiment spec

    direction = exp.objective.direction

    # Resolve pruner: explicit argument > experiment YAML > default
    effective_pruner_type = pruner_type if pruner_type is not None else exp.tuning.pruner_type

    # Minimum epochs before pruning: respect patience from experiment spec
    min_patience = _min_patience_from_spec(exp.training.patience)

    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_seed),
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
    )

    if effective_pruner_type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=max(HYPERBAND_MIN_RESOURCE, min_patience),
            max_resource=_infer_hyperband_max_resource(exp),
            reduction_factor=HYPERBAND_REDUCTION_FACTOR,
        )
    elif effective_pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=min_patience)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=_make_storage(storage),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Phase-transition seeding: enqueue top configs from previous study
    if (exp.tuning.prev_study is not None
            and not study.user_attrs.get("_prev_study_seeded")):
        study.set_user_attr("_prev_study_seeded", True)
        _, top_params = narrow_search_space_from_study(
            prev_study_name=exp.tuning.prev_study,
            storage=storage,
            top_fraction=exp.tuning.top_fraction,
            enqueue_top_k=exp.tuning.enqueue_top_k,
        )
        for params in top_params:
            study.enqueue_trial(params)
        if top_params:
            print(f"[Tuning] Seeded {len(top_params)} top configs from "
                  f"'{exp.tuning.prev_study}'", file=sys.stderr, flush=True)

    from Proteus import train_model_cv  # local import to avoid circulars

    def objective(trial: optuna.Trial) -> float:
        model_cfg, train_cfg, trial_cfg = _resolve_trial_configs(exp, trial)

        run_dir = os.path.join(exp.output_dir, f"trial_{trial.number:05d}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "trial_config.json"), "w") as f:
            json.dump(trial_cfg, f, indent=2)

        try:
            train_model_cv(
                model_name="BioDVN",
                training_set_tsv=exp.data.training_set,
                zarr_base=exp.data.zarr_base,
                test_fold=None,  # All folds
                out_dir=run_dir,
                model_config=model_cfg,
                training_config=train_cfg,
                genes_per_batch=32,
                chunk_size=256,
                shuffle_genes_in_batch=True,
                seed=exp.data.seed,
                subsample_frac=subsample_frac,
                train_only_cohorts=exp.data.train_only_cohorts,
                balance_cohorts=exp.data.balance_cohorts,
                source_groups=exp.data.source_groups,
                target_n_per_source=exp.data.target_n_per_source,
                batch_mode=exp.data.batch_mode,
            )
        except optuna.TrialPruned:
            raise  # Let Optuna handle pruning
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            # CUDA hardware failures (driver crash, ECC error, device-side
            # assert, etc.) are infrastructure problems — once CUDA enters
            # this error state, every subsequent .to(device) call in the
            # same process will fail.  Raise SystemExit so the worker
            # process terminates instead of spinning through hundreds of
            # doomed trials.  (SystemExit inherits from BaseException, so
            # study.optimize(catch=(Exception,)) will not intercept it.)
            _err = str(exc).lower()
            if 'cuda' in _err and ('unknown error' in _err or 'device-side assert' in _err):
                print(f"[Trial {trial.number}] CUDA infrastructure failure — "
                      f"terminating worker: {exc}", file=sys.stderr, flush=True)
                raise SystemExit(1)
            # Training diverged (NaN, OOM, etc.) — record failure and
            # return worst-case value so the trial completes rather than
            # crashing the entire study.
            print(f"[Trial {trial.number}] Training failed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            trial.set_user_attr("error", str(exc))
            return float("inf") if direction == "minimize" else float("-inf")

        obj = compute_cv_objective(run_dir)
        for k, v in obj.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                trial.set_user_attr(k, float(v))

        return _objective_value(obj, metric=exp.objective.metric, direction=direction)

    # Run trials, retrying any that were pruned as duplicates.
    _done = 0
    for _ in range(_DEDUP_MAX_RETRIES):
        if _done >= int(n_trials_effective):
            break
        study.optimize(objective, n_trials=1, catch=(Exception,))
        latest = study.trials[-1]
        if latest.state == optuna.trial.TrialState.PRUNED and latest.user_attrs.get("_duplicate"):
            print(f"[Tuning] Trial {latest.number} was a duplicate — retrying.")
            continue
        _done += 1

    _write_study_report(study, exp.output_dir)


# =============================================================================
# Cross-Fold Promotion
# =============================================================================
# When fold-parallel tuning discovers a config that scores in the top
# ``promote_top_frac`` of completed trials, that config is *promoted*:
# automatically trained on every fold that hasn't tested it yet.
#
# Coordination between HPC workers uses a JSON ledger on shared storage
# protected by fcntl file locking.
# =============================================================================

def _config_hash(params: Dict[str, Any]) -> str:
    """Deterministic 16-char hex hash of a parameter dict."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class _FixedParamTrial:
    """Minimal mock of ``optuna.Trial`` that replays fixed parameters.

    Used by the promotion system to feed a recorded parameter dict
    through ``_resolve_trial_configs`` without going through the Optuna
    suggest machinery.
    """

    def __init__(self, params: Dict[str, Any]):
        self._params = dict(params)
        self.number = -1  # sentinel for log messages

    # The three suggest_* signatures mirror what _suggest_from_param calls.
    def suggest_float(self, name: str, low: float = 0.0, high: float = 1.0,
                      *, step: Optional[float] = None,
                      log: bool = False) -> float:
        return float(self._params[name])

    def suggest_int(self, name: str, low: int = 0, high: int = 1,
                    *, step: int = 1, log: bool = False) -> int:
        return int(self._params[name])

    def suggest_categorical(self, name: str, choices: list) -> Any:
        return self._params[name]


class _PromotionManager:
    """File-based cross-fold promotion of top HP configurations.

    When fold-parallel tuning discovers a config that scores in the top
    fraction of completed trials, this manager coordinates promoting
    it — training that config on every fold that hasn't tested it yet.

    The JSON ledger is safe for concurrent SLURM workers because all
    reads/writes are protected by an ``fcntl.flock`` exclusive lock.

    Ledger schema::

        {
          "configs": {
            "<config_hash>": {
              "params": { ... },          # Optuna param dict
              "source_trial": int,
              "source_fold": int,
              "source_score": float,
              "fold_results": {
                "<fold>": {"status": "done", "score": float, "trial_dir": str}
                           | {"status": "running"}
                           | {"status": "failed", "error": str}
              }
            }
          }
        }
    """

    def __init__(self, ledger_path: str, all_folds: List[int]):
        self.ledger_path = ledger_path
        self.all_folds = sorted(all_folds)

    # ----- low-level lock helpers -----

    def _lock_and_read(self) -> Tuple[dict, Any]:
        """Open ledger under exclusive lock; return (data, file_handle)."""
        os.makedirs(os.path.dirname(self.ledger_path) or ".", exist_ok=True)
        fh = open(self.ledger_path, "a+")
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        content = fh.read()
        data = json.loads(content) if content.strip() else {"configs": {}}
        return data, fh

    def _write_and_unlock(self, data: dict, fh) -> None:
        fh.seek(0)
        fh.truncate()
        json.dump(data, fh, indent=2)
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()

    def _unlock(self, fh) -> None:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()

    # ----- public API -----

    def nominate(
        self,
        params: Dict[str, Any],
        source_fold: int,
        source_score: float,
        source_trial: int,
    ) -> str:
        """Nominate a config for cross-fold promotion (idempotent).

        Returns the config hash.
        """
        chash = _config_hash(params)
        data, fh = self._lock_and_read()
        try:
            if chash not in data["configs"]:
                data["configs"][chash] = {
                    "params": params,
                    "source_fold": source_fold,
                    "source_score": source_score,
                    "source_trial": source_trial,
                    "fold_results": {
                        str(source_fold): {
                            "status": "done",
                            "score": source_score,
                            "trial": source_trial,
                        }
                    },
                }
            else:
                # Already nominated; ensure this fold's result is recorded.
                entry = data["configs"][chash]
                fk = str(source_fold)
                if fk not in entry["fold_results"]:
                    entry["fold_results"][fk] = {
                        "status": "done",
                        "score": source_score,
                        "trial": source_trial,
                    }
            self._write_and_unlock(data, fh)
        except (OSError, KeyError, TypeError, ValueError):
            self._unlock(fh)
            raise
        return chash

    def claim_pending(self, fold: int) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Atomically claim a pending promotion for *fold*.

        Returns ``(config_hash, params)`` or ``None`` if nothing pending.
        """
        data, fh = self._lock_and_read()
        try:
            fk = str(fold)
            for chash, entry in data["configs"].items():
                if fk not in entry["fold_results"]:
                    entry["fold_results"][fk] = {"status": "running"}
                    self._write_and_unlock(data, fh)
                    return chash, entry["params"]
            self._write_and_unlock(data, fh)
            return None
        except (OSError, KeyError, TypeError, ValueError):
            self._unlock(fh)
            raise

    def record_result(
        self, config_hash: str, fold: int, score: float, trial_dir: str
    ) -> None:
        """Record a completed promotion result."""
        data, fh = self._lock_and_read()
        try:
            if config_hash in data["configs"]:
                data["configs"][config_hash]["fold_results"][str(fold)] = {
                    "status": "done",
                    "score": score,
                    "trial_dir": trial_dir,
                }
            self._write_and_unlock(data, fh)
        except (OSError, KeyError, TypeError, ValueError):
            self._unlock(fh)
            raise

    def record_failure(
        self, config_hash: str, fold: int, error: str
    ) -> None:
        """Record a failed promotion attempt."""
        data, fh = self._lock_and_read()
        try:
            if config_hash in data["configs"]:
                data["configs"][config_hash]["fold_results"][str(fold)] = {
                    "status": "failed",
                    "error": error,
                }
            self._write_and_unlock(data, fh)
        except (OSError, KeyError, TypeError, ValueError):
            self._unlock(fh)
            raise

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all promoted configs and their results."""
        data, fh = self._lock_and_read()
        try:
            self._write_and_unlock(data, fh)
        except (OSError, TypeError, ValueError):
            self._unlock(fh)
            raise

        summaries = []
        for chash, entry in data.get("configs", {}).items():
            done = {
                k: v for k, v in entry["fold_results"].items()
                if v.get("status") == "done"
            }
            scores = [
                v["score"] for v in done.values()
                if isinstance(v.get("score"), (int, float))
                and math.isfinite(v["score"])
            ]
            summaries.append({
                "config_hash": chash,
                "params": entry["params"],
                "source_fold": entry["source_fold"],
                "source_trial": entry["source_trial"],
                "folds_completed": len(done),
                "folds_total": len(self.all_folds),
                "mean_score": _nanmean(scores) if scores else None,
                "std_score": _nanstd(scores) if len(scores) > 1 else None,
                "fold_scores": {k: v.get("score") for k, v in done.items()},
            })
        return {"promoted_configs": summaries}


def _identify_top_configs(
    study: "optuna.Study",
    direction: str,
    top_frac: float,
    max_promotions: int,
) -> List["optuna.Trial"]:
    """Return the top-scoring unique configs from a completed study.

    Parameters
    ----------
    study : optuna.Study
    direction : 'minimize' | 'maximize'
    top_frac : fraction of trials to consider "top" (e.g. 0.10 → top 10%)
    max_promotions : hard cap on how many configs to return
    """
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and math.isfinite(t.value)
    ]
    if not completed:
        return []

    # Sort best-first
    if direction == "minimize":
        completed.sort(key=lambda t: t.value)
    else:
        completed.sort(key=lambda t: t.value, reverse=True)

    # Percentile threshold: keep trials in the top ``top_frac``.
    n_top = max(1, int(math.ceil(len(completed) * top_frac)))
    candidates = completed[:n_top]

    # De-duplicate by config hash (keep the best trial per config)
    seen: set[str] = set()
    unique: List[optuna.Trial] = []
    for trial in candidates:
        chash = _config_hash(trial.params)
        if chash not in seen:
            seen.add(chash)
            unique.append(trial)
            if len(unique) >= max_promotions:
                break

    return unique


def _run_promotion_phase(
    study: "optuna.Study",
    exp: "ExperimentSpec",
    fold_num: int,
    direction: str,
    train_fn: Callable,
) -> None:
    """Post-trial promotion phase: test top configs on this fold.

    Called after ``study.optimize()`` completes for a fold worker.
    Identifies top configs across the study, nominates them, then
    trains any that haven't been tested on ``fold_num``.

    Parameters
    ----------
    study : the shared Optuna study (read-only access to trials)
    exp : loaded ExperimentSpec
    fold_num : this worker's fold
    direction : 'minimize' | 'maximize'
    train_fn : ``train_model_cv`` function (imported in caller)
    """
    top_frac = exp.tuning.promote_top_frac
    max_promo = exp.tuning.max_promotions
    if top_frac <= 0.0 or max_promo <= 0:
        return

    all_folds = _folds_from_tsv(exp.data.training_set)
    ledger_path = os.path.join(exp.output_dir, "promotion_ledger.json")
    pm = _PromotionManager(ledger_path, all_folds)

    # Nominate the top configs discovered so far.
    top_trials = _identify_top_configs(study, direction, top_frac, max_promo)
    for t in top_trials:
        src_fold = t.user_attrs.get("fold_num", fold_num)
        pm.nominate(
            params=t.params,
            source_fold=int(src_fold),
            source_score=float(t.value),
            source_trial=int(t.number),
        )
    if top_trials:
        print(
            f"[Promotion] Fold {fold_num}: nominated "
            f"{len(top_trials)} config(s) for cross-fold testing"
        )

    # Process pending promotions for this fold.
    n_promoted = 0
    while True:
        claimed = pm.claim_pending(fold_num)
        if claimed is None:
            break
        chash, params = claimed
        n_promoted += 1
        print(
            f"[Promotion] Fold {fold_num}: training promoted config "
            f"{chash[:8]}… ({n_promoted})"
        )

        promo_run_dir = os.path.join(
            exp.output_dir, f"promoted_{chash[:8]}_fold{fold_num}"
        )
        try:
            mock_trial = _FixedParamTrial(params)
            model_cfg, train_cfg, _tcfg = _resolve_trial_configs(exp, mock_trial)

            os.makedirs(promo_run_dir, exist_ok=True)
            _write_json_atomic(
                os.path.join(promo_run_dir, "trial_config.json"), _tcfg
            )

            train_fn(
                model_name="BioDVN",
                training_set_tsv=exp.data.training_set,
                zarr_base=exp.data.zarr_base,
                test_fold=int(fold_num),
                out_dir=promo_run_dir,
                model_config=model_cfg,
                training_config=train_cfg,
                genes_per_batch=32,
                chunk_size=256,
                shuffle_genes_in_batch=True,
                seed=exp.data.seed,
                subsample_frac=exp.data.subsample_frac,
                train_only_cohorts=exp.data.train_only_cohorts,
                balance_cohorts=exp.data.balance_cohorts,
                source_groups=exp.data.source_groups,
                target_n_per_source=exp.data.target_n_per_source,
            )

            # Read fold metric
            fold_dir = os.path.join(
                promo_run_dir, f"CV-fold_{int(fold_num)}"
            )
            metrics_path = os.path.join(
                fold_dir,
                f"validation-fold_{int(fold_num)}_metrics.json",
            )
            with open(metrics_path, "r") as f:
                fold_metrics = json.load(f)

            tumor = fold_metrics.get("tumor", {})
            bg = fold_metrics.get("background", {})
            obj = {
                "tumor_mae_mean": tumor.get("mae", float("nan")),
                "tumor_r2_mean": tumor.get("r2", float("nan")),
                "tumor_gvw_crps_mean": tumor.get("gvw_crps", float("nan")),
                "background_mae_mean": bg.get("mae", float("nan")),
                "background_r2_mean": bg.get("r2", float("nan")),
                "background_gvw_crps_mean": bg.get("gvw_crps", float("nan")),
            }
            # Balanced metrics (single-fold: mean across non-PHT cohorts)
            if fold_metrics.get("balanced_tumor_gvw_crps") is not None:
                obj["balanced_tumor_gvw_crps_mean"] = float(fold_metrics["balanced_tumor_gvw_crps"])
                obj["balanced_tumor_gvw_crps_std"] = float(fold_metrics.get("balanced_tumor_gvw_crps_std", float("nan")))
            if fold_metrics.get("balanced_tumor_gvw_r") is not None:
                obj["balanced_tumor_gvw_r_mean"] = float(fold_metrics["balanced_tumor_gvw_r"])
                obj["balanced_tumor_gvw_r_std"] = float(fold_metrics.get("balanced_tumor_gvw_r_std", float("nan")))
            score = _objective_value(
                obj, metric=exp.objective.metric, direction=direction
            )
            pm.record_result(chash, fold_num, score, promo_run_dir)
            print(
                f"[Promotion] Fold {fold_num}: config {chash[:8]}… "
                f"score={score:.4f}"
            )

        except optuna.TrialPruned as exc:
            pm.record_failure(chash, fold_num, f"pruned: {exc}")
            print(
                f"[Promotion] Fold {fold_num}: config {chash[:8]}… "
                f"pruned ({exc})"
            )
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ExperimentSpecError) as exc:
            pm.record_failure(chash, fold_num, str(exc))
            print(
                f"[Promotion] Fold {fold_num}: config {chash[:8]}… "
                f"failed ({exc})"
            )

    # Write promotion summary
    if n_promoted > 0 or top_trials:
        summary = pm.summary()
        summary_path = os.path.join(exp.output_dir, "promotion_summary.json")
        _write_json_atomic(summary_path, summary)
        print(
            f"[Promotion] Fold {fold_num}: completed {n_promoted} "
            f"promotion run(s); summary → {summary_path}"
        )


# =============================================================================
# Fold-Parallel Tuning (one fold per process, each fold = different trial)
# =============================================================================

def run_optuna_fold_tuning(
    experiment_path: str,
    fold_num: int,
    storage: str,
    n_trials: Optional[int] = None,
    study_name: Optional[str] = None,
    sampler_seed: int = 42,
    pruner_type: Optional[Literal['hyperband', 'median', 'none']] = None,
) -> None:
    """Fold-parallel tuning: each fold is an independent Optuna trial.

    Launch one job per fold on HPC.  All point at the same Optuna
    study/storage.  Each job calls ``study.optimize(objective, n_trials=N)``
    where the objective trains only the assigned fold.

    This produces 5 × N total trials (5 folds × N trials per fold),
    each reporting a single-fold metric.  Cross-fold aggregation happens
    post-hoc.

    Args:
        experiment_path: Path to experiment YAML.
        fold_num: The fold this process trains.
        storage: Optuna storage URL.
        n_trials: Number of trials for this fold worker.  Defaults to
            ``exp.tuning.n_trials_per_fold``.
        study_name: Study name override (default: experiment name).
        sampler_seed: Base seed.  Actual seed = sampler_seed + fold_num
            so each fold explores slightly different regions.
        pruner_type: Pruning strategy.  If None, reads from experiment
            YAML tuning.pruner_type (default 'hyperband').
    """
    exp = load_experiment_spec(experiment_path)
    study_name = str(study_name) if study_name is not None else exp.name
    n_trials_effective = n_trials if n_trials is not None else exp.tuning.n_trials_per_fold
    subsample_frac = exp.data.subsample_frac

    direction = exp.objective.direction

    # Resolve pruner: explicit argument > experiment YAML > default
    effective_pruner_type = pruner_type if pruner_type is not None else exp.tuning.pruner_type

    # Minimum epochs before pruning: respect patience from experiment spec
    min_patience = _min_patience_from_spec(exp.training.patience)

    # Seed varies per fold so each worker explores different regions
    fold_seed = int(sampler_seed) + int(fold_num)
    sampler = optuna.samplers.TPESampler(
        seed=fold_seed,
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
    )

    if effective_pruner_type == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=max(HYPERBAND_MIN_RESOURCE, min_patience),
            max_resource=_infer_hyperband_max_resource(exp),
            reduction_factor=HYPERBAND_REDUCTION_FACTOR,
        )
    elif effective_pruner_type == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=min_patience)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=_make_storage(storage),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Phase-transition seeding: enqueue top configs from previous study
    if (exp.tuning.prev_study is not None
            and not study.user_attrs.get("_prev_study_seeded")):
        study.set_user_attr("_prev_study_seeded", True)
        _, top_params = narrow_search_space_from_study(
            prev_study_name=exp.tuning.prev_study,
            storage=storage,
            top_fraction=exp.tuning.top_fraction,
            enqueue_top_k=exp.tuning.enqueue_top_k,
        )
        for params in top_params:
            study.enqueue_trial(params)
        if top_params:
            print(f"[Tuning] Seeded {len(top_params)} top configs from "
                  f"'{exp.tuning.prev_study}'", file=sys.stderr, flush=True)

    from Proteus import train_model_cv  # local import to avoid circulars

    def objective(trial: optuna.Trial) -> float:
        model_cfg, train_cfg, trial_cfg = _resolve_trial_configs(exp, trial)

        run_dir = os.path.join(exp.output_dir, f"trial_{trial.number:05d}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "trial_config.json"), "w") as f:
            json.dump(trial_cfg, f, indent=2)

        # Tag trial with fold metadata
        trial.set_user_attr("fold_num", int(fold_num))

        def _epoch_end_callback(
            epoch: int, val_metrics: Dict[str, float], val_score: float
        ) -> None:
            """Report intermediate metric for Hyperband pruning.

            The reported metric must align with the study direction:
            - maximize → report a metric where higher = better
            - minimize → report a metric where lower = better

            For maximize: prefer val_min_tumor_gvw_r (min across sources,
            higher = better), fall back to val_balanced_tumor_gvw_r, then
            val_score.  For minimize: val_tumor_gvw_crps.

            Pruning is suppressed for the first ``patience`` epochs so the
            model has time to complete warmup and stabilize.
            """
            if direction == 'maximize':
                # Prefer min-source GVW-r when available (direct proxy
                # for min_tumor_gvw_r objective); fall back to balanced.
                metric_val = val_metrics.get(
                    'val_min_tumor_gvw_r',
                    val_metrics.get('val_balanced_tumor_gvw_r', val_score),
                )
            else:
                metric_val = val_metrics.get('val_tumor_gvw_crps', val_score)
            if metric_val is not None and isinstance(metric_val, (int, float)):
                trial.report(float(metric_val), step=int(epoch))
                if epoch >= int(train_cfg.patience) and trial.should_prune():
                    raise optuna.TrialPruned(f'Pruned at epoch {epoch}')

        try:
            train_model_cv(
                model_name='BioDVN',
                training_set_tsv=exp.data.training_set,
                zarr_base=exp.data.zarr_base,
                test_fold=int(fold_num),
                out_dir=run_dir,
                model_config=model_cfg,
                training_config=train_cfg,
                genes_per_batch=32,
                chunk_size=256,
                shuffle_genes_in_batch=True,
                seed=exp.data.seed,
                subsample_frac=subsample_frac,
                epoch_end_callback=_epoch_end_callback,
                train_only_cohorts=exp.data.train_only_cohorts,
                balance_cohorts=exp.data.balance_cohorts,
                source_groups=exp.data.source_groups,
                target_n_per_source=exp.data.target_n_per_source,
                batch_mode=exp.data.batch_mode,
            )
        except optuna.TrialPruned:
            raise  # Let Optuna handle it
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            # CUDA hardware failures (driver crash, ECC error, etc.) —
            # once CUDA is broken, every .to(device) in this process will
            # fail.  Terminate the worker immediately instead of burning
            # through trial IDs.  SystemExit bypasses catch=(Exception,).
            _err = str(exc).lower()
            if 'cuda' in _err and ('unknown error' in _err or 'device-side assert' in _err):
                print(f"[Trial {trial.number}, Fold {fold_num}] CUDA "
                      f"infrastructure failure — terminating worker: {exc}", file=sys.stderr, flush=True)
                raise SystemExit(1)
            # Training diverged (NaN, OOM, etc.) — record failure and
            # return worst-case value so the trial completes rather than
            # crashing the entire study.
            print(f"[Trial {trial.number}, Fold {fold_num}] Training failed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            trial.set_user_attr("error", str(exc))
            return float("inf") if direction == "minimize" else float("-inf")

        # Read single-fold metric
        fold_dir = os.path.join(run_dir, f'CV-fold_{int(fold_num)}')
        metrics_path = os.path.join(fold_dir, f'validation-fold_{int(fold_num)}_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"[Trial {trial.number}, Fold {fold_num}] Metrics file not found: {metrics_path}", file=sys.stderr, flush=True)
            trial.set_user_attr("error", f"Metrics file not found: {metrics_path}")
            return float("inf") if direction == "minimize" else float("-inf")

        with open(metrics_path, 'r') as f:
            fold_metrics = json.load(f)

        # Build a single-fold objective dict compatible with _objective_value
        tumor = fold_metrics.get("tumor", {})
        bg = fold_metrics.get("background", {})
        obj = {
            "tumor_mae_mean": tumor.get("mae", float("nan")),
            "tumor_r2_mean": tumor.get("r2", float("nan")),
            "tumor_gvw_crps_mean": tumor.get("gvw_crps", float("nan")),
            "background_mae_mean": bg.get("mae", float("nan")),
            "background_r2_mean": bg.get("r2", float("nan")),
            "background_gvw_crps_mean": bg.get("gvw_crps", float("nan")),
            "val_loss": _nanmean([
                tumor.get("mae", float("nan")),
                bg.get("mae", float("nan")),
            ]),
        }
        # Balanced fine-tuning objective (single-fold value)
        balanced_val = fold_metrics.get("balanced_tumor_gvw_crps", None)
        if balanced_val is not None:
            obj["balanced_tumor_gvw_crps_mean"] = float(balanced_val)
            obj["balanced_tumor_gvw_crps_std"] = float(fold_metrics.get("balanced_tumor_gvw_crps_std", float("nan")))
        balanced_r = fold_metrics.get("balanced_tumor_gvw_r", None)
        if balanced_r is not None:
            obj["balanced_tumor_gvw_r_mean"] = float(balanced_r)
            obj["balanced_tumor_gvw_r_std"] = float(fold_metrics.get("balanced_tumor_gvw_r_std", float("nan")))

        # Composite lower-bound / upper-bound objectives
        _r_m = obj.get("balanced_tumor_gvw_r_mean")
        _r_s = obj.get("balanced_tumor_gvw_r_std")
        if (_r_m is not None and _r_s is not None
                and math.isfinite(float(_r_m)) and math.isfinite(float(_r_s))):
            obj["balanced_tumor_gvw_r_lower_bound"] = float(_r_m) - float(_r_s)
        _c_m = obj.get("balanced_tumor_gvw_crps_mean")
        _c_s = obj.get("balanced_tumor_gvw_crps_std")
        if (_c_m is not None and _c_s is not None
                and math.isfinite(float(_c_m)) and math.isfinite(float(_c_s))):
            obj["balanced_tumor_gvw_crps_upper_bound"] = float(_c_m) + float(_c_s)

        # Min-source GVW-r: minimum per-source GVW-r from fold metrics.
        per_source = fold_metrics.get("per_source", {})
        if per_source:
            src_r_vals = [
                float(v["tumor"]["gvw_r"])
                for v in per_source.values()
                if "tumor" in v and "gvw_r" in v["tumor"]
                   and math.isfinite(float(v["tumor"]["gvw_r"]))
            ]
            if src_r_vals:
                obj["min_tumor_gvw_r"] = min(src_r_vals)
        elif fold_metrics.get("min_tumor_gvw_r") is not None:
            obj["min_tumor_gvw_r"] = float(fold_metrics["min_tumor_gvw_r"])

        for k, v in obj.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                trial.set_user_attr(k, float(v))

        return _objective_value(obj, metric=exp.objective.metric, direction=direction)

    # Run trials, retrying any that were pruned as duplicates.
    _done = 0
    for _ in range(_DEDUP_MAX_RETRIES):
        if _done >= int(n_trials_effective):
            break
        study.optimize(objective, n_trials=1, catch=(Exception,))
        latest = study.trials[-1]
        if latest.state == optuna.trial.TrialState.PRUNED and latest.user_attrs.get("_duplicate"):
            print(f"[Tuning] Trial {latest.number} was a duplicate — retrying.")
            continue
        _done += 1

    # Cross-fold promotion phase (runs after main trial loop)
    _run_promotion_phase(study, exp, fold_num, direction, train_model_cv)

    _write_study_report(study, exp.output_dir)


# =============================================================================
# Report Helper
# =============================================================================

def _compute_fold_stability(
    study: optuna.Study, output_dir: str, direction: str,
) -> Optional[Dict[str, Any]]:
    """Compute cross-fold stability analysis from a fold-parallel study.

    Groups completed trials by their ``fold_num`` user attribute, then
    reports:

    * **per_fold** — best score, mean score, trial count per fold.
    * **study_fold_spread** — std of best-per-fold scores (how much
      fold difficulty varies).
    * **stability_score** — mean-of-best + std-of-best for ``minimize``
      (or mean - std for ``maximize``).  Penalises configs that only
      perform well on easy folds.
    * **promoted_configs** — if a promotion ledger exists, per-config
      cross-fold mean/std/stability from configs tested on >1 fold.

    Returns the report dict (also written to ``fold_stability.json``).
    """
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and math.isfinite(t.value)
    ]
    if not completed:
        return None

    # --- per-fold grouping ---
    fold_trials: Dict[int, List[Tuple[int, float]]] = {}
    for t in completed:
        fnum = t.user_attrs.get("fold_num")
        if fnum is None:
            continue
        fnum = int(fnum)
        fold_trials.setdefault(fnum, []).append((t.number, float(t.value)))

    if not fold_trials:
        return None

    is_minimize = direction == "minimize"
    per_fold: Dict[str, Any] = {}
    best_per_fold: List[float] = []
    for fnum in sorted(fold_trials):
        trials = fold_trials[fnum]
        scores = [s for _, s in trials]
        best_score = min(scores) if is_minimize else max(scores)
        best_trial = min(trials, key=lambda x: x[1]) if is_minimize else max(trials, key=lambda x: x[1])
        per_fold[str(fnum)] = {
            "n_complete": len(trials),
            "best_score": round(best_score, 6),
            "mean_score": round(_nanmean(scores), 6),
            "best_trial": best_trial[0],
        }
        best_per_fold.append(best_score)

    mean_of_best = _nanmean(best_per_fold)
    std_of_best = _nanstd(best_per_fold) if len(best_per_fold) > 1 else 0.0
    if is_minimize:
        stability_score = mean_of_best + std_of_best
    else:
        stability_score = mean_of_best - std_of_best

    report: Dict[str, Any] = {
        "per_fold": per_fold,
        "study_fold_spread": {
            "best_per_fold": [round(v, 6) for v in best_per_fold],
            "mean_of_best": round(mean_of_best, 6),
            "std_of_best": round(std_of_best, 6),
            "stability_score": round(stability_score, 6),
        },
    }

    # --- promoted-config stability (if ledger exists) ---
    ledger_path = os.path.join(output_dir, "promotion_ledger.json")
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, "r") as f:
                ledger = json.load(f)
            promoted: List[Dict[str, Any]] = []
            for chash, entry in ledger.get("configs", {}).items():
                done = {
                    k: v for k, v in entry.get("fold_results", {}).items()
                    if isinstance(v, dict) and v.get("status") == "done"
                }
                scores = [
                    float(v["score"]) for v in done.values()
                    if isinstance(v.get("score"), (int, float))
                    and math.isfinite(v["score"])
                ]
                if len(scores) < 2:
                    continue
                pmean = _nanmean(scores)
                pstd = _nanstd(scores)
                pstab = (pmean + pstd) if is_minimize else (pmean - pstd)
                promoted.append({
                    "config_hash": chash,
                    "params": entry.get("params", {}),
                    "fold_scores": {k: round(v["score"], 6) for k, v in done.items()
                                    if isinstance(v.get("score"), (int, float))},
                    "n_folds": len(scores),
                    "mean": round(pmean, 6),
                    "std": round(pstd, 6),
                    "stability_score": round(pstab, 6),
                })
            if promoted:
                promoted.sort(key=lambda x: x["stability_score"],
                              reverse=not is_minimize)
                report["promoted_configs"] = promoted
        except (json.JSONDecodeError, OSError):
            pass  # ledger unreadable, skip

    # --- write report ---
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fold_stability.json")
    _write_json_atomic(out_path, report)

    # --- console summary ---
    spread = report["study_fold_spread"]
    print(f"\n[Fold Stability] best-per-fold: {spread['best_per_fold']}")
    print(f"[Fold Stability] mean={spread['mean_of_best']:.4f}  "
          f"std={spread['std_of_best']:.4f}  "
          f"stability_score={spread['stability_score']:.4f}")
    if "promoted_configs" in report:
        for pc in report["promoted_configs"]:
            print(f"  promoted {pc['config_hash'][:8]}… "
                  f"mean={pc['mean']:.4f} std={pc['std']:.4f} "
                  f"stability={pc['stability_score']:.4f} "
                  f"({pc['n_folds']} folds)")

    return report


def _write_study_report(study: optuna.Study, output_dir: str) -> None:
    """Write best-trial summary and param importance to disk."""
    try:
        best = study.best_trial
    except ValueError:
        # No completed trials
        print("[Tuning] No completed trials in study.")
        return

    direction = str(study.direction).split(".")[-1].lower()

    out = {
        "study_name": study.study_name,
        "direction": direction,
        "n_trials": len(study.trials),
        "best_trial": {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": dict(best.user_attrs),
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "study_best.json")
    _write_json_atomic(out_path, out)

    # Param importance report (best-effort)
    try:
        from optuna import importance as optuna_importance
        imp = optuna_importance.get_param_importances(study)
        imp_path = os.path.join(output_dir, 'param_importance.json')
        _write_json_atomic(imp_path, {k: float(v) for k, v in imp.items()})
    except (ImportError, RuntimeError, ValueError, TypeError, OSError):
        pass

    # Fold-stability analysis (fold-parallel studies)
    _compute_fold_stability(study, output_dir, direction)


# =============================================================================
# Coordinated Fold-Parallel Tuning (ask/tell + fold workers)
# =============================================================================
# Each "round" the coordinator asks Optuna for N configs, writes them to
# disk, then the SLURM script launches 5 fold workers per config.  After
# all folds finish, the coordinator reads fold metrics and tells Optuna
# the cross-fold objective.  This gives fold-level parallelism with a
# proper cross-fold objective.
# =============================================================================


def coordinated_ask(
    experiment_path: str,
    n_configs: int,
    storage: str,
    study_name: Optional[str] = None,
    sampler_seed: int = 42,
) -> List[int]:
    """Ask Optuna for ``n_configs`` trial configs and write them to disk.

    For each config, creates ``{output_dir}/trial_{number:05d}/`` with
    a ``trial_config.json`` and a ``pending`` sentinel file.

    Returns the list of trial numbers (used by the SLURM script to
    dispatch fold workers).
    """
    exp = load_experiment_spec(experiment_path)
    study_name = str(study_name) if study_name is not None else exp.name
    direction = exp.objective.direction

    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_seed),
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
    )
    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=_make_storage(storage),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Phase-transition seeding (once per study)
    if (exp.tuning.prev_study is not None
            and not study.user_attrs.get("_prev_study_seeded")):
        study.set_user_attr("_prev_study_seeded", True)
        _, top_params = narrow_search_space_from_study(
            prev_study_name=exp.tuning.prev_study,
            storage=storage,
            top_fraction=exp.tuning.top_fraction,
            enqueue_top_k=exp.tuning.enqueue_top_k,
        )
        for params in top_params:
            study.enqueue_trial(params)
        if top_params:
            print(f"[Coordinated] Seeded {len(top_params)} top configs from "
                  f"'{exp.tuning.prev_study}'", flush=True)

    trial_numbers: List[int] = []
    for _ in range(n_configs):
        trial = study.ask()
        try:
            _, _, trial_cfg = _resolve_trial_configs(exp, trial)
        except optuna.TrialPruned as exc:
            # Config is geometrically invalid — tell Optuna and skip.
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            print(f"[Coordinated] Trial {trial.number} pruned: {exc}", flush=True)
            continue

        run_dir = os.path.join(exp.output_dir, f"trial_{trial.number:05d}")
        os.makedirs(run_dir, exist_ok=True)

        _write_json_atomic(
            os.path.join(run_dir, "trial_config.json"), trial_cfg
        )

        # Write pending sentinel with trial metadata
        _write_json_atomic(
            os.path.join(run_dir, "coordinated_pending.json"),
            {"trial_number": trial.number, "status": "pending"},
        )

        trial_numbers.append(trial.number)
        print(f"[Coordinated] Asked trial {trial.number:05d} → {run_dir}", flush=True)

    # Persist trial numbers for the SLURM script
    _write_json_atomic(
        os.path.join(exp.output_dir, "coordinated_active.json"),
        {"trial_numbers": trial_numbers},
    )

    return trial_numbers


def coordinated_tell(
    experiment_path: str,
    trial_numbers: Optional[List[int]] = None,
    storage: str = "sqlite:///results_training/optuna.db",
    study_name: Optional[str] = None,
    sampler_seed: int = 42,
) -> None:
    """Report cross-fold results for completed coordinated trials.

    Reads fold metrics from ``{output_dir}/trial_{number}/CV-fold_*/``,
    computes the CV objective, and calls ``study.tell()`` for each trial.
    If ``trial_numbers`` is None, reads from ``coordinated_active.json``.
    """
    exp = load_experiment_spec(experiment_path)
    study_name = str(study_name) if study_name is not None else exp.name
    direction = exp.objective.direction
    metric = exp.objective.metric

    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_seed),
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=_make_storage(storage),
        load_if_exists=True,
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )

    # Read active trial numbers if not provided
    if trial_numbers is None:
        active_path = os.path.join(exp.output_dir, "coordinated_active.json")
        if not os.path.exists(active_path):
            print("[Coordinated-tell] No coordinated_active.json found.", flush=True)
            return
        with open(active_path, "r") as f:
            trial_numbers = json.load(f)["trial_numbers"]

    # Map trial numbers to frozen trials for study.tell()
    # study.tell() needs the trial object returned by study.ask(),
    # but those are only available in the ask() process.  We use
    # study.tell(trial_number, ...) which Optuna supports via the
    # trial number when the trial is in WAITING or RUNNING state.
    for trial_num in trial_numbers:
        run_dir = os.path.join(exp.output_dir, f"trial_{trial_num:05d}")

        # Check that all folds completed
        folds = _folds_from_tsv(exp.data.training_set)
        all_done = True
        for fold_k in folds:
            metrics_path = os.path.join(
                run_dir, f"CV-fold_{fold_k}",
                f"validation-fold_{fold_k}_metrics.json",
            )
            if not os.path.exists(metrics_path):
                all_done = False
                break

        if not all_done:
            print(f"[Coordinated-tell] Trial {trial_num:05d}: "
                  f"not all folds complete — skipping.", flush=True)
            # Tell Optuna this trial failed
            study.tell(trial_num, state=optuna.trial.TrialState.FAIL)
            continue

        # Compute cross-fold objective (same as sequential CV)
        try:
            obj = compute_cv_objective(run_dir)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[Coordinated-tell] Trial {trial_num:05d}: "
                  f"objective computation failed: {exc}", flush=True)
            study.tell(trial_num, state=optuna.trial.TrialState.FAIL)
            continue

        try:
            value = _objective_value(obj, metric=metric, direction=direction)
        except ExperimentSpecError as exc:
            print(f"[Coordinated-tell] Trial {trial_num:05d}: {exc}", flush=True)
            study.tell(trial_num, state=optuna.trial.TrialState.FAIL)
            continue

        # Report to Optuna — cv_objective.json on disk preserves full
        # cross-fold metrics; the study only stores the scalar objective.
        study.tell(trial_num, value)

        print(f"[Coordinated-tell] Trial {trial_num:05d}: "
              f"{metric}={value:.4f}", flush=True)

        # Remove pending sentinel
        pending = os.path.join(run_dir, "coordinated_pending.json")
        if os.path.exists(pending):
            os.remove(pending)

    _write_study_report(study, exp.output_dir)
    print(f"[Coordinated-tell] Reported {len(trial_numbers)} trial(s).", flush=True)


def run_coordinated_fold(
    experiment_path: str,
    trial_number: int,
    fold_num: int,
) -> None:
    """Train a single fold for a coordinated trial.

    Reads the trial config from ``{output_dir}/trial_{number}/trial_config.json``,
    builds model/training configs, and runs ``train_model_cv`` for the
    specified fold.  No Optuna interaction — pure training.
    """
    exp = load_experiment_spec(experiment_path)
    run_dir = os.path.join(exp.output_dir, f"trial_{trial_number:05d}")
    config_path = os.path.join(run_dir, "trial_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Trial config not found: {config_path}\n"
            f"Run 'tune-ask' first to generate configs."
        )

    with open(config_path, "r") as f:
        trial_cfg = json.load(f)

    # Reconstruct model and training configs from the saved trial config
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
        reduction_dilation=int(sc["reduction_dilation"]),
        beta_background=float(s["beta_background"]),
        beta_tcvae=float(s["beta_tcvae"]),
        beta_contrastive=float(s["beta_contrastive"]),
        beta_reconstruction=float(s["beta_reconstruction"]),
        cl_margin=float(s["cl_margin"]),
        internal_weight=float(s["internal_weight"]),
        beta_grl=float(s["beta_grl"]),
        beta_point_mse=float(s["beta_point_mse"]),
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
        base_weights=tr.get("base_weights"),
        freeze_level=tr.get("freeze_level"),
        freeze_mode=tr.get("freeze_mode"),
        backbone_lr_multiplier=float(tr.get("backbone_lr_multiplier", 1.0)),
    )

    from Proteus import train_model_cv  # local import to avoid circulars

    print(f"[Coordinated] Training trial {trial_number:05d}, fold {fold_num}",
          flush=True)

    train_model_cv(
        model_name="BioDVN",
        training_set_tsv=exp.data.training_set,
        zarr_base=exp.data.zarr_base,
        test_fold=int(fold_num),
        out_dir=run_dir,
        model_config=model_cfg,
        training_config=train_cfg,
        genes_per_batch=32,
        chunk_size=256,
        shuffle_genes_in_batch=True,
        seed=exp.data.seed,
        subsample_frac=exp.data.subsample_frac,
        train_only_cohorts=exp.data.train_only_cohorts,
        balance_cohorts=exp.data.balance_cohorts,
        source_groups=exp.data.source_groups,
        target_n_per_source=exp.data.target_n_per_source,
        batch_mode=exp.data.batch_mode,
    )
