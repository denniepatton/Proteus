"""Experiment specification for Proteus training runs.

This module defines a strict YAML schema for training experiments.
Experiments specify model/training hyperparameters and a training_set TSV
that defines exactly which samples go into which fold.

Training set TSV format (tab-separated, with header):
    Cohort    Sample    Fold
    PC-PDX    LuCaP_173-1    1
    ...

HP Group Taxonomy:
    - Structural:  architecture, loss function, gate regularisation
    - Training:    optimizer, schedule, regularisation
    - Scale:       width/depth, capacity, conditional on architecture

Design goals:
- Fail loudly on unknown keys or invalid types.
- Support both fixed values and Optuna search spaces.
- Deterministic fold assignment via explicit TSV file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

import yaml


JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, List["JSONValue"], Dict[str, "JSONValue"]]


class ExperimentSpecError(ValueError):
    pass


def _fail(msg: str) -> "None":
    raise ExperimentSpecError(msg)


def _expect_dict(x: Any, ctx: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        _fail(f"{ctx} must be a mapping/dict, got {type(x).__name__}")
    return x


def _check_no_unknown_keys(d: Mapping[str, Any], allowed: Sequence[str], ctx: str) -> None:
    unknown = sorted(set(d.keys()) - set(allowed))
    if unknown:
        _fail(f"Unknown keys in {ctx}: {unknown}. Allowed: {sorted(allowed)}")


def _as_bool(x: Any, ctx: str) -> bool:
    if isinstance(x, bool):
        return x
    _fail(f"{ctx} must be a boolean, got {type(x).__name__}")


def _as_int(x: Any, ctx: str) -> int:
    if isinstance(x, bool):
        _fail(f"{ctx} must be an int, got bool")
    if isinstance(x, int):
        return x
    _fail(f"{ctx} must be an int, got {type(x).__name__}")


def _as_float(x: Any, ctx: str) -> float:
    if isinstance(x, bool):
        _fail(f"{ctx} must be a float, got bool")
    if isinstance(x, (int, float)):
        return float(x)
    _fail(f"{ctx} must be a float, got {type(x).__name__}")


def _as_str(x: Any, ctx: str) -> str:
    if isinstance(x, str):
        s = x.strip()
        if not s:
            _fail(f"{ctx} must be a non-empty string")
        return s
    _fail(f"{ctx} must be a string, got {type(x).__name__}")


@dataclass(frozen=True)
class ParamSpec:
    """A parameter that can be fixed or sampled via Optuna.

    YAML form (fixed):
      param_name: 0.001
    or
      param_name:
        fixed: 0.001
    
    YAML form (Optuna suggest):
      param_name:
        suggest:
          type: float
          low: 1e-5
          high: 1e-2
          log: true
    or
      param_name:
        suggest:
          type: categorical
          choices: [a, b, c]
    """

    fixed: Optional[JSONValue] = None
    suggest: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_yaml(x: Any, ctx: str) -> "ParamSpec":
        if isinstance(x, (str, int, float, bool)) or x is None or isinstance(x, list):
            return ParamSpec(fixed=x)

        d = _expect_dict(x, ctx)
        _check_no_unknown_keys(d, allowed=["fixed", "suggest"], ctx=ctx)

        has_fixed = "fixed" in d
        has_suggest = "suggest" in d
        if (not has_fixed) and (not has_suggest):
            _fail(f"{ctx} must define either 'fixed' or 'suggest'")
        if has_fixed and has_suggest:
            _fail(f"{ctx} cannot define both 'fixed' and 'suggest'")

        fixed = d.get("fixed", None) if has_fixed else None
        suggest = d.get("suggest", None) if has_suggest else None
        if suggest is not None:
            suggest = _expect_dict(suggest, f"{ctx}.suggest")
        return ParamSpec(fixed=fixed, suggest=suggest)


# =============================================================================
# GROUP 1: Structural HPs (architecture, input ablations, loss function)
# =============================================================================

@dataclass(frozen=True)
class StructuralSpec:
    """Structural / architectural hyperparameters (Group 1).

    Covers architecture choices, loss-function weights, and contrastive margin.
    """
    beta_background: ParamSpec
    beta_tcvae: ParamSpec
    beta_contrastive: ParamSpec
    beta_reconstruction: ParamSpec
    beta_grl: ParamSpec
    beta_point_mse: ParamSpec
    cl_margin: ParamSpec
    internal_weight: ParamSpec

    @staticmethod
    def from_yaml(x: Any) -> "StructuralSpec":
        d = _expect_dict(x, "structural")
        allowed = [
            "beta_background", "beta_tcvae", "beta_contrastive",
            "beta_reconstruction", "beta_grl", "beta_point_mse",
            "cl_margin", "internal_weight",
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="structural")

        def p(name: str, default: Any) -> ParamSpec:
            return ParamSpec.from_yaml(d.get(name, default), f"structural.{name}")

        return StructuralSpec(
            beta_background=p("beta_background", 1.0),
            beta_tcvae=p("beta_tcvae", 3.0),
            beta_contrastive=p("beta_contrastive", 0.5),
            beta_reconstruction=p("beta_reconstruction", 1.5),
            beta_grl=p("beta_grl", 1.0),
            beta_point_mse=p("beta_point_mse", 0.0),
            cl_margin=p("cl_margin", 0.5),
            internal_weight=p("internal_weight", 2.5),
        )


# =============================================================================
# GROUP 2: Training HPs (optimizer, schedule, regularisation)
# =============================================================================

@dataclass(frozen=True)
class TrainingSpec:
    """Training / optimiser hyperparameters (Group 2).

    Uses cosine-annealing-with-warmup scheduler.  total_steps is always
    auto-derived as epochs * steps_per_epoch in Trainer.fit().  Schedule
    fractions (warmup_frac, min_lr_ratio) are scale-invariant across
    different data fractions and epoch counts.
    """
    max_lr: ParamSpec
    min_lr_ratio: ParamSpec
    warmup_start_lr: ParamSpec
    weight_decay: ParamSpec
    max_grad_norm: ParamSpec
    warmup_frac: ParamSpec
    schedule_frac: ParamSpec
    epochs: ParamSpec
    patience: ParamSpec
    use_amp: ParamSpec

    @staticmethod
    def from_yaml(x: Any) -> "TrainingSpec":
        d = _expect_dict(x, "training")
        allowed = [
            "max_lr", "min_lr_ratio", "warmup_start_lr",
            "weight_decay", "max_grad_norm",
            "warmup_frac", "schedule_frac",
            "epochs", "patience", "use_amp",
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="training")

        def p(name: str, default: Any) -> ParamSpec:
            return ParamSpec.from_yaml(d.get(name, default), f"training.{name}")

        return TrainingSpec(
            max_lr=p("max_lr", 0.001),
            min_lr_ratio=p("min_lr_ratio", 1e-4),
            warmup_start_lr=p("warmup_start_lr", 0.0),
            weight_decay=p("weight_decay", 0.002),
            max_grad_norm=p("max_grad_norm", 1.0),
            warmup_frac=p("warmup_frac", 0.0),
            schedule_frac=p("schedule_frac", 1.0),
            epochs=p("epochs", 30),
            patience=p("patience", 5),
            use_amp=p("use_amp", True),
        )


# =============================================================================
# GROUP 3: Scale HPs (width / depth / capacity)
# =============================================================================

@dataclass(frozen=True)
class ScaleSpec:
    """Scale / capacity hyperparameters (Group 3).

    These control the width and depth of each sub-network and should
    be tuned *after* structural and training HPs are settled.
    """
    encoder_start: ParamSpec
    latent_size: ParamSpec
    decoder_start: ParamSpec
    mdn_hidden: ParamSpec
    mdn_final: ParamSpec
    signal_filters: ParamSpec
    signal_encoder_width: ParamSpec
    signal_ksize: ParamSpec
    interaction_ksize: ParamSpec
    initial_reduction_channels: ParamSpec
    n_reductions: ParamSpec
    reduction_dilation: ParamSpec
    dropout_rate: ParamSpec

    @staticmethod
    def from_yaml(x: Any) -> "ScaleSpec":
        d = _expect_dict(x, "scale")
        allowed = [
            "encoder_start", "latent_size", "decoder_start",
            "mdn_hidden", "mdn_final",
            "signal_filters", "signal_encoder_width",
            "signal_ksize", "interaction_ksize",
            "initial_reduction_channels", "n_reductions",
            "reduction_dilation",
            "dropout_rate",
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="scale")

        def p(name: str, default: Any) -> ParamSpec:
            return ParamSpec.from_yaml(d.get(name, default), f"scale.{name}")

        return ScaleSpec(
            encoder_start=p("encoder_start", 128),
            latent_size=p("latent_size", 32),
            decoder_start=p("decoder_start", 16),
            mdn_hidden=p("mdn_hidden", 64),
            mdn_final=p("mdn_final", 16),
            signal_filters=p("signal_filters", 16),
            signal_encoder_width=p("signal_encoder_width", 1024),
            signal_ksize=p("signal_ksize", 6),
            interaction_ksize=p("interaction_ksize", 3),
            initial_reduction_channels=p("initial_reduction_channels", 64),
            n_reductions=p("n_reductions", 3),
            reduction_dilation=p("reduction_dilation", 1),
            dropout_rate=p("dropout_rate", 0.15),
        )


# =============================================================================
# Transfer Learning
# =============================================================================

@dataclass(frozen=True)
class TransferSpec:
    """Transfer learning configuration."""
    base_weights: Optional[str]
    freeze_level: Optional[str]

    @staticmethod
    def from_yaml(x: Any) -> "TransferSpec":
        if x is None:
            return TransferSpec(base_weights=None, freeze_level=None)
        d = _expect_dict(x, "transfer")
        allowed = ["base_weights", "freeze_level"]
        _check_no_unknown_keys(d, allowed=allowed, ctx="transfer")

        base_weights = d.get("base_weights", None)
        if base_weights is not None:
            base_weights = _as_str(base_weights, "transfer.base_weights")
        
        freeze_level = d.get("freeze_level", None)
        if freeze_level is not None:
            freeze_level = _as_str(freeze_level, "transfer.freeze_level")

        return TransferSpec(base_weights=base_weights, freeze_level=freeze_level)


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass(frozen=True)
class DataSpec:
    """Data loading configuration.

    subsample_frac (optional, float in (0, 1]):
        Fraction of each fold to use during training.  Used by phased
        tuning to run cheap screening passes on 10% or 50% of data.
        None means use 100% of the data (default for production runs).
    """
    training_set: str
    zarr_base: str
    seed: int
    subsample_frac: Optional[float]

    @staticmethod
    def from_yaml(x: Any) -> "DataSpec":
        d = _expect_dict(x, "data")
        allowed = ["training_set", "zarr_base", "seed",
                    "subsample_frac"]
        _check_no_unknown_keys(d, allowed=allowed, ctx="data")
        subsample_raw = d.get("subsample_frac", None)
        subsample_frac: Optional[float] = None
        if subsample_raw is not None:
            subsample_frac = float(subsample_raw)
            if not (0.0 < subsample_frac <= 1.0):
                _fail(f"data.subsample_frac must be in (0.0, 1.0], "
                      f"got {subsample_frac}")
        return DataSpec(
            training_set=_as_str(d["training_set"], "data.training_set"),
            zarr_base=_as_str(
                d.get("zarr_base",
                      "/hpc/temp/ha_g/user/rpatton/ProteusZarrs"),
                "data.zarr_base"),
            seed=_as_int(d.get("seed", 42), "data.seed"),
            subsample_frac=subsample_frac,
        )


# =============================================================================
# Objective (for tuning)
# =============================================================================

@dataclass(frozen=True)
class ObjectiveSpec:
    """Optuna optimization objective."""
    metric: str
    direction: Literal["minimize", "maximize"]

    @staticmethod
    def from_yaml(x: Any) -> "ObjectiveSpec":
        if x is None:
            return ObjectiveSpec(metric="val_loss", direction="minimize")
        d = _expect_dict(x, "objective")
        allowed = ["metric", "direction"]
        _check_no_unknown_keys(d, allowed=allowed, ctx="objective")

        metric = _as_str(d.get("metric", "val_loss"), "objective.metric")
        direction = _as_str(d.get("direction", "minimize"), "objective.direction").lower()
        if direction not in ("minimize", "maximize"):
            _fail("objective.direction must be one of: minimize|maximize")
        return ObjectiveSpec(metric=metric, direction=direction)


# =============================================================================
# Tuning Phase Configuration
# =============================================================================

@dataclass(frozen=True)
class TuningSpec:
    """Tuning-specific configuration for phased HP search.

    phase:              human-readable phase label (e.g. "1.2")
    n_trials_per_fold:  how many trials each fold worker runs
    prev_study:         name of preceding Optuna study (for phase transition)
    top_fraction:       fraction of completed trials to keep when narrowing
    enqueue_top_k:      number of best trials to enqueue in next study
    promote_top_frac:   fraction of trials considered "top" for cross-fold
                        promotion (0.0 = disabled).  E.g. 0.10 → top 10%.
    max_promotions:     cap on how many distinct configs are promoted.
    pruner_type:        Optuna pruning strategy ('hyperband', 'median', 'none').
                        Default 'hyperband'.
    """
    phase: str
    n_trials_per_fold: int
    prev_study: Optional[str]
    top_fraction: float
    enqueue_top_k: int
    promote_top_frac: float
    max_promotions: int
    pruner_type: str

    @staticmethod
    def from_yaml(x: Any) -> "TuningSpec":
        if x is None:
            return TuningSpec(
                phase="1.0",
                n_trials_per_fold=10, prev_study=None,
                top_fraction=0.2, enqueue_top_k=3,
                promote_top_frac=0.0, max_promotions=3,
                pruner_type="hyperband",
            )
        d = _expect_dict(x, "tuning")
        allowed = [
            "hp_group", "phase", "n_trials_per_fold",
            "prev_study", "top_fraction", "enqueue_top_k",
            "promote_top_frac", "max_promotions", "pruner_type",
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="tuning")
        # hp_group is accepted for backward compatibility but ignored.
        pruner_raw = _as_str(
            d.get("pruner_type", "hyperband"), "tuning.pruner_type"
        ).lower()
        if pruner_raw not in ("hyperband", "median", "none"):
            _fail("tuning.pruner_type must be one of: hyperband|median|none")
        return TuningSpec(
            phase=_as_str(d.get("phase", "1.0"), "tuning.phase"),
            n_trials_per_fold=_as_int(
                d.get("n_trials_per_fold", 10),
                "tuning.n_trials_per_fold"),
            prev_study=d.get("prev_study", None),
            top_fraction=_as_float(
                d.get("top_fraction", 0.2), "tuning.top_fraction"),
            enqueue_top_k=_as_int(
                d.get("enqueue_top_k", 3), "tuning.enqueue_top_k"),
            promote_top_frac=_as_float(
                d.get("promote_top_frac", 0.0),
                "tuning.promote_top_frac"),
            max_promotions=_as_int(
                d.get("max_promotions", 3),
                "tuning.max_promotions"),
            pruner_type=pruner_raw,
        )


# =============================================================================
# Top-Level Experiment Spec
# =============================================================================

@dataclass(frozen=True)
class ExperimentSpec:
    """Complete experiment specification.

    New canonical YAML layout uses three HP groups::

        structural:  ...
        training:    ...
        scale:       ...

    For backward compatibility, old-style ``model`` + ``loss`` top-level
    sections are auto-mapped into the new groups.
    """
    name: str
    output_dir: str
    data: DataSpec
    structural: StructuralSpec
    training: TrainingSpec
    scale: ScaleSpec
    transfer: TransferSpec
    objective: ObjectiveSpec
    tuning: TuningSpec

    @staticmethod
    def from_yaml_doc(doc: Any) -> "ExperimentSpec":
        d = _expect_dict(doc, "experiment")
        allowed = [
            "name", "output_dir", "data",
            "structural", "training", "scale",
            "transfer", "objective", "tuning",
            # backward-compat aliases ----------------------------------
            "model", "loss",
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="experiment")

        name = _as_str(d.get("name", "experiment"), "name")
        output_dir = _as_str(d.get("output_dir", "results_training"),
                             "output_dir")

        structural_raw = d.get("structural", None)
        scale_raw = d.get("scale", None)
        model_raw = d.get("model", None)
        loss_raw = d.get("loss", None)
        training_raw = d.get("training", {})

        # ------------------------------------------------------------------
        # Backward compat: build structural + scale from old model + loss
        # ------------------------------------------------------------------
        if structural_raw is None and model_raw is not None:
            m = dict(model_raw) if model_raw else {}
            lo = dict(loss_raw) if loss_raw else {}
            structural_raw = {
                "beta_background": lo.get("beta_background", 1.0),
                "beta_tcvae": lo.get("beta_tcvae", lo.get("beta_vae", 3.0)),
                "beta_contrastive": lo.get("beta_contrastive", 0.5),
                "beta_reconstruction": lo.get("beta_reconstruction", 1.5),
                "cl_margin": lo.get("cl_margin", 0.5),
                "internal_weight": lo.get("internal_weight", 2.5),
            }

        if scale_raw is None and model_raw is not None:
            m = dict(model_raw) if model_raw else {}
            scale_raw = {
                "encoder_start": m.get("encoder_start", 128),
                "latent_size": m.get("latent_size", 32),
                "decoder_start": m.get("decoder_start", 16),
                "mdn_hidden": m.get("mdn_hidden", 64),
                "mdn_final": m.get("mdn_final", 16),
                "signal_filters": m.get("signal_filters", 16),
                "signal_encoder_width": m.get("signal_encoder_width", 1024),
                "signal_ksize": m.get("signal_ksize", 6),
                "interaction_ksize": m.get("interaction_ksize", 3),
                "initial_reduction_channels": m.get(
                    "initial_reduction_channels", 64),
                "n_reductions": m.get("n_reductions", 3),
                "reduction_dilation": m.get(
                    "reduction_dilation", 1),
            }

        # Move dropout_rate from old model to training
        if model_raw is not None:
            m = dict(model_raw) if model_raw else {}
            if "dropout_rate" in m and "dropout_rate" not in training_raw:
                training_raw = dict(training_raw)
                training_raw["dropout_rate"] = m["dropout_rate"]

        return ExperimentSpec(
            name=name,
            output_dir=output_dir,
            data=DataSpec.from_yaml(d["data"]),
            structural=StructuralSpec.from_yaml(structural_raw or {}),
            training=TrainingSpec.from_yaml(training_raw),
            scale=ScaleSpec.from_yaml(scale_raw or {}),
            transfer=TransferSpec.from_yaml(d.get("transfer", None)),
            objective=ObjectiveSpec.from_yaml(d.get("objective", None)),
            tuning=TuningSpec.from_yaml(d.get("tuning", None)),
        )


def load_experiment_spec(path: str) -> ExperimentSpec:
    """Load an experiment specification from a YAML file.
    
    Args:
        path: Path to YAML file, or experiment name (will look in config/experiments/).
              If path doesn't exist and doesn't end in .yaml, tries:
              1. config/experiments/{path}.yaml
              2. config/experiments/{path}
    """
    import os
    from pathlib import Path
    
    # If path exists as-is, use it
    if os.path.exists(path):
        resolved_path = path
    else:
        # Try common locations
        project_root = Path(__file__).resolve().parent.parent
        candidates = [
            project_root / "config" / "experiments" / f"{path}.yaml",
            project_root / "config" / "experiments" / path,
            Path(path),  # Original path (will fail with clear error)
        ]
        
        resolved_path = None
        for candidate in candidates:
            if candidate.exists():
                resolved_path = str(candidate)
                break
        
        if resolved_path is None:
            raise FileNotFoundError(
                f"Experiment spec not found: '{path}'\n"
                f"Searched:\n"
                + "\n".join(f"  - {c}" for c in candidates)
            )
    
    with open(resolved_path, "r") as f:
        doc = yaml.safe_load(f)
    return ExperimentSpec.from_yaml_doc(doc)
