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

from proteus_utils.data_helpers import PLCA_WEIGHTED_SOURCE_WEIGHTS


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
    beta_kl: ParamSpec
    beta_contrastive: ParamSpec
    beta_reconstruction: ParamSpec
    beta_tfx: ParamSpec
    beta_ccc: ParamSpec
    beta_pht_consistency: ParamSpec
    beta_pht_uncertainty: ParamSpec
    cl_margin: ParamSpec
    internal_weight: ParamSpec

    @staticmethod
    def from_yaml(x: Any) -> "StructuralSpec":
        d = _expect_dict(x, "structural")
        allowed = [
            "beta_background", "beta_kl", "beta_contrastive",
            "beta_reconstruction", "beta_tfx", "beta_ccc",
            "beta_pht_consistency", "beta_pht_uncertainty", "cl_margin",
            "internal_weight",
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="structural")

        def p(name: str, default: Any) -> ParamSpec:
            return ParamSpec.from_yaml(d.get(name, default), f"structural.{name}")

        return StructuralSpec(
            beta_background=p("beta_background", 2.0),
            beta_kl=p("beta_kl", 1.0),
            beta_contrastive=p("beta_contrastive", 1.0),
            beta_reconstruction=p("beta_reconstruction", 0.50),
            beta_tfx=p("beta_tfx", 0.20),
            beta_ccc=p("beta_ccc", 0.25),
            beta_pht_consistency=p("beta_pht_consistency", 0.10),
            beta_pht_uncertainty=p("beta_pht_uncertainty", 0.05),
            cl_margin=p("cl_margin", 1.0),
            internal_weight=p("internal_weight", 1.0),
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
    ema_decay: ParamSpec
    grad_accum_steps: ParamSpec

    @staticmethod
    def from_yaml(x: Any) -> "TrainingSpec":
        d = _expect_dict(x, "training")
        allowed = [
            "max_lr", "min_lr_ratio", "warmup_start_lr",
            "weight_decay", "max_grad_norm",
            "warmup_frac", "schedule_frac",
            "epochs", "patience", "use_amp",
            "ema_decay", "grad_accum_steps",
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="training")

        def p(name: str, default: Any) -> ParamSpec:
            return ParamSpec.from_yaml(d.get(name, default), f"training.{name}")

        return TrainingSpec(
            max_lr=p("max_lr", 0.0015),
            min_lr_ratio=p("min_lr_ratio", 5e-4),
            warmup_start_lr=p("warmup_start_lr", 2.5e-4),
            weight_decay=p("weight_decay", 1e-5),
            max_grad_norm=p("max_grad_norm", 18.0),
            warmup_frac=p("warmup_frac", 0.10),
            schedule_frac=p("schedule_frac", 0.90),
            epochs=p("epochs", 15),
            patience=p("patience", 3),
            use_amp=p("use_amp", True),
            ema_decay=p("ema_decay", 0.0),
            grad_accum_steps=p("grad_accum_steps", 4),
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
    signal_ksize: ParamSpec
    interaction_ksize: ParamSpec
    initial_reduction_channels: ParamSpec
    n_reductions: ParamSpec
    funnel_reduction_factor: ParamSpec
    reduction_dilation: ParamSpec
    dropout_rate: ParamSpec

    @staticmethod
    def from_yaml(x: Any) -> "ScaleSpec":
        d = _expect_dict(x, "scale")
        allowed = [
            "encoder_start", "latent_size", "decoder_start",
            "mdn_hidden", "mdn_final",
            "signal_filters",
            "signal_ksize", "interaction_ksize",
            "initial_reduction_channels", "n_reductions",
            "funnel_reduction_factor",
            "reduction_dilation",
            "dropout_rate",
            "signal_encoder_width",  # Legacy key; ignored by the current compact spatial encoder
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="scale")

        def p(name: str, default: Any) -> ParamSpec:
            return ParamSpec.from_yaml(d.get(name, default), f"scale.{name}")

        return ScaleSpec(
            encoder_start=p("encoder_start", 128),
            latent_size=p("latent_size", 64),
            decoder_start=p("decoder_start", 64),
            mdn_hidden=p("mdn_hidden", 64),
            mdn_final=p("mdn_final", 32),
            signal_filters=p("signal_filters", 16),
            signal_ksize=p("signal_ksize", 6),
            interaction_ksize=p("interaction_ksize", 3),
            initial_reduction_channels=p("initial_reduction_channels", 64),
            n_reductions=p("n_reductions", 3),
            funnel_reduction_factor=p("funnel_reduction_factor", 4),
            reduction_dilation=p("reduction_dilation", 3),
            dropout_rate=p("dropout_rate", 0.20),
        )


# =============================================================================
# Transfer Learning
# =============================================================================

@dataclass(frozen=True)
class TransferSpec:
    """Transfer learning configuration."""
    base_weights: Optional[str]

    @staticmethod
    def from_yaml(x: Any) -> "TransferSpec":
        if x is None:
            return TransferSpec(base_weights=None)
        d = _expect_dict(x, "transfer")
        allowed = ["base_weights"]
        _check_no_unknown_keys(d, allowed=allowed, ctx="transfer")

        base_weights = d.get("base_weights", None)
        if base_weights is not None:
            base_weights = _as_str(base_weights, "transfer.base_weights")

        return TransferSpec(base_weights=base_weights)


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

    source_groups (optional, Dict[str, List[str]]):
        Source grouping for evaluation and weighted selection metrics.
        Maps a source name to the list of cohort names that belong to
        that source.  PHT samples are automatically assigned to the
        source whose PDX cohort they derive from (parsed from sample
        name).  In the preferred chunk-based PLCAv2 path this does not
        alter training exposure; it defines how validation metrics are
        aggregated.  The legacy ``multi_source`` batch mode still uses
        these groups for explicit source-balanced sampling.
        Mutually exclusive with ``balance_cohorts``.

    steps_per_epoch (optional, int):
        Number of fixed stochastic gene-conditioned micro-batches per epoch.

    batch_mode (optional, str: 'chunk' | 'multi_source'):
        Batch construction strategy.  ``'multi_source'`` is the default
        gene-conditioned source/PHT-aware training batcher. ``'chunk'`` is
        retained for validation/inference-style contiguous reads.

    selection_metric (optional, str: 'min_balanced_crps' |
                                     'min_weighted_source_crps' |
                                     'max_gvw_r_lower_bound'):
        Model selection / early-stopping metric.  ``'min_balanced_crps'``
        (default) selects the epoch with lowest balanced GVW-CRPS.
        ``'min_weighted_source_crps'`` selects the epoch with lowest
        fixed-weight PLCAv2 source GVW-CRPS using the 4 source buckets.
        ``'max_gvw_r_lower_bound'`` selects the epoch with highest
        balanced GVW-r lower bound (mean − std across sources).
    """
    training_set: str
    zarr_base: str
    seed: int
    subsample_frac: Optional[float]
    train_only_cohorts: Optional[List[str]]
    balance_cohorts: Optional[List[str]]
    source_groups: Optional[Dict[str, List[str]]]
    target_n_per_source: Optional[int]
    steps_per_epoch: int
    batch_mode: str
    selection_metric: str

    @staticmethod
    def from_yaml(x: Any) -> "DataSpec":
        d = _expect_dict(x, "data")
        allowed = ["training_set", "zarr_base", "seed",
                    "subsample_frac", "train_only_cohorts",
                    "balance_cohorts", "source_groups",
                    "target_n_per_source", "steps_per_epoch", "batch_mode",
                    "selection_metric"]
        _check_no_unknown_keys(d, allowed=allowed, ctx="data")
        subsample_raw = d.get("subsample_frac", None)
        subsample_frac: Optional[float] = None
        if subsample_raw is not None:
            subsample_frac = float(subsample_raw)
            if not (0.0 < subsample_frac <= 1.0):
                _fail(f"data.subsample_frac must be in (0.0, 1.0], "
                      f"got {subsample_frac}")

        train_only_raw = d.get("train_only_cohorts", None)
        train_only: Optional[List[str]] = None
        if train_only_raw is not None:
            if not isinstance(train_only_raw, list):
                _fail("data.train_only_cohorts must be a list of strings")
            train_only = [str(c).strip() for c in train_only_raw]

        balance_raw = d.get("balance_cohorts", None)
        balance: Optional[List[str]] = None
        if balance_raw is not None:
            if not isinstance(balance_raw, list):
                _fail("data.balance_cohorts must be a list of strings")
            balance = [str(c).strip() for c in balance_raw]

        source_groups_raw = d.get("source_groups", None)
        source_groups: Optional[Dict[str, List[str]]] = None
        if source_groups_raw is not None:
            if not isinstance(source_groups_raw, dict):
                _fail("data.source_groups must be a mapping of source_name -> [cohort_names]")
            source_groups = {
                str(k).strip(): [str(c).strip() for c in v]
                for k, v in source_groups_raw.items()
            }

        target_n_raw = d.get("target_n_per_source", None)
        target_n: Optional[int] = None
        if target_n_raw is not None:
            target_n = _as_int(target_n_raw, "data.target_n_per_source")
            if target_n <= 0:
                _fail(f"data.target_n_per_source must be > 0, got {target_n}")

        steps_per_epoch = _as_int(
            d.get("steps_per_epoch", 4000), "data.steps_per_epoch"
        )
        if steps_per_epoch <= 0:
            _fail(f"data.steps_per_epoch must be > 0, got {steps_per_epoch}")

        if source_groups is not None and balance is not None:
            _fail("data.source_groups and data.balance_cohorts are "
                  "mutually exclusive — use one or the other")

        batch_mode_raw = _as_str(
            d.get("batch_mode", "multi_source"), "data.batch_mode"
        ).lower()
        if batch_mode_raw not in ("chunk", "multi_source"):
            _fail(f"data.batch_mode must be 'chunk' or 'multi_source', "
                  f"got '{batch_mode_raw}'")

        if batch_mode_raw == "multi_source" and source_groups is None:
            _fail("data.batch_mode='multi_source' requires "
                  "data.source_groups to be set")

        selection_metric_raw = _as_str(
            d.get("selection_metric", "min_balanced_crps"),
            "data.selection_metric",
        ).lower()
        if selection_metric_raw not in (
            "min_balanced_crps", "min_weighted_source_crps",
            "max_gvw_r_lower_bound",
        ):
            _fail(f"data.selection_metric must be 'min_balanced_crps' "
                  f"or 'min_weighted_source_crps' or 'max_gvw_r_lower_bound', "
                  f"got '{selection_metric_raw}'")

        if selection_metric_raw == 'min_weighted_source_crps':
            if source_groups is None:
                _fail(
                    "data.selection_metric='min_weighted_source_crps' requires "
                    "data.source_groups"
                )
            expected = set(PLCA_WEIGHTED_SOURCE_WEIGHTS)
            actual = set(source_groups)
            if actual != expected:
                _fail(
                    "Weighted PLCA selection requires source_groups with keys "
                    f"{sorted(expected)}, got {sorted(actual)}"
                )

        return DataSpec(
            training_set=_as_str(d["training_set"], "data.training_set"),
            zarr_base=_as_str(
                d.get("zarr_base",
                      "/hpc/temp/ha_g/user/rpatton/ProteusZarrs"),
                "data.zarr_base"),
            seed=_as_int(d.get("seed", 42), "data.seed"),
            subsample_frac=subsample_frac,
            train_only_cohorts=train_only,
            balance_cohorts=balance,
            source_groups=source_groups,
            target_n_per_source=target_n,
            steps_per_epoch=steps_per_epoch,
            batch_mode=batch_mode_raw,
            selection_metric=selection_metric_raw,
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
            "phase", "n_trials_per_fold",
            "prev_study", "top_fraction", "enqueue_top_k",
            "promote_top_frac", "max_promotions", "pruner_type",
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="tuning")
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

    YAML layout uses three HP groups::

        structural:  ...
        training:    ...
        scale:       ...
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
        ]
        _check_no_unknown_keys(d, allowed=allowed, ctx="experiment")

        name = _as_str(d.get("name", "experiment"), "name")
        output_dir = _as_str(d.get("output_dir", "results_training"),
                             "output_dir")

        return ExperimentSpec(
            name=name,
            output_dir=output_dir,
            data=DataSpec.from_yaml(d["data"]),
            structural=StructuralSpec.from_yaml(d.get("structural", {})),
            training=TrainingSpec.from_yaml(d.get("training", {})),
            scale=ScaleSpec.from_yaml(d.get("scale", {})),
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
