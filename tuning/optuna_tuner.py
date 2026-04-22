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

import csv
import json
import math
import os
import sys
import traceback
import warnings
from typing import Any, Dict, List, Literal, Optional

import optuna

# Suppress Optuna experimental feature warnings (multivariate, constant_liar are stable in practice)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

from config.experiment_spec import (
    ExperimentSpec,
    ExperimentSpecError,
    ParamSpec,
    load_experiment_spec,
)
from tuning.metrics import (
    collect_fold_metrics,
    compute_cv_objective,
    nanmean,
    nanstd,
    objective_value,
)
from tuning.param_resolution import (
    config_hash,
    configs_from_trial_json,
    resolve_trial_configs,
)
from tuning.promotion import run_promotion_phase
from tuning.reporting import (
    narrow_search_space_from_study,
    write_study_report,
)

__all__ = [
    "run_optuna_cv_tuning",
    "run_optuna_fold_tuning",
    "coordinated_ask",
    "coordinated_proxy_ask",
    "coordinated_proxy_tell",
    "coordinated_proxy_finalize",
    "coordinated_tell",
    "run_coordinated_fold",
]


PROXY_OBJECTIVE_FILENAME = "proxy_objective.json"
PROMOTED_CV_SUMMARY_FILENAME = "promoted_cv_summary.json"
DEFAULT_PROXY_FOLDS = (3, 5)


# =============================================================================
# Hyperband / Phased Search Configuration
# =============================================================================

HYPERBAND_MIN_RESOURCE = 3
HYPERBAND_REDUCTION_FACTOR = 3


# =============================================================================
# Small Utilities
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


def _fold_metrics_path(run_dir: str, fold_num: int) -> str:
    return os.path.join(
        run_dir,
        f"CV-fold_{int(fold_num)}",
        f"validation-fold_{int(fold_num)}_metrics.json",
    )


def _primary_fold_weighted_crps(fold_metrics: Dict[str, Any]) -> float:
    value = fold_metrics.get("weighted_tumor_gvw_crps")
    if value is None:
        value = fold_metrics.get("balanced_tumor_gvw_crps")
    if value is None:
        raise KeyError("fold metrics are missing weighted_tumor_gvw_crps")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("fold weighted_tumor_gvw_crps is non-finite")
    return value


def _existing_config_hashes(output_dir: str) -> set[str]:
    hashes: set[str] = set()
    if not os.path.isdir(output_dir):
        return hashes

    for entry in os.listdir(output_dir):
        if not entry.startswith("trial_"):
            continue
        cfg_path = os.path.join(output_dir, entry, "trial_config.json")
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        hashes.add(config_hash(cfg))

    return hashes


def compute_proxy_objective(
    run_dir: str,
    proxy_folds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute the scout objective from a fixed set of proxy folds.

    The proxy objective is the mean weighted tumor GVW-CRPS across the
    designated proxy folds. A proxy trial is only eligible for Optuna tell
    once every proxy fold has produced a metrics JSON.
    """
    folds = list(proxy_folds) if proxy_folds is not None else list(DEFAULT_PROXY_FOLDS)
    if not folds:
        raise ValueError("proxy_folds must be non-empty")

    proxy_scores: List[float] = []
    proxy_balanced_r: List[float] = []
    proxy_fold_metrics: Dict[str, Dict[str, float]] = {}

    for fold_num in folds:
        metrics_path = _fold_metrics_path(run_dir, int(fold_num))
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Proxy fold metrics not found: {metrics_path}")

        with open(metrics_path, "r") as f:
            fold_metrics = json.load(f)

        weighted_crps = _primary_fold_weighted_crps(fold_metrics)
        proxy_scores.append(weighted_crps)

        entry: Dict[str, float] = {
            "weighted_tumor_gvw_crps": weighted_crps,
        }
        balanced_r = fold_metrics.get("balanced_tumor_gvw_r")
        if balanced_r is not None and math.isfinite(float(balanced_r)):
            entry["balanced_tumor_gvw_r"] = float(balanced_r)
            proxy_balanced_r.append(float(balanced_r))

        proxy_fold_metrics[str(int(fold_num))] = entry

    obj: Dict[str, Any] = {
        "proxy_folds": [int(fold) for fold in folds],
        "proxy_folds_with_metrics": len(proxy_fold_metrics),
        "proxy_weighted_tumor_gvw_crps_mean": nanmean(proxy_scores),
        "proxy_weighted_tumor_gvw_crps_std": nanstd(proxy_scores),
        "proxy_fold_metrics": proxy_fold_metrics,
    }
    if proxy_balanced_r:
        obj["proxy_balanced_tumor_gvw_r_mean"] = nanmean(proxy_balanced_r)
        obj["proxy_balanced_tumor_gvw_r_std"] = nanstd(proxy_balanced_r)

    out_path = os.path.join(run_dir, PROXY_OBJECTIVE_FILENAME)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)

    return obj


def _create_study_with_seeding(
    exp: ExperimentSpec,
    study_name: str,
    direction: str,
    storage: str,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner,
) -> optuna.Study:
    """Create or load an Optuna study and seed from previous phase if configured."""
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

    return study


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
    subsample_frac = exp.data.subsample_frac

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

    study = _create_study_with_seeding(
        exp, study_name, direction, storage, sampler, pruner,
    )

    from Proteus import train_model_cv  # local import to avoid circulars

    def objective(trial: optuna.Trial) -> float:
        model_cfg, train_cfg, trial_cfg = resolve_trial_configs(exp, trial)

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
            # CUDA infrastructure failure — terminate worker immediately.
            _err = str(exc).lower()
            if 'cuda' in _err and ('unknown error' in _err or 'device-side assert' in _err):
                print(f"[Trial {trial.number}] CUDA infrastructure failure — "
                      f"terminating worker: {exc}", file=sys.stderr, flush=True)
                raise SystemExit(1)
            # Training diverged — return worst-case value.
            print(f"[Trial {trial.number}] Training failed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            trial.set_user_attr("error", str(exc))
            return float("inf") if direction == "minimize" else float("-inf")

        obj = compute_cv_objective(run_dir)
        for k, v in obj.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                trial.set_user_attr(k, float(v))

        return objective_value(obj, metric=exp.objective.metric, direction=direction)

    study.optimize(objective, n_trials=int(n_trials_effective), catch=(Exception,))

    write_study_report(study, exp.output_dir)


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

    study = _create_study_with_seeding(
        exp, study_name, direction, storage, sampler, pruner,
    )

    from Proteus import train_model_cv  # local import to avoid circulars

    def objective(trial: optuna.Trial) -> float:
        model_cfg, train_cfg, trial_cfg = resolve_trial_configs(exp, trial)

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
            val_score.  For minimize: prefer the weighted source metric when
            that is the study objective, else fall back to balanced / pooled
            GVW-CRPS.

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
                if exp.objective.metric == 'weighted_tumor_gvw_crps_mean':
                    metric_val = val_metrics.get(
                        'val_weighted_tumor_gvw_crps',
                        val_metrics.get('val_balanced_tumor_gvw_crps', val_score),
                    )
                else:
                    metric_val = val_metrics.get(
                        'val_balanced_tumor_gvw_crps',
                        val_metrics.get('val_tumor_gvw_crps', val_score),
                    )
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
            # CUDA infrastructure failure — terminate worker immediately.
            _err = str(exc).lower()
            if 'cuda' in _err and ('unknown error' in _err or 'device-side assert' in _err):
                print(f"[Trial {trial.number}, Fold {fold_num}] CUDA "
                      f"infrastructure failure — terminating worker: {exc}", file=sys.stderr, flush=True)
                raise SystemExit(1)
            # Training diverged — return worst-case value.
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

        # Build a single-fold objective dict compatible with objective_value
        tumor = fold_metrics.get("tumor", {})
        bg = fold_metrics.get("background", {})
        obj = {
            "tumor_mae_mean": tumor.get("mae", float("nan")),
            "tumor_r2_mean": tumor.get("r2", float("nan")),
            "tumor_gvw_crps_mean": tumor.get("gvw_crps", float("nan")),
            "background_mae_mean": bg.get("mae", float("nan")),
            "background_r2_mean": bg.get("r2", float("nan")),
            "background_gvw_crps_mean": bg.get("gvw_crps", float("nan")),
            "val_loss": nanmean([
                tumor.get("mae", float("nan")),
                bg.get("mae", float("nan")),
            ]),
        }
        # Balanced fine-tuning objective (single-fold value)
        balanced_val = fold_metrics.get("balanced_tumor_gvw_crps", None)
        if balanced_val is not None:
            obj["balanced_tumor_gvw_crps_mean"] = float(balanced_val)
            obj["balanced_tumor_gvw_crps_std"] = float(fold_metrics.get("balanced_tumor_gvw_crps_std", float("nan")))
        weighted_val = fold_metrics.get("weighted_tumor_gvw_crps", None)
        if weighted_val is not None:
            obj["weighted_tumor_gvw_crps_mean"] = float(weighted_val)
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

        return objective_value(obj, metric=exp.objective.metric, direction=direction)

    study.optimize(objective, n_trials=int(n_trials_effective), catch=(Exception,))

    # Cross-fold promotion phase (runs after main trial loop)
    run_promotion_phase(study, exp, fold_num, direction, train_model_cv)

    write_study_report(study, exp.output_dir)


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

    # Probe the study to count existing trials *before* building the sampler.
    # TPESampler(seed=S) resets its internal RNG to seed S each time it is
    # instantiated.  During the random startup phase (n_startup_trials=10 by
    # default), the sampler draws from this RNG without conditioning on
    # completed trials, so recreating the sampler with the same seed across
    # rounds produces identical suggestions.  Offsetting the seed by the
    # number of existing trials makes each round deterministic-but-unique.
    _probe_store = _make_storage(storage)
    try:
        _probe = optuna.load_study(
            study_name=study_name, storage=_probe_store)
        n_existing = len(_probe.trials)
    except KeyError:
        n_existing = 0
    effective_seed = int(sampler_seed) + n_existing

    sampler = optuna.samplers.TPESampler(
        seed=effective_seed,
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
        n_startup_trials=n_configs,
    )
    pruner = optuna.pruners.NopPruner()

    study = _create_study_with_seeding(
        exp, study_name, direction, storage, sampler, pruner,
    )

    trial_numbers: List[int] = []
    for _ in range(n_configs):
        trial = study.ask()
        try:
            _, _, trial_cfg = resolve_trial_configs(exp, trial)
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


def coordinated_proxy_ask(
    experiment_path: str,
    n_configs: int,
    storage: str,
    study_name: Optional[str] = None,
    sampler_seed: int = 42,
) -> List[int]:
    """Ask Optuna for proxy-fold trial configs.

    Proxy trials are canonical for the proxy racing scheduler: the same trial
    directory is later reused for any promoted full-CV completion folds.
    Duplicate configs are pruned before they reach the queue.
    """
    exp = load_experiment_spec(experiment_path)
    study_name = str(study_name) if study_name is not None else exp.name
    direction = exp.objective.direction

    _probe_store = _make_storage(storage)
    try:
        _probe = optuna.load_study(study_name=study_name, storage=_probe_store)
        n_existing = len(_probe.trials)
    except KeyError:
        n_existing = 0
    effective_seed = int(sampler_seed) + n_existing

    sampler = optuna.samplers.TPESampler(
        seed=effective_seed,
        multivariate=True,
        constant_liar=True,
        warn_independent_sampling=False,
        n_startup_trials=max(8, int(n_configs)),
    )
    pruner = optuna.pruners.NopPruner()

    study = _create_study_with_seeding(
        exp, study_name, direction, storage, sampler, pruner,
    )

    existing_hashes = _existing_config_hashes(exp.output_dir)
    trial_numbers: List[int] = []

    for _ in range(int(n_configs)):
        accepted = False
        for _attempt in range(32):
            trial = study.ask()
            try:
                _, _, trial_cfg = resolve_trial_configs(exp, trial)
            except optuna.TrialPruned as exc:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print(f"[Proxy-ask] Trial {trial.number} pruned: {exc}", flush=True)
                continue

            cfg_hash = config_hash(trial_cfg)
            if cfg_hash in existing_hashes:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print(
                    f"[Proxy-ask] Trial {trial.number:05d} duplicates an existing config "
                    f"({cfg_hash}); pruning.",
                    flush=True,
                )
                continue

            trial.set_user_attr("config_hash", cfg_hash)
            trial.set_user_attr("proxy_folds", list(DEFAULT_PROXY_FOLDS))

            run_dir = os.path.join(exp.output_dir, f"trial_{trial.number:05d}")
            os.makedirs(run_dir, exist_ok=True)

            _write_json_atomic(os.path.join(run_dir, "trial_config.json"), trial_cfg)
            _write_json_atomic(
                os.path.join(run_dir, "coordinated_pending.json"),
                {
                    "trial_number": trial.number,
                    "status": "proxy_pending",
                    "proxy_folds": list(DEFAULT_PROXY_FOLDS),
                },
            )

            existing_hashes.add(cfg_hash)
            trial_numbers.append(trial.number)
            print(f"[Proxy-ask] Asked trial {trial.number:05d} → {run_dir}", flush=True)
            accepted = True
            break

        if not accepted:
            print("[Proxy-ask] Exhausted attempts to find a non-duplicate config.", flush=True)

    _write_json_atomic(
        os.path.join(exp.output_dir, "coordinated_active.json"),
        {"trial_numbers": trial_numbers},
    )

    return trial_numbers


def coordinated_proxy_tell(
    experiment_path: str,
    trial_numbers: Optional[List[int]] = None,
    storage: str = "sqlite:///results_training/optuna.db",
    study_name: Optional[str] = None,
    sampler_seed: int = 42,
    proxy_folds: Optional[List[int]] = None,
) -> List[int]:
    """Report completed proxy-fold results to the proxy Optuna study.

    Unlike coordinated_tell(), incomplete trials are *not* failed here. They
    are simply skipped until every proxy fold has written its metrics JSON.
    """
    exp = load_experiment_spec(experiment_path)
    study_name = str(study_name) if study_name is not None else exp.name
    direction = exp.objective.direction
    folds = list(proxy_folds) if proxy_folds is not None else list(DEFAULT_PROXY_FOLDS)

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

    if trial_numbers is None:
        active_path = os.path.join(exp.output_dir, "coordinated_active.json")
        if not os.path.exists(active_path):
            print("[Proxy-tell] No coordinated_active.json found.", flush=True)
            return
        with open(active_path, "r") as f:
            trial_numbers = json.load(f)["trial_numbers"]

    frozen_id_map: Dict[int, int] = {
        trial.number: trial._trial_id for trial in study.trials  # type: ignore[attr-defined]
    }

    reported: List[int] = []
    for trial_num in trial_numbers:
        run_dir = os.path.join(exp.output_dir, f"trial_{trial_num:05d}")
        try:
            obj = compute_proxy_objective(run_dir, folds)
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"[Proxy-tell] Trial {trial_num:05d}: proxy folds not ready ({exc}).", flush=True)
            continue

        value = obj.get("proxy_weighted_tumor_gvw_crps_mean")
        if value is None or not math.isfinite(float(value)):
            print(f"[Proxy-tell] Trial {trial_num:05d}: proxy objective is non-finite.", flush=True)
            continue

        try:
            if trial_num in frozen_id_map:
                trial_id = frozen_id_map[trial_num]
                for step, fold_num in enumerate(folds):
                    fold_entry = obj["proxy_fold_metrics"].get(str(int(fold_num)), {})
                    fold_value = fold_entry.get("weighted_tumor_gvw_crps")
                    if fold_value is not None and math.isfinite(float(fold_value)):
                        study._storage.set_trial_intermediate_value(
                            trial_id, step, float(fold_value),
                        )
        except (RuntimeError, AttributeError, KeyError) as exc:
            print(
                f"[Proxy-tell] Trial {trial_num:05d}: could not set proxy intermediate values: {exc}",
                flush=True,
            )

        try:
            study.tell(trial_num, float(value))
        except (RuntimeError, ValueError) as exc:
            frozen = next((trial for trial in study.trials if trial.number == trial_num), None)
            if frozen is not None and frozen.state == optuna.trial.TrialState.COMPLETE:
                print(
                    f"[Proxy-tell] Trial {trial_num:05d}: already reported in study; "
                    f"marking as complete.",
                    flush=True,
                )
                reported.append(int(trial_num))
                continue
            print(f"[Proxy-tell] Trial {trial_num:05d}: tell failed: {exc}", flush=True)
            continue

        reported.append(int(trial_num))
        print(
            f"[Proxy-tell] Trial {trial_num:05d}: "
            f"proxy_weighted_tumor_gvw_crps_mean={float(value):.4f}",
            flush=True,
        )

    write_study_report(study, exp.output_dir)
    print(f"[Proxy-tell] Reported {len(reported)} proxy trial(s).", flush=True)
    return reported


def coordinated_proxy_finalize(
    experiment_path: str,
    trial_numbers: Optional[List[int]] = None,
) -> List[int]:
    """Finalize promoted trials by writing cross-fold CV objective summaries.

    This does not report to Optuna. It computes the full CV objective for any
    promoted trial with all folds complete and updates a sorted summary JSON.
    """
    exp = load_experiment_spec(experiment_path)
    folds = _folds_from_tsv(exp.data.training_set)

    if trial_numbers is None:
        active_path = os.path.join(exp.output_dir, "coordinated_active.json")
        if not os.path.exists(active_path):
            print("[Proxy-finalize] No coordinated_active.json found.", flush=True)
            return
        with open(active_path, "r") as f:
            trial_numbers = json.load(f)["trial_numbers"]

    summary_path = os.path.join(exp.output_dir, PROMOTED_CV_SUMMARY_FILENAME)
    existing: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            for row in summary.get("records", []):
                trial_num = int(row["trial_number"])
                existing[trial_num] = row
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            existing = {}

    finalized: List[int] = []
    for trial_num in trial_numbers:
        run_dir = os.path.join(exp.output_dir, f"trial_{trial_num:05d}")
        if not all(os.path.exists(_fold_metrics_path(run_dir, fold_num)) for fold_num in folds):
            print(f"[Proxy-finalize] Trial {trial_num:05d}: full CV not complete — skipping.", flush=True)
            continue

        try:
            obj = compute_cv_objective(run_dir)
        except (RuntimeError, FileNotFoundError, KeyError, ValueError) as exc:
            print(f"[Proxy-finalize] Trial {trial_num:05d}: CV objective failed: {exc}", flush=True)
            continue

        record: Dict[str, Any] = {"trial_number": int(trial_num), **obj}
        proxy_path = os.path.join(run_dir, PROXY_OBJECTIVE_FILENAME)
        if os.path.exists(proxy_path):
            try:
                with open(proxy_path, "r") as f:
                    proxy_obj = json.load(f)
                proxy_score = proxy_obj.get("proxy_weighted_tumor_gvw_crps_mean")
                if proxy_score is not None and math.isfinite(float(proxy_score)):
                    record["proxy_weighted_tumor_gvw_crps_mean"] = float(proxy_score)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass

        existing[int(trial_num)] = record
        finalized.append(int(trial_num))

        metric_value = record.get(exp.objective.metric)
        if metric_value is not None and math.isfinite(float(metric_value)):
            print(
                f"[Proxy-finalize] Trial {trial_num:05d}: "
                f"{exp.objective.metric}={float(metric_value):.4f}",
                flush=True,
            )

    def sort_key(row: Dict[str, Any]) -> Tuple[float, int]:
        value = row.get(exp.objective.metric)
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            fallback = float("inf") if exp.objective.direction == "minimize" else float("-inf")
            return fallback, int(row["trial_number"])
        return float(value), int(row["trial_number"])

    records = list(existing.values())
    reverse = exp.objective.direction == "maximize"
    records.sort(key=sort_key, reverse=reverse)

    _write_json_atomic(
        summary_path,
        {
            "objective_metric": exp.objective.metric,
            "direction": exp.objective.direction,
            "records": records,
        },
    )
    print(f"[Proxy-finalize] Finalized {len(finalized)} trial(s).", flush=True)
    return finalized


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

    # Build trial number → internal ID map for intermediate value reporting.
    # Intermediate values (per-fold metrics) must be set while the trial is
    # still in RUNNING state, i.e. before study.tell().  This uses Optuna's
    # storage-level API because coordinated mode separates ask/tell processes.
    _frozen_id_map: Dict[int, int] = {
        t.number: t._trial_id for t in study.trials  # type: ignore[attr-defined]
    }

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
            value = objective_value(obj, metric=metric, direction=direction)
        except ExperimentSpecError as exc:
            print(f"[Coordinated-tell] Trial {trial_num:05d}: {exc}", flush=True)
            study.tell(trial_num, state=optuna.trial.TrialState.FAIL)
            continue

        # Report per-fold objective-aligned intermediate values so the Optuna
        # Dashboard can display fold-level progress plots.
        try:
            if trial_num in _frozen_id_map:
                _tid = _frozen_id_map[trial_num]
                fold_rows = collect_fold_metrics(run_dir)
                for step, row in enumerate(fold_rows):
                    if metric == 'weighted_tumor_gvw_crps_mean':
                        fold_value = row.get('weighted_tumor_gvw_crps')
                    elif direction == 'maximize':
                        fold_value = row.get('balanced_tumor_gvw_r')
                    else:
                        fold_value = row.get('balanced_tumor_gvw_crps')
                    if (fold_value is not None
                            and isinstance(fold_value, (int, float))
                            and math.isfinite(float(fold_value))):
                        study._storage.set_trial_intermediate_value(
                            _tid, step, float(fold_value))
        except (RuntimeError, AttributeError) as exc:
            print(f"[Coordinated-tell] Trial {trial_num:05d}: "
                  f"could not set intermediate values: {exc}", flush=True)

        # Report to Optuna — cv_objective.json on disk preserves full
        # cross-fold metrics; the study only stores the scalar objective.
        study.tell(trial_num, value)

        print(f"[Coordinated-tell] Trial {trial_num:05d}: "
              f"{metric}={value:.4f}", flush=True)

        # Remove pending sentinel
        pending = os.path.join(run_dir, "coordinated_pending.json")
        if os.path.exists(pending):
            os.remove(pending)

    write_study_report(study, exp.output_dir)
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

    model_cfg, train_cfg = configs_from_trial_json(trial_cfg, exp)

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
        seed=exp.data.seed,
        subsample_frac=exp.data.subsample_frac,
        train_only_cohorts=exp.data.train_only_cohorts,
        balance_cohorts=exp.data.balance_cohorts,
        source_groups=exp.data.source_groups,
        target_n_per_source=exp.data.target_n_per_source,
        batch_mode=exp.data.batch_mode,
    )
