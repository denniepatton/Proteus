"""Cross-fold promotion of top hyperparameter configurations.

When fold-parallel tuning discovers a config that scores in the top
fraction of completed trials, the promotion system coordinates training
that config on every fold that hasn't tested it yet.  This gives a
complete cross-fold picture of the best configs discovered during
per-fold search.

Coordination between HPC workers uses a JSON ledger on shared storage
protected by ``fcntl`` file locking.
"""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna

from config.experiment_spec import ExperimentSpec, ExperimentSpecError
from tuning._io_utils import (
    folds_from_tsv as _folds_from_tsv,
    write_json_atomic as _write_json_atomic,
)
from tuning.metrics import nanmean, nanstd, objective_value
from tuning.param_resolution import (
    FixedParamTrial,
    config_hash,
    resolve_trial_configs,
)

__all__ = [
    "PromotionManager",
    "identify_top_configs",
    "run_promotion_phase",
]


class PromotionManager:
    """File-based cross-fold promotion of top HP configurations.

    The JSON ledger is safe for concurrent SLURM workers because all
    reads/writes are protected by an ``fcntl.flock`` exclusive lock.

    Ledger schema::

        {
          "configs": {
            "<config_hash>": {
              "params": { ... },
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
        chash = config_hash(params)
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
        self, config_hash_val: str, fold: int, score: float, trial_dir: str
    ) -> None:
        """Record a completed promotion result."""
        data, fh = self._lock_and_read()
        try:
            if config_hash_val in data["configs"]:
                data["configs"][config_hash_val]["fold_results"][str(fold)] = {
                    "status": "done",
                    "score": score,
                    "trial_dir": trial_dir,
                }
            self._write_and_unlock(data, fh)
        except (OSError, KeyError, TypeError, ValueError):
            self._unlock(fh)
            raise

    def record_failure(
        self, config_hash_val: str, fold: int, error: str
    ) -> None:
        """Record a failed promotion attempt."""
        data, fh = self._lock_and_read()
        try:
            if config_hash_val in data["configs"]:
                data["configs"][config_hash_val]["fold_results"][str(fold)] = {
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
                "mean_score": nanmean(scores) if scores else None,
                "std_score": nanstd(scores) if len(scores) > 1 else None,
                "fold_scores": {k: v.get("score") for k, v in done.items()},
            })
        return {"promoted_configs": summaries}


def identify_top_configs(
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
        chash = config_hash(trial.params)
        if chash not in seen:
            seen.add(chash)
            unique.append(trial)
            if len(unique) >= max_promotions:
                break

    return unique


def run_promotion_phase(
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
    pm = PromotionManager(ledger_path, all_folds)

    # Nominate the top configs discovered so far.
    top_trials = identify_top_configs(study, direction, top_frac, max_promo)
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
            mock_trial = FixedParamTrial(params)
            model_cfg, train_cfg, _tcfg = resolve_trial_configs(exp, mock_trial)

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
            if fold_metrics.get("weighted_tumor_gvw_crps") is not None:
                obj["weighted_tumor_gvw_crps_mean"] = float(fold_metrics["weighted_tumor_gvw_crps"])
            if fold_metrics.get("balanced_tumor_gvw_r") is not None:
                obj["balanced_tumor_gvw_r_mean"] = float(fold_metrics["balanced_tumor_gvw_r"])
                obj["balanced_tumor_gvw_r_std"] = float(fold_metrics.get("balanced_tumor_gvw_r_std", float("nan")))
            score = objective_value(
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
