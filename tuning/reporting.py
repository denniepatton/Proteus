"""Study reporting, phase-transition analysis, and fold stability.

Provides:

* ``write_study_report`` — best-trial summary and param importance.
* ``compute_fold_stability`` — cross-fold stability analysis for
  fold-parallel studies, including promoted-config stability from the
  promotion ledger.
* ``narrow_search_space_from_study`` — analyse a completed study and
  produce narrowed distributions for the next tuning phase.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import optuna

from tuning.metrics import nanmean, nanstd

__all__ = [
    "write_study_report",
    "compute_fold_stability",
    "narrow_search_space_from_study",
]


def _write_json_atomic(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def compute_fold_stability(
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
            "mean_score": round(nanmean(scores), 6),
            "best_trial": best_trial[0],
        }
        best_per_fold.append(best_score)

    mean_of_best = nanmean(best_per_fold)
    std_of_best = nanstd(best_per_fold) if len(best_per_fold) > 1 else 0.0
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
                pmean = nanmean(scores)
                pstd = nanstd(scores)
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


def write_study_report(study: optuna.Study, output_dir: str) -> None:
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
    compute_fold_stability(study, output_dir, direction)


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
