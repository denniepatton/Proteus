#!/usr/bin/env python3
"""Phase 2-3 Optuna tuning results analysis.

Walks all 209 trial directories in biodvn_phase2_3_training/, reads
trial_config.json for HP values and training_history.json for metrics,
and produces summary tables.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results_training" / "biodvn_phase2_3_training"
N_TRIALS = 209  # trial_00000 .. trial_00208

STRUCTURAL_HPS = [
    "beta_point_mse",
    "beta_background",
    "beta_tcvae",
    "beta_contrastive",
    "beta_grl",
    "beta_reconstruction",
    "cl_margin",
    "internal_weight",
]

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def load_trial(trial_dir: Path):
    """Return a dict with trial metadata, HPs, and per-fold results.

    Returns None if the trial directory is empty / has no config.
    """
    config_path = trial_dir / "trial_config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    # Extract the 8 structural HPs from the config
    structural = config.get("structural", {})
    hp_values = {hp: structural.get(hp) for hp in STRUCTURAL_HPS}

    # Discover fold subdirectories
    fold_results = {}
    for child in sorted(trial_dir.iterdir()):
        if child.is_dir() and child.name.startswith("CV-fold_"):
            fold_num = int(child.name.split("_")[1])
            fold_info = _load_fold(child, fold_num)
            fold_results[fold_num] = fold_info

    if not fold_results:
        return None

    return {
        "trial_dir": trial_dir.name,
        "trial_idx": int(trial_dir.name.split("_")[1]),
        "hp_values": hp_values,
        "folds": fold_results,
    }


def _load_fold(fold_dir: Path, fold_num: int) -> dict:
    """Load training history + metrics for a single fold."""
    result = {
        "fold": fold_num,
        "completed": False,
        "n_epochs": 0,
        "best_crps": None,
        "best_epoch": None,
        "final_crps": None,
        "best_r2": None,
        "has_metrics_json": False,
        "metrics_crps": None,
    }

    # Training history
    hist_path = fold_dir / "training_history.json"
    if hist_path.exists():
        with open(hist_path) as f:
            hist = json.load(f)
        crps_list = hist.get("val_tumor_gvw_crps", [])
        r2_list = hist.get("val_tumor_gvw_r2", [])
        train_loss = hist.get("train_loss", [])

        n_epochs = len(train_loss)
        result["n_epochs"] = n_epochs
        result["completed"] = True

        if crps_list:
            best_crps = min(crps_list)
            best_epoch = crps_list.index(best_crps) + 1
            result["best_crps"] = best_crps
            result["best_epoch"] = best_epoch
            result["final_crps"] = crps_list[-1]

        if r2_list:
            result["best_r2"] = max(r2_list)
    else:
        # No training history → pruned or still running
        # Check if best_model.pt exists (training started but was pruned)
        if (fold_dir / "best_model.pt").exists():
            result["completed"] = False  # pruned
            result["n_epochs"] = 0
        # Otherwise the fold directory is essentially empty

    # Validation metrics JSON (written after full completion)
    metrics_files = list(fold_dir.glob("validation-fold_*_metrics.json"))
    if metrics_files:
        result["has_metrics_json"] = True
        with open(metrics_files[0]) as f:
            metrics = json.load(f)
        tumor = metrics.get("tumor", {})
        result["metrics_crps"] = tumor.get("gvw_crps")
        result["metrics_r2"] = tumor.get("r2")
        result["metrics_gvw_r2"] = tumor.get("gvw_mean_r2")

    return result


def collect_all_trials() -> list:
    """Walk every trial directory and return a list of trial dicts."""
    trials = []
    for i in range(N_TRIALS):
        trial_dir = RESULTS_DIR / f"trial_{i:05d}"
        if not trial_dir.exists():
            continue
        info = load_trial(trial_dir)
        if info is not None:
            trials.append(info)
    return trials


# ---------------------------------------------------------------------------
# Analysis / Reporting
# ---------------------------------------------------------------------------

def print_overview(trials: list) -> None:
    """Print high-level overview."""
    total = len(trials)
    completed_folds = 0
    pruned_folds = 0
    fold_counter = defaultdict(int)
    fold_completed = defaultdict(int)

    for t in trials:
        for fold_num, finfo in t["folds"].items():
            fold_counter[fold_num] += 1
            if finfo["completed"]:
                completed_folds += 1
                fold_completed[fold_num] += 1
            else:
                pruned_folds += 1

    print("=" * 80)
    print("PHASE 2-3 OPTUNA TUNING RESULTS — OVERVIEW")
    print("=" * 80)
    print(f"Total trials with config:     {total}")
    print(f"Total fold runs:              {completed_folds + pruned_folds}")
    print(f"  Completed (has history):    {completed_folds}")
    print(f"  Pruned (no history):        {pruned_folds}")
    print()
    print("Per-fold breakdown:")
    print(f"  {'Fold':<8} {'Total':<8} {'Completed':<12} {'Pruned':<8}")
    for fold in sorted(fold_counter.keys()):
        total_f = fold_counter[fold]
        comp_f = fold_completed[fold]
        prun_f = total_f - comp_f
        print(f"  {fold:<8} {total_f:<8} {comp_f:<12} {prun_f:<8}")
    print()


def print_all_trials_table(trials: list) -> None:
    """Print a summary table of ALL completed fold-results, grouped by fold, sorted by CRPS."""
    # Flatten into (trial_idx, fold_num, fold_info) rows
    rows_by_fold = defaultdict(list)
    for t in trials:
        for fold_num, finfo in t["folds"].items():
            if finfo["completed"] and finfo["best_crps"] is not None:
                rows_by_fold[fold_num].append({
                    "trial_idx": t["trial_idx"],
                    "best_crps": finfo["best_crps"],
                    "best_epoch": finfo["best_epoch"],
                    "n_epochs": finfo["n_epochs"],
                    "best_r2": finfo["best_r2"],
                    "final_crps": finfo["final_crps"],
                    "metrics_crps": finfo.get("metrics_crps"),
                    "hp_values": t["hp_values"],
                })

    for fold in sorted(rows_by_fold.keys()):
        rows = sorted(rows_by_fold[fold], key=lambda r: r["best_crps"])
        print("=" * 100)
        print(f"FOLD {fold} — {len(rows)} completed trials (sorted by best CRPS, ascending)")
        print("=" * 100)
        header = f"{'Rank':<6}{'Trial':<10}{'Best CRPS':<12}{'Final CRPS':<12}{'Best R²':<10}{'BestEp':<8}{'Epochs':<8}{'Metrics CRPS':<14}"
        print(header)
        print("-" * len(header))
        for rank, row in enumerate(rows, 1):
            mcr = f"{row['metrics_crps']:.4f}" if row['metrics_crps'] is not None else "N/A"
            br2 = f"{row['best_r2']:.4f}" if row['best_r2'] is not None else "N/A"
            print(
                f"{rank:<6}{row['trial_idx']:<10}"
                f"{row['best_crps']:<12.4f}"
                f"{row['final_crps']:<12.4f}"
                f"{br2:<10}"
                f"{row['best_epoch']:<8}{row['n_epochs']:<8}"
                f"{mcr:<14}"
            )
        print()


def print_top_trials_hp_table(trials: list, top_n: int = 10) -> None:
    """For the top N trials per fold (by CRPS), print the full 8 structural HP values."""
    rows_by_fold = defaultdict(list)
    for t in trials:
        for fold_num, finfo in t["folds"].items():
            if finfo["completed"] and finfo["best_crps"] is not None:
                rows_by_fold[fold_num].append({
                    "trial_idx": t["trial_idx"],
                    "best_crps": finfo["best_crps"],
                    "hp_values": t["hp_values"],
                })

    for fold in sorted(rows_by_fold.keys()):
        rows = sorted(rows_by_fold[fold], key=lambda r: r["best_crps"])[:top_n]
        print("=" * 140)
        print(f"FOLD {fold} — TOP {len(rows)} TRIALS (HP values)")
        print("=" * 140)

        # Header
        hp_headers = [hp[:16] for hp in STRUCTURAL_HPS]
        header = f"{'Rank':<6}{'Trial':<8}{'CRPS':<10}" + "".join(f"{h:<18}" for h in hp_headers)
        print(header)
        print("-" * len(header))

        for rank, row in enumerate(rows, 1):
            hp_vals = "".join(
                f"{row['hp_values'].get(hp, 'N/A'):<18.6f}"
                if isinstance(row['hp_values'].get(hp), (int, float))
                else f"{'N/A':<18}"
                for hp in STRUCTURAL_HPS
            )
            print(f"{rank:<6}{row['trial_idx']:<8}{row['best_crps']:<10.4f}{hp_vals}")
        print()


def print_cross_fold_stats(trials: list) -> None:
    """For trials that completed on multiple folds, compute cross-fold mean CRPS."""
    # Find trials that appear on multiple folds
    trial_fold_crps = defaultdict(dict)  # trial_idx -> {fold: best_crps}
    trial_hps = {}
    for t in trials:
        for fold_num, finfo in t["folds"].items():
            if finfo["completed"] and finfo["best_crps"] is not None:
                trial_fold_crps[t["trial_idx"]][fold_num] = finfo["best_crps"]
                trial_hps[t["trial_idx"]] = t["hp_values"]

    print("=" * 100)
    print("CROSS-FOLD STATISTICS")
    print("=" * 100)

    # Group by number of folds completed
    by_n_folds = defaultdict(list)
    for trial_idx, fold_map in trial_fold_crps.items():
        by_n_folds[len(fold_map)].append(trial_idx)

    print("\nTrials by number of completed folds:")
    for n in sorted(by_n_folds.keys()):
        print(f"  {n} fold(s): {len(by_n_folds[n])} trials")

    # For each trial, compute mean CRPS across completed folds
    print("\n--- All trials with mean CRPS across completed folds (sorted) ---")
    summary_rows = []
    for trial_idx, fold_map in trial_fold_crps.items():
        mean_crps = sum(fold_map.values()) / len(fold_map)
        folds_str = ",".join(str(f) for f in sorted(fold_map.keys()))
        summary_rows.append({
            "trial_idx": trial_idx,
            "n_folds": len(fold_map),
            "mean_crps": mean_crps,
            "folds": folds_str,
            "fold_crps": fold_map,
        })

    summary_rows.sort(key=lambda r: r["mean_crps"])

    header = f"{'Rank':<6}{'Trial':<10}{'#Folds':<8}{'Mean CRPS':<12}{'Folds':<20}{'Per-fold CRPS':<40}"
    print(header)
    print("-" * len(header))
    for rank, row in enumerate(summary_rows, 1):
        per_fold = "  ".join(f"F{f}:{c:.4f}" for f, c in sorted(row["fold_crps"].items()))
        print(
            f"{rank:<6}{row['trial_idx']:<10}"
            f"{row['n_folds']:<8}{row['mean_crps']:<12.4f}"
            f"{row['folds']:<20}{per_fold}"
        )

    # If any trial has 5 folds, show it prominently
    full_cv = [r for r in summary_rows if r["n_folds"] == 5]
    if full_cv:
        print(f"\n--- Trials with all 5 folds completed ({len(full_cv)}) ---")
        for rank, row in enumerate(full_cv, 1):
            per_fold = "  ".join(f"F{f}:{c:.4f}" for f, c in sorted(row["fold_crps"].items()))
            print(f"  {rank}. Trial {row['trial_idx']:05d}  mean={row['mean_crps']:.4f}  {per_fold}")

    # Also show top-20 overall with HP values
    print(f"\n--- TOP 20 trials by mean CRPS (with HP values) ---")
    hp_headers = [hp[:14] for hp in STRUCTURAL_HPS]
    header2 = f"{'Rank':<6}{'Trial':<8}{'#F':<4}{'Mean CRPS':<11}" + "".join(f"{h:<16}" for h in hp_headers)
    print(header2)
    print("-" * len(header2))
    for rank, row in enumerate(summary_rows[:20], 1):
        hps = trial_hps.get(row["trial_idx"], {})
        hp_str = "".join(
            f"{hps.get(hp, 0):<16.6f}" if isinstance(hps.get(hp), (int, float)) else f"{'N/A':<16}"
            for hp in STRUCTURAL_HPS
        )
        print(f"{rank:<6}{row['trial_idx']:<8}{row['n_folds']:<4}{row['mean_crps']:<11.4f}{hp_str}")
    print()


def print_hp_range_summary(trials: list) -> None:
    """For completed trials, summarize the range of each HP and the range in top quartile."""
    all_crps_hp = []
    for t in trials:
        for fold_num, finfo in t["folds"].items():
            if finfo["completed"] and finfo["best_crps"] is not None:
                all_crps_hp.append((finfo["best_crps"], t["hp_values"]))

    if not all_crps_hp:
        return

    all_crps_hp.sort(key=lambda x: x[0])
    n = len(all_crps_hp)
    q25_cutoff = n // 4

    print("=" * 100)
    print("HP RANGE SUMMARY (all completed fold-results)")
    print("=" * 100)
    print(f"Total completed fold-results: {n}")
    print(f"Top 25% cutoff (rank {q25_cutoff}): CRPS <= {all_crps_hp[q25_cutoff][0]:.4f}")
    print()

    header = f"{'HP':<22}{'All Min':<12}{'All Max':<12}{'All Mean':<12}{'Top25 Min':<12}{'Top25 Max':<12}{'Top25 Mean':<12}"
    print(header)
    print("-" * len(header))

    for hp in STRUCTURAL_HPS:
        all_vals = [row[1].get(hp) for row in all_crps_hp if row[1].get(hp) is not None]
        top_vals = [row[1].get(hp) for row in all_crps_hp[:q25_cutoff] if row[1].get(hp) is not None]

        if all_vals:
            a_min, a_max, a_mean = min(all_vals), max(all_vals), sum(all_vals) / len(all_vals)
        else:
            a_min = a_max = a_mean = float("nan")

        if top_vals:
            t_min, t_max, t_mean = min(top_vals), max(top_vals), sum(top_vals) / len(top_vals)
        else:
            t_min = t_max = t_mean = float("nan")

        print(
            f"{hp:<22}{a_min:<12.4f}{a_max:<12.4f}{a_mean:<12.4f}"
            f"{t_min:<12.4f}{t_max:<12.4f}{t_mean:<12.4f}"
        )
    print()


def print_pruned_trials_summary(trials: list) -> None:
    """List trials that were pruned (fold started but no training history)."""
    pruned = []
    for t in trials:
        for fold_num, finfo in t["folds"].items():
            if not finfo["completed"]:
                pruned.append((t["trial_idx"], fold_num))

    if not pruned:
        print("No pruned trials found.\n")
        return

    print("=" * 60)
    print(f"PRUNED FOLD RUNS ({len(pruned)} total)")
    print("=" * 60)
    header = f"{'Trial':<10}{'Fold':<8}"
    print(header)
    print("-" * len(header))
    for trial_idx, fold_num in sorted(pruned):
        print(f"{trial_idx:<10}{fold_num:<8}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Scanning: {RESULTS_DIR}")
    print(f"Expected trials: {N_TRIALS} (trial_00000 .. trial_{N_TRIALS - 1:05d})")
    print()

    trials = collect_all_trials()

    print_overview(trials)
    print_all_trials_table(trials)
    print_top_trials_hp_table(trials, top_n=10)
    print_cross_fold_stats(trials)
    print_hp_range_summary(trials)
    print_pruned_trials_summary(trials)


if __name__ == "__main__":
    main()
