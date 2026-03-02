#!/usr/bin/env python3
"""Phase 2-3 deep HP analysis: per-HP convergence + cross-fold patterns."""
from __future__ import annotations
import json, statistics
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parent / "results_training" / "biodvn_phase2_3_training"
STRUCTURAL_HPS = [
    "beta_point_mse", "beta_background", "beta_tcvae", "beta_contrastive",
    "beta_grl", "beta_reconstruction", "cl_margin", "internal_weight",
]

def load_all():
    rows = []  # (trial_idx, fold, best_crps, hp_dict)
    for td in sorted(RESULTS_DIR.iterdir()):
        if not td.name.startswith("trial_"):
            continue
        cfg_p = td / "trial_config.json"
        if not cfg_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        struct = cfg.get("structural", {})
        for child in td.iterdir():
            if child.is_dir() and child.name.startswith("CV-fold_"):
                fold = int(child.name.split("_")[1])
                hist_p = child / "training_history.json"
                if not hist_p.exists():
                    continue
                hist = json.loads(hist_p.read_text())
                crps_list = hist.get("val_tumor_gvw_crps", [])
                r2_list = hist.get("val_tumor_gvw_r2", [])
                if not crps_list:
                    continue
                rows.append((
                    int(td.name.split("_")[1]),
                    fold,
                    min(crps_list),
                    max(r2_list) if r2_list else None,
                    len(crps_list),
                    struct,
                ))
    return rows

def main():
    rows = load_all()
    print(f"Completed fold-results: {len(rows)}")

    # ---- Per-fold best CRPS comparison with phase 2-2 ----
    p22_best = {1: 0.7763, 2: None, 3: 1.2100, 4: None, 5: 0.8374}
    by_fold = defaultdict(list)
    for trial_idx, fold, crps, r2, n_ep, hp in rows:
        by_fold[fold].append((crps, trial_idx, r2, n_ep, hp))
    
    print("\n=== FOLD COMPARISON: Phase 2-3 vs 2-2 ===")
    for fold in sorted(by_fold.keys()):
        entries = sorted(by_fold[fold])
        best_crps = entries[0][0]
        p22 = p22_best.get(fold)
        delta = f"  Δ={best_crps - p22:+.4f}" if p22 else "  (no 2-2 ref)"
        print(f"  F{fold}: n={len(entries):3d} completed | best={best_crps:.4f} (T{entries[0][1]:03d}){delta}")

    # ---- Per-HP analysis across TOP trials ----
    # Use top 5 from EACH fold that has >=3 trials
    print("\n=== PER-HP CONVERGENCE (top 5 per fold, or all if <5) ===")
    top_by_fold = {}
    for fold in sorted(by_fold.keys()):
        entries = sorted(by_fold[fold])
        top_by_fold[fold] = entries[:5]

    for hp in STRUCTURAL_HPS:
        print(f"\n  {hp}:")
        all_top_vals = []
        for fold in sorted(top_by_fold.keys()):
            vals = [e[4].get(hp) for e in top_by_fold[fold] if e[4].get(hp) is not None]
            all_top_vals.extend(vals)
            if vals:
                mn, mx, avg = min(vals), max(vals), sum(vals)/len(vals)
                med = statistics.median(vals)
                print(f"    F{fold} (n={len(vals)}): min={mn:.4f}  max={mx:.4f}  mean={avg:.4f}  median={med:.4f}")
        if all_top_vals:
            mn, mx, avg = min(all_top_vals), max(all_top_vals), sum(all_top_vals)/len(all_top_vals)
            med = statistics.median(all_top_vals)
            stdev = statistics.stdev(all_top_vals) if len(all_top_vals) > 1 else 0
            print(f"    CROSS-FOLD (n={len(all_top_vals)}): min={mn:.4f}  max={mx:.4f}  mean={avg:.4f}  median={med:.4f}  std={stdev:.4f}")

    # ---- Correlation: rank each HP by CRPS rank ----
    print("\n=== HP-CRPS CORRELATION (Spearman rank, per fold) ===")
    for fold in sorted(by_fold.keys()):
        entries = sorted(by_fold[fold])
        if len(entries) < 5:
            print(f"  F{fold}: too few trials ({len(entries)}) for correlation")
            continue
        n = len(entries)
        crps_ranks = list(range(n))  # already sorted by crps
        print(f"  F{fold} (n={n}):")
        for hp in STRUCTURAL_HPS:
            vals = [e[4].get(hp, 0) for e in entries]
            # Rank the HP values
            sorted_idx = sorted(range(n), key=lambda i: vals[i])
            hp_ranks = [0] * n
            for rank, idx in enumerate(sorted_idx):
                hp_ranks[idx] = rank
            # Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
            d_sq = sum((hp_ranks[i] - crps_ranks[i])**2 for i in range(n))
            rho = 1 - 6*d_sq / (n*(n**2 - 1))
            direction = "↑ higher=worse" if rho > 0.2 else ("↓ lower=worse" if rho < -0.2 else "— weak")
            print(f"    {hp:<22s} ρ={rho:+.3f}  {direction}")

    # ---- Global top 10 with full HP table ----
    print("\n=== GLOBAL TOP 10 (by CRPS, regardless of fold) ===")
    all_rows = sorted(rows, key=lambda r: r[2])
    print(f"{'Rank':<6}{'Trial':<8}{'Fold':<6}{'CRPS':<10}{'R²':<8}{'Ep':<5}", end="")
    for hp in STRUCTURAL_HPS:
        print(f"{hp[:14]:<16}", end="")
    print()
    print("-" * 150)
    for rank, (trial_idx, fold, crps, r2, n_ep, hp) in enumerate(all_rows[:10], 1):
        r2s = f"{r2:.4f}" if r2 else "N/A"
        print(f"{rank:<6}{trial_idx:<8}{fold:<6}{crps:<10.4f}{r2s:<8}{n_ep:<5}", end="")
        for h in STRUCTURAL_HPS:
            v = hp.get(h)
            print(f"{v:<16.4f}" if v is not None else f"{'N/A':<16}", end="")
        print()

    # ---- Epoch analysis: where do best models peak? ----
    print("\n=== PEAK EPOCH ANALYSIS (all completed trials) ===")
    for fold in sorted(by_fold.keys()):
        entries = sorted(by_fold[fold])
        best_epochs = [e[3] for e in entries]  # n_ep is total, we need best_epoch
        total_epochs = [e[3] for e in entries]
        print(f"  F{fold}: n={len(entries)}")
    # Reload with best_epoch info
    best_ep_data = []
    for td in sorted(RESULTS_DIR.iterdir()):
        if not td.name.startswith("trial_"):
            continue
        for child in td.iterdir():
            if child.is_dir() and child.name.startswith("CV-fold_"):
                fold = int(child.name.split("_")[1])
                hist_p = child / "training_history.json"
                if not hist_p.exists():
                    continue
                hist = json.loads(hist_p.read_text())
                crps_list = hist.get("val_tumor_gvw_crps", [])
                if not crps_list:
                    continue
                best_ep = crps_list.index(min(crps_list)) + 1
                best_ep_data.append((fold, min(crps_list), best_ep, len(crps_list)))

    by_fold_ep = defaultdict(list)
    for fold, crps, best_ep, total_ep in best_ep_data:
        by_fold_ep[fold].append((crps, best_ep, total_ep))
    
    for fold in sorted(by_fold_ep.keys()):
        entries = sorted(by_fold_ep[fold])
        best_eps = [e[1] for e in entries]
        total_eps = [e[2] for e in entries]
        print(f"  F{fold}: best_epoch: min={min(best_eps)} max={max(best_eps)} mean={sum(best_eps)/len(best_eps):.1f} | total_epoch: min={min(total_eps)} max={max(total_eps)}")

if __name__ == "__main__":
    main()
