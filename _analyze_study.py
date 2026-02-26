#!/usr/bin/env python3
"""One-shot analysis of biodvn_tune_phase1-6 Optuna study."""
import sqlite3
import json
import math
from collections import defaultdict

DB = "results_training/optuna.db"
STUDY = "biodvn_tune_phase1-6"

db = sqlite3.connect(DB)
c = db.cursor()

# --- Study ID ---
c.execute("SELECT study_id FROM studies WHERE study_name=?", (STUDY,))
row = c.fetchone()
if not row:
    print(f"Study '{STUDY}' not found"); exit(1)
study_id = row[0]

# --- All trials with state ---
c.execute("""
    SELECT t.trial_id, t.number, t.state, tv.value
    FROM trials t
    LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
    WHERE t.study_id = ?
    ORDER BY t.number
""", (study_id,))
all_trials = c.fetchall()

states = {0: "RUNNING", 1: "COMPLETE", 2: "PRUNED", 3: "FAIL", 4: "WAITING"}
state_counts = defaultdict(int)
for _, _, st, _ in all_trials:
    state_counts[states.get(st, str(st))] += 1
print(f"=== Study: {STUDY} ===")
print(f"Total trials: {len(all_trials)}")
for s, cnt in sorted(state_counts.items()):
    print(f"  {s}: {cnt}")

# --- Completed trials ---
completed = [(tid, num, val) for tid, num, st, val in all_trials if st == 1]
finite = [(tid, num, val) for tid, num, val in completed if val is not None and math.isfinite(val)]
inf_trials = [(tid, num, val) for tid, num, val in completed if val is None or not math.isfinite(val)]
print(f"\nCompleted: {len(completed)} (finite: {len(finite)}, inf: {len(inf_trials)})")

# --- Get params + attrs for all finite trials ---
def get_trial_data(trial_id):
    c.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id=?", (trial_id,))
    params = {r[0]: r[1] for r in c.fetchall()}
    c.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?", (trial_id,))
    attrs = {r[0]: json.loads(r[1]) if r[1] else None for r in c.fetchall()}
    return params, attrs

# Sort by value ascending
finite.sort(key=lambda x: x[2])

print(f"\n{'='*80}")
print(f"TOP 30 TRIALS (by tumor_gvw_crps_mean, lower is better)")
print(f"{'='*80}")
for i, (tid, num, val) in enumerate(finite[:30]):
    params, attrs = get_trial_data(tid)
    fold = attrs.get("fold_num", "?")
    tumor_mae = attrs.get("tumor_mae_mean", "?")
    tumor_r2 = attrs.get("tumor_r2_mean", "?")
    print(f"\n#{i+1} Trial {num} | fold={fold} | obj={val:.6f} | mae={tumor_mae} | r2={tumor_r2}")
    # Print searched params
    searched = [
        "structural.beta_tcvae",
        "scale.signal_ksize",
        "scale.interaction_ksize",
        "scale.n_reductions",
        "scale.reduction_dilation_small",
        "scale.reduction_dilation_large",
        "scale.encoder_start",
        "scale.latent_size",
        "scale.decoder_start",
        "scale.mdn_hidden",
        "scale.mdn_final",
        "scale.signal_filters",
        "scale.initial_reduction_channels",
        "scale.dropout_rate",
    ]
    vals = []
    for p in searched:
        v = params.get(p, "?")
        short = p.split(".")[-1]
        vals.append(f"{short}={v}")
    print(f"  {', '.join(vals)}")

# --- Per-fold analysis ---
print(f"\n{'='*80}")
print(f"PER-FOLD ANALYSIS")
print(f"{'='*80}")
fold_data = defaultdict(list)
for tid, num, val in finite:
    _, attrs = get_trial_data(tid)
    fold = attrs.get("fold_num", None)
    if fold is not None:
        fold_data[fold].append((num, val, attrs))

for fold in sorted(fold_data.keys()):
    trials_f = fold_data[fold]
    vals = [v for _, v, _ in trials_f]
    vals.sort()
    n = len(vals)
    mean_v = sum(vals) / n if n else float("nan")
    median_v = vals[n // 2] if n else float("nan")
    best_v = vals[0] if n else float("nan")
    print(f"\nFold {fold}: n={n}, best={best_v:.6f}, median={median_v:.6f}, mean={mean_v:.6f}")
    # Top 3
    trials_f.sort(key=lambda x: x[1])
    for num, val, attrs in trials_f[:3]:
        print(f"  Trial {num}: {val:.6f} (mae={attrs.get('tumor_mae_mean','?')}, r2={attrs.get('tumor_r2_mean','?')})")

# --- HP distributions in top N ---
print(f"\n{'='*80}")
print(f"HP VALUE DISTRIBUTIONS IN TOP-25 vs ALL FINITE TRIALS")
print(f"{'='*80}")

hp_names = [
    "structural.beta_tcvae",
    "scale.signal_ksize",
    "scale.interaction_ksize",
    "scale.n_reductions",
    "scale.reduction_dilation_small",
    "scale.reduction_dilation_large",
    "scale.encoder_start",
    "scale.latent_size",
    "scale.decoder_start",
    "scale.mdn_hidden",
    "scale.mdn_final",
    "scale.signal_filters",
    "scale.initial_reduction_channels",
    "scale.dropout_rate",
]

top_n = 25
top_trials = finite[:top_n]
all_data = {}
top_data = {}

for hp in hp_names:
    all_vals = []
    for tid, num, val in finite:
        params, _ = get_trial_data(tid)
        v = params.get(hp)
        if v is not None:
            all_vals.append(v)
    
    top_vals = []
    for tid, num, val in top_trials:
        params, _ = get_trial_data(tid)
        v = params.get(hp)
        if v is not None:
            top_vals.append(v)
    
    all_data[hp] = all_vals
    top_data[hp] = top_vals

for hp in hp_names:
    short = hp.split(".")[-1]
    all_v = all_data[hp]
    top_v = top_data[hp]
    
    # Check if categorical or numeric
    try:
        all_num = [float(x) for x in all_v]
        top_num = [float(x) for x in top_v]
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False
    
    print(f"\n  {short}:")
    if is_numeric:
        if all_num:
            all_mean = sum(all_num)/len(all_num)
            top_mean = sum(top_num)/len(top_num) if top_num else float("nan")
            all_sorted = sorted(all_num)
            top_sorted = sorted(top_num)
            print(f"    ALL (n={len(all_num)}): mean={all_mean:.4f}, min={min(all_num):.4f}, max={max(all_num):.4f}, p25={all_sorted[len(all_sorted)//4]:.4f}, p75={all_sorted[3*len(all_sorted)//4]:.4f}")
            if top_num:
                print(f"    TOP-{top_n} (n={len(top_num)}): mean={top_mean:.4f}, min={min(top_num):.4f}, max={max(top_num):.4f}, p25={top_sorted[len(top_sorted)//4]:.4f}, p75={top_sorted[3*len(top_sorted)//4]:.4f}")
    else:
        # Categorical: count distribution
        from collections import Counter
        all_counts = Counter(all_v)
        top_counts = Counter(top_v)
        print(f"    ALL (n={len(all_v)}): {dict(all_counts)}")
        print(f"    TOP-{top_n} (n={len(top_v)}): {dict(top_counts)}")

# --- FAILED trials info ---
print(f"\n{'='*80}")
print(f"FAILED TRIALS (state=3)")
print(f"{'='*80}")
failed = [(tid, num, val) for tid, num, st, val in all_trials if st == 3]
for tid, num, _ in failed[:15]:
    _, attrs = get_trial_data(tid)
    fold = attrs.get("fold_num", "?")
    err = attrs.get("error", "no error attr")
    if isinstance(err, str) and len(err) > 120:
        err = err[:120] + "..."
    print(f"  Trial {num} (fold={fold}): {err}")
if len(failed) > 15:
    print(f"  ... and {len(failed)-15} more")

# --- INF trials info ---
print(f"\n{'='*80}")
print(f"COMPLETED WITH INF VALUE ({len(inf_trials)} trials)")
print(f"{'='*80}")
for tid, num, val in inf_trials[:10]:
    _, attrs = get_trial_data(tid)
    fold = attrs.get("fold_num", "?")
    err = attrs.get("error", "no error attr")
    if isinstance(err, str) and len(err) > 150:
        err = err[:150] + "..."
    print(f"  Trial {num} (fold={fold}): {err}")
if len(inf_trials) > 10:
    print(f"  ... and {len(inf_trials)-10} more")

# --- PRUNED trials reasons ---
print(f"\n{'='*80}")
print(f"PRUNED TRIALS SUMMARY ({state_counts.get('PRUNED',0)})")
print(f"{'='*80}")
pruned = [(tid, num) for tid, num, st, _ in all_trials if st == 2]
prune_reasons = defaultdict(int)
for tid, num in pruned:
    _, attrs = get_trial_data(tid)
    # Check system attrs for pruning info
    c.execute("SELECT key, value_json FROM trial_system_attributes WHERE trial_id=?", (tid,))
    sys_attrs = {r[0]: r[1] for r in c.fetchall()}
    fail_reason = sys_attrs.get("fail_reason", "unknown")
    if "non-positive seq" in str(fail_reason):
        prune_reasons["geometry (non-positive seq length)"] += 1
    elif "beta_tcvae" in str(fail_reason):
        prune_reasons["beta_tcvae < 1.0"] += 1
    elif "signal_filters" in str(fail_reason):
        prune_reasons["signal_filters not div by 4"] += 1
    elif "Pruned at epoch" in str(fail_reason):
        prune_reasons["Hyperband epoch pruning"] += 1
    else:
        prune_reasons[str(fail_reason)[:80]] += 1

for reason, cnt in sorted(prune_reasons.items(), key=lambda x: -x[1]):
    print(f"  {cnt:>4}x  {reason}")

db.close()
print("\nDone.")
