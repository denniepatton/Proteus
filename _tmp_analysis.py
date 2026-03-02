import pandas as pd
import numpy as np

base = '/Volumes/fh/fast/ha_g/user/rpatton/TritonCNN/Proteus/results_training/biodvn_phase2_2_training'

def get_cohort(name):
    if name.startswith('LuCaP'):
        return 'PC-PDX'
    elif name.startswith('PHT'):
        return 'PHT'
    elif name in ['DIVE_CDX29','FHSC36','FHSC47','MSK_LX640','FHSC04','FHSC14','LX33B','LU86','SCLC150','SCLC152','DIVE_CDX30P','H211']:
        return 'SCLC-PDX'
    elif name.startswith('L-'):
        return 'SCLC'
    elif name.startswith('H'):
        return 'SCLC-PDX'
    elif name.startswith('BP'):
        return 'HD-L'
    elif name.startswith('AE-'):
        return 'CRPC-WCDT'
    elif name.startswith('04-') or name.startswith('CRPC'):
        return 'CRPC-RA'
    return 'unknown'

def per_sample_stats(path, label):
    df = pd.read_csv(path, sep='\t', engine='python')
    dfv = df.dropna(subset=['truth', 'prediction_mean'])
    rows = []
    for sample in sorted(dfv['sample'].unique()):
        sub = dfv[dfv['sample'] == sample]
        t = sub['truth'].values
        p = sub['prediction_mean'].values
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        mae = np.mean(np.abs(t - p))
        ci_width = (sub['ci_q95'] - sub['ci_q5']).mean()
        cohort = get_cohort(sample)
        rows.append({'sample': sample, 'cohort': cohort, 'n': len(sub), 'r2': r2, 'mae': mae, 'ci_width': ci_width, 'truth_std': t.std()})
    return pd.DataFrame(rows)

# Analyze all 3 completed folds
for fold, trial, crps_label in [(3, 44, '1.21'), (1, 38, '0.78'), (5, 50, '0.84')]:
    path = f'{base}/trial_{trial:05d}/CV-fold_{fold}/validation-fold_{fold}_predictions_tumor.tsv'
    df = per_sample_stats(path, f'F{fold}')
    print(f'\n=== FOLD {fold} (trial {trial}, CRPS~{crps_label}) by cohort ===')
    for c in ['PC-PDX', 'SCLC-PDX', 'PHT']:
        sub = df[df['cohort'] == c]
        if len(sub) > 0:
            print(f'  {c:10s}: n_samples={len(sub):2d}  R2={sub["r2"].mean():.4f}  MAE={sub["mae"].mean():.4f}  CI_w={sub["ci_width"].mean():.3f}  truth_std={sub["truth_std"].mean():.3f}')
    print(f'  {"ALL":10s}: n_samples={len(df):2d}  R2={df["r2"].mean():.4f}  MAE={df["mae"].mean():.4f}  CI_w={df["ci_width"].mean():.3f}')

# Also print individual samples for F3 sorted by CI width
print('\n=== F3 samples sorted by CI width (descending) ===')
df3 = per_sample_stats(f'{base}/trial_00044/CV-fold_3/validation-fold_3_predictions_tumor.tsv', 'F3')
df3_sorted = df3.sort_values('ci_width', ascending=False)
for _, row in df3_sorted.iterrows():
    print(f'  {row["sample"]:55s}  {row["cohort"]:10s}  R2={row["r2"]:.4f}  MAE={row["mae"]:.4f}  CI_w={row["ci_width"]:.3f}')

