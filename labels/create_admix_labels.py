import pandas as pd
import numpy as np

# Load the TSV files
mix_ids = pd.read_csv('admix_identities.tsv', sep='\t', header=None, names=['pdx', 'healthy', 'name'])
lucap_exp = pd.read_csv('PC-PDX_Log2CPMplus1.tsv', sep='\t', index_col=0)
sclc_exp = pd.read_csv('SCLC-PDX_Log2CPMplus1.tsv', sep='\t', index_col=0)
hd_exp = pd.read_csv('HD-L_Log2CPMplus1.tsv', sep='\t', index_col=0)

# Ensure all gene names match lucap_exp
all_genes = lucap_exp.index

# Create an empty DataFrame for each output
phh_tumor = pd.DataFrame(index=all_genes)
phs_tumor = pd.DataFrame(index=all_genes)
pht_tumor = pd.DataFrame(index=all_genes)
pht_background = pd.DataFrame(index=all_genes)

# Process PHH samples
phh_samples = mix_ids[mix_ids['name'].str.startswith('PHH')]
for _, row in phh_samples.iterrows():
    pdx = row['pdx']
    name = row['name']
    if pdx in lucap_exp.columns:
        phh_tumor[name] = lucap_exp[pdx]
    elif pdx in sclc_exp.columns:
        phh_tumor[name] = sclc_exp[pdx]
    else:
        raise ValueError(f"PDX sample {pdx} not found in LuCaP or SCLC expression files.")
    
# Process PHS samples
phs_samples = mix_ids[~mix_ids['name'].str.startswith('PHH') & ~mix_ids['name'].str.startswith('PHT')]
for _, row in phs_samples.iterrows():
    pdx = row['pdx']
    name = row['name'] + '_merge'
    if pdx in lucap_exp.columns:
        phs_tumor[name] = lucap_exp[pdx]
    elif pdx in sclc_exp.columns:
        phs_tumor[name] = sclc_exp[pdx]
    else:
        print(f"PDX sample {pdx} not found in LuCaP or SCLC expression files.")

# Process PHT-tumor samples
pht_samples = mix_ids[mix_ids['name'].str.startswith('PHT')]
for _, row in pht_samples.iterrows():
    pdx = row['pdx']
    name = row['name']
    if pdx in lucap_exp.columns:
        pht_tumor[name] = lucap_exp[pdx]
    elif pdx in sclc_exp.columns:
        pht_tumor[name] = sclc_exp[pdx]
    else:
        raise ValueError(f"PDX sample {pdx} not found in LuCaP or SCLC expression files.")

# Process PHT-background samples
for _, row in pht_samples.iterrows():
    healthy = row['healthy']
    name = row['name']
    if healthy in hd_exp.columns:
        pht_background[name] = hd_exp[healthy]
    else:
        raise ValueError(f"Healthy sample {healthy} not found in HD expression file.")

# Fill missing rows for PHT-tumor and PHT-background with NaN if they originate from sclc_exp
for column in pht_tumor.columns:
    if column in sclc_exp.columns:
        missing_genes = all_genes.difference(sclc_exp.index)
        pht_tumor.loc[missing_genes, column] = np.nan

for column in pht_background.columns:
    if column in sclc_exp.columns:
        missing_genes = all_genes.difference(sclc_exp.index)
        pht_background.loc[missing_genes, column] = np.nan

# Save the output files
phh_tumor.to_csv('PHH-tumor_Log2CPMplus1.tsv', sep='\t')
phs_tumor.to_csv('PHS-tumor_Log2CPMplus1.tsv', sep='\t')
pht_tumor.to_csv('PHT-tumor_Log2CPMplus1.tsv', sep='\t')
pht_background.to_csv('PHT-background_Log2CPMplus1.tsv', sep='\t')

print("Files created successfully.")
