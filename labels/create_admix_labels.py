import pandas as pd

# Read the admix_samples.txt file and extract the PDX names
with open('admix_samples.txt', 'r') as f:
    samples = f.read().splitlines()
pdx_names = [sample.rsplit('_', 1)[0] for sample in samples]

# Read the file1.tsv and file2.tsv files
df1 = pd.read_csv('LuCaP_Log2CPMplus1.tsv', sep='\t', index_col=0)
df2 = pd.read_csv('SCLC_Log2CPMplus1.tsv', sep='\t', index_col=0)

# Initialize a list to store the columns
columns = []

# For each PDX name, extract the corresponding column and add it to the list
for pdx, sample in zip(pdx_names, samples):
    if pdx in df1.columns:
        columns.append(df1[pdx].rename(sample))
    if pdx in df2.columns:
        columns.append(df2[pdx].rename(sample))

# Concatenate all columns at once
output_df = pd.concat(columns, axis=1)

# Write the output DataFrame to a new TSV file
output_df.to_csv('admix_Log2CPMplus1.tsv', sep='\t')