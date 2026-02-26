import pandas as pd


df = pd.read_csv("IHC.tsv", sep="\t", dtype=str).fillna("")

# 2. Clean up and convert columns to numeric, ignoring invalid strings
score_cols = ["NECTIN4", "Ki67", "ERBB2"]

for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Group by Patient ID and compute mean + median
grouped = df.groupby("Patient_ID").agg({
    "NECTIN4": ["mean", "median"],
    "Ki67": ["mean", "median"],
    "ERBB2": ["mean", "median"]
})

# Flatten multi-level column names
grouped.columns = [
    "NECTIN4_mean", "NECTIN4_median",
    "Ki67_mean", "Ki67_median",
    "ERBB2_mean", "ERBB2_median"
]

# 4. Write out the summarized TSV
out_df = grouped.reset_index().rename(columns={"Patient_ID": "sample"})
out_df.to_csv("BC-TAN_IHC-Scores.tsv", sep="\t", index=False)
