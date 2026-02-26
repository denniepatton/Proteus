import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use("Agg")  # For non-interactive backends (if desired, e.g. in scripts)

# ------------------ HARD-CODED PATHS -------------------
A_TSS_PATH = "../../../../LuCaP_data/Triton_TSS/results/TritonCompositeFM.tsv"
B_TSS_PATH = "../../../../SCLC_data/Triton_TSS/results/TritonCompositeFM.tsv"
A_GB_PATH  = "../../../../LuCaP_data/Triton_GB/results/TritonCompositeFM.tsv"
B_GB_PATH  = "../../../../SCLC_data/Triton_GB/results/TritonCompositeFM.tsv"

A_EXPR_PATH = "../../labels/LuCaP-PDX_Log2CPMplus1.tsv"
B_EXPR_PATH = "../../labels/SCLC-PDX_Log2CPMplus1.tsv"

# ------------------ CONSTANTS & FEATURE LISTS -------------------
FEATURES = [
    "fragment-ratio", "fragment-mean", "fragment-stdev", "fragment-entropy",
    "fragment-median", "fragment-diversity", "fragment-mad", "np-amplitude",
    "np-score", "np-period", "mean-depth", "var-ratio"
]
# Update the feature order to account for the mean-depth rename
RAW_FEATURE_ORDER = FEATURES.copy()
FEATURE_ORDER = [f if f != "mean-depth" else "log-mean-depth" for f in RAW_FEATURE_ORDER]

# ========== STEP 1 & 1.5: LOAD & COMBINE FEATURE DATA (TSS, GB) ==========
# A helper function to read a "tidy" TSV and restrict + transform features.

def load_and_process_feature_data(tss_path, gb_path):
    """
    Read two tidy TSVs (from sets A and B for either TSS or GB), 
    combine them, restrict to FEATURES, then standard-scale per sample+feature,
    except 'mean-depth' => log( mean-depth ) => renamed to 'log-mean-depth'.
    
    Returns a DataFrame with columns:
       ['sample', 'site', 'feature', 'scaled_value']
    """
    df_A = pd.read_csv(tss_path, sep="\t")
    df_B = pd.read_csv(gb_path, sep="\t")
    df   = pd.concat([df_A, df_B], ignore_index=True)
    
    # Restrict to only the desired features
    df = df[df["feature"].isin(FEATURES)].copy()
    
    # For convenience, let's keep everything as needed
    # We'll transform 'value' -> 'scaled_value'
    df["scaled_value"] = np.nan  # placeholder
    
    # Split out mean-depth vs. other features
    md_mask = (df["feature"] == "mean-depth")
    other_mask = ~md_mask
    
    # 1) For mean-depth => log transform => rename feature to "log-mean-depth"
    df.loc[md_mask, "scaled_value"] = np.log1p(df.loc[md_mask, "value"])
    df.loc[md_mask, "feature"] = "log-mean-depth"
    
    # 2) For other features => standard scale per (sample, feature)
    #    That is, for each sample+feature pair, we standardize across all sites.
    def standard_scale(x):
        return (x - x.mean()) / x.std(ddof=0)
    
    # We will group by sample + feature (the new feature which might still be "mean-depth"? 
    # Actually we changed the name for mean-depth, so it's excluded from this group.)
    grouping_cols = ["sample", "feature"]
    # standard scale only the "other" original features:
    # But note we have changed "mean-depth" to "log-mean-depth" above.
    # We only want to apply standard scaling to the original 11 that are not 'mean-depth'.
    # After that rename, they remain the same feature name. So let's do a sub-DataFrame.
    # Another approach is to do transform but skip rows that are "log-mean-depth".
    
    # Mark the subset that needs standard scaling: everything that is not "log-mean-depth"
    idx_other = df["feature"] != "log-mean-depth"
    df.loc[idx_other, "scaled_value"] = (
        df.loc[idx_other]
        .groupby(grouping_cols)["value"]
        .transform(standard_scale)
    )
    
    # We drop the original 'value' column so we do not confuse them later
    df.drop(columns=["value"], inplace=True)
    
    # Return the processed DataFrame
    # columns: sample, site, feature, scaled_value
    return df

# Load TSS data (combine A_TSS + B_TSS)
df_tss = load_and_process_feature_data(A_TSS_PATH, B_TSS_PATH)

# Load GB data (combine A_GB + B_GB)
df_gb = load_and_process_feature_data(A_GB_PATH, B_GB_PATH)


# ========== STEP 2: LOAD EXPRESSION MATRICES (A_expr, B_expr) ==========
def load_and_combine_expression(a_expr_path, b_expr_path):
    """
    Read the wide expression data (genes as rows, samples as columns),
    combine them, and pivot => tidy form with columns ['site', 'sample', 'expr_value'].
    
    We assume the expression data is already log-scaled (Log2(CPM+1) or similar).
    """
    expr_A = pd.read_csv(a_expr_path, sep="\t", index_col=0)  # index => gene
    expr_B = pd.read_csv(b_expr_path, sep="\t", index_col=0)
    
    # Combine (union of genes, union of samples). 
    # "Most logical" might be outer join. 
    # But typically, if the same genes exist, they share row indices, etc.
    # We'll do a simple outer join in columns. 
    # If some samples are identical, that's okay, we can keep them distinct or unify them if they are the same name.
    expr_comb = pd.concat([expr_A, expr_B], axis=1)
    
    # Now we go from wide => tidy
    # Genes are row index => "site"
    # Column names => "sample"
    expr_comb.index.name = "site"
    expr_tidy = expr_comb.stack().reset_index()
    expr_tidy.columns = ["site", "sample", "expr_value"]
    
    # Filter out expression values of exactly 0
    expr_tidy = expr_tidy[expr_tidy["expr_value"] > 0].copy()
    
    return expr_tidy

df_expr = load_and_combine_expression(A_EXPR_PATH, B_EXPR_PATH)

# ========== STEP 3: ENSURE UNION/INTERSECTION OF SAMPLES ==========
# We'll interpret "Ensure the union of samples between sets A and B is identical
# (inner join) between feature data and expression data" as:
# "We only keep samples that appear in both (the combined feature sets) and in expression."

# For TSS
samples_tss = set(df_tss["sample"].unique())
# For GB
samples_gb = set(df_gb["sample"].unique())
# For expression
samples_expr = set(df_expr["sample"].unique())

# We'll keep the intersection for TSS or GB separately
samples_keep_tss = samples_tss.intersection(samples_expr)
samples_keep_gb  = samples_gb.intersection(samples_expr)

df_tss = df_tss[df_tss["sample"].isin(samples_keep_tss)].copy()
df_gb  = df_gb[df_gb["sample"].isin(samples_keep_gb)].copy()
df_expr_tss = df_expr[df_expr["sample"].isin(samples_keep_tss)].copy()
df_expr_gb  = df_expr[df_expr["sample"].isin(samples_keep_gb)].copy()

# ========== STEP 4: DROP NAs ==========
df_tss.dropna(subset=["scaled_value"], inplace=True)
df_gb.dropna(subset=["scaled_value"], inplace=True)
df_expr_tss.dropna(subset=["expr_value"], inplace=True)
df_expr_gb.dropna(subset=["expr_value"], inplace=True)

# ========== STEP 5 & 6: SCATTER (DENSITY) PLOTS + CORRELATIONS ==========
# We will create TSS and GB plots for each feature, matching data on (sample, site).

def plot_feature_vs_expression(df_feature, df_expr, feature_name, region_label):
    """
    Given a feature DataFrame [sample, site, feature, scaled_value]
    and an expression DF [site, sample, expr_value],
    filter to feature_name, merge on (site, sample), plot density scatter,
    compute Pearson R^2 and Spearman rho (with p-value), and label the plot.
    
    region_label is either "TSS" or "GB", used in the axis labels and file naming.
    """
    # Filter to only the desired feature
    sub_feat = df_feature[df_feature["feature"] == feature_name].copy()
    # Merge with expression on (sample, site)
    merged = pd.merge(
        sub_feat,
        df_expr, 
        on=["sample","site"], 
        how="inner",
        validate="one_to_one"
    )
    if merged.empty:
        print(f"WARNING: No data to plot for {region_label} {feature_name}. Skipping.")
        return None, None, None, None

    x = merged["expr_value"].values
    y = merged["scaled_value"].values
    
    # Compute correlations
    pear_r, pear_p = pearsonr(x, y)
    r2 = pear_r**2
    spear_r, spear_p = spearmanr(x, y)
    
    # Make a density scatter plot with square bins
    plt.figure(figsize=(7, 7))  # Square figure for square bins
    
    # Get data ranges to set appropriate aspect ratio
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    
    # Calculate appropriate aspect ratio for square bins
    aspect_ratio = x_range / y_range if y_range != 0 else 1
    
    # Use hexbin with square bins by specifying extent and aspect
    extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
    hb = plt.hexbin(x, y, gridsize=50, cmap="viridis", mincnt=1, bins="log", 
                   extent=extent)
    plt.colorbar(hb, label="log10(Count)")
    
    plt.title(f"{region_label} {feature_name} vs. Expression")
    plt.xlabel("Log2(CPM+1)")  # expression is already log scale
    plt.ylabel(f"{region_label} {feature_name}")
    
    # Set y-axis limits based on feature name
    if feature_name != "log-mean-depth":
        plt.ylim([-7, 7])  # ~ square with expression
    
    # Ensure aspect ratio makes hexbins appear square
    plt.gca().set_aspect(aspect_ratio)
    
    # Annotate R^2 and Spearman
    text_str = f"R^2 = {r2:.3f}\nSpearman = {spear_r:.3f}\np(Spearman) = {spear_p:.1e}"
    plt.text(0.05, 0.95, text_str,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round", fc="w", alpha=0.7, ec="k"))
    
    # Save the plot
    outname = f"{region_label}_{feature_name.replace('/', '-')}_vs_expression.png"
    plt.tight_layout()
    plt.savefig(outname, dpi=150)
    plt.close()
    
    return r2, spear_r, spear_p, merged.shape[0]

# We'll store the Spearman correlations in a dictionary for building the heatmap
spearman_dict_tss = {}
spearman_dict_gb  = {}

for feat in FEATURE_ORDER:
    # For TSS
    r2_tss, spear_tss, p_tss, n_tss = plot_feature_vs_expression(df_tss, df_expr_tss, feat, "TSS")
    spearman_dict_tss[feat] = spear_tss if spear_tss is not None else 0.0
    
    # For GB
    r2_gb, spear_gb, p_gb, n_gb = plot_feature_vs_expression(df_gb, df_expr_gb, feat, "GB")
    spearman_dict_gb[feat] = spear_gb if spear_gb is not None else 0.0


# ========== STEP 7: HEATMAP OF SPEARMAN RHO (TSS vs. GB) FOR EACH FEATURE ==========
# Create custom colormap: blue (-1) to black (0) to red (1) with narrower black region
# This makes the gradient more abrupt around 0, highlighting positive and negative correlations
colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # Blue, Dark Blue, Black, Dark Red, Red
positions = [0, 0.5, 1]  # Narrow black region in the middle
cmap = LinearSegmentedColormap.from_list("blue_black_red", list(zip(positions, colors)))

heat_data = []
for feat in FEATURE_ORDER:
    val_tss = spearman_dict_tss.get(feat, 0.0)
    val_gb  = spearman_dict_gb.get(feat, 0.0)
    heat_data.append([val_tss, val_gb])

heat_array = np.array(heat_data)  # shape: (12, 2)

plt.figure(figsize=(3, 6))  # two columns, 12 rows => tall
# Show the heatmap with imshow using the custom colormap
im = plt.imshow(heat_array, vmin=-0.5, vmax=0.5, aspect="auto", origin="upper", cmap=cmap)

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label("Spearman Correlation")

# Ticks for rows (features)
plt.yticks(ticks=range(len(FEATURE_ORDER)), labels=FEATURE_ORDER)

# Ticks for columns
plt.xticks(ticks=[0, 1], labels=["TSS", "GB"])

plt.title("Spearman rho vs. Expression")

plt.tight_layout()
plt.savefig("spearman_rho_heatmap.png", dpi=150)
plt.close()

print("Done! Generated density scatter plots for TSS/GB features vs. expression and the spearman correlation heatmap.")