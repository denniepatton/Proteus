"""
analysis_utils.py
v1.1.0, 2/22/2024
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script contains helper functions for the analysis of the data.
"""

# Standard library imports
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr, mannwhitneyu

# Third party imports
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score


########################## HELPER FUNCTIONS ##########################################################################################

def load_file_as_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]
    
def rename_tan(df):
    df.columns = df.columns.str.replace(r'(\d{2}-\d{3})([A-Z]\d)', r'\1_\2-', regex=True)
    return df

def create_palette(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    subtype_to_color = dict(zip(df[0], df[1]))
    return subtype_to_color
    
def merge_validation(fold_dfs):
    # Reset the index of each DataFrame and concatenate them
    df_concat = pd.concat(df.reset_index() for df in fold_dfs)
    # Pivot the DataFrame so that genes are the index, samples are the columns, and predictions are the values
    df_merged = df_concat.pivot(index='index', columns='sample', values='prediction')
    return df_merged

def get_average_tpm(df):
    # Convert Log2(TPM+1) to TPM [also works for CPM]
    df_tpm = 2**df - 1

    # Extract the sample names from the column names
    sample_names = df_tpm.columns.str.split('_').str[0]

    # Average together columns which have the same sample name
    # This creates a DataFrame with one column per sample, where each value is the average TPM for that sample
    df_avg = df_tpm.groupby(sample_names, axis=1).mean()

    # Rename the columns by adding '_avg' to the sample names
    df_avg.columns = df_avg.columns + '_avg'

    # Convert back to Log2(TPM+1)
    # This transforms the TPM values back to the original scale
    df_log2tpm_plus_1 = np.log2(df_avg + 1)

    # Return the transformed DataFrame
    return df_log2tpm_plus_1


def get_fit_tpm(df_mets, df_pred):
    # Convert Log2(TPM+1) to TPM [also works for CPM - see block comment below]
    df_mets_tpm = 2**df_mets - 1
    df_pred_tpm = 2**df_pred - 1

    # Find the common genes
    common_genes = df_mets_tpm.index.intersection(df_pred_tpm.index)

    # Filter the dataframes to include only common genes
    df_mets_tpm = df_mets_tpm.loc[common_genes]
    df_pred_tpm = df_pred_tpm.loc[common_genes]

    # Initialize the result DataFrame and dict
    result_df = pd.DataFrame(index=df_mets_tpm.index, columns=df_pred_tpm.columns)
    splits_dict = {}

    # Iterate over each sample in df_pred
    for sample in df_pred_tpm.columns:
        # Get the sample name prefix
        prefix = sample.split('_')[0]

        # Get the columns in df_mets that match the prefix
        matching_cols = [col for col in df_mets_tpm.columns if col.startswith(prefix)]
        if len(matching_cols) < 2:
            continue

        # Define the objective function to minimize
        def objective(x):
            combined = np.dot(df_mets_tpm[matching_cols], x)
            if np.isnan(combined).any():
                return 1e6
            # Create a mask for non-NaN rows in df_pred_tpm[sample]
            mask = ~np.isnan(df_pred_tpm[sample])
            # Apply the mask to both df_pred_tpm[sample] and combined
            return mean_absolute_error(df_pred_tpm[sample][mask], combined[mask])

        # Define the constraints
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        # Define the bounds
        bnds = [(0, 1) for _ in matching_cols]
        # Initialize the weights
        x0 = [1/len(matching_cols)] * len(matching_cols)
        # Perform the optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

        # Store the result in the DataFrame and dict
        result_df[sample] = np.dot(df_mets_tpm[matching_cols], list(res.x))
        modified_keys = ["_".join(col.split("_")[1:]) for col in matching_cols]
        splits_dict[sample] = dict(zip(modified_keys, list(res.x)))

    # Drop any columns in result_df that did not receive values
    result_df = result_df.dropna(axis=1, how='all')
    """ Un-block if using TPM:
    # Re-scale the result_df to sum to 1,000,000
    result_df = result_df.div(result_df.sum()) * 1e6
    """
    # Convert the result back to Log2(TPM+1)
    result_df = result_df.applymap(lambda x: np.log2(x + 1))
    return result_df, splits_dict


def rescale_tmm(df, ref_sample='../../RNA-Seq/LuCaP_173-1_raw_counts.pkl'):
    """
    This function re-scales a set of samples based on TMM (see edgeR) normalization,
    using the same (default) reference sample used to scale model inputs. Input is expcted
    in Log2(CPM +1), which is 

    Parameters:
    df (pandas.DataFrame): DataFrame containing gene expression data.
    ref_sample (pandas.DataFrame): DataFrame containing the raw counts of the reference sample.

    Returns:
    pandas.DataFrame: DataFrame with the gene expression data in re-scaled Log2(CPM + 1) format.
    """
    ref_sample = pd.read_pickle(ref_sample)

    # Transform the data back to "raw counts" by reversing the log2 transformation and subtracting 1
    df.loc[:, df.columns != 'mRNA_size'] = 2**df.loc[:, df.columns != 'mRNA_size']
    df.loc[:, df.columns != 'mRNA_size'] -= 1
    # Align df and ref_sample
    df, ref_sample = df.align(ref_sample, join='inner', axis=0)

    # Drop rows where all values are 0 in both df and ref_sample
    mask = (df != 0).any(axis=1) | (ref_sample != 0).any()
    df_tmm_nz = df[mask]
    ref_sample = ref_sample[mask]

    # Calculate the library sizes
    lib_sizes_tmm = df_tmm_nz.sum()
    lib_sizes_ref = ref_sample.sum()

    # Calculate the M/A/V-values
    with np.errstate(divide='ignore'):
        M = np.log2(df_tmm_nz.div(lib_sizes_tmm, axis='columns').div(ref_sample / lib_sizes_ref, axis='index'))
        A = 0.5 * np.log2(df_tmm_nz.div(lib_sizes_tmm, axis='columns').multiply(ref_sample / lib_sizes_ref, axis='index'))
        V = ((lib_sizes_tmm - df_tmm_nz).div(lib_sizes_tmm, axis='columns') / df_tmm_nz).add((lib_sizes_ref - ref_sample).div(lib_sizes_ref, axis='index') / ref_sample, axis='rows')

    # Remove infinite values
    finite_indices = (np.isfinite(M) & np.isfinite(A) & (np.abs(A) > -1e10))
    M = M[finite_indices]
    A = A[finite_indices]
    V = V[finite_indices]

    def calculate_scaling_factor(M, A, V):
        # drop any NaN values (missing or masked as infinite)
        M = M.dropna()
        A = A.dropna()
        V = V.dropna()
        # Calculate the rank-based trimming thresholds
        n = len(M)
        loL = int(n * 0.3) + 1
        hiL = n + 1 - loL
        loS = int(n * 0.05) + 1
        hiS = n + 1 - loS

        # Trim the M-values and A-values based on their ranks
        keep = ((stats.rankdata(M) >= loL) & (stats.rankdata(M) <= hiL) & 
                (stats.rankdata(np.abs(A)) >= loS) & (stats.rankdata(np.abs(A)) <= hiS))

        # Calculate the scaling factor
        f = np.sum(M[keep] / V[keep]) / np.sum(1 / V[keep])

        return 2 ** f

    scaling_factors = pd.DataFrame({col: calculate_scaling_factor(M[col], A[col], V[col]) for col in M.columns}, index=['scaling_factor'])
    scaling_factors = scaling_factors / np.exp(np.mean(np.log(scaling_factors)))

    # Normalize the counts by the scaling factors
    lib_sizes_scaled = lib_sizes_tmm * scaling_factors.loc['scaling_factor']

    # Combine initial library sizes, scaling factors, and final library sizes into a single DataFrame
    combined_df = pd.concat([lib_sizes_tmm, scaling_factors.T, lib_sizes_scaled.T], axis=1)
    combined_df.columns = ['Initial Library Sizes', 'Scaling Factors', 'Final Library Sizes (TMM-scaled)']
    print(combined_df)

    df_log2cpm = np.log2(df.div(lib_sizes_scaled) * 1e6 + 1)  # return log2(CPM + 1)

    return df_log2cpm


########################## PLOTTING FUNCTIONS ########################################################################################

# Create a dictionary mapping subtypes to colors
subtype_to_color = create_palette('../reference/palette.tsv')


def plot_samples(df, filename):
    """
    Plots kernel density estimates and histograms for each column in the DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing gene expression data.
    filename (str): Name of the file to be used in the x-axis label.

    Returns:
    None
    """
    # Create a figure and axis for the kernel density plot
    fig_kde, ax_kde = plt.subplots()

    # Create a figure and axis for the histogram
    fig_hist, ax_hist = plt.subplots()

    # Iterate over each column in the DataFrame, excluding 'mRNA'
    for column in df.columns:
        if column != 'mRNA_size':
            # Plot the density of the column values
            sns.kdeplot(df[column], ax=ax_kde, label=column)

            # Plot the histogram of the column values
            sns.histplot(df[column], ax=ax_hist, bins=20, label=column, kde=False, element='step', stat='density', fill=False)

    # Shrink the labels
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Add a legend, title, and labels to the plots
    ax_kde.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 6})
    ax_kde.set_title('Kernel Density Plots')
    ax_kde.set_xlabel('Log2(CPM +1)')
    ax_kde.set_ylabel('Density')

    ax_hist.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 6})
    ax_hist.set_title('Histograms')
    ax_hist.set_xlabel('Log2(CPM +1)')
    ax_hist.set_ylabel('Density')

    # Save the plots as PDF files
    plt.figure(fig_kde.number)
    plt.savefig(filename + "_kernels.pdf", bbox_inches='tight')
    plt.close()

    plt.figure(fig_hist.number)
    plt.savefig(filename + "_histograms.pdf", bbox_inches='tight')
    plt.close()


def plot_vs_meta(df_x, df_y, name_x, name_y, name, tfx_dict=None, subtypes=None):
    if tfx_dict is not None:
        tfx_dict = {key[:6]: value for key, value in tfx_dict.items()}

    x_values = df_x.loc[name_x]
    y_values = df_y.loc[name_y]

    data = pd.DataFrame({'x': x_values, 'y': y_values})
    data = data.dropna()
    if len(data['x']) <= 2 or len(data['y']) <= 2:
        return

    def func(x, a, b):
        return a * x + b

    def error(params, x, y, func):
        return np.sum((y - func(x, *params))**2)

    p0 = [1, 1]
    res = minimize(error, p0, args=(data['x'], data['y'], func), method='L-BFGS-B', options={'maxiter': 1000})
    popt = res.x

    r2 = r2_score(data['y'], func(data['x'], *popt))
    spearman_coef, ps_val = spearmanr(data['x'], data['y'])
    pearson_coef, pr_val = pearsonr(data['x'], data['y'])

    plt.figure(figsize=(6, 6))

    if name_x == name_y:
        # plt.xlim([min(data['x'].min() - 0.5, data['y'].min() - 0.5), max(data['x'].max() + 0.5, data['y'].max() + 0.5)])
        # plt.ylim([min(data['x'].min() - 0.5, data['y'].min() - 0.5), max(data['x'].max() + 0.5, data['y'].max() + 0.5)])
        plt.xlim([min(data['x'].min() - 0.1, data['y'].min() - 0.1), max(data['x'].max() + 0.1, data['y'].max() + 0.1)])
        plt.ylim([min(data['x'].min() - 0.1, data['y'].min() - 0.1), max(data['x'].max() + 0.1, data['y'].max() + 0.1)])
    else:
        plt.xlim([data['x'].min(), data['x'].max()])
        plt.ylim([data['y'].min(), data['y'].max()])

    sizes = [tfx_dict.get(sample, 1) * 200 if tfx_dict is not None else 200 for sample in data.index]
    if subtypes is not None:
        colors = [subtype_to_color.get(subtype, 'black') for subtype in subtypes[data.index]]
        plt.scatter(data['x'], data['y'], s=sizes, c=colors, alpha=0.8)
        seen_subtypes = subtypes[data.index].unique()
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=subtype_to_color[subtype], markersize=10) for subtype in seen_subtypes]
        plt.legend(handles, seen_subtypes, title='Subtype', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(data['x'], data['y'], s=sizes, alpha=0.8)

    plt.plot(data['x'], func(data['x'], *popt), '--', color='gray', alpha=0.5)
    if name_x == name_y:
        # plt.xlabel(name_x + ' Log2(CPM+1) [RNA-seq]')
        plt.xlabel(name_x + ' Score [RNA-seq]')
    else:
        plt.xlabel(name_x)
    # plt.ylabel(name_y + ' Log2(CPM+1) [predicted]')
    plt.ylabel(name_y + ' Score [predicted]')
    plt.title(name_x)
    plt.text(0.05, 0.95, f'R^2 = {r2:.3f}\nSpearman = {spearman_coef:.3f}\nS. p-val = {ps_val:.3f}\nPearson = {pearson_coef:.3f}\nP. p-val = {pr_val:.3f}', transform=plt.gca().transAxes, verticalalignment='top')

    plt.savefig(name + '.pdf', bbox_inches='tight')
    plt.close()


def plot_vs_tfx(preds_df, tfx_dict, ref, tissue, name):
    # convert boths dfs to CPM
    # preds_df = 2**preds_df - 1
    ref_cpm = 2**ref - 1

    # Create a mask for each condition
    mask_ref = (ref_cpm['Whole_Blood'] < 0.5) & (ref_cpm[tissue] > 10)
    mask_whole_blood = (ref_cpm['Whole_Blood'] > 10) & (ref_cpm[tissue] < 0.5)

    # Apply the masks to the index of the dataframe to get the gene lists
    genes_ref = ref.index[mask_ref].tolist()
    genes_whole_blood = ref.index[mask_whole_blood].tolist()

    # print the number of genes in each list
    print(f'Number of genes in {tissue} list: {len(genes_ref)}')
    print(f'Number of genes in Whole_Blood list: {len(genes_whole_blood)}')
    
    def func(x, a, b, c, d, e):
        return a + b * x + c * np.power(x, 2) + d * np.power(x, 3) + e * np.power((1 - x), 2)

    def inverted_func(x, a, b, c, d, e):
        return a + b * x + c * np.power(x, 2) + d * np.power(x, 3) + e * np.power((1 - x), 2)

    # Define the error function to minimize
    def error(params, x, y, func):
        return np.sum((y - func(x, *params))**2)

    # Create the plots
    for gene_set, color, name_suffix in [(genes_ref, 'blue', tissue), (genes_whole_blood, 'red', 'Whole_Blood')]:
        
        plt.figure(figsize=(10, 6))
        # Check if each gene in gene_set is in the index of preds_df
        valid_genes = [gene for gene in gene_set if gene in preds_df.index]
        # Restrict to genes in the list also in preds_df
        restricted_df = preds_df.loc[valid_genes].copy()
        # If name_suffix is 'WholeBlood', divide every value by the corresponding value in the "Whole_Blood" column of blood_ref
        restricted_df = restricted_df.div(ref[name_suffix].loc[valid_genes], axis=0)
        # Pivot df into long form
        long_df = restricted_df.reset_index().melt(id_vars='index', var_name='Sample', value_name='GeneValues')
        long_df.columns = ['Gene', 'Sample', 'GeneValues']
        # Remove any rows with NaNs
        long_df = long_df.dropna()
        # Add a TFX column based on the Sample column
        long_df['TFX'] = long_df['Sample'].map(tfx_dict)
        # Proceed with the analysis/fit/plotting
        # plt.scatter(long_df['TFX'], long_df['GeneValues'], alpha=0.2)
        plt.hexbin(long_df['TFX'], long_df['GeneValues'], gridsize=300, cmap='hot_r', bins='log')
        plt.colorbar(label='log10(N)')
        plt.ylim(0.0, 2.0)
        plt.xlim(0.0, 1.0)

        # Choose the function to use based on name_suffix
        if name_suffix == 'WholeBlood':
            fit_func = func
        else:
            fit_func = inverted_func

        # Fit an exponential decay curve to the data and add it to the plot
        # p0 = [0.8, 2]  # Initial guess for the parameters
        # bounds = [(0, None), (0, None)]  # Bounds for the parameters
        # res = minimize(error, p0, args=(long_df['TFX'], long_df['GeneValues'], fit_func), method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})
        x0 = [0.5, 0.5, 0.5, 0.5, 0.5]
        res = minimize(error, x0, args=(long_df['TFX'], long_df['GeneValues'], fit_func), method='L-BFGS-B', options={'maxiter': 10000})
        popt = res.x        
        tfx_values_sorted = np.sort(long_df['TFX'])
        plt.plot(tfx_values_sorted, fit_func(tfx_values_sorted, *popt), color=color)

        # Print the coefficients on the plot
        plt.text(0.05, 0.95, f'Coefficients for {color} genes: {popt}', transform=plt.gca().transAxes)

        plt.xlabel('TFX')
        plt.title(f'Gene values vs TFX for {name_suffix} genes')

        # Save the plot to a file
        plt.savefig(name + name_suffix + '.pdf')
        plt.close()


def scatter_plot(df_ref, df_pred, genes_subset=None, name='scatter', goi_dict=None):
    """
    Function to create a scatter plot, with point densities or different colors for gene groups.

    Parameters:
    df_ref (DataFrame): The reference DataFrame (single column).
    df_pred (DataFrame): The DataFrame with predicted values (single column).
    genes_subset (tuple, optional): Tuple of gene groups to subset to. If None, all genes are considered.
    name (str, optional): Name of the plot.

    Returns:
    None
    """
    # print(df_ref)
    # print(df_pred)
    # Find the intersection of the genes in both DataFrames
    common_genes = df_ref.index.intersection(df_pred.index)
    # If a gene subset is provided, intersect it with the common genes
    if genes_subset is not None:
        common_genes = common_genes.intersection(sum(genes_subset, []))
        # Restrict each list in genes_subset to genes that are in common_genes
        genes_subset = tuple([list(set(gene_list).intersection(common_genes)) for gene_list in genes_subset])
    # Restrict the DataFrames to the common genes
    df_ref = df_ref.loc[common_genes]
    df_pred = df_pred.loc[common_genes]

    # If both DataFrames have more than one column, find the intersection of the columns
    if len(df_ref.shape) > 1 and len(df_pred.shape) > 1:
        common_samples = df_ref.columns.intersection(df_pred.columns)
        df_ref = df_ref[common_samples]
        df_pred = df_pred[common_samples]
    else:
        common_samples = [1]

    # Extract the values
    y_true = df_ref.values.flatten()
    y_pred = df_pred.values.flatten()
    # Remove NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_group = y_true[mask]
    y_pred_group = y_pred[mask]
    # Calculate metrics
    mae = mean_absolute_error(y_true_group, y_pred_group)
    r2 = r2_score(y_true_group, y_pred_group)
    rho, p_val = spearmanr(y_true_group, y_pred_group)
    # Calculate best fit line
    slope = np.polyfit(y_true_group, y_pred_group, 1)[0]

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    if genes_subset is None:
        # # Calculate the 2D histogram of the data
        # H, xedges, yedges = np.histogram2d(y_true_group, y_pred_group, bins=500)

        # # Plot the histogram as a heatmap
        # plt.imshow(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', aspect='auto', cmap='hot', norm=LogNorm())
        # plt.colorbar(label='Density')
        plt.hexbin(y_true_group, y_pred_group, gridsize=100, cmap='viridis_r', bins='log')
        plt.colorbar(label='log10(N)')
    else:
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        for i, gene_group in enumerate(genes_subset):
            # Subset df_ref and df_pred to the specific gene_group
            y_true_subset = df_ref.loc[gene_group].values.flatten()
            y_pred_subset = df_pred.loc[gene_group].values.flatten()

            # Create a mask that selects only pairs where both values are not NaN
            not_nan_mask = ~(np.isnan(y_true_subset) | np.isnan(y_pred_subset))

            # Apply the mask to y_true_subset and y_pred_subset
            y_true_subset = y_true_subset[not_nan_mask]
            y_pred_subset = y_pred_subset[not_nan_mask]

            plt.scatter(y_true_subset, y_pred_subset, alpha=0.5, color=colors[i % len(colors)], edgecolors='none', label=f'subset_{i+1}')
        plt.legend(title='Gene Set', bbox_to_anchor=(1.05, 1), loc='upper left')

    if goi_dict is not None:
        for subtype, genes in goi_dict.items():
            # Subset df_ref and df_pred to the specific genes
            y_true_subset = df_ref.loc[genes].values.flatten()
            y_pred_subset = df_pred.loc[genes].values.flatten()

            # Create a mask that selects only pairs where both values are not NaN
            not_nan_mask = ~(np.isnan(y_true_subset) | np.isnan(y_pred_subset))

            # Apply the mask to y_true_subset and y_pred_subset
            y_true_subset = y_true_subset[not_nan_mask]
            y_pred_subset = y_pred_subset[not_nan_mask]

            # Plot the points for these genes in the color indicated by the subtype
            plt.scatter(y_true_subset, y_pred_subset, color=subtype_to_color[subtype], edgecolors='none')

            # Convert genes to a numpy array to allow boolean indexing
            genes = np.array(genes)
            # Apply the mask to the genes array
            genes_subset = genes[not_nan_mask]

            # Annotate the points
            for i, gene in enumerate(genes_subset):
                plt.annotate(gene, (y_true_subset[i], y_pred_subset[i]), 
                            xytext=(5, 5), textcoords='offset points', 
                            color=subtype_to_color[subtype])

    # Plot best fit line
    x = np.linspace(0, max(y_true_group.max(), y_pred_group.max()) + 1, 100)
    # plt.plot(x, slope * x, 'k--', alpha=0.5)
    plt.plot(x, x, 'k--', alpha=0.5)

    plt.xlabel('True Values in Log2(CPM + 1)')
    plt.ylabel('Predicted Values in Log2(CPM + 1)')
    plt.title(f'Valid genes: {len(y_true)} / Total genes: {len(mask)} / Total comparisons: {len(common_samples)}')

    # Make the plot square and set the axis limits
    plt.axis('square')
    plt.xlim(0, max(y_true_group.max(), y_pred_group.max()) + 1)
    plt.ylim(0, max(y_true_group.max(), y_pred_group.max()) + 1)

    # Print metrics on the plot
    plt.text(0.05, 0.95, f'MAE: {mae:.2f}\nR-squared: {r2:.2f}\nSpearman rho: {rho:.2f}\np-value: {p_val:.2f}\nSlope: {slope:.2f}', 
            transform=plt.gca().transAxes, verticalalignment='top')

    plt.savefig(f"{name}.pdf")
    plt.close()

    return r2, slope, mae


def plot_heatmap(genes, df, tfx_dict, name='heatmap', breaks=None, tfx_sort=False, gsea=False, sub_dict=None, sample_order=None):
    # check if genes is a tuple, if so, unpack it into one list but record count divide as breaks
    if isinstance(genes, tuple):
        breaks, total = [], 0
        for gene_list in genes:
            total += len(gene_list)
            breaks.append(total)
        genes = sum(genes, [])

    # Filter df to include only common samples and genes
    common_samples = df.columns.intersection(tfx_dict.keys())
    if sub_dict is not None:
        common_samples = common_samples.intersection(sub_dict.keys())
    df = df[common_samples].reindex(genes)

    # drop any columns that are all NaN
    df = df.dropna(axis=1, how='all')

    # if sub_dict is provided, replace any non-string values with "Unknown"
    if sub_dict is not None:
        sub_dict = {key: value if isinstance(value, str) else 'Unknown' for key, value in sub_dict.items()}

    # Sort df by subtype and then by sample name if sub_dict is provided, else sort by TFX or alphabetically
    if sample_order is not None:
        df = df[sample_order]
    elif sub_dict is not None and '_' in list(sub_dict.values())[0]:  # If sub_dict is provided and the sample names are in the format "#_subtype1,#_subtype2,..."
        # Define a function to extract the dominant subtype from the subtype string
        def get_dominant_subtype(subtype_str):
            # Split the string into individual subtypes
            subtypes = subtype_str.split(',')
            # Extract the counts and names of the subtypes
            counts_and_names = [subtype.split('_') for subtype in subtypes]
            # Sort the counts and names by count in descending order
            counts_and_names.sort(key=lambda x: int(x[0]), reverse=True)
            # Return the name of the dominant subtype
            return counts_and_names[0][1]
        # Sort the dataframe by dominant subtype and TFX
        df = df.reindex(sorted(df.columns, key=lambda x: (get_dominant_subtype(sub_dict.get(x, '')), -tfx_dict.get(x, 0))), axis=1)
    elif sub_dict is not None:
        df = df.reindex(sorted(df.columns, key=lambda x: (sub_dict.get(x, ''), -tfx_dict.get(x, 0))), axis=1)
        df = df.apply(pd.to_numeric, errors='coerce')
    elif tfx_sort:
        df = df.reindex(sorted(df.columns, key=lambda x: tfx_dict[x], reverse=True), axis=1)
    else:
        df = df.sort_index(axis=1)
    common_samples = df.columns

    # Create a custom colormap
    colors = ["royalblue", "white", "darkred"] if gsea else ["darkblue", "royalblue", "skyblue", "yellow", "orange", "red", "darkred"]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    cmap.set_bad(color='lightgray')  # Set the color for NaN values

    # Calculate figure size
    num_genes, num_samples = len(genes), len(df.columns)
    fig_width, fig_height = max(10, num_samples * 0.2), max(10, num_genes * 0.4) + 1  # Add 1" for the bar plot

    # Create the figure and grid
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 30])  # Adjust the height ratios to account for the fixed height of the bar plot

    # Create the heatmap
    ax2 = plt.subplot(gs[1])
    ax2.set_aspect('equal')  # Make the cells square
    if gsea:
        sns.heatmap(df, cmap=cmap, ax=ax2, linewidths=0.1, linecolor='black', cbar=False, vmin=-1.0, vmax=1.0)
    else:
        sns.heatmap(df, cmap=cmap, ax=ax2, linewidths=0.1, linecolor='black', cbar=False, vmin=0)

    # Adjust the y-limits to show the full range of the heatmap
    ax2.set_ylim(0, len(df.index))
    ax2.invert_yaxis()  # Invert the y-axis to match the order of the genes in the DataFrame

    # Draw horizontal lines at the specified row numbers
    if breaks is not None:
        for b in breaks:
            ax2.axhline(b, color='white', linewidth=3)

    # Set the labels
    ax2.set_xticks(np.arange(len(common_samples)) + 0.5)  # Add 0.5 to center the labels
    ax2.set_xticklabels(df.columns, ha='center')
    ax2.set_yticks(np.arange(len(df.index)) + 0.5)  # Add 0.5 to center the labels with the rows
    ax2.set_yticklabels(df.index, rotation=0)
    if gsea:
        ax2.set_ylabel('Gene Set')
    else:
        ax2.set_ylabel('Gene')

    # Create the colorbar
    cbar = plt.colorbar(ax2.collections[0], ax=ax2, extend='min')
    cbar.set_label("GSVA Score") if gsea else cbar.set_label("Predicted Log2(CPM +1)")

    # Create the bar plot
    ax1 = fig.add_axes([ax2.get_position().x0, ax2.get_position().y0 + ax2.get_position().height + 0.01, ax2.get_position().width, 1.0/fig_height])  # Set the height of the bar plot to 1"

    # If sub_dict is provided, color the bars according to the sample's subtype
    if sub_dict is not None:
        # Check if the first value in sub_dict contains a '_'
        if '_' in list(sub_dict.values())[0]:
            unique_subtypes = set()
            # Iterate over each sample
            for i, sample in enumerate(df.columns):
                # Get the subtype string for the sample
                subtype_str = sub_dict[sample]
                # Split the string into individual subtypes
                subtypes = subtype_str.split(',')
                # Initialize the bottom of the bar
                bottom = 0
                # Iterate over each subtype
                for subtype in subtypes:
                    # Split the subtype into count and name
                    count, sub_name = subtype.split('_')
                    unique_subtypes.add(sub_name)
                    # Calculate the height of the bar for this subtype
                    height = tfx_dict.get(sample, 0) * int(count) / sum(int(s.split('_')[0]) for s in subtypes)
                    # Create the bar
                    ax1.bar(i, height, bottom=bottom, color=subtype_to_color[sub_name], width=1.0, edgecolor='black', linewidth=0.1)
                    # Update the bottom of the bar
                    bottom += height
            unique_subtypes = list(unique_subtypes)
        else:
            # Create a list of colors for the bars
            bar_colors = [subtype_to_color[sub_dict[sample]] for sample in df.columns]
            # Create the bars
            ax1.bar(df.columns, [tfx_dict.get(sample, 0) for sample in df.columns], color=bar_colors, width=1.0, edgecolor='black', linewidth=0.1)
            # Get the unique subtypes in the barplot
            unique_subtypes = df.columns.map(sub_dict).unique()
        # Create the legend
        handles = [Patch(color=subtype_to_color[subtype], label=subtype) for subtype in unique_subtypes]
        ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.5, 1))
    else:
        ax1.bar(df.columns, [tfx_dict.get(sample, 0) for sample in df.columns], color='darkgray', width=1.0, edgecolor='black', linewidth=0.1)

    ax1.set_ylim(0, 1)
    ax1.set_xlim(-0.5, len(df.columns) - 0.5)
    ax1.set_ylabel('TFX')
    ax1.set_xticks([])  # Remove x-axis labels for the bar plot

    # Save the figure to a PDF
    print('SAVING FIGURE: ', name)
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.close()  # Close the figure to free up memory


def plot_double_heatmap(genes, df_true, df_pred, name='double_heatmap', breaks=None, subtypes=None):
    # check if genes is a tuple, if so, unpack it into one list but record count divide as breaks
    if isinstance(genes, tuple):
        breaks, total = [], 0
        for gene_list in genes:
            total += len(gene_list)
            breaks.append(total)
        genes = sum(genes, [])

    # if any genes from genes are not in df_true or df_pred, add those rows with NaN values
    for gene in genes:
        if gene not in df_true.index:
            df_true.loc[gene] = np.nan
        if gene not in df_pred.index:
            df_pred.loc[gene] = np.nan

    # Find the intersection of samples in both DataFrames
    common_samples = df_true.columns.intersection(df_pred.columns)

    # Sort the samples by subtype and then alphabetically if subtypes is not None
    if subtypes is not None:
        sorted_samples = subtypes.loc[common_samples].sort_values().index
    else:
        # Calculate the sum for each sample in df_true for the first gene list
        sums = df_true.loc[genes[:breaks[0]], common_samples].sum()
        # Sort the samples by the calculated sums
        sorted_samples = sums.sort_values().index

    # Filter and sort the DataFrames to include only the common genes and samples
    filtered_df_true = df_true.loc[genes, sorted_samples]
    filtered_df_pred = df_pred.loc[genes, sorted_samples]

    # Define the colors
    colors = ["darkblue", "royalblue", "skyblue", "yellow", "orange", "red", "darkred"]
    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    cmap.set_bad(color='lightgray')  # Set the color for any NaN values

    # Calculate figure size to ensure all labels are visible
    num_genes = len(genes)
    num_samples = len(sorted_samples)
    fig_width = max(10, num_samples * 0.2)  # 0.2 inch per sample
    fig_height = max(10, num_genes * 0.4)  # 0.4 inch per gene (double the height for two heatmaps)

    # Calculate the global minimum and maximum values across both dataframes
    vmin = min(filtered_df_true.min().min(), filtered_df_pred.min().min())
    vmax = max(filtered_df_true.max().max(), filtered_df_pred.max().max())

    # Create the figure
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.02})

    # Create the heatmaps
    sns.heatmap(filtered_df_true, cmap=cmap, linewidths=0.1, linecolor='black', ax=ax1, vmin=vmin, vmax=vmax, cbar=True)
    sns.heatmap(filtered_df_pred, cmap=cmap, linewidths=0.1, linecolor='black', ax=ax2, vmin=vmin, vmax=vmax, cbar=True)

    # Draw horizontal lines at the specified row numbers
    if breaks is not None:
        for b in breaks:
            ax1.axhline(b, color='white', linewidth=5)
            ax2.axhline(b, color='white', linewidth=5)

    # Set the x-ticks and their labels
    ax2.set_xticks(np.arange(len(sorted_samples)) + 0.5)  # Add 0.5 to center the labels
    ax2.set_xticklabels(sorted_samples, ha='center')

    # Set the y-labels
    ax1.set_ylabel('Gene (True)')
    ax2.set_ylabel('Gene (Pred)')

    # Save the figure to a PDF
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.close()


def r2_mae_plot(df_true, df_pred, df_meta, genes, name):
    common_genes = df_true.index.intersection(df_pred.index)
    # If a gene subset is provided, intersect it with the common genes
    if genes is not None:
        common_genes = common_genes.intersection(sum(genes, []))
        # Restrict the DataFrames to the common genes
        df_true = df_true.loc[common_genes]
        df_pred = df_pred.loc[common_genes]

    # Compute R^2 and MAE for each sample
    r2_scores = []
    maes = []
    depths = []

    for sample in df_true.columns:
        if sample in df_pred.columns:
            # Drop NaN values before subsetting to common_genes
            y_true = df_true[sample].dropna()
            y_pred = df_pred[sample].dropna()
            # Find common genes after dropping NaNs
            common_genes = y_true.index.intersection(y_pred.index)
            # Subset to common_genes
            y_true = y_true.loc[common_genes].values
            y_pred = y_pred.loc[common_genes].values
            # Check if y_true and y_pred are not empty
            if y_true.size > 2 and y_pred.size > 2:
                r2_scores.append(r2_score(y_true, y_pred))
                maes.append(mean_absolute_error(y_true, y_pred))
                depths.append(df_meta.loc['depth', sample])

    # Create a DataFrame from the scores and depths
    scores_df = pd.DataFrame({
        'depth': depths,
        'r2_scores': r2_scores,
        'maes': maes
    })

    # Sort the DataFrame by depth
    scores_df = scores_df.sort_values(by='depth')

    # Create the plot
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('R^2', color=color)
    ax1.plot(scores_df['depth'], scores_df['r2_scores'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('MAE', color=color)
    ax2.plot(scores_df['depth'], scores_df['maes'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig(f'{name}.pdf')
    plt.close()


########################## UN-USED FUNCTIONS #########################################################################################
    
# def renormalize_tpm(df):
#     """Renormalize each sample to have total TPM of 1,000,000. Input DF has samples as columns and genes as rows,
#     and expects values to be in Log2(TPM + 1), so it will convert to TPM, renormalize, and convert back to Log2(TPM + 1).
#     """
#     # Convert log2(TPM+1) back to TPM
#     df = 2**df - 1
#     # Renormalize each sample to have total TPM of 1,000,000
#     df = df.div(df.sum()) * 1e6
#     # Convert to log2(TPM + 1)
#     df = np.log2(df + 1)
#     return df


# def impute_rescale_tpm(df):
#     # Convert from log2(tpm+1) to tpm
#     df_tpm = 2**df - 1
#     # Create a mask of the NaN values
#     mask = df_tpm.isna()
#     # Fill NaN values with the row (gene) mean
#     df_tpm.fillna(df_tpm.mean(axis=1), inplace=True)
#     # Rescale each column (sample) to sum to 1e6
#     df_tpm = df_tpm.div(df_tpm.sum(axis=0), axis=1) * 1e6
#     # Convert back to log2(tpm +1)
#     df_log2 = np.log2(df_tpm + 1)
#     # Re-apply the NaN values from the mask
#     df_log2[mask] = np.nan
#     return df_log2


# def perform_reference_deconvolution(all_predictions_df, tfx_dict, background_df, tissues=None):
#     print('Performing reference deconvolution...')
#     if tissues is not None:
#         background_df = background_df[tissues]

#     # Convert values from Log2(TPM +1) to raw TPM
#     all_predictions_df = 2**all_predictions_df - 1

#     # Find common genes between predictions and background
#     common_genes = all_predictions_df.index.intersection(background_df.index)
#     all_predictions_df = all_predictions_df.loc[common_genes]
#     background_df = background_df.loc[common_genes]

#     # Initialize output DataFrames
#     dc_tumor = pd.DataFrame(index=all_predictions_df.index, columns=all_predictions_df.columns)
#     dc_background = all_predictions_df.copy()
#     deconvolution_weights = pd.DataFrame(index=all_predictions_df.columns, columns=background_df.columns)

#     # Perform deconvolution for each sample
#     for sample in all_predictions_df.columns:
#         print(f'Processing sample {sample}...')
#         tfx = tfx_dict[sample]
#         # Filter out NaN values from pred_values
#         pred_values = all_predictions_df[sample].dropna()
#         # Use the same indices to subset background_df
#         background_df_sample = background_df.reindex(pred_values.index)
#         # Find the total TPM of the sample, excluding NaN values, save it, and re-scale to sum to 1,000,000
#         total_tpm = pred_values.sum()
#         pred_values = pred_values / total_tpm * 1e6
#         # Now do the same for each tissue in the background_df_sample
#         background_df_sample = background_df_sample.apply(lambda x: x / x.sum() * 1e6, axis=0)

#         # Make a preliminary guess for the tumor profile: tfx * pred_values
#         preliminary_tumor_profile = pred_values

#         # Objective function for minimization
#         def objective(weights):
#             reconstruction = tfx * preliminary_tumor_profile + (1 - tfx) * np.dot(weights, background_df_sample.values.T)
#             return np.sum((pred_values - reconstruction)**2)

#         # Constraints: weights must sum to <= 1 and be non-negative
#         constraints = ({'type': 'eq', 'fun': lambda x:  1 - np.sum(x)})
#         bounds = [(0, (1 - tfx)) for _ in range(len(background_df_sample.columns))]

#         # Perform minimization
#         # x0 = np.ones(len(background_df_sample.columns)) / len(background_df_sample.columns)
#         x0 = [0.01, 0.01, 0.01, 0.07, 0.9]
#         result = minimize(objective, x0, bounds=bounds, constraints=constraints)

#         # Store the weights
#         deconvolution_weights.loc[sample] = result.x

#         # Save the index before converting to numpy array
#         pred_values_index = pred_values.index

#         # Reconstruct the tumor and background profiles
#         background_profile = (1 - tfx) * np.dot(result.x, background_df_sample.values.T)
#         tumor_profile = pred_values - background_profile
#         tumor_profile = np.clip(tumor_profile, 0, None)

#         # Convert numpy arrays to pandas Series
#         background_profile = pd.Series(background_profile, index=pred_values_index)
#         tumor_profile = pd.Series(tumor_profile, index=pred_values_index)

#         # Re-normalize the profiles to sum to 1,000,000
#         tumor_profile = tumor_profile / tumor_profile.sum() * 1e6
#         background_profile = background_profile / background_profile.sum() * 1e6

#         # Store the profiles in the output DataFrames
#         dc_tumor[sample] = tumor_profile.reindex(dc_tumor.index)
#         dc_background[sample] = background_profile.reindex(dc_background.index)


#     # Convert values from raw TPM to Log2(TPM +1)
#     dc_tumor = np.log2(dc_tumor + 1)
#     dc_background = np.log2(dc_background + 1)
#     print('Done!')

#     return dc_tumor, dc_background, deconvolution_weights




