"""
analysis_utils.py
v1.1.0, 2/22/2024
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script contains helper functions for the analysis of the data.
"""

# Standard library imports
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr

# Third party imports
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score


def subtract_background(preds_df, tfx_dict, sub_sample, ref_file):
    import scipy.stats as stats
    # Load the reference file as a pandas DataFrame
    ref_df = pd.read_csv(ref_file, sep='\t', index_col=0, header=0)

    # Restrict the reference DataFrame to the specified sample
    ref_df = ref_df[[sub_sample]]

    # Un-log2 scale and subtract 1 from all values in both DataFrames
    preds_df = np.power(2, preds_df) - 1
    ref_df = np.power(2, ref_df) - 1

    # For each sample in preds_df, subtract the background linearly
    for sample in preds_df.columns:
        tfx = tfx_dict[sample]
        preds_df[sample] = (preds_df[sample] - (1 - tfx) * ref_df[sub_sample]) / tfx


    # Floor all values at 0
    preds_df = preds_df.clip(lower=0)
    df_tmm = preds_df * 1e6

    ### perform TMM normalization
    ref_sample = pd.read_pickle('../../RNA-Seq/LuCaP_173-1_raw_counts.pkl')
    # Ensure df_tmm and ref_sample have the same index
    df_tmm, ref_sample = df_tmm.align(ref_sample, join='inner', axis=0)
    # Drop rows where all values are 0 in both df_tmm and ref_sample
    mask = (df_tmm != 0).any(axis=1) | (ref_sample != 0).any()
    df_tmm_nz = df_tmm[mask]
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
    df_log2cpm = np.log2(df_tmm.div(lib_sizes_scaled) * 1e6 + 1)  # return log2(CPM + 1)

    return df_log2cpm


########################## HELPER FUNCTIONS ##########################################################################################

def format_pval(pval):
    if pval < 1e-10:
        return "< E-10"
    elif pval < 0.001:
        return f'= {pval:.2E}'
    else:
        return f'= {pval:.3f}'

def load_file_as_list(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

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

def sample_level_correction(df_preds, df_ref, ref_sample='LuCaP_173-1'):
    # Ensure the reference sample exists in the reference DataFrame
    if ref_sample not in df_ref.columns:
        raise ValueError(f"Reference sample '{ref_sample}' not found in df_ref columns")
    
    # Extract the reference sample, dropping NaNs
    ref_values = df_ref[ref_sample].dropna()
    
    # Function to apply quantile normalization without changing ranks
    def quantile_normalize(sample, ref_values):
        # Match the index with the reference values and drop NaNs
        matched_values = sample.dropna().reindex(ref_values.index).dropna()
        valid_ref_values = ref_values.loc[matched_values.index]
        
        # Sort both sample and reference values
        sorted_sample = matched_values.sort_values()
        sorted_ref_values = valid_ref_values.sort_values()
        
        # Map the sorted sample values to the sorted reference values
        result = pd.Series(np.nan, index=sample.index)
        result.loc[sorted_sample.index] = sorted_ref_values.values
        
        return result
    
    # Apply the quantile normalization to each column in df_preds
    df_normalized = df_preds.apply(lambda col: quantile_normalize(col, ref_values), axis=0)
    
    return df_normalized

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
    ax_kde.set_xlabel('log\u2082(CPM+1)')
    ax_kde.set_ylabel('Density')

    ax_hist.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 6})
    ax_hist.set_title('Histograms')
    ax_hist.set_xlabel('log\u2082(CPM+1)')
    ax_hist.set_ylabel('Density')

    # Save the plots as PDF files
    plt.figure(fig_kde.number)
    plt.savefig(filename + "_kernels.pdf", bbox_inches='tight')
    plt.close()

    plt.figure(fig_hist.number)
    plt.savefig(filename + "_histograms.pdf", bbox_inches='tight')
    plt.close()


def plot_vs_meta(df_x, df_y, name_x, name_y, name, x_unit='RNA-Seq log\u2082(CPM+1)', y_unit='Predicted log\u2082(CPM+1)', tfx_dict=None, subtypes=None):
    # if tfx_dict is not None:
    #     tfx_dict = {key[:6]: value for key, value in tfx_dict.items()}

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

    mae = mean_absolute_error(data['x'], data['y'])
    slope = np.polyfit(pd.to_numeric(data['x'], errors='coerce'), pd.to_numeric(data['y'], errors='coerce'), 1)[0]

    plt.figure(figsize=(6, 6))

    x_range = data['x'].max() - data['x'].min()
    y_range = data['y'].max() - data['y'].min()
    if name_x == name_y:
        combined_min = min(data['x'].min(), data['y'].min()) - 0.05 * max(x_range, y_range)
        combined_max = max(data['x'].max(), data['y'].max()) + 0.05 * max(x_range, y_range)
        plt.xlim([combined_min, combined_max])
        plt.ylim([combined_min, combined_max])
    else:
        plt.xlim([data['x'].min() - 0.05 * x_range, data['x'].max() + 0.05 * x_range])
        plt.ylim([data['y'].min() - 0.05 * y_range, data['y'].max() + 0.05 * y_range])

    sizes = [tfx_dict.get(sample, 1) * 200 if tfx_dict is not None else 200 for sample in data.index]
    if subtypes is not None:
        common_index = data.index.intersection(subtypes.index)
        filtered_data = data.loc[common_index]
        colors = [subtype_to_color.get(subtype, 'black') for subtype in subtypes[filtered_data.index]]
        # colors = [subtype_to_color.get(subtype, 'black') for subtype in subtypes[data.index]]
        plt.scatter(data['x'], data['y'], s=sizes, c=colors, alpha=0.8)
        seen_subtypes = subtypes[data.index].unique()
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=subtype_to_color[subtype], markersize=10) for subtype in seen_subtypes]
        plt.legend(handles, seen_subtypes, title='Subtype', bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        plt.scatter(data['x'], data['y'], s=sizes, alpha=0.8)

    if tfx_dict is not None:
        # Create a legend for the point sizes using list comprehension
        markers = [plt.scatter([], [], s=size*200, alpha=1.0, color='gray', label=str(size)) for size in [0.1, 0.3, 0.5, 1.0]]
        plt.legend(handles=markers, title='TFX', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.plot(data['x'], func(data['x'], *popt), '--', color='gray', alpha=0.5)
    plt.xlabel(name_x + ' ' + x_unit)
    plt.ylabel(name_y + ' ' + y_unit)
    plt.title(name_x)
    if name_x == name_y:  # testing for direct linear relationship - use PEARSON
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE: {mae:.2f}\nSlope: {slope:.2f}\nPearson r = {pearson_coef:.3f}\nr p-val {format_pval(pr_val)}', transform=plt.gca().transAxes, verticalalignment='top')
    else:  # testing for monotonic relationship - use SPEARMAN
        plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE: {mae:.2f}\nSlope: {slope:.2f}\nSpearman ρ = {spearman_coef:.3f}\nρ p-val {format_pval(ps_val)}', transform=plt.gca().transAxes, verticalalignment='top')

    plt.savefig(name + '.pdf', bbox_inches='tight')
    plt.close()
    

def scatter_plot(df_ref, df_pred, genes_subset=None, subset_names=None, name='scatter', goi_dict=None):
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
    pearson_coef, pr_val = pearsonr(y_true_group, y_pred_group)
    # Calculate best fit line
    slope = np.polyfit(y_true_group, y_pred_group, 1)[0]

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    if genes_subset is None:
        plt.hexbin(y_true_group, y_pred_group, gridsize=100, cmap='viridis_r', bins='log')
        plt.colorbar(label='$\log_{10}(N+1)$', shrink=0.8)
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

            # Check if subset_names is not None and use it to label the groups
            if subset_names is not None:
                label = f"{subset_names[i]} ({len(gene_group):,})"
            else:
                label = f'subset_{i+1}'

            plt.scatter(y_true_subset, y_pred_subset, alpha=0.5, color=colors[i % len(colors)], edgecolors='none', label=label)
        plt.legend(title='Gene Set', bbox_to_anchor=(1.02, 1), loc='upper left')

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

    plt.xlabel('RNA-Seq log\u2082(CPM+1)')
    plt.ylabel('Predicted log\u2082(CPM+1)')
    plt.title(f'Valid genes: {len(y_true):,} / Total genes: {len(mask):,} / Total samples: {len(common_samples):,}')

    # Make the plot square and set the axis limits
    plt.axis('square')
    plt.xlim(0, max(y_true_group.max(), y_pred_group.max()) + 1)
    plt.ylim(0, max(y_true_group.max(), y_pred_group.max()) + 1)

    # Print metrics on the plot
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE: {mae:.2f}\nSlope: {slope:.2f}\nPearson r = {pearson_coef:.3f}\nr p-val {format_pval(pr_val)}', transform=plt.gca().transAxes, verticalalignment='top')

    plt.savefig(f"{name}.pdf", bbox_inches='tight')
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
    cbar.set_label("GSVA Score") if gsea else cbar.set_label("Predicted log\u2082(CPM+1)")

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
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.close()  # Close the figure to free up memory


def plot_double_heatmap(genes, df_true, df_pred, name='double_heatmap', breaks=None, subtypes=None, gsea=False):
    df_true = df_true.apply(pd.to_numeric, errors='coerce')
    df_pred = df_pred.apply(pd.to_numeric, errors='coerce')
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
        if not gsea:
            # Calculate the sum for each sample in df_true for the first gene list
            sums = df_true.loc[genes[:breaks[0]], common_samples].sum()
            # Sort the samples by the calculated sums
            sorted_samples = sums.sort_values().index
        else:
            # Sort the columns of df_true by the values in the first row
            sorted_samples = df_true.T.sort_values(by=df_true.index[0]).T.columns

    # Filter and sort the DataFrames to include only the common genes and samples
    filtered_df_true = df_true.loc[genes, sorted_samples]
    filtered_df_pred = df_pred.loc[genes, sorted_samples]

    # Define the colors
    colors = ["royalblue", "white", "darkred"] if gsea else ["darkblue", "royalblue", "skyblue", "yellow", "orange", "red", "darkred"]
    # Create the colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    cmap.set_bad(color='lightgray')  # Set the color for any NaN values

    # Calculate figure size to ensure all labels are visible
    num_genes = len(genes)
    num_samples = len(sorted_samples)
    fig_width = max(10, num_samples * 0.2)  # 0.2 inch per sample
    fig_height = max(10, num_genes * 0.4)  # 0.4 inch per gene (double the height for two heatmaps)

    # Calculate the global minimum and maximum values across both dataframes
    if gsea:
        vmin = -1.0
        vmax = 1.0
    else:
        vmin = min(filtered_df_true.min().min(), filtered_df_pred.min().min())
        vmax = max(filtered_df_true.max().max(), filtered_df_pred.max().max())

    # Create the figure
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.02})

    # Create the heatmaps
    ax1 = sns.heatmap(filtered_df_true, cmap=cmap, linewidths=0.1, linecolor='black', ax=ax1, vmin=vmin, vmax=vmax, cbar=True)
    ax2 = sns.heatmap(filtered_df_pred, cmap=cmap, linewidths=0.1, linecolor='black', ax=ax2, vmin=vmin, vmax=vmax, cbar=True)
    for cbar in [ax1.collections[0].colorbar, ax2.collections[0].colorbar]:
        if gsea:
            cbar.set_label("GSVA Score")
        else:
            cbar.set_label('log\u2082(CPM+1)')

    # Draw horizontal lines at the specified row numbers
    if breaks is not None:
        for b in breaks:
            ax1.axhline(b, color='white', linewidth=5)
            ax2.axhline(b, color='white', linewidth=5)

    # Set the x-ticks and their labels
    ax2.set_xticks(np.arange(len(sorted_samples)) + 0.5)  # Add 0.5 to center the labels
    ax2.set_xticklabels(sorted_samples, ha='center')

    # Set the y-labels
    ax1.set_ylabel('True')
    ax2.set_ylabel('Predicted')

    # Save the figure to a PDF
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.close()


def r2_mae_plot(df_true, df_pred, df_meta, genes, name, mode='depth'):
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
    var = []

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
                # r2_scores.append(r2_score(y_true, y_pred))
                r2_scores.append(pearsonr(y_true, y_pred)[0])
                maes.append(mean_absolute_error(y_true, y_pred))
                if mode == 'depth':
                    var.append(df_meta.loc['depth', sample])
                else:
                    var.append(df_meta[sample])

    # Create a DataFrame from the scores and depths
    scores_df = pd.DataFrame({
        'var': var,
        'r2_scores': r2_scores,
        'maes': maes
    })

    # Sort the DataFrame by depth
    scores_df = scores_df.sort_values(by='var')

    # Create the plot
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    if mode == 'depth':
        ax1.set_xlabel('Depth')
    else:
        ax1.set_xlabel('TFX')
    ax1.set_ylabel('R²', color=color)
    ax1.plot(scores_df['var'], scores_df['r2_scores'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1])

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('MAE', color=color)
    ax2.plot(scores_df['var'], scores_df['maes'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Fit a logarithmic trendline to the R^2
    if (scores_df['r2_scores'] >= 0).all():  # Check if all R^2 values are non-negative
        try:
            r2_fit = np.polyfit(np.log(scores_df['var']), scores_df['r2_scores'], 1, w=np.sqrt(scores_df['r2_scores']))
            r2_trendline = r2_fit[1] + r2_fit[0] * np.log(scores_df['var'])
            ax1.plot(scores_df['var'], r2_trendline, color='tab:red', linestyle='dotted')
        except np.linalg.LinAlgError:
            print("Could not fit logarithmic trendline to the R^2")
    # else:
    #     print("Skipped fitting due to negative R^2 values")

    # Fit an exponential decay trendline to the MAE
    try:
        mae_fit = np.polyfit(scores_df['var'], np.log(scores_df['maes']), 1, w=np.sqrt(scores_df['maes']))
        mae_trendline = np.exp(mae_fit[1]) * np.exp(mae_fit[0] * scores_df['var'])
        ax2.plot(scores_df['var'], mae_trendline, color='tab:blue', linestyle='dotted')
    except np.linalg.LinAlgError:
        print("Could not fit exponential decay trendline to the MAE")


    fig.tight_layout()
    plt.savefig(f'{name}.pdf')
    plt.close()

