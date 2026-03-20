"""
analysis_utils.py
v2.0.1, 1/28/2025
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script contains helper functions for the analysis of the data.
"""

# Standard library imports
import os
import warnings
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr, bootstrap
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Third party imports
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as PathEffects


from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc, mean_squared_error
from sklearn.decomposition import PCA

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts

# Set Type 42 font embedding (editable text) for PDFs
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Set global font size
mpl.rcParams['font.size'] = 12

######################################################################################################################################################
### HELPER FUNCTIONS ###
######################################################################################################################################################

def cpm_to_tpm(df, length_ref='../reference/MANE.GRCh38.v1.3.UniversalReference.tsv'):
    # Read the gene length reference file
    length_df = pd.read_csv(length_ref, sep='\t')
    # Merge the gene length information with the input data frame
    merged_df = df.merge(length_df[['symbol', 'mRNA_size']], left_index=True, right_on='symbol', how='inner')
    # Set the 'symbol' column as the index
    merged_df.set_index('symbol', inplace=True)
    # Convert Log2(CPM+1) values to CPM values
    cpm_values = np.power(2, merged_df.drop(columns='mRNA_size')) - 1
    # Calculate RPK (Reads Per Kilobase)
    rpk = cpm_values.div(merged_df['mRNA_size'], axis=0)
    # Calculate the scaling factor for TPM
    scaling_factor = rpk.sum(axis=0) / 1e6
    # Calculate TPM values
    tpm_values = rpk.div(scaling_factor, axis=1)
    # Convert TPM values to Log2(TPM+1) values
    log2_tpm_values = np.log2(tpm_values + 1)
    
    return log2_tpm_values

def renormalize_cpm(preds_df, internal=False):
    import scipy.stats as stats
    # Un-log2 scale and subtract 1 from all values
    preds_df = np.power(2, preds_df) - 1

    # Floor all values at 0
    preds_df = preds_df.clip(lower=0)
    df_tmm = preds_df * 1e6

    ### perform TMM normalization
    if internal:  # use the sample with median overall expression as the reference sample
        column_sums = preds_df.sum(axis=0)
        median_sum = column_sums.median()
        median_column = column_sums.sub(median_sum).idxmin()
        print(f"The column with the median sum is: {median_column}")
        ref_sample = preds_df[median_column]
    else:
        # N.B. the sample below is also used to TMM-scale the data before training
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
    scaling_factors = scaling_factors = scaling_factors / np.exp(np.log(scaling_factors).mean(axis=0))

    # Normalize the counts by the scaling factors
    lib_sizes_scaled = lib_sizes_tmm * scaling_factors.loc['scaling_factor']

    # Combine initial library sizes, scaling factors, and final library sizes into a single DataFrame
    combined_df = pd.concat([lib_sizes_tmm, scaling_factors.T, lib_sizes_scaled.T], axis=1)
    combined_df.columns = ['Initial Library Sizes', 'Scaling Factors', 'Final Library Sizes (TMM-scaled)']
    df_log2cpm = np.log2(df_tmm.div(lib_sizes_scaled) * 1e6 + 1)  # return log2(CPM + 1)

    return df_log2cpm

def format_pval(pval):
    if np.isnan(pval):
        return "p = NaN"
    elif pval < 1e-10:
        return "p < 10⁻¹⁰"
    else:
        exponent = int(np.floor(np.log10(pval)))
        mantissa = pval / 10**exponent
        if exponent < -2 or exponent > 2:
            return f"p = {mantissa:.3f} × 10⁻{abs(exponent)}"
        else:
            return f"p = {pval:.3g}"

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


def get_gene_accuracy(df_truth, df_pred, name='results/AccuracyMetrics'):
    """
    Compute accuracy metrics (R2, RMSE, MAE, nRMSE_stdev, nRMSE_range, truth_stdev, truth_range) for each gene
    between truth and predicted values. Saves a TSV of metrics and distribution plots.
    Plots are restricted to genes with a truth_range >= 2.
    
    Parameters:
    -----------
    df_truth : pandas.DataFrame
        DataFrame containing ground truth gene expression values
    df_pred : pandas.DataFrame
        DataFrame containing predicted gene expression values
    name : str
        Base path for output files (directory will be created if it doesn't exist)
    """
    # Create the results directory (extract directory from name)
    output_dir = os.path.dirname(name)
    if output_dir:  # Only create if there's an actual directory part
        os.makedirs(output_dir, exist_ok=True)
    
    # Find common samples and genes
    common_samples = df_truth.columns.intersection(df_pred.columns)
    common_genes = df_truth.index.intersection(df_pred.index)
    
    # Restrict to common samples and genes
    df_truth_common = df_truth.loc[common_genes, common_samples]
    df_pred_common = df_pred.loc[common_genes, common_samples]
    
    # Initialize results DataFrame
    metric_columns = ['R2', 'RMSE', 'MAE', 'nRMSE_stdev', 'nRMSE_range', 'truth_stdev', 'truth_range']
    results = pd.DataFrame(index=common_genes, columns=metric_columns)
    
    # Calculate metrics for each gene
    for gene in common_genes:
        truth_vals_series = df_truth_common.loc[gene].astype(float)
        pred_vals_series = df_pred_common.loc[gene].astype(float)
        
        # Filter out NaN values
        valid_idx = ~(np.isnan(truth_vals_series) | np.isnan(pred_vals_series))
        truth_vals_clean = truth_vals_series[valid_idx].values
        pred_vals_clean = pred_vals_series[valid_idx].values
        
        # Skip genes with insufficient data
        if len(truth_vals_clean) < 2:
            results.loc[gene, metric_columns] = np.nan
            continue
            
        try:
            # Calculate R2 score
            r2 = r2_score(truth_vals_clean, pred_vals_clean)
            
            # Calculate RMSE (root mean squared error)
            rmse = np.sqrt(np.mean((truth_vals_clean - pred_vals_clean) ** 2))
            
            # Calculate MAE (mean absolute error)
            mae = np.mean(np.abs(truth_vals_clean - pred_vals_clean))
            
            # Calculate truth_stdev and nRMSE_stdev
            std_truth = np.std(truth_vals_clean)
            nrmse_stdev = rmse / std_truth if std_truth > 0 else np.nan
            
            # Calculate truth_range and nRMSE_range
            range_truth = np.max(truth_vals_clean) - np.min(truth_vals_clean)
            nrmse_range = rmse / range_truth if range_truth > 0 else np.nan
            
            # Store results
            results.loc[gene, 'R2'] = r2
            results.loc[gene, 'RMSE'] = rmse
            results.loc[gene, 'MAE'] = mae
            results.loc[gene, 'truth_stdev'] = std_truth
            results.loc[gene, 'nRMSE_stdev'] = nrmse_stdev
            results.loc[gene, 'truth_range'] = range_truth
            results.loc[gene, 'nRMSE_range'] = nrmse_range
        except Exception:
            results.loc[gene, metric_columns] = np.nan
    
    # Save results to TSV file
    results.to_csv(f"{name}_metrics.tsv", sep='\t', index_label='Gene')

    # Filter results for plotting: genes with truth_range >= 2
    results_for_plotting = results[results['truth_range'] >= 2]
    
    # Define columns to plot (excluding the new truth_stdev and truth_range from plotting directly as distributions)
    plot_metric_columns = ['R2', 'RMSE', 'MAE', 'nRMSE_stdev', 'nRMSE_range']

    # Generate distribution plots for each metric from the filtered set
    for metric in plot_metric_columns:
        plt.figure(figsize=(8, 6))
        # Use results_for_plotting here
        metric_values = results_for_plotting[metric].dropna()
        
        if not metric_values.empty:
            sns.histplot(metric_values, kde=True, stat="density", bins=30)
            plt.title(f'Distribution of {metric} (genes with truth range >= 2)')
            plt.xlabel(metric)
            plt.ylabel('Density')
            
            if metric == 'R2':
                plt.xlim([-1, 1])
            elif metric in ['nRMSE_stdev', 'nRMSE_range']:
                plt.xlim([0, 5])
                
            plt.tight_layout()
            plt.savefig(f"{name}_{metric}_distribution_filtered.pdf") # Added _filtered to filename
        plt.close()

# Create a dictionary mapping subtypes to colors
subtype_to_color = create_palette('../reference/palette.tsv')

######################################################################################################################################################
### PLOTTING FUNCTIONS ###
######################################################################################################################################################

def plot_samples_summary(depth_df, subtype_dict, output_file, tfx_dict=None, ho_samples=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Ensure "sample" column exists, filling with index values if it does not
    if 'sample' not in depth_df.columns:
        depth_df['sample'] = depth_df.index

    # Merge the depth data with the subtype data
    depth_df['subtype'] = depth_df['sample'].map(subtype_dict)
    if tfx_dict:
        depth_df['tfx'] = depth_df['sample'].map(tfx_dict).fillna(1.0)
    else:
        depth_df['tfx'] = 1.0

    # remove all columns from depth_df that are not 'sample', 'depth', 'subtype', or 'tfx'
    columns_to_keep = ['sample', 'depth', 'subtype', 'tfx']
    depth_df = depth_df[columns_to_keep]

    # Separate ho_samples if provided
    if ho_samples:
        ho_samples_set = set(ho_samples)
        ho_df = depth_df[depth_df['sample'].isin(ho_samples_set)]
        main_df = depth_df[~depth_df['sample'].isin(ho_samples_set)]
    else:
        ho_df = pd.DataFrame(columns=depth_df.columns)
        main_df = depth_df

    # Sort the dataframes by depth
    main_df = main_df.sort_values(by='depth', ascending=False)
    ho_df = ho_df.sort_values(by='depth', ascending=False)

    # Concatenate the dataframes back together
    if not ho_df.empty:
        plot_df = pd.concat([main_df, ho_df])
    else:
        plot_df = main_df.copy()

    # Determine figure size
    width = len(plot_df) * 0.3 + 2  # Adjusting width based on the number of samples
    height = 4

    # Plotting
    _, ax = plt.subplots(figsize=(width, height))

    seen_subtypes = set()  # Track seen subtypes

    for _, row in plot_df.iterrows():
        sample = row['sample']
        depth = row['depth']
        subtype = row['subtype']
        tfx = row['tfx']

        # Handle multiple subtypes
        if '_' in subtype:
            subtypes = subtype.split(',')
            total_parts = sum(int(sub.split('_')[0]) for sub in subtypes)
            current_bottom = 0
            for sub in subtypes:
                parts, sub_subtype = sub.split('_')
                parts = int(parts)
                seen_subtypes.add(sub_subtype)
                color = subtype_to_color.get(sub_subtype, '#d3d3d3')
                if tfx < 1.0:
                    ax.bar(sample, depth * tfx * (parts / total_parts), bottom=current_bottom, color=color)
                    current_bottom += depth * tfx * (parts / total_parts)
            # Color the remaining non-tfx component
            ax.bar(sample, depth * (1 - tfx), bottom=depth * tfx, color='gray')
        else:
            seen_subtypes.add(subtype)
            color = subtype_to_color.get(subtype, '#d3d3d3')
            if tfx < 1.0:
                ax.bar(sample, depth * tfx, color=color)
                ax.bar(sample, depth * (1 - tfx), bottom=depth * tfx, color='gray')
            else:
                ax.bar(sample, depth, color=color)

    ax.set_xlim(-0.5, len(plot_df) - 0.5)

    # Mark ho_samples section
    if ho_samples:
        ax.axvline(x=len(main_df) - 0.5, color='gray', linestyle='--')

    # Customizing the plot
    ax.set_ylabel('Mean Depth')
    ax.set_xlabel('Samples')

    # Before setting x-tick labels, set x-ticks to match the number of samples
    ax.set_xticks(range(len(plot_df)))
    # Now set x-tick labels with the 'sample' column, and rotate them vertically
    ax.set_xticklabels(plot_df['sample'], rotation='vertical')

    # Adding legend for seen subtypes only
    handles = [Patch(color=subtype_to_color[subtype], label=subtype) for subtype in seen_subtypes]
    ax.legend(handles=handles, title='Subtype', loc="upper left", bbox_to_anchor=(1, 1))

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
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
            col_data = df[column].dropna()
            # Plot the density of the column values
            sns.kdeplot(col_data, ax=ax_kde, label=column)
            # Plot the histogram of the column values
            sns.histplot(col_data, ax=ax_hist, bins=50, label=column, kde=False, element='step', stat='density', fill=False)

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

def plot_vs_meta(df_x, df_y, name_x, name_y, name, x_unit='RNA-Seq log\u2082(CPM+1)', y_unit='Predicted log\u2082(CPM+1)',
                 tfx_dict=None, subtypes=None, print_metrics=False, plot_conf_int=True):
    """
    Scatter-plot df_x[name_x] vs. df_y[name_y], optionally coloring/marking points
    by sample subtype and status, scaling point size by tfx_dict, and drawing a
    best-fit line (linear).  
    
    Parameters
    ----------
    df_x : pd.DataFrame
        DataFrame containing x-values.
    df_y : pd.DataFrame
        DataFrame containing y-values.
    name_x : str
        Column/key name in df_x to plot on the x-axis.
    name_y : str
        Column/key name in df_y to plot on the y-axis.
    name : str
        Filename (without extension) for saving the plot.
    x_unit : str, optional
        Label for x-axis units.
    y_unit : str, optional
        Label for y-axis units.
    tfx_dict : dict, optional
        Dictionary mapping sample IDs to a scaling factor for point sizes.
    subtypes : pd.DataFrame, optional
        DataFrame with 'subtype' (and optionally 'status') columns indexed by sample.
    print_metrics : bool, optional
        If True, prints out metrics (R², MAE, correlation).
    plot_conf_int : bool, default=True
        If True, plots the 95% confidence interval band around the best-fit line.
    """

    # 1. Extract data, coerce to numeric, drop NaNs
    x_values = pd.to_numeric(df_x.loc[name_x], errors='coerce')
    y_values = pd.to_numeric(df_y.loc[name_y], errors='coerce')
    data = pd.DataFrame({'x': x_values, 'y': y_values}).dropna()

    # Exit early if insufficient points
    if len(data) < 2:
        return

    # 2. Define linear function & error objective
    def func(x, a, b):
        return a * x + b

    def error(params, x_arr, y_arr, f):
        return np.sum((y_arr - f(x_arr, *params))**2)

    # 3. Fit linear model via L-BFGS-B
    p0 = [1, 1]  # initial guess
    res = minimize(error, p0, args=(data['x'], data['y'], func), 
                   method='L-BFGS-B', options={'maxiter': 1000})
    a_opt, b_opt = res.x

    # 4. Compute metrics: R², Spearman, Pearson, MAE
    predicted = func(data['x'], a_opt, b_opt)
    r2 = r2_score(data['y'], predicted)
    # Note: This is as in your original code, which uses x vs. y for MAE:
    mae = mean_absolute_error(data['x'], data['y'])  

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spearman_coef, ps_val = spearmanr(data['x'], data['y'])
        pearson_coef, pr_val = pearsonr(data['x'], data['y'])

    # 5. Prepare figure
    plt.figure(figsize=(6, 6))

    # Determine axis limits
    x_min, x_max = data['x'].min(), data['x'].max()
    y_min, y_max = data['y'].min(), data['y'].max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    # If comparing the *same* variable, make axes identical (square plot)
    if name_x == name_y:
        combined_min = min(x_min, y_min) - 0.05 * max(x_range, y_range)
        combined_max = max(x_max, y_max) + 0.05 * max(x_range, y_range)
        plt.xlim([combined_min, combined_max])
        plt.ylim([combined_min, combined_max])
    else:
        plt.xlim([x_min - 0.05 * x_range, x_max + 0.05 * x_range])
        plt.ylim([y_min - 0.05 * y_range, y_max + 0.05 * y_range])

    # 6. Compute point sizes from tfx_dict (default=200 if not found)
    if tfx_dict is not None:
        sizes = [tfx_dict.get(sample, 1) * 300 for sample in data.index]
    else:
        sizes = [300] * len(data)

    # 7. Plot the best-fit line (behind points, solid red)
    #    We'll do this before plotting points (or with lower zorder).
    plt.plot(
        data['x'], predicted, 
        color='black', linestyle='-', alpha=1.0,
        zorder=1  # Behind points
    )

    # If confidence interval should be shown, compute & plot it
    if plot_conf_int:
        x_data = data['x'].values
        n = len(x_data)
        mean_x = np.mean(x_data)
        Sxx = np.sum((x_data - mean_x) ** 2)
        residuals = data['y'].values - func(x_data, a_opt, b_opt)
        s_err = np.sqrt(np.sum(residuals**2) / (n - 2))

        # 95% confidence interval (two-sided)
        from scipy.stats import t
        t_val = t.ppf(0.975, n - 2)

        # Create sorted x for a smooth band
        sorted_idx = np.argsort(x_data)
        x_sorted = x_data[sorted_idx]
        predicted_sorted = func(x_sorted, a_opt, b_opt)
        conf_interval = (
            t_val
            * s_err
            * np.sqrt(1 / n + (x_sorted - mean_x) ** 2 / Sxx)
        )

        # Fill the light gray band
        plt.fill_between(
            x_sorted,
            predicted_sorted - conf_interval,
            predicted_sorted + conf_interval,
            color='gray',
            alpha=0.3,
            zorder=1  # Behind points
        )

    # 8. Plot points
    if subtypes is not None:
        common_index = data.index.intersection(subtypes.index)
        filtered_data = data.loc[common_index]

        # Re-sync sizes to the filtered data
        filtered_sizes = []
        for i, sample in enumerate(data.index):
            if sample in common_index:
                filtered_sizes.append(sizes[i])

        seen_subtypes = set()
        status_exists = ('status' in subtypes.columns)

        # Plot each point individually
        for i, sample in enumerate(filtered_data.index):
            row = subtypes.loc[sample]
            subtype = row['subtype']
            color = subtype_to_color.get(subtype, 'black')
            marker = 'o'  # default
            if status_exists:
                status = row['status']
                marker = 'o' if status == 'validation' else '^'

            plt.scatter(
                filtered_data.loc[sample, 'x'],
                filtered_data.loc[sample, 'y'],
                s=filtered_sizes[i],
                c=color,
                marker=marker,
                alpha=0.7,
                edgecolors='black',  # Add black outline
                linewidths=0.5,      # Make the outline thin
                zorder=2  # Above line
            )
            seen_subtypes.add(subtype)

        # Build legend handles for subtypes
        handles = [
            plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=subtype_to_color.get(sub, 'black'),
                markersize=10, label=sub
            )
            for sub in seen_subtypes
        ]
        plt.legend(
            handles=handles, title='Subtype',
            bbox_to_anchor=(1.02, 1), loc='upper left'
        )

    else:
        # No subtype logic, just scatter all data at once
        plt.scatter(
            data['x'], data['y'],
            s=sizes, alpha=0.7,
            edgecolors='black',  # Add black outline
            linewidths=0.5,      # Make the outline thin
            zorder=2  # Above line
        )

    # 9. If tfx_dict is used, add a legend for TFX sizes
    if tfx_dict is not None:
        legend_sizes = [0.1, 0.3, 0.5, 1.0]
        legend_handles = []
        for tfx in legend_sizes:
            handle = plt.scatter([], [], s=tfx*300, color='gray', alpha=1.0)
            legend_handles.append(handle)
        plt.legend(
            handles=legend_handles,
            labels=[str(s) for s in legend_sizes],
            title='TFX', bbox_to_anchor=(1.02, 0.7), loc='upper left'
        )

    # 10. Labels, title, and text
    plt.xlabel(f'{name_x} {x_unit}')
    plt.ylabel(f'{name_y} {y_unit}')
    plt.title(name_x)

    text_x, text_y = 0.05, 0.95
    if name_x == name_y:
        rmse = np.sqrt(np.mean((data['y'] - data['x'])**2))
        stats_text = (
            f'RMSE = {rmse:.2f}\n'
            f'Pearson r = {pearson_coef:.3f}\n'
            f'{format_pval(pr_val)}'
        )
        if print_metrics:
            print(
                f'{name_x} vs {name_y}: '
                f'R² = {r2:.3f}, MAE = {mae:.2f}, '
                f'Pearson r = {pearson_coef:.3f}, r {format_pval(pr_val)}'
            )
    else:
        rmse = np.sqrt(np.mean((data['y'] - data['x'])**2))
        stats_text = (
            f'RMSE = {rmse:.2f}\n'
            f'Spearman ρ = {spearman_coef:.3f}\n'
            f'{format_pval(ps_val)}'
        )
        if print_metrics:
            print(
                f'{name_x} vs {name_y}: '
                f'R² = {r2:.3f}, MAE = {mae:.2f}, '
                f'Spearman ρ = {spearman_coef:.3f}, ρ {format_pval(ps_val)}'
            )

    plt.text(text_x, text_y, stats_text,
             transform=plt.gca().transAxes, 
             verticalalignment='top')

    # Make the plotting box square (physical), not data ratio
    ax = plt.gca()
    ax.set_box_aspect(1)

    # 11. Save and close
    plt.savefig(name + '.pdf', bbox_inches='tight')
    plt.close()

def scatter_plot(df_ref, df_pred, genes_subset=None, subset_names=None, name='scatter', goi_dict=None, range=None):
    """
    Function to create a scatter plot, with point densities or different colors for gene groups.

    Parameters:
    df_ref (DataFrame): The reference DataFrame (single column).
    df_pred (DataFrame): The DataFrame with predicted values (single column).
    genes_subset (tuple, optional): Tuple of gene groups to subset to. If None, all genes are considered.
    name (str, optional): Name of the plot.
    goi_dict (dict, optional): Dictionary mapping a category (e.g., subtype) to a list of genes of interest.
                               These genes will be highlighted.
    range (tuple, optional): Tuple specifying (min, max) for x and y axes.


    Returns:
    None
    """
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

    # Check if the arrays are not empty
    if len(y_true_group) == 0 or len(y_pred_group) == 0:
        # print(y_true_group, y_pred_group)
        return None, None, None  # or any other appropriate default values

    # Calculate metrics
    mae = mean_absolute_error(y_true_group, y_pred_group)
    r2 = r2_score(y_true_group, y_pred_group)
    pearson_coef, pr_val = pearsonr(y_true_group, y_pred_group)

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

    texts = []  # Initialize list to store text objects for adjustText
    if goi_dict is not None:
        for subtype, genes_in_subtype in goi_dict.items():
            # Ensure genes_in_subtype is a list-like structure
            if not isinstance(genes_in_subtype, (list, pd.Series, np.ndarray)):
                print(f"Warning: Genes for subtype {subtype} in goi_dict are not in a list-like structure. Skipping.")
                continue

            # Filter genes_in_subtype to those present in df_ref (and thus df_pred)
            highlight_genes = [gene for gene in genes_in_subtype if gene in df_ref.index]
            if not highlight_genes:
                continue
            # Subset df_ref and df_pred to the specific highlight_genes
            y_true_subset = df_ref.loc[highlight_genes].values.flatten()
            y_pred_subset = df_pred.loc[highlight_genes].values.flatten()
            # Create a mask that selects only pairs where both values are not NaN
            not_nan_mask = ~(np.isnan(y_true_subset) | np.isnan(y_pred_subset))
            # Apply the mask to y_true_subset and y_pred_subset
            y_true_subset_masked = y_true_subset[not_nan_mask]
            y_pred_subset_masked = y_pred_subset[not_nan_mask]
            # Apply the mask to the gene names as well
            highlight_genes_masked = np.array(highlight_genes)[not_nan_mask]
            # Plot the points for these genes in the color indicated by the subtype
            plt.scatter(
                y_true_subset_masked, y_pred_subset_masked,
                color=subtype_to_color.get(subtype, 'gray'),
                edgecolors='black', linewidths=0.5, s=300, zorder=3
            )
            # Annotate the points
            for i, gene_label in enumerate(highlight_genes_masked):
                text_obj = plt.annotate(
                    gene_label,
                    (y_true_subset_masked[i], y_pred_subset_masked[i]),
                    xytext=(5, 5), textcoords='offset points',
                    color=subtype_to_color.get(subtype, 'black'),
                    fontsize=12,
                    path_effects=[PathEffects.withStroke(linewidth=0.75, foreground='black')],
                    zorder=4)
                texts.append(text_obj)
    # Plot best fit line
    x = np.linspace(0, max(y_true_group.max(), y_pred_group.max()) + 1, 100)
    # plt.plot(x, slope * x, 'k--', alpha=0.5)
    plt.plot(x, x, 'k--', alpha=0.5)

    plt.xlabel('RNA-Seq log\u2082(CPM+1)')
    plt.ylabel('Predicted log\u2082(CPM+1)')
    plt.title(f'Valid genes: {len(y_true):,} / Total genes: {len(mask):,} / Total samples: {len(common_samples):,}')

    # Make the plotting box square (physical), not data ratio
    ax = plt.gca()
    ax.set_box_aspect(1)

    if range is not None:
        plt.xlim(range)
        plt.ylim(range)
    else:
        # set axes from (0,0) to the overall max
        axis_max = max(y_true_group.max(), y_pred_group.max())
        plt.xlim(0, axis_max)
        plt.ylim(0, axis_max)

    # annotate Pearson r and p‑value in top-left
    ax = plt.gca()
    ax.text(
        0.05, 0.95,
        f"Pearson r = {pearson_coef:.3f}\np = {pr_val:.3g}",
        transform=ax.transAxes,
        verticalalignment='top'
    )

    plt.savefig(f"{name}.pdf", bbox_inches='tight')
    plt.close()
    return r2, pearson_coef, mae


def plot_heatmap(genes, df, tfx_dict, name='heatmap', breaks=None, tfx_sort=False, gsea=False, sub_dict=None, sample_order=None):
    """Plot a heatmap of gene expression data or GSEA results.
    Parameters:
    genes (list or tuple): List of gene names or a tuple of lists of gene names.
    df (pd.DataFrame): DataFrame containing gene expression data.
    tfx_dict (dict): Dictionary mapping sample IDs to TFX values.
    name (str): Name of the output file.
    breaks (list): List of row numbers to draw horizontal breaks.
    tfx_sort (bool): If True, sort samples by TFX values.
    gsea (bool): If True, plot GSEA results.
    sub_dict (dict): Dictionary mapping sample IDs to subtypes.
    sample_order (list): List of sample IDs in the desired order."""
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
    # colors = ["royalblue", "white", "darkred"] if gsea else ["darkblue", "royalblue", "skyblue", "yellow", "orange", "red", "darkred"]
    # cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    cmap = plt.get_cmap("inferno")
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
    cbar = plt.colorbar(ax2.collections[0], ax=ax2, extend='min', pad=0.02)
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

    # Handle tuple input for genes and derive breaks if not provided
    derived_breaks = []
    if isinstance(genes, tuple):
        gene_lists = list(genes)
        genes = []
        current_break_point = 0
        for gene_list_item in gene_lists:
            genes.extend(gene_list_item)
            current_break_point += len(gene_list_item)
            derived_breaks.append(current_break_point)
        if breaks is None and derived_breaks:
            breaks = derived_breaks[:-1] # Exclude the total count as a break line

    # Ensure all specified genes are present in dataframes, adding NaNs if not
    for gene_item in list(genes): 
        if gene_item not in df_true.index:
            df_true.loc[gene_item] = np.nan
        if gene_item not in df_pred.index:
            df_pred.loc[gene_item] = np.nan

    # Find common samples and remove samples that are all NaN in either dataframe
    common_samples = df_true.columns.intersection(df_pred.columns)
    common_samples = [
        col for col in common_samples 
        if not (df_true[col].isna().all() or df_pred[col].isna().all())
    ]
    
    if not common_samples:
        print(f"No common, non-empty samples found for {name}. Skipping plot.")
        return
        
    df_true = df_true[common_samples]
    df_pred = df_pred[common_samples]

    # Filter out genes that are all NaN across the selected common samples
    # This defines the actual genes we can plot
    genes_present_in_df = df_true.index.union(df_pred.index)
    valid_genes_for_plot = [
        g for g in genes 
        if g in genes_present_in_df and not (df_true.loc[g, common_samples].isna().all() and df_pred.loc[g, common_samples].isna().all())
    ]

    if not valid_genes_for_plot:
        print(f"No valid genes with non-NaN data found for {name}. Skipping plot.")
        return
    
    # Sort samples
    sorted_samples_list = [] 
    if subtypes is not None and not subtypes.empty:
        samples_for_subtype_sort = subtypes.index.intersection(common_samples)
        if not samples_for_subtype_sort.empty:
            # Add sorting by index name for tie-breaking if subtype index has a name
            sort_by_columns = ['_is_healthy', 'subtype']
            if subtypes.index.name: # Check if index has a name
                 # Temporarily reset index to use original index as a sort column
                subtypes_temp_for_sort = subtypes.loc[samples_for_subtype_sort].reset_index()
                sort_by_columns.append(subtypes.index.name)
                sorted_samples_df = (subtypes_temp_for_sort
                                  .assign(_is_healthy=lambda df_lambda: df_lambda['subtype'].ne('Healthy'))
                                  .sort_values(by=sort_by_columns))
                sorted_samples_list = sorted_samples_df[subtypes.index.name].tolist()
            else: # Fallback if index has no name
                 sorted_samples_list = (subtypes.loc[samples_for_subtype_sort]
                                  .assign(_is_healthy=lambda df_lambda: df_lambda['subtype'].ne('Healthy'))
                                  .sort_values(by=sort_by_columns)
                                  .index.tolist())
        else:
            sorted_samples_list = sorted(list(common_samples))
        # print(sorted_samples_list)
    else: # Original sorting logic if subtypes is None or empty
        if not gsea:
            # Ensure genes for sum are valid and present
            genes_for_sum_calc = [g for g in valid_genes_for_plot[:breaks[0] if breaks else len(valid_genes_for_plot)] if g in df_true.index]
            if genes_for_sum_calc and common_samples:
                sums = df_true.loc[genes_for_sum_calc, common_samples].sum()
                sorted_samples_list = sums.sort_values().index.tolist()
            else:
                sorted_samples_list = sorted(list(common_samples))
        else: # GSEA sorting
            # Ensure the first gene for sorting is valid and present
            first_gsea_sort_gene = next((g for g in valid_genes_for_plot if g in df_true.index), None)
            if first_gsea_sort_gene and common_samples:
                sorted_samples_list = df_true.loc[valid_genes_for_plot, common_samples].T.sort_values(by=first_gsea_sort_gene).T.columns.tolist()
            else:
                sorted_samples_list = sorted(list(common_samples))
    
    if not sorted_samples_list: # Should not happen if common_samples existed, but as a safeguard
        print(f"Sample sorting resulted in an empty list for {name}. Skipping plot.")
        return

    # Filter and sort the DataFrames to include only the final common genes and samples
    # Use valid_genes_for_plot and sorted_samples_list
    filtered_df_true = df_true.loc[valid_genes_for_plot, sorted_samples_list]
    filtered_df_pred = df_pred.loc[valid_genes_for_plot, sorted_samples_list]

    cmap = plt.get_cmap("inferno")
    cmap.set_bad(color='lightgray')

    num_genes_final = len(valid_genes_for_plot)
    num_samples_final = len(sorted_samples_list)


    # Adjusted figure size calculation for horizontal layout
    cell_size = 0.25
    fig_h = max(6, num_genes_final * cell_size + 2.5) 
    # Adjust fig_w slightly to account for a single colorbar and tighter spacing
    fig_w = max(10, (num_samples_final * cell_size) * 2 + 3.0) # Reduced padding slightly

    if gsea:
        vmin_val, vmax_val = -1.0, 1.0
    else:
        all_plot_values = pd.concat([filtered_df_true.stack(), filtered_df_pred.stack()]).dropna()
        if all_plot_values.empty:
            vmin_val, vmax_val = 0, 1 
        else:
            vmin_val = all_plot_values.min()
            vmax_val = all_plot_values.max()

    # Create the figure: 1 row, 2 columns, share y-axis, minimal wspace
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(fig_w, fig_h), sharey=True,
                                   gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.01}, # Adjusted wspace
                                   constrained_layout=True)

    # Heatmap 1 (True) - No colorbar
    if not filtered_df_true.empty:
        sns.heatmap(filtered_df_true, cmap=cmap, linewidths=0.1, linecolor='black',
                    ax=ax1, vmin=vmin_val, vmax=vmax_val, cbar=False, square=False) # cbar=False
        ax1.set_aspect('equal', adjustable='box') 
        ax1.set_title('True Values', fontsize=10)
        if num_genes_final > 0 : ax1.set_ylabel('Genes', fontsize=9)
    else:
        ax1.set_title('True Values (No Data)', fontsize=10)
        if num_genes_final > 0: ax1.set_ylabel('Genes', fontsize=9)
        ax1.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax1.transAxes)


    # Heatmap 2 (Predicted) - With colorbar
    if not filtered_df_pred.empty:
        sns.heatmap(filtered_df_pred, cmap=cmap, linewidths=0.1, linecolor='black',
                    ax=ax2, vmin=vmin_val, vmax=vmax_val, cbar=True, yticklabels=False, square=False,
                    cbar_kws={"shrink": 0.6, "aspect": 25, "pad": 0.04}) 
        ax2.set_aspect('equal', adjustable='box') 
        ax2.set_title('Predicted Values', fontsize=10)
        ax2.set_ylabel('') 
    else:
        ax2.set_title('Predicted Values (No Data)', fontsize=10)
        ax2.set_ylabel('')
        ax2.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax2.transAxes)

    # Set colorbar label only for ax2's colorbar
    if ax2.collections: # Check if heatmap was plotted on ax2
        cb = ax2.collections[0].colorbar
        if cb:
            cb.ax.tick_params(labelsize=7) 
            if gsea:
                cb.set_label("GSVA Score", size=8)
            else:
                cb.set_label('log\u2082(CPM+1)', size=8)


    # Draw horizontal lines for breaks
    if breaks is not None:
        for break_val in breaks:
            if 0 < break_val < num_genes_final: # Ensure break is within bounds
                ax1.axhline(break_val, color='white', linewidth=2)
                ax2.axhline(break_val, color='white', linewidth=2)

    # Set x-ticks and their labels for both axes
    for current_ax in [ax1, ax2]:
        current_ax.set_xticks(np.arange(num_samples_final) + 0.5)
        current_ax.set_xticklabels(sorted_samples_list, rotation=90, ha='center', fontsize=10)
        current_ax.tick_params(axis='x', pad=1) # Adjust padding for x-tick labels
        
        # Set y-tick labels for the first plot (ax1)
        if current_ax == ax1 and num_genes_final > 0:
            current_ax.set_yticks(np.arange(num_genes_final) + 0.5)
            current_ax.set_yticklabels(valid_genes_for_plot, rotation=0, fontsize=10)
            current_ax.tick_params(axis='y', pad=1)


    # plt.subplots_adjust(bottom=0.15, top=0.90, left=0.1, right=0.9) # Adjust overall layout
    if name: # Add a figure title if a name is provided
        fig.suptitle(f'Comparison: {name}', fontsize=12, y=0.97)

    plt.savefig(f'{name}.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_roc_curve(genes, df_true, df_pred, name, threshold=2.0):
    df_true = df_true.apply(pd.to_numeric, errors='coerce')
    df_pred = df_pred.apply(pd.to_numeric, errors='coerce')

    # Flatten the list of genes if it is a list of lists (used in heatmaps)
    if any(isinstance(i, list) for i in genes):
        genes = [gene for sublist in genes for gene in sublist]

    # If any genes from genes are not in df_true or df_pred, add those rows with NaN values
    for gene in genes:
        if gene not in df_true.index:
            df_true.loc[gene] = np.nan
        if gene not in df_pred.index:
            df_pred.loc[gene] = np.nan

    # Find the intersection of samples in both DataFrames
    common_samples = df_true.columns.intersection(df_pred.columns)
    df_true = df_true[common_samples]
    df_pred = df_pred[common_samples]

    # Filter the DataFrames to include only the common genes
    filtered_df_true = df_true.loc[genes]
    filtered_df_pred = df_pred.loc[genes]

    # Flatten the DataFrames to create a single list of values for true and predicted
    y_true = filtered_df_true.values.flatten()
    y_pred = filtered_df_pred.values.flatten()

    # Remove NaN values
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    # Calculate the binary labels based on the threshold
    true_labels = (y_true >= threshold).astype(int)
    # pred_labels = (y_pred >= threshold).astype(int)

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(true_labels, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.close()


def thresholding_boxes(df_true, df_pred, df_meta, genes, name):
    # Restrict to samples common to df_true, df_pred, and df_meta
    common_samples = df_true.columns.intersection(df_pred.columns).intersection(df_meta.columns)
    df_true = df_true[common_samples]
    df_pred = df_pred[common_samples]
    # Create a copy of the filtered df_meta to avoid SettingWithCopyWarning
    df_meta = df_meta[common_samples].copy()

    common_genes = df_true.index.intersection(df_pred.index)
    # If a gene subset is provided, intersect it with the common genes
    if genes is not None:
        common_genes = common_genes.intersection(sum(genes, []))
        # Restrict the DataFrames to the common genes
        df_true = df_true.loc[common_genes]
        df_pred = df_pred.loc[common_genes]

    # Determine which grouping variables are available
    grouping_vars = ['tfx']
    
    # Check how many unique tfx values we have
    unique_tfx_count = len(df_meta.loc['tfx'].unique())
    few_unique_tfx = unique_tfx_count < 10
    
    # Only add depth if we have normal binning (many unique tfx values)
    if not few_unique_tfx and 'depth' in df_meta.index:
        grouping_vars.append('depth')

    # Prepopulate the DataFrame with samples and metrics
    data_rows = []
    for sample in df_pred.columns:
        for metric in ['r', 'RMSE', 'AUC']:
            row = {'sample': sample, 'metric': metric}
            # Add each grouping variable to the row
            for var in grouping_vars:
                row[var] = df_meta.loc[var, sample]
            data_rows.append(row)

    # Create the DataFrame
    scores_df = pd.DataFrame(data_rows)

    # Fill in the score values
    for sample in df_pred.columns:
        if sample in df_true.columns:
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
                # Calculate metrics
                r = pearsonr(y_true, y_pred)[0]
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                true_labels = (y_true >= 2.0).astype(int)
                fpr, tpr, _ = roc_curve(true_labels, y_pred)
                roc_auc = auc(fpr, tpr)
                
                # Update scores in the DataFrame
                scores_df.loc[(scores_df['sample'] == sample) & (scores_df['metric'] == 'r'), 'score'] = r
                scores_df.loc[(scores_df['sample'] == sample) & (scores_df['metric'] == 'RMSE'), 'score'] = rmse
                scores_df.loc[(scores_df['sample'] == sample) & (scores_df['metric'] == 'AUC'), 'score'] = roc_auc

        # Create separate plots for each metric and grouping variable
        metrics = ['r', 'RMSE', 'AUC']
        
        # Save the complete metrics data to TSV
        tsv_output_path = f'{name}_metrics_by_sample.tsv'
        scores_df.to_csv(tsv_output_path, sep='\t', index=False)
        print(f"Saved complete metrics data to {tsv_output_path}")
        
        # Additionally, create and save a summary by TFX values
        summary_df = pd.DataFrame()
        
        for var in grouping_vars:
            for metric in metrics:
                metric_df = scores_df[scores_df['metric'] == metric]
                
                # Group by TFX or TFX bins
                if (var == 'tfx') and few_unique_tfx:
                    # For few unique TFX values, group directly by TFX
                    summary = metric_df.groupby(var)['score'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
                    summary['grouping_var'] = var
                    summary['metric'] = metric
                else:
                    # For many unique values, create and use bins
                    bin_col = f'{var}_bin'
                    if bin_col not in metric_df.columns:
                        # Create bins appropriate for each variable
                        if var == 'tfx':
                            bins = np.linspace(0, 1, 11)
                        else:
                            min_val = scores_df[var].min()
                            max_val = scores_df[var].max()
                            bins = np.linspace(min_val, max_val, 11)
                        
                        metric_df[bin_col] = pd.cut(metric_df[var], bins=bins, include_lowest=True)
                    
                    # Group by bins
                    summary = metric_df.groupby(bin_col)['score'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
                    summary['grouping_var'] = var
                    summary['metric'] = metric
                
                summary_df = pd.concat([summary_df, summary], ignore_index=True)
        
        # Save summary by TFX/bins to TSV
        summary_tsv_path = f'{name}_metrics_summary.tsv'
        summary_df.to_csv(summary_tsv_path, sep='\t', index=False)
        print(f"Saved metrics summary by grouping variables to {summary_tsv_path}")
    
    for var in grouping_vars:
        # For tfx with few unique values, use those exact values instead of bins
        if (var == 'tfx') and few_unique_tfx:
            bin_col = var  # Use the actual values directly
        else:
            # Create bins appropriate for each variable - reduced from 20 to 10 bins
            if var == 'tfx':
                # For tfx, create 10 bins of width 0.1
                bins = np.linspace(0, 1, 11)
            else:
                # For other variables, create 10 equally-spaced bins across the range
                min_val = scores_df[var].min()
                max_val = scores_df[var].max()
                bins = np.linspace(min_val, max_val, 11)
                
            # Add binned column
            bin_col = f'{var}_bin'
            scores_df[bin_col] = pd.cut(scores_df[var], bins=bins, include_lowest=True)
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_df = scores_df[scores_df['metric'] == metric]
            
            # Create the boxplot
            sns.boxplot(x=bin_col, y='score', data=metric_df, ax=ax)
            # Add individual points on top of the boxplot
            sns.swarmplot(x=bin_col, y='score', data=metric_df, ax=ax, color="black", alpha=0.5, size=3)
            
            # Set title and labels
            ax.set_title(f"{metric} by {var}")
            ax.set_xlabel(var)
            ax.set_ylabel(metric)
            
            # Set y-axis limits for r and AUC
            if metric in ['r', 'AUC']:
                ax.set_ylim(0.0, 1.0)
                
            # Make x-labels more readable
            plt.xticks(rotation=90)
            
            # Save the figure
            fig.tight_layout()
            plt.savefig(f'{name}_{metric}_by_{var}.pdf')
            plt.close()

    # Add ROC curve plot when few unique TFX values exist
    if few_unique_tfx:
        # Get unique TFX values
        unique_tfxs = sorted(df_meta.loc['tfx'].unique())
        
        # Create a figure for ROC curves
        plt.figure(figsize=(8, 8))
        
        # Color map for different TFX values
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_tfxs)))
        
        # Plot ROC curve for each TFX value
        for i, tfx_value in enumerate(unique_tfxs):
            # Get samples with this TFX value
            tfx_samples = df_meta.columns[df_meta.loc['tfx'] == tfx_value]
            
            # Prepare data for ROC curve
            all_true = []
            all_pred = []
            
            for sample in tfx_samples:
                # Only process if sample is in both true and pred dataframes
                if sample in df_true.columns and sample in df_pred.columns:
                    y_true = df_true[sample].dropna()
                    y_pred = df_pred[sample].dropna()
                    
                    # Find common genes after dropping NaNs
                    common_genes = y_true.index.intersection(y_pred.index)
                    
                    # Subset to common_genes
                    y_true = y_true.loc[common_genes].values
                    y_pred = y_pred.loc[common_genes].values
                    
                    # Check if data is valid
                    if y_true.size > 2 and y_pred.size > 2:
                        # Add to collections
                        true_labels = (y_true >= 2.0).astype(int)
                        all_true.extend(true_labels)
                        all_pred.extend(y_pred)
            
            # Skip if not enough data
            if len(all_true) < 10:
                continue
                
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(all_true, all_pred)
            roc_auc = auc(fpr, tpr)
            
            # Plot the ROC curve
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                     label=f'TFX={tfx_value:.2f} (AUC={roc_auc:.3f})')
        
        # Plot the random guess line
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        
        # Set labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by TFX Value')
        
        # Set axis limits
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Add legend
        plt.legend(loc="lower right")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'{name}_roc_by_tfx.pdf')
        plt.close()


def cluster_analysis(df_true, df_pred, name, subtypes=None):
    """
    Performs cluster analysis on gene expression data from true and predicted dataframes.
    
    Args:
        df_true (pd.DataFrame): DataFrame containing reference/true values
        df_pred (pd.DataFrame): DataFrame containing predicted values
        name (str): Base name for output files
        subtypes (pd.DataFrame, optional): DataFrame with 'subtype' column indexed by sample IDs
    """
    # 1. Restrict to common genes and samples
    common_genes = df_true.index.intersection(df_pred.index)
    common_samples = df_true.columns.intersection(df_pred.columns)
    
    df_true = df_true.loc[common_genes, common_samples]
    df_pred = df_pred.loc[common_genes, common_samples]
    
    # Drop any genes/rows with ANY NaN values in either dataframe
    mask_true = df_true.isna().any(axis=1)
    mask_pred = df_pred.isna().any(axis=1)
    genes_with_nans = df_true.index[mask_true | mask_pred]
    
    print(f"Removing {len(genes_with_nans)} genes with NaN values out of {len(common_genes)} total genes")
    
    # Remove genes with NaNs from both dataframes
    df_true = df_true.drop(genes_with_nans)
    df_pred = df_pred.drop(genes_with_nans)
    
    # Check if we have any valid genes left
    if len(df_true) == 0 or len(df_pred) == 0:
        print("No valid genes remaining after NaN removal. Skipping analysis.")
        return
    
    # 2. Find the top 1000 most variable genes from remaining valid genes (predictions - unsupervised)
    # Calculate variance for each gene across all samples
    var_pred = df_pred.var(axis=1)
    # Get the top 1000 most variable genes
    n_top_genes = min(1000, len(var_pred))
    top_genes = var_pred.nlargest(n_top_genes).index
    
    # Subset dataframes to top variable genes
    df_true_top = df_true.loc[top_genes]
    df_pred_top = df_pred.loc[top_genes]
    
    # 3. Create PCA plot
    # Transpose to have genes as features and samples as observations
    X_true = df_true_top.T.values
    X_pred = df_pred_top.T.values
    
    # Combine data for a joint PCA (no scaling - assume data is already scaled)
    X_combined = np.vstack([X_true, X_pred])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_combined)
    
    # Extract PCA results for each dataset
    n_samples = len(common_samples)
    pca_true = pca_result[:n_samples]
    pca_pred = pca_result[n_samples:]
    
    # Create PCA plot
    plt.figure(figsize=(10, 8))
    
    # Plot points with coloring based on subtypes if provided
    if subtypes is not None:
        # Get common samples with subtypes
        common_with_subtypes = common_samples.intersection(subtypes.index)
        
        # Get unique subtypes for legend
        unique_subtypes = subtypes.loc[common_with_subtypes, 'subtype'].unique()
        
        # Plot points colored by subtype, with different markers for true vs pred
        for i, sample in enumerate(common_samples):
            if sample in subtypes.index:
                subtype = subtypes.loc[sample, 'subtype']
                color = subtype_to_color.get(subtype, 'gray')
                
                # True data points (circles)
                plt.scatter(pca_true[i, 0], pca_true[i, 1], 
                          c=color, marker='o', s=80, alpha=0.7, edgecolors='black')
                
                # Predicted data points (squares)
                plt.scatter(pca_pred[i, 0], pca_pred[i, 1], 
                          c=color, marker='s', s=80, alpha=0.7, edgecolors='black')
                
                # Connect matching pairs
                plt.plot([pca_true[i, 0], pca_pred[i, 0]], 
                       [pca_true[i, 1], pca_pred[i, 1]], 
                       color=color, alpha=0.5, linewidth=0.5)
        
        # Create legend for subtypes and for true/predicted markers
        subtype_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=subtype_to_color.get(subtype, 'gray'), 
                                   markersize=8, label=subtype) for subtype in unique_subtypes]
        
        marker_handles = [plt.Line2D([0], [0], marker='o', color='black', markersize=8, label='True'),
                        plt.Line2D([0], [0], marker='s', color='black', markersize=8, label='Predicted')]
        
        # Combine legends
        all_handles = subtype_handles + marker_handles
        plt.legend(handles=all_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Default coloring (blue for true, red for predicted)
        plt.scatter(pca_true[:, 0], pca_true[:, 1], c='blue', marker='o', label='True', alpha=0.7)
        plt.scatter(pca_pred[:, 0], pca_pred[:, 1], c='red', marker='s', label='Predicted', alpha=0.7)
        
        # Draw lines connecting matching sample pairs
        for i in range(n_samples):
            plt.plot([pca_true[i, 0], pca_pred[i, 0]], 
                   [pca_true[i, 1], pca_pred[i, 1]], 
                   'gray', alpha=0.5, linewidth=0.5)
        
        plt.legend()
    
    # Add labels and title
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title(f'PCA of Top {n_top_genes} Most Variable Genes')
    
    # Save PCA plot
    plt.tight_layout()
    plt.savefig(f'{name}_pca.pdf', bbox_inches='tight')
    plt.close()
    
    # 4. Create hierarchical clustering dendrogram
    # Prepare labels
    labels_true = [f"{sample} (truth)" for sample in common_samples]
    labels_pred = [f"{sample} (pred)" for sample in common_samples]
    labels_combined = labels_true + labels_pred
    
    # Compute linkage using correlation distance without scaling
    corr_dist = pdist(X_combined, metric='correlation')
    Z_combined = linkage(corr_dist, method='average')
    
    # Create dendrogram with colored labels based on subtypes
    plt.figure(figsize=(12, 8))
    dendrogram_plot = dendrogram(
        Z_combined,
        labels=labels_combined,
        leaf_rotation=90,
        leaf_font_size=8,
    )
    
    # Color the labels in the dendrogram if subtypes are provided
    if subtypes is not None:
        ax = plt.gca()
        xlbls = ax.get_xticklabels()
        
        for lbl in xlbls:
            sample_name = lbl.get_text()
            
            # Extract original sample name by removing the ' (truth)' or ' (pred)' suffix
            original_name = sample_name.split(' ')[0]
            
            if original_name in subtypes.index:
                subtype = subtypes.loc[original_name, 'subtype']
                color = subtype_to_color.get(subtype, 'black')
                lbl.set_color(color)
    
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance (1 - Correlation)')
    plt.tight_layout()
    plt.savefig(f'{name}_dendrogram.pdf', bbox_inches='tight')
    plt.close()

def plot_pearson_heatmap(truth_df, preds_df, subtypes_df, name="spearman_heatmap"):
    """
    Create a heatmap of Pearson correlations between true and predicted values,
    organized by cohort.
    
    Parameters:
    -----------
    truth_df : pandas.DataFrame
        DataFrame with true values (indices as rows, samples as columns)
    preds_df : pandas.DataFrame
        DataFrame with predicted values (indices as rows, samples as columns)
    subtypes_df : pandas.DataFrame
        DataFrame with 'subtype' column containing cohort information, indexed by sample ID
    name : str, default="spearman_heatmap"
        Base name for the output file
        
    Returns:
    --------
    pandas.DataFrame: DataFrame with correlation results
    """
    # Find common indices between truth and prediction dataframes
    common_indices = truth_df.index.intersection(preds_df.index)
    if len(common_indices) == 0:
        print("No common indices found between truth and prediction dataframes")
        return pd.DataFrame()
    
    # Get unique cohorts from subtypes_df
    if 'subtype' not in subtypes_df.columns:
        print("'subtype' column not found in subtypes_df")
        return pd.DataFrame()
    
    unique_cohorts = sorted(subtypes_df['subtype'].unique())
    
    # Create a DataFrame to store correlation results
    # Columns: cohorts + "Combined"
    # Rows: common indices
    correlations = pd.DataFrame(index=common_indices, 
                               columns=unique_cohorts + ["Combined"])
    
    # Calculate correlations for each cohort
    for cohort in unique_cohorts:
        # Get samples for this cohort
        cohort_samples = subtypes_df[subtypes_df['subtype'] == cohort].index
        # Get common samples between cohort samples and both dataframes
        common_samples = cohort_samples.intersection(
            truth_df.columns.intersection(preds_df.columns))
        
        if len(common_samples) < 3:  # Need at least 3 samples for meaningful correlation
            print(f"Skipping cohort {cohort}: insufficient samples")
            continue
            
        # Calculate Pearson correlation for each index in this cohort
        for idx in common_indices:
            true_vals = pd.to_numeric(truth_df.loc[idx, common_samples], errors='coerce')
            pred_vals = pd.to_numeric(preds_df.loc[idx, common_samples], errors='coerce')
            
            # Filter out NaN values
            valid_mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
            true_vals_clean = true_vals[valid_mask]
            pred_vals_clean = pred_vals[valid_mask]
            
            # Calculate correlation if we have enough data points
            if len(true_vals_clean) >= 3:
                # Check if data has variance (pearsonr will fail if either array is constant)
                if true_vals_clean.var() > 0 and pred_vals_clean.var() > 0:
                    corr, _ = pearsonr(true_vals_clean, pred_vals_clean)
                    correlations.loc[idx, cohort] = corr
    
    # Calculate "Combined" correlations (across all samples)
    all_common_samples = truth_df.columns.intersection(preds_df.columns)
    
    for idx in common_indices:
        true_vals = pd.to_numeric(truth_df.loc[idx, all_common_samples], errors='coerce')
        pred_vals = pd.to_numeric(preds_df.loc[idx, all_common_samples], errors='coerce')
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
        true_vals_clean = true_vals[valid_mask]
        pred_vals_clean = pred_vals[valid_mask]
        
        # Calculate correlation if we have enough data points
        if len(true_vals_clean) >= 3:
            # Check if data has variance
            if true_vals_clean.var() > 0 and pred_vals_clean.var() > 0:
                corr, _ = pearsonr(true_vals_clean, pred_vals_clean)
                correlations.loc[idx, "Combined"] = corr
    
    # Drop rows with all NaN values
    correlations = correlations.dropna(how='all')
    
    # Save to TSV
    tsv_filename = f"{name}.tsv"
    correlations.to_csv(tsv_filename, sep='\t')
    
    # Create the heatmap
    if not correlations.empty:
        # Ensure the correlations DataFrame is of float type for heatmap
        correlations = correlations.astype(float)

        # Determine figure size based on data dimensions
        fig_height = max(8, len(correlations) * 0.4)
        fig_width = max(8, len(correlations.columns) * 0.7)
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Define colormap: blue (-1) to black (0) to red (1)
        colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # Blue, Black, Red
        positions = [0, 0.5, 1]
        cmap = mpl.colors.LinearSegmentedColormap.from_list("blue_black_red", list(zip(positions, colors)))
        
        # Create the heatmap
        ax = sns.heatmap(
            correlations,
            cmap=cmap,
            vmin=-1, vmax=1,
            annot=False,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Pearson\'s r'},
            mask=correlations.isna(),  # Mask NaN values - this is okay
            square=True  # Ensure cells are square
        )
        
        # Set labels and title
        plt.title('Pearson Correlations between True and Predicted Values by Cohort')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{name}.pdf")
        plt.close()
    
    return correlations

def ihc_spearman_heatmap(preds_df, truth_df, ihc_dfs, name="ihc_spearman_heatmap"):
    """
    Create heatmaps of Spearman correlations between:
    1. RNA-Seq vs Predicted gene expression
    2. RNA-Seq gene expression vs IHC scores 
    3. Predicted gene expression vs IHC scores
    
    Parameters:
    -----------
    preds_df : pandas.DataFrame
        Dataframe of predicted gene expression values (genes as rows, samples as columns)
    truth_df : pandas.DataFrame
        Dataframe of true gene expression values (genes as rows, samples as columns)
    ihc_dfs : list of pandas.DataFrame
        List of dataframes containing IHC scores (IHC metrics as rows, samples as columns)
        Expecting format where IHC scores have names like "GENE_mean" or "GENE_median"
    name : str
        Base name for the output file
        
    Returns:
    --------
    tuple: (mean_results_df, median_results_df) with correlation results
    """
    # Separate results for mean and median
    results_mean = []
    results_median = []
    
    # Process each IHC dataframe
    for ihc_df in ihc_dfs:
        # Determine cohort based on sample naming convention in the columns
        if any("PC-TAN_" in col for col in ihc_df.columns):
            cohort = "PC"
        elif any("BC-TAN_" in col for col in ihc_df.columns):
            cohort = "BC"
        else:
            continue  # Skip if we can't determine the cohort
        
        # Find common samples between all datasets - these will be used for ALL genes
        initial_common_samples = list(set(truth_df.columns).intersection(preds_df.columns).intersection(ihc_df.columns))
        if len(initial_common_samples) < 3:
            continue
        
        # Process genes with MEAN IHC scores
        ihc_mean_genes = [idx.split("_")[0] for idx in ihc_df.index if "_mean" in idx]
        
        # Pre-filter to find common samples that have valid data for ALL genes
        valid_samples_for_all_means = set(initial_common_samples)
        for gene in ihc_mean_genes:
            if gene in truth_df.index and gene in preds_df.index:
                ihc_key = f"{gene}_mean"
                
                # Check samples with valid data for this gene
                valid_for_truth = ~truth_df.loc[gene, initial_common_samples].isna()
                valid_for_pred = ~preds_df.loc[gene, initial_common_samples].isna()
                valid_for_ihc = ~ihc_df.loc[ihc_key, initial_common_samples].isna()
                
                # Find samples with valid data across all three measures
                valid_samples_for_gene = set([
                    s for i, s in enumerate(initial_common_samples) 
                    if valid_for_truth.iloc[i] and valid_for_pred.iloc[i] and valid_for_ihc.iloc[i]
                ])
                
                # Update the intersection of valid samples
                valid_samples_for_all_means.intersection_update(valid_samples_for_gene)
        
        # Convert to list for indexing
        common_samples_mean = list(valid_samples_for_all_means)
        
        # Skip if we don't have enough samples after filtering
        if len(common_samples_mean) < 3:
            print(f"Skipping mean IHC analysis for {cohort}: insufficient samples with complete data.")
            continue
            
        # Now process each gene using the consistent sample set
        for gene in ihc_mean_genes:
            if gene in truth_df.index and gene in preds_df.index:
                ihc_key = f"{gene}_mean"
                
                # Extract the data for this gene (using consistent sample set)
                data = pd.DataFrame({
                    'truth': truth_df.loc[gene, common_samples_mean],
                    'pred': preds_df.loc[gene, common_samples_mean],
                    'ihc': ihc_df.loc[ihc_key, common_samples_mean]
                })
                
                # Remove rows with any NaN values to get actual valid sample count per test
                data_clean = data.dropna()
                
                if len(data_clean) < 3:  # Need at least 3 samples for meaningful correlation
                    continue
                
                # Calculate correlations using only valid data
                true_vs_pred_rho, _ = spearmanr(data_clean['truth'], data_clean['pred'])
                true_vs_ihc_rho, _ = spearmanr(data_clean['truth'], data_clean['ihc'])
                pred_vs_ihc_rho, _ = spearmanr(data_clean['pred'], data_clean['ihc'])
                
                # Store results with actual sample count used
                results_mean.append({
                    'cohort': cohort,
                    'gene': gene,
                    'true_vs_pred': true_vs_pred_rho,
                    'true_vs_ihc': true_vs_ihc_rho,
                    'pred_vs_ihc': pred_vs_ihc_rho,
                    'n_samples': len(data_clean)
                })
        
        # Similar approach for MEDIAN IHC scores
        ihc_median_genes = [idx.split("_")[0] for idx in ihc_df.index if "_median" in idx]
        
        # Pre-filter for median scores
        valid_samples_for_all_medians = set(initial_common_samples)
        for gene in ihc_median_genes:
            if gene in truth_df.index and gene in preds_df.index:
                ihc_key = f"{gene}_median"
                if ihc_key not in ihc_df.index:
                    continue
                
                valid_for_truth = ~truth_df.loc[gene, initial_common_samples].isna()
                valid_for_pred = ~preds_df.loc[gene, initial_common_samples].isna()
                valid_for_ihc = ~ihc_df.loc[ihc_key, initial_common_samples].isna()
                
                valid_samples_for_gene = set([
                    s for i, s in enumerate(initial_common_samples) 
                    if valid_for_truth.iloc[i] and valid_for_pred.iloc[i] and valid_for_ihc.iloc[i]
                ])
                
                valid_samples_for_all_medians.intersection_update(valid_samples_for_gene)
        
        common_samples_median = list(valid_samples_for_all_medians)
        
        if len(common_samples_median) < 3:
            print(f"Skipping median IHC analysis for {cohort}: insufficient samples with complete data.")
            continue
            
        for gene in ihc_median_genes:
            if gene in truth_df.index and gene in preds_df.index:
                ihc_key = f"{gene}_median"
                if ihc_key not in ihc_df.index:
                    continue
                
                data = pd.DataFrame({
                    'truth': truth_df.loc[gene, common_samples_median],
                    'pred': preds_df.loc[gene, common_samples_median],
                    'ihc': ihc_df.loc[ihc_key, common_samples_median]
                })
                
                # Remove rows with any NaN values to get actual valid sample count per test
                data_clean = data.dropna()
                
                if len(data_clean) < 3:  # Need at least 3 samples for meaningful correlation
                    continue
                
                # Calculate correlations using only valid data
                true_vs_pred_rho, _ = spearmanr(data_clean['truth'], data_clean['pred'])
                true_vs_ihc_rho, _ = spearmanr(data_clean['truth'], data_clean['ihc'])
                pred_vs_ihc_rho, _ = spearmanr(data_clean['pred'], data_clean['ihc'])
                
                # Store results with actual sample count used
                results_median.append({
                    'cohort': cohort,
                    'gene': gene,
                    'true_vs_pred': true_vs_pred_rho,
                    'true_vs_ihc': true_vs_ihc_rho,
                    'pred_vs_ihc': pred_vs_ihc_rho,
                    'n_samples': len(data_clean)
                })
    
    # Create dataframes from results
    mean_results_df = None if not results_mean else pd.DataFrame(results_mean).sort_values(['cohort', 'gene'])
    median_results_df = None if not results_median else pd.DataFrame(results_median).sort_values(['cohort', 'gene'])
    
    # Create and save heatmaps if we have data
    if mean_results_df is not None and len(mean_results_df) > 0:
        create_ihc_heatmap(mean_results_df, f"{name}_mean.pdf", "Mean IHC Scores")

    if median_results_df is not None and len(median_results_df) > 0:
        create_ihc_heatmap(median_results_df, f"{name}_median.pdf", "Median IHC Scores")
    
    return (mean_results_df, median_results_df)


def create_ihc_heatmap(results_df, output_filename, subtitle):
    """Helper function to create a correlation heatmap for IHC data"""
    # Prepare data for heatmap
    heatmap_data = []
    row_labels = []
    cohort_boundaries = []
    
    current_cohort = None
    current_row = 0
    
    for _, row in results_df.iterrows():
        # Track cohort boundaries for visualization
        if current_cohort != row['cohort']:
            if current_row > 0:  # Don't add boundary at the start
                cohort_boundaries.append(current_row)
            current_cohort = row['cohort']
            
        heatmap_data.append([
            row['true_vs_pred'],
            row['true_vs_ihc'],
            row['pred_vs_ihc']
        ])
        
        # Include sample count in the label
        row_labels.append(f"{row['cohort']}: {row['gene']} (n={row['n_samples']})")
        current_row += 1
    
    # Save the heatmap data to a TSV file
    heatmap_df_to_save = pd.DataFrame(heatmap_data, index=row_labels, columns=['RNA-Seq vs Pred', 'RNA-Seq vs IHC', 'Pred vs IHC'])
    tsv_output_filename = output_filename.replace('.pdf', '.tsv')
    heatmap_df_to_save.to_csv(tsv_output_filename, sep='\t')

    # Create the heatmap
    plt.figure(figsize=(6, len(heatmap_data) * 0.4 + 1))
    
    # Define colormap: blue (-1) to black (0) to red (1)
    colors = [(0, 0, 1), (0, 0, 0), (1, 0, 0)]  # Blue, Black, Red
    positions = [0, 0.5, 1]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("blue_black_red", list(zip(positions, colors)))
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        vmin=-1, vmax=1,
        annot=False,  # Changed from True to False
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Spearman\'s ρ'},
        mask=np.isnan(heatmap_data)  # Mask NaN values
    )
    
    # Set labels
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_xticklabels(['RNA-Seq vs Pred', 'RNA-Seq vs IHC', 'Pred vs IHC'], rotation=30)
    
    plt.title(f'Spearman Correlations between Expression and {subtitle}')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_filename)
    plt.close()
    
    return results_df

def confidence_analysis(preds_df, confs_df, truth_df, name='confidence_analysis'):
    """
    Analyze the relationship between prediction confidence and both expression values and prediction errors.
    
    Parameters:
    -----------
    preds_df : pandas.DataFrame
        DataFrame containing predicted expression values
    confs_df : pandas.DataFrame
        DataFrame containing confidence values (0-1) for each prediction
    truth_df : pandas.DataFrame
        DataFrame containing true expression values, used for error calculation (required)
    name : str, optional
        Base name for output files
        
    Outputs:
    --------
    Several plots exploring the relationships between confidence and expression/error values:
    - pred-expression-vs-activity_score.pdf : Scatter plot of predicted expression vs activity score
    - truth-expression-vs-activity_score.pdf : Scatter plot of true expression vs activity score
    - true_confidence-vs-activity_score.pdf : Scatter plot of true confidence vs activity score
    - residuals-vs-activity_score.pdf : Scatter plot of error vs activity score
    - confidence_vs_rmse.pdf : Confidence vs RMSE plot
    - risk_coverage_curve.pdf : Risk–Coverage curve
    """
    # Get common genes and samples between all dataframes
    common_genes = preds_df.index.intersection(confs_df.index).intersection(truth_df.index)
    common_samples = preds_df.columns.intersection(confs_df.columns).intersection(truth_df.columns)
    
    # Filter dataframes to common genes and samples
    preds_subset = preds_df.loc[common_genes, common_samples]
    confs_subset = confs_df.loc[common_genes, common_samples]
    truth_subset = truth_df.loc[common_genes, common_samples]
    
    # Flatten the matrices into 1D arrays for plotting
    conf_values = confs_subset.values.flatten()
    pred_expr_values = preds_subset.values.flatten()
    truth_expr_values = truth_subset.values.flatten()
    
    # Calculate true confidence values using the sigmoid function
    truth_conf_values = 1 / (1 + np.exp(-(truth_expr_values-2)/0.5))
    
    # Remove NaN values for the first plot (pred expression vs activity score)
    mask = ~np.isnan(conf_values) & ~np.isnan(pred_expr_values)
    filtered_conf_values = conf_values[mask]
    filtered_pred_expr_values = pred_expr_values[mask]
    
    # 1. Create predicted expression vs activity score plot
    plt.figure(figsize=(10, 8))
    plt.hexbin(filtered_conf_values, np.abs(filtered_pred_expr_values), gridsize=100, cmap='viridis', bins='log')
    plt.colorbar(label='Count (log scale)')
    plt.title('Predicted Expression Magnitude vs Activity Score')
    plt.xlabel('Activity Score (0-1)')
    plt.ylabel('Absolute Predicted Expression |log₂(CPM+1)|')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{name}/pred-expression-vs-activity_score.pdf')
    plt.close()
    
    # Remove NaN values for all remaining plots that include truth values
    mask = ~np.isnan(conf_values) & ~np.isnan(pred_expr_values) & ~np.isnan(truth_expr_values)
    filtered_conf_values = conf_values[mask]
    filtered_pred_expr_values = pred_expr_values[mask]
    filtered_truth_expr_values = truth_expr_values[mask]
    filtered_truth_conf_values = truth_conf_values[mask]
    
    # 2. Create true expression vs activity score plot
    plt.figure(figsize=(10, 8))
    plt.hexbin(filtered_conf_values, np.abs(filtered_truth_expr_values), gridsize=100, cmap='viridis', bins='log')
    plt.colorbar(label='Count (log scale)')
    plt.title('True Expression Magnitude vs Activity Score')
    plt.xlabel('Activity Score (0-1)')
    plt.ylabel('Absolute True Expression |log₂(CPM+1)|')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{name}/truth-expression-vs-activity_score.pdf')
    plt.close()
    
    # 3. Create true confidence vs activity score plot
    plt.figure(figsize=(10, 8))
    plt.hexbin(filtered_conf_values, filtered_truth_conf_values, gridsize=100, cmap='viridis', bins='log')
    plt.colorbar(label='Count (log scale)')
    plt.plot([0, 1], [0, 1], 'r--')  # Add diagonal line for reference
    plt.title('True Confidence vs Activity Score')
    plt.xlabel('Activity Score (0-1)')
    plt.ylabel('True Confidence (0-1)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{name}/true_confidence-vs-activity_score.pdf')
    plt.close()
    
    # 4. Calculate residuals and create residuals vs activity score plot
    residuals = filtered_pred_expr_values - filtered_truth_expr_values
    abs_residuals = np.abs(residuals)
    
    plt.figure(figsize=(10, 8))
    plt.hexbin(filtered_conf_values, abs_residuals, gridsize=100, cmap='viridis', bins='log')
    plt.colorbar(label='Count (log scale)')
    
    # Calculate mean residual and RMSE across confidence bins
    bin_width = 0.02  # 50 bins
    bins = np.arange(0, 1 + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    
    mean_residuals = []
    mean_rmse = []
    
    for i in range(len(bins)-1):
        bin_mask = (filtered_conf_values >= bins[i]) & (filtered_conf_values < bins[i+1])
        if np.sum(bin_mask) > 0:  # Ensure we have data in this bin
            # Mean residual for this bin (not absolute, to show bias)
            mean_residuals.append(np.mean(residuals[bin_mask]))
            
            # Mean RMSE for this bin
            bin_rmse = np.sqrt(np.mean(residuals[bin_mask]**2))
            mean_rmse.append(bin_rmse)
        else:
            mean_residuals.append(np.nan)
            mean_rmse.append(np.nan)
    
    # Add the mean residual line
    line1, = plt.plot(bin_centers, mean_residuals, 'b-', linewidth=3, label='Mean Residual')
    
    # Add the RMSE line
    line2, = plt.plot(bin_centers, mean_rmse, 'r-', linewidth=3, label='Root Mean Squared Error')
    
    # Create legend with both lines
    plt.legend(handles=[line1, line2])
    
    plt.title('Prediction Error vs Activity Score')
    plt.xlabel('Activity Score (0-1)')
    plt.ylabel('Error (prediction - truth)')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{name}/residuals-vs-activity_score.pdf')
    plt.close()
    
    # 5. Additional plot: Confidence vs RMSE (Confidence = 2·|s−0.5|)
    conf_trans = (2 * np.abs(conf_values - 0.5))**2
    mask_conf = ~np.isnan(conf_trans) & ~np.isnan(pred_expr_values) & ~np.isnan(truth_expr_values)
    ct = conf_trans[mask_conf]
    resid = (pred_expr_values - truth_expr_values)[mask_conf]
    bins2 = np.linspace(0, 1, 101)
    centers2 = bins2[:-1] + (bins2[1] - bins2[0]) / 2
    rmse2 = []
    for i in range(len(bins2) - 1):
        m = (ct >= bins2[i]) & (ct < bins2[i+1])
        rmse2.append(np.sqrt(np.mean(resid[m]**2))) if m.any() else rmse2.append(np.nan)
    plt.figure(figsize=(10, 8))
    plt.plot(centers2, rmse2, 'r-', linewidth=3)
    plt.title('Confidence vs RMSE')
    plt.xlabel('Confidence (2×|Activity Score - 0.5|)')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig(f'{name}/confidence_vs_rmse.pdf')
    plt.close()
    
    # 6. Risk–Coverage curve using transformed confidence
    #    (use the same confidence bins as in plot 5)
    conf_all = ct
    err_abs  = np.abs(pred_expr_values[mask_conf] - truth_expr_values[mask_conf])

    # reuse bins2 and centers2 from above
    bins = bins2
    centers = centers2

    coverage = [(conf_all >= b).sum() / len(conf_all) for b in bins[:-1]]
    risk     = [err_abs[conf_all >= b].mean() if (conf_all >= b).any() else np.nan
                for b in bins[:-1]]

    plt.figure(figsize=(8, 6))
    plt.plot(centers, risk, marker='o')
    plt.title('Risk–Coverage Curve')
    plt.xlabel('Confidence (2×|Activity Score - 0.5|)')
    plt.ylabel('Risk (mean absolute error)')
    plt.tight_layout()
    plt.savefig(f'{name}/risk_coverage_curve.pdf')
    plt.close()

    # RMSE vs confidence threshold (transformed)
    plt.figure(figsize=(8, 6))
    plt.plot(centers, rmse2, marker='o')
    plt.title('RMSE vs Confidence Threshold')
    plt.xlabel('Confidence (2×|Activity Score - 0.5|)')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig(f'{name}/rmse_vs_threshold.pdf')
    plt.close()

    # for t, c, r in zip(bins2, coverage, risk):
    #     print(f"threshold ≥ {t:0.2f} → coverage {c:0.2f}, mean error {r:0.2f}")

    return


def hl_boxes(preds_df, truth_df, hl_df_param, output_dir_path):
    """
    Generates line plots comparing Pearson r and AUC metrics
    for predicted vs. true gene expression, binned by gene half-life.
    Metrics are calculated per sample for genes within each bin, then averaged.
    95% confidence intervals are shown. A secondary axis displays average true expression.

    Args:
        preds_df (pd.DataFrame): DataFrame of predicted expression values (genes as index, samples as columns).
        truth_df (pd.DataFrame): DataFrame of true expression values (genes as index, samples as columns).
        hl_df_param (pd.DataFrame): DataFrame with gene names in the first data column 
                                   and half-life values in the second data column. Index is ignored.
        output_dir_path (str): Path to the directory where the output PDF will be saved.
    """
    # --- Constants ---
    N_TARGET_BINS = 10 # Reduced from 500 for faster testing, can be increased
    MIN_GENES_PER_BIN = 5
    MIN_PAIRS_FOR_PEARSONR = 3
    MIN_SAMPLES_FOR_CI_BOOTSTRAP = 5 # Min samples contributing to a bin's metric for CI

    # 1. Prepare hl_df
    hl_df_processed = hl_df_param.iloc[:, [0, 1]].copy()
    hl_df_processed.columns = ['gene', 'half_life']
    hl_df_processed['half_life'] = pd.to_numeric(hl_df_processed['half_life'], errors='coerce')
    hl_df_processed.dropna(subset=['gene', 'half_life'], inplace=True)
    hl_df_processed.set_index('gene', inplace=True)

    # 2. Find common genes and filter DataFrames
    common_genes = preds_df.index.intersection(truth_df.index).intersection(hl_df_processed.index)
    
    preds_df_common = preds_df.loc[common_genes]
    truth_df_common = truth_df.loc[common_genes]
    hl_df_common = hl_df_processed.loc[common_genes]

    preds_df_common = preds_df_common.dropna(axis=1, how='all')
    truth_df_common = truth_df_common.dropna(axis=1, how='all')

    common_samples_final = preds_df_common.columns.intersection(truth_df_common.columns)
    preds_df_common = preds_df_common[common_samples_final]
    truth_df_common = truth_df_common[common_samples_final]
    
    if preds_df_common.empty or truth_df_common.empty or hl_df_common.empty or len(common_samples_final) == 0:
        print("One of the DataFrames is empty or no common samples after processing. Skipping plot.")
        return

    # 3. Create Half-Life Bins
    # Ensure there are enough unique half-life values for qcut
    if hl_df_common['half_life'].nunique() < 2: # Need at least 2 unique values for qcut to make bins
        print("Not enough unique half-life values to create bins. Skipping plot.")
        return
    
    # Adjust N_TARGET_BINS if fewer unique half-lives than target bins
    # or if fewer genes than target bins
    effective_n_bins = min(N_TARGET_BINS, hl_df_common['half_life'].nunique() -1, len(hl_df_common) // MIN_GENES_PER_BIN)
    if effective_n_bins < 1:
        effective_n_bins = 1 # Ensure at least one bin if possible
    
    if effective_n_bins == 1 and hl_df_common['half_life'].nunique() > 1 : # if only one bin can be made, but there are multiple HL values
         # Create bins manually if qcut struggles with too few effective_n_bins
        hl_df_common_sorted = hl_df_common.sort_values(by='half_life')
        # if we can only make one bin, assign all to bin 0
        hl_df_common['bin_id'] = 0
    elif hl_df_common['half_life'].nunique() <= effective_n_bins : # if unique HL values are less than or equal to effective_n_bins
        # Assign unique half-life values to unique bins
        hl_to_bin_map = {hl: i for i, hl in enumerate(sorted(hl_df_common['half_life'].unique()))}
        hl_df_common['bin_id'] = hl_df_common['half_life'].map(hl_to_bin_map)
    else:
        try:
            hl_df_common['bin_id'] = pd.qcut(hl_df_common['half_life'], effective_n_bins, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"Error during pd.qcut: {e}. Trying with fewer bins or manual assignment.")
            # Fallback: if qcut fails, assign unique half-life values to unique bins if possible
            if  hl_df_common['half_life'].nunique() > 1:
                hl_to_bin_map = {hl: i for i, hl in enumerate(sorted(hl_df_common['half_life'].unique()))}
                hl_df_common['bin_id'] = hl_df_common['half_life'].map(hl_to_bin_map)
            else: # Single half-life value
                hl_df_common['bin_id'] = 0


    actual_num_bins = hl_df_common['bin_id'].nunique()
    if actual_num_bins == 0:
        print("No bins created. Skipping plot.")
        return

    # 4. Calculate Metrics per Bin
    bin_results = []
    
    for bin_idx in sorted(hl_df_common['bin_id'].unique()):
        if pd.isna(bin_idx):
            continue
        
        genes_in_bin = hl_df_common[hl_df_common['bin_id'] == bin_idx].index
        
        if len(genes_in_bin) < MIN_GENES_PER_BIN:
            continue

        mean_hl_in_bin = hl_df_common.loc[genes_in_bin, 'half_life'].mean()
        
        truth_bin_genes = truth_df_common.loc[genes_in_bin]
        preds_bin_genes = preds_df_common.loc[genes_in_bin]

        sample_pearson_rs = []
        sample_aucs = []
        sample_mean_true_exprs = []

        for sample in common_samples_final:
            y_true_s_all = truth_bin_genes[sample]
            y_pred_s_all = preds_bin_genes[sample]
            
            temp_df = pd.DataFrame({'true': y_true_s_all, 'pred': y_pred_s_all}).dropna()
            y_true_s = temp_df['true'].values
            y_pred_s = temp_df['pred'].values

            # Calculate Mean True Expression for this sample in this bin
            if len(y_true_s) > 0:
                sample_mean_true_exprs.append(np.mean(y_true_s))

            # Calculate Pearson r
            if len(y_true_s) >= MIN_PAIRS_FOR_PEARSONR:
                if not (np.all(y_true_s == y_true_s[0]) or np.all(y_pred_s == y_pred_s[0])):
                    r_val, _ = pearsonr(y_true_s, y_pred_s)
                    if not np.isnan(r_val):
                        sample_pearson_rs.append(r_val)
            
            # Calculate AUC
            if len(y_true_s) >= MIN_PAIRS_FOR_PEARSONR: # Often need similar amount of data
                true_labels_binary = (y_true_s >= 2.0).astype(int)
                if len(np.unique(true_labels_binary)) > 1: # Need both classes
                    fpr, tpr, _ = roc_curve(true_labels_binary, y_pred_s)
                    roc_auc = auc(fpr, tpr)
                    if not np.isnan(roc_auc):
                        sample_aucs.append(roc_auc)
        
        # Aggregate metrics for the bin
        current_bin_data = {'mean_hl': mean_hl_in_bin, 'num_genes': len(genes_in_bin)}

        for metric_name, sample_values in [('pearson_r', sample_pearson_rs), 
                                           ('auc', sample_aucs), 
                                           ('mean_true_expr', sample_mean_true_exprs)]:
            if len(sample_values) >= MIN_SAMPLES_FOR_CI_BOOTSTRAP:
                # Convert to numpy array for bootstrap
                values_arr = np.array(sample_values)
                # Filter out NaNs that might have slipped through, for bootstrap
                values_arr_clean = values_arr[~np.isnan(values_arr)]
                if len(values_arr_clean) >= MIN_SAMPLES_FOR_CI_BOOTSTRAP:
                    mean_val = np.mean(values_arr_clean)
                    # Bootstrap CI
                    # Use try-except for bootstrap as it can sometimes fail with small/degenerate samples
                    try:
                        ci_result = bootstrap((values_arr_clean,), np.mean, confidence_level=0.95, 
                                              random_state=1, method='BCa') # BCa is generally good
                        ci_low, ci_high = ci_result.confidence_interval.low, ci_result.confidence_interval.high
                    except Exception: # Catch any bootstrap error
                        ci_low, ci_high = np.nan, np.nan
                else: # Not enough clean samples for bootstrap
                    mean_val = np.mean(values_arr_clean) if len(values_arr_clean) > 0 else np.nan
                    ci_low, ci_high = np.nan, np.nan

            elif len(sample_values) > 0: # Not enough for CI, but can calculate mean
                mean_val = np.mean(np.array(sample_values)[~np.isnan(sample_values)])
                ci_low, ci_high = np.nan, np.nan
            else: # No values
                mean_val, ci_low, ci_high = np.nan, np.nan, np.nan
            
            current_bin_data[f'mean_{metric_name}'] = mean_val
            current_bin_data[f'{metric_name}_ci_low'] = ci_low
            current_bin_data[f'{metric_name}_ci_high'] = ci_high
        
        bin_results.append(current_bin_data)

    plot_df = pd.DataFrame(bin_results)
    plot_df.dropna(subset=['mean_hl', 'mean_pearson_r', 'mean_auc', 'mean_mean_true_expr'], how='any', inplace=True)
    plot_df.sort_values(by='mean_hl', inplace=True)

    if plot_df.empty or len(plot_df) < 2: # Need at least 2 points for a line plot
        print("Not enough valid data points after binning to generate plots. Skipping.")
        return

    # 5. Create Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    fig.suptitle('Performance Metrics vs. Gene Half-Life', fontsize=16)

    # --- Pearson r Plot ---
    ax1 = axes[0]
    # Primary axis (Pearson r)
    ax1.plot(plot_df['mean_hl'], plot_df['mean_pearson_r'], color='dodgerblue', label='Mean Pearson r')
    ax1.fill_between(plot_df['mean_hl'], plot_df['pearson_r_ci_low'], plot_df['pearson_r_ci_high'], 
                     color='dodgerblue', alpha=0.2, label='95% CI (Pearson r)')
    ax1.set_xlabel('Mean Gene Half-Life in Bin')
    ax1.set_ylabel('Pearson Correlation Coefficient (r)', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    # Consider log scale for x-axis if half-lives span many orders of magnitude
    # ax1.set_xscale('log') 

    # Secondary axis (True Expression)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(plot_df['mean_hl'], plot_df['mean_mean_true_expr'], color='forestgreen', linestyle='--', label='Avg. True Expression')
    ax1_twin.fill_between(plot_df['mean_hl'], plot_df['mean_true_expr_ci_low'], plot_df['mean_true_expr_ci_high'],
                          color='forestgreen', alpha=0.2, label='95% CI (True Expr.)')
    ax1_twin.set_ylabel('Average True log2(CPM+1)', color='forestgreen')
    ax1_twin.tick_params(axis='y', labelcolor='forestgreen')
    
    ax1.set_title('Pearson r by Gene Half-Life')
    # Combine legends for ax1
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')


    # --- AUC Plot ---
    ax2 = axes[1]
    # Primary axis (AUC)
    ax2.plot(plot_df['mean_hl'], plot_df['mean_auc'], color='darkorange', label='Mean AUC')
    ax2.fill_between(plot_df['mean_hl'], plot_df['auc_ci_low'], plot_df['auc_ci_high'],
                     color='darkorange', alpha=0.2, label='95% CI (AUC)')
    ax2.set_xlabel('Mean Gene Half-Life in Bin')
    ax2.set_ylabel('Area Under ROC Curve (AUC)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    # ax2.set_xscale('log')

    # Secondary axis (True Expression)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(plot_df['mean_hl'], plot_df['mean_mean_true_expr'], color='forestgreen', linestyle='--', label='Avg. True Expression')
    ax2_twin.fill_between(plot_df['mean_hl'], plot_df['mean_true_expr_ci_low'], plot_df['mean_true_expr_ci_high'],
                          color='forestgreen', alpha=0.2, label='95% CI (True Expr.)')
    ax2_twin.set_ylabel('Average True log2(CPM+1)', color='forestgreen')
    ax2_twin.tick_params(axis='y', labelcolor='forestgreen')

    ax2.set_title('AUC by Gene Half-Life')
    # Combine legends for ax2
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 6. Save Plot
    os.makedirs(output_dir_path, exist_ok=True)
    plot_filename = os.path.join(output_dir_path, "gene_hl_binned_metrics.pdf")
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved binned line plot to {plot_filename}")


def combined_auc_roc(
    truth_df, preds_df, cohort_dict, name,
    genes=None, variable=None, roc_threshold=2.0,
    n_bootstraps=1000, random_seed=42
):
    """
    Plot cohort-level ROC curves with 95% bootstrap CIs (clustered by sample)
    and report ROC AUC ± CI.

    Parameters
    ----------
    truth_df, preds_df : pd.DataFrame  (rows = genes/variables, columns = samples)
    cohort_dict        : dict {sample_id: cohort_name}
    name               : str   basename for the output PDF
    genes              : list / 'all'  – optional gene set (continuous truth values, binarized by roc_threshold)
    variable           : str           – optional single 0/1 row in truth_df
    roc_threshold      : float         – threshold to binarize continuous truth values for ROC analysis when using 'genes'.
    n_bootstraps       : int           – number of cluster bootstrap replicates
    random_seed        : int           – RNG seed for reproducibility
    """
    # ---------- housekeeping ----------
    # Ensure common samples
    common_samples = (
        truth_df.columns.intersection(preds_df.columns)
        .intersection(list(cohort_dict.keys())) # Use list() for robustness
    )
    truth_df_processed = truth_df[common_samples].copy() # Use .copy() to avoid SettingWithCopyWarning
    preds_df_processed = preds_df[common_samples].copy()

    # Align rows (genes/variables) based on 'genes' or 'variable' parameter
    if genes is not None: # Operating in "gene set" mode
        if variable is not None:
            raise ValueError("Cannot specify both 'genes' and 'variable'. Pick one.")
        
        common_rows = pd.Index([]) # Initialize as empty pandas Index
        if isinstance(genes, str) and genes.lower() == 'all':
            # Use all common genes between the two dataframes
            common_rows = truth_df_processed.index.intersection(preds_df_processed.index)
        elif isinstance(genes, list):
            # Use genes from the provided list that are common to both DataFrames
            genes_index = pd.Index(genes)
            common_rows_in_dfs = truth_df_processed.index.intersection(preds_df_processed.index)
            # Filter genes_index to those present in common_rows_in_dfs, maintaining order from genes_index
            common_rows = genes_index[genes_index.isin(common_rows_in_dfs)]
        else:
            raise ValueError("Invalid 'genes' parameter. Must be a list of gene names, 'all', or None.")

        if common_rows.empty:
            print(f"Warning: No common genes found for ROC analysis based on 'genes' parameter: {genes}. ROC AUC will likely be N/A.")
            # Create empty DFs with correct columns to prevent errors, but allow graceful N/A handling
            truth_df_processed = pd.DataFrame(index=common_rows, columns=truth_df_processed.columns)
            preds_df_processed = pd.DataFrame(index=common_rows, columns=preds_df_processed.columns)
        else:
            truth_df_processed = truth_df_processed.loc[common_rows]
            preds_df_processed = preds_df_processed.loc[common_rows]
            
    elif variable is not None: # Operating in "single variable" mode
        # Ensure the variable exists in both dataframes
        if variable not in truth_df_processed.index:
            raise ValueError(f"Variable '{variable}' not found in truth_df_processed index.")
        if variable not in preds_df_processed.index:
            raise ValueError(f"Variable '{variable}' not found in preds_df_processed index.")
        # No further global row filtering needed; _flatten_for_roc will select the variable.
    else: # Both genes and variable are None
        raise ValueError("Either 'genes' or 'variable' must be specified for ROC analysis.")

    # Filter common_samples again by those actually present in cohort_dict after potential modifications
    cohorts = sorted({cohort_dict[s] for s in common_samples if s in cohort_dict})
    
    base_fpr = np.linspace(0, 1, 101) # Base FPR for interpolation
    rng = np.random.default_rng(random_seed)

    fig, ax = plt.subplots(figsize=(7, 7))
    text_blurbs = []

    # ---------- iterate over cohorts ----------
    for cohort_idx, cohort in enumerate(cohorts):
        # Select samples for the current cohort that are still in common_samples
        cohort_samples = [s for s in common_samples if cohort_dict.get(s) == cohort and s in truth_df_processed.columns]
        if not cohort_samples: # Skip if cohort has no valid samples
            text_blurbs.append(f'{cohort}: ROC AUC = N/A (no samples)')
            continue
            
        truth_c = truth_df_processed[cohort_samples]
        preds_c = preds_df_processed[cohort_samples]

        # ---- helper to make 1-D y_true (binary) / y_score for *this* (gene/variable, sample) matrix
        def _flatten_for_roc(sel_truth, sel_pred, current_genes_param, current_variable_param, threshold_param):
            if current_genes_param is not None:  # gene set case
                y_t_flat_raw = sel_truth.values.flatten()
                y_p_flat_raw = sel_pred.values.flatten()
                
                y_t_flat = pd.to_numeric(y_t_flat_raw, errors='coerce')
                y_p_flat = pd.to_numeric(y_p_flat_raw, errors='coerce')
                
                if len(y_t_flat) != len(y_p_flat):
                    raise ValueError(f"Internal error: Flattened truth and prediction arrays have different lengths: {len(y_t_flat)} vs {len(y_p_flat)}. This should not happen if input DataFrames are correctly aligned.")

                mask = ~(np.isnan(y_t_flat) | np.isnan(y_p_flat))
                y_true_continuous = y_t_flat[mask]
                y_pred_scores = y_p_flat[mask]
                y_true_binary = (y_true_continuous >= threshold_param).astype(int)
            else:  # single binary variable (current_variable_param is not None)
                y_t_series = sel_truth.loc[current_variable_param]
                y_p_series = sel_pred.loc[current_variable_param]

                y_t_numeric = pd.to_numeric(y_t_series, errors='coerce')
                y_p_numeric = pd.to_numeric(y_p_series, errors='coerce')

                y_t_flat = y_t_numeric.values.ravel()
                y_p_flat = y_p_numeric.values.ravel()
                
                mask = ~(np.isnan(y_t_flat) | np.isnan(y_p_flat))
                y_true_binary = y_t_flat[mask].astype(int) 
                y_pred_scores = y_p_flat[mask]
            return y_true_binary, y_pred_scores
            
        y_true_all_bin, y_pred_all_scores = _flatten_for_roc(truth_c, preds_c, genes, variable, roc_threshold)
        
        if len(y_true_all_bin) == 0 or len(np.unique(y_true_all_bin)) < 2:
            text_blurbs.append(f'{cohort}: ROC AUC = N/A (insufficient data)')
            continue

        fpr_main, tpr_main, _ = roc_curve(y_true_all_bin, y_pred_all_scores)
        tpr_main_interp = np.interp(base_fpr, fpr_main, tpr_main)
        tpr_main_interp[0] = 0.0 
        auc_main = auc(fpr_main, tpr_main)

        # ---------- cluster bootstrap by *sample* ----------
        all_interp_tprs, aucs_bootstrap = [], []
        for _ in range(n_bootstraps):
            if not cohort_samples: break 
            bs_samples_indices = rng.choice(len(cohort_samples),
                                            size=len(cohort_samples),
                                            replace=True)
            bs_samples = [cohort_samples[i] for i in bs_samples_indices]
            
            bs_truth_sel = truth_df_processed[bs_samples]
            bs_preds_sel = preds_df_processed[bs_samples]

            y_t_bs_bin, y_p_bs_scores = _flatten_for_roc(bs_truth_sel, bs_preds_sel, genes, variable, roc_threshold)
            
            if len(y_t_bs_bin) == 0 or len(np.unique(y_t_bs_bin)) < 2: 
                continue

            fpr_b, tpr_b, _ = roc_curve(y_t_bs_bin, y_p_bs_scores)
            tpr_b_interp = np.interp(base_fpr, fpr_b, tpr_b)
            tpr_b_interp[0] = 0.0
            
            all_interp_tprs.append(tpr_b_interp)
            aucs_bootstrap.append(auc(fpr_b, tpr_b))

        if not all_interp_tprs: 
            mean_tpr = tpr_main_interp 
            lower_tpr = tpr_main_interp
            upper_tpr = tpr_main_interp
            auc_ci_low, auc_ci_high = auc_main, auc_main
        else:
            all_interp_tprs_np = np.array(all_interp_tprs) # Ensure it's a NumPy array for vstack if not already
            if all_interp_tprs_np.ndim == 1: # Handle case with only one successful bootstrap
                 all_interp_tprs_np = all_interp_tprs_np.reshape(1, -1)

            mean_tpr = np.mean(all_interp_tprs_np, axis=0)
            lower_tpr, upper_tpr = np.percentile(all_interp_tprs_np, [2.5, 97.5], axis=0)
            if aucs_bootstrap: 
                auc_ci_low, auc_ci_high = np.percentile(aucs_bootstrap, [2.5, 97.5])
            else: 
                auc_ci_low, auc_ci_high = auc_main, auc_main
        
        # ---------- plotting ----------
        color_to_use = None
        try:
            if 'subtype_to_color' in globals() and isinstance(globals()['subtype_to_color'], dict):
                color_to_use = globals()['subtype_to_color'].get(cohort)
        except NameError: # Should be caught by 'in globals()' but as a safeguard
            pass 
        
        if color_to_use is None: 
            prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_to_use = prop_cycle_colors[cohort_idx % len(prop_cycle_colors)]

        ax.plot(base_fpr, mean_tpr, lw=2, color=color_to_use, label=f'{cohort}')
        ax.fill_between(base_fpr, lower_tpr, upper_tpr, alpha=0.25, color=color_to_use)

        text_blurbs.append(
            f'{cohort}: ROC AUC = {auc_main:.2f} '
            f'({auc_ci_low:.2f}–{auc_ci_high:.2f})')

    # ---------- cosmetics ----------
    ax.plot([0, 1], [0, 1], ls=':', color='grey', lw=1, label='Chance') 
    ax.set(xlim=(0, 1), ylim=(0, 1.05),
           xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           title='ROC Curves by Cohort')
    ax.set_aspect('equal', adjustable='box')
    
    handles, labels = ax.get_legend_handles_labels()
    if handles: 
        ax.legend(handles=handles, labels=labels, title='Cohorts', loc='center left', bbox_to_anchor=(1.02, 0.5))

    ax.text(1.02, 0.95, '\n'.join(text_blurbs), 
            transform=ax.transAxes,
            va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
            fontsize=9)

    plt.savefig(f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # ---------- calculate and save per-sample Pearson correlations ----------
    sample_correlations = []
    
    for sample in common_samples:
        if genes is not None:  # gene set mode
            # Get truth and prediction values for this sample across all genes
            truth_sample = truth_df_processed[sample]
            preds_sample = preds_df_processed[sample]
            
            # Convert to numeric and remove NaN pairs
            truth_numeric = pd.to_numeric(truth_sample, errors='coerce')
            preds_numeric = pd.to_numeric(preds_sample, errors='coerce')
            
            # Create a mask for valid (non-NaN) pairs
            valid_mask = ~(pd.isna(truth_numeric) | pd.isna(preds_numeric))
            
            if valid_mask.sum() >= 3:  # Need at least 3 pairs for correlation
                truth_valid = truth_numeric[valid_mask]
                preds_valid = preds_numeric[valid_mask]
                
                # Calculate Pearson correlation
                correlation, p_value = pearsonr(truth_valid, preds_valid)
                
                sample_correlations.append({
                    'Sample': sample,
                    'Cohort': cohort_dict.get(sample, 'Unknown'),
                    'Pearson_r': correlation,
                    'p_value': p_value,
                    'n_genes': valid_mask.sum()
                })
            else:
                sample_correlations.append({
                    'Sample': sample,
                    'Cohort': cohort_dict.get(sample, 'Unknown'),
                    'Pearson_r': np.nan,
                    'p_value': np.nan,
                    'n_genes': valid_mask.sum()
                })
        
        elif variable is not None:  # single variable mode
            # For single variable mode, we can't calculate meaningful gene-set correlations
            # since there's only one variable per sample
            sample_correlations.append({
                'Sample': sample,
                'Cohort': cohort_dict.get(sample, 'Unknown'),
                'Pearson_r': np.nan,
                'p_value': np.nan,
                'n_genes': 1
            })
    
    # Convert to DataFrame and save as TSV
    if sample_correlations:
        correlations_df = pd.DataFrame(sample_correlations)
        correlations_df = correlations_df.sort_values(['Cohort', 'Sample'])
        
        # Save to TSV file
        tsv_filename = f'{name}_Sample-Pearsons.tsv'
        correlations_df.to_csv(tsv_filename, sep='\t', index=False)
        print(f"Saved per-sample Pearson correlations to {tsv_filename}")


def _calculate_all_hr_data(df, duration_col, event_col, variables_for_hr_calculation, transform_type):
    """
    Helper function to calculate hazard ratio data.
    
    Transform types:
    - 'zscore': Z-score normalization only (mean=0, std=1)
    - 'log2p1': Log2(x+1) transformation only  
    - 'log2p1_zscore': Log2(x+1) transformation followed by z-score normalization
    
    All transform types result in standardized variables for HR calculation.
    """
    cph = CoxPHFitter()
    all_hr_data = []

    for var in variables_for_hr_calculation:
        subset_cols = [var, duration_col, event_col]
        df_subset = df[subset_cols].dropna().copy()

        if df_subset.shape[0] < 10 or df_subset[var].nunique() < 2:
            print(f"Skipping HR calculation for {var} due to insufficient data (<10 obs) or variance (<2 unique values).")
            continue

        if transform_type == 'log2p1':
            if (df_subset[var] < -1).any():
                print(f"Warning: Variable '{var}' contains values < -1. Log2(x+1) transform might produce NaNs or complex numbers. Affected values will be NaNs.")
            transformed_var = np.log2(df_subset[var] + 1)
            transformed_var.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_subset[var] = transformed_var
            df_subset.dropna(subset=[var], inplace=True)

            if df_subset.shape[0] < 10 or df_subset[var].nunique() < 2:
                print(f"Skipping HR for {var} after log2p1 transform due to insufficient data or variance.")
                continue
        elif transform_type == 'log2p1_zscore':
            # First apply log2(x+1) transformation
            if (df_subset[var] < -1).any():
                print(f"Warning: Variable '{var}' contains values < -1. Log2(x+1) transform might produce NaNs or complex numbers. Affected values will be NaNs.")
            transformed_var = np.log2(df_subset[var] + 1)
            transformed_var.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_subset[var] = transformed_var
            df_subset.dropna(subset=[var], inplace=True)
            
            if df_subset.shape[0] < 10 or df_subset[var].nunique() < 2:
                print(f"Skipping HR for {var} after log2p1 transform due to insufficient data or variance.")
                continue
                
            # Then apply z-score transformation
            mean_val = df_subset[var].mean()
            std_val = df_subset[var].std()
            if std_val == 0 or pd.isna(std_val):
                print(f"Skipping HR calculation for {var} due to zero or NaN standard deviation after log2p1 (cannot z-score).")
                continue
            df_subset[var] = (df_subset[var] - mean_val) / std_val
        elif transform_type == 'zscore':
            mean_val = df_subset[var].mean()
            std_val = df_subset[var].std()
            if std_val == 0 or pd.isna(std_val):
                print(f"Skipping HR calculation for {var} due to zero or NaN standard deviation (cannot z-score).")
                continue
            df_subset[var] = (df_subset[var] - mean_val) / std_val
        else:
            raise ValueError(f"Unknown transform_type: {transform_type}")

        try:
            cph.fit(df_subset, duration_col=duration_col, event_col=event_col, formula=f"`{var}`")
            summary = cph.summary
            hr = summary.loc[var, 'exp(coef)']
            ci_lower = summary.loc[var, 'exp(coef) lower 95%']
            ci_upper = summary.loc[var, 'exp(coef) upper 95%']
            p_val = summary.loc[var, 'p']
            all_hr_data.append({'Variable': var, 'HR': hr, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper, 'p': p_val})
        except Exception as e:
            print(f"Could not fit Cox model for variable '{var}' (transform: {transform_type}): {e}")

    if not all_hr_data:
        return pd.DataFrame()
    return pd.DataFrame(all_hr_data)


def _calculate_multivariate_hr_data(df, duration_col, event_col, variables_list, transform_types_dict):
    """
    Helper function to calculate multivariate hazard ratio data.
    
    Args:
        df: DataFrame with survival data
        duration_col: Column name for duration
        event_col: Column name for events
        variables_list: List of variables to include in the multivariate model
        transform_types_dict: Dict mapping variable names to transform types ('log2p1', 'zscore', or None)
    """
    if not variables_list:
        return pd.DataFrame()
    
    # Prepare the subset with all variables
    subset_cols = variables_list + [duration_col, event_col]
    df_subset = df[subset_cols].dropna().copy()
    
    if df_subset.shape[0] < 10:
        print(f"Insufficient data for multivariate Cox model ({df_subset.shape[0]} observations after removing NaN).")
        return pd.DataFrame()
    
    # Apply transformations
    for var in variables_list:
        transform_type = transform_types_dict.get(var)
        
        if transform_type == 'log2p1':
            if (df_subset[var] < -1).any():
                print(f"Warning: Variable '{var}' contains values < -1. Log2(x+1) transform might produce NaNs.")
            transformed_var = np.log2(df_subset[var] + 1)
            transformed_var.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_subset[var] = transformed_var
            
        elif transform_type == 'log2p1_zscore':
            # First apply log2(x+1) transformation
            if (df_subset[var] < -1).any():
                print(f"Warning: Variable '{var}' contains values < -1. Log2(x+1) transform might produce NaNs.")
            transformed_var = np.log2(df_subset[var] + 1)
            transformed_var.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_subset[var] = transformed_var
            
            # Then apply z-score transformation after removing NaNs
            df_subset.dropna(subset=[var], inplace=True)
            if df_subset.shape[0] < 10:
                print(f"Insufficient data for variable '{var}' after log2p1 transform.")
                return pd.DataFrame()
                
            mean_val = df_subset[var].mean()
            std_val = df_subset[var].std()
            if std_val == 0 or pd.isna(std_val):
                print(f"Cannot z-score variable '{var}' due to zero/NaN standard deviation after log2p1.")
                return pd.DataFrame()
            df_subset[var] = (df_subset[var] - mean_val) / std_val
            
        elif transform_type == 'zscore':
            mean_val = df_subset[var].mean()
            std_val = df_subset[var].std()
            if std_val == 0 or pd.isna(std_val):
                print(f"Cannot z-score variable '{var}' due to zero/NaN standard deviation.")
                return pd.DataFrame()
            df_subset[var] = (df_subset[var] - mean_val) / std_val
    
    # Remove any rows with NaN after transformations
    df_subset.dropna(inplace=True)
    
    if df_subset.shape[0] < 10:
        print(f"Insufficient data for multivariate Cox model after transformations ({df_subset.shape[0]} observations).")
        return pd.DataFrame()
    
    # Check for sufficient variance in each variable
    for var in variables_list:
        if df_subset[var].nunique() < 2:
            print(f"Variable '{var}' has insufficient variance for multivariate model.")
            return pd.DataFrame()
    
    # Fit multivariate Cox model without penalization
    cph = CoxPHFitter()
    
    try:
        # Create formula with all variables
        formula_vars = [f"`{var}`" for var in variables_list]
        formula = " + ".join(formula_vars)
        
        cph.fit(df_subset, duration_col=duration_col, event_col=event_col, formula=formula)
        
        # Extract results
        summary = cph.summary
        all_hr_data = []
        
        for var in variables_list:
            if var in summary.index:
                hr = summary.loc[var, 'exp(coef)']
                ci_lower = summary.loc[var, 'exp(coef) lower 95%']
                ci_upper = summary.loc[var, 'exp(coef) upper 95%']
                p_val = summary.loc[var, 'p']
                all_hr_data.append({
                    'Variable': var, 
                    'HR': hr, 
                    'CI_Lower': ci_lower, 
                    'CI_Upper': ci_upper, 
                    'p': p_val
                })
        
        return pd.DataFrame(all_hr_data)
        
    except Exception as e:
        print(f"Could not fit multivariate Cox model: {e}")
        return pd.DataFrame()
    

def plot_hazard_ratios(hr_df_full, # Pre-calculated HR data
                       output_path,
                       plot_title,
                       xlabel_base,
                       vars_to_display=None,
                       top_n_to_display=0,
                       bottom_n_to_display=0,
                       fixed_x_lims=None # Tuple (min_hr, max_hr) for x-axis
                       ):
    """
    Plots univariate hazard ratios from pre-calculated data.
    The selection of which calculated HRs to plot is controlled by vars_to_display or top/bottom_n_to_display.
    """
    if hr_df_full.empty:
        print(f"No hazard ratio data provided for the '{plot_title}' plot. Skipping.")
        return

    hr_df_to_plot = pd.DataFrame()
    # Save the full HR data to a TSV file for reference
    hr_df_full.to_csv(output_path.replace('.pdf', '.tsv'), sep='\t', index=False)

    if vars_to_display is not None:
        hr_df_to_plot = hr_df_full[hr_df_full['Variable'].isin(vars_to_display)].copy()
    elif top_n_to_display > 0 or bottom_n_to_display > 0:
        temp_df = hr_df_full.copy()
        if not temp_df.empty:
            actual_top_n = min(top_n_to_display, len(temp_df))
            actual_bottom_n = min(bottom_n_to_display, len(temp_df))
            top_n_vars = temp_df.nlargest(actual_top_n, 'HR')['Variable'].tolist()
            bottom_n_vars = temp_df.nsmallest(actual_bottom_n, 'HR')['Variable'].tolist()
            selected_vars_set = set(top_n_vars) | set(bottom_n_vars)
            hr_df_to_plot = temp_df[temp_df['Variable'].isin(list(selected_vars_set))].copy()
    else:
        hr_df_to_plot = hr_df_full.copy()

    if hr_df_to_plot.empty:
        print(f"No variables selected for HR plot '{plot_title}' after filtering/selection.")
        return

    hr_df_to_plot.sort_values(by='HR', inplace=True)

    fig, ax = plt.subplots(figsize=(4, max(2.0, len(hr_df_to_plot) * 0.4)))

    y_pos = np.arange(len(hr_df_to_plot))
    xerr_lower = (hr_df_to_plot['HR'] - hr_df_to_plot['CI_Lower']).clip(lower=0)
    xerr_upper = (hr_df_to_plot['CI_Upper'] - hr_df_to_plot['HR']).clip(lower=0)
    
    ax.errorbar(hr_df_to_plot['HR'], y_pos, xerr=[xerr_lower, xerr_upper],
                fmt='o', color='black', capsize=5, label='Hazard Ratio (95% CI)')

    ax.axvline(1.0, color='gray', linestyle='--', lw=1, label='HR=1 (Reference)')
    ax.axvline(0.5, color='green', linestyle='--', lw=1, label='HR=0.5')
    ax.axvline(2.0, color='red', linestyle='--', lw=1, label='HR=2.0')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hr_df_to_plot['Variable'])
    ax.set_xlabel(f'Hazard Ratio ({xlabel_base}, Log Scale)')
    ax.set_xscale('log')
    if fixed_x_lims:
        ax.set_xlim(fixed_x_lims)

    ax.set_title(plot_title)
    ax.invert_yaxis() 
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Hazard Ratio plot to {output_path}")


def survival_analysis(df, duration_col, event_col, os_unit, name, keep_vars=None):
    """
    Main function to run survival analysis. Generates two HR plots and Kaplan-Meier plots.
    HR plots will have synchronized x-axis ranges.
    
    ALL VARIABLES ARE Z-SCORED FOR HAZARD RATIO CALCULATIONS:
    - Clinical variables (excluding TFX): log2(x+1) transformation THEN z-scoring
    - TFX variables: z-scoring only (raw values)  
    - Discovery variables: z-scoring only (raw values)

    Args:
        df (pd.DataFrame): DataFrame with patients as rows/indices.
                           Must contain duration and event columns.
                           Other columns are treated as variables.
        duration_col (str): Column name for survival duration (time).
        event_col (str): Column name indicating if the event occurred (1=event, 0=censored).
        os_unit (str): Units of survival time for plotting (e.g., "Months").
        name (str): Base path/name for output files. Output directory will be created.
        keep_vars (list, optional): List of variable names for the 'clinical' HR plot.
                                    These are log2(x+1) transformed then z-scored, except for 'TFX' which is z-scored only.
                                    Other variables will be considered for a 'discovery' HR plot (z-scored only).
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if duration_col not in df.columns:
        raise ValueError(f"Duration column '{duration_col}' not found in DataFrame.")
    if event_col not in df.columns:
        raise ValueError(f"Event column '{event_col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[duration_col]):
         raise ValueError(f"Duration column '{duration_col}' must be numeric.")
    if not pd.api.types.is_numeric_dtype(df[event_col]):
         raise ValueError(f"Event column '{event_col}' must be numeric (0 or 1).")
    if not set(df[event_col].dropna().unique()).issubset({0, 1}):
        print(f"Warning: Event column '{event_col}' contains values other than 0 and 1. Assuming 1=event, 0=censored.")

    # --- Setup ---
    output_dir = os.path.dirname(name)
    base_name = os.path.basename(name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    km_output_dir = os.path.join(output_dir, f"{base_name}_KM_plots")
    if not os.path.exists(km_output_dir):
        os.makedirs(km_output_dir)
        print(f"Created Kaplan-Meier plot directory: {km_output_dir}")

    all_numeric_variables = [col for col in df.columns if col not in [duration_col, event_col] and pd.api.types.is_numeric_dtype(df[col])]
    if not all_numeric_variables:
        print("No numeric variable columns found for analysis (excluding duration/event cols).")
        return
    print(f"Found {len(all_numeric_variables)} total numeric variables for potential analysis.")

    # --- Calculate HR data for clinical variables with log2+zscore transformations ---
    clinical_hr_df_full = pd.DataFrame()
    valid_clinical_vars = []
    if keep_vars and isinstance(keep_vars, list) and len(keep_vars) > 0:
        valid_clinical_vars = [v for v in keep_vars if v in all_numeric_variables]
        if valid_clinical_vars:
            print(f"Calculating Clinical Hazard Ratios for: {', '.join(valid_clinical_vars)}")
            
            # Calculate HRs for clinical variables with log2+zscore transformations
            clinical_hr_data = []
            
            # Group clinical variables by transformation type
            tfx_vars = [v for v in valid_clinical_vars if v.upper() == 'TFX']
            log2p1_vars = [v for v in valid_clinical_vars if v.upper() != 'TFX']
            
            # Calculate HRs for TFX (z-scored only)
            if tfx_vars:
                print(f"  - Z-scoring TFX variables (raw values): {', '.join(tfx_vars)}")
                tfx_hr_df = _calculate_all_hr_data(df.copy(), duration_col, event_col, tfx_vars, 'zscore')
                if not tfx_hr_df.empty:
                    clinical_hr_data.append(tfx_hr_df)
            
            # Calculate HRs for other clinical variables (log2p1 then zscore)
            if log2p1_vars:
                print(f"  - Log2(x+1) then Z-scoring other clinical variables: {', '.join(log2p1_vars)}")
                log2p1_zscore_hr_df = _calculate_all_hr_data(df.copy(), duration_col, event_col, log2p1_vars, 'log2p1_zscore')
                if not log2p1_zscore_hr_df.empty:
                    clinical_hr_data.append(log2p1_zscore_hr_df)
            
            # Combine clinical HR results
            if clinical_hr_data:
                clinical_hr_df_full = pd.concat(clinical_hr_data, ignore_index=True)
        else:
            print("No valid 'keep_vars' found in DataFrame's numeric columns for clinical HR plot.")
    else:
        print("No 'keep_vars' provided or empty list, skipping clinical HR calculations.")

    # --- Calculate HR data for discovery variables (all z-scored) ---
    discovery_hr_df_full = pd.DataFrame()
    discovery_vars_list = [var for var in all_numeric_variables if not (keep_vars and var in keep_vars)]
    if discovery_vars_list:
        print(f"Calculating Discovery Hazard Ratios from {len(discovery_vars_list)} variables (z-scored, excluding clinical).")
        discovery_hr_df_full = _calculate_all_hr_data(df.copy(), duration_col, event_col,
                                                      discovery_vars_list, 'zscore')
    else:
        print("No remaining variables for discovery HR calculations after excluding keep_vars (if any).")

    # --- Determine global x-axis limits for HR plots ---
    min_ci_val = np.inf
    max_ci_val = -np.inf
    has_hr_data_for_xlim = False

    if not clinical_hr_df_full.empty and 'CI_Lower' in clinical_hr_df_full.columns:
        min_ci_val = min(min_ci_val, clinical_hr_df_full['CI_Lower'].min())
        max_ci_val = max(max_ci_val, clinical_hr_df_full['CI_Upper'].max())
        has_hr_data_for_xlim = True
    
    if not discovery_hr_df_full.empty and 'CI_Lower' in discovery_hr_df_full.columns:
        min_ci_val = min(min_ci_val, discovery_hr_df_full['CI_Lower'].min())
        max_ci_val = max(max_ci_val, discovery_hr_df_full['CI_Upper'].max())
        has_hr_data_for_xlim = True

    shared_hr_plot_xlims = None
    if has_hr_data_for_xlim and min_ci_val > 0 and max_ci_val > 0 and min_ci_val < max_ci_val :
        lower_bound = max(0.01, min_ci_val / 1.5)
        upper_bound = max_ci_val * 1.5
        if lower_bound < upper_bound:
             shared_hr_plot_xlims = (lower_bound, upper_bound)
        else:
             shared_hr_plot_xlims = (max(0.01, min_ci_val * 0.9), max_ci_val * 1.1)

    # --- Generate Plots ---

    # 1. Clinical Hazard Ratio Plot (with log2+zscore transformation labels)
    if not clinical_hr_df_full.empty:
        clinical_hr_plot_path = os.path.join(output_dir, f"{base_name}_clinical_hazard_ratios.pdf")
        
        # Create appropriate xlabel for log2+zscore transformations
        tfx_vars = [v for v in valid_clinical_vars if v.upper() == 'TFX']
        log2p1_vars = [v for v in valid_clinical_vars if v.upper() != 'TFX']
        
        if tfx_vars and log2p1_vars:
            xlabel_text = 'per 1 SD change [TFX: z-scored; others: log2(x+1) then z-scored]'
        elif tfx_vars:
            xlabel_text = 'per 1 SD change [z-scored]'
        else:
            xlabel_text = 'per 1 SD change [log2(x+1) then z-scored]'
            
        plot_hazard_ratios(hr_df_full=clinical_hr_df_full,
                           output_path=clinical_hr_plot_path,
                           plot_title='Clinical Univariate Hazard Ratios',
                           xlabel_base=xlabel_text,
                           vars_to_display=valid_clinical_vars,
                           fixed_x_lims=shared_hr_plot_xlims)
    elif keep_vars and isinstance(keep_vars, list) and len(keep_vars) > 0 and valid_clinical_vars:
        print("Clinical HR plot skipped as no HR data was generated.")

    # 2. Discovery Hazard Ratio Plot
    if not discovery_hr_df_full.empty:
        discovery_hr_plot_path = os.path.join(output_dir, f"{base_name}_discovery_hazard_ratios.pdf")
        plot_hazard_ratios(hr_df_full=discovery_hr_df_full,
                           output_path=discovery_hr_plot_path,
                           plot_title='Discovery Univariate Hazard Ratios (Top/Bottom 5)',
                           xlabel_base='per 1 SD change [z-scored]',
                           top_n_to_display=5,
                           bottom_n_to_display=5,
                           fixed_x_lims=shared_hr_plot_xlims)
    elif discovery_vars_list:
        print("Discovery HR plot skipped as no HR data was generated.")

    # --- Select variables with p < 0.05 from univariate analyses for multivariate ---
    multivariate_hr_df_full = pd.DataFrame()
    significant_vars_for_multivariate = []
    transform_types_dict = {}
    
    # Get significant clinical variables (p < 0.05) with appropriate transformations
    if not clinical_hr_df_full.empty:
        significant_clinical = clinical_hr_df_full[clinical_hr_df_full['p'] < 0.05]['Variable'].tolist()
        significant_vars_for_multivariate.extend(significant_clinical)
        
        # Set transformation types for clinical variables
        for var in significant_clinical:
            if var.upper() == 'TFX':
                transform_types_dict[var] = 'zscore'
            else:
                transform_types_dict[var] = 'log2p1_zscore'
        
        print(f"Found {len(significant_clinical)} significant clinical variables (p < 0.05): {', '.join(significant_clinical)}")
    
    # Get significant discovery variables (p < 0.05)
    if not discovery_hr_df_full.empty:
        significant_discovery = discovery_hr_df_full[discovery_hr_df_full['p'] < 0.05]['Variable'].tolist()
        significant_vars_for_multivariate.extend(significant_discovery)
        for var in significant_discovery:
            transform_types_dict[var] = 'zscore'
        print(f"Found {len(significant_discovery)} significant discovery variables (p < 0.05): {', '.join(significant_discovery)}")
    
    # Calculate multivariate model with significant variables only
    if significant_vars_for_multivariate:
        print(f"Calculating Multivariate Hazard Ratios for {len(significant_vars_for_multivariate)} significant variables (p < 0.05 in univariate).")
        multivariate_hr_df_full = _calculate_multivariate_hr_data(
            df.copy(), duration_col, event_col, 
            significant_vars_for_multivariate, transform_types_dict
        )
    else:
        print("No variables with p < 0.05 found in univariate analyses for multivariate modeling.")

    # --- Update x-axis limits to include multivariate data ---
    if not multivariate_hr_df_full.empty and 'CI_Lower' in multivariate_hr_df_full.columns:
        min_ci_val = min(min_ci_val, multivariate_hr_df_full['CI_Lower'].min())
        max_ci_val = max(max_ci_val, multivariate_hr_df_full['CI_Upper'].max())
        has_hr_data_for_xlim = True
    
    # 3. Multivariate Hazard Ratio Plot
    if not multivariate_hr_df_full.empty:
        multivariate_hr_plot_path = os.path.join(output_dir, f"{base_name}_multivariate_hazard_ratios.pdf")
        
        # Create appropriate xlabel based on transformation types
        significant_tfx = [var for var in significant_vars_for_multivariate if transform_types_dict.get(var) == 'zscore' and var.upper() == 'TFX']
        significant_clinical_log2p1_zscore = [var for var in significant_vars_for_multivariate if transform_types_dict.get(var) == 'log2p1_zscore']
        significant_discovery_zscore = [var for var in significant_vars_for_multivariate if transform_types_dict.get(var) == 'zscore' and var.upper() != 'TFX']
        
        # Create appropriate xlabel - all variables are z-scored, but with different pre-processing
        xlabel_parts = []
        if significant_clinical_log2p1_zscore:
            xlabel_parts.append('clinical: log2(x+1) then z-scored')
        if significant_tfx:
            xlabel_parts.append('TFX: z-scored')
        if significant_discovery_zscore:
            xlabel_parts.append('discovery: z-scored')
            
        if len(xlabel_parts) > 1:
            xlabel_text = f'per 1 SD change [{"; ".join(xlabel_parts)}]'
        elif significant_clinical_log2p1_zscore:
            xlabel_text = 'per 1 SD change [log2(x+1) then z-scored]'
        else:
            xlabel_text = 'per 1 SD change [z-scored]'
            
        plot_hazard_ratios(hr_df_full=multivariate_hr_df_full,
                           output_path=multivariate_hr_plot_path,
                           plot_title=f'Multivariate Hazard Ratios (n={len(significant_vars_for_multivariate)} vars with p<0.05)',
                           xlabel_base=xlabel_text,
                           vars_to_display=None,
                           fixed_x_lims=None)
    elif significant_vars_for_multivariate:
        print("Multivariate HR plot skipped as no HR data was generated.")

    # 4. Kaplan-Meier Plots (one per variable, using all_numeric_variables)
    print(f"\nGenerating Kaplan-Meier plots in {km_output_dir} for all {len(all_numeric_variables)} numeric variables...")
    for var in all_numeric_variables:
        km_plot_path = os.path.join(km_output_dir, f"{base_name}_{var}_KM.pdf")
        plot_kaplan_meier(df.copy(), duration_col, event_col, var, os_unit, km_plot_path)
    print("Kaplan-Meier plot generation complete.")

    print("\nSurvival analysis finished.")


def plot_kaplan_meier(df, duration_col, event_col, variable_col, os_unit, output_path):
    """
    Generates and saves a Kaplan-Meier plot for a given variable.

    Args:
        df (pd.DataFrame): DataFrame containing survival data and the variable.
        duration_col (str): Column name for survival duration.
        event_col (str): Column name for event occurrence (1=event, 0=censored).
        variable_col (str): Column name for the variable to stratify by.
                            The variable will be dichotomized by its median.
        os_unit (str): Units for the time axis label (e.g., "Months").
        output_path (str): Path to save the generated plot.
    """
    data_subset = df[[variable_col, duration_col, event_col]].copy()
    data_subset.dropna(inplace=True)

    if data_subset.empty or data_subset[variable_col].nunique() < 2 or len(data_subset) < 10:
        print(f"Skipping Kaplan-Meier plot for '{variable_col}' due to insufficient data (n={len(data_subset)}, unique_var_values={data_subset[variable_col].nunique()}).")
        return

    median_val = data_subset[variable_col].median()
    
    # Handle cases where median split results in only one group due to data distribution
    if data_subset[variable_col].min() == median_val and data_subset[variable_col].max() == median_val:
         print(f"Skipping Kaplan-Meier plot for '{variable_col}' as all values are identical to the median, cannot stratify.")
         return
    if (data_subset[variable_col] < median_val).sum() == 0 or (data_subset[variable_col] >= median_val).sum() == 0 :
        print(f"Skipping Kaplan-Meier plot for '{variable_col}' as median split results in only one group.")
        return

    data_subset['group'] = np.where(data_subset[variable_col] >= median_val, f'{variable_col}_High', f'{variable_col}_Low')

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(8, 6))

    group_high_data = data_subset[data_subset['group'] == f'{variable_col}_High']
    group_low_data = data_subset[data_subset['group'] == f'{variable_col}_Low']

    if len(group_high_data) < 5 or len(group_low_data) < 5:
        print(f"Skipping Kaplan-Meier plot for '{variable_col}' due to insufficient samples in one or both groups after median split (High: {len(group_high_data)}, Low: {len(group_low_data)}).")
        plt.close(fig)
        return
        
    kmf_high.fit(group_high_data[duration_col], 
                 event_observed=group_high_data[event_col], 
                 label=f'High (≥ {median_val:.2f}, n={len(group_high_data)})')
    kmf_high.plot_survival_function(ax=ax)

    kmf_low.fit(group_low_data[duration_col], 
                event_observed=group_low_data[event_col], 
                label=f'Low (< {median_val:.2f}, n={len(group_low_data)})')
    kmf_low.plot_survival_function(ax=ax)

    try:
        logrank_results = logrank_test(
            group_high_data[duration_col], group_low_data[duration_col],
            event_observed_A=group_high_data[event_col], event_observed_B=group_low_data[event_col]
        )
        p_value_text = format_pval(logrank_results.p_value)
        ax.text(0.05, 0.05, f'Log-rank {p_value_text}', transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    except Exception as e:
        print(f"Could not perform logrank test for {variable_col}: {e}")
        ax.text(0.05, 0.05, 'Log-rank: N/A', transform=ax.transAxes, fontsize=10)


    add_at_risk_counts(kmf_high, kmf_low, ax=ax)
    
    plt.title(f'Kaplan-Meier Plot for {variable_col}')
    plt.xlabel(f'Time ({os_unit})')
    plt.ylabel('Survival Probability')
    plt.ylim(0, 1.05)
    plt.legend(loc='best', title=f'{variable_col} Groups')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
