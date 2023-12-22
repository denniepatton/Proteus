"""
process_raw_counts.py
v1.0.0, 12/14/2023
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script converts raw count data from LuCaPs into a format that can be used by the CNN.
Namely, gene level Log2(TPM + 1) for direct regression and gene level active z-score for binary classification.
"""

# Standard library imports
import os

# Third party imports
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def plot_heatmap(df, categories, gene_list, output_file):
    """
    Plots a heatmap of gene expression data.

    Parameters:
    df (pandas.DataFrame): DataFrame containing gene expression data.
    categories (dict): Dictionary mapping sample names to categories.
    gene_list (list): List of genes to include in the heatmap.
    output_file (str): Path to the output file (without file extension).

    Returns:
    None
    """
    # Filter the DataFrame to include only valid samples and specified genes
    valid_samples = [sample for sample in categories.keys() if sample in df.columns]
    df = df.loc[gene_list, valid_samples]

    # Sort the DataFrame columns by the categories
    sorted_columns = sorted(df.columns, key=lambda x: categories[x])
    df = df[sorted_columns]

    # Determine the colormap for the heatmap based on the data
    vmin, vmax = df.min().min(), df.max().max()
    vcenter = (vmin + vmax) / 2
    is_binary = df.isin([0, 1]).all().all()

    cmap = plt.get_cmap('bwr') if is_binary else plt.get_cmap('RdBu_r')
    norm = None if is_binary else TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    # Set color for NaN values
    cmap.set_bad(color='gray')

    # Plot the heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(df, cmap=cmap, norm=norm, cbar_kws={'label': 'Expression'}, yticklabels=True)

    # Save the plot to a PDF file
    plt.savefig(f'{output_file}.pdf')

def active_z_score(df):
    """
    Calculates the active z-scores for each column in the DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing gene expression data.

    Returns:
    df_z (pandas.DataFrame): DataFrame with the same structure as df, but with the original values replaced by z-scores.
    """
    # Create a new directory if it doesn't exist
    if not os.path.exists('sample-level_AZ-Score_fits'):
        os.makedirs('sample-level_AZ-Score_fits')

    # Copy the DataFrame to avoid modifying the original data
    df_z = df.copy()

    # Iterate over each column in the DataFrame
    for column in df_z.columns:
        # Estimate the kernel density
        kernel = stats.gaussian_kde(df_z[column].dropna())

        # Find the value that maximizes the kernel density estimate
        x = np.linspace(min(df_z[column]), max(df_z[column]), 1000)
        density = kernel.evaluate(x)
        max_density_value = x[np.argmax(density)]
        median = max_density_value

        # Mirror the values greater than or equal to the median
        values = df_z[column][df_z[column] >= median]
        mirrored_values = median - (values - median)

        # Combine the original and mirrored values
        combined_values = np.concatenate([values, mirrored_values])

        # Find the mean and stdev of the combined values
        mean = median
        std_dev = np.std(combined_values)

        # Calculate the z-scores based off the active Gaussian distribution
        z_scores = (df_z[column] - mean) / std_dev

        # Replace the original column values with the z-scores
        df_z[column] = z_scores

        # Plot the original density of the values and the fitted Gaussian
        plt.figure(figsize=(10, 6))
        kde = sns.kdeplot(df[column], label='Original')

        # Get the maximum y-value of the KDE plot
        y_values = kde.get_lines()[0].get_data()[1]
        max_y = np.max(y_values)

        # Generate the fitted Gaussian
        x = np.linspace(min(df[column]), max(df[column]), 100)
        fitted_gaussian = stats.norm.pdf(x, mean, std_dev)

        # Normalize the fitted Gaussian to the maximum y-value of the KDE plot
        fitted_gaussian *= max_y / np.max(fitted_gaussian)

        plt.plot(x, fitted_gaussian, label='Fitted Gaussian (active)')
        plt.axvline(median, color='black', linestyle='--', label='Kernel Maximum, AZ-Score == 0')
        plt.axvline((median - 3 * std_dev), color='red', linestyle='--', label='Active Threshold, AZ-Score == -3')

        # Save the plot as a PDF
        plt.legend()
        plt.title(f'Value Density and Gaussian Fit for {column}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.savefig(f'sample-level_AZ-Score_fits/{column}.pdf')
        plt.close()

    return df_z



def plot_samples(df, filename, log_x=False, log_y=False):
    """
    Plots kernel density estimates and histograms for each column in the DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing gene expression data.
    filename (str): Name of the file to be used in the x-axis label.
    log_x (bool): If True, sets the x-axis scale to logarithmic. Default is False.
    log_y (bool): If True, sets the y-axis scale to logarithmic. Default is False.

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

    # If log_x or log_y is True, set the x and/or y scale to log
    if log_x:
        ax_kde.set_xscale('log')
        ax_hist.set_xscale('log')
    if log_y:
        ax_kde.set_yscale('log')
        ax_hist.set_yscale('log')

    # Shrink the labels
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    # Add a legend, title, and labels to the plots
    ax_kde.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 6})
    ax_kde.set_title('Kernel Density Plots')
    ax_kde.set_xlabel(filename)
    ax_kde.set_ylabel('Density')

    ax_hist.legend(loc="upper left", bbox_to_anchor=(1, 1), prop={'size': 6})
    ax_hist.set_title('Histograms')
    ax_hist.set_xlabel(filename)
    ax_hist.set_ylabel('Density')

    # Save the plots as PDF files
    plt.figure(fig_kde.number)
    plt.savefig(filename + "_kernels.pdf", bbox_inches='tight')

    plt.figure(fig_hist.number)
    plt.savefig(filename + "_histograms.pdf", bbox_inches='tight')

def convert_to_tpm(df):
    """
    Converts raw count data to Transcripts Per Million (TPM).

    Parameters:
    df (pandas.DataFrame): DataFrame containing raw count data.

    Returns:
    df_tpm (pandas.DataFrame): DataFrame with the same structure as df, but with the original values replaced by TPM.
    """
    # Copy the DataFrame to avoid modifying the original data
    df_tpm = df.copy()

    # Divide each value by the corresponding value in the 'mRNA' column and multiply by 1000
    df_tpm.loc[:, df_tpm.columns != 'mRNA_size'] = df_tpm.loc[:, df_tpm.columns != 'mRNA_size'].div(df_tpm['mRNA_size'], axis=0)
    df_tpm.loc[:, df_tpm.columns != 'mRNA_size'] *= 1000

    # Drop the 'mRNA' column
    df_tpm.drop(columns='mRNA_size', inplace=True)

    # Calculate the sample level scaling factor for each column
    scaling_factors = df_tpm.sum() / 1e6

    # Divide each value by the corresponding sample level scaling factor
    df_tpm /= scaling_factors

    return df_tpm

def main():
    # Define paths and parameters
    raw_data = 'LuCaP_PDX_125_RNAseq_raw_counts.txt'
    data_labels = 'LuCaP_PDX_125_RNAseq_phenodata.txt'
    with open('../reference/AR12.txt', 'r') as f:
        ar_12 = f.read().splitlines()
    with open('../reference/NE35.txt', 'r') as f:
        ne_35 = f.read().splitlines()
    with open('../reference/Targets.txt', 'r') as f:
        targets = f.read().splitlines()
    categories = pd.read_table('../reference/LuCaP_phenotype.tsv', sep='\t', header=None)
    categories = dict(zip(categories[0], categories[1]))

    # Read in raw count data and labels
    df = pd.read_csv(raw_data, sep='\t', index_col=0)
    df_labels = pd.read_csv(data_labels, sep='\t', index_col=0)

    # Replace the current index with the values in "Gene_symbol"
    df.set_index('Gene_symbol', inplace=True)
    # Remove the column 'Gene_name'
    df.drop(columns='Gene_name', inplace=True)

    # Create a mapping from the 'sample' column to the 'line' column in df_labels
    column_mapping = df_labels.set_index('sample')['line'].to_dict()
    # Replace '.' with '-' in the mapping
    column_mapping = {k: v.replace('.', '-') for k, v in column_mapping.items()}

    # Replace the column names in df using the mapping
    df.columns = df.columns.map(lambda x: column_mapping[x] if x != 'mRNA_size' else x)

    # Group by the new column names and sum the values
    df = df.groupby(df.columns, axis=1).sum()

    # Remove sample 235-3, which appears to be an outlier
    df.drop(columns='LuCaP_235-3', inplace=True)

    # Convert raw counts to TPM
    df_tpm = convert_to_tpm(df)
    df_tpm.to_csv('LuCaP_TPM.tsv', sep='\t')
    print('range of LuCaP_TPM: ', df_tpm.min().min(), df_tpm.max().max())
    plot_samples(np.log2(df_tpm), "LuCaP_Log2TPM")

    # Apply the log2 + 1 transformation
    df_log2tpm = np.log2(df_tpm + 1)
    df_log2tpm.to_csv('LuCaP_Log2TPMp1.tsv', sep='\t')
    print('range of LuCaP_Log2TPMp1: ', df_log2tpm.min().min(), df_log2tpm.max().max())
    plot_samples(df_log2tpm, "LuCaP_Log2TPMp1")
    plot_heatmap(df_log2tpm, categories, targets, 'LuCaP_Log2TPMp1_Targets_HeatMap')
    plot_heatmap(df_log2tpm, categories, ar_12, 'LuCaP_Log2TPMp1_AR12_HeatMap')
    plot_heatmap(df_log2tpm, categories, ne_35, 'LuCaP_Log2TPMp1_NE35_HeatMap')

    # Apply the active z-score transformation
    df_log2tpm = np.log2(df_tpm).replace(-np.inf, np.nan)
    df_log2tpm_z = active_z_score(df_log2tpm)
    df_log2tpm_z.to_csv('LuCaP_Log2TPM_AZ-Score.tsv', sep='\t')
    print('range of LuCaP_Log2TPM_AZ-Score: ', df_log2tpm_z.min().min(), df_log2tpm_z.max().max())
    plot_samples(df_log2tpm_z, "LuCaP_Log2TPM_AZ-Score")
    plot_heatmap(df_log2tpm_z, categories, targets, 'LuCaP_Log2TPM_AZ-Score_Targets_HeatMap')
    plot_heatmap(df_log2tpm_z, categories, ar_12, 'LuCaP_Log2TPM_AZ-Score_AR12_HeatMap')
    plot_heatmap(df_log2tpm_z, categories, ne_35, 'LuCaP_Log2TPM_AZ-Score_NE35_HeatMap')

    # produce a mask of df_log2tpm where all values <= 1 are 0 and all others are 1
    # df_log2tpm_bin = df_log2tpm.copy()
    # df_log2tpm_bin[df_log2tpm_bin <= 1] = 0
    # df_log2tpm_bin[df_log2tpm_bin > 1] = 1
    # df_log2tpm_bin = df_log2tpm_bin.fillna(0)
    # plot_heatmap(df_log2tpm_bin, categories, targets, palette, 'LuCaP_Log2TPM_1Binary_Targets_HeatMap')
    # plot_heatmap(df_log2tpm_bin, categories, ar_12, palette, 'LuCaP_Log2TPM_1Binary_AR12_HeatMap')
    # plot_heatmap(df_log2tpm_bin, categories, ne_35, palette, 'LuCaP_Log2TPM_1Binary_NE35_HeatMap')
    # df_log2tpm_bin.to_csv('LuCaP_Log2TPM_1Binary.tsv', sep='\t')

    # produce a mask of df_log2tpm_z where all values <= -3 are 0 and all others are 1
    # df_log2tpm_z_bin = df_log2tpm_z.copy()
    # df_log2tpm_z_bin[df_log2tpm_z_bin <= -3] = 0
    # df_log2tpm_z_bin[df_log2tpm_z_bin > -3] = 1
    # df_log2tpm_z_bin = df_log2tpm_z_bin.fillna(0)
    # plot_heatmap(df_log2tpm_z_bin, categories, targets, palette, 'LuCaP_Log2TPM_AZ-Score_-3Binary_Targets_HeatMap')
    # plot_heatmap(df_log2tpm_z_bin, categories, ar_12, palette, 'LuCaP_Log2TPM_AZ-Score_-3Binary_AR12_HeatMap')
    # plot_heatmap(df_log2tpm_z_bin, categories, ne_35, palette, 'LuCaP_Log2TPM_AZ-Score_-3Binary_NE35_HeatMap')
    # df_log2tpm_z_bin.to_csv('LuCaP_Log2TPM_AZ-Score_-3Binary.tsv', sep='\t')

    # Produce a mask of df_log2tpm_z where all values <= 0 are 0 and all others are 1
    df_log2tpm_z_bin = df_log2tpm_z.copy()
    df_log2tpm_z_bin[df_log2tpm_z_bin <= 0] = 0
    df_log2tpm_z_bin[df_log2tpm_z_bin > 0] = 1
    df_log2tpm_z_bin = df_log2tpm_z_bin.fillna(0)
    # plot_heatmap(df_log2tpm_z_bin, categories, targets, palette, 'LuCaP_Log2TPM_AZ-Score_0Binary_Targets_HeatMap')
    # plot_heatmap(df_log2tpm_z_bin, categories, ar_12, palette, 'LuCaP_Log2TPM_AZ-Score_0Binary_AR12_HeatMap')
    # plot_heatmap(df_log2tpm_z_bin, categories, ne_35, palette, 'LuCaP_Log2TPM_AZ-Score_0Binary_NE35_HeatMap')
    df_log2tpm_z_bin.to_csv('LuCaP_Log2TPM_AZ-Score_0Binary.tsv', sep='\t')




if __name__ == "__main__":
    main()