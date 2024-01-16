"""
Proteus.py
v1.4.1, 1/9/2024
Author: Robert D Patton, rpatton@fredhutch.org
with help from: Alex Netzley
Ha Lab, Fred Hutchinson Cancer Research Center

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/app/software/CUDA/12.0.0
"""

# Standard library imports
import os
import csv
import argparse

# Third party imports
import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

# Create a dictionary that maps each alias to the corresponding gene name
gene_aliases = {
    'Rhombotin-3': 'LM03',
    'RBTN3': 'LM03',
    'RHOM3': 'LM03',
    'DAT1': 'LM03',
    'LMO-3': 'LM03',
    'CD3G': 'TARP',
    'TCRG': 'TARP',
    'TCRGC1': 'TARP',
    'TCRGC2': 'TARP',
    'TCRGV': 'TARP',
    'TCR': 'TARP'
}

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_heatmap(phenotyping_genes, W_df):
    # Replace the aliases in W_df's index with the corresponding gene names
    W_df.index = W_df.index.map(lambda x: gene_aliases.get(x, x))

    # Filter the list to include only the genes that are present in W_df
    phenotyping_genes = [gene for gene in phenotyping_genes if gene in W_df.index]

    # Filter the DataFrame to include only the genes in the list
    filtered_df = W_df.loc[phenotyping_genes]

    # Sort the DataFrame by column names
    filtered_df = filtered_df.sort_index(axis=1)

    # Create a custom colormap
    colors = ["darkblue", "lightblue", "yellow", "orange", "red", "darkred"]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(20, 10))  # Increase figure size
    sns.heatmap(filtered_df, cmap=cmap, ax=ax, linewidths=0)

    # Draw horizontal lines at the specified row numbers
    ax.hlines([8, 14, 22], *ax.get_xlim(), color='white', linewidth=2)

    # Set the labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)  # Decrease font size
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Set the colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Log2(TPM +1)")

    # Save the figure to a PDF
    plt.savefig('heatmap.pdf')

from sklearn.decomposition import NMF


def perform_deconvolution(all_predictions_df, psma_tfx_subtype_dict, beta_loss='frobenius'):
    # Store the indices of the NaN values
    nan_indices = pd.isnull(all_predictions_df)

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(all_predictions_df), 
                              index=all_predictions_df.index, 
                              columns=all_predictions_df.columns)

    # Initialize DataFrames to store the profiles
    profiles_of_interest = pd.DataFrame(index=all_predictions_df.index)
    background_profiles = []

    # Apply NMF to each sample individually
    for sample in all_predictions_df.columns:
        nmf = NMF(n_components=2, init='random', random_state=0, beta_loss=beta_loss)
        W = nmf.fit_transform(df_imputed[[sample]])
        H = nmf.components_

        # Convert to DataFrames
        W_df = pd.DataFrame(W, index=all_predictions_df.index, columns=['Profile of Interest', 'Background Profile'])
        H_df = pd.DataFrame(H, columns=[sample], index=['Profile of Interest', 'Background Profile'])

        # Determine which component corresponds to the "profile of interest" and which corresponds to the "background profile"
        if H_df[sample]['Profile of Interest'] / H_df[sample]['Background Profile'] < float(psma_tfx_subtype_dict[sample]):
            # Swap the profiles for this sample
            H_df[sample] = H_df[sample][::-1]
            W_df = W_df[['Background Profile', 'Profile of Interest']]

        # Store the profiles
        profiles_of_interest[sample] = W_df['Profile of Interest']
        background_profiles.append(W_df['Background Profile'])

    # Reapply the NaNs to the original positions
    profiles_of_interest[nan_indices] = np.nan

    return profiles_of_interest, background_profiles


def main():
    # Load data
    all_predictions_df = pd.read_csv('all_predictions.tsv', sep='\t', index_col=0)
    with open('PSMA_TFX_Subtype.tsv', 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        psma_tfx_subtype_dict = {rows[0]: rows[1] for rows in reader}
    with open('reference/PhenotypingGenes.txt', 'r') as file:
        phenotyping_genes = [line.strip() for line in file]

    # Perform deconvolution
    W_df, H_df = perform_deconvolution(all_predictions_df, psma_tfx_subtype_dict)

    # Plot the heatmap
    plot_heatmap(phenotyping_genes, W_df)


if __name__ == "__main__":
    main()
