"""
plot_utils.py
v1.0.1, 12/15/2023
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This file contains utility functions for visualizing models/results.
"""

# Standard library imports
import os

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc

# Keras imports
import keras


def plot_model(model, output, filename = "architecture"):
    keras.utils.plot_model(model, to_file = os.path.join(output, f"{filename}.png"), show_shapes=True, show_layer_activations=True)


def plot_convergence(history, output, filename = "model"):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('categorical accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output, f"convergence_acc_{filename}.pdf"))
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output, f"convergence_loss_{filename}.pdf"))
    plt.close()


def plot_convergence_regression(history, output, filename = "model"):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(output, f"convergence_mse_{filename}.pdf"))
    plt.close()


def plot_residuals(residuals, output):
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins='auto')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of residuals')
    plt.show()
    plt.savefig(os.path.join(output, "residuals.pdf"))
    plt.close() 


def roc_plot(data, output, filename, curve_names=None):
    plt.figure()

    for i, (truths, predictions) in enumerate(data):
        fpr, tpr, _ = roc_curve(truths, predictions)
        roc_auc = auc(fpr, tpr)
        if curve_names is not None:
            plt.plot(fpr, tpr, color=plt.cm.tab10(i), lw=2, label=f'ROC curve for {curve_names[i]} (area = {roc_auc:.2f})')
        else:
            plt.plot(fpr, tpr, color=plt.cm.tab10(i), lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output, filename + ".pdf"))
    plt.close()


def subset_roc_plot(data, output, filename, subsets):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Calculate and plot the ROC curve for all genes
    fpr, tpr, _ = roc_curve(data['true_label'], data['prediction'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'All genes (area = {roc_auc:.2f}, n = {len(data)})')

    # Iterate over each key-value pair in the subsets dictionary
    for subset_name, genes in subsets.items():
        # Find the intersection of the genes in the DataFrame and the genes in the subset
        common_genes = np.intersect1d(data.index, genes)
        # Subset the data DataFrame to include only the genes in the current list
        subset_data = data.loc[common_genes]

        # Calculate and plot the ROC curve for the subsetted data
        fpr, tpr, _ = roc_curve(subset_data['true_label'], subset_data['prediction'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{subset_name} (area = {roc_auc:.2f}, n = {len(subset_data)} / {len(genes)})')

    # Add a legend, title, and labels to the plot
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    # Save the plot as a PDF file
    plt.savefig(os.path.join(output, filename + ".pdf"))
    plt.close(fig)


def scatter_plot(y_true, y_pred, output, filename, title):
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)

    # Create scatter plot
    plt.figure(figsize=(8, 6))

    # Calculate the point densities
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)

    plt.scatter(y_true, y_pred, c=z, norm=LogNorm(), alpha=0.5)
    plt.xlabel('True Values in Log2(TPM + 1)')
    plt.ylabel('Predicted Values in Log2(TPM + 1)')
    plt.title(title)
    plt.colorbar(label='Density')

    # Print metrics on the plot
    plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR-squared: {r2:.2f}\nPCC: {pcc:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top')

    plt.savefig(os.path.join(output, filename + ".pdf"))
    plt.close()


def subset_scatter_plot(data, output, filename, subsets):
    # Iterate over each key-value pair in the subsets dictionary
    for subset_name, genes in subsets.items():
        # Find the intersection of the genes in the DataFrame and the genes in the subset
        common_genes = np.intersect1d(data.index, genes)
        # Subset the data DataFrame to include only the genes in the current list
        subset_data = data.loc[common_genes]

        # Separate true and predicted values
        y_true = subset_data['true_label']
        y_pred = subset_data['prediction']

        # Calculate the number of present genes in the subset
        count_name = f"{subset_name} ({len(common_genes)}/{len(genes)})"

        # Call scatter_plot for each group
        scatter_plot(y_true, y_pred, output, f"{filename}_{subset_name}", count_name)
