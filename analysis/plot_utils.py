"""
plot_utils.py
v1.0.2, 1/9/2024
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This module provides utility functions for visualizing models and results in TritonCNNv3.
"""

# Standard library imports
import os

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr, gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, auc
from tensorflow.keras.utils import plot_model


def plot_model_architecture(model, output_dir, filename="architecture"):
    """
    Function to plot the architecture of the model.

    Parameters:
    model (keras.Model): The model whose architecture is to be plotted.
    output_dir (str): The directory where the plot will be saved.
    filename (str, optional): The name of the file. Defaults to "architecture".

    Returns:
    None
    """
    # Create the full output path
    output_path = os.path.join(output_dir, f"{filename}.png")

    # Plot the model and save the plot to the output path
    plot_model(model, to_file=output_path, show_shapes=True, show_dtype=True)


def plot_model_loss(history, output_dir, filename="model"):
    """
    Function to plot the model's loss over epochs.

    Parameters:
    history (History): The history object obtained from the fit method of the model.
    output_dir (str): The directory where the plot will be saved.
    filename (str, optional): The name of the file. Defaults to "model".

    Returns:
    None
    """
    # Create a new figure
    plt.figure()

    # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    # Set the title and labels
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Add a legend
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{filename}_loss.pdf"))

    # Close the figure
    plt.close()


def plot_residuals(residuals, output_dir, filename="residuals"):
    """
    Function to plot the distribution of residuals.

    Parameters:
    residuals (array-like): The residuals to be plotted.
    output_dir (str): The directory where the plot will be saved.
    filename (str, optional): The name of the file. Defaults to "residuals".

    Returns:
    None
    """
    # Create a new figure
    plt.figure(figsize=(8, 6))

    # Plot the histogram of residuals
    plt.hist(residuals, bins='auto')

    # Set the title and labels
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of residuals')

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))

    # Close the figure
    plt.close()


def plot_roc_curves(data, output_dir, filename, curve_names=None):
    """
    Function to plot ROC curves for multiple datasets.

    Parameters:
    data (list of tuples): Each tuple contains two arrays: the true labels and the predicted probabilities.
    output_dir (str): The directory where the plot will be saved.
    filename (str): The name of the file.
    curve_names (list of str, optional): The names of the curves. Defaults to None.

    Returns:
    None
    """
    plt.figure()

    for i, (truths, predictions) in enumerate(data):
        fpr, tpr, _ = roc_curve(truths, predictions)
        roc_auc = auc(fpr, tpr)
        label = f'ROC curve for {curve_names[i]} (area = {roc_auc:.2f})' if curve_names else f'ROC curve (area = {roc_auc:.2f})'
        plt.plot(fpr, tpr, color=plt.cm.tab10(i), lw=2, label=label)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
    plt.close()


def plot_roc_curves(data, output_dir, filename, curve_names=None):
    """
    Function to plot ROC curves for multiple datasets.

    Parameters:
    data (list of tuples): Each tuple contains two arrays: the true labels and the predicted probabilities.
    output_dir (str): The directory where the plot will be saved.
    filename (str): The name of the file.
    curve_names (list of str, optional): The names of the curves. Defaults to None.

    Returns:
    None
    """
    plt.figure()

    for i, (truths, predictions) in enumerate(data):
        fpr, tpr, _ = roc_curve(truths, predictions)
        roc_auc = auc(fpr, tpr)
        label = f'ROC curve for {curve_names[i]} (area = {roc_auc:.2f})' if curve_names else f'ROC curve (area = {roc_auc:.2f})'
        plt.plot(fpr, tpr, color=plt.cm.tab10(i), lw=2, label=label)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
    plt.close()


def scatter_plot(y_true, y_pred, output_dir, filename, title, annotations=None):
    """
    Function to create a scatter plot of true vs predicted values, with point densities.

    Parameters:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.
    output_dir (str): The directory where the plot will be saved.
    filename (str): The name of the file.
    title (str): The title of the plot.
    annotations (list of str, optional): The annotations for each point. Defaults to None.

    Returns:
    None
    """
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
    
    # Add annotations if provided
    if annotations is not None:
        for i, txt in enumerate(annotations):
            plt.annotate(txt, (y_true[i], y_pred[i]), fontsize='small')

    plt.xlabel('True Values in Log2(TPM + 1)')
    plt.ylabel('Predicted Values in Log2(TPM + 1)')
    plt.title(title)
    plt.colorbar(label='Density')

    # Print metrics on the plot
    plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR-squared: {r2:.2f}\nPCC: {pcc:.2f}', 
             transform=plt.gca().transAxes, verticalalignment='top')

    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
    plt.close()


def subset_scatter_plot(data, output_dir, filename, subsets):
    """
    Function to create scatter plots of true vs predicted values for multiple subsets of data.

    Parameters:
    data (DataFrame): The data containing the true labels and predictions.
    output_dir (str): The directory where the plot will be saved.
    filename (str): The name of the file.
    subsets (dict): A dictionary where the keys are subset names and the values are lists of genes in each subset.

    Returns:
    None
    """
    # Create a scatter plot for all genes
    scatter_plot(data['true_label'], data['prediction'], output_dir, filename, "All genes")

    # Iterate over each subset
    for subset_name, genes in subsets.items():
        # Subset the data
        subset_data = data.loc[data.index.intersection(genes)]

        # Separate true and predicted values
        y_true = subset_data['true_label']
        y_pred = subset_data['prediction']

        # Calculate the number of present genes in the subset
        count_name = f"{subset_name} ({len(subset_data)}/{len(genes)})"

        # Create a scatter plot for the subset
        scatter_plot(y_true, y_pred, output_dir, f"{filename}_{subset_name}", count_name, subset_data.index if len(subset_data) < 30 else None)


def residuals_plot(y_true, y_pred, output_dir, filename, title, annotations=None, var=None):
    """
    Function to create a residuals plot of true vs predicted values.

    Parameters:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.
    output_dir (str): The directory where the plot will be saved.
    filename (str): The name of the file.
    title (str): The title of the plot.
    annotations (list of str, optional): The annotations for each point. Defaults to None.
    var (array-like, optional): The variable to plot against the residuals. Defaults to None.

    Returns:
    None
    """
    # Calculate residuals
    residuals = y_true - y_pred

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Create residuals plot
    plt.figure(figsize=(8, 6))

    # Calculate the point densities
    xy = np.vstack([y_pred if var is None else var, residuals])
    z = gaussian_kde(xy)(xy)

    plt.scatter(y_pred if var is None else var, residuals, c=z, norm=LogNorm(), alpha=0.5)
    
    # Add annotations if provided
    if annotations is not None:
        for i, txt in enumerate(annotations):
            plt.annotate(txt, (y_pred[i] if var is None else var[i], residuals[i]), fontsize='small')

    plt.xlabel('Predicted Values in Log2(TPM + 1)' if var is None else 'Standard Scaled ' + var.name)
    plt.ylabel('Residuals')
    plt.title(title)
    plt.colorbar(label='Density')

    # Print metrics on the plot
    if var is None:
        plt.text(0.05, 0.95, f'MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR-squared: {r2:.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top')

    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
    plt.close()


def subset_residuals_plot(data, output_dir, filename, subsets, var=None):
    """
    Function to create residuals plots of true vs predicted values for multiple subsets of data.

    Parameters:
    data (DataFrame): The data containing the true labels and predictions.
    output_dir (str): The directory where the plot will be saved.
    filename (str): The name of the file.
    subsets (dict): A dictionary where the keys are subset names and the values are lists of genes in each subset.
    var (str, optional): The variable to plot against the residuals. Defaults to None.

    Returns:
    None
    """
    # Create a residuals plot for all genes
    residuals_plot(data['true_label'], data['prediction'], output_dir, filename, "All genes", var=data[var] if var else None)

    # Iterate over each subset
    for subset_name, genes in subsets.items():
        # Subset the data
        subset_data = data.loc[data.index.intersection(genes)]

        # Separate true and predicted values
        y_true = subset_data['true_label']
        y_pred = subset_data['prediction']

        # Calculate the number of present genes in the subset
        count_name = f"{subset_name} ({len(subset_data)}/{len(genes)})"

        # Create a residuals plot for the subset
        residuals_plot(y_true, y_pred, output_dir, f"{filename}_{subset_name}", count_name, subset_data.index if len(subset_data) < 30 else None, var=subset_data[var] if var else None)
