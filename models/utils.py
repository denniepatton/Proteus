#!/usr/bin/env python
'''
@File    :   utils.py
@Author  :   Akira Nair
@Contact :   akira_nair@brown.edu / anair3@fredhutch.org
@Desc    :   Utility functions
'''

import keras
from keras.callbacks import Callback
import logging
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, r2_score
from itertools import cycle
import seaborn as sns
import pandas as pd
from keras.utils import set_random_seed
# model training callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=0.001)
es = keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy', mode='max', patience=7, verbose=1)
es_rg = keras.callbacks.EarlyStopping(
            monitor='val_mse', mode='min', patience=7, verbose=1)
# logging callback
class LoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        epoch_number = epoch + 1
        loss = logs['loss']
        accuracy = logs['categorical_accuracy']
        val_loss = logs['val_loss']
        val_accuracy = logs['val_categorical_accuracy']
        logging.info(f"Epoch {epoch_number}: "
                     f"Train Loss = {loss:.4f}, Train Accuracy = {accuracy:.4f}, "
                     f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

class RegLoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        epoch_number = epoch + 1
        loss = logs['loss']
        mse = logs['mse']
        val_loss = logs['val_loss']
        val_mse = logs['val_mse']
        logging.info(f"Epoch {epoch_number}: "
                     f"Train Loss (MSE) = {loss:.4f}, Train MSE = {mse:.4f}, "
                     f"Val Loss (MSE) = {val_loss:.4f}, Val MSE = {val_mse:.4f}")

class MAELoggerCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        epoch_number = epoch + 1
        loss = logs['loss']
        mse = logs['mse']
        val_loss = logs['val_loss']
        val_mse = logs['val_mse']
        logging.info(f"Epoch {epoch_number}: "
                     f"Train Loss (MSE) = {loss:.4f}, Train MSE = {mse:.4f}, "
                     f"Val Loss (MSE) = {val_loss:.4f}, Val MSE = {val_mse:.4f}")


def categorical_label_distribution(labels):
    categorical_labels = np.argmax(labels, axis=1)
    unique_values, counts = np.unique(categorical_labels, return_counts=True)
    total = counts.sum()
    for value, count in zip(unique_values, counts):
        logging.info(f"Category {value}: {count} / {round(count/total, 4)}")

def plot_model(model, output, filename = "architecture"):
    keras.utils.plot_model(model, to_file = os.path.join(output, f"{filename}.png"), show_shapes=True, show_layer_activations=True)

def plot_convergence(history, output, filename = "model"):
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
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


def init(output, name = "my_model", use_time = True):
    if use_time:
        time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        directory_name = f"{name}_{time_stamp}"
    else:
        directory_name = f"{name}"
    path = os.path.join(output, directory_name)
    
    if not os.path.exists(path):
        os.makedirs(path)
        logging.basicConfig(filename=os.path.join(path, f"model.log"), 
                        filemode='w', 
                        level=logging.INFO,
                        force=True,
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename=os.path.join(path, f"model.log"), 
                        filemode='w', 
                        level=logging.INFO,
                        force=True,
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
        logging.warning("Directory exists.")
    return path

def create_model_directory(output, name = "sarah_model"):
    time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    directory_name = f"{name}__{time_stamp}"
    path = os.path.join(output, directory_name)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        logging.warning("Directory exists.")
    return path

def evaluation_metrics(predictions, truths):
    # Calculate MAE
    mae = np.mean(np.abs(truths - predictions))
    # Calculate R2 score
    r2 = r2_score(truths, predictions)
    # Calculate residuals
    residuals = truths - predictions
    # metric text
    metric_txt = f"MAE: {mae:.3f} | R^2: {r2:.3f}"
    logging.info(metric_txt)
    return mae, r2, residuals

def plot_residuals(residuals, output):
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins='auto')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of residuals')
    plt.show()
    plt.savefig(os.path.join(output, "residuals.pdf"))
    plt.close() 
def plot_regression_predictions(predictions, truths, output):
    # Create a DataFrame with predictions and truths
    
    data = pd.DataFrame({'Truths': truths, 'Predictions': predictions})

    # Plotting predictions vs truths using Seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='Truths', y='Predictions', color='b', alpha=0.5)
    plt.plot([min(truths), max(truths)], [min(truths), max(truths)], color='r')
    # plt.text(0.05, 0.95, metric_txt, transform=plt.gca().transAxes,
    #      fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.xlabel('Truths')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Truths')
    plt.show()
    plt.savefig(os.path.join(output, "performance.pdf"))
    plt.close()

def open_data(data, split_path=None, keys=None):
    if split_path is not None:
        training_samples, validation_samples, testing_samples = parse_splits(split_path)
    if keys is None or keys=="all":
        npz_keys = list(data.keys())
    else:
        npz_keys = ["ids", "samples", "genes", "labels"]
        npz_keys.extend([keys])
    # load data
    data_opened = {}
    for key in npz_keys:
        logging.info(f"Loading {key}")
        data_opened[key] = data[key]   
    all_samples = data_opened["samples"]
    # get training data
    training = {}
    indices = np.where(np.isin(all_samples, training_samples))[0]
    for key in data_opened:
        training[key] = data[key][indices]        
    # get validation data
    validation = {}
    indices = np.where(np.isin(all_samples, validation_samples))[0]
    for key in data_opened:
        validation[key] = data[key][indices]   
    # get testing data
    testing = {}
    indices = np.where(np.isin(all_samples, testing_samples))[0]
    for key in data_opened:
        testing[key] = data[key][indices]   
    return training, validation, testing




def filter_compiled_by_samples(data, samples, keys=None):
    if keys is None:
        npz_keys = list(data.keys())
    else:
        npz_keys = ["ids", "samples", "genes", "labels"]
        npz_keys.extend(keys)
    all_samples = data["samples"]
    indices = np.where(np.isin(all_samples, samples))[0]
    out = {"indices": indices}
    for key in npz_keys:
        logging.info(f"Loading {key}")
        out[key] = data[key][indices]        
    return out

def parse_splits(splits):
    with open(splits, 'r') as f:
        training = f.readline().strip()
        validation = f.readline().strip()
        testing = f.readline().strip()
    training = training.split(',')
    validation = validation.split(',')
    testing = testing.split(',')
    return training[1:], validation[1:], testing[1:]

def roc_plot(predictions, truths, output):
    n_classes = truths.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(truths[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # fpr, tpr, _ = roc_curve(truths, predictions)
    # roc_auc = auc(fpr, tpr)
    tdic = {0: '_downregulate_ROC',
            1: '_no_change_ROC',
            2: '_upregulate_ROC'}
    plt.figure()
    colors = cycle(sns.color_palette("Set2", n_colors=3))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Label {tdic[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC for multi-class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output, "roc.pdf"))
    plt.close()

def save_predictions(predictions, truths, samples, genes, output, extension=""):
    predictions_df = pd.DataFrame(predictions)
    predictions_df["truths"] = truths
    predictions_df["samples"] = samples
    predictions_df["genes"] = genes
    predictions_df.to_csv(os.path.join(output, f"predictions{extension}.csv"))

def set_seed(seed):
    set_random_seed(seed)