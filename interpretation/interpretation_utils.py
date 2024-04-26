# Alex Netzley
# 2/26/2024

import os
import csv
import argparse
from multiprocessing import Pool, cpu_count

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# Loading In Model
from models.model_utils import CosineDecay, ScalarAttention, CrossModalAttention, TemporalAttention 
from models.models import get_model_instance

# Packages for Base-Pair Kernel Visualization
import matplotlib.pyplot as plt
import numpy as np

# Packages for Motif Plots for Base-Pair Kernels
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Packages for SHAP
import shap
import pickle

import sys
sys.path.append("..") 

from preprocessing.generate_tf_splits import *
from preprocessing.compile_sample_data import *

###############################
### Model Loading Functions ###
###############################
def load_best_model(model_path):
    """
    Load a model from a specified path.

    Args:
        model_path (str): The path to the model to be loaded.
    
    Returns:
        best_model: The loaded model.
    """
    hypermodel = get_model_instance("BioCNN")
    best_model = hypermodel.build_default()
    best_model.load_weights(model_path)
    return best_model

##############################
### Data Loading Functions ###
##############################
def get_data(origin_filepath, regenerate_data=False, regenerate_num_samples=None, save_filepath=None):
    """
    Load the data from a specified path, or regenerate the data if specified.

    Args:
        origin_filepath (str): The path to the data (either a .tfrecord if regenerate_data=True, or .npz if False).
        regenerate_data (bool, optional): Whether to regenerate the data. (default: False)
        regenerate_num_samples (int, optional): The number of samples to regenerate. (default: None)
        save_filepath (str, optional): The path to save the data as a .npz. (default: None)

    Returns:
        input_data (np.array): The loaded or regenerated data.
    """
    if regenerate_data:
        #Check we have num_samples specified
        #if (regenerate_num_samples is None) or (save_filepath is None):
        #    raise ValueError("If regenerate_data is True, regenerate_num_samples and save_path must be specified.")
        #Generate the data
        ds = get_raw_tf_dataset(origin_filepath)
        input_data = tf_ds_np_array(ds, regenerate_num_samples)
        #Save the data
        save_npz(input_data, save_filepath)

    else:
        #Load the data
        input_data = load_npz(origin_filepath)

    return input_data

def get_gene_subset_data(origin_filepath, gene_list_filepath=None, regenerate_data=False, save_filepath=None):
    """
    Load the gene subset data from a specified path, or regenerate the data if specified.

    Args:
        origin_filepath (str): The path to the data (either a .tfrecord if regenerate_data=True, or .npz if False).
        gene_list_filepath (str (OR list), optional): The path to the gene list .txt file OR the gene list iteslf. (default: None)
        regenerate_data (bool, optional): Whether to regenerate the data. (default: False)
        save_filepath (str, optional): The path to save the data as a .npz. (default: None)

    Returns:
        input_data (np.array): The loaded or regenerated gene-specific data.
    """
    if regenerate_data:
        #Check we have num_samples specified
        if (gene_list_filepath is None) or (save_filepath is None):
            raise ValueError("If regenerate_data is True, gene_list_filepath and save_path must be specified.")
        #Generate the data
        ds = get_raw_tf_dataset(origin_filepath)
        ds_subset = get_gene_subset(ds, gene_list_filepath)
        #print(ds_subset)
        input_data = tf_ds_np_array(ds_subset)
        #Save the data
        save_npz(input_data, save_filepath)

    else:
        #Load the data
        input_data = load_npz(origin_filepath)

    return input_data

def get_raw_tf_dataset(tfrecord_path):
    """
    Load a raw TF Dataset from a specified path.

    Args:
        tfrecord_path (str): The path to the TFRecord file.
    
    Returns:
        ds: The loaded TF Dataset.
    """
    ds_raw = tf.data.TFRecordDataset(tfrecord_path)
    ds = ds_raw.cache()
    ds = ds.shuffle(1024*8)
    opt = tf.data.Options()
    opt.experimental_deterministic = False
    ds = ds.with_options(opt)
    ds = ds.map(lambda x: parse_serialized_example(x, labeled=True))
    return ds

def tf_ds_np_array(ds, num_elements=None):
    """
    Converts from a TF Dataset object to a numpy array.

    Args:
        ds (tf.data.Dataset): The incoming TF Dataset object.
        num_elements (int, optional): The number of elements to take from the full dataset.

    Returns:
        input_data: The prepared dataset as a numpy array.
    """
    TSS_image_list = []
    TSS_scalar_list = []
    GB_scalar_list = []
    ref_scalar_list = []
    if num_elements is not None:
        ds = ds.take(num_elements)
    for element in ds:
        TSS_image_list.append(element['TSS_image'].numpy())
        TSS_scalar_list.append(element['TSS_scalars'].numpy())
        GB_scalar_list.append(element['GB_scalars'].numpy())
        ref_scalar_list.append(element['ref_scalars'].numpy())

        TSS_image_stack = np.stack(TSS_image_list)
        TSS_scalar_stack = np.stack(TSS_scalar_list)
        GB_scalar_stack = np.stack(GB_scalar_list)
        ref_scalar_stack = np.stack(ref_scalar_list)

    input_data = [TSS_image_stack, TSS_scalar_stack, GB_scalar_stack, ref_scalar_stack]
    return input_data


def normalize(x):
    return (x - np.mean(x, axis = 0)[np.newaxis]) / np.std(x, axis = 0)[np.newaxis]


def subsample(input_data, sample_size = 20, seed=None):
    """
    Generates a random subsample of the data.

    Args:
        input_data (np.array): The incoming numpy array.
        sample_size (int, optional): The size of the sample.
        seed (int, optional): The random seed for reproducibility. (default: None)

    Returns:
        input_data_sample: The sampled data with sample_size elements.
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(200)
    sample_indices = np.random.choice(input_data[0].shape[0], sample_size, replace=False)
    input_data_sample = [input_data[0][sample_indices], input_data[1][sample_indices], input_data[2][sample_indices], input_data[3][sample_indices]]

    return input_data_sample

def save_npz(input_data, filepath):
    """
    Saves the numpy file to specified location.

    Args:
        input_data (np.array): The numpy array to be saved.
        filepath (str): The folder and filename to save the array as.
    """
    TSS_image_stack = input_data[0]
    TSS_scalar_stack = input_data[1]
    GB_scalar_stack = input_data[2]
    ref_scalar_stack = input_data[3]
    np.savez(filepath, TSS_image_stack = TSS_image_stack, TSS_scalar_stack = TSS_scalar_stack, GB_scalar_stack = GB_scalar_stack, ref_scalar_stack = ref_scalar_stack)

def load_npz(filepath):
    """
    Load a saved .npz file to specified location.

    Args:
        filepath (str): The folder and filename where the .npz is located.
    
    Returns:
        input_data: The loaded numpy array.
    """
    #Load input_data from my local folder
    input_data_npz = np.load(filepath, allow_pickle=True)

    #unpack the input_data, store it in a list
    input_data = []
    for key in input_data_npz:
        input_data.append(input_data_npz[key])
    return input_data

def get_gene_subset(input_data, gene_list_path):
    """
    Extracts all entries of a dataset that are a gene that is present in gene_list_path.

    Args:
        input_data (tf.data.Dataset): The input TFDataset data.
        gene_list_path (str OR list): The path to the gene list file. OR the gene list itself.
    
    Returns:
        ds_subset: The subsetted tf.data.Dataset with only the genes from the gene list.
    """
    #gene_list_path = "/fh/fast/ha_g/user/rpatton/TritonCNN/RNA-Seq/LuCaPvsGTEx_up-regulated.txt"
    if isinstance(gene_list_path, str):
        #print(gene_list_path)
        gene_list = pd.read_table(gene_list_path, header=None, names=['gene_name'])
        #convert dataframe to list
        gene_list = gene_list['gene_name'].tolist()
    else:
        gene_list = gene_list_path

    ds_subset = input_data.filter(lambda x: tf.reduce_any(tf.equal(x['gene_name'], gene_list)))
    return ds_subset

######################################
### Kernel Visualization Functions ###
######################################
def plot_nucleotide_conv_filters(model, folder_path=None):
    """
    Plots all the 2D convolutional filters for the nucleotide layer.

    Args:
        model (Model): The Keras model.
        folder_path (str, optional): The base folder location to save the figure. (default: None)
    """
    conv_layer_name = [layer.name for layer in model.layers if layer.name.startswith('conv2d')]
    actg_conv_layer = model.get_layer(name=conv_layer_name[0])
    weights = actg_conv_layer.get_weights()[0]
    #print(weights.shape)
    fig, axs = plt.subplots(4, 8, figsize=(15, 7))
    for j in range(4):
        for i in range(8):
            axs[j, i].imshow(weights[:, :, 0, 8*j + i], cmap='viridis')
            axs[j, i].set_yticks(np.arange(4))
            axs[j, i].set_yticklabels(['A', 'C', 'G', 'T'])
    cbar = fig.colorbar(axs[0, 0].imshow(weights[:, :, 0, 0], cmap='viridis'), ax=axs, orientation='vertical', fraction=0.05)
    # Add in a label on the y axis of each plot
    cbar.set_label('Weight')
    fig.suptitle(f'Nucleotide Conv Filters (32)', fontsize=16)
    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'nucleotide_conv_filters.png'))
    else:
        plt.show()

def plot_signal_kernels(best_model, folder_path=None):
    """
    Generally not used in analyses
    Plot the 7 signal first layer 3x1 kernels.

    Args:
        best_model (Model): The Keras model.
        folder_path (str, optional): The base folder location to save the figure. (default: None)
    """
    ## Visualizing all 7 signal first layer 3x1 kernels
    triton_signals = ['depth', 'nucleosome cp', 'phased-nucleosome cp', 'fl short/long ratio', 'fl diversity', 'fl entropy', 'peak locations']

    for signal_num in range(len(triton_signals)):
        #Extract the weights of the corresponding first convolutional layer
        if signal_num == 0:
            first_conv_layer = best_model.get_layer(name='conv1d')
        else:
            first_conv_layer = best_model.get_layer(name=f'conv1d_{signal_num}')
        weights = first_conv_layer.get_weights()[0]
        print(weights.shape)

        fig, axs = plt.subplots(6, 8, figsize=(15, 7))

        for i in range(6):
            for j in range(8):
                axs[i, j].imshow(weights[:, :, 8*i + j], cmap='viridis')
                axs[i, j].axis('off')

        # Add color bar
        cbar = fig.colorbar(axs[0, 0].imshow(weights[:, :, 0], cmap='viridis'), ax=axs, orientation='vertical', fraction=0.05)
        cbar.set_label('Weight')

        # Add title
        signal_name = triton_signals[signal_num]
        fig.suptitle(f'Convolutional Layer 1 Filters for {signal_name}', fontsize=16)
        if folder_path is not None:
            fig.savefig(os.path.join(folder_path, f'{signal_name}_conv_filters.png'))
        else:
            fig.show()

def plot_signal_combination_conv3d_filters(model, folder_path=None):
    """
    Plots 8 of the convolutional filters for the 3d signal combination layer.

    Args:
        model (Model): The Keras model.
        folder_path (str, optional): The base folder location to save the figure. (default: None)
    """
    ## Visualizing 1 kernel from the depthwise 2d layer
    conv_layer_name = [layer.name for layer in model.layers if layer.name.startswith('conv3d')]
    conv3d_layer = model.get_layer(name=conv_layer_name[0])
    weights = conv3d_layer.get_weights()[0]
    #print(weights.shape)
    fig, axs = plt.subplots(4, 2, figsize=(25,5))
    for j in range(4):
        for i in range(2):
            axs[j, i].imshow(np.squeeze(weights[:, :, :, :, 4*j + i]), cmap='viridis')
            axs[j, i].set_yticks(np.arange(8))
            if i == 0:
                axs[j, i].set_yticklabels(['depth', 'nucleosome cp', 'phased-nucleosome cp', 'fl short/long ratio', 'fl diversity', 'fl entropy', 'peak locations', 'nucleotides'])
            else:
                axs[j, i].set_yticklabels([])
    cbar = fig.colorbar(axs[0, 0].imshow(np.squeeze(weights[:, :, :, :, 0]), cmap='viridis'), ax=axs, orientation='vertical', fraction=0.05)
    # Add in a label on the y axis of each plot
    #plt.subplots_adjust(hspace=0.5)
    cbar.set_label('Weight')
    fig.suptitle(f'Depthwise-Conv Filters ({(j+1)*(i+1)} of 128)', fontsize=16)
    if folder_path is not None:
        fig.savefig(os.path.join(folder_path, 'signal_combination_conv3d_filters.png'))
    else:
        plt.show()

def plot_summed_signal_combination_conv3d_filters(model, folder_path=None):
    """
    Plots the summed signal combination conv3d filters.

    Args:
        model (Model): The Keras model.
        folder_path (str, optional): The base folder location to save the figure. (default: None)
    """
    #Extract all the weights from the first depthwise 3d convolutional layer
    conv_layer_name = [layer.name for layer in model.layers if layer.name.startswith('conv3d')]
    conv3d_layer = model.get_layer(name=conv_layer_name[0])
    weights = conv3d_layer.get_weights()[0]
    #print(weights.shape)
    fig, axs = plt.subplots(4, 32, figsize=(25,5))
    for j in range(4):
        for i in range(32):
            axs[j, i].imshow(np.sum(np.squeeze(weights[:, :, :, :, 4*j + i]), axis=1)[:, np.newaxis], cmap='viridis')
            axs[j, i].set_yticks(np.arange(8))
            axs[j, i].set_xticks([])
            if i == 0:
                axs[j, i].set_yticklabels(['depth', 'nucleosome cp', 'phased-nucleosome cp', 'fl short/long ratio', 'fl diversity', 'fl entropy', 'peak locations', 'nucleotides'])
            else:
                axs[j, i].set_yticklabels([])
    cbar = fig.colorbar(axs[0, 0].imshow(np.sum(np.squeeze(weights[:, :, :, :, 0]), axis=1)[:, np.newaxis], cmap='viridis'), ax=axs, orientation='vertical', fraction=0.05)
    # Add in a label on the y axis of each plot
    #plt.subplots_adjust(hspace=0.5)
    cbar.set_label('Weight')
    fig.suptitle(f'Summed Depthwise-Conv Filters ({(j+1)*(i+1)} of 128)', fontsize=16)
    if folder_path is not None:
        fig.savefig(os.path.join(folder_path, 'summed_signal_combination_conv3d_filters.png'))
    else:
        plt.show()



################################
### Motif Plotting Functions ###
################################
def plot_nucleotide_motifs(model, folder_path=None):
    """
    Plots all the motif plots for the Conv2d Nucleotide filter.

    Args:
        model (Model): The Keras model.
        folder_path (str, optional): The base folder location to save the figure. (default: None)
    """
    #Extract the Weights
    conv_layer_name = [layer.name for layer in model.layers if layer.name.startswith('conv2d')]
    actg_conv_layer = model.get_layer(name=conv_layer_name[0])
    weights = actg_conv_layer.get_weights()[0]

    # Data Preprocessing
    def shift(x): # column shift
        x = x - np.min(x, axis=0)  # Subtracting the min value from each column to make each column non-negative
        return x / x.sum(axis=0)

    max_weight = np.max(weights)
    min_weight = np.min(weights)
    diff  = max_weight - min_weight

    def global_shift(x, diff = diff): 
        return x / diff # global shift # Divide each value by the max - min of all filter values across all filters

    processed_motif_data = {}
    for filter_number in range(weights.shape[3]):
        #print(filter_number)
        processed_filter = global_shift(weights[:, :, 0, filter_number])

        processed_motif_data_list = []
        for position in range(weights.shape[1]):
            processed_motif_data_list.append([
                ('A', processed_filter[0, position]),
                ('C', processed_filter[1, position]),
                ('G', processed_filter[2, position]),
                ('T', processed_filter[3, position])
            ])
        processed_motif_data[filter_number] = processed_motif_data_list


    # Define Letter-Drawing Patch Code
    globscale = 1.35
    LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1),
                "G" : TextPath((-0.384, 0), "G", size=1),
                "A" : TextPath((-0.35, 0), "A", size=1),
                "C" : TextPath((-0.366, 0), "C", size=1) }
    COLOR_SCHEME = {'G': 'orange', 
                    'A': 'red', 
                    'C': 'blue', 
                    'T': 'darkgreen'}

    def letterAt(letter, x, y, yscale=1, ax=None, neg=False):
        text = LETTERS[letter]
        if neg and yscale <= 0: # If the yscale is negative, flip the letter ONLY when neg is True
            t = mpl.transforms.Affine2D().scale(1*globscale, -yscale*globscale) + \
                mpl.transforms.Affine2D().translate(x, y) + ax.transData
        else:
            t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
                mpl.transforms.Affine2D().translate(x,y) + ax.transData
                
        p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
        if ax != None:
            ax.add_artist(p)
        return p

    # Plotting the motif filters
    fig, axs = plt.subplots(4, 8, figsize=(20,6))

    all_scores = processed_motif_data_list

    #All values shifted postive, scaled to [0,1]
    for j in range(4):
        for i in range(8):
            #Extract a specfic filter
            all_scores = processed_motif_data[j*8 + i]
            x = 1
            maxi = 0
            for scores in all_scores:
                y_down_score = 0
                y_up_score = 0
                for base, score in scores:
                    if score > 0:
                        letter = letterAt(base, x, y_up_score, score, axs[j, i], True)
                        y_up_score += score
                    else:
                        y_down_score += score
                        letter = letterAt(base, x, y_down_score, score, axs[j, i], True)

                    axs[j, i].add_patch(letter)
                    axs[j, i].add_patch(patches.Rectangle((0, -1), 4,1,alpha = 0.05)) #transparency
                x += 1

            axs[j, i].set_xticks(range(1, x))
            axs[j, i].set_xlim((0, x))
            axs[j, i].set_ylim((-0.8, 1))  

    plt.suptitle('Motif Plots for the Nucleotide Conv Filters (32)', fontsize=16)
    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'nucleotide_motifs.png'))
    else:
        plt.show()

def kernel_visualization(model, save_filepath=None):
    """
    Wrapper function to visualizes the kernels for the model.

    Args:
        model (Model): The Keras model.
        save_filepath (str, optional): The base folder location to save the figures. (default: None)
    """
    folder_path = None
    if save_filepath is not None:
        folder_path = os.path.join(save_filepath, 'kernel_visualizations')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    plot_nucleotide_conv_filters(model, folder_path)
    plot_signal_combination_conv3d_filters(model, folder_path)
    plot_summed_signal_combination_conv3d_filters(model, folder_path)
    plot_nucleotide_motifs(model, folder_path)

######################
### SHAP Functions ###
######################
def plot_shap_TSS_Images(shap_values, folder_path=None):
    '''
    Plots the mean absolute SHAP values for the TSS Images.

    Args:
        shap_values: The SHAP values.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    '''
    images = np.array(shap_values[0][0])
    images_t = np.transpose(images, (2,0,1))
    means = np.mean(np.abs(images_t), axis=1)
    plt.clf()
    cols = ['1. Depth', '2. Probable Nucleosome Center Profile', '3. Phased Nucleosome Profile', '4. Fragment Length Short:Long Ratio', '5. Fragment Length Diversity', '6. Fragment Length Entropy', '7. Peaks', '8. A-Adenine', '9. C-Cytosine', '10. G-Guanine', '11. T-Tyrosine']
    plt.figure(figsize=(6, 4))
    plt.imshow(means, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar for reference
    plt.title('Mean Abs SHAP Values for TSS Images')
    plt.colorbar()
    plt.yticks(np.arange(len(cols)), cols)

    # Show the plot
    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'TSS_Images_SHAP.png'))
    else:
        plt.show()


def plot_shap_images_nucleotide(shap_values, folder_path=None):
    '''
    Plots the mean absolute SHAP values for the TSS Images, with the nucleotide SHAPs summed to one signal.

    Args:
        shap_values (np.array): The SHAP values.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    '''
    images = np.array(shap_values[0][0])
    images_t = np.transpose(images, (2,0,1))
    images_t.shape
    new_images_t = images_t.copy()
    summed_bp_shap = np.sum(new_images_t[7:11], axis=0)
    #print(summed_bp_shap.shape)
    new_images_t[7] = summed_bp_shap
    new_images_t = new_images_t[:8]
    new_means = np.mean(np.abs(new_images_t), axis=1)
    #print(new_images_t.shape)

    plt.clf()
    new_cols = ['1. Depth', '2. Probable Nucleosome Center Profile', '3. Phased Nucleosome Profile', '4. Fragment Length Short:Long Ratio', '5. Fragment Length Diversity', '6. Fragment Length Entropy', '7. Peaks', '8. Nucleotide']
    plt.figure(figsize=(6, 4))
    plt.imshow(new_means, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar for reference
    plt.title(f'Mean Abs SHAP Values for TSS Images Nucleotide Summed')#, with {new_images_t.shape[1]} samples in the test set')
    plt.colorbar()
    plt.yticks(np.arange(len(new_cols)), new_cols)

    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'TSS_Images_Nucleotide_SHAP.png'))
    else:
        plt.show()


def plot_shap_images_individual(shap_values, folder_path=None):
    '''
    Plots the mean absolute SHAP values for the TSS Images, with the nucleotide SHAPs summed to one signal.
    Additionally, each signal is it own subplot, and the color scales are now local to each individual signal.

    Args:
        shap_values (np.array): The SHAP values.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    '''
    images = np.array(shap_values[0][0])
    images_t = np.transpose(images, (2,0,1))
    images_t.shape
    new_images_t = images_t.copy()
    summed_bp_shap = np.sum(new_images_t[7:11], axis=0)
    #print(summed_bp_shap.shape)
    new_images_t[7] = summed_bp_shap
    new_images_t = new_images_t[:8]
    new_means = np.mean(np.abs(new_images_t), axis=1)
    plt.clf()
    new_cols = ['1. Depth', '2. PNuc CP', '3. PNuc P', '4. FL S:L R', '5. FL Div', '6. FL Ent', '7. Peaks', '8. Nucleotide']
    plt.figure(figsize=(6, 4))
    fig, axs = plt.subplots(8, 1, figsize=(20,8))
    for j in range(8):
        axs[j].imshow(new_means[j, :][np.newaxis], cmap='viridis', aspect='auto', interpolation='nearest')
        axs[j].set_yticks([])
        axs[j].set_ylabel(new_cols[j])

    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'Mean Abs SHAP Values for TSS Images Nucleotide Summed (Color Scale Local)')
    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'TSS_Images_Nucleotide_SHAP_Individual.png'))
    else:
        plt.show()

def plot_shap_global(shap_values, include_images, scale_include_images = False, folder_path=None):
    '''
    Plots the mean absolute SHAP values for all four inputs (TSS Images, TSS Scalars, GB Scalars, and Reference Scalars).
    A global colorbar scale is included, with (optionally) an adjustment to account for extremely large TSS Image Summed Nucleotide SHAP values.

    Args:
        shap_values (np.array): The SHAP values.
        include_images (bool): Whether to include the TSS Images in the plot.
        scale_images_special (bool, optional): Whether to include the TSS Image Summed SHAP values from the colorbar min and max. (default: False)
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    '''
    images = np.array(shap_values[0][0])
    images_t = np.transpose(images, (2,0,1))
    images_t.shape
    new_images_t = images_t.copy()
    summed_bp_shap = np.sum(new_images_t[7:11], axis=0)
    #print(summed_bp_shap.shape)
    new_images_t[7] = summed_bp_shap
    new_images_t = new_images_t[:8]

    # Plot mean abs SHAP Values for the TSS Scalars, GB Scalars, and Reference Scalars, and summed signals
    TSS_image_cols = ['1. Depth', '2. Probable Nucleosome Center Profile', '3. Phased Nucleosome Profile', '4. Fragment Length Short:Long Ratio', '5. Fragment Length Diversity', '6. Fragment Length Entropy', '7. Peaks', '8. Nucleotide']
    TSS_scalar_cols = ['1. fragment-diversity', '2. fragment-entropy', '3. fragment-mad', '4. fragment-mean', '5. fragment-median', '6. fragment-ratio', '7. fragment-stdev', '8. mean-depth', '9. np-amplitude', '10. np-period', '11. np-score', '12. var-ratio' ]
    ref_scalar_cols = ['1. numexons', '2. transcript_length', '3. mRNA_size']
    
    TSS_images_summed = np.mean(np.abs(np.sum(new_images_t, axis = 2)), axis = 1)[:, np.newaxis]

    TSS_scalars = np.mean(np.abs(shap_values[0][1]), axis = 0)[:, np.newaxis]
    TSS_scalars_std = np.std(shap_values[0][1], axis = 0)[:, np.newaxis]

    GB_scalars = np.mean(np.abs(shap_values[0][2]), axis = 0)[:, np.newaxis]
    GB_scalars_std = np.std(shap_values[0][2], axis = 0)[:, np.newaxis]

    ref_scalars = np.mean(np.abs(shap_values[0][3]), axis = 0)[:, np.newaxis]
    ref_scalars_std = np.std(shap_values[0][3], axis = 0)[:, np.newaxis]

    if include_images:
        cols_list = [TSS_image_cols, TSS_scalar_cols, TSS_scalar_cols, ref_scalar_cols]
        name_list = ['TSS Images', 'TSS Scalars', 'GB Scalars', 'Reference Scalars']
        fig, axs = plt.subplots(1,4, figsize=(16, 4))
        shap_values_summed = [TSS_images_summed, TSS_scalars, GB_scalars, ref_scalars]

    else:
        cols_list = [TSS_scalar_cols, TSS_scalar_cols, ref_scalar_cols]
        name_list = ['TSS Scalars', 'GB Scalars', 'Reference Scalars']
        fig, axs = plt.subplots(1,3, figsize=(12, 4))
        shap_values_summed = [TSS_scalars, GB_scalars, ref_scalars]

    if scale_include_images:
        vmin = min(np.min(TSS_images_summed), np.min(TSS_scalars), np.min(GB_scalars), np.min(ref_scalars))
        vmax = max(np.max(TSS_images_summed), np.max(TSS_scalars), np.max(GB_scalars), np.max(ref_scalars))
    else:
        vmin = min(np.min(TSS_scalars), np.min(GB_scalars), np.min(ref_scalars))
        vmax = max(np.max(TSS_scalars), np.max(GB_scalars), np.max(ref_scalars))

    for i in range(len(cols_list)):
        if i == 0:
            im = axs[i].imshow(shap_values_summed[i][np.newaxis], cmap='viridis', aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin)
        axs[i].imshow(shap_values_summed[i], cmap='viridis', aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin)
        axs[i].set_yticks(np.arange(len(cols_list[i])), cols_list[i])
        axs[i].set_xticks([])
        axs[i].set_title(f'Mean SHAP Values for {name_list[i]}')

    fig.subplots_adjust(wspace=5) 
    fig.colorbar(im, ax=axs.ravel().tolist())
    if include_images and scale_include_images:
        fig.suptitle(f'Mean Abs SHAP Values for All Inputs (Color Scale Global)', y=1.1)
    elif include_images and not scale_include_images:
        fig.suptitle(f'Mean Abs SHAP Values for All Inputs (Color Scale Global, Excluding TSS Images)', y=1.1)
    else:
        fig.suptitle(f'Mean Abs SHAP Values for Input Scalars (Color Scale Global)', y=1.1)

    if folder_path is not None:
        if include_images and scale_include_images:
            fig.savefig(os.path.join(folder_path, 'Global_SHAP_All_CS_Global.png'))
        elif include_images and not scale_include_images:
            fig.savefig(os.path.join(folder_path, 'Global_SHAP_All_CS_Global_Exc_Images.png'))
        else:
            fig.savefig(os.path.join(folder_path, 'Global_SHAP_Scalars.png'))
    else:
        fig.show()

def plot_shap_scalars_individual(shap_values, folder_path=None):
    '''
    Plots the mean absolute SHAP values for th three scalar inputs, with local colorbars for each.

    Args:
        shap_values (np.array): The SHAP values.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    '''
    TSS_scalars = np.mean(np.abs(shap_values[0][1]), axis = 0)[:, np.newaxis]
    GB_scalars = np.mean(np.abs(shap_values[0][2]), axis = 0)[:, np.newaxis]
    ref_scalars = np.mean(np.abs(shap_values[0][3]), axis = 0)[:, np.newaxis]

    TSS_scalar_cols = ['1. fragment-diversity', '2. fragment-entropy', '3. fragment-mad', '4. fragment-mean', '5. fragment-median', '6. fragment-ratio', '7. fragment-stdev', '8. mean-depth', '9. np-amplitude', '10. np-period', '11. np-score', '12. var-ratio' ]
    ref_scalar_cols = ['1. numexons', '2. transcript_length', '3. mRNA_size']

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Define data and titles
    data = [TSS_scalars, GB_scalars, ref_scalars]
    titles = ['Mean Shap Values for TSS Scalars', 'Mean Shap Values for GB Scalars', 
              'Mean Shap Values for reference Scalars']
    cols = [TSS_scalar_cols, TSS_scalar_cols, ref_scalar_cols]

    # Plot each subplot and add colorbar
    for ax, d, t, c in zip(axs, data, titles, cols):
        im = ax.imshow(d, cmap='viridis', aspect='auto', interpolation='nearest')
        ax.set_yticks(np.arange(len(c)), c)
        ax.set_xticks([])
        ax.set_title(t)
        fig.colorbar(im, ax=ax)

    fig.subplots_adjust(wspace=5) 
    fig.suptitle(f'Mean Abs SHAP Values for Input Scalars (Color Scale Local)', y=1.1)

    if folder_path is not None:
        fig.savefig(os.path.join(folder_path, 'SHAP_Scalars_Individual.png'))
    else:
        fig.show()
    

def plot_shap_scalar_distributions(shap_values):
    '''
    Plots the dstribution of shap values across the different test data samples for TSS and GB Scalars.

    Args:
        shap_values (np.array): The SHAP values.
    '''
    TSS_scalar_cols = ['1. fragment-diversity', '2. fragment-entropy', '3. fragment-mad', '4. fragment-mean', '5. fragment-median', '6. fragment-ratio', '7. fragment-stdev', '8. mean-depth', '9. np-amplitude', '10. np-period', '11. np-score', '12. var-ratio' ]
    #plot the distribution of shap values for each scalar
    plt.clf()
    fig, axs = plt.subplots(3, 4, figsize=(10, 10))
    for i in range(3):
        for j in range(4):
            axs[i, j].hist(shap_values[0][1][:, 4*i + j], bins=25)
            axs[i, j].set_ylabel(f'{i}', rotation=0)
            axs[i, j].set_title(TSS_scalar_cols[4*i + j])
            axs[i, j].set_xlim(-0.5, 0.5)

    fig.suptitle('TSS Scalars Shap Value Distributions', fontsize=16)
    plt.show()

    plt.clf()
    fig, axs = plt.subplots(3, 4, figsize=(10, 10))
    for i in range(3):
        for j in range(4):
            axs[i, j].hist(shap_values[0][2][:, 4*i + j], bins=25)
            axs[i, j].set_ylabel(f'{i}', rotation=0)
            axs[i, j].set_title(TSS_scalar_cols[4*i + j])
            axs[i, j].set_xlim(-0.5, 0.5)

    fig.suptitle('GB Scalars Shap Value Distributions', fontsize=16)
    plt.show()

def shap_analysis(model, input_data, background_size, test_size, save_filepath=None, random_seed=False):
    '''
    Wrapper function to perform SHAP analysis on the model.

    Args:
        model (Model): The Keras model.
        input_data (np.array): The input data.
        background_size (int): The size of the background data.
        test_size (int): The size of the test data.
        save_filepath (str, optional): The base folder location to save the figures. (default: None)
        random_seed (bool, optional): Whether to use a random seed. (default: False)
    '''
    folder_path = None
    if save_filepath is not None:
        folder_path = os.path.join(save_filepath, 'shap_analysis')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    #Randomly Subsample the Data
    if isinstance(background_size, int):
        input_data_background = subsample(input_data, sample_size = background_size, seed=random_seed)
    else:
        input_data_background = background_size
    
    if isinstance(test_size, int):
        input_data_test = subsample(input_data, sample_size = test_size, seed=random_seed)
    else:
        input_data_test = test_size

    # Execute the SHAP Analysis
    explainer = shap.GradientExplainer(model, input_data_background)
    shap_values = explainer.shap_values(input_data_test)

    # Plot mean abs SHAP Values for the TSS Images
    plot_shap_TSS_Images(shap_values, folder_path=folder_path)

    # Plot mean abs SHAP Values for the TSS Images with the Nucleotides summed
    plot_shap_images_nucleotide(shap_values, folder_path=folder_path)

    # Plot mean abs SHAP Values for the TSS Images with the Nucleotides summed, each signal color scaled individually
    plot_shap_images_individual(shap_values, folder_path=folder_path)

    # Plot global comparison of mean abs SHAP Values for all inputs, with the signals summed
    plot_shap_global(shap_values, include_images = True, scale_include_images = True, folder_path=folder_path)

    # Plot global comparison of mean abs SHAP Values for three scalars
    plot_shap_global(shap_values, include_images = False, folder_path=folder_path)

    # Plot scalars with each color bar scaled locally
    plot_shap_scalars_individual(shap_values, folder_path=folder_path)

#################################
### Gradient Ascent Functions ###
#################################
def prep_data_trainable(input_data, only_signal=False):
    """
    Prepares the input data for gradient ascent by making the specified inputs trainable.

    Args:
        input_data (np.array): The input data.
        only_signal (bool, optional): Whether to only make the signal inputs trainable. (default: False)
    """
    x_tf = []
    freeze_set = [3, 12] if only_signal else [3]

    for data in input_data:
        #Freeze Ref Scalars
        if data.shape[1] in freeze_set:
            data = tf.Variable(data, trainable=False)
        #Unfreeze the GB and TSS Scalars (and the TSS Image, mask will be applied later to zero out the nucleotide signals gradient)
        else:
            data = tf.Variable(data, trainable=True)

        x_tf.append(data)
    return x_tf

def gradient_ascent_train(input_data, model, iterations, step_size, label, only_signal=False):
    """
    Main Training Loop to perform gradient ascent on the specified dataset

    Args:
        input_data (tf.Variable): The input data, as a trainable input image.
        model (Model): The untrainable, frozen Keras model.
        iterations (int): The number of iterations to perform.
        step_size (float): The step size for the gradient ascent.
        label (float): The label to optimize for.
        only_signal (bool, optional): Whether to only optimize the signal inputs. (default: False)
    
    Returns:
        updated_data (tf.Variable): The optimized input image.
        losses (list): The loss values for each iteration.
        outputs (list): The output values for each iteration.
    """
    freeze_set = [3, 2, 1] if only_signal else [3]
    losses = []
    outputs = []
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.reset()
            # Forward pass
            output = model(input_data)
            # Define the loss as the mean of the output
            loss = tf.square(output - label)
            losses.append(loss)
            outputs.append(output)

            gradients = tape.gradient(loss, input_data)

        for d in range(len(gradients)):
            if d in freeze_set:
                continue
            if d == 0:
                mask = tf.concat([tf.ones((7, 2000), dtype=tf.float32), tf.zeros((4, 2000), dtype=tf.float32)], axis=0)
                mask = tf.expand_dims(mask, axis=0)
                mask = tf.transpose(mask, perm=[0, 2, 1])
                gradients[d] = tf.multiply(gradients[d], mask)

            input_data[d].assign_add(-step_size * gradients[d])
    #print(len(input_data))
    return input_data, losses, outputs

def plot_gradient_training_curves(num_trials, outputs_list, losses_list, folder_path=None):
    """
    Plots the training curves for the gradient ascent.

    Args:
        num_trials (int): The number of independent gradient trials to peform.
        outputs_list (list): The output values for each iteration.
        losses_list (list): The loss values for each iteration.
        folder_path (str, optional): The base folder location to save the figure. (default: None)
    """
    fig, axs = plt.subplots(num_trials, 2, figsize=(6, 2*num_trials))
    for i in range(num_trials):
        if num_trials == 1:
            axs[0].plot(np.squeeze(outputs_list))
            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('Predicted Expression (log2(TPM))')

            axs[1].plot(np.squeeze(losses_list))
            axs[1].set_xlabel('Iteration')
            axs[1].set_ylabel('Loss')
        else:
            axs[i, 0].plot(np.squeeze(outputs_list[i]))
            axs[i, 0].set_xlabel('Iteration')
            axs[i, 0].set_ylabel('Predicted Expression (log2(TPM))')

            axs[i, 1].plot(np.squeeze(losses_list[i]))
            axs[i, 1].set_xlabel('Iteration')
            axs[i, 1].set_ylabel('Loss')
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle('Gradient Ascent Training Curves', fontsize=14)
    if folder_path is not None:
        fig.savefig(os.path.join(folder_path, 'gradient_ascent_training_curves.png'))
    else:
        plt.show()


def plot_gradient_updated_images(num_trials, original_data, updated_data, folder_path=None):
    """
    Plots the original vs updated image data for the gradient ascent.

    Args:
        num_trials (int): The number of independent gradient trials performed.
        original_data (np.array): The original input data.
        updated_data (np.array): The updated input data.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    """
    plt.clf()
    cols = ['1. Depth', '2. Probable Nucleosome Center Profile', '3. Phased Nucleosome Profile', '4. Fragment Length Short:Long ratio', '5. Fragment Length Diversity', '6. Fragment Length Entropy', '7. Peaks']
    for trial in range(num_trials):
        fig, axs = plt.subplots(len(cols), 1, figsize=(12, 8))
        for i in range(len(cols)):
            axs[i].plot(original_data[trial][0][0][:, i])
            axs[i].set_ylabel(cols[i], rotation=0)
            axs[i].plot(updated_data[trial][0][0][:, i])
            
        #plt.subplots_adjust(left=0.5)
        fig.suptitle(f'TSS Images Pre and Post Gradient Ascent for Trial {trial}', fontsize=16)
        fig.legend(['Original', 'Updated'], loc='upper right', fontsize=16)
        if folder_path is not None:
            fig.savefig(os.path.join(folder_path, f'gradient_ascent_images_trial_{trial}.png'))
        else:
            plt.show()

def plot_gradient_updated_scalars(num_trials, original_data, updated_data, folder_path=None):
    """
    Plots the original vs updated scalar data for the gradient ascent.

    Args:
        num_trials (int): The number of independent gradient trials performed.
        original_data (np.array): The original input data.
        updated_data (np.array): The updated input data.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    """
    TSS_scalar_cols = ['1. fragment-diversity', '2. fragment-entropy', '3. fragment-mad', '4. fragment-mean', '5. fragment-median', '6. fragment-ratio', '7. fragment-stdev', '8. mean-depth', '9. np-amplitude', '10. np-period', '11. np-score', '12. var-ratio' ]
    ref_scalar_cols = ['1. numexons', '2. transcript_length', '3. mRNA_size']
    cols_list = [TSS_scalar_cols, TSS_scalar_cols, ref_scalar_cols]
    bar_width = 0.5

    for trial in range(num_trials):
        fig, axs = plt.subplots(3,1, figsize=(12, 12))

        #Creating shading based on relative difference
        rel_diff_prelim_list = []
        for i in range(3):
            rel_diff = (np.abs(updated_data[trial][i+1][0] - original_data[trial][i+1][0])) #/ np.abs(original_data[it][i+1][0])
            if i == 0:
                min_val = np.min(rel_diff)
                max_val = np.max(rel_diff)
            else:
                if np.min(rel_diff) < min_val:
                    min_val = np.min(rel_diff)
                if np.max(rel_diff) > max_val:
                    max_val = np.max(rel_diff)
            rel_diff_prelim_list.append(rel_diff)

        rel_diff_list = []
        for i in range(3):
            rel_diff_list.append((((rel_diff_prelim_list[i] - min_val) / (max_val - min_val))*(4/5) + 1/5).tolist())


        title_list = ['TSS Scalars', 'GB Scalars', 'Ref Scalars']
        for i in range(3):
            bars = axs[i].barh(np.arange(len(cols_list[i])) + bar_width/4, original_data[trial][i+1][0], height=bar_width/2, label='Original')
            for bar, alpha in zip(bars, rel_diff_list[i]):
                bar.set_alpha(alpha)
            bars = axs[i].barh(np.arange(len(cols_list[i])) - bar_width/4, updated_data[trial][i+1][0], height=bar_width/2, label='Updated')
            for bar, alpha in zip(bars, rel_diff_list[i]):
                bar.set_alpha(alpha)
            axs[i].set_xlabel('Value')
            axs[i].set_yticks(np.arange(len(cols_list[i])), cols_list[i])
            axs[i].set_title(f'{title_list[i]} Pre and Post Gradient Ascent')

        fig.suptitle(f'Scalars Pre and Post Gradient Ascent for Trial {trial}', fontsize=16)
        fig.legend(['Original', 'Updated'], loc='upper right', fontsize = 16)
        if folder_path is not None:
            fig.savefig(os.path.join(folder_path, f'gradient_ascent_scalars_trial_{trial}.png'))
        else:
            fig.show()

def gradient_ascent(input_data, model, label, num_trials, iterations=100, step_size=5e-2,  only_signal=False, random_seed=False, save_filepath=None):
    """
    Main training loop to perform gradient ascent.

    Args:
        input_data (np.array): The input data to sample from.
        model (Model): The Keras model to use for gradient ascent.
        label (float): The label to optimize for, a log2(TPM) value, usually low (0) or high (>5).
        num_trials (int): The number of independent gradient trials to peform.
        iterations (int, optional): The number of iterations to perform. (default: 100)
        step_size (float, optional): The step size for the gradient ascent. (default: 5e-2)
        only_signal (bool, optional): Whether to only optimize the signal inputs. (default: False)
        random_seed (bool, optional): Whether to use a random seed per trial or not. (default: False)
        save_filepath (str, optional): The base folder location to save the figures. (default: None)
    
    Returns:
        input_image (tf.Variable): The optimized input image.
        losses (list): The loss values for each iteration.
        outputs (list): The output values for each iteration.
    """

    folder_path = None
    if save_filepath is not None:
        folder_path = os.path.join(save_filepath, 'gradient_ascent')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Redirect stdout to devnull
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info messages
    os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings

    # Freeze the model
    ascent_model = model
    for layer in ascent_model.layers:
        layer.trainable = False
    
    original_data = []
    updated_data = []
    losses_list = []
    outputs_list = []
    for i in range(num_trials):
        if random_seed:
            seed = np.random.randint(0, 100000)
        else:
            seed = 0
        sampled_input_data = subsample(input_data, sample_size = 1, seed=seed+i)
        prepped_input_data = prep_data_trainable(sampled_input_data, only_signal)
        updated_dat, losses, outputs = gradient_ascent_train(prepped_input_data, ascent_model, iterations, step_size, label, only_signal)
        losses_list.append(losses)
        outputs_list.append(outputs)
        original_data.append(sampled_input_data)
        updated_data.append(updated_dat)
    
    # Plotting Results
    plot_gradient_training_curves(num_trials, outputs_list, losses_list, folder_path)
    plot_gradient_updated_images(num_trials, original_data, updated_data, folder_path)
    if not only_signal:
        plot_gradient_updated_scalars(num_trials, original_data, updated_data, folder_path)

    # Restore stdout
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['PYTHONWARNINGS'] = 'default'
    gradient_output_data = losses_list, outputs_list, original_data, updated_data
    return gradient_output_data

#######################
### Other Functions ###
#######################
def plot_gene_subset_predicted_expression(input_data_test_up, input_data_test_down, best_model):
    """
    Plots a histogram of the distributions of the predicted expression for the gene subset data.

    Args:
        input_data_up (np.array): The input data for the up-regulated genes.
        input_data_down (np.array): The input data for the down-regulated genes.
        best_model (Model): The Keras model.
    """

    # Redirect stdout to devnull
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info messages
    os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings

    pred_list_up = []
    for i in range(100):
        input_data_temp = [input_data_test_up[0][i][np.newaxis], input_data_test_up[1][i][np.newaxis], input_data_test_up[2][i][np.newaxis], input_data_test_up[3][i][np.newaxis]]
        pred = best_model.predict(input_data_temp, verbose = 0)
        pred_list_up.append(pred[0, 0])

    pred_list_down = []
    for i in range(100):
        input_data_temp = [input_data_test_down[0][i][np.newaxis], input_data_test_down[1][i][np.newaxis], input_data_test_down[2][i][np.newaxis], input_data_test_down[3][i][np.newaxis]]
        pred = best_model.predict(input_data_temp, verbose = 0)
        pred_list_down.append(pred[0, 0])

    plt.hist(pred_list_up, bins = np.arange(0, 10, 0.5), alpha = 0.5)
    #plt.title('Predicted Expression Levels for Up-Regulated Genes')
    plt.hist(pred_list_down, bins = np.arange(0, 10, 0.5), alpha = 0.5)
    plt.title('Predicted Expression Levels for Up and Down-Regulated Genes')
    plt.legend(['Up-Regulated', 'Down-Regulated'])
    plt.xlabel('Predicted Expression Level')
    plt.show()

    # Restore stdout
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['PYTHONWARNINGS'] = 'default'