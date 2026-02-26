"""
interpretation_utils.py
v1.1.1, 9/10/2024
Author: Alexander Netzley, anetzley@fredhutch.org
Updates: Robert Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This module provides all the function code for Proteus interpretability analysis,
including kernel visualization, SHAP, and gradient ascent.
"""

import os
import sys
import shap
import logomaker
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Add parent directory to sys.path
sys.path.append("..")

# Import custom modules
from models.models import get_model_instance
from proteus_utils.compile_sample_data import *
from proteus_utils.proteus_utils import *

# define a custom two-color colormap for SHAP analysis
# Convert HEX to RGB
blue_rgb = (0x00 / 255, 0x8a / 255, 0xfb / 255)  # HEX #008afb
red_rgb = (0xff / 255, 0x00 / 255, 0x51 / 255)   # HEX #ff0051

colors = [blue_rgb,  # Blue at the lowest value
          (0, 0, 0),  # Black at 0
          red_rgb]    # Red at the highest value
n_bins = 100  # Number of bins for interpolation
cmap_name = 'blue_black_red'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

triton_profiles_ids = ['GC-Corrected Depth', 'Fragment End Coverage', 'Nucleosome Center Profile', 'Fragment Short:Long Ratio', 'Fragment Length Diversity', 'Fragment Length Entropy', 'Peak Calls', 'Adenine', 'Cytosine', 'Guanine', 'Tyrosine']
triton_profiles_seq_ids = ['GC-Corrected Depth', 'Fragment End Coverage', 'Nucleosome Center Profile', 'Fragment Short:Long Ratio', 'Fragment Length Diversity', 'Fragment Length Entropy', 'Peak Calls', 'Reference Sequence']
triton_scalars_ids = ['fragment-diversity', 'fragment-entropy', 'fragment-mad', 'fragment-mean', 'fragment-median', 'fragment-ratio', 'fragment-stdev', 'mean-depth', 'np-amplitude', 'np-period', 'np-score', 'var-ratio' ]
ref_scalars_ids = ['num_exons', 'transcript_length', 'mRNA_length', 'tumor_fraction']

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
def get_data(origin_filepath, regenerate_data=False, generate_num_samples=None, save_filepath=None, gene_list_filepath=None, subset_data=None):
    """
    Load the data from a specified path, or regenerate the data if specified.

    Args:
        origin_filepath (str): The path to the data (either a .tfrecord if regenerate_data=True, or .npz if False).
        regenerate_data (bool, optional): Whether to regenerate the data. (default: False)
        regenerate_num_samples (int, optional): The number of samples to generate. (default: None)
        save_filepath (str, optional): The path to save the data as a .npz. (default: None)
        gene_list_filepath (str (OR list), optional): The path to the gene list .txt file OR the gene list iteslf. (default: None)

    Returns:
        input_data (np.array): The loaded or regenerated data.
    """
    if regenerate_data or not os.path.exists(save_filepath):
        #Generate the data
        print(f'Generating new data, num_samples: {generate_num_samples}')
        ds = get_raw_tf_dataset(origin_filepath)
        if gene_list_filepath is not None:
            ds = get_gene_subset(ds, gene_list_filepath)
            generate_num_samples = None
        elif subset_data is not None:
            filtered_elements = []
            count_total = 0
            for element in ds:  # direct Python iteration
                count_total += 1
                name = element['sample_name'].numpy().decode('utf-8')
                if subset_data == 'pdx':
                    # Keep samples whose name starts with any of these prefixes
                    if any(name.startswith(prefix) for prefix in ["LuCaP", "DIVE", "LU", "SCLC", "MSK"]):
                        filtered_elements.append(element)
                elif subset_data == 'healthy':
                    if name.startswith("BP112"):
                        filtered_elements.append(element)
                else:  # Mixtures
                    if name.startswith("PHT"):
                        filtered_elements.append(element)

            if filtered_elements:
                print(f"{len(filtered_elements)} out of {count_total} matched the criteria.")
                # Build a new TF dataset from the filtered elements
                ds = tf.data.Dataset.from_generator(
                    lambda: (x for x in filtered_elements),
                    output_signature={
                        key: tf.TensorSpec(element[key].shape, element[key].dtype)
                        for key in element.keys()
                    }
                )
            else:
                print("Warning: no elements matched the filtering criteria.")
            generate_num_samples = None
        input_data = tf_ds_np_array(ds, generate_num_samples)
        #Save the data
        save_npz(input_data, save_filepath)
    else:
        #Load the data
        print(f'Loading existing data... ({save_filepath})')
        input_data = load_npz(save_filepath)

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
    if num_elements is None:
        num_elements = num_elements = sum(1 for _ in ds)
        print(f"Taking all {num_elements} elements from the dataset.")

    ds = ds.take(num_elements)
    TSS_image_list = [None] * num_elements
    TSS_scalar_list = [None] * num_elements
    GB_scalar_list = [None] * num_elements
    ref_scalar_list = [None] * num_elements
    tfx_scalar_list = [None] * num_elements
    labels_list = [None] * num_elements

    for i, element in enumerate(ds):
        TSS_image_list[i] = element['TSS_images'].numpy()
        TSS_scalar_list[i] = element['TSS_scalars'].numpy()
        GB_scalar_list[i] = element['GB_scalars'].numpy()
        ref_scalar_list[i] = element['ref_scalars'].numpy()
        tfx_scalar_list[i] = element['TFX_label'].numpy()
        labels_list[i] = element['tumor_exp'].numpy()

    TSS_image_stack = np.stack(TSS_image_list)
    TSS_scalar_stack = np.stack(TSS_scalar_list)
    GB_scalar_stack = np.stack(GB_scalar_list)
    ref_scalar_stack = np.stack(ref_scalar_list)
    tfx_scalar_stack = np.stack(tfx_scalar_list)
    labels_stack = np.stack(labels_list)

    input_data = [TSS_image_stack, TSS_scalar_stack, GB_scalar_stack, ref_scalar_stack, tfx_scalar_stack, labels_stack]
    return input_data


def normalize(x):
    return (x - np.mean(x, axis = 0)[np.newaxis]) / np.std(x, axis = 0)[np.newaxis]


def subsample(input_data, sample_size=20000, seed=None, restriction='all_genes', min_depth=10.0):
    """
    Generates a random subsample of the data.

    Args:
        input_data (np.array): The incoming numpy array.
        sample_size (int, optional): The size of the sample.
        seed (int, optional): The random seed for reproducibility. (default: None)
        restriction (str, optional): The restriction type for the subsample, ['all_genes', 'active_genes', 'inactive_genes']. (default: 'all_genes')
            N.B. 'active_genes' and 'inactive_genes' are determined by log2(CPM+1) >= 2 or < 2, respectively.
        min_depth (float, optional): The minimum depth for a gene to be considered. (default: 0.0) within tfrecords
            mean-depth is encoded as log(fragment depth) so the threshold is also logged.

    Returns:
        input_data_sample: The sampled data with sample_size elements.
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(200)

    if min_depth > 0.0:
        min_depth = np.log(min_depth)

    # Restrict based on active or inactive genes
    if restriction == 'active_genes':
        valid_indices = np.where((input_data[1][:, 7] > min_depth) & (input_data[5] >= 2.0))[0]  # Active genes (log2(CPM+1) >= 1)
    elif restriction == 'inactive_genes':
        valid_indices = np.where((input_data[1][:, 7] > min_depth) & (input_data[5] < 2.0))[0]  # Inactive genes (log2(CPM+1) < 1)
    else:
        valid_indices = np.where(input_data[1][:, 7] > min_depth)[0]  # No restriction (all genes)

    # Ensure sample_size does not exceed the number of valid_indices
    if sample_size > len(valid_indices):
        print(f"Warning: sample_size ({sample_size}) is greater than the number of valid_indices ({len(valid_indices)}). Using all valid_indices.")
        sample_size = len(valid_indices)

    # Sample from the valid indices
    sample_indices = np.random.choice(valid_indices, sample_size, replace=False)

    # Subsample the input data based on the selected indices
    input_data_sample = [input_data[i][sample_indices] for i in range(len(input_data))]

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
    tfx_scalar_stack = input_data[4]
    labels_stack = input_data[5]
    np.savez(filepath, TSS_image_stack = TSS_image_stack, TSS_scalar_stack = TSS_scalar_stack, GB_scalar_stack = GB_scalar_stack, ref_scalar_stack = ref_scalar_stack, tfx_scalar_stack = tfx_scalar_stack, labels_stack = labels_stack)

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
    (Requires num_filters to be at least 8, and a multiple of 8)

    Args:
        model (Model): The Keras model.
        folder_path (str, optional): The base folder location to save the figure. (default: None)
    """
    conv_layer_name = [layer.name for layer in model.layers if layer.name.startswith('conv2d')]
    actg_conv_layer = model.get_layer(name=conv_layer_name[0])
    weights = actg_conv_layer.get_weights()[0]
    num_filters = actg_conv_layer.filters

    subplot_height = int(num_filters / 8)
    #print(weights.shape)
    fig, axs = plt.subplots(subplot_height, 8, figsize=(15, 7))
    for j in range(subplot_height):
        for i in range(8):
            axs[j, i].imshow(weights[:, :, 0, 8*j + i], cmap='viridis')
            axs[j, i].set_yticks(np.arange(4))
            axs[j, i].set_yticklabels(['A', 'C', 'G', 'T'])
    cbar = fig.colorbar(axs[0, 0].imshow(weights[:, :, 0, 0], cmap='viridis'), ax=axs, orientation='vertical', fraction=0.05)
    # Add in a label on the y axis of each plot
    cbar.set_label('Weight')
    fig.suptitle(f'Nucleotide Conv Filters ({num_filters})', fontsize=16)
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
    triton_signals = triton_profiles_seq_ids[:-1]

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
    Plots all of the convolutional filters for the 3d signal combination layer.

    Args:
        model (Model): The Keras model.
        folder_path (str, optional): The base folder location to save the figure. (default: None)
    """
    ## Visualizing 1 kernel from the depthwise 2d layer
    conv_layer_name = [layer.name for layer in model.layers if layer.name.startswith('conv3d')]
    conv3d_layer = model.get_layer(name=conv_layer_name[0])
    weights = conv3d_layer.get_weights()[0]
    #print(weights.shape)

    #Hardcoded to index at 0
    fig, axs = plt.subplots(4, 4, figsize=(25,5))
    for j in range(4):
        for i in range(4):
            axs[j, i].imshow(np.squeeze(weights[0, :, :, :, 4*j + i]), cmap='viridis')
            axs[j, i].set_yticks(np.arange(8))
            if i == 0:
                axs[j, i].set_yticklabels(triton_profiles_seq_ids)
            else:
                axs[j, i].set_yticklabels([])
    cbar = fig.colorbar(axs[0, 0].imshow(np.squeeze(weights[0, :, :, :, 0]), cmap='viridis'), ax=axs, orientation='vertical', fraction=0.05)
    # Add in a label on the y axis of each plot
    #plt.subplots_adjust(hspace=0.5)
    cbar.set_label('Weight')
    fig.suptitle(f'Depthwise-Conv Filters ({(j+1)*(i+1)} of {weights.shape[4]})', fontsize=16)
    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'signal_combination_conv3d_filters.png'))
    else:
        plt.show()


def plot_all_conv2d_filters(model, folder_path=None):
    """
    Plots all conv2d filters from the model.
    Each filter displays WIDTH (first dimension) on x-axis and HEIGHT (second dimension) on y-axis.
    
    Args:
        model (Model): The Keras model.
        folder_path (str, optional): Where to save the figures.
    """
    # Get all conv2d layer names
    conv_layer_names = [layer.name for layer in model.layers if 'conv2d' in layer.name]
    
    for layer_name in conv_layer_names:
        # Get weights
        conv2d_layer = model.get_layer(name=layer_name)
        weights = conv2d_layer.get_weights()[0]  # Shape: [WIDTH, HEIGHT, IN_CHANNELS, OUT_FILTERS]
        filter_shape = weights.shape
        num_filters = filter_shape[3]  # Number of filters = last dimension
        
        # Get dilation info
        dilation_rate = getattr(conv2d_layer, 'dilation_rate', (1, 1))
        if isinstance(dilation_rate, int):
            dilation_rate = (dilation_rate, dilation_rate)
        
        # Determine grid layout
        if num_filters <= 16:
            rows, cols = 4, 4
        elif num_filters <= 32:
            rows, cols = 4, 8
        elif num_filters <= 64:
            rows, cols = 8, 8
        else:
            cols = 8
            rows = int(np.ceil(num_filters / cols))
        
        # Check for special handling (64 output channels)
        is_deep_filter = num_filters == 64 and filter_shape[1] == 128
        print(f"Filter shape: {filter_shape}, Special handling: {is_deep_filter}")
        
        # Create figure
        fig = plt.figure(figsize=(cols*2.5, rows*2.2))
        
        # Create a grid with extra space for colorbar
        if is_deep_filter:
            gs = fig.add_gridspec(rows, cols+1, width_ratios=[1]*cols + [0.05])
        else:
            gs = fig.add_gridspec(rows, cols+1, width_ratios=[1]*cols + [0.05])
        
        # Plot each filter
        for i in range(num_filters):
            row, col = i // cols, i % cols
            
            # Create subplot
            if is_deep_filter:
                cell_gs = gs[row, col].subgridspec(1, 12)
                ax_filter = fig.add_subplot(cell_gs[0, :11])
                ax_activation = fig.add_subplot(cell_gs[0, 11:])
            else:
                ax_filter = fig.add_subplot(gs[row, col])
            
            # Extract filter (first two dimensions only)
            filter_img = weights[:, :, 0, i]
                
            # Plot the filter
            im = ax_filter.imshow(filter_img.T, cmap='viridis', aspect='auto', interpolation='none')
            ax_filter.set_title(f'Filter {i+1}', fontsize=9)
            
            # Special handling for 64 output channels
            if is_deep_filter:
                # Add white lines every 16 rows
                for j in range(1, 8):
                    ax_filter.axhline(y=j*16-0.5, color='black', linestyle='-', linewidth=0.5)
                
                # Add channel labels
                y_positions = np.arange(8)*16 + 8
                if col == 0:
                    ax_filter.set_yticks(y_positions)
                    ax_filter.set_yticklabels(triton_profiles_seq_ids, fontsize=8)
                else:
                    ax_filter.set_yticks([])
                
                # Calculate mean absolute activation per channel
                mean_abs_weights = []
                for j in range(8):
                    channel_slice = filter_img[:, j*16:(j+1)*16]
                    if channel_slice.size > 0:
                        mean_abs = np.mean(np.abs(channel_slice))
                    else:
                        mean_abs = 0.0
                    mean_abs_weights.append(mean_abs)
                
                # Normalize weights to sum to 1
                sum_abs_weights = sum(mean_abs_weights)
                if sum_abs_weights > 0:
                    mean_abs_weights = np.array(mean_abs_weights) / sum_abs_weights
                
                # Plot mean absolute weights bar chart
                ax_activation.barh(y_positions, mean_abs_weights, height=14, color='grey', alpha=0.7)
                ax_activation.set_yticks([])
                ax_activation.set_xticks([])
                ax_activation.set_ylim(ax_filter.get_ylim())
                ax_activation.set_title("AMS", fontsize=6)
            else:
                ax_filter.set_yticks([])
            
            # No x-axis labels
            ax_filter.set_xticks([])
            
            # Show dilation indicator if needed
            if dilation_rate != (1, 1):
                ax_filter.text(0.95, 0.05, f"dil={dilation_rate}",
                               transform=ax_filter.transAxes,
                               fontsize=6, ha='right', va='bottom',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Hide unused subplots
        for i in range(num_filters, rows*cols):
            row, col = i // cols, i % cols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
        
        # Add colorbar to the right of all subplots
        cax = fig.add_subplot(gs[:, -1])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Weight Value')
        
        # Create title
        title = f'{layer_name} Filters ({num_filters} filters)'
        subtitle = f"Kernel: {filter_shape[0]}×{filter_shape[1]}"
        if dilation_rate != (1, 1):
            subtitle += f", Dilation: {dilation_rate[0]}×{dilation_rate[1]}"
            
        fig.suptitle(f"{title}\n{subtitle}", fontsize=14)
        
        # Adjust layout
        plt.subplots_adjust(top=0.9, right=0.95, hspace=0.3, wspace=0.3)
        
        # Save or display
        if folder_path is not None:
            os.makedirs(folder_path, exist_ok=True)
            plt.savefig(os.path.join(folder_path, f'{layer_name}_filters.png'), dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


################################
### Motif Plotting Functions ###
################################

def compute_PWM(model, data, save_filepath=None, kernel_size=6):
    """
    Wrapper function to compute and visualize PWM for the model 1st layer nucleotide conv filters.

    Args:
        model (Model): The Keras model.
        data (np.array): The input data (or a subset) for computing activations
        save_filepath (str, optional): The base folder location to save the figures. (default: None)
    """

    def kernel_2_PWM(kernel):
        #helper function to split the nucleotide sequnce up into the 1998 length 3 windows
        nucleotide_windows = []
        for i in range(nucleotides.shape[0] - (kernel_size - 1)):
            nucleotide_windows.append(nucleotides[i:i+kernel_size])

        activated_sequences = np.zeros((kernel_size, 4, 0))
        for window in nucleotide_windows:
            activation = np.sum(window.T * kernel)
            if activation > 0:
                activated_sequences = np.concatenate((activated_sequences, window[:, :, np.newaxis]), axis=2)

        activated_sequences_prob = np.mean(activated_sequences, axis=2)
        return activated_sequences_prob

    conv_layer_name = [layer.name for layer in model.layers if layer.name.startswith('conv2d')]
    actg_conv_layer = model.get_layer(name=conv_layer_name[0])
    weights = actg_conv_layer.get_weights()[0]
    num_filters = actg_conv_layer.filters
    subplot_height = int(num_filters / 8)
    pwm_filters_alldata = np.zeros((kernel_size, 4, num_filters, 0))

    #Calculate background frequencies of each nucleotide
    nuc_freqs = np.zeros((4, 0))
    for data_entry in range(data[0].shape[0]):
        nucleotides_freq = np.mean(data[0][data_entry, :, 7:].squeeze(), axis=0)
        nuc_freqs = np.concatenate((nuc_freqs, nucleotides_freq[:, np.newaxis]), axis=1)
    background_nucleotide_frequency = np.mean(nuc_freqs, axis=1)

    #Loop through all data to compute PWM scores
    for data_entry in range(10):
        nucleotides = data[0][data_entry, :, 7:].squeeze()

        pwm_filters = np.zeros((kernel_size, 4, 0))
        #loop through and apply all 16 kernels
        for filt in range(num_filters):
            conv_filter = weights[:, :, 0, filt].squeeze()
            activated_sequences_prob = kernel_2_PWM(conv_filter)
            pwm_filters = np.concatenate((pwm_filters, activated_sequences_prob[:, :, np.newaxis]), axis=2)

        pwm_filters_alldata = np.concatenate((pwm_filters_alldata, pwm_filters[:, :, :, np.newaxis]), axis=3)

    pwm_filters_average = np.mean(pwm_filters_alldata, axis=3)

    subplot_height = int(num_filters / 8)
    fig, axs = plt.subplots(subplot_height, 8, figsize=(15, 7))
    for j in range(subplot_height):
        for i in range(8):
            axs[j, i].imshow(pwm_filters_average[:, :, 8*j + i].T, cmap='viridis')
            axs[j, i].set_yticks(np.arange(4))
            axs[j, i].set_yticklabels(['A', 'C', 'G', 'T'])
    cbar = fig.colorbar(axs[0, 0].imshow(pwm_filters_average[:, :, 0].T, cmap='viridis'), ax=axs, orientation='vertical', fraction=0.05)
    # Add in a label on the y axis of each plot
    cbar.set_label('Weight')
    fig.suptitle(f'Nucleotide PWM ({num_filters})', fontsize=16)

    if save_filepath is not None:
        plt.savefig(os.path.join(save_filepath, 'nucleotide_motifs_PWM.png'))
    else:
        plt.show()

    return pwm_filters_average, background_nucleotide_frequency

def export_PWM_to_MEME(pwm_filters_average, background_nucleotide_frequency, save_filepath = None):
    """
    Export the PWM to a MEME formatted file.

    Args:
        pwm_filters_average (np.array): The averaged PWM filters for the nucleotide layer.
        background_nucleotide_frequency (np.array): The background nucleotide frequencies.
        save_filepath (str, optional): The base folder location to save the MEME file. (default: None)
    """
    # Background frequencies (for A, C, G, T)
    background_frequencies = {
        'A': background_nucleotide_frequency[0],
        'C': background_nucleotide_frequency[1],
        'G': background_nucleotide_frequency[2],
        'T': background_nucleotide_frequency[3]
    }

    # Create MEME formatted string
    meme_str = "MEME version 4\n\n"
    meme_str += "ALPHABET= ACGT\n\n"
    meme_str += "strands: + -\n\n"
    meme_str += "Background letter frequencies\n"
    for base, freq in background_frequencies.items():
        meme_str += f"{base} {freq:.4f} "
    meme_str += "\n\n"

    # Add each PWM to the MEME string
    for i in range(pwm_filters_average.shape[2]):
        pwm = pwm_filters_average[:, :, i].squeeze()
        motif_name = f"kernel_{i+1}"
        motif_length = pwm.shape[0]
        
        meme_str += f"MOTIF {motif_name}\n"
        meme_str += f"letter-probability matrix: alength= 4 w= {motif_length}\n"
        
        for col in range(motif_length):
            for row in range(4):
                meme_str += f"{pwm[col, row]:.6f} "
            meme_str += "\n"
        meme_str += "\n"

    # Write to a file
    if save_filepath is not None:
        with open(os.path.join(save_filepath, 'pwm_motif.meme'), "w") as f:
            f.write(meme_str)

def plot_nucleotide_motifs(model, pwm_filters_average, save_filepath=None):
    """
    Plot the nucleotide motifs for the model.

    Args:
        model (Model): The Keras model.
        pwm_filters_average (np.array): The averaged PWM filters for the nucleotide layer.
        save_filepath (str, optional): The base folder location to save the figures. (default: None)
    """
    conv_layer_name = [layer.name for layer in model.layers if layer.name.startswith('conv2d')]
    actg_conv_layer = model.get_layer(name=conv_layer_name[0])
    num_filters = actg_conv_layer.filters

    subplot_height = int(num_filters / 2)
    # Use constrained_layout instead of tight_layout
    fig_1, axs_1 = plt.subplots(subplot_height, 2, figsize=(4, 16), 
                               constrained_layout=True, 
                               gridspec_kw={'wspace': 0.05})
                               
    fig_2, axs_2 = plt.subplots(subplot_height, 2, figsize=(4, 16), 
                               constrained_layout=True,
                               gridspec_kw={'wspace': 0.05})
                               
    for j in range(subplot_height):
        for i in range(2):
            filt_df = pd.DataFrame(pwm_filters_average[:, :, 2*j + i].squeeze(),columns=['A','C','G','T'])
            filt_df_info = logomaker.transform_matrix(filt_df,from_type='probability',to_type='information')
            logo = logomaker.Logo(filt_df_info, ax = axs_1[j,i])
            axs_1[j, i].set_ylim(0, 1)
            if i == 0:  # Only show y-axis labels and ticks on the first column
                axs_1[j, i].set_yticks([0, 0.5, 1])
                axs_1[j, i].set_ylabel('bits')
            else:
                axs_1[j, i].set_yticks([])
            axs_1[j, i].set_xticks([])  # Do not show x-axis ticks/labels
            axs_1[j, i].set_title(f'kernel {2*j + i + 1}')

            logo = logomaker.Logo(filt_df, ax = axs_2[j,i])
            axs_2[j, i].set_ylim(0, 1)
            if i == 0:  # Only show y-axis labels and ticks on the first column
                axs_2[j, i].set_yticks([0, 0.5, 1])
                axs_2[j, i].set_ylabel('probability')
            else:
                axs_2[j, i].set_yticks([])
            axs_2[j, i].set_xticks([])  # Do not show x-axis ticks/labels
            axs_2[j, i].set_title(f'kernel {2*j + i + 1}')

    fig_1.suptitle('PWMs in Information Content', fontsize=16)
    fig_2.suptitle('PWMs in Probability', fontsize=16)

    if save_filepath is not None:
        fig_1.savefig(os.path.join(save_filepath, 'nucleotide_motifs_PWM_IC.png'))
        fig_2.savefig(os.path.join(save_filepath, 'nucleotide_motifs_PWM_Prob.png'))
    else:
        plt.show()

def plot_motifs(model, data, save_filepath=None):
    """
    Wrapper function to compute and visualize PWM for the model 1st layer nucleotide conv filters.

    Args:
        model (Model): The Keras model.
        data (np.array): The input data (or a subset) for computing activations
        save_filepath (str, optional): The base folder location to save the figures. (default: None)
    """
    pwm_filters_average, background_nucleotide_frequency = compute_PWM(model, data, save_filepath)
    export_PWM_to_MEME(pwm_filters_average, background_nucleotide_frequency, save_filepath)
    plot_nucleotide_motifs(model, pwm_filters_average, save_filepath)


def kernel_visualization(model, data, save_filepath=None):
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
    plot_all_conv2d_filters(model, folder_path)
    plot_motifs(model, data, folder_path)

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
    images_t = np.transpose(shap_values[0], (2,0,1))
    means = np.mean(np.abs(images_t), axis=1)
    plt.clf()
    cols = triton_profiles_ids
    plt.figure(figsize=(10,4))
    plt.imshow(means, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar for reference
    plt.title('Mean Absolute SHAP Values for TSS Images')
    plt.colorbar()
    plt.yticks(np.arange(len(cols)), cols)

    # Show the plot
    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'TSS_Images_SHAP.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


def plot_shap_images_nucleotide(shap_values, folder_path=None):
    '''
    Plots the mean absolute SHAP values for the TSS Images, with the nucleotide SHAPs summed to one signal.

    Args:
        shap_values (np.array): The SHAP values.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    '''
    images_t = np.transpose(shap_values[0], (2,0,1))
    new_images_t = images_t.copy()
    summed_bp_shap = np.sum(new_images_t[7:11], axis=0)
    new_images_t[7] = summed_bp_shap
    new_images_t = new_images_t[:8]
    new_means = np.mean(np.abs(new_images_t), axis=1)

    plt.clf()
    new_cols = triton_profiles_seq_ids
    plt.figure(figsize=(10,4))
    plt.imshow(new_means, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar for reference
    plt.title(f'Mean Absolute SHAP Values for TSS Images Nucleotide Summed')
    plt.colorbar()
    plt.yticks(np.arange(len(new_cols)), new_cols)

    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'TSS_Images_Nucleotide_SHAP.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


def plot_shap_images_individual(shap_values, folder_path=None):
    '''
    Plots the median SHAP values for the TSS Images, with the nucleotide SHAPs summed to one signal.
    Additionally, each signal is it own subplot, and the color scales are now local to each individual signal.

    Args:
        shap_values (np.array): The SHAP values.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    '''
    # process SHAP values into mean model impact
    images_t = np.transpose(shap_values[0], (2,0,1))
    new_images_t = images_t.copy()
    concat_bp_shap = np.sum(new_images_t[7:11], axis=0)
    new_images_t[7] = concat_bp_shap
    new_images_t = new_images_t[:8]
    new_means = np.mean(np.abs(new_images_t), axis=1)
    new_cols = triton_profiles_seq_ids
    _, axs = plt.subplots(8, 1, figsize=(10,4), gridspec_kw={'hspace': 0.1})
    for j in range(8):
        axs[j].imshow(new_means[j, :][np.newaxis], cmap='viridis', aspect='auto', interpolation='nearest')
        axs[j].set_yticks([])
        axs[j].set_ylabel(new_cols[j], rotation=0, labelpad=10, ha='right')
        # Remove x-ticks and labels for all but the bottom subplot
        if j < 7:
            axs[j].set_xticks([])
        else:
            axs[j].set_xticks(np.linspace(0, new_means[j, :].size, 9))
            axs[j].set_xticklabels(np.linspace(-1000, 1000, 9).astype(int))

    plt.suptitle(f'Mean Absolute SHAP Values for TSS Images Nucleotide Summed (Color Scale Local)')
    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'TSS_Images_Nucleotide_SHAP_Individual.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


def plot_signal_polarity(shap_values, test_values, folder_path=None):
    '''
    Plots the signal impact-polarity, where:
    - The magnitude is the mean absolute SHAP value at each point.
    - The color represents polarity, where polarity of -1 means all values above 1 drive the model output down, 
      and all values below 1 drive the output up, and vice versa for a polarity of 1.

    Args:
        shap_values (np.array): The SHAP values.
        test_values (np.array): The test values corresponding to the same feature space.
        folder_path (str, optional): The base folder location to save the figures. (default: None)
    '''
    print(shap_values[0].shape)
    print(test_values[0].shape)
    # Extract the SHAP values for the signal inputs (assuming they are the first input).
    shap_images = np.transpose(shap_values[0], (2, 0, 1))  # Shape: (signals, samples, length)
    test_images = np.transpose(test_values[0], (2, 0, 1))  # Corresponding test values for the signals.

    # Initialize an array to store the polarity-impact scores.
    impact_polarity = np.zeros((shap_images.shape[0], shap_images.shape[2]))

    # Calculate the mean absolute SHAP value (impact) and the polarity.
    for j in range(11):  # Loop over each signal.
        """ shape coming in is (signals, batch, 2000)"""
        above_mean = test_images[j, :, :] > 1.0  # Boolean mask for values above the mean.

        """
        RED if higher signal = higher output
        BLUE if lower signal = higher output
        """
        # Determine the SHAP polarity:
        # +1: SHAP > 0 for values above mean, SHAP < 0 for values below mean
        # -1: SHAP < 0 for values above mean, SHAP > 0 for values below mean
        pos_above = (shap_images[j, :, :] > 0) & above_mean # increased depth = increased output
        neg_below = (shap_images[j, :, :] < 0) & ~above_mean # decreased depth = decreased output
        pos_below = (shap_images[j, :, :] > 0) & ~above_mean # decreased depth = increased output
        neg_above = (shap_images[j, :, :] < 0) & above_mean # increased depth = decreased output

        # Calculate polarity score based on how consistently SHAP values correspond to increases or decreases.
        polarity = (np.sum(pos_above, axis=0) + np.sum(neg_below, axis=0) - np.sum(pos_below, axis=0) - np.sum(neg_above, axis=0)) / shap_images.shape[1]
        impact_polarity[j, :] = polarity * np.median(np.abs(shap_images[j, :, :]), axis=0)  # Combine with mean abs SHAP.
        
        # Scale the polarity-impact scores to be between -1 and +1.
        positive_values = impact_polarity[j, impact_polarity[j, :] > 0]
        negative_values = impact_polarity[j, impact_polarity[j, :] < 0]

        if positive_values.size > 0:
            max_positive = np.max(positive_values)
            impact_polarity[j, impact_polarity[j, :] > 0] /= max_positive
        if negative_values.size > 0:
            min_negative = np.min(negative_values)
            impact_polarity[j, impact_polarity[j, :] < 0] /= np.abs(min_negative)

    # Plot the polarity-impact score for each signal.
    fig, axs = plt.subplots(11, 1, figsize=(10, 4), gridspec_kw={'hspace': 0.1})
    for j in range(11):
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = axs[j].imshow(impact_polarity[j, :].reshape(1, -1), cmap=custom_cmap, aspect='auto', interpolation='nearest', norm=norm)
        axs[j].set_yticks([])
        axs[j].set_ylabel(triton_profiles_ids[j], rotation=0, labelpad=10, ha='right')

        # Remove x-ticks and labels for all but the bottom subplot
        if j < 10:
            axs[j].set_xticks([])
        else:
            axs[j].set_xticks(np.linspace(0, impact_polarity[j, :].size, 9))
            axs[j].set_xticklabels(np.linspace(-1000, 1000, 9).astype(int))

    plt.suptitle(f'SHAP Polarity-Impact (Color Scale Local)')
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.set_label('Direction and impact of signal:output relationship')

    # Save or display the plot
    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'TSS_Images_Polarity-Impact.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


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
    images_t = np.transpose(shap_values[0], (2,0,1))
    new_images_t = images_t.copy()
    summed_bp_shap = np.sum(new_images_t[7:11], axis=0)
    new_images_t[7] = summed_bp_shap
    new_images_t = new_images_t[:8]

    # Plot mean abs SHAP Values for the TSS Scalars, GB Scalars, and Reference Scalars, and summed signals
    TSS_image_cols = triton_profiles_seq_ids
    TSS_scalar_cols = triton_scalars_ids
    ref_scalar_cols = ref_scalars_ids
    
    TSS_images_summed = np.median(np.sum(new_images_t, axis = 2), axis = 1)[:, np.newaxis]
    TSS_scalars = np.median(shap_values[0][1], axis = 0)[:, np.newaxis]
    GB_scalars = np.median(shap_values[0][2], axis = 0)[:, np.newaxis]
    ref_scalars = np.median(shap_values[0][3], axis = 0)[:, np.newaxis]
    tfx_scalars = np.median(shap_values[0][4], axis = 0)[:, np.newaxis]
    # combine tfx_scalars with ref_scalars
    ref_scalars = np.concatenate((ref_scalars, tfx_scalars), axis=0)

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
            im = axs[i].imshow(shap_values_summed[i][np.newaxis], cmap=custom_cmap, aspect='auto', interpolation='nearest', norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
        axs[i].imshow(shap_values_summed[i], cmap=custom_cmap, aspect='auto', interpolation='nearest', vmax=vmax, vmin=vmin)
        axs[i].set_yticks(np.arange(len(cols_list[i])), cols_list[i])
        axs[i].set_xticks([])
        axs[i].set_title(f'Median SHAP Values for {name_list[i]}')

    fig.subplots_adjust(wspace=5) 
    fig.colorbar(im, ax=axs.ravel().tolist())
    if include_images and scale_include_images:
        fig.suptitle(f'Median SHAP Values for All Inputs (Color Scale Global)', y=1.1)
    elif include_images and not scale_include_images:
        fig.suptitle(f'Median SHAP Values for All Inputs (Color Scale Global, Excluding TSS Images)', y=1.1)
    else:
        fig.suptitle(f'Median SHAP Values for Input Scalars (Color Scale Global)', y=1.1)

    if folder_path is not None:
        if include_images and scale_include_images:
            fig.savefig(os.path.join(folder_path, 'Global_SHAP_All_CS_Global.png'), bbox_inches='tight')
        elif include_images and not scale_include_images:
            fig.savefig(os.path.join(folder_path, 'Global_SHAP_All_CS_Global_Exc_Images.png'), bbox_inches='tight')
        else:
            fig.savefig(os.path.join(folder_path, 'Global_SHAP_Scalars.png'), bbox_inches='tight')
    else:
        fig.show()
    plt.clf()


def plot_shap_summary(shap_values, test_values, folder_path=None):
    """
    Creates a SHAP summary plot using a custom colormap for scalar features and saves or displays the plot.
    This function assumes 'feature_names' and 'custom_cmap' are available globally.

    Args:
        shap_values (np.array): Array of SHAP values for scalar features.
        folder_path (str, optional): Directory to save the plot image. If None, plot will be displayed.

    'TSS_images': tf.io.FixedLenFeature([2000, 11], tf.float32),
    'TSS_scalars': tf.io.FixedLenFeature([12], tf.float32),
    'GB_scalars': tf.io.FixedLenFeature([12], tf.float32),
    'ref_scalars': tf.io.FixedLenFeature([3], tf.float32),
    'TFX_label': tf.io.FixedLenFeature([], tf.float32),
    """
    # concatenate tfx_scalars with ref_scalars
    # print('shaps')
    # print(shap_values[0].shape)
    # print(shap_values[1].shape)
    # print(shap_values[2].shape)
    # print(shap_values[3].shape)
    # print(shap_values[4].shape)
    # print('tests')
    # print(test_values[0].shape)
    # print(test_values[1].shape)
    # print(test_values[2].shape)
    # print(test_values[3].shape)
    # print(test_values[4].shape)
    shap_values[4] = np.expand_dims(shap_values[4], axis=1)
    shap_values[3] = np.concatenate((shap_values[3], shap_values[4]), axis=1)
    test_values[3] = np.concatenate((test_values[3], test_values[4]), axis=1)
    shap_groups = ['TSS_Scalars', 'GB_Scalars', 'Reference_Scalars']  

    # Ensure the folder path exists if provided
    if folder_path and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Function to group and sort features
    def group_and_sort_features(shap_values, feature_names):
        fragment_indices = [i for i, name in enumerate(feature_names) if 'fragment-' in name]
        np_indices = [i for i, name in enumerate(feature_names) if 'np-' in name]
        other_indices = [i for i, name in enumerate(feature_names) if 'fragment-' not in name and 'np-' not in name]

        # Calculate absolute average SHAP values for sorting
        abs_avg_shap_values = np.abs(shap_values).mean(axis=0)

        # Sort each group based on the absolute average SHAP values
        fragment_indices = sorted(fragment_indices, key=lambda i: abs_avg_shap_values[i], reverse=True)
        np_indices = sorted(np_indices, key=lambda i: abs_avg_shap_values[i], reverse=True)
        other_indices = sorted(other_indices, key=lambda i: abs_avg_shap_values[i], reverse=True)

        # Combine the sorted indices
        sorted_indices = fragment_indices + np_indices + other_indices
        return sorted_indices

    # Iterate over each input feature's SHAP values
    for i, shap_value in enumerate(shap_values):
        if i not in {1, 2, 3}:
            continue
        if i == 1 or i == 2:
            feat_names = triton_scalars_ids
        else:
            feat_names = ref_scalars_ids

        # Group and sort features for TSS_Scalars and GB_Scalars
        if i in {1, 2}:
            sorted_indices = group_and_sort_features(shap_values[2], feat_names)  # Use GB_Scalars for sorting
            shap_value = shap_value[:, sorted_indices]
            test_values[i] = test_values[i][:, sorted_indices]
            feat_names = [feat_names[j] for j in sorted_indices]

        plt.figure()
        shap.summary_plot(shap_value, features=test_values[i], feature_names=feat_names, sort=False, show=False)
        plt.xlim(-2.0, 2.0)
        plt.title(f'SHAP Beeswarm Plot for {shap_groups[i-1]}')
        plt.xlabel("SHAP value (impact on model prediction) in log$_2$(CPM + 1)")

        # Save or show the plot
        if folder_path:
            plt.savefig(os.path.join(folder_path, f'Explainer_{shap_groups[i-1]}.png'), bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def shap_analysis(model, input_data, background_size, test_size, save_filepath=None, random_seed=False, output_index=0, batch_size=500):
    '''
    Wrapper function to perform SHAP analysis on the model.

    Args:
        model (Model): The Keras model.
        input_data (np.array): The input data.
        background_size (int): The size of the background data.
        test_size (int): The size of the test data.
        save_filepath (str, optional): The base folder location to save the figures. (default: None)
        random_seed (bool, optional): Whether to use a random seed. (default: False)
        output_index (int, optional): Which output to analyze for multi-output models. (default: 0)
        batch_size (int, optional): Batch size for SHAP value computation to avoid OOM. (default: 500)
    '''
    inputs = model.inputs
    outputs = model.outputs[output_index]  # Select just one of the outputs (tumor expression)
    single_output_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Rest of the function uses this single_output_model
    input_data[4] = input_data[4].reshape(-1, 1)

    for analysis in ['all_genes', 'active_genes', 'inactive_genes']:
        # Randomly Subsample the Data (background)
        if isinstance(background_size, int):
            print('Subsampling Background Data to Size:', background_size)
            input_data_background = subsample(input_data, sample_size = background_size, seed=random_seed, restriction=analysis)
        else:
            input_data_background = background_size

        print('Creating SHAP Explainer...')
        explainer = shap.GradientExplainer(single_output_model, input_data_background[:-1])  # Use the single output model

        print(f'Performing SHAP Analysis for {analysis}...')
        folder_path = None
        if save_filepath is not None:
            folder_path = os.path.join(save_filepath, f'shap_analysis/{analysis}')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        print(f'Subsampling Test Data to Size: {test_size} restricted to {analysis}')
        input_data_test = subsample(input_data, sample_size=test_size, seed=random_seed, restriction=analysis)

        # Process SHAP values in batches to avoid OOM
        num_samples = len(input_data_test[0])
        num_batches = int(np.ceil(num_samples / batch_size))
        print(f'Processing SHAP values in {num_batches} batches of size {batch_size}')
        
        # Initialize lists to collect batched results
        all_shap_values = None
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            batch_size_actual = end_idx - start_idx
            
            print(f'Processing batch {batch_idx+1}/{num_batches} (samples {start_idx} to {end_idx-1})')
            
            # Extract batch data
            batch_data = [x[start_idx:end_idx] for x in input_data_test[:-1]]
            
            # Get SHAP values for this batch
            batch_shap_values = explainer.shap_values(batch_data)
            batch_shap_values = [np.squeeze(s) for s in batch_shap_values]
            
            # Accumulate results
            if all_shap_values is None:
                all_shap_values = batch_shap_values
            else:
                # Append the new values to the accumulated values
                all_shap_values = [np.concatenate((all_values, batch_values), axis=0) 
                                   for all_values, batch_values in zip(all_shap_values, batch_shap_values)]
        
        # Use the accumulated SHAP values for visualization
        shap_values = all_shap_values
        print('Shape of shap_values:', [s.shape for s in shap_values])
        print('Shape of input_data_test:', [i.shape for i in input_data_test])

        # Plot polarity-impact scores across signals
        plot_signal_polarity(shap_values, input_data_test, folder_path=folder_path)

        # Plot shap summaries (force plot)
        plot_shap_summary(shap_values, input_data_test, folder_path=folder_path)


# def shap_analysis(model, input_data, background_size, test_size, save_filepath=None, random_seed=False, output_index=0):
#     '''
#     Wrapper function to perform SHAP analysis on the model.

#     Args:
#         model (Model): The Keras model.
#         input_data (np.array): The input data.
#         background_size (int): The size of the background data.
#         test_size (int): The size of the test data.
#         save_filepath (str, optional): The base folder location to save the figures. (default: None)
#         random_seed (bool, optional): Whether to use a random seed. (default: False)
#         output_index (int, optional): Which output to analyze for multi-output models. (default: 0)
#     '''
#     inputs = model.inputs
#     outputs = model.outputs[output_index]  # Select just one of the outputs (tumor expression)
#     single_output_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     # Rest of the function uses this single_output_model
#     input_data[4] = input_data[4].reshape(-1, 1)

#     for analysis in ['all_genes', 'active_genes', 'inactive_genes']:
#         # Randomly Subsample the Data (background)
#         if isinstance(background_size, int):
#             print('Subsampling Background Data to Size:', background_size)
#             input_data_background = subsample(input_data, sample_size = background_size, seed=random_seed, restriction=analysis)
#         else:
#             input_data_background = background_size

#         print('Creating SHAP Explainer...')
#         explainer = shap.GradientExplainer(single_output_model, input_data_background[:-1])  # Use the single output model

#         print(f'Performing SHAP Analysis for {analysis}...')
#         folder_path = None
#         if save_filepath is not None:
#             folder_path = os.path.join(save_filepath, f'shap_analysis/{analysis}')
#             if not os.path.exists(folder_path):
#                 os.makedirs(folder_path)
        
#         print(f'Subsampling Test Data to Size: {test_size} restricted to {analysis}')
#         input_data_test = subsample(input_data, sample_size=test_size, seed=random_seed, restriction=analysis)

#         # Execute the SHAP Analysis
#         shap_values = explainer.shap_values(input_data_test[:-1]) # Exclude the labels
#         shap_values = [np.squeeze(s) for s in shap_values]
#         print('Shape of shap_values:', [s.shape for s in shap_values])
#         print('Shape of input_data_test:', [i.shape for i in input_data_test])

#         # Plot polarity-impact scores across signals
#         plot_signal_polarity(shap_values, input_data_test, folder_path=folder_path)

#         # # Plot mean abs SHAP Values for the TSS Images
#         # plot_shap_TSS_Images(shap_values, folder_path=folder_path)

#         # # Plot mean abs SHAP Values for the TSS Images with the Nucleotides summed
#         # plot_shap_images_nucleotide(shap_values, folder_path=folder_path)

#         # # Plot mean abs SHAP Values for the TSS Images with the Nucleotides summed, each signal color scaled individually
#         # plot_shap_images_individual(shap_values, folder_path=folder_path)

#         # Plot global comparison of mean abs SHAP Values for all inputs, with the signals summed
#         # plot_shap_global(shap_values, include_images=True, scale_include_images=True, folder_path=folder_path)

#         # # Plot global comparison of mean abs SHAP Values for three scalars
#         # plot_shap_global(shap_values, include_images=False, folder_path=folder_path)

#         # Plot scalars with each color bar scaled locally
#         # plot_shap_scalars_individual(shap_values, folder_path=folder_path)

#         # Plot shap summaries (force plot)
#         plot_shap_summary(shap_values, input_data_test, folder_path=folder_path)

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
        fig.savefig(os.path.join(folder_path, 'gradient_ascent_training_curves.pdf'))
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
    cols = triton_profiles_seq_ids[:-1]
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
            fig.savefig(os.path.join(folder_path, f'gradient_ascent_images_trial_{trial}.pdf'))
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
    TSS_scalar_cols = triton_scalars_ids
    ref_scalar_cols = ref_scalars_ids
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
            fig.savefig(os.path.join(folder_path, f'gradient_ascent_scalars_trial_{trial}.pdf'))
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


""" Not using - just clearing space """

# def plot_shap_scalars_individual(shap_values, folder_path=None):
#     '''
#     Plots the mean absolute SHAP values for th three scalar inputs, with local colorbars for each.

#     Args:
#         shap_values (np.array): The SHAP values.
#         folder_path (str, optional): The base folder location to save the figures. (default: None)
#     '''
#     TSS_scalars = np.median(shap_values[0][1], axis = 0)[:, np.newaxis]
#     GB_scalars = np.median(shap_values[0][2], axis = 0)[:, np.newaxis]
#     ref_scalars = np.median(shap_values[0][3], axis = 0)[:, np.newaxis]
#     tfx_scalars = np.median(shap_values[0][4], axis = 0)[:, np.newaxis]
#     # combine tfx_scalars with ref_scalars
#     ref_scalars = np.concatenate((ref_scalars, tfx_scalars), axis=0)

#     TSS_scalar_cols = triton_scalars_ids
#     ref_scalar_cols = ref_scalars_ids

#     fig, axs = plt.subplots(1, 3, figsize=(12, 4))

#     # Define data and titles
#     data = [TSS_scalars, GB_scalars, ref_scalars]
#     titles = ['Median Values for TSS Scalars', 'Median Shap Values for GB Scalars', 'Median Values for reference Scalars']
#     cols = [TSS_scalar_cols, TSS_scalar_cols, ref_scalar_cols]

#     # Plot each subplot and add colorbar
#     for ax, d, t, c in zip(axs, data, titles, cols):
#         im = ax.imshow(d, cmap=custom_cmap, aspect='auto', interpolation='nearest', norm=TwoSlopeNorm(vmin=np.min(d), vcenter=0, vmax=np.max(d)))
#         ax.set_yticks(np.arange(len(c)), c)
#         ax.set_xticks([])
#         ax.set_title(t)
#         fig.colorbar(im, ax=ax)

#     fig.subplots_adjust(wspace=5) 
#     fig.suptitle(f'Median SHAP Values for Input Scalars (Color Scale Local)', y=1.1)

#     if folder_path is not None:
#         fig.savefig(os.path.join(folder_path, 'SHAP_Scalars_Individual.pdf'))
#     else:
#         fig.show()
#     plt.clf()
    
# def plot_shap_scalar_distributions(shap_values):
#     '''
#     Plots the dstribution of shap values across the different test data samples for TSS and GB Scalars.

#     Args:
#         shap_values (np.array): The SHAP values.
#     '''
#     TSS_scalar_cols = triton_scalars_ids
#     #plot the distribution of shap values for each scalar
#     plt.clf()
#     fig, axs = plt.subplots(3, 4, figsize=(10, 10))
#     for i in range(3):
#         for j in range(4):
#             axs[i, j].hist(shap_values[0][1][:, 4*i + j], bins=25)
#             axs[i, j].set_ylabel(f'{i}', rotation=0)
#             axs[i, j].set_title(TSS_scalar_cols[4*i + j])
#             axs[i, j].set_xlim(-0.5, 0.5)

#     fig.suptitle('TSS Scalars Shap Value Distributions', fontsize=16)
#     plt.show()

#     plt.clf()
#     fig, axs = plt.subplots(3, 4, figsize=(10, 10))
#     for i in range(3):
#         for j in range(4):
#             axs[i, j].hist(shap_values[0][2][:, 4*i + j], bins=25)
#             axs[i, j].set_ylabel(f'{i}', rotation=0)
#             axs[i, j].set_title(TSS_scalar_cols[4*i + j])
#             axs[i, j].set_xlim(-0.5, 0.5)

#     fig.suptitle('GB Scalars Shap Value Distributions', fontsize=16)
#     plt.show()
#     plt.clf()
