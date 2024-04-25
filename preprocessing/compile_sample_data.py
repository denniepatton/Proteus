"""
compile_sample_data.py
v1.0.1, 12/14/2023
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script contains functions for compiling sample data from raw Triton and reference files into a single HDF5 file.
"""

# Standard library imports
import os

# Third party imports
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_samples(modal):
    """
    Retrieves sample names based on the modality of the data.

    Args:
        modal (tuple): A tuple where the first element is the modality of the data 
                       ("image", "panel", "scalars", "label", "ref") and the second element 
                       is the path to the directory containing the files.

    Returns:
        list: A list containing the sample names. Returns an empty list for "ref" modality 
              or unrecognized modalities.
    """
    modality, path = modal[0], modal[-1]

    if modality in ["image", "panel"]:
        return _get_samples_from_files(path)
    elif modality == "scalars":
        return _get_samples_from_scalars(path)
    elif modality == "label":
        return _get_samples_from_label(path)
    elif modality == "ref":
        return []
    else:
        print(f'Modality "{modality}" not recognized. Skipping.')
        return []


def _get_samples_from_files(path):
    """Helper function to get samples from file names in a directory."""
    samples = []
    for filename in os.listdir(path):
        parts = filename.split("_TritonProfiles.npz")
        if len(parts) == 2:
            samples.append(parts[0])
    return samples


def _get_samples_from_scalars(path):
    """Helper function to get samples from a scalars file."""
    df = pd.read_csv(path + 'TritonCompositeFM.tsv', sep='\t')
    return df['sample'].unique().tolist()


def _get_samples_from_label(path):
    """Helper function to get samples from a label file."""
    df = pd.read_csv(path, sep='\t', nrows=1)
    return df.dropna(axis=1).columns.unique().tolist()


def process_sample(sample, regions_dict, ref_path, out_dir, overwrite):
    """
    Processes a sample and writes the data to an HDF5 file.

    Args:
        sample (str): The name of the sample to process.
        regions_dict (dict): A dictionary mapping labels to modalities and paths.
        ref_path (str): The path to the reference data.
        out_dir (str): The directory to write the HDF5 file to.
        overwrite (bool): Whether to overwrite existing files.

    Note:
        The original data files are not modified.
    """
    if os.path.isfile(f'{out_dir}{sample}_CompiledProcessedData.h5') and not overwrite:
        print(f'Processed data for sample {sample} already exists. Skipping.')
        return

    print(f'Processing sample {sample} . . .')
    sample_dict = {}
    for label, modal in regions_dict.items():
        modality, path = modal[0], modal[1]
        if modality == 'image':
            sample_dict[label] = _process_image(f'{path}{sample}_TritonProfiles.npz')
        elif modality == 'scalars':
            sample_dict[label] = _process_scalar(f'{path}TritonCompositeFM.tsv', sample)
        elif modality == 'label':
            sample_dict[label] = _process_label(path, sample)
        elif modality == 'ref':
            sample_dict[label] = _process_ref(ref_path)
        else:
            print(f'Modality "{modality}" not recognized. Skipping.')

    # Initialize the gene-based dictionary
    gene_dict = {}

    # Iterate over the original dictionary
    print('Inverting sample dictionary . . . (to gene outer, modal inner))')
    for outer_key, inner_dict in sample_dict.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in gene_dict:
                gene_dict[inner_key] = {}
            gene_dict[inner_key][outer_key] = value

    _save_dict_to_hdf5(gene_dict, out_dir + sample + '_CompiledProcessedData.h5')

def _process_image(filepath):
    """
    Processes an image file and returns the data as a dictionary.
    N.B. that "non-functional" means Triton did not report a signal matrix,
    which means the corresponding region lacked >90% base pair coverage.

    Args:
        filepath (str): The path to the image file.

    Returns:
        dict: A dictionary where each key is a 'site' and the value is a 2D numpy array of the signals.
    """
    print(f'Processing image file {filepath} . . .')

    # Load the NPZ file
    input_object = np.load(filepath, allow_pickle=True)

    counter = 0
    out_dict = {}
    for key, signals in input_object.items():
        # Check if signals is a functional input matrix (Triton or TritonMe) - will be empty if not
        if signals.shape == (11, 2000) or signals.shape == (15, 2000):
            signals = signals.T 
        else:
            counter += 1
            continue

        # Normalize to mean = 1 all signals, excluding peak locs and nucleotide freqs
        for col in range(6):
            signals[:, col] = _scale_signal(signals[:, col])

        out_dict[key] = signals

    print(f'Found {counter} non-functional input matrices.')
    return out_dict

def _scale_signal(input_array):
    """
    Scales an input array by dividing each element by the mean of the array.

    Args:
        input_array (numpy.ndarray): The input array to be scaled.

    Returns:
        numpy.ndarray: The scaled array.
    """
    
    mean_val = np.mean(input_array)
    if mean_val != 0:
        return input_array / mean_val
    else:
        return input_array


def _process_scalar(filepath, sample, drop_list=None):
    """
    Processes a scalar file and returns the data as a dictionary.

    N.B. mean-depth and ONLY mean-depth is not standard scaled as this has an absolute meaning and
    is also used downstream for thresholding and plotting.

    Args:
        filepath (str): The path to the scalar file.
        sample (str): The name of the sample to process.
        drop_list (list, optional): A list of column names to drop. Defaults to 
            ["central-diversity", "central-depth", "central-loc", "plus-minus-ratio", 
            "minus-one-pos", "plus-one-pos"].

    Returns:
        dict: A dictionary where each key is a 'site' and the value is a 1D numpy array of the values.
    """
    # Default drop list: these features are NaN for gene bodies and may be NaN for noisy promoters
    if drop_list is None:
        drop_list = ["central-diversity", "central-depth", "central-loc", 
                     "plus-minus-ratio", "minus-one-pos", "plus-one-pos"]

    print(f'Processing scalar file {filepath} . . .')

    df = pd.read_csv(filepath, sep='\t')
    df = df[df['sample'] == sample].drop(columns=['sample'])

    # Pivot the DataFrame
    df = df.pivot(index='site', columns='feature', values='value')

    # Remove the columns found in drop_list
    df = df.drop(columns=drop_list)

    # Sort the remaining columns
    df = df.sort_index(axis=1)

    # Apply StandardScaler to each feature across all sites
    scaler = StandardScaler()
    columns_to_scale = df.columns.difference(['mean-depth'])
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # Convert the DataFrame to a dictionary where each key is a 'site' and the value is a 1D numpy array of the values
    site_dict = {site: np.array(df.loc[site].values) for site in df.index}

    return site_dict


def _process_label(filepath, sample):
    """
    Processes a label file and returns the data as a dictionary.

    Args:
        filepath (str): The path to the label file.
        sample (str): The name of the sample to process.

    Returns:
        dict: A dictionary where each key is an index (row name) and the value comes from the sample column.
    """
    print(f'Processing label file {filepath} . . .')

    df = pd.read_csv(filepath, sep='\t', index_col=0)

    # Restrict to the column that is sample
    df = df[[sample]]
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Convert the DataFrame to a dictionary where each key is an index (row name) and the value comes from the sample column
    label_dict = df[sample].to_dict()

    return label_dict


def _process_ref(filepath, cols=['NumExons', 'transcript_length', 'mRNA_size']):
    """
    Processes a reference file and returns the data as a dictionary.

    Args:
        filepath (str): The path to the reference file.
        cols (list, optional): A list of column names to process. Defaults to ['NumExons', 'transcript_length', 'mRNA_size'].

    Returns:
        dict: A dictionary where each key is a 'name' and the value is a 1D numpy array of the values.
    """
    print(f'Processing reference file {filepath} . . .')

    # Load the TSV file
    df = pd.read_csv(filepath, sep='\t')

    # Coerce the values in cols to numeric
    df[cols] = df[cols].apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Drop rows with NaNs in cols
    df = df.dropna(subset=cols)

    # Apply StandardScaler to the columns indicated by cols
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    # Write a dictionary where the keys come from the column 'symbol' and the values are 1D numpy arrays based on the values for that row/dymbol in cols
    ref_dict = {row['symbol']: np.array(row[cols].values, dtype=float) for _, row in df.iterrows()}

    return ref_dict

def _save_dict_to_hdf5(dic, filename):
    """
    Saves a dictionary to an HDF5 file.

    Args:
        dic (dict): The dictionary to save.
        filename (str): The name of the HDF5 file.

    Note:
        This function uses recursion to handle nested dictionaries.
    """
    with h5py.File(filename, 'w') as h5file:
        _recursively_save_dict_contents_to_group(h5file, '/', dic)


def _recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Recursively saves the contents of a dictionary to an HDF5 group.

    Args:
        h5file (h5py.File): The HDF5 file.
        path (str): The path to the HDF5 group.
        dic (dict): The dictionary to save.

    Note:
        This function is intended to be used internally by the _save_dict_to_hdf5() function.
    """
    for key, item in dic.items():
        if isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif isinstance(item, (np.ndarray, int, np.int64, float, np.float64, str, bytes)):
            h5file[path + key] = item
        else:
            print(f"Cannot save {key} as it is of type {type(item)}.")

