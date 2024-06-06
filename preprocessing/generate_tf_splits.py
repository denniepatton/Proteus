"""
generate_tf_splits.py
v1.0.3, 6/6/2024
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script contains functions for generating the training and testing splits 
for TensorFlow models and saving them as .tfrecord files.
"""

# Standard library imports
import os
import glob

# Third party imports
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf


def check_sample_genes(sample_data_file_path_modes):
    """
    Checks the genes in a sample data file and returns a dictionary of valid genes.

    Args:
        sample_data_file_path_modes (tuple): A tuple containing the sample name,
                                             path to the sample data, and list of modalities to use.

    Returns:
        tuple: A tuple containing the sample name and a DataFrame containing the valid genes.
            
    """
    sample, data_file, use_modes, min_depth, min_label = sample_data_file_path_modes
    gene_data = []
    with h5py.File(data_file, 'r') as f:
        for gene in f.keys():
            if 'truth_labels' in f[gene]:
                min_label_met = f[gene]['truth_labels'][()] >= np.log2(min_label)  # labels stored as log2(TPM+1), so min_label 1.1 means min TPM of 0.1, etc.
            else:
                continue  # skip genes without a continuous label, i.e. not represented in the truth labels
            no_nans = all(modality in f[gene] and not np.isnan(np.array(f[gene][modality])).any() for modality in use_modes)
            if 'TSS_scalars' in f[gene]:  # Also reports false for unmatched genes
                min_depth_met = f[gene]['TSS_scalars'][7] >= min_depth  # see extract_gene_depths() below: mean-depth is raw (not scaled)
            else:
                min_depth_met = False  # no depth data does not meet min_depth
            
            gene_data.append({'gene': gene, 'no_nans': no_nans, 'min_depth': min_depth_met, 'min_label': min_label_met})

    df = pd.DataFrame(gene_data).set_index('gene')

    return sample, df


def split_df_gene(df, folds, gene_set=None):
    """
    Splits the data into folds, stratified by gene name.

    Args:
        df (pandas.DataFrame): The DataFrame to split. Each row corresponds to a gene, 
                               and each column corresponds to a sample.
        folds (int): The number of folds to split the data into.
        gene_set (string, optional): Path to a txt file with genes of interest to restrict to, one per line.

    Returns:
        pandas.DataFrame: The split DataFrame, where each True value has been replaced 
                          with the fold number it belongs to.
    """
    print('Splitting the data into folds (stratified by gene name) . . .')

    # If gene_set is not None, load the gene names from the .txt file
    if gene_set is not None:
        with open(gene_set, 'r') as f:
            gene_set = [line.strip() for line in f]

    # Loop over the rows in the DataFrame
    for row in df.itertuples(index=True):
        # If gene_set is not None and the gene for this row is not in gene_set, cast all True values to False
        if gene_set is not None and row[0] not in gene_set:
            df.loc[row[0]] = False
        else:
            # Get the indices of the True values in the row
            true_indices = [i for i, x in enumerate(row[1:]) if x]

            # Draw a random sample of fold numbers for this row
            fold_numbers = np.random.choice(range(1, folds + 1), len(true_indices), replace=True)

            # Assign a fold number to each True value in the row
            for i, fold in zip(true_indices, fold_numbers):
                df.at[row[0], df.columns[i]] = fold

    return df


def split_df_sample(df, folds):
    """
    Splits the data into folds, grouping samples together.

    Args:
        df (pandas.DataFrame): The DataFrame to split. Each row corresponds to a gene, 
                               and each column corresponds to a sample.
        folds (int): The number of folds to split the data into.

    Returns:
        pandas.DataFrame: The split DataFrame, where each True value has been replaced 
                          with the fold number it belongs to.
    """
    print('Splitting the data into folds (grouping samples together) . . .')

    # Calculate the sum of True values in each column (total number of genes for each sample)
    sums = df.sum()

    # Sort the samples based on the total number of genes
    sorted_samples = sums.sort_values().index.tolist()

    # Initialize a list to hold the total number of genes for each fold
    fold_sums = [0] * folds

    # Initialize a dictionary to hold the fold number for each sample
    sample_folds = {}

    # Assign each sample to the fold with the least total number of genes so far
    for sample in sorted_samples:
        fold = fold_sums.index(min(fold_sums))
        sample_folds[sample] = fold + 1
        fold_sums[fold] += sums[sample]

    # Replace the True values in the DataFrame with the fold numbers
    for sample, fold in sample_folds.items():
        df[sample] = df[sample].replace(True, fold)

    return df


def serialize_example(feature_dict, gene_name, sample_name):
    """
    Creates a tf.Example message ready to be written to a file.

    Args:
        feature_dict (dict): A dictionary where each key is a feature name and each value is a feature value.

    Returns:
        bytes: The serialized tf.Example message.
    """
    # Create a dictionary to hold the features
    features = {}
    features['gene_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gene_name.encode('utf-8')]))
    features['sample_name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample_name.encode('utf-8')]))

    # For each feature in the feature_dict
    for key, value in feature_dict.items():
        # Check the type of the value
        if isinstance(value, np.ndarray):
            # Flatten the array and store it
            value = value.ravel()
            if value.dtype == np.float32 or value.dtype == np.float64:
                features[key] = _float_feature(value)
            elif value.dtype == np.int64:
                features[key] = _int64_feature(value)
        elif isinstance(value, (int, np.integer)):
            features[key] = _int64_feature(value)
        elif isinstance(value, (float, np.floating)):
            features[key] = _float_feature(value)
        else:
            raise ValueError(f'Unknown type for feature "{key}"')

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def _float_feature(value):
    """Returns a float_list from a float / double."""
    if isinstance(value, (float, np.floating)):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if isinstance(value, (int, np.integer)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def load_key_from_hdf5(key, h5file):
    """
    Loads a key from an HDF5 file.

    Args:
        key (str): The key to load.
        h5file (h5py.File): The HDF5 file to load the key from.

    Returns:
        dict: A dictionary where each key is a subkey of the key in the HDF5 file and each value is the corresponding data.
        If the key is not found in the HDF5 file, returns None.
    """
    if key in h5file:
        # Load the inner dictionary
        inner_dict = {}
        for inner_key, item in h5file[key].items():
            inner_dict[inner_key] = item[()]
        return inner_dict
    else:
        print(f'WARNING: Key "{key}" not found in HDF5 file: {h5file.filename}')
        return None
    

def parse_serialized_example(serialized_example, labeled=True):
    """
    Parses a serialized tf.Example message.

    Args:
        serialized_example (bytes): The serialized tf.Example message.

    Returns:
        dict: A dictionary where each key is a feature name and each value is the corresponding feature value.
    """
    # Define the feature description dictionary
    if labeled:
        feature_description = {
            'gene_name': tf.io.FixedLenFeature([], tf.string),
            'sample_name': tf.io.FixedLenFeature([], tf.string),
            'TSS_images': tf.io.FixedLenFeature([2000, 11], tf.float32),
            'TSS_scalars': tf.io.FixedLenFeature([12], tf.float32),
            'GB_scalars': tf.io.FixedLenFeature([12], tf.float32),
            'ref_scalars': tf.io.FixedLenFeature([3], tf.float32),
            'TFX_label': tf.io.FixedLenFeature([], tf.float32),
            'truth_labels': tf.io.FixedLenFeature([], tf.float32)
        }
    else:
        feature_description = {
            'gene_name': tf.io.FixedLenFeature([], tf.string),
            'sample_name': tf.io.FixedLenFeature([], tf.string),
            'TSS_images': tf.io.FixedLenFeature([2000, 11], tf.float32),
            'TSS_scalars': tf.io.FixedLenFeature([12], tf.float32),
            'GB_scalars': tf.io.FixedLenFeature([12], tf.float32),
            'ref_scalars': tf.io.FixedLenFeature([3], tf.float32),
            'TFX_label': tf.io.FixedLenFeature([], tf.float32)
        }

    # Parse the serialized example
    example = tf.io.parse_single_example(serialized_example, feature_description)

    return example


def process_sample_folds(folds_df, sample_name, samples, data_files, out_dir):
    """
    Processes the folds for a given sample and writes the data to TFRecord files.

    Args:
        folds_df (pandas.DataFrame): The DataFrame containing the fold assignments.
        sample_name (str): The name of the sample to process.
        samples (list): The list of all sample names.
        data_files (list): The list of all data file paths.
        out_dir (str): The directory to write the TFRecord files to.
    """
    print(f'Processing {sample_name} . . .')

    # Group the genes by fold
    groups = folds_df[folds_df[sample_name] != False].groupby(sample_name)
    folds_dict = {fold: group.index.tolist() for fold, group in groups}

    # Get the index of the sample in the samples list
    sample_index = samples.index(sample_name)

    # Open the HDF5 file for the sample
    with h5py.File(data_files[sample_index], 'r') as f:
        # For each fold
        for fold in folds_dict.keys():
            # Create a TFRecord file for the fold
            out_file = f'{out_dir}{sample_name}_fold-{fold}.tfrecord'
            with tf.io.TFRecordWriter(out_file) as writer:
                # For each gene in the fold
                for gene in folds_dict[fold]:
                    # Load the gene data from the HDF5 file
                    gene_data = load_key_from_hdf5(gene, f)
                    if gene_data is None:
                        continue  
                    # Serialize the gene data
                    serialized = serialize_example(gene_data, gene, sample_name)
                    # Write the serialized data to the TFRecord file
                    writer.write(serialized)


def process_holdout_sample(sample, sample_data, use_modes, out_dir, min_depth=0.0):
    """
    Processes a holdout (or single) sample and writes the data to a TFRecord file.

    Args:
        sample (str): The name of the sample to process.
        sample_data (str): The path to the sample data file.
        use_modes (list): The list of modes to use.
        out_dir (str): The directory to write the TFRecord file to.
        min_depth (float, optional): The minimum depth to use. Defaults to 0.0.
    """
    print(f'Processing {sample} . . .')

    with h5py.File(sample_data, 'r') as f:
        # Create a dictionary of valid genes
        valid_genes = {gene: sample for gene in f.keys()
                    if all(modality in f[gene] and not np.isnan(np.array(f[gene][modality])).any() for modality in use_modes)
                    and f[gene]['TSS_scalars'][7] >= min_depth}

        # Create a TFRecord file for the sample
        out_file = f'{out_dir}{sample}.tfrecord'
        with tf.io.TFRecordWriter(out_file) as writer:
            # For each valid gene
            for gene in valid_genes.keys():
                # Load the gene data from the HDF5 file
                gene_data = load_key_from_hdf5(gene, f)
                # Serialize the gene data
                serialized = serialize_example(gene_data, gene, sample)
                # Write the serialized data to the TFRecord file
                writer.write(serialized)


def combine_tfrecords(fold_number, out_dir):
    """
    Combines all TFRecord files for a given fold into a single TFRecord file.

    Args:
        fold_number (int): The fold number.
        out_dir (str): The directory containing the TFRecord files.

    Note:
        The original TFRecord files are deleted after they have been combined.
    """
    print(f'Combining TFRecord files for fold {fold_number} . . .')

    # Get all .tfrecord files for the given fold
    tfrecord_files = glob.glob(f'{out_dir}*fold-{fold_number}.tfrecord')

    # Create a new .tfrecord file to write the combined data
    with tf.io.TFRecordWriter(f'{out_dir}combined_fold-{fold_number}.tfrecord') as writer:
        for tfrecord_file in tfrecord_files:
            # Read the .tfrecord file
            for record in tf.data.TFRecordDataset(tfrecord_file):
                # Write the record to the new file
                writer.write(record.numpy())

    # Delete the original .tfrecord files
    for tfrecord_file in tfrecord_files:
        os.remove(tfrecord_file)


def get_dataset(use_set, fold_numbers=None, samples=None, shuffle=True, batch_size=32, labeled=True):
    """
    Loads and prepares a dataset from TFRecord files.

    Args:
        fold_numbers (list): The fold numbers to load.
        use_set (str): The (named) split-set to use.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        batch_size (int, optional): The batch size. Defaults to 32.

    Returns:
        tf.data.Dataset: The prepared dataset.
    """
    if samples is not None:
        print(f'Loading dataset for samples {samples} from {use_set} set . . .')
        # Load the .tfrecord file for each sample
        files = [f'data/{use_set}/{sample}.tfrecord' for sample in samples]
    elif fold_numbers is not None:
        print(f'Loading dataset for fold(s) {fold_numbers} from {use_set} set . . .')
        # Load the .tfrecord file for the given fold(s)
        files = [f'data/{use_set}/combined_fold-{fold_number}.tfrecord' for fold_number in fold_numbers]
    else:
        print('Neither samples nor fold numbers were specified.')
        return None

    ds = tf.data.TFRecordDataset(files)
    ds = ds.cache()

    # Shuffle the data if specified
    if shuffle: 
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    # Parse the data
    ds = ds.map(lambda x: parse_serialized_example(x, labeled))
    # ds = ds.map(parse_serialized_example, labeled)

    # Batch the data
    ds = ds.batch(batch_size)

    return ds

def extract_gene_names(batch):
    return batch['gene_name']

def extract_sample_names(batch):
    return batch['sample_name']

def extract_gene_depths(batch):
    """ ASSUMES:
    drop_list = ["central-diversity", "central-depth", "central-loc", 
                 "plus-minus-ratio", "minus-one-pos", "plus-one-pos"]
    leaving 12 features which are sorted alphanumericlaly by default, so 'mean-depth' is index 7

    0: fragment-diversity
    1: fragment-entropy
    2: fragment-mad
    3: fragment-mean
    4: fragment-median
    5: fragment-ratio
    6: fragment-stdev
    7: mean-depth
    8: np-amplitude
    9: np-period
    10: np-score
    11: var-ratio
    """
    return batch['TSS_scalars'][:, 7]
