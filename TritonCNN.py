# v3.0.1, 11/23/2023
# Robert D Patton, rpatton@fredhutch.org
# Ha Lab, Fred Hutchinson Cancer Research Center

"""
Newest version of the Trion-based CNN.
This provides a main and wrapper for running TritonCNN and associated pre-processing and post-processing steps.
Should me modified to work with batch submission to workflow managers (e.g. SLURM, SGE, etc.),
and/or run as a standlone script with command line arguments.
"""

import os
import csv
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool, cpu_count

import models
from preprocessing.compile_sample_data import *
from preprocessing.generate_tf_splits import *

def generate_data(regions_dict, ref_path, overwrite=False):
    """
    Generates data for each modality, for each sample. Each sample is saved as an h5 file, and nested to mimic a dictionary
    structure where the outer keys are genes, and the inner key:value pairs are modalities:data.

    Args:
        regions_dict (dict): A dictionary where each key is a label and the value is a list of modality and path.
        ref_path (str): The path to the reference file.
        overwrite (bool, optional): If True, overwrite existing processed data. Defaults to False.

    Returns:
        None
    """
    out_dir = 'data/samples/'
    os.makedirs(out_dir, exist_ok=True)

    # Find common samples across all modalities
    common_samples = set()
    for _, modal in regions_dict.items():
        samples = get_samples(modal)
        if not common_samples:
            common_samples = set(samples)
        elif samples:
            common_samples &= set(samples)

    print(f'Found {len(common_samples)} common samples across all modalities.')
    print(f'Samples: {common_samples}')

    # Create a pool of workers and process each sample
    args = [(sample, regions_dict, ref_path, out_dir, overwrite) for sample in common_samples]
    # with Pool(1) as p:  # for debugging
    with Pool(cpu_count()) as p:
        print(f'Using {cpu_count()} cores to process samples . . .')
        p.starmap(process_sample, args)

def generate_sets(name, strat_method, sample_data_path, use_modes, validation_samples, folds=5):
    """
    Generates sets of data for training, validation, and testing.

    Args:
        name (str): The name of the dataset.
        strat_method (str): The strategy for splitting the data ('gene' or 'sample').
        sample_data_path (str): The path to the sample data.
        use_modes (list): The list of modalities to use.
        validation_samples (list): The list of samples to use for validation.
        folds (int, optional): The number of folds for cross-validation. Defaults to 5.

    Returns:
        None
    """
    out_dir = 'data/' + name + '/'
    os.makedirs(out_dir, exist_ok=True)
    # Write the arguments to a TSV file
    with open(f'{out_dir}{name}.tsv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['name', 'sample_data_path', 'use_modes', 'validation_samples', 'folds'])
        writer.writerow([name, sample_data_path, use_modes, validation_samples, folds])

    # Get all files in the directory
    all_files = os.listdir(sample_data_path)
    # Filter the list of files
    data_files = [f for f in all_files if f.endswith('_CompiledProcessedData.h5')]
    samples = [f.replace('_CompiledProcessedData.h5', '') for f in data_files]
    # Filter out validation samples and their corresponding data files
    samples, data_files = zip(*[(sample, data_file) for sample, data_file in zip(samples, data_files) if sample not in validation_samples])

    if os.path.exists(out_dir + 'gene_sample_df.tsv'):
        print('Loading existing gene-sample DataFrame . . .')
        gene_sample_df = pd.read_csv(out_dir + 'gene_sample_df.tsv', sep='\t', index_col=0)
    else:
        # create a dictionary to hold what genes have no NaN values for all modalities for each sample
        # N.B. that ALL available modalities are saved so long as those specified with use_modes have no NaN values
        print('Checking samples for NaN values in modalities specified . . .')
        with Pool(cpu_count()) as p:
            results = p.map(check_sample_genes, zip(samples, data_files, [sample_data_path]*len(samples), [use_modes]*len(samples)))
        # Create a DataFrame to hold the results
        gene_sample_df = pd.DataFrame()
        # Loop over the results
        for result in results:
            # Get the sample name from one of the values in the dictionary
            first_key = next(iter(result), None)
            if first_key is None:
                continue
            sample = result[first_key]
            # Create a DataFrame for this result
            result_df = pd.DataFrame({sample: {gene: True for gene in result.keys()}})
            # Concatenate the result DataFrame to the main DataFrame
            gene_sample_df = pd.concat([gene_sample_df, result_df], axis=1)
        # Fill NaN values with False
        gene_sample_df.fillna(False, inplace=True)
        # Sort the DataFrame by row and column indices
        gene_sample_df.sort_index(axis=0, inplace=True)
        gene_sample_df.sort_index(axis=1, inplace=True)
        gene_sample_df.to_csv(out_dir + 'gene_sample_df.tsv', sep='\t')

    if strat_method == 'gene':
        # Split the DataFrame into folds
        folds_df = split_df_gene(gene_sample_df, folds)
    elif strat_method == 'sample':
        # Split the DataFrame into folds
        folds_df = split_df_sample(gene_sample_df, folds)
    folds_df.to_csv(out_dir + 'folds_df.tsv', sep='\t')
    # Flatten the DataFrame into a Series
    flattened = pd.Series(folds_df.values.flatten())
    # Get the count of each unique value
    value_counts = flattened.value_counts()
    # Print the counts
    print('Fold counts:')
    print(value_counts)
    
    # Loop through folds_df by sample, open up the corresponding sample .h5 file from data/samples, 
    # and write the data to the corresponding .tf record file based on the fold specified in folds_df
    print('Starting new .tf splits . . .')

    args_list = [(folds_df, sample_name, samples, sample_data_path, data_files, out_dir) for sample_name in folds_df.columns]
    with Pool(cpu_count()) as p:
        # Use the pool to process each sample in parallel
        p.starmap(process_sample_folds, args_list)

    print('Merge the .tf splits by fold . . .')
    # Combine the tfrecords for each fold
    for fold_number in range(1, folds + 1):
        combine_tfrecords(fold_number, out_dir)
    print('Finished generating sets.')


def generate_validation(name, sample_data_path, use_modes, validation_samples):
    """
    Generates validation data.

    Args:
        name (str): The name of the dataset.
        sample_data_path (str): The path to the sample data.
        use_modes (list): The list of modalities to use.
        validation_samples (list): The list of samples to use for validation.

    Returns:
        None
    """
    out_dir = 'data/' + name + '/'
    os.makedirs(out_dir, exist_ok=True)
    # Get all files in the directory
    all_files = os.listdir(sample_data_path)
    # Filter the list of files
    data_files = [f for f in all_files if f.endswith('_CompiledProcessedData.h5')]
    samples = [f.replace('_CompiledProcessedData.h5', '') for f in data_files]
    # Filter out validation samples and their corresponding data files
    samples, data_files = zip(*[(sample, data_file) for sample, data_file in zip(samples, data_files) if sample in validation_samples])
    print('Starting new validation .tf . . .')
    args_list = [(sample, sample_data, sample_data_path, use_modes, out_dir) for sample, sample_data in zip(samples, data_files)]
    with Pool(cpu_count()) as p:
        # Use the pool to process each sample in parallel
        p.starmap(process_validation_sample, args_list)
    print('Finished generating validation data.')
        

def main():
    """
    The main function to process the data into a format that is easily accessible by the CNN, 
    and then generate the training, validation, and test sets.
    . . .
    """
    # Define paths and parameters
    process_data = False  # if True, will check but not overwrite existing data
    use_set = 'GeneStrat_CompleteInputs_11-21-2023'
    scalars_reference_path = '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/Ensembl109_reference.bed'
    training_home = '/fh/fast/ha_g/user/rpatton/LuCaP_data/'
    sample_data_path = '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/data/samples/'

    # the outer structure of the saved, sample-level data will be a dictionary with gene names as keys
    # the inner structure of the saved, sample-level data will be a dictionary with the following keys (per gene):
    modality_dict = {'TSS_image': ['image', training_home + 'Triton_TSS/results/'],
                     'TSS_scalars': ['scalars', training_home + 'Triton_TSS/results/'],
                     'GB_scalars': ['scalars', training_home + 'Triton_GB/results/'],
                     'ref_scalars': ['ref', ['exon_count', 'transcript_length'], scalars_reference_path],
                     'continuous_label': ['label', '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/labels/continuous_tpm.tsv'],
                     'binary_label': ['label', '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/labels/binary_zfpkm.tsv']}
    # 'TFBS_panel': ['panel', training_home + 'Triton_TFBS/results/']
    
     # Process the data if necessary
    if process_data == True:
        print('Starting data processing . . .')
        generate_data(modality_dict, scalars_reference_path, overwrite=False)
        print('Finished processing data.')

    # TODO: this is the kinda stuff that should go in a config file, ultimately
    if use_set == 'GeneStrat_CompleteInputs_11-21-2023':
        # define train/test/vallidation set generation parameters
        strat_method = 'gene' # 'gene' or 'sample'
        folds = 5
        # N.B. that use_modes are which features are required to contain no NaNs for a feature to pass, but that all modes are saved
        use_modes = ['TSS_image', 'TSS_scalars', 'GB_scalars', 'ref_scalars', 'continuous_label', 'binary_label']
        validation_samples = ['LuCaP_93', 'LuCaP_235-3']  # LuCaP 93 and 235-3 are the deepest NEPC and ARPC samples, respectively

        # Check if the output files already exist
        output_files_exist = all(os.path.exists(f'data/{use_set}/combined_fold-{i}.tfrecord') for i in range(1, folds + 1))
        # Only run generate_sets if the output files do not exist
        if not output_files_exist:
            generate_sets(use_set, strat_method, sample_data_path, use_modes, validation_samples, folds=folds)
            print('Finished generating sets.')
        else:
            print('Data already exists. Continuing.')

        # Check if the output files (validation) already exist
        output_valids_exist = all(os.path.exists(f'data/{use_set}/{i}.tfrecord') for i in validation_samples)
        # Only run generate_sets if the output files do not exist
        if not output_valids_exist:
            generate_validation(use_set, sample_data_path, use_modes, validation_samples)
            print('Finished generating sets.')
        else:
            print('Data already exists. Continuing.')
    elif use_set == 'SampleStrat_CompleteInputs_11-21-2023':
        # define train/test/vallidation set generation parameters
        strat_method = 'sample' # 'gene' or 'sample'
        folds = 5
        # N.B. that use_modes are which features are required to contain no NaNs for a feature to pass, but that all modes are saved
        use_modes = ['TSS_image', 'TSS_scalars', 'GB_scalars', 'ref_scalars', 'continuous_label', 'binary_label']
        validation_samples = ['LuCaP_93', 'LuCaP_235-3']  # LuCaP 93 and 235-3 are the deepest NEPC and ARPC samples, respectively
        
        # Check if the output files already exist
        output_files_exist = all(os.path.exists(f'data/{use_set}/combined_fold-{i}.tfrecord') for i in range(1, folds + 1))
        # Only run generate_sets if the output files do not exist
        if not output_files_exist:
            generate_sets(use_set, strat_method, sample_data_path, use_modes, validation_samples, folds=folds)
            print('Finished generating sets.')
        else:
            print('Data already exists. Continuing.')

        # Check if the output files (validation) already exist
        output_valids_exist = all(os.path.exists(f'data/{use_set}/{i}.tfrecord') for i in validation_samples)
        # Only run generate_sets if the output files do not exist
        if not output_valids_exist:
            generate_validation(use_set, sample_data_path, use_modes, validation_samples)
            print('Finished generating sets.')
        else:
            print('Data already exists. Continuing.')
    else:
        print('No set specified. Exiting.')

    # TODO: to load data for a specific model train/test/run, we'll use get_dataset() from generate_tf_splits.py
    # TODO: which can load all train files at once and shuffle them, as well as individual test files


if __name__ == "__main__":
    main()
