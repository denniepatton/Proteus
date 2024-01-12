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
from multiprocessing import Pool, cpu_count

# Third party imports
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner.tuners import BayesianOptimization

# Local application imports
from analysis import plot_utils
from models.models import *
from models.model_utils import *
from preprocessing.generate_tf_splits import *
from preprocessing.compile_sample_data import *


def generate_data(regions_dict, ref_path, out_dir, overwrite=False):
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
    # Create the output directory if it doesn't exist
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

    # Prepare arguments for multiprocessing
    args = [(sample, regions_dict, ref_path, out_dir, overwrite) for sample in common_samples]

    # Create a pool of workers and process each sample
    with Pool(cpu_count()) as p:
        print(f'Using {cpu_count()} cores to process samples . . .')
        p.starmap(process_sample, args)


def generate_sets(name, strat_method, sample_data_path, use_modes, validation_samples, folds=5, min_depth=0.0, use_zeros=True):
    """
    Generates sets of data for training. Will use all available samples in the sample_data_path passed, excluding those in the
    validation_samples list. Will split the data into folds according to the stratification method specified.

    Args:
        name (str): The name of the dataset.
        strat_method (str): The strategy for splitting the data ('gene' or 'sample').
        sample_data_path (str): The path to the sample data.
        use_modes (list): The list of modalities to use.
        validation_samples (list): The list of samples to use for validation.
        folds (int, optional): The number of folds for cross-validation. Defaults to 5.
        min_depth (float, optional): The minimum depth to use for training. Defaults to 0.0.
        use_zeros (bool, optional): If True, will use samples with zero TPM true label. Defaults to True.

    Returns:
        None
    """
    # Define output directory and create it if it doesn't exist
    out_dir = 'data/' + name + '/'
    os.makedirs(out_dir, exist_ok=True)

    # Write the arguments to a TSV file
    with open(f'{out_dir}{name}.tsv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['name', 'sample_data_path', 'use_modes', 'validation_samples', 'folds', 'min_depth', 'use_zeros'])
        writer.writerow([name, sample_data_path, use_modes, validation_samples, folds, min_depth, use_zeros])

    # Get all files in the directory and filter the list of files
    all_files = os.listdir(sample_data_path)
    data_files = [f for f in all_files if f.endswith('_CompiledProcessedData.h5')]
    samples = [f.replace('_CompiledProcessedData.h5', '') for f in data_files]

    # Filter out validation samples and their corresponding data files
    samples, data_files = zip(*[(sample, data_file) for sample, data_file in zip(samples, data_files) if sample not in validation_samples])

    # Load existing gene-sample DataFrame or create a new one
    if os.path.exists(out_dir + 'gene_sample_df.tsv'):
        print('Loading existing gene-sample DataFrame . . .')
        gene_sample_df = pd.read_csv(out_dir + 'gene_sample_df.tsv', sep='\t', index_col=0)
    else:
        # create a dataframe to hold what genes have no NaN / meet the depth and zeros requirement values for all modalities for each sample
        # N.B. that ALL available modalities are saved so long as those specified with use_modes have no NaN values and threhsolds are met.
        print('Checking samples for minimum depth, 0 labels, and NaN values in modalities specified . . .')
        with Pool(cpu_count()) as p:
            results = p.map(check_sample_genes, zip(samples, data_files, [sample_data_path]*len(samples), [use_modes]*len(samples), [min_depth]*len(samples)))
        # Create dataframes to hold the results
        gene_sample_df = pd.DataFrame()
        summary_df = pd.DataFrame(columns=['valid_genes', 'invalid_nans', 'invalid_depth', 'zero_tpm'])
        # Loop over the results
        for sample, df in results:
            # Create a new column 'valid' that is True if no_nans and min_depth are both True 
            # and zero_tpm is EITHER True or False if use_zeros==True or False if use_zeros==False
            df['valid'] = df['no_nans'] & df['min_depth'] & (use_zeros | ~df['zero_tpm'])
            # Add the 'valid' column to the gene_sample_df DataFrame
            gene_sample_df[sample] = df['valid']

            # Calculate the summary statistics
            valid_genes = df['valid'].sum()
            invalid_nans = (~df['no_nans']).sum()
            invalid_depth = (~df['min_depth']).sum()
            zero_tpm = df['zero_tpm'].sum()

            # Add the summary statistics to the summary DataFrame
            summary_df.loc[sample] = [valid_genes, invalid_nans, invalid_depth, zero_tpm]

    # Write the summary DataFrame to a CSV file
    summary_df.to_csv(out_dir + 'valid_genes_summary.csv')
    # Sort the gene_sample_df DataFrame by row and column indices
    gene_sample_df.sort_index(axis=0, inplace=True)
    gene_sample_df.sort_index(axis=1, inplace=True)
    # Write the gene_sample_df DataFrame to a TSV file
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


def generate_validation(sample_data_path, use_modes, validation_samples, out_dir, min_depth=0.0, use_zeros=True, filter_samples=True):
    """
    Generates validation data.

    Args:
        sample_data_path (str): The path to the sample data.
        use_modes (list): The list of modalities to use.
        validation_samples (list): The list of samples to use for validation.
        out_dir (str, optional): The path to the output directory. Defaults to None.
        min_depth (float, optional): The minimum depth to use for training. Defaults to 0.0.
        use_zeros (bool, optional): If True, will use samples with zero TPM true label. Defaults to True.
        filter_samples (bool, optional): If True, will filter the samples to only those in the validation_samples list. Defaults to True.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Get all files in the directory and filter the list of files
    all_files = os.listdir(sample_data_path)
    data_files = [f for f in all_files if f.endswith('_CompiledProcessedData.h5')]
    samples = [f.replace('_CompiledProcessedData.h5', '') for f in data_files]

    # Filter out validation samples and their corresponding data files
    if filter_samples:
        # Check if there are any matching samples
        matching_samples = [sample for sample in samples if sample in validation_samples]
        if not matching_samples:
            print(f"No matching samples found. Skipping.")
            return
        samples, data_files = zip(*[(sample, data_file) for sample, data_file in zip(samples, data_files) if sample in validation_samples])

    print('Starting new validation .tf . . .')

    # Prepare arguments for multiprocessing
    args_list = [(sample, sample_data, sample_data_path, use_modes, out_dir, min_depth, use_zeros) for sample, sample_data in zip(samples, data_files)]

    # Create a pool of workers and process each sample
    with Pool(cpu_count()) as p:
        p.starmap(process_validation_sample, args_list)

    print('Finished generating validation data.')


def train_model(model_name, use_set, folds, test_fold=None, tune_hyperparameters=True):
    """
    Trains a model using the specified dataset and folds.

    Args:
        model_name (str): The name of the model to train.
        use_set (str): The name of the dataset to use.
        folds (int): The number of folds for cross-validation.
        test_fold (int, optional): If specified, will only conduct the CV with test_fold as test. Defaults to None.
        tune_hyperparameters (bool, optional): If True, hyperparameters will be tuned. Defaults to True.

    Returns:
        None
    """
    print('Training model . . .')
    out_dir = 'results_training/' + model_name + '_' + use_set + '/'
    os.makedirs(out_dir, exist_ok=True)

    # Instantiate the hypermodel
    hypermodel = get_model_instance(model_name)

    # # Define the strategy for parallel training
    # strategy = tf.distribute.MirroredStrategy()
    # # Open a strategy scope
    # with strategy.scope():

    # Everything that creates variables should be under the strategy scope
    model = hypermodel.build_default()

    def extract_features_wrapper(batch):
        return hypermodel.feature_extraction(batch)
    
    # Print the model summary
    print('Model summary (base model if tuning hyperparameters):')
    model.summary()
    plot_utils.plot_model_architecture(model, out_dir)

    # Loop over the folds
    prediction_data = []
    best_model = None
    best_val_performance = float('inf')
    for i in range(1, folds + 1):
        # If test_fold is specified and this is not the test_fold, skip this iteration
        if test_fold is not None and i != test_fold:
            continue

        print(f'Starting fold {i}/{folds}')

        # Create a new subdirectory for the current fold
        fold_dir = os.path.join(out_dir, f'CV-fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)

        # Use the current fold for validation and all other folds for training
        train_folds = [j for j in range(1, folds + 1) if j != i]
        val_fold = [i]

        trainset = get_dataset(use_set=use_set, fold_numbers=train_folds, shuffle=True, batch_size=32).map(extract_features_wrapper)
        valset = get_dataset(use_set=use_set, fold_numbers=val_fold, shuffle=False, batch_size=32).map(extract_features_wrapper)
        valset_genes = get_dataset(use_set=use_set, fold_numbers=val_fold, shuffle=False, batch_size=32).map(extract_gene_names)
        # Flatten the gene_names_list and decode the byte strings to strings
        gene_names_list = [item.numpy().decode('utf-8') for batch in valset_genes for item in batch]
        print('Completed extracting data.')

        if tune_hyperparameters:
            # Nested split (shuffled) for hyperparameter tuning
            DATASET_SIZE = sum(1 for _ in trainset)
            TRAIN_SIZE = int(0.8 * DATASET_SIZE)

            train_dataset = trainset.take(TRAIN_SIZE)
            val_dataset = trainset.skip(TRAIN_SIZE)

            print('Tuning hyperparameters . . .')
            tuner = BayesianOptimization(
                hypermodel,
                objective='val_loss',
                max_trials=20,
                executions_per_trial=3,
                directory=fold_dir + '/bayesian_optimization',
                project_name=model_name + '_' + use_set + '_fold_' + str(i)
            )

            # Define early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)

            # Run the hyperparameter search
            tuner.search(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[early_stopping])
            best_model = tuner.get_best_models(num_models=1)[0]
            print('Best model summary:')
            best_model.summary()
            plot_utils.plot_model_architecture(best_model, fold_dir)
            model = best_model

        # Re-define callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        # model training
        history = model.fit(trainset, epochs=100, batch_size=32, verbose=1, shuffle=True, validation_data=valset, callbacks=[early_stopping, reduce_lr])
        model.save_weights(os.path.join(fold_dir, f"validation-fold_{val_fold[0]}_weights.hdf5"))
        print('Finished training model.')
        # Plot the training and validation loss
        plot_utils.plot_model_loss(history, fold_dir, f"validation-fold_{val_fold[0]}")

        # Generate predictions on the validation set
        predictions = model.predict(valset).flatten()
        # Extract the true labels from the validation set
        true_labels = np.concatenate([y.numpy() for _, y in valset], axis=0)
        prediction_data.append((true_labels, predictions))
        df = pd.DataFrame({'true_label': true_labels, 'prediction': predictions}, index=gene_names_list)
        df.to_csv(os.path.join(fold_dir, f'validation-fold_{val_fold[0]}_predictions.tsv'), sep='\t')
        # Save the predictions and true labels as .npy files
        np.save(os.path.join(fold_dir, f"validation-fold_{val_fold[0]}_predictions.npy"), predictions)
        np.save(os.path.join(fold_dir, f"validation-fold_{val_fold[0]}_true_labels.npy"), true_labels)

        # Plot the output predictions
        plot_utils.scatter_plot(true_labels, predictions, fold_dir, f"validation-fold_{val_fold[0]}_predictions-scatter", val_fold[0])
        plot_utils.residuals_plot(true_labels, predictions, fold_dir, f"validation-fold_{val_fold[0]}_residuals-scatter", val_fold[0])


        # Check if this model has better validation performance than the current best model (always will if doing seperate folds)
        val_performance = np.min(history.history['val_loss']) 
        if val_performance < best_val_performance:
            best_val_performance = val_performance
            best_model = model

        best_model.save(os.path.join(fold_dir, 'best_model.h5'))
        with open(os.path.join(fold_dir, 'best_model_val_loss.txt'), 'w') as f:
            f.write(str(best_val_performance))
        print(f'Finished fold {i}/{folds}.')

    print('Finished complete training across all folds.')

    return


def test_model(model_name, use_set, validation_samples, gene_sets):
    """
    Tests a model using the specified dataset and validation samples.

    Args:
        model_name (str): The name of the model to test.
        use_set (str): The name of the dataset to use.
        validation_samples (list): The list of samples to use for validation.
        gene_sets (list): The list of gene sets to use.

    Returns:
        None
    """
    print('Starting testing model . . .')
    out_dir = 'results_training/' + model_name + '_' + use_set + '/'
    os.makedirs(out_dir, exist_ok=True)

    hypermodel = get_model_instance(model_name)
    def extract_features_wrapper(batch):
        return hypermodel.feature_extraction(batch)

    # Loop through the validation samples
    prediction_data = []
    for sample in validation_samples:
        print(f'Starting validation for {sample}')
        # Create a new subdirectory for the current sample
        sample_dir = os.path.join(out_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)

        # Get the dataset for the current sample
        valset = get_dataset(use_set='validation', samples=[sample], shuffle=False, batch_size=32).map(extract_features_wrapper)
        valset_genes = get_dataset(use_set='validation', samples=[sample], shuffle=False, batch_size=32).map(extract_gene_names)
        valset_depths = get_dataset(use_set='validation', samples=[sample], shuffle=False, batch_size=32).map(extract_gene_depths)

        # Flatten the gene_names_list and decode the byte strings to strings, etc.
        gene_names_list = [item.numpy().decode('utf-8') for batch in valset_genes for item in batch]
        gene_depths_list = [item.numpy() for batch in valset_depths for item in batch]

        # Load the best model
        best_model = tf.keras.models.load_model(os.path.join(out_dir, 'best_model.h5'), custom_objects={'ScalarAttention': ScalarAttention, 'CrossModalAttention': CrossModalAttention})
        # Generate predictions on the validation set
        predictions = best_model.predict(valset).flatten()

        # Extract the true labels from the validation set
        true_labels = np.concatenate([y.numpy() for _, y in valset], axis=0)
        prediction_data.append((true_labels, predictions))

        df = pd.DataFrame({'true_label': true_labels, 'prediction': predictions, 'tss_depth': gene_depths_list}, index=gene_names_list)
        df.to_csv(os.path.join(sample_dir, f'validation-{sample}_predictions.tsv'), sep='\t')

        plot_utils.subset_scatter_plot(df, sample_dir, f"scatter_validation_{sample}", gene_sets)
        plot_utils.subset_residuals_plot(df, sample_dir, f"residual_validation_{sample}", gene_sets)
        plot_utils.subset_residuals_plot(df, sample_dir, f"residual-depth_validation_{sample}", gene_sets, var="tss_depth")
            
        print(f'Finished testing model for {sample}.')
    print('Finished testing model.')


def run_model(model_path, samples_folder, out_dir):
    print('Running model . . .')
    os.makedirs(out_dir, exist_ok=True)
    hypermodel = get_model_instance('BioCNN')
    def extract_features_wrapper(batch):
        return hypermodel.patient_feature_extraction(batch)
    model = tf.keras.models.load_model(model_path, custom_objects={'ScalarAttention': ScalarAttention, 'CrossModalAttention': CrossModalAttention})

    # Get list of .tfrecord files in samples_folder
    samples_files = [f for f in os.listdir(samples_folder) if f.endswith('.tfrecord')]

    # Initialize an empty DataFrame to store all predictions
    all_predictions = pd.DataFrame()

    for sample_file in samples_files:
        # Remove the file extension to get the sample name
        sample = sample_file.replace('.tfrecord', '')
        print(f'Starting run for {sample}')

        # Get the dataset for the current sample
        valset = get_dataset(use_set='patient_validation', samples=[sample], shuffle=False, batch_size=32, labeled=False).map(extract_features_wrapper)
        valset_genes = get_dataset(use_set='patient_validation', samples=[sample], shuffle=False, batch_size=32, labeled=False).map(extract_gene_names)

        # Flatten the gene_names_list and decode the byte strings to strings, etc.
        gene_names_list = [item.numpy().decode('utf-8') for batch in valset_genes for item in batch]

        # Generate predictions
        predictions = model.predict(valset).flatten()

        # Create a DataFrame for the current sample's predictions
        df = pd.DataFrame({sample: predictions}, index=gene_names_list)

        # Concatenate the current sample's DataFrame to the all_predictions DataFrame
        all_predictions = pd.concat([all_predictions, df], axis=1)

        print(f'Finished running model on {sample}.')

    # Save the all_predictions DataFrame to a .tsv file
    all_predictions.to_csv(os.path.join(out_dir, 'all_predictions.tsv'), sep='\t')

    print('Finished running model on all samples.')
    return all_predictions


def main():
    print(tf.__version__)
    """
    The main function to run the Proteus pipeline.

    This function sets up the necessary parameters for the Proteus pipeline, including paths to various data and model files, 
    the modalities to be used in the model, and the gene sets used for validation plots. 

    If the `process_data` flag is set, it will process the data according to the defined modalities and save the processed data 
    in a hierarchical format (gene -> feature-key -> data).

    Finally, it defines the parameters for the dataset to be used in the model training/testing.

    Command line arguments:
    --scalars_reference_path: Path to the scalars reference file.
    --training_home: Path to the training home directory.
    --sample_data_path: Path to the sample data directory.
    --model_name: Name of the model to use.
    --use_set: The dataset to use.
    --process_data: If True, will check but not overwrite existing sample-level h5 data.
    --test_fold: Specify the test fold for parallel training on different GPU nodes.
    --set_parameters: Path to the set_parameters.tsv file.
    --run_mode: Specify whether to 'train' or 'run' the model.
    --tune_hps: Specify whether to tune hyperparameters.

    NOTE: this function has been optimized to run on as many GPUs as are available on the machine. At Fred Hutch some
    dunce decided to put every GPU core on its own node, so the native methods for multi-GPU training don't work (for now).
    Instead, training folds can be run in parallel on different GPU (nodes) by specifying a test fold on the command line.

    Notes:
    The `use_set` parameter is a key in the `set_parameters` dictionary defined in the function. This dictionary defines
    the parameters for the dataset to be used in the model training/testing. Here are the current options:

    GeneStrat_0x_IncZeros
    GeneStrat_0x_NonZeros
    GeneStrat_10x_IncZeros
    GeneStrat_10x_NonZeros
    SampleStrat_0x_IncZeros
    SampleStrat_0x_NonZeros
    SampleStrat_10x_IncZeros
    SampleStrat_10x_NonZeros
    """
    # Create the parser and add the arguments
    parser = argparse.ArgumentParser(description="Run Proteus")
    parser.add_argument("--scalars_reference_path", type=str, default="/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/Ensembl109_reference.bed", help="Path to the scalars reference file")
    parser.add_argument("--training_home", type=str, default="/fh/fast/ha_g/user/rpatton/LuCaP_data/", help="Path to the training home directory")
    parser.add_argument("--model_name", type=str, default="BioCNN", help="Name of the model to use (in models/models.py))")
    parser.add_argument("--use_set", type=str, default='GeneStrat_10x_NonZeros', help="The set_parameters dataset to use (see Proteus.py main() for options)")
    parser.add_argument("--process_data", type=bool, default=False, help="If True, will check but not overwrite existing sample-level h5 data")
    parser.add_argument("--test_fold", type=int, default=None, help="Specify the test fold for parallel training on different GPU nodes")
    parser.add_argument("--set_parameters", type=str, default='data/set_parameters.tsv', help="Path to the set_parameters.tsv file")
    parser.add_argument("--run_mode", type=str, default='train', help="Specify whether to 'train' or 'run' the model")
    parser.add_argument("--tune_hps", type=bool, default=True, help="Specify whether to tune hyperparameters")

    # Parse the arguments
    args = parser.parse_args()

    # Define the gene sets used for validation plots (hard-coded for now)
    gene_sets = {'Housekeeping': '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/HousekeepingGenes.txt',
                 'Non-Housekeeping': '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/NonHousekeepingGenes.txt',
                 'Phenotyping': '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/PhenotypingGenes.txt',
                 'Targets': '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/Targets.txt'}
    gene_sets = {key: open(path, 'r').read().splitlines() for key, path in gene_sets.items()}

    # Define the modality dictionary
    """ MODALITY DICT: this is the dictionary that defines the global modalities to be used in the model, and where to find them.
    These are also hard-coded into the preprocessing scripts, so if you change them here, you must change them there as well."""
    modality_dict = {'TSS_image': ['image', args.training_home + 'Triton_TSS/results/'],
                     'TSS_scalars': ['scalars', args.training_home + 'Triton_TSS/results/'],
                     'GB_scalars': ['scalars', args.training_home + 'Triton_GB/results/'],
                     'ref_scalars': ['ref', ['exon_count', 'transcript_length'], args.scalars_reference_path],
                     'continuous_label': ['label', '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/labels/LuCaP_Log2TPMp1.tsv']}
    
    # Define the set parameters / use_sets
    """ KEYS: these are pre-defined, hard-coded fold splits with different holdout strategies and stratification methods.
    min_depth is used to enforce a depth cutoff for heldout ('run') data as well if specfied. use_zeros applies only to training."""
    set_parameters = {}
    with open(args.set_parameters, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            set_parameters[row[0]] = {
                'strat_method': row[1],
                'min_depth': float(row[2]),
                'use_zeros': row[3].lower() == 'true',
                'validation_samples': row[4].split(',')
            }

    # Run Proteus.py in training mode if specified
    if args.run_mode == 'train':

        # Process data fresh, if the flag is set
        """ PROCESS DATA: if True, will check but not overwrite existing SAMPLE-LEVEL h5 data
        -> the outer structure of the saved, sample-level data will be a dictionary with gene names as keys
        -> the inner structure of the saved, sample-level data will be a dictionary with feature-keys defined by the modality_dict"""
        if args.process_data:
            print('Starting data processing . . .')
            generate_data(modality_dict, args.scalars_reference_path, 'data/samples_training/', overwrite=False)
            print('Finished processing data.')

        # Check if use_set key is in the dictionary
        if args.use_set in set_parameters:
            print(set_parameters[args.use_set])
            # Get the parameters for this use_set
            strat_method = set_parameters[args.use_set]['strat_method']
            min_depth = set_parameters[args.use_set]['min_depth']
            validation_samples = set_parameters[args.use_set]['validation_samples']
            use_zeros = set_parameters[args.use_set]['use_zeros']
            # Define common parameters
            folds = 5
            use_modes = ['TSS_image', 'TSS_scalars', 'GB_scalars', 'ref_scalars', 'continuous_label']
            # Check if the output files already exist
            output_files_exist = all(os.path.exists(f'data/{args.use_set}/combined_fold-{i}.tfrecord') for i in range(1, folds + 1))
            # Only run generate_sets if the output files do not exist
            if not output_files_exist:
                generate_sets(args.use_set, strat_method, 'data/samples_training/', use_modes, validation_samples, folds=folds, min_depth=min_depth, use_zeros=use_zeros)
                print('Finished generating sets.')
            else:
                print('Training data already exists. Continuing.')
        else:
            print('No training use_set specified which matches the keys in ' + args.set_parameters + '. Exiting.')
            quit()

        # Define the validation data path based on min_depth
        validation_data_path = f'data/validation_{min_depth}x/'
        # Check if the output files (validation) already exist
        output_valids_exist = all(os.path.exists(f'{validation_data_path}{i}.tfrecord') for i in validation_samples)

        # Only run generate_sets if the output files do not exist
        if not output_valids_exist:
            generate_validation('data/samples_training/', use_modes, validation_samples, validation_data_path, min_depth=min_depth, use_zeros=use_zeros)
            print('Finished generating sets.')
        else:
            print('Holdout data already exists. Continuing.')

        # Check for GPU availability
        n_gpus = len(tf.config.list_physical_devices('GPU'))
        if n_gpus == 0:
            print('No GPUs available - please check your environemnt/drivers/machine. Exiting.')
            quit()
        elif n_gpus == 1:
            print('1 GPU available: running in single GPU mode.')
        else:
            print(f'{n_gpus} GPUs available: running in multi GPU mode.')

        # Train and test as separated folds for parallelization
        """ If this is ever run on a proper system with multiple GPUs per node, this can be changed to run all folds in parallel,
        e.g. only the else needs to be evaluated. The if will catch a larger batch submission where each job is a single fold."""
        if args.test_fold is not None:
            if args.test_fold > 0:
                # train only on the specified fold, then exit
                print(f'Running fold {args.test_fold} in parallel . . .')
                train_model(args.model_name, args.use_set, folds, test_fold=args.test_fold, tune_hyperparameters=args.tune_hps)
            elif args.test_fold == 0:
                # After ALL folds have been run load the best model and test on hold out data:
                # Directory where the fold directories are located
                base_dir = 'results/' + args.model_name + '_' + args.use_set + '/'
                best_val_loss = float('inf')
                best_model_path = None

                # Loop over all directories in base_dir
                for i in range(1, folds+1):
                    fold_dir = os.path.join(base_dir, f'CV-fold_{i}')
                    # Load the validation loss from the text file
                    with open(os.path.join(fold_dir, 'best_model_val_loss.txt'), 'r') as f:
                        val_loss = float(f.read().strip())
                    # If this model has a lower validation loss than the current best model, update the best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(fold_dir, 'best_model.h5')

                # Load the best model and save it to out_dir
                best_model = tf.keras.models.load_model(best_model_path, custom_objects={'ScalarAttention': ScalarAttention, 'CrossModalAttention': CrossModalAttention})
                best_model.save(os.path.join(base_dir, 'best_model.h5'))
                plot_utils.plot_model_architecture(best_model, base_dir, filename='best_model')

                # test the model on hold out data
                test_model(args.model_name, args.use_set, validation_samples, gene_sets)
            else:
                print('Invalid test fold specified. Exiting.')
        else:
            # Train and test compeltely using all availble GPUs
            train_model(args.model_name, args.use_set, folds, tune_hyperparameters=args.tune_hps)
            # Test the model on hold out data
            test_model(args.model_name, args.use_set, validation_samples, gene_sets)

    # Run Proteus.py in run mode if specified
    elif args.run_mode == 'run':
        ### Patient data ###
        patient_home = '/fh/fast/ha_g/user/rpatton/PSMA_data/'
        min_val_depth = 0.0
        modality_dict = {'TSS_image': ['image', patient_home + 'Triton_TSS/results/'],
                        'TSS_scalars': ['scalars', patient_home + 'Triton_TSS/results/'],
                        'GB_scalars': ['scalars', patient_home + 'Triton_GB/results/'],
                        'ref_scalars': ['ref', ['exon_count', 'transcript_length'], args.scalars_reference_path]}
        use_modes = ['TSS_image', 'TSS_scalars', 'GB_scalars', 'ref_scalars']
        print('Starting data processing . . .')
        generate_data(modality_dict, args.scalars_reference_path, 'data/samples_patient/', overwrite=False)
        print('Finished processing data.')
        # Get list of files in each directory
        samples_files = os.listdir('data/samples_patient/')
        validation_path = f'data/validation_patient_{min_val_depth}x/'
        os.makedirs(validation_path, exist_ok=True)
        validation_files = os.listdir(validation_path)
        # Remove file extensions and keep only unique names
        samples_names = set([f.replace('_CompiledProcessedData.h5', '') for f in samples_files])
        validation_names = set([f.replace('.tfrecord', '') for f in validation_files])
        # Find names that are in samples_names but not in validation_names
        missing_names = samples_names - validation_names
        # Call generate_validation with missing_names
        generate_validation('data/samples_patient/', use_modes, list(missing_names), validation_path, min_depth=min_val_depth, filter_samples=True)
        model_path = 'results_old/BioCNN_GeneStrat_CompleteInputs_12-14-2023_RESULTS-1/best_model.h5'
        run_model(model_path, validation_path, 'results_holdout/BioCNN_GeneStrat_CompleteInputs_12-14-2023_RESULTS-1')


    else:
        print('Invalid run mode specified. Exiting.')
        quit()



if __name__ == "__main__":
    main()
