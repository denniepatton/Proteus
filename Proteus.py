"""
Proteus.py
v1.6.1, 5/9/2024
Author: Robert D Patton, rpatton@fredhutch.org
with help from: Alex Netzley
Ha Lab, Fred Hutchinson Cancer Research Center
"""

# Standard library imports
import re
import os
import csv
import argparse
from multiprocessing import Pool, cpu_count

# Third party imports
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch, BayesianOptimization

# Local application imports
from analysis import plot_utils
from models.models import *
from models.model_utils import *
from preprocessing.generate_tf_splits import *
from preprocessing.compile_sample_data import *


def generate_data(dataset, regions_dict, ref_path, overwrite=False):
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
    out_dir = f'data/{dataset}/h5_files/'
    os.makedirs(out_dir, exist_ok=True)

    # Find common samples across all modalities
    common_samples = set()
    for _, modal in regions_dict.items():
        samples = get_samples(modal, dataset)
        if samples:  # Ignore empty lists
            if not common_samples:
                common_samples = set(samples)
            elif common_samples:
                common_samples &= set(samples)


    print(f'Found {len(common_samples)} common samples across all modalities.')
    print(f'Samples: {common_samples}')

    # Prepare arguments for multiprocessing
    args = [(sample, regions_dict, ref_path, out_dir, overwrite) for sample in common_samples]

    # Create a pool of workers and process each sample
    with Pool(cpu_count()) as p:
        print(f'Using {cpu_count()} cores to process samples . . .')
        p.starmap(process_sample, args)


def generate_sets(name, strat_method, data_files, use_modes, holdout_samples, folds=5, min_depth=0.0, min_label=1.0):
    """
    Generates sets of data for training. Will use all available samples in the sample_data_path passed, excluding those in the
    holdout_samples list. Will split the data into folds according to the stratification method specified.

    Args:
        name (str): The name of the dataset.
        strat_method (str): The strategy for splitting the data ('gene' or 'sample').
        sample_paths (list): The paths to each h5 sample file
        use_modes (list): The list of modalities to use.
        holdout_samples (list): The list of samples to use for holdout.
        folds (int, optional): The number of folds for cross-validation. Defaults to 5.
        min_depth (float, optional): The minimum depth to use for training. Defaults to 0.0.
        use_zeros (bool, optional): If True, will use samples with zero TPM true label. Defaults to True.

    Returns:
        None
    """
    # Define output directory and create it if it doesn't exist
    out_dir = 'data/' + name + '/'
    os.makedirs(out_dir, exist_ok=True)
    samples = [os.path.basename(f).replace('_CompiledProcessedData.h5', '') for f in data_files]

    # Write the arguments to a TSV file
    with open(f'{out_dir}{name}.tsv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['name', 'use_modes', 'holdout_samples', 'folds', 'min_depth', 'min_label'])
        writer.writerow([name, use_modes, holdout_samples, folds, min_depth, min_label])
        f.write("\n")
        f.write("samples:\n")
        for path in data_files:
            f.write(f"{path}\n")

    # Filter out holdout samples and their corresponding data files
    samples, data_files = zip(*[(sample, data_file) for sample, data_file in zip(samples, data_files) if sample not in holdout_samples])

    # Load existing gene-sample DataFrame or create a new one
    if os.path.exists(out_dir + 'gene_sample_df.tsv'):
        print('Loading existing gene availability . . .')
        gene_sample_df = pd.read_csv(out_dir + 'gene_sample_df.tsv', sep='\t', index_col=0)
    else:
        # create a dataframe to hold what genes have no NaN / meet the depth and zeros requirement values for all modalities for each sample
        # N.B. that ALL available modalities are saved so long as those specified with use_modes have no NaN values and threhsolds are met.
        print('Checking samples for minimum depth, minimum labels, and NaN values in modalities specified . . .')
        with Pool(cpu_count()) as p:
            results = p.map(check_sample_genes, zip(samples, data_files, [use_modes]*len(samples), [min_depth]*len(samples), [min_label]*len(samples)))
        # Create dataframes to hold the results
        gene_sample_df = pd.DataFrame()
        summary_df = pd.DataFrame(columns=['valid_genes', 'invalid_nans', 'invalid_depth', 'invalid_label'])
        # Loop over the results
        for sample, df in results:
            # Create a new column 'valid' that is True if all modalities have no NaN values and meet the thresholds
            df['valid'] = df['no_nans'] & df['min_depth'] & df['min_label']
            # Add the 'valid' column to the gene_sample_df DataFrame
            gene_sample_df[sample] = df['valid']

            # Calculate the summary statistics
            valid_genes = df['valid'].sum()
            invalid_nans = (~df['no_nans']).sum()
            invalid_depth = (~df['min_depth']).sum()
            invalid_label = (~df['min_label']).sum()

            # Add the summary statistics to the summary DataFrame
            summary_df.loc[sample] = [valid_genes, invalid_nans, invalid_depth, invalid_label]

        # Write the summary DataFrame to a CSV file
        summary_df.sort_index(axis=0, inplace=True)
        summary_df.to_csv(out_dir + 'valid_genes_summary.csv')
        # Sort the gene_sample_df DataFrame by row and column indices
        gene_sample_df.sort_index(axis=0, inplace=True)
        gene_sample_df.sort_index(axis=1, inplace=True)
        # Fill NaN values with False (NaNs occur for genes which exist in on cohort but not another)
        gene_sample_df.fillna(False, inplace=True)
        # Write the gene_sample_df DataFrame to a TSV file
        gene_sample_df.to_csv(out_dir + 'gene_sample_df.tsv', sep='\t')

    if os.path.exists(out_dir + 'folds_df.tsv'):
        print('Loading existing fold splits . . .')
        folds_df = pd.read_csv(out_dir + 'folds_df.tsv', sep='\t', index_col=0)
    else:
        if strat_method == 'gene':
            # Split the DataFrame into folds
            folds_df = split_df_gene(gene_sample_df, folds)
        elif strat_method == 'var':
            # Split the DataFrame into folds
            folds_df = split_df_gene(gene_sample_df, folds, gene_set='reference/variable_training.txt')
        elif strat_method == 'sample':
            # Split the DataFrame into folds
            folds_df = split_df_sample(gene_sample_df, folds)
        folds_df = folds_df.sort_index(axis=1)
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
    args_list = [(folds_df, sample_name, samples, data_files, out_dir) for sample_name in folds_df.columns]
    with Pool(cpu_count()) as p:
        # Use the pool to process each sample in parallel
        p.starmap(process_sample_folds, args_list)

    print('Merging .tf splits by fold . . .')
    # Combine the tfrecords for each fold
    for fold_number in range(1, folds + 1):
        combine_tfrecords(fold_number, out_dir)
    print('Finished generating sets.')


def generate_holdout(use_modes, holdout_samples, out_dir, min_depth=0.0, filter_samples=True):
    """
    Generates sample-level holdout data (tfrecords).

    Args:
        use_modes (list): The list of modalities to use.
        holdout_samples (list): The list of samples to use for holdout.
        out_dir (str, optional): The path to the output directory. Defaults to None.
        min_depth (float, optional): The minimum depth to use for training. Defaults to 0.0.
        use_zeros (bool, optional): If True, will use samples with zero TPM true label. Defaults to True.
        filter_samples (bool, optional): If True, will filter the samples to only those in the holdout_samples list. Defaults to True.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Get all files in the root data directory and filter the list of files
    data_files = glob.glob('data/**/*_CompiledProcessedData.h5', recursive=True)
    samples = [os.path.basename(f).replace('_CompiledProcessedData.h5', '') for f in data_files]

    # Filter out holdout samples and their corresponding data files
    if filter_samples:
        # Check if there are any matching samples
        matching_samples = [sample for sample in samples if sample in holdout_samples]
        if not matching_samples:
            print(f"No matching samples found. Skipping.")
            return
        samples, data_files = zip(*[(sample, data_file) for sample, data_file in zip(samples, data_files) if sample in holdout_samples])

    print('Starting holdout .tfrecord generation . . .')

    # Prepare arguments for multiprocessing
    args_list = [(sample, sample_data, use_modes, out_dir, min_depth) for sample, sample_data in zip(samples, data_files)]

    # Create a pool of workers and process each sample
    with Pool(cpu_count()) as p:
        p.starmap(process_holdout_sample, args_list)

    print('Finished generating holdout data.')


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
    def extract_features_wrapper(batch):
        return hypermodel.feature_extraction(batch)
    
    # Print the model summary
    print('Model summary (base model if tuning hyperparameters):')
    model = hypermodel.build_default()
    model.summary()
    plot_utils.plot_model_architecture(model, out_dir)

    # Loop over the folds
    prediction_data = []
    best_model = None
    best_val_performance = float('inf')

    for i in range(1, folds + 1):
        # Re-instantiate the model for each fold to ensure a fresh model and no leakage
        model = hypermodel.build_default()

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
        valset_samples = get_dataset(use_set=use_set, fold_numbers=val_fold, shuffle=False, batch_size=32).map(extract_sample_names)
        # Flatten the gene_names_list and decode the byte strings to strings
        gene_names_list = [item.numpy().decode('utf-8') for batch in valset_genes for item in batch]
        sample_names_list = [item.numpy().decode('utf-8') for batch in valset_samples for item in batch]
        print('Completed extracting data.')

        if tune_hyperparameters:
            print('Tuning hyperparameters . . .')
            total_batches = sum(1 for _ in trainset)
            print(f'Total training batches (size 32): {total_batches}')

            # # Define early stopping (fairly critical threshold here, as it is used for random_tuner only)
            # early_stopping = EarlyStopping(monitor='val_loss', patience=5)

            # # Start with random search
            # random_tuner = RandomSearch(
            #     lambda hp: hypermodel.build(hp, epoch_batches=total_batches),
            #     objective='val_mse',
            #     max_trials=10,  # Number of random search trials - should go quickly with strict early stopping
            #     executions_per_trial=2,  # Number of models that should be built and fit for each trial
            #     directory=fold_dir + '/random_search',
            #     project_name=model_name + '_' + use_set + '_fold_' + str(i)
            # )

            # random_tuner.search(trainset, validation_data=valset, epochs=15, callbacks=[early_stopping])
            # # Get the best hyperparameters from random search
            # rt_hyperparameters = random_tuner.get_best_hyperparameters()[0]

            bayesian_tuner = BayesianOptimization(
                lambda hp: hypermodel.build(hp, epoch_batches=total_batches),
                objective='val_mse',
                max_trials=30,
                executions_per_trial=1,
                directory=fold_dir + '/bayesian_optimization',
                project_name=model_name + '_' + use_set + '_fold_' + str(i)
                # Use the best hyperparameters from random search as a starting point
                # hyperparameters=rt_hyperparameters
            )

            # Run the hyperparameter search - early stopping is NOT used as we want the best performance AT the final epoch
            bayesian_tuner.search(trainset, validation_data=valset, epochs=15)
            best_hyperparameters = bayesian_tuner.get_best_hyperparameters()[0]
            best_model = bayesian_tuner.get_best_models(num_models=1)[0]
            print('Best model summary:')
            best_model.summary()
            plot_utils.plot_model_architecture(best_model, fold_dir)
            model = best_model
            # Save the best hyperparameters to a file
            best_hyperparameters_values = best_hyperparameters.values
            with open(os.path.join(fold_dir, 'best_hyperparameters.txt'), 'w') as f:
                for key, value in best_hyperparameters_values.items():
                    f.write(f'{key}: {value}\n')
            
        # train the (best hps if tuned) model
        history = model.fit(trainset, epochs=15, batch_size=32, verbose=1, shuffle=True, validation_data=valset)
        model.save_weights(os.path.join(fold_dir, f"validation-fold_{val_fold[0]}_weights.hdf5"))
        print('Finished training model.')
        # Plot the training and validation loss
        plot_utils.plot_model_loss(history, fold_dir, f"validation-fold_{val_fold[0]}")

        # Generate predictions on the validation set
        predictions = model.predict(valset).flatten()
        # Extract the true labels from the validation set
        true_labels = np.concatenate([y.numpy() for _, y in valset], axis=0)
        prediction_data.append((true_labels, predictions))
        df = pd.DataFrame({'true_label': true_labels, 'prediction': predictions, 'sample': sample_names_list}, index=gene_names_list)
        df.to_csv(os.path.join(fold_dir, f'validation-fold_{val_fold[0]}_predictions.tsv'), sep='\t')

        # Plot the output predictions
        plot_utils.scatter_plot(true_labels, predictions, fold_dir, f"validation-fold_{val_fold[0]}_predictions-scatter", val_fold[0])
        plot_utils.residuals_plot(true_labels, predictions, fold_dir, f"validation-fold_{val_fold[0]}_residuals-scatter", val_fold[0])

        # Check if this model has better validation performance than the current best model (always will if doing seperate folds)
        val_performance = np.min(history.history['val_mse']) 
        if val_performance < best_val_performance:
            best_val_performance = val_performance
            best_model = model

        best_model.save(os.path.join(fold_dir, 'best_model.h5'))
        with open(os.path.join(fold_dir, 'best_model_val_loss.txt'), 'w') as f:
            f.write(str(best_val_performance))
        print(f'Finished fold {i}/{folds}.')

    print('Finished complete training across all folds.')

    return


def test_model(model_name, use_set, holdout_samples, gene_sets, min_depth=0.0):
    """
    Tests a model using the specified dataset and holdout samples.

    Args:
        model_name (str): The name of the model to test.
        use_set (str): The name of the dataset to use.
        holdout_samples (list): The list of samples to use for holdout.
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

    # Loop through the holdout samples
    prediction_data = []
    for sample in holdout_samples:
        print(f'Starting holdout testing for {sample}')
        # Create a new subdirectory for the current sample
        sample_dir = os.path.join(out_dir, sample)
        os.makedirs(sample_dir, exist_ok=True)

        # Get the dataset for the current sample
        valset = get_dataset(use_set=f'tfrecords_{min_depth}x', samples=[sample], shuffle=False, batch_size=32).map(extract_features_wrapper)
        valset_genes = get_dataset(use_set=f'tfrecords_{min_depth}x', samples=[sample], shuffle=False, batch_size=32).map(extract_gene_names)
        valset_depths = get_dataset(use_set=f'tfrecords_{min_depth}x', samples=[sample], shuffle=False, batch_size=32).map(extract_gene_depths)

        # Flatten the gene_names_list and decode the byte strings to strings, etc.
        gene_names_list = [item.numpy().decode('utf-8') for batch in valset_genes for item in batch]
        gene_depths_list = [item.numpy() for batch in valset_depths for item in batch]

        # Load the best model
        best_model = tf.keras.models.load_model(os.path.join(out_dir, 'best_model.h5'), custom_objects={'CosineDecay': CosineDecay, 'WeightedMSELoss': WeightedMSELoss(), 'ClippedReLU': ClippedReLU()})
        # Generate predictions on the holdout set
        predictions = best_model.predict(valset).flatten()

        # Extract the true labels from the holdout set
        true_labels = np.concatenate([y.numpy() for _, y in valset], axis=0)
        prediction_data.append((true_labels, predictions))

        df = pd.DataFrame({'true_label': true_labels, 'prediction': predictions, 'tss_depth': gene_depths_list}, index=gene_names_list)
        df.to_csv(os.path.join(sample_dir, f'holdout-{sample}_predictions.tsv'), sep='\t')

        plot_utils.subset_scatter_plot(df, sample_dir, f"scatter_holdout_{sample}", gene_sets)
        plot_utils.subset_residuals_plot(df, sample_dir, f"residual_holdout_{sample}", gene_sets)
        plot_utils.subset_residuals_plot(df, sample_dir, f"residual-depth_holdout_{sample}", gene_sets, var="tss_depth")
            
        print(f'Finished testing model for {sample}.')
    print('Finished testing model.')


def run_model(model_path, samples_folder, out_dir):
    print('Running model . . .')
    os.makedirs(out_dir, exist_ok=True)
    # hypermodel = get_model_instance('BioCNN')
    hypermodel = get_model_instance('BioCNN')
    def extract_features_wrapper(batch):
        return hypermodel.patient_feature_extraction(batch)
    model = tf.keras.models.load_model(model_path, custom_objects={'CosineDecay': CosineDecay, 'WeightedMSELoss': WeightedMSELoss(), 'ClippedReLU': ClippedReLU()})

    # Get list of .tfrecord files in samples_folder
    samples_files = [f for f in os.listdir(samples_folder) if f.endswith('.tfrecord')]
    sample_home = "/".join(samples_folder.split("/")[1:]).rstrip('/')

    # Initialize an empty DataFrame to store all predictions
    all_predictions = pd.DataFrame()

    for sample_file in samples_files:
        # Remove the file extension to get the sample name
        sample = sample_file.replace('.tfrecord', '')
        print(f'Starting run for {sample}')

        # Get the dataset for the current sample
        valset = get_dataset(use_set=sample_home, samples=[sample], shuffle=False, batch_size=32, labeled=False).map(extract_features_wrapper)
        valset_genes = get_dataset(use_set=sample_home, samples=[sample], shuffle=False, batch_size=32, labeled=False).map(extract_gene_names)

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
    all_predictions.to_csv(os.path.join(out_dir, 'predictions.tsv'), sep='\t')

    print('Finished running model on all samples.')
    return all_predictions


def main():
    print(tf.__version__)
    """
    The main function to run the Proteus pipeline.

    This function sets up the necessary parameters for the Proteus pipeline, including paths to various data and model files, 
    and the modalities to be used in the model. If the `process_data` flag is set, it will process the data according to the
    defined modalities and save the processed data in a hierarchical format (gene -> feature-key -> data).

    Finally, it defines the parameters for the dataset to be used in the model training/testing.

    Command line arguments:
    --scalars_reference_path: Path to the scalars reference file.
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

        use_set:
    GeneStrat_0x_0CPM
    SampleStrat_0x_0CPM
    PC-HO_0x_0CPM
    SCLC-HO_0x_0CPM
    """
    # Create the parser and add the arguments
    parser = argparse.ArgumentParser(description="Run Proteus")
    parser.add_argument("--scalars_reference_path", type=str, default="/fh/fast/ha_g/user/rpatton/TritonCNN/Proteus/reference/MANE.GRCh38.v1.3.UniversalReference.tsv", help="Path to the scalars reference file")
    parser.add_argument("--model_name", type=str, default="BioCNN", help="Name of the model to use (in models/models.py))")
    parser.add_argument("--use_set", type=str, default='GeneStrat_0x_0CPM', help="The set_parameters dataset to use (see Proteus.py main() for options)")
    parser.add_argument("--process_data", type=bool, default=False, help="If True, will check but not overwrite existing sample-level h5 data")
    parser.add_argument("--test_fold", type=int, default=None, help="Specify the test fold for parallel training on different GPU nodes")
    parser.add_argument("--set_parameters", type=str, default='data/set_parameters.tsv', help="Path to the set_parameters.tsv file")
    parser.add_argument("--run_mode", type=str, default='train', help="Specify whether to 'train' (+test using CV), 'test', 'full-train', 'transfer-train',  or 'run' the model")
    parser.add_argument("--tune_hps", type=bool, default=False, help="Specify whether to tune hyperparameters")

    # Parse the arguments
    args = parser.parse_args()

    # Define the set parameters / use_sets
    """ KEYS: these are pre-defined, hard-coded fold splits with different holdout strategies and stratification methods.
    min_depth is used to enforce a depth cutoff for heldout ('run') data as well if specfied. use_zeros applies only to training."""
    set_parameters = {}
    with open(args.set_parameters, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            set_parameters[row[0]] = {'strat_method': row[1],
                                      'min_depth': float(row[2]),
                                      'min_label': float(row[3]),
                                      'holdout_samples': row[4].split(',')}

    # Define dataset paths and label files (name: [Triton top directory, RNA-Seq/truth path, TFX labels path])
    datasets = {'LuCaP-PDX':['/fh/fast/ha_g/user/rpatton/LuCaP_data/', 'labels/LuCaP-PDX_Log2CPMplus1.tsv', None],
                'SCLC-PDX':['/fh/fast/ha_g/user/rpatton/SCLC_data/', 'labels/SCLC-PDX_Log2CPMplus1.tsv', None],
                'BC-TAN':['/fh/fast/ha_g/user/rpatton/BC-TAN_data/', 'labels/BC-TAN_Log2CPMplus1.tsv', 'labels/BC-TAN_TFX_Subtype.tsv'],
                'PC-TAN':['/fh/fast/ha_g/user/rpatton/PC-TAN_data/', 'labels/PC-TAN_Log2CPMplus1.tsv', 'labels/PC-TAN_TFX_Subtype.tsv'],
                'SCLC':[None, 'labels/SCLC_Log2CPMplus1.tsv', 'labels/SCLC_TFX_Subtype.tsv'],
                'PSMA':['/fh/fast/ha_g/user/rpatton/PSMA_data/', None, 'labels/PSMA_TFX_Subtype.tsv'],
                'WCDT':[None, 'labels/WCDT_Log2CPMplus1.tsv', None]}
    
    # Process all data into sample-level h5 files (or check if it needs to be processed)
    if args.process_data:
        for dataset, set_paths in datasets.items():
            if set_paths[0] is None:
                continue
            # Define the modality dictionary
            """ MODALITY DICT: this is the dictionary that defines the global modalities to be used in the model, and where to find them.
            These are also hard-coded into the preprocessing scripts, so if you change them here, you must change them there as well."""
            modality_dict = {'TSS_images': ['image', set_paths[0] + 'Triton_TSS/results/'],
                            'TSS_scalars': ['scalars', set_paths[0] + 'Triton_TSS/results/'],
                            'GB_scalars': ['scalars', set_paths[0] + 'Triton_GB/results/'],
                            'ref_scalars': ['ref', ['NumExons', 'transcript_length', 'mRNA_size'], args.scalars_reference_path]}
            if set_paths[1] is not None:
                modality_dict['truth_labels'] = ['label', set_paths[1]]
            if set_paths[2] is not None:
                modality_dict['TFX_label'] = ['tfx', set_paths[2]]
            # Process the data
            """ PROCESS DATA: if True, will check but not overwrite existing SAMPLE-LEVEL h5 data
            -> the outer structure of the saved, sample-level data will be a dictionary with gene names as keys
            -> the inner structure of the saved, sample-level data will be a dictionary with feature-keys defined by the modality_dict"""
            print('Starting data processing for ' + dataset + ' . . .')
            generate_data(dataset, modality_dict, args.scalars_reference_path, overwrite=False)
        print('Finished processing data.')

    # Run Proteus.py in training mode if specified
    if args.run_mode == 'train' or args.run_mode == 'test':
        # Check if use_set key is in the dictionary
        if args.use_set in set_parameters:
            print(set_parameters[args.use_set])
            # Get the parameters for this use_set
            strat_method = set_parameters[args.use_set]['strat_method']
            min_depth = set_parameters[args.use_set]['min_depth']
            min_label = set_parameters[args.use_set]['min_label']
            holdout_samples = set_parameters[args.use_set]['holdout_samples']
            # Define common parameters
            folds = 5
            use_modes = ['TSS_images', 'TSS_scalars', 'GB_scalars', 'ref_scalars', 'TFX_label', 'truth_labels']
            # Check if the output files already exist
            output_files_exist = all(os.path.exists(f'data/{args.use_set}/combined_fold-{i}.tfrecord') for i in range(1, folds + 1))
            # Get all files in the training directories (default to PDX training)
            all_files = ['data/LuCaP-PDX/h5_files/' + f for f in os.listdir('data/LuCaP-PDX/h5_files/')] + ['data/SCLC-PDX/h5_files/' + f for f in os.listdir('data/SCLC-PDX/h5_files/')]
            data_files = [f for f in all_files if f.endswith('_CompiledProcessedData.h5')]
            # Check if holdout samples are named or bulk:
            if len(holdout_samples) == 1:
                single_sample = holdout_samples[0]
                samples = [f.replace('_CompiledProcessedData.h5', '') for f in data_files]
                if single_sample.startswith("PLUS_"):
                    group_id = single_sample.replace("PLUS_", "")
                    holdout_samples = [sample for sample in samples if group_id in sample]
                elif single_sample.startswith("MINUS_"):
                    group_id = single_sample.replace("MINUS_", "")
                    holdout_samples = [sample for sample in samples if group_id not in sample]
            # Only run generate_sets if the output files do not exist
            if not output_files_exist:
                generate_sets(args.use_set, strat_method, data_files, use_modes, holdout_samples, folds=folds, min_depth=min_depth, min_label=min_label)
                print('Finished generating sets.')
            else:
                print('Training data already exists. Continuing.')
        else:
            print('No training use_set specified which matches the keys in ' + args.set_parameters + '. Exiting.')
            quit()

        # Define the holdout data path based on min_depth
        holdout_data_path = f'data/tfrecords_{min_depth}x/'
        # Check if the output files (holdout) already exist
        output_ho_exist = all(os.path.exists(f'{holdout_data_path}{i}.tfrecord') for i in holdout_samples)

        # Only run generate_sets if the output files do not exist
        if not output_ho_exist:
            generate_holdout(use_modes, holdout_samples, holdout_data_path, min_depth=min_depth)
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
        if args.run_mode == 'train':
            if args.test_fold is not None and args.test_fold > 0:
                # train only on the specified fold, then exit
                print(f'Running fold {args.test_fold} in parallel . . .')
                train_model(args.model_name, args.use_set, folds, test_fold=args.test_fold, tune_hyperparameters=args.tune_hps)
            else:
                # train on all folds
                train_model(args.model_name, args.use_set, folds, tune_hyperparameters=args.tune_hps)
            print('Finished training on all CV folds.')

        # After ALL folds have been run load the default (presumed best - update default based on HPs) model and test on hold out data:
        if args.test_fold is None or args.test_fold == 0:
            print('Training model using all CV folds for holdout evaluation . . .')
            out_dir = 'results_training/' + args.model_name + '_' + args.use_set + '/'
            hypermodel = get_model_instance(args.model_name)

            def extract_features_wrapper(batch):
                return hypermodel.feature_extraction(batch)

            trainset = get_dataset(use_set=args.use_set, fold_numbers=range(1, folds + 1), shuffle=True, batch_size=32).map(extract_features_wrapper)
            print('Finished combining training data.')
            print('Training model . . .')
            total_batches = sum(1 for _ in trainset)
            print(f'Total training batches (size 32): {total_batches}')  # used for setting the Cosine Decay schedule to ensure it decays in N epochs
            model = hypermodel.build_default(epoch_batches=total_batches)

            # train the model
            model.fit(trainset, epochs=15, batch_size=32, verbose=1, shuffle=True)
            model.save_weights(os.path.join(out_dir, 'All-CV_weights.hdf5'))
            model.save(os.path.join(out_dir, 'All-CV_model.h5'))
            print('Finished training.')
            print('Testing model on holdout samples . . .')
            test_model(args.model_name, args.use_set, holdout_samples, min_depth=min_depth)
            print('Finished!')

    # Transfer learning mode
    elif args.run_mode == 'transfer-train':

        training_home = 'data/samples_training/'
        training_samples = ['18-101',
                            '19-001',
                            '19-048']
        # Define output directory and create it if it doesn't exist
        out_dir = 'data/composite_training/'
        os.makedirs(out_dir, exist_ok=True)
        # Define the training data path based on min_depth
        training_data_path = f'data/tfrecords_0.0x/'
        # Check if the output files (holdout) already exist
        output_training_exist = all(os.path.exists(f'{training_data_path}{i}.tfrecord') for i in training_samples)
        # Only run generate_sets if the output files do not exist
        use_modes = ['TSS_images', 'TSS_scalars', 'GB_scalars', 'ref_scalars', 'TFX_label', 'truth_labels']
        if not output_training_exist:
            generate_holdout('data/samples_training/', use_modes, training_samples, training_data_path, min_depth=0.0)
            print('Finished generating sets.')
        else:
            print('Training data already exists. Continuing.')

        # Create a new .tfrecord file to write the combined data
        with tf.io.TFRecordWriter(f'{out_dir}TransferSamples_0x_0CPM.tfrecord') as writer:
            for tfrecord_file in [f'{training_data_path}{sample}.tfrecord' for sample in training_samples]:
                # Read the .tfrecord file
                for record in tf.data.TFRecordDataset(tfrecord_file):
                    # Write the record to the new file
                    writer.write(record.numpy())
        print('Finished combining training data.')

        print('Training model . . .')
        out_dir = 'final_models/'
        os.makedirs(out_dir, exist_ok=True)

        # Instantiate the hypermodel
        hypermodel = get_model_instance('BioCNN')
        model = hypermodel.build_default(epoch_batches=10000)

        def extract_features_wrapper(batch):
            return hypermodel.feature_extraction(batch)

        # Load transfer training data
        ds = tf.data.TFRecordDataset('data/composite_training/TransferSamples_0x_0CPM.tfrecord')
        ds = ds.cache()
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        # Parse the data
        ds = ds.map(lambda x: parse_serialized_example(x, labeled=True))
        # Batch the data
        ds = ds.batch(32)
        trainset = ds.map(extract_features_wrapper)

        total_batches = sum(1 for _ in trainset)
        print(f'Total training batches (size 32): {total_batches}')  # used for setting the Cosine Decay schedule to ensure it decays in N epochs
        model = hypermodel.build_default(epoch_batches=total_batches,
                                         initial_learning_rate=0.00001,  # 10% of initial learning rate
                                         decay_epochs=10)  # decay to 0 over 10 instead of 15 epochs
        model.load_weights('final_models/NUM_weights.hdf5')
        print('Completed extracting data and loading base model.')

        # train the model
        model.fit(trainset, epochs=10, batch_size=32, verbose=1, shuffle=True)
        model.save_weights(os.path.join(out_dir, f"NUM-TT_weights.hdf5"))
        model.save(os.path.join(out_dir, 'NUM-TT_model.h5'))
        print('Finished transfer training.')


    # Run Proteus.py in run mode if specified
    elif args.run_mode == 'run':
        ### Patient data ###
        min_val_depth = 0.0
        use_modes = ['TSS_images', 'TSS_scalars', 'GB_scalars', 'TFX_label', 'ref_scalars']
        patient_homes = ['/fh/fast/ha_g/user/rpatton/BC-TAN_data/',
                         '/fh/fast/ha_g/user/rpatton/PSMA_data/',
                         '/fh/fast/ha_g/user/rpatton/PC-TAN_data/']
        for patient_home in patient_homes:
            modality_dict = {'TSS_images': ['image', patient_home + 'Triton_TSS/results/'],
                            'TSS_scalars': ['scalars', patient_home + 'Triton_TSS/results/'],
                            'GB_scalars': ['scalars', patient_home + 'Triton_GB/results/'],
                            'TFX_label': ['tfx', patient_home + 'labels/'],
                            'ref_scalars': ['ref', ['NumExons', 'transcript_length', 'mRNA_size'], args.scalars_reference_path]}
            sample_group = patient_home.split('/')[-2]
            group_dir = 'data/' + sample_group + '/'
            os.makedirs(group_dir, exist_ok=True)
            h5_dir = group_dir + 'h5_files/'
            os.makedirs(h5_dir, exist_ok=True)
            print('Starting data processing for ' + sample_group + ' . . .')
            generate_data(modality_dict, args.scalars_reference_path, h5_dir, overwrite=False)
            print('Finished processing data.')
            # Get list of files in each directory
            samples_files = [f for f in os.listdir(h5_dir) if f.endswith('_CompiledProcessedData.h5')]
            tf_dir = f'{group_dir}tfrecords_{min_val_depth}x/'
            os.makedirs(tf_dir, exist_ok=True)
            tf_files = os.listdir(tf_dir)
            # Remove file extensions and keep only unique names
            samples_names = set([f.replace('_CompiledProcessedData.h5', '') for f in samples_files])
            tf_names = set([f.replace('.tfrecord', '') for f in tf_files])
            # Find names that are in samples_names but not in tf_names
            missing_names = samples_names - tf_names
            # Call generate_holdout with missing_names
            generate_holdout(h5_dir, use_modes, list(missing_names), tf_dir, min_depth=min_val_depth, filter_samples=True)
            output_base = 'results_holdout/' + sample_group + '/'
            os.makedirs(output_base, exist_ok=True)
            model = 'final_models/NUM-TT_model.h5'
            run_model(model, tf_dir, output_base)

    # train a complete model on all training data
    elif args.run_mode == 'full-train':
        min_depth = 0.0
        use_modes = ['TSS_images', 'TSS_scalars', 'GB_scalars', 'ref_scalars', 'TFX_label', 'truth_labels']
        # convert the training data to a single tfrecord
        training_home = 'data/samples_training/'
        training_samples = [f.replace('_CompiledProcessedData.h5', '') for f in os.listdir(training_home)]
        # Define output directory and create it if it doesn't exist
        out_dir = 'data/composite_training/'
        os.makedirs(out_dir, exist_ok=True)

        # Define the training data path based on min_depth
        training_data_path = f'data/tfrecords_{min_depth}x/'
        # Check if the output files (holdout) already exist
        output_training_exist = all(os.path.exists(f'{training_data_path}{i}.tfrecord') for i in training_samples)

        # Only run generate_holdout if the output files do not exist
        if not output_training_exist:
            generate_holdout('data/samples_training/', use_modes, training_samples, training_data_path, min_depth=min_depth)
            print('Finished generating sets.')
        else:
            print('Training data already exists. Continuing.')

        # Create a new .tfrecord file to write the combined data
        with tf.io.TFRecordWriter(f'{out_dir}AllSamples_0x_0CPM.tfrecord') as writer:
            for tfrecord_file in glob.glob(f'{training_data_path}*.tfrecord'):
                # Read the .tfrecord file
                for record in tf.data.TFRecordDataset(tfrecord_file):
                    # Write the record to the new file
                    writer.write(record.numpy())
        print('Finished combining training data.')

        print('Training model . . .')
        out_dir = 'final_models/'
        os.makedirs(out_dir, exist_ok=True)

        # Instantiate the hypermodel
        hypermodel = get_model_instance('BioCNN')
        model = hypermodel.build_default(epoch_batches=10000)

        def extract_features_wrapper(batch):
            return hypermodel.feature_extraction(batch)
        
        # Print the model summary
        print('Model summary (base model if tuning hyperparameters):')
        model.summary()
        plot_utils.plot_model_architecture(model, out_dir)

        # Load all combined training data
        ds = tf.data.TFRecordDataset('data/composite_training/AllSamples_0x_0CPM.tfrecord')
        ds = ds.cache()
        ds = ds.shuffle(1024*8)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        # Parse the data
        ds = ds.map(lambda x: parse_serialized_example(x, labeled=True))
        # Batch the data
        ds = ds.batch(32)
        trainset = ds.map(extract_features_wrapper)

        total_batches = sum(1 for _ in trainset)
        print(f'Total training batches (size 32): {total_batches}')  # used for setting the Cosine Decay schedule to ensure it decays in N epochs
        model=hypermodel.build_default(epoch_batches=total_batches)
        print('Completed extracting data.')

        # train the model
        model.fit(trainset, epochs=15, batch_size=32, verbose=1, shuffle=True)
        model.save_weights(os.path.join(out_dir, f"NUM_weights.hdf5"))
        model.save(os.path.join(out_dir, 'NUM_model.h5'))
        print('Finished complete training on all data.')

                
    else:
        print('Invalid run mode specified. Exiting.')
        quit()



if __name__ == "__main__":
    main()
