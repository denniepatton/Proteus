"""
Proteus.py
v1.3.2, 12/20/2023
Author: Robert D Patton, rpatton@fredhutch.org
with help from: Alex Netzley
Ha Lab, Fred Hutchinson Cancer Research Center
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
from kerastuner.tuners import RandomSearch, BayesianOptimization

# Local application imports
from analysis import plot_utils
from models.models import *
from preprocessing.generate_tf_splits import *
from preprocessing.compile_sample_data import *


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
    # Define output directory and create it if it doesn't exist
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

    # Prepare arguments for multiprocessing
    args = [(sample, regions_dict, ref_path, out_dir, overwrite) for sample in common_samples]

    # Create a pool of workers and process each sample
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
    # Define output directory and create it if it doesn't exist
    out_dir = 'data/' + name + '/'
    os.makedirs(out_dir, exist_ok=True)

    # Write the arguments to a TSV file
    with open(f'{out_dir}{name}.tsv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['name', 'sample_data_path', 'use_modes', 'validation_samples', 'folds'])
        writer.writerow([name, sample_data_path, use_modes, validation_samples, folds])

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


def generate_validation(sample_data_path, use_modes, validation_samples):
    """
    Generates validation data.

    Args:
        sample_data_path (str): The path to the sample data.
        use_modes (list): The list of modalities to use.
        validation_samples (list): The list of samples to use for validation.

    Returns:
        None
    """
    # Define output directory and create it if it doesn't exist
    out_dir = 'data/validation/'
    os.makedirs(out_dir, exist_ok=True)

    # Get all files in the directory and filter the list of files
    all_files = os.listdir(sample_data_path)
    data_files = [f for f in all_files if f.endswith('_CompiledProcessedData.h5')]
    samples = [f.replace('_CompiledProcessedData.h5', '') for f in data_files]

    # Filter out validation samples and their corresponding data files
    samples, data_files = zip(*[(sample, data_file) for sample, data_file in zip(samples, data_files) if sample in validation_samples])

    print('Starting new validation .tf . . .')

    # Prepare arguments for multiprocessing
    args_list = [(sample, sample_data, sample_data_path, use_modes, out_dir) for sample, sample_data in zip(samples, data_files)]

    # Create a pool of workers and process each sample
    with Pool(cpu_count()) as p:
        p.starmap(process_validation_sample, args_list)

    print('Finished generating validation data.')


def train_model(model_name, use_set, folds, test_fold=None, regression_output=True, tune_hyperparameters=True):
    """
    Trains a model using the specified dataset and folds.

    Args:
        model_name (str): The name of the model to train.
        use_set (str): The name of the dataset to use.
        folds (int): The number of folds for cross-validation.
        test_fold (int, optional): If specified, will only conduct the CV with test_fold as test. Defaults to None.
        regression_output (bool, optional): If True, the model will output regression values. Defaults to True.
        tune_hyperparameters (bool, optional): If True, hyperparameters will be tuned. Defaults to True.

    Returns:
        None
    """
    print('Training model . . .')
    out_dir = 'results/' + model_name + '_' + use_set + '/'
    os.makedirs(out_dir, exist_ok=True)

    # Instantiate the hypermodel
    hypermodel = get_model_from_string(model_name)

    # Define the strategy for parallel training
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope
        model = hypermodel.build_default(regression_output=regression_output)

        def extract_features_wrapper(batch):
            return hypermodel.feature_extraction(batch, regression_output)
        
        # Print the model summary
        print('Model summary (base model if tuning hyperparameters):')
        model.summary()
        plot_utils.plot_model(model, out_dir)

        # Loop over the folds
        prediction_data = []
        best_model = None
        best_val_performance = float('inf')
        for i in range(folds):
            # If test_fold is specified and this is not the test_fold, skip this iteration
            if test_fold is not None and i != test_fold:
                continue

            print(f'Starting fold {i+1}/{folds}')

            # Create a new subdirectory for the current fold
            fold_dir = os.path.join(out_dir, f'CV-fold_{i+1}')
            os.makedirs(fold_dir, exist_ok=True)

            # Use the current fold for validation and all other folds for training
            train_folds = [j+1 for j in range(folds) if j != i]
            val_fold = [i+1]

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
                    max_trials=30,
                    executions_per_trial=3,
                    directory=fold_dir + '/bayesian_optimization',
                    project_name=model_name + '_' + use_set + '_fold_' + str(i+1)
                )

                # Define early stopping
                early_stopping = EarlyStopping(monitor='val_loss', patience=5)

                # Run the hyperparameter search
                tuner.search(train_dataset, validation_data=val_dataset, epochs=10, callbacks=[early_stopping])
                best_model = tuner.get_best_models(num_models=1)[0]
                print('Best model summary:')
                best_model.summary()
                plot_utils.plot_model(best_model, fold_dir)
                model = best_model

            # Re-define callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

            # model training
            history = model.fit(trainset, epochs=100, batch_size=32, verbose=1, shuffle=True, validation_data=valset, callbacks=[early_stopping, reduce_lr])
            model.save_weights(os.path.join(fold_dir, f"validation-fold_{val_fold[0]}_weights.hdf5"))
            print('Finished training model.')

            # Plot the training and validation loss
            if regression_output:
                plot_utils.plot_convergence_regression(history, fold_dir, f"validation-fold_{val_fold[0]}")
            else:
                plot_utils.plot_convergence_classification(history, fold_dir, f"validation-fold_{val_fold[0]}")

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
            if regression_output:
                plot_utils.scatter_plot(true_labels, predictions, fold_dir, f"scatter_validation-fold_{val_fold[0]}", val_fold[0])
            else:
                plot_utils.roc_plot([(true_labels, predictions)], fold_dir, f"ROC_validation-fold_{val_fold[0]}")

            # Check if this model has better validation performance than the current best model (always will if doing seperate folds)
            val_performance = np.min(history.history['val_loss']) 
            if val_performance < best_val_performance:
                best_val_performance = val_performance
                best_model = model

            best_model.save(os.path.join(fold_dir, 'best_model.h5'))
            with open(os.path.join(fold_dir, 'best_model_val_loss.txt'), 'w') as f:
                f.write(str(best_val_performance))
            print(f'Finished fold {i+1}/{folds}.')

    print('Finished complete training across all folds.')

    return


def test_model(model_name, use_set, validation_samples, gene_sets, regression_output=True):
    """
    Tests a model using the specified dataset and validation samples.

    Args:
        model_name (str): The name of the model to test.
        use_set (str): The name of the dataset to use.
        validation_samples (list): The list of samples to use for validation.
        gene_sets (list): The list of gene sets to use.
        regression_output (bool, optional): If True, the model will output regression values. Defaults to True.

    Returns:
        None
    """
    print('Starting testing model . . .')
    out_dir = 'results/' + model_name + '_' + use_set + '/'
    os.makedirs(out_dir, exist_ok=True)

    hypermodel = get_model_from_string(model_name)
    def extract_features_wrapper(batch):
        return hypermodel.feature_extraction(batch, regression_output)

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

        # Flatten the gene_names_list and decode the byte strings to strings
        gene_names_list = [item.numpy().decode('utf-8') for batch in valset_genes for item in batch]

        # Load the best model
        best_model = tf.keras.models.load_model(os.path.join(out_dir, 'best_model.h5'))
        # Generate predictions on the validation set
        predictions = best_model.predict(valset).flatten()

        # Extract the true labels from the validation set
        true_labels = np.concatenate([y.numpy() for _, y in valset], axis=0)
        prediction_data.append((true_labels, predictions))

        df = pd.DataFrame({'true_label': true_labels, 'prediction': predictions}, index=gene_names_list)
        df.to_csv(os.path.join(sample_dir, f'validation-{sample}_predictions.tsv'), sep='\t')

        if regression_output:
            plot_utils.subset_scatter_plot(df, sample_dir, f"scatter_validation_{sample}", gene_sets)
        else:
            plot_utils.subset_roc_plot(df, sample_dir, f"ROC_validation_{sample}", gene_sets)
            
        print(f'Finished testing model for {sample}.')

    if not regression_output:  # Plot the composite ROC curve
        plot_utils.roc_plot(prediction_data, out_dir, 'ROC_all_validation_samples', curve_names=validation_samples)
    print('Finished testing model.')


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

    NOTE: this function has been optimized to run on as many GPUs as are available on the machine. At Fred Hutch some
    dunce decided to put every GPU core on its own node, so the native methods for multi-GPU training don't work (for now).
    Instead, training folds can be run in parallel on different GPU (nodes) by specifying a test fold on the command line.
    """
    # Create the parser and add the arguments
    parser = argparse.ArgumentParser(description="Run Proteus")
    parser.add_argument("--scalars_reference_path", type=str, default="/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/Ensembl109_reference.bed", help="Path to the scalars reference file")
    parser.add_argument("--training_home", type=str, default="/fh/fast/ha_g/user/rpatton/LuCaP_data/", help="Path to the training home directory")
    parser.add_argument("--sample_data_path", type=str, default="/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/data/samples/", help="Path to the sample data directory")
    parser.add_argument("--model_name", type=str, default="BioCNN", help="Name of the model to use (in models/models.py))")
    parser.add_argument("--use_set", type=str, default="GeneStrat_CompleteInputs_12-14-2023", help="The dataset to use (see Proteus.py main() for options)")
    parser.add_argument("--process_data", type=bool, default=False, help="If True, will check but not overwrite existing sample-level h5 data")
    parser.add_argument("--test_fold", type=int, default=None, help="Specify the test fold for parallel training on different GPU nodes")

    # Parse the arguments
    args = parser.parse_args()

    # Define the gene sets used for validation plots (hard-coded for now)
    gene_sets = {'Housekeeping': '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/HousekeepingGenes.txt',
                 'Non-Housekeeping': '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/NonHousekeepingGenes.txt',
                 'Targets': '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/Targets.txt'}
    gene_sets = {key: open(path, 'r').read().splitlines() for key, path in gene_sets.items()}

    # Define the modality dictionary
    """ MODALITY DICT: this is the dictionary that defines the modalities to be used in the model, and where to find them.
    These are also hard-coded into the preprocessing scripts, so if you change them here, you must change them there as well."""
    modality_dict = {'TSS_image': ['image', args.training_home + 'Triton_TSS/results/'],
                     'TSS_scalars': ['scalars', args.training_home + 'Triton_TSS/results/'],
                     'GB_scalars': ['scalars', args.training_home + 'Triton_GB/results/'],
                     'ref_scalars': ['ref', ['exon_count', 'transcript_length'], args.scalars_reference_path],
                     'continuous_label': ['label', '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/labels/LuCaP_Log2TPMp1.tsv'],
                     'binary_label': ['label', '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/labels/LuCaP_Log2TPM_AZ-Score_0Binary.tsv']}
    
    # Process data if the flag is set
    """ PROCESS DATA: if True, will check but not overwrite existing sample-level h5 data
    -> the outer structure of the saved, sample-level data will be a dictionary with gene names as keys
    -> the inner structure of the saved, sample-level data will be a dictionary with feature-keys defined by the modality_dict"""
    if args.process_data:
        print('Starting data processing . . .')
        generate_data(modality_dict, args.scalars_reference_path, overwrite=False)
        print('Finished processing data.')

    # Define the set parameters
    """ USE_SETS: these are pre-defined, hard-coded fold splits with different holdout strategies and stratification methods
    'GeneStrat_CompleteInputs_12-14-2023' -> gene stratified
    'SampleStrat_CompleteInputs_12-14-2023' -> sample stratified
    'NEPC-HO_CompleteInputs_12-14-2023' -> all NEPCs held out """
    set_parameters = {
        'GeneStrat_CompleteInputs_12-14-2023': {
            'strat_method': 'gene',
            'validation_samples': ['LuCaP_93', 'LuCaP_136', 'LuCaP_96']
        },
        'SampleStrat_CompleteInputs_12-14-2023': {
            'strat_method': 'sample',
            'validation_samples': ['LuCaP_93', 'LuCaP_136', 'LuCaP_96']
        },
        'NEPC-HO_CompleteInputs_12-14-2023': {
            'strat_method': 'gene',
            'validation_samples': ['LuCaP_93', 'LuCaP_145-1', 'LuCaP_173-1', 'LuCaP_208-4', 'LuCaP_145-2', 'LuCaP_49']
        }
    }

    # Check if use_set is in the dictionary
    if args.use_set in set_parameters:
        # Get the parameters for this use_set
        strat_method = set_parameters[args.use_set]['strat_method']
        validation_samples = set_parameters[args.use_set]['validation_samples']
        # Define common parameters
        folds = 5
        use_modes = ['TSS_image', 'TSS_scalars', 'GB_scalars', 'ref_scalars', 'continuous_label', 'binary_label']
        # Check if the output files already exist
        output_files_exist = all(os.path.exists(f'data/{args.use_set}/combined_fold-{i}.tfrecord') for i in range(1, folds + 1))
        # Only run generate_sets if the output files do not exist
        if not output_files_exist:
            generate_sets(args.use_set, strat_method, args.sample_data_path, use_modes, validation_samples, folds=folds)
            print('Finished generating sets.')
        else:
            print('Training data already exists. Continuing.')
    else:
        print('No set specified. Exiting.')
        quit()

    # Check if the output files (validation) already exist
    output_valids_exist = all(os.path.exists(f'data/validation/{i}.tfrecord') for i in validation_samples)
    # Only run generate_sets if the output files do not exist
    if not output_valids_exist:
        generate_validation(args.sample_data_path, use_modes, validation_samples)
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
            train_model(args.model_name, args.use_set, folds, test_fold=args.test_fold)
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
            best_model = tf.keras.models.load_model(best_model_path)
            best_model.save(os.path.join(base_dir, 'best_model.h5'))
            # test the model on hold out data
            test_model(args.model_name, args.use_set, validation_samples, gene_sets)
        else:
            print('Invalid test fold specified. Exiting.')
            quit()
    else:
        # Train and test compeltely using all availble GPUs
        train_model(args.model_name, args.use_set, folds)
        # Test the model on hold out data
        test_model(args.model_name, args.use_set, validation_samples, gene_sets)


if __name__ == "__main__":
    main()
