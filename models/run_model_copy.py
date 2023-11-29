#!/usr/bin/env python
'''
@File    :   run_model.py
@Author  :   Alex Netzley, adapted from Akira Nair
'''

import argparse
import sys
import json
import pickle
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()


def args_parse(args):
    parser = argparse.ArgumentParser(description="Run deep learning model on Triton output to infer gene expression.")
    parser.add_argument("-n", "--name", required=True, help="Specify a name for the model (REQUIRED)")
    parser.add_argument("-o", "--output", default="/fh/fast/users/anetzley/TritonCNNv3/model/_models", help="Output folder for model")
    #parser.add_argument("-d", "--data", default=None, help="Path to .npz file with compiled data")
    parser.add_argument("--model_version", default=None, type=str, help="If specified, uses a pre-defined architecture")
    parser.add_argument("--modalities", choices=["images", "scalars", "all"], default="images", help="Which modalities to include in the model")
    parser.add_argument("--task", choices=[0, 2, 3], default=0, type=int, help="Number of classes to predict from. If 0, creates a regression model." )
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs to run model for")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate of model")
    parser.add_argument("--use_attention", action='store_true', default=False, help="Use attention model")
    parser.add_argument("--seed", default=206, type=int, help="Global seed")
    #parser.add_argument("--split", default="/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/splits_seed41.csv", help="A file for training, validation, testing splits")
    args = parser.parse_args(args)
    return args

def run(args):
    ### init... ###
    import numpy as np
    import logging
    import tensorflow as tf
    import utils
    import os
    OUTPUT = utils.init(output=args.output, name=args.name, use_time=True)
    utils.set_seed(args.seed)
    logging.info(f"***SETTINGS --- {args}")
    
    
    # ### Load data ###
    # if args.data is None:
    #     if args.task==0:
    #         data_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/compiled_data_l2tpm.npz'
    #     elif args.task==3:
    #         data_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/categorical/compiled_data_three_class.npz'
    #     elif args.task==2:
    #         data_path = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/categorical/compiled_data_strict_binary.npz'
    #     else:
    #         raise ValueError(f"User specified a model task of value {args.task} which does not have a default data path. Please "
    #                          "specify a path to a .npz data file using the --data flag.")
    # else:
    #     data_path = args.data
    # data = np.load(data_path)
    # filtered_training, filtered_validation, filtered_testing = utils.open_data(data, args.split, keys=args.modalities)
    # logging.info("Loaded data.")
    import sys
    sys.path.append("..")
    from preprocessing.generate_tf_splits import *


    ### Prepare data for model input ###
    if args.modalities == "images":
        full_trainset = get_dataset(fold_numbers = [1, 2, 3, 4], use_set = 'GeneStrat_CompleteInputs_11-21-2023', shuffle=True, batch_size=32)
        full_testset = get_dataset(fold_numbers = [5], use_set = 'GeneStrat_CompleteInputs_11-21-2023', shuffle=True, batch_size=32)
        print('Completed Extracting Full Sets')

        
        # Extract only the features you need
        def extract_features(batch):
            return {'input_1': batch['TSS_image']}, {'dense_2': batch['binary_label']}

        trainset = full_trainset.map(extract_features)
        testset = full_testset.map(extract_features)
        # images_train = tf.convert_to_tensor(filtered_training["images"])
        # images_val = tf.convert_to_tensor(filtered_validation["images"])
        # images_test = tf.convert_to_tensor(filtered_testing["images"])
        # model_training_inputs = images_train
        # model_val_inputs = images_val
        # model_test_inputs = images_test
        # args.i_shape = (images_train.shape[1], images_train.shape[2])
    elif args.modalities == "scalars":
        scalars_train = tf.convert_to_tensor(filtered_training["scalars"])
        scalars_val = tf.convert_to_tensor(filtered_validation["scalars"])
        scalars_test = tf.convert_to_tensor(filtered_testing["scalars"])
        model_training_inputs = scalars_train
        model_val_inputs = scalars_val
        model_test_inputs = scalars_test
        args.s_shape = scalars_train.shape[1]
    else:
        images_train = tf.convert_to_tensor(filtered_training["images"])
        images_val = tf.convert_to_tensor(filtered_validation["images"])
        images_test = tf.convert_to_tensor(filtered_testing["images"])
        scalars_train = tf.convert_to_tensor(filtered_training["scalars"])
        scalars_val = tf.convert_to_tensor(filtered_validation["scalars"])
        scalars_test = tf.convert_to_tensor(filtered_testing["scalars"])
        model_training_inputs = [images_train, scalars_train]
        model_val_inputs = [images_val, scalars_val]
        model_test_inputs = [images_test, scalars_test]
        args.i_shape = (images_train.shape[1], images_train.shape[2])
        args.s_shape = scalars_train.shape[1]

    ### Regression task ###
    if args.task==0:
        import regression_models
        logging.info(f"Creating a regression model with {args.modalities}.")
        
        # get model
        model = regression_models.get_model_type(args)
        utils.plot_model(model, OUTPUT)
        
        # convert labels to floats for regression
        labels_train = tf.convert_to_tensor(filtered_training["labels"], dtype=tf.float32)
        labels_val = tf.convert_to_tensor(filtered_validation["labels"], dtype=tf.float32)

        # model training
        logging.info(f"Training on {len(labels_train)}, validating on {len(labels_val)}")
        logging.info(f"Training model for {args.epochs} epochs...")
        history = model.fit(model_training_inputs, labels_train, 
                            epochs=args.epochs, batch_size=args.batch_size, verbose=1,
                            shuffle=True, validation_data=(model_val_inputs, labels_val), 
                            callbacks=[utils.RegLoggerCallback(), utils.es_rg])
        utils.plot_convergence_regression(history, OUTPUT)
    
    ### Classification task ###
    else:
        import classification_models
        logging.info(f"Creating a classification model to predict {args.task} classes with {args.modalities}.")
        
        # get model
        model = classification_models.get_model_type(args)
        utils.plot_model(model, OUTPUT)
        
        ## convert labels to one-hot encodings for classifications
        #labels_train_ohe = tf.one_hot(tf.convert_to_tensor(filtered_training["labels"], dtype=tf.int8), depth=args.task)
        #labels_val_ohe = tf.one_hot(tf.convert_to_tensor(filtered_validation["labels"], dtype=tf.int8), depth=args.task)

        # model training
        logging.info(f"Training on {len(labels_train_ohe)}, validating on {len(labels_val_ohe)}")
        logging.info(f"Training model for {args.epochs} epochs...")
        history = model.fit(trainset, 
                            epochs=args.epochs, batch_size=args.batch_size, verbose=1,
                            shuffle=True, validation_data=testset, 
                            callbacks=[utils.LoggerCallback(), utils.es])
        utils.plot_convergence(history, OUTPUT)
        
    ### Save model ###
    model.save_weights(os.path.join(OUTPUT, "weights.hdf5"))
        
    # ### Evaluate model on all subsets of data ###
    # # predictions for training
    # predictions = model.predict(trainset)
    # utils.save_predictions(predictions=predictions,
    #                         truths=filtered_training["labels"],
    #                         samples=filtered_training["samples"],
    #                         genes=filtered_training["genes"],
    #                     output=OUTPUT,
    #                     extension="-training")
    
    # # predictions for validation
    # predictions = model.predict(model_val_inputs)
    # utils.save_predictions(predictions=predictions,
    #                         truths=filtered_validation["labels"],
    #                         samples=filtered_validation["samples"],
    #                         genes=filtered_validation["genes"],
    #                     output=OUTPUT,
    #                     extension="-validation")
    
    # # predictions for testing
    # predictions = model.predict(model_test_inputs)
    # utils.save_predictions(predictions=predictions,
    #                         truths=filtered_testing["labels"],
    #                         samples=filtered_testing["samples"],
    #                         genes=filtered_testing["genes"],
    #                     output=OUTPUT,
    #                     extension="-testing")
    
    
    with open(os.path.join(OUTPUT, "settings.json"), 'w') as f:
        t = json.dumps(vars(args))
        f.write(t)
    
    # done
    logging.info("Complete.")


def main(args):
    settings = args_parse(args)
    print(settings)
    run(settings)

if __name__ == "__main__":
    main(sys.argv[1:])
