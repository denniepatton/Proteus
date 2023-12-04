# Alex Netzley
# 12/01/2023
#!/usr/bin/env python
# '''
# @File    :   run_model.py
# @Author  :   Alex Netzley, adapted from Akira Nair
# '''
import os
import pandas as pd
import tensorflow as tf

import argparse
import sys
import json
import pickle

import numpy as np
import logging

from preprocessing import generate_tf_splits
from models import utils, classification_models, regression_models
import os
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()


def args_parse(args):
    parser = argparse.ArgumentParser(description="Run deep learning model on Triton output to infer gene expression.")
    parser.add_argument("-n", "--name", required=True, help="Specify a name for the model (REQUIRED)")
    parser.add_argument("-o", "--output", default="/fh/fast/ha_g/user/anetzley/TritonCNNv3/model/_models", help="Output folder for model")
    #parser.add_argument("-d", "--data", default=None, help="Path to .npz file with compiled data")
    parser.add_argument("--model_version", default=None, type=str, help="If specified, uses a pre-defined architecture")
    parser.add_argument("--modalities", choices=["images", "scalars", "all"], default="images", help="Which modalities to include in the model")
    parser.add_argument("--task", choices=[0, 1], default=0, type=int, help="0 specifies regression model, 1 specifies binary classification" )
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs to run model for")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate of model")
    parser.add_argument("--use_attention", action='store_true', default=False, help="Use attention model")
    parser.add_argument("--seed", default=206, type=int, help="Global seed")
    args = parser.parse_args(args)
    return args

def run(args):
    ### init... ###
    OUTPUT = utils.init(output=args.output, name=args.name, use_time=True)
    utils.set_seed(args.seed)
    logging.info(f"***SETTINGS --- {args}")
    
    ### Extract Split Data ###
    full_trainset = generate_tf_splits.get_dataset(fold_numbers = [1, 2, 3, 4], use_set = 'GeneStrat_CompleteInputs_11-21-2023', shuffle=True, batch_size=args.batch_size)
    full_testset = generate_tf_splits.get_dataset(fold_numbers = [5], use_set = 'GeneStrat_CompleteInputs_11-21-2023', shuffle=True, batch_size=args.batch_size)
    print('Completed Extracting Full Sets')
    
    ### Process args for future use ###
    if args.task == 1:
        label = 'binary_label'
    else:
        label = 'continuous_label'

    if args.modalities == 'images':
        modalities = ['TSS_image']
    elif args.modalities == 'scalars':
        modalities = ['TSS_scalars']
    else:
        modalities = ['TSS_image', 'TSS_scalars']

    ### Prepare data for model input ###
    #Define a function to parse the dataset and extract only the features we want
    def extract_features(batch, modalities, labels):
        #Extract all inputs
        inputs = {}
        for modality in modalities:
            inputs[modality] = batch[modality]

        return inputs, {'output': batch[labels]}

    trainset = full_trainset.map(lambda batch: extract_features(batch, modalities, label))
    testset = full_testset.map(lambda batch: extract_features(batch, modalities, label))

    #Extract the first element from the dataset so we can get the shape of each input
    first_element = next(iter(trainset.take(1)))
    #print(first_element)
    
    #Extract the shape of all items in first_element
    for modality in modalities:
        #print(first_element[0][modality][0].shape)
        if modality == 'TSS_image':
            input_img_shape = first_element[0][modality][0].shape
            args.i_shape = ([input_img_shape[0], input_img_shape[1]])
            #print(args.i_shape)
        else:
            input_s_shape = first_element[0][modality][0].shape
            args.s_shape = (input_s_shape[0])


    ### Regression task ###
    if args.task==0:
        logging.info(f"Creating a regression model with {args.modalities}.")
        
        # get model
        model = regression_models.get_model_type(args)
        utils.plot_model(model, OUTPUT)
        
        # # convert labels to floats for regression
        # labels_train = tf.convert_to_tensor(filtered_training["labels"], dtype=tf.float32)
        # labels_val = tf.convert_to_tensor(filtered_validation["labels"], dtype=tf.float32)

        # model training
        #logging.info(f"Training on {len(labels_train)}, validating on {len(labels_val)}")
        logging.info(f"Training model for {args.epochs} epochs...")
        history = model.fit(trainset,
                            epochs=args.epochs, batch_size=args.batch_size, verbose=1,
                            shuffle=True, validation_data=testset, 
                            callbacks=[utils.RegLoggerCallback(), utils.es_rg])
        utils.plot_convergence_regression(history, OUTPUT)
    
    ### Classification task ###
    else:
        
        logging.info(f"Creating a classification model to predict {args.task} classes with {args.modalities}.")
        
        # get model
        model = classification_models.get_model_type(args)
        utils.plot_model(model, OUTPUT)

        # model training
        #logging.info(f"Training on {len(labels_train)}, validating on {len(labels_test)}")
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
    #main(['test_classifier', '/fh/fast/users/anetzley/TritonCNNv3/model/_models', 'GeneStrat_CompleteInputs_11-21-2023', 'images', '0', '1', '32', '1e-3', '0', '206'])
