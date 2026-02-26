"""
interpretation.py
v1.2.0, 3/5/2025
Author: Alexander Netzley, anetzley@fredhutch.org
Updates: Robert Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This module provides a wrapper for the model interpretation analysis, including
kernel visualization, SHAP, and gradient ascent.
"""

import argparse
import os
import sys
import tensorflow as tf

# Local imports
sys.path.append("..")
from interpretation_utils import *
from models.model_utils import *
from proteus_utils.compile_sample_data import *
from proteus_utils.proteus_utils import *

def interpretation_analysis(model, data, save_filepath, background_size, test_size, kernel_viz=True, shap=True):
    """ 
    Perform interpretation analysis on the model using kernel visualization and SHAP analysis.
    """
    print("Performing interpretation analysis...")
    # Kernel Visualization
    if kernel_viz:
        print("Performing kernel visualization...")
        kernel_visualization(model, data, save_filepath)
        
    # SHAP Analysis
    if shap:
        print("Performing SHAP analysis...")
        shap_analysis(model, data, background_size=background_size, test_size=test_size, save_filepath=save_filepath)

def main():
    parser = argparse.ArgumentParser(description='Interpretation Analysis')
    # parser.add_argument('--model_name', type=str, default='PDX-Base_model', help='Name of model to perform analysis on')
    parser.add_argument('--kernel_viz', action='store_true', default=True, help='Whether to perform kernel visualization')
    parser.add_argument('--shap', action='store_true', default=True, help='Whether to perform SHAP analysis')
    parser.add_argument('--background_size', type=int, default=50000, help='Size of background for SHAP analysis')
    parser.add_argument('--test_size', type=int, default=5000, help='Size of test set for SHAP analysis')
    args = parser.parse_args()

    for model_name in ['BTS_bval-v2']:
        best_model = tf.keras.models.load_model(os.path.join(os.pardir, 'final_models', f'{model_name}_model.h5'), 
                                                custom_objects = {'Sampling': Sampling,
                                                                  'MergeLatents': MergeLatents,
                                                                  'TrainingChoice': TrainingChoice,
                                                                  'KLDivergence': KLDivergence,
                                                                  'ContrastiveLoss': ContrastiveLoss,
                                                                  'ReconstructionLoss': ReconstructionLoss,
                                                                  'InternalLoss': InternalLoss,
                                                                  'ClippedReLU': ClippedReLU,
                                                                  'MaskedHuberLoss': MaskedHuberLoss,
                                                                  'MaskedMSE': MaskedMSE,
                                                                  'MaskedR2': MaskedR2,
                                                                  'MaskedFocalMSE': MaskedFocalMSE,
                                                                  'MaskedFocalLoss': MaskedFocalLoss})
       
        data_ref = f'../data/BTS/combined_fold-3.tfrecord'
        for subset in ['pdx', 'mix', None]:  # 'healthy'
            # Define folder location, extract model and data
            save_filepath = os.path.join(f'{model_name}_{subset}/')
            os.makedirs(save_filepath, exist_ok=True)
            
            input_data = get_data(data_ref,
                                    regenerate_data=False,
                                    generate_num_samples=args.background_size,
                                    save_filepath=f'{model_name}_{subset}/fold-3_examples_{args.background_size}.npz',
                                    subset_data=subset)

            interpretation_analysis(best_model, 
                                    input_data, 
                                    save_filepath=save_filepath, 
                                    background_size=50000, 
                                    test_size=5000)

if __name__ == "__main__":
    main()

