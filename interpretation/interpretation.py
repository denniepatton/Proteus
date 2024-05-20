"""
interpretation.py
v1.0.0, 4/29/2024
Author: Alexander Netzley, anetzley@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This module provides a wrapper for the model interpretation analysis, including
kernel visualization, SHAP, and gradient ascent.
"""

import os
import csv
import argparse
from multiprocessing import Pool, cpu_count
import importlib
from importlib import reload

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

import sys
import argparse
sys.path.append("..") 

# Loading In Model
from models.model_utils import CosineDecay, ScalarAttention, CrossModalAttention, TemporalAttention 
from models.models import get_model_instance

# Packages for Base-Pair Kernel Visualization
import matplotlib.pyplot as plt
import numpy as np

# Packages for Motif Plots for Base-Pair Kernels
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Packages for SHAP
import shap
import pickle
from preprocessing.generate_tf_splits import *
from preprocessing.compile_sample_data import *
from interpretation_utils import *

def interpretation_analysis(model, data, save_filepath, background_size=1000, test_size=100,  kernel_viz=False, shap=False):
    """ 
    Perform interpretation analysis on the model using kernel visualization and SHAP analysis.
    """
    # Kernel Visualization
    if kernel_viz:
        kernel_visualization(model, save_filepath)
        
    # SHAP Analysis
    if shap:
        shap_analysis(model, data, background_size=background_size, test_size=test_size, save_filepath=save_filepath)

def main():
    parser = argparse.ArgumentParser(description='Interpretation Analysis')
    parser.add_argument('--kernel_viz', action='store_true', default=False, help='Whether to perform kernel visualization')
    parser.add_argument('--shap', action='store_true', default=False, help='Whether to perform SHAP analysis')
    parser.add_argument('--background_size', type=int, default=1000, help='Size of background for SHAP analysis')
    parser.add_argument('--test_size', type=int, default=100, help='Size of test set for SHAP analysis')
    args = parser.parse_args()

    #Define folder loaciton, extract model and data
    save_filepath = os.path.join(os.path.dirname(os.getcwd())+'/interpretation_results/')
    best_model = tf.keras.models.load_model(os.path.join(os.pardir, 'final_models', 'TPU-0x_model.h5'), 
                                            custom_objects={'CosineDecay': CosineDecay, 
                                                            'ScalarAttention': ScalarAttention, 
                                                            'CrossModalAttention': CrossModalAttention, 
                                                            'TemporalAttention': TemporalAttention})
    
    #TODO think about how to remove this hardcooding (assuming we won't be hosting the data on git? but maybe we will be)
    input_data = get_data('/fh/fast/ha_g/user/anetzley/data_for_shap/full_dataset/input_data_CPM.npz')

    #print(args.kernel_viz, args.shap, args.background_size, args.test_size)
    interpretation_analysis(best_model, input_data, save_filepath=save_filepath, background_size=args.background_size, test_size=args.test_size, kernel_viz=args.kernel_viz, shap=args.shap)

    #print('All Done!')


if __name__ == "__main__":
    main()

