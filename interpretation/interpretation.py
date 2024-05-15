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

#model_path = "/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/results_holdout/UW-WGS_data/model.h5"
#model_path = "/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/results_holdout/UW-WGS_data/best_model.h5"
#best_model = tf.keras.models.load_model(model_path, custom_objects={'CosineDecay': CosineDecay, 'ScalarAttention': ScalarAttention, 'CrossModalAttention': CrossModalAttention, 'TemporalAttention': TemporalAttention})
# print(best_model.summary())

#filepath = "/fh/fast/ha_g/user/rpatton/TritonCNN/Proteus/data/composite_training/AllSamples_0x_0CPM.tfrecord"

#Optional load filepaths
#save_filepath = '/fh/fast/ha_g/user/anetzley/data_for_shap/full_dataset/input_data_CPM_Full.npz'

#input_data = get_data(filepath, regenerate_data=True, regenerate_num_samples=None, save_filepath=save_filepath)

# input_ASCL1 = get_gene_subset_data("/fh/fast/ha_g/user/rpatton/TritonCNN/Proteus/data/composite_training/AllSamples_0x_0CPM.tfrecord", gene_list_filepath=['ASCL1'], regenerate_data=True, save_filepath='/fh/fast/ha_g/user/anetzley/data_for_shap/ASCL1/ascl1_cpm.npz')

def interpretation_analysis(model, data, save_filepath, background_size=5000, test_size=1000,  kernel_viz=True, shap=True):
    """ 
    """
    # Kernel Visualization
    if kernel_viz:
        kernel_visualization(best_model, save_filepath)
    # SHAP Analysis
    if shap:
        shap_analysis(best_model, input_data, background_size=5000, test_size=1000, save_filepath=save_filepath)

#Define folder loaciton, extract model and data
save_filepath = os.path.join(os.path.dirname(os.getcwd())+'/interpretation_results/')
best_model = tf.keras.models.load_model(os.path.join(os.pardir, 'final_models', 'TPU-0x_model.h5'), custom_objects={'CosineDecay': CosineDecay, 'ScalarAttention': ScalarAttention, 'CrossModalAttention': CrossModalAttention, 'TemporalAttention': TemporalAttention})
input_data = get_data('/fh/fast/ha_g/user/anetzley/data_for_shap/full_dataset/input_data_CPM.npz')

interpretation_analysis(best_model, input_data, save_filepath=save_filepath, background_size=5000, test_size=1000 )