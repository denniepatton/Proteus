"""
model_utils.py
v1.0.1, 12/15/2023
Author: Robert D Patton, rpatton@fredhutch.org
Originally implemented by: Akira Nair
Ha Lab, Fred Hutchinson Cancer Research Center

This file contains utility functions for models.
"""

# Third party imports
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Reshape, Multiply


class ScalarAttention(Layer):
    def __init__(self, **kwargs):
        super(ScalarAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(input_shape[-1], activation='softmax')
        super(ScalarAttention, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, scalars)
        # compute attention scores
        attention_scores = self.dense(x)
        # apply attention scores
        attention_output = attention_scores * x
        return attention_output

    def compute_output_shape(self, input_shape):
        return input_shape
    

class CrossModalAttention(Layer):
    def __init__(self, **kwargs):
        super(CrossModalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(input_shape[0][-1], activation='softmax')
        super(CrossModalAttention, self).build(input_shape)

    def call(self, inputs):
        signals, scalars = inputs
        # signals shape: (batch, flattened)
        # scalars shape: (batch, scalars)
        # compute attention scores based on the scalars
        attention_scores = self.dense(scalars)
        # apply attention scores to the signals
        attention_output = Multiply()([signals, attention_scores])
        return attention_output

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def print_continuous_label_distribution(dataset):
    continuous_labels = []

    for _, labels in dataset:
        # labels is a batch of 'continuous_label'
        continuous_labels.extend(labels.numpy())

    total_samples = len(continuous_labels)

    # Bin the 'continuous_label' labels into 10 bins of equal size
    hist, bin_edges = np.histogram(continuous_labels, bins=10)
    for i in range(10):
        percentage = (hist[i] / total_samples) * 100
        print(f"Continuous Label Bin {i} ({bin_edges[i]} to {bin_edges[i+1]}): {percentage}%")
