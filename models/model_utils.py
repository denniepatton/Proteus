"""
model_utils.py
v1.0.2, 1/9/2024
Author: Robert D Patton, rpatton@fredhutch.org
Originally implemented by: Akira Nair
Ha Lab, Fred Hutchinson Cancer Research Center

This file contains utility functions for models.
"""

# Third party imports
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Multiply


class ScalarAttention(Layer):
    """
    This class implements a scalar attention mechanism as a Keras Layer.
    It computes attention scores using a dense layer with softmax activation,
    and applies these scores to the input.
    """
    def build(self, input_shape):
        """
        This method creates the dense layer with softmax activation.
        It is called by Keras once the input shape is known.
        """
        self.dense = Dense(input_shape[-1], activation='softmax')
        super().build(input_shape)

    def call(self, x):
        """
        This method computes and applies the attention scores.
        It is called by Keras for each input batch.

        Parameters:
        x (Tensor): The input tensor. Shape: (batch_size, num_scalars)

        Returns:
        Tensor: The output tensor. Shape: (batch_size, num_scalars)
        """
        attention_scores = self.dense(x)
        return attention_scores * x

    def compute_output_shape(self, input_shape):
        """
        This method returns the output shape.
        It is called by Keras to determine the output shape of the Layer.

        Parameters:
        input_shape (Tuple): The shape of the input.

        Returns:
        Tuple: The shape of the output.
        """
        return input_shape
    

class CrossModalAttention(Layer):
    """
    This class implements a cross-modal attention mechanism as a Keras Layer.
    It computes attention scores using a dense layer with softmax activation,
    and applies these scores to the input signals.
    """
    def build(self, input_shape):
        """
        This method creates the dense layer with softmax activation.
        It is called by Keras once the input shape is known.

        Parameters:
        input_shape (Tuple): The shape of the input. It should be a tuple of two tuples,
                             where the first tuple is the shape of the signals and the second
                             tuple is the shape of the scalars.
        """
        self.dense = Dense(input_shape[1][-1], activation='softmax')
        super().build(input_shape)

    def call(self, inputs):
        """
        This method computes and applies the attention scores.
        It is called by Keras for each input batch.

        Parameters:
        inputs (Tuple): The input tensors. It should be a tuple of two tensors,
                        where the first tensor is the signals and the second tensor is the scalars.

        Returns:
        Tensor: The output tensor. Shape: (batch_size, num_signals)
        """
        signals, scalars = inputs
        attention_scores = self.dense(scalars)
        return Multiply()([signals, attention_scores])

    def compute_output_shape(self, input_shape):
        """
        This method returns the output shape.
        It is called by Keras to determine the output shape of the Layer.

        Parameters:
        input_shape (Tuple): The shape of the input.

        Returns:
        Tuple: The shape of the output.
        """
        return input_shape[0]


def print_continuous_label_distribution(dataset):
    """
    Function to print the distribution of continuous labels in a dataset.

    Parameters:
    dataset (tf.data.Dataset): The dataset containing the labels.

    Returns:
    None
    """
    # Extract the continuous labels from the dataset
    continuous_labels = [labels.numpy() for _, labels in dataset]

    # Flatten the list of arrays into a single array
    continuous_labels = np.concatenate(continuous_labels)

    # Calculate the total number of samples
    total_samples = len(continuous_labels)

    # Bin the 'continuous_label' labels into 10 bins of equal size
    hist, bin_edges = np.histogram(continuous_labels, bins=10)

    # Print the percentage of samples in each bin
    for i in range(10):
        percentage = (hist[i] / total_samples) * 100
        print(f"Continuous Label Bin {i} ({bin_edges[i]} to {bin_edges[i+1]}): {percentage:.2f}%")
