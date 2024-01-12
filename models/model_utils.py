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
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Multiply, Conv1D, Concatenate


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
    

class TemporalAttention(Layer):
    """
    This class implements a temporal attention mechanism as a Keras Layer.
    It computes attention scores using a 1D convolutional layer with softmax activation,
    and applies these scores to the input. This allows the model to consider the spatial 
    relationship between timesteps when computing the attention scores.

    The attention scores are computed independently for each channel in the input.
    """
    def build(self, input_shape):
        """
        This method creates the 1D convolutional layer with softmax activation.
        It is called by Keras once the input shape is known.

        Parameters:
        input_shape (Tuple): The shape of the input. Expected to be (batch_size, timesteps, channels)
        """
        self.conv = Conv1D(input_shape[-1], kernel_size=3, padding='same', activation='softmax')
        super().build(input_shape)

    def call(self, x):
        """
        This method computes and applies the attention scores.
        It is called by Keras for each input batch.

        Parameters:
        x (Tensor): The input tensor. Shape: (batch_size, timesteps, channels)

        Returns:
        Tensor: The output tensor. Shape: (batch_size, timesteps, channels)
        """
        attention_scores = self.conv(x)
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
    It computes attention scores from one input and applies these scores to another input.
    This allows the model to focus more on certain elements of one input based on the values of the other input.

    The attention scores are computed using a dense layer with softmax activation.
    """
    def build(self, input_shape):
        """
        This method creates the dense layer with softmax activation.
        It is called by Keras once the input shape is known.

        Parameters:
        input_shape (Tuple): The shape of the input. Expected to be a tuple of two shapes, 
                             each representing the shape of one of the inputs.
        """
        self.dense = Dense(1, activation='softmax')
        super().build(input_shape)

    def call(self, inputs):
        """
        This method computes and applies the attention scores.
        It is called by Keras for each input batch.

        Parameters:
        inputs (List[Tensor]): The list of input tensors. Expected to contain two tensors.

        Returns:
        Tensor: The output tensor. Shape: (batch_size, timesteps + channels)
        """
        x, y = inputs
        attention_scores = self.dense(y)
        attention_scores = tf.expand_dims(attention_scores, -1)
        x = Multiply()([x, attention_scores])
        x = tf.squeeze(x, axis=1)
        return Concatenate()([x, y])

    def compute_output_shape(self, input_shape):
        """
        This method returns the output shape.
        It is called by Keras to determine the output shape of the Layer.

        Parameters:
        input_shape (Tuple): The shape of the input.

        Returns:
        Tuple: The shape of the output.
        """
        x_shape, y_shape = input_shape
        return (x_shape[0], x_shape[1] + y_shape[1])


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
