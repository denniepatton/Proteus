"""
regression_models.py
v2.0.1, 12/14/2023
Author: Robert D Patton, rpatton@fredhutch.org
with help from: Akira Nair, Alex Netzley
Ha Lab, Fred Hutchinson Cancer Research Center

This file contains models for predicting gene expression from cfDNA.

Available features are:
    'gene_name': tf.io.FixedLenFeature([], tf.string),
    'TSS_image': tf.io.FixedLenFeature([2000, 11], tf.float32),
    'TSS_scalars': tf.io.FixedLenFeature([12], tf.float32),
    'GB_scalars': tf.io.FixedLenFeature([12], tf.float32),
    'ref_scalars': tf.io.FixedLenFeature([2], tf.float32),
    'continuous_label': tf.io.FixedLenFeature([], tf.float32),
    'binary_label': tf.io.FixedLenFeature([], tf.float32)

"""

# Third party imports
import numpy as np
import tensorflow as tf

# Keras imports
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (Dense, Flatten, Dropout, Input, BatchNormalization, Conv2D,
                          Reshape, Concatenate, MultiHeadAttention, TimeDistributed,
                          Lambda, Attention, Add, DepthwiseConv2D, LSTM, Conv1D, MaxPooling1D, MaxPooling2D)
from keras.regularizers import l1_l2
from keras.losses import MeanSquaredError
from kerastuner import HyperModel

# Local application imports
from .model_utils import ScalarAttention, CrossModalAttention

def get_model_from_string(model_name):
    # Get the class from globals and instantiate it
    model_class = globals()[model_name]
    model_instance = model_class()
    return model_instance


class BioCNN(HyperModel):

    def build_default(self, regression_output=True):
        return self.bio_cnn(regression_output=regression_output)

    def build(self, hp):
        # Define the hyperparameters
        signal_filters = hp.Int('signal_filters', min_value=8, max_value=32, step=8)
        relational_filters = hp.Int('relational_filters', min_value=8, max_value=32, step=8)
        n_output_channels = hp.Int('n_output_channels', min_value=0, max_value=128, step=16)
        n_reductions = hp.Int('n_reductions', min_value=0, max_value=3, step=1)
        n_lstms = hp.Int('n_lstms', min_value=0, max_value=3, step=1)
        lstm_channels = hp.Int('lstm_channels', min_value=64, max_value=256, step=64)
        n_hidden_signals = hp.Int('n_hidden_signals', min_value=1, max_value=3, step=1)
        n_hidden_scalars = hp.Int('n_hidden_scalars', min_value=1, max_value=3, step=1)
        n_hidden_combined = hp.Int('n_hidden_combined', min_value=1, max_value=3, step=1)
        use_image_attention = hp.Boolean('use_image_attention')
        use_scalar_attention = hp.Boolean('use_scalar_attention')
        use_cross_modal_attention = hp.Boolean('use_cross_modal_attention')
        dense_l1 = hp.Float('dense_l1', min_value=1e-5, max_value=1, sampling='LOG')
        dense_l2 = hp.Float('dense_l2', min_value=1e-5, max_value=1, sampling='LOG')
        conv_l1 = hp.Float('conv_l1', min_value=1e-5, max_value=1, sampling='LOG')
        conv_l2 = hp.Float('conv_l2', min_value=1e-5, max_value=1, sampling='LOG')
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        # Define non-hyperparameter arguments
        regression_output = hp.Fixed('regression_output', value=True)  # or value=False for binary classification
        # Call your function with the hyperparameters
        model = self.bio_cnn(signal_filters, relational_filters, n_reductions, n_lstms, lstm_channels,
                             n_hidden_signals, n_hidden_scalars, n_hidden_combined, n_output_channels, use_image_attention,
                             use_scalar_attention, use_cross_modal_attention, learning_rate, regression_output,
                             dense_l1, dense_l2, conv_l1, conv_l2)
        return model
    
    def feature_extraction(self, batch, regression_output=True):
        if regression_output:
            output_label = 'continuous_label'
        else:
            output_label = 'binary_label'
        return {'TSS_image': batch['TSS_image'],
                'TSS_scalars': batch['TSS_scalars'],
                'GB_scalars': batch['GB_scalars'],
                'ref_scalars': batch['ref_scalars'],}, batch[output_label]

    def bio_cnn(self, signal_filters=32, relational_filters=16, n_reductions=0, n_lstms=0, lstm_channels=128,
                n_hidden_signals=1, n_hidden_scalars=1, n_hidden_combined=1, n_output_channels=64, use_image_attention=False,
                use_scalar_attention=False, use_cross_modal_attention=False, learning_rate=1e-3, regression_output=True,
                dense_l1=0.01, dense_l2=0.01, conv_l1=0.01, conv_l2=0.01):
        """ SIGNAL PROCESSING BLOCK:
        Takes in the 11 signal channels and applies a 1D convolution to each channel independently in order to smooth/extract
        features from the signals. Then, the 11 channels are concatenated back into a single tensor with signal_filters channels.
        Kernel size of 3 is chosen specifically to highlight amino acid triplets and perform rough smoothing. Likewise, the
        number of signal_filters should exceed the number of amino acids (20) to allow for complete feature extraction.
        """
        # First handle the signals input
        signal_input = Input(shape=(2000, 11), name='TSS_image')
        # Split the input into 11 separate tensors
        split_layers = [Lambda(lambda x: x[:, :, i:i+1]) for i in range(11)]
        x_splits = [layer(signal_input) for layer in split_layers]  # Now each x_split is of shape (batch_size, 2000, 1)
        # Apply a Conv1D layer to each x_split
        conv_layers = [Conv1D(filters=signal_filters, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=conv_l1, l2=conv_l2)) for _ in range(11)]
        x_convs = [layer(x_split) for layer, x_split in zip(conv_layers, x_splits)]  # Now each x_conv is of shape (batch_size, 1998, signal_filters)
        # Concatenate the results
        x = Concatenate(axis=2)(x_convs)  # Now x is of shape (batch_size, 1998, 11 * signal_filters)
        x = Reshape((1998, 11, signal_filters))(x)  # Now x is of shape (batch_size, 1998, 11, signal_filters)
        x = BatchNormalization()(x) # batch normalization after each convolution
        """ RELATIONAL CHANNEL GENERATION BLOCK:
        Apply a DepthwiseConv2D layer to the signals to find inter-signal relationships. The DepthwiseConv2D layer will
        act as a scanning window that will slide across the time axis of the signals and find relationships between the
        signal_filters independently extracted signals. These are added as additional relational channels beyond the 11
        signal channels, each with signal_filters channels, e.g. so the total number of *signals* is 11 + relational filters
        and the total number of *channels* is signal_filters.
        """
        # Assuming x has shape (batch, 1998, 11, signal_filters)
        # Apply DepthwiseConv2D along the time axis and across all signals
        x_channels = DepthwiseConv2D(kernel_size=(1, 11), data_format='channels_last', depth_multiplier=relational_filters, activation='relu', kernel_regularizer=l1_l2(l1=conv_l1, l2=conv_l2))(x)
        # Now x_channels has shape (batch, 1998, 1, signal_filters * relational_filters)
        x_channels = Reshape((1998, relational_filters, signal_filters))(x_channels)  # remove the extra dimension
        x_channels = BatchNormalization()(x_channels)
        # Now x_channels has shape (batch, 1998, relational_filters, signal_filters)
        # Concatenate x and x_channels along the signal axis
        x = Concatenate(axis=2)([x, x_channels])
        # Now x has shape (batch, 1998, 11 + n_channels, n_filters)
        # Get the true shape of the tensor 'x'
        shape = K.int_shape(x)
        # Reshape 'x' into shape (batch, 1998, (11 + n_channels) * n_filters) such that each timestep is a vector of length (11 + n_channels) * n_filters
        x = Reshape((-1, shape[2] * shape[3]))(x)
        """ CHANNEL-REDUCTION BLOCK [optional]:
        Apply a 1D convolution to signal tensors to reduce the total number of channels to n_output_channels.
        """
        if n_output_channels > 0:
            x = Conv1D(filters=n_output_channels, kernel_size=1, activation='relu', kernel_regularizer=l1_l2(l1=conv_l1, l2=conv_l2))(x)
            x = BatchNormalization()(x)
        """ IMAGE ATTENTION BLOCK [optional]:
        Apply self-attention across timesteps.
        """
        if use_image_attention:
            x = ScalarAttention()(x)
        """ TIME-REDUCTION BLOCK [optional]:
        Apply a series of MaxPooling1D and Conv1D layers to reduce the number of time steps and increase the number
        of channels. This is done to reduce the number of parameters in the model and to increase the number of channels, which
        may also be helpful for downstream LSTMs, if used.
        """
        for _ in range(n_reductions):
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=signal_filters, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=conv_l1, l2=conv_l2))(x)
            x = BatchNormalization()(x)
        """ LSTM BLOCK [optional]:
        Apply a series of LSTM layers to the signals to find more complex time-dependent relationships between the signals.
        """
        if n_lstms > 0:  # use LSTM layers to flatten the signals
            for _ in range(n_lstms-1):
                x = LSTM(lstm_channels, return_sequences=True)(x)
            x = LSTM(lstm_channels)(x)
        else:  # simply flatten the signals
            x = Flatten()(x)
        """ SIGNALS' HIDDEN LAYERS BLOCK:
        Apply hidden layers to the flattened signals before combining with scalars.
        """
        for i in range(n_hidden_signals):
            # start with the number of channels corresponding the number of LSTM channels (if or if not used) and reduce by half each time
            n_dense = int(lstm_channels / (2**i))
            x = Dense(n_dense, activation='relu', kernel_regularizer=l1_l2(l1=dense_l1, l2=dense_l2))(x)
            x = BatchNormalization()(x)
        """ SCLARS' HIDDEN LAYERS BLOCK:
        Separate input for scalars and apply hidden layers to the scalars.
        """
        # Process scalars
        tss_scalars = Input(shape=(12,), name='TSS_scalars')
        gb_scalars = Input(shape=(12,), name='GB_scalars')
        ref_scalars = Input(shape=(2,), name='ref_scalars')
        # Concatenate the scalars
        y = Concatenate()([tss_scalars, gb_scalars, ref_scalars])
        if use_scalar_attention:
            y = ScalarAttention()(y)
        n_dense_scalars_start = 32
        for i in range(n_hidden_scalars):
            # start with n = 32 > n_scalars and reduce by half each time
            n_dense_scalars = int(n_dense_scalars_start / (2**i))
            y = Dense(n_dense_scalars, activation='relu', kernel_regularizer=l1_l2(l1=dense_l1, l2=dense_l2))(y)
            y = BatchNormalization()(y)
        """ COMBINE AND FINAL HIDDEN LAYERS BLOCK:
        Concatenate the signals and scalars and apply hidden layers to the combined data.
        """
        if use_cross_modal_attention:
            z = CrossModalAttention()([x, y])
        else:
            z = Concatenate()([x, y])
        n_dense_combined_start = 80
        for i in range(n_hidden_combined):
            # start with n = 80 channels and reduce by half each time
            n_dense_combined = int(n_dense_combined_start / (2**i))
            z = Dense(n_dense_combined, activation='relu', kernel_regularizer=l1_l2(l1=dense_l1, l2=dense_l2))(z)
            z = BatchNormalization()(z)
        """ OUTPUT BLOCK:
        Final output uses softplus activation to ensure positive values for log2(TPM + 1) with more range close to 0.
        """
        if regression_output:
            output = Dense(1, activation='softplus', name='output')(z)
        else:
            output = Dense(1, activation='sigmoid', name='output')(z)
        # Create model
        model = Model(inputs=[signal_input, tss_scalars, gb_scalars, ref_scalars], outputs=output, name='BioCNN')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=MeanSquaredError(),
                    optimizer=optimizer, 
                    metrics=['mse'])
        return model
    
def main():
    pass

if __name__ == "__main__":
    main()