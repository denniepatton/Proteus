"""
models.py
v2.0.2, 1/9/2024
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
    'continuous_label': tf.io.FixedLenFeature([], tf.float32)

"""

# Third party imports
from tensorflow import keras

# Keras imports
from keras import backend as K
from keras.models import Model
from keras.layers import (Dense, Flatten, Input, BatchNormalization,
                          Reshape, Concatenate, 
                          Lambda, DepthwiseConv2D, Conv1D, MaxPooling1D)
from keras.regularizers import l1_l2
from keras.losses import MeanSquaredError
from kerastuner import HyperModel

# Local application imports
from .model_utils import ScalarAttention, TemporalAttention, CrossModalAttention

def get_model_instance(model_name):
    """
    Function to get an instance of a model class from its name.

    Parameters:
    model_name (str): The name of the model class.

    Returns:
    object: An instance of the model class.
    """
    # Get the class from globals and instantiate it
    model_class = globals().get(model_name)
    
    if model_class is None:
        raise ValueError(f"No model found with name: {model_name}")

    model_instance = model_class()
    return model_instance


class BioCNN(HyperModel):
    """
    This class represents a biological Convolutional Neural Network (BioCNN) model.
    It extends the HyperModel class from Keras Tuner, allowing hyperparameter tuning.
    """

    def build_default(self):
        """
        This method builds the default BioCNN model.

        Parameters:
        regression_output (bool): Whether the model should output regression predictions.

        Returns:
        tf.keras.Model: The default BioCNN model.
        """
        return self.bio_cnn()


    def build(self, hp):
        """
        This method builds a BioCNN model with hyperparameters.

        Parameters:
        hp (kerastuner.HyperParameters): The hyperparameters for the model.

        Returns:
        tf.keras.Model: The BioCNN model with hyperparameters.
        """
        # Define the hyperparameters
        signal_filters = hp.Int('signal_filters', min_value=8, max_value=64, step=8)
        relational_filters = hp.Int('relational_filters', min_value=8, max_value=64, step=8)
        n_reductions = hp.Int('n_reductions', min_value=1, max_value=10, step=1)
        n_hidden_start = hp.Int('n_hidden_start', min_value=64, max_value=256, step=64)
        n_hidden_signals = hp.Int('n_hidden_signals', min_value=0, max_value=4, step=1)
        n_hidden_scalars = hp.Int('n_hidden_scalars', min_value=0, max_value=4, step=1)
        n_hidden_combined = hp.Int('n_hidden_combined', min_value=0, max_value=4, step=1)
        n_output_channels = hp.Int('n_output_channels', min_value=0, max_value=64, step=16)
        use_image_attention = hp.Boolean('use_image_attention')
        use_scalar_attention = hp.Boolean('use_scalar_attention')
        use_cross_modal_attention = hp.Boolean('use_cross_modal_attention')
        dense_l1 = hp.Float('dense_l1', min_value=1e-5, max_value=1, sampling='LOG')
        dense_l2 = hp.Float('dense_l2', min_value=1e-5, max_value=1, sampling='LOG')
        conv_l1 = hp.Float('conv_l1', min_value=1e-5, max_value=1, sampling='LOG')
        conv_l2 = hp.Float('conv_l2', min_value=1e-5, max_value=1, sampling='LOG')
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        # Call your function with the hyperparameters
        model = self.bio_cnn(signal_filters, relational_filters, n_reductions, n_hidden_start,
                             n_hidden_signals, n_hidden_scalars, n_hidden_combined, n_output_channels, use_image_attention,
                             use_scalar_attention, use_cross_modal_attention, learning_rate,
                             dense_l1, dense_l2, conv_l1, conv_l2)
        return model
    
    def feature_extraction(self, batch):
        """
        This method extracts features from a batch of data.

        Parameters:
        batch (tf.data.Dataset): The batch of data.
        regression_output (bool): Whether the model should output regression predictions.

        Returns:
        str: The name of the output label.
        """
        return {'TSS_image': batch['TSS_image'],
                'TSS_scalars': batch['TSS_scalars'],
                'GB_scalars': batch['GB_scalars'],
                'ref_scalars': batch['ref_scalars'],}, batch['continuous_label']
    
    def patient_feature_extraction(self, batch):
        """
        This method extracts features from a batch of data EXCLUDING TRUTH LABELS.

        Parameters:
        batch (tf.data.Dataset): The batch of data.
        regression_output (bool): Whether the model should output regression predictions.

        Returns:
        str: The name of the output label.
        """
        return {'TSS_image': batch['TSS_image'],
                'TSS_scalars': batch['TSS_scalars'],
                'GB_scalars': batch['GB_scalars'],
                'ref_scalars': batch['ref_scalars']}

    def bio_cnn(self, signal_filters=48, relational_filters=24, n_reductions=6, n_hidden_start=128,
                n_hidden_signals=0, n_hidden_scalars=0, n_hidden_combined=3, n_output_channels=0, use_image_attention=False,
                use_scalar_attention=False, use_cross_modal_attention=True, learning_rate=1e-3,
                dense_l1=1e-4, dense_l2=1e-4, conv_l1=1e-3, conv_l2=1e-3):
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
            x = TemporalAttention()(x)
        """ TIME-REDUCTION BLOCK [optional]:
        Apply a series of MaxPooling1D and Conv1D layers to reduce the number of time steps.
        Then, flatten the signals into a single vector before going into hidden layers.
        """
        for _ in range(n_reductions):
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=signal_filters, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=conv_l1, l2=conv_l2))(x)
            x = BatchNormalization()(x)
        x = Flatten()(x)
        """ SIGNALS' HIDDEN LAYERS BLOCK:
        Apply hidden layers to the flattened signals before combining with scalars.
        """
        for i in range(n_hidden_signals):
            # start with n_hidden_start and reduce by half each time
            n_dense = int(n_hidden_start / (2**i))
            x = Dense(n_dense, activation='relu', kernel_regularizer=l1_l2(l1=dense_l1, l2=dense_l2))(x)
            x = BatchNormalization()(x)
        """ SCALARS' HIDDEN LAYERS BLOCK:
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
        Concatenate the signals and scalars and apply hidden layers to the combined data. If cross-modal attention is used,
        signals are weighted by scalars' attention mechanism before concatenation.
        """
        if use_cross_modal_attention:
            z = CrossModalAttention()([x, y])
        else:
            z = Concatenate()([x, y])
        n_dense_combined_start = 128
        for i in range(n_hidden_combined):
            # start with n = 80 channels and reduce by half each time
            n_dense_combined = int(n_dense_combined_start / (2**i))
            z = Dense(n_dense_combined, activation='relu', kernel_regularizer=l1_l2(l1=dense_l1, l2=dense_l2))(z)
            z = BatchNormalization()(z)
        """ OUTPUT BLOCK:
        Final output uses softplus activation to ensure positive values for log2(TPM + 1) with more range close to 0.
        """
        output = Dense(1, activation='softplus', name='output')(z)
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