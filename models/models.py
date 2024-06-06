"""
models.py
v2.1.0, 5/30/2024
Author: Robert D Patton, rpatton@fredhutch.org
with help from: Akira Nair, Alex Netzley
Ha Lab, Fred Hutchinson Cancer Research Center

This file contains models for predicting gene expression from cfDNA.

Available features are:
    'gene_name': tf.io.FixedLenFeature([], tf.string),
    'sample_name': tf.io.FixedLenFeature([], tf.string),
    'TSS_images': tf.io.FixedLenFeature([2000, 11], tf.float32),
    'TSS_scalars': tf.io.FixedLenFeature([12], tf.float32),
    'GB_scalars': tf.io.FixedLenFeature([12], tf.float32),
    'ref_scalars': tf.io.FixedLenFeature([3], tf.float32),
    'TFX_label': tf.io.FixedLenFeature([], tf.float32),
    'truth_label': tf.io.FixedLenFeature([], tf.float32)

"""

# Third party imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import (Dense, Flatten, Input, BatchNormalization,
                          Concatenate, Dropout,
                          Conv1D, Conv2D, Conv3D)
from kerastuner import HyperModel

# Local local imports
from .model_utils import CosineDecay


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


class ClippedReLU(tf.keras.layers.Layer):
    def __init__(self, max_value=17, **kwargs):
        super(ClippedReLU, self).__init__(**kwargs)
        self.max_value = max_value

    def call(self, inputs):
        return tf.keras.activations.relu(inputs, max_value=self.max_value)


class WeightedMSELoss(tf.keras.losses.Loss):
    """
    Custom loss function that applies an exponential weighting to the mean squared error.
    This exponential weighting is designed to punish errors in large predictions more than in
    small, as the model tends to over-predict by nature.
    """
    def __init__(self, loss_weight=2, reduction=tf.keras.losses.Reduction.AUTO, name='WeightedMSELoss'):
        super().__init__(reduction=reduction, name=name)
        self.loss_weight = loss_weight
    
    def call(self, y_true, y_pred):
        """
        Here if a loss_weight of 0 is passed, the loss will be the same as the standard MSE.
        Otherwise it represents the exponential scaling factor, with larger values producing
        a less severe/steep exponential. 10 is used as the horizontal shift to keep the funciton
        relatively level through 10, the most common range of predictions (raw CPM ~= 1000).
        """
        if self.loss_weight == 0:
            return tf.reduce_mean(tf.square(y_true - y_pred))
        else:
            # Define the weighting function
            weight = 1 + tf.exp((y_pred - 10) / self.loss_weight)
            # Calculate the mean squared error
            weighted_mse = tf.square(y_true - y_pred) * weight
            # Return the mean of the weighted mse
            return tf.reduce_mean(weighted_mse)


class BioCNN(HyperModel):
    """
    This class represents a biological Convolutional Neural Network (BioCNN) model.
    It extends the HyperModel class from Keras Tuner, allowing hyperparameter tuning.
    """

    def feature_extraction(self, batch):
        """
        This method extracts features from a batch of data.

        Parameters:
        batch (tf.data.Dataset): The batch of data.
        regression_output (bool): Whether the model should output regression predictions.

        Returns:
        str: The name of the output label.
        """
        return {'TSS_images': batch['TSS_images'],
                'TSS_scalars': batch['TSS_scalars'],
                'GB_scalars': batch['GB_scalars'],
                'TFX_label': batch['TFX_label'],
                'ref_scalars': batch['ref_scalars']}, batch['truth_labels']
    
    def patient_feature_extraction(self, batch):
        """
        This method extracts features from a batch of data EXCLUDING TRUTH LABELS.

        Parameters:
        batch (tf.data.Dataset): The batch of data.
        regression_output (bool): Whether the model should output regression predictions.

        Returns:
        str: The name of the output label.
        """
        return {'TSS_images': batch['TSS_image'],
                'TSS_scalars': batch['TSS_scalars'],
                'GB_scalars': batch['GB_scalars'],
                'TFX_label': batch['TFX_label'],
                'ref_scalars': batch['ref_scalars']}
    

    def build_default(self, epoch_batches=10000, initial_learning_rate=0.0001, decay_epochs=15):
        """
        This method builds the default BioCNN model.

        Parameters:
        epoch_batches (int): The number of batches per epoch, which is used to create a cosine decay schedule.
        initial_learning_rate (float): The initial learning rate for the cosine decay schedule.
        decay_epochs (int): The number of epochs with decay, after which the learning rate will be 0.

        Returns:
        tf.keras.Model: The default BioCNN model.
        """
        decay_steps = int(decay_epochs * epoch_batches)
        # Create the cosine decay schedule
        cosine_decay_schedule = CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps)
        return self.bio_cnn(learning_rate=cosine_decay_schedule)
    
    def build(self, hp, epoch_batches=10000):
        """
        This method builds a BioCNN model with hyperparameters.

        Parameters:
        hp (kerastuner.HyperParameters): The hyperparameters for the model.
        epoch_batches (int): The number of batches per epoch, which is used to create a cosine decay schedule.

        Returns:
        tf.keras.Model: The BioCNN model with hyperparameters.

        Notes: best HPs so far are in the comments to the right.
        """
        # Define the architecture hyperparameters and their ranges
        signal_filters = hp.Choice('signal_filters', values=[16, 32]) # 32
        conv3d_width = hp.Int('conv3d_width', min_value=3, max_value=5, step=1) # 4
        signal_output_channels = hp.Choice('signal_output_channels', values=[32, 64]) # 64
        n_reductions = hp.Int('n_reductions', min_value=2, max_value=4, step=1) # 3
        n_hidden_signals = hp.Int('n_hidden_signals', min_value=2, max_value=4, step=1) # 3
        signal_hidden_start = hp.Choice('signal_hidden_start', values=[256, 512, 1024]) # 512
        n_hidden_start = hp.Choice('n_hidden_start', values=[128, 256, 512]) # 256
        n_hidden_layers = hp.Int('n_hidden_layers', min_value=2, max_value=4, step=1) # 3
        dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.5, step=0.05) # 0.35
        loss_weight = hp.Choice('loss_weight', values=[2, 3, 5, 10]) # 2
        # learning rate schedule (these HPs are not passed to the model, only the schedule is)
        initial_learning_rate = hp.Choice('initial_learning_rate', values=[0.001, 0.0005, 0.0001]) # 0.001
        decay_epochs = hp.Choice('decay_epochs', values=[15]) # decay epochs, after which learning rate will be 0 (model runs on 15)
        decay_steps = int(decay_epochs * epoch_batches) # total steps
        # Create the cosine decay schedule
        cosine_decay_schedule = CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps)
        # Call bio_cnn with the hyperparameters
        model = self.bio_cnn(cosine_decay_schedule, signal_filters, conv3d_width, signal_output_channels, n_reductions, n_hidden_signals,
                             signal_hidden_start, n_hidden_start, n_hidden_layers, dropout_rate, loss_weight)
        return model
    

    
    def bio_cnn(self, learning_rate, signal_filters=32, conv3d_width=3, signal_output_channels=64, n_reductions=2, n_hidden_signals=4,
                signal_hidden_start=1024, n_hidden_start=512, n_hidden_layers=2, dropout_rate=0.4, loss_weight=0):
        """ SIGNAL PROCESSING BLOCK (initial signal processing):
        Takes in the 11 signal channels and separates them into 7 phasing/fragmentomic signals and 4 O.H. nt signals, then applies
        a 1D convolution to each channel independently in order to smooth/extract initial elements of interest from the signals,
        while applying a 4x3 2D convolution to the O.H. nt signals to capture the trimer contexts and condense them into a single signal.
        Afterwards the 8 (7 + 4 -> 1) channels are concatenated back into a single tensor with signal_filters channels.
        Kernel size of 3 is chosen specifically to highlight amino acid triplets and perform rough smoothing. 
        """
        # First handle the signals input
        signal_input = Input(shape=(2000, 11), name='TSS_images')
        # Split the input into 7 separate tensors and 1 tensor with 4 channels
        x_split_0 = signal_input[:, :, 0:1]
        x_split_1 = signal_input[:, :, 1:2]
        x_split_2 = signal_input[:, :, 2:3]
        x_split_3 = signal_input[:, :, 3:4]
        x_split_4 = signal_input[:, :, 4:5]
        x_split_5 = signal_input[:, :, 5:6]
        x_split_6 = signal_input[:, :, 6:7]
        x_split_nt = signal_input[:, :, 7:]
        # Apply a Conv1D layer to each of the first 7 x_splits
        x_conv_0 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_0)
        x_conv_1 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_1)
        x_conv_2 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_2)
        x_conv_3 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_3)
        x_conv_4 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_4)
        x_conv_5 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_5)
        x_conv_6 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_6)
        # Apply a Conv2D layer to the last x_split
        conv_layer_2d = Conv2D(filters=signal_filters, kernel_size=(4, 3), activation='relu')
        # Transpose the last split of x_splits to make the 2000 dimension the width and the 4 dimension the height
        x_transposed = tf.transpose(x_split_nt, perm=[0, 2, 1])
        # Expand dimensions at the end to add a channel dimension
        x_expanded = tf.expand_dims(x_transposed, axis=-1)
        # Apply the Conv2D layer
        x_2d_conv = conv_layer_2d(x_expanded)
        # Squeeze the second dimension
        x_2d_conv = tf.squeeze(x_2d_conv, axis=1)
        # Concatenate the results
        x_convs = tf.stack([x_conv_0, x_conv_1, x_conv_2, x_conv_3, x_conv_4, x_conv_5, x_conv_6, x_2d_conv], axis=2)
        x = BatchNormalization()(x_convs)
        """ RELATIONAL CHANNEL GENERATION BLOCK:
        Apply a 3D Convolution to the signals to find inter-signal relationships. The 3D Convolution layer will
        act as a scanning window that will slide across the time axis of the signals and find relationships between the
        signal_filters' independently extracted signals, outputting signal_output_channels combination channels."""
        # Assuming x has shape (batch, 1998, 8, signal_filters)
        # Add an extra dimension to x for the channels dimension expected by Conv3D
        x = x[..., None]
        # Apply Conv3D along the time axis and across all signals and channels
        x_channels = Conv3D(signal_output_channels, kernel_size=(conv3d_width, 8, signal_filters), activation='relu')(x)
        # Now x_channels has shape (batch, 1998, 1, 1, signal_output_channels)
        # Remove the empty dimensions so shape is (batch, 1998, signal_output_channels)
        x = tf.squeeze(x_channels, axis=[2, 3])
        x = BatchNormalization()(x)
        """ TIME-REDUCTION BLOCK:
        Apply a series of strided Conv1D layers to reduce the number of time steps by a factor of 2 each time,
        while maintaining the number of channels.
        """
        for _ in range(n_reductions):
            # use strided convolutions of kernel size 4 and stride 2 to see each timestep twice
            x = Conv1D(filters=signal_output_channels, kernel_size=4, strides=2, activation='relu')(x)
            x = BatchNormalization()(x)
        """ PROCESS SCALARS' INPUT:
        Combine the TSS and GB scalars with the reference scalars as an additional head.
        """
        # Process scalars
        tss_scalars = Input(shape=(12,), name='TSS_scalars')
        gb_scalars = Input(shape=(12,), name='GB_scalars')
        ref_scalars = Input(shape=(3,), name='ref_scalars')
        tfx = Input(shape=(1,), name='TFX_label')
        # Concatenate the scalars
        y = Concatenate()([tss_scalars, gb_scalars, ref_scalars, tfx])
        """ COMBINE SIGNALS AND SCALARS:
        A series of hidden layers reduced by half each time are applied to the flattened signals to make the output
        dimensionality more similar to the scalars, while giving the signals more solo processing.
        """
        x = Flatten()(x)
        for i in range(n_hidden_signals):
            n_dense_signal = int(signal_hidden_start / (2**i))
            x = Dense(n_dense_signal, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        z = Concatenate()([x, y])
        """ APPLY FINAL HIDDEN LAYERS BLOCK:
        These can be though of as reduction blocks, with the number of nodes halving each time to make a sparser
        representation of the data before the final output layer.
        """
        for i in range(n_hidden_layers):
            n_dense_combined = int(n_hidden_start / (2**i))
            z = Dense(n_dense_combined, activation='relu')(z)
            z = BatchNormalization()(z)
            z = Dropout(dropout_rate)(z)
        """ OUTPUT BLOCK:
        Final output uses a clipped ReLU activation to ensure positive values for log2(X + 1) which stay in positive bounds;
        a reasonable max is 20, as 2^20 is > 10^6, e.g. the upper bound of possible CPM/TPM values.
        """
        output = Dense(1, activation=ClippedReLU(max_value=20), name='output')(z)
        # Create model
        model = Model(inputs=[signal_input, tss_scalars, gb_scalars, ref_scalars, tfx], outputs=output, name='BioCNN')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=WeightedMSELoss(loss_weight=loss_weight),
                      optimizer=optimizer, 
                      metrics=['mse'])
        return model


def main():
    pass

if __name__ == "__main__":
    main()

### PREVIOUS BEST MODEL USED IN TPU TRAINING AND VALIDATION ###############################################################
# class BioCNN(HyperModel):
#     """
#     This class represents a biological Convolutional Neural Network (BioCNN) model.
#     It extends the HyperModel class from Keras Tuner, allowing hyperparameter tuning.
#     """

#     def feature_extraction(self, batch):
#         """
#         This method extracts features from a batch of data.

#         Parameters:
#         batch (tf.data.Dataset): The batch of data.
#         regression_output (bool): Whether the model should output regression predictions.

#         Returns:
#         str: The name of the output label.
#         """
#         return {'TSS_image': batch['TSS_image'],
#                 'TSS_scalars': batch['TSS_scalars'],
#                 'GB_scalars': batch['GB_scalars'],
#                 'ref_scalars': batch['ref_scalars'],}, batch['continuous_label']
    
#     def patient_feature_extraction(self, batch):
#         """
#         This method extracts features from a batch of data EXCLUDING TRUTH LABELS.

#         Parameters:
#         batch (tf.data.Dataset): The batch of data.
#         regression_output (bool): Whether the model should output regression predictions.

#         Returns:
#         str: The name of the output label.
#         """
#         return {'TSS_image': batch['TSS_image'],
#                 'TSS_scalars': batch['TSS_scalars'],
#                 'GB_scalars': batch['GB_scalars'],
#                 'ref_scalars': batch['ref_scalars']}
    

#     def build_default(self, epoch_batches=10000):
#         """
#         This method builds the default BioCNN model.

#         Parameters:
#         epoch_batches (int): The number of batches per epoch, which is used to create a cosine decay schedule.

#         Returns:
#         tf.keras.Model: The default BioCNN model.
#         """
#         # learning rate schedule - determined in HP tuning
#         initial_learning_rate = 0.0 # initial learning rate
#         decay_epochs = 20 # epochs with decay, after which learning rate will be 0
#         decay_steps = int(decay_epochs * epoch_batches)
#         warmup_target = 0.001 # target learning rate after warmup
#         warmup_steps = epoch_batches * 5 # warmup for 5 epochs
#         # Create the cosine decay schedule
#         cosine_decay_schedule = CosineDecay(initial_learning_rate=initial_learning_rate,
#                                             decay_steps=decay_steps,
#                                             warmup_steps=warmup_steps,
#                                             warmup_target=warmup_target)
#         return self.bio_cnn(learning_rate=cosine_decay_schedule)
    
#     def build(self, hp, epoch_batches=10000):
#         """
#         This method builds a BioCNN model with hyperparameters.

#         Parameters:
#         hp (kerastuner.HyperParameters): The hyperparameters for the model.
#         epoch_batches (int): The number of batches per epoch, which is used to create a cosine decay schedule.

#         Returns:
#         tf.keras.Model: The BioCNN model with hyperparameters.

#         Notes: best HPs so far are in the comments to the right.
#         """
#         # Define the architecture hyperparameters and their ranges
#         signal_filters = hp.Choice('signal_filters', values=[16, 32]) # 32
#         conv3d_width = hp.Int('conv3d_width', min_value=3, max_value=3, step=1) # 3
#         signal_output_channels = hp.Choice('signal_output_channels', values=[64]) # 64
#         n_reductions = hp.Int('n_reductions', min_value=3, max_value=4, step=1) # 3
#         n_hidden_signals = hp.Int('n_hidden_signals', min_value=4, max_value=5, step=1) # 4
#         signal_hidden_start = hp.Choice('signal_hidden_start', values=[1024, 2048, 4096]) # 1024
#         n_hidden_start = hp.Choice('n_hidden_start', values=[256, 512]) # 512
#         n_hidden_layers = hp.Int('n_hidden_layers', min_value=3, max_value=4, step=1) # 4
#         dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.2, step=0.1) # 0.2
#         # learning rate schedule (these HPs are not passed to the model, only the schedule is)
#         initial_learning_rate = hp.Choice('initial_learning_rate', values=[0.0]) # 0
#         decay_epochs = hp.Choice('decay_epochs', values=[20]) # decay epochs, after which learning rate will be 0 (model runs on 20)
#         decay_steps = int(decay_epochs * epoch_batches) # total steps
#         warmup_target = hp.Choice('warmup_target', values=[0.001]) # 0.001
#         warmup_epochs = hp.Choice('warmup_fraction', values=[2, 5, 10]) # 3
#         warmup_steps = int(warmup_epochs * decay_steps)
#         # Create the cosine decay schedule
#         cosine_decay_schedule = CosineDecay(initial_learning_rate=initial_learning_rate,
#                                             decay_steps=decay_steps,
#                                             warmup_steps=warmup_steps,
#                                             warmup_target=warmup_target)
#         # Call bio_cnn with the hyperparameters
#         model = self.bio_cnn(cosine_decay_schedule, signal_filters, conv3d_width, signal_output_channels, n_reductions,
#                              n_hidden_signals, signal_hidden_start, n_hidden_start, n_hidden_layers, dropout_rate)
#         return model
    

    
#     def bio_cnn(self, learning_rate, signal_filters=16, conv3d_width=3, signal_output_channels=64, n_reductions=3,
#                 n_hidden_signals=4, signal_hidden_start=1024, n_hidden_start=256, n_hidden_layers=4, dropout_rate=0.2):
#         """ SIGNAL PROCESSING BLOCK (initial signal processing):
#         Takes in the 11 signal channels and separates them into 7 phasing/fragmentomic signals and 4 O.H. nt signals, then applies
#         a 1D convolution to each channel independently in order to smooth/extract initial elements of interest from the signals,
#         while applying a 4x3 2D convolution to the O.H. nt signals to capture the trimer contexts and condense them into a single.
#         Afterwards the 8 (7 + 4 -> 1) channels are concatenated back into a single tensor with signal_filters channels.
#         Kernel size of 3 is chosen specifically to highlight amino acid triplets and perform rough smoothing. 
#         """
#         # First handle the signals input
#         signal_input = Input(shape=(2000, 11), name='TSS_image')
#         # Split the input into 7 separate tensors and 1 tensor with 4 channels
#         x_split_0 = signal_input[:, :, 0:1]
#         x_split_1 = signal_input[:, :, 1:2]
#         x_split_2 = signal_input[:, :, 2:3]
#         x_split_3 = signal_input[:, :, 3:4]
#         x_split_4 = signal_input[:, :, 4:5]
#         x_split_5 = signal_input[:, :, 5:6]
#         x_split_6 = signal_input[:, :, 6:7]
#         x_split_nt = signal_input[:, :, 7:]
#         # Apply a Conv1D layer to each of the first 7 x_splits
#         x_conv_0 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_0)
#         x_conv_1 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_1)
#         x_conv_2 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_2)
#         x_conv_3 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_3)
#         x_conv_4 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_4)
#         x_conv_5 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_5)
#         x_conv_6 = Conv1D(filters=signal_filters, kernel_size=3, activation='relu')(x_split_6)
#         # Apply a Conv2D layer to the last x_split
#         conv_layer_2d = Conv2D(filters=signal_filters, kernel_size=(4, 3), activation='relu')
#         # Transpose the last split of x_splits to make the 2000 dimension the width and the 4 dimension the height
#         x_transposed = tf.transpose(x_split_nt, perm=[0, 2, 1])
#         # Expand dimensions at the end to add a channel dimension
#         x_expanded = tf.expand_dims(x_transposed, axis=-1)
#         # Apply the Conv2D layer
#         x_2d_conv = conv_layer_2d(x_expanded)
#         # Squeeze the second dimension
#         x_2d_conv = tf.squeeze(x_2d_conv, axis=1)
#         # Concatenate the results
#         x_convs = tf.stack([x_conv_0, x_conv_1, x_conv_2, x_conv_3, x_conv_4, x_conv_5, x_conv_6, x_2d_conv], axis=2)
#         x = BatchNormalization()(x_convs)
#         """ RELATIONAL CHANNEL GENERATION BLOCK:
#         Apply a 3D Convolution to the signals to find inter-signal relationships. The 3D Convolution layer will
#         act as a scanning window that will slide across the time axis of the signals and find relationships between the
#         signal_filters' independently extracted signals, outputting signal_output_channels combination channels."""
#         # Assuming x has shape (batch, 1998, 8, signal_filters)
#         # Add an extra dimension to x for the channels dimension expected by Conv3D
#         x = x[..., None]
#         # Apply Conv3D along the time axis and across all signals and channels
#         x_channels = Conv3D(signal_output_channels, kernel_size=(conv3d_width, 8, signal_filters), activation='relu')(x)
#         # Now x_channels has shape (batch, 1998, 1, 1, signal_output_channels)
#         # Remove the empty dimensions so shape is (batch, 1998, signal_output_channels)
#         x = tf.squeeze(x_channels, axis=[2, 3])
#         x = BatchNormalization()(x)
#         """ TIME-REDUCTION BLOCK:
#         Apply a series of strided Conv1D layers to reduce the number of time steps by a factor of 2 each time,
#         while maintaining the number of channels.
#         """
#         for _ in range(n_reductions):
#             # use strided convolutions of kernel size 4 and stride 2 to see each timestep twice
#             x = Conv1D(filters=signal_output_channels, kernel_size=4, strides=2, activation='relu')(x)
#             x = BatchNormalization()(x)
#         """ PROCESS SCALARS' INPUT:
#         Combine the TSS and GB scalars with the reference scalars as an additional head.
#         """
#         # Process scalars
#         tss_scalars = Input(shape=(12,), name='TSS_scalars')
#         gb_scalars = Input(shape=(12,), name='GB_scalars')
#         ref_scalars = Input(shape=(3,), name='ref_scalars')
#         # Concatenate the scalars
#         y = Concatenate()([tss_scalars, gb_scalars, ref_scalars])
#         """ COMBINE SIGNALS AND SCALARS:
#         A series of hidden layers reduced by half each time are applied to the flattened signals to make the output
#         dimensionality more similar to the scalars, while giving the signals more solo processing.
#         """
#         x = Flatten()(x)
#         for i in range(n_hidden_signals):
#             n_dense_signal = int(signal_hidden_start / (2**i))
#             x = Dense(n_dense_signal, activation='relu')(x)
#             x = BatchNormalization()(x)
#             x = Dropout(dropout_rate)(x)
#         z = Concatenate()([x, y])
#         """ APPLY FINAL HIDDEN LAYERS BLOCK:
#         These can be though of as reduction blocks, with the number of nodes halving each time to make a sparser
#         representation of the data before the final output layer.
#         """
#         for i in range(n_hidden_layers):
#             n_dense_combined = int(n_hidden_start / (2**i))
#             z = Dense(n_dense_combined, activation='relu')(z)
#             z = BatchNormalization()(z)
#             z = Dropout(dropout_rate)(z)
#         """ OUTPUT BLOCK:
#         Final output uses ReLU activation to ensure positive values for log2(X + 1).
#         """
#         output = Dense(1, activation='relu', name='output')(z)
#         # Create model
#         model = Model(inputs=[signal_input, tss_scalars, gb_scalars, ref_scalars], outputs=output, name='BioCNN')
#         optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#         model.compile(loss='mse',
#                       optimizer=optimizer, 
#                       metrics=['mse'])
#         return model
####################################################################################################################################