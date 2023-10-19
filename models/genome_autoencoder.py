import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Activation, \
    GlobalAveragePooling1D, Reshape, Concatenate, MultiHeadAttention, TimeDistributed, Lambda, Attention, Add
import pandas as pd
import numpy as np
import logging


class GenomeAutoencoder(Model):
    def __init__(self, gene_pool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gene_pool = gene_pool
        self.d_model = len(gene_pool)
        self.dense1 = Dense(1000, activation='relu')
        self.dense2 = Dense(100, activation='relu')
        self.decoded = Dense(self.d_model, activation='linear')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        z = self.dense2(x)
        decoded = self.decoded(z)
        return decoded
    
    def get_embedding(self, inputs):
        x = self.dense1(inputs)
        z = self.dense2(x)
        return z

    def scale(self, data):
        scaled = [data.get(gene, 0) for gene in self.gene_pool]
        return np.array(scaled)
            


def create_autoencoder(labels_data):
    d_model = labels_data.shape[1]
    input_layer = Input(shape=(d_model,))
    dense = Dense(1000, activation='relu')(input_layer) 
    z = Dense(100, activation='relu')(dense)
    decoded = Dense(d_model, activation='linear')(z)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def noise_data(array, noise_mean=0, noise_std_dev=0.1, dropout=30, noise_factor=50):
    n_genes_dropped = dropout
    n_noise_samples = noise_factor
    modified_rows = []
    original_rows = []

    for row in array:
        for _ in range(n_noise_samples):
            # get random indices to dropout
            indices_to_replace = np.random.choice(range(len(row)), size=n_genes_dropped, replace=False)
            # make a copy of the current row
            modified_row = np.copy(row)
            # create noise for the row
            noise = np.random.normal(noise_mean, noise_std_dev, size=modified_row.shape)
            # add noise to the data
            modified_row += noise
            # set dropout indices to 0
            modified_row[indices_to_replace] = 0
            modified_rows.append(modified_row)
            original_rows.append(row)
    noise_array = np.array(modified_rows)
    original_array = np.array(original_rows)
    return noise_array, original_array

def iterative_correction(genome_autoencoder: GenomeAutoencoder, predictions, n_iters=1, learning_rate=0.1):
    # unpack predictions to full genome
    predictions_resized = genome_autoencoder.scale(predictions)
    prediction_table = np.zeros((len(predictions_resized), n_iters+1))
    prediction_table[:, 0] = predictions_resized
    for n in range(n_iters):
        logging.info(f"Iter {n}")
        current_predictions = prediction_table[:, n]
        model_predictions = genome_autoencoder.predict(current_predictions).reshape(-1)
        new_predictions = current_predictions + learning_rate*(model_predictions-current_predictions)
        prediction_table[:, n+1] = new_predictions
    return prediction_table



