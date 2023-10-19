import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Activation, \
    GlobalAveragePooling1D, Reshape, Concatenate, MultiHeadAttention, TimeDistributed, Lambda, Attention, Add
from keras.regularizers import l1, l2, l1_l2
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, AveragePooling1D
from keras.losses import MeanSquaredError
# from . import utils
import utils
import argparse
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import os
import sys

def get_model_type(args):
    if args.model_version is not None:
        models = {
            "v1": v1,
            "v2": v2,
            "v3": v3,
            "v4": v4,
            "v5": v5,
            "v6": v6,
            "v9": v9,
        }
        model = models[args.model_version]
        return model(steps=args.i_steps, features=args.i_features, scalar_features=args.s_features)
    if args.modalities == "images":
        if args.use_attention:
            return regression_only_images_attention(input_dim=args.i_shape, 
                                                    learning_rate=args.learning_rate)
        else:
            return regression_only_images(input_dim=args.i_shape,
                                                    learning_rate=args.learning_rate)
    elif args.modalities == "scalars":
        if args.use_attention:
            return regression_only_scalars_attention(learning_rate=args.learning_rate, 
                                                     n_features=args.s_features)
        else:
            return regression_only_scalars(learning_rate=args.learning_rate, 
                                           n_features=args.s_features)
    else:
        if args.use_attention:
            return regression_all_attention(input_dim_i=args.i_shape,
                                            input_dim_s=args.s_shape,
                                            learning_rate=args.learning_rate)
        else:
            return regression_all(input_dim_i=args.i_shape,
                                    input_dim_s=args.s_shape,
                                    learning_rate=args.learning_rate)


def regression_only_images(input_dim=(2000, 11), learning_rate=1e-3, n_filters=64, n_convolutions=1):
    sequential_input = Input(shape=input_dim)
    x = Reshape((input_dim[0], input_dim[1], 1))(sequential_input)
    bp_model = Sequential([
        Flatten(),
        BatchNormalization(),
        Dense(3)
    ])
    x = TimeDistributed(bp_model)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=n_filters, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)        
    for _ in range(n_convolutions-1):
        x = Conv1D(filters=n_filters, kernel_size=3, activation='relu',
               kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(1)(x)

    # Create the model
    model = Model(inputs=sequential_input, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def regression_only_scalars(input_dim=14, learning_rate=1e-3, n_hidden=1):
    scalars_in = Input(shape=(input_dim,))
    x = Dense(50, activation='relu')(scalars_in)
    x = BatchNormalization()(x)
    for _ in range(n_hidden-1):
        x = Dense(50, activation='relu')(x)
        x = BatchNormalization()(x)
    output = Dense(1)(x)
    # Create model
    model = Model(inputs=scalars_in, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def regression_all(input_dim_i=(2000, 11), input_dim_s=14, learning_rate=1e-3, n_filters=64, n_convolutions=1, n_hidden=1):
    sequential_input = Input(shape=input_dim_i)
    x = Reshape((input_dim_i[0], input_dim_i[1], 1))(sequential_input)
    bp_model = Sequential([
        Flatten(),
        BatchNormalization(),
        Dense(3)
    ])
    x = TimeDistributed(bp_model)(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=n_filters, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    for _ in range(n_convolutions-1):
        x = Conv1D(filters=n_filters, kernel_size=3, activation='relu',
               kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)        
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    
    scalars_in = Input(shape=(input_dim_s,))
    y = Dense(50, activation='relu')(scalars_in)
    y = BatchNormalization()(y)
    for _ in range(n_hidden-1):
        y = Dense(50, activation='relu')(y)
        y = BatchNormalization()(y)
    
    z = Concatenate()([x, y])
    z = Dense(200, activation='relu')(z)
    output = Dense(1)(z)

    # Create model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def regression_only_images_attention(input_dim=(2000, 11), learning_rate=1e-3):
    sequential = Input(shape=input_dim)
    x = Attention()([sequential, sequential])
    x = BatchNormalization()(x)
    x = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(1)(x)
    # Create model
    model = Model(inputs=sequential, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def regression_only_scalars_attention(input_dim=14, learning_rate=1e-3):
    scalars_in = Input(shape=(input_dim,))
    x = Attention()([scalars_in, scalars_in])
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(1)(x)
    # Create model
    model = Model(inputs=scalars_in, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def regression_all_attention(input_dim_i=(2000, 11),input_dim_s=14,learning_rate=1e-3):
    sequential = Input(shape=input_dim_i)
    x = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(sequential)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)

    scalars_in = Input(shape=(input_dim_s,))
    y = Flatten()(scalars_in)
    y = BatchNormalization()(y)
    y = Dense(50, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dense(50, activation='relu')(y)

    # apply cross attention
    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)
    xy = MultiHeadAttention(num_heads=2, key_dim=64)(x, y, y)
    yx = MultiHeadAttention(num_heads=2, key_dim=64)(y, x, x)
    xy = Flatten()(xy)
    yx = Flatten()(yx)

    z = Concatenate()([xy, yx])
    z = BatchNormalization()(z)
    z = Dense(100, activation='relu')(z)
    output = Dense(1)(z)
    
    
    # Create model
    model = Model(inputs=[sequential, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

### SPECIFIC ARCHITECTURE VERSIONS ###

def v1(steps=2000, features=11, scalar_features=18):
    """
    This model performed with R^2: 0.79, MAE: 0.34, RMSE: 0.469

    Args:
        steps (int, optional): _description_. Defaults to 2000.
        features (int, optional): _description_. Defaults to 11.
        scalar_features (int, optional): _description_. Defaults to 18.

    Returns:
        _type_: _description_
    """
    sequential_input = Input(shape=(steps, features))
    x = Reshape((steps, features, 1))(sequential_input)
    x = TimeDistributed(Conv1D(filters=8, kernel_size=3, padding='same'), input_shape=(steps, features, 1))(x)
    x = Lambda(lambda t: tf.reduce_sum(t, axis=-1))(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(200)(x)
    scalars_in = Input(shape=(scalar_features,))
    y = Dense(50, activation='relu')(scalars_in)
    y = Dropout(0.1)(y)
    y = Dense(50, activation='relu')(y)

    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)
    xy = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(x, y, y)
    yx = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(y, x, x)
    xy = Flatten()(xy)
    yx = Flatten()(yx)
    z = Concatenate()([xy, yx])
    
    z = Dense(100, activation='relu')(z)
    output = Dense(1)(z)

    # Create model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def v2(steps=2000, features=11, scalar_features=18):
    """
    This model performed with...
    sbatch model/run_model.sh --use_full_data --name "REGRESSION_V2" --model_version "v2" --epochs 30 --task 0 --modalities "all" --output "/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY18_RUN"
    Args:
        steps (int, optional): _description_. Defaults to 2000.
        features (int, optional): _description_. Defaults to 11.
        scalar_features (int, optional): _description_. Defaults to 18.

    Returns:
        _type_: _description_
    """
    # images
    sequential_input = Input(shape=(steps, features))
    x = Reshape((steps, features, 1))(sequential_input)
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    # small-kernel convolution
    x1 = TimeDistributed(Conv1D(filters=4, kernel_size=3, padding='valid'))(x)
    x1 = Lambda(lambda t: tf.reduce_sum(t, axis=-1))(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    
    # large-kernel convolution
    x2 = TimeDistributed(Conv1D(filters=4, kernel_size=166, padding='valid'))(x)
    x2 = Lambda(lambda t: tf.reduce_sum(t, axis=-1))(x2)
    x2 = BatchNormalization()(x2)
    
    # linear projection of small and large kernel convolution
    # operations
    x1 = Flatten()(x1)
    x1 = Dense(60)(x1)
    x2 = Flatten()(x2)
    x2 = Dense(60)(x2)
    
    # concatenation and joint representation projection
    x = Concatenate()([x1, x2])
    x = Dropout(0.3)(x)
    x = Dense(100, activation='relu')(x)

    # scalars
    scalars_in = Input(shape=(scalar_features,))
    y = Dense(30, activation='relu')(scalars_in)
    y = Dropout(0.1)(y)
    y = Dense(30, activation='relu')(y)

    # cross-attention
    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)
    xy = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(x, y, y)
    yx = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(y, x, x)
    xy = Flatten()(xy)
    yx = Flatten()(yx)
    z = Concatenate()([xy, yx])
    
    z = Dense(100, activation='relu')(z)
    output = Dense(1)(z)

    # Create model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def v3(steps=2000, features=11, scalar_features=18):
    """
    This model performed with...
    sbatch model/run_model.sh --use_full_data --name "REGRESSION_V3" --model_version "v3" --epochs 30 --task 0 --modalities "all" --output "/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY19_RUN"
    Args:
        steps (int, optional): _description_. Defaults to 2000.
        features (int, optional): _description_. Defaults to 11.
        scalar_features (int, optional): _description_. Defaults to 18.

    Returns:
        _type_: _description_
    """
    # images
    sequential_input = Input(shape=(steps, features))

    # calculate positional encoding 
    pos_enc = np.zeros((steps, features))
    for pos in range(steps):
        for i in range(features):
            if i%2==0:
                pos_enc[pos, i]=np.sin(pos/(10000**(i/features)))
            else:
                pos_enc[pos, i]=np.cos(pos/(10000**((i-1)/features)))
    pos_enc_reshaped = pos_enc[np.newaxis, :, :]
    pos_enc = tf.constant(pos_enc_reshaped)
    x = Add()([pos_enc, sequential_input])
    x = Attention()([x, x])
    x = Reshape((steps, features, 1))(x)
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    # small-kernel convolution
    x1 = TimeDistributed(Conv1D(filters=4, kernel_size=3, padding='valid'))(x)
    x1 = Lambda(lambda t: tf.reduce_sum(t, axis=-1))(x1)
    x1 = BatchNormalization()(x1)
    x1 = tf.transpose(x1, perm=[0, 2, 1])
    x1 = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    
    # large-kernel convolution
    x2 = TimeDistributed(Conv1D(filters=4, kernel_size=166, padding='valid'))(x)
    x2 = Lambda(lambda t: tf.reduce_sum(t, axis=-1))(x2)
    x2 = BatchNormalization()(x2)
    
    # flatten small and large kernel convolutions
    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    
    # concatenation and joint representation projection
    x = Concatenate()([x1, x2])
    x = Dropout(0.3)(x)
    x = Dense(100, activation='relu')(x)

    # scalars
    scalars_in = Input(shape=(scalar_features,))
    y = Dense(30, activation='relu')(scalars_in)
    y = Dropout(0.1)(y)
    y = Dense(30, activation='relu')(y)

    # cross-attention
    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)
    xy = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(x, y, y)
    yx = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(y, x, x)
    xy = Flatten()(xy)
    yx = Flatten()(yx)
    z = Concatenate()([xy, yx])
    
    z = Dense(100, activation='relu')(z)
    output = Dense(1)(z)

    # Create model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def v4(steps=2000, features=11, scalar_features=18):
    """
    This model performed with...
    sbatch model/run_model.sh --use_full_data --name "REGRESSION_V4" --model_version "v4" --epochs 30 --task 0 --modalities "all" --output "/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY20_RUN"
    Args:
        steps (int, optional): _description_. Defaults to 2000.
        features (int, optional): _description_. Defaults to 11.
        scalar_features (int, optional): _description_. Defaults to 18.

    Returns:
        _type_: _description_
    """
    # images
    sequential_input = Input(shape=(steps, features))

    # calculate positional encoding 
    pos_enc = np.zeros((steps, features))
    for pos in range(steps):
        for i in range(features):
            if i%2==0:
                pos_enc[pos, i]=np.sin(pos/(10000**(i/features)))
            else:
                pos_enc[pos, i]=np.cos(pos/(10000**((i-1)/features)))
    pos_enc_reshaped = pos_enc[np.newaxis, :, :]
    pos_enc = tf.constant(pos_enc_reshaped)
    x = Add()([pos_enc, sequential_input])
    x = Reshape((steps, features, 1))(x)
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    # small-kernel convolution
    x1 = TimeDistributed(Conv1D(filters=4, kernel_size=3, padding='valid'))(x)
    x1 = Lambda(lambda t: tf.reduce_sum(t, axis=-1))(x1)
    x1 = BatchNormalization()(x1)
    x1 = tf.transpose(x1, perm=[0, 2, 1])
    x1 = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    
    # large-kernel convolution
    x2 = TimeDistributed(Conv1D(filters=4, kernel_size=166, padding='valid'))(x)
    x2 = Lambda(lambda t: tf.reduce_sum(t, axis=-1))(x2)
    x2 = BatchNormalization()(x2)
    x2 = tf.transpose(x2, perm=[0, 2, 1])
    x2 = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)
    
    # flatten small and large kernel convolutions
    x1 = Flatten()(x1)
    x2 = Flatten()(x2)
    
    # concatenation and joint representation projection
    x = Concatenate()([x1, x2])
    x = Dropout(0.3)(x)
    x = Dense(1000, activation='relu')(x)

    # scalars
    scalars_in = Input(shape=(scalar_features,))
    y = Dense(50, activation='relu')(scalars_in)
    y = Dropout(0.1)(y)
    y = Dense(50, activation='relu')(y)

    # cross-attention
    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)
    xy = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(x, y, y)
    yx = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(y, x, x)
    xy = Flatten()(xy)
    yx = Flatten()(yx)
    z = Concatenate()([xy, yx])
    
    z = Dense(100, activation='relu')(z)
    output = Dense(1)(z)

    # Create model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def v5(steps=2000, features=11, scalar_features=18):
    """
    This model performed with ...
    sbatch model/run_model.sh --use_full_data --name "REGRESSION_V5" --model_version "v5" --epochs 30 --task 0 --modalities "all" --output "/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY20_RUN"
    Args:
        steps (int, optional): _description_. Defaults to 2000.
        features (int, optional): _description_. Defaults to 11.
        scalar_features (int, optional): _description_. Defaults to 18.

    Returns:
        _type_: _description_
    """
    sequential_input = Input(shape=(steps, features))
    # calculate positional encoding 
    pos_enc = np.zeros((steps, features))
    for pos in range(steps):
        for i in range(features):
            if i%2==0:
                pos_enc[pos, i]=np.sin(pos/(10000**(i/features)))
            else:
                pos_enc[pos, i]=np.cos(pos/(10000**((i-1)/features)))
    pos_enc_reshaped = pos_enc[np.newaxis, :, :]
    pos_enc = tf.constant(pos_enc_reshaped)
    x = Add()([pos_enc, sequential_input])
    x = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(200)(x)
    scalars_in = Input(shape=(scalar_features,))
    y = Dense(50, activation='relu')(scalars_in)
    y = Dropout(0.1)(y)
    y = Dense(50, activation='relu')(y)

    x = tf.expand_dims(x, -1)
    y = tf.expand_dims(y, -1)
    xy = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(x, y, y)
    yx = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(y, x, x)
    xy = Flatten()(xy)
    yx = Flatten()(yx)
    z = Concatenate()([xy, yx])
    
    z = Dense(100, activation='relu')(z)
    output = Dense(1)(z)

    # Create model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def v6(steps=2000, features=11, scalar_features=18):
    """
    This model performed with ...
    sbatch model/run_model.sh --use_full_data --name "REGRESSION_V6" --model_version "v6" --epochs 30 --task 0 --modalities "all" --output "/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/JULY20_RUN"
    Args:
        steps (int, optional): _description_. Defaults to 2000.
        features (int, optional): _description_. Defaults to 11.
        scalar_features (int, optional): _description_. Defaults to 18.

    Returns:
        _type_: _description_
    """
    sequential_input = Input(shape=(steps, features))
    # calculate positional encoding 
    pos_enc = np.zeros((steps, features))
    for pos in range(steps):
        for i in range(features):
            if i%2==0:
                pos_enc[pos, i]=np.sin(pos/(10000**(i/features)))
            else:
                pos_enc[pos, i]=np.cos(pos/(10000**((i-1)/features)))
    pos_enc_reshaped = pos_enc[np.newaxis, :, :]
    pos_enc = tf.constant(pos_enc_reshaped)
    x = Add()([pos_enc, sequential_input])
    x = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(200)(x)
    scalars_in = Input(shape=(scalar_features,))
    y = Dense(50, activation='relu')(scalars_in)
    y = Dropout(0.1)(y)
    y = Dense(50, activation='relu')(y)

    # x = tf.expand_dims(x, -1)
    # y = tf.expand_dims(y, -1)
    # xy = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(x, y, y)
    # yx = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(y, x, x)
    # xy = Flatten()(xy)
    # yx = Flatten()(yx)
    z = Concatenate()([x, y])
    
    z = Dense(100, activation='relu')(z)
    output = Dense(1)(z)

    # Create model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def v9(steps=2000, features=11, scalar_features=18):
    """
    sbatch model/run_model.sh --use_full_data --name "REGRESSION_V9" --model_version "v9" --epochs 10 --task 0 --modalities "images" --learning_rate 0.0001 --output "/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/model/_models/NEW_REGRESSION"

    Args:
        steps (int, optional): _description_. Defaults to 2000.
        features (int, optional): _description_. Defaults to 11.
        scalar_features (int, optional): _description_. Defaults to 18.

    Returns:
        _type_: _description_
    """
    sequential_input = Input(shape=(steps, features))
    x = Reshape((steps, features, 1))(sequential_input)
    bp_model = Sequential([
        Flatten(),
        BatchNormalization(),
        Dense(3)
    ])
    x = TimeDistributed(bp_model)(x)
    
    x = BatchNormalization()(x)
    x = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # for _ in range(n_convolutions-1):
    #     x = Conv1D(filters=n_filters, kernel_size=3, activation='relu', input_shape=(steps, features),
    #            kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    #     x = BatchNormalization()(x)
    #     if use_avg_pool:
    #         x = AveragePooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(1)(x)

    # Create model
    model = Model(inputs=sequential_input, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(loss=MeanSquaredError(),
                  optimizer=optimizer, 
                  metrics=['mse'])
    return model

def main():
    pass

if __name__ == "__main__":
    main()