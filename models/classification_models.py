#!/usr/bin/env python
'''
@File    :   classification_models.py
@Author  :   Akira Nair
@Contact :   akira_nair@brown.edu / anair3@fredhutch.org
@Desc    :   Classification models (for predicting gene activity status)
'''

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Activation, \
    GlobalAveragePooling1D, Reshape, Concatenate, Attention, TimeDistributed, Lambda, MultiHeadAttention, Add
from keras.regularizers import l1, l2, l1_l2
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, AveragePooling1D
import numpy as np
import utils

def get_model_type(args):
    if args.model_version is not None:
        models = {
            "v1" : v1,
            "comparison_images" : comparison_images,
            "comparison_scalars" : comparison_scalars,
            "comparison_both" : comparison_both
        }
        vmodel = models[args.model_version]
        return vmodel()
    if args.modalities == "images":
        if args.use_attention:
            return classification_only_images_attention(input_dim=args.i_shape, 
                                                        output_dim=args.task, 
                                                        learning_rate=args.learning_rate, )
        else:
            return classification_only_images(input_dim=args.i_shape,
                                              output_dim=args.task,
                                            learning_rate=args.learning_rate)
    elif args.modalities == "scalars":
        if args.use_attention:
            return classification_only_scalars_attention(learning_rate=args.learning_rate,
                                                          output_dim=args.task, 
                                                          n_features=args.s_features)
        else:
            return classification_only_scalars(input_dim=args.s_shape,
                                            output_dim=args.task, 
                                            learning_rate=args.learning_rate)
    else:
        if args.use_attention:
            return classification_all_attention(input_dim_i=args.i_shape,
                                                input_dim_s=args.s_shape,
                                                output_dim=args.task, 
                                                learning_rate=args.learning_rate)
        else:
            return classification_all(input_dim_i=args.i_shape,
                                      input_dim_s=args.s_shape,
                                            output_dim=args.task, 
                                            learning_rate=args.learning_rate)


def classification_only_images(input_dim=(2000, 11), output_dim=3, learning_rate=1e-3, n_filters=64, n_convolutions=1):
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
    output = Dense(output_dim, activation='softmax')(x)

    # Create the model
    model = Model(inputs=sequential_input, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    return model

def classification_only_scalars(input_dim=14, output_dim=3, learning_rate=1e-3, n_hidden=1):
    scalars_in = Input(shape=(input_dim,))
    x = Dense(50, activation='relu')(scalars_in)
    x = BatchNormalization()(x)
    for _ in range(n_hidden-1):
        x = Dense(50, activation='relu')(x)
        x = BatchNormalization()(x)
    output = Dense(output_dim, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=scalars_in, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    return model

def classification_all(input_dim_i=(2000, 11), input_dim_s=14, output_dim=3, learning_rate=1e-3, n_filters=64, n_convolutions=1, n_hidden=1):
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
    output = Dense(output_dim, activation='softmax')(z)
    # Create the model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    return model

def classification_only_images_attention(input_dim=(2000, 11), output_dim=3, learning_rate=1e-3):
    sequential = Input(shape=input_dim)
    x = Attention()([sequential, sequential])
    x = BatchNormalization()(x)
    x = Conv1D(filters=1, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(output_dim, activation='softmax')(x)
    # Create model
    model = Model(inputs=sequential, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    return model

def classification_only_scalars_attention(input_dim=14, output_dim=3, learning_rate=1e-3):
    scalars_in = Input(shape=(input_dim,))
    x = Attention()([scalars_in, scalars_in])
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(output_dim, activation='softmax')(x)
    # Create the model
    model = Model(inputs=scalars_in, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    return model

def classification_all_attention(input_dim_i=(2000, 11),input_dim_s=14,output_dim=3, learning_rate=1e-3):
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
    output = Dense(output_dim, activation='softmax')(z)
    # Create the model
    model = Model(inputs=[sequential, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    return model

### SPECIFIC ARCHITECTURE VERSIONS ###

def v1():
    steps=2000
    features=11
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
    scalars_in = Input(shape=(14,))
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
    output = Dense(3, activation='softmax')(z)

    # Create model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['categorical_accuracy'])
    return model

def comparison_images():
    sequential_input = Input(shape=(2000, 11))

    x = Reshape((2000, 11, 1))(sequential_input)
    bp_model = Sequential([
        Flatten(),
        BatchNormalization(),
        Dense(3)
    ])
    x = TimeDistributed(bp_model)(x)
    
    x = BatchNormalization()(x)
    x = Conv1D(filters=2, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)        
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)

    # Create the model
    model = Model(inputs=sequential_input, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, 
                metrics=['categorical_accuracy'])
    return model

def comparison_scalars():
    scalars_in = Input(shape=(14,))
    x = Dense(50, activation='relu')(scalars_in)
    x = BatchNormalization()(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    output = Dense(2, activation='softmax')(x)
    # Create the model
    model = Model(inputs=scalars_in, outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, 
                metrics=['categorical_accuracy'])
    return model

def comparison_both():
    sequential_input = Input(shape=(2000, 11))

    x = Reshape((2000, 11, 1))(sequential_input)
    bp_model = Sequential([
        Flatten(),
        BatchNormalization(),
        Dense(3)
    ])
    x = TimeDistributed(bp_model)(x)
    
    x = BatchNormalization()(x)
    x = Conv1D(filters=2, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.02, l2=0.02))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)        
    x = Flatten()(x)
    x = BatchNormalization()(x)

    scalars_in = Input(shape=(14,))
    y = Dense(50, activation='relu')(scalars_in)
    y = BatchNormalization()(y)
    y = Dense(50, activation='relu')(y)
    y = BatchNormalization()(y)


    z = Concatenate()([x, y])
    z = Dense(100, activation='relu')(z)
    output = Dense(2, activation='softmax')(z)

    # Create the model
    model = Model(inputs=[sequential_input, scalars_in], outputs=output)
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, 
                metrics=['categorical_accuracy'])
    return model


def main():
    pass

if __name__ == "__main__":
    main()