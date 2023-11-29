# anetzley
# 11/29/2023


import os
import csv
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool, cpu_count


from preprocessing.compile_sample_data import *
from preprocessing.generate_tf_splits import *

full_trainset = get_dataset(fold_numbers = [1, 2, 3, 4], use_set = 'GeneStrat_CompleteInputs_11-21-2023', shuffle=True, batch_size=32)
full_testset = get_dataset(fold_numbers = [5], use_set = 'GeneStrat_CompleteInputs_11-21-2023', shuffle=True, batch_size=32)
print('Completed Extracting Full Sets')

#Check for GPU availability
from tensorflow.python.client import device_lib

# Get list of devices
local_device_protos = device_lib.list_local_devices()

# Check if GPU is available
gpu_available = any(device.device_type == 'GPU' for device in local_device_protos)
print("Is GPU available:", gpu_available)

# Extract only the features you need
def extract_features(batch):
    return {'input_1': batch['TSS_image']}, {'dense_2': batch['binary_label']}

trainset = full_trainset.map(extract_features)
testset = full_testset.map(extract_features)

#Print out the shapes of the datasets
print(trainset)
print(testset)

#from models.utils import *
#from models.classification_models import classification_only_images
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, Activation, \
    GlobalAveragePooling1D, Reshape, Concatenate, Attention, TimeDistributed, Lambda, MultiHeadAttention, Add
from keras.regularizers import l1, l2, l1_l2
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, AveragePooling1D

def classification_only_images(input_dim=(2000, 11), output_dim=1, learning_rate=1e-3, n_filters=64, n_convolutions=1):
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
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    return model

#logging.info(f"Creating a classification model to predict {args.task} classes with {args.modalities}.")

# get model
#model = classification_models.classification_only_images()
model = classification_only_images()
#Print the model summary]
model.summary()

# #utils.plot_model(model, OUTPUT)

# # convert labels to one-hot encodings for classifications
# labels_train_ohe = tf.one_hot(tf.convert_to_tensor(filtered_training["labels"], dtype=tf.int8), depth=args.task)
# labels_val_ohe = tf.one_hot(tf.convert_to_tensor(filtered_validation["labels"], dtype=tf.int8), depth=args.task)

# # model training
# logging.info(f"Training on {len(labels_train_ohe)}, validating on {len(labels_val_ohe)}")
# logging.info(f"Training model for {args.epochs} epochs...")
history = model.fit(trainset, 
                    epochs=5, batch_size=32, verbose=1,
                    shuffle=True, validation_data=testset)
                    # utils.plot_convergence(history, OUTPUT)
