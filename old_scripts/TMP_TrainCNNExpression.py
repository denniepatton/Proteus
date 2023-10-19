#! /usr/bin/env python3
# ml TensorFlow/2.4.1-fosscuda-2020b
# sbatch -o out.txt --gres=gpu CNN_prof.py

import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import regularizers
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, LSTM, BatchNormalization, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, r2_score
from keras.layers.merge import concatenate


# fit and evaluate the model
def evaluate_model(train_data, train_labels, test_data, test_labels, params):
    ks1, ks2, ks3, kn1, kn2, kn3, mp1, mp2, mp3, drop, lstm, dense1, dense2, batch = [var for var in params]
    verbose, epochs, batch_size = 0, 3, batch
    steps, features, outputs = train_data.shape[1], train_data.shape[2], 1
    steps, features = train_data.shape[1], train_data.shape[2]
    model = Sequential()
    model.add(Conv1D(filters=kn1, kernel_size=ks1, activation='relu', input_shape=(steps, features)))
    if not mp1 == 0:
        model.add(MaxPooling1D(pool_size=mp1))
    if not ks2 == 0:
        model.add(Conv1D(filters=kn2, kernel_size=ks2, activation='relu'))
    if not mp2 == 0:
        model.add(MaxPooling1D(pool_size=mp2))
    if not ks3 == 0:
        model.add(Conv1D(filters=kn3, kernel_size=ks3, activation='relu'))
    if not mp3 == 0:
        model.add(MaxPooling1D(pool_size=mp3))
    if not drop == 0:
        model.add(Dropout(drop))
    if not lstm == 0:
        model.add(LSTM(lstm))
    model.add(Flatten())
    model.add(Dense(dense1, activation='relu'))
    if not dense2 == 0:
        model.add(Dense(dense2, activation='relu'))
    # model.add(Dense(outputs, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.add(Dense(outputs, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    # fit network
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=0)
    # loss = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=0)
    # multi headed approach below (not up-to-date)#
    ####################################################################################################################
    # head 1
    inputs1 = Input(shape=(steps, features))
    conv1 = Conv1D(filters=kn1, kernel_size=ks1, activation='relu')(inputs1)
    pool1 = MaxPooling1D(pool_size=mp1)(conv1)
    flat1 = Flatten()(pool1)
    # head 2
    inputs2 = Input(shape=(steps, features))
    conv2 = Conv1D(filters=kn2, kernel_size=ks2, activation='relu')(inputs2)
    pool2 = MaxPooling1D(pool_size=mp2)(conv2)
    flat2 = Flatten()(pool2)
    # head 3
    inputs3 = Input(shape=(steps, features))
    conv3 = Conv1D(filters=kn3, kernel_size=ks3, activation='relu')(inputs3)
    pool3 = MaxPooling1D(pool_size=mp3)(conv3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense_layer = Dense(dense, activation='relu')(merged)
    soft = Dense(outputs, activation='softmax')(dense_layer)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=soft)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit([train_data, train_data, train_data], train_labels,
              epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate([test_data, test_data, test_data], test_labels, batch_size=batch_size, verbose=0)
    ####################################################################################################################
    return accuracy
    # return loss


# run an experiment
def run_experiment(images, labels, params, do_skf):
    # repeat experiment
    scores = list()
    if do_skf:
        skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
        # do k-fold cross validation (stratified)
        for train_index, test_index in skf.split(images, labels):
            images_train = images[train_index]
            labels_train = labels[train_index]
            images_test = images[test_index]
            labels_test = labels[test_index]
            # evaluate model
            score = evaluate_model(images_train, labels_train, images_test, labels_test, params)
            score = score * 100.0
            scores.append(score)
    else:
        kf = KFold(n_splits=3, random_state=42, shuffle=True)
        # do k-fold cross validation (stratified)
        for train_index, test_index in kf.split(images, labels):
            images_train = images[train_index]
            labels_train = labels[train_index]
            images_test = images[test_index]
            labels_test = labels[test_index]
            # evaluate model
            score = evaluate_model(images_train, labels_train, images_test, labels_test, params)
            scores.append(score)
    m, s = np.mean(scores), np.std(scores)
    ks1, ks2, ks3, kn1, kn2, kn3, mp1, mp2, mp3, drop, lstm, dense1, dense2, batch = [var for var in params]
    print('> ks1=%d ks2=%d ks3=%d kn1=%d kn2=%d kn3=%d mp1=%d mp2=%d mp3=%d '
          'drop=%d lstm=%d dense1=%d dense2=%d batch=%d : %.3f%% (+/-%.3f)' %
          (ks1, ks2, ks3, kn1, kn2, kn3, mp1, mp2, mp3, drop, lstm, dense1, dense2, batch, m, s))
    return m


# subset data to speed up training
def subset(images, labels, count):
    indices = list(range(images.shape[0]))
    random.shuffle(indices)
    return images[indices[:count]], labels[indices[:count]]


mutate = [True, False]
step = [-1, 1]


def mutate_val(value, pool):
    if random.choice(mutate):
        index = pool.index(value)
        if index == 0:
            return pool[index + 1]
        elif index == len(pool) - 1:
            return pool[index - 1]
        else:
            return pool[index + random.choice(step)]
    else:
        return value


def mutate_params(params):
    ks1, ks2, ks3, kn1, kn2, kn3, mp1, mp2, mp3, drop, lstm, dense1, dense2, batch = [var for var in params]
    kernel_size_pool = [0, 2, 3, 4, 5, 8, 12, 16, 32, 148]
    ks1_pool = [2, 3, 4, 5, 8, 12, 16, 32, 148]
    kernel_num_pool = [16, 32, 64]
    max_pool_pool = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    drop_pool = [0, 0.1, 0.25, 0.5]
    lstm_pool = [0, 128, 512, 1024]
    dense_pool = [0, 100, 200, 300]
    dense1_pool = [10, 50, 100, 200, 300, 500, 1000]
    batch_pool = [16, 32, 64, 128]
    new_ks1 = mutate_val(ks1, ks1_pool)
    new_ks2 = mutate_val(ks2, kernel_size_pool)
    new_ks3 = mutate_val(ks3, kernel_size_pool)
    new_kn1 = mutate_val(kn1, kernel_num_pool)
    new_kn2 = mutate_val(kn2, kernel_num_pool)
    new_kn3 = mutate_val(kn3, kernel_num_pool)
    new_mp1 = mutate_val(mp1, max_pool_pool)
    new_mp2 = mutate_val(mp2, max_pool_pool)
    new_mp3 = mutate_val(mp3, max_pool_pool)
    new_drop = mutate_val(drop, drop_pool)
    new_lstm = mutate_val(lstm, lstm_pool)
    new_dense1 = mutate_val(dense1, dense1_pool)
    new_dense2 = mutate_val(dense2, dense_pool)
    new_batch = mutate_val(batch, batch_pool)
    return [new_ks1, new_ks2, new_ks3, new_kn1, new_kn2, new_kn3, new_mp1, new_mp2, new_mp3, new_drop, new_lstm,
            new_dense1, new_dense2, new_batch]


def evolve_model(images, labels, seeds, generations):
    kernel_size_pool = [0, 2, 3, 4, 8, 16, 32, 148]
    ks1_pool = [2, 3, 6]
    kernel_num_pool = [16, 32, 64]
    max_pool_pool = [0, 2, 3, 4]
    drop_pool = [0, 0.5]
    lstm_pool = [0, 128, 512, 1024]
    dense_pool = [0, 100, 200, 300]
    dense1_pool = [100, 200, 300]
    batch_pool = [32]
    best_score = 0  # 0 for accuracy, 100 for loss
    best_params = []
    print('Beginning seeding process: ' + str(seeds) + ' seeds\n')
    if seeds > 0:
        for _ in range(seeds):
            ks1 = random.choice(ks1_pool)
            ks2 = random.choice(kernel_size_pool)
            ks3 = random.choice(kernel_size_pool)
            kn1 = random.choice(kernel_num_pool)
            kn2 = random.choice(kernel_num_pool)
            kn3 = random.choice(kernel_num_pool)
            mp1 = random.choice(max_pool_pool)
            mp2 = random.choice(max_pool_pool)
            mp3 = random.choice(max_pool_pool)
            drop = random.choice(drop_pool)
            lstm = random.choice(lstm_pool)
            dense1 = random.choice(dense1_pool)
            dense2 = random.choice(dense_pool)
            batch = random.choice(batch_pool)
            params = [ks1, ks2, ks3, kn1, kn2, kn3, mp1, mp2, mp3, drop, lstm, dense1, dense2, batch]
            score = run_experiment(images, labels, params, False)
            if score < best_score:
                best_score = score
                best_params = params
    else:
        ks1 = 5
        ks2 = 3
        ks3 = 8
        kn1 = 64
        kn2 = 32
        kn3 = 32
        mp1 = 3
        mp2 = 2
        mp3 = 0
        drop = 0
        lstm = 128
        dense1 = 1000
        dense2 = 0
        batch = 32
        params = [ks1, ks2, ks3, kn1, kn2, kn3, mp1, mp2, mp3, drop, lstm, dense1, dense2, batch]
        score = run_experiment(images, labels, params, False)
        if score < best_score:
            best_score = score
            best_params = params
    print('Beginning mutations: ' + str(generations) + ' generations\n')
    for _ in range(generations):
        params = mutate_params(best_params)
        score = run_experiment(images, labels, params, False)
        if score < best_score:
            best_score = score
            best_params = params
    return best_score, best_params


def calibration_plot(true_labels, pred_labels, title, ece):
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i in range(true_labels.shape[1]):
        name = str(i)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            [item[i] for item in true_labels], [item[i] for item in pred_labels], n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,))
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title('Calibration plot (reliability curve)')
    plt.text(0.1, 0.9, 'ECE: ' + ece)
    plt.tight_layout()
    plt.savefig(title)
    plt.close()


def roc_plot(predictions, truths, title):
    fpr, tpr, _ = roc_curve(truths, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='indigo', lw=2, label='ROC curve (AUC = {0:0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.savefig(title)
    plt.close()


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


def sa_fit(input_array):
    steps, features = input_array.shape[1], input_array.shape[2]
    input_layer = keras.layers.Input(shape=(steps, features))
    x = keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = keras.layers.MaxPooling1D(2, padding='same')(x)
    x = keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling1D(2, padding='same')(x)
    x = keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x)
    encoded = keras.layers.MaxPooling1D(2, padding='same')(x)
    # at this point the representation is (17, 8)
    sparse = keras.layers.Dense(8, activation='relu', activity_regularizer=regularizers.l1(10e-5), name='sparse')(encoded)
    x = keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(sparse)
    x = keras.layers.UpSampling1D(2)(x)
    x = keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.UpSampling1D(2)(x)
    x = keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(x)
    x = keras.layers.UpSampling1D(2)(x)
    decoded = keras.layers.Conv1D(filters=1, kernel_size=3, activation='linear', padding='same')(x)
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(input_array, input_array, epochs=10, batch_size=128, shuffle=True,
                    validation_data=(input_array, input_array))
    # decoded_array = autoencoder.predict(input_array)
    # return decoded_array
    intermediate = keras.Model(autoencoder.input, autoencoder.get_layer('sparse').output)
    intermediate_output = intermediate.predict(input_array)
    return intermediate_output


# resnet model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
def c1d_resnet_model(steps, features, output, steps_2=None, features_2=None):
    n_feature_maps = 32
    input_layer = keras.layers.Input(shape=(steps, features))
    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    # BLOCK 2
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    # BLOCK 3
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    gap_layer = keras.layers.Flatten()(gap_layer)
    if steps_2 is None and features_2 is None:  # not doing a multi-head
        if output == 'Activity':  # binary classification
            output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)
            model = keras.models.Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif output == 'FPKM':  # regression
            output_layer = keras.layers.Dense(1, activation='linear')(gap_layer)
            model = keras.models.Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        else:  # multi-class
            output_layer = keras.layers.Dense(output, activation='softmax')(gap_layer)
            model = keras.models.Model(inputs=input_layer, outputs=output_layer)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:  # build second head
        input_layer_2 = keras.layers.Input(shape=(steps_2, features_2))
        # BLOCK 1
        conv_x_2 = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer_2)
        conv_x_2 = keras.layers.BatchNormalization()(conv_x_2)
        conv_x_2 = keras.layers.Activation('relu')(conv_x_2)
        conv_y_2 = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x_2)
        conv_y_2 = keras.layers.BatchNormalization()(conv_y_2)
        conv_y_2 = keras.layers.Activation('relu')(conv_y_2)
        conv_z_2 = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y_2)
        conv_z_2 = keras.layers.BatchNormalization()(conv_z_2)
        # expand channels for the sum
        shortcut_y_2 = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer_2)
        shortcut_y_2 = keras.layers.BatchNormalization()(shortcut_y_2)
        output_block_1_2 = keras.layers.add([shortcut_y_2, conv_z_2])
        output_block_1_2 = keras.layers.Activation('relu')(output_block_1_2)
        # BLOCK 2
        conv_x_2 = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1_2)
        conv_x_2 = keras.layers.BatchNormalization()(conv_x_2)
        conv_x_2 = keras.layers.Activation('relu')(conv_x_2)
        conv_y_2 = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x_2)
        conv_y_2 = keras.layers.BatchNormalization()(conv_y_2)
        conv_y_2 = keras.layers.Activation('relu')(conv_y_2)
        conv_z_2 = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y_2)
        conv_z_2 = keras.layers.BatchNormalization()(conv_z_2)
        # expand channels for the sum
        shortcut_y_2 = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1_2)
        shortcut_y_2 = keras.layers.BatchNormalization()(shortcut_y_2)
        output_block_2_2 = keras.layers.add([shortcut_y_2, conv_z_2])
        output_block_2_2 = keras.layers.Activation('relu')(output_block_2_2)
        # BLOCK 3
        conv_x_2 = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2_2)
        conv_x_2 = keras.layers.BatchNormalization()(conv_x_2)
        conv_x_2 = keras.layers.Activation('relu')(conv_x_2)
        conv_y_2 = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x_2)
        conv_y_2 = keras.layers.BatchNormalization()(conv_y_2)
        conv_y_2 = keras.layers.Activation('relu')(conv_y_2)
        conv_z_2 = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y_2)
        conv_z_2 = keras.layers.BatchNormalization()(conv_z_2)
        # no need to expand channels because they are equal
        shortcut_y_2 = keras.layers.BatchNormalization()(output_block_2_2)
        output_block_3_2 = keras.layers.add([shortcut_y_2, conv_z_2])
        output_block_3_2 = keras.layers.Activation('relu')(output_block_3_2)
        # FINAL
        gap_layer_2 = keras.layers.GlobalAveragePooling1D()(output_block_3_2)
        gap_layer_2 = keras.layers.Flatten()(gap_layer_2)
        merged = concatenate([gap_layer, gap_layer_2], axis=1)
        if output == 'Activity':  # binary classification
            output_layer = keras.layers.Dense(1, activation='sigmoid')(merged)
            model = keras.models.Model(inputs=[input_layer, input_layer_2], outputs=output_layer)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif output == 'FPKM':  # regression
            output_layer = keras.layers.Dense(1, activation='linear')(merged)
            model = keras.models.Model(inputs=[input_layer, input_layer_2], outputs=output_layer)
            model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        else:  # multi-class
            output_layer = keras.layers.Dense(output, activation='softmax')(merged)
            model = keras.models.Model(inputs=[input_layer, input_layer_2], outputs=output_layer)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# replicate the favored architecture and hyper parameters below before doing final training
def finalized_model_activity(steps, features, outputs):
    # loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)  # loss=loss_fn
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(steps, features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# replicate the favored architecture and hyper parameters below before doing final training
def finalized_model_fpkm(steps, features, outputs):
    # loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)  # loss=loss_fn
    model = Sequential()
    # model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(steps, features)))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    # model.add(MaxPooling1D(pool_size=3))
    model.add(LSTM(128))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(outputs, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model


# will print model summary, accuracy/loss vs epochs, calibration curves,  and save the completed file to a directory
def save_model_activity(images, labels, name, images_2=None):
    if not os.path.exists(name):
        os.makedirs(name)
    # initial training with validation hold-out
    verbose, epochs, batch_size = 1, 10, 32
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold = 1
    predictions = []
    ground_truth = []
    for train_index, test_index in skf.split(images, labels):
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
        file_path = name + '/best_model.hdf5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
        print('#################### FOLD ' + str(fold) + ' ####################')
        images_train = images[train_index]
        labels_train = to_categorical(labels[train_index])  ###########################################################
        images_test = images[test_index]
        labels_test = to_categorical(labels[test_index])  #############################################################
        # steps, features, outputs = images_train.shape[1], images_train.shape[2], 1
        steps, features = images_train.shape[1], images_train.shape[2]
        if images_2 is not None:  # two headed model
            images_train_2 = images_2[train_index]
            images_test_2 = images_2[test_index]
            steps_2, features_2 = images_train_2.shape[1], images_train_2.shape[2]
            print('#################### TRAINING ####################')
            model = c1d_resnet_model(steps, features, 'Activity', steps_2, features_2)
            history = model.fit([images_train, images_train_2], labels_train, epochs=epochs, batch_size=batch_size,
                                verbose=verbose, shuffle=True,
                                validation_data=([images_test, images_test_2], labels_test),
                                callbacks=[reduce_lr, model_checkpoint])
            print('#################### TESTING ####################')
            model.load_weights(file_path)
            predictions.append(model.predict([images_test, images_test_2]))
        else:
            print('#################### TRAINING ####################')
            # model = finalized_model_activity(steps, features, outputs)
            model = c1d_resnet_model(steps, features, labels_train.shape[1])
            # model = c1d_resnet_model(steps, features, 'Activity')
            history = model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                                shuffle=True, validation_data=(images_test, labels_test),
                                callbacks=[reduce_lr, model_checkpoint])
            print('#################### TESTING ####################')
            model.load_weights(file_path)
            predictions.append(model.predict(images_test))
        ground_truth.append(labels_test)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(name + '/training_accuracy_fold' + str(fold) + '.pdf')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(name + '/training_loss_fold' + str(fold) + '.pdf')
        plt.close()
        fold += 1
    # predictions = np.concatenate(predictions)
    # labels_test = np.concatenate(ground_truth)
    # print ROC and AUC from the K-fold validation
    # roc_plot(predictions, labels_test, name + '/ROC.pdf')
    # do final training using all data
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    file_path = name + '/best_model.hdf5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
    steps, features = images.shape[1], images.shape[2]
    labels = to_categorical(labels)
    model = c1d_resnet_model(steps, features, labels.shape[1])
    model.fit(images, labels, epochs=20, batch_size=batch_size, verbose=verbose, shuffle=True,
              callbacks=[reduce_lr, model_checkpoint])
    model.load_weights(file_path)
    model.save(name)
    # print(model.summary())


# this save model does regression based analysis and uses 20% hold-out
def save_model_fpkm(images, labels, name, images_2=None):
    if not os.path.exists(name):
        os.makedirs(name)
    # initial training with validation hold-out
    verbose, epochs, batch_size = 1, 20, 128
    index_80 = round(0.8 * len(labels))
    images_train = images[:index_80]
    labels_train = labels[:index_80]
    images_test = images[(index_80 + 1):]
    labels_test = labels[(index_80 + 1):]
    steps, features = images_train.shape[1], images_train.shape[2]
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    file_path = name + '/best_model.hdf5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
    if images_2 is not None:  # two headed model
        images_train_2 = images_2[:index_80]
        images_test_2 = images_2[(index_80 + 1):]
        steps_2, features_2 = images_train_2.shape[1], images_train_2.shape[2]
        print('#################### TRAINING ####################')
        model = c1d_resnet_model(steps, features, 'FPKM', steps_2, features_2)
        history = model.fit([images_train, images_train_2], labels_train, epochs=epochs, batch_size=batch_size,
                            verbose=verbose, shuffle=True, validation_data=([images_test, images_test_2], labels_test),
                            callbacks=[reduce_lr, model_checkpoint])
        print('#################### TESTING ####################')
        model.load_weights(file_path)
        predictions = model.predict([images_test, images_test_2])
    else:
        print('#################### TRAINING ####################')
        model = c1d_resnet_model(steps, features, 'FPKM')
        history = model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                            shuffle=True, validation_data=(images_test, labels_test),
                            callbacks=[reduce_lr, model_checkpoint])
        print('#################### TESTING ####################')
        model.load_weights(file_path)
        predictions = model.predict(images_test)
    r2_score_model = r2_score(labels_test, predictions)
    print("R2 Score: ", r2_score_model)
    fig, ax = plt.subplots()
    ax.plot(labels_test, predictions, linewidth=0, marker=".", alpha=0.2)
    ax.plot([0, 1], [0, 1], ls="--", c=".3", alpha=0.5)
    ax.set_xlabel("True Normalized Log2 FPKM", fontweight="bold")
    ax.set_ylabel("Predicted Normalized Log2 FPKM", fontweight="bold")
    plt.text(0.1, 0.9, 'R2: ' + str(r2_score_model))
    plt.savefig(name + '/R2Plot.pdf')
    plt.close()
    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot mse during training
    plt.subplot(212)
    plt.title('Mean Squared Error')
    plt.plot(history.history['mse'], label='train')
    plt.plot(history.history['val_mse'], label='test')
    plt.legend()
    plt.savefig(name + '/LossMSEPlot.pdf')
    plt.close()
    # finish - re-train on all data
    steps, features = images.shape[1], images.shape[2]
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    file_path = name + '/best_model.hdf5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
    if images_2 is not None:  # two headed model
        steps_2, features_2 = images_2.shape[1], images_2.shape[2]
        print('#################### TRAINING ####################')
        model = c1d_resnet_model(steps, features, 'FPKM', steps_2, features_2)
        model.fit([images, images_2], labels, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True,
                  callbacks=[reduce_lr, model_checkpoint])
    else:
        print('#################### TRAINING ####################')
        model = c1d_resnet_model(steps, features, 'FPKM')
        model.fit(images, labels, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True,
                  callbacks=[reduce_lr, model_checkpoint])
    model.load_weights(file_path)
    model.save(name)


# normalize expression values to be in the range [-1, 1]
def standardize_fpkm(labels):  # potentially use z-score in the future?
    log_cov = np.log(labels)
    return (log_cov - min(log_cov))/(max(log_cov) - min(log_cov))


def quick_test(images, labels, name, subs, frags=None):
    model = keras.models.load_model(name)
    if frags is not None:
        predictions = model.predict([images, frags])
    else:
        predictions = model.predict(images)
    r2_score_model = r2_score(labels, predictions)
    print("R2 Score: ", r2_score_model)
    fig, ax = plt.subplots()
    ax.plot(labels, predictions, linewidth=0, marker=".", alpha=0.2)
    ax.plot([0, 1], [0, 1], ls="--", c=".3", alpha=0.5)
    ax.set_xlabel("True Normalized Log2 FPKM", fontweight="bold")
    ax.set_ylabel("Predicted Normalized Log2 FPKM", fontweight="bold")
    plt.text(0.1, 0.9, 'R2: ' + str(r2_score_model))
    plt.savefig(name + '/' + subs + '_R2Plot.pdf')
    plt.close()


# def main():  # coverage main
#     # references . . .
#     hk_genes = pd.read_table('../references/HousekeepingGenes.txt', header=None)
#     pam50_genes = pd.read_table('../references/PC_gene_lists/PAM50.txt', header=None)
#     pcs37_genes = pd.read_table('../references/PC_gene_lists/PCS37.txt', header=None)
#     matched_ids = pd.read_table('CoverageData/EAP1_TSS-1EIJ_OrderedGeneNames.txt', header=None)
#     lo_genes = pd.concat([hk_genes, pam50_genes, pcs37_genes], ignore_index=True)
#     non_lo_indices = np.where(~matched_ids[0].str.contains('|'.join(list(lo_genes[0]))))[0]
#     pam50_indices = np.where(matched_ids[0].str.contains('|'.join(list(pam50_genes[0]))))[0]
#     pcs37_indices = np.where(matched_ids[0].str.contains('|'.join(list(pcs37_genes[0]))))[0]
#     # load data generated by GenerateProfileCovInputs.py
#     images_input = np.load('CoverageData/EAP1_TSS-1EIJ_1bp-Seq_EAP1_cov-profiles.npy')
#     frags_input = np.load('CoverageData/EAP1_TSS-1EIJ_1bp-Seq_EAP1_frag-profiles.npy')
#     labels_input = np.load('CoverageData/EAP1_TSS-1EIJ_1bp-Seq_EAP1_fpkm-labels.npy')
#     # labels_input = np.load('CoverageData/EAP1_TSS-1EIJ_1bp-Seq_EAP1_activity-labels.npy')
#     labels_input = standardize_fpkm(labels_input)
#     # images_input, labels_input = subset(images_input, labels_input, 50000)
#     images_pam50, labels_pam50, frags_pam50 =\
#         images_input[pam50_indices], labels_input[pam50_indices], frags_input[pam50_indices]
#     images_pcs37, labels_pcs37, frags_pcs37 =\
#         images_input[pcs37_indices], labels_input[pcs37_indices], frags_input[pcs37_indices]
#     images_input, labels_input, frags_input =\
#         images_input[non_lo_indices], labels_input[non_lo_indices], frags_input[non_lo_indices]
#     idx = np.random.permutation(len(images_input))
#     images_input, labels_input, frags_input = images_input[idx], labels_input[idx], frags_input[idx]
#     # images_input = sa_fit(images_input)
#     # save/evaluate model . . .
#     ####################################################################################################################
#     # save_model_activity(images_input, labels_input, 'Activity_model_EAP1_TSS-1EIJ_1H-1DRN')
#     # save_model_fpkm(images_input, labels_input, 'FPKM_model_EAP1_TSS-1EIJ_2H-1DRN', frags_input)
#     save_model_fpkm(images_input, labels_input, 'FPKM_model_EAP1_TSS-1EIJ_1H-1DRN')
#     quick_test(images_pam50, labels_pam50, 'FPKM_model_EAP1_TSS-1EIJ_1H-1DRN', 'PAM50')
#     quick_test(images_pcs37, labels_pcs37, 'FPKM_model_EAP1_TSS-1EIJ_1H-1DRN', 'PCS37')
#     # generate/mutate model . . .
#     ####################################################################################################################
#     # score, params = evolve_model(images_input, labels_input, 10, 1000)
#     # print(score)
#     # print(params)


def main():  # subtyping main
    # load data
    images_input = np.load('SubtypingData/profiles_TFBS_3S.npy')
    labels_input = np.load('SubtypingData/labels_TFBS_3S.npy')
    idx = np.random.permutation(len(images_input))
    images_input, labels_input = images_input[idx], labels_input[idx]
    # save/evaluate model . . .
    ####################################################################################################################
    save_model_activity(images_input, labels_input, 'Subtyping-3S_model_TFBS_1DRN')
    # generate/mutate model . . .
    ####################################################################################################################


if __name__ == "__main__":
    main()
