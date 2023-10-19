#! /usr/bin/env python3
# first load: ml Python/3.8.6-GCCcore-10.2.0, TensorFlow/2.4.1-fosscuda-2020b

# v2.0.0, 12/13/2022
# modified to take-in and use Triton output files

import random
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, LSTM, BatchNormalization, Activation, GlobalAveragePooling1D, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, AveragePooling1D
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
from keras.layers.merge import concatenate


# subset data to speed up training
def subset(images, labels, count):
    indices = list(range(images.shape[0]))
    random.shuffle(indices)
    return images[indices[:count]], labels[indices[:count]]


# plot a calibration curve
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


# plot ROC curve
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


# for use in creating inception architectures
def inception_module(x, filters_1, filters_3_reduce, filters_3, filters_5_reduce, filters_5, filters_pool_proj):
    conv_1 = Conv1D(filters_1, kernel_size=1, padding='same', activation='relu')(x)
    conv_3 = Conv1D(filters_3_reduce, kernel_size=1, padding='same', activation='relu')(x)
    conv_3 = Conv1D(filters_3, kernel_size=3, padding='same', activation='relu')(conv_3)
    conv_5 = Conv1D(filters_5_reduce, kernel_size=1, padding='same', activation='relu')(x)
    conv_5 = Conv1D(filters_5, kernel_size=5, padding='same', activation='relu')(conv_5)
    pool_proj = MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    pool_proj = Conv1D(filters_pool_proj, kernel_size=1, padding='same', activation='relu')(pool_proj)
    output = concatenate([conv_1, conv_3, conv_5, pool_proj], axis=2)
    return output


def bio_model(steps, features, outputs):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(steps, features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(filters=32, kernel_size=6, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def bio_model_2d(steps, features, outputs):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, features), activation='relu', input_shape=(steps, features, 1)))
    model.add(Reshape((1998, 32)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(filters=32, kernel_size=6, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def inception_model(steps, features, outputs):
    input_layer = Input(shape=(steps, features))
    x = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu', strides=2)(input_layer)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = Conv1D(filters=64, kernel_size=1, activation='relu', padding='same', strides=1)(x)
    x = Conv1D(filters=192, kernel_size=3, activation='relu', padding='same', strides=1)(x)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = inception_module(x, filters_1=64, filters_3_reduce=96, filters_3=128, filters_5_reduce=16, filters_5=32,
                         filters_pool_proj=32)
    x = inception_module(x, filters_1=128, filters_3_reduce=128, filters_3=192, filters_5_reduce=32, filters_5=96,
                         filters_pool_proj=64)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = inception_module(x, filters_1=192, filters_3_reduce=96, filters_3=208, filters_5_reduce=16, filters_5=48,
                         filters_pool_proj=64)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    x = Dense(outputs, activation='sigmoid', name='output')(x)
    model = Model(input_layer, x, name='inception')
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', loss_weights=0.3, optimizer=sgd, metrics=['accuracy'])
    return model


def bio_inception_model(steps, features, outputs):
    input_layer = Input(shape=(steps, features))
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', strides=2)(input_layer)
    x = inception_module(x, filters_1=64, filters_3_reduce=96, filters_3=128, filters_5_reduce=16, filters_5=32,
                         filters_pool_proj=32)
    x = inception_module(x, filters_1=128, filters_3_reduce=128, filters_3=192, filters_5_reduce=32, filters_5=96,
                         filters_pool_proj=64)
    x = MaxPooling1D(pool_size=3, strides=2)(x)
    x = inception_module(x, filters_1=192, filters_3_reduce=96, filters_3=208, filters_5_reduce=16, filters_5=48,
                         filters_pool_proj=64)
    x = GlobalAveragePooling1D()(x)
    x = Dense(outputs, activation='sigmoid', name='output')(x)
    model = Model(input_layer, x, name='bio_inception')
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', loss_weights=0.3, optimizer=sgd, metrics=['accuracy'])
    return model


# will print model summary, accuracy/loss vs epochs, calibration curves, and save the completed file to a directory
def save_model_activity(images, labels, name, model_type):
    if not os.path.exists(name + '_' + model_type):
        os.makedirs(name + '_' + model_type)
    # initial training with validation hold-out
    verbose, epochs, batch_size = 1, 50, 64
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold = 1
    predictions = []
    ground_truth = []
    for train_index, test_index in skf.split(images, labels):
        file_path = name + '_' + model_type + '/best_model.hdf5'
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.001)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_accuracy', mode='max',
                                                           save_best_only=True, verbose=1)
        print('##################### FOLD ' + str(fold) + ' #####################')
        images_train, labels_train = images[train_index], labels[train_index]
        images_test, labels_test = images[test_index], labels[test_index]
        steps, features = images_train.shape[1], images_train.shape[2]
        print(images_train.shape)
        print(steps, features)
        print('#################### TRAINING ####################')
        if model_type == 'inception':
            model = inception_model(steps, features, 1)
        elif model_type == 'bio_2d':
            model = bio_model_2d(steps, features, 1)
        elif model_type == 'bio_inception':
            model = bio_inception_model(steps, features, 1)
        else:  # currently 'Bio'
            model = bio_model(steps, features, 1)
        history = model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                            shuffle=True, validation_data=(images_test, labels_test),
                            callbacks=[reduce_lr, es, model_checkpoint])
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
        plt.savefig(name + '_' + model_type + '/training_accuracy_fold' + str(fold) + '.pdf')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(name + '_' + model_type + '/training_loss_fold' + str(fold) + '.pdf')
        plt.close()
        fold += 1
    predictions = np.concatenate(predictions)
    labels_test = np.concatenate(ground_truth)
    # print ROC and AUC from the K-fold validation
    roc_plot(predictions, labels_test, name + '_' + model_type + '/' + name + '_' + model_type + '_ROC.pdf')
    # do final training using all data
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.0001)
    # file_path = name + '_' + model_type + '/best_model.hdf5'
    # model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
    # steps, features = images.shape[1], images.shape[2]
    # labels = to_categorical(labels)
    # model = c1d_resnet_model(steps, features, labels.shape[1])
    # model.fit(images, labels, epochs=20, batch_size=batch_size, verbose=verbose, shuffle=True,
    #           callbacks=[reduce_lr, model_checkpoint])
    # model.load_weights(file_path)
    # model.save(name + '_' + model_type)
    # print(model.summary())


def test_model_activity(images, labels, name, model_type):
    print('#################### TESTING ' + name + '_' + model_type + ' ####################')
    if not os.path.exists(name + '_' + model_type + '_TESTING'):
        os.makedirs(name + '_' + model_type + '_TESTING')
    file_path = name + '_' + model_type + '/best_model.hdf5'
    steps, features = images.shape[1], images.shape[2]
    print(images.shape)
    print(steps, features)
    if model_type == '1DRN':
        model = c1d_resnet_model(steps, features, 'Activity')
    elif model_type == 'inception':
        model = inception_model(steps, features, 1)
    elif model_type == 'bio_2d':
        model = bio_model_2d(steps, features, 1)
    elif model_type == 'bio_inception':
        model = bio_inception_model(steps, features, 1)
    else:  # currently 'Bio'
        model = bio_model(steps, features, 1)
    model.load_weights(file_path)
    predictions = model.predict(images)
    print(predictions)
    print(labels)
    roc_plot(predictions, labels, name + '_' + model_type + '_TESTING/' + name + '_' + model_type + '_ROC.pdf')


def scale_arr(input_array):
    # normalize:
    # mean_val = np.nanmean(input_array)
    # input_array = input_array / mean_val
    # standardize:
    mean_val, std_val = np.nanmean(input_array), np.nanstd(input_array)
    if std_val != 0:
        input_array = (input_array - mean_val) / (3 * std_val)  # the 3 is to capture the 99.7 rule at +/- 1
        input_array[input_array > 1] = 1
        input_array[input_array < -1] = -1
    else:
        input_array = np.nan
    return input_array


def clean_input(input_object, identifier):
    out_dict = {}
    print('Cleaning sample ' + identifier + ': ' + str(len(input_object.items())) + ' initial profiles.')
    for key, signals in input_object.items():
        if signals.shape == (2000, 13):  # functional input matrix
            if np.mean(signals[:, 0]) > 10:  # at least 10x depth in the profile
                for col in range(9):  # normalize to mean = 1 all signals, excluding peak locs and nucleotide freqs
                    signals[:, col] = scale_arr(signals[:, col])
                if not np.isnan(signals).any():
                    out_dict[identifier + '_' + key] = signals
    print(' -> ' + str(len(out_dict)) + ' profiles met clean-up criteria.')
    return out_dict


# TODO: add feature matrix compatibility as additional info for two (or more) heads
def main():
    out_dir = '/fh/fast/ha_g/user/rpatton/CNNTesting/Triton_Activity_TSS/'
    images_file = out_dir + 'activity_images.npy'
    scalars_file = out_dir + 'activity_scalars.npy'
    labels_file = out_dir + 'activity_labels.npy'
    idents_file = out_dir + 'activity_identifiers.npy'
    # labels_ref = '/fh/fast/ha_g/user/rpatton/references/LuCaP_activity_z-fpkm.tsv'
    labels_ref = '/fh/fast/ha_g/user/rpatton/references/LuCaP_AvgL2FPKM.tsv'
    if not os.path.isfile(images_file) or not os.path.isfile(labels_file) or not os.path.isfile(scalars_file):
        # process fresh data into easily accessible image/scalar and label files (with shared indices)
        input_folder = '/fh/fast/ha_g/user/rpatton/LuCaP_data/Triton_TSS/results/'
        print('Processing data fresh from ' + input_folder + ' . . .')
        images_dict = {}
        input_paths = [y for x in os.walk(input_folder) for y in glob(os.path.join(x[0], '*TritonProfiles.npz'))]
        samples = [os.path.basename(path).split('_TritonProfiles.npz')[0] for path in input_paths]
        scalars_df = pd.read_table(input_folder + 'TritonCompositeFM.tsv', sep='\t', index_col=0)
        print(scalars_df)
        for sample, input_path in zip(samples, input_paths):
            sample = sample.replace('_LuCaP', '')
            temp_data = np.load(input_path)
            images_dict.update(clean_input(temp_data, sample))
        labels = pd.read_table(labels_ref, sep='\t', index_col=0)
        labels_dict = {}
        # TODO: you left off here . . . need to match scalar data and save as a 1D array
        for _, row in pd.melt(labels.reset_index(), id_vars='index').iterrows():
            labels_dict[row['variable'] + '_' + row['index']] = row['value']
        images_input, labels_input, idents = [], [], []
        for key, val in images_dict.items():
            if key in labels_dict:
                images_input.append(val)
                labels_input.append(labels_dict[key])
                idents.append(key)
        images_input, labels_input, idents = np.stack(images_input), np.stack(labels_input), np.stack(idents)
        np.save(images_file, images_input)
        np.save(labels_file, labels_input)
        np.save(idents_file, idents)
        print('Finished.')
    else:
        images_input, labels_input, idents = np.load(images_file), np.load(labels_file), np.load(idents_file)
    # KEEP below lines for subsetting for future TSS sites
    # if os.path.isfile(images_file) and os.path.isfile(labels_file) and not subset:
    #     print("Loading " + images_file + ' and matched ' + labels_file + ' . . .')
    #     images_input, labels_input = np.load(images_file), np.load(labels_file)
    #     print('Finished.')
    # elif os.path.isfile('subset_' + images_file) and os.path.isfile('subset_' + labels_file):
    #     print("Loading " + 'subset_' + images_file + ' and matched ' + 'subset_' + labels_file + ' . . .')
    #     images_input, labels_input = np.load('subset_' + images_file), np.load('subset_' + labels_file)
    #     print('Finished.')
    # elif not subset:
    #
    # else:
    #     print('Subsetting and saving data from established file . . .')
    #     images_input, labels_input = np.load(images_file), np.load(labels_file)
    #     subset_idx = random.sample(range(len(images_input)), 50000)
    #     images_input, labels_input = images_input[subset_idx], labels_input[subset_idx]
    #     np.save('subset_' + images_file, images_input)
    #     np.save('subset_' + labels_file, labels_input)
    #     print('Finished.')
    print('Total inputs: ' + str(len(images_input)))
    print(images_input.shape)
    with open('/fh/fast/ha_g/user/rpatton/references/PSMA44.txt') as f:
        ho_sites = f.read().splitlines()
    ho_idx = []
    for site in ho_sites:
        ho_idx += [i for i, x in enumerate(idents.flatten().tolist()) if '_' + site in x]
    ho_images, ho_labels, ho_idents =\
        images_input.take(ho_idx, axis=0), labels_input.take(ho_idx, axis=0), idents.take(ho_idx, axis=0)
    images_input, labels_input, idents = \
        np.delete(images_input, ho_idx, axis=0), np.delete(labels_input, ho_idx, axis=0), np.delete(idents, ho_idx, axis=0)
    print('Hold out set dimensions:')
    print(ho_images.shape)
    idx = np.random.permutation(len(images_input))
    images_input, labels_input, idents = images_input[idx], labels_input[idx], idents[idx]
    active_count = np.count_nonzero(labels_input == 1.0)
    print('Total active in training set: ' + str(round(100*active_count/len(images_input))) + '%')
    # TEST/evaluate finished models:
    images_2d = np.expand_dims(ho_images, axis=-1)
    test_model_activity(images_2d, ho_labels, 'Activity_TSS', 'bio_2d')
    # save/evaluate model . . .
    # 1D CNN models: ###################################################################################################
    # save_model_activity(images_input, labels_input, 'Activity_TSS', 'bio')
    # save_model_activity(images_input, labels_input, 'Activity_TSS', '1DRN')
    # save_model_activity(images_input, labels_input, 'Activity_TSS', 'inception')
    # save_model_activity(images_input, labels_input, 'Activity_TSS', 'bio_inception')
    # # no sequence
    # save_model_activity(np.delete(images_input, [9, 10, 11, 12], 2), labels_input, 'Activity_TSS_no-seq', 'bio')
    # save_model_activity(np.delete(images_input, [9, 10, 11, 12], 2), labels_input, 'Activity_TSS_no-seq', '1DRN')
    # save_model_activity(np.delete(images_input, [9, 10, 11, 12], 2), labels_input, 'Activity_TSS_no-seq', 'inception')
    # save_model_activity(np.delete(images_input, [9, 10, 11, 12], 2), labels_input, 'Activity_TSS_no-seq', 'bio_inception')
    # # nucleosome and hetero signals, only
    # save_model_activity(np.delete(images_input, [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], 2), labels_input,
    #                     'Activity_TSS_signals', 'bio')
    # save_model_activity(np.delete(images_input, [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], 2), labels_input,
    #                     'Activity_TSS_signals', '1DRN')
    # save_model_activity(np.delete(images_input, [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], 2), labels_input,
    #                     'Activity_TSS_signals', 'inception')
    # save_model_activity(np.delete(images_input, [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], 2), labels_input,
    #                     'Activity_TSS_signals', 'bio_inception')
    # 2D CNN models: ###################################################################################################
    # images_2d = np.expand_dims(images_input, axis=-1)
    # save_model_activity(images_2d, labels_input, 'Activity_TSS', 'bio_2d')
    # no sequence
    # images_2d = np.expand_dims(np.delete(images_input, [9, 10, 11, 12], 2), axis=-1)
    # save_model_activity(images_2d, labels_input, 'Activity_TSS_no-seq', 'bio_2d')
    # nucleosome and hetero signals, only
    # images_2d = np.expand_dims(np.delete(images_input, [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12], 2), axis=-1)
    # save_model_activity(images_2d, labels_input, 'Activity_TSS_signals', 'bio_2d')
    # Deep models: #####################################################################################################
    # Combination models: ##############################################################################################
    ####################################################################################################################
    """
    to remove specific lines from the signal profile, pass the following [lines to delete]:                
    np.delete(images_input, [1, 2, 3, 4, 5, 6, 7, 8, 9], 2) -> 2 here refers to the second dim of the array
    1: Depth (GC-corrected, if provided)
    2: Nucleosome-level phased profile
    3: Nucleosome center profile (GC-corrected, if provided)
    4: Mean fragment size
    5: Fragment size Shannon entropy
    6: Fragment heterogeneity (unique fragment lengths / total fragments)
    7: Fragment MAD (Mean Absolute Deviation)
    8: Short:long ratio (x <= 120 / 140 <= x <= 250)
    9: Peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***
    10: A (Adenine) frequency**
    11: C (Cytosine) frequency**
    12: G (Guanine) frequency**
    13: T (Tyrosine) frequency**
    """
    ####################################################################################################################
    # score, params = evolve_model(images_input, labels_input, 10, 1000)
    # print(score)
    # print(params)

    print('Finished.')


if __name__ == "__main__":
    main()
