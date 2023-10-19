# ml TensorFlow/2.4.1-fosscuda-2020b
# sbatch -o out.txt --gres=gpu CNN_prof.py

import random
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as backend
from glob import glob
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, LSTM, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from itertools import cycle
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, r2_score
# from keras.layers.merge import concatenate


# fit and evaluate the model
def evaluate_model(train_data, train_labels, test_data, test_labels, params):
    ks1, ks2, ks3, kn1, kn2, kn3, mp1, mp2, mp3, drop, lstm, dense1, dense2, batch = [var for var in params]
    verbose, epochs, batch_size = 0, 5, batch
    steps, features, outputs = train_data.shape[1], train_data.shape[2], train_labels.shape[1]
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
    model.add(Dense(outputs, activation='softmax'))
    # model.add(Dense(1, activation='linear'))  # tanh? linear?
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=0)
    # loss = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=0)
    # multi headed approach below (not up-to-date)#
    ####################################################################################################################
    # # head 1
    # inputs1 = Input(shape=(steps, features))
    # conv1 = Conv1D(filters=kn1, kernel_size=ks1, activation='relu')(inputs1)
    # pool1 = MaxPooling1D(pool_size=mp1)(conv1)
    # flat1 = Flatten()(pool1)
    # # head 2
    # inputs2 = Input(shape=(steps, features))
    # conv2 = Conv1D(filters=kn2, kernel_size=ks2, activation='relu')(inputs2)
    # pool2 = MaxPooling1D(pool_size=mp2)(conv2)
    # flat2 = Flatten()(pool2)
    # # head 3
    # inputs3 = Input(shape=(steps, features))
    # conv3 = Conv1D(filters=kn3, kernel_size=ks3, activation='relu')(inputs3)
    # pool3 = MaxPooling1D(pool_size=mp3)(conv3)
    # flat3 = Flatten()(pool3)
    # # merge
    # merged = concatenate([flat1, flat2, flat3])
    # # interpretation
    # dense_layer = Dense(dense, activation='relu')(merged)
    # soft = Dense(outputs, activation='softmax')(dense_layer)
    # model = Model(inputs=[inputs1, inputs2, inputs3], outputs=soft)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # fit network
    # model.fit([train_data, train_data, train_data], train_labels,
    #           epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # evaluate model
    # _, accuracy = model.evaluate([test_data, test_data, test_data], test_labels, batch_size=batch_size, verbose=0)
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
            labels_train = to_categorical(labels[train_index])
            images_test = images[test_index]
            labels_test = to_categorical(labels[test_index])
            # evaluate model
            score = evaluate_model(images_train, labels_train, images_test, labels_test, params)
            score = score * 100.0
            scores.append(score)
    else:
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
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
    kernel_size_pool = [0, 2, 3, 4, 5, 8, 12, 20, 50, 100]
    ks1_pool = [2, 3, 4, 5, 6, 8, 12, 20, 50, 100]
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
    kernel_size_pool = [0, 2, 3, 4, 8]
    ks1_pool = [2, 3, 6]
    kernel_num_pool = [16, 32, 64]
    max_pool_pool = [0, 2, 3, 4]
    drop_pool = [0, 0.5]
    lstm_pool = [0, 128, 512, 1024]
    dense_pool = [0, 100, 200, 300]
    dense1_pool = [100, 200, 300]
    batch_pool = [32]
    best_score = 100  # 0 for accuracy, 100 for loss
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
            score = run_experiment(images, labels, params, True)
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
        score = run_experiment(images, labels, params, True)
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
    # Compute ROC curve and ROC area for each class
    n_classes = truths.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(truths[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(truths.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'indigo', 'darkolivegreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-class receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.savefig(title)
    plt.close()


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


# replicate the favored architecture and hyper parameters below before doing final training
def finalized_model(steps, features, outputs):
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(steps, features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(filters=32, kernel_size=12, activation='relu'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(outputs, activation='relu'))
    model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
    ####################################################################################################################
    # model = Sequential()
    # model.add(Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=(steps, features)))
    # model.add(MaxPooling1D(pool_size=3))
    # model.add(Conv1D(filters=32, kernel_size=12, activation='relu'))
    # model.add(MaxPooling1D(pool_size=5))
    # model.add(Flatten())
    # model.add(Dense(200, activation='relu'))
    # model.add(Dense(outputs, activation='linear'))  # was linear
    # model.compile(loss='mse', optimizer='adam')
    ####################################################################################################################
    # ### see below for 2-headed model ###
    # verbose, epochs, batch_size = 1, 50, 32
    # steps, features, outputs = images.shape[1], images.shape[2], labels.shape[1]
    # # head 1
    # inputs1 = Input(shape=(steps, features))
    # conv1a = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    # pool1a = MaxPooling1D(pool_size=8)(conv1a)
    # drop1a = Dropout(0.5)(pool1a)
    # conv1b = Conv1D(filters=128, kernel_size=8, activation='relu')(drop1a)
    # conv1c = Conv1D(filters=128, kernel_size=8, activation='relu')(conv1b)
    # pool1b = MaxPooling1D(pool_size=5)(conv1c)
    # # head 2
    # inputs2 = Input(shape=(steps, features))
    # conv2a = Conv1D(filters=64, kernel_size=167, activation='relu')(inputs2)
    # pool2a = MaxPooling1D(pool_size=4)(conv2a)
    # drop2a = Dropout(0.5)(pool2a)
    # conv2b = Conv1D(filters=128, kernel_size=6, activation='relu')(drop2a)
    # conv2c = Conv1D(filters=128, kernel_size=6, activation='relu')(conv2b)
    # pool2b = MaxPooling1D(pool_size=9)(conv2c)
    # # merge and continue
    # merged = concatenate([pool1b, pool2b])
    # lstm1 = LSTM(512, dropout=0.5)(merged)
    # # lstm2 = LSTM(128, dropout=0.5)(lstm1)
    # dense1 = Dense(200, activation='relu')(lstm1)
    # dense2 = Dense(100, activation='relu')(dense1)
    # soft = Dense(outputs, activation='softmax')(dense2)
    # model = Model(inputs=[inputs1, inputs2], outputs=soft)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ####################################################################################################################
    return model


# will print model summary, accuracy/loss vs epochs, calibration curves,  and save the completed file to a directory
def save_model(images, labels, name):
    if not os.path.exists(name):
        os.makedirs(name)
    # initial training with validation hold-out
    temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
    verbose, epochs, batch_size = 1, 10, 32
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold = 1
    predictions = []
    ground_truth = []
    for train_index, test_index in skf.split(images, labels):
        print('#################### FOLD ' + str(fold) + ' ####################')
        images_train = images[train_index]
        labels_train = to_categorical(labels[train_index])
        images_test = images[test_index]
        labels_test = to_categorical(labels[test_index])
        steps, features, outputs = images_train.shape[1], images_train.shape[2], labels_train.shape[1]
        model = finalized_model(steps, features, outputs)
        history = model.fit(images_train, labels_train, epochs=epochs, validation_data=(images_test, labels_test),
                            batch_size=batch_size, verbose=verbose, shuffle=True)
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
    # perform temperature scaling and plot along with history (using validation set)
    predictions = np.concatenate(predictions)
    labels_test = np.concatenate(ground_truth)
    soft_predictions = [softmax(item) for item in predictions]
    labels_true = tf.convert_to_tensor(np.argmax(labels_test, axis=1), dtype=tf.int32, name='labels_true')
    labels_predicted = tf.convert_to_tensor(predictions, dtype=tf.float32, name='logits')
    ece = tfp.stats.expected_calibration_error(num_bins=50, logits=labels_predicted, labels_true=labels_true)
    ece = backend.get_value(ece)
    print('ECE before temperature scaling: ' + str(ece))
    calibration_plot(labels_test, soft_predictions, name + '/Reliability_Diagram_Pre-Scaling.pdf', str(ece))
    def compute_loss():
        predictions_w_temp = tf.math.divide(predictions, temp)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            tf.convert_to_tensor(labels_test), predictions_w_temp))
        return loss
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    print('Temperature Initial value: {}'.format(temp.numpy()))
    for _ in range(100):  # training epochs
        optimizer.minimize(compute_loss, var_list=[temp])
    print('Temperature Final value: {}'.format(temp.numpy()))
    scaled_predictions = tf.math.divide(predictions, temp)
    soft_scaled_predictions = np.asarray([softmax(item) for item in scaled_predictions])
    labels_predicted = tf.convert_to_tensor(scaled_predictions, dtype=tf.float32, name='logits')
    ece_scaled = tfp.stats.expected_calibration_error(num_bins=50, logits=labels_predicted, labels_true=labels_true)
    ece_scaled = backend.get_value(ece_scaled)
    print('ECE after temperature scaling: ' + str(ece_scaled))
    calibration_plot(labels_test, soft_scaled_predictions,
                     name + '/Reliability_Diagram_Post-Scaling.pdf', str(ece_scaled))
    with open(name + '/temperature.txt', 'w') as temp_file:
        temp_file.write(str(temp))
    # print ROC and AUC from the K-fold validation
    roc_plot(soft_scaled_predictions, labels_test, name + '/ROC.pdf')
    # do final training using all data
    labels = to_categorical(labels)
    steps, features, outputs = images.shape[1], images.shape[2], labels.shape[1]
    model = finalized_model(steps, features, outputs)
    model.fit(images, labels, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
    model.save(name)
    print(model.summary())


# this save model does regression based analysis and uses 20% hold-out
def save_model_fpkm(images, labels, name):
    if not os.path.exists(name):
        os.makedirs(name)
    # initial training with validation hold-out
    verbose, epochs, batch_size = 1, 20, 128
    index_80 = round(0.8 * len(labels))
    images_train = images[:index_80]
    labels_train = labels[:index_80]
    images_test = images[(index_80 + 1):]
    labels_test = labels[(index_80 + 1):]
    print('#################### TRAINING ####################')
    steps, features = images_train.shape[1], images_train.shape[2]
    model = finalized_model(steps, features, 1)
    model.fit(images_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True)
    print('#################### TESTING ####################')
    predictions = model.predict(images_test)
    r2_score_model = r2_score(labels_test, predictions)
    print("R2 Score: ", r2_score_model)
    fig, ax = plt.subplots()
    ax.plot(labels_test, predictions, linewidth=0, marker=".")
    ax.plot([-1.1, 1.1], [-1.1, 1.1], ls="--", c=".3", alpha=.5)
    ax.set_xlabel("True Normalized Log2 FPKM", fontweight="bold")
    ax.set_ylabel("Predicted Normalized Log2 FPKM", fontweight="bold")
    plt.text(-0.9, 0.9, 'R2: ' + str(r2_score_model))
    plt.savefig(name + '/R2Plot.pdf')
    plt.close()
    model.save(name)
    print(model.summary())


def loo(folders, name):
    if not os.path.exists(name):
        os.makedirs(name)
    # initial training with validation hold-out
    temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
    verbose, epochs, batch_size = 1, 50, 32
    image_paths = []
    label_paths = []
    for folder in folders:
        image_paths.append([y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*profiles.npy'))])
        label_paths.append([y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*labels.npy'))])
    folds = min(map(len, image_paths))  # folds = number of fewest samples
    predictions = []
    ground_truth = []
    loo_indices = list(range(0, 50, folds))
    for i in range(folds):
        print('#################### FOLD ' + str(i + 1) + ' ####################')
        images_train = []
        labels_train = []
        images_test = []
        labels_test = []
        for j in range(len(image_paths)):
            looj = loo_indices[:math.trunc(len(image_paths[j])/folds)]
            loo_image_paths = [image_paths[j][k] for k in looj]
            loo_label_paths = [label_paths[j][k] for k in looj]
            train_image_paths = [element for k, element in enumerate(image_paths[j]) if k not in looj]
            train_label_paths = [element for k, element in enumerate(label_paths[j]) if k not in looj]
            for path in loo_image_paths:
                images_test.append(np.load(path))
            for path in loo_label_paths:
                labels_test.append(np.load(path))
            for path in train_image_paths:
                images_train.append(np.load(path))
            for path in train_label_paths:
                labels_train.append(np.load(path))
        images_train = np.concatenate(images_train)
        labels_train = np.concatenate(labels_train)
        images_test = np.concatenate(images_test)
        labels_test = np.concatenate(labels_test)
        labels_test = to_categorical(labels_test)
        labels_train = to_categorical(labels_train)
        # idx = np.random.permutation(len(images_train))
        # images_train, labels_train = images_train[idx], labels_train[idx]
        # idy = np.random.permutation(len(images_test))
        # images_test, labels_test = images_test[idy], labels_test[idy]
        steps, features, outputs = images_train.shape[1], images_train.shape[2], labels_train.shape[1]
        model = finalized_model(steps, features, outputs)
        history = model.fit(images_train, labels_train, epochs=epochs, validation_data=(images_test, labels_test),
                            batch_size=batch_size, verbose=verbose, shuffle=True)
        predictions.append(model.predict(images_test))
        ground_truth.append(labels_test)
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(name + '/training_accuracy_fold' + str(i) + '.pdf')
        plt.close()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(name + '/training_loss_fold' + str(i) + '.pdf')
        plt.close()
        loo_indices = [x + 1 for x in loo_indices]
    # perform temperature scaling and plot along with history (using validation set)
    predictions = np.concatenate(predictions)
    labels_test = np.concatenate(ground_truth)
    soft_predictions = [softmax(item) for item in predictions]
    labels_true = tf.convert_to_tensor(np.argmax(labels_test, axis=1), dtype=tf.int32, name='labels_true')
    labels_predicted = tf.convert_to_tensor(predictions, dtype=tf.float32, name='logits')
    ece = tfp.stats.expected_calibration_error(num_bins=50, logits=labels_predicted, labels_true=labels_true)
    ece = backend.get_value(ece)
    print('ECE before temperature scaling: ' + str(ece))
    calibration_plot(labels_test, soft_predictions, name + '/Reliability_Diagram_Pre-Scaling.pdf', str(ece))
    def compute_loss():
        predictions_w_temp = tf.math.divide(predictions, temp)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            tf.convert_to_tensor(labels_test), predictions_w_temp))
        return loss
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    print('Temperature Initial value: {}'.format(temp.numpy()))
    for _ in range(100):  # training epochs
        optimizer.minimize(compute_loss, var_list=[temp])
    print('Temperature Final value: {}'.format(temp.numpy()))
    scaled_predictions = tf.math.divide(predictions, temp)
    soft_scaled_predictions = np.asarray([softmax(item) for item in scaled_predictions])
    labels_predicted = tf.convert_to_tensor(scaled_predictions, dtype=tf.float32, name='logits')
    ece_scaled = tfp.stats.expected_calibration_error(num_bins=50, logits=labels_predicted, labels_true=labels_true)
    ece_scaled = backend.get_value(ece_scaled)
    print('ECE after temperature scaling: ' + str(ece_scaled))
    calibration_plot(labels_test, soft_scaled_predictions,
                     name + '/Reliability_Diagram_Post-Scaling.pdf', str(ece_scaled))
    with open(name + '/temperature.txt', 'w') as temp_file:
        temp_file.write(str(temp))
    # print ROC and AUC from the K-fold validation
    roc_plot(soft_scaled_predictions, labels_test, name + '/ROC.pdf')


# normalize expression values to be in the range [-1, 1]
def normalize_fpkm(labels):  # potentially use z-score in the future?
    log_cov = np.log(labels)
    return (log_cov - min(log_cov))/(max(log_cov) - min(log_cov))
    # return 2 * (log_cov - min(log_cov))/(max(log_cov) - min(log_cov)) - 1


# def main():
#     images_input = np.load('training/coverageprofiles.npy')
#     labels_input = np.load('training/coveragelabels.npy')
#     idx = np.random.permutation(len(images_input))
#     images_input, labels_input = images_input[idx], labels_input[idx]
#     # images_input, labels_input = subset(images_input, labels_input, 1000)
#     labels_input = normalize_fpkm(labels_input)
#     save_model_fpkm(images_input, labels_input, 'FPKM_model')


# def main():
#     # load data generated by GenerateProfileInputs.py
#     folders = ['training/TFBS-1bp/ARPC/', 'training/TFBS-1bp/NEPC/', 'training/TFBS-1bp/HD/']
#     loo(folders, 'TFBS-3S_model_sample-wise')

def main():
    # load data generated by GenerateProfileInputs.py
    #images_input = np.load('training/profiles_TSS_3S.npy')
    #labels_input = np.load('training/labels_TSS_3S.npy')
    images_input = np.load('BTCUSD_1f_1440l_10p_CV_profiles.npy')
    labels_input = np.load('BTCUSD_1f_1440l_10p_CV_labels.npy') + 1  # for categorization
    # images, labels = subset(images_input, labels_input, 20000)
    # be sure to double-check desired hyper parameters
    # double-check architecture and parameters before saving
    idx = np.random.permutation(len(images_input))
    images_input, labels_input = images_input[idx], labels_input[idx]
    #save_model(images_input, labels_input, 'BTCUSD_1f_1440l_10p_CV_labels')
    score, params = evolve_model(images_input, labels_input, 10, 1000)
    print(score)
    print(params)


if __name__ == "__main__":
    main()
