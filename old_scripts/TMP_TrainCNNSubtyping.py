#! /usr/bin/env python3
# ml Python/3.8.2-GCCcore-9.3.0

import random
from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

# load data generate by GenerateInputTSS.py
images_input = np.load('images_500.npy')
labels_input = np.load('labels.npy')
labels_input = to_categorical(labels_input)  # one-hot encode labels: ARPC, AMPC, ARlow, NEPC = 0,1,2,3


# define the maximization (will be minimized, so negative objective function)
def obj_fun(theta, obs):  # theta=unknown params, x=observations
    return -np.sum(np.log(np.sum(obs*theta, axis=1)))


# likelihood maximization
def max_likelihood_estimate(obs):
    # bounds for the coefficients
    bounds = Bounds(0, 1)
    # constraint: theta sums to 1
    constraint = [{"type": "eq", "fun": lambda theta: np.sum(theta) - 1}]
    # initial guess
    theta_init = np.full(obs.shape[1], 1/obs.shape[1])
    res = minimize(fun=obj_fun, args=obs, x0=theta_init, bounds=bounds, constraints=constraint)
    return res.x


# increase all class samples numbers to match the largest group
# method: randomly duplicate inputs and add random normal noise
def augment_samples(init_images, init_labels, input_type):
    if input_type == "profile":
        class_counts = np.sum(init_labels, axis=0).astype(int)
        print("Old Sample Mix (initial inputs): " + str(class_counts))
        out_images = init_images
        out_labels = init_labels
        highest_count = max(class_counts)
        for i in range(class_counts.shape[0]):
            temp_images = []
            temp_labels = []
            label = np.eye(class_counts.shape[0])[i]
            samples = class_counts[i]
            if samples >= highest_count:
                continue
            indices = (init_labels == label).all(axis=1).nonzero()[0]
            while samples < highest_count:
                aug_sample = np.transpose(init_images[np.random.choice(indices)])
                aug_sample[0] *= np.random.normal(1, 0.3, aug_sample.shape[1])
                aug_sample[0] /= np.max(np.abs(aug_sample[0]), axis=0)
                temp_images.append(np.transpose(np.asmatrix(aug_sample)))
                temp_labels.append(label)
                samples += 1
            out_images = np.append(out_images, temp_images, axis=0)
            out_labels = np.append(out_labels, temp_labels, axis=0)
        new_class_counts = np.sum(out_labels, axis=0).astype(int)
        print("New Sample Mix (initial inputs): " + str(new_class_counts))
        del init_images, init_labels
        return out_images, out_labels
    elif input_type == "fragment":
        print("not built yet!")
    else:
        print("###ERROR### invalid input_type specified for augment_samples")


# summarize scores
def summarize_results(scores, kernel_size, kernel_number):
    print(scores, kernel_size, kernel_number)
    # summarize mean and standard deviation
    for i in range(len(scores)):
        m, s = mean(scores[i]), std(scores[i])
        if len(kernel_size) > 0:
            print('Kernel Size =%d: %.3f%% (+/-%.3f)' % (kernel_size[i], m, s))
            pyplot.boxplot(scores, labels=kernel_size)
            pyplot.savefig('kernel-size_cnn_filters.png')
        elif len(kernel_number) > 0:
            print('Kernel Number =%d: %.3f%% (+/-%.3f)' % (kernel_number[i], m, s))
            pyplot.boxplot(scores, labels=kernel_number)
            pyplot.savefig('kernel-number_cnn_filters.png')


# fit and evaluate a model
def evaluate_model(train_data, train_labels, test_data, test_labels, k_size, k_num):
    verbose, epochs, batch_size = 0, 10, 32
    steps, features, outputs = train_data.shape[1], train_data.shape[2], train_labels.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=k_num, kernel_size=k_size, activation='relu', input_shape=(steps, features)))
    model.add(Conv1D(filters=k_num, kernel_size=k_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=0)
    predictions = model.predict(test_data)
    true_mix = np.sum(test_labels, axis=0)/test_labels.shape[0]
    pred_mix = np.sum(predictions, axis=0)/predictions.shape[0]
    mle_mix = max_likelihood_estimate(predictions)
    errors_pred = np.abs(true_mix - pred_mix) / true_mix * 100
    errors_mle = np.abs(true_mix - mle_mix) / true_mix * 100
    print("True Mixture: " + str(true_mix))
    print("Predicted Mixture: " + str(pred_mix))
    print("Error (%): " + str(errors_pred))
    print("MLE Mixture: " + str(mle_mix))
    print("Error (%): " + str(errors_mle))
    return accuracy


# run an experiment
def run_experiment(k_params, n_params, repeats=10):
    # repeat experiment
    all_scores = list()
    for k in k_params:
        for n in n_params:
            # repeat experiment
            scores = list()
            for r in range(repeats):
                # subset data for testing (random 100,000 from samples)
                # divide into 80,000 (training) and 20,000 (testing)
                subset = random.sample(range(images.shape[0]), 100000)
                images_sub = images[subset]
                labels_sub = labels[subset]
                images_train = images_sub[:80000]
                images_test = images_sub[80001:]
                labels_train = labels_sub[:80000]
                labels_test = labels_sub[80001:]
                # evaluate model
                score = evaluate_model(images_train, labels_train, images_test, labels_test, k, n)
                score = score * 100.0
                print('>p=%d #%d: %.3f' % (n, r + 1, score))
                scores.append(score)
            all_scores.append(scores)
        # summarize results
    summarize_results(all_scores, k_params, n_params)


images, labels = augment_samples(images_input, labels_input, "profile")
kernel_size_params = [3, 6, 12, 24, 50, 100, 200]
kernel_n_params = [8, 16, 32, 64, 128]
run_experiment(kernel_size_params, [8])
run_experiment([6], kernel_n_params)
