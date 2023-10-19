#! /usr/bin/env python3
# ml TensorFlow/2.4.1-fosscuda-2020b

import os
import re
import seaborn
import statistics
import numpy as np
import pandas as pd
from glob import glob
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds


# define the maximization (will be minimized, so negative objective function)
def obj_fun(theta, obs):  # theta=unknown params, x=observations
    return -np.sum(np.log(np.sum(obs * theta, axis=1)))


def threshold(list, min):
    return np.asarray([array for array in list if (array > min).any()])


def one_max(list):
    classes = list.shape[1]
    return np.asarray([to_categorical(array.argmax(), num_classes=classes) for array in list])


def softmax(vector, temp):
    e = np.exp(vector / temp)
    return e / e.sum()


# likelihood maximization
def max_likelihood_estimate(obs):
    # bounds for the coefficients
    bounds = Bounds(0, 1)
    # constraint: theta sums to 1
    constraint = [{"type": "eq", "fun": lambda theta: np.sum(theta) - 1}]
    # initial guess
    theta_init = np.full(obs.shape[1], 1 / obs.shape[1])
    res = minimize(fun=obj_fun, args=obs, x0=theta_init, bounds=bounds, constraints=constraint)
    return res.x


def plot_heatmap(df, name, sample, sub):
    plt.imshow(df, aspect=0.2, cmap="RdYlBu")
    plt.colorbar()
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df)), df.index)
    plt.title(sample + ' CNN confidence/predictions heatmap ' + sub)
    plt.savefig(name + '_' + sample + '_' + sub + '_heatmap.pdf')
    plt.close()


def plot_confidence(df, name):
    df = df[['sample', 'gene', 'ARPC', 'NEPC', 'HD']]
    all_samples = df['sample'].unique().tolist()
    goi = ['BRCA1', 'ERG', 'FOXA1', 'AR', 'HOXB13']
    gen = (sample for sample in all_samples if len(sample) < 10)
    for sample in gen:
        sample_df = df.loc[df['sample'] == sample]
        sample_df = sample_df.drop(columns=['sample'])
        sample_df = sample_df.set_index('gene')
        # heatmaps:
        top_arpc = sample_df.sort_values(['ARPC'], ascending=False).head(30)
        top_nepc = sample_df.sort_values(['NEPC'], ascending=False).head(30)
        top_hd = sample_df.sort_values(['HD'], ascending=False).head(30)
        top_goi = sample_df.reindex(index=goi)
        plot_heatmap(top_arpc, name, sample, 'Top30APRC')
        plot_heatmap(top_nepc, name, sample, 'Top30NEPC')
        plot_heatmap(top_hd, name, sample, 'Top30HD')
        plot_heatmap(top_goi, name, sample, 'GOI')
        # distributions:
        arpc_mean = round(statistics.mean(sample_df['ARPC']), 3)
        arpc_std = round(statistics.stdev(sample_df['ARPC']), 3)
        nepc_mean = round(statistics.mean(sample_df['NEPC']), 3)
        nepc_std = round(statistics.stdev(sample_df['NEPC']), 3)
        hd_mean = round(statistics.mean(sample_df['HD']), 3)
        hd_std = round(statistics.stdev(sample_df['HD']), 3)
        summary = 'mean, stdev | ARPC: ' + str(arpc_mean) + ', ' + str(arpc_std) + ' | NEPC: ' + str(nepc_mean) + ', '\
                  + str(nepc_std) + ' | HD: ' + str(hd_mean) + ', ' + str(hd_std)
        seaborn.kdeplot(sample_df['ARPC'], shade=True, label='ARPC')
        seaborn.kdeplot(sample_df['NEPC'], shade=True, label='NEPC')
        seaborn.kdeplot(sample_df['HD'], shade=True, label='HD')
        plt.legend(prop={'size': 16}, title='Subtype')
        plt.suptitle(sample + ' CNN confidence/predictions densities')
        plt.title(summary)
        plt.xlabel('confidence')
        plt.ylabel('density')
        plt.savefig(name + '_' + sample + '_densities.pdf')
        plt.close()


def test_model():
    # from tensorflow import keras
    model_dir = 'Subtyping-3S_model_TFBS_1DRN'
    model = keras.models.load_model(model_dir)
    temp = 1.00
    input_direct = 'TFBS-wise_samples/'
    input_paths = [y for x in os.walk(input_direct) for y in glob(os.path.join(x[0], '*profiles.npy'))]
    individual = True
    if individual:
        df = []
        df_scaled = []
        for path in input_paths:
            name = os.path.basename(path)
            images = np.load(path)
            logits = model.predict(images)
            predictions = [softmax(item, 1) for item in logits]
            predictions_scaled = [softmax(item, temp) for item in logits]
            sample = re.sub('_LuCaP.*\.npy', '', name)
            gene = re.sub('.*_LuCaP\.', '', name)
            gene = re.sub('\.hg38.*', '', gene)
            d = {'sample': sample, 'gene': gene, 'ARPC': predictions[0][0],
                 'HD': predictions[0][1], 'NEPC': predictions[0][2]}
            d_scaled = {'sample': sample, 'gene': gene, 'ARPC': predictions_scaled[0][0],
                        'HD': predictions_scaled[0][1], 'NEPC': predictions_scaled[0][2]}
            df.append(d)
            df_scaled.append(d_scaled)
        df = pd.DataFrame(df)
        df_scaled = pd.DataFrame(df_scaled)
        df.to_csv('predictions.tsv', sep='\t')
        df_scaled.to_csv('predictions_scaled.tsv', sep='\t')
    else:
        for path in input_paths:
            name = os.path.basename(path)
            print(name)
            images = np.load(path)
            logits = model.predict(images)
            predictions = np.asarray([softmax(item, 1) for item in logits])
            predictions_scaled = np.asarray([softmax(item, temp) for item in logits])
            pred_mix = np.sum(predictions, axis=0) / predictions.shape[0]
            mle_mix = max_likelihood_estimate(predictions)
            pred_mix_scaled = np.sum(predictions_scaled, axis=0) / predictions_scaled.shape[0]
            mle_mix_scaled = max_likelihood_estimate(predictions_scaled)
            print("Sum Mixture:\t[ " + '  '.join(["{:.2%}".format(val) for val in pred_mix]) + ' ]')
            print("MLE Mixture:\t[ " + '  '.join(["{:.2%}".format(val) for val in mle_mix]) + ' ]')
            print("Sum Mixture (T-Scaled):\t[ " + '  '.join(["{:.2%}".format(val) for val in pred_mix_scaled]) + ' ]')
            print("MLE Mixture (T-Scaled):\t[ " + '  '.join(["{:.2%}".format(val) for val in mle_mix_scaled]) + ' ]')


def plot_kernels():
    import keras.backend as K
    model_dir = 'TFBS-3S_model'
    model = keras.models.load_model(model_dir)
    cnn1 = model.get_layer('conv1d_60').kernel
    cnn1_np = K.eval(cnn1)  # need to be transposed? wrong shape format


def main():
    # test_model()
    df = pd.read_csv('predictions.tsv', sep='\t')
    df_scaled = pd.read_csv('predictions_scaled.tsv', sep='\t')
    plot_confidence(df, 'TFBS-3S-Confidences')
    plot_confidence(df_scaled, 'TFBS-3S-ScaledConfidences')


if __name__ == "__main__":
    main()
