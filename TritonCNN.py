# v3.0.0, 10/06/2023
# Robert D Patton, rpatton@fredhutch.org
# Ha Lab, Fred Hutchinson Cancer Research Center

"""
Newest version of the Trion-based CNN. This provides a main and wrapper for running
TritonCNN and associated pre-processing and post-processing steps, and was modified from TritonCNN_Huang.py.
This version is built on the imporvements made by Akira Nair.
"""

import os
from glob import glob
import numpy as np
import pandas as pd

import models
from preprocessing import generate_scalars, generate_images, compile_data

process_data = True

def generate_data(home, name, regions, ref_path):
    """
    Generate data for the CNN from the LuCaP data. This function will generate the data for each region
    in the regions dictionary, and save the data as a numpy array in the home directory.
    :param home: the home directory where the data will be saved #ERROR??
    :param name: the name of the data set
    :param regions: a dictionary of regions to be processed, with the region name as the key and a list of
    data types as the value
    :param ref_path: the path to the reference file
    :return: None
    """
    if not os.path.isdir('data/'):
        os.mkdir('data/')
    for region, data_types in regions.items():
        for data_type in data_types:
            print('Processing ' + region + ' ' + data_type + ' data . . .')
            out_dir = 'data/' + name + '_' + region + '_' + data_type + '/'
            data_dir = home + 'Triton_' + region + '/results/'
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            if data_type == 'scalars':
                generate_scalars.generate(out_dir, data_dir, ref_path)
            if data_type == 'images':
                generate_images.generate(out_dir, data_dir)
    print('Finished generating data.')

def preprocess_data(images_path, scalars_path, labels_path, output):
    """
    Preprocess the data. This function calls preprocessing/compile_data.py and preprocessing/splits.py to 
    compile and convert all data into an , and split into customizable test/train sets. Full compiled data 
    is stored as compiled_data.npz in output location, while train/test splits are stored as ??.npz in ?? location
    :param ?: ?
    :return: None
    """
    print('Compiling Data, storing outputs in '+ output)
    compile_data.compile(images_path, scalars_path, labels_path, output)
    print('Finished compiling data.')

def main():
    """
    First, process the data into a format that is easily accessible by the CNN.
    . . . flesh out this docstring
    """
    reference_path = '/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/Ensembl109_reference.bed'
    training_home = '/fh/fast/ha_g/user/rpatton/LuCaP_data/'
    training_name = 'LuCaP'
    # training_regions = {'TSS': ['images', 'scalars'],
    #                     'GB': ['scalars'],
    #                     'TFBS': ['image-panels']}
    
    training_regions = {'TSS': ['images', 'scalars']}

    images_path = '/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/images'
    scalars_path = '/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/scalars'
    labels_path = '/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/labels/labels_continuous_tpm.csv'
    output = '/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/continuous'
    
    if process_data:
        #generate_data(training_home, training_name, training_regions, reference_path)
        preprocess_data(images_path, scalars_path, labels_path, output)



    quit()




    out_dir = '/fh/fast/ha_g/user/rpatton/CNNTesting/Triton_Activity_TSS/'
    images_file = out_dir + 'activity_images.npy'
    scalars_file = out_dir + 'activity_scalars.npy'
    labels_file = out_dir + 'activity_labels.npy'
    idents_file = out_dir + 'activity_identifiers.npy'
    # labels_ref = '/fh/fast/ha_g/user/rpatton/references/LuCaP/LuCaP_activity_z-fpkm.tsv'
    labels_ref = '/fh/fast/ha_g/user/rpatton/references/LuCaP/LuCaP_AvgL2FPKM.tsv'
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


    print('Finished.')


if __name__ == "__main__":
    main()
