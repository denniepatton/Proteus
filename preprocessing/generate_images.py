import pandas as pd
import numpy as np
import os
import logging
import multiprocessing
import concurrent.futures

"""
This script generates images from the NPZ files from Triton stored in data_dir. 
The generated images are saved as CSV to output_dir, along with a JSON file containing image counts information.

Main Function
    - generate(output_dir, data_dir)

"""

### LOGGING CONFIGURATION
logging.basicConfig(filename="generate_images.log", filemode='w', level=logging.INFO)
### ---------------------

def read_npz(filepath):
    npz = np.load(filepath)
    return npz

def process_npz(input_object, identifier="", tss_window_size=2000, min_seq_depth=10):
    """
    Cleans a npz input.

    Args:
        input_object (np.ndarray): The input object to be cleaned.
        identifier (str, optional): An identifier for the input object. Defaults to "".
        tss_window_size (int, optional): The size of the TSS (Transcription Start Site) window. Defaults to 2000.
        min_seq_depth (int, optional): The minimum sequence depth required in the profile. Defaults to 10.
    
    Calls:
        normalize_v2

    Returns:
        dict: A dictionary containing the cleaned profiles that meet the clean-up criteria.
    """
    out_dict = {}
    logging.info('Cleaning sample ' + identifier + ': ' +
          str(len(input_object.items())) + ' initial profiles.')
    for key, signals in input_object.items():
        if signals.shape == (11, tss_window_size):  # functional input matrix
            signals = signals.T  # functional input matrix
            if np.mean(signals[:, 0]) > min_seq_depth:  # at least 10x depth in the profile
                # normalize to mean = 1 all signals, excluding peak locs and nucleotide freqs
                for col in range(7):
                    signals[:, col] = normalize_v2(signals[:, col])
                if not np.isnan(signals).any():
                    out_dict[identifier + '_' + key] = signals
    logging.info(' -> ' + str(len(out_dict)) + ' profiles met clean-up criteria.')
    return out_dict

def normalize(input_array):
    mean_val, std_val = np.nanmean(input_array), np.nanstd(input_array)
    if std_val != 0:
        input_array = (input_array - mean_val) / (3 * std_val)
        input_array[input_array > 1] = 1
        input_array[input_array < -1] = -1
    else:
        input_array = np.nan
    return input_array

# NOTE: JULY 19, implemented this normalization function
def normalize_v2(input_array):
    mean_val, std_val = np.nanmean(input_array), np.nanstd(input_array)
    if std_val != 0 and mean_val != 0:
        input_array /= mean_val
    else:
        input_array = np.nan
    return input_array


def get_samples(triton_results_path):
    """
    This function retrieves sample names and their corresponding paths from a specified directory. 
    The function iterates over all files in the given directory, splits the filename on the string 
    "_TritonProfiles.npz", and if the split results in two parts, it assumes the first part is a 
    sample name and adds it to the samples list. The full path to the file is also added to the 
    paths list.

    Args:
        triton_results_path (str): The path to the directory containing the files.
        
    Calls:

    Returns:
        tuple: A tuple containing two lists. The first list contains the sample names, and the 
        second list contains the corresponding paths to the files.
    """
    samples = []
    paths = []
    for p in os.listdir(triton_results_path):
        s = p.split("_TritonProfiles.npz")
        if len(s)==2:
            samples.append(s[0])
            paths.append(os.path.join(triton_results_path, p))
    return samples, paths

import os
import logging

def make_sample_folders(output_directory, samples):
    """
    Create folders for each sample in the output directory.

    Args:
        output_directory (str): The path to the output directory where the sample folders will be created.
        samples (str or list): The samples for which the folders will be created. If a string is provided, 
            it is assumed to be a file path containing a list of samples.
    
    Calls:
        get_samples

    Returns:
        None
    """
    if isinstance(samples, str):
        samples = get_samples(samples)
    assert isinstance(samples, list)
    for sample in samples:
        new_sample_folder = os.path.join(output_directory, sample)
        try:
            os.makedirs(new_sample_folder)
        except Exception:
            logging.warning(f"Directory exists for sample {sample}")

def save_as_table(processed_data, output):
    """
    Save processed data as tables in CSV format.

    Args:
        processed_data (dict): A dictionary containing processed data.
        output (str): The output directory where the CSV files will be saved.
    
    Calls:

    Returns:
        None
    """
    image_vars = ["depth", "probable-nucleosome-center-profile", "phased-nucleosome-profile", "fragment-short-long-ratio", "fragment-diversity", \
              "fragment-entropy", "peak-location", "A-freq", "C-freq", "G-freq", "T-freq"]
    for k in processed_data:
        df = pd.DataFrame(processed_data[k])
        df = df.T
        df.index = image_vars
        df.to_csv(os.path.join(output, f"{k}.csv"))
        
def process_sample_path_pair(sample, path, out_dir):
    """
    Process a sample and its corresponding path.
    Also, save the counts information to a JSON file.

    Args:
        sample (str): The identifier of the sample.
        path (str): The path to the NPZ file.
        out_dir (str): The output directory.

    Calls:
        read_npz
        process_npz
        save_as_table

    Returns:
        None
    """
    npz = read_npz(path)
    proc_npz = process_npz(npz, identifier=sample)
    with open(out_dir + 'counts_images.json', 'a') as f:
        f.write(f"\"{sample}\": {list(proc_npz.keys())},\n")
    save_as_table(proc_npz, os.path.join(out_dir, sample))
    logging.info(f"Saved tables for sample {sample}.")

def generate(output_dir, data_dir):
    """
    Main generate images function. Generate images from the given data directory and save them to 
    the output directory.

    Args:
        output_dir (str): The directory where the generated images will be saved.
        data_dir (str): The directory containing the input data.
        
    Calls:
        get_samples
        make_sample_folders
        process_sample_path_pair

    Returns:
        None
    """
    samples, paths = get_samples(data_dir)
    make_sample_folders(output_dir, samples)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_sample_path_pair, sample, path, output_dir) for sample, path in zip(samples, paths)]
        concurrent.futures.wait(futures)
    logging.info(f"Complete")