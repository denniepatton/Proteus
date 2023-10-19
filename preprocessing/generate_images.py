import pandas as pd
import numpy as np
import os
import logging
import multiprocessing
import concurrent.futures

### LOGGING CONFIGURATION
logging.basicConfig(filename="generate_images.log", filemode='w', level=logging.INFO)
### ---------------------

def read_npz(filepath):
    npz = np.load(filepath)
    return npz

def process_npz(input_object, identifier = "", tss_window_size = 2000, min_seq_depth = 10):
    """
    Cleans a npz input

    Args:
        input_object (_type_): _description_
        identifier (_type_): _description_
        tss_window_size (int, optional): _description_. Defaults to 2000.
        min_seq_depth (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
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
    samples = []
    paths = []
    for p in os.listdir(triton_results_path):
        s = p.split("_TritonProfiles.npz")
        if len(s)==2:
            samples.append(s[0])
            paths.append(os.path.join(triton_results_path, p))
    return samples, paths

def make_sample_folders(output_directory, samples):
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
    image_vars = ["depth", "probable-nucleosome-center-profile", "phased-nucleosome-profile", "fragment-short-long-ratio", "fragment-diversity", \
              "fragment-entropy", "peak-location", "A-freq", "C-freq", "G-freq", "T-freq"]
    for k in processed_data:
        df = pd.DataFrame(processed_data[k])
        df = df.T
        df.index = image_vars
        df.to_csv(os.path.join(output, f"{k}.csv"))
        
def process_sample_path_pair(sample, path, out_dir):
    npz = read_npz(path)
    proc_npz = process_npz(npz, identifier=sample)
    with open(out_dir + 'counts_images.json', 'a') as f:
        f.write(f"\"{sample}\": {list(proc_npz.keys())},\n")
    save_as_table(proc_npz, os.path.join(out_dir, sample))
    logging.info(f"Saved tables for sample {sample}.")

def generate(output_dir, data_dir):
    samples, paths = get_samples(data_dir)
    make_sample_folders(output_dir, samples)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_sample_path_pair, sample, path, output_dir) for sample, path in zip(samples, paths)]
        concurrent.futures.wait(futures)
    logging.info(f"Complete")

def main():
    pass

if __name__ == "__main__":
    main()

