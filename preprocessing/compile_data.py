import os
import pandas as pd
import tensorflow as tf
import pickle
import logging
from multiprocessing import Pool
import json
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor

### LOGGING CONFIGURATION
# logging.basicConfig(filename="convert_to_tensors.log", filemode='w', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
### ---------------------
def get_sample_and_gene(filename):
    filename = filename.replace(".csv", "")
    s = filename.split("_")
    if len(s) == 3:
        return s[1], s[2]
    else:
        raise AttributeError(f"Couldn't parse {filename} into sample and gene")
    
def calculate_genes_across_samples_images(path_to_images):
    """
    Generates a dictionary mapping each sample to a list of 
    genes with imaging data available for that sample
    """
    out = {}
    # samples = os.listdir(path_to_images)
    for sample in next(os.walk(path_to_images))[1]:
        out[sample] = [get_sample_and_gene(f)[1] for f in os.listdir(os.path.join(path_to_images, sample))]
    return out

def calculate_genes_across_samples_scalars(path_to_scalars):
    """
    Generates a dictionary mapping each sample to a list of 
    genes with scalar data available for that sample
    """
    out = {}
    samples = os.listdir(path_to_scalars)
    for sample in samples:
        df = pd.read_csv(os.path.join(path_to_scalars, sample))
        out[sample.replace(".csv", "")] = list(set(df["site"]))
    return out

def calculate_genes_across_samples_labels(path_to_labels):
    """
    Generates a dictionary mapping each sample to a list of 
    genes with label data available for that sample
    """
    out = {}
    y = pd.read_csv(path_to_labels, index_col=0)
    y_genes = list(y.index)
    y_samples = list(y.columns)
    for sample in y_samples:
        sample_label = "LuCaP_" + sample.split("_")[0]
        out[sample_label] = y_genes
    return out

def create_gene_json(path_to_images, path_to_scalars, path_to_labels, output):
    # images
    i = calculate_genes_across_samples_images(path_to_images)
    with open(os.path.join(output, "genes_images.json"), 'w') as f:
        f.write(json.dumps(i))
    # scalars
    s = calculate_genes_across_samples_scalars(path_to_scalars)
    with open(os.path.join(output, "genes_scalars.json"), 'w') as f:
        f.write(json.dumps(s))
    # labels
    l = calculate_genes_across_samples_labels(path_to_labels)
    with open(os.path.join(output, "genes_labels.json"), 'w') as f:
        f.write(json.dumps(l))
    return i, s, l

def process_gene(gene, sample, images_path, scalars_path, y):
    logging.info(f"Processing gene {gene} for sample {sample}")
    # load images
    image = np.array(pd.read_csv(os.path.join(images_path, sample, f"{sample}_{gene}.csv"), index_col=0), dtype=np.float32).T
    
    # load scalar
    scalar = np.array(pd.read_csv(os.path.join(scalars_path, f"{sample}.csv"), index_col=0).loc[gene])
    
    # load label
    try:
        label = y.at[gene, sample]
    except KeyError:
        logging.warning(f"Couldn't find label for {sample}, gene: {gene}")
        print(gene, sample)
        return None
    
    # check for missing values
    if np.isnan(image).any() or np.isnan(scalar).any() or np.isnan(label):
        logging.warning(f"NAs found for {sample}, gene: {gene}")
        return None
    else:
        return (f"{sample}_{gene}", sample, gene, image, scalar, label)

def process_sample(sample, combined, images_path, scalars_path, y):
    logging.info(f"Starting sample {sample}")
    genes = combined[sample]
    n_genes=len(genes)
    logging.info(f"Processing {n_genes} genes")
    
    ids_sample = []
    samples_sample = []
    genes_sample = []
    i_sample = []
    s_sample = []
    l_sample = []
    
    # You can specify the number of threads based on the number of available cores
    num_threads = 64

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Use the executor to process genes in parallel
        results = list(executor.map(lambda gene: process_gene(gene, sample, images_path, scalars_path, y), genes))
    
    results = [result for result in results if result is not None]
    ids_sample, samples_sample, genes_sample, i_sample, s_sample, l_sample = zip(*results)
    logging.info(f"Completed sample {sample}, processed {n_genes} genes")
    return ids_sample, samples_sample, genes_sample, i_sample, s_sample, l_sample



def save_data(combined, images_path, scalars_path, labels_path, output):
    ids = []
    samples = []
    genes = []
    i = []
    s = []
    l = []
    y = pd.read_csv(labels_path, index_col=0)
    cols = y.columns
    renamed = []
    for col in cols:
        sample_label = "LuCaP_" + col.split("_")[0]
        renamed.append(sample_label)
    y.columns = renamed

    process_sample_partial = partial(
        process_sample,
        images_path=images_path,
        scalars_path=scalars_path,
        combined=combined,
        y=y
    )

    with Pool(processes=50) as pool:
        results = pool.map(process_sample_partial, combined.keys())


    for ids_sample, samples_sample, genes_sample, i_sample, s_sample, l_sample in results:
        ids.extend(ids_sample)
        samples.extend(samples_sample)
        genes.extend(genes_sample)
        i.extend(i_sample)
        s.extend(s_sample)
        l.extend(l_sample)

    ids = np.array(ids)
    samples = np.array(samples)
    genes = np.array(genes)
    i = np.array(i)
    s = np.array(s)
    l = np.array(l)
    np.savez(os.path.join(output, "compiled_data.npz"), 
            ids=ids, samples=samples, genes=genes,
            images=i, scalars=s, labels=l)
    

def compile(images_path, scalars_path, labels_path, output):
     logging.basicConfig(filename="compile_data.log", filemode='w', level=logging.INFO, force=True,
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    # calculate sample to gene dictionarys for images, scalars, labels
    i, s, l = create_gene_json(images_path, scalars_path, labels_path, output)
    # remove samples not found in all three modalities
    combined = {}
    for sample in i:
        if sample in s and sample in l:
            i_genes, s_genes, l_genes = set(i[sample]), set(s[sample]), set(l[sample])
            common_genes = list(i_genes.intersection(s_genes, l_genes))
            combined[sample] = common_genes
    # remove samples with less than 50 genes available
    combined_filtered = {}
    for sample in combined:
        if len(combined[sample]) > 50:
            combined_filtered[sample] = combined[sample]
        else:
            logging.info(f"removing sample {sample}...")
    save_data(combined_filtered, images_path, scalars_path, labels_path, output) 

    logging.info("Complete")

def main():
    pass
"""     logging.basicConfig(filename="compile_data.log", filemode='w', level=logging.INFO, force=True,
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    # calculate sample to gene dictionarys for images, scalars, labels
    IMAGES_PATH = '/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/images'
    SCALARS_PATH = '/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/scalars'
    LABELS_PATH = '/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/labels/labels_continuous_tpm.csv'
    OUTPUT = '/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/continuous'
    i, s, l = create_gene_json(IMAGES_PATH, SCALARS_PATH, LABELS_PATH, OUTPUT )
    # remove samples not found in all three modalities
    combined = {}
    for sample in i:
        if sample in s and sample in l:
            i_genes, s_genes, l_genes = set(i[sample]), set(s[sample]), set(l[sample])
            common_genes = list(i_genes.intersection(s_genes, l_genes))
            combined[sample] = common_genes
    # remove samples with less than 50 genes available
    combined_filtered = {}
    for sample in combined:
        if len(combined[sample]) > 50:
            combined_filtered[sample] = combined[sample]
        else:
            logging.info(f"removing sample {sample}...")
    save_data(combined_filtered, IMAGES_PATH, SCALARS_PATH, LABELS_PATH, OUTPUT) """


    # USE_SMALL_DATA = False
    # REGRESSION = False
    # if USE_SMALL_DATA:
    #     logging.basicConfig(filename="compile_data_small.log", filemode='w', level=logging.INFO)
    # else:
    #     ### LOGGING CONFIGURATION
    #     logging.basicConfig(filename="compile_data.log", filemode='w', level=logging.INFO, 
    #                         format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    #     ### ---------------------
    # logging.info("Compiling data...")
    # splits = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/splits.csv'
    # images = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/images'
    # scalars = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/scalars'
    # if REGRESSION:
    #     labels = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/labels/labels_continuous_AvgL2fpkm.csv'
    #     output = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/regression'
    # else:
    #     labels = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/labels/labels_three_class_fpkm.csv'
    #     output = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/classification'
    # gene_counts = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/counts.json'
    # training, testing = parse_splits(splits)
    # with open(gene_counts) as f:
    #     gene_counts = json.load(f)
    # logging.info("Parsed splits.")
    # y = pd.read_csv(labels, index_col=0)
    # y_genes = list(y.index)
    # y_samples = list(y.columns)
    # logging.info("Loaded labels")
    # # training
    # if USE_SMALL_DATA:
    #     training = [t for t in training if gene_counts[t] < 400]
    #     testing = [t for t in testing if gene_counts[t] < 400]
    #     output = '/fh/scratch/delete90/ha_g/users/anair3/repositories/cfDNA-gene-exp-inference/data/continuous/data_small'
        
    # image_data_training = load_images(images, training)
    # logging.info("Loaded training images")
    # scalar_data_training = load_scalars(scalars, training)
    # logging.info("Loaded training scalars")
    # image_keys_training = list(image_data_training.keys())
    # scalar_keys_training = list(scalar_data_training.keys())
    # ids = []
    # samples = []
    # genes = []
    # i = []
    # s = []
    # l = []
    # for g in image_keys_training:
    #     if g in scalar_keys_training:
    #         s_name, g_name = get_sample_and_gene(g)
    #         if s_name in y_samples and g_name in y_genes:
    #             i_data = image_data_training[g]
    #             s_data = scalar_data_training[g]
    #             l_data = y.at[g_name, s_name]
    #             if np.isnan(i_data).any() or np.isnan(s_data).any() or pd.isna(l_data):
    #                 logging.warning(f"Sample {s_name} with gene {g_name} has NA values...skipping entry")
    #                 continue
    #             else:
    #                 ids.append(f"{s_name}_{g_name}")
    #                 samples.append(s_name)
    #                 genes.append(g_name)
    #                 i.append(i_data)
    #                 s.append(s_data)
    #                 l.append(l_data)
    # id_train = np.array(ids)
    # samples_train = np.array(samples)
    # genes_train = np.array(genes)
    # i_train = np.array(i)
    # s_train = np.array(s)
    # l_train = np.array(l)
    # logging.info("Converted training samples to np arrays")
    # np.savez(os.path.join(output, "training.npz"), 
    #         ids=id_train, samples=samples_train, genes=genes_train,
    #         images=i_train, scalars=s_train, labels=l_train)
    
    # # testing
    # image_data_testing = load_images(images, testing)
    # logging.info("Loaded testing images")
    # scalar_data_testing = load_scalars(scalars, testing)
    # logging.info("Loaded testing scalars")
    # image_keys_testing = list(image_data_testing.keys())
    # scalar_keys_testing = list(scalar_data_testing.keys())
    # ids = []
    # samples = []
    # genes = []
    # i = []
    # s = []
    # l = []
    # for g in image_keys_testing:
    #     if g in scalar_keys_testing:
    #         s_name, g_name = get_sample_and_gene(g)
    #         if s_name in y_samples and g_name in y_genes:
    #             i_data = image_data_testing[g]
    #             s_data = scalar_data_testing[g]
    #             l_data = y.at[g_name, s_name]
    #             if np.isnan(i_data).any() or np.isnan(s_data).any() or pd.isna(l_data):
    #                 logging.warning(f"Sample {s_name} with gene {g_name} has NA values...skipping entry")
    #                 continue
    #             ids.append(f"{s_name}_{g_name}")
    #             samples.append(s_name)
    #             genes.append(g_name)
    #             i.append(i_data)
    #             s.append(s_data)
    #             l.append(l_data)
    # id_test = np.array(ids)
    # samples_test = np.array(samples)
    # genes_test = np.array(genes)
    # i_test = np.array(i)
    # s_test = np.array(s)
    # l_test = np.array(l)
    # logging.info("Converted testing samples to np arrays")
    # np.savez(os.path.join(output, "testing.npz"), 
    #         ids=id_test, samples=samples_test, genes=genes_test,
    #         images=i_test, scalars=s_test, labels=l_test)
    #logging.info("Complete")

if __name__ == "__main__":
    main()