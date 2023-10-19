# from sklearn.model_selection import train_test_split
import os
import logging
import json
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

### LOGGING CONFIGURATION
logging.basicConfig(filename="train_test_split.log", filemode='w', level=logging.INFO)
### ---------------------

def train_test_stratified_split(dataframe, ratio=[0.7, 0.1, 0.2], seed=42):
    shuffler = random.Random(seed)
    assert sum(ratio)==1
    assert "samples" in dataframe.columns
    assert "genes" in dataframe.columns
    assert type(seed) is int
    
    total_gene_instances = len(dataframe)
    
    train_set = []
    val_set = []
    test_set = []
    
    train_size = 0
    val_size = 0
    test_size = 0
    
    max_train_size = 0
    max_val_size = 0
    max_test_size = 0
    
    gene_counts = dict(dataframe.groupby("samples").count()["genes"])
    # first add ARPC samples to all sets
    arpc_samples = list(dataframe.loc[dataframe["phenotype"]!="NEPC"]["samples"].unique())
    shuffler.shuffle(arpc_samples)
    total_arpc = sum([gene_counts[s] for s in arpc_samples])
    max_train_size += ratio[0]*total_arpc
    max_val_size += ratio[1]*total_arpc
    max_test_size += ratio[2]*total_arpc
    for arpc in arpc_samples:
        if train_size < max_train_size:
            train_set.append(arpc)
            train_size += gene_counts[arpc]
        elif val_size < max_val_size:
            val_set.append(arpc)
            val_size += gene_counts[arpc]
        elif test_size < max_test_size:
            test_set.append(arpc)
            test_size += gene_counts[arpc]
    # next add NEPC samples to all sets
    nepc_samples = list(dataframe.loc[dataframe["phenotype"]=="NEPC"]["samples"].unique())
    shuffler.shuffle(arpc_samples)
    total_nepc = sum([gene_counts[s] for s in nepc_samples])
    max_train_size += ratio[0]*total_nepc
    max_val_size += ratio[1]*total_nepc
    max_test_size += ratio[2]*total_nepc
    for nepc in nepc_samples:
        if train_size < max_train_size:
            train_set.append(nepc)
            train_size += gene_counts[nepc]
        elif val_size < max_val_size:
            val_set.append(nepc)
            val_size += gene_counts[nepc]
        elif test_size < max_test_size:
            test_set.append(nepc)
            test_size += gene_counts[nepc]
    return train_set, val_set, test_set

def plot_split(train, val, test, data):
    def add_subset_info(x):
        if x in train:
            return "train"
        elif x in val:
            return "val"
        elif x in test:
            return "test"
        else:
            return "Omit"
    data["subset"] = data["samples"].apply(add_subset_info)
    fig, ax = plt.subplots(2, figsize = (10, 15))
    grouped_df = data.groupby(['subset', 'phenotype'])['genes'].count().reset_index()
    sns.barplot(x='phenotype', y='genes', hue='subset', data=grouped_df, ax=ax[0])
    grouped_df = data.groupby(['subset', 'phenotype'])['samples'].nunique().reset_index()
    sns.barplot(x='phenotype', y='samples', hue='subset', data=grouped_df, ax=ax[1])
    plt.show()
    plt.close()
    return fig
    

def main():
    OUTPUT = "/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/continuous/"
    SEED = 206
    data = np.load("/fh/fast/ha_g/user/rpatton/TritonCNN/cfDNA-gene-exp-inference/data/continuous/compiled_data.npz")
    samples = data["samples"]
    genes = data["genes"]
    data_df = pd.DataFrame({"samples": samples, "genes": genes})
    sample_info = pd.read_csv("/fh/fast/ha_g/user/rpatton/references/LuCaP/LuCaP_AllVars.tsv", sep="\t")[["sample", "phenotype"]]
    phenotype = {}
    for row in sample_info.iterrows():
        sample = row[1][0]
        # splits = sample.split("_")
        # sample_reformat= f"{splits[1]}_{splits[0]}"
        # phenotype[sample_reformat]=row[1][1]
        phenotype[sample]=row[1][1]
    data_df["phenotype"] = data_df["samples"].map(phenotype)

    training, validation, testing = train_test_stratified_split(data_df, seed=SEED)
    fig = plot_split(training, validation, testing, data_df)
    fig.savefig(os.path.join(OUTPUT, f"splits_info_seed{SEED}.pdf"))
    with open(os.path.join(OUTPUT, f"splits_seed{SEED}.csv"), 'w') as f:
        f.write("Training," + ",".join(training))
        f.write("\n")
        f.write("Validation," + ",".join(validation))
        f.write("\n")
        f.write("Testing," + ",".join(testing))
    logging.info("Complete.")


if __name__ == "__main__":
    main()