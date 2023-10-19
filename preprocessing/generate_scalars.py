import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

### LOGGING CONFIGURATION
logging.basicConfig(filename="generate_scalars.log", level=logging.INFO)
### ---------------------

def get_scalar_data(scalar_data_path):
    all_scalar = pd.read_csv(scalar_data_path, sep = "\t")
    pivoted_df = all_scalar.pivot_table(index=["sample", "site"],
                                        columns="feature",
                                        values="value")
    all_scalar_proc = pivoted_df.reset_index()
    return all_scalar_proc

def get_samples(scalar_data):
    return list(scalar_data["sample"].unique())

def open_gene_metadata(path):
    gene_meta = pd.read_csv(path, sep='\t')[["name", "exon_count", "transcript_length"]]
    # TODO: log exon count first? See distirbution
    gene_meta['exon_count'] = np.log(gene_meta['exon_count'])
    min_value = gene_meta['exon_count'].min()
    max_value = gene_meta['exon_count'].max()
    gene_meta['exon_count'] = (gene_meta['exon_count'] - min_value) / (max_value - min_value)
    gene_meta['transcript_length'] = np.log(gene_meta['transcript_length'])
    # TODO: log transcript length first? See distirbution
    min_value = gene_meta['transcript_length'].min()
    max_value = gene_meta['transcript_length'].max()
    gene_meta['transcript_length'] = (gene_meta['transcript_length'] - min_value) / (max_value - min_value)
    exon_count_info = dict(zip(gene_meta["name"], gene_meta["exon_count"]))
    transcript_length_info = dict(zip(gene_meta["name"], gene_meta["transcript_length"]))
    return exon_count_info, transcript_length_info

def generate_scalar_tables_for_samples(scalar_data, output_dir, meta_path, samples=None):
    exon_count_info, transcript_length_info = open_gene_metadata(meta_path)
    if samples is None:
        samples = get_samples(scalar_data)
    for s in samples:
        filtered_for_sample = scalar_data.loc[scalar_data["sample"] == s].drop(columns=["sample"]).set_index("site")
                # Drop the "central" features for scaalars
        filtered_for_sample = filtered_for_sample.drop(columns=["central-diversity", "central-depth", "central-loc", "plus-minus-ratio", "minus-one-pos", "plus-one-pos"])
        # TODO: confirm scaling is what we want
        standardized = StandardScaler().fit_transform(filtered_for_sample)
        filtered_for_sample = pd.DataFrame(standardized, 
                                           index=filtered_for_sample.index, 
                                           columns=filtered_for_sample.columns)
        filtered_for_sample["exon_count"] = filtered_for_sample.index.map(exon_count_info)
        filtered_for_sample["transcript_length"] = filtered_for_sample.index.map(transcript_length_info)
        logging.info(f"Before {s} had NAs in {filtered_for_sample.columns[filtered_for_sample.isna().any()].tolist()}")
        logging.info(f"{filtered_for_sample.isna().sum()}")
        # Drop rows with NA values
        filtered_for_sample = filtered_for_sample.dropna()
        logging.info(f"Any NAs? {filtered_for_sample.isnull().values.any()}")
        # TODO switch to using a different output format (e.g. feather or pickle)
        filtered_for_sample.to_csv(os.path.join(output_dir, f"{s}.csv"))


def generate(output_dir, data_dir, reference_path):
    # TODO: parallelize with avaliable cores
    scalar_data = get_scalar_data(os.path.join(data_dir, "TritonCompositeFM.tsv"))
    generate_scalar_tables_for_samples(scalar_data, output_dir, reference_path)

def main():
    pass

if __name__ == "__main__":
    main()
