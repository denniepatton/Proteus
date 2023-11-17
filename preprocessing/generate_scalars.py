import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
This script generates scalar tables from the Triton output stored in data_dir, using the reference file at reference_path. 
The generated tables are saved as CSV to output_dir.

Main Function
    - generate(output_dir, data_dir, reference_path)

"""

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
        """
    Opens gene metadata from a given file path and performs data preprocessing.

    Args:
        path (str): The file path to the gene metadata file.

    Calls:

    Returns:
        tuple: A tuple containing two dictionaries:
            - exon_count_info: A dictionary mapping gene names to preprocessed exon count values.
            - transcript_length_info: A dictionary mapping gene names to preprocessed transcript length values.
    """
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
    """
    Generate scalar tables for given samples.

    Args:
        scalar_data (pd.DataFrame): The input scalar data.
        output_dir (str): The directory to save the generated tables.
        meta_path (str): The path to the gene metadata file.
        samples (list, optional): The list of samples to generate tables for. If not provided, tables will be generated for all samples in the scalar data.

    Calls:
        open_gene_metadata
        get_samples

    Returns:
        None
    """
    exon_count_info, transcript_length_info = open_gene_metadata(meta_path)
    if samples is None:
        samples = get_samples(scalar_data)
    for s in samples:
        filtered_for_sample = scalar_data.loc[scalar_data["sample"] == s].drop(columns=["sample"]).set_index("site")
        # Drop the "central" features for scalars
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
    """
    Main generate scalars function. Generate scalar tables for data from Triton output stored at data_dir, referenced 
    from reference_path. Writes output scalars as CSV files to output_dir.

    Args:
        output_dir (str): The directory to save the generated scalar tables.
        data_dir (str): The directory containing the input data.
        reference_path (str): The path to the reference file.

    Calls:
        get_scalar_data
        generate_scalar_tables_for_samples

    Returns:
        None
    """
    # TODO: parallelize with available cores
    scalar_data = get_scalar_data(os.path.join(data_dir, "TritonCompositeFM.tsv"))
    generate_scalar_tables_for_samples(scalar_data, output_dir, reference_path)
    logging.info("Finished generating scalar tables.")
