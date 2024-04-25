"""
run_analysis.py
v1.0.2, 2/1/2024
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script runs a complete analysis of PROTEUS and its outputs
"""

# Standard library imports
import os
import csv

# Third party imports
import numpy as np
import pandas as pd

# Local application imports
from analysis_utils import *


def main():
    # results_training folders to run LuCaP / SCLC analysis on
    validation_analyses = ['GeneStrat_0x',
                           'SampleStrat_0x',
                           'PC-HO_0x',
                           'SCLC-HO_0x']
    # patient validation sets to run:  [ID TFX Subtype] (no header)
    patient_analyses = {'UW-WGS': 'UW-WGS_TFX_Subtype.tsv',
                        'PSMA': 'PSMA_TFX_Subtype_Curated-WGS.tsv',
                        'BC-TAN': 'BC-TAN_TFX_Subtype.tsv',
                        'PC-TAN': 'PC-TAN_TFX_Subtype.tsv'}
    healthy_analyses = ['HD']
    # load reference gene lists
    gene_lists = {'all': None,
                  'CCP': ['CCP31.txt'],
                  'Targets': ['Targets.txt'],
                  'PAM50': ['PAM50.txt'],
                  'EMT': ['EMT.txt'],
                  'Hypoxia': ['Hypoxia.txt'],
                  'PC-Phenotyping': [[8, 14, 22], 'PC_PhenotypingGenes.txt'],
                  'BC-Phenotyping': ['DF-List', 'BC_MolecularSubtypes.txt'],
                  'PSMA': ['PSMA-high.txt', 'PSMA-low.txt'],
                  'LuCaP-vs-WholeBlood': ['LuCaPvsGTEx_up-regulated.txt', 'LuCaPvsGTEx_down-regulated.txt'],
                  'LuCaP-vs-PBMC': ['LuCaPvsPBMC_up-regulated.txt', 'LuCaPvsPBMC_down-regulated.txt'],
                  'SCLC-vs-WholeBlood': ['SCLCvsGTEx_up-regulated.txt', 'SCLCvsGTEx_down-regulated.txt'],
                  'SCLC-vs-PBMC': ['SCLCvsPBMC_up-regulated.txt', 'SCLCvsPBMC_down-regulated.txt'],
                  'LuCaP-vs-SCLC': ['LuCaPvsSCLC_up-regulated.txt', 'LuCaPvsSCLC_down-regulated.txt']}
    print('Loading gene lists...')
    for key, value in gene_lists.items():
        if value is None:
            continue
        elif isinstance(value[0], list):
            # Load the file as a list of strings
            file_content = load_file_as_list(os.path.join('../reference/', value[1]))
            # Define the ranges
            ranges = [0] + value[0] + [None]
            # Split the list into the desired ranges
            new_value = [file_content[start:end] for start, end in zip(ranges[:-1], ranges[1:])]
        elif value[0] == 'DF-List':
            # Read the file into a DataFrame
            df = pd.read_csv(os.path.join('../reference/', value[1]), sep='\t')  # adjust the separator if needed
            # Convert each column to a list and store them in a tuple
            new_value = tuple(df[col].tolist() for col in df.columns)
        else:
            # Load each file as a list of strings
            new_value = [load_file_as_list(os.path.join('../reference/', filename)) for filename in value]
        # Replace the original value with the new tuple
        gene_lists[key] = tuple(new_value)
    gene_lists['Targets_SCLC'] = (['ASCL1', 'NEUROD1', 'POU2F3', 'ATOH1', 'DLL3', 'SEZ6', 'NOTCH1', 'NOTCH3',
                                   'MYC', 'MGA', 'SYP', 'INSM1', 'PTEN', 'EGFR', 'RB1', 'TP53', 'PIK3CA', 'TACSTD2',
                                   'HES1', 'EZH2', 'FOS', 'JUN', 'KRAS', 'REST'],) # TACSTD2 = TROP2

    print('Loading reference data...')
    # load ground truth /reference data
    lucap_truth = pd.read_csv('../labels/LuCaP_Log2CPMplus1.tsv', sep='\t', index_col=0)
    lucap_meta = pd.read_csv('../reference/LuCaP_metadata.tsv', sep='\t', index_col=0).T
    sclc_truth = pd.read_csv('../labels/SCLC_Log2CPMplus1.tsv', sep='\t', index_col=0)
    sclc_meta = pd.read_csv('../reference/SCLC_metadata.tsv', sep='\t', index_col=0).T
    pbmc_ref = pd.read_csv('../labels/PBMC-Joined_Log2CPMplus1.tsv', sep='\t', index_col=0)
    gtex_ref = pd.read_csv('../labels/GTEx_Log2CPMplus1.tsv', sep='\t', index_col=0)
    pc_biopsies = pd.read_csv('../labels/PC_TAN_Log2CPMplus1.tsv', sep='\t', index_col=0)
    bc_biopsies = pd.read_csv('../labels/BC_TAN_Log2CPMplus1.tsv', sep='\t', index_col=0)
    pctan_meta = pd.read_csv('../reference/PC-TAN_IHC-Scores.tsv', sep='\t', index_col=0).T
    # Restrict/rename TAN biopsies based on sample_aliases
    pc_biopsies = rename_tan(pc_biopsies)
    bc_biopsies = rename_tan(bc_biopsies)


    ############################################################################################################
    # Run training validation analyses
    ############################################################################################################
    print('Running training validation analyses.')
    for analysis in validation_analyses:
        """ For each use_set / training regime:
        1. Load the predictions for each fold, found in ../results_training/analysis/CV-fold_*/validation-fold_*_predictions.tsv
        2. Merge all folds into one prediction df for lucap and one for sclc, corresponding to the truth dfs
        3. Makes results/analysis/LuCaP, /SCLC, and /Combined folders if they don't exist and save the merged predictions there
        4. Run the following analyses:
        ### Run combined analyses
        - Plot scatter of truth vs pred for all genes and all samples
        ### Run LuCaP-specific analyses:
        - Plot combined scatter of truth vs pred for all samples and all genes, as well as  for [PC-Phenotyping, LuCaP-vs-WholeBlood, LuCaP-vs-PBMC]
        - Plot two-figure heatmaps for truth and pred for [CCP, Targets, PAM50, EMT, Hypoxia, PC-Phenotyping, LuCaP-vs-WholeBlood, LuCaP-vs-PBMC]
        - Compute R^2 and MAE for each sample in both all genes, [PC-Phenotyping, LuCaP-vs-WholeBlood, LuCaP-vs-PBMC] and plot those
          metrics vs depth (from lucap_ref)
        - Plot sample-wise scatters of truth vs pred CCP-score vs CCP-mean, PSMA-score vs PSMA-mean, and PSMA-score vs FOLH1 (PSMA) prediction 
        ### Run SCLC-specific analyses
        - Plot combined scatter of truth vs pred for all samples and all genes, as well as  for [SCLC-vs-WholeBlood, SCLC-vs-PBMC]
        - Plot two-figure heatmaps for truth and pred for [CCP, Targets, PAM50, EMT, Hypoxia, SCLC-vs-WholeBlood, SCLC-vs-PBMC]
        """
        print(f'Running analyses for {analysis}')
        # Load predictions
        val_preds = [pd.read_csv(f'../results_training/BioCNN_{analysis}/CV-fold_{fold}/validation-fold_{fold}_predictions.tsv', sep='\t', index_col=0) for fold in range(1, 6)]
        samples = [name for name in os.listdir(f'../results_training/BioCNN_{analysis}') if "CV" not in name and os.path.isdir(os.path.join(f'../results_training/BioCNN_{analysis}', name))]
        ho_preds = {sample: pd.read_csv(f'../results_training/BioCNN_{analysis}/{sample}/validation-{sample}_predictions.tsv', sep='\t', index_col=0) for sample in samples}
        df_ho = pd.concat([ho_preds[sample]['prediction'].rename(sample) for sample in ho_preds], axis=1)
        merged_truth = lucap_truth.merge(sclc_truth, how='outer', left_index=True, right_index=True)
        os.makedirs(f'results/{analysis}', exist_ok=True)
        os.makedirs(f'results/{analysis}/LuCaP', exist_ok=True)
        os.makedirs(f'results/{analysis}/SCLC', exist_ok=True)
        os.makedirs(f'results/{analysis}/Combined', exist_ok=True)
        file_path = f'results/{analysis}/Combined/predictions.tsv'
        if os.path.exists(file_path):
            df_val = pd.read_csv(file_path, sep='\t', index_col=0)
        else:
            df_val = merge_validation(val_preds)
            df_val.to_csv(file_path, sep='\t')
        # Plot scatter of truth vs pred for all genes and all samples
        scatter_plot(merged_truth, df_val, name=f'results/{analysis}/Combined/true-vs-pred_val_scatter')
        scatter_plot(merged_truth, df_ho, name=f'results/{analysis}/Combined/true-vs-pred_ho_scatter')
        df_val = pd.concat([df_val, df_ho], axis=1)
        scatter_plot(merged_truth, df_val, name=f'results/{analysis}/Combined/true-vs-pred_scatter')
        ### LuCaP-specific analyses
        common_lucap = df_val.columns.intersection(lucap_truth.columns)
        lucap_val = df_val.loc[:, common_lucap]
        lucap_truth = lucap_truth.loc[:, common_lucap]
        lucap_subtypes = lucap_meta.loc['phenotype', common_lucap]
        lucap_pam50 = lucap_meta.loc['PAM50', common_lucap]
        for gene_set in ['all', 'PC-Phenotyping', 'LuCaP-vs-WholeBlood', 'LuCaP-vs-PBMC', 'LuCaP-vs-SCLC']:
            scatter_plot(lucap_truth, lucap_val, genes_subset=gene_lists[gene_set], name=f'results/{analysis}/LuCaP/{gene_set}_true-vs-pred_scatter')
        for gene_set in ['CCP', 'Targets', 'PAM50', 'PC-Phenotyping', 'LuCaP-vs-SCLC']:
            plot_double_heatmap(gene_lists[gene_set], lucap_truth, lucap_val, name=f'results/{analysis}/LuCaP/{gene_set}_true-vs-pred_heatmap', subtypes=lucap_subtypes)
        for gene_set in ['all', 'PC-Phenotyping', 'LuCaP-vs-WholeBlood', 'LuCaP-vs-PBMC', 'LuCaP-vs-SCLC']:
            r2_mae_plot(lucap_truth, lucap_val, lucap_meta, gene_lists[gene_set], name=f'results/{analysis}/LuCaP/{gene_set}_r2_mae_plot')
        # Scatter plot FOLH1 expression vs PSMA-mean from lucap_meta, for both truth and pred
        plot_vs_meta(lucap_meta, lucap_val, 'PSMA-mean', 'FOLH1', name=f'results/{analysis}/LuCaP/FOLH1-pred-vs-PSMA-mean_scatter', subtypes=lucap_subtypes)
        plot_vs_meta(lucap_meta, lucap_truth, 'PSMA-mean', 'FOLH1', name=f'results/{analysis}/LuCaP/FOLH1-truth-vs-PSMA-mean_scatter', subtypes=lucap_subtypes)
        for gene in gene_lists['Targets'][0]:
            plot_vs_meta(lucap_truth, lucap_val, gene, gene, name=f'results/{analysis}/LuCaP/{gene}_true-vs-pred_scatter', subtypes=lucap_subtypes)
        # Load GSVA and PAM50 scores for truth and pred (LuCaP ONLY)
        if analysis == 'GeneStrat_0x':
            gsva_pred = pd.read_csv('GeneScores/Training-Predictions_gsva.sig.es.obs.txt', sep='\t', index_col=0)
            gsva_truth = pd.read_csv('GeneScores/Training-Truth_gsva.sig.es.obs.txt', sep='\t', index_col=0)
            pam50_pred = pd.read_csv('GeneScores/Training-Predictions_pam50scores.txt', sep='\t', index_col=0)
            pam50_truth = pd.read_csv('GeneScores/Training-Truth_pam50scores.txt', sep='\t', index_col=0)
            # transpose the pam50 scores
            pam50_pred = pam50_pred.T
            pam50_truth = pam50_truth.T
            # Select only the 'Basal', 'LumA', and 'LumB' rows
            pam50_pred = pam50_pred.loc[['Basal', 'LumA', 'LumB'], :]
            pam50_truth = pam50_truth.loc[['Basal', 'LumA', 'LumB'], :]
            # Find samples with 'LuCaP_' in the name
            gsva_samples_pred = [sample for sample in gsva_pred.columns if 'LuCaP_' in sample]
            gsva_samples_truth = [sample for sample in gsva_truth.columns if 'LuCaP_' in sample]
            pam50_samples_pred = [sample for sample in pam50_pred.columns if 'LuCaP_' in sample]
            pam50_samples_truth = [sample for sample in pam50_truth.columns if 'LuCaP_' in sample]
            # Find common samples
            common_samples = list(set(gsva_samples_pred).intersection(gsva_samples_truth, pam50_samples_pred, pam50_samples_truth))
            # Filter data frames based on common samples
            rows_to_keep = ['ARG.10', 'NE.10', 'EMT.12', 'EMT', 'PSMA-high', 'PSMA-low', 'UC', 'PUC', 'NE', 'SQ']
            gsva_pred = gsva_pred.loc[rows_to_keep, common_samples]
            gsva_truth = gsva_truth.loc[rows_to_keep, common_samples]
            pam50_pred = pam50_pred.loc[:, common_samples]
            pam50_truth = pam50_truth.loc[:, common_samples]
            # plot scores as scatter plots
            plot_vs_meta(pam50_truth, pam50_pred, 'Basal', 'Basal', name=f'results/{analysis}/Basal_scatter', subtypes=lucap_subtypes)
            plot_vs_meta(pam50_truth, pam50_pred, 'LumA', 'LumA', name=f'results/{analysis}/LumA_scatter', subtypes=lucap_subtypes)
            plot_vs_meta(pam50_truth, pam50_pred, 'LumB', 'LumB', name=f'results/{analysis}/LumB_scatter', subtypes=lucap_subtypes)
            # make a dictionary of lucap subtypes, mapping every sample in common_samples to 1.0
            lucap_tfx = {sample: 1.0 for sample in common_samples}
            plot_heatmap(gsva_pred.index.tolist(), gsva_pred, lucap_tfx, name=f'results/{analysis}/GSVA_scores_predicted', gsea=True, sub_dict=lucap_subtypes.to_dict())
            plot_heatmap(gsva_truth.index.tolist(), gsva_truth, lucap_tfx, name=f'results/{analysis}/GSVA_scores_truth', gsea=True, sub_dict=lucap_subtypes.to_dict())
            plot_heatmap(pam50_pred.index.tolist(), pam50_pred, lucap_tfx, name=f'results/{analysis}/PAM50_scores_predicted', gsea=True, sub_dict=lucap_pam50.to_dict())
            plot_heatmap(pam50_truth.index.tolist(), pam50_truth, lucap_tfx, name=f'results/{analysis}/PAM50_scores_truth', gsea=True, sub_dict=lucap_pam50.to_dict())
        ### SCLC-specific analyses
        common_sclc = df_val.columns.intersection(sclc_truth.columns)
        sclc_val = df_val.loc[:, common_sclc]
        sclc_truth = sclc_truth.loc[:, common_sclc]
        sclc_subtypes = sclc_meta.loc['subtype', common_sclc]
        for gene_set in ['all', 'SCLC-vs-WholeBlood', 'SCLC-vs-PBMC', 'LuCaP-vs-SCLC', 'Targets_SCLC']:
            scatter_plot(sclc_truth, sclc_val, genes_subset=gene_lists[gene_set], name=f'results/{analysis}/SCLC/{gene_set}_true-vs-pred_scatter')
        for gene_set in ['CCP', 'Targets', 'PAM50', 'EMT', 'LuCaP-vs-SCLC', 'Targets_SCLC']:
            plot_double_heatmap(gene_lists[gene_set], sclc_truth, sclc_val, name=f'results/{analysis}/SCLC/{gene_set}_true-vs-pred_heatmap', subtypes=sclc_subtypes)
        for gene in gene_lists['Targets_SCLC'][0]:
            plot_vs_meta(sclc_truth, sclc_val, gene, gene, name=f'results/{analysis}/SCLC/{gene}_true-vs-pred_scatter', subtypes=sclc_subtypes)

    quit()


    ############################################################################################################
    # Run patient validation analyses
    ############################################################################################################
    # TODO: Add functionality to create a pi-chart of estimated tissues-of-origin (including 0s), to be in-set into TAN analyses
    # TODO: see if TMM-re-scaling helps or hurts.
    for analysis in patient_analyses:
        """ For each patient validation set:
        1. Load the predictions, found in ../results_holdout/analysis_data/predictions.tsv
        2. Make results/analysis folder if it doesn't exist for outputs
        3. Run the following analyses:
        - Plot heatmaps for [CCP, Targets, PAM50, PC-Phenotyping, PSMA] (+ TFX bars)
        - Plot GSEA scores as (identically sorted) heatmaps (+ TFX bars)
        - Save GSEA scores as a TSV file
        ### if analysis == 'UW-WGS':
        - Trying both fit-to-biopsy  and biopsy-avg:
          plot scatter of truth vs pred for matching biopsy sets on all genes as well as [Targets, PC-Phenotyping, LuCaP-vs-WholeBlood, LuCaP-vs-PBMC]
          (and also print TFX on the scatter)
        """
        # Load data
        preds_df = pd.read_csv(f'../results_holdout/penultimate_MSE/{analysis}_data/predictions.tsv', sep='\t', index_col=0)
        # Load TFX, Subtypes
        with open(f'../reference/{patient_analyses[analysis]}', 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            rows = list(reader)  # Store the rows in a list
            tfx_dict = {row[0]: float(row[1]) for row in rows}
            subtype_dict = {row[0]: row[2] for row in rows}
        os.makedirs(f'results/{analysis}', exist_ok=True)

        ### Re-normalize and deconvolve the predictions at the sample level based on TFX ###
        """ No renormalization / subtraction process of any kind so far has yeilded better results than the raw predictions
        """
        # preds_df = subtract_background_and_rescale(preds_df, gtex_ref, tfx_dict, ref_tissue='Prostate')
        # preds_df.to_csv(f'results/{analysis}_deconvolved-predictions.tsv', sep='\t')
        # preds_df, _, dcw = perform_reference_deconvolution(preds_df, tfx_dict, gtex_ref, tissues=['Artery_-_Aorta', 'Artery_-_Coronary', 'Artery_-_Tibial', 'Liver', 'Whole_Blood'])
        # print(dcw)
        
        plot_samples(preds_df, f'results/{analysis}/sample-expression-profiles')
        # load GSEA scores
        gsea_scores = pd.read_csv(f'GeneScores/{analysis}_gsva.sig.es.obs.txt', sep='\t', index_col=0)
        # Perform cohort-specific analyses:
        if analysis == 'UW-WGS':
            # Get order from paper
            order = load_file_as_list('UW-WGS_order.txt')
            # Drop samples with "CRPC" in their name
            preds_df = preds_df.drop(columns=[col for col in preds_df.columns if 'CRPC' in col])
            gsea_scores = gsea_scores.drop(index=[idx for idx in gsea_scores.index if 'CRPC' in idx])
            # Define the new order of the indices
            new_order = ['ARG.6', 'ARG.10', 'NE.6', 'NE.10', 'PSMA-high', 'PSMA-low',
                        'CCP.31', 'EMT.12', 'EMT',
                        'Hypoxia', 'Buffa_2010_Hypoxia_51', 'Ragnum_2014_Hypoxia_32', 'Ragnum_201 4_Hypoxia_woProliferation_12', 'Eustace_West_2013_Hypoxia_26', 'Eustace_West_2013_Hypoxia_woProliferation_22', 'Yang_West_2018_Hypoxia_up_19', 'Yang_West_2018_Hypoxia_up_woProliferation_12']
            # Reorder the indices of gsea_scores
            gsea_scores = gsea_scores.reindex(new_order)
            plot_heatmap(gsea_scores.index.tolist(), gsea_scores, tfx_dict, name=f'results/{analysis}/GSEA_scores_heatmap', gsea=True, breaks=[6, 9], sub_dict=subtype_dict, sample_order=order)
            gsea_scores.to_csv(f'results/{analysis}/GSEA_scores.tsv', sep='\t')
            # Do GSEAs again with only AR/NE:
            new_order = ['ARG.6', 'NE.6']
            gsea_scores = gsea_scores.reindex(new_order)
            plot_heatmap(gsea_scores.index.tolist(), gsea_scores, tfx_dict, name=f'results/{analysis}/GSEA_scores_AR-NE_heatmap', gsea=True, sub_dict=subtype_dict, sample_order=order)
            # Use tfx_dict to subset gsea_scores to those samples with tfx > 0.2
            gsea_scores = gsea_scores.loc[:, [sample for sample, tfx in tfx_dict.items() if tfx > 0.1 and 'CRPC' not in sample]]


            # Calculate the 'call' row
            call_row = np.where(gsea_scores.loc['NE.6'] > gsea_scores.loc['ARG.6'], 1, -1)
            # Create a new DataFrame with 'call' as the first row
            new_gsea_scores = pd.DataFrame(call_row.reshape(1,-1), columns=gsea_scores.columns, index=['call'])
            # Append the rest of gsea_scores to it
            new_gsea_scores = pd.concat([new_gsea_scores, gsea_scores])
            # Ensure 'call' row is the first row
            gsea_scores = new_gsea_scores.loc[['call'] + [index for index in new_gsea_scores.index if index != 'call']]


            print(gsea_scores)
            plot_heatmap(gsea_scores.index.tolist(), gsea_scores, tfx_dict, name=f'results/{analysis}/GSEA_scores_AR-NE_heatmap_subset', gsea=True, sub_dict=subtype_dict)
            for gene_set in ['Targets', 'PC-Phenotyping']:
                # Plot gene set heatmaps
                plot_heatmap(gene_lists[gene_set], preds_df, tfx_dict, name=f'results/{analysis}/{gene_set}_base_heatmap', sub_dict=subtype_dict, sample_order=order)
        elif analysis == 'PSMA':
            # Define the new order of the indices
            new_order = ['ARG.6', 'ARG.10', 'NE.6', 'NE.10', 'PSMA-high', 'PSMA-low',
                        'CCP.31', 'EMT.12', 'EMT',
                        'Hypoxia', 'Buffa_2010_Hypoxia_51', 'Ragnum_2014_Hypoxia_32', 'Ragnum_201 4_Hypoxia_woProliferation_12', 'Eustace_West_2013_Hypoxia_26', 'Eustace_West_2013_Hypoxia_woProliferation_22', 'Yang_West_2018_Hypoxia_up_19', 'Yang_West_2018_Hypoxia_up_woProliferation_12']
            # Reorder the indices of gsea_scores
            gsea_scores = gsea_scores.reindex(new_order)
            plot_heatmap(gsea_scores.index.tolist(), gsea_scores, tfx_dict, name=f'results/{analysis}/GSEA_scores_heatmap', gsea=True, breaks=[6, 9], sub_dict=subtype_dict)
            gsea_scores.to_csv(f'results/{analysis}/GSEA_scores.tsv', sep='\t')
            for gene_set in ['Targets', 'PC-Phenotyping']:
                # Plot gene set heatmaps
                plot_heatmap(gene_lists[gene_set], preds_df, tfx_dict, name=f'results/{analysis}/{gene_set}_base_heatmap', sub_dict=subtype_dict)
        elif analysis == 'BC-TAN':
            # Define the new order of the indices
            new_order = ['UC', 'PUC', 'SQ', 'NE', 'Fanconi-Anemia',
                         'CCP.31', 'EMT.12', 'EMT',
                         'Hypoxia', 'Buffa_2010_Hypoxia_51', 'Ragnum_2014_Hypoxia_32', 'Ragnum_201 4_Hypoxia_woProliferation_12', 'Eustace_West_2013_Hypoxia_26', 'Eustace_West_2013_Hypoxia_woProliferation_22', 'Yang_West_2018_Hypoxia_up_19', 'Yang_West_2018_Hypoxia_up_woProliferation_12']
            # Reorder the indices of gsea_scores
            gsea_scores = gsea_scores.reindex(new_order)
            plot_heatmap(gsea_scores.index.tolist(), gsea_scores, tfx_dict, name=f'results/{analysis}/GSEA_scores_heatmap', gsea=True, breaks=[5, 8], sub_dict=subtype_dict)
            gsea_scores.to_csv(f'results/{analysis}/GSEA_scores.tsv', sep='\t')
            # Do GSEAs again with only molecular types:
            new_order = ['UC', 'PUC', 'SQ', 'NE', 'Fanconi-Anemia']
            gsea_scores = gsea_scores.reindex(new_order)
            plot_heatmap(gsea_scores.index.tolist(), gsea_scores, tfx_dict, name=f'results/{analysis}/GSEA_scores_MolSub_heatmap', gsea=True, sub_dict=subtype_dict)
            # plot_vs_tfx(preds_df, tfx_dict, gtex_ref, 'Bladder', name=f'results/{analysis}/TFX-vs-Ratio_')
            bc_log2_tpm_fit, splits = get_fit_tpm(bc_biopsies, preds_df)
            # Convert the dictionary to a DataFrame
            splits_df = pd.DataFrame(splits)
            # Transpose the DataFrame so that the outer keys become the rows and the inner keys become the columns
            splits_df = splits_df.transpose()
            # Save the DataFrame as a TSV file
            splits_df.to_csv(f'results/{analysis}/splits.tsv', sep='\t')
            for gene_set in ['all', 'Targets', 'BC-Phenotyping']:
                # Create a new DataFrame df_metrics with the same columns as preds_df and rows for r2, slope, and intercept
                df_metrics = pd.DataFrame(index=['r2', 'slope'], columns=preds_df.columns)
                for col in bc_log2_tpm_fit.columns:
                    # Find the matching column in preds_df
                    matching_col = next((c for c in preds_df.columns if c == col), None)
                    if matching_col:
                        r2, slope, mae = scatter_plot(bc_log2_tpm_fit[col], preds_df[matching_col], genes_subset=gene_lists[gene_set], name=f'results/{analysis}/{gene_set}_{col}_true-fit-vs-pred_scatter')
                        # Fill up df_metrics
                        df_metrics.loc['r2', matching_col] = r2
                        df_metrics.loc['slope', matching_col] = slope
                # Convert df_metrics to numeric, errors='coerce' will convert non-numeric data to NaN
                df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                print(f'For gene set {gene_set}, average r2: {df_metrics.loc["r2"].mean()}, average MAE: {mae}')
                # Plot gene set heatmaps
                if gene_set != 'all':
                    plot_heatmap(gene_lists[gene_set], preds_df, tfx_dict, name=f'results/{analysis}/{gene_set}_base_heatmap', sub_dict=subtype_dict)
        elif analysis == 'PC-TAN':
            # Drop "Unknown" samples from subtype_dict
            subtype_dict = {sample: subtype for sample, subtype in subtype_dict.items() if subtype != 'Unknown'}
            # Define the new order of the indices
            new_order = ['ARG.6', 'ARG.10', 'NE.6', 'NE.10', 'PSMA-high', 'PSMA-low',
                        'CCP.31', 'EMT.12', 'EMT',
                        'Hypoxia', 'Buffa_2010_Hypoxia_51', 'Ragnum_2014_Hypoxia_32', 'Ragnum_201 4_Hypoxia_woProliferation_12', 'Eustace_West_2013_Hypoxia_26', 'Eustace_West_2013_Hypoxia_woProliferation_22', 'Yang_West_2018_Hypoxia_up_19', 'Yang_West_2018_Hypoxia_up_woProliferation_12']
            # Reorder the indices of gsea_scores
            gsea_scores = gsea_scores.reindex(new_order)
            plot_heatmap(gsea_scores.index.tolist(), gsea_scores, tfx_dict, name=f'results/{analysis}/GSEA_scores_heatmap', gsea=True, breaks=[6, 9], sub_dict=subtype_dict)
            gsea_scores.to_csv(f'results/{analysis}/GSEA_scores.tsv', sep='\t')
            # plot_vs_tfx(preds_df, tfx_dict, gtex_ref, 'Prostate', name=f'results/{analysis}/TFX-vs-Ratio_')
            # tan_log2_tpm_avg = get_average_tpm(pc_biopsies)
            tan_log2_tpm_fit, splits = get_fit_tpm(pc_biopsies, preds_df)

            # Convert the dictionary to a DataFrame
            splits_df = pd.DataFrame(splits)
            # Transpose the DataFrame so that the outer keys become the rows and the inner keys become the columns
            splits_df = splits_df.transpose()
            # Save the DataFrame as a TSV file
            splits_df.to_csv(f'results/{analysis}/splits.tsv', sep='\t')
            plot_vs_meta(pctan_meta, preds_df, 'PSMA_mean', 'FOLH1',  name=f'results/{analysis}/FOLH1-vs-PSMA-mean_scatter', tfx_dict=tfx_dict)
            plot_vs_meta(pctan_meta, preds_df, 'PSMA_median', 'FOLH1', name=f'results/{analysis}/FOLH1-vs-PSMA-med_scatter', tfx_dict=tfx_dict)
            plot_vs_meta(tan_log2_tpm_fit, preds_df, 'FOLH1', 'FOLH1', name=f'results/{analysis}/FOLH1_pred-vs-biopsy_scatter', tfx_dict=tfx_dict)
            for gene in ['AR', 'NKX3-1', 'SYP', 'INSM1', 'ASCL1', 'SOX2', 'FOXA2', 'MKI67']:
                plot_vs_meta(pctan_meta, preds_df, gene + '_mean', gene, name=f'results/{analysis}/{gene}-vs-IHC-mean_scatter', tfx_dict=tfx_dict)
                plot_vs_meta(pctan_meta, preds_df, gene + '_median', gene, name=f'results/{analysis}/{gene}-vs-IHC-median_scatter', tfx_dict=tfx_dict)
                plot_vs_meta(tan_log2_tpm_fit, preds_df, gene, gene, name=f'results/{analysis}/{gene}_pred-vs-biopsy_scatter', tfx_dict=tfx_dict)
            for gene_set in ['all', 'Targets', 'PC-Phenotyping']:
                # Create a new DataFrame df_metrics with the same columns as preds_df and rows for r2, slope, and intercept
                df_metrics = pd.DataFrame(index=['r2', 'slope'], columns=preds_df.columns)
                for col in tan_log2_tpm_fit.columns:
                    # Find the matching column in preds_df
                    matching_col = next((c for c in preds_df.columns if c == col), None)
                    if matching_col:
                        if gene_set == 'all':
                            goi_dict = {'ARPC': ['AR', 'KLK2', 'KLK3', 'NKX3-1', 'TMPRSS2', 'PMEPA1'],
                                        'NEPC': ['SYP', 'CHGA', 'SCG3', 'ASCL1', 'INSM1', 'ACTL6B']}
                            r2, slope, mae = scatter_plot(tan_log2_tpm_fit[col], preds_df[matching_col], genes_subset=gene_lists[gene_set], name=f'results/{analysis}/{gene_set}_{col}_true-fit-vs-pred_scatter', goi_dict=goi_dict)
                        else:
                            r2, slope, mae = scatter_plot(tan_log2_tpm_fit[col], preds_df[matching_col], genes_subset=gene_lists[gene_set], name=f'results/{analysis}/{gene_set}_{col}_true-fit-vs-pred_scatter')
                        # Fill up df_metrics
                        df_metrics.loc['r2', matching_col] = r2
                        df_metrics.loc['slope', matching_col] = slope
                # for col in preds_df.columns:
                #     # Find the matching column in preds_df
                #     scatter_plot(gtex_ref['Whole_Blood'], preds_df[col], name=f'results/{analysis}/{col}_WB-vs-pred_scatter')
                #     scatter_plot(gtex_ref['Prostate'], preds_df[col], name=f'results/{analysis}/{col}_Prostate-vs-pred_scatter')
                # Convert df_metrics to numeric, errors='coerce' will convert non-numeric data to NaN
                df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                print(f'For gene set {gene_set}, average r2: {df_metrics.loc["r2"].mean()}, average MAE: {mae}')
                if gene_set != 'all':
                    # preds_df = preds_df.loc[:, tan_log2_tpm_fit.columns]
                    plot_heatmap(gene_lists[gene_set], preds_df, {key[:6]: value for key, value in tfx_dict.items() if value > 0.15}, name=f'results/{analysis}/{gene_set}_pred_heatmap', sub_dict={key[:6]: value for key, value in subtype_dict.items()})
                    plot_heatmap(gene_lists[gene_set], tan_log2_tpm_fit, {key[:6]: value for key, value in tfx_dict.items() if value > 0.15}, name=f'results/{analysis}/{gene_set}_true_heatmap', sub_dict={key[:6]: value for key, value in subtype_dict.items()})
                    # plot_double_heatmap(gene_lists[gene_set], tan_log2_tpm_fit, preds_df, name=f'results/{analysis}/{gene_set}_double_heatmap')

    ############################################################################################################
    # Run healthy analyses
    ############################################################################################################
    for analysis in healthy_analyses:
        preds_df = pd.read_csv(f'../results_holdout/penultimate_MSE/{analysis}_data/predictions.tsv', sep='\t', index_col=0)
        os.makedirs(f'results/{analysis}', exist_ok=True)
        plot_samples(preds_df, f'results/{analysis}/sample-expression-profiles')
        # load GSEA scores
        gsea_scores = pd.read_csv(f'GeneScores/{analysis}_gsva.sig.es.obs.txt', sep='\t', index_col=0)
        plot_heatmap(gsea_scores.index.tolist(), gsea_scores, tfx_dict, name=f'results/{analysis}/GSEA_scores_heatmap', gsea=True, breaks=[6, 9], sub_dict=subtype_dict)
        gsea_scores.to_csv(f'results/{analysis}/GSEA_scores.tsv', sep='\t')
        for sample in preds_df.columns:
            scatter_plot(gtex_ref['Whole_Blood'], preds_df[sample], name=f'results/{analysis}/{sample}_vs_Whole-Blood')
            scatter_plot(pbmc_ref['PBMC'], preds_df[sample], name=f'results/{analysis}/{sample}_vs_PBMC')


if __name__ == "__main__":
    main()
