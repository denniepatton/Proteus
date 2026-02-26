"""
run_analysis.py
v2.1.1, 5/26/2025
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script runs a complete analysis of PROTEUS and its outputs
### Make sure to run GenerateScores.R in GeneScores before running this script with run_gsea = True ###
"""

# Standard library imports
import os
import csv

# Third party imports
import pandas as pd
import numpy as np

# Local application imports
from analysis_utils import *

run_gsea = True  # Set to True to run GSEA and PAM50 analyses
tfx_limit = 0.03  # Minimum TFX threshold for patient analyses

def main():
    # results_training folders to run LuCaP / SCLC analysis on
    # validation_analyses = {'BioDVN-v2_BTS': 'BTS_HO_samples'}
                        #    'BioDVN_NoSeq_BTS': 'BTS_HO_samples',
                        #    'BioDVN_OnlySeq_BTS': 'BTS_HO_samples',}
    validation_analyses = {}
    # patient sets to run (header: sample   tfx subtype)
    patient_results = ['TL-NoBC-v3-2_BioDVN']
    patient_analyses = {'Pluvicto': 'Pluvicto_TFX_Subtype.tsv',
                        'PC-TAN': 'PC-TAN_TFX_Subtype.tsv',
                        'SCLC': 'SCLC_TFX_Subtype.tsv',
                        'BC-TAN': 'BC-TAN_TFX_Subtype.tsv',
                        'WCDT': 'WCDT_TFX_Subtype.tsv',
                        'UW': 'UW_TFX_Subtype.tsv'}

    # load reference gene lists
    gene_lists = {'all': None,
                  'GavinsSet': ['GavinsSet.txt'],
                  'SCLC-Targeted': ['SCLC-Targeted_FullGenes.txt'],
                  'CCP': ['CCP31.txt'],
                  'Targets_PC': ['Targets_PC.txt'],
                  'Targets_Vipul': ['Targets_Vipul.txt'],
                  'Targets_Pfizer': ['Targets_Pfizer.txt'],
                  'Targets_SCLC': ['Targets_SCLC.txt'],
                  'Targets_BC': ['Targets_BC.txt'],
                  'PAM50': ['PAM50.txt'],
                  'EMT': ['EMT.txt'],
                  'Hypoxia': ['Hypoxia.txt'],
                  'PlymGlyco': ['PlymateGlycolitic.txt'],
                  'PC-Phenotyping': [[8, 14], 'PC_PhenotypingGenes.txt'],
                  'BLCA': ['BLCA_LamList.txt'],
                  'Housekeeping': ['Housekeeping_HumanProteinAtlas.tsv'],
                  'Non-Housekeeping': ['NonHousekeeping_HumanProteinAtlas.tsv'],
                  'DruggableGenome': ['DruggableGenome_Tier1.txt'],
                  'CancerSurfaceome': ['CancerSurfaceome.txt']}
    set_names = {'all': None, 'GavinsSet': None, 'SCLC-Targeted': None,
                 'Targets_PC': None, 'Targets_SCLC': None, 'Targets_Vipul': None, 'Targets_Pfizer': None,
                 'Targets_BC': None, 'Housekeeping': None, 'Non-Housekeeping': None,
                 'DruggableGenome': None, 'CancerSurfaceome': None,
                 'PC-Phenotyping': ['NEURO I', 'NEURO II', 'AR']}
    print('Loading gene lists...')
    for key, value in gene_lists.items():
        if value is None:
            continue
        elif isinstance(value[0], list):
            file_content = load_file_as_list(os.path.join('../reference/', value[1]))
            ranges = [0] + value[0] + [None]
            new_value = [file_content[start:end] for start, end in zip(ranges[:-1], ranges[1:])]
        elif value[0] == 'DF-List':
            df = pd.read_csv(os.path.join('../reference/', value[1]), sep='\t')
            new_value = tuple(df[col].tolist() for col in df.columns)
        else:
            new_value = [load_file_as_list(os.path.join('../reference/', filename)) for filename in value]
        gene_lists[key] = tuple(new_value)

    print('Loading data...')
    # load ground truth and reference data
    hd_truth = pd.read_csv('../labels/HD-L_Log2CPMplus1.tsv', sep='\t', index_col=0)
    lucap_truth = pd.read_csv('../labels/LuCaP-PDX_Log2CPMplus1.tsv', sep='\t', index_col=0)
    lucap_meta = pd.read_csv('../reference/LuCaP-PDX_metadata.tsv', sep='\t', index_col=0).T
    sclc_truth = pd.read_csv('../labels/SCLC-PDX_Log2CPMplus1.tsv', sep='\t', index_col=0)
    sclc_meta = pd.read_csv('../reference/SCLC-PDX_metadata.tsv', sep='\t', index_col=0).T
    pht_truth_tm = pd.read_csv('../labels/PHT-tumor_Log2CPMplus1.tsv', sep='\t', index_col=0)
    pht_truth_bg = pd.read_csv('../labels/PHT-background_Log2CPMplus1.tsv', sep='\t', index_col=0)
    pht_meta = pd.read_csv('../labels/PHT_TFX_Subtype.tsv', sep='\t', index_col=0).T
    phh_truth = pd.read_csv('../labels/PHH-tumor_Log2CPMplus1.tsv', sep='\t', index_col=0)
    phs_truth = pd.read_csv('../labels/PHS-tumor_Log2CPMplus1.tsv', sep='\t', index_col=0)
    phs_meta = pd.read_csv('../labels/PHS_TFX_Subtype.tsv', sep='\t', index_col=0).T
    pctan_mean_base = pd.read_csv('../labels/PC-TAN_Log2CPMplus1.tsv', sep='\t', index_col=0)
    pctan_meta = pd.read_csv('../reference/PC-TAN_IHC-Scores.tsv', sep='\t', index_col=0)
    pctan_meta['LogPSA'] = np.log2(pd.to_numeric(pctan_meta['PSA_final'], errors='coerce') + 1)
    pctan_meta = pctan_meta.T
    pctan_meta.columns = "PC-TAN_" + pctan_meta.columns
    bctan_mean_base = pd.read_csv('../labels/BC-TAN_Log2CPMplus1.tsv', sep='\t', index_col=0)
    bctan_meta = pd.read_csv('../reference/BC-TAN_IHC-Scores.tsv', sep='\t', index_col=0)
    sclcpatient_mean = pd.read_csv('../labels/SCLC_Log2CPMplus1.tsv', sep='\t', index_col=0)           
    bctan_meta.index = "BC-TAN_" + bctan_meta.index.str.slice(0, 6)
    bctan_meta = bctan_meta.groupby(level=0).mean().T
    wcdt_truth = pd.read_csv('../labels/WCDT_Log2CPMplus1.tsv', sep='\t', index_col=0)
    wcdt_meta = pd.read_csv('../reference/WCDT_metadata.tsv', sep='\t', index_col=0)
    wcdt_meta = wcdt_meta.T
    psma_meta = pd.read_csv('../reference/Pluvicto_metadata.tsv', sep='\t', index_col=0).drop(columns=['LogPSA'])  # -> do internally
    psma_meta['LogPSA'] = np.log2(pd.to_numeric(psma_meta['PSA'], errors='coerce') + 1)
    psma_meta = psma_meta.T
    uw_meta = pd.read_csv('../reference/UW_metadata.tsv', sep='\t', index_col=0)
    uw_meta['LogPSA'] = np.log2(pd.to_numeric(uw_meta['PSA'], errors='coerce') + 1)
    uw_meta = uw_meta.T
    mrna_hl = pd.read_csv('../reference/mRNA-HL_13059_2022_2811_MOESM3_ESM_NonHousekeeping.tsv', sep='\t', index_col=0)

    training_samples = ["PC-TAN_23-008","PC-TAN_16-071","PC-TAN_19-048","PC-TAN_21-032", "PC-TAN_05-217",
                        "MEDO-PL-1441-004","MEDO-PL-1644-003","GUHA-PL-0037-005",
                        "DTB-261-BL","DTB-266-BL","DTB-214-BL",  # TL-NoBC set
                        "DTB-174-BL","DTB-149-PRO","DTB-261-PRO", "DTB-210-PRO"]  # Masked WCDT late-comers

    ############################################################################################################
    # Run initial training validation analyses
    ############################################################################################################
    print('*** Running training validation analyses.')
    for analysis, ho_set in validation_analyses.items():
        print(f'##### Running {analysis} combined validation analyses.')
        os.makedirs(f'results_CV/{analysis}', exist_ok=True)
        os.makedirs(f'results_CV/{analysis}/Combined', exist_ok=True)
        ### Load predictions and manage data
        """ Construct the following dataframes:
        df_val_tm: Combined validation predictions for tfx > 0.0 samples (No pure HD samples)
        df_val_bg: Combined validation predictions for tfx < 1.0 samples (No pure PDX samples)
        df_val_pure: Combined validation predictions for pure samples (Only PDX tumor preds and HD background preds)
        df_ho_tm: Holdout predictions for tfx > 0.0 samples (No pure HD samples)
        df_ho_pure: Holdout predictions for pure samples (Only PDX tumor preds)
        ~ truth dfs may contain samples matching both val and ho sets
        df_truth_tm: Combined validation truth for tfx > 0.0 samples (No pure HD samples)
        df_truth_bg: Combined validation truth for tfx < 1.0 samples (No pure PDX samples)
        df_truth_pure: Combined validation truth for pure samples (Only PDX tumor truth and HD background truth)
        ~ combined dfs are used for validation + holdout analyses
        df_combined_pure: Combined validation + holdout predictions for pure samples

        _tm and _bg dfs are used for benchmarking tfx/depth/(tfx+depth) dependence
        _pure dfs are used for pure sample analysis
        """
        # --- 1. Load/merge cross-validation predictions ---
        # Tumor
        cv_preds_tm = [pd.read_csv(f'../results_training/{analysis}/CV-fold_{fold}/validation-fold_{fold}_predictions_tumor.tsv', sep='\t', index_col=0)for fold in range(1, 6)]
        df_val_tm = merge_validation(cv_preds_tm)
        df_val_tm = df_val_tm.loc[:, ~df_val_tm.columns.str.startswith('BP')]  # Healthies in the triaing folds have BP identifiers
        # Background
        cv_preds_bg = [pd.read_csv(f'../results_training/{analysis}/CV-fold_{fold}/validation-fold_{fold}_predictions_background.tsv',sep='\t', index_col=0)for fold in range(1, 6)]
        df_val_bg = merge_validation(cv_preds_bg)
        df_val_bg = df_val_bg.loc[:, df_val_bg.columns.str.startswith(('BP', 'PHT'))]  # Healthies start with BP, healthy mixes with PHT
        # --- 2. Holdout predictions (tumor only) ---
        df_ho_tm = pd.read_csv(f'../results_holdout/{analysis}/{ho_set}/predictions_tumor.tsv',sep='\t', index_col=0)
        # --- 3. Truth DataFrames ---
        df_truth_tm = pd.concat([lucap_truth, sclc_truth, pht_truth_tm, phh_truth, phs_truth],axis=1, join='outer')
        df_truth_bg = pd.concat([hd_truth, pht_truth_bg],axis=1, join='outer')
        # --- 4. Build "pure" subsets for validation ---
        df_val_pure = df_val_tm.loc[:, ~df_val_tm.columns.str.startswith('PHT')]
        # Select columns from df_val_bg that start with "BP"
        bp_samples = df_val_bg.loc[:, df_val_bg.columns.str.startswith('BP')]
        df_val_pure = pd.concat([df_val_pure, bp_samples], axis=1)
        # --- 5. Build "pure" subsets for holdout ---
        # Tumor > 0.0 fraction only => exclude columns starting with mixture prefixes
        df_ho_pure = df_ho_tm.loc[:, ~df_ho_tm.columns.str.startswith(('PHH', 'KP6'))]
        df_ho_pure = df_ho_pure.loc[:, ~df_ho_pure.columns.str.contains('merge')]
        # --- 6. Build "pure" truth subsets (optional) ---
        # e.g. exclude partial columns from df_truth_tm and df_truth_bg
        df_truth_tm_pure = df_truth_tm.loc[:, ~df_truth_tm.columns.str.startswith('PHT')]
        df_truth_bg_pure = df_truth_bg.loc[:, ~df_truth_bg.columns.str.startswith('PHT')]
        df_truth_pure = pd.concat([df_truth_tm_pure, df_truth_bg_pure], axis=1)
        # --- 7. Combine validation + holdout for pure samples ---
        df_combined_pure = pd.concat([df_val_pure, df_ho_pure], axis=1)
        """ Global analysis of pure predictions """
        # Plot scatter of truth vs pred for all genes and all samples
        scatter_plot(df_truth_pure, df_val_pure, name=f'results_CV/{analysis}/Combined/true-vs-pred_val_scatter')
        scatter_plot(df_truth_pure, df_ho_pure, name=f'results_CV/{analysis}/Combined/true-vs-pred_ho_scatter')
        for ho_sample in df_ho_pure.columns:
            if ho_sample.startswith('LuCaP'):
                goi_dict = {'ARPC': ['AR', 'NKX3-1', 'FOLH1', 'KLK3'],
                            'NEPC': ['ASCL1', 'CHGA', 'SOX2', 'SYP'],}
            else:
                goi_dict = {'POU2F3': ['POU2F3'],
                            'ASCL1': ['ASCL1'],
                            'NEUROD1': ['NEUROD1'],
                            'ATOH1': ['ATOH1']}
            scatter_plot(df_truth_pure, df_ho_pure[[ho_sample]], name=f'results_CV/{analysis}/Combined/{ho_sample}_true-vs-pred_scatter', range=(0, 15), goi_dict=goi_dict)
        # Save predictions to file after re-normalizing and removine NaN genes (BP healthy donor)
        hd_temp = df_val_pure.loc[:, df_val_pure.columns.str.startswith('BP')]
        hd_temp = hd_temp.dropna()
        hd_temp.to_csv(f'results_CV/{analysis}/HD-Val_predictions.tsv', sep='\t')
        # Gather the subtypes for the samples
        lucap_subs = lucap_meta.loc['phenotype']
        sclc_subs = sclc_meta.loc['subtype']
        hd_subs = pd.Series('Healthy', index=hd_truth.columns, name='subtype')
        training_subtypes = pd.concat([lucap_subs, sclc_subs, hd_subs], axis=0).to_frame(name='subtype')
        training_subtypes['status'] = ['validation' if sample in df_val_pure.columns else 'holdout' for sample in training_subtypes.index]
        os.makedirs(f'results_CV/{analysis}/Combined/Target_scatters', exist_ok=True)
        for gene in gene_lists['GavinsSet'][0]:
            plot_vs_meta(df_truth_pure, df_combined_pure, gene, gene, name=f'results_CV/{analysis}/Combined/Target_scatters/{gene}_true-vs-pred_scatter', subtypes=training_subtypes)
        # Generate accuracy metrics
        combined_meta = pd.concat([lucap_meta, sclc_meta, pht_meta, phs_meta], axis=1, join='outer')
        # fill all NaN values in the row 'tfx' with 1.0 (pure PDX samples)
        combined_meta.loc['tfx'] = combined_meta.loc['tfx'].astype(float).fillna(1.0)
        for gene_set in ['all', 'DruggableGenome', 'CancerSurfaceome', 'Housekeeping', 'PC-Phenotyping']:
            thresholding_boxes(df_truth_tm, df_val_tm, combined_meta, gene_lists[gene_set], name=f'results_CV/{analysis}/Combined/validation_{gene_set}')
            # thresholding_boxes(df_truth_tm.loc[:, df_truth_tm.columns.str.contains('merge')],
            #                    df_ho_tm.loc[:, df_ho_tm.columns.str.contains('merge')],
            #                    combined_meta, gene_lists[gene_set], name=f'results_CV/{analysis}/Combined/synthetic_{gene_set}')
            thresholding_boxes(df_truth_tm.loc[:, df_truth_tm.columns.str.contains('merge') & df_truth_tm.columns.str.contains('LuCaP')],
                    df_ho_tm.loc[:, df_ho_tm.columns.str.contains('merge') & df_ho_tm.columns.str.contains('LuCaP')],
                    combined_meta, gene_lists[gene_set], name=f'results_CV/{analysis}/Combined/synthetic_{gene_set}')
        hl_boxes(df_val_tm, df_truth_tm, mrna_hl, f'results_CV/{analysis}/Combined/')
        # Run tissue-specific analyses
        for pdx in ['LuCaP-PDX', 'SCLC-PDX']:
            print(f'##### Running analyses for {analysis} in {pdx}.')
            os.makedirs(f'results_CV/{analysis}/{pdx}', exist_ok=True)
            if pdx == 'LuCaP-PDX':
                pdx_truth = lucap_truth
                pdx_meta = lucap_meta
                # pdx_subtypes = lucap_meta.loc['phenotype']
                # pdx_pam50 = lucap_meta.loc['PAM50']
                scatter_sets = ['all', 'PC-Phenotyping', 'Targets_PC', 'Non-Housekeeping', 'DruggableGenome', 'CancerSurfaceome', 'GavinsSet']
                heatmap_sets = ['PC-Phenotyping']
                ihc_pairs = [('PSMA-mean', 'FOLH1')]
                # targets = gene_lists['Targets_PC'][0]
                targets = gene_lists['Targets_Pfizer'][0]
                gseas = ['ARG.10', 'NE.10',  'CCP.31', 'HALLMARK_HYPOXIA']
            elif pdx == 'SCLC-PDX':
                pdx_truth = sclc_truth
                pdx_meta = sclc_meta
                # pdx_subtypes = sclc_meta.loc['subtype']
                # pdx_pam50 = None
                scatter_sets = ['all', 'Targets_SCLC', 'Non-Housekeeping', 'DruggableGenome', 'CancerSurfaceome']
                heatmap_sets = ['Targets_SCLC']
                ihc_pairs = []
                targets = gene_lists['Targets_SCLC'][0]
                gseas = ['ARG.10', 'NE.10', 'CCP.31']
            target_samples = pdx_truth.columns.union(hd_truth.columns)
            val_subset = df_combined_pure.loc[:, df_combined_pure.columns.intersection(target_samples)]
            val_temp = val_subset.loc[:, ~val_subset.columns.str.startswith('BP')].dropna()
            val_temp.to_csv(f'results_CV/{analysis}/{pdx}_predictions.tsv', sep='\t')
            # Plot sample overview
            plot_samples(val_subset, f'results_CV/{analysis}/{pdx}/sample-expression-profiles')
            cluster_analysis(df_truth_pure, val_subset, name=f'results_CV/{analysis}/{pdx}/cluster-analysis', subtypes=training_subtypes)
            # Combined samples' scatter plots (including group, gene, and against IHC scores) and heatmaps
            for gene_set in scatter_sets:
                scatter_plot(df_truth_pure, val_subset, genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results_CV/{analysis}/{pdx}/{gene_set}_true-vs-pred_scatter')
                # r2_mae_plot(df_truth_pure, val_subset, pdx_meta, gene_lists[gene_set], name=f'results_CV/{analysis}/{pdx}/{gene_set}_r2_mae_plot')
            for gene_set in heatmap_sets:
                plot_double_heatmap(gene_lists[gene_set], df_truth_pure, val_subset, name=f'results_CV/{analysis}/{pdx}/{gene_set}_heatmap', subtypes=training_subtypes)
                plot_roc_curve(gene_lists[gene_set], df_truth_pure, val_subset, name=f'results_CV/{analysis}/{pdx}/{gene_set}_ROC')
            os.makedirs(f'results_CV/{analysis}/{pdx}/IHC_scatters', exist_ok=True)
            for ihc_pair in ihc_pairs:
                plot_vs_meta(pdx_meta, val_subset, ihc_pair[0], ihc_pair[1], x_unit='IHC Score', name=f'results_CV/{analysis}/{pdx}/IHC_scatters/{ihc_pair[1]}-pred-vs-{ihc_pair[0]}_scatter', subtypes=training_subtypes)
                plot_vs_meta(pdx_meta, df_truth_pure, ihc_pair[0], ihc_pair[1], x_unit='IHC Score', y_unit='RNA-Seq Log\u2082(CPM+1)', name=f'results_CV/{analysis}/{pdx}/IHC_scatters/{ihc_pair[1]}-truth-vs-{ihc_pair[0]}_scatter', subtypes=training_subtypes)
            os.makedirs(f'results_CV/{analysis}/{pdx}/Target_scatters', exist_ok=True)
            for gene in targets:
                plot_vs_meta(df_truth_pure, val_subset, gene, gene, name=f'results_CV/{analysis}/{pdx}/Target_scatters/{gene}_true-vs-pred_scatter', subtypes=training_subtypes)

            if run_gsea:  # GSVA and PAM50 heatmaps/scatters
                os.makedirs(f'results_CV/{analysis}/{pdx}/GSEA_PAM50', exist_ok=True)
                gsva_truth = pd.read_csv(f'GeneScores/BioDVN-v2_BTS/{pdx}_truth_gsva.txt', sep='\t', index_col=0)
                gsva_pred = pd.read_csv(f'GeneScores/BioDVN-v2_BTS/{pdx}_predicted_gsva.txt', sep='\t', index_col=0)
                pam50_truth = pd.read_csv(f'GeneScores/BioDVN-v2_BTS/{pdx}_truth_pam50scores.txt', sep='\t', index_col=0).T
                pam50_pred = pd.read_csv(f'GeneScores/BioDVN-v2_BTS/{pdx}_predicted_pam50scores.txt', sep='\t', index_col=0).T
                # # Select only the 'Basal', 'LumA', and 'LumB' rows of PAM50
                # pam50_truth = pam50_truth.loc[['Basal', 'LumA', 'LumB'], :]
                # pam50_pred = pam50_pred.loc[['Basal', 'LumA', 'LumB'], :]
                # os.makedirs(f'results_CV/{analysis}/{pdx}/GSEA_PAM50', exist_ok=True)
                # # plot scores as scatter plots
                # if pdx_pam50 is not None:  # do not know PAM50 truth for SCLC data
                #     for psub in ['LumA', 'LumB', 'Basal']:
                #         plot_vs_meta(pam50_truth, pam50_pred, psub, psub, x_unit='Score (RNA-Seq)', y_unit='Score (Predicted)', name=f'results_CV/{analysis}/{pdx}/GSEA_PAM50/{psub}_scatter', subtypes=pdx_pam50)
                # Restrict the GSEA scores to the groups in gseas
                # Filter target_samples to only include those present in both dataframes
                available_samples_truth = gsva_truth.columns.intersection(target_samples)
                available_samples_pred = gsva_pred.columns.intersection(target_samples)
                common_samples = available_samples_truth.intersection(available_samples_pred)
                
                gsva_truth = gsva_truth.loc[gseas, common_samples]
                gsva_pred = gsva_pred.loc[gseas, common_samples]
                for gsea in gseas:
                    plot_vs_meta(gsva_truth, gsva_pred, gsea, gsea, x_unit='Score (RNA-Seq)', y_unit='Score (Predicted)', name=f'results_CV/{analysis}/{pdx}/GSEA_PAM50/{gsea}_scatter', subtypes=training_subtypes)
                # make a dictionary of lucap subtypes, mapping every sample in common_samples to 1.0
                tfx_dict = {sample: 1.0 for sample in target_samples}
                plot_double_heatmap(gsva_pred.index.tolist(), gsva_truth, gsva_pred, name=f'results_CV/{analysis}/{pdx}/GSEA_PAM50/GSVA-Scores_true-vs-pred_heatmap', gsea=True, subtypes=training_subtypes)
                if pdx_pam50 is not None:  # do not know PAM50 truth for SCLC data
                    plot_double_heatmap(pam50_pred.index.tolist(), pam50_truth, pam50_pred, name=f'results_CV/{analysis}/{pdx}/GSEA_PAM50/PAM50-Scores_true-vs-pred_heatmap', gsea=True, subtypes=pdx_pam50)


    ############################################################################################################
    # Run hold out patient analyses
    ############################################################################################################
    print('*** Running patient analyses.')
    for direct in patient_results:
        # Load all predictions/truth/meta data and perform initial cleaning:
        preds_dfs = {}
        ft_dfs = {}
        activity_dfs = {}
        tfx_dicts = {}
        subtype_dicts = {}
        subtype_dfs = {}
        gsea_preds = {}
        gsea_truths = {}
        pam50_preds = {}
        pam50_truths = {}
        for analysis in patient_analyses.keys():
            print(f'### Loading and cleaning data for {analysis} patient cohort.')
            preds_df = pd.read_csv(f'../results_holdout/{direct}/{analysis}/predictions_tumor.tsv', sep='\t', index_col=0)
            preds_activity = pd.read_csv(f'../results_holdout/{direct}/{analysis}/classifications_tumor.tsv', sep='\t', index_col=0)
            # Load TFX and subtypes
            with open(f'../labels/{patient_analyses[analysis]}', 'r') as file:
                reader = csv.reader(file, delimiter='\t')
                next(reader, None)  # Skip the header
                rows = list(reader)  # Store the rows in a list
                tfx_dict = {row[0]: float(row[1]) for row in rows}
                subtype_dict = {row[0]: row[2] for row in rows}
                subtype_df = pd.DataFrame.from_dict({row[0]: analysis for row in rows}, orient='index', columns=['subtype'])
            if run_gsea:  # Load GSVA scores
                gsea_pred = pd.read_csv(f'GeneScores/{direct}/{analysis}_predicted_gsva.txt', sep='\t', index_col=0)
                pam50_pred = pd.read_csv(f'GeneScores/{direct}/{analysis}_predicted_pam50scores.txt', sep='\t', index_col=0).T
                pam50_pred = pam50_pred.iloc[:3, :]  # Select only the 'Basal', 'LumA', and 'LumB' rows of PAM50
                pam50_pred.apply(pd.to_numeric, errors='coerce', axis=1)  # Ensure numeric values
                gsea_preds[analysis] = gsea_pred
                pam50_preds[analysis] = pam50_pred
                if analysis != 'Pluvicto' and analysis != 'UW':  # Pluvicto and UW do not have truth data / RNA-seq
                    gsea_truth = pd.read_csv(f'GeneScores/{direct}/{analysis}_truth_gsva.txt', sep='\t', index_col=0)
                    pam50_truth = pd.read_csv(f'GeneScores/{direct}/{analysis}_truth_pam50scores.txt', sep='\t', index_col=0).T
                    pam50_truth = pam50_truth.iloc[:3, :]
                    gsea_truths[analysis] = gsea_truth
                    pam50_truths[analysis] = pam50_truth
            # Restrict to patients with TFX >= threhsold
            preds_df = preds_df.drop(columns=['MEDO-PL-1437-001'], errors='ignore')  # MEDO-PL-1437-001 a known outlier based on expression profile.
            dropped_samples = [sample for sample in preds_df.columns if sample not in tfx_dict or tfx_dict[sample] < tfx_limit]
            if len(dropped_samples) > 0:
                print(f"Dropping {len(dropped_samples)} samples (due to TFX):", dropped_samples)
                preds_df = preds_df.loc[:, [sample for sample in preds_df.columns if sample in tfx_dict and tfx_dict[sample] >= tfx_limit]]
            # Drop samples used in training, if applicable
            existing_training_samples = [sample for sample in training_samples if sample in preds_df.columns]
            ft_df = preds_df.loc[:, existing_training_samples] # Get fine-tuning set samples
            preds_df = preds_df.drop(columns=training_samples, errors='ignore')
            # Special cohort-specific processing
            if analysis == 'Pluvicto':
                # Drop samples not ending in C1 (baseline) from preds_df, then rename s.t. e.g. FH0829_E_C1 -> FH0829 (patient level)
                preds_df = preds_df.loc[:, preds_df.columns.str.endswith('_C1')]
                preds_df.columns = preds_df.columns.str.replace(r'_E_C1', '')
                tfx_dict = {key.replace('_E_C1', ''): value for key, value in tfx_dict.items()}
            if analysis == 'PC-TAN':
                pctan_mean = pctan_mean_base.reindex(columns=preds_df.columns)
            if analysis == 'BC-TAN':
                bctan_mean = bctan_mean_base.reindex(columns=preds_df.columns)
            # populate dictionaries
            preds_dfs[analysis] = preds_df
            ft_dfs[analysis] = ft_df
            activity_dfs[analysis] = preds_activity
            tfx_dicts[analysis] = tfx_dict
            subtype_dicts[analysis] = subtype_dict
            subtype_dfs[analysis] = subtype_df

        ############################################################################################################
        # Run cross-cohort analyses
        ############################################################################################################
        combined_name = 'Combined_PC-SCLC-BC'
        combined_analyses = ['PC-TAN', 'WCDT', 'SCLC', 'BC-TAN']
        truth_dfs = [pctan_mean, wcdt_truth, sclcpatient_mean, bctan_mean]
        genes = ['CD274', 'CD276', 'KRT7', 'KRT20', 'FGFR1', 'FGFR3', 'FGFR4', 'ERBB2', 'NECTIN4', 'PROX1']
        genes += gene_lists['Targets_BC'][0]
        genes += gene_lists['Targets_PC'][0]
        genes += gene_lists['Targets_SCLC'][0]
        gseas = ['ARG.10', 'NE.10', 'CCP.31']
        pam50s = ['LumA', 'LumB', 'Basal']

        combined_preds = []
        combined_truths = []
        combined_fts = []
        combined_tfx_dict = {}
        combined_cohort_dict = {}
        print(f'*** Running combined (cross-cohort) analysis.')
        for i, analysis in enumerate(combined_analyses):
            preds_df = preds_dfs[analysis]
            ft_df = ft_dfs[analysis]
            preds_activity = activity_dfs[analysis]
            if not os.path.exists(f'results_patient/{direct}/{combined_name}/{analysis}_confidence'):
                os.makedirs(f'results_patient/{direct}/{combined_name}/{analysis}_confidence')
            confidence_analysis(preds_df, preds_activity, truth_dfs[i], name=f'results_patient/{direct}/{combined_name}/{analysis}_confidence')
            combined_preds.append(preds_df)
            combined_truths.append(truth_dfs[i])
            combined_fts.append(ft_df)
            combined_tfx_dict.update(tfx_dicts[analysis])
            cohort_dict = {sample: analysis for sample in preds_df.columns}
            combined_cohort_dict.update(cohort_dict)

        preds_df = pd.concat(combined_preds, axis=1)
        fts_df = pd.concat(combined_fts, axis=1)
        truth_df = pd.concat(combined_truths, axis=1)
        # Convert subtypes dict to a df with keys as index and subtypes as the one columns:
        cohort_df = pd.DataFrame.from_dict(combined_cohort_dict, orient='index', columns=['subtype'])
        os.makedirs(f'results_patient/{direct}/{combined_name}', exist_ok=True)
        # Half-life and IHC analyses
        gene_ranges = truth_df.max(axis=1) - truth_df.min(axis=1)
        truth_df_filtered = truth_df[gene_ranges > 2]
        hl_boxes(preds_df, truth_df_filtered, mrna_hl, f'results_patient/{direct}/{combined_name}/2PLUS')
        ihc_spearman_heatmap(preds_df, truth_df, [pctan_meta, bctan_meta], name=f'results_patient/{direct}/{combined_name}/IHC_correlation_heatmap')
        # Target scatters
        for gene in genes:
            os.makedirs(f'results_patient/{direct}/{combined_name}/Target_scatters', exist_ok=True)
            plot_vs_meta(truth_df, preds_df, gene, gene, name=f'results_patient/{direct}/{combined_name}/Target_scatters/{gene}_pred-vs-biopsy_scatter', tfx_dict=combined_tfx_dict, subtypes=cohort_df)
        # Gene activity AUC ROC curves (slow due to bootstrapping)
        # for gene_set in ['all', 'DruggableGenome', 'CancerSurfaceome', 'BLCA', 'PC-Phenotyping']:
        #     if gene_set == 'all':
        #         combined_auc_roc(truth_df, preds_df, combined_cohort_dict, name=f'results_patient/{direct}/{combined_name}/{gene_set}_AUC_ROC', genes=preds_df.index.tolist())
        #     elif gene_set == 'PC-Phenotyping':  # Need to use all sub-groups of the gene list
        #         genes_temp = [gene for sublist in gene_lists[gene_set] for gene in sublist]
        #         combined_auc_roc(truth_df, preds_df, combined_cohort_dict, name=f'results_patient/{direct}/{combined_name}/{gene_set}_AUC_ROC', genes=genes_temp)   
        #     else:
        #         combined_auc_roc(truth_df, preds_df, combined_cohort_dict, name=f'results_patient/{direct}/{combined_name}/{gene_set}_AUC_ROC', genes=gene_lists[gene_set][0])
        # Fine-tuning set gene-level accuracy analysis
        get_gene_accuracy(truth_df, fts_df, name=f'results_patient/{direct}/{combined_name}/FT_Accuracy-Thresholding')
        get_gene_accuracy(truth_df, preds_df, name=f'results_patient/{direct}/{combined_name}/Patient_Accuracy-Thresholding')
        # GSEA analyses
        if run_gsea:
            gsea_pred = [gsea_preds[analysis] for analysis in combined_analyses]
            gsea_truth = [gsea_truths[analysis] for analysis in combined_analyses]
            # Concatenate the GSEA scores
            gsea_pred = pd.concat(gsea_pred, axis=1)
            gsea_truth = pd.concat(gsea_truth, axis=1)
            plot_pearson_heatmap(gsea_truth, gsea_pred, cohort_df, name=f'results_patient/{direct}/{combined_name}/GSEA_scores_spearman_heatmap')
            for gsea in gseas:
                plot_vs_meta(gsea_truth, gsea_pred, gsea, gsea, x_unit='score (RNA-Seq)', y_unit='score (Predicted)', name=f'results_patient/{direct}/{combined_name}/{gsea}_scatter', tfx_dict=combined_tfx_dict, subtypes=cohort_df)
            pam50_pred = [pam50_preds[analysis] for analysis in combined_analyses]
            pam50_truth = [pam50_truths[analysis] for analysis in combined_analyses]
            # Concatenate the PAM50 scores
            pam50_pred = pd.concat(pam50_pred, axis=1)
            pam50_truth = pd.concat(pam50_truth, axis=1)
            for psub in pam50s:
                plot_vs_meta(pam50_truth, pam50_pred, psub, psub, x_unit='score (RNA-Seq)', y_unit='score (Predicted)', name=f'results_patient/{direct}/{combined_name}/{psub}_scatter', tfx_dict=combined_tfx_dict, subtypes=cohort_df)
            # Cast PAM50 to dominant scores for cohort-level AUC-ROCs
            dominant_pam50 = pam50_truth.idxmax(axis=0)
            basal_dominant_df = pd.DataFrame({'Basal': (dominant_pam50 == 'Basal').astype(int)})
            combined_auc_roc(basal_dominant_df.T, pam50_pred, combined_cohort_dict, name=f'results_patient/{direct}/{combined_name}/PAM50_AUC_ROC', variable='Basal')
            # Assemble NE-like status for the combined cohort (ignoring AMPC)
            ne_status_dict = {}
            positive_ne_subtypes = ["NEPC", "AMPC", "NE"]
            drop_subtypes = ["MIX", "Unknown"]  # "MIX" may include DNPC and/or be ARPC-dominant, with infiltration not reflecting NE-like status -> use pure sampling
            for analysis_key in ['PC-TAN', 'WCDT', 'BC-TAN']:
                current_subtype_dict = subtype_dicts[analysis_key]
                for sample, subtype in current_subtype_dict.items():
                    if subtype in drop_subtypes:
                        continue # Skip these samples
                    elif subtype in positive_ne_subtypes:
                        ne_status_dict[sample] = 1
                    else:
                        ne_status_dict[sample] = 0
            ne_status_df = pd.DataFrame.from_dict(ne_status_dict, orient='index', columns=['NE-like'])
            ne_preds_df = gsea_pred.loc[['NE.10'], :]
            ne_preds_df.index = ['NE-like']
            combined_auc_roc(ne_status_df.T, ne_preds_df, combined_cohort_dict, name=f'results_patient/{direct}/{combined_name}/NE_AUC_ROC', variable='NE-like')

        ############################################################################################################
        # Run single-cohort analyses
        ############################################################################################################
        for analysis in patient_analyses:
            print(f'##### Running analyses for {analysis} using {direct} model.')
            os.makedirs(f'results_patient/{direct}/{analysis}', exist_ok=True)
            preds_df = preds_dfs[analysis]
            tfx_dict = tfx_dicts[analysis]
            subtype_dict = subtype_dicts[analysis]
            subtype_df = subtype_dfs[analysis]
            if run_gsea:
                gsea_pred = gsea_preds[analysis]
                pam50_pred = pam50_preds[analysis]
                if analysis != 'Pluvicto' and analysis != 'UW':
                    gsea_truth = gsea_truths[analysis]
                    pam50_truth = pam50_truths[analysis]

            # Cohort by cohort analysis:
            if analysis == 'Pluvicto':
                # Create sample distributions, true and predicted, for examining transformations
                plot_samples(preds_df, f'results_patient/{direct}/{analysis}/predicted-expression-profiles_raw')
                preds_df.to_csv(f'results_patient/{direct}/{analysis}/predictions_tumor.tsv', sep='\t')
                plot_vs_meta(psma_meta, preds_df, 'LogPSA', 'KLK3', x_unit='Log2 Serum PSA (ng/ml)', name=f'results_patient/{direct}/{analysis}/KLK3pred-vs-SerumPSA_scatter', tfx_dict=tfx_dict)
                if run_gsea:
                    ### Survival Analysis ###
                    os_df = psma_meta.T
                    os_df.drop(columns=['Radiographic_progression', 'LogPSA'], inplace=True)
                    # left join pam50_pred to the df (keep all rows in os_df, even if they are not in pam50_pred)
                    os_df = os_df.join(pam50_pred.T, how='left')
                    # left join gsea_pred to the df (keep all rows in os_df, even if they are not in gsea_pred)
                    os_df = os_df.join(gsea_pred.T, how='left')
                    # add "FOLH1" from preds_df (left-join)
                    folh1_pred = preds_df.loc['FOLH1']
                    os_df = os_df.join(folh1_pred.T, how='left')
                    # add "TFX" from tfx_dict (left-join)
                    os_df['TFX'] = os_df.index.map(tfx_dict)
                    # drop any rows with NaN in "T_Cycles"
                    os_df = os_df.dropna(subset=['T_Cycles'])
                    survival_analysis(os_df, 'T_Cycles', 'PSA_progression', 'Cycles', name=f'results_patient/{direct}/{analysis}/OS_analysis', keep_vars=['PSA', 'Volume_PSMA', 'Mean_SUV_PSMA', 'Median_SUV_PSMA', 'Max_SUV_PSMA', 'Volume_FDG', 'TFX'])
                plot_vs_meta(psma_meta, preds_df, 'Max_SUV_PSMA', 'FOLH1',  x_unit='Max SUV', name=f'results_patient/{direct}/{analysis}/FOLH1-vs-PSMA-Max_scatter', tfx_dict=tfx_dict, print_metrics=True)
                plot_vs_meta(psma_meta, preds_df, 'Mean_SUV_PSMA', 'FOLH1',  x_unit='Mean SUV', name=f'results_patient/{direct}/{analysis}/FOLH1-vs-PSMA-Mean_scatter', tfx_dict=tfx_dict, print_metrics=True)
                plot_vs_meta(psma_meta, preds_df, 'Median_SUV_PSMA', 'FOLH1',  x_unit='Median SUV', name=f'results_patient/{direct}/{analysis}/FOLH1-vs-PSMA-Median_scatter', tfx_dict=tfx_dict, print_metrics=True)
                plot_vs_meta(psma_meta, preds_df, 'Volume_PSMA', 'FOLH1',  x_unit='Volume PSMA', name=f'results_patient/{direct}/{analysis}/FOLH1-vs-PSMA-Volume_scatter', tfx_dict=tfx_dict, print_metrics=True)
            elif analysis == 'PC-TAN':
                # Create sample distributions, true and predicted, for examining transformations
                plot_samples(preds_df, f'results_patient/{direct}/{analysis}/predicted-expression-profiles_raw')
                plot_samples(pctan_mean, f'results_patient/{direct}/{analysis}/truth-expression-profiles_raw')
                preds_df.to_csv(f'results_patient/{direct}/{analysis}/predictions_tumor.tsv', sep='\t')
                pctan_mean.to_csv(f'results_patient/{direct}/{analysis}/predictions_truth.tsv', sep='\t')
                heatmap_sets = ['CCP', 'Targets_PC', 'PAM50', 'PC-Phenotyping', 'Targets_Vipul', 'PlymGlyco']
                # targets = gene_lists['Targets_PC'][0]
                targets = gene_lists['PlymGlyco'][0]
                if run_gsea:
                    gseas = ['ARG.10', 'NE.10', 'CCP.31']
                    plot_heatmap(gseas, gsea_pred, tfx_dict, name=f'results_patient/{direct}/{analysis}/GSEA_scores_heatmap', gsea=True, sub_dict=subtype_dict)
                    plot_double_heatmap(gseas, gsea_truth, gsea_pred, name=f'results_patient/{direct}/{analysis}/GSEA_scores_true-vs-pred_heatmap', gsea=True)
                    gsea_pred.to_csv(f'results_patient/{direct}/{analysis}/GSEA_scores.tsv', sep='\t')
                    for psub in ['LumA', 'LumB', 'Basal']:
                        plot_vs_meta(pam50_truth, pam50_pred, psub, psub, x_unit='Score (RNA-Seq)', y_unit='Score (Predicted)', name=f'results_patient/{direct}/{analysis}/{psub}_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                    for gsea in gseas:
                        plot_vs_meta(gsea_truth, gsea_pred, gsea, gsea, x_unit='Score (RNA-Seq)', y_unit='Score (Predicted)', name=f'results_patient/{direct}/{analysis}/{gsea}_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                # IHC and target scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/IHC_scatters', exist_ok=True)
                os.makedirs(f'results_patient/{direct}/{analysis}/Target_scatters', exist_ok=True)
                plot_vs_meta(pctan_meta, preds_df, 'FOLH1_mean', 'FOLH1',  x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/FOLH1pred-vs-PSMA-mean_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                plot_vs_meta(pctan_meta, preds_df, 'FOLH1_median', 'FOLH1', x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/FOLH1pred-vs-PSMA-med_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                plot_vs_meta(pctan_meta, preds_df, 'LogPSA', 'KLK3', x_unit='Log2 Serum PSA (ng/ml)', name=f'results_patient/{direct}/{analysis}/IHC_scatters/KLK3pred-vs-SerumPSA_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                plot_vs_meta(pctan_meta, pctan_mean, 'FOLH1_mean', 'FOLH1',  x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/FOLH1true-vs-PSMA-mean_scatter', subtypes=subtype_df)
                plot_vs_meta(pctan_meta, pctan_mean, 'FOLH1_median', 'FOLH1', x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/FOLH1true-vs-PSMA-med_scatter', subtypes=subtype_df)
                plot_vs_meta(pctan_meta, pctan_mean, 'LogPSA', 'KLK3', x_unit='Log2 Serum PSA (ng/ml)', name=f'results_patient/{direct}/{analysis}/IHC_scatters/KLK3true-vs-SerumPSA_scatter', subtypes=subtype_df)
                for gene in ['AR', 'NKX3-1', 'SYP', 'INSM1', 'ASCL1', 'SOX2', 'FOXA2', 'MKI67']:
                    plot_vs_meta(pctan_meta, preds_df, gene + '_mean', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}pred-vs-IHC-mean_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                    plot_vs_meta(pctan_meta, preds_df, gene + '_median', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}pred-vs-IHC-median_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                    plot_vs_meta(pctan_meta, pctan_mean, gene + '_mean', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}true-vs-IHC-mean_scatter', subtypes=subtype_df)
                    plot_vs_meta(pctan_meta, pctan_mean, gene + '_median', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}true-vs-IHC-median_scatter', subtypes=subtype_df)
                for gene in ['STEAP1', 'DLL3']:  # Only mean available
                    plot_vs_meta(pctan_meta, preds_df, gene + '_mean', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}pred-vs-IHC-mean_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                    plot_vs_meta(pctan_meta, pctan_mean, gene + '_mean', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}true-vs-IHC-mean_scatter', subtypes=subtype_df)
                for gene in targets + ['AFF3', 'CDH3', 'GALNT12', 'KRT19']:
                    plot_vs_meta(pctan_mean, preds_df, gene, gene, name=f'results_patient/{direct}/{analysis}/Target_scatters/{gene}_pred-vs-biopsy_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                # Sample-wise scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters', exist_ok=True)
                for gene_set in ['all', 'Targets_PC', 'PC-Phenotyping', 'Non-Housekeeping', 'DruggableGenome', 'CancerSurfaceome']:
                    # Create a new DataFrame df_metrics with the same columns as preds_df and rows for r2, pearson, and intercept
                    df_metrics = pd.DataFrame(index=['r2', 'pearson', 'mae'], columns=preds_df.columns)
                    for col in preds_df.columns:
                        if col not in pctan_mean.columns:
                            continue
                        if gene_set == 'all':
                            goi_dict = {'ARPC': ['AR', 'NKX3-1', 'FOLH1', 'KLK3'],
                                        'NEPC': ['ASCL1', 'CHGA', 'SOX2', 'SYP'],}
                            r2, pearson, mae = scatter_plot(pctan_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], name=f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter', goi_dict=goi_dict)
                        else:
                            r2, pearson, mae = scatter_plot(pctan_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter')
                        # Fill up df_metrics
                        df_metrics.loc['r2', col] = r2
                        df_metrics.loc['pearson', col] = pearson
                        df_metrics.loc['mae', col] = mae
                    # Convert df_metrics to numeric, errors='coerce' will convert non-numeric data to NaN
                    df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                    print(f'For gene set {gene_set}, median R²: {df_metrics.loc["r2"].median()}, median Pearson r: {df_metrics.loc["pearson"].median()}, median MAE: {df_metrics.loc["mae"].median()}')
                for gene_set in heatmap_sets:
                    plot_double_heatmap(gene_lists[gene_set], pctan_mean, preds_df, name=f'results_patient/{direct}/{analysis}/{gene_set}_double_heatmap', subtypes=subtype_df)
                    plot_roc_curve(gene_lists[gene_set], pctan_mean, preds_df, name=f'results_patient/{direct}/{analysis}/{gene_set}_ROC')
                scatter_plot(pctan_mean, preds_df, genes_subset=gene_lists['PC-Phenotyping'], subset_names=set_names['PC-Phenotyping'], name=f'results_patient/{direct}/{analysis}/PC-Phenotyping_true-vs-pred_scatter')
            elif analysis == 'BC-TAN':
                # Create sample distributions, true and predicted, for examining transformations
                plot_samples(preds_df, f'results_patient/{direct}/{analysis}/predicted-expression-profiles_raw')
                plot_samples(bctan_mean, f'results_patient/{direct}/{analysis}/truth-expression-profiles_raw')
                preds_df.to_csv(f'results_patient/{direct}/{analysis}/predictions_tumor.tsv', sep='\t')
                bctan_mean.to_csv(f'results_patient/{direct}/{analysis}/predictions_truth.tsv', sep='\t')
                heatmap_sets = ['CCP', 'Targets_BC', 'PAM50']
                targets = gene_lists['Targets_BC'][0]
                if run_gsea:
                    gseas = ['ARG.10', 'NE.10', 'CCP.31']
                    plot_heatmap(gseas, gsea_pred, tfx_dict, name=f'results_patient/{direct}/{analysis}/GSEA_scores_heatmap', gsea=True, sub_dict=subtype_dict)
                    plot_double_heatmap(gseas, gsea_truth, gsea_pred, name=f'results_patient/{direct}/{analysis}/GSEA_scores_true-vs-pred_heatmap', gsea=True)
                    gsea_pred.to_csv(f'results_patient/{direct}/{analysis}/GSEA_scores.tsv', sep='\t')
                # IHC scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/IHC_scatters', exist_ok=True)
                for gene in ['NECTIN4', 'MKI67', 'ERBB2']:
                    plot_vs_meta(bctan_meta, preds_df, gene + '_mean', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}pred-vs-IHC-mean_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                    plot_vs_meta(bctan_meta, preds_df, gene + '_median', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}pred-vs-IHC-median_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                    plot_vs_meta(bctan_meta, bctan_mean, gene + '_mean', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}true-vs-IHC-mean_scatter', subtypes=subtype_df)
                    plot_vs_meta(bctan_meta, bctan_mean, gene + '_median', gene, x_unit='IHC Score', name=f'results_patient/{direct}/{analysis}/IHC_scatters/{gene}true-vs-IHC-median_scatter', subtypes=subtype_df)
                # Target gene scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/Target_scatters', exist_ok=True)
                for gene in targets:
                    plot_vs_meta(bctan_mean, preds_df, gene, gene, name=f'results_patient/{direct}/{analysis}/Target_scatters/{gene}_pred-vs-biopsy_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                # Sample-wise scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters', exist_ok=True)
                for gene_set in ['all', 'Targets_BC', 'Non-Housekeeping', 'DruggableGenome', 'CancerSurfaceome']:
                    # Create a new DataFrame df_metrics with the same columns as preds_df and rows for r2, pearson, and intercept
                    df_metrics = pd.DataFrame(index=['r2', 'pearson', 'mae'], columns=preds_df.columns)
                    for col in preds_df.columns:
                        if col not in bctan_mean.columns:
                            continue
                        if gene_set == 'all':
                            goi_dict = {'UC': ['GATA3', 'FOXA1'],
                                        'SQ': ['KRT14', 'KRT15', 'EGFR'],
                                        'NE': ['CHGA', 'SYP']}
                            r2, pearson, mae = scatter_plot(bctan_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], name=f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter', goi_dict=goi_dict)
                        else:
                            r2, pearson, mae = scatter_plot(bctan_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter')
                        # Fill up df_metrics
                        df_metrics.loc['r2', col] = r2
                        df_metrics.loc['pearson', col] = pearson
                        df_metrics.loc['mae', col] = mae
                    # Convert df_metrics to numeric, errors='coerce' will convert non-numeric data to NaN
                    df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                    print(f'For gene set {gene_set}, median R²: {df_metrics.loc["r2"].median()}, median Pearson r: {df_metrics.loc["pearson"].median()}, median MAE: {df_metrics.loc["mae"].median()}')
                # Plot gene set heatmaps
                for gene_set in heatmap_sets:
                    plot_double_heatmap(gene_lists[gene_set], bctan_mean, preds_df, name=f'results_patient/{direct}/{analysis}/{gene_set}_double_heatmap')
                    plot_roc_curve(gene_lists[gene_set], bctan_mean, preds_df, name=f'results_patient/{direct}/{analysis}/{gene_set}_ROC')
            elif analysis == 'SCLC':
                # Create sample distributions, true and predicted, for examining transformations
                plot_samples(preds_df, f'results_patient/{direct}/{analysis}/predicted-expression-profiles_raw')
                plot_samples(sclcpatient_mean, f'results_patient/{direct}/{analysis}/truth-expression-profiles_raw')
                preds_df.to_csv(f'results_patient/{direct}/{analysis}/predictions_tumor.tsv', sep='\t')
                sclcpatient_mean.to_csv(f'results_patient/{direct}/{analysis}/predictions_truth.tsv', sep='\t')
                heatmap_sets = ['CCP', 'Targets_SCLC']
                targets = gene_lists['Targets_SCLC'][0]
                if run_gsea:
                    gseas = ['ARG.10','NE.10', 'CCP.31']
                    plot_heatmap(gseas, gsea_pred, tfx_dict, name=f'results_patient/{direct}/{analysis}/GSEA_scores_heatmap', gsea=True, sub_dict=subtype_dict)
                    plot_double_heatmap(gseas, gsea_truth, gsea_pred, name=f'results_patient/{direct}/{analysis}/GSEA_scores_true-vs-pred_heatmap', gsea=True)
                    gsea_pred.to_csv(f'results_patient/{direct}/{analysis}/GSEA_scores.tsv', sep='\t')
                # Target gene scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/Target_scatters', exist_ok=True)
                for gene in targets:
                    if gene not in sclcpatient_mean.index or gene not in preds_df.index:
                        print(f'Gene {gene} not found in sclcpatient_mean or preds_df, skipping...')
                        continue
                    plot_vs_meta(sclcpatient_mean, preds_df, gene, gene, name=f'results_patient/{direct}/{analysis}/Target_scatters/{gene}_pred-vs-biopsy_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                # Sample-wise scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters', exist_ok=True)
                for gene_set in ['all', 'Targets_SCLC', 'Non-Housekeeping', 'DruggableGenome', 'CancerSurfaceome']:
                    # Create a new DataFrame df_metrics with the same columns as preds_df and rows for r2, pearson, and intercept
                    # r2_mae_plot(sclcpatient_mean, preds_df, tfx_dict, gene_lists[gene_set], name=f'results_patient/{direct}/{analysis}/{gene_set}_r2_mae_plot', mode='TFX')
                    df_metrics = pd.DataFrame(index=['r2', 'pearson', 'mae'], columns=preds_df.columns)
                    for col in preds_df.columns:
                        if col not in sclcpatient_mean.columns:
                            continue
                        if gene_set == 'all':
                            goi_dict = {'POU2F3': ['POU2F3'],
                                        'ASCL1': ['ASCL1'],
                                        'NEUROD1': ['NEUROD1'],
                                        'ATOH1': ['ATOH1']}
                            r2, pearson, mae = scatter_plot(sclcpatient_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter', goi_dict=goi_dict)
                        else:
                            r2, pearson, mae = scatter_plot(sclcpatient_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter')
                        # Fill up df_metrics
                        df_metrics.loc['r2', col] = r2
                        df_metrics.loc['pearson', col] = pearson
                        df_metrics.loc['mae', col] = mae
                    # Convert df_metrics to numeric, errors='coerce' will convert non-numeric data to NaN
                    df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                    print(f'For gene set {gene_set}, median R²: {df_metrics.loc["r2"].median()}, median Pearson r: {df_metrics.loc["pearson"].median()}, median MAE: {df_metrics.loc["mae"].median()}')
                    # Plot gene set heatmaps
                for gene_set in heatmap_sets:
                    plot_double_heatmap(gene_lists[gene_set], sclcpatient_mean, preds_df, name=f'results_patient/{direct}/{analysis}/{gene_set}_double_heatmap')
                    plot_roc_curve(gene_lists[gene_set], sclcpatient_mean, preds_df, name=f'results_patient/{direct}/{analysis}/{gene_set}_ROC')
            elif analysis == 'WCDT':
                # Create sample distributions, true and predicted, for examining transformations
                plot_samples(preds_df, f'results_patient/{direct}/{analysis}/predicted-expression-profiles_raw')
                plot_samples(wcdt_truth, f'results_patient/{direct}/{analysis}/truth-expression-profiles_raw')
                preds_df.to_csv(f'results_patient/{direct}/{analysis}/predictions_tumor.tsv', sep='\t')
                wcdt_truth.to_csv(f'results_patient/{direct}/{analysis}/predictions_truth.tsv', sep='\t')
                heatmap_sets = ['CCP', 'Targets_PC', 'PAM50', 'PC-Phenotyping']
                targets = gene_lists['Targets_PC'][0]
                if run_gsea:
                    gseas = ['ARG.10', 'NE.10', 'CCP.31']
                    plot_heatmap(gseas, gsea_pred, tfx_dict, name=f'results_patient/{direct}/{analysis}/GSEA_scores_heatmap', gsea=True, sub_dict=subtype_dict)
                    plot_double_heatmap(gseas, gsea_truth, gsea_pred, name=f'results_patient/{direct}/{analysis}/GSEA_scores_true-vs-pred_heatmap', gsea=True)
                    gsea_pred.to_csv(f'results_patient/{direct}/{analysis}/GSEA_scores.tsv', sep='\t')
                    for psub in ['LumA', 'LumB', 'Basal']:
                        plot_vs_meta(pam50_truth, pam50_pred, psub, psub, x_unit='Score (RNA-Seq)', y_unit='Score (Predicted)', name=f'results_patient/{direct}/{analysis}/{psub}_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                    for gsea in gseas:
                        plot_vs_meta(gsea_truth, gsea_pred, gsea, gsea, x_unit='Score (RNA-Seq)', y_unit='Score (Predicted)', name=f'results_patient/{direct}/{analysis}/{gsea}_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                # Target scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/Target_scatters', exist_ok=True)
                for gene in targets:
                    plot_vs_meta(wcdt_truth, preds_df, gene, gene, name=f'results_patient/{direct}/{analysis}/Target_scatters/{gene}_pred-vs-biopsy_scatter', tfx_dict=tfx_dict, subtypes=subtype_df)
                # Sample-wise scatters
                os.makedirs(f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters', exist_ok=True)
                for gene_set in ['all', 'Targets_PC', 'PC-Phenotyping', 'Non-Housekeeping', 'DruggableGenome', 'CancerSurfaceome']:
                    # Create a new DataFrame df_metrics with the same columns as preds_df and rows for r2, pearson, and intercept
                    df_metrics = pd.DataFrame(index=['r2', 'pearson', 'mae'], columns=preds_df.columns)
                    for col in preds_df.columns:
                        if col not in wcdt_truth.columns:
                            continue
                        if gene_set == 'all':
                            goi_dict = {'ARPC': ['AR', 'NKX3-1', 'FOLH1', 'KLK3'],
                                        'NEPC': ['ASCL1', 'CHGA', 'SOX2', 'SYP'],}
                            r2, pearson, mae = scatter_plot(wcdt_truth[col], preds_df[col], genes_subset=gene_lists[gene_set], name=f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter', goi_dict=goi_dict)
                        else:
                            r2, pearson, mae = scatter_plot(wcdt_truth[col], preds_df[col], genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results_patient/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter')
                        # Fill up df_metrics
                        df_metrics.loc['r2', col] = r2
                        df_metrics.loc['pearson', col] = pearson
                        df_metrics.loc['mae', col] = mae
                    # Convert df_metrics to numeric, errors='coerce' will convert non-numeric data to NaN
                    df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                    print(f'For gene set {gene_set}, median R²: {df_metrics.loc["r2"].median()}, median Pearson r: {df_metrics.loc["pearson"].median()}, median MAE: {df_metrics.loc["mae"].median()}')
                for gene_set in heatmap_sets:
                    plot_double_heatmap(gene_lists[gene_set], wcdt_truth, preds_df, name=f'results_patient/{direct}/{analysis}/{gene_set}_double_heatmap')
                    plot_roc_curve(gene_lists[gene_set], wcdt_truth, preds_df, name=f'results_patient/{direct}/{analysis}/{gene_set}_ROC')
            elif analysis == 'UW':
                # Create sample distributions, true and predicted, for examining transformations
                plot_samples(preds_df, f'results_patient/{direct}/{analysis}/predicted-expression-profiles_raw')
                preds_df.to_csv(f'results_patient/{direct}/{analysis}/predictions_tumor.tsv', sep='\t')
                plot_vs_meta(uw_meta, preds_df, 'LogPSA', 'KLK3', x_unit='Log2 Serum PSA (ng/ml)', name=f'results_patient/{direct}/{analysis}/KLK3pred-vs-SerumPSA_scatter', tfx_dict=tfx_dict)
                heatmap_sets = ['CCP', 'Targets_PC', 'PAM50', 'PC-Phenotyping']
                targets = gene_lists['Targets_PC'][0]
                if run_gsea:
                    gseas = ['ARG.10', 'NE.10', 'CCP.31']
                    plot_heatmap(gseas, gsea_pred, tfx_dict, name=f'results_patient/{direct}/{analysis}/GSEA_scores_heatmap', gsea=True, sub_dict=subtype_dict)
                    gsea_pred.to_csv(f'results_patient/{direct}/{analysis}/GSEA_scores.tsv', sep='\t')
                for hset in ['CCP', 'Targets_PC', 'PAM50', 'PC-Phenotyping']:
                    plot_heatmap(gene_lists[hset], preds_df, tfx_dict, name=f'results_patient/{direct}/{analysis}/{hset}_heatmap', sub_dict=subtype_dict)



if __name__ == "__main__":
    main()
