"""
run_analysis.py
v1.2.0, 5/24/2024
Author: Robert D Patton, rpatton@fredhutch.org
Ha Lab, Fred Hutchinson Cancer Research Center

This script runs a complete analysis of PROTEUS and its outputs
### Make sure to run GenerateScores.R in GeneScores before running this script ###
"""

# Standard library imports
import os
import csv

# Third party imports
import pandas as pd

# Local application imports
from analysis_utils import *

def main():
    # results_training folders to run LuCaP / SCLC analysis on
    validation_analyses = ['GeneStrat_0x_0CPM']
                        #    'SampleStrat_0x_0CPM',
                        #    'PC-Only_0x_0CPM',
                        #    'SCLC-Only_0x_0CPM']
    # patient sets to run (header: sample   tfx subtype)
    patient_analyses = {'PC-TAN': 'PC-TAN_TFX_Subtype.tsv',
                        'BC-TAN': 'BC-TAN_TFX_Subtype.tsv',
                        'PSMA': 'PSMA_TFX_Subtype.tsv'}
    patient_results = ['NUM-TT']  # 'TPU-0x_0CPM', 'NUM
    # load reference gene lists
    gene_lists = {'all': None,
                  'CCP': ['CCP31.txt'],
                  'Targets_PC': ['Targets_PC.txt'],
                  'Targets_SCLC': ['Targets_SCLC.txt'],
                  'Targets_BC': ['Targets_BC.txt'],
                  'PAM50': ['PAM50.txt'],
                  'EMT': ['EMT.txt'],
                  'Hypoxia': ['Hypoxia.txt'],
                  'PC-Phenotyping': [[8, 14], 'PC_PhenotypingGenes.txt'],
                  'BC-Phenotyping': ['DF-List', 'BC_MolecularSubtypes.txt'],
                  'PSMA': ['PSMA-high.txt', 'PSMA-low.txt'],
                  'LuCaP-vs-WholeBlood': ['LuCaPvsGTEx_up-regulated.txt', 'LuCaPvsGTEx_down-regulated.txt'],
                  'SCLC-vs-WholeBlood': ['SCLCvsGTEx_up-regulated.txt', 'SCLCvsGTEx_down-regulated.txt'],
                  'StaticTraining': ['static_training.txt'],
                  'VariableTraining': ['variable_training.txt'],}
    set_names = {'all': None, 'Targets_PC': None, 'Targets_SCLC': None, 'Targets_BC': None, 'StaticTraining': None, 'VariableTraining': None,
                 'PC-Phenotyping': ['NEURO I', 'NEURO II', 'AR'],
                 'BC-Phenotyping': ['UC', 'PUC', 'NE', 'SQ'],
                 'PSMA': ['PSMA-high', 'PSMA-low'],
                 'LuCaP-vs-WholeBlood': ['LuCaP-vs-WholeBlood up-regulated', 'LuCaP-vs-WholeBlood down-regulated'],
                 'SCLC-vs-WholeBlood': ['SCLC-vs-WholeBlood up-regulated', 'SCLC-vs-WholeBlood down-regulated']}
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

    print('Loading reference data...')
    # load ground truth /reference data
    lucap_truth = pd.read_csv('../labels/LuCaP_Log2CPMplus1.tsv', sep='\t', index_col=0)
    lucap_meta = pd.read_csv('../reference/LuCaP_metadata.tsv', sep='\t', index_col=0).T
    sclc_truth = pd.read_csv('../labels/SCLC_Log2CPMplus1.tsv', sep='\t', index_col=0)
    sclc_meta = pd.read_csv('../reference/SCLC_metadata.tsv', sep='\t', index_col=0).T
    pc_tan_mean = pd.read_csv('../labels/PC_TAN_Log2CPMplus1.tsv', sep='\t', index_col=0)
    pctan_meta = pd.read_csv('../reference/PC-TAN_IHC-Scores.tsv', sep='\t', index_col=0).T
    bc_tan_mean = pd.read_csv('../labels/BC_TAN_Log2CPMplus1.tsv', sep='\t', index_col=0)
    bctan_meta = pd.read_csv('../reference/BC-TAN_IHC-Scores.tsv', sep='\t', index_col=0)
    bctan_meta.index = bctan_meta.index.str.slice(0, 6)
    bctan_meta = bctan_meta.groupby(level=0).mean().T

    ############################################################################################################
    # Run training validation analyses
    ############################################################################################################
    print('*** Running training validation analyses.')
    for analysis in validation_analyses:
        """ For each use_set / training regime:
        1. Load the predictions for each fold, found in ../results_training/analysis/CV-fold_*/validation-fold_*_predictions.tsv
        2. Merge all folds into one prediction df for lucap and one for sclc, corresponding to the truth dfs
        3. Makes results/analysis/LuCaP, /SCLC, and /Combined folders if they don't exist and save the merged predictions there
        4. Run the following analyses:
        ### Run combined analyses
        - Plot scatter of truth vs pred for all genes and all samples
        ### Run LuCaP-specific analyses:
        - Plot combined scatter of truth vs pred for all samples and all genes, as well as  for [PC-Phenotyping, LuCaP-vs-WholeBlood]
        - Plot two-figure heatmaps for truth and pred for [CCP, Targets, PAM50, EMT, Hypoxia, PC-Phenotyping, LuCaP-vs-WholeBlood]
        - Compute R^2 and MAE for each sample in both all genes, [PC-Phenotyping, LuCaP-vs-WholeBlood] and plot those
          metrics vs depth (from lucap_ref)
        - Plot sample-wise scatters of truth vs pred CCP-score vs CCP-mean, PSMA-score vs PSMA-mean, and PSMA-score vs FOLH1 (PSMA) prediction 
        ### Run SCLC-specific analyses
        - Plot combined scatter of truth vs pred for all samples and all genes, as well as  for [SCLC-vs-WholeBlood]
        - Plot two-figure heatmaps for truth and pred for [CCP, Targets, PAM50, EMT, Hypoxia, SCLC-vs-WholeBlood]
        """
        print(f'##### Running {analysis} combined validation analyses.')
        # Load predictions
        cv_preds = [pd.read_csv(f'../results_training/BioCNN_{analysis}/CV-fold_{fold}/validation-fold_{fold}_predictions.tsv', sep='\t', index_col=0) for fold in range(1, 6)]
        ho_samples = [name for name in os.listdir(f'../results_training/BioCNN_{analysis}') if "CV" not in name and os.path.isdir(os.path.join(f'../results_training/BioCNN_{analysis}', name))]
        ho_preds = {sample: pd.read_csv(f'../results_training/BioCNN_{analysis}/{sample}/validation-{sample}_predictions.tsv', sep='\t', index_col=0) for sample in ho_samples}
        df_ho = pd.concat([ho_preds[sample]['prediction'].rename(sample) for sample in ho_preds], axis=1)
        merged_truth = lucap_truth.merge(sclc_truth, how='outer', left_index=True, right_index=True)
        df_val = merge_validation(cv_preds)
        #### REMOVE LuCaP damples with "depth" < 5x ####
        samples_to_remove = lucap_meta.columns[lucap_meta.loc['depth'] < 5.0]
        df_val = df_val.drop(columns=samples_to_remove, errors='ignore')
        df_ho = df_ho.drop(columns=samples_to_remove, errors='ignore')
        os.makedirs(f'results/{analysis}', exist_ok=True)
        os.makedirs(f'results/{analysis}/Combined', exist_ok=True)
        # Plot scatter of truth vs pred for all genes and all samples
        scatter_plot(merged_truth, df_val, name=f'results/{analysis}/Combined/true-vs-pred_val_scatter')
        scatter_plot(merged_truth, df_ho, name=f'results/{analysis}/Combined/true-vs-pred_ho_scatter')
        for ho_sample in ho_samples:
            scatter_plot(merged_truth, df_ho[[ho_sample]], name=f'results/{analysis}/Combined/{ho_sample}_true-vs-pred_scatter')
        df_val = pd.concat([df_val, df_ho], axis=1)
        combined_preds_path = f'results/{analysis}/Combined/predictions.tsv'
        df_val.to_csv(combined_preds_path, sep='\t')
        scatter_plot(merged_truth, df_val, name=f'results/{analysis}/Combined/true-vs-pred_all-training_scatter')
        ##### Find the best and worst genes based on MAE and save them for DAVID analysis #####
        # from sklearn.metrics import mean_squared_error

        # # Calculate MSE for each gene
        # mse_values = {}
        # for gene in merged_truth.index:
        #     if gene in df_val.index:
        #         truth_values = merged_truth.loc[[gene]].dropna(axis=1)
        #         pred_values = df_val.loc[[gene]].dropna(axis=1)
        #         common_samples = truth_values.columns.intersection(pred_values.columns)
        #         mse_values[gene] = mean_squared_error(truth_values[common_samples].values[0], pred_values[common_samples].values[0])

        # # Sort genes by MSE
        # sorted_genes = sorted(mse_values.items(), key=lambda x: x[1])

        # # Get the top and bottom 10% of genes
        # num_genes = len(sorted_genes)
        # top_10_percent = sorted_genes[:int(num_genes * 0.1)]
        # bottom_10_percent = sorted_genes[-int(num_genes * 0.1):]

        # # Save the top and bottom 10% of genes to separate .txt files
        # with open(f'results/{analysis}/Combined/top10_genes.txt', 'w') as f:
        #     for gene, mse in top_10_percent:
        #         f.write(f'{gene}\n')

        # with open(f'results/{analysis}/Combined/bot10_genes.txt', 'w') as f:
        #     for gene, mse in bottom_10_percent:
        #         f.write(f'{gene}\n')
        ########################################################################################
        # Run tissue-specific analyses
        for pdx in ['LuCaP', 'SCLC']:
            print(f'##### Running analyses for {analysis} in {pdx}.')
            os.makedirs(f'results/{analysis}/{pdx}', exist_ok=True)
            if pdx == 'LuCaP':
                pdx_truth = lucap_truth
                pdx_meta = lucap_meta
                pdx_subtypes = lucap_meta.loc['phenotype']
                pdx_pam50 = lucap_meta.loc['PAM50']
                scatter_sets = ['all', 'LuCaP-vs-WholeBlood', 'PC-Phenotyping', 'Targets_PC', 'StaticTraining', 'VariableTraining']
                heatmap_sets = ['CCP', 'Targets_PC', 'PAM50', 'PC-Phenotyping']
                ihc_pairs = [('PSMA-mean', 'FOLH1')]
                targets = gene_lists['Targets_PC'][0]
                gseas = ['ARG.10', 'NE.10', 'CCP.31', 'EMT', 'Hypoxia', 'PSMA-high', 'PSMA-low']
            elif pdx == 'SCLC':
                pdx_truth = sclc_truth
                pdx_meta = sclc_meta
                pdx_subtypes = sclc_meta.loc['subtype']
                pdx_pam50 = None
                scatter_sets = ['all', 'SCLC-vs-WholeBlood', 'Targets_SCLC', 'StaticTraining', 'VariableTraining']
                heatmap_sets = ['CCP', 'Targets_SCLC']
                ihc_pairs = []
                targets = gene_lists['Targets_SCLC'][0]
                gseas = ['CCP.31', 'Hypoxia', 'Fanconi-Anemia']
            pdx_samples = df_val.columns.intersection(pdx_truth.columns)
            pdx_val = df_val.loc[:, pdx_samples]
            pdx_truth = pdx_truth.loc[:, pdx_samples]
            # Combined samples' scatter plots (including group, gene, and against IHC scores) and heatmaps
            for gene_set in scatter_sets:
                scatter_plot(pdx_truth, pdx_val, genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results/{analysis}/{pdx}/{gene_set}_true-vs-pred_scatter')
                if pdx == 'SCLC':  # NO DEPTH INFO - YET!
                    continue
                r2_mae_plot(pdx_truth, pdx_val, pdx_meta, gene_lists[gene_set], name=f'results/{analysis}/{pdx}/{gene_set}_r2_mae_plot')
            for gene_set in heatmap_sets:
                plot_double_heatmap(gene_lists[gene_set], pdx_truth, pdx_val, name=f'results/{analysis}/{pdx}/{gene_set}_true-vs-pred_heatmap', subtypes=pdx_subtypes)
            os.makedirs(f'results/{analysis}/{pdx}/IHC_scatters', exist_ok=True)
            for ihc_pair in ihc_pairs:
                plot_vs_meta(pdx_meta, pdx_val, ihc_pair[0], ihc_pair[1], x_unit='IHC Score', name=f'results/{analysis}/{pdx}/IHC_scatters/{ihc_pair[1]}-pred-vs-{ihc_pair[0]}_scatter', subtypes=pdx_subtypes)
                plot_vs_meta(pdx_meta, pdx_truth, ihc_pair[0], ihc_pair[1], x_unit='IHC Score', y_unit='RNA-Seq Log\u2082(CPM+1)', name=f'results/{analysis}/{pdx}/IHC_scatters/{ihc_pair[1]}-truth-vs-{ihc_pair[0]}_scatter', subtypes=pdx_subtypes)
            os.makedirs(f'results/{analysis}/{pdx}/Target_scatters', exist_ok=True)
            for gene in targets:
                plot_vs_meta(pdx_truth, pdx_val, gene, gene, name=f'results/{analysis}/{pdx}/Target_scatters/{gene}_true-vs-pred_scatter', subtypes=pdx_subtypes)
            # GSVA and PAM50 heatmaps/scatters
            gsva_truth = pd.read_csv(f'GeneScores/{pdx}_Truth_gsva.sig.es.obs.txt', sep='\t', index_col=0)
            gsva_pred = pd.read_csv(f'GeneScores/{pdx}_Predicted_gsva.sig.es.obs.txt', sep='\t', index_col=0)
            pam50_truth = pd.read_csv(f'GeneScores/{pdx}_Truth_pam50scores.txt', sep='\t', index_col=0).T
            pam50_pred = pd.read_csv(f'GeneScores/{pdx}_Predicted_pam50scores.txt', sep='\t', index_col=0).T
            # Select only the 'Basal', 'LumA', and 'LumB' rows of PAM50
            pam50_truth = pam50_truth.loc[['Basal', 'LumA', 'LumB'], :]
            pam50_pred = pam50_pred.loc[['Basal', 'LumA', 'LumB'], :]
            os.makedirs(f'results/{analysis}/{pdx}/GSEA_PAM50', exist_ok=True)
            # plot scores as scatter plots
            if pdx_pam50 is not None:  # do not know PAM50 truth for SCLC data
                for psub in ['LumA', 'LumB', 'Basal']:
                    plot_vs_meta(pam50_truth, pam50_pred, psub, psub, x_unit='Score (RNA-Seq)', y_unit='Score (Predicted)', name=f'results/{analysis}/{pdx}/GSEA_PAM50/{psub}_scatter', subtypes=pdx_pam50)
            # Restrict the GSEA scores to the groups in gseas
            gsva_truth = gsva_truth.loc[gseas, pdx_samples]
            gsva_pred = gsva_pred.loc[gseas, pdx_samples]
            for gsea in gseas:
                plot_vs_meta(gsva_truth, gsva_pred, gsea, gsea, x_unit='Score (RNA-Seq)', y_unit='Score (Predicted)', name=f'results/{analysis}/{pdx}/GSEA_PAM50/{gsea}_scatter', subtypes=pdx_subtypes)
            # make a dictionary of lucap subtypes, mapping every sample in common_samples to 1.0
            tfx_dict = {sample: 1.0 for sample in pdx_samples}
            plot_double_heatmap(gsva_pred.index.tolist(), gsva_truth, gsva_pred, name=f'results/{analysis}/{pdx}/GSEA_PAM50/GSVA-Scores_true-vs-pred_heatmap', gsea=True, subtypes=pdx_subtypes)
            if pdx_pam50 is not None:  # do not know PAM50 truth for SCLC data
                plot_double_heatmap(pam50_pred.index.tolist(), pam50_truth, pam50_pred, name=f'results/{analysis}/{pdx}/GSEA_PAM50/PAM50-Scores_true-vs-pred_heatmap', gsea=True, subtypes=pdx_pam50)


    ############################################################################################################
    # Run patient analyses
    ############################################################################################################
    print('*** Running patient analyses.')
    for direct in patient_results:
        for analysis in patient_analyses:
            """ For each patient set:
            1. Load the predictions, found in ../results_holdout/direct/analysis_data/predictions.tsv
            2. Make results/analysis folder if it doesn't exist for outputs
            3. Run the following analyses:
            - Plot heatmaps for [CCP, Targets, PAM50, PC-Phenotyping, PSMA, etc.] (+ TFX bars)
            - Plot GSEA scores as (identically sorted) heatmaps (+ TFX bars)
            - Save GSEA scores as a TSV file
            ### if analysis == 'UW-WGS':
            - Trying both fit-to-biopsy  and biopsy-avg:
            plot scatter of truth vs pred for matching biopsy sets on all genes as well as [Targets, PC-Phenotyping, LuCaP-vs-WholeBlood]
            (and also print TFX on the scatter)
            """
            print(f'##### Running analyses for {analysis} using {direct} model.')
            # Load data
            preds_df = pd.read_csv(f'../results_holdout/{direct}/{analysis}_data/predictions.tsv', sep='\t', index_col=0)
            if "TAN" in analysis:
                # Restrict sample names to the first 6 characters (patient ID)
                preds_df.columns = [col[:6] for col in preds_df.columns]
            # Load TFX, Subtypes
            with open(f'../reference/{patient_analyses[analysis]}', 'r') as file:
                reader = csv.reader(file, delimiter='\t')
                next(reader, None)  # Skip the header
                rows = list(reader)  # Store the rows in a list
                tfx_dict = {row[0]: float(row[1]) for row in rows}
                subtype_dict = {row[0]: row[2] for row in rows}
            ###
            # preds_df = subtract_background(preds_df, tfx_dict, 'Whole_Blood', '../labels/GTEx_Log2CPMplus1.tsv')
            ###
            gsea_pred = pd.read_csv(f'GeneScores/{analysis}_Predicted_gsva.sig.es.obs.txt', sep='\t', index_col=0)
            gsea_truth = pd.read_csv(f'GeneScores/{analysis}_Truth_gsva.sig.es.obs.txt', sep='\t', index_col=0)
            pam50_pred = pd.read_csv(f'GeneScores/{analysis}_Predicted_pam50scores.txt', sep='\t', index_col=0).T
            pam50_truth = pd.read_csv(f'GeneScores/{analysis}_Truth_pam50scores.txt', sep='\t', index_col=0).T
            os.makedirs(f'results/{direct}/{analysis}', exist_ok=True)
            plot_samples(preds_df, f'results/{direct}/{analysis}/sample-expression-profiles')
            # Perform cohort-specific analyses:
            if analysis == 'PSMA':
                continue
            elif analysis == 'PC-TAN':
                # Drop "Unknown" samples from subtype_dict
                subtype_dict = {sample: subtype for sample, subtype in subtype_dict.items() if subtype != 'Unknown'}
                heatmap_sets = ['CCP', 'Targets_PC', 'PAM50', 'PC-Phenotyping']
                targets = gene_lists['Targets_PC'][0]
                gseas = ['ARG.10', 'NE.10', 'CCP.31', 'EMT', 'Hypoxia', 'PSMA-high', 'PSMA-low']
                plot_heatmap(gseas, gsea_pred, tfx_dict, name=f'results/{direct}/{analysis}/GSEA_scores_heatmap', gsea=True, sub_dict=subtype_dict)
                plot_double_heatmap(gseas, gsea_truth, gsea_pred, name=f'results/{direct}/{analysis}/GSEA_scores_true-vs-pred_heatmap', gsea=True)
                gsea_pred.to_csv(f'results/{direct}/{analysis}/GSEA_scores.tsv', sep='\t')
                # IHC and target scatters
                os.makedirs(f'results/{direct}/{analysis}/IHC_scatters', exist_ok=True)
                os.makedirs(f'results/{direct}/{analysis}/Target_scatters', exist_ok=True)
                plot_vs_meta(pctan_meta, preds_df, 'PSMA_mean', 'FOLH1',  x_unit='IHC Score', name=f'results/{direct}/{analysis}/IHC_scatters/FOLH1-vs-PSMA-mean_scatter', tfx_dict=tfx_dict)
                plot_vs_meta(pctan_meta, preds_df, 'PSMA_median', 'FOLH1', x_unit='IHC Score', name=f'results/{direct}/{analysis}/IHC_scatters/FOLH1-vs-PSMA-med_scatter', tfx_dict=tfx_dict)
                for gene in ['AR', 'NKX3-1', 'SYP', 'INSM1', 'ASCL1', 'SOX2', 'FOXA2', 'MKI67']:
                    # TODO: print total IHC metrics?
                    plot_vs_meta(pctan_meta, preds_df, gene + '_mean', gene, x_unit='IHC Score', name=f'results/{direct}/{analysis}/IHC_scatters/{gene}-vs-IHC-mean_scatter', tfx_dict=tfx_dict)
                    plot_vs_meta(pctan_meta, preds_df, gene + '_median', gene, x_unit='IHC Score', name=f'results/{direct}/{analysis}/IHC_scatters/{gene}-vs-IHC-median_scatter', tfx_dict=tfx_dict)
                for gene in targets:
                    plot_vs_meta(pc_tan_mean, preds_df, gene, gene, name=f'results/{direct}/{analysis}/Target_scatters/{gene}_pred-vs-biopsy_scatter', tfx_dict=tfx_dict)
                # Sample-wise scatters
                os.makedirs(f'results/{direct}/{analysis}/Sample-wise_gene-set_scatters', exist_ok=True)
                for gene_set in ['all', 'Targets_PC', 'PC-Phenotyping', 'StaticTraining', 'VariableTraining']:
                    # Create a new DataFrame df_metrics with the same columns as preds_df and rows for r2, slope, and intercept
                    r2_mae_plot(pc_tan_mean, preds_df, tfx_dict, gene_lists[gene_set], name=f'results/{direct}/{analysis}/{gene_set}_r2_mae_plot', mode='TFX')
                    df_metrics = pd.DataFrame(index=['r2', 'slope', 'mae'], columns=preds_df.columns)
                    for col in preds_df.columns:
                        if col not in pc_tan_mean.columns:
                            continue
                        if gene_set == 'all':
                            goi_dict = {'ARPC': ['AR', 'KLK2', 'KLK3', 'NKX3-1', 'TMPRSS2', 'PMEPA1'],
                                        'NEPC': ['SYP', 'CHGA', 'SCG3', 'ASCL1', 'INSM1', 'ACTL6B']}
                            r2, slope, mae = scatter_plot(pc_tan_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], name=f'results/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter', goi_dict=goi_dict)
                        else:
                            r2, slope, mae = scatter_plot(pc_tan_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter')
                        # Fill up df_metrics
                        df_metrics.loc['r2', col] = r2
                        df_metrics.loc['slope', col] = slope
                        df_metrics.loc['mae', col] = mae
                    # Convert df_metrics to numeric, errors='coerce' will convert non-numeric data to NaN
                    df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                    print(f'For gene set {gene_set}, median R²: {df_metrics.loc["r2"].median()}, mean R²: {df_metrics.loc["r2"].mean()}, mean MAE: {df_metrics.loc["mae"].mean()}')
                    if gene_set not in ['all', 'StaticTraining', 'VariableTraining']:
                        plot_double_heatmap(gene_lists[gene_set], pc_tan_mean, preds_df, name=f'results/{direct}/{analysis}/{gene_set}_double_heatmap')
            elif analysis == 'BC-TAN':
                # Restrict tfx_dict and subtype_dict to deep samples, and rename to patient IDs
                tfx_dict = {sample[:6]: tfx for sample, tfx in tfx_dict.items() if 'Deep' in sample}
                subtype_dict = {sample[:6]: subtype for sample, subtype in subtype_dict.items() if 'Deep' in sample}
                heatmap_sets = ['CCP', 'Targets_BC', 'PAM50', 'BC-Phenotyping']
                targets = gene_lists['Targets_BC'][0]
                gseas = ['UC', 'PUC', 'SQ', 'NE', 'CCP.31', 'EMT', 'Hypoxia', 'Fanconi-Anemia']
                plot_heatmap(gseas, gsea_pred, tfx_dict, name=f'results/{direct}/{analysis}/GSEA_scores_heatmap', gsea=True, sub_dict=subtype_dict)
                plot_double_heatmap(gseas, gsea_truth, gsea_pred, name=f'results/{direct}/{analysis}/GSEA_scores_true-vs-pred_heatmap', gsea=True)
                gsea_pred.to_csv(f'results/{direct}/{analysis}/GSEA_scores.tsv', sep='\t')
                # IHC scatters
                os.makedirs(f'results/{direct}/{analysis}/IHC_scatters', exist_ok=True)
                for gene in ['NECTIN4', 'TACSTD2']:
                    plot_vs_meta(bctan_meta, preds_df, gene + '_membrane', gene, x_unit='IHC Score', name=f'results/{direct}/{analysis}/IHC_scatters/{gene}-vs-IHC-membrane_scatter', tfx_dict=tfx_dict)
                    plot_vs_meta(bctan_meta, preds_df, gene + '_cytoplasm', gene, x_unit='IHC Score', name=f'results/{direct}/{analysis}/IHC_scatters/{gene}-vs-IHC-cytoplasm_scatter', tfx_dict=tfx_dict)
                # Target gene scatters
                os.makedirs(f'results/{direct}/{analysis}/Target_scatters', exist_ok=True)
                for gene in targets:
                    if gene == 'NECTIN4': # N.B. ../../RNA-Seq/BC_TAN/Raw_values_91samples_19712genes_4_7_2021.txt DOES NOT CONTAIN NECTIN4
                        continue
                    plot_vs_meta(bc_tan_mean, preds_df, gene, gene, name=f'results/{direct}/{analysis}/Target_scatters/{gene}_pred-vs-biopsy_scatter', tfx_dict=tfx_dict)
                # Sample-wise scatters
                os.makedirs(f'results/{direct}/{analysis}/Sample-wise_gene-set_scatters', exist_ok=True)
                for gene_set in ['all', 'Targets_BC', 'BC-Phenotyping', 'StaticTraining', 'VariableTraining']:
                    # Create a new DataFrame df_metrics with the same columns as preds_df and rows for r2, slope, and intercept
                    r2_mae_plot(bc_tan_mean, preds_df, tfx_dict, gene_lists[gene_set], name=f'results/{direct}/{analysis}/{gene_set}_r2_mae_plot', mode='TFX')
                    df_metrics = pd.DataFrame(index=['r2', 'slope', 'mae'], columns=preds_df.columns)
                    for col in preds_df.columns:
                        if col not in bc_tan_mean.columns:
                            continue
                        r2, slope, mae = scatter_plot(bc_tan_mean[col], preds_df[col], genes_subset=gene_lists[gene_set], subset_names=set_names[gene_set], name=f'results/{direct}/{analysis}/Sample-wise_gene-set_scatters/{gene_set}_{col}_true-fit-vs-pred_scatter')
                        # Fill up df_metrics
                        df_metrics.loc['r2', col] = r2
                        df_metrics.loc['slope', col] = slope
                        df_metrics.loc['mae', col] = mae
                    # Convert df_metrics to numeric, errors='coerce' will convert non-numeric data to NaN
                    df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                    print(f'For gene set {gene_set}, median R²: {df_metrics.loc["r2"].median()}, mean R²: {df_metrics.loc["r2"].mean()}, mean MAE: {df_metrics.loc["mae"].mean()}')
                    # Plot gene set heatmaps
                    if gene_set not in ['all', 'StaticTraining', 'VariableTraining']:
                        plot_double_heatmap(gene_lists[gene_set], bc_tan_mean, preds_df, name=f'results/{direct}/{analysis}/{gene_set}_double_heatmap')


if __name__ == "__main__":
    main()
