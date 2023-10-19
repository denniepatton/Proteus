# Proteus <img src="misc/logo_v1.png" width="140" align="left">
A CNN-based methodology for predicting gene expression and activity from cell free DNA (cfDNA), Proteus uses biomarkers and signals output by [Triton](https://github.com/denniepatton/Triton) to infer coverage, nucleosome occupancy, sequence context, and fragmentation profiles from regions of interest.
<br/><br/>


# TritonCNNv3
This is the newest working repo for the 3rd version of TritonCNN, working title "Proteus" (Another Greek god / merman like Triton who is prophetic and changing). Until we have a finalized model and are cleaning up and testing, let's use this README to track jobs and status, at the coding level. More general ideas about architectures, larger tasks, and scientific analyses will be stored in the Ha Lab One Drive in TritonCNN/OngoingIdeas.docx, while that parent directory will hold figures, write-ups, etc. See [Triton](https://github.com/denniepatton/Triton) for specifics on input data formatting; TritonCNNv3 is currently setup to use default output data/directory formatting from Triton. Akira's previous README can be found below for reference - let's leave it in until it becomes obsolete.

## NOTES AND TO-DO
First read the REPO STRUCTURE section below.

It would be good to confirm the structure of this repo is following best practices, and correct it as needed. There may be subtlties to Tensorflow/deep repos we should adhere to!

For now, let's keep everything hard coded in TritonCNN.py, the primary wrapper, and assume runs are local with a GPU (e.g. use grabnode for now with no bash or snakemake submission protocol).

### Task 1: Flesh out TritonCNN.py (main wrapper)

The TritonCNN.py script is designed to be a wrapper for running pre-processing routines, training and testing models, and doing basic output analysis. Ultimately, the user should only have to run TritonCNN.py and nothing else in order to train, test, and validate the final model (e.g. paper ready). A larger method implementing SnakeMake or WDL for faster cloud computing plus a config folder should also be made, eventually. The main function should:
- check for the existence of pre-processed data in a pre-defined directory structure, and generate it by default if it is not found (by calling scripts in preprocessing)
- take in (hardcoded is fine for now) train/test splits and convert pre-processed data above into fast-access binary/image files with matched label data (by calling scripts in preprocessing)
- allow for the easy modification/selection of models, so that multiple models can be tested with varying parameters and tuning methods
- perform basic output analysis (plots of accuracy vs epochs, ROC, validation metrics, interesting plotting, sanity checks, visuals and interpretation of each model - contained in analysis/)
- output all results to a clearly labeled subfolder(s) in results/
Furthermore TritonCNN.py should do no heavy lifting, but only contain a main and wrapper functions which call the scripts in analysis / models / preprocessing

A. Pre-process data:
I've already implimented a main in the wrapper which contains some hardcoded paths and data modalities; calling generate_data() should pre-process all data in those modalities by calling preprocessing/generate_scalars.py and preprocessing/generate_images.py (made by Akira, already tested and working). Those scripts made by Akira need to be double-checked and cleaned up (see #TODO:'s), and a third script "generate_image-panels.py" needs to be made to process the 3D TFBS tensors, and then incorporated into generate_data().

B. Split pre-processed data and make easy-access:
This step was previously done by preprocessing/compile_data.py - let's make that script callable in the main, and move any inputs for it into the wrapper. A single function in TritonCNN.py should match labels via this script and make the outputs, possibly in a labeled sub folder in data/

C. Make splits and subset data:
This should be an easily modifiable function that calls OR ABSORBS preprocessing/create_smaller_set.py, cross-validation_splits.py, and train_test_split.py in order to select training sets from the data above.

D. Code in method for testing multiple models:
Eventually the TritonCNN.py script should take in inputs which define a specific model and training attributes, but for now let's just hard code a method for doing this within. This is similar to what Akira was doing with models/test_model.py in conjunction with models/models.json, but should make it easier to make the main a true script when everything is finished. When D is finished, the main should call at most a single config file.

E. Impliment basic output analysis:
Confirm all basic outputs we want are happening by default, including interpretibility work. Should call or absorb analysis/visualization_clusters_NYC.py and analysis/akira_analysis_notebook_NYC.pynb.

F. Streamline outputs:
Make sure each of the models (plus params) tested in D get their own sub directory in results/, which contains the model weights, output test predictions, model design and interpretation, and basic outputs from E.


## REPO STRUCTURE

### analysis
Will contain any and all scripts meant to process final predictions output by TritonCNNv3. At some point the .gitignore should be modified to only upload scripts from this folder, since final figures, tables, etc. should get moved to the working paper directory. I have not looked at the scripts inside in depth, but they will be used/called/absorbed in Task 1.E

### data
The folder itself should be kept on the repo; all contents are in the .gitignore as they are huge and private data. The TritonCNN.py wrapper (in progress) should automatically save parsed, processed Triton output data in subfolders labeled label_region_dtype, e.g. LuCaP_TSS_images. For now "LuCaP" is hardcoded, the region choices are TSS (Transcription Start Site | primary data used for images and scalars), GB (Gene Body | secondary data used only for scalars) and TFBS (Transcription Factor Binding Site | will be used as tertiary data image-panels, i.e. 3D tensors at the sample level) and the dtypes are scalars (scalar, region-level values from "TritonCompositeFM.tsv" Triton output), images (2000x11 signal matrices from "_TritonProfiles.npz" Triton output), and image-panels (UNCODED but will be Nx2000x11 image panels, also produced from "_TritonProfiles.npz").

### labels
Contains training labels, which are not generated by Triton. Currently contains matched labels for LuCaP lines:
- continuous_zfpkm.tsv (for expression training/prediction)
- continuous_fpkm.tsv (for expression training/prediction)
- continuous_tpm.tsv (for expression training/prediction)
- binary_zfpkm.tsv (for activity training/prediction)
We should try multiple continuous/binary expression types and see what works/has the most support in the literature. Currently scripts to convert fpkm to zFPKM, implemented in R, are in labels/rna_seq_analysis/. If we end up using zFPKM we should supply an easy-to-use python script that works in the same environment to get the same outputs, replacing these.

### misc
Contains miscellaneous files, e.g. to flesh out the README someday (currently only contains the new logo)

### models
Should ultimately contain ONLY models and scripts that run models, to be referenced by the main wrapper (TritonCNN.py)
- models.json: appears to contain info for testing multiple models at a time (see Task 1.C)
- classification_models.py: should contain all models for activity prediction
- gene_model_hpt.py: old? let's wait to delete for now
- genome_autoencoder.py: may need to be absorbed by utils.py or a similar master script of sub-architectures to be called by the main models scripts
- interpretation.py: ultimately move to analysis/, after cleaning up and using in Task 1.E
- regression_models.py: should contain all models for expression prediction
- run_model.py: *Akira's primary script for running models! Needs to be looked at for Task 1.D*
- test_model.py: to be absorbed/used in conjunction with Task 1.C
- utils.py: contains general utils called by models - should seperate out analysis/plotting functions and move to analysis/, leaving only model-related functions

### old_scripts
Previous scripts for TritonCNN which may be useful as a reference: we can delete this folder once everything is up and running.

### preprocessing
Should contain any scripts used for pre-processing data, including making splits and performing scale transformations
- comppile_data.py: matches data to labels and saves as fast access
- create_smaller_set.py: subsets data - should be absorbed by another script
- cross_validation_splits.py: makes split info (can be combined with train_test_split.py into something like "make_splits.py")
- generate_images.py: generates csv data from Triton npz signal outputs
- generate_scalars.py: generates csv data from Triton tsv scalar outputs
- train_test_split.py: see cross_validation_splits.py

### reference
Contains reference files used by TritonCNN
- HousekeepingGenes.txt: list of housekeeping genes for subsetting
- NonHousekeepingGenes.txt: list of non-housekeeping genes for subsetting
- Ensembl109_reference.bed: reference BED file (which should match version used by Triton) used for getting transcript length and exon count scalar labels

### results
Will be populated by outputs, but only this folder needs to go in the repo (not its innards)




# [cfDNA-gene-exp-inference](https://github.com/GavinHaLab/cfDNA-gene-exp-inference) / [production-cfDNA-gene-exp-inference](https://github.com/GavinHaLab/production-cfDNA-gene-exp-inference)
Akira Nair's extension of Triton CNN to infer gene expression in tumor cfDNA samples using neural networks.

### Run a model on existing data
The easiest way to train a model is using the `run_model.py` script located in the model folder. 
Many of the model training parameters can be adjusted using command line arguments. Here are the available arguments.
```
  -h, --help                    
            show this help message and exit
  
  -n NAME, --name NAME          
            Specify a name for the model (REQUIRED)
  
  -o OUTPUT, --output OUTPUT    
            Output folder for model
  
  --use_full_data               
            If true, will use the large dataset
  
  --model_version {v1,v2}       
            If specified, uses a pre-defined architecture
  
  --modalities {images,scalars,all}
            Which modalities to include in the model
  
  --task {0,2,3}        
            Number of classes to predict from. If 0, creates a regression model.
  
  --epochs EPOCHS       
            Number of epochs to run model for
  
  --batch_size BATCH_SIZE
            Batch size for training

  --learning_rate LEARNING_RATE
            Learning rate of model
  
  --n_filters N_FILTERS
            Number of filters used in convolution (must use image modality)
  
  --n_convolutions N_CONVOLUTIONS
            Number of convolution layers (must use image modality)
  
  --mean_normalize      
            If true, divides sequential data by mean for each stream
  
  --use_avg_pool        
            If true, use average pooling instead of max pooling
  
  --use_dropout         
            Use dropout layers
  
  --dropout_rate DROPOUT_RATE
            If using dropout, sets the dropout rate
  
  --use_attention       
            Use attention model

  --lsconv              
            Create two streams of convolution with long and short kernel lengths

  --seed SEED
            Global seed
```

To see a list of options, run
```sh
python model/run_model.py --help
```
Models can also be run using sbatch as well. 
Here is my recommended way of incorporating the arguments into a shell script called `run_model.sh`:
```sh
#!/bin/bash
#SBATCH --job-name=model
#SBATCH --partition=campus-new
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=10000
#SBATCH --time=1:0:0
#SBATCH -o run_model.out

module purge
cd /path/to/cfDNA-gene-exp-inference/model
source ~/.bashrc
# load relevant modules
ml Anaconda3/2022.05
ml Python/3.9.6-GCCcore-11.2.0
# activate environment
conda activate ENV_NAME
# run script
python3 run_model.py $@
```

Here are a few examples...
1. Run a regression model for 2 epochs.
```sh
sbatch model/run_model.sh --name "my_regression" --use_full_data --epochs 1 --task 0
```
2. Run a classification model that predicts between 3 classes for 10 epochs.
```sh
sbatch model/run_model.sh --name "three_class" --use_full_data --epochs 1 --task 3
```
3. Run an attention classification model that predicts between 2 classes for 5 epochs and with a learning rate of 0.0001
```sh
sbatch model/run_model.sh --name "attention_two_class" --use_full_data --epochs 5 --task 2 --use_attention --learning_rate 0.0001
```


## Documentation
### `models/`
#### `run_model.py`
the main python script to run models
#### `classification_models.py`
contains architectures for classification models
#### `regression_models.py`
contains architectures for regression models
#### `interpretation.py`
used to create interpretation results for a trained model using SHAP
#### `normalization.py`
creates different scaling methods for preprocessing features before they are fed into models
#### `utils.py`
various utility functions used in running models, such as plotting and
evaluation metrics

## Abstract
Cell-free DNA (cfDNA) consists of DNA fragments released from cells during necrosis or apoptosis, circulating freely in plasma. The arrangement of nucleosomes within cfDNA structure can encode insights into a tumor's transcriptome, thereby delineating the tumor's phenotype. Leveraging the power of deep learning, we present an approach that uses cfDNA fragmentomics to infer gene expression in the tumor. The models were trained on a dataset encompassing cfDNA sequences and bulk tissue RNA-sequencing data sourced from metastatic castration-resistant prostate cancer patient-derived xenograft mouse lines. Using Triton, a nucleosome profiling analysis tool, we generated nucleosome profiles for gene bodies and corresponding transcription start sites. The two covariate groups were used to train a two-headed convolutional neural network model. The model is able to classify binary gene activity status with 88.5% accuracy. We extend the model to a regression task, where it predicts log2 transcripts per million (TPM) values with a root mean squared error (RMSE) of 2.20 and an R-squared value of 0.79.  Finally, we show that our models’ predictions align with gene expression patterns that define prostate cancer phenotype. Our results highlight the potential of cfDNA fragmentomics to unravel the landscape of tumor transcriptomics. 

## Data

#### Scalars
| Variable | Definition |
| ---------| -----------|
site | the gene region name
sample | which LuCap patient the data is from
central-depth | window-normalized phased signal value at the central-loc
central-diversity | window-normalized fragment diversity value in the +/-5 bp region about the central-loc 
central-loc | location of central inflection relative to window center (0)
fragment-diversity | fragment lengths' diversity (unique fragment lengths / total fragments, i.e. multiset support / cardinality)
fragment-entropy | fragment lengths' Shannon entropy
fragment-mad | fragment lengths' median absolute deviation  
fragment-mean | fragment lengths' mean 
fragment-median | fragment lengths' median 
fragment-ratio | fragment lengths' short:long ratio (x <= 150 / x > 150)
fragment-stdev | fragment lengths' standard deviation 
mean-depth | mean depth in the region (GC-corrected, if provided)  
minus-one-pos | location relative to central-loc of minus-one nucleosome 
np-amplitude | phased-nucleosome mean amplitude  
np-period | phased-nucleosome period 
np-score | Nucleosome Phasing Score (NPS)
plus-minus-ratio | ratio of height of +1 nucleosome to -1 nucleosome 
plus-one-pos | location relative to central-loc of plus-one nucleosome 
var-ratio | ratio of variation in total phased signal
exon_count | number of exons the gene encodes
transcript_length | most common transcript length associated with the gene

#### Images
| Variable | Definition |
| ---------| -----------|
1-depth (GC-corrected, if provided)  
2-probable nucleosome center profile (fragment length re-weighted depth)  
3-phased-nucleosome profile (Fourier filtered probable nucleosome center profile)  
4-fragment lengths' short:long ratio (x <= 150 / x > 150)  
5-fragment lengths' diversity (unique fragment lengths / total fragments, i.e. multiset support / cardinality)  
6-fragment lengths' Shannon Entropy (normalized to window Shannon Entropy)  
7-peak locations (-1: trough, 1: peak, -2: minus-one peak, 2: plus-one peak, 3: inflection point)***  
8-A (Adenine) frequency**  
9-C (Cytosine) frequency**  
10-G (Guanine) frequency**  
11-T (Tyrosine) frequency**  

#### Target
L2TPM - log2 transcripts per million for continuous expression \
zFPKM - FPKM standardized using a Gaussian fit to the main peak, threshold of -3 for non-expressed/expressed genes
