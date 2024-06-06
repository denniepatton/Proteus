# Proteus <img src="misc/logo_v2.png" width="140" align="left">
As a CNN-based gene expression model, Proteus utilizes information contained in the cell-free DNA (cfDNA) coverage profiles of individual genes
to accurately and directly predict underlying mRNA abundance providing a wholistic, liquid-biopsy-based analog to tradtional RNA-seq assays.

_Proteus_, also known as the "old man of the sea," is named for a prophetic Greek god of the sea who would change forms to disquise himself before ultimately revealing the truth.  
Like Proteus, this tool works in the deep and changes the form of data to reveal new information.
<br/><br/>

## Description
Proteus is a CNN-based and biologically informed deep learning framework which utilizes features and signal profiles from cfDNA WGS to predict
corresponding mRNA abundance directly, at the individual gene level, and in a manner agnostic of sample or tissue identity. To do this Proteus
uses promoter region coverage, nucleosome positioning, and fragmentation profiles along with full-length-gene features output by [Triton](https://github.com/GavinHaLab/Triton)
in conjunction with the underlying reference sequence. Proteus was trained to predict individual genes' Log2(CPM+1), where CPM is TMM-normalized across training samples,
in a 1:1 manner given a set of Triton-derived input features. These predictions can be used analogously to RNA-seq assays, meaning differential gene expression
analyses, gene set enrichment analyses, molecular phenotyping, specific target monitoring and more can be achieved using standard WGS of cfDNA. Unlike tissue biopsy-based
RNA-seq, this method allows for the comprehensive evaluation of expression from a liquid biopsy, meaning results wholistically reflect the underlying profiles from
which the cfDNA originated.

TODO: make a run-mode only script, which uses a pre-packaged model and takes as inputs the Triton output files? E.g. the most stripped-down version that's easy to run/use
for people who only want to make predictions?

### Outputs

TBD: only handle the prediction use case? Seperate section if people want to train?

### Uses

Proteus should be used as an endpoint in cfDNA data analysis by outputting ready-to-use expression predictions. These can be used without additional formatting
in many standard RNA-seq methods and analyses, including gene set enrichment analyses. As the outputs mimic those created by [edgeR](https://bioconductor.org/packages/release/bioc/html/edgeR.html)
they are especially suitable for, and can be immediately integrated into, edgeR-based differential expression analyses. Proteus can also be used to directly monitor
the expression of therapeutic targets in order to tailor treatment protocols.

### Publications

TBD

## Usage

Proteus was designed to run on a () GPU as a command line tool . . .

### Inputs to Triton.py:

```
--scalars_reference_path	: Path to the scalars reference file (string, default="/fh/fast/ha_g/user/rpatton/TritonCNN/TritonCNNv3/reference/MANE.GRCh38.v1.3.UniversalReference.tsv")
--training_home		: Path to the training home directory (string, default="/fh/fast/ha_g/user/rpatton/SCLC_data/")
--model_name		: Name of the model to use (in models/models.py) (string, default="BioCNN")
--use_set			: The set_parameters dataset to use (see Proteus.py main() for options) (string, default='GeneStrat_0x')
--process_data		: If True, will check but not overwrite existing sample-level h5 data (bool, default=True)
--test_fold			: Specify the test fold for parallel training on different GPU nodes (int, optional)
--set_parameters		: Path to the set_parameters.tsv file (string, default='data/set_parameters.tsv')
--run_mode			: Specify whether to 'train' (+test using CV), 'test', 'full-train' or 'run' the model (string, default='train')
--tune_hps			: Specify whether to tune hyperparameters (bool, default=False)
```

### Inputs (extra details):

**scalars_reference_path:** how much detail should we add?

### Contained Scripts:

**Proteus.py** | primary script which calls those in preprocessing/ and models/ to train and run Proteus
**preprocessing/compile_sample_data.py** | ...
**preprocessing/generate_tf_splits.py** | ...
**models/models.py** | ...
**models/model_utils.py** | ...
N.B. analysis/ will be re-structured, we may not include all scripts . . .

#### compile_sample_data.py

Summary of this specific script, and so on.

### Methodology

We'll want to fill this out, include a diagram of the model architecture, and highlight some model interp. (with images)

### TO RUN

instructions

## Requirements

GPU recs.

The provided environment.yaml describes a conda environment capable of . . .
Info on TF?

On Fred Hutch servers, Proteus can be run by loading the conda environment from environment.yaml and entering the following lines on cmd:
ml GCC/12.2.0 CUDA/12.0.0 cuDNN/8.8.0.121-CUDA-12.0.0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/app/software/CUDA/12.0.0

## Contact
If you have any questions or feedback, please contact me at:  
**Email:** <rpatton@fredhutch.org>

## Acknowledgments
Proteus is developed and maintained by Robert D. Patton in the Gavin Ha Lab, Fred Hutchinson Cancer Center.
Soon to be just the Ha Lab, maybe?

## License

The MIT License (MIT)

Copyright (c) 2023 Fred Hutchinson Cancer Center

Permission is hereby granted, free of charge, to any government or not-for-profit entity, or to any person employed at one of the foregoing (each, an "Academic Licensee") who obtains a copy of this software and associated documentation files (the “Software”), to deal in the Software purely for non-commercial research and educational purposes, including the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or share copies of the Software, and to permit other Academic Licensees to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

No Academic Licensee shall be permitted to sell or use the Software or derivatives thereof in any service for commercial benefit. For the avoidance of doubt, any use by or transfer to a commercial entity shall be considered a commercial use and will require a separate license with Fred Hutchinson Cancer Center.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
