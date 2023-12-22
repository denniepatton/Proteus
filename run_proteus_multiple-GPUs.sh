#!/bin/bash
#SBATCH --job-name=Proteus
#SBATCH --output=Proteus_%j.out
#SBATCH --error=Proteus_%j.err
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --gpus=rtx2080ti:20

# Activate the conda environment
source /home/rpatton/miniconda3/etc/profile.d/conda.sh
conda activate tfenv

# Load the additional, needed drivers
ml GCC/12.2.0 CUDA/12.0.0 cuDNN/8.8.0.121-CUDA-12.0.0

# Run Proteus
python Proteus.py
