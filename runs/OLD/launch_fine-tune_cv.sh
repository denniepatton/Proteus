#!/bin/bash
set -euo pipefail

# =====================================================================
# PAN-FT CV — 5-fold cross-validation with final fine-tuning HPs
# Submits 5 SLURM jobs, one per fold, each on an RTX 2080 Ti.
# patience=999 → all 20 epochs run; final-epoch predictions for CV.
# =====================================================================

YAML="config/experiments/fine-tuning/PAN-FT_cv.yaml"
MODEL="BioDVN"
TAG="PANFT_cv"
ENV_NAME="proteus"

for fold in 1 2 3 4 5; do
    job_script=$(mktemp "job_${TAG}_f${fold}_XXXXXX.sh")
    cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${TAG}_F${fold}
#SBATCH --output=logs/${TAG}_F${fold}_%j.out
#SBATCH --error=logs/${TAG}_F${fold}_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx2080ti:1
#SBATCH --time=48:00:00
#SBATCH -A nelson_p

eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}
python Proteus.py \\
    --run_mode train \\
    --experiment ${YAML} \\
    --model_name ${MODEL} \\
    --test_fold ${fold}
EOL
    sbatch "${job_script}"
    rm -f "${job_script}"
done

echo "Submitted 5 CV fold jobs for ${TAG} (${MODEL})"
