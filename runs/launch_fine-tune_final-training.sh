#!/bin/bash
set -euo pipefail

# =====================================================================
# PAN-FT final training (BioDVN) — single L40S GPU
# Fine-tunes base model on ALL 80 PAN-FT samples (44 CRPC-RA + 16 SCLC
# + 20 PHT).  Gene-level 10% chunk split for early stopping (patience=3).
# Output: trained_models/proteus_PAN-FT/full_train/best_model.pt
#         trained_models/proteus_PAN-FT/full_train/final_model.pt
# =====================================================================

YAML="config/experiments/fine-tuning/PAN-FT_final-training.yaml"
MODEL="BioDVN"
TAG="PANFT_final"
ENV_NAME="proteus"

job_script=$(mktemp "job_${TAG}_XXXXXX.sh")
cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${TAG}
#SBATCH --output=logs/${TAG}_%j.out
#SBATCH --error=logs/${TAG}_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx2080ti:1
#SBATCH --time=72:00:00
#SBATCH -A nelson_p

eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}
python Proteus.py \\
    --run_mode full-train \\
    --experiment ${YAML} \\
    --model_name ${MODEL}
EOL

sbatch "${job_script}"
rm -f "${job_script}"

echo "Submitted full-train job for ${TAG} (${MODEL})"
