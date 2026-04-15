#!/bin/bash
set -euo pipefail

# =====================================================================
# PLCA model no-seq final training (BioDVN_NoSeq) — single L40S GPU
# Trains on ALL samples with gene-level 10% chunk split for early stopping.
# Output: trained_models/proteus_plca-model_no-seq/full_train/final_model.pt
# =====================================================================

YAML="config/experiments/plca-no-seq_final-training.yaml"
MODEL="BioDVN_NoSeq"
TAG="plca-noseq_final"
ENV_NAME="proteus"

job_script=$(mktemp "job_${TAG}_XXXXXX.sh")
cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${TAG}
#SBATCH --output=logs/${TAG}_%j.out
#SBATCH --error=logs/${TAG}_%j.err
#SBATCH --partition=chorus
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40s:1
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
