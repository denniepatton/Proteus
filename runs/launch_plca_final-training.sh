#!/bin/bash
set -euo pipefail

# =====================================================================
# PLCA model final training (BioDVN) — single L40S GPU
# Trains on all PLCA samples with the fixed PLCAv2 trial-3 recipe.
# Full-train runs a fixed 25-epoch chunk-based schedule and saves
# trained_models/proteus_plca-model/full_train/final_model.pt.
# =====================================================================

YAML="config/experiments/plca_final-training.yaml"
MODEL="BioDVN"
TAG="plca_final"
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
