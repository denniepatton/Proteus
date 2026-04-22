#!/bin/bash
set -euo pipefail

# =====================================================================
# PLCA model CV — 5-fold cross-validation (BioDVN)
# Submits 5 independent RTX jobs, one per fold.
# OOF predictions are written under each fold directory after monitored
# checkpoint selection with the fixed PLCAv2 trial-3 recipe.
# =====================================================================

YAML="config/experiments/plca_cv.yaml"
MODEL="BioDVN"
TAG="plca_cv"
ENV_NAME="proteus"

for fold in {1..5}; do
    job_script=$(mktemp "job_${TAG}_f${fold}_XXXXXX.sh")
    cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${TAG}_F${fold}
#SBATCH --output=logs/${TAG}_F${fold}_%j.out
#SBATCH --error=logs/${TAG}_F${fold}_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx2080ti:1
#SBATCH --time=72:00:00
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
