#!/bin/bash
set -euo pipefail

# =====================================================================
# Launch PLCA inference for a selected BioDVN variant on one L40S GPU.
#
# Usage:
#   bash runs/launch_inference.sh [default|no-seq|only-seq]
#
# This submits one SLURM job that runs inference sequentially across the
# pre-specified cohort list below. The script points inference at the
# PLCA full-train checkpoint for the selected variant.
# =====================================================================

ENV_NAME="proteus"
VARIANT="${1:-default}"

COHORTS=(
    "BC-Ghana"
    "BLCA-RA"
    "Crompton"
    "CRPC-Pfizer"
    "CRPC-Pluvicto"
    "CRPC-UW"
    "CRPC-WCDT"
    "PC-PDX"
    "PHS"
    "SCLC-PDX"
)

case "${VARIANT}" in
    default)
        YAML="config/experiments/plca_final-training.yaml"
        MODEL="BioDVN"
        TAG="plca_final_inference"
        ;;
    no-seq)
        YAML="config/experiments/plca-no-seq_final-training.yaml"
        MODEL="BioDVN_NoSeq"
        TAG="plca-noseq_inference"
        ;;
    only-seq)
        YAML="config/experiments/plca-only-seq_final-training.yaml"
        MODEL="BioDVN_OnlySeq"
        TAG="plca-onlyseq_inference"
        ;;
    *)
        echo "Unknown variant: ${VARIANT}" >&2
        echo "Usage: bash runs/launch_inference.sh [default|no-seq|only-seq]" >&2
        exit 1
        ;;
esac

COHORT_LIST="${COHORTS[*]}"

job_script=$(mktemp "job_${TAG}_XXXXXX.sh")
cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${TAG}
#SBATCH --output=logs/${TAG}_%j.out
#SBATCH --error=logs/${TAG}_%j.err
#SBATCH --partition=chorus
#SBATCH --cpus-per-task=4
#SBATCH --gpus=l40s:1
#SBATCH --time=12:00:00
#SBATCH -A nelson_p

eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}

COHORTS=(${COHORT_LIST})

echo "Running PLCA inference variant: ${VARIANT}"
echo "Model name: ${MODEL}"
echo "Experiment YAML: ${YAML}"
echo "Cohorts: \${COHORTS[*]}"
echo "Predictions will be written under: results_holdout/${MODEL}/<COHORT>/"

for cohort in "\${COHORTS[@]}"; do
    echo "=== Inference: \${cohort} ==="
        python Proteus.py \
                --run_mode run \
                --experiment ${YAML} \
                --model_name ${MODEL} \
        --use_set "\${cohort}"
done
EOL

sbatch "${job_script}"
rm -f "${job_script}"

echo "Submitted inference job for ${TAG} (${MODEL})"
echo "Configured cohorts: ${COHORT_LIST}"
echo "Per-cohort prediction outputs will go to: results_holdout/${MODEL}/<COHORT>/"
