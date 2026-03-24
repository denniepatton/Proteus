#!/bin/bash
set -euo pipefail

# =====================================================================
# XGBoost Fine-Tune — Final Model Training + Inference (CPU-only)
# =====================================================================
# Trains the final deployable fine-tuned model on ALL PAN-FT data
# using best HPs from Optuna fine-tune study (trial 382).
# Optionally runs inference on CRPC-RA (or another use_set).
#
# Usage:
#   sbatch analysis/XGBoost/slurm/train_finetune_final_xgb.sh
#   sbatch --export=INFERENCE_SET=CRPC-RA analysis/XGBoost/slurm/train_finetune_final_xgb.sh
# =====================================================================

#SBATCH --job-name=XGB_ft_final
#SBATCH --output=logs/XGB_ft_final_%j.out
#SBATCH --error=logs/XGB_ft_final_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -A nelson_p

ENV_NAME="proteus"
NTHREAD=16
BASE_MODEL_DIR="analysis/XGBoost/trained_models/final"
FT_MODEL_DIR="analysis/XGBoost/trained_models/fine-tuned"

cd "$(git rev-parse --show-toplevel)" || exit 1

echo "=== XGBoost Fine-Tune: Final Model ==="
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "nthread:   ${NTHREAD}"
echo "Base model:${BASE_MODEL_DIR}"
echo "FT model:  ${FT_MODEL_DIR}"
echo "Start:     $(date)"
echo ""

# ---- Full-train final fine-tuned model ----
PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.train_finetune \
    --mode full-train \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --out_dir "${FT_MODEL_DIR}" \
    --nthread "${NTHREAD}" \
    --seed 42

# ---- Optional inference on a use_set ----
if [[ -n "${INFERENCE_SET:-}" ]]; then
    echo ""
    echo "=== Inference on ${INFERENCE_SET} ==="
    PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.inference \
        --model_dir "${FT_MODEL_DIR}" \
        --use_set "${INFERENCE_SET}"
fi

echo ""
echo "Finished:  $(date)"
