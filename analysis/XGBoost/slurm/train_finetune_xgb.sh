#!/bin/bash
set -euo pipefail

# =====================================================================
# XGBoost Fine-Tune — CV Evaluation + Full-Train (CPU-only)
# =====================================================================
# Step 1: 5-fold CV on PAN-FT with warm-start from BTS base model.
# Step 2: Full-train final fine-tuned model on all PAN-FT data.
# Step 3 (optional): Inference on a use_set with the fine-tuned model.
#
# Uses best HPs from tune_finetune_xgb.sh (or cemented defaults).
#
# Usage:
#   sbatch analysis/XGBoost/slurm/train_finetune_xgb.sh
#   sbatch --export=INFERENCE_SET=CRPC-WCDT analysis/XGBoost/slurm/train_finetune_xgb.sh
# =====================================================================

#SBATCH --job-name=XGB_ft_train
#SBATCH --output=logs/XGB_ft_train_%j.out
#SBATCH --error=logs/XGB_ft_train_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -A nelson_p

ENV_NAME="proteus"
NTHREAD=16
BASE_MODEL_DIR="analysis/XGBoost/trained_models/final"
FT_MODEL_DIR="analysis/XGBoost/trained_models/fine-tuned"
CV_OUT_DIR="results_training/xgboost_fine-tune_cv"

cd "$(git rev-parse --show-toplevel)" || exit 1

echo "=== XGBoost Fine-Tune: CV + Full-Train ==="
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "nthread:   ${NTHREAD}"
echo "Base model:${BASE_MODEL_DIR}"
echo "CV output: ${CV_OUT_DIR}"
echo "FT model:  ${FT_MODEL_DIR}"
echo "Start:     $(date)"
echo ""

# ---- 5-fold CV evaluation ----
echo "=== Step 1: 5-fold CV ==="
PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.train_finetune \
    --mode cv \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --out_dir "${CV_OUT_DIR}" \
    --nthread "${NTHREAD}"

echo ""

# ---- Full-train final fine-tuned model ----
echo "=== Step 2: Full-Train ==="
PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.train_finetune \
    --mode full-train \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --out_dir "${FT_MODEL_DIR}" \
    --nthread "${NTHREAD}" \
    --seed 42

# ---- Optional inference on a use_set ----
if [[ -n "${INFERENCE_SET:-}" ]]; then
    echo ""
    echo "=== Step 3: Inference on ${INFERENCE_SET} ==="
    PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.inference \
        --model_dir "${FT_MODEL_DIR}" \
        --use_set "${INFERENCE_SET}"
fi

echo ""
echo "Finished:  $(date)"
