#!/bin/bash
set -euo pipefail

# =====================================================================
# XGBoost Baseline — Train Final Model (CPU-only)
# =====================================================================
# Trains the final deployable model on ALL BTS data using best HPs
# from Optuna tuning, then optionally runs inference on a use_set.
#
# Usage:
#   sbatch analysis/XGBoost/slurm/train_final_xgb.sh
#   sbatch --export=INFERENCE_SET=CRPC-WCDT analysis/XGBoost/slurm/train_final_xgb.sh
# =====================================================================

#SBATCH --job-name=XGB_final
#SBATCH --output=logs/XGB_final_%j.out
#SBATCH --error=logs/XGB_final_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -A nelson_p

ENV_NAME="proteus"
MODEL_DIR="analysis/XGBoost/trained_models/final"
NTHREAD=16

cd "$(git rev-parse --show-toplevel)" || exit 1

echo "=== XGBoost Final Model Training ==="
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "nthread:   ${NTHREAD}"
echo "Output:    ${MODEL_DIR}"
echo "Start:     $(date)"
echo ""

# ---- Train final model ----
PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.train_final \
    --out_dir "${MODEL_DIR}" \
    --nthread "${NTHREAD}" \
    --seed 42

# ---- Optional inference on a use_set ----
if [[ -n "${INFERENCE_SET:-}" ]]; then
    echo ""
    echo "=== Inference on ${INFERENCE_SET} ==="
    PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.inference \
        --model_dir "${MODEL_DIR}" \
        --use_set "${INFERENCE_SET}"
fi

echo ""
echo "Finished:  $(date)"
