#!/bin/bash
set -euo pipefail

# =====================================================================
# XGBoost Fine-Tune — Optuna HP Tuning (CPU-only)
# =====================================================================
# Searches fine-tuning HPs (learning_rate, n_rounds, subsample) via
# Optuna over the PAN-FT cohorts (CRPC-RA + SCLC + PHT).
# Warm-starts from the BTS base model each trial.
#
# Usage:
#   sbatch analysis/XGBoost/slurm/tune_finetune_xgb.sh
# =====================================================================

#SBATCH --job-name=XGB_ft_tune
#SBATCH --output=logs/XGB_ft_tune_%j.out
#SBATCH --error=logs/XGB_ft_tune_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH -A nelson_p

ENV_NAME="proteus"
N_TRIALS=100
STUDY_NAME="xgboost_finetune"
NTHREAD=16
N_JOBS=4          # 4 trials × 16 threads = 64 cores
BASE_MODEL_DIR="analysis/XGBoost/trained_models/final"

cd "$(git rev-parse --show-toplevel)" || exit 1

echo "=== XGBoost Fine-Tune HP Tuning ==="
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Trials:    ${N_TRIALS} (${N_JOBS} parallel × ${NTHREAD} threads each)"
echo "Study:     ${STUDY_NAME}"
echo "Base model:${BASE_MODEL_DIR}"
echo "Start:     $(date)"
echo ""

PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.tune_finetune \
    --n_trials "${N_TRIALS}" \
    --study_name "${STUDY_NAME}" \
    --nthread "${NTHREAD}" \
    --n_jobs "${N_JOBS}" \
    --base_model_dir "${BASE_MODEL_DIR}" \
    --seed 42

echo ""
echo "Finished:  $(date)"
