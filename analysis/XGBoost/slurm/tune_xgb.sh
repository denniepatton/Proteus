#!/bin/bash
set -euo pipefail

# =====================================================================
# XGBoost Baseline — Optuna HP Tuning (CPU-only)
# =====================================================================
# Submits a single CPU job to run ~200 Optuna trials.
# Runs N_JOBS trials in parallel, each using NTHREAD cores.
#
# Usage:
#   sbatch analysis/XGBoost/slurm/tune_xgb.sh
# =====================================================================

#SBATCH --job-name=XGB_tune
#SBATCH --output=logs/XGB_tune_%j.out
#SBATCH --error=logs/XGB_tune_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -A nelson_p

ENV_NAME="proteus"
N_TRIALS=200
STUDY_NAME="xgboost_baseline"
NTHREAD=16
N_JOBS=4          # 4 trials × 16 threads = 64 cores

cd "$(git rev-parse --show-toplevel)" || exit 1

echo "=== XGBoost HP Tuning ==="
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Trials:    ${N_TRIALS} (${N_JOBS} parallel × ${NTHREAD} threads each)"
echo "Study:     ${STUDY_NAME}"
echo "Start:     $(date)"
echo ""

PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.tune \
    --n_trials "${N_TRIALS}" \
    --study_name "${STUDY_NAME}" \
    --nthread "${NTHREAD}" \
    --n_jobs "${N_JOBS}" \
    --seed 42

echo ""
echo "Finished:  $(date)"
