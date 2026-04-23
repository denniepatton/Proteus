#!/bin/bash
set -euo pipefail

# =====================================================================
# XGBoost PLCA baseline — Optuna HP tuning (CPU-only)
# =====================================================================
# Fresh PLCA study using the Proteus-aligned weighted source-balanced
# GVW-MAE objective and the narrowed PLCA search profile.
#
# Usage:
#   sbatch analysis/XGBoost/slurm/tune_xgb_plca.sh
# =====================================================================

#SBATCH --job-name=XGB_tune_PLCA
#SBATCH --output=logs/XGB_tune_PLCA_%j.out
#SBATCH --error=logs/XGB_tune_PLCA_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -A nelson_p

ENV_NAME="proteus"
N_TRIALS=80
STUDY_NAME="xgboost_plca_weighted_mae"
STORAGE_REL="analysis/XGBoost/optuna_xgb_plca_weighted_mae.db"
RESULTS_DIR_REL="analysis/XGBoost/tuning_results_plca_weighted_mae"
TSV_REL="data/PLCA_sample-splits.tsv"
NTHREAD=16
N_JOBS=4          # 4 trials × 16 threads = 64 cores
N_ESTIMATORS=8000

cd "$(git rev-parse --show-toplevel)" || exit 1

echo "=== XGBoost PLCA HP Tuning ==="
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Trials:    ${N_TRIALS} (${N_JOBS} parallel × ${NTHREAD} threads each)"
echo "Study:     ${STUDY_NAME}"
echo "Storage:   ${STORAGE_REL}"
echo "Results:   ${RESULTS_DIR_REL}"
echo "Splits:    ${TSV_REL}"
echo "Metric:    weighted_source_gvw_mae"
echo "Profile:   plca"
echo "Rounds:    ${N_ESTIMATORS}"
echo "Start:     $(date)"
echo ""

PYTHONPATH="${PWD}/analysis" micromamba run -n "${ENV_NAME}" python -m XGBoost.tune \
    --n_trials "${N_TRIALS}" \
    --study_name "${STUDY_NAME}" \
    --storage "sqlite:///${PWD}/${STORAGE_REL}" \
    --results_dir "${PWD}/${RESULTS_DIR_REL}" \
    --tsv "${PWD}/${TSV_REL}" \
    --metric weighted_source_gvw_mae \
    --search_profile plca \
    --n_estimators "${N_ESTIMATORS}" \
    --nthread "${NTHREAD}" \
    --n_jobs "${N_JOBS}" \
    --seed 42

echo ""
echo "Finished:  $(date)"