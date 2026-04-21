#!/bin/bash
set -euo pipefail
MANIFEST_FILE="$1"
YAML_FILE="$2"
ENV_NAME="$3"

eval "$(micromamba shell hook -s bash)"
micromamba activate "${ENV_NAME}"

# Look up this task's trial and fold from the manifest
TASK_LINE=$(awk -v id="${SLURM_PROCID}" '$1 == id' "${MANIFEST_FILE}")
TRIAL_NUM=$(echo "${TASK_LINE}" | cut -f2)
FOLD_NUM=$(echo "${TASK_LINE}" | cut -f3)

echo "[Task ${SLURM_PROCID}] trial=${TRIAL_NUM} fold=${FOLD_NUM} node=$(hostname)"

python Proteus.py \
    --run_mode tune-fold \
    --experiment "${YAML_FILE}" \
    --trial_number "${TRIAL_NUM}" \
    --test_fold "${FOLD_NUM}"
