#!/bin/bash
set -euo pipefail

# Submit proxy-fold PLCAv2-2 tuning workers.
# Start the coordinator manually first:
#   bash runs/biodvn_tune-plcav2-2_opportunistic.sh coordinator

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WORKER_SCRIPT="runs/biodvn_tune-plcav2-2_opportunistic.sh"
ENV_NAME="${ENV_NAME:-proteus}"
ACCOUNT="${ACCOUNT:-ha_g}"

RTX_COUNT="${RTX_COUNT:-16}"
RTX_PARTITION="${RTX_PARTITION:-campus-new}"
RTX_GPU="${RTX_GPU:-rtx2080ti:1}"
RTX_TIME="${RTX_TIME:-120:00:00}"

L40_COUNT="${L40_COUNT:-4}"
L40_PARTITION="${L40_PARTITION:-chorus}"
L40_GPU="${L40_GPU:-l40s:1}"
L40_TIME="${L40_TIME:-120:00:00}"

submit_worker() {
    local role="$1"
    local worker_idx="$2"
    local partition="$3"
    local gpu_spec="$4"
    local time_limit="$5"
    local job_name="plcav22_${role}_${worker_idx}"

    sbatch \
        --job-name="${job_name}" \
        --output="logs/${job_name}_%j.out" \
        --error="logs/${job_name}_%j.err" \
        --partition="${partition}" \
        --cpus-per-task=8 \
        --mem=200G \
        --gpus="${gpu_spec}" \
        --time="${time_limit}" \
        -A "${ACCOUNT}" \
        --wrap="cd ${REPO_ROOT} && ENV_NAME=${ENV_NAME} bash ${WORKER_SCRIPT} ${role}"
}

echo "Submitting ${RTX_COUNT} RTX proxy scouts (paired folds 3/5) and ${L40_COUNT} L40 completion workers for the plcav2-2 study."
echo "Coordinator should already be running via: bash ${WORKER_SCRIPT} coordinator"

for idx in $(seq 1 "${RTX_COUNT}"); do
    submit_worker rtx "${idx}" "${RTX_PARTITION}" "${RTX_GPU}" "${RTX_TIME}"
done

for idx in $(seq 1 "${L40_COUNT}"); do
    submit_worker l40 "${idx}" "${L40_PARTITION}" "${L40_GPU}" "${L40_TIME}"
done

echo "Submitted proxy-fold PLCAv2-2 tuning workers."