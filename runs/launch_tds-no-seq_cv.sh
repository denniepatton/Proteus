#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export EXPERIMENT="${EXPERIMENT:-config/experiments/tds-no-seq_cv.yaml}"
export MODEL_NAME="${MODEL_NAME:-BioDVN_NoSeq}"
export ENV_NAME="${ENV_NAME:-proteus}"
export ACCOUNT="${ACCOUNT:-nelson_p}"
export PARTITION="${PARTITION:-campus-new}"
export GPU="${GPU:-rtx2080ti:1}"
export TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export JOB_PREFIX="${JOB_PREFIX:-tds_no_seq_cv}"

exec "${REPO_ROOT}/runs/launch_train_cv.sh"