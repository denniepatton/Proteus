#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export EXPERIMENT="${EXPERIMENT:-config/experiments/tds-only-seq_final-training.yaml}"
export MODEL_NAME="${MODEL_NAME:-BioDVN_OnlySeq}"
export ENV_NAME="${ENV_NAME:-proteus}"
export ACCOUNT="${ACCOUNT:-nelson_p}"
export PARTITION="${PARTITION:-chorus}"
export GPU="${GPU:-l40s:1}"
export TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
export JOB_PREFIX="${JOB_PREFIX:-tds_only_seq_final_train}"

exec "${REPO_ROOT}/runs/launch_full_train.sh"