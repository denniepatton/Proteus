#!/bin/bash
set -euo pipefail

EXPERIMENT="config/experiments/PLCA_tuning/biodvn_tune-plcav3_finalize_v3-4.yaml" \
STUDY_NAME="biodvn_tune-plcav3_finalize_v3-4" \
JOB_PREFIX="plcav3_finalize_v3_4" \
"$(dirname "${BASH_SOURCE[0]}")/launch_plcav3_finalize.sh"