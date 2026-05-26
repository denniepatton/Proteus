#!/bin/bash
set -euo pipefail

EXPERIMENT="config/experiments/PLCA_tuning/biodvn_tune-plcav3_finalize_v3-5.yaml" \
STUDY_NAME="biodvn_tune-plcav3_finalize_v3-5" \
JOB_PREFIX="plcav3_finalize_v3_5" \
"$(dirname "${BASH_SOURCE[0]}")/launch_plcav3_finalize.sh"