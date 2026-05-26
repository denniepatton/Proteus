#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-}"
MODEL_NAME="${MODEL_NAME:-BioDVN}"
ENV_NAME="${ENV_NAME:-proteus}"
ACCOUNT="${ACCOUNT:-nelson_p}"
PARTITION="${PARTITION:-campus-new}"
GPU="${GPU:-rtx2080ti:1}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOB_PREFIX="${JOB_PREFIX:-proteus_full_train}"
SUBSAMPLE="${SUBSAMPLE:-}"
BASE_WEIGHTS="${BASE_WEIGHTS:-}"
ZARR_OUTPUT_DIR="${ZARR_OUTPUT_DIR:-}"

if [[ -z "${EXPERIMENT}" ]]; then
    echo "Set EXPERIMENT to a config/experiments/*.yaml file." >&2
    exit 1
fi

mkdir -p logs
job_script=$(mktemp "${JOB_PREFIX}_XXXXXX.sh")

cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${JOB_PREFIX}
#SBATCH --output=logs/${JOB_PREFIX}_%j.out
#SBATCH --error=logs/${JOB_PREFIX}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus=${GPU}
#SBATCH --time=${TIME_LIMIT}
#SBATCH -A ${ACCOUNT}

set -euo pipefail
cd "${REPO_ROOT}"
eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}

cmd=("${PYTHON_BIN}" Proteus.py --run_mode full-train --experiment "${EXPERIMENT}" --model_name "${MODEL_NAME}")
if [[ -n "${SUBSAMPLE}" ]]; then
    cmd+=(--subsample "${SUBSAMPLE}")
fi
if [[ -n "${BASE_WEIGHTS}" ]]; then
    cmd+=(--base_weights "${BASE_WEIGHTS}")
fi
if [[ -n "${ZARR_OUTPUT_DIR}" ]]; then
    cmd+=(--zarr_output_dir "${ZARR_OUTPUT_DIR}")
fi

echo "[proteus-full-train] model=${MODEL_NAME} experiment=${EXPERIMENT}"
"\${cmd[@]}"
EOL

sbatch "${job_script}"
rm -f "${job_script}"

echo "Submitted full-train job ${JOB_PREFIX}"
