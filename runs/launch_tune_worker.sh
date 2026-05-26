#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-}"
MODEL_NAME="${MODEL_NAME:-BioDVN}"
RUN_MODE="${RUN_MODE:-tune}"
ENV_NAME="${ENV_NAME:-proteus}"
ACCOUNT="${ACCOUNT:-nelson_p}"
PARTITION="${PARTITION:-campus-new}"
GPU="${GPU:-rtx2080ti:1}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOB_PREFIX="${JOB_PREFIX:-proteus_tune}"
TEST_FOLD="${TEST_FOLD:-}"
N_TRIALS="${N_TRIALS:-50}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-}"
STUDY_NAME="${STUDY_NAME:-}"
SAMPLER_SEED="${SAMPLER_SEED:-42}"
SUBSAMPLE="${SUBSAMPLE:-}"

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

cmd=("${PYTHON_BIN}" Proteus.py --run_mode "${RUN_MODE}" --experiment "${EXPERIMENT}" --model_name "${MODEL_NAME}" --n_trials "${N_TRIALS}" --sampler_seed "${SAMPLER_SEED}")
if [[ -n "${TEST_FOLD}" ]]; then
    cmd+=(--test_fold "${TEST_FOLD}")
fi
if [[ -n "${OPTUNA_STORAGE}" ]]; then
    cmd+=(--optuna_storage "${OPTUNA_STORAGE}")
fi
if [[ -n "${STUDY_NAME}" ]]; then
    cmd+=(--study_name "${STUDY_NAME}")
fi
if [[ -n "${SUBSAMPLE}" ]]; then
    cmd+=(--subsample "${SUBSAMPLE}")
fi

echo "[proteus-tune-worker] run_mode=${RUN_MODE} model=${MODEL_NAME} experiment=${EXPERIMENT}"
"\${cmd[@]}"
EOL

sbatch "${job_script}"
rm -f "${job_script}"

echo "Submitted tuning worker job ${JOB_PREFIX}"