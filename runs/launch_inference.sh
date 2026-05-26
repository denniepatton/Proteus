#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-proteus}"
ACCOUNT="${ACCOUNT:-nelson_p}"
PARTITION="${PARTITION:-chorus}"
GPU="${GPU:-l40s:1}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MODEL_NAME="${MODEL_NAME:-BioDVN}"
EXPERIMENT="${EXPERIMENT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOB_PREFIX="${JOB_PREFIX:-proteus_infer}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash runs/launch_inference.sh <DATASET> [DATASET ...]" >&2
    echo "Optional env: MODEL_NAME, EXPERIMENT, PARTITION, GPU, TIME_LIMIT, ACCOUNT, ENV_NAME" >&2
    exit 1
fi

DATASET_LIST="$*"
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

MODEL_NAME="${MODEL_NAME}"
EXPERIMENT="${EXPERIMENT}"
DATASETS=(${DATASET_LIST})

echo "[proteus-inference] model=${MODEL_NAME} datasets=\${DATASETS[*]} experiment=\${EXPERIMENT:-<bundled checkpoint>}"

for dataset in "\${DATASETS[@]}"; do
    cmd=("${PYTHON_BIN}" Proteus.py --run_mode run --model_name "\${MODEL_NAME}" --use_set "\${dataset}")
    if [[ -n "\${EXPERIMENT}" ]]; then
        cmd+=(--experiment "\${EXPERIMENT}")
    fi
    echo "[proteus-inference] dataset=\${dataset}"
    "\${cmd[@]}"
done
EOL

sbatch "${job_script}"
rm -f "${job_script}"

echo "Submitted inference job ${JOB_PREFIX} for datasets: ${DATASET_LIST}"

