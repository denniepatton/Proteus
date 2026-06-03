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
MODEL_SELECTOR="${MODEL:-${MODEL_NAME:-default}}"
MODEL_NAME=""
EXPERIMENT="${EXPERIMENT:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JOB_PREFIX="${JOB_PREFIX:-proteus_infer}"
USE_SETS="${USE_SETS:-}"

POSITIONAL_DATASETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --model" >&2
                exit 1
            fi
            MODEL_SELECTOR="$2"
            shift 2
            ;;
        --use_sets|--datasets)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --use_sets" >&2
                exit 1
            fi
            USE_SETS="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: bash runs/launch_inference.sh [--model default|no-seq|only-seq] [--use_sets all|SET1,SET2,...]

Selection options:
  --model      Choose one of the three TDS final models.
  --use_sets   Choose datasets by comma-separated names, or 'all'.

Compatibility:
  - Positional dataset names still work.
  - MODEL_NAME and MODEL env vars still work.
  - EXPERIMENT still overrides the auto-selected experiment.

Examples:
  bash runs/launch_inference.sh --model default --use_sets all
  bash runs/launch_inference.sh --model no-seq --use_sets PHS,CRPC-RA,SCLC
  MODEL=only-seq bash runs/launch_inference.sh PHT PC-PDX
EOF
            exit 0
            ;;
        *)
            POSITIONAL_DATASETS+=("$1")
            shift
            ;;
    esac
done

case "${MODEL_SELECTOR}" in
    default|BioDVN)
        MODEL_NAME="BioDVN"
        DEFAULT_EXPERIMENT="config/experiments/tds_final-training.yaml"
        ;;
    no-seq|noseq|BioDVN_NoSeq)
        MODEL_NAME="BioDVN_NoSeq"
        DEFAULT_EXPERIMENT="config/experiments/tds-no-seq_final-training.yaml"
        ;;
    only-seq|onlyseq|BioDVN_OnlySeq)
        MODEL_NAME="BioDVN_OnlySeq"
        DEFAULT_EXPERIMENT="config/experiments/tds-only-seq_final-training.yaml"
        ;;
    *)
        echo "Unknown model selector: ${MODEL_SELECTOR}. Use default, no-seq, only-seq, BioDVN, BioDVN_NoSeq, or BioDVN_OnlySeq." >&2
        exit 1
        ;;
esac

if [[ -z "${EXPERIMENT}" ]]; then
    EXPERIMENT="${DEFAULT_EXPERIMENT}"
fi

AUTO_DATASETS=0
DATASET_LIST=""
if [[ -n "${USE_SETS}" && ${#POSITIONAL_DATASETS[@]} -gt 0 ]]; then
    echo "Use either --use_sets or positional dataset names, not both." >&2
    exit 1
fi

if [[ -n "${USE_SETS}" ]]; then
    if [[ "${USE_SETS}" == "all" ]]; then
        AUTO_DATASETS=1
    else
        DATASET_LIST="${USE_SETS//,/ }"
    fi
elif [[ ${#POSITIONAL_DATASETS[@]} -eq 0 ]]; then
    AUTO_DATASETS=1
else
    DATASET_LIST="${POSITIONAL_DATASETS[*]}"
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

MODEL_NAME="${MODEL_NAME}"
EXPERIMENT="${EXPERIMENT}"
AUTO_DATASETS="${AUTO_DATASETS}"
DATASET_LIST="${DATASET_LIST}"

if [[ "${AUTO_DATASETS}" == "1" ]]; then
    mapfile -t DATASETS < <("${PYTHON_BIN}" - <<'PY'
from config.dataset_config import list_datasets

for dataset_name in list_datasets():
    print(dataset_name)
PY
    )
else
    DATASETS=(${DATASET_LIST})
fi

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

if [[ "${AUTO_DATASETS}" == "1" ]]; then
    echo "Submitted inference job ${JOB_PREFIX} for all configured use sets"
else
    echo "Submitted inference job ${JOB_PREFIX} for datasets: ${DATASET_LIST}"
fi

