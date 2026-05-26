#!/bin/bash
set -euo pipefail

# Submit the phase-4 PLCAv4 fast sweep on fold 5 only.
#
# This phase uses a fresh study name inside the shared fast-sweep SQLite
# storage, with no previous-study seeding.
#
# It uses the direct fold-parallel Optuna worker path:
#   python Proteus.py --run_mode tune --test_fold 5 ...
#
# Each worker points at the same shared minimize study, but uses a unique
# sampler seed so the fold-5-only scout does not collapse onto duplicate
# random startup suggestions.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-config/experiments/PLCA_tuning/biodvn_tune-plcav4-4_fast-sweep.yaml}"
ENV_NAME="${ENV_NAME:-proteus}"
ACCOUNT="${ACCOUNT:-nelson_p}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///results_training/optuna_plcav4_fast_sweep.db}"
STUDY_NAME="${STUDY_NAME:-biodvn_tune-plcav4-4_fast-sweep}"
ALLOW_EXISTING_DB="${ALLOW_EXISTING_DB:-1}"
FOLD_NUM="${FOLD_NUM:-5}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-10}"
BASE_SAMPLER_SEED="${BASE_SAMPLER_SEED:-6200}"
JOB_PREFIX="${JOB_PREFIX:-plcav4_4_fast}"
PYTHON_BIN="${HOME}/.local/share/mamba/envs/${ENV_NAME}/bin/python"

RTX_WORKERS="${RTX_WORKERS:-15}"
RTX_PARTITION="${RTX_PARTITION:-campus-new}"
RTX_GPU="${RTX_GPU:-rtx2080ti:1}"
RTX_TIME="${RTX_TIME:-72:00:00}"

L40_WORKERS="${L40_WORKERS:-5}"
L40_PARTITION="${L40_PARTITION:-chorus}"
L40_GPU="${L40_GPU:-l40s:1}"
L40_TIME="${L40_TIME:-72:00:00}"

CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_CPU="${MEM_PER_CPU:-20G}"

mkdir -p logs

resolve_python() {
    if [[ -x "${PYTHON_BIN}" ]]; then
        echo "${PYTHON_BIN}"
    else
        echo "python"
    fi
}

sqlite_path_from_url() {
    local storage_url="$1"
    if [[ "${storage_url}" =~ ^sqlite:///(.+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo ""
    fi
}

bootstrap_study() {
    OPTUNA_STORAGE_ENV="${OPTUNA_STORAGE}" \
    STUDY_NAME_ENV="${STUDY_NAME}" \
    "${WORKER_PYTHON}" - <<'PY'
import os

import optuna

storage = os.environ["OPTUNA_STORAGE_ENV"]
study_name = os.environ["STUDY_NAME_ENV"]

if storage.startswith("sqlite"):
    storage_obj = optuna.storages.RDBStorage(
        url=storage,
        engine_kwargs={
            "connect_args": {"timeout": 300},
            "pool_pre_ping": True,
        },
    )
else:
    storage_obj = storage

optuna.create_study(
    study_name=study_name,
    direction="minimize",
    storage=storage_obj,
    load_if_exists=True,
)
print(f"[plcav4-4-fast-bootstrap] ready study={study_name} storage={storage}", flush=True)
PY
}

submit_worker() {
    local gpu_class="$1"
    local worker_idx="$2"
    local partition="$3"
    local gpu_spec="$4"
    local time_limit="$5"
    local sampler_seed="$6"
    local job_name="${JOB_PREFIX}_${gpu_class}_w${worker_idx}_f${FOLD_NUM}"
    local worker_script
    worker_script=$(mktemp "${JOB_PREFIX}_${gpu_class}_XXXXXX.sh")

    cat > "${worker_script}" <<'WORKER_EOF'
#!/bin/bash
set -euo pipefail

worker_idx="$1"
fold_num="$2"
experiment_path="$3"
worker_python="$4"
optuna_storage="$5"
study_name="$6"
trials_per_worker="$7"
sampler_seed="$8"

echo "[plcav4-4-fast] host=$(hostname) worker=${worker_idx} fold=${fold_num} sampler_seed=${sampler_seed}" 

"${worker_python}" Proteus.py \
    --run_mode tune \
    --experiment "${experiment_path}" \
    --test_fold "${fold_num}" \
    --n_trials "${trials_per_worker}" \
    --optuna_storage "${optuna_storage}" \
    --study_name "${study_name}" \
    --sampler_seed "${sampler_seed}"
WORKER_EOF
    chmod +x "${worker_script}"

    local job_id
    if ! job_id=$(sbatch --parsable \
        --job-name="${job_name}" \
        --output="logs/${job_name}_%j.out" \
        --error="logs/${job_name}_%j.err" \
        --partition="${partition}" \
        --ntasks=1 \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem-per-cpu="${MEM_PER_CPU}" \
        --gpus="${gpu_spec}" \
        --time="${time_limit}" \
        -A "${ACCOUNT}" \
        "${worker_script}" \
        "${worker_idx}" "${FOLD_NUM}" "${EXPERIMENT}" "${WORKER_PYTHON}" \
        "${OPTUNA_STORAGE}" "${STUDY_NAME}" "${TRIALS_PER_WORKER}" "${sampler_seed}"); then
        rm -f "${worker_script}"
        return 1
    fi

    echo "Submitted ${job_name} as job ${job_id}"
}

WORKER_PYTHON="$(resolve_python)"
TOTAL_WORKERS=$((RTX_WORKERS + L40_WORKERS))

if [[ "${TOTAL_WORKERS}" -le 0 ]]; then
    echo "Need at least one worker" >&2
    exit 1
fi

SQLITE_PATH="$(sqlite_path_from_url "${OPTUNA_STORAGE}")"
if [[ -n "${SQLITE_PATH}" ]]; then
    if [[ "${SQLITE_PATH}" != /* ]]; then
        SQLITE_PATH="${REPO_ROOT}/${SQLITE_PATH}"
    fi
    if [[ -e "${SQLITE_PATH}" && "${ALLOW_EXISTING_DB:-0}" != "1" ]]; then
        echo "Refusing to reuse existing Optuna DB: ${SQLITE_PATH}" >&2
        echo "Set ALLOW_EXISTING_DB=1 to reuse it explicitly." >&2
        exit 1
    fi
fi

echo "[plcav4-4-fast] experiment=${EXPERIMENT} fold=${FOLD_NUM} workers=${TOTAL_WORKERS} trials_per_worker=${TRIALS_PER_WORKER} storage=${OPTUNA_STORAGE} study=${STUDY_NAME}"
bootstrap_study

worker_idx=0
while [[ "${worker_idx}" -lt "${RTX_WORKERS}" ]]; do
    sampler_seed=$((BASE_SAMPLER_SEED + worker_idx))
    submit_worker "rtx" "${worker_idx}" "${RTX_PARTITION}" "${RTX_GPU}" "${RTX_TIME}" "${sampler_seed}"
    worker_idx=$((worker_idx + 1))
done

while [[ "${worker_idx}" -lt "${TOTAL_WORKERS}" ]]; do
    sampler_seed=$((BASE_SAMPLER_SEED + worker_idx))
    submit_worker "l40" "${worker_idx}" "${L40_PARTITION}" "${L40_GPU}" "${L40_TIME}" "${sampler_seed}"
    worker_idx=$((worker_idx + 1))
done

echo "Submitted ${TOTAL_WORKERS} PLCAv4 phase-4 fast-sweep workers for fold ${FOLD_NUM}"