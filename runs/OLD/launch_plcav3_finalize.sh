#!/bin/bash
set -euo pipefail

# Submit a coordinated full-CV PLCAv3 finalize sweep.
#
# Each Optuna config is evaluated across all 5 folds as one coordinated trial.
# Every fold is submitted as an independent one-GPU SLURM job; no multi-node or
# multi-task GPU jobs are used.
#
# Important: prev_study seeding only works when the previous study lives in the
# same Optuna storage backend, so this launcher intentionally reuses the refine
# SQLite DB with a new study name.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-config/experiments/PLCA_tuning/biodvn_tune-plcav3_finalize.yaml}"
ENV_NAME="${ENV_NAME:-proteus}"
ACCOUNT="${ACCOUNT:-nelson_p}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///results_training/optuna_plcav3_refine.db}"
STUDY_NAME="${STUDY_NAME:-biodvn_tune-plcav3_finalize}"
TOTAL_CONFIGS="${TOTAL_CONFIGS:-32}"
CONFIGS_PER_ROUND=4
BASE_SAMPLER_SEED="${BASE_SAMPLER_SEED:-5200}"
PYTHON_BIN="${HOME}/.local/share/mamba/envs/${ENV_NAME}/bin/python"
JOB_PREFIX="${JOB_PREFIX:-plcav3_finalize}"

RTX_PARTITION="${RTX_PARTITION:-campus-new}"
RTX_GPU="${RTX_GPU:-rtx2080ti:1}"
RTX_TIME="${RTX_TIME:-24:00:00}"

L40_PARTITION="${L40_PARTITION:-chorus}"
L40_GPU="${L40_GPU:-l40s:1}"
L40_TIME="${L40_TIME:-24:00:00}"

FOLDS=(1 2 3 4 5)
GPU_CLASS_SEQUENCE=(rtx rtx rtx l40)
mkdir -p logs

resolve_output_dir() {
    local output_dir
    output_dir=$(grep -E '^[[:space:]]*output_dir:' "${EXPERIMENT}" | head -1 | cut -d: -f2- | tr -d ' "')
    if [[ -z "${output_dir}" ]]; then
        echo "Could not resolve output_dir from ${EXPERIMENT}" >&2
        return 1
    fi
    echo "${output_dir}"
}

OUTPUT_DIR="$(resolve_output_dir)"

resolve_python() {
    if [[ -x "${PYTHON_BIN}" ]]; then
        echo "${PYTHON_BIN}"
    else
        echo "python"
    fi
}

WORKER_PYTHON="$(resolve_python)"

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
print(f"[plcav3-finalize-bootstrap] ready study={study_name} storage={storage}", flush=True)
PY
}

parse_trial_numbers() {
    local ask_output="$1"
    local trial_line
    trial_line=$(printf '%s\n' "${ask_output}" | grep '^TRIAL_NUMBERS=' | tail -1 || true)
    if [[ -z "${trial_line}" ]]; then
        echo ""
    else
        echo "${trial_line#TRIAL_NUMBERS=}"
    fi
}

ask_configs() {
    local ask_n="$1"
    "${WORKER_PYTHON}" Proteus.py \
        --run_mode tune-ask \
        --experiment "${EXPERIMENT}" \
        --n_configs "${ask_n}" \
        --optuna_storage "${OPTUNA_STORAGE}" \
        --study_name "${STUDY_NAME}" \
        --sampler_seed "${BASE_SAMPLER_SEED}"
}

tell_configs() {
    local trial_csv="$1"
    "${WORKER_PYTHON}" Proteus.py \
        --run_mode tune-tell \
        --experiment "${EXPERIMENT}" \
        --trial_numbers "${trial_csv}" \
        --optuna_storage "${OPTUNA_STORAGE}" \
        --study_name "${STUDY_NAME}" \
        --sampler_seed "${BASE_SAMPLER_SEED}"
}

submit_single_fold() {
    local gpu_class="$1"
    local config_idx="$2"
    local trial_num="$3"
    local fold_num="$4"
    local partition="$5"
    local gpu_spec="$6"
    local time_limit="$7"
    local job_name="${JOB_PREFIX}_${gpu_class}_cfg${config_idx}_t${trial_num}_f${fold_num}"
    local worker_script
    worker_script=$(mktemp "${JOB_PREFIX}_${gpu_class}_XXXXXX.sh")

    cat > "${worker_script}" <<'WORKER_EOF'
#!/bin/bash
set -euo pipefail

trial_num="$1"
fold_num="$2"
experiment_path="$3"
worker_python="$4"

echo "[plcav3-finalize] host=$(hostname) trial=${trial_num} fold=${fold_num} task=0"

"${worker_python}" Proteus.py \
    --run_mode tune-fold \
    --experiment "${experiment_path}" \
    --trial_number "${trial_num}" \
    --test_fold "${fold_num}"
WORKER_EOF
    chmod +x "${worker_script}"

    local job_id
    if ! job_id=$(sbatch --parsable \
        --job-name="${job_name}" \
        --output="logs/${job_name}_%j.out" \
        --error="logs/${job_name}_%j.err" \
        --partition="${partition}" \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem-per-cpu=20G \
        --gpus="${gpu_spec}" \
        --time="${time_limit}" \
        -A "${ACCOUNT}" \
        "${worker_script}" "${trial_num}" "${fold_num}" "${EXPERIMENT}" "${WORKER_PYTHON}"); then
        rm -f "${worker_script}"
        return 1
    fi

    if [[ -z "${job_id}" ]]; then
        rm -f "${worker_script}"
        echo "sbatch returned an empty job id for trial ${trial_num}, fold ${fold_num}" >&2
        return 1
    fi

    echo "${job_id}:${worker_script}"
}

wait_for_job() {
    local job_id="$1"
    while true; do
        sleep 30
        local state
        state=$(squeue -j "${job_id}" -h -o '%T' 2>/dev/null || true)
        if [[ -z "${state}" ]]; then
            break
        fi
    done
}

check_job_exit() {
    local job_id="$1"
    local exit_code
    exit_code=$(sacct -j "${job_id}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')
    if [[ "${exit_code}" != "0:0" ]]; then
        echo "Job ${job_id} exited with ${exit_code}" >&2
        sacct -j "${job_id}" --format=JobID,NodeList,ExitCode,State,Elapsed -n >&2 || true
        return 1
    fi
    return 0
}

fold_metrics_path() {
    local trial_num="$1"
    local fold_num="$2"
    printf '%s/trial_%05d/CV-fold_%d/validation-fold_%d_metrics.json' \
        "${OUTPUT_DIR}" "${trial_num}" "${fold_num}" "${fold_num}"
}

fold_complete() {
    local trial_num="$1"
    local fold_num="$2"
    [[ -s "$(fold_metrics_path "${trial_num}" "${fold_num}")" ]]
}

trial_complete() {
    local trial_num="$1"
    local fold_num
    for fold_num in "${FOLDS[@]}"; do
        if ! fold_complete "${trial_num}" "${fold_num}"; then
            return 1
        fi
    done
    return 0
}

echo "Submitting coordinated PLCAv3 finalize CV sweep."
echo "Study: ${STUDY_NAME}"
echo "Storage: ${OPTUNA_STORAGE}"
echo "Total configs: ${TOTAL_CONFIGS} (${CONFIGS_PER_ROUND} per round)"
echo "Output: ${OUTPUT_DIR}"
echo "Resources: one GPU per fold job; GPU assignment cycles rtx,rtx,rtx,l40"
echo "Walltime: RTX=${RTX_TIME}, L40=${L40_TIME}"

bootstrap_study

configs_started=0
round=0

while (( configs_started < TOTAL_CONFIGS )); do
    round=$((round + 1))
    remaining=$((TOTAL_CONFIGS - configs_started))
    ask_n=${CONFIGS_PER_ROUND}
    if (( remaining < ask_n )); then
        ask_n=${remaining}
    fi

    echo "================================================================="
    echo "  Round ${round}: asking for ${ask_n} config(s)"
    echo "================================================================="

    ask_output=$(ask_configs "${ask_n}")
    trial_csv=$(parse_trial_numbers "${ask_output}")
    if [[ -z "${trial_csv}" ]]; then
        echo "No trial numbers returned by tune-ask. Output:" >&2
        echo "${ask_output}" >&2
        exit 1
    fi

    IFS=',' read -r -a trial_nums <<< "${trial_csv}"
    if (( ${#trial_nums[@]} == 0 )); then
        echo "No configs returned; stopping early."
        break
    fi

    echo "  Trial numbers: ${trial_csv}"

    job_records=()
    worker_scripts=()
    fold_task_idx=0

    for idx in "${!trial_nums[@]}"; do
        trial_num="${trial_nums[$idx]}"
        trial_job_records=()
        trial_worker_scripts=()
        submitted_ok=1

        for fold_num in "${FOLDS[@]}"; do
            if fold_complete "${trial_num}" "${fold_num}"; then
                echo "  Trial ${trial_num} fold ${fold_num}: metrics already present; skipping submission"
                continue
            fi

            gpu_class="${GPU_CLASS_SEQUENCE[$((fold_task_idx % ${#GPU_CLASS_SEQUENCE[@]}))]}"
            fold_task_idx=$((fold_task_idx + 1))

            if [[ "${gpu_class}" == "rtx" ]]; then
                partition="${RTX_PARTITION}"
                gpu_spec="${RTX_GPU}"
                time_limit="${RTX_TIME}"
            elif [[ "${gpu_class}" == "l40" ]]; then
                partition="${L40_PARTITION}"
                gpu_spec="${L40_GPU}"
                time_limit="${L40_TIME}"
            else
                echo "Unknown GPU class in GPU_CLASS_SEQUENCE: ${gpu_class}" >&2
                exit 1
            fi

            if record=$(submit_single_fold "${gpu_class}" "$((idx + 1))" "${trial_num}" "${fold_num}" "${partition}" "${gpu_spec}" "${time_limit}"); then
                trial_job_records+=("${record%%:*}")
                trial_worker_scripts+=("${record#*:}")
            else
                submitted_ok=0
                break
            fi
        done

        if (( submitted_ok == 0 )); then
            echo "  Failed to submit all missing folds for trial ${trial_num}; it will not be reported this round." >&2
        fi

        for job_id in "${trial_job_records[@]}"; do
            job_records+=("${job_id}")
        done
        for worker_script in "${trial_worker_scripts[@]}"; do
            worker_scripts+=("${worker_script}")
        done

        if (( ${#trial_job_records[@]} > 0 )); then
            echo "  Submitted trial ${trial_num} as job(s) $(IFS=,; echo "${trial_job_records[*]}")"
        elif trial_complete "${trial_num}"; then
            echo "  Trial ${trial_num}: all fold metrics already present"
        fi
    done

    for job_id in "${job_records[@]}"; do
        wait_for_job "${job_id}"
        check_job_exit "${job_id}"
    done

    for worker_script in "${worker_scripts[@]}"; do
        rm -f "${worker_script}"
    done

    completed_trial_nums=()
    for trial_num in "${trial_nums[@]}"; do
        if trial_complete "${trial_num}"; then
            completed_trial_nums+=("${trial_num}")
        else
            echo "  Trial ${trial_num}: one or more fold metrics are missing; not reporting this round" >&2
        fi
    done

    if (( ${#completed_trial_nums[@]} == 0 )); then
        echo "No trials completed all folds in round ${round}; stopping." >&2
        exit 1
    fi

    completed_trial_csv=$(IFS=,; echo "${completed_trial_nums[*]}")
    echo "  Reporting ${#completed_trial_nums[@]} completed config(s) to Optuna"
    tell_configs "${completed_trial_csv}"

    configs_started=$((configs_started + ${#completed_trial_nums[@]}))
done

echo "Submitted and reported ${configs_started} finalize config(s)."