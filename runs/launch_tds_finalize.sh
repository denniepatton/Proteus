#!/bin/bash
set -euo pipefail

# Submit a coordinated full-CV TDS v2 finalize sweep.
#
# Each Optuna config is evaluated across all 5 folds as one coordinated trial.
# Every fold is submitted as an independent one-GPU SLURM job; no multi-node or
# multi-task GPU jobs are used.
#
# The launcher resumes pending coordinated trials from the output directory and
# auto-retries CUDA init/node failures on the same GPU class while excluding
# the failed node.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="${EXPERIMENT:-config/experiments/TDS_tuning/biodvn_tune-tds-v2_finalize.yaml}"
ENV_NAME="${ENV_NAME:-proteus}"
ACCOUNT="${ACCOUNT:-nelson_p}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///results_training/optuna_tds_v2_finalize.db}"
STUDY_NAME="${STUDY_NAME:-biodvn_tune-tds-v2_finalize}"
TOTAL_CONFIGS="${TOTAL_CONFIGS:-48}"
CONFIGS_PER_ROUND="${CONFIGS_PER_ROUND:-4}"
BASE_SAMPLER_SEED="${BASE_SAMPLER_SEED:-7200}"
PYTHON_BIN="${HOME}/.local/share/mamba/envs/${ENV_NAME}/bin/python"
JOB_PREFIX="${JOB_PREFIX:-tds_v2_finalize}"
# Site-specific Zarr location. Override by exporting PROTEUS_ZARR_BASE before
# launching (e.g. a fast-scratch path on your cluster).
DEFAULT_PROTEUS_ZARR_BASE="${DEFAULT_PROTEUS_ZARR_BASE:-./data/zarr}"
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
export PROTEUS_ZARR_BASE="${PROTEUS_ZARR_BASE:-${DEFAULT_PROTEUS_ZARR_BASE}}"

validate_data_paths() {
    EXPERIMENT_ENV="${EXPERIMENT}" \
    "${WORKER_PYTHON}" - <<'PY'
import csv
import os

from config.experiment_spec import load_experiment_spec
from proteus_utils.data_helpers import zarr_path_for_sample

exp = load_experiment_spec(os.environ["EXPERIMENT_ENV"])
training_set = exp.data.training_set
zarr_base = exp.data.zarr_base

missing = []
checked = 0
with open(training_set, "r", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        cohort = row["Cohort"].strip()
        if cohort == "PHT":
            continue
        sample = row["Sample"].strip()
        checked += 1
        zarr_path = zarr_path_for_sample(sample, cohort, zarr_base)
        if not os.path.isdir(zarr_path):
            missing.append(zarr_path)

if missing:
    preview = "\n".join(f"  {path}" for path in missing[:8])
    remainder = "" if len(missing) <= 8 else f"\n  ... and {len(missing) - 8} more"
    raise SystemExit(
        f"[tds-v2-finalize] Missing {len(missing)} non-PHT Zarr store(s) under {zarr_base} "
        f"for {training_set}.\nFirst missing paths:\n{preview}{remainder}"
    )

print(
    f"[tds-v2-finalize] validated {checked} non-PHT Zarr store(s) under {zarr_base}",
    flush=True,
)
PY
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
print(f"[tds-v2-finalize-bootstrap] ready study={study_name} storage={storage}", flush=True)
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

count_completed_trials() {
    OPTUNA_STORAGE_ENV="${OPTUNA_STORAGE}" \
    STUDY_NAME_ENV="${STUDY_NAME}" \
    "${WORKER_PYTHON}" - <<'PY'
import os

import optuna

storage = os.environ["OPTUNA_STORAGE_ENV"]
study_name = os.environ["STUDY_NAME_ENV"]

study = optuna.load_study(study_name=study_name, storage=storage)
complete_count = sum(
    trial.state == optuna.trial.TrialState.COMPLETE
    for trial in study.trials
)
print(complete_count, flush=True)
PY
}

discover_pending_trials() {
    OUTPUT_DIR_ENV="${OUTPUT_DIR}" \
    "${WORKER_PYTHON}" - <<'PY'
import glob
import json
import os
import re

output_dir = os.environ["OUTPUT_DIR_ENV"]
pending_trials = []

for pending_path in sorted(glob.glob(os.path.join(output_dir, "trial_*", "coordinated_pending.json"))):
    trial_dir = os.path.basename(os.path.dirname(pending_path))
    match = re.fullmatch(r"trial_(\d+)", trial_dir)
    if match is None:
        continue
    try:
        with open(pending_path, "r") as handle:
            payload = json.load(handle)
        if str(payload.get("status", "")).strip().lower() != "pending":
            continue
    except (OSError, json.JSONDecodeError):
        continue
    pending_trials.append(str(int(match.group(1))))

print(",".join(pending_trials), flush=True)
PY
}

submit_single_fold() {
    local gpu_class="$1"
    local config_idx="$2"
    local trial_num="$3"
    local fold_num="$4"
    local partition="$5"
    local gpu_spec="$6"
    local time_limit="$7"
    local exclude_nodes="${8:-}"
    local job_name="${JOB_PREFIX}_${gpu_class}_cfg${config_idx}_t${trial_num}_f${fold_num}"
    local worker_script
    worker_script=$(mktemp "${JOB_PREFIX}_${gpu_class}_XXXXXX.sh")
    local exclude_args=()

    if [[ -n "${exclude_nodes}" ]]; then
        exclude_args+=("--exclude=${exclude_nodes}")
    fi

    cat > "${worker_script}" <<'WORKER_EOF'
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

trial_num="$1"
fold_num="$2"
experiment_path="$3"
worker_python="$4"

echo "[tds-v2-finalize] host=$(hostname) trial=${trial_num} fold=${fold_num} task=0"

"${worker_python}" -u Proteus.py \
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
        --export="ALL,PROTEUS_ZARR_BASE=${PROTEUS_ZARR_BASE}" \
        --partition="${partition}" \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem-per-cpu=20G \
        --gpus="${gpu_spec}" \
        --time="${time_limit}" \
        -A "${ACCOUNT}" \
        "${exclude_args[@]}" \
        "${worker_script}" "${trial_num}" "${fold_num}" "${EXPERIMENT}" "${WORKER_PYTHON}"); then
        rm -f "${worker_script}"
        return 1
    fi

    if [[ -z "${job_id}" ]]; then
        rm -f "${worker_script}"
        echo "sbatch returned an empty job id for trial ${trial_num}, fold ${fold_num}" >&2
        return 1
    fi

    printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "${job_id}" \
        "${job_name}" \
        "${worker_script}" \
        "${gpu_class}" \
        "${partition}" \
        "${gpu_spec}" \
        "${time_limit}" \
        "${trial_num}" \
        "${fold_num}" \
        "${config_idx}" \
        "${exclude_nodes}"
}

job_is_active() {
    local job_id="$1"
    local state
    state=$(squeue -j "${job_id}" -h -o '%T' 2>/dev/null || true)
    [[ -n "${state}" ]]
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

job_nodelist() {
    local job_id="$1"
    local node
    node=$(sacct -j "${job_id}" -n -o NodeList -X 2>/dev/null | head -1 | tr -d '[:space:]')
    if [[ -z "${node}" || "${node}" == "None" ]]; then
        return 0
    fi
    echo "${node}"
}

job_failed_for_cuda_init() {
    local job_name="$1"
    local job_id="$2"
    local err_path="logs/${job_name}_${job_id}.err"
    local out_path="logs/${job_name}_${job_id}.out"
    local cuda_init_pattern='CUDA error: CUDA unknown error|CUDA-capable device\(s\) is/are busy or unavailable|No CUDA GPUs are available|CUDA_VISIBLE_DEVICES|torch\._C\._cuda_init'

    if grep -Eq "${cuda_init_pattern}" "${err_path}" 2>/dev/null; then
        return 0
    fi
    if grep -Eq "${cuda_init_pattern}" "${out_path}" 2>/dev/null; then
        return 0
    fi
    return 1
}

merge_exclude_nodes() {
    local existing_nodes="$1"
    local new_node="$2"
    local node

    if [[ -z "${new_node}" ]]; then
        echo "${existing_nodes}"
        return 0
    fi

    if [[ -z "${existing_nodes}" ]]; then
        echo "${new_node}"
        return 0
    fi

    IFS=',' read -r -a existing_node_list <<< "${existing_nodes}"
    for node in "${existing_node_list[@]}"; do
        if [[ "${node}" == "${new_node}" ]]; then
            echo "${existing_nodes}"
            return 0
        fi
    done

    echo "${existing_nodes},${new_node}"
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

fold_job_active() {
    local trial_num="$1"
    local fold_num="$2"
    local job_names
    job_names=$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)
    if [[ -z "${job_names}" ]]; then
        return 1
    fi
    grep -Eq "^${JOB_PREFIX}_(rtx|l40)_cfg[0-9]+_t${trial_num}_f${fold_num}$" <<< "${job_names}"
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

run_trial_batch() {
    local trial_nums=("$@")
    local job_records=()
    local worker_scripts=()
    local failed_jobs=()
    local completed_trial_nums=()
    local incomplete_trial_nums=()
    local fold_task_idx=0
    local idx

    for idx in "${!trial_nums[@]}"; do
        local trial_num="${trial_nums[$idx]}"
        local trial_job_records=()
        local trial_worker_scripts=()
        local submitted_ok=1
        local fold_num

        for fold_num in "${FOLDS[@]}"; do
            local gpu_class
            local partition
            local gpu_spec
            local time_limit

            if fold_complete "${trial_num}" "${fold_num}"; then
                echo "  Trial ${trial_num} fold ${fold_num}: metrics already present; skipping submission"
                continue
            fi

            if fold_job_active "${trial_num}" "${fold_num}"; then
                echo "  Trial ${trial_num} fold ${fold_num}: job already queued/running; skipping duplicate submission"
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
                return 1
            fi

            if record=$(submit_single_fold "${gpu_class}" "$((idx + 1))" "${trial_num}" "${fold_num}" "${partition}" "${gpu_spec}" "${time_limit}"); then
                local _job_id _job_name _worker_script _record_gpu_class _record_partition _record_gpu_spec _record_time _record_trial _record_fold _record_config_idx _record_exclude_nodes
                IFS='|' read -r _job_id _job_name _worker_script _record_gpu_class _record_partition _record_gpu_spec _record_time _record_trial _record_fold _record_config_idx _record_exclude_nodes <<< "${record}"
                trial_job_records+=("${record}")
                trial_worker_scripts+=("${_worker_script}")
            else
                submitted_ok=0
                break
            fi
        done

        if (( submitted_ok == 0 )); then
            echo "  Failed to submit all missing folds for trial ${trial_num}; it will remain pending." >&2
        fi

        local record
        for record in "${trial_job_records[@]}"; do
            job_records+=("${record}")
        done
        local worker_script
        for worker_script in "${trial_worker_scripts[@]}"; do
            worker_scripts+=("${worker_script}")
        done

        if (( ${#trial_job_records[@]} > 0 )); then
            local submitted_job_ids=()
            for record in "${trial_job_records[@]}"; do
                submitted_job_ids+=("${record%%|*}")
            done
            echo "  Submitted trial ${trial_num} as job(s) $(IFS=,; echo "${submitted_job_ids[*]}")"
        elif trial_complete "${trial_num}"; then
            echo "  Trial ${trial_num}: all fold metrics already present"
        else
            echo "  Trial ${trial_num}: still missing one or more fold metrics"
        fi
    done

    local pending_job_records=("${job_records[@]}")
    while (( ${#pending_job_records[@]} > 0 )); do
        local next_pending_job_records=()
        local saw_completion=0

        local record
        for record in "${pending_job_records[@]}"; do
            local job_id job_name worker_script gpu_class partition gpu_spec time_limit trial_num fold_num config_idx exclude_nodes
            IFS='|' read -r job_id job_name worker_script gpu_class partition gpu_spec time_limit trial_num fold_num config_idx exclude_nodes <<< "${record}"

            if job_is_active "${job_id}"; then
                next_pending_job_records+=("${record}")
                continue
            fi

            saw_completion=1
            if ! check_job_exit "${job_id}"; then
                if job_failed_for_cuda_init "${job_name}" "${job_id}"; then
                    local failed_node retry_exclude_nodes retry_record
                    failed_node=$(job_nodelist "${job_id}")
                    retry_exclude_nodes=$(merge_exclude_nodes "${exclude_nodes}" "${failed_node}")
                    echo "  Job ${job_id} hit a CUDA init failure; resubmitting trial ${trial_num} fold ${fold_num} on ${gpu_class}${retry_exclude_nodes:+ while excluding ${retry_exclude_nodes}}"
                    if retry_record=$(submit_single_fold "${gpu_class}" "${config_idx}" "${trial_num}" "${fold_num}" "${partition}" "${gpu_spec}" "${time_limit}" "${retry_exclude_nodes}"); then
                        local retry_job_id retry_job_name retry_worker_script _retry_gpu_class _retry_partition _retry_gpu_spec _retry_time _retry_trial_num _retry_fold_num _retry_config_idx _retry_exclude_nodes
                        IFS='|' read -r retry_job_id retry_job_name retry_worker_script _retry_gpu_class _retry_partition _retry_gpu_spec _retry_time _retry_trial_num _retry_fold_num _retry_config_idx _retry_exclude_nodes <<< "${retry_record}"
                        worker_scripts+=("${retry_worker_script}")
                        next_pending_job_records+=("${retry_record}")
                    else
                        echo "  Retry submission failed for trial ${trial_num} fold ${fold_num}" >&2
                        failed_jobs+=("${job_id}")
                    fi
                else
                    failed_jobs+=("${job_id}")
                fi
            fi
        done

        pending_job_records=("${next_pending_job_records[@]}")
        if (( ${#pending_job_records[@]} > 0 && saw_completion == 0 )); then
            sleep 30
        fi
    done

    local worker_script
    for worker_script in "${worker_scripts[@]}"; do
        rm -f "${worker_script}"
    done

    local trial_num
    for trial_num in "${trial_nums[@]}"; do
        if trial_complete "${trial_num}"; then
            completed_trial_nums+=("${trial_num}")
        else
            incomplete_trial_nums+=("${trial_num}")
            echo "  Trial ${trial_num}: one or more fold metrics are missing; leaving pending" >&2
        fi
    done

    if (( ${#completed_trial_nums[@]} > 0 )); then
        local completed_trial_csv
        completed_trial_csv=$(IFS=,; echo "${completed_trial_nums[*]}")
        echo "  Reporting ${#completed_trial_nums[@]} completed config(s) to Optuna"
        tell_configs "${completed_trial_csv}"
    fi

    if (( ${#failed_jobs[@]} > 0 )); then
        echo "  One or more fold jobs failed: $(IFS=,; echo "${failed_jobs[*]}")" >&2
        return 1
    fi

    if (( ${#incomplete_trial_nums[@]} > 0 )); then
        echo "  Pending trial(s) still incomplete: $(IFS=,; echo "${incomplete_trial_nums[*]}")" >&2
        return 1
    fi

    return 0
}

echo "Submitting coordinated TDS v2 finalize CV sweep."
echo "Study: ${STUDY_NAME}"
echo "Storage: ${OPTUNA_STORAGE}"
echo "Total configs: ${TOTAL_CONFIGS} (${CONFIGS_PER_ROUND} per round)"
echo "Output: ${OUTPUT_DIR}"
echo "Zarr base: ${PROTEUS_ZARR_BASE}"
echo "Objective: mean cohort_tumor_soft_objective across folds"
echo "Resources: one GPU per fold job; GPU assignment cycles rtx,rtx,rtx,l40"
echo "Walltime: RTX=${RTX_TIME}, L40=${L40_TIME}"

validate_data_paths
bootstrap_study

configs_started=$(count_completed_trials)
echo "Existing COMPLETE trial count: ${configs_started}"

carryover_trial_nums=()
pending_trial_csv=$(discover_pending_trials)
if [[ -n "${pending_trial_csv}" ]]; then
    IFS=',' read -r -a pending_trial_nums <<< "${pending_trial_csv}"
    echo "Resuming pending trial(s) before asking for new configs: ${pending_trial_csv}"

    pending_complete_trial_nums=()
    pending_incomplete_trial_nums=()
    for trial_num in "${pending_trial_nums[@]}"; do
        if trial_complete "${trial_num}"; then
            pending_complete_trial_nums+=("${trial_num}")
        else
            pending_incomplete_trial_nums+=("${trial_num}")
        fi
    done

    if (( ${#pending_complete_trial_nums[@]} > 0 )); then
        completed_trial_csv=$(IFS=,; echo "${pending_complete_trial_nums[*]}")
        echo "Reporting already-complete pending trial(s): ${completed_trial_csv}"
        tell_configs "${completed_trial_csv}"
    fi

    configs_started=$(count_completed_trials)
    echo "After reporting complete pending trials, COMPLETE count is ${configs_started}"

    if (( ${#pending_incomplete_trial_nums[@]} > 0 )); then
        carryover_trial_nums=("${pending_incomplete_trial_nums[@]}")
        echo "Carrying ${#carryover_trial_nums[@]} incomplete pending trial(s) into the next batch: $(IFS=,; echo "${carryover_trial_nums[*]}")"
    fi
fi

round=0

while (( configs_started < TOTAL_CONFIGS )); do
    round=$((round + 1))
    remaining=$((TOTAL_CONFIGS - configs_started))
    ask_n=${CONFIGS_PER_ROUND}
    if (( ${#carryover_trial_nums[@]} > 0 )); then
        ask_n=$((ask_n - ${#carryover_trial_nums[@]}))
    fi
    if (( ask_n < 0 )); then
        ask_n=0
    fi
    if (( remaining < ask_n )); then
        ask_n=${remaining}
    fi

    echo "================================================================="
    if (( ${#carryover_trial_nums[@]} > 0 )); then
        echo "  Round ${round}: ${#carryover_trial_nums[@]} carryover pending + asking for ${ask_n} new config(s)"
    else
        echo "  Round ${round}: asking for ${ask_n} config(s)"
    fi
    echo "================================================================="

    trial_nums=()
    if (( ask_n > 0 )); then
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
    elif (( ${#carryover_trial_nums[@]} == 0 )); then
        echo "No new configs requested and no carryover trials remain; stopping."
        break
    else
        echo "  No new configs asked this round; filling the batch with carryover pending trials only"
    fi

    batch_trial_nums=("${carryover_trial_nums[@]}" "${trial_nums[@]}")
    carryover_trial_nums=()

    if ! run_trial_batch "${batch_trial_nums[@]}"; then
        configs_started=$(count_completed_trials)
        echo "Round ${round} stopped with failures. Reported COMPLETE trials so far: ${configs_started}" >&2
        exit 1
    fi

    configs_started=$(count_completed_trials)
    echo "Completed configs reported so far: ${configs_started}/${TOTAL_CONFIGS}"
done

echo "Submitted and reported ${configs_started} finalize config(s)."