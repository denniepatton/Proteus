#!/bin/bash
set -euo pipefail

# Submit refinement fold-parallel PLCAv3 tuning workers.
# This path uses Proteus.py --run_mode tune --test_fold <K>, so Hyperband
# pruning and the built-in promotion phase both stay active.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="config/experiments/PLCA_tuning/biodvn_tune-plcav3_refine.yaml"
ENV_NAME="${ENV_NAME:-proteus}"
ACCOUNT="${ACCOUNT:-nelson_p}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///results_training/optuna_plcav3_refine.db}"
STUDY_NAME="${STUDY_NAME:-biodvn_tune-plcav3_refine}"
TRIALS_PER_WORKER=64
BASE_SAMPLER_SEED=4200
PYTHON_BIN="${HOME}/.local/share/mamba/envs/${ENV_NAME}/bin/python"

RTX_COUNT="${RTX_COUNT:-15}"
RTX_PARTITION="${RTX_PARTITION:-campus-new}"
RTX_GPU="${RTX_GPU:-rtx2080ti:1}"
RTX_TIME="${RTX_TIME:-72:00:00}"

L40_COUNT="${L40_COUNT:-5}"
L40_PARTITION="${L40_PARTITION:-chorus}"
L40_GPU="${L40_GPU:-l40s:1}"
L40_TIME="${L40_TIME:-72:00:00}"

RTX_FOLDS=(1 2 3 4 5)
L40_FOLDS=(1 2 3 4 5)

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
print(f"[plcav3-refine-bootstrap] ready study={study_name} storage={storage}", flush=True)
PY
}

submit_worker() {
    local role="$1"
    local worker_idx="$2"
    local fold_num="$3"
    local partition="$4"
    local gpu_spec="$5"
    local time_limit="$6"
    local sampler_seed="$7"
    local job_name="plcav3_refine_f${fold_num}_${role}_${worker_idx}"
    local worker_python="${WORKER_PYTHON}"

    sbatch \
        --job-name="${job_name}" \
        --output="logs/${job_name}_%j.out" \
        --error="logs/${job_name}_%j.err" \
        --partition="${partition}" \
        --cpus-per-task=8 \
        --mem=200G \
        --gpus="${gpu_spec}" \
        --time="${time_limit}" \
        -A "${ACCOUNT}" \
        --wrap="cd ${REPO_ROOT} && export PYTHONUNBUFFERED=1 && echo '[plcav3-refine] host='\"\$(hostname)\"' fold=${fold_num} role=${role} seed=${sampler_seed} python=${worker_python}' && ${worker_python} Proteus.py --run_mode tune --experiment ${EXPERIMENT} --test_fold ${fold_num} --optuna_storage ${OPTUNA_STORAGE} --study_name ${STUDY_NAME} --sampler_seed ${sampler_seed} --n_trials ${TRIALS_PER_WORKER}"
}

echo "Submitting refinement PLCAv3 fold-parallel Hyperband tuning workers."
echo "Study: ${STUDY_NAME}"
echo "Storage: ${OPTUNA_STORAGE}"
echo "Each worker runs fold-parallel Optuna tuning with Hyperband pruning and promotion."

bootstrap_study

for idx in $(seq 1 "${RTX_COUNT}"); do
    fold_idx=$(( (idx - 1) % ${#RTX_FOLDS[@]} ))
    fold_num="${RTX_FOLDS[$fold_idx]}"
    sampler_seed=$(( BASE_SAMPLER_SEED + idx ))
    submit_worker rtx "${idx}" "${fold_num}" "${RTX_PARTITION}" "${RTX_GPU}" "${RTX_TIME}" "${sampler_seed}"
done

for idx in $(seq 1 "${L40_COUNT}"); do
    fold_idx=$(( (idx - 1) % ${#L40_FOLDS[@]} ))
    fold_num="${L40_FOLDS[$fold_idx]}"
    sampler_seed=$(( BASE_SAMPLER_SEED + 100 + idx ))
    submit_worker l40 "${idx}" "${fold_num}" "${L40_PARTITION}" "${L40_GPU}" "${L40_TIME}" "${sampler_seed}"
done

echo "Submitted refinement PLCAv3 tuning workers."