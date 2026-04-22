#!/bin/bash
set -euo pipefail

# =====================================================================
# Coordinated tuning — distributed folds on 2080ti (15 GPUs per round)
#
# Designed to run IN PARALLEL with biodvn_tune-plca_serial.sh.
# Both share the same Optuna study/DB — coordinated_ask atomically
# reserves trial numbers, so no duplicate configs.
#
# Resource budget: 15 2080ti (3 configs/round × 5 folds distributed)
# Partner script uses 5 L40s (5 configs/round, folds serial)
# Combined: 8 configs/round, 5 rounds = 40 total trials
#
# Each round:
#   1. tune-ask:  Optuna suggests N_CONFIGS trial configs
#   2. Single sbatch allocation with N_CONFIGS × N_FOLDS tasks (1 GPU each)
#   3. tune-tell: Report cross-fold results to Optuna
# =====================================================================
EXPERIMENT="PLCA_tuning/biodvn_tune-plca_phase2-1"
TAG="PLCA_2-1b"
YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
N_CONFIGS=3          # 3 configs × 5 folds = 15 GPUs per round
N_FOLDS=5
N_ROUNDS=5           # 5 rounds × 3 configs = 15 trials from this script
BASE_SAMPLER_SEED=42
MAX_RETRIES=2
WALL_TIME="72:00:00"

BLACKLIST_FILE="runs/bad_nodes.txt"
touch "${BLACKLIST_FILE}"

# =====================================================================
# Build SLURM --exclude list from the blacklist file
# =====================================================================
_build_exclude() {
    local nodes
    nodes=$(sort -u "${BLACKLIST_FILE}" | paste -sd ',' -)
    if [[ -n "${nodes}" ]]; then
        echo "--exclude=${nodes}"
    fi
}

n_gpus_per_round=$((N_CONFIGS * N_FOLDS))

echo "=== PLCA Phase 2.1: Batched-fold tuning (2080ti, ${N_CONFIGS}×${N_FOLDS}=${n_gpus_per_round} GPUs/round) ==="
echo "    ${N_CONFIGS} configs/round × ${N_FOLDS} folds × ${N_ROUNDS} rounds = $((N_CONFIGS * N_ROUNDS)) trials from this script"
echo "    Shares Optuna study with serial script (combined target: 40 trials)"
echo ""

for round in $(seq 1 "${N_ROUNDS}"); do
    echo "================================================================="
    echo "  Round ${round}/${N_ROUNDS}: Asking Optuna for ${N_CONFIGS} configs"
    echo "================================================================="

    # Step 1: tune-ask — get trial configs from Optuna
    ask_output=$(python Proteus.py \
        --run_mode tune-ask \
        --experiment "${YAML}" \
        --n_configs "${N_CONFIGS}" \
        --sampler_seed "${BASE_SAMPLER_SEED}")

    # Parse trial numbers from output (line: TRIAL_NUMBERS=0,1,2,3)
    trial_line=$(echo "${ask_output}" | grep '^TRIAL_NUMBERS=' | tail -1)
    if [[ -z "${trial_line}" ]]; then
        echo "  ERROR: tune-ask did not return trial numbers. Output:"
        echo "${ask_output}"
        exit 1
    fi
    trial_csv="${trial_line#TRIAL_NUMBERS=}"
    IFS=',' read -ra TRIAL_NUMS <<< "${trial_csv}"
    n_trials=${#TRIAL_NUMS[@]}

    echo "  Got ${n_trials} trial(s): ${trial_csv}"

    if (( n_trials == 0 )); then
        echo "  No trials returned — stopping early."
        break
    fi

    n_tasks=$((n_trials * N_FOLDS))

    # Step 2: Build a task manifest mapping SLURM_PROCID → (trial, fold)
    MANIFEST=$(mktemp "manifest_round${round}_XXXXXX.tsv")
    task_id=0
    for trial_num in "${TRIAL_NUMS[@]}"; do
        for fold in $(seq 1 "${N_FOLDS}"); do
            echo -e "${task_id}\t${trial_num}\t${fold}" >> "${MANIFEST}"
            task_id=$((task_id + 1))
        done
    done

    # Step 3: Write the per-task worker script
    WORKER_SCRIPT=$(mktemp "worker_round${round}_XXXXXX.sh")
    cat > "${WORKER_SCRIPT}" <<'WORKER_EOF'
#!/bin/bash
set -euo pipefail
MANIFEST_FILE="$1"
YAML_FILE="$2"
ENV_NAME="$3"

eval "$(micromamba shell hook -s bash)"
micromamba activate "${ENV_NAME}"

# Look up this task's trial and fold from the manifest
TASK_LINE=$(awk -v id="${SLURM_PROCID}" '$1 == id' "${MANIFEST_FILE}")
TRIAL_NUM=$(echo "${TASK_LINE}" | cut -f2)
FOLD_NUM=$(echo "${TASK_LINE}" | cut -f3)

echo "[Task ${SLURM_PROCID}] trial=${TRIAL_NUM} fold=${FOLD_NUM} node=$(hostname)"

python Proteus.py \
    --run_mode tune-fold \
    --experiment "${YAML_FILE}" \
    --trial_number "${TRIAL_NUM}" \
    --test_fold "${FOLD_NUM}"
WORKER_EOF
    chmod +x "${WORKER_SCRIPT}"

    # Step 4: Submit single allocation with all tasks
    exclude_flag=$(_build_exclude)
    job_name="${TAG}_r${round}"

    echo "  Submitting 1 job with ${n_tasks} tasks (${n_tasks} GPUs)..."

    job_id=$(sbatch --parsable \
        --job-name="${job_name}" \
        --output="logs/${job_name}_task%t_%j.out" \
        --error="logs/${job_name}_task%t_%j.err" \
        --partition=campus-new \
        --ntasks="${n_tasks}" \
        --cpus-per-task=8 \
        --mem-per-cpu=20G \
        --gpus-per-task=rtx2080ti:1 \
        --time="${WALL_TIME}" \
        -A ha_g \
        ${exclude_flag} \
        --wrap="srun bash ${WORKER_SCRIPT} ${MANIFEST} ${YAML} ${ENV_NAME}")

    echo "  Job ${job_id}: ${n_tasks} tasks × 1 GPU each"

    # Step 5: Wait for the single job to finish
    echo "  Waiting for job ${job_id}..."
    while true; do
        sleep 30
        state=$(squeue -j "${job_id}" -h -o '%T' 2>/dev/null || true)
        if [[ -z "${state}" ]]; then
            break
        fi
    done

    # Check exit status
    exit_code=$(sacct -j "${job_id}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')
    if [[ "${exit_code}" != "0:0" ]]; then
        echo "  WARNING: Job ${job_id} exited with ${exit_code}"
        echo "  Checking individual task statuses..."
        sacct -j "${job_id}" --format=JobID,NodeList,ExitCode,State,Elapsed -n
        # Blacklist failed nodes
        while IFS= read -r node; do
            node=$(echo "${node}" | tr -d '[:space:]')
            if [[ -n "${node}" && "${node}" != "None" ]]; then
                if ! grep -qxF "${node}" "${BLACKLIST_FILE}" 2>/dev/null; then
                    echo "${node}" >> "${BLACKLIST_FILE}"
                    echo "  Blacklisted node: ${node}"
                fi
            fi
        done < <(sacct -j "${job_id}" -n -o NodeList --state=FAILED 2>/dev/null)
    fi

    # Clean up temp files
    rm -f "${MANIFEST}" "${WORKER_SCRIPT}"

    # Step 6: tune-tell — report results back to Optuna
    echo "  Reporting results to Optuna..."
    python Proteus.py \
        --run_mode tune-tell \
        --experiment "${YAML}" \
        --trial_numbers "${trial_csv}" \
        --sampler_seed "${BASE_SAMPLER_SEED}"

    echo "  Round ${round} complete."
    echo ""
done

echo "=== PLCA Phase 2.1 tuning complete (${N_ROUNDS} rounds). ==="
