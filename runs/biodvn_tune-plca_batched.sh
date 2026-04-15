#!/bin/bash
set -euo pipefail

# =====================================================================
# Coordinated fold-parallel tuning
#
# Each round:
#   1. tune-ask:  Optuna suggests N_CONFIGS trial configs, writes to disk
#   2. tune-fold: 5 fold workers per config train in parallel (N_CONFIGS×5 jobs)
#   3. tune-tell: Read fold metrics, compute cross-fold objective, report to Optuna
#
# This gives true cross-fold optimization with fold-level parallelism:
# 4 configs × 5 folds = 20 simultaneous GPU jobs per round.
# =====================================================================
EXPERIMENT="PLCA_tuning/biodvn_tune-plca_phase2-1"
TAG="PLCA_2-1"
YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
N_CONFIGS=4          # Configs to ask per round
N_FOLDS=5            # Folds per config
N_ROUNDS=10          # Total rounds (N_ROUNDS × N_CONFIGS = total trials)
BASE_SAMPLER_SEED=42
POLL_INTERVAL=30
MAX_RETRIES=2

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

# =====================================================================
# Helper: submit a fold worker for a specific trial + fold
# =====================================================================
submit_fold_worker() {
    local trial_num=$1 fold_num=$2
    local job_name="${TAG}_t${trial_num}_f${fold_num}"
    local exclude_flag
    exclude_flag=$(_build_exclude)
    local job_script
    job_script=$(mktemp "job_t${trial_num}_f${fold_num}_XXXXXX.sh")

    cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus=rtx2080ti:1
#SBATCH --time=16:00:00
#SBATCH -A nelson_p

eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}
python Proteus.py \\
    --run_mode tune-fold \\
    --experiment ${YAML} \\
    --trial_number ${trial_num} \\
    --test_fold ${fold_num}
EOL

    local job_id
    job_id=$(sbatch --parsable ${exclude_flag} "${job_script}")
    rm -f "${job_script}"
    echo "${job_id}"
}

# =====================================================================
# Record the node a failed job ran on into the blacklist file.
# =====================================================================
_blacklist_node() {
    local jid=$1
    local node
    node=$(sacct -j "${jid}" -n -o NodeList -X 2>/dev/null | head -1 | tr -d '[:space:]')
    if [[ -n "${node}" && "${node}" != "None" ]]; then
        if ! grep -qxF "${node}" "${BLACKLIST_FILE}" 2>/dev/null; then
            echo "${node}" >> "${BLACKLIST_FILE}"
            echo "  [$(date '+%H:%M:%S')] Added ${node} to blacklist (${BLACKLIST_FILE})"
        fi
    fi
}

# =====================================================================
# Wait for all jobs in an array to finish, resubmitting failures.
# Args: parallel arrays JOB_IDS, JOB_TRIALS, JOB_FOLDS, JOB_RETRIES
# =====================================================================
wait_for_jobs() {
    local n_slots=${#JOB_IDS[@]}
    while true; do
        sleep "${POLL_INTERVAL}"
        local alive=0
        for i in $(seq 0 $((n_slots - 1))); do
            local jid="${JOB_IDS[$i]}"
            local trial="${JOB_TRIALS[$i]}"
            local fold="${JOB_FOLDS[$i]}"
            local retries="${JOB_RETRIES[$i]}"

            local state
            state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
            if [[ -n "${state}" ]]; then
                alive=$((alive + 1))
                continue
            fi

            # Job no longer in queue — check exit status
            local exit_code
            exit_code=$(sacct -j "${jid}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')
            if [[ "${exit_code}" == "0:0" ]]; then
                continue
            fi

            # Job failed
            _blacklist_node "${jid}"
            if (( retries >= MAX_RETRIES )); then
                echo "  [$(date '+%H:%M:%S')] trial=${trial} fold=${fold} (job ${jid}) failed — retry cap reached, skipping."
                continue
            fi

            echo "  [$(date '+%H:%M:%S')] trial=${trial} fold=${fold} (job ${jid}) failed (exit ${exit_code}) — resubmitting..."
            local new_jid
            new_jid=$(submit_fold_worker "${trial}" "${fold}")
            JOB_IDS[$i]="${new_jid}"
            JOB_RETRIES[$i]=$((retries + 1))
            echo "    New job: ${new_jid} (retry $((retries + 1))/${MAX_RETRIES})"
            alive=$((alive + 1))
        done

        if (( alive == 0 )); then
            break
        fi
    done
}

echo "=== PLCA Phase 2.1: Coordinated tuning (20 epochs, GRL tuning) ==="
echo "    ${N_CONFIGS} configs/round × ${N_FOLDS} folds × ${N_ROUNDS} rounds = $((N_CONFIGS * N_ROUNDS)) total trials"
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

    # Step 2: Submit fold workers for each trial
    echo "  Submitting $((n_trials * N_FOLDS)) fold workers..."
    declare -a JOB_IDS=()
    declare -a JOB_TRIALS=()
    declare -a JOB_FOLDS=()
    declare -a JOB_RETRIES=()

    for trial_num in "${TRIAL_NUMS[@]}"; do
        for fold in $(seq 1 "${N_FOLDS}"); do
            jid=$(submit_fold_worker "${trial_num}" "${fold}")
            JOB_IDS+=("${jid}")
            JOB_TRIALS+=("${trial_num}")
            JOB_FOLDS+=("${fold}")
            JOB_RETRIES+=(0)
            echo "    trial=${trial_num} fold=${fold}: job ${jid}"
        done
    done

    # Step 3: Wait for all fold workers to finish
    echo "  Waiting for $((n_trials * N_FOLDS)) fold workers..."
    wait_for_jobs

    # Step 4: tune-tell — report results back to Optuna
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
