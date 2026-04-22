#!/bin/bash
set -euo pipefail

# =====================================================================
# Coordinated tuning — serial folds on L40s (phase 2.2)
#
# Targeted ablation of internal-loss weights: beta_tcvae, beta_grl,
# beta_reconstruction, internal_weight.  Dropout fixed at 0.30,
# LR narrowed to [5e-5, 4e-4].
#
# Resource budget: 5 L40s (5 configs/round, each trains folds 1-5 serial)
# =====================================================================
EXPERIMENT="PLCA_tuning/biodvn_tune-plca_phase2-2"
TAG="PLCA_2-2s"
YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
N_CONFIGS=5          # 5 L40s, 1 config per GPU
N_FOLDS=5
N_ROUNDS=2           # 2 rounds × 5 configs = 10 trials
BASE_SAMPLER_SEED=42
POLL_INTERVAL=60
MAX_RETRIES=2
WALL_TIME="72:00:00"

BLACKLIST_FILE="runs/bad_nodes.txt"
touch "${BLACKLIST_FILE}"

# =====================================================================
_build_exclude() {
    local nodes
    nodes=$(sort -u "${BLACKLIST_FILE}" | paste -sd ',' -)
    if [[ -n "${nodes}" ]]; then
        echo "--exclude=${nodes}"
    fi
}

# =====================================================================
submit_serial_trial() {
    local trial_num=$1
    local job_name="${TAG}_t${trial_num}"
    local exclude_flag
    exclude_flag=$(_build_exclude)
    local job_script
    job_script=$(mktemp "job_serial_t${trial_num}_XXXXXX.sh")

    cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --partition=chorus
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus=l40s:1
#SBATCH --time=${WALL_TIME}
#SBATCH -A ha_g

eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}

echo "[Serial] Trial ${trial_num}: running folds 1-${N_FOLDS} on \$(hostname)"

for fold in \$(seq 1 ${N_FOLDS}); do
    echo ""
    echo "========================================"
    echo "[Serial] Trial ${trial_num}, fold \${fold}/${N_FOLDS}"
    echo "========================================"
    python Proteus.py \\
        --run_mode tune-fold \\
        --experiment ${YAML} \\
        --trial_number ${trial_num} \\
        --test_fold \${fold}
done

echo ""
echo "[Serial] Trial ${trial_num}: all ${N_FOLDS} folds complete."
EOL

    local job_id
    job_id=$(sbatch --parsable ${exclude_flag} "${job_script}")
    rm -f "${job_script}"
    echo "${job_id}"
}

# =====================================================================
_blacklist_node() {
    local jid=$1
    local node
    node=$(sacct -j "${jid}" -n -o NodeList -X 2>/dev/null | head -1 | tr -d '[:space:]')
    if [[ -n "${node}" && "${node}" != "None" ]]; then
        if ! grep -qxF "${node}" "${BLACKLIST_FILE}" 2>/dev/null; then
            echo "${node}" >> "${BLACKLIST_FILE}"
            echo "  [$(date '+%H:%M:%S')] Blacklisted node: ${node}"
        fi
    fi
}

# =====================================================================
wait_for_jobs() {
    local n_slots=${#JOB_IDS[@]}
    while true; do
        sleep "${POLL_INTERVAL}"
        local alive=0
        for i in $(seq 0 $((n_slots - 1))); do
            local jid="${JOB_IDS[$i]}"
            local trial="${JOB_TRIALS[$i]}"
            local retries="${JOB_RETRIES[$i]}"

            local state
            state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
            if [[ -n "${state}" ]]; then
                alive=$((alive + 1))
                continue
            fi

            local exit_code
            exit_code=$(sacct -j "${jid}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')
            if [[ "${exit_code}" == "0:0" ]]; then
                continue
            fi

            _blacklist_node "${jid}"
            if (( retries >= MAX_RETRIES )); then
                echo "  [$(date '+%H:%M:%S')] trial=${trial} (job ${jid}) failed — retry cap reached, skipping."
                continue
            fi

            echo "  [$(date '+%H:%M:%S')] trial=${trial} (job ${jid}) failed (exit ${exit_code}) — resubmitting..."
            local new_jid
            new_jid=$(submit_serial_trial "${trial}")
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

# =====================================================================
# Main loop
# =====================================================================
echo "=== PLCA Phase 2.2: Serial-fold tuning (L40s, ${N_CONFIGS} GPUs) ==="
echo "    ${N_CONFIGS} configs/round × ${N_FOLDS} folds(serial) × ${N_ROUNDS} rounds = $((N_CONFIGS * N_ROUNDS)) trials"
echo ""

for round in $(seq 1 "${N_ROUNDS}"); do
    echo "================================================================="
    echo "  Round ${round}/${N_ROUNDS}: Asking Optuna for ${N_CONFIGS} configs"
    echo "================================================================="

    # Step 1: tune-ask
    ask_output=$(python Proteus.py \
        --run_mode tune-ask \
        --experiment "${YAML}" \
        --n_configs "${N_CONFIGS}" \
        --sampler_seed "${BASE_SAMPLER_SEED}")

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

    # Step 2: Submit one job per config
    echo "  Submitting ${n_trials} serial-fold jobs (1 GPU each)..."
    declare -a JOB_IDS=()
    declare -a JOB_TRIALS=()
    declare -a JOB_RETRIES=()

    for trial_num in "${TRIAL_NUMS[@]}"; do
        jid=$(submit_serial_trial "${trial_num}")
        JOB_IDS+=("${jid}")
        JOB_TRIALS+=("${trial_num}")
        JOB_RETRIES+=(0)
        echo "    trial=${trial_num}: job ${jid} (folds 1-${N_FOLDS} serial)"
    done

    # Step 3: Wait for all config jobs to finish
    echo "  Waiting for ${n_trials} jobs..."
    wait_for_jobs

    # Step 4: tune-tell
    echo "  Reporting results to Optuna..."
    python Proteus.py \
        --run_mode tune-tell \
        --experiment "${YAML}" \
        --trial_numbers "${trial_csv}" \
        --sampler_seed "${BASE_SAMPLER_SEED}"

    echo "  Round ${round} complete."
    echo ""
done

echo "=== PLCA Phase 2.2 tuning complete (${N_ROUNDS} rounds). ==="
