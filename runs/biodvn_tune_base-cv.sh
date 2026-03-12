#!/bin/bash
set -euo pipefail

# =====================================================================
# Phase 3.2R — Sequential 5-fold CV per trial (20 parallel workers)
#
# Each worker runs ALL 5 folds sequentially per trial (no --test_fold).
# With 20 GPUs, we get 20 concurrent trial evaluations.
# Each worker runs n_trials_per_fold=1.
# =====================================================================
EXPERIMENT="biodvn_tune_phase3-2R"
TAG="BTP_3-2R"
YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
POLL_INTERVAL=15
MONITOR_INTERVAL=120
MAX_RETRIES=3

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
# Helper: submit a sequential-CV worker (no --test_fold → all 5 folds)
# =====================================================================
submit_worker() {
    local partition=$1 gpu_spec=$2 time_limit=$3 suffix=$4
    local job_name="${TAG}${suffix}"
    local exclude_flag
    exclude_flag=$(_build_exclude)
    local job_script
    job_script=$(mktemp "job_cv_${suffix}_XXXXXX.sh")

    cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=8
#SBATCH --gpus=${gpu_spec}
#SBATCH --time=${time_limit}
#SBATCH -A nelson_p

micromamba run -n ${ENV_NAME} python Proteus.py \\
    --run_mode tune \\
    --experiment ${YAML} \\
    --n_trials 1
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
# Phase 1: Submit worker 1 on RTX 2080 Ti to initialize the Optuna DB
# =====================================================================
echo "=== Phase 1: Submitting worker 1 (RTX 2080 Ti) to initialize study ==="
INIT_JOB=$(submit_worker campus-new "rtx2080ti:1" "72:00:00" "_w01")
echo "  Worker 1 submitted: job ${INIT_JOB}"

echo "=== Waiting for worker 1 (job ${INIT_JOB}) to start running ==="
while true; do
    state=$(squeue -j "${INIT_JOB}" -h -o '%T' 2>/dev/null || true)
    if [[ "${state}" == "RUNNING" ]]; then
        echo "  Worker 1 is RUNNING — proceeding."
        sleep 10
        break
    elif [[ -z "${state}" ]]; then
        echo "  WARNING: Worker 1 job ${INIT_JOB} no longer in queue."
        echo "  Proceeding anyway (the DB may already exist from a prior run)."
        break
    fi
    sleep "${POLL_INTERVAL}"
done

# =====================================================================
# Phase 2: Submit remaining 19 workers (14 RTX + 5 L40S)
#
# Each worker runs sequential 5-fold CV. No --test_fold.
# RTX 2080 Ti: ~4h/fold × 5 folds × 3 trials = ~60h → 72h limit
# L40S:        ~2h/fold × 5 folds × 3 trials = ~30h → 72h limit
# =====================================================================
echo "=== Phase 2: Submitting remaining 19 workers (14 RTX + 5 L40S) ==="

declare -a JOB_IDS=()
declare -a JOB_SUFFIXES=()
declare -a RETRIES=()

# Worker 1 (the initializer) is already an RTX worker
JOB_IDS+=("${INIT_JOB}"); JOB_SUFFIXES+=("_w01"); RETRIES+=(0)

# RTX workers 2–15
for i in $(seq 2 15); do
    suffix=$(printf "_w%02d" "$i")
    jid=$(submit_worker campus-new "rtx2080ti:1" "72:00:00" "${suffix}")
    echo "  RTX worker ${i}: job ${jid}"
    JOB_IDS+=("${jid}"); JOB_SUFFIXES+=("${suffix}"); RETRIES+=(0)
done

# L40S workers 16–20
for i in $(seq 16 20); do
    suffix=$(printf "_w%02d" "$i")
    jid=$(submit_worker chorus "l40s:1" "72:00:00" "${suffix}")
    echo "  L40S worker ${i}: job ${jid}"
done

n_slots=${#JOB_IDS[@]}
echo "=== All jobs submitted (${n_slots} RTX tracked + 5 L40S untracked = 20 workers) ==="

# =====================================================================
# Phase 3: Monitor RTX workers — resubmit on failure, blacklist nodes
# =====================================================================
echo ""
echo "=== Phase 3: Monitoring ${n_slots} RTX workers (every ${MONITOR_INTERVAL}s, max ${MAX_RETRIES} retries/slot) ==="

while true; do
    sleep "${MONITOR_INTERVAL}"

    alive=0
    for i in $(seq 0 $((n_slots - 1))); do
        jid="${JOB_IDS[$i]}"
        suffix="${JOB_SUFFIXES[$i]}"
        retries="${RETRIES[$i]}"

        state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
        if [[ -n "${state}" ]]; then
            alive=$((alive + 1))
            continue
        fi

        exit_code=$(sacct -j "${jid}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')

        if [[ "${exit_code}" == "0:0" ]]; then
            echo "  [$(date '+%H:%M:%S')]${suffix} (job ${jid}) completed successfully."
            continue
        fi

        _blacklist_node "${jid}"

        if (( retries >= MAX_RETRIES )); then
            echo "  [$(date '+%H:%M:%S')]${suffix} (job ${jid}) failed — retry cap (${MAX_RETRIES}) reached, skipping."
            continue
        fi

        new_jid=$(submit_worker campus-new "rtx2080ti:1" "72:00:00" "${suffix}")
        RETRIES[$i]=$((retries + 1))
        JOB_IDS[$i]="${new_jid}"
        echo "  [$(date '+%H:%M:%S')]${suffix} (was ${jid}, exit ${exit_code}) — resubmitted as ${new_jid} (retry ${RETRIES[$i]}/${MAX_RETRIES})"
        alive=$((alive + 1))
    done

    if (( alive == 0 )); then
        echo "  [$(date '+%H:%M:%S')] All RTX workers finished or exhausted retries. Exiting monitor."
        break
    fi
done

echo "=== Monitoring complete ==="
