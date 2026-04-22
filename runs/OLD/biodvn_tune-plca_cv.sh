#!/bin/bash
set -euo pipefail

# =====================================================================
# PLCA Phase 1.7 — Sequential CV tuning (cross-fold objective)
#
# Each worker runs ALL 5 folds per trial → proper cross-fold optimization.
# 20 SLURM workers, each with 1 GPU, sharing an Optuna study via SQLite.
# Each trial trains 15 epochs × 5 folds ≈ 42h → expect ~1.5 trials/worker.
# Top-10 configs from phase 1.6 are seeded; remaining trials are
# TPE-sampled from narrowed HP ranges.
# =====================================================================
EXPERIMENT="PLCA_tuning/biodvn_tune-plca_phase1-7"
TAG="PLCA_1-7"
YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
N_WORKERS=20
TRIALS_PER_WORKER=3
BASE_SAMPLER_SEED=42
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
# Helper: submit a CV worker (no --test_fold → all folds per trial)
# =====================================================================
submit_cv_worker() {
    local worker_id=$1 partition=$2 gpu_spec=$3 time_limit=$4 sampler_seed=$5
    local job_name="${TAG}_w${worker_id}"
    local exclude_flag
    exclude_flag=$(_build_exclude)
    local job_script
    job_script=$(mktemp "job_w${worker_id}_XXXXXX.sh")

    cat > "${job_script}" <<EOL
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus=${gpu_spec}
#SBATCH --time=${time_limit}
#SBATCH -A nelson_p

eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}
python Proteus.py \\
    --run_mode tune \\
    --experiment ${YAML} \\
    --n_trials ${TRIALS_PER_WORKER} \\
    --sampler_seed ${sampler_seed}
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

echo "=== PLCA Phase 1.7: ${N_WORKERS} CV workers, ${TRIALS_PER_WORKER} trials/worker, 5 folds/trial ==="

# =====================================================================
# Phase 1: Submit worker 0 to initialize the Optuna study + seed prev
# =====================================================================
echo "=== Phase 1: Submitting worker 0 to initialize study ==="
INIT_SEED="${BASE_SAMPLER_SEED}"
INIT_JOB=$(submit_cv_worker 0 campus-new "rtx2080ti:1" "72:00:00" "${INIT_SEED}")
echo "  Worker 0 submitted: job ${INIT_JOB}"

echo "=== Waiting for worker 0 (job ${INIT_JOB}) to start running ==="
while true; do
    state=$(squeue -j "${INIT_JOB}" -h -o '%T' 2>/dev/null || true)
    if [[ "${state}" == "RUNNING" ]]; then
        echo "  Worker 0 is RUNNING — proceeding."
        sleep 10
        break
    elif [[ -z "${state}" ]]; then
        echo "  WARNING: Worker 0 job ${INIT_JOB} no longer in queue."
        echo "  Proceeding anyway (the DB may already exist from a prior run)."
        break
    fi
    sleep "${POLL_INTERVAL}"
done

# =====================================================================
# Phase 2: Submit remaining 19 workers
# =====================================================================
echo "=== Phase 2: Submitting remaining $((N_WORKERS - 1)) workers ==="

declare -a JOB_IDS=()
declare -a JOB_LABELS=()
declare -a JOB_WIDS=()
declare -a JOB_SEEDS=()
declare -a RETRIES=()

# Track the initializer
JOB_IDS+=("${INIT_JOB}"); JOB_LABELS+=("w0"); JOB_WIDS+=(0); JOB_SEEDS+=("${INIT_SEED}"); RETRIES+=(0)

for w in $(seq 1 $((N_WORKERS - 1))); do
    seed=$((BASE_SAMPLER_SEED + w))
    jid=$(submit_cv_worker "${w}" campus-new "rtx2080ti:1" "72:00:00" "${seed}")
    label="w${w}"
    echo "  ${label}: job ${jid} (seed ${seed})"
    JOB_IDS+=("${jid}"); JOB_LABELS+=("${label}"); JOB_WIDS+=("${w}"); JOB_SEEDS+=("${seed}"); RETRIES+=(0)
done

n_slots=${#JOB_IDS[@]}
echo "=== All ${n_slots} workers submitted ==="
echo ""

# =====================================================================
# Phase 3: Monitor and resubmit failed workers
# =====================================================================
echo "=== Phase 3: Monitoring ${n_slots} workers (every ${MONITOR_INTERVAL}s, max ${MAX_RETRIES} retries) ==="
while true; do
    sleep "${MONITOR_INTERVAL}"
    alive=0
    for i in $(seq 0 $((n_slots - 1))); do
        jid="${JOB_IDS[$i]}"
        label="${JOB_LABELS[$i]}"
        wid="${JOB_WIDS[$i]}"
        seed="${JOB_SEEDS[$i]}"
        retries="${RETRIES[$i]}"

        state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
        if [[ -n "${state}" ]]; then
            alive=$((alive + 1))
            continue
        fi

        # Job no longer in queue — check exit status
        exit_code=$(sacct -j "${jid}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')
        if [[ "${exit_code}" == "0:0" ]]; then
            echo "  [$(date '+%H:%M:%S')] ${label} (job ${jid}) completed successfully."
            continue
        fi

        # Job failed — blacklist node and resubmit
        _blacklist_node "${jid}"
        if (( retries >= MAX_RETRIES )); then
            echo "  [$(date '+%H:%M:%S')] ${label} (job ${jid}) failed — retry cap (${MAX_RETRIES}) reached, skipping."
            continue
        fi

        echo "  [$(date '+%H:%M:%S')] ${label} (job ${jid}) failed (exit ${exit_code}) — resubmitting..."
        new_jid=$(submit_cv_worker "${wid}" campus-new "rtx2080ti:1" "72:00:00" "${seed}")
        JOB_IDS[$i]="${new_jid}"
        RETRIES[$i]=$((retries + 1))
        echo "    New job: ${new_jid} (retry $((retries + 1))/${MAX_RETRIES})"
        alive=$((alive + 1))
    done

    if (( alive == 0 )); then
        echo "=== All workers have finished. ==="
        break
    fi
done

echo "=== PLCA Phase 1.7 tuning complete. ==="
