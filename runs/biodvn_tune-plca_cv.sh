#!/bin/bash
set -euo pipefail

# =====================================================================
# PLCA Phase 1.3 — Min-source GVW-r objective, narrowed LR, tunable epochs
#
# 4 workers per fold × 5 folds = 20 SLURM jobs.
# All share the same Optuna study via SQLite storage.
# Each worker runs 50 trials; 4 workers × 50 = 200 trials per fold.
# Workers for the same fold get different sampler seeds so each
# explores a distinct region of HP space.
# =====================================================================
EXPERIMENT="PLCA_tuning/biodvn_tune-plca_phase1-3"
TAG="PLCA_1-3"
YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
N_WORKERS_PER_FOLD=4
TRIALS_PER_WORKER=50
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
# Helper: submit a fold worker
# =====================================================================
submit_fold_worker() {
    local fold=$1 worker_id=$2 partition=$3 gpu_spec=$4 time_limit=$5 sampler_seed=$6
    local job_name="${TAG}_F${fold}w${worker_id}"
    local exclude_flag
    exclude_flag=$(_build_exclude)
    local job_script
    job_script=$(mktemp "job_f${fold}w${worker_id}_XXXXXX.sh")

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

eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}
python Proteus.py \\
    --run_mode tune \\
    --experiment ${YAML} \\
    --test_fold ${fold} \\
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

TOTAL_WORKERS=$((N_WORKERS_PER_FOLD * 5))
echo "=== PLCA Phase 1.2: ${N_WORKERS_PER_FOLD} workers/fold × 5 folds = ${TOTAL_WORKERS} workers, ${TRIALS_PER_WORKER} trials/worker ==="

# =====================================================================
# Phase 1: Submit first worker for Fold 1 to initialize the Optuna study
# =====================================================================
echo "=== Phase 1: Submitting Fold 1 worker 0 (RTX 2080 Ti) to initialize study ==="
INIT_SEED="${BASE_SAMPLER_SEED}"
INIT_JOB=$(submit_fold_worker 1 0 campus-new "rtx2080ti:1" "72:00:00" "${INIT_SEED}")
echo "  Fold 1 worker 0 submitted: job ${INIT_JOB}"

echo "=== Waiting for Fold 1 worker 0 (job ${INIT_JOB}) to start running ==="
while true; do
    state=$(squeue -j "${INIT_JOB}" -h -o '%T' 2>/dev/null || true)
    if [[ "${state}" == "RUNNING" ]]; then
        echo "  Fold 1 worker 0 is RUNNING — proceeding."
        sleep 10
        break
    elif [[ -z "${state}" ]]; then
        echo "  WARNING: Fold 1 worker 0 job ${INIT_JOB} no longer in queue."
        echo "  Proceeding anyway (the DB may already exist from a prior run)."
        break
    fi
    sleep "${POLL_INTERVAL}"
done

# =====================================================================
# Phase 2: Submit remaining workers (3 more for fold 1, 4 each for 2-5)
# =====================================================================
echo "=== Phase 2: Submitting remaining $((TOTAL_WORKERS - 1)) workers ==="

declare -a JOB_IDS=()
declare -a JOB_LABELS=()
declare -a JOB_FOLDS=()
declare -a JOB_WIDS=()
declare -a JOB_SEEDS=()
declare -a RETRIES=()

# Track the initializer
JOB_IDS+=("${INIT_JOB}"); JOB_LABELS+=("F1w0"); JOB_FOLDS+=(1); JOB_WIDS+=(0); JOB_SEEDS+=("${INIT_SEED}"); RETRIES+=(0)

global_idx=1  # worker 0 already submitted
for fold in 1 2 3 4 5; do
    start_w=0
    if [[ "${fold}" -eq 1 ]]; then
        start_w=1  # worker 0 for fold 1 already submitted
    fi
    for w in $(seq "${start_w}" $((N_WORKERS_PER_FOLD - 1))); do
        seed=$((BASE_SAMPLER_SEED + global_idx))
        jid=$(submit_fold_worker "${fold}" "${w}" campus-new "rtx2080ti:1" "72:00:00" "${seed}")
        label="F${fold}w${w}"
        echo "  ${label}: job ${jid} (seed ${seed})"
        JOB_IDS+=("${jid}"); JOB_LABELS+=("${label}"); JOB_FOLDS+=("${fold}"); JOB_WIDS+=("${w}"); JOB_SEEDS+=("${seed}"); RETRIES+=(0)
        global_idx=$((global_idx + 1))
    done
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
        fold="${JOB_FOLDS[$i]}"
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
        new_jid=$(submit_fold_worker "${fold}" "${wid}" campus-new "rtx2080ti:1" "72:00:00" "${seed}")
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

echo "=== PLCA Phase 1.0 tuning complete. ==="
