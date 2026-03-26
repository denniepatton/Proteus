#!/bin/bash
set -euo pipefail

# =====================================================================
# PLCA Phase 1.0 — Fold-parallel HP screening (10% data)
#
# 5 SLURM jobs, one per fold (--test_fold 1..5).
# All share the same Optuna study via SQLite storage.
# Each fold worker runs n_trials_per_fold=50 trials independently.
# =====================================================================
EXPERIMENT="PLCA_tuning/biodvn_tune-plca_phase1-0"
TAG="PLCA_1-0"
YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
TRIALS_PER_FOLD=50
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
    local fold=$1 partition=$2 gpu_spec=$3 time_limit=$4
    local job_name="${TAG}_F${fold}"
    local exclude_flag
    exclude_flag=$(_build_exclude)
    local job_script
    job_script=$(mktemp "job_f${fold}_XXXXXX.sh")

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
    --n_trials ${TRIALS_PER_FOLD} \\
    --sampler_seed ${BASE_SAMPLER_SEED}
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
# Phase 1: Submit Fold 1 to initialize the Optuna study
# =====================================================================
echo "=== Phase 1: Submitting Fold 1 (RTX 2080 Ti) to initialize study ==="
INIT_JOB=$(submit_fold_worker 1 campus-new "rtx2080ti:1" "72:00:00")
echo "  Fold 1 submitted: job ${INIT_JOB}"

echo "=== Waiting for Fold 1 (job ${INIT_JOB}) to start running ==="
while true; do
    state=$(squeue -j "${INIT_JOB}" -h -o '%T' 2>/dev/null || true)
    if [[ "${state}" == "RUNNING" ]]; then
        echo "  Fold 1 is RUNNING — proceeding."
        sleep 10
        break
    elif [[ -z "${state}" ]]; then
        echo "  WARNING: Fold 1 job ${INIT_JOB} no longer in queue."
        echo "  Proceeding anyway (the DB may already exist from a prior run)."
        break
    fi
    sleep "${POLL_INTERVAL}"
done

# =====================================================================
# Phase 2: Submit Folds 2-5
# =====================================================================
echo "=== Phase 2: Submitting Folds 2-5 ==="

declare -a JOB_IDS=()
declare -a JOB_FOLDS=()
declare -a RETRIES=()

# Fold 1 is already submitted
JOB_IDS+=("${INIT_JOB}"); JOB_FOLDS+=(1); RETRIES+=(0)

for fold in 2 3 4 5; do
    jid=$(submit_fold_worker "${fold}" campus-new "rtx2080ti:1" "72:00:00")
    echo "  Fold ${fold}: job ${jid}"
    JOB_IDS+=("${jid}"); JOB_FOLDS+=("${fold}"); RETRIES+=(0)
done

n_slots=${#JOB_IDS[@]}
echo "=== All ${n_slots} fold workers submitted ==="
echo ""

# =====================================================================
# Phase 3: Monitor and resubmit failed fold workers
# =====================================================================
echo "=== Phase 3: Monitoring ${n_slots} fold workers (every ${MONITOR_INTERVAL}s, max ${MAX_RETRIES} retries) ==="
while true; do
    sleep "${MONITOR_INTERVAL}"
    alive=0
    for i in $(seq 0 $((n_slots - 1))); do
        jid="${JOB_IDS[$i]}"
        fold="${JOB_FOLDS[$i]}"
        retries="${RETRIES[$i]}"

        state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
        if [[ -n "${state}" ]]; then
            alive=$((alive + 1))
            continue
        fi

        # Job no longer in queue — check exit status
        exit_code=$(sacct -j "${jid}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')
        if [[ "${exit_code}" == "0:0" ]]; then
            echo "  [$(date '+%H:%M:%S')] Fold ${fold} (job ${jid}) completed successfully."
            continue
        fi

        # Job failed — blacklist node and resubmit
        _blacklist_node "${jid}"
        if (( retries >= MAX_RETRIES )); then
            echo "  [$(date '+%H:%M:%S')] Fold ${fold} (job ${jid}) failed — retry cap (${MAX_RETRIES}) reached, skipping."
            continue
        fi

        echo "  [$(date '+%H:%M:%S')] Fold ${fold} (job ${jid}) failed (exit ${exit_code}) — resubmitting..."
        new_jid=$(submit_fold_worker "${fold}" campus-new "rtx2080ti:1" "72:00:00")
        JOB_IDS[$i]="${new_jid}"
        RETRIES[$i]=$((retries + 1))
        echo "    New job: ${new_jid} (retry $((retries + 1))/${MAX_RETRIES})"
        alive=$((alive + 1))
    done

    if (( alive == 0 )); then
        echo "=== All fold workers have finished. ==="
        break
    fi
    echo "  [$(date '+%H:%M:%S')] ${alive}/${n_slots} workers still active."
done

echo "=== PLCA Phase 1.0 tuning complete. ==="
