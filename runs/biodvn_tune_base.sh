#!/bin/bash
set -euo pipefail

# =====================================================================
# Experiment name — single source of truth
# =====================================================================
EXPERIMENT="biodvn_tune_phase2-3"
TAG="BTP_2-3"  # short tag for job names / logs

YAML="${EXPERIMENT}.yaml"
ENV_NAME="proteus"               # micromamba environment (pytorch-cuda=12.1)
POLL_INTERVAL=15                 # seconds between squeue checks while waiting for init job
MONITOR_INTERVAL=120             # seconds between monitoring sweeps
MAX_RETRIES=5                    # max resubmissions per RTX worker slot

# Node blacklist: nodes that have caused CUDA failures are recorded here.
# One hostname per line.  The file is both read (to build --exclude) and
# appended (when a failed job's node is identified).
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
# Helper: generate and submit a fold job
# Prints the SLURM job ID to stdout.
# =====================================================================
submit_fold() {
    local fold=$1 partition=$2 gpu_spec=$3 time_limit=$4 suffix=$5
    local job_name="F${fold}_${TAG}${suffix}"
    local exclude_flag
    exclude_flag=$(_build_exclude)
    local job_script
    job_script=$(mktemp "job_fold_${fold}_XXXXXX.sh")

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
    --test_fold ${fold}
EOL

    local job_id
    job_id=$(sbatch --parsable ${exclude_flag} "${job_script}")
    rm -f "${job_script}"
    echo "${job_id}"
}

# =====================================================================
# Record the node a failed job ran on into the blacklist file.
#   _blacklist_node <job_id>
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
# Phase 1: Submit Fold 1 on an RTX 2080 Ti (fast to schedule) to
#           create / initialize the Optuna DB before other workers.
# =====================================================================
echo "=== Phase 1: Submitting Fold 1 (RTX 2080 Ti) to initialize study ==="
FOLD1_JOB=$(submit_fold 1 campus-new "rtx2080ti:1" "72:00:00" "")
echo "  Fold 1 submitted: job ${FOLD1_JOB}"

# =====================================================================
# Wait until Fold 1 is RUNNING so the DB exists before the fleet.
# =====================================================================
echo "=== Waiting for Fold 1 (job ${FOLD1_JOB}) to start running ==="
while true; do
    state=$(squeue -j "${FOLD1_JOB}" -h -o '%T' 2>/dev/null || true)
    if [[ "${state}" == "RUNNING" ]]; then
        echo "  Fold 1 is RUNNING — proceeding."
        sleep 10  # give Python a moment to create the study
        break
    elif [[ -z "${state}" ]]; then
        echo "  WARNING: Fold 1 job ${FOLD1_JOB} no longer in queue."
        echo "  Proceeding anyway (the DB may already exist from a prior run)."
        break
    fi
    sleep "${POLL_INTERVAL}"
done

# =====================================================================
# Phase 2: Submit the remaining 19 workers (14 RTX + 5 L40S)
#
#   RTX 2080 Ti (campus-new): Fold 1 × 2 more + Folds 2-5 × 3 = 14
#   L40S (chorus):            Folds 1-5 × 1 = 5
# =====================================================================
echo "=== Phase 2: Submitting remaining workers (14 RTX + 5 L40S) ==="

# Track all RTX jobs for the monitor (parallel arrays).
declare -a JOB_IDS=()
declare -a JOB_FOLDS=()
declare -a JOB_SUFFIXES=()
declare -a RETRIES=()

# The Phase 1 initializer is also an RTX worker.
JOB_IDS+=("${FOLD1_JOB}"); JOB_FOLDS+=(1); JOB_SUFFIXES+=(""); RETRIES+=(0)

# Fold 1: 2 additional RTX workers
for worker in b c; do
    jid=$(submit_fold 1 campus-new "rtx2080ti:1" "72:00:00" "_${worker}")
    echo "  Fold 1 RTX worker ${worker}: job ${jid}"
    JOB_IDS+=("${jid}"); JOB_FOLDS+=(1); JOB_SUFFIXES+=("_${worker}"); RETRIES+=(0)
done

# Folds 2-5: 3 RTX workers each
for fold in 2 3 4 5; do
    for worker in a b c; do
        jid=$(submit_fold "${fold}" campus-new "rtx2080ti:1" "72:00:00" "_${worker}")
        echo "  Fold ${fold} RTX worker ${worker}: job ${jid}"
        JOB_IDS+=("${jid}"); JOB_FOLDS+=("${fold}"); JOB_SUFFIXES+=("_${worker}"); RETRIES+=(0)
    done
done

# L40S workers (1 per fold) — not monitored for resubmission
for fold in 1 2 3 4 5; do
    jid=$(submit_fold "${fold}" chorus "l40s:1" "72:00:00" "_L")
    echo "  Fold ${fold} L40S: job ${jid}"
done

n_slots=${#JOB_IDS[@]}
echo "=== All jobs submitted (${n_slots} RTX + 5 L40S = $((n_slots + 5)) workers) ==="

# =====================================================================
# Phase 3: Monitor RTX workers — resubmit on failure, blacklist nodes
#
# L40S jobs are excluded: they queue for a long time and are resilient
# once running.  RTX 2080 Ti workers are the ones that hit transient
# CUDA node failures.
# =====================================================================
echo ""
echo "=== Phase 3: Monitoring ${n_slots} RTX workers (every ${MONITOR_INTERVAL}s, max ${MAX_RETRIES} retries/slot) ==="

while true; do
    sleep "${MONITOR_INTERVAL}"

    alive=0
    for i in $(seq 0 $((n_slots - 1))); do
        jid="${JOB_IDS[$i]}"
        fold="${JOB_FOLDS[$i]}"
        suffix="${JOB_SUFFIXES[$i]}"
        retries="${RETRIES[$i]}"

        # Still in the queue? (PENDING or RUNNING)
        state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
        if [[ -n "${state}" ]]; then
            alive=$((alive + 1))
            continue
        fi

        # Job left the queue — check exit code.
        exit_code=$(sacct -j "${jid}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')

        if [[ "${exit_code}" == "0:0" ]]; then
            echo "  [$(date '+%H:%M:%S')] Fold ${fold}${suffix} (job ${jid}) completed successfully."
            continue
        fi

        # Failed — blacklist the node and resubmit if under retry cap.
        _blacklist_node "${jid}"

        if (( retries >= MAX_RETRIES )); then
            echo "  [$(date '+%H:%M:%S')] Fold ${fold}${suffix} (job ${jid}) failed — retry cap (${MAX_RETRIES}) reached, skipping."
            continue
        fi

        new_jid=$(submit_fold "${fold}" campus-new "rtx2080ti:1" "72:00:00" "${suffix}")
        RETRIES[$i]=$((retries + 1))
        JOB_IDS[$i]="${new_jid}"
        echo "  [$(date '+%H:%M:%S')] Fold ${fold}${suffix} (was ${jid}, exit ${exit_code}) — resubmitted as ${new_jid} (retry ${RETRIES[$i]}/${MAX_RETRIES})"
        alive=$((alive + 1))
    done

    if (( alive == 0 )); then
        echo "  [$(date '+%H:%M:%S')] All RTX workers finished or exhausted retries. Exiting monitor."
        break
    fi
done

echo "=== Monitoring complete ==="
