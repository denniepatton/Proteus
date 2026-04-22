#!/bin/bash
set -euo pipefail

# =====================================================================
# Experiment name — single source of truth
# =====================================================================
EXPERIMENT="fine-tuning/biodvn_fine-tune_phase1-1"
TAG="PANFT_1-1"

YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
TRIALS_PER_WORKER=8           # each trial trains all 5 folds (~7-10h on RTX)
N_RTX_WORKERS=15              # campus-new partition
N_L40S_WORKERS=5              # chorus partition
BASE_SAMPLER_SEED=42
POLL_INTERVAL=15
MONITOR_INTERVAL=120
MAX_RETRIES=5
BLACKLIST_FILE="runs/bad_nodes.txt"
touch "${BLACKLIST_FILE}"

_build_exclude() {
    local nodes
    nodes=$(sort -u "${BLACKLIST_FILE}" | paste -sd ',' -)
    if [[ -n "${nodes}" ]]; then
        echo "--exclude=${nodes}"
    fi
}

submit_worker() {
    local worker_id=$1 partition=$2 gpu_spec=$3 time_limit=$4 sampler_seed=$5
    local job_name="W${worker_id}_${TAG}"
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

TOTAL_WORKERS=$((N_RTX_WORKERS + N_L40S_WORKERS))
echo "=== Sequential 5-fold CV tuning: ${TRIALS_PER_WORKER} trials/worker × ${TOTAL_WORKERS} workers = $((TRIALS_PER_WORKER * TOTAL_WORKERS)) total trials ==="

# ── Phase 1: Submit one worker to create the Optuna study ──
echo "=== Phase 1: Submitting Worker 0 (RTX 2080 Ti) to initialize study ==="
INIT_SEED="${BASE_SAMPLER_SEED}"
INIT_JOB=$(submit_worker 0 campus-new "rtx2080ti:1" "72:00:00" "${INIT_SEED}")
echo "  Worker 0 submitted: job ${INIT_JOB}"

echo "=== Waiting for Worker 0 (job ${INIT_JOB}) to start running ==="
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

# ── Phase 2: Submit remaining workers ──
echo "=== Phase 2: Submitting remaining workers ($((N_RTX_WORKERS - 1)) RTX + ${N_L40S_WORKERS} L40S) ==="
declare -a JOB_IDS=()
declare -a JOB_WIDS=()
declare -a JOB_SEEDS=()
declare -a RETRIES=()
JOB_IDS+=("${INIT_JOB}"); JOB_WIDS+=(0); JOB_SEEDS+=("${INIT_SEED}"); RETRIES+=(0)

# Additional RTX workers
for wid in $(seq 1 $((N_RTX_WORKERS - 1))); do
    seed=$((BASE_SAMPLER_SEED + wid))
    jid=$(submit_worker "${wid}" campus-new "rtx2080ti:1" "72:00:00" "${seed}")
    echo "  RTX Worker ${wid}: job ${jid}"
    JOB_IDS+=("${jid}"); JOB_WIDS+=("${wid}"); JOB_SEEDS+=("${seed}"); RETRIES+=(0)
done

# L40S workers — not monitored for resubmission
for i in $(seq 0 $((N_L40S_WORKERS - 1))); do
    wid=$((N_RTX_WORKERS + i))
    seed=$((BASE_SAMPLER_SEED + wid))
    jid=$(submit_worker "${wid}" chorus "l40s:1" "72:00:00" "${seed}")
    echo "  L40S Worker ${wid}: job ${jid}"
done

n_slots=${#JOB_IDS[@]}
echo "=== All jobs submitted (${n_slots} RTX monitored + ${N_L40S_WORKERS} L40S unmonitored) ==="
echo ""

# ── Phase 3: Monitor RTX workers ──
echo "=== Phase 3: Monitoring ${n_slots} RTX workers (every ${MONITOR_INTERVAL}s, max ${MAX_RETRIES} retries/slot) ==="
while true; do
    sleep "${MONITOR_INTERVAL}"
    alive=0
    for i in $(seq 0 $((n_slots - 1))); do
        jid="${JOB_IDS[$i]}"
        wid="${JOB_WIDS[$i]}"
        seed="${JOB_SEEDS[$i]}"
        retries="${RETRIES[$i]}"
        state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
        if [[ -n "${state}" ]]; then
            alive=$((alive + 1))
            continue
        fi
        exit_code=$(sacct -j "${jid}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')
        if [[ "${exit_code}" == "0:0" ]]; then
            echo "  [$(date '+%H:%M:%S')] Worker ${wid} (job ${jid}) completed successfully."
            continue
        fi
        _blacklist_node "${jid}"
        if (( retries >= MAX_RETRIES )); then
            echo "  [$(date '+%H:%M:%S')] Worker ${wid} (job ${jid}) failed — retry cap (${MAX_RETRIES}) reached, skipping."
            continue
        fi
        new_jid=$(submit_worker "${wid}" campus-new "rtx2080ti:1" "72:00:00" "${seed}")
        RETRIES[$i]=$((retries + 1))
        JOB_IDS[$i]="${new_jid}"
        echo "  [$(date '+%H:%M:%S')] Worker ${wid} (was ${jid}, exit ${exit_code}) — resubmitted as ${new_jid} (retry ${RETRIES[$i]}/${MAX_RETRIES})"
        alive=$((alive + 1))
    done
    if (( alive == 0 )); then
        echo "  [$(date '+%H:%M:%S')] All RTX workers finished or exhausted retries. Exiting monitor."
        break
    fi
done
echo "=== Monitoring complete ==="
