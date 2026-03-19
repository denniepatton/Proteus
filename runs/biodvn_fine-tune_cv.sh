#!/bin/bash
set -euo pipefail

# =====================================================================
# Experiment name — single source of truth
# =====================================================================
EXPERIMENT="fine-tuning/biodvn_fine-tune_phase1-1"
TAG="PANFT_1-1"

YAML="config/experiments/${EXPERIMENT}.yaml"
ENV_NAME="proteus"
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

submit_fold() {
    local fold=$1 partition=$2 gpu_spec=$3 time_limit=$4 suffix=$5 sampler_seed=$6
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

eval "\$(micromamba shell hook -s bash)"
micromamba activate ${ENV_NAME}
python Proteus.py \\
    --run_mode tune \\
    --experiment ${YAML} \\
    --test_fold ${fold} \\
    --sampler_seed ${sampler_seed}
EOL

    local job_id
    job_id=$(sbatch --parsable ${exclude_flag} "${job_script}")
    rm -f "${job_script}"
    echo "${job_id}"
}

slot_sampler_seed() {
    local fold=$1 slot=$2
    echo $((BASE_SAMPLER_SEED + fold * 100 + slot))
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

echo "=== Phase 1: Submitting Fold 1 (RTX 2080 Ti) to initialize study ==="
FOLD1_SEED=$(slot_sampler_seed 1 0)
FOLD1_JOB=$(submit_fold 1 campus-new "rtx2080ti:1" "72:00:00" "" "${FOLD1_SEED}")
echo "  Fold 1 submitted: job ${FOLD1_JOB}"

echo "=== Waiting for Fold 1 (job ${FOLD1_JOB}) to start running ==="
while true; do
    state=$(squeue -j "${FOLD1_JOB}" -h -o '%T' 2>/dev/null || true)
    if [[ "${state}" == "RUNNING" ]]; then
        echo "  Fold 1 is RUNNING — proceeding."
        sleep 10
        break
    elif [[ -z "${state}" ]]; then
        echo "  WARNING: Fold 1 job ${FOLD1_JOB} no longer in queue."
        echo "  Proceeding anyway (the DB may already exist from a prior run)."
        break
    fi
    sleep "${POLL_INTERVAL}"
done

echo "=== Phase 2: Submitting remaining workers (14 RTX + 5 L40S) ==="
declare -a JOB_IDS=()
declare -a JOB_FOLDS=()
declare -a JOB_SUFFIXES=()
declare -a JOB_SEEDS=()
declare -a RETRIES=()
JOB_IDS+=("${FOLD1_JOB}"); JOB_FOLDS+=(1); JOB_SUFFIXES+=(""); JOB_SEEDS+=("${FOLD1_SEED}"); RETRIES+=(0)

# Fold 1: 2 additional RTX workers
slot=1
for worker in b c; do
    seed=$(slot_sampler_seed 1 "${slot}")
    jid=$(submit_fold 1 campus-new "rtx2080ti:1" "72:00:00" "_${worker}" "${seed}")
    echo "  Fold 1 RTX worker ${worker}: job ${jid}"
    JOB_IDS+=("${jid}"); JOB_FOLDS+=(1); JOB_SUFFIXES+=("_${worker}"); JOB_SEEDS+=("${seed}"); RETRIES+=(0)
    slot=$((slot + 1))
done

# Folds 2-5: 3 RTX workers each
for fold in 2 3 4 5; do
    slot=0
    for worker in a b c; do
        seed=$(slot_sampler_seed "${fold}" "${slot}")
        jid=$(submit_fold "${fold}" campus-new "rtx2080ti:1" "72:00:00" "_${worker}" "${seed}")
        echo "  Fold ${fold} RTX worker ${worker}: job ${jid}"
        JOB_IDS+=("${jid}"); JOB_FOLDS+=("${fold}"); JOB_SUFFIXES+=("_${worker}"); JOB_SEEDS+=("${seed}"); RETRIES+=(0)
        slot=$((slot + 1))
    done
done

# L40S workers (1 per fold) — not monitored for resubmission
for fold in 1 2 3 4 5; do
    seed=$(slot_sampler_seed "${fold}" 9)
    jid=$(submit_fold "${fold}" chorus "l40s:1" "72:00:00" "_L" "${seed}")
    echo "  Fold ${fold} L40S: job ${jid}"
done

n_slots=${#JOB_IDS[@]}
echo "=== All jobs submitted (${n_slots} RTX + 5 L40S = $((n_slots + 5)) workers) ==="
echo ""
echo "=== Phase 3: Monitoring ${n_slots} RTX workers (every ${MONITOR_INTERVAL}s, max ${MAX_RETRIES} retries/slot) ==="
while true; do
    sleep "${MONITOR_INTERVAL}"
    alive=0
    for i in $(seq 0 $((n_slots - 1))); do
        jid="${JOB_IDS[$i]}"
        fold="${JOB_FOLDS[$i]}"
        suffix="${JOB_SUFFIXES[$i]}"
        seed="${JOB_SEEDS[$i]}"
        retries="${RETRIES[$i]}"
        state=$(squeue -j "${jid}" -h -o '%T' 2>/dev/null || true)
        if [[ -n "${state}" ]]; then
            alive=$((alive + 1))
            continue
        fi
        exit_code=$(sacct -j "${jid}" -n -o ExitCode -X 2>/dev/null | head -1 | tr -d '[:space:]')
        if [[ "${exit_code}" == "0:0" ]]; then
            echo "  [$(date '+%H:%M:%S')] Fold ${fold}${suffix} (job ${jid}) completed successfully."
            continue
        fi
        _blacklist_node "${jid}"
        if (( retries >= MAX_RETRIES )); then
            echo "  [$(date '+%H:%M:%S')] Fold ${fold}${suffix} (job ${jid}) failed — retry cap (${MAX_RETRIES}) reached, skipping."
            continue
        fi
        new_jid=$(submit_fold "${fold}" campus-new "rtx2080ti:1" "72:00:00" "${suffix}" "${seed}")
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
