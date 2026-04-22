#!/bin/bash
set -euo pipefail

ROLE="${1:-}"

if [[ -z "${ROLE}" ]]; then
    echo "Usage: $0 <coordinator|rtx|l40|status>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="PLCA_tuning/biodvn_tune-plcav2_weighted"
YAML="config/experiments/${EXPERIMENT}.yaml"
OUTPUT_DIR="results_training/biodvn_plcav2_weighted_tuning"
STATE_DIR="${OUTPUT_DIR}/coordination"
STATE_JSON="${STATE_DIR}/queue_state.json"

ENV_NAME="${ENV_NAME:-proteus}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///results_training/optuna.db}"
STUDY_NAME="${STUDY_NAME:-}"
BASE_SAMPLER_SEED="${BASE_SAMPLER_SEED:-42}"

TARGET_ACTIVE_TRIALS="${TARGET_ACTIVE_TRIALS:-15}"
ASK_BATCH="${ASK_BATCH:-3}"
COORDINATOR_POLL_SECONDS="${COORDINATOR_POLL_SECONDS:-45}"
WORKER_IDLE_SECONDS="${WORKER_IDLE_SECONDS:-60}"
STALE_SECONDS="${STALE_SECONDS:-7200}"
HEARTBEAT_SECONDS=60

mkdir -p "${STATE_DIR}" logs

activate_env() {
    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook -s bash)"
        micromamba activate "${ENV_NAME}"
    fi
}

study_args=()
if [[ -n "${STUDY_NAME}" ]]; then
    study_args+=(--study_name "${STUDY_NAME}")
fi

state_cmd() {
    python - "${STATE_JSON}" "${OUTPUT_DIR}" "$@" <<'PY'
import json
import math
import os
import sys
import time
from pathlib import Path
import fcntl

state_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
cmd = sys.argv[3]
args = sys.argv[4:]

state_path.parent.mkdir(parents=True, exist_ok=True)


def metrics_path(trial_num: str, fold_num: str) -> Path:
    t = int(trial_num)
    f = int(fold_num)
    return output_dir / f"trial_{t:05d}" / f"CV-fold_{f}" / f"validation-fold_{f}_metrics.json"


def checkpoint_path(trial_num: str, fold_num: str) -> Path:
    t = int(trial_num)
    f = int(fold_num)
    return output_dir / f"trial_{t:05d}" / f"CV-fold_{f}" / "best_model.pt"


def read_partial_score(path: Path) -> float:
    try:
        with open(path, 'r') as handle:
            metrics = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return float('inf')
    value = metrics.get('weighted_tumor_gvw_crps')
    if value is None:
        value = metrics.get('balanced_tumor_gvw_crps')
    if value is None:
        return float('inf')
    value = float(value)
    return value if math.isfinite(value) else float('inf')


with open(state_path, 'a+') as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.seek(0)
    raw = fh.read().strip()
    state = json.loads(raw) if raw else {'next_order': 0, 'trials': {}}

    def save() -> None:
        fh.seek(0)
        fh.truncate()
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write('\n')
        fh.flush()

    now = int(time.time())

    if cmd == 'register_trials':
        trial_csv = args[0] if args else ''
        for token in [x.strip() for x in trial_csv.split(',') if x.strip()]:
            if token in state['trials']:
                continue
            state['trials'][token] = {
                'order': int(state['next_order']),
                'told': False,
                'serial_owner': None,
                'folds': {
                    str(fold): {
                        'status': 'pending',
                        'worker': None,
                        'claimed_at': 0,
                    }
                    for fold in range(1, 6)
                },
            }
            state['next_order'] += 1
        save()
    elif cmd == 'reconcile':
        stale_seconds = int(args[0]) if args else 0
        for trial_num, trial_data in state['trials'].items():
            serial_owner = trial_data.get('serial_owner')
            for fold_num, fold_data in trial_data['folds'].items():
                fold_metrics = metrics_path(trial_num, fold_num)
                if fold_metrics.exists():
                    fold_data['status'] = 'done'
                    continue
                status = fold_data.get('status', 'pending')
                claimed_at = int(fold_data.get('claimed_at', 0) or 0)
                if status == 'pending' and checkpoint_path(trial_num, fold_num).exists():
                    # Recover queue state after a coordinator restart or an
                    # over-eager stale reset: if the fold has already produced
                    # a checkpoint but not final metrics, it is in-flight.
                    fold_data['status'] = 'running'
                    fold_data['worker'] = None
                    fold_data['claimed_at'] = now
                    continue
                if status == 'running' and claimed_at > 0 and stale_seconds > 0:
                    if now - claimed_at > stale_seconds:
                        fold_data['status'] = 'pending'
                        fold_data['worker'] = None
                        fold_data['claimed_at'] = 0
            if all(f['status'] == 'done' for f in trial_data['folds'].values()):
                trial_data['serial_owner'] = None
            elif serial_owner is not None:
                owner_active = any(
                    f['status'] == 'running' and f.get('worker') == serial_owner
                    for f in trial_data['folds'].values()
                )
                if not owner_active:
                    trial_data['serial_owner'] = None
        save()
    elif cmd == 'ready_to_tell':
        ready = [
            int(trial_num)
            for trial_num, trial_data in state['trials'].items()
            if not trial_data.get('told', False)
            and all(f['status'] == 'done' for f in trial_data['folds'].values())
        ]
        ready.sort()
        print(','.join(str(x) for x in ready))
    elif cmd == 'mark_told':
        trial_csv = args[0] if args else ''
        for token in [x.strip() for x in trial_csv.split(',') if x.strip()]:
            if token in state['trials']:
                state['trials'][token]['told'] = True
                state['trials'][token]['serial_owner'] = None
        save()
    elif cmd == 'count_active':
        active = sum(1 for trial in state['trials'].values() if not trial.get('told', False))
        print(active)
    elif cmd == 'heartbeat':
        trial_num, fold_num, worker_id = args
        trial_data = state['trials'].get(str(trial_num))
        if trial_data is not None:
            fold_data = trial_data['folds'][str(fold_num)]
            if fold_data.get('worker') in (None, worker_id):
                fold_data['status'] = 'running'
                fold_data['worker'] = worker_id
                fold_data['claimed_at'] = now
                save()
    elif cmd == 'claim_rtx':
        worker_id = args[0]
        candidates = []
        for trial_num, trial_data in state['trials'].items():
            if trial_data.get('told', False):
                continue
            serial_owner = trial_data.get('serial_owner')
            if serial_owner is not None and serial_owner != worker_id:
                continue
            folds = trial_data['folds']
            done_count = sum(1 for f in folds.values() if f['status'] == 'done')
            order = int(trial_data.get('order', 0))
            for fold_num, fold_data in folds.items():
                if fold_data.get('status') == 'pending':
                    candidates.append((done_count, order, int(fold_num), int(trial_num), fold_num))
        if not candidates:
            print('')
        else:
            _, _, _, trial_int, fold_num = min(candidates)
            trial_data = state['trials'][str(trial_int)]
            trial_data['folds'][fold_num]['status'] = 'running'
            trial_data['folds'][fold_num]['worker'] = worker_id
            trial_data['folds'][fold_num]['claimed_at'] = now
            save()
            print(f'{trial_int}:{fold_num}')
    elif cmd == 'claim_l40':
        worker_id = args[0]
        for trial_num, trial_data in state['trials'].items():
            if trial_data.get('told', False):
                continue
            running_folds = [
                fold_num for fold_num, fold_data in trial_data['folds'].items()
                if fold_data.get('status') == 'running' and fold_data.get('worker') == worker_id
            ]
            if running_folds:
                running_folds.sort(key=int)
                print(f"{trial_num}:{','.join(running_folds)}")
                break
        else:
            candidates = []
            seeded = []
            fallback = []
            for trial_num, trial_data in state['trials'].items():
                if trial_data.get('told', False):
                    continue
                serial_owner = trial_data.get('serial_owner')
                if serial_owner is not None and serial_owner != worker_id:
                    continue
                folds = trial_data['folds']
                pending = [fold_num for fold_num, fold_data in folds.items() if fold_data.get('status') == 'pending']
                if not pending:
                    continue
                done = [fold_num for fold_num, fold_data in folds.items() if fold_data.get('status') == 'done']
                running = [
                    fold_num for fold_num, fold_data in folds.items()
                    if fold_data.get('status') == 'running'
                ]
                order = int(trial_data.get('order', 0))
                if done:
                    scores = [read_partial_score(metrics_path(trial_num, fold_num)) for fold_num in done]
                    finite_scores = [score for score in scores if math.isfinite(score)]
                    partial_mean = sum(finite_scores) / len(finite_scores) if finite_scores else float('inf')
                    candidates.append((
                        partial_mean,
                        -len(done),
                        -len(running),
                        order,
                        int(trial_num),
                        pending,
                    ))
                elif running:
                    # Before any fold has finished, use the fast L40s to help
                    # complete already-seeded trials rather than opening more
                    # one-fold scouts.
                    seeded.append((-len(running), order, int(trial_num), pending))
                else:
                    fallback.append((order, int(trial_num), pending))

            if candidates:
                _, _, _, _, trial_int, pending = min(candidates)
            elif seeded:
                _, _, trial_int, pending = min(seeded)
            elif fallback:
                _, trial_int, pending = min(fallback)
            else:
                print('')
                fcntl.flock(fh, fcntl.LOCK_UN)
                sys.exit(0)

            trial_data = state['trials'][str(trial_int)]
            trial_data['serial_owner'] = worker_id
            claimed_folds = []
            for fold_num in pending:
                fold_data = trial_data['folds'][str(fold_num)]
                fold_data['status'] = 'running'
                fold_data['worker'] = worker_id
                fold_data['claimed_at'] = now
                claimed_folds.append(str(fold_num))
            save()
            print(f"{trial_int}:{','.join(claimed_folds)}")
    elif cmd == 'mark_done':
        trial_num, fold_num, worker_id = args
        trial_data = state['trials'].get(str(trial_num))
        if trial_data is not None:
            fold_data = trial_data['folds'][str(fold_num)]
            if fold_data.get('worker') in (None, worker_id):
                fold_data['status'] = 'done'
                fold_data['worker'] = worker_id
                fold_data['claimed_at'] = now
            if all(f['status'] == 'done' for f in trial_data['folds'].values()):
                trial_data['serial_owner'] = None
        save()
    elif cmd == 'mark_failed':
        trial_num, fold_num, worker_id = args
        trial_data = state['trials'].get(str(trial_num))
        if trial_data is not None:
            fold_data = trial_data['folds'][str(fold_num)]
            if fold_data.get('worker') in (None, worker_id):
                fold_data['status'] = 'pending'
                fold_data['worker'] = None
                fold_data['claimed_at'] = 0
            if trial_data.get('serial_owner') == worker_id:
                owner_active = any(
                    f['status'] == 'running' and f.get('worker') == worker_id
                    for f in trial_data['folds'].values()
                )
                if not owner_active:
                    trial_data['serial_owner'] = None
        save()
    elif cmd == 'status':
        lines = []
        for trial_num, trial_data in sorted(state['trials'].items(), key=lambda item: int(item[0])):
            done = sum(1 for f in trial_data['folds'].values() if f['status'] == 'done')
            running = sum(1 for f in trial_data['folds'].values() if f['status'] == 'running')
            pending = sum(1 for f in trial_data['folds'].values() if f['status'] == 'pending')
            lines.append(
                f"trial={int(trial_num):05d} told={trial_data.get('told', False)} "
                f"done={done} running={running} pending={pending} serial_owner={trial_data.get('serial_owner')}"
            )
        print('\n'.join(lines))
    else:
        raise ValueError(f'Unknown state command: {cmd}')

    fcntl.flock(fh, fcntl.LOCK_UN)
PY
}

parse_trial_numbers() {
    local ask_output="$1"
    local trial_line
    trial_line=$(echo "${ask_output}" | grep '^TRIAL_NUMBERS=' | tail -1 || true)
    if [[ -z "${trial_line}" ]]; then
        echo ""
    else
        echo "${trial_line#TRIAL_NUMBERS=}"
    fi
}

fold_metrics_path() {
    local trial_num="$1"
    local fold_num="$2"
    printf '%s/trial_%05d/CV-fold_%d/validation-fold_%d_metrics.json\n' \
        "${OUTPUT_DIR}" "${trial_num}" "${fold_num}" "${fold_num}"
}

run_fold() {
    local trial_num="$1"
    local fold_num="$2"
    local worker_id="${3:-}"
    local fold_pid=""
    local heartbeat_pid=""
    local exit_code=0
    echo "[$(date '+%F %T')] role=${ROLE} trial=${trial_num} fold=${fold_num} host=$(hostname)"
    python Proteus.py \
        --run_mode tune-fold \
        --experiment "${YAML}" \
        --trial_number "${trial_num}" \
        --test_fold "${fold_num}" &
    fold_pid=$!

    if [[ -n "${worker_id}" ]]; then
        (
            while kill -0 "${fold_pid}" 2>/dev/null; do
                state_cmd heartbeat "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null || true
                sleep "${HEARTBEAT_SECONDS}"
            done
        ) &
        heartbeat_pid=$!
    fi

    wait "${fold_pid}" || exit_code=$?

    if [[ -n "${heartbeat_pid}" ]]; then
        kill "${heartbeat_pid}" 2>/dev/null || true
        wait "${heartbeat_pid}" 2>/dev/null || true
        state_cmd heartbeat "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null || true
    fi

    return "${exit_code}"
}

coordinator_loop() {
    activate_env
    state_cmd reconcile "${STALE_SECONDS}" >/dev/null
    while true; do
        state_cmd reconcile "${STALE_SECONDS}" >/dev/null

        ready_trials=$(state_cmd ready_to_tell)
        if [[ -n "${ready_trials}" ]]; then
            echo "[$(date '+%F %T')] Reporting completed trials: ${ready_trials}"
            python Proteus.py \
                --run_mode tune-tell \
                --experiment "${YAML}" \
                --trial_numbers "${ready_trials}" \
                --optuna_storage "${OPTUNA_STORAGE}" \
                --sampler_seed "${BASE_SAMPLER_SEED}" \
                "${study_args[@]}"
            state_cmd mark_told "${ready_trials}" >/dev/null
        fi

        active_trials=$(state_cmd count_active)
        if (( active_trials < TARGET_ACTIVE_TRIALS )); then
            ask_n=$((TARGET_ACTIVE_TRIALS - active_trials))
            if (( ask_n > ASK_BATCH )); then
                ask_n=${ASK_BATCH}
            fi
            if (( ask_n > 0 )); then
                echo "[$(date '+%F %T')] Asking Optuna for ${ask_n} config(s)"
                ask_output=$(python Proteus.py \
                    --run_mode tune-ask \
                    --experiment "${YAML}" \
                    --n_configs "${ask_n}" \
                    --optuna_storage "${OPTUNA_STORAGE}" \
                    --sampler_seed "${BASE_SAMPLER_SEED}" \
                    "${study_args[@]}")
                trial_csv=$(parse_trial_numbers "${ask_output}")
                if [[ -n "${trial_csv}" ]]; then
                    echo "[$(date '+%F %T')] Registered trials: ${trial_csv}"
                    state_cmd register_trials "${trial_csv}" >/dev/null
                fi
            fi
        fi

        sleep "${COORDINATOR_POLL_SECONDS}"
    done
}

rtx_worker_loop() {
    activate_env
    local worker_id
    worker_id="rtx:$(hostname):$$"
    while true; do
        state_cmd reconcile "${STALE_SECONDS}" >/dev/null
        task=$(state_cmd claim_rtx "${worker_id}")
        if [[ -z "${task}" ]]; then
            sleep "${WORKER_IDLE_SECONDS}"
            continue
        fi

        IFS=':' read -r trial_num fold_num <<< "${task}"
        if run_fold "${trial_num}" "${fold_num}" "${worker_id}"; then
            metrics_path=$(fold_metrics_path "${trial_num}" "${fold_num}")
            if [[ -f "${metrics_path}" ]]; then
                state_cmd mark_done "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null
            else
                echo "Missing metrics after successful fold run: ${metrics_path}" >&2
                state_cmd mark_failed "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null
            fi
        else
            state_cmd mark_failed "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null
            sleep "${WORKER_IDLE_SECONDS}"
        fi
    done
}

l40_worker_loop() {
    activate_env
    local worker_id
    worker_id="l40:$(hostname):$$"
    while true; do
        state_cmd reconcile "${STALE_SECONDS}" >/dev/null
        claim=$(state_cmd claim_l40 "${worker_id}")
        if [[ -z "${claim}" ]]; then
            sleep "${WORKER_IDLE_SECONDS}"
            continue
        fi

        IFS=':' read -r trial_num fold_csv <<< "${claim}"
        IFS=',' read -ra folds <<< "${fold_csv}"
        for fold_num in "${folds[@]}"; do
            metrics_path=$(fold_metrics_path "${trial_num}" "${fold_num}")
            if [[ -f "${metrics_path}" ]]; then
                state_cmd mark_done "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null
                continue
            fi
            if run_fold "${trial_num}" "${fold_num}" "${worker_id}"; then
                if [[ -f "${metrics_path}" ]]; then
                    state_cmd mark_done "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null
                else
                    echo "Missing metrics after successful fold run: ${metrics_path}" >&2
                    state_cmd mark_failed "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null
                    break
                fi
            else
                state_cmd mark_failed "${trial_num}" "${fold_num}" "${worker_id}" >/dev/null
                break
            fi
        done
    done
}

case "${ROLE}" in
    coordinator)
        coordinator_loop
        ;;
    rtx)
        rtx_worker_loop
        ;;
    l40)
        l40_worker_loop
        ;;
    status)
        state_cmd reconcile "${STALE_SECONDS}" >/dev/null
        state_cmd status
        ;;
    *)
        echo "Usage: $0 <coordinator|rtx|l40|status>"
        exit 1
        ;;
esac