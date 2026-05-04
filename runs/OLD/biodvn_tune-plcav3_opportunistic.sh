#!/bin/bash
set -euo pipefail

ROLE="${1:-}"

if [[ -z "${ROLE}" ]]; then
    echo "Usage: $0 <coordinator|rtx|l40|status>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXPERIMENT="PLCA_tuning/biodvn_tune-plcav3"
YAML="config/experiments/${EXPERIMENT}.yaml"
OUTPUT_DIR="results_training/biodvn_plcav3_proxy135_tuning"
STATE_DIR="${OUTPUT_DIR}/coordination"
STATE_JSON="${STATE_DIR}/queue_state.json"

ENV_NAME="${ENV_NAME:-proteus}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///results_training/optuna_plcav3_proxy135.db}"
STUDY_NAME="${STUDY_NAME:-}"
BASE_SAMPLER_SEED="${BASE_SAMPLER_SEED:-42}"

TARGET_PROXY_TRIALS="${TARGET_PROXY_TRIALS:-5}"
ASK_BATCH="${ASK_BATCH:-5}"
COORDINATOR_POLL_SECONDS="${COORDINATOR_POLL_SECONDS:-20}"
WORKER_IDLE_SECONDS="${WORKER_IDLE_SECONDS:-20}"
STALE_SECONDS="${STALE_SECONDS:-7200}"
HEARTBEAT_SECONDS=60

PROXY_FOLDS_CSV="1,3,5"

mkdir -p "${STATE_DIR}" logs

activate_env() {
    if command -v micromamba >/dev/null 2>&1; then
        eval "$(micromamba shell hook -s bash)"
        micromamba activate "${ENV_NAME}"
    fi
}

state_cmd() {
    python - "${STATE_JSON}" "${OUTPUT_DIR}" "$@" <<'PY'
import fcntl
import json
import math
import sys
import time
from pathlib import Path

state_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
cmd = sys.argv[3]
args = sys.argv[4:]

PROXY_FOLDS = (1, 3, 5)
COMPLETION_FOLDS = (2, 4)
ALL_FOLDS = tuple(sorted(PROXY_FOLDS + COMPLETION_FOLDS))

state_path.parent.mkdir(parents=True, exist_ok=True)


def metrics_path(trial_num: str, fold_num: str) -> Path:
    t = int(trial_num)
    f = int(fold_num)
    return output_dir / f"trial_{t:05d}" / f"CV-fold_{f}" / f"validation-fold_{f}_metrics.json"


def checkpoint_path(trial_num: str, fold_num: str) -> Path:
    t = int(trial_num)
    f = int(fold_num)
    return output_dir / f"trial_{t:05d}" / f"CV-fold_{f}" / "best_model.pt"


def cv_objective_path(trial_num: str) -> Path:
    t = int(trial_num)
    return output_dir / f"trial_{t:05d}" / "cv_objective.json"


def read_fold_score(path: Path) -> float | None:
    try:
        with open(path, 'r') as handle:
            metrics = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None

    value = metrics.get('weighted_tumor_gvw_crps')
    if value is None:
        value = metrics.get('balanced_tumor_gvw_crps')
    if value is None:
        return None

    value = float(value)
    return value if math.isfinite(value) else None


def compute_proxy_score(trial_num: str) -> float | None:
    scores = []
    for fold_num in PROXY_FOLDS:
        score = read_fold_score(metrics_path(trial_num, str(fold_num)))
        if score is None:
            return None
        scores.append(float(score))
    return sum(scores) / len(scores)


def ensure_trial_shape(trial_data: dict) -> None:
    trial_data.setdefault('order', 0)
    trial_data.setdefault('proxy_told', False)
    trial_data.setdefault('full_written', False)
    trial_data.setdefault('proxy_score', None)
    folds = trial_data.setdefault('folds', {})
    for fold in ALL_FOLDS:
        key = str(fold)
        fold_data = folds.get(key)
        if not isinstance(fold_data, dict):
            folds[key] = {'status': 'pending', 'worker': None, 'claimed_at': 0}
            continue
        fold_data.setdefault('status', 'pending')
        fold_data.setdefault('worker', None)
        fold_data.setdefault('claimed_at', 0)


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
                'proxy_told': False,
                'full_written': False,
                'proxy_score': None,
                'folds': {
                    str(fold): {
                        'status': 'pending',
                        'worker': None,
                        'claimed_at': 0,
                    }
                    for fold in ALL_FOLDS
                },
            }
            state['next_order'] += 1
        save()
    elif cmd == 'reconcile':
        stale_seconds = int(args[0]) if args else 0
        for trial_num, trial_data in state['trials'].items():
            ensure_trial_shape(trial_data)
            for fold_num, fold_data in trial_data['folds'].items():
                fold_metrics = metrics_path(trial_num, fold_num)
                if fold_metrics.exists():
                    fold_data['status'] = 'done'
                    continue
                status = fold_data.get('status', 'pending')
                claimed_at = int(fold_data.get('claimed_at', 0) or 0)
                if status == 'pending' and checkpoint_path(trial_num, fold_num).exists():
                    fold_data['status'] = 'running'
                    fold_data['worker'] = None
                    fold_data['claimed_at'] = now
                    continue
                if status == 'running' and claimed_at > 0 and stale_seconds > 0:
                    if now - claimed_at > stale_seconds:
                        fold_data['status'] = 'pending'
                        fold_data['worker'] = None
                        fold_data['claimed_at'] = 0

            proxy_score = compute_proxy_score(trial_num)
            trial_data['proxy_score'] = proxy_score if proxy_score is not None else None
            if cv_objective_path(trial_num).exists():
                trial_data['full_written'] = True
        save()
    elif cmd == 'count_proxy_active':
        active = sum(1 for trial in state['trials'].values() if not trial.get('proxy_told', False))
        print(active)
    elif cmd == 'ready_proxy_tell':
        ready = []
        for trial_num, trial_data in state['trials'].items():
            ensure_trial_shape(trial_data)
            proxy_score = trial_data.get('proxy_score')
            if trial_data.get('proxy_told', False):
                continue
            if proxy_score is None or not math.isfinite(float(proxy_score)):
                continue
            ready.append(int(trial_num))
        ready.sort()
        print(','.join(str(x) for x in ready))
    elif cmd == 'mark_proxy_told':
        trial_csv = args[0] if args else ''
        for token in [x.strip() for x in trial_csv.split(',') if x.strip()]:
            trial_data = state['trials'].get(token)
            if trial_data is None:
                continue
            ensure_trial_shape(trial_data)
            trial_data['proxy_told'] = True
        save()
    elif cmd == 'ready_full_finalize':
        ready = []
        for trial_num, trial_data in state['trials'].items():
            ensure_trial_shape(trial_data)
            if trial_data.get('full_written', False):
                continue
            if all(trial_data['folds'][str(fold)]['status'] == 'done' for fold in ALL_FOLDS):
                ready.append(int(trial_num))
        ready.sort()
        print(','.join(str(x) for x in ready))
    elif cmd == 'mark_full_written':
        trial_csv = args[0] if args else ''
        for token in [x.strip() for x in trial_csv.split(',') if x.strip()]:
            trial_data = state['trials'].get(token)
            if trial_data is None:
                continue
            ensure_trial_shape(trial_data)
            trial_data['full_written'] = True
        save()
    elif cmd == 'heartbeat':
        trial_num, fold_num, worker_id = args
        trial_data = state['trials'].get(str(trial_num))
        if trial_data is not None:
            ensure_trial_shape(trial_data)
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
            ensure_trial_shape(trial_data)
            folds = trial_data['folds']
            proxy_pending = [
                fold for fold in PROXY_FOLDS
                if folds[str(fold)].get('status') == 'pending'
            ]
            if not proxy_pending:
                continue
            proxy_started = sum(
                1 for fold in PROXY_FOLDS
                if folds[str(fold)].get('status') in ('running', 'done')
            )
            order = int(trial_data.get('order', 0))
            candidates.append((
                -proxy_started,
                order,
                int(trial_num),
                int(proxy_pending[0]),
            ))

        if not candidates:
            print('')
            fcntl.flock(fh, fcntl.LOCK_UN)
            sys.exit(0)

        _, _, trial_int, fold_num = min(candidates)
        trial_data = state['trials'][str(trial_int)]
        trial_data['folds'][str(fold_num)]['status'] = 'running'
        trial_data['folds'][str(fold_num)]['worker'] = worker_id
        trial_data['folds'][str(fold_num)]['claimed_at'] = now
        save()
        print(f'{trial_int}:{fold_num}')
    elif cmd == 'claim_l40':
        worker_id = args[0]
        candidates = []
        for trial_num, trial_data in state['trials'].items():
            ensure_trial_shape(trial_data)
            proxy_score = trial_data.get('proxy_score')
            if proxy_score is None or not math.isfinite(float(proxy_score)):
                continue
            pending = [
                fold for fold in COMPLETION_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'pending'
            ]
            if not pending:
                continue
            done_count = sum(
                1 for fold in COMPLETION_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'done'
            )
            running_count = sum(
                1 for fold in COMPLETION_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'running'
            )
            order = int(trial_data.get('order', 0))
            candidates.append((
                float(proxy_score),
                -(done_count + running_count),
                -done_count,
                order,
                int(trial_num),
                int(pending[0]),
            ))

        if not candidates:
            print('')
            fcntl.flock(fh, fcntl.LOCK_UN)
            sys.exit(0)

        _, _, _, _, trial_int, fold_num = min(candidates)
        trial_data = state['trials'][str(trial_int)]
        trial_data['folds'][str(fold_num)]['status'] = 'running'
        trial_data['folds'][str(fold_num)]['worker'] = worker_id
        trial_data['folds'][str(fold_num)]['claimed_at'] = now
        save()
        print(f'{trial_int}:{fold_num}')
    elif cmd == 'mark_done':
        trial_num, fold_num, worker_id = args
        trial_data = state['trials'].get(str(trial_num))
        if trial_data is not None:
            ensure_trial_shape(trial_data)
            fold_data = trial_data['folds'][str(fold_num)]
            if fold_data.get('worker') in (None, worker_id):
                fold_data['status'] = 'done'
                fold_data['worker'] = worker_id
                fold_data['claimed_at'] = now
        save()
    elif cmd == 'mark_failed':
        trial_num, fold_num, worker_id = args
        trial_data = state['trials'].get(str(trial_num))
        if trial_data is not None:
            ensure_trial_shape(trial_data)
            fold_data = trial_data['folds'][str(fold_num)]
            if fold_data.get('worker') in (None, worker_id):
                fold_data['status'] = 'pending'
                fold_data['worker'] = None
                fold_data['claimed_at'] = 0
        save()
    elif cmd == 'status':
        lines = []
        for trial_num, trial_data in sorted(state['trials'].items(), key=lambda item: int(item[0])):
            ensure_trial_shape(trial_data)
            proxy_done = sum(
                1 for fold in PROXY_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'done'
            )
            proxy_running = sum(
                1 for fold in PROXY_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'running'
            )
            proxy_pending = sum(
                1 for fold in PROXY_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'pending'
            )
            full_done = sum(
                1 for fold in COMPLETION_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'done'
            )
            full_running = sum(
                1 for fold in COMPLETION_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'running'
            )
            full_pending = sum(
                1 for fold in COMPLETION_FOLDS
                if trial_data['folds'][str(fold)].get('status') == 'pending'
            )
            proxy_score = trial_data.get('proxy_score')
            score_text = (
                f"{float(proxy_score):.4f}"
                if proxy_score is not None and math.isfinite(float(proxy_score))
                else "NA"
            )
            lines.append(
                f"trial={int(trial_num):05d} proxy_told={trial_data.get('proxy_told', False)} "
                f"full_written={trial_data.get('full_written', False)} proxy_score={score_text} "
                f"proxy_done={proxy_done} proxy_running={proxy_running} proxy_pending={proxy_pending} "
                f"full_done={full_done} full_running={full_running} full_pending={full_pending}"
            )
        print('\n'.join(lines))
    else:
        raise ValueError(f'Unknown state command: {cmd}')

    fcntl.flock(fh, fcntl.LOCK_UN)
PY
}

parse_csv_from_output() {
    local key="$1"
    local output="$2"
    local line
    line=$(printf '%s\n' "${output}" | grep "^${key}=" | tail -1 || true)
    if [[ -z "${line}" ]]; then
        echo ""
    else
        echo "${line#${key}=}"
    fi
}

proxy_ask_cmd() {
    local n_configs="$1"
    python - "${YAML}" "${OPTUNA_STORAGE}" "${STUDY_NAME}" "${BASE_SAMPLER_SEED}" "${n_configs}" <<'PY'
import json
import os
import sys

from config.experiment_spec import load_experiment_spec
from tuning.optuna_tuner import coordinated_proxy_ask

experiment_path, storage, study_name, sampler_seed, n_configs = sys.argv[1:6]
proxy_folds = [1, 3, 5]
study_name = study_name or None

trial_numbers = coordinated_proxy_ask(
    experiment_path=experiment_path,
    n_configs=int(n_configs),
    storage=storage,
    study_name=study_name,
    sampler_seed=int(sampler_seed),
)

exp = load_experiment_spec(experiment_path)
for trial_num in trial_numbers:
    run_dir = os.path.join(exp.output_dir, f"trial_{int(trial_num):05d}")
    pending_path = os.path.join(run_dir, "coordinated_pending.json")
    if not os.path.exists(pending_path):
        continue
    with open(pending_path, 'r') as handle:
        pending = json.load(handle)
    pending['proxy_folds'] = list(proxy_folds)
    with open(pending_path, 'w') as handle:
        json.dump(pending, handle, indent=2)
        handle.write('\n')

print("TRIAL_NUMBERS=" + ",".join(str(int(t)) for t in trial_numbers), flush=True)
PY
}

proxy_tell_cmd() {
    local trial_csv="$1"
    python - "${YAML}" "${OPTUNA_STORAGE}" "${STUDY_NAME}" "${BASE_SAMPLER_SEED}" "${trial_csv}" <<'PY'
import sys

from tuning.optuna_tuner import coordinated_proxy_tell

experiment_path, storage, study_name, sampler_seed, trial_csv = sys.argv[1:6]
study_name = study_name or None
trial_numbers = [int(x.strip()) for x in trial_csv.split(',') if x.strip()]

reported = coordinated_proxy_tell(
    experiment_path=experiment_path,
    trial_numbers=trial_numbers,
    storage=storage,
    study_name=study_name,
    sampler_seed=int(sampler_seed),
    proxy_folds=[1, 3, 5],
)

print("REPORTED_TRIAL_NUMBERS=" + ",".join(str(int(t)) for t in reported), flush=True)
PY
}

proxy_finalize_cmd() {
    local trial_csv="$1"
    python - "${YAML}" "${trial_csv}" <<'PY'
import sys

from tuning.optuna_tuner import coordinated_proxy_finalize

experiment_path, trial_csv = sys.argv[1:3]
trial_numbers = [int(x.strip()) for x in trial_csv.split(',') if x.strip()]

finalized = coordinated_proxy_finalize(
    experiment_path=experiment_path,
    trial_numbers=trial_numbers,
)

print("FINALIZED_TRIAL_NUMBERS=" + ",".join(str(int(t)) for t in finalized), flush=True)
PY
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

        ready_full=$(state_cmd ready_full_finalize)
        if [[ -n "${ready_full}" ]]; then
            echo "[$(date '+%F %T')] Finalizing promoted full-CV trials: ${ready_full}"
            if finalize_output=$(proxy_finalize_cmd "${ready_full}" 2>&1); then
                finalized_csv=$(parse_csv_from_output "FINALIZED_TRIAL_NUMBERS" "${finalize_output}")
                if [[ -n "${finalized_csv}" ]]; then
                    state_cmd mark_full_written "${finalized_csv}" >/dev/null
                fi
            else
                echo "${finalize_output}" >&2
            fi
        fi

        ready_proxy=$(state_cmd ready_proxy_tell)
        if [[ -n "${ready_proxy}" ]]; then
            echo "[$(date '+%F %T')] Reporting completed proxy trials: ${ready_proxy}"
            if tell_output=$(proxy_tell_cmd "${ready_proxy}" 2>&1); then
                reported_csv=$(parse_csv_from_output "REPORTED_TRIAL_NUMBERS" "${tell_output}")
                if [[ -n "${reported_csv}" ]]; then
                    state_cmd mark_proxy_told "${reported_csv}" >/dev/null
                fi
            else
                echo "${tell_output}" >&2
            fi
        fi

        active_proxy=$(state_cmd count_proxy_active)
        if (( active_proxy < TARGET_PROXY_TRIALS )); then
            ask_n=$((TARGET_PROXY_TRIALS - active_proxy))
            if (( ask_n > ASK_BATCH )); then
                ask_n=${ASK_BATCH}
            fi
            if (( ask_n > 0 )); then
                echo "[$(date '+%F %T')] Asking proxy study for ${ask_n} config(s)"
                if ask_output=$(proxy_ask_cmd "${ask_n}" 2>&1); then
                    trial_csv=$(parse_csv_from_output "TRIAL_NUMBERS" "${ask_output}")
                    if [[ -n "${trial_csv}" ]]; then
                        echo "[$(date '+%F %T')] Registered proxy trials: ${trial_csv}"
                        state_cmd register_trials "${trial_csv}" >/dev/null
                    fi
                else
                    echo "${ask_output}" >&2
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
        task=$(state_cmd claim_l40 "${worker_id}")
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