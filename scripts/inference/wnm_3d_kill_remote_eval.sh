#!/usr/bin/env bash
# Stop this checkout's GN-Bench remote evaluation and WNM-3D server processes.

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

DRY_RUN="false"
GRACE_PERIOD="10"
GN0_ROOT_OVERRIDE=""

print_help() {
    cat <<'EOF'
Usage: bash scripts/inference/wnm_3d_kill_remote_eval.sh [OPTIONS]

Stop WNM-3D servers launched from this checkout and GN-Bench remote evaluation
workers launched from the selected sibling GN0 checkout.

Options:
  --dry-run                 List matching processes without sending signals
  --grace-period SEC        Seconds to wait after SIGTERM before SIGKILL (default: 10)
  --gn0-root PATH           GN0 checkout to scope evaluation-process matching
  -h, --help                Show this help message
EOF
}

require_option_value() {
    local option="$1"
    local value="${2-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        echo "Error: $option requires a value." >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --grace-period)
            require_option_value "$1" "${2-}"
            GRACE_PERIOD="$2"
            shift 2
            ;;
        --gn0-root)
            require_option_value "$1" "${2-}"
            GN0_ROOT_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help >&2
            exit 2
            ;;
    esac
done

if ! [[ "$GRACE_PERIOD" =~ ^[0-9]+$ ]]; then
    echo "Error: --grace-period must be a non-negative integer." >&2
    exit 2
fi
if ! command -v pgrep >/dev/null 2>&1; then
    echo "Error: pgrep is required to locate inference and evaluation processes." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
SERVER_ENTRYPOINT="${REPO_ROOT}/scripts/inference/wnm_3d_server.py"
CURRENT_UID="$(id -u)"
RUNTIME_DIR="${WNM3D_RUNTIME_DIR:-${XDG_RUNTIME_DIR:-/tmp}/wnm-3d-${CURRENT_UID}}"

GN0_ROOT_CANDIDATE="${GN0_ROOT_OVERRIDE:-${WNM3D_GN0_ROOT:-${REPO_ROOT}/../GN0}}"
GN0_ROOT=""
if [[ -d "$GN0_ROOT_CANDIDATE" ]]; then
    GN0_ROOT="$(cd "$GN0_ROOT_CANDIDATE" && pwd -P)"
else
    echo "Warning: GN0 checkout not found at: $GN0_ROOT_CANDIDATE" >&2
    echo "         GN-Bench evaluation processes will not be selected." >&2
fi

process_start_id() {
    ps -p "$1" -o lstart= 2>/dev/null \
        | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

process_command() {
    ps -p "$1" -o command= 2>/dev/null || true
}

process_cwd() {
    local pid="$1"
    if [[ -e "/proc/${pid}/cwd" ]]; then
        readlink "/proc/${pid}/cwd" 2>/dev/null || true
        return
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -a -p "$pid" -d cwd -Fn 2>/dev/null \
            | sed -n 's/^n//p' \
            | head -n 1
    fi
}

is_owned_process() {
    local pid="$1"
    local owner_uid
    owner_uid="$(ps -p "$pid" -o uid= 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$owner_uid" && "$owner_uid" == "$CURRENT_UID" ]]
}

TARGET_PIDS=()
TARGET_START_IDS=()
TARGET_DESCRIPTIONS=()

add_target() {
    local pid="$1"
    local description="$2"
    local existing_pid

    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 0
    [[ "$pid" != "$$" ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    is_owned_process "$pid" || return 0

    for existing_pid in "${TARGET_PIDS[@]}"; do
        [[ "$existing_pid" == "$pid" ]] && return
    done

    TARGET_PIDS+=("$pid")
    TARGET_START_IDS+=("$(process_start_id "$pid")")
    TARGET_DESCRIPTIONS+=("$description")
}

remove_stale_pid_file() {
    local path="$1"
    if [[ "$DRY_RUN" == "false" ]]; then
        rm -f -- "$path"
    fi
}

collect_registered_servers() {
    local pid_file pid recorded_start recorded_root command current_start
    local -a pid_files=()

    [[ -d "$RUNTIME_DIR" ]] || return 0
    mapfile -t pid_files < <(find "$RUNTIME_DIR" -maxdepth 1 -type f -name 'server-*.pid' -print 2>/dev/null)
    for pid_file in "${pid_files[@]}"; do
        if [[ ! -O "$pid_file" ]]; then
            echo "Warning: ignoring PID file not owned by the current user: $pid_file" >&2
            continue
        fi

        pid="$(sed -n 's/^pid=//p' "$pid_file" | head -n 1)"
        recorded_start="$(sed -n 's/^start=//p' "$pid_file" | head -n 1)"
        recorded_root="$(sed -n 's/^repo_root=//p' "$pid_file" | head -n 1)"
        if [[ ! "$pid" =~ ^[1-9][0-9]*$ || "$recorded_root" != "$REPO_ROOT" ]]; then
            echo "Warning: removing invalid WNM-3D PID file: $pid_file" >&2
            remove_stale_pid_file "$pid_file"
            continue
        fi
        if ! kill -0 "$pid" 2>/dev/null || ! is_owned_process "$pid"; then
            remove_stale_pid_file "$pid_file"
            continue
        fi

        current_start="$(process_start_id "$pid")"
        command="$(process_command "$pid")"
        if [[ -z "$recorded_start" || "$current_start" != "$recorded_start" || "$command" != *"scripts/inference/wnm_3d_server.sh"* ]]; then
            echo "Warning: ignoring stale WNM-3D PID registration: $pid_file" >&2
            remove_stale_pid_file "$pid_file"
            continue
        fi
        add_target "$pid" "WNM-3D server launcher"
    done
}

collect_server_workers() {
    local pid command
    local -a candidates=()

    mapfile -t candidates < <(pgrep -u "$CURRENT_UID" -f 'wnm_3d_server[.]py' || true)
    for pid in "${candidates[@]}"; do
        command="$(process_command "$pid")"
        if [[ "$command" == *"$SERVER_ENTRYPOINT"* ]]; then
            add_target "$pid" "WNM-3D server worker"
        fi
    done
}

collect_gn0_evaluation() {
    local pid command cwd
    local -a candidates=()

    [[ -n "$GN0_ROOT" ]] || return 0
    mapfile -t candidates < <(
        {
            pgrep -u "$CURRENT_UID" -f 'gn0[.]evaluation[.]remote' || true
            pgrep -u "$CURRENT_UID" -f 'scripts/evaluation/[e]val_remote[.]sh' || true
        } | sort -u
    )
    for pid in "${candidates[@]}"; do
        cwd="$(process_cwd "$pid")"
        [[ "$cwd" == "$GN0_ROOT" ]] || continue
        command="$(process_command "$pid")"
        if [[ "$command" == *"-m gn0.evaluation.remote"* ]]; then
            add_target "$pid" "GN-Bench remote evaluation worker"
        elif [[ "$command" == *"scripts/evaluation/eval_remote.sh"* ]]; then
            add_target "$pid" "GN-Bench evaluation launcher"
        fi
    done
}

target_is_running() {
    local index="$1"
    local pid="${TARGET_PIDS[$index]}"
    local expected_start="${TARGET_START_IDS[$index]}"
    local current_start

    kill -0 "$pid" 2>/dev/null || return 1
    current_start="$(process_start_id "$pid")"
    [[ -n "$expected_start" && "$current_start" == "$expected_start" ]]
}

collect_registered_servers
collect_server_workers
collect_gn0_evaluation

if [[ ${#TARGET_PIDS[@]} -eq 0 ]]; then
    echo "No scoped WNM-3D server or GN-Bench evaluation processes found."
    exit 0
fi

echo "Scoped processes:"
for index in "${!TARGET_PIDS[@]}"; do
    printf '  PID %-8s %s\n' "${TARGET_PIDS[$index]}" "${TARGET_DESCRIPTIONS[$index]}"
done

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run: no signals sent."
    exit 0
fi

echo "Sending SIGTERM..."
kill -TERM "${TARGET_PIDS[@]}" 2>/dev/null || true

for ((elapsed=0; elapsed<GRACE_PERIOD; elapsed++)); do
    any_running="false"
    for index in "${!TARGET_PIDS[@]}"; do
        if target_is_running "$index"; then
            any_running="true"
            break
        fi
    done
    [[ "$any_running" == "false" ]] && break
    sleep 1
done

REMAINING_PIDS=()
for index in "${!TARGET_PIDS[@]}"; do
    if target_is_running "$index"; then
        REMAINING_PIDS+=("${TARGET_PIDS[$index]}")
    fi
done

if [[ ${#REMAINING_PIDS[@]} -gt 0 ]]; then
    echo "Sending SIGKILL to processes that did not exit: ${REMAINING_PIDS[*]}"
    kill -KILL "${REMAINING_PIDS[@]}" 2>/dev/null || true
fi

echo "Scoped GN-Bench evaluation and WNM-3D server processes cleaned."
