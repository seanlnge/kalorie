#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_ROOT="$ROOT/web"
PYTHON_SRC_PATH="$ROOT/src"
PYTHON_PATH_SEPARATOR=":"
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*)
    if command -v cygpath >/dev/null 2>&1; then
      PYTHON_SRC_PATH="$(cygpath -w "$PYTHON_SRC_PATH")"
      PYTHON_PATH_SEPARATOR=";"
    fi
    ;;
esac
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-600}"
MODEL_NAME="${MODEL_NAME:-}"
NO_POLLER=0

PIDS=()
NAMES=()
PYTHON_CMD=()
NPM_CMD=()
STOPPING=0

usage() {
  cat <<'EOF'
Usage: ./start-stack.sh [options]

Starts the local Kalorie2 stack and stops every child process on Ctrl+C.

Options:
  --api-port PORT          FastAPI port (default: 8000)
  --web-port PORT          Vite port (default: 5173)
  --poll-interval SECONDS  Market poll interval (default: 600)
  --model-name NAME        Model name for the poller (default: newest model)
  --no-poller              Start only API and web
  PYTHON=/path/python      Optional Python executable override
  -h, --help               Show this help
EOF
}

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("$PYTHON")
  elif command -v powershell.exe >/dev/null 2>&1; then
    local python_path
    python_path="$(
      powershell.exe -NoProfile -Command "(Get-Command python -ErrorAction Stop).Source" |
        tr -d '\r'
    )"
    if [[ -n "$python_path" ]]; then
      if command -v cygpath >/dev/null 2>&1; then
        python_path="$(cygpath -u "$python_path")"
      elif command -v wslpath >/dev/null 2>&1; then
        python_path="$(wslpath -u "$python_path")"
      fi
      PYTHON_CMD=("$python_path")
    fi
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
  elif command -v py >/dev/null 2>&1; then
    PYTHON_CMD=(py -3)
  fi

  if [[ "${#PYTHON_CMD[@]}" -eq 0 ]]; then
    echo "Could not find Python. Set PYTHON=/path/to/python and retry." >&2
    exit 127
  fi
}

windows_command_path() {
  local command_name="$1"
  local command_path
  command_path="$(
    powershell.exe -NoProfile -Command "(Get-Command $command_name -ErrorAction Stop).Source" |
      tr -d '\r'
  )"
  if [[ -n "$command_path" ]]; then
    if command -v cygpath >/dev/null 2>&1; then
      command_path="$(cygpath -u "$command_path")"
    elif command -v wslpath >/dev/null 2>&1; then
      command_path="$(wslpath -u "$command_path")"
    fi
    printf '%s\n' "$command_path"
  fi
}

resolve_npm() {
  if [[ -n "${NPM:-}" ]]; then
    NPM_CMD=("$NPM")
  elif command -v cmd.exe >/dev/null 2>&1; then
    NPM_CMD=(cmd.exe /c npm)
  elif command -v npm >/dev/null 2>&1; then
    NPM_CMD=(npm)
  fi

  if [[ "${#NPM_CMD[@]}" -eq 0 ]]; then
    echo "Could not find npm. Set NPM=/path/to/npm and retry." >&2
    exit 127
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-port)
      API_PORT="$2"
      shift 2
      ;;
    --web-port)
      WEB_PORT="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --no-poller)
      NO_POLLER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if command -v powershell.exe >/dev/null 2>&1 && [[ "${KALORIE_STACK_FORCE_BASH:-0}" != "1" ]]; then
  ps_script="$ROOT/start-stack.ps1"
  if command -v cygpath >/dev/null 2>&1; then
    ps_script="$(cygpath -w "$ps_script")"
  elif command -v wslpath >/dev/null 2>&1; then
    ps_script="$(wslpath -w "$ps_script")"
  fi

  ps_args=(-ApiPort "$API_PORT" -WebPort "$WEB_PORT" -PollIntervalSeconds "$POLL_INTERVAL_SECONDS")
  if [[ -n "$MODEL_NAME" ]]; then
    ps_args+=(-ModelName "$MODEL_NAME")
  fi
  if [[ "$NO_POLLER" -eq 1 ]]; then
    ps_args+=(-NoPoller)
  fi

  exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ps_script" "${ps_args[@]}"
fi

kill_tree() {
  local pid="$1"
  local child

  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r child; do
      [[ -n "$child" ]] && kill_tree "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
  fi

  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  if [[ "$STOPPING" -eq 1 ]]; then
    return
  fi
  STOPPING=1
  echo
  echo "Stopping Kalorie2 stack..."

  local i
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "  stopping ${NAMES[$i]} pid=${PIDS[$i]}"
      kill_tree "${PIDS[$i]}"
    fi
  done

  sleep 1
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
      kill -KILL "${PIDS[$i]}" 2>/dev/null || true
    fi
  done
}

trap cleanup INT TERM EXIT

resolve_python
resolve_npm

start_process() {
  local name="$1"
  local workdir="$2"
  shift 2

  echo "Starting $name..."
  (
    cd "$workdir"
    if [[ -n "${PYTHONPATH:-}" ]]; then
      export PYTHONPATH="${PYTHON_SRC_PATH}${PYTHON_PATH_SEPARATOR}${PYTHONPATH}"
    else
      export PYTHONPATH="$PYTHON_SRC_PATH"
    fi
    exec "$@"
  ) &
  local pid=$!
  PIDS+=("$pid")
  NAMES+=("$name")
  echo "  $name pid=$pid"
}

start_process \
  "api" \
  "$ROOT" \
  "${PYTHON_CMD[@]}" -m uvicorn kalorie2.webapi.main:create_app \
    --factory \
    --host 127.0.0.1 \
    --port "$API_PORT"

start_process \
  "web" \
  "$WEB_ROOT" \
  "${NPM_CMD[@]}" run dev -- \
    --host 127.0.0.1 \
    --port "$WEB_PORT" \
    --strictPort

if [[ "$NO_POLLER" -eq 0 ]]; then
  poller_args=(
    "${PYTHON_CMD[@]}" -c 'from kalorie2.market_poller import app; app()'
    loop
    --interval-seconds "$POLL_INTERVAL_SECONDS"
  )
  if [[ -n "$MODEL_NAME" ]]; then
    poller_args+=(--model-name "$MODEL_NAME")
  fi
  start_process "poller" "$ROOT" "${poller_args[@]}"
fi

echo
echo "Kalorie2 stack is running."
echo "  API: http://127.0.0.1:$API_PORT"
echo "  Web: http://127.0.0.1:$WEB_PORT"
if [[ "$NO_POLLER" -eq 0 ]]; then
  echo "  Poller: every $POLL_INTERVAL_SECONDS seconds"
fi
echo "Press Ctrl+C to stop all processes."

while true; do
  sleep 1
  for i in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "${NAMES[$i]} exited unexpectedly." >&2
      exit 1
    fi
  done
done
