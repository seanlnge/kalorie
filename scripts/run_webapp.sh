#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEB_ROOT="${PROJECT_ROOT}/web"
ORIGINAL_DIR="$(pwd)"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

BACKEND_PID=""

cleanup() {
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" >/dev/null 2>&1; then
    kill "${BACKEND_PID}" >/dev/null 2>&1 || true
    wait "${BACKEND_PID}" 2>/dev/null || true
    echo "[run_webapp] Stopped backend (${BACKEND_PID})."
  fi
}

trap cleanup EXIT INT TERM

if [[ ! -d "${WEB_ROOT}" ]]; then
  echo "[run_webapp] Missing frontend directory: ${WEB_ROOT}"
  exit 1
fi

if [[ ! -d "${WEB_ROOT}/node_modules" ]]; then
  echo "[run_webapp] Installing frontend dependencies..."
  cd "${WEB_ROOT}"
  npm install
  cd "${ORIGINAL_DIR}"
fi

echo "[run_webapp] Starting backend on http://${BACKEND_HOST}:${BACKEND_PORT}"
cd "${PROJECT_ROOT}"
python -m uvicorn kalorie.webapi.main:create_app --factory --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
BACKEND_PID="$!"

sleep 2
if ! kill -0 "${BACKEND_PID}" >/dev/null 2>&1; then
  echo "[run_webapp] Backend failed to start."
  exit 1
fi

export VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://${BACKEND_HOST}:${BACKEND_PORT}}"
echo "[run_webapp] Starting frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo "[run_webapp] Using VITE_API_BASE_URL=${VITE_API_BASE_URL}"
cd "${WEB_ROOT}"
npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}"
