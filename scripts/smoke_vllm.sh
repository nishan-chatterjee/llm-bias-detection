#!/usr/bin/env bash
# Launch a private temporary endpoint, validate it, then stop only that server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ALIAS="${1:-gemma-3-1b-it}"
GPU_IDS="${GPU_IDS:-0}"
PORT="${PORT:-18083}"
SMOKE_LOG_DIR="$(mktemp -d /tmp/llm-bias-vllm-smoke.XXXXXX)"
server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
GPU_IDS="${GPU_IDS}" TP_SIZE=1 HOST=127.0.0.1 PORT="${PORT}" \
MAX_MODEL_LEN=2048 GPU_MEMORY_UTILIZATION=0.30 ENFORCE_EAGER=1 \
  bash "${ROOT}/scripts/serve_vllm.sh" "${MODEL_ALIAS}" \
  > "${SMOKE_LOG_DIR}/server.log" 2>&1 &
server_pid="$!"
echo "Smoke-server log: ${SMOKE_LOG_DIR}/server.log"
"${PYTHON_BIN:-python}" "${ROOT}/scripts/smoke_openai_server.py" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${MODEL_ALIAS}" \
  --wait-ready 300
