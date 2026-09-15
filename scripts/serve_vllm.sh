#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ALIAS="${1:-gemma-3-1b-it}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"

MODEL_SOURCE="$(${PYTHON_BIN} "${ROOT}/scripts/resolve_model.py" "${MODEL_ALIAS}" --field source)"
REVISION="$(${PYTHON_BIN} "${ROOT}/scripts/resolve_model.py" "${MODEL_ALIAS}" --field revision)"
ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL_SOURCE}"
  --served-model-name "${MODEL_ALIAS}"
  --tensor-parallel-size "${TP_SIZE}"
  --port "${PORT}"
  --trust-remote-code
)
if [[ -n "${REVISION}" ]]; then
  ARGS+=(--revision "${REVISION}")
fi

echo "Serving ${MODEL_ALIAS} on GPUs ${GPU_IDS} at http://127.0.0.1:${PORT}/v1"
printf ' + CUDA_VISIBLE_DEVICES=%q' "${GPU_IDS}"
printf ' %q' "${PYTHON_BIN}" "${ARGS[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" "${ARGS[@]}"
