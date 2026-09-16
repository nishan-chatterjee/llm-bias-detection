#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/model.gguf [served-model-name]" >&2
  exit 2
fi

MODEL_PATH="$1"
MODEL_NAME="${2:-$(basename "${MODEL_PATH}" .gguf)}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
GPU_LAYERS="${GPU_LAYERS:--1}"
PORT="${PORT:-8080}"
CONTEXT_SIZE="${CONTEXT_SIZE:-8192}"
HOST="${HOST:-127.0.0.1}"
GPU_IDS="${GPU_IDS:-0}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "GGUF file not found: ${MODEL_PATH}" >&2
  exit 1
fi

echo "Serving ${MODEL_NAME} on GPUs ${GPU_IDS} with llama.cpp at http://${HOST}:${PORT}/v1"
printf ' + CUDA_VISIBLE_DEVICES=%q %q' "${GPU_IDS}" "${LLAMA_SERVER_BIN}"
printf ' %q' --model "${MODEL_PATH}" --alias "${MODEL_NAME}" \
  --n-gpu-layers "${GPU_LAYERS}" --ctx-size "${CONTEXT_SIZE}" \
  --host "${HOST}" --port "${PORT}"
printf '\n'
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
CUDA_VISIBLE_DEVICES="${GPU_IDS}" exec "${LLAMA_SERVER_BIN}" \
  --model "${MODEL_PATH}" \
  --alias "${MODEL_NAME}" \
  --n-gpu-layers "${GPU_LAYERS}" \
  --ctx-size "${CONTEXT_SIZE}" \
  --host "${HOST}" \
  --port "${PORT}"
