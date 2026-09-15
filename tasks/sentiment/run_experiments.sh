#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/tasks/sentiment/output/ibm-sentiment}"
MODELS="${MODELS:-all}"
ACTION="${1:-all}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

design() {
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/sentiment/run_ibm_sentiment.py" generate-design \
    --models "${MODELS}" --output "${OUTPUT_ROOT}"
}

run_full() {
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/sentiment/run_ibm_sentiment.py" run \
    --models "${MODELS}" --gpus "${GPU_IDS}" --output "${OUTPUT_ROOT}"
}

smoke() {
  local smoke_root="${OUTPUT_ROOT}/smoke"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/sentiment/run_ibm_sentiment.py" generate-design \
    --samples 1 --models gemma-3-1b-it --output "${smoke_root}"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/sentiment/run_ibm_sentiment.py" run \
    --models gemma-3-1b-it --gpus "${GPU_IDS%%,*}" --output "${smoke_root}"
}

case "${ACTION}" in
  design) design ;;
  run) run_full ;;
  smoke) smoke ;;
  all) design; run_full ;;
  *) echo "Usage: $0 {design|run|smoke|all}" >&2; exit 2 ;;
esac
