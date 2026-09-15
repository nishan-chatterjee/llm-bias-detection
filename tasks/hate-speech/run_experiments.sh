#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/tasks/hate-speech/output/hate-speech}"
MODEL_SELECTION="${MODELS:-all}"
HISTORICAL_ORDER="gemma-3-12b-it,Qwen3-4B,Qwen3-8B,Qwen3-14B,Qwen3-32B,gemma-3-1b-it,gemma-3-4b-it,gemma-3-27b-it"
RUN_MODELS="${MODEL_SELECTION}"
if [[ "${RUN_MODELS}" == "all" ]]; then
  RUN_MODELS="${HISTORICAL_ORDER}"
fi
PROMPTS="${PROMPTS:-${ROOT}/tasks/hate-speech/data/prompts/english.json}"
CORPUS="${CORPUS:-${ROOT}/tasks/hate-speech/data/corpus/english.jsonl}"
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
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/hate-speech/run_hate_speech.py" generate-design \
    --models "${MODEL_SELECTION}" --output "${OUTPUT_ROOT}/experimental_design.csv"
}

preflight() {
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/hate-speech/run_hate_speech.py" validate-inputs \
    --prompts "${PROMPTS}" --corpus "${CORPUS}"
}

run_full() {
  local model
  IFS=',' read -ra model_list <<< "${RUN_MODELS}"
  for model in "${model_list[@]}"; do
    run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/hate-speech/run_hate_speech.py" run \
      --model "${model}" --gpus "${GPU_IDS}" \
      --design "${OUTPUT_ROOT}/experimental_design.csv" \
      --prompts "${PROMPTS}" --corpus "${CORPUS}" --output "${OUTPUT_ROOT}"
  done
}

smoke() {
  local smoke_root="${OUTPUT_ROOT}/smoke"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/hate-speech/run_hate_speech.py" generate-design \
    --samples 1 --models gemma-3-1b-it --output "${smoke_root}/experimental_design.csv"
  preflight
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/hate-speech/run_hate_speech.py" run \
    --model gemma-3-1b-it --gpus "${GPU_IDS}" \
    --design "${smoke_root}/experimental_design.csv" \
    --prompts "${PROMPTS}" --corpus "${CORPUS}" --output "${smoke_root}" \
    --max-configs 1 --max-items 2
}

case "${ACTION}" in
  design) design ;;
  preflight) preflight ;;
  run) preflight; run_full ;;
  smoke) smoke ;;
  all) design; preflight; run_full ;;
  *) echo "Usage: $0 {design|preflight|run|smoke|all}" >&2; exit 2 ;;
esac
