#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/tasks/political-compass/output}"
PRIMARY="gemma-3-1b-it,gemma-3-4b-it,gemma-3-12b-it,gemma-3-27b-it,Qwen3-4B,Qwen3-8B,Qwen3-14B,Qwen3-32B"
CHAT_STANDARD="gemma-3-1b-it,gemma-3-4b-it,gemma-3-12b-it,gemma-3-27b-it,Qwen3-4B_no_think,Qwen3-8B_no_think,Qwen3-14B_no_think,Qwen3-32B_no_think"
CHAT_THINK="Qwen3-4B_think,Qwen3-8B_think,Qwen3-14B_think,Qwen3-32B_think"
MODELS="${MODELS:-${PRIMARY}}"
MCQ_LANGUAGES="${MCQ_LANGUAGES:-bulgarian,czech,english,french,german,italian,persian,polish,portuguese,romanian,russian,slovene,spanish,turkish}"
QUANTIZATIONS="${QUANTIZATIONS:-bf16,8bit,4bit}"
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

design_mcq() {
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/mcq.py" generate-design \
    --models "${MODELS}" --languages "${MCQ_LANGUAGES}" \
    --output "${OUTPUT_ROOT}/mcq/experimental_design.csv"
}

design_chat() {
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/chat.py" generate-design \
    --models "${PRIMARY}" --language english --output "${OUTPUT_ROOT}/chat"
}

design() {
  design_mcq
  design_chat
}

run_mcq() {
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/mcq.py" run \
    --models "${MODELS}" --quantizations "${QUANTIZATIONS}" \
    --gpus "${GPU_IDS}" --output "${OUTPUT_ROOT}/mcq"
}

run_chat_standard() {
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/chat.py" run \
    --models "${CHAT_STANDARD}" --gpus "${GPU_IDS}" --answer chat-classify \
    --language english --output "${OUTPUT_ROOT}/chat"
}

run_chat_think() {
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/chat.py" run \
    --models "${CHAT_THINK}" --gpus "${GPU_IDS}" --answer chat-classify \
    --language english --output "${OUTPUT_ROOT}/chat"
}

smoke_mcq() {
  local smoke_root="${OUTPUT_ROOT}/smoke"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/mcq.py" generate-design \
    --samples 1 --models gemma-3-1b-it --languages english \
    --output "${smoke_root}/mcq/experimental_design.csv"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/mcq.py" run \
    --models gemma-3-1b-it --quantizations bf16 --gpus "${GPU_IDS%%,*}" \
    --max-configs 1 --max-questions 2 --output "${smoke_root}/mcq"
}

smoke_chat() {
  local smoke_root="${OUTPUT_ROOT}/smoke"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/chat.py" generate-design \
    --samples 1 --models gemma-3-1b-it --language english --output "${smoke_root}/chat"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/chat.py" run \
    --models gemma-3-1b-it --gpus "${GPU_IDS%%,*}" --answer chat-classify \
    --language english --max-configs 1 --max-questions 2 \
    --output "${smoke_root}/chat"
}

smoke_chat_think() {
  local smoke_root="${OUTPUT_ROOT}/smoke"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/chat.py" generate-design \
    --samples 1 --models Qwen3-4B_think --language english --output "${smoke_root}/chat-think"
  run_cmd "${PYTHON_BIN}" "${ROOT}/tasks/political-compass/chat.py" run \
    --models Qwen3-4B_think --gpus "${GPU_IDS%%,*}" --answer chat-classify \
    --language english --max-configs 1 --max-questions 2 \
    --output "${smoke_root}/chat-think"
}

smoke() {
  smoke_mcq
  smoke_chat
  smoke_chat_think
}

case "${ACTION}" in
  design) design ;;
  design-mcq) design_mcq ;;
  design-chat) design_chat ;;
  mcq) run_mcq ;;
  chat) run_chat_standard ;;
  chat-think) run_chat_think ;;
  smoke-mcq) smoke_mcq ;;
  smoke-chat) smoke_chat ;;
  smoke-chat-think) smoke_chat_think ;;
  smoke) smoke ;;
  all) design; run_mcq; run_chat_standard; run_chat_think ;;
  *) echo "Usage: $0 {design|design-mcq|design-chat|mcq|chat|chat-think|smoke-mcq|smoke-chat|smoke-chat-think|smoke|all}" >&2; exit 2 ;;
esac
