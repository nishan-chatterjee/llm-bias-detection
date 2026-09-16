#!/usr/bin/env bash
set -Eeuo pipefail

# Bounded smoke test for a local llama.cpp completion binary and GGUF.
# Usage: LLAMA_BIN=/path/llama-completion MODEL=/path/model.gguf ./...sh
: "${LLAMA_BIN:?set LLAMA_BIN to llama-completion}"
: "${MODEL:?set MODEL to a GGUF checkpoint}"
[[ -x "$LLAMA_BIN" ]] || { echo "missing executable: $LLAMA_BIN" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing GGUF: $MODEL" >&2; exit 2; }

work="$(mktemp -d "${TMPDIR:-/tmp}/llama-smoke.XXXXXX")"
stdout_log="$work/completion.stdout"
stderr_log="$work/loader.stderr"
echo "logs=$work"

echo "host=$(hostname)"
echo "binary=$LLAMA_BIN"
echo "model=$MODEL"
echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
if git -C "$(dirname "$(dirname "$LLAMA_BIN")")" rev-parse --short HEAD >/dev/null 2>&1; then
  echo "llama_cpp_revision=$(git -C "$(dirname "$(dirname "$LLAMA_BIN")")" rev-parse --short HEAD)"
fi
before="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null || true)"
echo "vram_before=$before"

set +e
started="$(date +%s)"
display_prompt=()
if "$LLAMA_BIN" --help 2>&1 | grep -q -- '--no-display-prompt'; then
  display_prompt=(--no-display-prompt)
fi
timeout "${LLAMA_TIMEOUT_SECONDS:-240}s" "$LLAMA_BIN" \
  -m "$MODEL" -ngl "${LLAMA_GPU_LAYERS:-999}" -c 512 -n 8 \
  --temp 0 -p 'Write the uppercase words LLAMA, SMOKE and OK joined by underscores. Nothing else.' -no-cnv "${display_prompt[@]}" \
  >"$stdout_log" 2>"$stderr_log" &
timeout_pid=$!
wait "$timeout_pid"
rc=$?
elapsed=$(( $(date +%s) - started ))
set -e
cat "$stdout_log"
cat "$stderr_log" >&2
grep -Eq '^[[:space:]]*LLAMA_SMOKE_OK([[:space:]]|$)' "$stdout_log" || { echo "completion marker missing" >&2; exit 1; }
(( rc == 0 )) || { echo "llama.cpp exited with $rc" >&2; exit "$rc"; }
echo "elapsed_seconds=$elapsed"

# llama-completion is a child process of this script and has exited here.
after="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null || true)"
echo "vram_after=$after"
if kill -0 "$timeout_pid" 2>/dev/null; then
  echo "test process still present after exit: pid=$timeout_pid" >&2
  exit 1
fi
echo "own_process_pid=$timeout_pid own_process_after=none"
echo "LLAMA_LOAD_COMPLETION_UNLOAD_OK"
