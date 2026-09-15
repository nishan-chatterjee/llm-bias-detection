#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-run}"
case "${ACTION}" in
  design) exec bash "${HERE}/run_experiments.sh" design-chat ;;
  run) exec bash "${HERE}/run_experiments.sh" chat ;;
  smoke) exec bash "${HERE}/run_experiments.sh" smoke-chat ;;
  *) echo "Usage: $0 {design|run|smoke}" >&2; exit 2 ;;
esac
