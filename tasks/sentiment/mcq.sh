#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-run}"
case "${ACTION}" in
  design|run|smoke) exec bash "${HERE}/run_experiments.sh" "${ACTION}" ;;
  *) echo "Usage: $0 {design|run|smoke}" >&2; exit 2 ;;
esac
