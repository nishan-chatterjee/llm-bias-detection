#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-run}"
case "${ACTION}" in
  design) exec bash "${HERE}/run_experiments.sh" design-mcq ;;
  run) exec bash "${HERE}/run_experiments.sh" mcq ;;
  smoke) exec bash "${HERE}/run_experiments.sh" smoke-mcq ;;
  *) echo "Usage: $0 {design|run|smoke}" >&2; exit 2 ;;
esac
