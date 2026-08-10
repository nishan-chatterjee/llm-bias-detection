#!/usr/bin/env bash
set -euo pipefail

qa_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
primary_models="gemma-3-1b-it,gemma-3-4b-it,gemma-3-12b-it,gemma-3-27b-it,Qwen3-4B_no_think,Qwen3-8B_no_think,Qwen3-14B_no_think,Qwen3-32B_no_think,Qwen3-4B_think,Qwen3-8B_think,Qwen3-14B_think,Qwen3-32B_think"
models="${MODELS:-$primary_models}"

"$python_bin" "$qa_dir/scripts/00_build_trace_corpus.py" --models "$models"
"$python_bin" "$qa_dir/scripts/01_extract_features.py" --models "$models"
"$python_bin" "$qa_dir/scripts/02_analyze_existing_traces.py" --models "$models"
"$python_bin" "$qa_dir/scripts/03_phrase_analysis.py" --models "$models"
"$python_bin" "$qa_dir/scripts/07_question_feature_models.py"
"$python_bin" "$qa_dir/scripts/15_discover_expressive_heterogeneity.py" \
  --primary-only \
  --output-dir "$qa_dir/artifacts/tables/expressive_heterogeneity_primary"
"$python_bin" "$qa_dir/scripts/16_build_localized_notebooks.py"

echo "Primary-model qualitative tables and notebooks are ready."
