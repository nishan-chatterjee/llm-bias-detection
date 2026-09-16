#!/usr/bin/env python3
"""Preflight the local model, downloaded-data, and task-input layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
]


def check_file(path: Path, errors: list[str], label: str) -> None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-local-models", action="store_true")
    parser.add_argument("--require-downloaded-data", action="store_true")
    parser.add_argument("--require-analysis-data", action="store_true")
    parser.add_argument("--require-pct-inputs", action="store_true")
    parser.add_argument("--require-hate-inputs", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    config_path = ROOT / "models" / "serve" / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing_config = sorted(set(PRIMARY) - set(config))
    if missing_config:
        errors.append("model aliases absent from model_config.json: " + ", ".join(missing_config))

    local: list[str] = []
    remote: list[str] = []
    for alias in PRIMARY:
        if (ROOT / "models" / "serve" / alias / "config.json").is_file():
            local.append(alias)
        elif config.get(alias, {}).get("repo_id") and config.get(alias, {}).get("revision"):
            remote.append(alias)
        else:
            errors.append(f"{alias} has neither local files nor a pinned Hub source")
    if args.require_local_models and len(local) != len(PRIMARY):
        errors.append("local checkpoints missing: " + ", ".join(sorted(set(PRIMARY) - set(local))))

    question_dir = ROOT / "tasks" / "political-compass" / "data" / "questions"
    pct_questions = sorted(question_dir.glob("*.json")) if question_dir.is_dir() else []
    if args.require_pct_inputs and len(pct_questions) != 14:
        errors.append(
            f"Political Compass runner needs 14 question files in {question_dir}; found {len(pct_questions)}"
        )

    release = Path(os.environ.get("LLM_BIAS_DATA_DIR", ROOT / "data" / "release"))
    if args.require_downloaded_data:
        check_file(release / "metadata" / "manifest.json", errors, "dataset manifest")
        for name in [
            "political_compass_mcq",
            "political_compass_chat",
            "ibm_sentiment",
            "hate_speech",
        ]:
            if not (release / "data" / name).is_dir():
                errors.append(f"missing downloaded dataset configuration: {release / 'data' / name}")
    if args.require_analysis_data:
        for name in ["political_compass", "sentiment", "hate_speech"]:
            if not (release / "data" / "analysis_ready" / name).is_dir():
                errors.append(
                    "missing downloaded analysis tables: "
                    + str(release / "data" / "analysis_ready" / name)
                )

    check_file(
        ROOT / "tasks" / "sentiment" / "data" / "questions" / "ibm-claim-stance" / "ibm_test_topics.csv",
        errors,
        "IBM 30-topic input",
    )
    if args.require_hate_inputs:
        check_file(
            ROOT / "tasks" / "hate-speech" / "data" / "prompts" / "english.json",
            errors,
            "hate-speech prompt file",
        )
        check_file(
            ROOT / "tasks" / "hate-speech" / "data" / "corpus" / "english.jsonl",
            errors,
            "hate-speech corpus",
        )

    print(f"model resolution: {len(local)} local, {len(remote)} pinned Hub fallbacks")
    print(f"Political Compass question files: {len(pct_questions)}/14 (required only for fresh inference)")
    if errors:
        print("preflight failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("preflight passed")


if __name__ == "__main__":
    main()
