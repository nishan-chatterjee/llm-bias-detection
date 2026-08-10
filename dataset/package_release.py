#!/usr/bin/env python3
"""Build the companion Hugging Face dataset staging directory.

The script is intentionally allow-list based: only the paper's primary model
files (plus the named Gemma 27B ablation) can enter the release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.parquet as pq


PRIMARY = [
    "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
    "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
]
QUANTIZATIONS = ["bf16", "8bit", "4bit"]
CHAT_VARIANTS = [
    *[name for name in PRIMARY if name.startswith("gemma")],
    *[f"{name}_{mode}" for name in PRIMARY if name.startswith("Qwen") for mode in ("no_think", "think")],
]
ABLATION = "gemma-3-27b-it-abliterated-normpreserve-v1"


def ensure_empty_target(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty staging directory: {path}")


def copy(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def csv_to_parquet(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pacsv.read_csv(source)
    pq.write_table(table, target, compression="zstd", compression_level=5)
    return table.num_rows


def jsonl_to_parquet(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pajson.read_json(source)
    pq.write_table(table, target, compression="zstd", compression_level=5)
    return table.num_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    release = args.release.resolve()
    stage = args.stage.resolve()
    ensure_empty_target(stage)
    row_counts: dict[str, int] = {}

    # Shared metadata.
    copy(release / "dataset" / "DATASET_CARD.md", stage / "README.md")
    copy(release / "models" / "serve" / "model_config.json", stage / "metadata" / "model_config.json")

    # Political Compass MCQ: eight primary models x three quantizations.
    mcq_source = source / "tasks" / "political-compass" / "output" / "mcq-v1-300"
    copy(mcq_source / "experimental_design.csv", stage / "metadata" / "political_compass_mcq_design.csv")
    for model in PRIMARY:
        for quantization in QUANTIZATIONS:
            filename = f"results_{model}_{quantization}.csv"
            target = stage / "data" / "political_compass_mcq" / filename.replace(".csv", ".parquet")
            row_counts[str(target.relative_to(stage))] = csv_to_parquet(
                mcq_source / "results" / filename, target
            )

    # Political Compass chat: four Gemma and four Qwen think/no-think variants.
    chat_source = source / "tasks" / "political-compass" / "output" / "chat-v2-300"
    copy(chat_source / "experimental_design.csv", stage / "metadata" / "political_compass_chat_design.csv")
    for variant in CHAT_VARIANTS:
        target = stage / "data" / "political_compass_chat" / f"{variant}.parquet"
        row_counts[str(target.relative_to(stage))] = jsonl_to_parquet(
            chat_source / f"{variant}.jsonl", target
        )
    if args.include_ablation:
        target = stage / "data" / "political_compass_chat_ablation" / f"{ABLATION}.parquet"
        row_counts[str(target.relative_to(stage))] = jsonl_to_parquet(
            chat_source / f"{ABLATION}.jsonl", target
        )

    # Prompt templates are original experiment metadata. Proposition files are
    # staged only for a private/restricted upload pending Political Compass
    # redistribution review.
    for path in sorted((source / "tasks" / "political-compass" / "data" / "prompts").glob("*.json")):
        copy(path, stage / "inputs" / "political_compass" / "prompts" / path.name)
    copy(
        source / "tasks" / "political-compass" / "data" / "chat-prompts" / "english.json",
        stage / "inputs" / "political_compass" / "chat-prompts" / "english.json",
    )
    if args.include_restricted_pct_inputs:
        for path in sorted((source / "tasks" / "political-compass" / "data" / "questions").glob("*.json")):
            copy(path, stage / "restricted_inputs" / "political_compass" / "questions" / path.name)
        (stage / "restricted_inputs" / "README.md").write_text(
            "Political Compass proposition files. Keep private pending explicit redistribution clearance.\n",
            encoding="utf-8",
        )

    # IBM topic sentiment.
    ibm_source = source / "tasks" / "stance-sentiment" / "output" / "mcq-v1-300" / "ibm-sentiment"
    copy(ibm_source / "experimental_design.csv", stage / "metadata" / "ibm_sentiment_design.csv")
    for model in PRIMARY:
        target = stage / "data" / "ibm_sentiment" / f"{model}.parquet"
        row_counts[str(target.relative_to(stage))] = jsonl_to_parquet(
            ibm_source / f"{model}.jsonl", target
        )
    copy(
        release / "tasks" / "sentiment" / "data" / "questions" / "ibm-claim-stance" / "ibm_test_topics.csv",
        stage / "inputs" / "ibm_sentiment" / "ibm_test_topics.csv",
    )
    copy(
        release / "tasks" / "sentiment" / "data" / "prompts" / "english_2-way-sentiment.json",
        stage / "inputs" / "ibm_sentiment" / "english_2-way-sentiment.json",
    )

    # Hate speech: aggregate now, raw primary-model outputs after handoff.
    copy(
        release / "tasks" / "hate-speech" / "analysis" / "data" / "hs_configuration_scores.csv",
        stage / "data" / "hate_speech_aggregate" / "hs_configuration_scores.csv",
    )
    copy(
        release / "tasks" / "hate-speech" / "analysis" / "data" / "hs_factor_sensitivity.csv",
        stage / "data" / "hate_speech_aggregate" / "hs_factor_sensitivity.csv",
    )
    hate_design = release / "dataset" / "generated" / "hate_speech_design.csv"
    if hate_design.exists():
        copy(hate_design, stage / "metadata" / "hate_speech_design.csv")

    provenance = {
        "dataset_repo": args.dataset_repo,
        "visibility": "private",
        "primary_models": PRIMARY,
        "chat_variants": CHAT_VARIANTS,
        "included_secondary_ablation": bool(args.include_ablation),
        "political_compass_propositions_included_restricted": bool(args.include_restricted_pct_inputs),
        "political_compass_scorer_included": False,
        "model_weights_included": False,
        "hate_speech_raw_status": "pending colleague handoff",
        "row_counts": row_counts,
    }
    (stage / "metadata").mkdir(parents=True, exist_ok=True)
    (stage / "metadata" / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    manifest = []
    for path in sorted(p for p in stage.rglob("*") if p.is_file()):
        manifest.append(
            {
                "path": str(path.relative_to(stage)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rows": row_counts.get(str(path.relative_to(stage))),
            }
        )
    (stage / "metadata" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Packaged {len(manifest)} files ({sum(item['bytes'] for item in manifest):,} bytes) at {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="veritas-vox source checkout")
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--dataset-repo", default="nishan-chatterjee/polilean-evaluation-traces")
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument("--include-restricted-pct-inputs", action="store_true")
    package(parser.parse_args())


if __name__ == "__main__":
    main()
