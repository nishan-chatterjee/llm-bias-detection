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
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.parquet as pq

from materialize_hate_inputs import package_inputs
from chat_numeric import numeric_chat_table


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


def analysis_table_to_parquet(source: Path, target: Path) -> int:
    """Copy one tracked analysis table into a uniform Parquet release layer."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".parquet":
        table = pq.read_table(source)
    elif source.suffix == ".csv":
        table = pacsv.read_csv(source)
    else:
        raise ValueError(f"Unsupported analysis table format: {source}")
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
    if args.include_restricted_pct_inputs:
        raise ValueError('Restricted Political Compass questionnaires are not exported; retain authorized inputs locally.')
    ensure_empty_target(stage)
    row_counts: dict[str, int] = {}

    # Shared metadata.
    copy(release / "dataset" / "DATASET_CARD.md", stage / "README.md")
    copy(release / "models" / "serve" / "model_config.json", stage / "metadata" / "model_config.json")
    row_counts.update(package_inputs(stage, release))

    # Political Compass MCQ: eight primary models x three quantizations.
    mcq_source = source / "tasks" / "political-compass" / "output" / "mcq-v1-300"
    mcq_design_parts = []
    for model in PRIMARY:
        for quantization in QUANTIZATIONS:
            filename = f"results_{model}_{quantization}.csv"
            source_path = mcq_source / "results" / filename
            target = stage / "data" / "political_compass_mcq" / filename.replace(".csv", ".parquet")
            row_counts[str(target.relative_to(stage))] = csv_to_parquet(
                source_path, target
            )
            if quantization == "bf16":
                columns = [
                    "config_id", "model", "language", "ideology", "context_id",
                    "instr_type", "instr_idx", "persona_class", "persona_idx",
                    "key_type", "perm_id",
                ]
                mcq_design_parts.append(pd.read_csv(source_path, usecols=columns))
    mcq_design = (
        pd.concat(mcq_design_parts, ignore_index=True)
        .drop_duplicates("config_id")
        .sort_values("config_id")
        .reset_index(drop=True)
    )
    if len(mcq_design) != 14_400 or not mcq_design["config_id"].is_unique:
        raise ValueError("Canonical Political Compass MCQ design must contain 14,400 unique configurations")
    mcq_design_path = stage / "metadata" / "political_compass_mcq_design.csv"
    mcq_design.to_csv(mcq_design_path, index=False)
    row_counts[str(mcq_design_path.relative_to(stage))] = len(mcq_design)

    # Political Compass chat: four Gemma and four Qwen think/no-think variants.
    chat_source = source / "tasks" / "political-compass" / "output" / "chat-v2-300"
    chat_design = pd.read_csv(chat_source / "experimental_design.csv")
    chat_design = (
        chat_design[chat_design["model_variant"].isin(CHAT_VARIANTS)]
        .sort_values("config_id")
        .reset_index(drop=True)
    )
    if len(chat_design) != 21_600 or not chat_design["config_id"].is_unique:
        raise ValueError("Canonical Political Compass chat design must contain 21,600 unique configurations")
    chat_design_path = stage / "metadata" / "political_compass_chat_design.csv"
    chat_design.to_csv(chat_design_path, index=False)
    row_counts[str(chat_design_path.relative_to(stage))] = len(chat_design)
    for variant in CHAT_VARIANTS:
        target = stage / "data" / "political_compass_chat_numeric" / f"{variant}.parquet"
        table = numeric_chat_table(pajson.read_json(chat_source / f"{variant}.jsonl"))
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression="zstd", compression_level=5)
        row_counts[str(target.relative_to(stage))] = table.num_rows
    if args.include_ablation:
        target = stage / "data" / "political_compass_chat_ablation_numeric" / f"{ABLATION}.parquet"
        table = numeric_chat_table(pajson.read_json(chat_source / f"{ABLATION}.jsonl"))
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression="zstd", compression_level=5)
        row_counts[str(target.relative_to(stage))] = table.num_rows

    # Original prompt templates may be exported, but not proposition files or
    # raw chat traces containing the questionnaire wording.
    for path in sorted((source / "tasks" / "political-compass" / "data" / "prompts").glob("*.json")):
        copy(path, stage / "inputs" / "political_compass" / "prompts" / path.name)
    copy(
        source / "tasks" / "political-compass" / "data" / "chat-prompts" / "english.json",
        stage / "inputs" / "political_compass" / "chat-prompts" / "english.json",
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

    # Hate speech: compact aggregate plus optional item-level conversion from
    # the historical wide CSV handoff. The converter writes long Parquets and
    # never treats the first target's legacy q-identifiers as global item IDs.
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
    hate_status = "aggregate only; item-level handoff not supplied"
    if args.hate_results_dir:
        if not args.hate_design:
            raise ValueError("--hate-design is required with --hate-results-dir")
        conversion_manifest = stage / "metadata" / "hate_speech_conversion.json"
        subprocess.run(
            [
                sys.executable,
                str(
                    release
                    / "tasks"
                    / "hate-speech"
                    / "analysis"
                    / "convert_historical_results.py"
                ),
                "--raw",
                str(args.hate_results_dir.resolve()),
                "--design",
                str(args.hate_design.resolve()),
                "--output",
                str(stage / "data" / "hate_speech"),
                "--manifest",
                str(conversion_manifest),
            ],
            check=True,
        )
        for path in sorted((stage / "data" / "hate_speech").glob("*.parquet")):
            row_counts[str(path.relative_to(stage))] = pq.read_metadata(path).num_rows
        if len(list((stage / "data" / "hate_speech").glob("*.parquet"))) != 8:
            raise ValueError("Historical hate conversion did not produce eight Parquets")
        hate_status = (
            "eight primary-model item-prediction Parquets included; matching input "
            "corpus and prompt JSON supplied separately; original raw logits absent"
        )

    # Analysis-ready tables let every canonical notebook run from the pinned
    # Hugging Face snapshot without model weights or the non-redistributed PCT
    # scorer. These tables are derived only from the primary released outputs.
    analysis_tables = {
        "political_compass": {
            "pct_configuration_scores": release / "tasks" / "political-compass" / "analysis" / "data" / "pct_configuration_scores.csv",
            "chat_configuration_scores": release / "tasks" / "political-compass" / "analysis" / "data" / "chat_configuration_scores.parquet",
            "chat_stage_agreement_summary": release / "tasks" / "political-compass" / "analysis" / "data" / "chat_stage_agreement_summary.csv",
            "pct_factor_sensitivity": release / "tasks" / "political-compass" / "analysis" / "data" / "pct_factor_sensitivity.csv",
        },
        "sentiment": {
            name: release / "tasks" / "sentiment" / "analysis" / "data" / f"{name}.csv"
            for name in (
                "coverage_by_model",
                "config_level_metrics",
                "summary_by_model",
                "summary_by_model_persona",
                "factor_sensitivity_macro_f1",
                "topic_descriptive_metrics",
            )
        },
        "hate_speech": {
            name: release / "tasks" / "hate-speech" / "analysis" / "data" / f"{name}.csv"
            for name in (
                "hs_configuration_scores",
                "hs_factor_sensitivity",
                "hs_item_metrics_by_persona_target",
                "hs_calibration_by_model",
            )
        },
    }
    for task, tables in analysis_tables.items():
        for name, source_path in tables.items():
            target = stage / "data" / "analysis_ready" / task / f"{name}.parquet"
            row_counts[str(target.relative_to(stage))] = analysis_table_to_parquet(
                source_path, target
            )

    provenance = {
        "dataset_repo": args.dataset_repo,
        "visibility": "set by uploader; restricted questionnaire inputs are not exported",
        "primary_models": PRIMARY,
        "chat_variants": CHAT_VARIANTS,
        "included_secondary_ablation": bool(args.include_ablation),
        "political_compass_propositions_included_restricted": bool(args.include_restricted_pct_inputs),
        "political_compass_scorer_included": False,
        "political_compass_chat_raw_text_included": False,
        "model_weights_included": False,
        "hate_speech_item_predictions_status": hate_status,
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
    parser.add_argument("--dataset-repo", default="nishan-chatterjee/llm-bias-detection")
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument("--include-restricted-pct-inputs", action="store_true")
    parser.add_argument(
        "--hate-results-dir",
        type=Path,
        help="Directory containing the eight supplied historical wide result CSVs",
    )
    parser.add_argument(
        "--hate-design",
        type=Path,
        help="Supplied experimental_design_hs.csv used by the historical run",
    )
    package(parser.parse_args())


if __name__ == "__main__":
    main()
