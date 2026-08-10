#!/usr/bin/env python3
"""Stream chat JSONL files into compact, analysis-ready Parquet partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import (  # noqa: E402
    ALL_MODELS,
    CORPUS_DIR,
    DESIGN_PATH,
    GENERAL_MODEL_ORDER_1,
    GENERAL_MODEL_ORDER_2,
    QUESTIONS_PATH,
    RAW_CHAT_DIR,
    ensure_artifact_directories,
)
from io_utils import (  # noqa: E402
    ParquetBatchWriter,
    canonical_distribution,
    design_lhs_lookup,
    explicit_stance_index,
    iter_jsonl,
    model_columns,
    normalize_text,
    parse_models,
    predicted_canonical_index,
    probability_entropy,
    probability_margin,
    split_reasoning_trace,
    stable_hash,
    word_count,
    write_json,
)


def integer_or(value: object, default: int = -1) -> int:
    return default if value is None or value == "" else int(value)


def rationale_only(visible_text: str, statement: str) -> str:
    text = visible_text
    if statement:
        text = text.replace(statement, " ")
        text = text.replace(f'"{statement}"', " ")
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(
            ("stance:", "answer:", "final answer:", "position:", "choice:")
        ):
            continue
        kept.append(stripped)
    return normalize_text(" ".join(kept))


def build_row(record: dict, model_variant: str, lhs_lookup: dict[int, int]) -> dict:
    stage1_text = str(record.get("stage1_text") or "")
    think_text, visible_text, has_think_block = split_reasoning_trace(stage1_text)
    statement = str(record.get("statement") or "")
    probabilities = canonical_distribution(record)
    predicted_index = predicted_canonical_index(record)
    explicit_index, explicit_source = explicit_stance_index(record, visible_text)
    model_meta = model_columns(model_variant)
    config_id = integer_or(record.get("config_id"))
    question_id = integer_or(record.get("question_id"))
    ideology = str(record.get("ideology", ""))

    trace_id = stable_hash(model_variant, ideology, config_id, question_id)
    full_normalized = normalize_text(stage1_text)
    visible_normalized = normalize_text(visible_text)
    think_normalized = normalize_text(think_text)
    rationale = rationale_only(visible_text, statement)

    row = {
        "trace_id": trace_id,
        "item_id": str(record.get("item_id") or ""),
        "model_variant": model_variant,
        **model_meta,
        "group_1": model_variant in GENERAL_MODEL_ORDER_1,
        "group_2": model_variant in GENERAL_MODEL_ORDER_2,
        "generation_mode": str(record.get("generation_mode") or ""),
        "ideology": ideology,
        "language": str(record.get("language") or ""),
        "config_id": config_id,
        "lhs_row": int(lhs_lookup.get(config_id, -1)),
        "question_id": question_id,
        "question_page": integer_or(record.get("question_page")),
        "statement": statement,
        "prompt": str(record.get("prompt") or ""),
        "context_id": integer_or(record.get("context_id")),
        "instr_type": str(record.get("instr_type") or ""),
        "reasoning_mode": str(record.get("reasoning_mode") or ""),
        "instr_idx": int(record.get("instr_idx") or 0),
        "persona_class": str(record.get("persona_class") or ""),
        "persona_idx": integer_or(record.get("persona_idx")),
        "key_type": str(record.get("key_type") or ""),
        "perm_id": int(record.get("perm_id") or 0),
        "suffix_idx": int(record.get("suffix_idx") or 0),
        "stage1_text": stage1_text,
        "full_text_normalized": full_normalized,
        "think_text": think_text,
        "think_text_normalized": think_normalized,
        "visible_text": visible_text,
        "visible_text_normalized": visible_normalized,
        "rationale_text": rationale,
        "has_think_block": bool(has_think_block),
        "stage1_tokens": int(record.get("stage1_tokens") or 0),
        "full_words": word_count(stage1_text),
        "think_words": word_count(think_text),
        "visible_words": word_count(visible_text),
        "rationale_words": word_count(rationale),
        "finish_reason": str(record.get("stage1_finish_reason") or ""),
        "finish_length": str(record.get("stage1_finish_reason") or "") == "length",
        "error": str(record.get("error") or ""),
        "successful": not bool(record.get("error")),
        "predicted_canonical_index": predicted_index,
        "explicit_stance_index": explicit_index,
        "explicit_stance_source": explicit_source,
        "explicit_stance_found": explicit_index == explicit_index,
        "stage1_stage2_agree": (
            bool(explicit_index == predicted_index)
            if explicit_index == explicit_index and predicted_index == predicted_index
            else False
        ),
        "stage1_stage2_comparable": (
            explicit_index == explicit_index and predicted_index == predicted_index
        ),
        "prob_0_strongly_disagree": probabilities[0],
        "prob_1_disagree": probabilities[1],
        "prob_2_agree": probabilities[2],
        "prob_3_strongly_agree": probabilities[3],
        "classification_entropy": probability_entropy(probabilities),
        "classification_margin": probability_margin(probabilities),
        "content_hash": hashlib.sha256(
            full_normalized.casefold().encode("utf-8")
        ).hexdigest()[:20],
    }
    return row


def process_model(
    model_variant: str,
    raw_dir: Path,
    output_dir: Path,
    lhs_lookup: dict[int, int],
    batch_size: int,
    max_rows: int | None,
    overwrite: bool,
    expected_rows: int,
) -> dict:
    source = raw_dir / f"{model_variant}.jsonl"
    destination = output_dir / f"{model_variant}.parquet"
    partial = output_dir / f".{model_variant}.parquet.partial"
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        expected = min(expected_rows, max_rows) if max_rows is not None else expected_rows
        try:
            parquet = pq.ParquetFile(destination)
            valid = (
                parquet.metadata.num_rows == expected
                and {"trace_id", "prompt", "stage1_text"}.issubset(parquet.schema.names)
            )
        except Exception:  # noqa: BLE001
            valid = False
        if valid:
            return {
                "model_variant": model_variant,
                "status": "existing_validated",
                "path": str(destination),
                "rows": expected,
            }
        print(
            f"{model_variant}: existing partition is invalid or incomplete; rebuilding",
            flush=True,
        )
        destination.unlink()
    if destination.exists():
        destination.unlink()
    if partial.exists():
        partial.unlink()

    rows = 0
    seen_ids: set[str] = set()
    duplicates = 0
    errors = 0
    with ParquetBatchWriter(partial, batch_size=batch_size) as writer:
        for record in iter_jsonl(source):
            item_id = str(record.get("item_id") or "")
            if item_id and item_id in seen_ids:
                duplicates += 1
                continue
            if item_id:
                seen_ids.add(item_id)
            writer.append(build_row(record, model_variant, lhs_lookup))
            rows += 1
            errors += bool(record.get("error"))
            if max_rows is not None and rows >= max_rows:
                break
    partial.replace(destination)

    return {
        "model_variant": model_variant,
        "status": "built",
        "source": str(source),
        "path": str(destination),
        "rows": rows,
        "duplicates_skipped": duplicates,
        "error_rows": errors,
        "sample_limited": max_rows is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="Comma-separated model variants")
    parser.add_argument("--raw-dir", type=Path, default=RAW_CHAT_DIR)
    parser.add_argument("--output-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--max-rows-per-model", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_artifact_directories()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.models, ALL_MODELS)
    lhs_lookup = design_lhs_lookup(DESIGN_PATH)
    design = pd.read_csv(DESIGN_PATH)
    question_count = len(json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")))
    expected_rows_by_model = (
        design.groupby("model_variant", observed=True).size() * question_count
    ).to_dict()
    manifest = []
    for model in models:
        if model not in expected_rows_by_model:
            raise KeyError(f"No experimental-design rows found for {model}")
        result = process_model(
            model,
            args.raw_dir,
            args.output_dir,
            lhs_lookup,
            args.batch_size,
            args.max_rows_per_model,
            args.overwrite,
            int(expected_rows_by_model[model]),
        )
        manifest.append(result)
        print(result, flush=True)

    write_json(args.output_dir / "manifest.json", manifest)
    written = sum(
        int(row.get("rows", 0))
        for row in manifest
        if row.get("status") == "built"
    )
    reused = sum(row.get("status") == "existing_validated" for row in manifest)
    print(
        f"Finished {len(models)} models; newly written rows={written:,}; "
        f"validated/reused models={reused}"
    )


if __name__ == "__main__":
    main()
