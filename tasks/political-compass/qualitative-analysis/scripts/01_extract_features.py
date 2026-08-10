#!/usr/bin/env python3
"""Extract transparent response features and scorer-aware question metadata."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import (  # noqa: E402
    ALL_MODELS,
    CORPUS_DIR,
    FEATURE_DIR,
    PROMPTS_PATH,
    QUESTIONS_PATH,
    SCORER_PATH,
    ensure_artifact_directories,
)
from feature_definitions import (  # noqa: E402
    ABSOLUTE_RE,
    CONCESSIVE_RE,
    MODAL_RE,
    NEGATION_RE,
    QUESTION_TOPICS,
    SENSITIVE_QUESTION_IDS,
    TOKEN_RE,
    lexicon_features,
    token_counts,
)
from io_utils import (  # noqa: E402
    ParquetBatchWriter,
    normalize_text,
    parse_models,
    stable_hash,
    write_json,
)


TARGET_SIGNS = {
    "libertarian_left": {"economic": -1.0, "social": -1.0},
    "libertarian_right": {"economic": 1.0, "social": -1.0},
    "authoritarian_left": {"economic": -1.0, "social": 1.0},
    "authoritarian_right": {"economic": 1.0, "social": 1.0},
}


def question_table() -> pd.DataFrame:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    scorer = np.load(SCORER_PATH)
    economic = scorer["weights_economic"]
    social = scorer["weights_social"]
    rows = []
    for question in questions:
        question_id = int(question["id"])
        statement = str(question["statement"])
        econ_range = float(np.ptp(economic[question_id]))
        social_range = float(np.ptp(social[question_id]))
        if econ_range > 1e-9:
            axis = "economic"
            weights = economic[question_id].astype(float)
        elif social_range > 1e-9:
            axis = "social"
            weights = social[question_id].astype(float)
        else:
            axis = "unweighted"
            weights = np.zeros(4, dtype=float)
        direction = 0
        if axis != "unweighted":
            direction = 1 if weights[-1] > weights[0] else -1
        tokens = TOKEN_RE.findall(statement)
        rows.append(
            {
                "question_id": question_id,
                "question_page": int(question["page"]),
                "statement": statement,
                "topic": QUESTION_TOPICS[question_id],
                "sensitive_topic": question_id in SENSITIVE_QUESTION_IDS,
                "scorer_axis": axis,
                "scorer_weight_range": float(np.ptp(weights)),
                "agree_pole_sign": direction,
                "weight_0_strongly_disagree": float(weights[0]),
                "weight_1_disagree": float(weights[1]),
                "weight_2_agree": float(weights[2]),
                "weight_3_strongly_agree": float(weights[3]),
                "statement_words": len(tokens),
                "statement_chars": len(statement),
                "absolute_term_count": len(ABSOLUTE_RE.findall(statement)),
                "negation_count": len(NEGATION_RE.findall(statement)),
                "modal_count": len(MODAL_RE.findall(statement)),
                "concessive_count": len(CONCESSIVE_RE.findall(statement)),
                "clause_marker_count": statement.count(",")
                + statement.count(";")
                + len(CONCESSIVE_RE.findall(statement)),
                "question_feature_version": "v1",
            }
        )
    return pd.DataFrame(rows)


def persona_vocabulary() -> dict[str, set[str]]:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for ideology, templates in prompts["ideology_insertions"].items():
        vocabulary: set[str] = set()
        for insertions in templates.values():
            for value in insertions.values():
                vocabulary.update(token_counts(str(value)))
        # Remove generic labels that make every mention a trivial hit.
        vocabulary.difference_update(
            {
                "the",
                "a",
                "an",
                "and",
                "of",
                "for",
                "left",
                "right",
                "authoritarian",
                "libertarian",
                "center",
                "centrist",
                "centrism",
                "political",
            }
        )
        result[ideology] = vocabulary
    return result


def target_metrics(row: dict, question: dict) -> dict:
    ideology = str(row["ideology"])
    axis = str(question["scorer_axis"])
    predicted = row.get("predicted_canonical_index")
    probabilities = np.asarray(
        [
            row.get("prob_0_strongly_disagree"),
            row.get("prob_1_disagree"),
            row.get("prob_2_agree"),
            row.get("prob_3_strongly_agree"),
        ],
        dtype=float,
    )
    weights = np.asarray(
        [
            question["weight_0_strongly_disagree"],
            question["weight_1_disagree"],
            question["weight_2_agree"],
            question["weight_3_strongly_agree"],
        ],
        dtype=float,
    )
    target_sign = TARGET_SIGNS.get(ideology, {}).get(axis, float("nan"))
    result = {
        "target_sign": target_sign,
        "target_projection_hard": float("nan"),
        "target_projection_expected": float("nan"),
        "target_aligned": False,
        "target_comparable": False,
    }
    if not math.isfinite(target_sign) or axis == "unweighted":
        return result
    centered = weights - float(weights.mean())
    if predicted is not None and math.isfinite(float(predicted)):
        hard = target_sign * centered[int(predicted)]
        result["target_projection_hard"] = float(hard)
        result["target_aligned"] = bool(hard > 0)
        result["target_comparable"] = True
    if np.isfinite(probabilities).all() and probabilities.sum() > 0:
        probabilities = probabilities / probabilities.sum()
        result["target_projection_expected"] = float(
            target_sign * np.dot(probabilities, centered)
        )
    return result


def prompt_overlap(text: str, ideology: str, vocabulary: dict[str, set[str]]) -> dict:
    response_tokens = token_counts(text)
    persona_tokens = vocabulary.get(ideology, set())
    if not persona_tokens:
        return {
            "persona_vocab_overlap_count": 0,
            "persona_vocab_overlap_fraction": 0.0,
        }
    overlap = sum(min(1, response_tokens[token]) for token in persona_tokens)
    return {
        "persona_vocab_overlap_count": int(overlap),
        "persona_vocab_overlap_fraction": float(overlap / len(persona_tokens)),
    }


def process_model(
    model: str,
    corpus_dir: Path,
    output_dir: Path,
    question_lookup: dict[int, dict],
    vocabulary: dict[str, set[str]],
    batch_size: int,
    overwrite: bool,
) -> dict:
    source = corpus_dir / f"{model}.parquet"
    destination = output_dir / f"{model}.parquet"
    partial = output_dir / f".{model}.parquet.partial"
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        try:
            source_rows = pq.ParquetFile(source).metadata.num_rows
            existing = pq.ParquetFile(destination)
            valid = (
                existing.metadata.num_rows == source_rows
                and {
                    "trace_id",
                    "target_projection_expected",
                    "visible_hedging_present",
                }.issubset(existing.schema.names)
            )
        except Exception:  # noqa: BLE001
            valid = False
        if valid:
            return {
                "model_variant": model,
                "status": "existing_validated",
                "path": str(destination),
                "rows": source_rows,
            }
        print(
            f"{model}: existing feature partition is invalid or incomplete; rebuilding",
            flush=True,
        )
        destination.unlink()
    if destination.exists():
        destination.unlink()
    if partial.exists():
        partial.unlink()
    columns = [
        "trace_id",
        "model_variant",
        "base_model",
        "family",
        "protocol",
        "size_b",
        "group_1",
        "group_2",
        "ideology",
        "config_id",
        "lhs_row",
        "question_id",
        "context_id",
        "instr_type",
        "reasoning_mode",
        "instr_idx",
        "persona_class",
        "persona_idx",
        "key_type",
        "perm_id",
        "suffix_idx",
        "full_text_normalized",
        "think_text_normalized",
        "visible_text_normalized",
        "rationale_text",
        "stage1_tokens",
        "full_words",
        "think_words",
        "visible_words",
        "rationale_words",
        "finish_length",
        "successful",
        "predicted_canonical_index",
        "explicit_stance_index",
        "explicit_stance_found",
        "stage1_stage2_agree",
        "stage1_stage2_comparable",
        "prob_0_strongly_disagree",
        "prob_1_disagree",
        "prob_2_agree",
        "prob_3_strongly_agree",
        "classification_entropy",
        "classification_margin",
        "content_hash",
    ]
    rows = 0
    parquet = pq.ParquetFile(source)
    with ParquetBatchWriter(partial, batch_size=batch_size) as writer:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                question = question_lookup[int(row["question_id"])]
                output = {key: value for key, value in row.items() if not key.endswith("_text_normalized")}
                output.update(
                    {
                        "topic": question["topic"],
                        "sensitive_topic": question["sensitive_topic"],
                        "scorer_axis": question["scorer_axis"],
                        "scorer_weight_range": question["scorer_weight_range"],
                    }
                )
                output.update(target_metrics(row, question))
                output.update(
                    lexicon_features(str(row["visible_text_normalized"]), prefix="visible_")
                )
                output.update(
                    lexicon_features(str(row["think_text_normalized"]), prefix="think_")
                )
                output.update(prompt_overlap(
                    str(row["visible_text_normalized"]),
                    str(row["ideology"]),
                    vocabulary,
                ))
                output["visible_text_id"] = stable_hash(
                    row["model_variant"],
                    row["ideology"],
                    row["lhs_row"],
                    row["question_id"],
                    normalize_text(row["visible_text_normalized"]),
                )
                writer.append(output)
                rows += 1
    partial.replace(destination)
    return {
        "model_variant": model,
        "status": "built",
        "path": str(destination),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--output-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_artifact_directories()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = question_table()
    questions.to_csv(args.output_dir / "question_features.csv", index=False)
    question_lookup = {
        int(row["question_id"]): row for row in questions.to_dict("records")
    }
    vocabulary = persona_vocabulary()
    write_json(
        args.output_dir / "persona_vocabulary.json",
        {key: sorted(value) for key, value in vocabulary.items()},
    )

    manifest = []
    for model in parse_models(args.models, ALL_MODELS):
        result = process_model(
            model,
            args.corpus_dir,
            args.output_dir,
            question_lookup,
            vocabulary,
            args.batch_size,
            args.overwrite,
        )
        manifest.append(result)
        print(result, flush=True)
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
