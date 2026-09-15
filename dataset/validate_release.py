#!/usr/bin/env python3
"""Validate a packaged LLM Bias Detection Hugging Face dataset directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


PRIMARY = {
    "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
    "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_struct_probabilities(table, column: str, expected_candidates: int) -> None:
    values = table[column].combine_chunks()
    probability_sum = np.zeros(len(values), dtype=float)
    present = np.zeros(len(values), dtype=int)
    for index in range(values.type.num_fields):
        child = values.field(index)
        valid = np.asarray(pc.is_valid(child))
        present += valid.astype(int)
        probability_sum += np.asarray(pc.fill_null(child, 0.0), dtype=float)
    assert np.all(present == expected_candidates), (column, np.unique(present, return_counts=True))
    assert np.allclose(probability_sum, 1.0, atol=1e-8), (
        column,
        float(np.max(np.abs(probability_sum - 1.0))),
    )


def validate(stage: Path) -> None:
    manifest_path = stage / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest:
        path = stage / item["path"]
        assert path.exists(), path
        assert path.stat().st_size == item["bytes"], path
        assert digest(path) == item["sha256"], path

    forbidden_suffixes = {".npz", ".bin", ".safetensors", ".gguf", ".pt", ".pth"}
    forbidden = [path for path in stage.rglob("*") if path.suffix.lower() in forbidden_suffixes]
    assert not forbidden, forbidden

    mcq_files = sorted((stage / "data" / "political_compass_mcq").glob("*.parquet"))
    chat_files = sorted((stage / "data" / "political_compass_chat").glob("*.parquet"))
    ibm_files = sorted((stage / "data" / "ibm_sentiment").glob("*.parquet"))
    hate_files = sorted((stage / "data" / "hate_speech").glob("*.parquet"))
    assert len(mcq_files) == 24
    assert len(chat_files) == 12
    assert len(ibm_files) == 8
    assert len(hate_files) == 8

    analysis_expected = {
        "political_compass": {
            "pct_configuration_scores.parquet": 43_200,
            "chat_configuration_scores.parquet": 21_600,
            "chat_stage_agreement_summary.parquet": 12,
            "pct_factor_sensitivity.parquet": 16,
        },
        "sentiment": {
            "coverage_by_model.parquet": 8,
            "config_level_metrics.parquet": 14_400,
            "summary_by_model.parquet": 8,
            "summary_by_model_persona.parquet": 48,
            "factor_sensitivity_macro_f1.parquet": 8,
            "topic_descriptive_metrics.parquet": 1_440,
        },
        "hate_speech": {
            "hs_configuration_scores.parquet": 14_400,
            "hs_factor_sensitivity.parquet": 8,
            "hs_item_metrics_by_persona_target.parquet": 480,
            "hs_calibration_by_model.parquet": 80,
        },
    }
    for task, expected in analysis_expected.items():
        task_dir = stage / "data" / "analysis_ready" / task
        assert task_dir.is_dir(), task_dir
        observed = {
            path.name: pq.read_metadata(path).num_rows
            for path in task_dir.glob("*.parquet")
        }
        assert observed == expected, (task, observed, expected)

    for path in mcq_files:
        frame = pq.read_table(path).to_pandas()
        assert len(frame) == 1800
        assert frame["config_id"].is_unique, path
        probability = frame.filter(regex=r"^q\d+_prob_ans\d+$").to_numpy(float).reshape(len(frame), 62, 4)
        assert np.allclose(probability.sum(axis=2), 1.0, atol=2e-2), path
        assert set(frame["model"].map(lambda value: str(value).split("/")[-1])).issubset(PRIMARY)

    for path in chat_files:
        table = pq.read_table(
            path,
            columns=[
                "config_id", "question_id", "error", "classification_probs",
                "stage1_text", "classification_logprobs", "statement",
            ],
        )
        # 1,800 prompt configurations x 62 propositions per chat variant.
        assert table.num_rows == 111_600, (path, table.num_rows)
        assert {"stage1_text", "classification_logprobs", "classification_probs", "statement"}.issubset(table.column_names)
        assert table["error"].null_count == table.num_rows, path
        pairs = np.asarray(table["config_id"]) * 100 + np.asarray(table["question_id"])
        assert len(np.unique(pairs)) == table.num_rows, path
        assert len(np.unique(np.asarray(table["config_id"]))) == 1800, path
        assert len(np.unique(np.asarray(table["question_id"]))) == 62, path
        validate_struct_probabilities(table, "classification_probs", 4)

    for path in ibm_files:
        table = pq.read_table(
            path,
            columns=[
                "item_uid", "error", "classification_logprobs",
                "classification_probs", "gold_label", "target",
            ],
        )
        assert table.num_rows == 54_000, (path, table.num_rows)
        assert {"classification_logprobs", "classification_probs", "gold_label", "target"}.issubset(table.column_names)
        assert table["error"].null_count == table.num_rows, path
        assert len(np.unique(np.asarray(table["item_uid"]))) == table.num_rows, path
        validate_struct_probabilities(table, "classification_probs", 2)

    for path in hate_files:
        parquet = pq.ParquetFile(path)
        assert parquet.metadata.num_rows == 1_800 * 1_332, path
        assert "text" not in parquet.schema.names
        assert "candidate_logits" not in parquet.schema.names
        pairs = np.empty(parquet.metadata.num_rows, dtype=np.int64)
        aliases: set[str] = set()
        offset = 0
        for batch in parquet.iter_batches(
            batch_size=100_000,
            columns=[
                "config_id", "item_index", "model_alias", "target",
                "p_hate", "p_not_hate",
            ],
        ):
            config_id = np.asarray(batch.column("config_id"), dtype=np.int64)
            item_index = np.asarray(batch.column("item_index"), dtype=np.int64)
            p_hate = np.asarray(batch.column("p_hate"), dtype=float)
            p_not_hate = np.asarray(batch.column("p_not_hate"), dtype=float)
            assert np.isfinite(p_hate).all() and np.isfinite(p_not_hate).all(), path
            assert ((p_hate >= 0) & (p_hate <= 1)).all(), path
            assert ((p_not_hate >= 0) & (p_not_hate <= 1)).all(), path
            # Historical candidate probabilities are bf16-rounded; the
            # observed maximum sum deviation is 1/512 (0.001953125).
            assert np.allclose(p_hate + p_not_hate, 1.0, atol=3e-3), path
            assert ((item_index >= 1) & (item_index <= 1_332)).all(), path
            aliases.update(batch.column("model_alias").to_pylist())
            assert all(value for value in batch.column("target").to_pylist()), path
            count = len(config_id)
            pairs[offset : offset + count] = config_id * 2_000 + item_index
            offset += count
        assert offset == parquet.metadata.num_rows
        assert aliases.issubset(PRIMARY) and len(aliases) == 1, (path, aliases)
        assert len(np.unique(pairs)) == parquet.metadata.num_rows, path
        assert len(np.unique(pairs // 2_000)) == 1_800, path

    ablation_files = sorted(
        (stage / "data" / "political_compass_chat_ablation").glob("*.parquet")
    )
    assert len(ablation_files) <= 1
    if ablation_files:
        assert pq.read_metadata(ablation_files[0]).num_rows == 111_600

    hate = pd.read_csv(stage / "data" / "hate_speech_aggregate" / "hs_configuration_scores.csv")
    assert len(hate) == 14_400
    assert hate["model"].map(lambda value: str(value).split("/")[-1]).isin(PRIMARY).all()

    topics = pd.read_csv(stage / "inputs" / "ibm_sentiment" / "ibm_test_topics.csv")
    assert len(topics) == 30
    assert topics["topicSentiment"].value_counts().to_dict() == {1: 19, -1: 11}
    print(f"Validated {len(manifest)} manifested files in {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    validate(args.stage.resolve())


if __name__ == "__main__":
    main()
