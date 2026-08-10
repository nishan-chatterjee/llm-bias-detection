#!/usr/bin/env python3
"""Build a deterministic, balanced queue for rationale embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import ALL_MODELS, CORPUS_DIR, EMBEDDING_DIR, FEATURE_DIR, TABLE_DIR  # noqa: E402
from io_utils import parse_models, stable_hash, write_json  # noqa: E402


def deterministic_sample(
    frame: pd.DataFrame, groups: list[str], per_cell: int, salt: str
) -> pd.DataFrame:
    frame = frame.copy()
    frame["_sample_key"] = frame["trace_id"].map(lambda value: stable_hash(salt, value))
    return (
        frame.sort_values("_sample_key")
        .groupby(groups, observed=True, dropna=False, group_keys=False)
        .head(per_cell)
        .drop(columns="_sample_key")
    )


def load_trace_rows(model: str, corpus_dir: Path, feature_dir: Path) -> pd.DataFrame:
    corpus = pd.read_parquet(
        corpus_dir / f"{model}.parquet",
        columns=[
            "trace_id",
            "statement",
            "think_text_normalized",
            "visible_text_normalized",
            "rationale_text",
            "successful",
        ],
    )
    features = pd.read_parquet(
        feature_dir / f"{model}.parquet",
        columns=[
            "trace_id",
            "model_variant",
            "base_model",
            "family",
            "protocol",
            "size_b",
            "ideology",
            "lhs_row",
            "question_id",
            "topic",
            "sensitive_topic",
            "context_id",
            "reasoning_mode",
            "persona_class",
            "target_comparable",
            "target_aligned",
            "target_projection_expected",
            "stage1_tokens",
            "think_words",
            "visible_words",
            "classification_entropy",
        ],
    )
    return features.merge(corpus, on="trace_id", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=EMBEDDING_DIR / "queue")
    parser.add_argument("--per-cell", type=int, default=500)
    parser.add_argument("--per-think-outcome-cell", type=int, default=500)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument(
        "--representations",
        default="rationale,think",
        help="Comma-separated: rationale, visible, think, or think_plus_rationale.",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"Embedding queue already exists at {args.output_dir}; reusing it.")
        return
    models = parse_models(args.models, ALL_MODELS)
    blocks: list[pd.DataFrame] = []
    for model in models:
        frame = load_trace_rows(model, args.corpus_dir, args.feature_dir)
        frame = frame[frame["successful"] & frame["rationale_text"].str.len().gt(0)]
        selected = (
            frame
            if args.full
            else deterministic_sample(
                frame,
                ["model_variant", "ideology", "reasoning_mode"],
                args.per_cell,
                "balanced-embedding-v1",
            )
        )
        selected = selected.copy()
        selected["selection_reason"] = "full" if args.full else "balanced"
        blocks.append(selected)
        print(f"{model}: selected {len(selected):,}", flush=True)

    queue = pd.concat(blocks, ignore_index=True)
    queue["think_outcome"] = None
    pair_path = TABLE_DIR / "qwen_think_question_pairs.parquet"
    if pair_path.exists() and not args.full:
        pairs = pd.read_parquet(pair_path)
        required = {
            "trace_id_no_think",
            "trace_id_think",
            "pair_model",
            "think_outcome",
            "reasoning_mode_no_think",
        }
        if required.issubset(pairs.columns):
            candidate_pairs = pairs[
                pairs["think_outcome"].isin(
                    ["think_rescue", "think_harm", "both_aligned", "both_unaligned"]
                )
            ].copy()
            candidate_pairs["trace_id"] = candidate_pairs["trace_id_think"]
            sampled = deterministic_sample(
                candidate_pairs,
                ["pair_model", "think_outcome", "reasoning_mode_no_think"],
                args.per_think_outcome_cell,
                "think-outcome-embedding-v1",
            )
            outcome_map: dict[str, str] = {}
            selected_ids: set[str] = set()
            for row in sampled.itertuples(index=False):
                for trace_id in [str(row.trace_id), str(row.trace_id_no_think)]:
                    selected_ids.add(trace_id)
                    outcome_map[trace_id] = str(row.think_outcome)
            existing = set(queue["trace_id"])
            missing = selected_ids - existing
            extras = []
            for model in models:
                frame = load_trace_rows(model, args.corpus_dir, args.feature_dir)
                extra = frame[frame["trace_id"].isin(missing)].copy()
                if not extra.empty:
                    extra["selection_reason"] = "qwen_think_outcome"
                    extras.append(extra)
            if extras:
                queue = pd.concat([queue, *extras], ignore_index=True)
            queue["think_outcome"] = queue["trace_id"].map(outcome_map)

    queue = queue.drop_duplicates("trace_id").reset_index(drop=True)
    representations = [
        value.strip() for value in args.representations.split(",") if value.strip()
    ]
    unknown = set(representations) - {
        "rationale",
        "visible",
        "think",
        "think_plus_rationale",
    }
    if unknown:
        raise ValueError(f"Unknown representations: {sorted(unknown)}")
    represented = []
    for representation in representations:
        part = queue.copy()
        if representation == "rationale":
            part["embedding_text"] = part["rationale_text"].fillna("")
        elif representation == "visible":
            part["embedding_text"] = part["visible_text_normalized"].fillna("")
        elif representation == "think":
            part["embedding_text"] = part["think_text_normalized"].fillna("")
        else:
            part["embedding_text"] = (
                part["think_text_normalized"].fillna("")
                + "\n"
                + part["rationale_text"].fillna("")
            )
        part = part[part["embedding_text"].str.strip().str.len().gt(0)].copy()
        part["representation"] = representation
        part["embedding_id"] = part["trace_id"].map(
            lambda value: stable_hash("embedding", representation, value)
        )
        represented.append(part)
    queue = pd.concat(represented, ignore_index=True)
    queue["shard_id"] = queue["embedding_id"].map(
        lambda value: int(stable_hash("embedding-shard-v1", value), 16)
        % args.num_shards
    )
    manifests = []
    for shard_id in range(args.num_shards):
        path = args.output_dir / f"queue_{shard_id:02d}.parquet"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        part = queue[queue["shard_id"].eq(shard_id)].reset_index(drop=True)
        part.to_parquet(path, index=False)
        manifests.append({"shard_id": shard_id, "rows": len(part), "path": str(path)})
    write_json(
        manifest_path,
        {
            "models": models,
            "full": args.full,
            "per_cell": args.per_cell,
            "per_think_outcome_cell": args.per_think_outcome_cell,
            "num_shards": args.num_shards,
            "representations": representations,
            "rows": len(queue),
            "shards": manifests,
        },
    )
    print(f"Wrote {len(queue):,} unique traces across {args.num_shards} shards")


if __name__ == "__main__":
    main()
