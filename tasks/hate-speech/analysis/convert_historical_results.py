#!/usr/bin/env python3
"""Convert historical wide hate-speech CSVs to validated long Parquet files.

The historical runner appended rows for different target groups to a CSV whose
header was written by the first configuration. Every row has the same 1,332
item positions, but the ``q...`` identifiers in that header therefore belong
only to the first target-specific list. This converter deliberately exposes a
stable one-based ``item_index`` and does not mislabel those positions as global
question IDs.

The historical files contain candidate-only probabilities for literal
``True`` and ``False``. They do not contain raw vocabulary logits or source
statements, and this converter does not manufacture either field.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PRIMARY = {
    "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
    "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
}
ITEM_PATTERN = re.compile(r"^(q[^_]+)_is_hate_speech$")
ITEM_SUFFIXES = (
    "_is_hate_speech", "_target", "_meta", "_prob_ans_0", "_prob_ans_1",
)
METADATA = (
    "config_id", "model", "quantization", "language", "ideology",
    "context_id", "instr_idx", "persona_class", "persona_idx",
)


def short_model(value: str) -> str:
    return str(value).split("/")[-1]


def inspect_columns(columns: list[str]) -> list[str]:
    missing_metadata = sorted(set(METADATA).difference(columns))
    if missing_metadata:
        raise ValueError(f"Missing historical metadata columns: {missing_metadata}")
    prefixes = []
    for column in columns:
        match = ITEM_PATTERN.match(column)
        if match:
            prefixes.append(match.group(1))
    if len(prefixes) != 1_332:
        raise ValueError(f"Expected 1,332 item positions, found {len(prefixes):,}")
    expected = {prefix + suffix for prefix in prefixes for suffix in ITEM_SUFFIXES}
    missing_items = sorted(expected.difference(columns))
    if missing_items:
        raise ValueError(f"Missing {len(missing_items)} item columns; first: {missing_items[:3]}")
    return prefixes


def bool_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy()
    if values.dtype == bool:
        return values
    normalized = np.char.lower(values.astype(str))
    valid = np.isin(normalized, ["true", "false", "1", "0"])
    if not valid.all():
        bad = np.unique(normalized[~valid])[:5]
        raise ValueError(f"Unexpected hate labels: {bad.tolist()}")
    return np.isin(normalized, ["true", "1"])


def normalize_design(path: Path) -> pd.DataFrame:
    design = pd.read_csv(path)
    required = {
        "config_id", "model", "language", "target", "context_id", "instr_idx",
        "persona_class", "persona_idx", "ideology",
    }
    missing = sorted(required.difference(design.columns))
    if missing:
        raise ValueError(f"Design is missing columns: {missing}")
    design = design.copy()
    design["model_alias"] = design["model"].map(short_model)
    if set(design["model_alias"]) != PRIMARY:
        raise ValueError(f"Unexpected design models: {sorted(set(design['model_alias']) - PRIMARY)}")
    design["quantization"] = "bf16"
    design["context_id"] = pd.to_numeric(design["context_id"], errors="coerce").fillna(-1).astype(int)
    design["persona_idx"] = pd.to_numeric(design["persona_idx"], errors="coerce").fillna(-1).astype(int)
    design["persona_class"] = design["persona_class"].fillna("").astype(str)
    columns = [
        "config_id", "language", "target", "context_id", "instr_idx",
        "persona_class", "persona_idx", "model_alias", "model", "ideology",
        "quantization",
    ]
    result = design[columns].sort_values("config_id").reset_index(drop=True)
    if len(result) != 14_400 or not result["config_id"].is_unique:
        raise ValueError("Expected 14,400 unique design configurations")
    return result


def _long_chunk(chunk: pd.DataFrame, prefixes: list[str], expected: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(chunk)
    n_items = len(prefixes)
    config_ids = pd.to_numeric(chunk["config_id"], errors="raise").astype(int)
    if config_ids.duplicated().any():
        raise ValueError("Duplicate config_id within historical result chunk")
    expected_rows = expected.loc[config_ids.to_numpy()]

    aliases = chunk["model"].map(short_model).astype(str)
    if not np.array_equal(aliases.to_numpy(), expected_rows["model_alias"].to_numpy()):
        raise ValueError("Result model does not match supplied design")
    for column, absent in (("context_id", -1), ("persona_idx", -1)):
        actual = pd.to_numeric(chunk[column], errors="coerce").fillna(absent).astype(int).to_numpy()
        if not np.array_equal(actual, expected_rows[column].to_numpy()):
            raise ValueError(f"Result {column} does not match supplied design")
    for column in ("language", "ideology", "instr_idx"):
        actual = chunk[column].astype(str).to_numpy()
        wanted = expected_rows[column].astype(str).to_numpy()
        if not np.array_equal(actual, wanted):
            raise ValueError(f"Result {column} does not match supplied design")
    actual_persona = chunk["persona_class"].fillna("").astype(str).to_numpy()
    wanted_persona = expected_rows["persona_class"].fillna("").astype(str).to_numpy()
    if not np.array_equal(actual_persona, wanted_persona):
        raise ValueError("Result persona_class does not match supplied design")

    label_columns = [prefix + "_is_hate_speech" for prefix in prefixes]
    target_columns = [prefix + "_target" for prefix in prefixes]
    meta_columns = [prefix + "_meta" for prefix in prefixes]
    true_columns = [prefix + "_prob_ans_0" for prefix in prefixes]
    false_columns = [prefix + "_prob_ans_1" for prefix in prefixes]

    gold = bool_matrix(chunk[label_columns])
    targets = chunk[target_columns].astype(str).to_numpy()
    source_meta = chunk[meta_columns].fillna("").astype(str).to_numpy()
    p_hate = chunk[true_columns].to_numpy(dtype=float)
    p_not_hate = chunk[false_columns].to_numpy(dtype=float)
    if not np.isfinite(p_hate).all() or not np.isfinite(p_not_hate).all():
        raise ValueError("Non-finite candidate probabilities")
    if ((p_hate < 0) | (p_hate > 1) | (p_not_hate < 0) | (p_not_hate > 1)).any():
        raise ValueError("Candidate probability outside [0, 1]")
    # The historical softmax was computed/stored at bf16 precision. Observed
    # sums differ from one by at most 1/512 (0.001953125).
    if not np.allclose(p_hate + p_not_hate, 1.0, atol=3e-3):
        raise ValueError("True/False candidate probabilities do not sum to one")
    expected_targets = expected_rows["target"].astype(str).to_numpy()
    if not np.all(targets == expected_targets[:, None]):
        raise ValueError("Embedded item target does not match supplied design")

    repeat = lambda values: np.repeat(np.asarray(values), n_items)
    persona_class = chunk["persona_class"].fillna("").astype(str)
    persona_idx = pd.to_numeric(chunk["persona_idx"], errors="coerce").fillna(-1).astype(int)
    context_id = pd.to_numeric(chunk["context_id"], errors="coerce").fillna(-1).astype(int)
    return pd.DataFrame(
        {
            "config_id": repeat(config_ids),
            "model": repeat(chunk["model"].astype(str)),
            "model_alias": repeat(aliases),
            "quantization": repeat(chunk["quantization"].astype(str)),
            "language": repeat(chunk["language"].astype(str)),
            "ideology": repeat(chunk["ideology"].astype(str)),
            "target": targets.reshape(n_rows * n_items),
            "context_id": repeat(context_id),
            "instr_idx": repeat(pd.to_numeric(chunk["instr_idx"], errors="raise").astype(int)),
            "persona_class": repeat(persona_class),
            "persona_idx": repeat(persona_idx),
            "item_index": np.tile(np.arange(1, n_items + 1, dtype=np.int16), n_rows),
            "gold_hate": gold.reshape(n_rows * n_items),
            "source_meta": source_meta.reshape(n_rows * n_items),
            "p_hate": p_hate.reshape(n_rows * n_items),
            "p_not_hate": p_not_hate.reshape(n_rows * n_items),
            "predicted_hate": (p_hate >= p_not_hate).reshape(n_rows * n_items),
        }
    )


def convert_file(source: Path, target: Path, design: pd.DataFrame, chunk_rows: int = 25) -> dict:
    header = pd.read_csv(source, nrows=0).columns.tolist()
    prefixes = inspect_columns(header)
    expected = design.set_index("config_id", drop=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    config_ids: list[int] = []
    row_count = 0
    try:
        for chunk in pd.read_csv(source, chunksize=chunk_rows, low_memory=False):
            long = _long_chunk(chunk, prefixes, expected)
            config_ids.extend(pd.to_numeric(chunk["config_id"], errors="raise").astype(int).tolist())
            table = pa.Table.from_pandas(long, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    target, table.schema, compression="zstd", compression_level=5
                )
            writer.write_table(table)
            row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if len(config_ids) != 1_800 or len(set(config_ids)) != 1_800:
        raise ValueError(f"{source.name}: expected 1,800 unique configurations")
    if row_count != 1_800 * 1_332:
        raise ValueError(f"{source.name}: unexpected converted row count {row_count:,}")
    model_aliases = set(design.loc[design["config_id"].isin(config_ids), "model_alias"])
    if len(model_aliases) != 1:
        raise ValueError(f"{source.name}: design resolves to {sorted(model_aliases)}")
    return {
        "source": source.name,
        "output": target.name,
        "model_alias": next(iter(model_aliases)),
        "configurations": len(config_ids),
        "items_per_configuration": len(prefixes),
        "rows": row_count,
        "contains_source_text": False,
        "contains_raw_logits": False,
        "candidate_order": ["True", "False"],
    }


def convert_directory(
    raw_dir: Path,
    output_dir: Path,
    design_path: Path,
    normalized_design_path: Path | None = None,
    manifest_path: Path | None = None,
    chunk_rows: int = 25,
) -> list[dict]:
    design = normalize_design(design_path)
    if normalized_design_path is not None:
        normalized_design_path.parent.mkdir(parents=True, exist_ok=True)
        design.to_csv(normalized_design_path, index=False)
    files = sorted(raw_dir.glob("*.csv"))
    if len(files) != 8:
        raise ValueError(f"Expected eight historical result CSVs, found {len(files)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in files:
        model = short_model(pd.read_csv(source, usecols=["model"], nrows=1)["model"].iloc[0])
        if model not in PRIMARY:
            raise ValueError(f"Unsupported model in {source}: {model}")
        target = output_dir / f"{model}.parquet"
        print(f"Converting {source.name} -> {target.name}", flush=True)
        records.append(convert_file(source, target, design, chunk_rows=chunk_rows))
    if {record["model_alias"] for record in records} != PRIMARY:
        raise ValueError("Converted result set does not contain exactly the eight primary models")
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Directory containing eight wide CSVs")
    parser.add_argument("--design", type=Path, required=True, help="Supplied experimental_design_hs.csv")
    parser.add_argument("--output", type=Path, required=True, help="Long-Parquet output directory")
    parser.add_argument("--normalized-design", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=25)
    args = parser.parse_args()
    records = convert_directory(
        args.raw.resolve(),
        args.output.resolve(),
        args.design.resolve(),
        args.normalized_design.resolve() if args.normalized_design else None,
        args.manifest.resolve() if args.manifest else None,
        args.chunk_rows,
    )
    print(f"Converted {sum(record['rows'] for record in records):,} item predictions")


if __name__ == "__main__":
    main()
