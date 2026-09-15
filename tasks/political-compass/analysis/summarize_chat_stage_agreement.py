#!/usr/bin/env python3
"""Summarize explicit Stage-1 versus Stage-2 agreement from chat Parquets."""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
QUALITATIVE = HERE.parent / "qualitative-analysis"
sys.path.insert(0, str(QUALITATIVE))

from io_utils import (  # noqa: E402
    explicit_stance_index,
    predicted_canonical_index,
    split_reasoning_trace,
)


COLUMNS = [
    "candidate_keys",
    "candidate_texts",
    "stage1_text",
    "classification_pred_key",
    "item_metadata",
    "error",
]


def summarize_file(path_string: str) -> dict:
    path = Path(path_string)
    total = comparable = agreements = errors = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=COLUMNS, batch_size=10_000):
        for record in batch.to_pylist():
            total += 1
            if record.get("error"):
                errors += 1
                continue
            _, visible, _ = split_reasoning_trace(record.get("stage1_text"))
            explicit, _ = explicit_stance_index(record, visible)
            predicted = predicted_canonical_index(record)
            if math.isfinite(explicit) and math.isfinite(predicted):
                comparable += 1
                agreements += int(explicit == predicted)
    return {
        "model_variant": path.stem,
        "rows": total,
        "error_rows": errors,
        "comparable_rows": comparable,
        "agreement_rows": agreements,
        "comparable_rate": comparable / total,
        "conditional_agreement_rate": agreements / comparable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "data" / "chat_stage_agreement_summary.csv",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    files = sorted(args.input.expanduser().resolve().glob("*.parquet"))
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 primary chat Parquets under {args.input}")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(summarize_file, map(str, files)))
    frame = pd.DataFrame(rows).sort_values("model_variant")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    total_comparable = int(frame["comparable_rows"].sum())
    total_agreements = int(frame["agreement_rows"].sum())
    print(frame.to_string(index=False))
    print(
        f"Overall conditional agreement: {total_agreements / total_comparable:.6f} "
        f"({total_agreements:,}/{total_comparable:,})"
    )
    print(args.output)


if __name__ == "__main__":
    main()
