#!/usr/bin/env python3
"""Collapse raw hate-speech outputs to one row per prompt configuration."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_ALIASES = {
    "gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it",
    "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
}
ABSENT = "absent"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "hs_configuration_scores.csv"


def short_model(value: str) -> str:
    return re.sub(r"^.*/", "", value)


def historical_csv(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        if frame.empty or "model" not in frame:
            continue
        model = str(frame["model"].iloc[0])
        if short_model(model) not in FINAL_ALIASES:
            continue
        probability_columns = [col for col in frame if col.endswith("_prob_ans_0")]
        if not probability_columns:
            raise ValueError(f"No *_prob_ans_0 columns in {path}")
        prefixes = [col[: -len("_prob_ans_0")] for col in probability_columns]
        probabilities = frame[probability_columns].to_numpy(float)
        target = frame[prefixes[0] + "_target"].fillna("None").astype(str).values
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "ideology": frame["ideology"].fillna(ABSENT).astype(str),
                    "target": target,
                    "context_combo": [
                        ABSENT if pd.isna(value) or value == -1 else f"Ctx_{int(value)}"
                        for value in frame["context_id"]
                    ],
                    "instr_combo": frame["instr_idx"].fillna(-1).astype(int).astype(str),
                    "persona_combo": [
                        ABSENT if pd.isna(kind) or kind == "" else f"{kind}_idx{int(index)}"
                        for kind, index in zip(frame["persona_class"], frame["persona_idx"])
                    ],
                    "mean_p_hate": np.nanmean(probabilities, axis=1),
                    "mean_item_dispersion": np.nanmean(
                        probabilities * (1.0 - probabilities), axis=1
                    ),
                }
            )
        )
    if not rows:
        raise ValueError(f"No supported historical result CSVs in {raw_dir}")
    return pd.concat(rows, ignore_index=True)


def release_jsonl(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("error"):
                    continue
                rows.append(
                    {
                        "config_id": int(record["config_id"]),
                        "question_id": int(record["question_id"]),
                        "model": str(record["model"]),
                        "ideology": str(record["ideology"]),
                        "target": str(record["target"]),
                        "context_combo": ABSENT
                        if int(record["context_id"]) == -1
                        else f"Ctx_{int(record['context_id'])}",
                        "instr_combo": str(int(record["instr_idx"])),
                        "persona_combo": ABSENT
                        if not record.get("persona_class")
                        else f"{record['persona_class']}_idx{int(record['persona_idx'])}",
                        "p_hate": float(record["classification_probs"]["True"]),
                    }
                )
    if not rows:
        raise ValueError(f"No successful release JSONL records in {raw_dir}")
    item = pd.DataFrame(rows).drop_duplicates(
        ["model", "config_id", "question_id"], keep="last"
    )
    groups = [
        "model", "ideology", "target", "context_combo", "instr_combo", "persona_combo",
        "config_id",
    ]
    return (
        item.groupby(groups, observed=True, sort=False)
        .agg(
            mean_p_hate=("p_hate", "mean"),
            mean_item_dispersion=("p_hate", lambda values: np.mean(values * (1.0 - values))),
        )
        .reset_index()
        .drop(columns="config_id")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--format", choices=["historical-csv", "release-jsonl"], required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = historical_csv(args.raw) if args.format == "historical-csv" else release_jsonl(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} configurations to {args.output}")


if __name__ == "__main__":
    main()
