#!/usr/bin/env python3
"""Build compact classification and calibration summaries from release Parquets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


GROUPS = ["model_alias", "ideology", "target"]


def summarize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["tp"] = frame["gold_hate"] & frame["predicted_hate"]
    frame["tn"] = ~frame["gold_hate"] & ~frame["predicted_hate"]
    frame["fp"] = ~frame["gold_hate"] & frame["predicted_hate"]
    frame["fn"] = frame["gold_hate"] & ~frame["predicted_hate"]
    frame["p_hate_gold"] = frame["p_hate"].where(frame["gold_hate"])
    frame["p_hate_nonhate"] = frame["p_hate"].where(~frame["gold_hate"])
    grouped = frame.groupby(GROUPS, observed=True, sort=False)
    out = grouped.agg(
        n=("p_hate", "size"),
        prevalence=("gold_hate", "mean"),
        positive_rate=("predicted_hate", "mean"),
        mean_p_hate=("p_hate", "mean"),
        mean_p_hate_gold=("p_hate_gold", "mean"),
        mean_p_hate_nonhate=("p_hate_nonhate", "mean"),
        tp=("tp", "sum"), tn=("tn", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
    ).reset_index()
    out["accuracy"] = (out["tp"] + out["tn"]) / out["n"]
    out["precision"] = out["tp"] / (out["tp"] + out["fp"]).replace(0, np.nan)
    out["recall"] = out["tp"] / (out["tp"] + out["fn"]).replace(0, np.nan)
    out["f1"] = 2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"])
    return out


def calibration_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["probability_bin"] = np.minimum((work["p_hate"] * 10).astype(int), 9)
    return (
        work.groupby(["model_alias", "probability_bin"], observed=True)
        .agg(n=("gold_hate", "size"), mean_p_hate=("p_hate", "mean"), observed_hate_rate=("gold_hate", "mean"))
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing eight long Parquets")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    summaries = []
    calibration = []
    columns = ["model_alias", "ideology", "target", "gold_hate", "predicted_hate", "p_hate"]
    for path in sorted(args.input.glob("*.parquet")):
        frame = pq.read_table(path, columns=columns).to_pandas()
        summaries.append(summarize_frame(frame))
        calibration.append(calibration_frame(frame))
        print(f"summarized {path.name}: {len(frame):,} predictions")
    if len(summaries) != 8:
        raise ValueError(f"Expected eight Parquets, found {len(summaries)}")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(summaries, ignore_index=True)
    bins = pd.concat(calibration, ignore_index=True)
    metrics.to_csv(args.output / "hs_item_metrics_by_persona_target.csv", index=False)
    bins.to_csv(args.output / "hs_calibration_by_model.csv", index=False)
    print(f"wrote {len(metrics):,} persona-target rows and {len(bins):,} calibration rows")


if __name__ == "__main__":
    main()
