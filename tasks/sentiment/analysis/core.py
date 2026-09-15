#!/usr/bin/env python3
"""Build annotation-free IBM topic-sentiment tables and figures.

Only fields emitted by ``run_ibm_sentiment.py`` are used.  There is no target
taxonomy and no LLM-authored annotation in this analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score


MODELS = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
]
IDEOLOGIES = [
    "base",
    "libertarian_left",
    "libertarian_right",
    "authoritarian_left",
    "authoritarian_right",
    "centrism",
]
FACTOR_COLS = ["context_id", "instr_combo", "key_type", "perm_id", "persona_combo"]
FACTOR_LABELS = {
    "context_id": "Context",
    "instr_combo": "Instruction",
    "key_type": "Key Type",
    "perm_id": "Permutation",
    "persona_combo": "Persona",
}
EXPECTED_CONFIGS_PER_MODEL = 6 * 300
EXPECTED_ITEMS = 30
EXPECTED_ROWS_PER_MODEL = EXPECTED_CONFIGS_PER_MODEL * EXPECTED_ITEMS


def iter_records(path: Path):
    if path.suffix == ".parquet":
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def compact_rows(path: Path) -> tuple[pd.DataFrame, int]:
    """Keep the last record for each item UID, matching resume semantics."""
    latest: dict[int, dict] = {}
    error_records = 0
    for record in iter_records(path):
        uid = record.get("item_uid")
        if uid is None:
            continue
        latest[int(uid)] = record

    rows = []
    for uid, record in latest.items():
        if record.get("error"):
            error_records += 1
            continue
        probs = record.get("classification_probs") or {}
        pred = record.get("classification_pred_label")
        gold = record.get("gold_label")
        if pred not in {"NEGATIVE", "POSITIVE"} or gold not in {"NEGATIVE", "POSITIVE"}:
            error_records += 1
            continue
        probability_values = [
            float(value)
            for value in probs.values()
            if value is not None and not pd.isna(value)
        ]
        rows.append(
            {
                "model": str(record["model"]),
                "ideology": str(record["ideology"]),
                "config_id": int(record["config_id"]),
                "item_uid": uid,
                "item_index": int(record["item_index"]),
                "source_item_id": str(record["source_item_id"]),
                "topic": str(record["topic"]),
                "target": str(record["target"]),
                "gold_label": gold,
                "pred_label": pred,
                "is_correct": bool(pred == gold),
                "confidence": max(probability_values, default=np.nan),
                "context_id": int(record.get("context_id", -1)),
                "instr_type": str(record["instr_type"]),
                "instr_idx": int(record["instr_idx"]),
                "persona_class": str(record.get("persona_class") or "base"),
                "persona_idx": int(record.get("persona_idx", -1)),
                "key_type": str(record["key_type"]),
                "perm_id": int(record["perm_id"]),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["instr_combo"] = frame["instr_type"] + "_idx" + frame["instr_idx"].astype(str)
        frame["persona_combo"] = np.where(
            frame["ideology"].eq("base"),
            "base",
            frame["persona_class"] + "_idx" + frame["persona_idx"].astype(str),
        )
    return frame, error_records


def metric_row(frame: pd.DataFrame) -> dict:
    gold = frame["gold_label"]
    pred = frame["pred_label"]
    return {
        "n": len(frame),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(
            gold, pred, labels=["NEGATIVE", "POSITIVE"], average="macro", zero_division=0
        ),
        "negative_f1": f1_score(
            gold, pred, labels=["NEGATIVE"], average="macro", zero_division=0
        ),
        "positive_f1": f1_score(
            gold, pred, labels=["POSITIVE"], average="macro", zero_division=0
        ),
        "mean_confidence": frame["confidence"].mean(),
        "positive_pred_rate": pred.eq("POSITIVE").mean(),
        "positive_gold_rate": gold.eq("POSITIVE").mean(),
    }


def grouped_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(groups, observed=True, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row.update(metric_row(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def between_group_ss(frame: pd.DataFrame, factor: str, target: str) -> float:
    grouped = frame.groupby(factor, dropna=False, observed=True)[target]
    means = grouped.mean()
    counts = grouped.count()
    grand = frame[target].mean()
    value = float((((means - grand) ** 2) * counts).sum())
    return max(value, 0.0) if np.isfinite(value) else 0.0


def residual_rmse(frame: pd.DataFrame, target: str) -> float:
    predictors = [
        factor for factor in FACTOR_COLS
        if frame[factor].nunique(dropna=False) >= 2
    ]
    tmp = frame[[target] + predictors].dropna(subset=[target]).copy()
    for col in predictors:
        tmp[col] = tmp[col].astype(str)
    design = pd.get_dummies(tmp[predictors], drop_first=True, dtype=float)
    design.insert(0, "_intercept_", 1.0)
    x = design.to_numpy(dtype=float)
    y = tmp[target].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    return float(np.sqrt(np.sum(residuals**2) / max(len(y) - rank, 1)))


def factor_sensitivity(config_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, subset in config_metrics.groupby("model", observed=True, sort=False):
        total_n = len(subset)
        row = {
            "model": model,
            "metric": "macro_f1",
            "Ideology SD": subset.groupby("ideology", observed=True)["macro_f1"].mean().std(),
            "Total Var SD": subset["macro_f1"].std(),
            "Residual RMSE": residual_rmse(subset, "macro_f1"),
        }
        for factor in FACTOR_COLS:
            ss = sum(
                between_group_ss(ideology_rows, factor, "macro_f1")
                for _, ideology_rows in subset.groupby("ideology", observed=True)
                if ideology_rows[factor].nunique(dropna=False) >= 2
            )
            row[FACTOR_LABELS[factor]] = np.sqrt(ss / total_n)
        rows.append(row)
    return pd.DataFrame(rows)


def save_figures(summary_persona: pd.DataFrame, sensitivity: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    heat = (
        summary_persona.pivot(index="model", columns="ideology", values="macro_f1")
        .reindex(index=MODELS, columns=IDEOLOGIES)
    )
    plt.figure(figsize=(11, 5.5))
    sns.heatmap(heat, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="YlGnBu")
    plt.title("IBM topic sentiment: macro-F1 by model and persona")
    plt.xlabel("Assigned persona")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output / "ibm_sentiment_macro_f1_model_persona_heatmap.png", dpi=180)
    plt.close()

    ordered = sensitivity.set_index("model").reindex(MODELS)
    columns = [
        "Ideology SD", "Total Var SD", "Residual RMSE",
        "Context", "Instruction", "Key Type", "Permutation", "Persona",
    ]
    plt.figure(figsize=(11, 5.5))
    sns.heatmap(ordered[columns], cmap="magma", annot=True, fmt=".3f")
    plt.title("IBM topic sentiment: config-level macro-F1 sensitivity")
    plt.xlabel("Descriptive variation component")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output / "ibm_sentiment_factor_sensitivity_macro_f1.png", dpi=180)
    plt.close()


def build(raw_dir: Path, tables_dir: Path, figures_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    config_parts = []
    model_summary_parts = []
    persona_summary_parts = []
    topic_parts = []

    for model in MODELS:
        jsonl_path = raw_dir / f"{model}.jsonl"
        parquet_path = raw_dir / f"{model}.parquet"
        path = jsonl_path if jsonl_path.exists() else parquet_path
        if not path.exists():
            coverage_rows.append(
                {"model": model, "successful_rows": 0, "error_rows": 0,
                 "unique_configs": 0, "unique_items": 0,
                 "expected_rows": EXPECTED_ROWS_PER_MODEL, "coverage": 0.0,
                 "path_exists": False}
            )
            continue
        frame, errors = compact_rows(path)
        coverage_rows.append(
            {
                "model": model,
                "successful_rows": len(frame),
                "error_rows": errors,
                "unique_configs": frame["config_id"].nunique(),
                "unique_items": frame["item_index"].nunique(),
                "expected_rows": EXPECTED_ROWS_PER_MODEL,
                "coverage": len(frame) / EXPECTED_ROWS_PER_MODEL,
                "path_exists": True,
            }
        )
        if frame.empty:
            continue
        config_parts.append(
            grouped_metrics(
                frame,
                ["model", "ideology", "config_id", "context_id", "instr_combo",
                 "key_type", "perm_id", "persona_combo"],
            )
        )
        model_summary_parts.append(grouped_metrics(frame, ["model"]))
        persona_summary_parts.append(grouped_metrics(frame, ["model", "ideology"]))
        topic_parts.append(
            frame.groupby(
                ["model", "ideology", "item_index", "source_item_id", "topic", "target", "gold_label"],
                observed=True,
                sort=False,
            )
            .agg(
                n=("is_correct", "size"),
                accuracy=("is_correct", "mean"),
                positive_pred_rate=("pred_label", lambda values: values.eq("POSITIVE").mean()),
                mean_confidence=("confidence", "mean"),
            )
            .reset_index()
        )

    coverage = pd.DataFrame(coverage_rows)
    config_metrics = pd.concat(config_parts, ignore_index=True)
    summary_model = pd.concat(model_summary_parts, ignore_index=True)
    summary_persona = pd.concat(persona_summary_parts, ignore_index=True)
    topic_metrics = pd.concat(topic_parts, ignore_index=True)
    sensitivity = factor_sensitivity(config_metrics)

    coverage.to_csv(tables_dir / "coverage_by_model.csv", index=False)
    config_metrics.to_csv(tables_dir / "config_level_metrics.csv", index=False)
    summary_model.to_csv(tables_dir / "summary_by_model.csv", index=False)
    summary_persona.to_csv(tables_dir / "summary_by_model_persona.csv", index=False)
    topic_metrics.to_csv(tables_dir / "topic_descriptive_metrics.csv", index=False)
    sensitivity.to_csv(tables_dir / "factor_sensitivity_macro_f1.csv", index=False)
    save_figures(summary_persona, sensitivity, figures_dir)

    print(coverage.to_string(index=False))
    print(f"Wrote {len(config_metrics):,} configuration rows and {len(topic_metrics):,} topic rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Directory containing eight runner JSONLs or eight downloaded release Parquets",
    )
    parser.add_argument("--tables", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--figures", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    build(args.raw, args.tables, args.figures)


if __name__ == "__main__":
    main()
