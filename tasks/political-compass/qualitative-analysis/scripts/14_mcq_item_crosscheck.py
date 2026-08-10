#!/usr/bin/env python3
"""Compare matched MCQ and chat answers at the individual-question level.

The existing MCQ/chat notebook compares the final two-dimensional compass
coordinates.  This script adds a more local diagnostic: for the same model and
prompt-design cell, does MCQ assign probability to the same four answer options
as the chat answer classifier?

Rows are averaged within the exact key used by the MCQ/chat notebook before
comparison.  Consequently this is a protocol-agreement diagnostic, not a new
independent model run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
QA_ROOT = HERE.parent.parent
PC_ROOT = QA_ROOT.parent
MCQ_DIR = PC_ROOT / "output" / "mcq-v1-300" / "results"
CHAT_DIR = PC_ROOT / "output" / "chat-v2-300"
OUT_DIR = QA_ROOT / "artifacts" / "tables" / "mcq_crosscheck"

PRIMARY_MODELS = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
]
KEY = [
    "base_model",
    "ideology",
    "context_combo",
    "instr_combo",
    "persona_combo",
    "key_type",
    "perm_id",
    "question_id",
]
PROB = [f"p{i}" for i in range(4)]


def strip_qwen_mode(model: str) -> str:
    for suffix in ("_no_think", "_think"):
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def method_for_variant(model: str) -> str:
    if model.endswith("_no_think"):
        return "Standard Chat"
    if model.endswith("_think"):
        return "Chat Think"
    return "Standard Chat"


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["context_id"] = pd.to_numeric(frame["context_id"], errors="coerce")
    frame["instr_idx"] = pd.to_numeric(frame["instr_idx"], errors="coerce")
    frame["persona_idx"] = pd.to_numeric(frame["persona_idx"], errors="coerce")
    frame["perm_id"] = pd.to_numeric(frame["perm_id"], errors="coerce").astype("Int64")
    frame["context_combo"] = frame["context_id"].map(
        lambda value: "None"
        if pd.isna(value) or int(value) < 0
        else f"Ctx_{int(value)}"
    )
    frame["instr_combo"] = (
        frame["instr_type"].fillna("None").astype(str)
        + "_idx"
        + frame["instr_idx"].fillna(-1).astype(int).astype(str)
    )

    def persona(row: pd.Series) -> str:
        name = str(row.get("persona_class", "") or "").strip()
        idx = row.get("persona_idx")
        if not name or pd.isna(idx) or int(idx) < 0:
            return "None"
        return f"{name}_idx{int(idx)}"

    frame["persona_combo"] = frame.apply(persona, axis=1)
    return frame


def scalar_keys(row: dict) -> dict:
    context_id = row.get("context_id")
    instr_idx = row.get("instr_idx")
    persona_idx = row.get("persona_idx")
    persona_class = str(row.get("persona_class", "") or "").strip()
    context_combo = (
        "None"
        if context_id is None or int(context_id) < 0
        else f"Ctx_{int(context_id)}"
    )
    instr_combo = (
        f"{str(row.get('instr_type') or 'None')}_idx"
        f"{-1 if instr_idx is None else int(instr_idx)}"
    )
    persona_combo = (
        "None"
        if not persona_class or persona_idx is None or int(persona_idx) < 0
        else f"{persona_class}_idx{int(persona_idx)}"
    )
    return {
        "ideology": row["ideology"],
        "context_combo": context_combo,
        "instr_combo": instr_combo,
        "persona_combo": persona_combo,
        "key_type": row["key_type"],
        "perm_id": int(row["perm_id"]),
    }


def canonicalize(probabilities: np.ndarray, perm_id: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if int(perm_id) in (2, 3):
        values = values[::-1]
    total = np.nansum(values)
    if not np.isfinite(total) or total <= 0:
        return np.full(4, np.nan)
    return values / total


def load_mcq() -> pd.DataFrame:
    records: list[dict] = []
    for model in PRIMARY_MODELS:
        path = MCQ_DIR / f"results_{model}_bf16.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["language"].eq("english")].copy()
        frame["base_model"] = model
        frame = add_keys(frame)
        for row in frame.itertuples(index=False):
            for question_id in range(62):
                raw = [
                    getattr(row, f"q{question_id}_prob_ans{answer}")
                    for answer in range(4)
                ]
                probs = canonicalize(raw, row.perm_id)
                records.append(
                    {
                        **{column: getattr(row, column) for column in KEY[:-1]},
                        "question_id": question_id,
                        "method": "MCQ",
                        **{column: probs[i] for i, column in enumerate(PROB)},
                    }
                )
    return pd.DataFrame.from_records(records)


def load_chat() -> pd.DataFrame:
    records: dict[tuple[str, str], dict] = {}
    variants = [
        model
        for base in PRIMARY_MODELS
        for model in (
            [base]
            if base.startswith("gemma")
            else [f"{base}_no_think", f"{base}_think"]
        )
    ]
    for variant in variants:
        path = CHAT_DIR / f"{variant}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                item_id = row.get("item_id")
                if item_id:
                    records[(variant, item_id)] = row

    output: list[dict] = []
    for (variant, _), row in records.items():
        if row.get("error") is not None:
            continue
        keys = row.get("candidate_keys") or []
        probability_map = row.get("classification_probs") or {}
        if len(keys) != 4:
            continue
        raw = [probability_map.get(key, np.nan) for key in keys]
        probs = canonicalize(raw, int(row["perm_id"]))
        meta = scalar_keys(row)
        output.append(
            {
                "base_model": strip_qwen_mode(variant),
                **meta,
                "question_id": int(row["question_id"]),
                "method": method_for_variant(variant),
                **{column: probs[i] for i, column in enumerate(PROB)},
            }
        )
    return pd.DataFrame.from_records(output)


def summarize_pairs(mcq: pd.DataFrame, chat: pd.DataFrame) -> pd.DataFrame:
    mcq_mean = mcq.groupby(KEY, dropna=False)[PROB].mean().reset_index()
    chat_mean = (
        chat.groupby(["method", *KEY], dropna=False)[PROB].mean().reset_index()
    )
    pairs = chat_mean.merge(mcq_mean, on=KEY, suffixes=("_chat", "_mcq"))

    chat_probs = pairs[[f"{p}_chat" for p in PROB]].to_numpy()
    mcq_probs = pairs[[f"{p}_mcq" for p in PROB]].to_numpy()
    pairs["chat_choice"] = chat_probs.argmax(axis=1)
    pairs["mcq_choice"] = mcq_probs.argmax(axis=1)
    pairs["four_way_agree"] = pairs["chat_choice"].eq(pairs["mcq_choice"])
    pairs["chat_binary"] = pairs["chat_choice"].ge(2).astype(int)
    pairs["mcq_binary"] = pairs["mcq_choice"].ge(2).astype(int)
    pairs["binary_stance_agree"] = pairs["chat_binary"].eq(pairs["mcq_binary"])
    pairs["total_variation"] = 0.5 * np.abs(chat_probs - mcq_probs).sum(axis=1)
    ordinal = np.arange(4, dtype=float)
    pairs["expected_answer_chat"] = chat_probs @ ordinal
    pairs["expected_answer_mcq"] = mcq_probs @ ordinal
    pairs["expected_answer_delta"] = (
        pairs["expected_answer_chat"] - pairs["expected_answer_mcq"]
    )
    return pairs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mcq = load_mcq()
    chat = load_chat()
    pairs = summarize_pairs(mcq, chat)
    pairs.to_parquet(OUT_DIR / "matched_question_pairs.parquet", index=False)

    group_columns = [
        "method",
        "base_model",
        "ideology",
    ]
    summaries: list[pd.DataFrame] = []
    for grouping in [
        ["method"],
        ["method", "base_model"],
        ["method", "ideology"],
        group_columns,
    ]:
        summary = (
            pairs.groupby(grouping, observed=True)
            .agg(
                question_pairs=("question_id", "size"),
                four_way_agreement=("four_way_agree", "mean"),
                binary_stance_agreement=("binary_stance_agree", "mean"),
                mean_total_variation=("total_variation", "mean"),
                median_total_variation=("total_variation", "median"),
                mean_expected_answer_delta=("expected_answer_delta", "mean"),
            )
            .reset_index()
        )
        summary["scope"] = "+".join(grouping)
        summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True, sort=False)
    result.to_csv(OUT_DIR / "matched_question_summary.csv", index=False)

    question = (
        pairs.groupby(["method", "question_id"], observed=True)
        .agg(
            question_pairs=("question_id", "size"),
            four_way_agreement=("four_way_agree", "mean"),
            binary_stance_agreement=("binary_stance_agree", "mean"),
            mean_total_variation=("total_variation", "mean"),
            mean_abs_expected_answer_delta=(
                "expected_answer_delta",
                lambda values: values.abs().mean(),
            ),
        )
        .reset_index()
    )
    question.to_csv(OUT_DIR / "matched_question_difficulty.csv", index=False)

    print(f"MCQ item rows: {len(mcq):,}")
    print(f"Chat item rows: {len(chat):,}")
    print(f"Matched method/config/question pairs: {len(pairs):,}")
    print(
        result[result["scope"].eq("method")][
            [
                "method",
                "question_pairs",
                "four_way_agreement",
                "binary_stance_agreement",
                "mean_total_variation",
                "mean_expected_answer_delta",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
