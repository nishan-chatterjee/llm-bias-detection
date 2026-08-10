#!/usr/bin/env python3
"""Leakage-controlled quadrant phrase discovery and held-out confirmation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import fisher_exact


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import (  # noqa: E402
    ALL_MODELS,
    CORPUS_DIR,
    FEATURE_DIR,
    QUADRANT_IDEOLOGIES,
    TABLE_DIR,
    ensure_artifact_directories,
)
from feature_definitions import TOKEN_RE  # noqa: E402
from io_utils import model_columns, parse_models, stable_hash, write_json  # noqa: E402


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "may",
    "might",
    "must",
    "not",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "stance",
    "strongly",
    "agree",
    "disagree",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "which",
    "while",
    "with",
    "would",
    "answer",
    "option",
    "perspective",
    "viewpoint",
    "ideology",
    "libertarian",
    "authoritarian",
    "left",
    "right",
}


def bh_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    n = len(values)
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = n - reverse_rank + 1
        running = min(running, values[index] * n / rank)
        adjusted[index] = running
    return adjusted.clip(0, 1).tolist()


def question_split(question_id: int) -> str:
    return "discovery" if int(stable_hash(question_id, length=2), 16) % 2 == 0 else "confirmation"


def document_ngrams(
    text: str,
    statement: str,
    persona_words: set[str],
    max_n: int,
) -> set[str]:
    statement_words = {
        token.casefold() for token in TOKEN_RE.findall(statement) if len(token) > 2
    }
    tokens = []
    for token in TOKEN_RE.findall(text):
        token = token.casefold()
        if len(token) < 2:
            continue
        if token in statement_words or token in persona_words:
            tokens.append("<redacted>")
        elif token in {"0", "1", "2", "3", "4", "a", "b", "c", "d"}:
            tokens.append("<answer>")
        else:
            tokens.append(token)
    output: set[str] = set()
    for n in range(1, max_n + 1):
        for start in range(0, len(tokens) - n + 1):
            parts = tokens[start : start + n]
            if "<redacted>" in parts or "<answer>" in parts:
                continue
            if all(part in STOPWORDS for part in parts):
                continue
            phrase = " ".join(parts)
            output.add(phrase)
    return output


def log_odds(
    target_count: int,
    target_total: int,
    other_count: int,
    other_total: int,
    alpha: float = 0.5,
) -> float:
    target_absent = max(0, target_total - target_count)
    other_absent = max(0, other_total - other_count)
    return math.log((target_count + alpha) / (target_absent + alpha)) - math.log(
        (other_count + alpha) / (other_absent + alpha)
    )


def collect_counts(
    models: list[str],
    corpus_dir: Path,
    persona_vocabulary: dict[str, set[str]],
    max_n: int,
    max_docs_per_model: int | None,
) -> tuple[dict, dict]:
    counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    totals: dict[tuple[str, str, str], int] = defaultdict(int)
    columns = [
        "model_variant",
        "ideology",
        "question_id",
        "statement",
        "persona_class",
        "rationale_text",
    ]
    for model in models:
        metadata = model_columns(model)
        scopes = [
            "all",
            f"model::{model}",
            f"family::{metadata['family']}::{metadata['protocol']}",
        ]
        seen = 0
        parquet = pq.ParquetFile(corpus_dir / f"{model}.parquet")
        for batch in parquet.iter_batches(batch_size=5_000, columns=columns):
            for row in batch.to_pylist():
                if row["ideology"] not in QUADRANT_IDEOLOGIES:
                    continue
                # Primary phrase analysis uses short personas to minimize supplied vocabulary.
                if row["persona_class"] != "short":
                    continue
                split = question_split(int(row["question_id"]))
                ideology = str(row["ideology"])
                phrases = document_ngrams(
                    str(row["rationale_text"] or ""),
                    str(row["statement"] or ""),
                    persona_vocabulary.get(ideology, set()),
                    max_n,
                )
                for scope in scopes:
                    totals[(scope, split, ideology)] += 1
                    counts[(scope, split, ideology)].update(phrases)
                seen += 1
                if max_docs_per_model is not None and seen >= max_docs_per_model:
                    break
            if max_docs_per_model is not None and seen >= max_docs_per_model:
                break
        print(f"Phrase scan {model}: {seen:,} short-persona documents", flush=True)
    return counts, totals


def discover_and_confirm(
    counts: dict,
    totals: dict,
    min_discovery_df: int,
    min_confirmation_df: int,
    top_k: int,
) -> pd.DataFrame:
    scopes = sorted({key[0] for key in totals})
    rows = []
    for scope in scopes:
        for ideology in QUADRANT_IDEOLOGIES:
            target_discovery = counts[(scope, "discovery", ideology)]
            other_discovery = Counter()
            target_total_discovery = totals[(scope, "discovery", ideology)]
            other_total_discovery = 0
            for other in QUADRANT_IDEOLOGIES:
                if other == ideology:
                    continue
                other_discovery.update(counts[(scope, "discovery", other)])
                other_total_discovery += totals[(scope, "discovery", other)]

            candidates = []
            vocabulary = set(target_discovery) | set(other_discovery)
            for phrase in vocabulary:
                total_df = target_discovery[phrase] + other_discovery[phrase]
                if total_df < min_discovery_df:
                    continue
                score = log_odds(
                    target_discovery[phrase],
                    target_total_discovery,
                    other_discovery[phrase],
                    other_total_discovery,
                )
                candidates.append((score, phrase))
            candidates.sort(reverse=True)

            for discovery_score, phrase in candidates[:top_k]:
                target_count = counts[(scope, "confirmation", ideology)][phrase]
                target_total = totals[(scope, "confirmation", ideology)]
                other_count = 0
                other_total = 0
                for other in QUADRANT_IDEOLOGIES:
                    if other == ideology:
                        continue
                    other_count += counts[(scope, "confirmation", other)][phrase]
                    other_total += totals[(scope, "confirmation", other)]
                if target_count + other_count < min_confirmation_df:
                    continue
                odds_ratio, p_value = fisher_exact(
                    [
                        [target_count, max(0, target_total - target_count)],
                        [other_count, max(0, other_total - other_count)],
                    ],
                    alternative="greater",
                )
                confirmation_score = log_odds(
                    target_count,
                    target_total,
                    other_count,
                    other_total,
                )
                rows.append(
                    {
                        "scope": scope,
                        "ideology": ideology,
                        "phrase": phrase,
                        "ngram_n": len(phrase.split()),
                        "discovery_log_odds": discovery_score,
                        "confirmation_log_odds": confirmation_score,
                        "same_direction": confirmation_score > 0,
                        "target_confirmation_df": target_count,
                        "other_confirmation_df": other_count,
                        "target_confirmation_rate": target_count / max(1, target_total),
                        "other_confirmation_rate": other_count / max(1, other_total),
                        "odds_ratio": odds_ratio,
                        "p_value": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["q_value_bh"] = np.nan
    for _, indices in result.groupby(["scope"], observed=True).groups.items():
        result.loc[indices, "q_value_bh"] = bh_adjust(
            result.loc[indices, "p_value"].tolist()
        )
    result["confirmed_q05"] = (
        result["same_direction"]
        & result["q_value_bh"].le(0.05)
        & result["target_confirmation_df"].ge(min_confirmation_df)
    )
    return result.sort_values(
        ["scope", "ideology", "confirmed_q05", "confirmation_log_odds"],
        ascending=[True, True, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None)
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--min-discovery-df", type=int, default=40)
    parser.add_argument("--min-confirmation-df", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--max-docs-per-model", type=int, default=None)
    args = parser.parse_args()

    ensure_artifact_directories()
    models = parse_models(args.models, ALL_MODELS)
    vocabulary_path = args.feature_dir / "persona_vocabulary.json"
    vocabulary = {
        key: set(value)
        for key, value in json.loads(
            vocabulary_path.read_text(encoding="utf-8")
        ).items()
    }
    counts, totals = collect_counts(
        models,
        args.corpus_dir,
        vocabulary,
        args.max_ngram,
        args.max_docs_per_model,
    )
    results = discover_and_confirm(
        counts,
        totals,
        args.min_discovery_df,
        args.min_confirmation_df,
        args.top_k,
    )
    path = args.output_dir / "quadrant_phrase_discovery_confirmation.csv"
    results.to_csv(path, index=False)
    totals_rows = [
        {"scope": scope, "split": split, "ideology": ideology, "n": n}
        for (scope, split, ideology), n in totals.items()
    ]
    pd.DataFrame(totals_rows).to_csv(
        args.output_dir / "quadrant_phrase_population.csv", index=False
    )
    write_json(
        args.output_dir / "phrase_analysis_manifest.json",
        {
            "models": models,
            "short_persona_only": True,
            "statement_and_persona_vocabulary_redacted": True,
            "max_ngram": args.max_ngram,
            "candidate_rows": int(len(results)),
            "confirmed_rows": int(results.get("confirmed_q05", pd.Series(dtype=bool)).sum()),
        },
    )
    print(f"Wrote {path}: {len(results):,} candidate confirmations")


if __name__ == "__main__":
    main()
