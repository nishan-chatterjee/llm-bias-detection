"""I/O and text helpers shared by the qualitative-analysis scripts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import IDEOLOGY_ORDER, MODEL_METADATA, qwen_base_model


THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def stable_hash(*parts: object, length: int = 16) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_text(text: object) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "")).strip()


def split_reasoning_trace(text: object) -> tuple[str, str, bool]:
    raw = str(text or "")
    matches = list(THINK_RE.finditer(raw))
    if not matches:
        opening = re.search(r"<think>", raw, flags=re.IGNORECASE)
        if opening:
            return raw[opening.end() :].strip(), raw[: opening.start()].strip(), True
        return "", raw.strip(), False
    think_text = "\n".join(match.group(1).strip() for match in matches).strip()
    visible_text = THINK_RE.sub("", raw).strip()
    return think_text, visible_text, True


def word_count(text: object) -> int:
    return len(TOKEN_RE.findall(str(text or "")))


def _candidate_to_canonical(record: dict) -> tuple[dict[str, int], dict[str, int]]:
    original = list(record.get("item_metadata", {}).get("original_choices", []))
    if not original:
        original = ["Strongly disagree", "Disagree", "Agree", "Strongly agree"]
    normalized_original = {normalize_text(label).casefold(): idx for idx, label in enumerate(original)}
    key_to_index: dict[str, int] = {}
    label_to_index: dict[str, int] = {}
    for key, label in zip(
        record.get("candidate_keys") or [],
        record.get("candidate_texts") or [],
    ):
        normalized_label = normalize_text(label).casefold()
        if normalized_label in normalized_original:
            index = normalized_original[normalized_label]
            key_to_index[str(key)] = index
            label_to_index[normalized_label] = index
    return key_to_index, label_to_index


def canonical_distribution(record: dict) -> list[float]:
    key_to_index, _ = _candidate_to_canonical(record)
    result = [float("nan")] * 4
    probs = record.get("classification_probs") or {}
    if not probs:
        return result
    result = [0.0] * 4
    seen = False
    for key, probability in probs.items():
        if str(key) in key_to_index:
            result[key_to_index[str(key)]] = float(probability)
            seen = True
    if not seen:
        return [float("nan")] * 4
    total = sum(result)
    if total > 0:
        result = [value / total for value in result]
    return result


def predicted_canonical_index(record: dict) -> float:
    key_to_index, _ = _candidate_to_canonical(record)
    key = record.get("classification_pred_key")
    if key is None or str(key) not in key_to_index:
        return float("nan")
    return float(key_to_index[str(key)])


def _last_nonempty_lines(text: str, count: int = 4) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-count:])


def explicit_stance_index(record: dict, visible_text: str) -> tuple[float, str]:
    """Extract a final explicit Likert stance conservatively.

    The parser prioritizes the last few visible lines, requires an unambiguous
    canonical label/key, and returns NaN when multiple options remain plausible.
    """
    key_to_index, label_to_index = _candidate_to_canonical(record)
    tail = _last_nonempty_lines(visible_text, count=5)
    tail_folded = tail.casefold()
    candidates: list[tuple[int, int, str]] = []

    for label, index in label_to_index.items():
        prefix_guard = r"(?<!strongly\s)" if label in {"agree", "disagree"} else ""
        for match in re.finditer(
            rf"(?<![a-z]){prefix_guard}{re.escape(label)}(?![a-z])",
            tail_folded,
        ):
            candidates.append((match.start(), index, f"label:{label}"))

    for key, index in key_to_index.items():
        # Single alphabetic keys are accepted only near stance/answer language
        # or as a line-leading/trailing answer token.
        escaped = re.escape(key)
        patterns = [
            rf"(?:stance|answer|position|choice|option)\s*[:=-]?\s*\**{escaped}\b",
            rf"(?m)^\s*\**{escaped}(?:[.)]|\s*$)",
            rf"\b{escaped}\s*[.)]\s*(?:strongly\s+)?(?:dis)?agree\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, tail, flags=re.IGNORECASE):
                candidates.append((match.start(), index, f"key:{key}"))

    if not candidates:
        return float("nan"), "none"
    last_position = max(position for position, _, _ in candidates)
    last = [candidate for candidate in candidates if candidate[0] == last_position]
    indices = {index for _, index, _ in last}
    if len(indices) != 1:
        return float("nan"), "ambiguous"
    _, index, source = last[-1]
    return float(index), source


def probability_entropy(probabilities: Iterable[float]) -> float:
    values = np.asarray(list(probabilities), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return float("nan")
    return float(-(values * np.log(values)).sum())


def probability_margin(probabilities: Iterable[float]) -> float:
    values = np.asarray(list(probabilities), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2])


def model_columns(model_variant: str) -> dict:
    family, protocol, size_b = MODEL_METADATA[model_variant]
    return {
        "family": family,
        "protocol": protocol,
        "size_b": size_b,
        "base_model": qwen_base_model(model_variant),
    }


def design_lhs_lookup(design_path: Path) -> dict[int, int]:
    design = pd.read_csv(design_path)
    design["lhs_row"] = design.groupby(
        ["model_variant", "language", "ideology"],
        sort=False,
        dropna=False,
    ).cumcount()
    return dict(
        zip(
            design["config_id"].astype(int),
            design["lhs_row"].astype(int),
        )
    )


def parse_models(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    requested = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [model for model in requested if model not in MODEL_METADATA]
    if unknown:
        raise ValueError(f"Unknown model variants: {unknown}")
    return requested


class ParquetBatchWriter:
    """Append dictionaries to a Parquet file without retaining the full corpus."""

    def __init__(self, path: Path, batch_size: int = 10_000):
        self.path = path
        self.batch_size = batch_size
        self.rows: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        self.writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> "ParquetBatchWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")
