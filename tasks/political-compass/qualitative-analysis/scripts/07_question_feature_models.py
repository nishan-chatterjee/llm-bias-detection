#!/usr/bin/env python3
"""Test whether question wording/features predict family-specific difficulty."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import FEATURE_DIR, TABLE_DIR  # noqa: E402
from io_utils import write_json  # noqa: E402


METADATA = ["family", "protocol", "ideology"]
STRUCTURED = [
    "topic",
    "sensitive_topic",
    "scorer_axis",
    "scorer_weight_range",
    "agree_pole_sign",
    "statement_words",
    "statement_chars",
    "absolute_term_count",
    "negation_count",
    "modal_count",
    "concessive_count",
    "clause_marker_count",
]
TARGETS = [
    "target_alignment_rate",
    "mean_entropy",
    "mean_margin",
    "stage1_stage2_agreement_rate",
    "mean_tokens",
    "hedging_rate",
    "moral_distance_rate",
]


def make_pipeline(mode: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = list(METADATA)
    numeric = []
    transformers = []
    if mode in {"structured", "combined"}:
        categorical += ["topic", "sensitive_topic", "scorer_axis"]
        numeric += [column for column in STRUCTURED if column not in categorical]
    if mode in {"text", "combined"}:
        transformers.append(
            (
                "statement",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=5_000,
                    sublinear_tf=True,
                ),
                "statement",
            )
        )
    transformers.append(
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        )
    )
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def evaluate(frame: pd.DataFrame, target: str, mode: str) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    data = frame[frame[target].notna()].copy()
    if len(data) < 2 or data["question_id"].nunique() < 2:
        return {
            "target": target,
            "feature_set": mode,
            "rows": len(data),
            "question_folds": int(data["question_id"].nunique()),
            "mae": np.nan,
            "r2": np.nan,
            "correlation": np.nan,
            "status": "insufficient_rows_or_questions",
        }
    predictions = np.full(len(data), np.nan)
    groups = data["question_id"].to_numpy()
    splits = min(10, data["question_id"].nunique())
    for train, test in GroupKFold(n_splits=splits).split(data, groups=groups):
        pipeline = make_pipeline(mode)
        pipeline.fit(data.iloc[train], data.iloc[train][target])
        predictions[test] = pipeline.predict(data.iloc[test])
    return {
        "target": target,
        "feature_set": mode,
        "status": "ok",
        "rows": len(data),
        "question_folds": splits,
        "mae": float(mean_absolute_error(data[target], predictions)),
        "r2": float(r2_score(data[target], predictions)),
        "correlation": float(np.corrcoef(data[target], predictions)[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--difficulty",
        type=Path,
        default=TABLE_DIR / "question_difficulty_by_family.csv",
    )
    parser.add_argument(
        "--question-features",
        type=Path,
        default=FEATURE_DIR / "question_features.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=TABLE_DIR / "question_feature_models"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    difficulty = pd.read_csv(args.difficulty)
    questions = pd.read_csv(args.question_features)
    frame = difficulty.merge(
        questions.drop(columns=["topic", "sensitive_topic", "scorer_axis"]),
        on="question_id",
        how="left",
        validate="many_to_one",
    )
    results = []
    for target in TARGETS:
        if target not in frame:
            continue
        for mode in ["metadata", "structured", "text", "combined"]:
            result = evaluate(frame, target, mode)
            results.append(result)
            print(result, flush=True)
    pd.DataFrame(results).to_csv(args.output_dir / "grouped_question_cv.csv", index=False)
    write_json(
        args.output_dir / "manifest.json",
        {
            "rows": len(frame),
            "questions": int(frame["question_id"].nunique()),
            "holdout": "GroupKFold by question_id",
            "feature_sets": ["metadata", "structured", "text", "combined"],
            "interpretation": (
                "Incremental performance over metadata-only estimates whether "
                "question wording/general features predict unseen-item difficulty."
            ),
        },
    )


if __name__ == "__main__":
    main()
