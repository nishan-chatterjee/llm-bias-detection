#!/usr/bin/env python3
"""Analyze Gemma-3-27B normal versus norm-preserving abliterated traces.

The primary estimand is the exact one-to-one 27B contrast. Pooled normal-Gemma
and all-non-abliterated comparisons are descriptive benchmarks because they
also change model family, size, or generation protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import ALL_MODELS, FEATURE_DIR, TABLE_DIR  # noqa: E402
from io_utils import write_json  # noqa: E402


ABLITERATED = "gemma-3-27b-it-abliterated-normpreserve-v1"
STANDARD_27B = "gemma-3-27b-it"
STANDARD_GEMMA = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
]
PAIR_KEYS = ["ideology", "lhs_row", "question_id"]

ITEM_METRICS = {
    "stage1_tokens": "continuous",
    "visible_words": "continuous",
    "classification_entropy": "continuous",
    "classification_margin": "continuous",
    "explicit_stance_found": "binary",
    "stage1_stage2_agree": "binary",
    "target_aligned": "binary",
    "target_projection_expected": "continuous",
    "visible_hedging_present": "binary",
    "visible_balance_present": "binary",
    "visible_actual_refusal_present": "binary",
    "visible_moral_distance_present": "binary",
    "visible_persona_meta_present": "binary",
    "visible_self_correction_present": "binary",
    "visible_counterargument_present": "binary",
    "persona_vocab_overlap_fraction": "continuous",
}

CONFIG_METRICS = [
    "economic",
    "social",
    "mean_stage1_tokens",
    "mean_visible_words",
    "target_alignment_rate",
    "mean_target_projection",
    "stage1_stage2_agreement",
    "mean_entropy",
    "target_quadrant_correct",
    "target_axis_projection",
    "distance_from_origin",
]

QUESTION_FEATURES = [
    "statement_words",
    "statement_chars",
    "absolute_term_count",
    "negation_count",
    "modal_count",
    "concessive_count",
    "clause_marker_count",
    "scorer_weight_range",
    "sensitive_topic",
]


def bh_adjust(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment, preserving missing values and row order."""
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().astype(float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(
        (ordered.to_numpy() * len(ordered) / ranks)[::-1]
    )[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def load_features(path: Path) -> pd.DataFrame:
    metadata = [
        "trace_id",
        "model_variant",
        "family",
        "protocol",
        "size_b",
        *PAIR_KEYS,
        "reasoning_mode",
        "context_id",
        "persona_class",
        "instr_type",
        "topic",
        "sensitive_topic",
        "scorer_axis",
        "target_comparable",
        "stage1_stage2_comparable",
    ]
    return pd.read_parquet(path, columns=metadata + list(ITEM_METRICS))


def validate_pair_metadata(pairs: pd.DataFrame) -> None:
    for column in ["reasoning_mode", "context_id", "persona_class", "instr_type"]:
        left = pairs[f"{column}_normal"].fillna("<NA>").astype(str)
        right = pairs[f"{column}_abliterated"].fillna("<NA>").astype(str)
        mismatches = int(left.ne(right).sum())
        if mismatches:
            raise ValueError(f"{column} differs in {mismatches:,} exact pairs")


def exact_pairs(feature_dir: Path) -> pd.DataFrame:
    normal = load_features(feature_dir / f"{STANDARD_27B}.parquet")
    ablated = load_features(feature_dir / f"{ABLITERATED}.parquet")
    if normal.duplicated(PAIR_KEYS).any() or ablated.duplicated(PAIR_KEYS).any():
        raise ValueError("Gemma exact-pair keys are not unique")
    pairs = normal.merge(
        ablated,
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_normal", "_abliterated"),
        validate="one_to_one",
    )
    if len(pairs) != len(normal) or len(pairs) != len(ablated):
        raise ValueError(
            f"Incomplete exact match: {len(pairs):,}/{len(normal):,}/{len(ablated):,}"
        )
    validate_pair_metadata(pairs)
    for metric in ITEM_METRICS:
        pairs[f"delta__{metric}"] = (
            pd.to_numeric(
                pairs[f"{metric}_abliterated"], errors="coerce"
            ).astype(float)
            - pd.to_numeric(pairs[f"{metric}_normal"], errors="coerce").astype(float)
        )
    pairs["alignment_transition"] = "not_comparable"
    comparable = (
        pairs["target_comparable_normal"].fillna(False).astype(bool)
        & pairs["target_comparable_abliterated"].fillna(False).astype(bool)
    )
    normal_aligned = pairs["target_aligned_normal"].fillna(False).astype(bool)
    ablated_aligned = pairs["target_aligned_abliterated"].fillna(False).astype(bool)
    pairs.loc[comparable & ~normal_aligned & ablated_aligned, "alignment_transition"] = (
        "abliteration_rescue"
    )
    pairs.loc[comparable & normal_aligned & ~ablated_aligned, "alignment_transition"] = (
        "abliteration_harm"
    )
    pairs.loc[comparable & normal_aligned & ablated_aligned, "alignment_transition"] = (
        "both_aligned"
    )
    pairs.loc[comparable & ~normal_aligned & ~ablated_aligned, "alignment_transition"] = (
        "both_unaligned"
    )
    return pairs


def contrast_summary(
    pairs: pd.DataFrame,
    metrics: list[str],
    normal_suffix: str = "normal",
    ablated_suffix: str = "abliterated",
    level: str = "item",
) -> pd.DataFrame:
    rows: list[dict] = []
    groups = [("all", pairs), *pairs.groupby("ideology", observed=True)]
    for ideology, group in groups:
        for metric in metrics:
            left = pd.to_numeric(
                group[f"{metric}_{normal_suffix}"], errors="coerce"
            ).astype(float)
            right = pd.to_numeric(
                group[f"{metric}_{ablated_suffix}"], errors="coerce"
            ).astype(float)
            valid = left.notna() & right.notna()
            if level == "item" and metric in {"target_aligned", "target_projection_expected"}:
                valid &= (
                    group["target_comparable_normal"].fillna(False).astype(bool)
                    & group["target_comparable_abliterated"].fillna(False).astype(bool)
                )
            if level == "configuration" and metric in {
                "target_alignment_rate",
                "mean_target_projection",
                "target_quadrant_correct",
                "target_axis_projection",
            }:
                valid &= (
                    group["target_quadrant_comparable_normal"]
                    .fillna(False)
                    .astype(bool)
                    & group["target_quadrant_comparable_abliterated"]
                    .fillna(False)
                    .astype(bool)
                )
            delta_frame = group.loc[valid, ["ideology", "lhs_row"]].copy()
            delta_frame["delta"] = (right[valid] - left[valid]).astype(float)
            if delta_frame.empty:
                continue
            cluster = (
                delta_frame.groupby(["ideology", "lhs_row"], observed=True)["delta"]
                .mean()
                .dropna()
            )
            if len(cluster) > 1:
                test = ttest_1samp(cluster.to_numpy(), popmean=0.0)
                se = float(cluster.std(ddof=1) / np.sqrt(len(cluster)))
                ci_low = float(cluster.mean() - 1.96 * se)
                ci_high = float(cluster.mean() + 1.96 * se)
                p_value = float(test.pvalue)
            else:
                ci_low = ci_high = p_value = np.nan
            rows.append(
                {
                    "level": level,
                    "ideology": ideology,
                    "metric": metric,
                    "metric_type": ITEM_METRICS.get(metric, "continuous"),
                    "n_pairs": int(valid.sum()),
                    "n_design_clusters": int(len(cluster)),
                    "normal_mean": float(left[valid].mean()),
                    "abliterated_mean": float(right[valid].mean()),
                    "difference_abliterated_minus_normal": float(
                        (right[valid] - left[valid]).mean()
                    ),
                    "cluster_robust_ci_low": ci_low,
                    "cluster_robust_ci_high": ci_high,
                    "cluster_mean_ttest_p": p_value,
                }
            )
    result = pd.DataFrame(rows)
    result["q_bh_within_level"] = result.groupby(
        ["level"], observed=True
    )["cluster_mean_ttest_p"].transform(bh_adjust)
    return result


def pooled_benchmark(
    feature_dir: Path,
    comparators: list[str],
    label: str,
    ablated: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = list(ITEM_METRICS)
    frames = []
    for model in comparators:
        frame = load_features(feature_dir / f"{model}.parquet")
        frame["comparator_model"] = model
        frames.append(frame)
    normals = pd.concat(frames, ignore_index=True)
    pooled = (
        normals.groupby(PAIR_KEYS, observed=True)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    matched = pooled.merge(
        ablated[PAIR_KEYS + ["target_comparable"] + metric_columns],
        on=PAIR_KEYS,
        suffixes=("_normal_pool", "_abliterated"),
        validate="one_to_one",
    )
    rows = []
    for ideology, group in [("all", matched), *matched.groupby("ideology", observed=True)]:
        for metric in metric_columns:
            left = pd.to_numeric(
                group[f"{metric}_normal_pool"], errors="coerce"
            ).astype(float)
            right = pd.to_numeric(
                group[f"{metric}_abliterated"], errors="coerce"
            ).astype(float)
            valid = left.notna() & right.notna()
            if metric in {"target_aligned", "target_projection_expected"}:
                valid &= group["target_comparable"].fillna(False).astype(bool)
            if not valid.any():
                continue
            rows.append(
                {
                    "benchmark": label,
                    "ideology": ideology,
                    "metric": metric,
                    "n_matched_cells": int(valid.sum()),
                    "normal_pool_mean": float(left[valid].mean()),
                    "abliterated_mean": float(right[valid].mean()),
                    "difference_abliterated_minus_normal_pool": float(
                        (right[valid] - left[valid]).mean()
                    ),
                    "comparators": len(comparators),
                }
            )
    return pd.DataFrame(rows)


def per_model_benchmark(
    feature_dir: Path, ablated: pd.DataFrame, models: list[str]
) -> pd.DataFrame:
    rows = []
    metric_columns = list(ITEM_METRICS)
    for model in models:
        normal = load_features(feature_dir / f"{model}.parquet")
        matched = normal.merge(
            ablated,
            on=PAIR_KEYS,
            suffixes=("_normal", "_abliterated"),
            validate="one_to_one",
        )
        for ideology, group in [
            ("all", matched),
            *matched.groupby("ideology", observed=True),
        ]:
            for metric in metric_columns:
                left = pd.to_numeric(
                    group[f"{metric}_normal"], errors="coerce"
                ).astype(float)
                right = pd.to_numeric(
                    group[f"{metric}_abliterated"], errors="coerce"
                ).astype(float)
                valid = left.notna() & right.notna()
                if metric in {"target_aligned", "target_projection_expected"}:
                    valid &= (
                        group["target_comparable_normal"].fillna(False).astype(bool)
                        & group["target_comparable_abliterated"]
                        .fillna(False)
                        .astype(bool)
                    )
                if valid.any():
                    rows.append(
                        {
                            "comparator_model": model,
                            "ideology": ideology,
                            "metric": metric,
                            "n_matched_cells": int(valid.sum()),
                            "comparator_mean": float(left[valid].mean()),
                            "abliterated_mean": float(right[valid].mean()),
                            "difference_abliterated_minus_comparator": float(
                                (right[valid] - left[valid]).mean()
                            ),
                            "primary_exact_27b_comparison": model == STANDARD_27B,
                        }
                    )
    return pd.DataFrame(rows)


def question_summary(pairs: pd.DataFrame, question_features: pd.DataFrame) -> pd.DataFrame:
    delta_columns = [f"delta__{metric}" for metric in ITEM_METRICS]
    grouped = (
        pairs.groupby(["ideology", "question_id"], observed=True)[delta_columns]
        .agg(["count", "mean", "std"])
    )
    grouped.columns = ["__".join(column) for column in grouped.columns]
    grouped = grouped.reset_index()
    transitions = (
        pairs[pairs["alignment_transition"].ne("not_comparable")]
        .groupby(
            ["ideology", "question_id", "alignment_transition"], observed=True
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in [
        "abliteration_rescue",
        "abliteration_harm",
        "both_aligned",
        "both_unaligned",
    ]:
        if column not in transitions:
            transitions[column] = 0
    transitions["alignment_switch_rate"] = (
        transitions["abliteration_rescue"] + transitions["abliteration_harm"]
    ) / transitions[
        [
            "abliteration_rescue",
            "abliteration_harm",
            "both_aligned",
            "both_unaligned",
        ]
    ].sum(axis=1).replace(0, np.nan)
    return (
        grouped.merge(transitions, on=["ideology", "question_id"], how="left")
        .merge(question_features, on="question_id", how="left")
        .sort_values(
            "delta__target_projection_expected__mean",
            key=lambda values: values.abs(),
            ascending=False,
        )
    )


def question_feature_associations(question: pd.DataFrame) -> pd.DataFrame:
    targets = [
        "delta__target_projection_expected__mean",
        "delta__target_aligned__mean",
        "delta__stage1_stage2_agree__mean",
        "delta__classification_entropy__mean",
        "delta__stage1_tokens__mean",
        "delta__visible_hedging_present__mean",
        "alignment_switch_rate",
    ]
    rows = []
    for ideology, group in [
        ("all", question.groupby("question_id", observed=True).mean(numeric_only=True).reset_index()),
        *question.groupby("ideology", observed=True),
    ]:
        for target in targets:
            for feature in QUESTION_FEATURES:
                x = pd.to_numeric(group[feature], errors="coerce")
                y = pd.to_numeric(group[target], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 10 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
                    continue
                rho, p_value = spearmanr(x[valid], y[valid])
                rows.append(
                    {
                        "ideology": ideology,
                        "target": target,
                        "question_feature": feature,
                        "n_questions": int(valid.sum()),
                        "spearman_rho": float(rho),
                        "p_value": float(p_value),
                    }
                )
    result = pd.DataFrame(rows)
    result["q_bh"] = result.groupby(
        ["ideology", "target"], observed=True
    )["p_value"].transform(bh_adjust)
    return result.sort_values(["q_bh", "p_value"])


def config_pairs(table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_parquet(table_dir / "config_scores.parquet")
    keys = ["ideology", "lhs_row"]
    normal = scores[scores["model_variant"].eq(STANDARD_27B)]
    ablated = scores[scores["model_variant"].eq(ABLITERATED)]
    pairs = normal.merge(
        ablated,
        on=keys,
        suffixes=("_normal", "_abliterated"),
        validate="one_to_one",
    )
    for metric in CONFIG_METRICS:
        pairs[f"delta__{metric}"] = (
            pd.to_numeric(
                pairs[f"{metric}_abliterated"], errors="coerce"
            ).astype(float)
            - pd.to_numeric(pairs[f"{metric}_normal"], errors="coerce").astype(float)
        )
    summary = contrast_summary(
        pairs,
        CONFIG_METRICS,
        normal_suffix="normal",
        ablated_suffix="abliterated",
        level="configuration",
    )
    return pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=TABLE_DIR / "gemma_abliteration"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = exact_pairs(args.feature_dir)
    ablated = load_features(args.feature_dir / f"{ABLITERATED}.parquet")
    question_features = pd.read_csv(args.feature_dir / "question_features.csv")

    exact = contrast_summary(pairs, list(ITEM_METRICS), level="item")
    transitions = (
        pairs.groupby(["ideology", "alignment_transition"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    question = question_summary(pairs, question_features)
    associations = question_feature_associations(question)

    normal_models = [model for model in ALL_MODELS if model != ABLITERATED]
    pooled = pd.concat(
        [
            pooled_benchmark(
                args.feature_dir,
                STANDARD_GEMMA,
                "all_standard_gemma_sizes",
                ablated,
            ),
            pooled_benchmark(
                args.feature_dir,
                normal_models,
                "all_non_abliterated_models",
                ablated,
            ),
        ],
        ignore_index=True,
    )
    per_model = per_model_benchmark(args.feature_dir, ablated, normal_models)
    configs, config_summary = config_pairs(args.table_dir)

    pairs.to_parquet(args.output_dir / "exact_item_pairs.parquet", index=False)
    configs.to_parquet(args.output_dir / "exact_config_pairs.parquet", index=False)
    exact.to_csv(args.output_dir / "exact_item_contrast.csv", index=False)
    config_summary.to_csv(
        args.output_dir / "exact_configuration_contrast.csv", index=False
    )
    transitions.to_csv(
        args.output_dir / "alignment_transitions.csv", index=False
    )
    question.to_csv(args.output_dir / "question_specific_effects.csv", index=False)
    associations.to_csv(
        args.output_dir / "question_feature_associations.csv", index=False
    )
    pooled.to_csv(args.output_dir / "pooled_normal_benchmarks.csv", index=False)
    per_model.to_csv(args.output_dir / "per_model_benchmarks.csv", index=False)

    write_json(
        args.output_dir / "manifest.json",
        {
            "primary_estimand": (
                "gemma-3-27b-it-abliterated-normpreserve-v1 minus "
                "gemma-3-27b-it, exactly matched by ideology, LHS row, and question"
            ),
            "primary_item_pairs": int(len(pairs)),
            "primary_configuration_pairs": int(len(configs)),
            "secondary_benchmarks": [
                "mean over all standard Gemma sizes within matched design cells",
                "mean over all non-abliterated models within matched design cells",
                "each non-abliterated model within matched design cells",
            ],
            "inference": (
                "Primary confidence intervals and tests use means clustered by "
                "ideology and LHS row; BH q-values cover tests within analysis level."
            ),
            "caveat": (
                "Only the same-size standard Gemma-3-27B contrast holds family, "
                "size, prompts, and design cells fixed. Pooled benchmarks are descriptive."
            ),
        },
    )
    print(
        f"Gemma ablation analysis complete: {len(pairs):,} item pairs and "
        f"{len(configs):,} configuration pairs.",
        flush=True,
    )


if __name__ == "__main__":
    main()
