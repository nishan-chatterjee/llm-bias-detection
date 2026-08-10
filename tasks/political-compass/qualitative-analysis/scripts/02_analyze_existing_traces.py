#!/usr/bin/env python3
"""Analyze existing traces: depth, contexts, item difficulty, and think rescue."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import (  # noqa: E402
    ALL_MODELS,
    FEATURE_DIR,
    GENERAL_MODEL_ORDER_1,
    GENERAL_MODEL_ORDER_2,
    SCORER_PATH,
    TABLE_DIR,
    ensure_artifact_directories,
)
from io_utils import parse_models, qwen_base_model, write_json  # noqa: E402


RESPONSE_METRICS = [
    "stage1_tokens",
    "full_words",
    "think_words",
    "visible_words",
    "classification_entropy",
    "classification_margin",
    "explicit_stance_found",
    "stage1_stage2_agree",
    "stage1_stage2_comparable",
    "target_aligned",
    "target_projection_hard",
    "target_projection_expected",
    "visible_hedging_present",
    "visible_hedging_per_100_words",
    "visible_contrast_present",
    "visible_balance_present",
    "visible_disclaimer_present",
    "visible_ai_identity_present",
    "visible_actual_refusal_present",
    "visible_moral_distance_present",
    "visible_persona_meta_present",
    "visible_self_correction_present",
    "visible_counterargument_present",
    "visible_certainty_present",
    "think_hedging_present",
    "think_self_correction_present",
    "think_persona_meta_present",
    "think_counterargument_present",
    "persona_vocab_overlap_fraction",
]


def load_model(path: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    columns = [
        name
        for name in pq.ParquetFile(path).schema.names
        if name not in {"rationale_text"}
    ]
    return pd.read_parquet(path, columns=columns)


def summarize(
    frame: pd.DataFrame,
    groups: list[str],
    metrics: list[str] = RESPONSE_METRICS,
) -> pd.DataFrame:
    available = [metric for metric in metrics if metric in frame.columns]
    summary = frame.groupby(groups, observed=True, dropna=False)[available].agg(
        ["count", "mean", "median", "std"]
    )
    summary.columns = ["__".join(column) for column in summary.columns]
    return summary.reset_index()


def context_depth_relation(row: pd.Series) -> str:
    context = int(row["context_id"])
    mode = str(row["reasoning_mode"])
    anti_depth = context in {0, 1, 2}
    if anti_depth and mode == "think":
        return "conflicting"
    if anti_depth and mode == "short":
        return "congruent_short"
    if context == -1:
        return "no_context"
    if context == 3:
        return "anti_correction_context"
    if context == 4:
        return "spatial_context"
    return "neutral_or_mixed"


def empirical_logit(successes: pd.Series, totals: pd.Series) -> pd.Series:
    return np.log((successes + 0.5) / (totals - successes + 0.5))


def compute_dif(question_cells: pd.DataFrame) -> pd.DataFrame:
    eligible = question_cells[
        question_cells["ideology"].isin(
            [
                "libertarian_left",
                "libertarian_right",
                "authoritarian_left",
                "authoritarian_right",
            ]
        )
        & question_cells["scorer_axis"].ne("unweighted")
    ].copy()
    eligible["cell_logit"] = empirical_logit(
        eligible["aligned_n"], eligible["comparable_n"]
    )
    group_cols = ["family", "protocol", "ideology"]
    group_rate = (
        eligible.groupby(group_cols, observed=True)[["aligned_n", "comparable_n"]]
        .sum()
        .reset_index()
    )
    group_rate["group_logit"] = empirical_logit(
        group_rate["aligned_n"], group_rate["comparable_n"]
    )
    question_rate = (
        eligible.groupby(["question_id"], observed=True)[["aligned_n", "comparable_n"]]
        .sum()
        .reset_index()
    )
    question_rate["question_logit"] = empirical_logit(
        question_rate["aligned_n"], question_rate["comparable_n"]
    )
    grand_success = eligible["aligned_n"].sum()
    grand_total = eligible["comparable_n"].sum()
    grand_logit = math.log(
        (grand_success + 0.5) / (grand_total - grand_success + 0.5)
    )
    eligible = eligible.merge(
        group_rate[group_cols + ["group_logit"]], on=group_cols, how="left"
    ).merge(
        question_rate[["question_id", "question_logit"]],
        on="question_id",
        how="left",
    )
    eligible["dif_logit"] = (
        eligible["cell_logit"]
        - eligible["group_logit"]
        - eligible["question_logit"]
        + grand_logit
    )
    eligible["dif_abs"] = eligible["dif_logit"].abs()
    return eligible.sort_values("dif_abs", ascending=False)


def config_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scorer = np.load(SCORER_PATH)
    economic = scorer["weights_economic"]
    social = scorer["weights_social"]
    bias_economic = float(scorer["bias_economic"])
    bias_social = float(scorer["bias_social"])
    qid = frame["question_id"].to_numpy(dtype=int)
    probs = frame[
        [
            "prob_0_strongly_disagree",
            "prob_1_disagree",
            "prob_2_agree",
            "prob_3_strongly_agree",
        ]
    ].to_numpy(dtype=float)
    frame = frame.copy()
    frame["item_economic_expected"] = np.einsum(
        "ij,ij->i", probs, economic[qid]
    )
    frame["item_social_expected"] = np.einsum("ij,ij->i", probs, social[qid])
    groups = [
        "model_variant",
        "base_model",
        "family",
        "protocol",
        "size_b",
        "ideology",
        "lhs_row",
        "reasoning_mode",
        "context_id",
        "persona_class",
    ]
    scores = (
        frame.groupby(groups, observed=True, dropna=False)
        .agg(
            economic=("item_economic_expected", "sum"),
            social=("item_social_expected", "sum"),
            mean_stage1_tokens=("stage1_tokens", "mean"),
            mean_visible_words=("visible_words", "mean"),
            target_alignment_rate=("target_aligned", "mean"),
            mean_target_projection=("target_projection_expected", "mean"),
            stage1_stage2_agreement=("stage1_stage2_agree", "mean"),
            mean_entropy=("classification_entropy", "mean"),
        )
        .reset_index()
    )
    scores.loc[
        ~scores["ideology"].isin(
            [
                "libertarian_left",
                "libertarian_right",
                "authoritarian_left",
                "authoritarian_right",
            ]
        ),
        ["target_alignment_rate", "mean_target_projection"],
    ] = np.nan
    scores["economic"] += bias_economic
    scores["social"] += bias_social
    target_econ = scores["ideology"].map(
        {
            "libertarian_left": -1,
            "libertarian_right": 1,
            "authoritarian_left": -1,
            "authoritarian_right": 1,
        }
    )
    target_social = scores["ideology"].map(
        {
            "libertarian_left": -1,
            "libertarian_right": -1,
            "authoritarian_left": 1,
            "authoritarian_right": 1,
        }
    )
    scores["target_quadrant_comparable"] = target_econ.notna()
    scores["target_quadrant_correct"] = (
        (scores["economic"] * target_econ > 0)
        & (scores["social"] * target_social > 0)
        & target_econ.notna()
    )
    scores["target_axis_projection"] = (
        scores["economic"] * target_econ + scores["social"] * target_social
    )
    scores["distance_from_origin"] = np.sqrt(
        scores["economic"] ** 2 + scores["social"] ** 2
    )
    return scores


def match_qwen_question_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    qwen = frame[
        frame["family"].eq("Qwen3")
        & frame["protocol"].isin(["think", "no_think"])
    ].copy()
    if qwen.empty:
        return pd.DataFrame()
    qwen["pair_model"] = qwen["model_variant"].map(qwen_base_model)
    keys = ["pair_model", "ideology", "lhs_row", "question_id"]
    value_cols = [
        "trace_id",
        "reasoning_mode",
        "context_id",
        "persona_class",
        "scorer_axis",
        "topic",
        "sensitive_topic",
        "stage1_tokens",
        "visible_words",
        "think_words",
        "classification_entropy",
        "classification_margin",
        "stage1_stage2_agree",
        "target_comparable",
        "target_aligned",
        "target_projection_expected",
        "visible_hedging_present",
        "visible_moral_distance_present",
        "visible_actual_refusal_present",
        "visible_persona_meta_present",
        "visible_self_correction_present",
        "think_hedging_present",
        "think_self_correction_present",
        "think_persona_meta_present",
    ]
    think = qwen[qwen["protocol"].eq("think")][keys + value_cols].copy()
    no_think = qwen[qwen["protocol"].eq("no_think")][keys + value_cols].copy()
    pairs = no_think.merge(
        think,
        on=keys,
        how="inner",
        suffixes=("_no_think", "_think"),
        validate="one_to_one",
    )
    for metric in [
        "stage1_tokens",
        "visible_words",
        "classification_entropy",
        "classification_margin",
        "target_projection_expected",
    ]:
        pairs[f"delta_{metric}"] = (
            pairs[f"{metric}_think"] - pairs[f"{metric}_no_think"]
        )
    comparable = (
        pairs["target_comparable_no_think"] & pairs["target_comparable_think"]
    )
    pairs["think_outcome"] = "not_comparable"
    pairs.loc[
        comparable
        & ~pairs["target_aligned_no_think"]
        & pairs["target_aligned_think"],
        "think_outcome",
    ] = "think_rescue"
    pairs.loc[
        comparable
        & pairs["target_aligned_no_think"]
        & ~pairs["target_aligned_think"],
        "think_outcome",
    ] = "think_harm"
    pairs.loc[
        comparable
        & pairs["target_aligned_no_think"]
        & pairs["target_aligned_think"],
        "think_outcome",
    ] = "both_aligned"
    pairs.loc[
        comparable
        & ~pairs["target_aligned_no_think"]
        & ~pairs["target_aligned_think"],
        "think_outcome",
    ] = "both_unaligned"
    return pairs


def match_qwen_config_pairs(scores: pd.DataFrame) -> pd.DataFrame:
    qwen = scores[
        scores["family"].eq("Qwen3")
        & scores["protocol"].isin(["think", "no_think"])
    ].copy()
    qwen["pair_model"] = qwen["model_variant"].map(qwen_base_model)
    keys = ["pair_model", "ideology", "lhs_row"]
    value_cols = [
        "reasoning_mode",
        "context_id",
        "persona_class",
        "economic",
        "social",
        "mean_stage1_tokens",
        "mean_visible_words",
        "target_alignment_rate",
        "mean_target_projection",
        "stage1_stage2_agreement",
        "mean_entropy",
        "target_quadrant_comparable",
        "target_quadrant_correct",
        "target_axis_projection",
    ]
    think = qwen[qwen["protocol"].eq("think")][keys + value_cols]
    no_think = qwen[qwen["protocol"].eq("no_think")][keys + value_cols]
    pairs = no_think.merge(
        think,
        on=keys,
        suffixes=("_no_think", "_think"),
        validate="one_to_one",
    )
    for metric in [
        "economic",
        "social",
        "mean_stage1_tokens",
        "mean_visible_words",
        "target_alignment_rate",
        "mean_target_projection",
        "stage1_stage2_agreement",
        "mean_entropy",
        "target_axis_projection",
    ]:
        pairs[f"delta_{metric}"] = (
            pairs[f"{metric}_think"] - pairs[f"{metric}_no_think"]
        )
    comparable = (
        pairs["target_quadrant_comparable_no_think"]
        & pairs["target_quadrant_comparable_think"]
    )
    pairs["think_config_outcome"] = "not_comparable"
    pairs.loc[
        comparable
        & ~pairs["target_quadrant_correct_no_think"]
        & pairs["target_quadrant_correct_think"],
        "think_config_outcome",
    ] = "think_rescue"
    pairs.loc[
        comparable
        & pairs["target_quadrant_correct_no_think"]
        & ~pairs["target_quadrant_correct_think"],
        "think_config_outcome",
    ] = "think_harm"
    pairs.loc[
        comparable
        & pairs["target_quadrant_correct_no_think"]
        & pairs["target_quadrant_correct_think"],
        "think_config_outcome",
    ] = "both_correct"
    pairs.loc[
        comparable
        & ~pairs["target_quadrant_correct_no_think"]
        & ~pairs["target_quadrant_correct_think"],
        "think_config_outcome",
    ] = "both_incorrect"
    return pairs


def matched_left_right_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare right/left personas on exactly matched model/LHS/question cells."""
    keys = [
        "model_variant",
        "family",
        "protocol",
        "size_b",
        "lhs_row",
        "question_id",
    ]
    metrics = [
        "stage1_tokens",
        "visible_words",
        "classification_entropy",
        "stage1_stage2_agree",
        "target_projection_expected",
        "visible_hedging_present",
        "visible_actual_refusal_present",
        "visible_moral_distance_present",
        "visible_persona_meta_present",
        "visible_self_correction_present",
    ]
    blocks = []
    for regime, left_ideology, right_ideology in [
        ("libertarian", "libertarian_left", "libertarian_right"),
        ("authoritarian", "authoritarian_left", "authoritarian_right"),
    ]:
        left = frame[frame["ideology"].eq(left_ideology)][keys + metrics]
        right = frame[frame["ideology"].eq(right_ideology)][keys + metrics]
        paired = left.merge(
            right,
            on=keys,
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        paired["regime"] = regime
        for metric in metrics:
            paired[f"delta_right_minus_left__{metric}"] = (
                pd.to_numeric(paired[f"{metric}_right"], errors="coerce").astype(float)
                - pd.to_numeric(paired[f"{metric}_left"], errors="coerce").astype(float)
            )
        blocks.append(paired)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=TABLE_DIR)
    args = parser.parse_args()

    ensure_artifact_directories()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.models, ALL_MODELS)
    frames = []
    model_summaries = []
    question_cells = []
    depth_summaries = []
    context_summaries = []
    factor_summaries = []
    suffix_controls = []
    duplicate_summaries = []

    for model in models:
        path = args.feature_dir / f"{model}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = load_model(path)
        frame["context_depth_relation"] = frame.apply(context_depth_relation, axis=1)
        frames.append(frame)
        model_summaries.append(summarize(frame, ["model_variant", "ideology"]))
        depth_summaries.append(
            summarize(frame, ["model_variant", "ideology", "reasoning_mode"])
        )
        context_summaries.append(
            summarize(
                frame,
                [
                    "model_variant",
                    "ideology",
                    "context_id",
                    "context_depth_relation",
                ],
            )
        )
        for factor in [
            "context_id",
            "instr_type",
            "reasoning_mode",
            "instr_idx",
            "persona_class",
            "persona_idx",
            "key_type",
            "perm_id",
            "suffix_idx",
        ]:
            factor_summary = summarize(
                frame,
                ["model_variant", "ideology", factor],
            )
            factor_summary = factor_summary.rename(columns={factor: "factor_level"})
            factor_summary.insert(2, "factor", factor)
            factor_summaries.append(factor_summary)
        suffix_controls.append(
            summarize(
                frame,
                ["model_variant", "ideology", "suffix_idx"],
                [
                    "stage1_tokens",
                    "visible_words",
                    "visible_hedging_present",
                    "visible_actual_refusal_present",
                    "visible_persona_meta_present",
                ],
            )
        )
        duplicate_summaries.append(
            frame.groupby(["model_variant", "ideology"], observed=True)
            .agg(
                n=("trace_id", "size"),
                unique_texts=("content_hash", "nunique"),
                empty_visible=("visible_words", lambda values: int((values == 0).sum())),
            )
            .reset_index()
        )
        cell = (
            frame.groupby(
                [
                    "model_variant",
                    "family",
                    "protocol",
                    "size_b",
                    "ideology",
                    "question_id",
                    "topic",
                    "sensitive_topic",
                    "scorer_axis",
                    "scorer_weight_range",
                ],
                observed=True,
                dropna=False,
            )
            .agg(
                n=("trace_id", "size"),
                comparable_n=("target_comparable", "sum"),
                aligned_n=("target_aligned", "sum"),
                target_alignment_rate=("target_aligned", "mean"),
                mean_target_projection=("target_projection_expected", "mean"),
                mean_entropy=("classification_entropy", "mean"),
                mean_margin=("classification_margin", "mean"),
                stage1_stage2_comparable_n=("stage1_stage2_comparable", "sum"),
                stage1_stage2_agreement_rate=("stage1_stage2_agree", "mean"),
                mean_tokens=("stage1_tokens", "mean"),
                mean_visible_words=("visible_words", "mean"),
                hedging_rate=("visible_hedging_present", "mean"),
                refusal_rate=("visible_actual_refusal_present", "mean"),
                moral_distance_rate=("visible_moral_distance_present", "mean"),
                self_correction_rate=("visible_self_correction_present", "mean"),
            )
            .reset_index()
        )
        question_cells.append(cell)
        print(f"Loaded and summarized {model}: {len(frame):,} rows", flush=True)

    all_frame = pd.concat(frames, ignore_index=True)
    del frames

    response_by_model = pd.concat(model_summaries, ignore_index=True)
    group_tables = []
    for group_number, order in [
        (1, GENERAL_MODEL_ORDER_1),
        (2, GENERAL_MODEL_ORDER_2),
    ]:
        group = response_by_model[response_by_model["model_variant"].isin(order)].copy()
        group["model_variant"] = pd.Categorical(
            group["model_variant"], categories=order, ordered=True
        )
        group = group.sort_values(["model_variant", "ideology"])
        group["display_group"] = group_number
        group_tables.append(group)

    outputs = {
        "response_summary_by_model_ideology.csv": response_by_model,
        "response_summary_display_group_1.csv": group_tables[0],
        "response_summary_display_group_2.csv": group_tables[1],
        "response_summary_by_family_protocol_ideology.csv": summarize(
            all_frame, ["family", "protocol", "ideology"]
        ),
        "reasoning_instruction_summary.csv": pd.concat(
            depth_summaries, ignore_index=True
        ),
        "context_interaction_summary.csv": pd.concat(
            context_summaries, ignore_index=True
        ),
        "design_factor_summary_long.csv": pd.concat(
            factor_summaries, ignore_index=True
        ),
        "suffix_stage1_negative_control.csv": pd.concat(
            suffix_controls, ignore_index=True
        ),
        "duplicate_and_empty_summary.csv": pd.concat(
            duplicate_summaries, ignore_index=True
        ),
    }
    question_frame = pd.concat(question_cells, ignore_index=True)
    outputs["question_difficulty_by_model.csv"] = question_frame
    outputs["question_dif_family_protocol.csv"] = compute_dif(question_frame)

    family_question = (
        question_frame.groupby(
            [
                "family",
                "protocol",
                "ideology",
                "question_id",
                "topic",
                "sensitive_topic",
                "scorer_axis",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            models=("model_variant", "nunique"),
            n=("n", "sum"),
            comparable_n=("comparable_n", "sum"),
            aligned_n=("aligned_n", "sum"),
            mean_entropy=("mean_entropy", "mean"),
            mean_margin=("mean_margin", "mean"),
            stage1_stage2_agreement_rate=("stage1_stage2_agreement_rate", "mean"),
            mean_tokens=("mean_tokens", "mean"),
            hedging_rate=("hedging_rate", "mean"),
            refusal_rate=("refusal_rate", "mean"),
            moral_distance_rate=("moral_distance_rate", "mean"),
        )
        .reset_index()
    )
    family_question["target_alignment_rate"] = (
        family_question["aligned_n"]
        / family_question["comparable_n"].replace(0, np.nan)
    )
    outputs["question_difficulty_by_family.csv"] = family_question

    left_right = matched_left_right_contrasts(all_frame)
    left_right.to_parquet(
        args.output_dir / "matched_left_right_item_contrasts.parquet", index=False
    )
    contrast_metrics = [
        column
        for column in left_right.columns
        if column.startswith("delta_right_minus_left__")
    ]
    if not left_right.empty:
        outputs["matched_left_right_summary_by_model.csv"] = summarize(
            left_right,
            ["model_variant", "family", "protocol", "regime"],
            contrast_metrics,
        )
        outputs["matched_left_right_summary_by_family.csv"] = summarize(
            left_right,
            ["family", "protocol", "regime"],
            contrast_metrics,
        )

    scores = config_scores(all_frame)
    scores.to_parquet(args.output_dir / "config_scores.parquet", index=False)
    outputs["config_score_summary.csv"] = summarize(
        scores,
        ["model_variant", "ideology", "reasoning_mode"],
        [
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
        ],
    )

    question_pairs = match_qwen_question_pairs(all_frame)
    question_pairs.to_parquet(
        args.output_dir / "qwen_think_question_pairs.parquet", index=False
    )
    if not question_pairs.empty:
        outputs["qwen_think_question_outcomes.csv"] = (
            question_pairs.groupby(
                [
                    "pair_model",
                    "ideology",
                    "reasoning_mode_no_think",
                    "question_id",
                    "topic_no_think",
                    "think_outcome",
                ],
                observed=True,
            )
            .size()
            .rename("n")
            .reset_index()
        )
        outputs["qwen_think_length_content_summary.csv"] = summarize(
            question_pairs,
            ["pair_model", "ideology", "think_outcome"],
            [
                "delta_stage1_tokens",
                "delta_visible_words",
                "delta_classification_entropy",
                "delta_classification_margin",
                "delta_target_projection_expected",
                "think_words_think",
                "think_hedging_present_think",
                "think_self_correction_present_think",
                "think_persona_meta_present_think",
            ],
        )

    config_pairs = match_qwen_config_pairs(scores)
    config_pairs.to_parquet(
        args.output_dir / "qwen_think_config_pairs.parquet", index=False
    )
    if not config_pairs.empty:
        outputs["qwen_think_config_outcomes.csv"] = (
            config_pairs.groupby(
                [
                    "pair_model",
                    "ideology",
                    "reasoning_mode_no_think",
                    "think_config_outcome",
                ],
                observed=True,
            )
            .agg(
                n=("lhs_row", "size"),
                mean_delta_target_projection=("delta_target_axis_projection", "mean"),
                mean_delta_tokens=("delta_mean_stage1_tokens", "mean"),
                mean_delta_visible_words=("delta_mean_visible_words", "mean"),
                mean_delta_entropy=("delta_mean_entropy", "mean"),
            )
            .reset_index()
        )

    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
        print(f"Wrote {filename}: {len(frame):,} rows")

    write_json(
        args.output_dir / "existing_trace_analysis_manifest.json",
        {
            "models": models,
            "response_rows": int(len(all_frame)),
            "config_rows": int(len(scores)),
            "qwen_question_pairs": int(len(question_pairs)),
            "qwen_config_pairs": int(len(config_pairs)),
            "outputs": sorted(list(outputs) + [
                "config_scores.parquet",
                "qwen_think_question_pairs.parquet",
                "qwen_think_config_pairs.parquet",
                "matched_left_right_item_contrasts.parquet",
            ]),
        },
    )


if __name__ == "__main__":
    main()
