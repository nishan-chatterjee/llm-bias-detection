#!/usr/bin/env python3
"""Discover localized left/right expressive-behavior contrasts.

This is deliberately a discovery analysis, not a test that every model or
question shares one population-average effect.  It starts from the exact
model/LHS/question pairs produced by ``02_analyze_existing_traces.py`` and
asks where a behavior is sufficiently prevalent to be observable, where the
matched risk difference is substantively large, and whether the same
question-level direction recurs in other models or families.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
QA_ROOT = HERE.parent
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from config import GENERAL_MODEL_ORDER_1, TABLE_DIR  # noqa: E402
from feature_definitions import QUESTION_TOPICS, SENSITIVE_QUESTION_IDS  # noqa: E402


FEATURES = {
    "hedging": "visible_hedging_present",
    "moral_distance": "visible_moral_distance_present",
    "actual_refusal": "visible_actual_refusal_present",
    "persona_meta": "visible_persona_meta_present",
    "self_correction": "visible_self_correction_present",
}

PARENT_FAMILY = {
    "Gemma-3": "Gemma-3",
    "Gemma-3-abliterated": "Gemma-3",
    "Qwen3": "Qwen3",
}


def normal_ci(mean: pd.Series, std: pd.Series, n: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Normal CI for the mean paired difference, clustered at the LHS row."""
    se = std / np.sqrt(n)
    return mean - 1.96 * se, mean + 1.96 * se


def classify_recurrence(row: pd.Series) -> str:
    if not row["informative"]:
        return "uninformative_low_exposure"
    if not row["substantive"]:
        return "no_substantive_pocket"
    if row["support_parent_families"] >= 2:
        return "cross_family_recurrence"
    if row["support_models"] >= 2:
        return "within_family_recurrence"
    return "isolated_model_pocket"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=TABLE_DIR / "matched_left_right_item_contrasts.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLE_DIR / "expressive_heterogeneity",
    )
    parser.add_argument(
        "--minimum-pooled-events",
        type=int,
        default=30,
        help="Minimum left+right events in a 600-response question cell.",
    )
    parser.add_argument(
        "--minimum-risk-difference",
        type=float,
        default=0.05,
        help="Absolute matched risk difference defining a discovery pocket.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Restrict recurrence and summaries to the 12 primary chat variants.",
    )
    args = parser.parse_args()

    pairs = pd.read_parquet(args.pairs)
    if args.primary_only:
        pairs = pairs[pairs["model_variant"].isin(GENERAL_MODEL_ORDER_1)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = [
        "model_variant",
        "family",
        "protocol",
        "size_b",
        "regime",
        "question_id",
    ]
    blocks: list[pd.DataFrame] = []
    for feature, source in FEATURES.items():
        delta = f"delta_right_minus_left__{source}"
        grouped = (
            pairs.groupby(keys, observed=True, dropna=False)
            .agg(
                matched_pairs=("lhs_row", "size"),
                left_events=(f"{source}_left", "sum"),
                right_events=(f"{source}_right", "sum"),
                risk_difference=(delta, "mean"),
                paired_difference_sd=(delta, "std"),
                discordant_pairs=(delta, lambda value: int((value != 0).sum())),
            )
            .reset_index()
        )
        grouped.insert(0, "feature", feature)
        blocks.append(grouped)

    cells = pd.concat(blocks, ignore_index=True)
    cells["left_rate"] = cells["left_events"] / cells["matched_pairs"]
    cells["right_rate"] = cells["right_events"] / cells["matched_pairs"]
    cells["pooled_events"] = cells["left_events"] + cells["right_events"]
    cells["pooled_rate"] = cells["pooled_events"] / (2 * cells["matched_pairs"])
    cells["ci_low"], cells["ci_high"] = normal_ci(
        cells["risk_difference"],
        cells["paired_difference_sd"].fillna(0),
        cells["matched_pairs"],
    )
    cells["ci_excludes_zero"] = (cells["ci_low"] > 0) | (cells["ci_high"] < 0)
    cells["informative"] = cells["pooled_events"] >= args.minimum_pooled_events
    cells["substantive"] = (
        cells["informative"]
        & (cells["risk_difference"].abs() >= args.minimum_risk_difference)
    )
    cells["direction"] = np.select(
        [cells["risk_difference"] > 0, cells["risk_difference"] < 0],
        ["right_more", "left_more"],
        default="tie",
    )
    cells["topic"] = cells["question_id"].map(QUESTION_TOPICS)
    cells["sensitive_topic"] = cells["question_id"].isin(SENSITIVE_QUESTION_IDS)
    cells["analysis_scope"] = np.where(
        cells["model_variant"].isin(GENERAL_MODEL_ORDER_1),
        "primary",
        "auxiliary",
    )
    cells["parent_family"] = cells["family"].map(PARENT_FAMILY).fillna(cells["family"])

    # Recurrence is defined for the same feature, regime, question, and sign.
    support = (
        cells[cells["substantive"]]
        .groupby(
            ["feature", "regime", "question_id", "direction"],
            observed=True,
        )
        .agg(
            support_models=("model_variant", "nunique"),
            support_families=("family", "nunique"),
            support_parent_families=("parent_family", "nunique"),
            support_family_protocols=(
                "protocol",
                lambda value: len(
                    set(
                        zip(
                            cells.loc[value.index, "family"],
                            value,
                        )
                    )
                ),
            ),
        )
        .reset_index()
    )
    cells = cells.merge(
        support,
        on=["feature", "regime", "question_id", "direction"],
        how="left",
    )
    for column in [
        "support_models",
        "support_families",
        "support_parent_families",
        "support_family_protocols",
    ]:
        cells[column] = cells[column].fillna(0).astype(int)
    cells["discovery_class"] = cells.apply(classify_recurrence, axis=1)

    # Qwen protocol-pair replication: same size/question/regime and direction.
    qwen = cells[
        cells["family"].eq("Qwen3")
        & cells["substantive"]
        & cells["protocol"].isin(["think", "no_think"])
    ]
    qwen_support = (
        qwen.groupby(
            ["feature", "size_b", "regime", "question_id", "direction"],
            observed=True,
        )["protocol"]
        .nunique()
        .rename("qwen_protocol_support")
        .reset_index()
    )
    cells = cells.merge(
        qwen_support,
        on=["feature", "size_b", "regime", "question_id", "direction"],
        how="left",
    )
    cells["qwen_protocol_support"] = (
        cells["qwen_protocol_support"].fillna(0).astype(int)
    )
    cells["qwen_protocol_replicated"] = cells["qwen_protocol_support"].eq(2)

    model_summary = (
        cells.groupby(
            [
                "feature",
                "model_variant",
                "family",
                "protocol",
                "size_b",
                "regime",
                "analysis_scope",
            ],
            observed=True,
        )
        .agg(
            questions=("question_id", "nunique"),
            informative_questions=("informative", "sum"),
            substantive_pockets=("substantive", "sum"),
            right_more_pockets=(
                "direction",
                lambda value: int(
                    (
                        value.eq("right_more")
                        & cells.loc[value.index, "substantive"]
                    ).sum()
                ),
            ),
            left_more_pockets=(
                "direction",
                lambda value: int(
                    (
                        value.eq("left_more")
                        & cells.loc[value.index, "substantive"]
                    ).sum()
                ),
            ),
            mean_risk_difference=("risk_difference", "mean"),
            median_risk_difference=("risk_difference", "median"),
            max_abs_risk_difference=("risk_difference", lambda value: value.abs().max()),
        )
        .reset_index()
    )
    model_summary["informative_fraction"] = (
        model_summary["informative_questions"] / model_summary["questions"]
    )
    model_summary["pocket_fraction_of_informative"] = (
        model_summary["substantive_pockets"]
        / model_summary["informative_questions"].replace(0, np.nan)
    )

    recurrence = (
        cells[cells["substantive"]]
        .groupby(
            ["feature", "regime", "question_id", "topic", "direction"],
            observed=True,
        )
        .agg(
            support_models=("model_variant", "nunique"),
            support_families=("family", "nunique"),
            support_parent_families=("parent_family", "nunique"),
            max_abs_risk_difference=("risk_difference", lambda value: value.abs().max()),
            median_risk_difference=("risk_difference", "median"),
            all_ci_exclude_zero=("ci_excludes_zero", "all"),
        )
        .reset_index()
        .sort_values(
            ["feature", "support_families", "support_models", "max_abs_risk_difference"],
            ascending=[True, False, False, False],
        )
    )

    # Treat Qwen's think/no-think modes as a direct same-question replication
    # check. These are complete deployment protocols, not a pure thinking effect.
    qwen_cells = cells[
        cells["family"].eq("Qwen3")
        & cells["protocol"].isin(["think", "no_think"])
    ].copy()
    qwen_wide = qwen_cells.pivot(
        index=["feature", "size_b", "regime", "question_id"],
        columns="protocol",
        values=["risk_difference", "informative", "substantive", "direction"],
    )
    qwen_wide.columns = [f"{metric}__{protocol}" for metric, protocol in qwen_wide]
    qwen_wide = qwen_wide.reset_index()
    stability_rows = []
    for keys_, part in qwen_wide.groupby(
        ["feature", "size_b", "regime"], observed=True
    ):
        both_informative = (
            part["informative__think"].astype(bool)
            & part["informative__no_think"].astype(bool)
        )
        both_substantive = (
            part["substantive__think"].astype(bool)
            & part["substantive__no_think"].astype(bool)
        )
        same_direction = (
            part["direction__think"].eq(part["direction__no_think"])
            & ~part["direction__think"].eq("tie")
        )
        eligible = part[both_informative]
        stability_rows.append(
            {
                "feature": keys_[0],
                "size_b": keys_[1],
                "regime": keys_[2],
                "questions": len(part),
                "both_informative": int(both_informative.sum()),
                "risk_difference_correlation": (
                    eligible["risk_difference__think"].corr(
                        eligible["risk_difference__no_think"]
                    )
                    if len(eligible) >= 2
                    else np.nan
                ),
                "both_substantive": int(both_substantive.sum()),
                "same_direction_substantive": int(
                    (both_substantive & same_direction).sum()
                ),
                "opposite_direction_substantive": int(
                    (both_substantive & ~same_direction).sum()
                ),
                "think_only_substantive": int(
                    (
                        part["substantive__think"].astype(bool)
                        & ~part["substantive__no_think"].astype(bool)
                    ).sum()
                ),
                "no_think_only_substantive": int(
                    (
                        ~part["substantive__think"].astype(bool)
                        & part["substantive__no_think"].astype(bool)
                    ).sum()
                ),
            }
        )
    qwen_stability = pd.DataFrame(stability_rows)

    cells = cells.sort_values(
        ["feature", "substantive", "support_families", "support_models", "risk_difference"],
        ascending=[True, False, False, False, False],
    )
    cells.to_csv(args.output_dir / "question_cells.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    recurrence.to_csv(args.output_dir / "question_recurrence.csv", index=False)
    qwen_stability.to_csv(args.output_dir / "qwen_protocol_stability.csv", index=False)
    manifest = {
        "source": str(args.pairs),
        "rows": int(len(cells)),
        "matched_pairs_per_cell": sorted(cells["matched_pairs"].unique().tolist()),
        "minimum_pooled_events": args.minimum_pooled_events,
        "minimum_risk_difference": args.minimum_risk_difference,
        "features": FEATURES,
        "class_counts": cells["discovery_class"].value_counts().to_dict(),
        "interpretation": (
            "Thresholds rank discovery pockets; confidence intervals are descriptive "
            "and are not used as a significance gate."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
