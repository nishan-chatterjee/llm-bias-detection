#!/usr/bin/env python3
"""Recompute the paper-facing numerical checks from the released tables.

This is a claim ledger, not a statistical test suite.  It prints the exact
aggregation behind each release-facing statement and fails only when the
released snapshot no longer reproduces the recorded values.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


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
PERSONA_SIGNS = {
    "libertarian_left": (-1, -1),
    "libertarian_right": (1, -1),
    "authoritarian_left": (-1, 1),
    "authoritarian_right": (1, 1),
}


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"expected {expected:.6f}, got {actual:.6f}")


def read(root: Path, task: str, name: str) -> pd.DataFrame:
    path = root / "data" / "analysis_ready" / task / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_dataset.py --component analysis`."
        )
    return pd.read_parquet(path)


def political_compass(root: Path) -> None:
    mcq = read(root, "political_compass", "pct_configuration_scores")
    chat = read(root, "political_compass", "chat_configuration_scores")
    agreement = read(root, "political_compass", "chat_stage_agreement_summary")
    assert len(mcq) == 43_200 and len(chat) == 21_600

    centroids = mcq.groupby(["model", "ideology"], observed=True)[
        ["score_econ", "score_soc"]
    ].mean()
    gemma12 = centroids.loc["gemma-3-12b-it", "score_econ"]
    print("\nPolitical Compass MCQ — full 14-language × 3-quantization design mean")
    for ideology in ["base", "libertarian_right", "authoritarian_left"]:
        value = float(gemma12.loc[ideology])
        print(f"  Gemma 3 12B {ideology:>20}: economic {value:+.3f}")
    close(float(gemma12.loc["base"]), -1.642386)
    close(float(gemma12.loc["libertarian_right"]), 5.512698)
    close(float(gemma12.loc["authoritarian_left"]), -6.095540)

    for ideology, (econ_sign, social_sign) in PERSONA_SIGNS.items():
        mask = mcq["ideology"].eq(ideology)
        mcq.loc[mask, "target_quadrant"] = (
            (mcq.loc[mask, "score_econ"] * econ_sign > 0)
            & (mcq.loc[mask, "score_soc"] * social_sign > 0)
        )
    rates = (
        mcq[mcq["ideology"].isin(PERSONA_SIGNS)]
        .groupby("ideology", observed=True)["target_quadrant"]
        .mean()
        .sort_values()
    )
    print("  MCQ target-quadrant rates pooled over models:")
    for ideology, value in rates.items():
        print(f"    {ideology:>20}: {value:.1%}")

    qwen = chat[chat["family"].eq("Qwen3")]
    keys = [
        "base_model",
        "ideology",
        "lhs_row",
        "reasoning_mode",
        "context_id",
        "persona_class",
    ]
    wide = qwen.pivot_table(
        index=keys,
        columns="protocol",
        values=["economic", "social"],
        aggfunc="first",
    ).dropna()
    economic_shift = float((wide[("economic", "think")] - wide[("economic", "no_think")]).mean())
    social_shift = float((wide[("social", "think")] - wide[("social", "no_think")]).mean())
    print(f"  Matched Qwen pairs: {len(wide):,}")
    print(f"  think − no-think economic shift: {economic_shift:+.3f}")
    print(f"  think − no-think social shift:   {social_shift:+.3f}")
    close(economic_shift, -0.334614)
    close(social_shift, -0.198890)

    contraction = []
    for (model, ideology), group in qwen.groupby(["base_model", "ideology"], observed=True):
        row = {"model": model, "ideology": ideology}
        for protocol in ["no_think", "think"]:
            values = group[group["protocol"].eq(protocol)][["economic", "social"]].to_numpy(float)
            covariance = np.cov(values, rowvar=False, ddof=1)
            row[f"economic_{protocol}"] = values[:, 0].std(ddof=1)
            row[f"social_{protocol}"] = values[:, 1].std(ddof=1)
            row[f"area_{protocol}"] = np.sqrt(max(0.0, np.linalg.det(covariance)))
        contraction.append(row)
    contraction = pd.DataFrame(contraction)
    print("  Qwen model × persona regions with smaller spread under think:")
    expected = {"economic": (16, 0.854259), "social": (24, 0.557081), "area": (24, 0.513197)}
    for metric, (expected_count, expected_median) in expected.items():
        ratio = contraction[f"{metric}_think"] / contraction[f"{metric}_no_think"]
        count = int((ratio < 1).sum())
        median = float(ratio.median())
        print(f"    {metric:>8}: {count}/24; median think/no-think ratio {median:.3f}")
        assert count == expected_count
        close(median, expected_median)

    chat_persona = chat[chat["ideology"].isin(PERSONA_SIGNS)]
    pooled = chat_persona.groupby("ideology", observed=True)["target_quadrant_correct"].mean()
    by_variant = chat_persona.pivot_table(
        index="model_variant",
        columns="ideology",
        values="target_quadrant_correct",
        aggfunc="mean",
    )
    lowest = by_variant.idxmin(axis=1).value_counts()
    print("  Chat target-quadrant rates pooled over variants:")
    for ideology, value in pooled.sort_values().items():
        print(f"    {ideology:>20}: {value:.1%}")
    print("  Lowest-rate persona among the 12 variants:", dict(lowest))
    assert lowest.to_dict() == {
        "authoritarian_left": 9,
        "authoritarian_right": 2,
        "libertarian_right": 1,
    }
    conditional_agreement = float(
        agreement["agreement_rows"].sum() / agreement["comparable_rows"].sum()
    )
    print(
        "  Explicit Stage-1/Stage-2 agreement among comparable rows: "
        f"{conditional_agreement:.1%}"
    )
    close(conditional_agreement, 0.950929)


def sentiment(root: Path) -> None:
    persona = read(root, "sentiment", "summary_by_model_persona")
    sensitivity = read(root, "sentiment", "factor_sensitivity_macro_f1")
    assert len(persona) == 48 and len(sensitivity) == 8
    wide = persona.pivot(index="model", columns="ideology", values="macro_f1").reindex(MODELS)
    condition_means = wide.mean().sort_values(ascending=False)
    print("\nIBM topic sentiment")
    print("  Mean macro-F1 over eight models:")
    for ideology, value in condition_means.items():
        print(f"    {ideology:>20}: {value:.3f}")
    assert wide.idxmax(axis=1).eq("base").all()
    assert wide.rank(axis=1, ascending=False, method="min")["centrism"].eq(2).all()
    close(float(condition_means["base"]), 0.631587)
    close(float(condition_means["centrism"]), 0.603485)
    print("  Base ranks first and centrist ranks second for all 8 checkpoints.")

    indexed = sensitivity.set_index("model").reindex(MODELS)
    gemma = indexed.loc[MODELS[:4], "Ideology SD"].to_numpy()
    qwen = indexed.loc[MODELS[4:], "Ideology SD"].to_numpy()
    assert np.all(np.diff(gemma) > 0) and np.all(np.diff(qwen) > 0)
    print("  Ideology SD by increasing checkpoint size:")
    print("    Gemma:", ", ".join(f"{value:.3f}" for value in gemma))
    print("    Qwen: ", ", ".join(f"{value:.3f}" for value in qwen))
    factors = ["Context", "Instruction", "Key Type", "Permutation", "Persona"]
    largest = indexed[factors].idxmax(axis=1)
    print("  Largest named prompt factor:", dict(largest.value_counts()))
    assert largest.value_counts().to_dict() == {"Instruction": 7, "Permutation": 1}


def hate_speech(root: Path, include_auc: bool) -> None:
    sensitivity = read(root, "hate_speech", "hs_factor_sensitivity")
    metrics = read(root, "hate_speech", "hs_item_metrics_by_persona_target")
    assert len(sensitivity) == 8 and len(metrics) == 480
    print("\nHate-speech detection")
    for column in ["Ideology SD", "persona_combo_sd", "context_combo_sd", "instr_combo_sd"]:
        print(
            f"  {column:>18}: {sensitivity[column].min():.3f}–"
            f"{sensitivity[column].max():.3f}"
        )
    close(float(sensitivity["Ideology SD"].max()), 0.131350)

    base = metrics[metrics["ideology"].eq("base")]
    f1_winners = base.loc[base.groupby("target")["f1"].idxmax(), "model_alias"].value_counts()
    precision_winners = base.loc[
        base.groupby("target")["precision"].idxmax(), "model_alias"
    ].value_counts()
    print("  Base-condition target wins by F1:", dict(f1_winners))
    print("  Base-condition target wins by precision:", dict(precision_winners))
    assert f1_winners.to_dict() == {"gemma-3-27b-it": 9, "Qwen3-8B": 1}
    assert precision_winners.to_dict() == {"Qwen3-14B": 10}

    if not include_auc:
        print("  AUC scan skipped (pass --include-hate-auc after downloading full hate outputs).")
        return
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for --include-hate-auc") from exc
    rows = []
    raw_dir = root / "data" / "hate_speech"
    files = sorted(raw_dir.glob("*.parquet"))
    if len(files) != 8:
        raise FileNotFoundError(f"Expected 8 hate-speech Parquets under {raw_dir}")
    for path in files:
        frame = pd.read_parquet(
            path, columns=["model_alias", "ideology", "target", "gold_hate", "p_hate"]
        )
        frame = frame[frame["ideology"].eq("base")]
        for (model, target), group in frame.groupby(["model_alias", "target"], observed=True):
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "auc": roc_auc_score(group["gold_hate"], group["p_hate"]),
                }
            )
    auc = pd.DataFrame(rows)
    target_means = auc.groupby("target", observed=True)["auc"].mean()
    print(
        "  Target mean-AUC range across the eight models: "
        f"{target_means.min():.3f}–{target_means.max():.3f}"
    )
    close(float(target_means.min()), 0.568837)
    close(float(target_means.max()), 0.692709)


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(os.environ.get("LLM_BIAS_DATA_DIR", "data/release"))
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--include-hate-auc", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    political_compass(root)
    sentiment(root)
    hate_speech(root, args.include_hate_auc)
    print("\nAll selected release-value checks passed.")


if __name__ == "__main__":
    main()
