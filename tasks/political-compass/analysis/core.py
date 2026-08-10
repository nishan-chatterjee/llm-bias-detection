"""Shared primary-model analysis helpers for Political Compass notebooks."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"

GEMMA_MODELS = [
    "gemma-3-1b-it",
    "gemma-3-4b-it",
    "gemma-3-12b-it",
    "gemma-3-27b-it",
]
QWEN_MODELS = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]
PRIMARY_MODELS = GEMMA_MODELS + QWEN_MODELS
IDEOLOGY_ORDER = [
    "base",
    "libertarian_left",
    "libertarian_right",
    "authoritarian_left",
    "authoritarian_right",
    "centrism",
]
IDEOLOGY_LABELS = {
    "base": "Base",
    "libertarian_left": "Libertarian Left",
    "libertarian_right": "Libertarian Right",
    "authoritarian_left": "Authoritarian Left",
    "authoritarian_right": "Authoritarian Right",
    "centrism": "Centrism",
}
IDEOLOGY_COLORS = {
    "base": "#4b5563",
    "libertarian_left": "#2ca02c",
    "libertarian_right": "#9467bd",
    "authoritarian_left": "#d62728",
    "authoritarian_right": "#1f77b4",
    "centrism": "#7f7f7f",
}


def size_label(name: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
    return f"{match.group(1)}B" if match else name


def load_mcq_configurations(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path or DATA_DIR / "pct_configuration_scores.csv")
    frame = frame[frame["model"].isin(PRIMARY_MODELS)].copy()
    frame["source"] = "MCQ"
    frame["base_model"] = frame["model"]
    frame["protocol"] = "mcq"
    return frame


def load_chat_configurations(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path or DATA_DIR / "chat_configuration_scores.parquet")
    return frame[frame["base_model"].isin(PRIMARY_MODELS)].copy()


def method_order() -> list[tuple[str, str, str]]:
    """Family-grouped paper order for combined MCQ/chat displays."""
    rows: list[tuple[str, str, str]] = []
    rows.extend((model, "mcq", f"Gemma MCQ {size_label(model)}") for model in GEMMA_MODELS)
    rows.extend((model, "standard", f"Gemma Chat {size_label(model)}") for model in GEMMA_MODELS)
    rows.extend((model, "mcq", f"Qwen MCQ {size_label(model)}") for model in QWEN_MODELS)
    rows.extend((model, "no_think", f"Qwen Chat no-think {size_label(model)}") for model in QWEN_MODELS)
    rows.extend((model, "think", f"Qwen Chat think {size_label(model)}") for model in QWEN_MODELS)
    return rows


def draw_compass_background(ax) -> None:
    ax.add_patch(patches.Rectangle((-10, 0), 10, 10, color="#ef4444", alpha=0.06))
    ax.add_patch(patches.Rectangle((0, 0), 10, 10, color="#3b82f6", alpha=0.06))
    ax.add_patch(patches.Rectangle((-10, -10), 10, 10, color="#22c55e", alpha=0.06))
    ax.add_patch(patches.Rectangle((0, -10), 10, 10, color="#a855f7", alpha=0.06))
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.12, linewidth=0.5)


def plot_centroid_grid(
    frame: pd.DataFrame,
    panels: list[tuple[str, str]],
    model_col: str = "base_model",
    protocol_col: str = "protocol",
    ncols: int = 4,
    title: str | None = None,
):
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.1 * nrows), squeeze=False)
    for ax, (panel_key, panel_label) in zip(axes.flat, panels):
        model, protocol = panel_key.split("|", 1)
        sub = frame[(frame[model_col] == model) & (frame[protocol_col] == protocol)]
        draw_compass_background(ax)
        for ideology in IDEOLOGY_ORDER:
            ide = sub[sub["ideology"] == ideology]
            if ide.empty:
                continue
            ax.scatter(
                ide["economic"].mean(),
                ide["social"].mean(),
                s=65 if ideology == "base" else 42,
                marker="*" if ideology == "base" else "o",
                color=IDEOLOGY_COLORS[ideology],
                edgecolor="black",
                linewidth=0.4,
                label=IDEOLOGY_LABELS[ideology],
            )
        ax.set_title(panel_label, fontsize=10)
        ax.set_xlabel("Economic: left ← → right")
        ax.set_ylabel("Social: libertarian ← → authoritarian")
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    if title:
        fig.suptitle(title, y=1.01, fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def qwen_think_outcomes(chat: pd.DataFrame) -> pd.DataFrame:
    """Pair configs and classify binary target-quadrant changes.

    A rescue means ``no_think`` was outside the assigned persona's target
    quadrant and ``think`` was inside it. A harm is the reverse. This definition
    uses quadrant membership only; it is not a change in distance from the
    quadrant centre.
    """
    qwen = chat[chat["base_model"].isin(QWEN_MODELS)].copy()
    keys = ["base_model", "ideology", "lhs_row", "reasoning_mode", "context_id", "persona_class"]
    wide = qwen.pivot_table(
        index=keys,
        columns="protocol",
        values="target_quadrant_correct",
        aggfunc="first",
    ).dropna(subset=["think", "no_think"])
    no_think = wide["no_think"].astype(bool)
    think = wide["think"].astype(bool)
    wide["outcome"] = np.select(
        [~no_think & think, no_think & ~think, no_think & think],
        ["rescue", "harm", "both_correct"],
        default="both_incorrect",
    )
    return (
        wide.reset_index()
        .groupby(["base_model", "outcome"], observed=True)
        .size()
        .rename("configurations")
        .reset_index()
    )
