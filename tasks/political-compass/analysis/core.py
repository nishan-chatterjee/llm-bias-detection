"""Shared primary-model analysis helpers for Political Compass notebooks."""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
REPO_ROOT = HERE.parents[2]

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


def analysis_table_path(name: str, path: Path | None = None) -> Path:
    """Resolve a downloaded analysis-ready table before the tracked fallback."""
    if path is not None:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return resolved
    release_root = Path(
        os.environ.get("LLM_BIAS_DATA_DIR", REPO_ROOT / "data" / "release")
    ).expanduser().resolve()
    candidates = [
        release_root / "data" / "analysis_ready" / "political_compass" / f"{name}.parquet",
        DATA_DIR / f"{name}.parquet",
        DATA_DIR / f"{name}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No analysis table {name!r}. Download it with "
        "`python scripts/download_dataset.py --component analysis`, or use the tracked fallback."
    )


def read_analysis_table(name: str, path: Path | None = None) -> pd.DataFrame:
    resolved = analysis_table_path(name, path)
    if resolved.suffix == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved)


def size_label(name: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)[Bb]", name)
    return f"{match.group(1)}B" if match else name


def load_mcq_configurations(path: Path | None = None) -> pd.DataFrame:
    frame = read_analysis_table("pct_configuration_scores", path)
    frame = frame[frame["model"].isin(PRIMARY_MODELS)].copy()
    frame["source"] = "MCQ"
    frame["base_model"] = frame["model"]
    frame["protocol"] = "mcq"
    return frame


def load_chat_configurations(path: Path | None = None) -> pd.DataFrame:
    frame = read_analysis_table("chat_configuration_scores", path)
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
    for panel_index, (ax, (panel_key, panel_label)) in enumerate(zip(axes.flat, panels)):
        model, protocol = panel_key.split("|", 1)
        sub = frame[(frame[model_col] == model) & (frame[protocol_col] == protocol)]
        draw_compass_background(ax)
        for ideology in IDEOLOGY_ORDER:
            ide = sub[sub["ideology"] == ideology]
            if ide.empty:
                continue
            ax.scatter(
                ide["economic"], ide["social"], s=7,
                color=IDEOLOGY_COLORS[ideology], alpha=0.10,
                edgecolor="none", rasterized=True,
            )
            if len(ide) >= 3:
                values = ide[["economic", "social"]].dropna().to_numpy(float)
                covariance = np.cov(values, rowvar=False)
                if np.isfinite(covariance).all():
                    eigvals, eigvecs = np.linalg.eigh(covariance)
                    eigvals = np.maximum(eigvals, 0)
                    order = eigvals.argsort()[::-1]
                    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                    ellipse = patches.Ellipse(
                        (values[:, 0].mean(), values[:, 1].mean()),
                        width=2 * np.sqrt(eigvals[0]),
                        height=2 * np.sqrt(eigvals[1]),
                        angle=angle,
                        fill=False,
                        color=IDEOLOGY_COLORS[ideology],
                        linewidth=0.8,
                        alpha=0.75,
                    )
                    ax.add_patch(ellipse)
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
        row, column = divmod(panel_index, ncols)
        ax.set_xlabel("Economic axis" if row == nrows - 1 else "")
        ax.set_ylabel("Social axis" if column == 0 else "")
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


def descriptive_factor_spread(
    frame: pd.DataFrame,
    score_columns: tuple[str, ...],
    factors: tuple[str, ...],
    model_columns: tuple[str, ...] = ("base_model", "protocol"),
) -> pd.DataFrame:
    """Return within-persona RMS between-level spread for released tables.

    This is deliberately descriptive.  For each method, persona, score and
    factor, it measures the weighted spread of factor-level means around the
    persona mean and then pools those spreads over personas.  It does not
    assume that factors are causal or mutually independent.
    """
    rows = []
    for keys, method in frame.groupby(list(model_columns), observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = dict(zip(model_columns, keys))
        for score in score_columns:
            result[f"{score}: total SD"] = method[score].std(ddof=0)
            result[f"{score}: persona SD"] = (
                method.groupby("ideology", observed=True)[score].mean().std(ddof=0)
            )
            for factor in factors:
                ss = 0.0
                n = 0
                for _, subset in method.groupby("ideology", observed=True):
                    means = subset.groupby(factor, observed=True, dropna=False)[score].mean()
                    counts = subset.groupby(factor, observed=True, dropna=False)[score].count()
                    grand = subset[score].mean()
                    ss += float((((means - grand) ** 2) * counts).sum())
                    n += int(counts.sum())
                result[f"{score}: {factor}"] = np.sqrt(ss / n) if n else np.nan
        rows.append(result)
    return pd.DataFrame(rows)


def plot_sensitivity_heatmap(
    table: pd.DataFrame,
    value_prefix: str,
    index: list[str],
    title: str,
    figsize: tuple[float, float] = (13, 6),
):
    """Plot one score's columns from :func:`descriptive_factor_spread`."""
    columns = [column for column in table if column.startswith(value_prefix + ":")]
    display = table.set_index(index)[columns]
    if index == ["base_model", "protocol"]:
        desired = [
            *((model, "standard") for model in GEMMA_MODELS),
            *((model, protocol) for model in QWEN_MODELS for protocol in ("no_think", "think")),
        ]
        display = display.reindex(pd.MultiIndex.from_tuples(desired, names=index)).dropna(how="all")
    display.columns = [column.split(": ", 1)[1].replace("_", " ").title() for column in columns]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(display, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Descriptive SD-sized component")
    ax.set_ylabel(" / ".join(index))
    fig.tight_layout()
    return fig
