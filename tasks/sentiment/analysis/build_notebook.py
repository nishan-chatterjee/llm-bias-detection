#!/usr/bin/env python3
"""Build the compact, paper-facing IBM sentiment analysis notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks" / "01_ibm_sentiment_core.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# IBM topic sentiment: core analysis

This notebook intentionally excludes the target taxonomy and every
LLM-authored annotation. It uses only gold topic-sentiment labels, model
candidate scores, assigned persona, and randomized prompt factors."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ANALYSIS = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if not (ANALYSIS / "data").exists():
    ANALYSIS = Path.cwd() / "tasks" / "sentiment" / "analysis"
DATA = ANALYSIS / "data"
FIGURES = ANALYSIS / "figures"

coverage = pd.read_csv(DATA / "coverage_by_model.csv")
summary_model = pd.read_csv(DATA / "summary_by_model.csv")
summary_persona = pd.read_csv(DATA / "summary_by_model_persona.csv")
sensitivity = pd.read_csv(DATA / "factor_sensitivity_macro_f1.csv")
topics = pd.read_csv(DATA / "topic_descriptive_metrics.csv")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Coverage

The archived run should contain 1,800 configurations × 30 topics = 54,000
successful rows per model. Coverage is a prerequisite for the comparisons
below; it is not a performance metric."""
        ),
        nbf.v4.new_code_cell(
            """display(coverage)
assert coverage["coverage"].eq(1.0).all()
assert coverage["error_rows"].eq(0).all()
assert coverage["successful_rows"].sum() == 432_000"""
        ),
        nbf.v4.new_markdown_cell(
            """## Model and assigned-persona performance

Macro-F1 gives equal weight to the negative and positive labels. Each cell in
the heatmap pools 300 prompt configurations for each of the 30 topics (9,000
classifications). The base column has no political persona. The other columns
use the named assigned persona; they do not identify a model's own ideology.

In this run, the base condition has the highest macro-F1 for every model. The
gap is small for some model/persona combinations and much larger for others,
so the defensible conclusion is prompt-condition sensitivity rather than a
uniform family- or size-level law."""
        ),
        nbf.v4.new_code_cell(
            """display(summary_model[["model", "n", "accuracy", "macro_f1", "positive_pred_rate"]])
display(Image(filename=str(FIGURES / "ibm_sentiment_macro_f1_model_persona_heatmap.png")))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Prompt-factor sensitivity

Each named factor cell is a descriptive root-mean-square spread in
configuration-level macro-F1 between the factor's levels, calculated within
persona conditions. `Ideology SD` is the spread of the six persona-condition
means. `Residual RMSE` comes from an additive dummy-coded model. These values
show where measured robustness variation is concentrated; they are not causal
effect estimates and do not sum to total variance.

Instruction wording is the largest named prompt-factor component for six of
eight models. The ordering of the other components varies by checkpoint."""
        ),
        nbf.v4.new_code_cell(
            """display(sensitivity)
display(Image(filename=str(FIGURES / "ibm_sentiment_factor_sensitivity_macro_f1.png")))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Topic-level descriptive table

There is one gold label per topic, so per-topic macro-F1 would be misleading.
The released table instead reports accuracy, positive-prediction rate, and
mean candidate confidence for each model × persona × topic over 300 prompt
configurations. It supports item-level inspection without an added taxonomy."""
        ),
        nbf.v4.new_code_cell(
            """display(topics.head(12))
assert len(topics) == 8 * 6 * 30
assert topics.groupby(["model", "ideology"])["item_index"].nunique().eq(30).all()"""
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
