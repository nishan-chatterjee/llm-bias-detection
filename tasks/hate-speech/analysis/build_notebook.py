#!/usr/bin/env python3
"""Build the compact hate-speech aggregate analysis notebook."""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks" / "01_hate_speech_factor_sensitivity.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Hate-speech configuration sensitivity

This notebook analyzes the released 14,400-row configuration aggregate. Raw
item-level model outputs are pending the colleague handoff, so this notebook
does not claim to reproduce item-level classification metrics.

**Sensitive-content note:** the underlying corpus contains identity-targeted
hate speech. This aggregate contains target-group labels but no source text."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ANALYSIS = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if not (ANALYSIS / "data").exists():
    ANALYSIS = Path.cwd() / "tasks" / "hate-speech" / "analysis"
DATA = ANALYSIS / "data"
FIGURES = ANALYSIS / "figures"

scores = pd.read_csv(DATA / "hs_configuration_scores.csv")
sensitivity = pd.read_csv(DATA / "hs_factor_sensitivity.csv")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Aggregate completeness

The design has 300 sampled configurations for each of six persona conditions
and eight models. Each row is already averaged over the items belonging to the
configuration's sampled target group."""
        ),
        nbf.v4.new_code_cell(
            """display(scores.head())
display(scores.groupby("model").size().rename("configurations"))
assert len(scores) == 8 * 6 * 300
assert scores.groupby("model").size().eq(1800).all()
assert scores["target"].nunique() == 10"""
        ),
        nbf.v4.new_markdown_cell(
            """## What the sensitivity values mean

`target` is a data-composition block because each configuration samples one
identity target; it is not presented as a prompt manipulation. Within each
persona condition, additive models estimate spread associated with persona
wording, context wording, and instruction wording. `Ideology SD` is the spread
between persona-condition means. `Intrinsic Blur` comes from mean `p(1-p)`
across item probabilities, and `Residual Jitter` is unexplained
configuration-level spread.

The largest component differs by checkpoint. Instruction wording is especially
large for Gemma 3 1B, whereas context or the persona-condition spread is larger
for some other checkpoints. This supports prompt sensitivity and
checkpoint-specific heterogeneity; it does not support a uniform size or
family law."""
        ),
        nbf.v4.new_code_cell(
            """columns = [
    "model", "Mean P(Hate)", "Total Var SD", "Ideology SD",
    "persona_combo_sd", "context_combo_sd", "instr_combo_sd",
    "Intrinsic Blur", "Residual Jitter",
]
display(sensitivity[columns])
display(Image(filename=str(FIGURES / "hate_speech_factor_sensitivity.png")))"""
        ),
        nbf.v4.new_markdown_cell(
            """## Limits

The very small model-fit p-values are not the main scientific claim: there are
many repeated configurations, and statistical detectability does not imply a
large or general effect. The component magnitudes and their inconsistency
across checkpoints are the more useful descriptive result. Item-level AUC,
F1, calibration, and target-level metrics must be regenerated after the raw
outputs and exact corpus are supplied and validated."""
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
