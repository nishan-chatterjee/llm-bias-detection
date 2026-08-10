#!/usr/bin/env python3
"""Build the compact, primary-model Political Compass analysis notebooks."""

from pathlib import Path
import textwrap

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


SETUP = """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANALYSIS_DIR = Path.cwd()
if not (ANALYSIS_DIR / 'core.py').exists():
    candidate = Path('tasks/political-compass/analysis').resolve()
    if (candidate / 'core.py').exists():
        ANALYSIS_DIR = candidate
    else:
        candidate = Path.cwd().parent
        if (candidate / 'core.py').exists():
            ANALYSIS_DIR = candidate
        else:
            raise FileNotFoundError('Run this notebook from the repository or analysis directory.')
sys.path.insert(0, str(ANALYSIS_DIR))

from core import (
    GEMMA_MODELS, QWEN_MODELS, PRIMARY_MODELS, IDEOLOGY_ORDER,
    load_mcq_configurations, load_chat_configurations, plot_centroid_grid,
    qwen_think_outcomes, size_label,
)

sns.set_theme(style='whitegrid')
"""


def make_notebook(title: str, intro: str, cells: list):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    notebook["cells"] = [md(f"# {title}\n\n{intro}"), code(SETUP), *cells]
    return notebook


mcq = make_notebook(
    "Political Compass MCQ — primary models",
    """
    This notebook uses the released configuration-level MCQ table. One row is
    one sampled prompt configuration after the 62 answer distributions have
    been mapped to an economic and social coordinate. It contains only the four
    Gemma and four Qwen checkpoints used in the paper.
    """,
    [
        code("""
        mcq = load_mcq_configurations()
        summary = pd.Series({
            'rows': len(mcq),
            'models': mcq['model'].nunique(),
            'quantizations': mcq['quantization'].nunique(),
            'ideologies': mcq['ideology'].nunique(),
            'languages': mcq['language'].nunique(),
            'unique config IDs': mcq['config_id'].nunique(),
        })
        display(summary.to_frame('value'))
        assert len(mcq) == 43_200
        assert set(mcq['model']) == set(PRIMARY_MODELS)
        """),
        md("""
        ## Persona centroids

        Each point below is the mean coordinate over all sampled prompt and
        quantization configurations for one assigned persona. The plot shows
        where the average lands; it does not imply that every configuration or
        proposition lands there.
        """),
        code("""
        panels = [(f'{model}|mcq', f'{model} MCQ') for model in PRIMARY_MODELS]
        plot_frame = mcq.rename(columns={'score_econ': 'economic', 'score_soc': 'social'})
        fig = plot_centroid_grid(plot_frame, panels, ncols=4, title='MCQ persona centroids')
        plt.show()
        """),
        md("""
        ## Factor-sensitivity summary

        A value is a standard-deviation-sized contribution in compass points
        after fitting each ideology condition separately. This is a measurement
        of variation associated with the sampled factor, not a universal claim
        about a family or checkpoint size.
        """),
        code("""
        sensitivity = pd.read_csv(ANALYSIS_DIR / 'data' / 'pct_factor_sensitivity.csv')
        display(sensitivity.head(12))
        """),
    ],
)


chat = make_notebook(
    "Political Compass chat — primary models",
    """
    This notebook uses the compact English chat configuration table. The rows
    cover four Gemma checkpoints plus Qwen think/no-think variants. Text-level
    diagnostics are kept in the separate qualitative-analysis notebooks.
    """,
    [
        code("""
        chat = load_chat_configurations()
        display(pd.Series({
            'rows': len(chat),
            'base models': chat['base_model'].nunique(),
            'model/protocol variants': chat['model_variant'].nunique(),
            'rows per variant': int(chat.groupby('model_variant').size().min()),
        }).to_frame('value'))
        assert len(chat) == 21_600
        assert set(chat['base_model']) == set(PRIMARY_MODELS)
        """),
        md("""
        ## Persona centroids in the required chat order

        Gemma checkpoints are shown by increasing size, followed by Qwen
        no-think and Qwen think checkpoints, each by increasing size.
        """),
        code("""
        panels = []
        panels += [(f'{m}|standard', f'Gemma Chat {size_label(m)}') for m in GEMMA_MODELS]
        panels += [(f'{m}|no_think', f'Qwen no-think {size_label(m)}') for m in QWEN_MODELS]
        panels += [(f'{m}|think', f'Qwen think {size_label(m)}') for m in QWEN_MODELS]
        fig = plot_centroid_grid(chat, panels, ncols=4, title='Chat persona centroids')
        plt.show()
        """),
        md("""
        ## What “rescue” means

        For matched Qwen configurations, a **rescue** is a binary transition:
        the no-think coordinate is outside the assigned persona's target
        quadrant and the think coordinate is inside it. A **harm** is the
        reverse. This definition uses quadrant membership; it is not distance
        from the quadrant centre and does not say that the rationale became
        better.
        """),
        code("""
        outcomes = qwen_think_outcomes(chat)
        outcomes['share_within_model'] = outcomes['configurations'] / outcomes.groupby('base_model')['configurations'].transform('sum')
        outcome_order = ['rescue', 'harm', 'both_correct', 'both_incorrect']
        display(outcomes.pivot(index='base_model', columns='outcome', values='share_within_model').reindex(QWEN_MODELS)[outcome_order].round(3))
        """),
    ],
)


comparison = make_notebook(
    "Political Compass MCQ–chat comparison — primary models",
    """
    This notebook compares English MCQ configuration coordinates with English
    chat coordinates. It is a protocol comparison, not a test of one universal
    family or size effect. Displays follow the paper ordering rule.
    """,
    [
        code("""
        mcq = load_mcq_configurations()
        mcq = mcq[mcq['language'].str.lower().eq('english')].rename(
            columns={'score_econ': 'economic', 'score_soc': 'social'}
        )
        chat = load_chat_configurations()
        display(pd.DataFrame({
            'source': ['MCQ (English)', 'Chat (English)'],
            'configuration rows': [len(mcq), len(chat)],
            'models': [mcq['base_model'].nunique(), chat['base_model'].nunique()],
        }))
        """),
        md("""
        ## Family-grouped method order

        The rows are Gemma MCQ, Gemma chat, Qwen MCQ, Qwen no-think chat, and
        Qwen think chat. Within every row, checkpoint size increases from left
        to right. Every marker is a persona centroid for that method/model.
        """),
        code("""
        combined = pd.concat([
            mcq[['base_model', 'protocol', 'ideology', 'economic', 'social']],
            chat[['base_model', 'protocol', 'ideology', 'economic', 'social']],
        ], ignore_index=True)
        panels = []
        panels += [(f'{m}|mcq', f'Gemma MCQ {size_label(m)}') for m in GEMMA_MODELS]
        panels += [(f'{m}|standard', f'Gemma Chat {size_label(m)}') for m in GEMMA_MODELS]
        panels += [(f'{m}|mcq', f'Qwen MCQ {size_label(m)}') for m in QWEN_MODELS]
        panels += [(f'{m}|no_think', f'Qwen no-think {size_label(m)}') for m in QWEN_MODELS]
        panels += [(f'{m}|think', f'Qwen think {size_label(m)}') for m in QWEN_MODELS]
        fig = plot_centroid_grid(combined, panels, ncols=4, title='English MCQ and chat persona centroids')
        plt.show()
        """),
        md("""
        ## Centroid displacement table

        The value below is the Euclidean distance between a persona's English
        MCQ centroid and its chat centroid. It summarizes a protocol-specific
        shift for that model/persona. A larger value does not identify why the
        shift happened and is not, by itself, a quality score.
        """),
        code("""
        mcq_centroids = mcq.groupby(['base_model', 'ideology'])[['economic', 'social']].mean()
        chat_centroids = chat.groupby(['base_model', 'protocol', 'ideology'])[['economic', 'social']].mean()
        rows = []
        for (model, protocol, ideology), point in chat_centroids.iterrows():
            if (model, ideology) not in mcq_centroids.index:
                continue
            ref = mcq_centroids.loc[(model, ideology)]
            rows.append({
                'model': model, 'protocol': protocol, 'ideology': ideology,
                'MCQ–chat centroid distance': float(np.linalg.norm(point.to_numpy() - ref.to_numpy())),
            })
        distance = pd.DataFrame(rows)
        display(distance.pivot_table(index=['model', 'protocol'], columns='ideology', values='MCQ–chat centroid distance').round(2))
        """),
    ],
)


OUT.mkdir(parents=True, exist_ok=True)
for name, notebook in {
    "01_mcq_analysis_primary_models.ipynb": mcq,
    "02_chat_analysis_primary_models.ipynb": chat,
    "03_mcq_chat_comparison_primary_models.ipynb": comparison,
}.items():
    path = OUT / name
    nbf.write(notebook, path)
    print(path)
