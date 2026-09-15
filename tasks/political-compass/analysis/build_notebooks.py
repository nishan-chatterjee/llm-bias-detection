#!/usr/bin/env python3
"""Build complete, primary-model Political Compass analysis notebooks."""

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
    candidates = [Path('tasks/political-compass/analysis').resolve(), Path.cwd().parent]
    ANALYSIS_DIR = next((path for path in candidates if (path / 'core.py').exists()), None)
if ANALYSIS_DIR is None:
    raise FileNotFoundError('Run from the repository root or tasks/political-compass/analysis.')
sys.path.insert(0, str(ANALYSIS_DIR))

from core import (
    GEMMA_MODELS, QWEN_MODELS, PRIMARY_MODELS, IDEOLOGY_ORDER,
    load_mcq_configurations, load_chat_configurations, method_order,
    plot_centroid_grid, qwen_think_outcomes, descriptive_factor_spread,
    plot_sensitivity_heatmap, size_label, analysis_table_path,
    read_analysis_table,
)

sns.set_theme(style='whitegrid', context='notebook')
def source_label(path):
    try: return path.relative_to(ANALYSIS_DIR.parents[2])
    except ValueError: return path
print('Political Compass analysis source:')
print('  MCQ:', source_label(analysis_table_path('pct_configuration_scores')))
print('  chat:', source_label(analysis_table_path('chat_configuration_scores')))
"""


def notebook(title: str, intro: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [md(f"# {title}\n\n{intro}"), code(SETUP), *cells]
    return nb


mcq = notebook(
    "Political Compass MCQ analysis — primary models",
    """
    This is the complete release analysis for the paper's four Gemma 3 IT and
    four Qwen3 checkpoints. It uses the released configuration-score table,
    where each row is one sampled prompt configuration after its 62 response
    distributions have been mapped to economic and social coordinates.

    The generated scorer is not redistributed. Consequently, this notebook
    starts from the released coordinates and does not pretend to reconstruct
    them from candidate probabilities. The separate instrument notebook shows
    how the scorer was reconstructed locally. When `data/release/` is present,
    the notebook reads the pinned Hugging Face `analysis_ready` tables; it
    otherwise uses the byte-equivalent tracked compact tables.
    """,
    [
        md("""## Coverage and design balance

        The MCQ experiment crosses eight models, three quantizations, fourteen
        languages and six persona conditions. Every model/persona block has
        the same number of configurations."""),
        code("""
        mcq = load_mcq_configurations()
        coverage = pd.DataFrame({
            'value': [len(mcq), mcq['model'].nunique(), mcq['quantization'].nunique(),
                      mcq['language'].nunique(), mcq['ideology'].nunique(), mcq['config_id'].nunique()],
        }, index=['configuration rows', 'models', 'quantizations', 'languages', 'personas', 'unique config IDs'])
        display(coverage)
        display(mcq.groupby(['model', 'ideology'], observed=True).size().unstack())
        assert len(mcq) == 43_200
        assert set(mcq['model']) == set(PRIMARY_MODELS)
        assert mcq.groupby(['model', 'ideology']).size().eq(900).all()
        """),
        md("""## Configuration distributions and persona centroids

        Small transparent points are individual prompt configurations. Large
        markers are persona means. Ellipses summarize one standard deviation
        along the covariance axes of the
        configuration cloud; they are descriptive, not confidence intervals.
        The plot averages over languages and quantizations."""),
        code("""
        panels = [(f'{model}|mcq', f'{model} MCQ') for model in PRIMARY_MODELS]
        plot_frame = mcq.rename(columns={'score_econ': 'economic', 'score_soc': 'social'})
        fig = plot_centroid_grid(plot_frame, panels, ncols=4, title='MCQ persona centroids')
        plt.show()
        """),
        md("""## Baseline displacement and persona separation

        `distance_from_origin` is the distance of a condition mean from the
        compass origin. `persona_spread` is the mean distance between each
        non-base persona centroid and that model's base centroid. These are
        compact summaries of this experiment, not quality scores."""),
        code("""
        centroids = (mcq.groupby(['model', 'ideology'], observed=True)[['score_econ', 'score_soc']]
                     .mean().reset_index())
        centroids['distance_from_origin'] = np.hypot(centroids['score_econ'], centroids['score_soc'])
        base = centroids[centroids.ideology.eq('base')].set_index('model')[['score_econ', 'score_soc']]
        shifted = centroids[~centroids.ideology.eq('base')].copy()
        shifted['distance_from_base'] = [
            float(np.linalg.norm(row[['score_econ','score_soc']].to_numpy(float) - base.loc[row.model].to_numpy(float)))
            for _, row in shifted.iterrows()
        ]
        summary = (shifted.groupby('model', observed=True)['distance_from_base'].mean()
                   .rename('mean persona-centroid distance from base').to_frame()
                   .join(centroids[centroids.ideology.eq('base')].set_index('model')['distance_from_origin']))
        display(summary.reindex(PRIMARY_MODELS).round(3))
        """),
        md("""## Language-specific position variation

        The table reports the largest distance between a language-specific
        persona centroid and that model/persona's overall centroid. A large
        entry means language matters somewhere in that block; it does not say
        that every language shifts in the same direction."""),
        code("""
        lang = mcq.groupby(['model','ideology','language'], observed=True)[['score_econ','score_soc']].mean()
        overall = mcq.groupby(['model','ideology'], observed=True)[['score_econ','score_soc']].mean()
        lang = lang.join(overall, rsuffix='_overall')
        lang['centroid_shift'] = np.hypot(lang.score_econ-lang.score_econ_overall,
                                         lang.score_soc-lang.score_soc_overall)
        display(lang.groupby(['model','ideology'])['centroid_shift'].max().unstack().reindex(PRIMARY_MODELS).round(2))
        """),
        md("""## Factor sensitivity

        Each factor cell is a standard-deviation-sized contribution in compass
        points from within-persona Type-III ANOVA fits. Stars and p-values are
        intentionally not used as the story: with many configurations, tiny
        differences are readily detectable. Compare magnitudes and recurrence
        instead. `Response Entropy` represents candidate-probability blur."""),
        code("""
        sensitivity = read_analysis_table('pct_factor_sensitivity')
        magnitude_columns = ['Ideology','Total Variation','Residual Jitter','Response Entropy',
                             'quantization_sd','language_sd','context_combo_sd','instr_combo_sd',
                             'key_type_sd','perm_id_sd','persona_combo_sd']
        for axis in ['econ','soc']:
            display(sensitivity[sensitivity.axis.eq(axis)].set_index('model')[magnitude_columns]
                    .reindex(PRIMARY_MODELS).round(2))
        """),
        md("""## Quantization and temperature diagnostics

        The released experiment directly supports bf16/8-bit/4-bit comparisons.
        It does **not** directly run ten decoding temperatures. The old research
        notebook's temperature curves were post-hoc transformations of stored
        probabilities, not additional model generations, so they are retained
        only in the historical notebook under `legacy/` and are not presented
        here as an experimental temperature result."""),
        code("""
        quant = (mcq.groupby(['model','quantization','ideology'], observed=True)[['score_econ','score_soc']]
                 .mean().reset_index())
        bf16 = quant[quant.quantization.eq('bf16')].set_index(['model','ideology'])
        rows=[]
        for q in ['8bit','4bit']:
            other=quant[quant.quantization.eq(q)].set_index(['model','ideology'])
            delta=np.hypot(other.score_econ-bf16.score_econ, other.score_soc-bf16.score_soc)
            rows.extend({'model':m,'quantization':q,'mean centroid shift from bf16':v}
                        for m,v in delta.groupby(level='model').mean().items())
        display(pd.DataFrame(rows).pivot(index='model',columns='quantization',
                                         values='mean centroid shift from bf16').reindex(PRIMARY_MODELS).round(3))
        """),
        md("""## Interpretation boundary

        This notebook establishes model-, persona-, language-, quantization-,
        and prompt-factor heterogeneity within the sampled experiment. It does
        not establish one universally hardest quadrant or a monotonic family or
        size law. Proposition-local chat behavior is analyzed separately."""),
    ],
)


chat = notebook(
    "Political Compass chat analysis — primary models",
    """
    This complete release notebook covers four Gemma chat variants and the four
    primary Qwen checkpoints in matched think/no-think protocols. It uses the
    released 21,600-row configuration table; text-level proposition diagnostics
    remain in `qualitative-analysis/`. When available, the table is read from
    the pinned Hugging Face `analysis_ready` download.
    """,
    [
        md("""## Coverage and protocol balance

        Every displayed model/protocol/persona block contains 300 configurations.
        The original exploratory notebook also displayed GaMS and Qwen 0.6B/1.7B;
        those are outside the paper model set and are intentionally absent."""),
        code("""
        chat = load_chat_configurations()
        counts = chat.groupby(['base_model','protocol','ideology'], observed=True).size()
        display(pd.Series({'configuration rows':len(chat), 'base models':chat.base_model.nunique(),
                           'model/protocol variants':chat.model_variant.nunique(),
                           'minimum block size':counts.min(), 'maximum block size':counts.max()}).to_frame('value'))
        assert len(chat)==21_600 and counts.eq(300).all()
        assert set(chat.base_model)==set(PRIMARY_MODELS)
        """),
        md("""## Configuration distributions and persona centroids

        Gemma sizes are grouped first, then Qwen no-think sizes, then Qwen
        think sizes. Points are configurations; markers are persona means; the
        ellipses show one standard deviation along covariance axes."""),
        code("""
        panels=[]
        panels += [(f'{m}|standard', f'Gemma Chat {size_label(m)}') for m in GEMMA_MODELS]
        panels += [(f'{m}|no_think', f'Qwen no-think {size_label(m)}') for m in QWEN_MODELS]
        panels += [(f'{m}|think', f'Qwen think {size_label(m)}') for m in QWEN_MODELS]
        fig=plot_centroid_grid(chat,panels,ncols=4,title='Chat persona centroids')
        plt.show()
        """),
        md("""## Response length and classification diagnostics

        Stage-1 tokens measure the visible response before the final candidate
        scoring step. Entropy measures the candidate distribution in Stage 2.
        Neither is a direct measure of rationale quality."""),
        code("""
        diagnostic=(chat.groupby(['base_model','protocol'],observed=True)
                    .agg(configurations=('lhs_row','size'), mean_stage1_tokens=('mean_stage1_tokens','mean'),
                         median_visible_words=('mean_visible_words','median'), mean_entropy=('mean_entropy','mean'),
                         mean_target_alignment=('target_alignment_rate','mean')))
        display(diagnostic.round(3))
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        sns.boxplot(data=chat,x='model_variant',y='mean_stage1_tokens',showfliers=False,ax=axes[0])
        axes[0].tick_params(axis='x',rotation=70); axes[0].set_title('Stage-1 token length')
        sns.boxplot(data=chat,x='model_variant',y='mean_entropy',showfliers=False,ax=axes[1])
        axes[1].tick_params(axis='x',rotation=70); axes[1].set_title('Stage-2 candidate entropy')
        fig.tight_layout(); plt.show()
        """),
        md("""## Explicit Stage-1 stance versus Stage-2 classification

        The parser searches the final visible lines of Stage 1 for one
        unambiguous answer label or key, maps both stages back to the canonical
        four choices, and compares them only when both are recoverable. Thus,
        `conditional agreement` is agreement divided by comparable rows—not by
        all generated rows. This checks the answer-mapping step; it does not
        establish that the explanation is faithful or high quality."""),
        code("""
        agreement=read_analysis_table('chat_stage_agreement_summary')
        display(agreement[['model_variant','rows','comparable_rows','comparable_rate',
                           'conditional_agreement_rate']].round(4))
        overall=agreement.agreement_rows.sum()/agreement.comparable_rows.sum()
        print(f'Overall conditional agreement: {overall:.3%} '
              f'({agreement.agreement_rows.sum():,}/{agreement.comparable_rows.sum():,})')
        assert len(agreement)==12 and np.isclose(overall,0.950929,atol=5e-7)
        """),
        md("""## Prompt-factor sensitivity

        These are descriptive within-persona spreads of factor-level means.
        Missing factor dimensions are left out rather than filled with zeros.
        The values do not sum to total variation and should not be interpreted
        causally."""),
        code("""
        chat_factors=('reasoning_mode','context_id','persona_class')
        spread=descriptive_factor_spread(chat,('economic','social'),chat_factors)
        display(spread.round(3))
        plot_sensitivity_heatmap(spread,'economic',['base_model','protocol'],
                                 'Chat economic-coordinate sensitivity',(12,7)); plt.show()
        plot_sensitivity_heatmap(spread,'social',['base_model','protocol'],
                                 'Chat social-coordinate sensitivity',(12,7)); plt.show()
        """),
        md("""## Matched Qwen think/no-think shifts

        The table uses matched LHS configuration keys. Positive coordinate
        deltas mean think mode moved right/up relative to no-think; signs vary
        by persona, so an overall average is insufficient."""),
        code("""
        keys=['base_model','ideology','lhs_row','reasoning_mode','context_id','persona_class']
        wide=chat[chat.base_model.isin(QWEN_MODELS)].pivot_table(index=keys,columns='protocol',
                 values=['economic','social','mean_stage1_tokens','mean_entropy'],aggfunc='first').dropna()
        shifts=pd.DataFrame(index=wide.index)
        for metric in ['economic','social','mean_stage1_tokens','mean_entropy']:
            shifts[metric+'_delta_think_minus_no_think']=wide[(metric,'think')]-wide[(metric,'no_think')]
        display(shifts.reset_index().groupby(['base_model','ideology'],observed=True)
                .mean(numeric_only=True).round(3))
        """),
        md("""## Quadrant rescues and harms

        A rescue means no-think was outside the assigned persona quadrant and
        think was inside it. A harm is the reverse. This is binary quadrant
        membership—not distance from a quadrant centre, response correctness,
        or rationale quality."""),
        code("""
        outcomes=qwen_think_outcomes(chat)
        outcomes['share']=outcomes.configurations/outcomes.groupby('base_model').configurations.transform('sum')
        order=['rescue','harm','both_correct','both_incorrect']
        display(outcomes.pivot(index='base_model',columns='outcome',values='share').reindex(QWEN_MODELS)[order].round(3))
        """),
        md("""## Interpretation boundary

        Think mode produces recurring rescues and harms, with directions that
        depend on model and persona/proposition exposure. These diagnostics do
        not support a uniformly beneficial protocol or a simple monotonic size
        effect."""),
    ],
)


comparison = notebook(
    "Political Compass MCQ–chat comparison — primary models",
    """
    This notebook compares English MCQ and chat configuration coordinates for
    the paper's eight primary checkpoints. Displays follow the paper ordering:
    Gemma MCQ then chat by size; Qwen MCQ, no-think chat and think chat by size.
    Both inputs resolve from the pinned Hugging Face `analysis_ready` download
    before falling back to the compact copies tracked with the code.
    """,
    [
        md("""## Coverage and comparable methods"""),
        code("""
        mcq=load_mcq_configurations()
        mcq=mcq[mcq.language.str.lower().eq('english')].rename(columns={'score_econ':'economic','score_soc':'social'})
        chat=load_chat_configurations()
        display(pd.DataFrame({'method':['MCQ English','Chat English'],
                              'configuration rows':[len(mcq),len(chat)],
                              'base models':[mcq.base_model.nunique(),chat.base_model.nunique()]}))
        # Language is sampled by the Latin-hypercube design rather than crossed
        # exhaustively, so the English MCQ subset contains 3,024 rows.
        assert len(mcq)==3_024 and len(chat)==21_600
        """),
        md("""## Family-grouped configuration maps

        Every subplot uses the same −10…10 axes. Individual points show prompt
        configurations; large markers show assigned-persona centroids; ellipses
        show one standard deviation along covariance axes."""),
        code("""
        combined=pd.concat([mcq[['base_model','protocol','ideology','economic','social']],
                            chat[['base_model','protocol','ideology','economic','social']]],ignore_index=True)
        panels=[(f'{m}|{p}',label) for m,p,label in method_order()]
        fig=plot_centroid_grid(combined,panels,ncols=4,title='English MCQ and chat persona centroids')
        plt.show()
        """),
        md("""## Persona-centroid displacement

        Each cell is the Euclidean distance between the English MCQ persona
        centroid and its chat counterpart. Larger values identify a larger
        protocol shift for this model/persona; they are not quality scores."""),
        code("""
        mcq_cent=mcq.groupby(['base_model','ideology'])[['economic','social']].mean()
        chat_cent=chat.groupby(['base_model','protocol','ideology'])[['economic','social']].mean()
        rows=[]
        for (model,protocol,ideology),point in chat_cent.iterrows():
            ref=mcq_cent.loc[(model,ideology)]
            rows.append({'model':model,'protocol':protocol,'ideology':ideology,
                         'distance':float(np.linalg.norm(point.to_numpy()-ref.to_numpy())),
                         'economic_delta':point.economic-ref.economic,
                         'social_delta':point.social-ref.social})
        displacement=pd.DataFrame(rows)
        display(displacement.pivot_table(index=['model','protocol'],columns='ideology',values='distance').round(2))
        """),
        md("""## Axis-specific protocol shifts

        Signed deltas preserve direction: positive economic values move right;
        positive social values move upward. This table makes clear when a large
        distance is economic, social, or both."""),
        code("""
        axis_summary=(displacement.groupby(['model','protocol'],observed=True)
                      .agg(mean_distance=('distance','mean'),
                           mean_abs_economic_shift=('economic_delta',lambda x:np.abs(x).mean()),
                           mean_abs_social_shift=('social_delta',lambda x:np.abs(x).mean()),
                           max_persona_distance=('distance','max')))
        display(axis_summary.round(3))
        """),
        md("""## Shared factor sensitivity

        MCQ and chat do not expose identical factors. The table compares only
        descriptive spreads available in each released configuration table and
        leaves non-applicable factors absent. Do not read white/missing cells
        as zero sensitivity."""),
        code("""
        mcq_for_spread=mcq.copy()
        mcq_for_spread['context_id']=mcq_for_spread.context_combo
        mcq_for_spread['persona_class']=mcq_for_spread.persona_combo
        mcq_spread=descriptive_factor_spread(mcq_for_spread,('economic','social'),
                     ('context_id','instr_combo','key_type','perm_id','persona_class'))
        chat_spread=descriptive_factor_spread(chat,('economic','social'),
                     ('reasoning_mode','context_id','persona_class'))
        sensitivity=pd.concat([mcq_spread,chat_spread],ignore_index=True,sort=False)
        display(sensitivity.round(3))
        """),
        md("""## Exact matched Qwen protocol comparison

        Think/no-think configurations share their sampled LHS keys. MCQ uses a
        different experimental design, so MCQ–chat comparisons above are made
        at persona-centroid level rather than falsely pairing individual rows."""),
        code("""
        keys=['base_model','ideology','lhs_row','reasoning_mode','context_id','persona_class']
        paired=chat[chat.base_model.isin(QWEN_MODELS)].pivot_table(index=keys,columns='protocol',
                    values=['economic','social','target_quadrant_correct'],aggfunc='first').dropna()
        result=[]
        for model in QWEN_MODELS:
            sub=paired.xs(model,level='base_model')
            result.append({'model':model,'matched configurations':len(sub),
                           'mean economic shift':(sub[('economic','think')]-sub[('economic','no_think')]).mean(),
                           'mean social shift':(sub[('social','think')]-sub[('social','no_think')]).mean(),
                           'think quadrant rate':sub[('target_quadrant_correct','think')].mean(),
                           'no-think quadrant rate':sub[('target_quadrant_correct','no_think')].mean()})
        display(pd.DataFrame(result).set_index('model').round(3))
        """),
        md("""## Interpretation boundary

        Protocol differences are large in some model/persona cells and modest
        in others. The analysis supports method-specific heterogeneity; it does
        not justify a universal MCQ-versus-chat or think-versus-no-think law."""),
    ],
)


OUT.mkdir(parents=True, exist_ok=True)
for name, nb in {
    "01_mcq_analysis_primary_models.ipynb": mcq,
    "02_chat_analysis_primary_models.ipynb": chat,
    "03_mcq_chat_comparison_primary_models.ipynb": comparison,
}.items():
    path = OUT / name
    nbf.write(nb, path)
    print(path)
