#!/usr/bin/env python3
"""Build the complete released hate-speech analysis notebook."""

from pathlib import Path
import textwrap
import nbformat as nbf


HERE=Path(__file__).resolve().parent
OUT=HERE/'notebooks'/'01_hate_speech_analysis_primary_models.ipynb'


def md(text): return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text): return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def main():
    nb=nbf.v4.new_notebook()
    nb['metadata']={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3'},
    }
    nb['cells']=[
        md("""# Hate-speech analysis — primary models

        This notebook analyzes the eight primary Gemma 3 IT and Qwen3
        checkpoints. It uses compact summaries derived from 19,180,800 released
        item predictions, plus the exact 14,400-row configuration aggregate.

        **Sensitive-content notice:** the source corpus contains identity-targeted
        hate speech. Historical prediction Parquets do not contain source statements, but
        they do contain target-group labels and source metadata. After
        `scripts/download_dataset.py --component analysis`, the notebook reads
        the pinned Hugging Face analysis tables directly."""),
        code("""
        from pathlib import Path
        import sys
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display

        ANALYSIS=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
        if not (ANALYSIS/'data_access.py').exists(): ANALYSIS=Path.cwd()/'tasks'/'hate-speech'/'analysis'
        if not (ANALYSIS/'data_access.py').exists(): raise FileNotFoundError('Run from the repository root or hate-speech analysis directory.')
        sys.path.insert(0,str(ANALYSIS))
        from data_access import read_table, table_path
        names=['hs_configuration_scores','hs_factor_sensitivity','hs_item_metrics_by_persona_target','hs_calibration_by_model']
        source=table_path(names[0]).parent
        try: source=source.relative_to(ANALYSIS.parents[2])
        except ValueError: pass
        print('Hate-speech analysis source:', source)
        scores,sensitivity,metrics,calibration=[read_table(name) for name in names]
        model_order=['gemma-3-1b-it','gemma-3-4b-it','gemma-3-12b-it','gemma-3-27b-it',
                     'Qwen3-4B','Qwen3-8B','Qwen3-14B','Qwen3-32B']
        ideology_order=['base','centrism','libertarian_left','libertarian_right',
                        'authoritarian_left','authoritarian_right']
        sns.set_theme(style='whitegrid')
        """),
        md("""## Completeness and design

        The design has 300 configurations per model/persona block. Each
        configuration selects one of ten identity targets and scores 1,332
        target-specific item positions. The historical CSV header was reused
        positionally across targets, so `item_index` is a within-target position,
        not a global question ID."""),
        code("""
        display(pd.Series({'configuration rows':len(scores),'models':scores.model.nunique(),
                           'personas':scores.ideology.nunique(),'targets':scores.target.nunique(),
                           'released item predictions':19_180_800}).to_frame('value'))
        display(scores.groupby(['model','ideology'],observed=True).size().unstack())
        assert len(scores)==8*6*300
        assert scores.groupby(['model','ideology']).size().eq(300).all()
        assert len(metrics)==8*6*10 and len(calibration)==8*10
        """),
        md("""## Configuration-level P(hate)

        Each row below is a configuration mean over its 1,332 items. Differences
        combine prompt sensitivity, persona assignment and the sampled target;
        they are not statements about the prevalence of hate speech in a group."""),
        code("""
        scores['model_alias']=scores.model.str.split('/').str[-1]
        config_summary=(scores.groupby(['model_alias','ideology'],observed=True)
                        .agg(configurations=('mean_p_hate','size'),mean_p_hate=('mean_p_hate','mean'),
                             sd_p_hate=('mean_p_hate','std'),mean_item_dispersion=('mean_item_dispersion','mean'))
                        .reset_index())
        display(config_summary.pivot(index='model_alias',columns='ideology',values='mean_p_hate')
                .reindex(model_order).reindex(columns=ideology_order).round(3))
        """),
        md("""## Item-level classification performance

        The historical run stores candidate-normalized probabilities for
        literal `True` and `False`. At threshold 0.5, the table reports accuracy,
        precision, recall and F1. These are classification diagnostics, not a
        toxicity score for a persona or identity target."""),
        code("""
        overall=(metrics.groupby('model_alias',observed=True)
                 .agg(n=('n','sum'),tp=('tp','sum'),tn=('tn','sum'),fp=('fp','sum'),fn=('fn','sum')))
        overall['accuracy']=(overall.tp+overall.tn)/overall.n
        overall['precision']=overall.tp/(overall.tp+overall.fp)
        overall['recall']=overall.tp/(overall.tp+overall.fn)
        overall['f1']=2*overall.precision*overall.recall/(overall.precision+overall.recall)
        display(overall.reindex(model_order)[['n','accuracy','precision','recall','f1']].round(3))
        """),
        md("""## Performance by assigned persona

        Metrics are micro-aggregated from target-specific counts. Variation is
        model-specific; a persona name describes the prompt condition and does
        not identify the model itself."""),
        code("""
        persona=(metrics.groupby(['model_alias','ideology'],observed=True)
                 .agg(n=('n','sum'),tp=('tp','sum'),tn=('tn','sum'),fp=('fp','sum'),fn=('fn','sum')))
        persona['accuracy']=(persona.tp+persona.tn)/persona.n
        persona['precision']=persona.tp/(persona.tp+persona.fp)
        persona['recall']=persona.tp/(persona.tp+persona.fn)
        persona['f1']=2*persona.precision*persona.recall/(persona.precision+persona.recall)
        f1=persona.f1.unstack().reindex(model_order).reindex(columns=ideology_order)
        display(f1.round(3))
        fig,ax=plt.subplots(figsize=(10,5)); sns.heatmap(f1,annot=True,fmt='.3f',cmap='YlGnBu',ax=ax)
        ax.set_title('Hate-speech F1 by model and assigned persona'); fig.tight_layout(); plt.show()
        """),
        md("""## Target-specific diagnostics

        Targets identify which subset of the evaluation data was sampled. The
        heatmap shows F1 for each model/target after pooling persona conditions.
        Differences can reflect the corpus composition and should not be
        interpreted as properties of the identity groups."""),
        code("""
        target=(metrics.groupby(['model_alias','target'],observed=True)
                .agg(n=('n','sum'),tp=('tp','sum'),tn=('tn','sum'),fp=('fp','sum'),fn=('fn','sum')))
        target['precision']=target.tp/(target.tp+target.fp)
        target['recall']=target.tp/(target.tp+target.fn)
        target['f1']=2*target.precision*target.recall/(target.precision+target.recall)
        target_f1=target.f1.unstack().reindex(model_order)
        fig,ax=plt.subplots(figsize=(13,5)); sns.heatmap(target_f1,annot=True,fmt='.2f',cmap='mako',ax=ax)
        ax.set_title('Hate-speech F1 by model and sampled target subset'); fig.tight_layout(); plt.show()
        """),
        md("""## Probability separation and calibration

        `mean_p_hate_gold` and `mean_p_hate_nonhate` summarize score separation.
        The calibration plot groups predictions into ten probability bins; good
        calibration lies near the diagonal. This is supported without raw
        vocabulary logits."""),
        code("""
        separation=(metrics.groupby('model_alias',observed=True)
                    .agg(mean_p_hate_gold=('mean_p_hate_gold','mean'),
                         mean_p_hate_nonhate=('mean_p_hate_nonhate','mean')).reindex(model_order))
        separation['mean separation']=separation.mean_p_hate_gold-separation.mean_p_hate_nonhate
        display(separation.round(3))
        fig,ax=plt.subplots(figsize=(7,6)); ax.plot([0,1],[0,1],'--',color='gray',label='perfect calibration')
        for model in model_order:
            sub=calibration[calibration.model_alias.eq(model)]
            ax.plot(sub.mean_p_hate,sub.observed_hate_rate,marker='o',label=model)
        ax.set(xlabel='Mean predicted P(hate)',ylabel='Observed hate rate',title='Calibration by model')
        ax.legend(bbox_to_anchor=(1.02,1),loc='upper left'); fig.tight_layout(); plt.show()
        """),
        md("""## Prompt-factor sensitivity

        `target` is treated as a data-composition block, not a prompt factor.
        Persona wording, context and instruction components are calculated
        within persona conditions. Magnitudes are more informative here than
        tiny p-values produced by the large repeated design."""),
        code("""
        columns=['model','Mean P(Hate)','Total Var SD','Ideology SD','persona_combo_sd',
                 'context_combo_sd','instr_combo_sd','Intrinsic Blur','Residual Jitter']
        display(sensitivity[columns].round(3))
        heat=(sensitivity.set_index('model').reindex(model_order)[
              ['Ideology SD','persona_combo_sd','context_combo_sd','instr_combo_sd','Intrinsic Blur','Residual Jitter']]
              .rename(columns={'persona_combo_sd':'Persona','context_combo_sd':'Context','instr_combo_sd':'Instruction'}))
        fig,ax=plt.subplots(figsize=(11,5.5)); sns.heatmap(heat,annot=True,fmt='.3f',cmap='viridis',ax=ax)
        ax.set(title='Hate-speech P(hate): SD-sized sensitivity components',
               xlabel='Descriptive variation component',ylabel='Model')
        fig.tight_layout(); plt.show()
        """),
        md("""## What is not reproduced here

        The historical `visuals_offensive_speech.ipynb` combined this task with
        a separate offensive-speech experiment and included raw-logit temperature
        transforms. The offensive dataset is not part of this paper release and
        the handoff contains no raw vocabulary logits. Those cells and their
        preserved old renderings are stored only under `legacy/`; they are not
        presented as runnable evidence.

        Historical prediction Parquets omit source statements and prompt hashes.
        The separately released `hate_speech_inputs` table and original corpus/
        prompt files now permit inspection and fresh inference. Join historical
        rows to inputs by `(target, item_index)`. All labels and source metadata
        match at these positions, but historical byte identity cannot be proved
        because text/prompt hashes were not recorded. Source statements contain
        sensitive language; these diagnostics do not print them by default."""),
    ]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(nb,OUT)
    print(OUT)
    # Keep the historical short entry notebook runnable rather than displaying
    # a cached PNG as if it were a newly computed result.
    short=nbf.v4.new_notebook(metadata=nb.metadata)
    short.cells=[md('''# Hate-speech factor sensitivity — primary models

    This compact entry uses the same portable data and computed heatmap as the
    complete notebook. Components are descriptive SD-sized spreads, not
    independent causal variance components.'''),nb.cells[1],nb.cells[-3],nb.cells[-2]]
    nbf.write(short,OUT.parent/'01_hate_speech_factor_sensitivity.ipynb')


if __name__=='__main__': main()
