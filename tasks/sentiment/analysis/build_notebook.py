#!/usr/bin/env python3
"""Build the complete annotation-free IBM topic-sentiment notebook."""

from pathlib import Path
import textwrap
import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks" / "01_ibm_sentiment_core.ipynb"


def md(text): return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
def code(text): return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [
        md("""# IBM topic sentiment — complete primary-model analysis

        This notebook uses only fields produced by the IBM sentiment experiment:
        gold topic sentiment, model predictions and candidate probabilities,
        assigned persona, and randomized prompt factors. It contains the paper's
        four Gemma 3 IT and four Qwen3 checkpoints.

        The exploratory LLM-authored target taxonomy and its downstream figures
        are deliberately excluded. They are not needed to reproduce the paper's
        core sentiment diagnostics and would add an unvalidated annotation layer.
        After `scripts/download_dataset.py --component analysis`, every table is
        read from the pinned Hugging Face snapshot; otherwise the notebook uses
        the byte-equivalent compact tables tracked with the code."""),
        code("""
        from pathlib import Path
        import sys
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import display

        ANALYSIS=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
        if not (ANALYSIS/'data_access.py').exists(): ANALYSIS=Path.cwd()/'tasks'/'sentiment'/'analysis'
        if not (ANALYSIS/'data_access.py').exists(): raise FileNotFoundError('Run from the repository root or sentiment analysis directory.')
        sys.path.insert(0,str(ANALYSIS))
        from data_access import read_table, table_path
        names=['coverage_by_model','summary_by_model','summary_by_model_persona',
               'factor_sensitivity_macro_f1','config_level_metrics','topic_descriptive_metrics']
        source=table_path(names[0]).parent
        try: source=source.relative_to(ANALYSIS.parents[2])
        except ValueError: pass
        print('Sentiment analysis source:', source)
        coverage,summary_model,summary_persona,sensitivity,config,topics=[read_table(name) for name in names]
        model_order=['gemma-3-1b-it','gemma-3-4b-it','gemma-3-12b-it','gemma-3-27b-it',
                     'Qwen3-4B','Qwen3-8B','Qwen3-14B','Qwen3-32B']
        ideology_order=['base','centrism','libertarian_left','libertarian_right',
                        'authoritarian_left','authoritarian_right']
        sns.set_theme(style='whitegrid')
        """),
        md("""## Coverage and label support

        Each model has 1,800 configurations × 30 topics = 54,000 predictions.
        Coverage is checked before any comparison. The topic table also makes
        the single gold label per topic explicit; per-topic macro-F1 would be
        inappropriate because each topic supplies only one class."""),
        code("""
        display(coverage)
        support=topics.drop_duplicates('item_index').groupby('gold_label').size().rename('topics')
        display(support.to_frame())
        assert coverage.coverage.eq(1).all() and coverage.error_rows.eq(0).all()
        assert coverage.successful_rows.sum()==432_000
        assert len(topics)==8*6*30
        """),
        md("""## Overall and assigned-persona performance

        Macro-F1 gives equal weight to positive and negative labels. The overall
        table pools all persona conditions. Every heatmap cell pools 9,000
        predictions (300 configurations × 30 topics). Persona names denote the
        assigned prompt condition, not a model's intrinsic ideology."""),
        code("""
        display(summary_model[['model','n','accuracy','macro_f1','negative_f1','positive_f1',
                               'mean_confidence','positive_pred_rate']])
        heat=summary_persona.pivot(index='model',columns='ideology',values='macro_f1').reindex(model_order).reindex(columns=ideology_order)
        fig,ax=plt.subplots(figsize=(11,5.5)); sns.heatmap(heat,annot=True,fmt='.3f',vmin=0,vmax=1,cmap='YlGnBu',ax=ax)
        ax.set(title='IBM topic sentiment: macro-F1 by model and persona',xlabel='Assigned persona',ylabel='Model')
        fig.tight_layout(); plt.show()
        """),
        md("""## Persona deltas from the base condition

        Subtracting each model's base macro-F1 makes robustness loss or gain
        visible without adding a topic taxonomy. Negative values mean lower
        macro-F1 under that assigned persona in this experiment."""),
        code("""
        wide=summary_persona.pivot(index='model',columns='ideology',values='macro_f1').reindex(model_order)
        delta=wide.subtract(wide['base'],axis=0).drop(columns='base')
        display(delta.round(3))
        fig,ax=plt.subplots(figsize=(10,4.8))
        sns.heatmap(delta.reindex(columns=[x for x in ideology_order if x!='base']),annot=True,fmt='.3f',
                    center=0,cmap='vlag',ax=ax)
        ax.set_title("Macro-F1 change relative to each model's base condition")
        fig.tight_layout(); plt.show()
        """),
        md("""## Configuration-level distributions

        Means can hide unstable configurations. The boxplots show the full
        distribution of configuration-level macro-F1. Each configuration
        contains all 30 topics."""),
        code("""
        fig,axes=plt.subplots(2,4,figsize=(17,8),sharey=True)
        for ax,model in zip(axes.flat,model_order):
            sub=config[config.model.eq(model)]
            sns.boxplot(data=sub,x='ideology',y='macro_f1',order=ideology_order,showfliers=False,ax=ax)
            ax.set_title(model); ax.tick_params(axis='x',rotation=65); ax.set_xlabel('')
        fig.suptitle('Configuration-level macro-F1 by assigned persona',y=1.01)
        fig.tight_layout(); plt.show()
        """),
        md("""## Prompt-factor sensitivity

        Each named factor is a descriptive root-mean-square spread in
        configuration macro-F1 between factor levels, calculated within persona
        conditions. `Ideology SD` is the spread of persona-condition means;
        `Residual RMSE` comes from an additive dummy-coded model. Components are
        not causal and do not add up to total variance."""),
        code("""
        display(sensitivity.round(3))
        sensitivity_columns=['Ideology SD','Total Var SD','Residual RMSE','Context','Instruction','Key Type','Permutation','Persona']
        fig,ax=plt.subplots(figsize=(11,5.5)); sns.heatmap(sensitivity.set_index('model').reindex(model_order)[sensitivity_columns],
                    annot=True,fmt='.3f',cmap='magma',ax=ax)
        ax.set(title='IBM topic sentiment: configuration-level macro-F1 sensitivity',
               xlabel='Descriptive variation component',ylabel='Model')
        fig.tight_layout(); plt.show()
        named=['Context','Instruction','Key Type','Permutation','Persona']
        largest=sensitivity.set_index('model')[named].idxmax(axis=1).rename('largest named factor')
        display(largest.to_frame().join(sensitivity.set_index('model')[named].max(axis=1).rename('SD-sized magnitude')))
        """),
        md("""## Topic-level descriptive diagnostics

        For each model × persona × topic, the table reports accuracy, positive
        prediction rate and confidence over 300 randomized prompt
        configurations. The largest within-topic persona ranges locate fragile
        topics without claiming a universal political direction."""),
        code("""
        topic_range=(topics.groupby(['model','item_index','topic','gold_label'],observed=True)
                     .agg(min_accuracy=('accuracy','min'),max_accuracy=('accuracy','max'),
                          min_positive_rate=('positive_pred_rate','min'),max_positive_rate=('positive_pred_rate','max'))
                     .reset_index())
        topic_range['accuracy_range']=topic_range.max_accuracy-topic_range.min_accuracy
        topic_range['positive_rate_range']=topic_range.max_positive_rate-topic_range.min_positive_rate
        display(topic_range.sort_values('accuracy_range',ascending=False).head(20))
        """),
        md("""## Prediction-skew diagnostics

        Accuracy does not reveal how errors move. `positive_pred_rate −
        positive_gold_rate` is positive when a condition over-predicts positive
        sentiment and negative when it under-predicts it."""),
        code("""
        skew=(config.groupby(['model','ideology'],observed=True)
              .agg(accuracy=('accuracy','mean'),macro_f1=('macro_f1','mean'),
                   positive_pred_rate=('positive_pred_rate','mean'),
                   positive_gold_rate=('positive_gold_rate','mean')).reset_index())
        skew['positive_prediction_skew']=skew.positive_pred_rate-skew.positive_gold_rate
        display(skew.pivot(index='model',columns='ideology',values='positive_prediction_skew')
                .reindex(model_order).reindex(columns=ideology_order).round(3))
        """),
        md("""## Model and size summaries

        Family/size tables are descriptive only. Four checkpoints per family
        are insufficient to infer a general scaling law, and prompt sensitivity
        is not monotonic across all conditions."""),
        code("""
        model_summary=summary_model.set_index('model').reindex(model_order).copy()
        model_summary['family']=['Gemma 3']*4+['Qwen3']*4
        model_summary['size_b']=[1,4,12,27,4,8,14,32]
        display(model_summary[['family','size_b','accuracy','macro_f1','mean_confidence','positive_pred_rate']])
        """),
        md("""## Interpretation boundary

        The defensible result is that IBM topic-sentiment performance and
        positive/negative prediction balance vary across assigned personas and
        prompt factors, with checkpoint-specific magnitudes. This notebook does
        not use LLM-authored target labels, does not establish an intrinsic
        political sentiment tendency, and does not support a universal family
        or size effect."""),
    ]
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(nb,OUT)
    print(OUT)


if __name__=='__main__': main()
