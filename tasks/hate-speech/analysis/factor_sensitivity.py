"""
Hate-speech factor sensitivity: the decomposition figure and its Fisher p-values.

Reproduces, from `data/hs_configuration_scores.csv` alone, the variance
decomposition of P(hate) and the combined p-value behind each prompt factor.

Two modelling choices carry over from the paper and are worth restating here,
because they are the difference between this figure and a naive one:

  * The gold label and the target identity group are properties of the *data*,
    not of the prompt. Their variance is not comparable with a prompt factor's,
    so `target` enters only as a blocking term and the label is not a factor at
    all. Every cell therefore describes how much a prompt-side choice moves
    P(hate) on the test set taken as a whole, pooled over all ten target groups.

  * `Mean P(Hate)` is reported in different units from the SD components and is
    kept out of the shared colour scale when plotting.

Ideology is handled separately from the prompt factors. It is the variable the
per-condition fits stratify on, so it is constant within each stratum and cannot
be combined across them with Fisher's method. It is instead tested once per
model, in a single fit that includes the target block, and reported as an
ordinary Type III F-test.

Usage
-----
    python factor_sensitivity_hs.py            # tables to stdout + CSV
    python factor_sensitivity_hs.py --figure   # also render the heatmap
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from _common import chi2_log10sf, model_sort_key, read_table, size_label

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "hs_configuration_scores.csv")
CSV_OUT = os.path.join(HERE, "data", "hs_factor_sensitivity.csv")
FIG_OUT = os.path.join(HERE, "figures", "hate_speech_factor_sensitivity.png")

PROMPT_FACTORS = ['persona_combo', 'context_combo', 'instr_combo']
BLOCK = ['target']

LABEL = {
    'persona_combo': 'Persona',
    'context_combo': 'Context',
    'instr_combo':   'Instruction',
}


def decompose(md):
    """SD components for one model, matching the published figure."""
    # remove the target effect so the figure describes the test as a whole
    adj = (md['mean_p_hate']
           - md.groupby(BLOCK)['mean_p_hate'].transform('mean')
           + md['mean_p_hate'].mean())
    mean_blur = md['mean_item_dispersion'].mean()

    ss = {f: 0.0 for f in PROMPT_FACTORS}
    ss_resid = 0.0
    for _ideology, sub in md.groupby('ideology'):
        terms = [f for f in (BLOCK + PROMPT_FACTORS) if sub[f].nunique() > 1]
        if terms and len(sub) > len(terms):
            formula = "mean_p_hate ~ " + " + ".join(f"C({f}, Sum)" for f in terms)
            try:
                tab = anova_lm(ols(formula, data=sub).fit(), typ=3)
                for f in PROMPT_FACTORS:
                    if f in terms:
                        ss[f] += tab.loc[f"C({f}, Sum)", 'sum_sq']
                ss_resid += tab.loc['Residual', 'sum_sq']
                continue
            except Exception:
                pass
        ss_resid += sub['mean_p_hate'].var(ddof=0) * len(sub)

    r = {
        'Mean P(Hate)': md['mean_p_hate'].mean(),
        'Total Var SD': np.sqrt(adj.var(ddof=0) + mean_blur),
        'Ideology SD': adj.groupby(md['ideology']).mean().std(),
        'Intrinsic Blur': np.sqrt(mean_blur),
        'Residual Jitter': np.sqrt(ss_resid / len(md)),
    }
    for f in PROMPT_FACTORS:
        r[f + '_sd'] = np.sqrt(ss[f] / len(md))
    return r


def fisher_prompt_factors(md):
    """Per-ideology Type III tests for the prompt factors, combined by Fisher."""
    logp = {f: [] for f in PROMPT_FACTORS}
    for _ideology, sub in md.groupby('ideology'):
        terms = [f for f in (BLOCK + PROMPT_FACTORS) if sub[f].nunique() > 1]
        if not (terms and len(sub) > len(terms)):
            continue
        formula = "mean_p_hate ~ " + " + ".join(f"C({f}, Sum)" for f in terms)
        try:
            tab = anova_lm(ols(formula, data=sub).fit(), typ=3)
        except Exception:
            continue
        resid_df = tab.loc['Residual', 'df']
        for f in PROMPT_FACTORS:
            if f not in terms:
                continue
            F = tab.loc[f"C({f}, Sum)", 'F']
            if np.isfinite(F) and F > 0:
                logp[f].append(float(f_dist.logsf(F, tab.loc[f"C({f}, Sum)", 'df'], resid_df)))

    out = {}
    for f in PROMPT_FACTORS:
        k = len(logp[f])
        if k:
            X = -2.0 * float(np.sum(logp[f]))
            out[f + '_k'] = k
            out[f + '_log10p'] = chi2_log10sf(X, 2 * k)
        else:
            out[f + '_k'] = 0
            out[f + '_log10p'] = np.nan
    return out


def ideology_pvalue(md):
    """Ideology tested once per model, with the target group as a block.

    Cannot go through Fisher: ideology is the stratifying variable, so it is
    constant inside each of the per-condition fits above.
    """
    terms = [f for f in (BLOCK + ['ideology'] + PROMPT_FACTORS) if md[f].nunique() > 1]
    if 'ideology' not in terms:
        return np.nan
    formula = "mean_p_hate ~ " + " + ".join(f"C({f}, Sum)" for f in terms)
    try:
        tab = anova_lm(ols(formula, data=md).fit(), typ=3)
    except Exception:
        return np.nan
    F = tab.loc["C(ideology, Sum)", 'F']
    if not (np.isfinite(F) and F > 0):
        return np.nan
    return float(f_dist.logsf(F, tab.loc["C(ideology, Sum)", 'df'],
                              tab.loc['Residual', 'df'])) / np.log(10.0)


def plot(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    summary = ['Mean P(Hate)', 'Total Var SD']
    sd_cols = ['Ideology SD', 'persona_combo_sd', 'context_combo_sd',
               'instr_combo_sd', 'Intrinsic Blur', 'Residual Jitter']
    names = {'Ideology SD': 'Ideology', 'persona_combo_sd': 'Persona',
             'context_combo_sd': 'Context', 'instr_combo_sd': 'Instruction',
             'Intrinsic Blur': 'Intrinsic Blur', 'Residual Jitter': 'Residual Jitter'}

    d_sum = res.set_index('model')[summary]
    d_sd = res.set_index('model')[sd_cols].rename(columns=names)

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(13, len(res) * 0.55 + 2.2),
        gridspec_kw={'width_ratios': [len(summary), len(sd_cols)], 'wspace': 0.04})

    # different units from the SD components: deliberately uncoloured
    sns.heatmap(d_sum, annot=d_sum.map(lambda v: f"{v:.2f}"), fmt='',
                cmap=matplotlib.colors.ListedColormap(['#f2f2f2']), cbar=False,
                annot_kws={"color": "#111111"}, ax=ax0,
                linewidths=0.4, linecolor='white')
    ax0.set_yticklabels([size_label(m) for m in d_sum.index], rotation=0)
    ax0.set_ylabel(""), ax0.set_xlabel("")
    ax0.set_xticklabels(summary, rotation=40, ha='right')

    sns.heatmap(d_sd, annot=d_sd.map(lambda v: f"{v:.2f}"), fmt='', cmap='viridis',
                cbar_kws={'label': 'SD contribution (P(hate) units)'},
                ax=ax1, linewidths=0.4, linecolor='white')
    ax1.set_yticks([]), ax1.set_ylabel(""), ax1.set_xlabel("")
    ax1.set_xticklabels(d_sd.columns, rotation=40, ha='right')

    plt.suptitle("Variance decomposition: hate-speech P(hate)", fontweight='bold')
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.savefig(FIG_OUT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"figure -> {os.path.relpath(FIG_OUT, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figure', action='store_true')
    args = ap.parse_args()

    df = read_table(DATA)
    print(f"{len(df)} configurations, {df['model'].nunique()} models")

    rows = []
    for model, md in df.groupby('model'):
        md = md.reset_index(drop=True)
        r = {'model': model}
        r.update(decompose(md))
        r.update(fisher_prompt_factors(md))
        r['ideology_log10p'] = ideology_pvalue(md)
        rows.append(r)

    res = pd.DataFrame(rows)
    res['sk'] = res['model'].apply(model_sort_key)
    res = res.sort_values('sk').drop(columns='sk').reset_index(drop=True)
    res.to_csv(CSV_OUT, index=False)
    print(f"table -> {os.path.relpath(CSV_OUT, HERE)}\n")

    sd_cols = (['Mean P(Hate)', 'Total Var SD', 'Ideology SD']
               + [f + '_sd' for f in PROMPT_FACTORS]
               + ['Intrinsic Blur', 'Residual Jitter'])
    print("=== SD components (P(hate) units) ===")
    print(res.set_index('model')[sd_cols]
          .rename(columns={f + '_sd': LABEL[f] for f in PROMPT_FACTORS})
          .round(3).to_string())

    p_cols = [f + '_log10p' for f in PROMPT_FACTORS] + ['ideology_log10p']
    p_names = {f + '_log10p': LABEL[f] for f in PROMPT_FACTORS}
    p_names['ideology_log10p'] = 'Ideology (single fit)'
    print("\n=== log10 p ===")
    print(res.set_index('model')[p_cols].rename(columns=p_names).round(1).to_string())

    print("\n=== weakest evidence per factor ===")
    for c in p_cols:
        i = res[c].idxmax()
        flag = "   <-- p >= 0.05" if res.loc[i, c] >= np.log10(0.05) else ""
        print(f"  {p_names[c]:<22} p = {10 ** res.loc[i, c]:.2e}  "
              f"({res.loc[i, 'model']}){flag}")

    if args.figure:
        plot(res)


if __name__ == '__main__':
    main()
