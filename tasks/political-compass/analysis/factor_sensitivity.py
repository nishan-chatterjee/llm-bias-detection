"""
Political Compass factor sensitivity: the heatmap figure and its Fisher p-values.

Reproduces, from `data/pct_configuration_scores.csv` alone:

  * every cell of the factor-sensitivity heatmap (per-factor SD in compass
    points, plus the ideology signal, total variation, residual jitter and
    response-entropy columns), and
  * the Fisher-combined p-value behind each of those cells.

Method
------
For each model x ideology condition we fit an ordinary least-squares ANOVA with
Type III sums of squares and orthogonal Sum contrasts, then convert each
factor's accumulated sum of squares into a standard deviation in compass-point
units. Stratifying by ideology first matters: a factor that pushes one ideology
condition left and another right would otherwise cancel and read as no effect.

Because that produces one p-value per ideology condition rather than one per
factor, the six are combined with Fisher's method,

    X = -2 * sum_i ln p_i  ~  chi^2(2k)   under the null,

with k = 6 conditions for every factor except persona wording, which does not
exist in the base condition and therefore has k = 5. The six fits use disjoint
subsets of rows, so the p-values are independent and Fisher's method applies.

All p-value arithmetic is done in log space. At n = 900 rows per fit the
individual p-values routinely fall below the smallest representable double, so
computing them on the natural scale and combining afterwards silently underflows
to zero; `scipy.stats.combine_pvalues` then returns inf. `f.logsf` and
`chi2.logsf` avoid this, and the asymptotic branch in `chi2_log10sf` covers the
range where even `chi2.logsf` saturates.

Usage
-----
    python factor_sensitivity_pct.py            # tables to stdout + CSV
    python factor_sensitivity_pct.py --figure   # also render the heatmap
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from _common import chi2_log10sf, model_sort_key, read_table

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "pct_configuration_scores.csv")
CSV_OUT = os.path.join(HERE, "data", "pct_factor_sensitivity.csv")
FIG_OUT = os.path.join(HERE, "figures", "factor_sensitivity_pct.png")

FACTORS = ['quantization', 'language', 'context_combo', 'instr_combo',
           'key_type', 'perm_id', 'persona_combo']

LABEL = {
    'quantization':  'Quantization',
    'language':      'Language',
    'context_combo': 'Disclaimer',
    'instr_combo':   'Instruction',
    'key_type':      'Key Type',
    'perm_id':       'Permutation',
    'persona_combo': 'Persona',
}

AXES = [('econ', 'score_econ', 'intrinsic_var_econ'),
        ('soc', 'score_soc', 'intrinsic_var_soc')]


def analyze_axis(df, score_col, blur_col):
    """One row per model: SD components and Fisher-combined p per factor."""
    rows = []
    for model_name, md in df.groupby('model'):
        n = len(md)
        ss = {f: 0.0 for f in FACTORS}
        logp = {f: [] for f in FACTORS}
        ss_resid = 0.0
        df_resid = 0.0
        ss_blur = md[blur_col].sum()
        sd_ideology = md.groupby('ideology')[score_col].mean().std()

        for _ideology, sub in md.groupby('ideology'):
            active = [f for f in FACTORS if sub[f].nunique() > 1]
            if not (active and len(sub) > len(active)):
                ss_resid += sub[score_col].var() * max(len(sub) - 1, 0)
                df_resid += max(len(sub) - 1, 0)
                continue
            formula = f"{score_col} ~ " + " + ".join(f"C({f}, Sum)" for f in active)
            try:
                tab = anova_lm(ols(formula, data=sub).fit(), typ=3)
            except Exception:
                ss_resid += sub[score_col].var() * max(len(sub) - 1, 0)
                df_resid += max(len(sub) - 1, 0)
                continue
            resid_df = tab.loc['Residual', 'df']
            for f in active:
                key = f"C({f}, Sum)"
                ss[f] += tab.loc[key, 'sum_sq']
                F = tab.loc[key, 'F']
                if np.isfinite(F) and F > 0:
                    logp[f].append(float(f_dist.logsf(F, tab.loc[key, 'df'], resid_df)))
            ss_resid += tab.loc['Residual', 'sum_sq']
            df_resid += resid_df

        # total variation = everything that is not the intended ideology signal
        non_persona = (sum(ss[f] for f in FACTORS if f != 'persona_combo')
                       + ss_resid + ss_blur)

        r = {
            'model': model_name,
            'Ideology': sd_ideology,
            'Total Variation': np.sqrt(non_persona / n) if n else 0.0,
            'Residual Jitter': np.sqrt(ss_resid / df_resid) if df_resid > 0 else 0.0,
            'Response Entropy': np.sqrt(ss_blur / n) if n else 0.0,
        }
        for f in FACTORS:
            r[f + '_sd'] = np.sqrt(ss[f] / n) if n else np.nan
            k = len(logp[f])
            if k:
                X = -2.0 * float(np.sum(logp[f]))
                r[f + '_k'] = k
                r[f + '_chi2'] = X
                r[f + '_log10p'] = chi2_log10sf(X, 2 * k)
            else:
                r[f + '_k'] = 0
                r[f + '_chi2'] = np.nan
                r[f + '_log10p'] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def plot(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    cols = (['Ideology', 'Total Variation', 'Residual Jitter', 'Response Entropy']
            + [f + '_sd' for f in FACTORS])
    names = {f + '_sd': LABEL[f] for f in FACTORS}

    fig, axes = plt.subplots(1, 2, figsize=(22, 0.6 * (len(res) // 2) + 4))
    panels = [('econ', 'ECONOMIC Impact'), ('soc', 'SOCIAL Impact')]
    sub_all = pd.concat([res[res.axis == a][cols] for a, _ in panels])
    vmin, vmax = float(np.nanmin(sub_all.values)), float(np.nanmax(sub_all.values))

    for ax, (axis, title) in zip(axes, panels):
        d = res[res.axis == axis].set_index('model')[cols].rename(columns=names)
        sns.heatmap(d, annot=d.map(lambda v: f"{v:.2f}"), fmt='', cmap='viridis',
                    vmin=vmin, vmax=vmax, ax=ax,
                    cbar=(axis == 'soc'),
                    cbar_kws={'label': 'Size of standard deviation'})
        ax.set_title(title, fontweight='bold', pad=14)
        ax.set_xlabel("")
        ax.set_ylabel("Model" if axis == 'econ' else "")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        if axis == 'soc':
            ax.set_yticks([])

    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"figure -> {os.path.relpath(FIG_OUT, HERE)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--figure', action='store_true', help="also render the heatmap")
    args = ap.parse_args()

    df = read_table(DATA)
    print(f"{len(df)} configurations, {df['model'].nunique()} models, "
          f"quantizations {sorted(df['quantization'].astype(str).unique())}")

    out = []
    for axis, score_col, blur_col in AXES:
        r = analyze_axis(df, score_col, blur_col)
        r.insert(1, 'axis', axis)
        out.append(r)
    res = pd.concat(out, ignore_index=True)
    res['sk'] = res['model'].apply(model_sort_key)
    res = res.sort_values(['axis', 'sk']).drop(columns='sk')
    res.to_csv(CSV_OUT, index=False)
    print(f"table -> {os.path.relpath(CSV_OUT, HERE)}\n")

    for axis, _, _ in AXES:
        s = res[res.axis == axis].set_index('model')
        print(f"=== {axis}: SD per factor (compass points) ===")
        print(s[[f + '_sd' for f in FACTORS]]
              .rename(columns={f + '_sd': LABEL[f] for f in FACTORS})
              .round(2).to_string())
        print(f"\n=== {axis}: log10 Fisher-combined p ===")
        print(s[[f + '_log10p' for f in FACTORS]]
              .rename(columns={f + '_log10p': LABEL[f] for f in FACTORS})
              .round(1).to_string(), "\n")

    print("=== cells failing p < 0.05 ===")
    found = False
    for f in FACTORS:
        c = f + '_log10p'
        for _, r in res[res[c] >= np.log10(0.05)].iterrows():
            found = True
            print(f"  {r['model']:<16} {r['axis']:<5} {LABEL[f]:<13} p = {10 ** r[c]:.3f}")
    if not found:
        print("  (none)")

    if args.figure:
        plot(res)


if __name__ == '__main__':
    main()
