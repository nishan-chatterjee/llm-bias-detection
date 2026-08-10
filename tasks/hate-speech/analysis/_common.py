"""Shared helpers for the analysis scripts."""
import re

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import chi2

# Sentinel level for a conditionally-sampled factor that is switched off.
# Must match `export_aggregates.ABSENT`.
ABSENT = "absent"


def read_table(path):
    """Read an exported table without pandas turning factor levels into NaN.

    Every table in `data/` is read through here rather than through a bare
    `pd.read_csv`. Default NA handling converts a dozen ordinary-looking strings
    ("None", "NA", "null", "nan", ...) into NaN, and statsmodels then drops
    those rows from every fit silently. Because roughly half of this design is
    conditionally sampled, a single such collision changes almost every number
    downstream while leaving the code looking correct.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def chi2_log10sf(x, k):
    """log10 P(chi^2_k > x), without underflowing to -inf for large x.

    `chi2.logsf` computes the survival function first and logs it afterwards, so
    it saturates at -inf once sf drops below ~1e-308 -- which happens routinely
    here. Beyond that point we use the asymptotic expansion of the upper
    incomplete gamma function, log Gamma(a, y) ~ (a - 1) log y - y for y >> a:
        log sf ~ (k/2 - 1) log(x/2) - x/2 - lgamma(k/2).
    """
    v = float(chi2.logsf(x, k))
    if not np.isfinite(v):
        a, y = k / 2.0, x / 2.0
        v = (a - 1.0) * np.log(y) - y - gammaln(a)
    return v / np.log(10.0)


def model_sort_key(name):
    """Sort models by family, then by parameter count."""
    size = re.search(r'(\d+\.?\d*)\s*[BbMm]', name)
    family = re.split(r'[-_\d]', re.sub(r'^.*/', '', name))[0].lower()
    return (family, float(size.group(1)) if size else 0.0)


def size_label(name):
    m = re.search(r'(\d+\.?\d*)[Bb]', name)
    return f"{m.group(1)}B" if m else name


def format_p(log10p):
    """Render a log10 p-value the way it is quoted in the paper."""
    if not np.isfinite(log10p):
        return "n/a"
    if log10p > -4:
        return f"{10 ** log10p:.3g}"
    return f"1e{log10p:.0f}"
