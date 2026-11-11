"""Bifurcation / polarization metrics for the distribution of trust in a checker across agents."""
import numpy as np
from scipy import stats


def bc(x):
    """Sample bimodality coefficient (SAS / Pfister et al. 2013); > 5/9 suggests bimodality.
    Same formula as utils.bimodality_coefficient in the repo."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or np.std(x) < 1e-12:
        return np.nan
    sk = stats.skew(x) * np.sqrt((n - 1) / n) * (n / (n - 2))
    ku = stats.kurtosis(x, fisher=True)
    return (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def bc_rows(X):
    X = np.asarray(X, float)
    n = X.shape[1]
    sk = stats.skew(X, axis=1) * np.sqrt((n - 1) / n) * (n / (n - 2))
    ku = stats.kurtosis(X, axis=1, fisher=True)
    out = (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    out[np.std(X, axis=1) < 1e-12] = np.nan
    return out


def dip_p(x):
    import diptest
    x = np.asarray(x, float)
    if np.std(x) < 1e-12:
        return 1.0
    d, p = diptest.diptest(x)
    return p


def otsu_split(x):
    """Best two-group split of a 1-D sample (maximises between-group variance, i.e. 1-D 2-means).
    Returns dict with the share of the lower group, group means, gap, eta = between/total variance,
    and the two-group Esteban-Ray-type index 4 s (1-s) * gap."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n < 2 or x[-1] - x[0] < 1e-12:
        return dict(s_low=np.nan, m_low=np.nan, m_high=np.nan, gap=0.0, eta=0.0, er=0.0, cut=np.nan)
    cs = np.cumsum(x)
    tot = cs[-1]
    k = np.arange(1, n)                      # size of lower group
    m1 = cs[:-1] / k
    m2 = (tot - cs[:-1]) / (n - k)
    w = k / n
    between = w * (1 - w) * (m2 - m1) ** 2
    # only split between distinct values
    valid = x[1:] > x[:-1]
    between = np.where(valid, between, -1)
    j = int(np.argmax(between))
    s = w[j]
    var = x.var()
    eta = between[j] / var if var > 0 else 0.0
    gap = m2[j] - m1[j]
    return dict(s_low=s, m_low=m1[j], m_high=m2[j], gap=gap, eta=eta, er=4 * s * (1 - s) * gap,
                cut=0.5 * (x[j] + x[j + 1]))
