"""PAWN global sensitivity analysis (Pianosi & Wagener 2015) on a Latin hypercube design.
Same estimator as deepfake-percolation/analyze_results.py (median KS over equal-frequency slices, bootstrap CIs),
plus a dummy input to give the noise floor (indices below the dummy's are not distinguishable from zero)."""
import numpy as np, pandas as pd
from scipy.stats import ks_2samp


def lhs(n, bounds, rng):
    """bounds: dict name -> (lo, hi, 'lin'|'log'). Returns DataFrame of n LHS points (+ 'dummy' column)."""
    cols = {}
    for name, (lo, hi, scale) in list(bounds.items()) + [("dummy", (0.0, 1.0, "lin"))]:
        u = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(u)
        cols[name] = lo + u * (hi - lo) if scale == "lin" else np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
    return pd.DataFrame(cols)


def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (sup distance between the empirical CDFs); same value as
    scipy.stats.ks_2samp(a, b).statistic, without the p-value computation."""
    a = np.sort(a); b = np.sort(b); x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, side="right") / len(a) - np.searchsorted(b, x, side="right") / len(b))))


def pawn_indices(X, y, n_slices=10):
    rows = []
    for col in X.columns:
        xi = X[col].values
        edges = np.unique(np.percentile(xi, np.linspace(0, 100, n_slices + 1)))
        ks = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (xi >= lo) & (xi <= hi)
            if m.sum() < 10:
                continue
            ks.append(ks_stat(y[m], y))
        rows.append(dict(param=col, ks_median=float(np.median(ks)), ks_max=float(np.max(ks))))
    return pd.DataFrame(rows).set_index("param")


def pawn_bootstrap(X, y, n_slices=10, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    est = pawn_indices(X, y, n_slices)
    B = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        B.append(pawn_indices(X.iloc[idx].reset_index(drop=True), y[idx], n_slices)["ks_median"].values)
    B = np.array(B)
    est["lo"] = np.percentile(B, 2.5, axis=0); est["hi"] = np.percentile(B, 97.5, axis=0)
    return est
