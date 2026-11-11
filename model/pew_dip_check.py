"""Pew ideology surveys used by Thurner, Hofer & Korbel (PNAS 2025, Fig. 1D): per survey year, the ideology
s_i = mean of 10 binary-coded opinions (+1 conservative, -1 liberal, 0 don't know), as in their
modules/4_figure_generation/utilities.jl (process_survey_data) and data/polarisation/pew/config/pew_survey_params.jl.
Outputs per year: n, their polarisation psi = Var(a_ij) (pairs sampled; 'strict' similarity, all 10 questions and
the leave-one-out average they report), and Hartigan's dip of the ideology distribution (raw, and with each value
spread uniformly over one full step of the 21-point scale, +-0.1). The scale has a sawtooth: odd tenths need an odd
number of 'neither/don't know' answers, so they are rare. A half-step jitter (+-0.05, used until 6 Nov) keeps the
sawtooth and the dip then mostly measures it (0.015-0.032); the full step averages neighbouring values (0.002-0.008,
the level that sampling noise alone gives a single hump of these sample sizes). Also the shape of the distribution: SD, excess kurtosis and the bimodality coefficient (BC > 5/9 suggests two humps).
Checked on 6 Nov whether the dip of these real data rises and levels off like ours (it does not; RESULTS_v3.md,
section 4, point 10). The survey files ship with their repository:
github.com/complexity-science-hub/ConnectivityPolarizationTransition (needs pyreadstat).
Usage: python pew_dip_check.py <their repo root> <out csv>"""
import sys, os, numpy as np, pandas as pd, pyreadstat, diptest
from scipy import stats
root, out = sys.argv[1], sys.argv[2]
A = {1.0: 1, 2.0: 1, 3.0: -1, 4.0: -1, 5.0: 0, 9.0: 0}          # 1999-2011 (1999 has no 5)
B = {1.0: 1, 2.0: -1, 5.0: 0, 9.0: 0}                           # 2014-2017
P = [(1999, "data/polarisation/pew/values99/values99c.sav", ["Q9K", "Q9I", "Q15E", "Q15F", "Q15M", "Q15N", "Q15P", "Q9P", "Q15A", "Q21E"], [3, 4, 8, 9], A),
     (2004, "data/polarisation/pew/Typology04/Typology04c.sav", ["q11a", "q11b", "q11c", "q11d", "q11f", "q11g", "q11i", "q11n", "q20r", "q20u"], [2, 4, 5, 6, 8, 10], A),
     (2011, "data/polarisation/pew/2011+Political+Typology+Survey/2011 Political Typology public.sav", ["q17a", "q17b", "q17c", "q17d", "q17f", "q17g", "q17i", "q17n", "q37r", "q37u"], [2, 4, 5, 6, 8, 10], A),
     (2014, "data/polarisation/pew/Polarization-2014/Polarization 2014 public.sav", ["q25a", "q25b", "q25c", "q25d", "q25f", "q25g", "q25i", "q25n", "q50r", "q50u"], [2, 4, 5, 6, 8, 10], B),
     (2015, "data/polarisation/pew/Governance 2015.public.sav", ["q42a", "q42b", "q42c", "q42d", "q42f", "q42g", "q42i", "q42h", "q106o", "q42m"], [2, 4, 5, 6, 7, 10], B),
     (2017, "data/polarisation/pew/Typology-17/Typology 17 public.sav", ["q25a", "q25b", "q25c", "q25d", "q25f", "q25g", "q25i", "q25n", "q50r", "q50u"], [2, 4, 5, 6, 8, 10], B)]
rng = np.random.default_rng(1)

def psi(M, pairs=200_000):
    i = rng.integers(0, len(M), pairs); j = rng.integers(0, len(M), pairs); ok = i != j
    a = 2 * (M[i[ok]] == M[j[ok]]).mean(1) - 1
    return a.var()

rows = []
for year, f, q, flip, mp in P:
    df, meta = pyreadstat.read_sav(os.path.join(root, f), usecols=q)
    M = df[q].apply(lambda col: col.map(lambda v: mp.get(float(v), np.nan) if pd.notna(v) else np.nan))
    unknown = int(((~df[q].isna()) & M.isna()).values.sum())         # codes outside the mapping (dropped, as 'missing')
    M = M.dropna().astype(int).values.copy()
    M[:, [k - 1 for k in flip]] *= -1
    s = M.mean(1)
    psi_all = psi(M); psi_loo = np.mean([psi(np.delete(M, k, 1), 50_000) for k in range(9)])
    dip_raw = diptest.dipstat(s)
    dip_j = np.mean([diptest.dipstat(s + rng.uniform(-0.1, 0.1, len(s))) for _ in range(20)])
    p_j = diptest.diptest(s + rng.uniform(-0.1, 0.1, len(s)))[1]
    g1 = stats.skew(s, bias=False); g2 = stats.kurtosis(s, bias=False); n = len(s)
    bc = (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    rows.append(dict(year=year, n=n, unknown_codes=unknown, mean_s=s.mean(), sd_s=s.std(), excess_kurtosis=g2, bimod_coef=bc,
                     psi_all10=psi_all, psi_loo9=psi_loo, dip_raw=dip_raw, dip_jitter=dip_j, p_jitter=p_j))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
