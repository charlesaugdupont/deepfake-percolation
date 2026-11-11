"""Global sensitivity analysis design (LHS) for PAWN: one run per design point."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from sweep import run_many
from gsa import lhs

BOUNDS = {
    "phi": (0.05, 0.95, "lin"),          # share of fakes
    "cost": (0.005, 1.0, "log"),         # checking payoff / sharing gain (average agent)
    "lam": (1.0, 5.0, "lin"),            # loss / gain
    "J0": (0.05, 0.65, "lin"),           # shared prior trust in the checker (t0 - f0)
    "kappa": (2.0, 30.0, "log"),         # prior strength (pseudo-observations per rate)
    "spread": (0.0, 0.3, "lin"),         # half-width of individual variation in prior means
    "tpr": (0.60, 0.95, "lin"),          # checker's true hit rate
    "fpr": (0.02, 0.30, "lin"),          # checker's true false-alarm rate
}

if __name__ == "__main__":
    net = sys.argv[1]; n = int(sys.argv[2]); outp = sys.argv[3]
    X = lhs(n, BOUNDS, np.random.default_rng(2026))
    jobs = []
    for i, r in X.iterrows():
        t0, f0, w = 0.5 + r.J0 / 2, 0.5 - r.J0 / 2, r.spread
        kw = C.kwargs(net, phi=r.phi, cost=r.cost, rm=r.lam, tpr=r.tpr, fpr=r.fpr, seed=int(60000 + i),
                      prior_mode=3, t_lo=max(t0 - w, 0.01), t_hi=min(t0 + w, 0.99), f_lo=max(f0 - w, 0.01), f_hi=min(f0 + w, 0.99),
                      kappa=r.kappa)
        jobs.append((kw, dict(i=i, **r.to_dict())))
    D = run_many(jobs, chunksize=1 if net not in (None, "hk") else 4)
    D.to_parquet(outp)
