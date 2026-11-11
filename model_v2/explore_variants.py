"""Exploratory: how do candidate model tweaks change where trust in the checker splits?
Coarse phi x cost grids for several model variants (stakes relative to mean degree, so cost is in units of an
average agent's gain from sharing one genuine item)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/work")
from sweep import run_many

PHIS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
COSTS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
REPS = 8
base = dict(T=2000, burn=1000, rp=1.0, rm=2.0, tpr=0.85, fpr=0.10, kscale="mean")
ident = dict(prior_mode=2, t0=2 / 3, f0=1 / 3, kappa=3.0, g0=0.5, kappa_g=2.0, ptrue_mode=2, seed_mode=1)
VARIANTS = {
    "V0 original (N=300)": dict(N=300, prior_mode=0, ptrue_mode=0, seed_mode=0, kappa=2.0),
    "V1 +posterior fix": dict(N=300, prior_mode=0, ptrue_mode=2, seed_mode=1, kappa=2.0),
    "V2 +identical priors": dict(N=300, **ident),
    "V3 +N=1000": dict(N=1000, **ident),
    "V4 max-ignorance het. priors": dict(N=1000, prior_mode=3, t_lo=0.01, t_hi=0.99, f_lo=0.01, f_hi=0.99, kappa=2.0,
                                         g0=0.5, kappa_g=2.0, ptrue_mode=2, seed_mode=1),
    "V5 V3, stakes indep. of degree": dict(N=1000, gamma=0.0, **ident),
    "V6 V3, explore eps=0.01": dict(N=1000, eps_explore=0.01, **ident),
}
if __name__ == "__main__":
    jobs = []
    for vname, v in VARIANTS.items():
        for phi in PHIS:
            for c in COSTS:
                for r in range(REPS):
                    kw = dict(base); kw.update(v); kw.update(phi=phi, cost=c, seed=1000 * r + 7)
                    jobs.append((kw, dict(variant=vname, phi=phi, cost=c, rep=r)))
    D = run_many(jobs, n_proc=2, opts=dict(with_dip=False), chunksize=8)
    D.to_csv("/home/claude/work/out_explore_variants.csv", index=False)
