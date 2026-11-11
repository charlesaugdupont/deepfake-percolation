"""Robustness of the split: exploration (random checks), partial feedback, posterior rule, stakes scaling, prior strength.
Long runs with snapshots so we can see whether the split is a long-lived transient or permanent."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from sweep import run_many

SNAPS = [0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000]
VARIANTS = {
    "baseline": {},
    "explore 1e-4": dict(eps_explore=1e-4), "explore 1e-3": dict(eps_explore=1e-3), "explore 1e-2": dict(eps_explore=1e-2),
    "feedback 0.5": dict(feedback=0.5), "feedback 0.2": dict(feedback=0.2),
    "posterior v1 (double count)": dict(ptrue_mode=0, seed_mode=0, n0=0.5),
    "stakes indep. of degree": dict(gamma=0.0),
    "stakes sqrt(degree)": dict(gamma=0.5),
    "prior strength x3 (kappa=9)": dict(kappa=9.0),
}

if __name__ == "__main__":
    net = sys.argv[1]; reps = int(sys.argv[2]); outp = sys.argv[3]
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    costs = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.05, 0.1]
    jobs = []
    for vname, v in VARIANTS.items():
        for c in costs:
            for r in range(reps):
                kw = C.kwargs(net, phi=0.3, cost=c, seed=500 + r, T=T, burn=T // 2, snap_times=[s for s in SNAPS if s <= T])
                kw.update(v)
                jobs.append((kw, dict(variant=vname, net=net, cost=c, rep=r)))
    D = run_many(jobs, opts=dict(snap_metrics=True), chunksize=1)
    D.to_parquet(outp)
