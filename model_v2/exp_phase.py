"""Phase diagram: share of fakes x relative checking payoff x relative loss, many replicates."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from sweep import run_many

PHIS = np.round(np.arange(0.05, 0.951, 0.05), 3)
COSTS = np.round(np.exp(np.linspace(np.log(0.005), np.log(1.0), 15)), 5)
LOSSES = [1.0, 2.0, 4.0]
import os as _os
if _os.environ.get("LOSSES"): LOSSES = [float(x) for x in _os.environ["LOSSES"].split(",")]

if __name__ == "__main__":
    net = sys.argv[1]; reps = int(sys.argv[2]); outp = sys.argv[3]
    extra = eval(sys.argv[4]) if len(sys.argv) > 4 else {}
    reps_other = int(os.environ.get("REPS_OTHER", reps))          # replicates for l/g != 2
    G = C.grid(net); PHIS, COSTS = G["phis"], G["costs"]
    jobs = []
    for lam in LOSSES:
        for phi in PHIS:
            for c in COSTS:
                for r in range(reps if lam == 2.0 else reps_other):
                    kw = C.kwargs(net, phi=float(phi), cost=float(c), rm=lam, seed=int(1e6 * lam + 1e4 * r + 7), **extra)
                    jobs.append((kw, dict(net=net, lam=lam, phi=float(phi), cost=float(c), rep=r)))
    D = run_many(jobs, chunksize=4 if net in (None, "hk") else 1)
    D.to_parquet(outp)
