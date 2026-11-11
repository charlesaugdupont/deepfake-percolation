"""Quick scan of the bifurcation index over cost at phi = 0.3, l/g = 2, to pick the representative cost c*."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from sweep import run_many
net = sys.argv[1]; reps = int(sys.argv[2]); outp = sys.argv[3]
jobs = [(C.kwargs(net, phi=0.3, cost=float(c), seed=900 + r), dict(cost=float(c), rep=r)) for c in C.grid(net)["costs"] for r in range(reps)]
D = run_many(jobs, chunksize=1); D.to_parquet(outp)
print(D.groupby("cost")[["dip4", "frac_locked", "frac_trapped", "frac_ever", "mean_J", "delta"]].mean().round(3).to_string())
