"""Reference checkers R1-R4: final trust distributions and split metrics across costs."""
import sys, os, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import config as C
from sweep import summarize

def one(args):
    kw, extra = args
    R = E.run(**kw); s = summarize(R); s.update(extra)
    return s, (R["J"][:, 0].astype(np.float32), R["n_checks"][:, 0] > 0, R["J"][:, 0] <= R["theta"][:, 0])

if __name__ == "__main__":
    net = sys.argv[1]; reps = int(sys.argv[2]); outp = sys.argv[3]; phi = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    costs = [float(c) for c in C.grid(net)["costs"] if c <= 0.6][::max(1, len([c for c in C.grid(net)["costs"] if c <= 0.6]) // 9)]
    jobs = []
    for ref, (t, f) in C.CHECKERS.items():
        for c in costs:
            for r in range(reps):
                jobs.append((C.kwargs(net, phi=phi, cost=c, tpr=t, fpr=f, seed=70000 + r), dict(ref=ref, tpr=t, fpr=f, cost=c, rep=r)))
    E.run(seed=0, T=5, N=50)
    with Pool(C.NPROC) as p:
        res = p.map(one, jobs, chunksize=1)
    S = pd.DataFrame([r[0] for r in res]); S.to_parquet(outp + ".parquet")
    pickle.dump([(j[1], r[1]) for j, r in zip(jobs, res) if j[1]["rep"] < 4], open(outp + "_J.pkl", "wb"))
    print(S.groupby(["ref", "cost"])[["dip4", "frac_locked", "frac_trapped", "mean_J", "frac_ever", "delta"]].mean().round(2).to_string())
