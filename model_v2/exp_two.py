"""Two checkers. Symmetric: two identical R2 checkers (same accuracy, same cost, same prior); options are share,
discard, check A, check B, or check both (cost 2c; share under the better of the AND / OR rules).
Asymmetric: cheap AI detector (R2) vs a slower, more accurate professional fact-checker (assumed (0.95, 0.03))."""
import sys, os, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import config as C
from sweep import summarize

PRO = (0.95, 0.03)


def one(args):
    kw, extra, keep = args
    R = E.run(**kw)
    s = summarize(R); s.update(extra)
    agents = None
    if keep:
        agents = dict(JA=R["J"][:, 0].astype(np.float32), JB=R["J"][:, 1].astype(np.float32), nA=R["n_checks"][:, 0],
                      nB=R["n_checks"][:, 1], nboth=R["n_both"], deg=R["deg"], thA=R["theta"][:, 0], thB=R["theta"][:, 1],
                      firstA=R["first_check"][:, 0], firstB=R["first_check"][:, 1], badA=R["first_bad"][:, 0], badB=R["first_bad"][:, 1])
    return s, agents, extra


if __name__ == "__main__":
    net = sys.argv[1]; reps = int(sys.argv[2]); outp = sys.argv[3]
    keep_costs = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.05]
    costs = [float(c) for c in C.grid(net)["costs"] if c <= 0.6]
    costs = sorted(set(costs) | set(keep_costs))
    jobs = []
    for phi in [0.3]:
        for c in costs:
            for r in range(reps):
                kw = C.kwargs(net, K=2, phi=phi, cost=c, seed=40000 + r)
                jobs.append((kw, dict(design="symmetric", phi=phi, cost=c, ratio=1.0, rep=r), (c in keep_costs and r < 5)))
        for ratio in [3.0, 10.0]:
            for c in costs:
                for r in range(max(2, reps // 2)):
                    kw = C.kwargs(net, K=2, phi=phi, cost=[c, ratio * c], tpr=[C.CHECKER["tpr"], PRO[0]],
                                  fpr=[C.CHECKER["fpr"], PRO[1]], seed=50000 + r)
                    jobs.append((kw, dict(design="asymmetric", phi=phi, cost=c, ratio=ratio, rep=r), (c in keep_costs and r < 5)))
    E.run(seed=0, T=5, N=50, K=2)
    with Pool(C.NPROC) as p:
        res = p.map(one, jobs, chunksize=1)
    S = pd.DataFrame([r[0] for r in res]); S["net"] = net
    S.to_parquet(outp + ".parquet")
    pickle.dump([(r[2], r[1]) for r in res if r[1] is not None], open(outp + "_agents.pkl", "wb"))
    print("saved", len(S))
