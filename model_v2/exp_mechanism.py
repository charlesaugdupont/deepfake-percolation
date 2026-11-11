"""Data for the mechanism figure: per-agent end states (degree, threshold, locked, first-check outcome) from network
runs at several costs, plus the share of checks made on fake items (q) used by the first-passage model."""
import sys, os, pickle, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import config as C


def one(kw):
    R = E.run(**kw)
    chk = R["c_checks"][:, 0]
    q = chk[R["c_fake"]].sum() / max(chk.sum(), 1)
    return dict(cost=kw["cost"], phi=kw["phi"], seed=kw["seed"], q=q, deg=R["deg"], theta=R["theta"][:, 0],
                J=R["J"][:, 0].astype(np.float32), nchecks=R["n_checks"][:, 0], first_bad=R["first_bad"][:, 0],
                first_check=R["first_check"][:, 0], stake=R["stake"])


if __name__ == "__main__":
    net = sys.argv[1]; reps = int(sys.argv[2]); outp = sys.argv[3]
    costs = [float(x) for x in sys.argv[4].split(",")]
    phi = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
    kws = [C.kwargs(net, phi=phi, cost=c, seed=3000 + r) for c in costs for r in range(reps)]
    E.run(seed=0, T=5, N=50)
    with Pool(C.NPROC) as p:
        res = p.map(one, kws, chunksize=1)
    pickle.dump(dict(net=net, phi=phi, runs=res), open(outp, "wb"))
    print("saved", len(res))
