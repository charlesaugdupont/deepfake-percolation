"""Does the split depend on the initial distribution of trust in the checker?
(i) identical priors: grid of shared starting trust J0 x checking cost (prior strength kappa = 3, as in the default);
(ii) J0 x prior strength kappa at a fixed cost;
(iii) individual variation: prior means spread uniformly +/- w around the default (2/3, 1/3), and two alternative designs."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from sweep import run_many

J0S = np.round(np.linspace(0.0, 0.6, 13), 4)
COSTS = np.round(np.exp(np.linspace(np.log(0.005), np.log(0.5), 11)), 5)
KAPPAS = [2.0, 3.0, 10.0, 30.0]
WIDTHS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def ident(J0, kap):
    return dict(prior_mode=2, t0=0.5 + J0 / 2, f0=0.5 - J0 / 2, kappa=kap)


if __name__ == "__main__":
    net = sys.argv[1]; reps = int(sys.argv[2]); outp = sys.argv[3]
    phi = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    c_fix = [float(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0.05]
    if net is not None and net.startswith("twitter"):           # dense network: coarser grids
        J0S = np.round(np.linspace(0.0, 0.6, 7), 4)
        COSTS = np.array([c for c in C.grid(net)["costs"] if c <= 0.6])
    jobs = []
    for J0 in J0S:
        for c in COSTS:
            for r in range(reps):
                jobs.append((C.kwargs(net, phi=phi, cost=float(c), seed=7000 + r, **ident(J0, 3.0)),
                             dict(block="J0xcost", design="identical", cost=float(c), J0=float(J0), kappa=3.0, width=0.0, rep=r)))
    for c in c_fix:
        for J0 in J0S:
            for kap in KAPPAS:
                if kap == 3.0:
                    continue
                for r in range(reps + 1):
                    jobs.append((C.kwargs(net, phi=phi, cost=c, seed=7500 + r, **ident(J0, kap)),
                                 dict(block="J0xkappa", design="identical", cost=c, J0=float(J0), kappa=kap, width=0.0, rep=r)))
        for w in WIDTHS:
            for r in range(reps + 1):
                pk = dict(prior_mode=3, t_lo=2 / 3 - w, t_hi=2 / 3 + w, f_lo=1 / 3 - w, f_hi=1 / 3 + w, kappa=3.0)
                jobs.append((C.kwargs(net, phi=phi, cost=c, seed=8000 + r, **pk),
                             dict(block="spread", design="spread", cost=c, J0=1 / 3, kappa=3.0, width=w, rep=r)))
        for name, pk in [("seminar prior U[0.3,0.7]", dict(prior_mode=3, t_lo=0.3, t_hi=0.7, f_lo=0.3, f_hi=0.7, kappa=2.0)),
                         ("max-ignorance U(0,1)", dict(prior_mode=3, t_lo=0.01, t_hi=0.99, f_lo=0.01, f_hi=0.99, kappa=2.0))]:
            for r in range(reps + 1):
                jobs.append((C.kwargs(net, phi=phi, cost=c, seed=9000 + r, **pk),
                             dict(block="alt", design=name, cost=c, J0=np.nan, kappa=pk["kappa"], width=np.nan, rep=r)))
    D = run_many(jobs, chunksize=2 if net in (None, "hk") else 1)
    D.to_parquet(outp)
