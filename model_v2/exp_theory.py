"""Pure first-passage theory: probability of ending up locked out as a function of the gap between starting trust and
the threshold, for several prior strengths and checkers (no network)."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trapmodel import p_trap
J0 = 1 / 3
thetas = np.linspace(0.0, 0.5, 51)
out = {}
for kap in [2.0, 3.0, 10.0, 30.0]:
    prior = (J0 * 0 + (0.5 + J0 / 2) * kap, (0.5 - J0 / 2) * kap, (0.5 - J0 / 2) * kap, (0.5 + J0 / 2) * kap)
    for ref, (t, f) in {"R2": (0.80, 0.10), "R3": (0.85, 0.15)}.items():
        for q in [0.15, 0.3]:
            out[(kap, ref, q)] = p_trap(thetas, prior=prior, t=t, f=f, q=q, n_checks=300, n_mc=8000, seed=1)
pickle.dump(dict(thetas=thetas, J0=J0, res=out), open(sys.argv[1], "wb"))
print({k: np.round(v[[10, 20, 30]], 3) for k, v in out.items() if k[1] == "R2" and k[2] == 0.15})
