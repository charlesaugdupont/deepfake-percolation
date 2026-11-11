"""Run every paper experiment on one network, in order. Usage: python3 run_all.py <net> [quick]"""
import sys, os, subprocess, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
net = sys.argv[1]; quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
R = dict(phase=10, time=50, mech=10, two=10, priors=6, checkers=8, robust=6, gsa=4000)
if quick:
    R = dict(phase=2, time=6, mech=3, two=2, priors=2, checkers=2, robust=2, gsa=300)


def sh(args, env=None):
    t0 = time.time(); print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True, env=env)
    print(f"   done in {time.time() - t0:.0f}s", flush=True)


phase = f"{OUT}/phase_{net}.parquet"
if not os.path.exists(phase):
    sh(["exp_phase.py", net, str(R["phase"]), phase])
D = pd.read_parquet(phase)
d = D[(D.lam == 2.0) & np.isclose(D.phi, 0.3)].groupby("cost").dip4.mean()
cstar = float(d.idxmax())
costs = np.sort(D.cost.unique()); k = int(np.searchsorted(costs, cstar))
c_lo, c_hi = float(costs[max(k - 4, 0)]), float(costs[min(k + 4, len(costs) - 1)])
print(f"c* = {cstar} (max bifurcation at phi=0.3, l/g=2); also {c_lo}, {c_hi}", flush=True)
open(f"{OUT}/cstar_{net}.txt", "w").write(f"{cstar}\n{c_lo}\n{c_hi}\n")
for c in [cstar, c_lo, c_hi]:
    p = f"{OUT}/time_{net}_c{c:g}.pkl"
    if not os.path.exists(p):
        sh(["exp_time.py", net, str(c), "0.3", str(R["time"]), p])
mcosts = ",".join(f"{x:g}" for x in sorted(set([c_lo, float(costs[max(k - 2, 0)]), cstar, float(costs[min(k + 2, len(costs) - 1)])])))
if not os.path.exists(f"{OUT}/mech_{net}.pkl"):
    sh(["exp_mechanism.py", net, str(R["mech"]), f"{OUT}/mech_{net}.pkl", mcosts, "0.3"])
if not os.path.exists(f"{OUT}/two_{net}.parquet"):
    sh(["exp_two.py", net, str(R["two"]), f"{OUT}/two_{net}", f"{cstar:g}"])
if not os.path.exists(f"{OUT}/priors_{net}.parquet"):
    sh(["exp_priors.py", net, str(R["priors"]), f"{OUT}/priors_{net}.parquet", "0.3", f"{cstar:g}"])
if not os.path.exists(f"{OUT}/checkers_{net}.parquet"):
    sh(["exp_checkers.py", net, str(R["checkers"]), f"{OUT}/checkers_{net}", "0.3"])
if not os.path.exists(f"{OUT}/robust_{net}.parquet"):
    sh(["exp_robust.py", net, str(R["robust"]), f"{OUT}/robust_{net}.parquet", "20000", f"{c_lo:g},{cstar:g}"])
if not os.path.exists(f"{OUT}/gsa_{net}.parquet"):
    sh(["exp_gsa.py", net, str(R["gsa"]), f"{OUT}/gsa_{net}.parquet"])
print("ALL DONE", flush=True)
