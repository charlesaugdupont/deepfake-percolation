"""Time evolution of trust in the checker (Figure: bifurcation over time).
For each run, at each snapshot: histogram of J-hat by group (never checked / checked-then-locked-out / still checking),
dip, BC, shares; plus a random subset of individual trajectories."""
import sys, os, time, pickle
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import diptest
from metrics import bc
import config as C

BINS = np.linspace(-1, 1, 81)


def snaps_default(T):
    s = sorted(set([0] + list(range(1, 50)) + list(range(50, 200, 5)) + list(range(200, 1000, 20)) + list(range(1000, T + 1, 50))))
    return [x for x in s if x <= T]


def one(args):
    kw, n_traj = args
    R = E.run(**kw)
    S = R["snap_times"]; th = R["theta"][:, 0]; N = R["N"]
    rng = np.random.default_rng(int(kw["seed"]) + 99)
    traj_idx = rng.choice(N, n_traj, replace=False)
    out = dict(times=S, hist=np.zeros((len(S), 3, len(BINS) - 1), np.int32), dip4=np.zeros(len(S)), bc=np.zeros(len(S)),
               ever=np.zeros(len(S)), locked=np.zeros(len(S)), burned=np.zeros(len(S)),
               mean_by_group=np.full((len(S), 3), np.nan), traj=np.zeros((len(S), n_traj), np.float32),
               traj_deg=R["deg"][traj_idx], traj_nchecks=R["n_checks"][traj_idx, 0], deg=R["deg"],
               J_end=R["J"][:, 0].astype(np.float32), ever_end=R["n_checks"][:, 0] > 0, theta=th,
               first_bad=R["first_bad"][:, 0], first_check=R["first_check"][:, 0],
               params={k: v for k, v in kw.items() if k not in ("graph", "phi_sched", "tpr_sched", "fpr_sched")})
    for si in range(len(S)):
        J = R["snapJ"][si, :, 0].astype(float); ever = R["snapNC"][si, :, 0] > 0; lock = J <= th
        groups = [~ever, ever & lock, ever & ~lock]
        for g, m in enumerate(groups):
            out["hist"][si, g] = np.histogram(J[m], BINS)[0]
            if m.any():
                out["mean_by_group"][si, g] = J[m].mean()
        out["dip4"][si] = 4 * diptest.dipstat(J) if J.std() > 1e-12 else 0.0
        out["bc"][si] = bc(J)
        out["ever"][si] = ever.mean(); out["locked"][si] = lock.mean(); out["burned"][si] = (ever & lock).mean()
        out["traj"][si] = J[traj_idx]
    return out


def run(kw_list, n_traj=40, n_proc=None):
    E.run(seed=0, T=5, N=50)
    t0 = time.time()
    with Pool(n_proc or C.NPROC) as p:
        res = p.map(one, [(kw, n_traj) for kw in kw_list], chunksize=1)
    print(f"{len(kw_list)} runs in {time.time() - t0:.1f}s", flush=True)
    return res


if __name__ == "__main__":
    net = sys.argv[1]; cost = float(sys.argv[2]); phi = float(sys.argv[3]); reps = int(sys.argv[4]); outp = sys.argv[5]
    T = 2000
    kws = [C.kwargs(net, phi=phi, cost=cost, seed=s, T=T, snap_times=snaps_default(T)) for s in range(reps)]
    res = run(kws)
    pickle.dump(dict(net=net, cost=cost, phi=phi, bins=BINS, runs=res), open(outp, "wb"))
