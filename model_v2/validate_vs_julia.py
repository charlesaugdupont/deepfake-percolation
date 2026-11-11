"""Validate the numba engine (original-model switches) against the Julia sweep sweep_2026_04_10_1044.
For a sample of LHS points, run 50 seeds each and compare per-point means of key outputs."""
import sys, os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "/home/claude/work")
import engine as E
from metrics import bc

S = pd.read_csv(os.environ.get("DFP_SIM_SUMMARY", "/mnt/user-data/uploads/deepfake-percolation/figures/seminar_2026_10/data/sim_summary.csv"))
pts = S.groupby("lhs_id").agg(v_cost=("v_cost", "first"), loss=("loss", "first"), tpr=("tpr", "first"),
                              fpr=("fpr", "first"), p_fake=("p_fake", "first")).reset_index()
rng = np.random.default_rng(11)
pick = pts.sample(40, random_state=3)
REPS = 50
METRICS = ["frac_verified_ever", "bc_J", "avg_true_cascade", "avg_fake_cascade", "avg_verify_rate", "mean_true_belief",
           "mean_tpr_belief_verifiers", "mean_J_nonverifiers"]


def one(args):
    lid, phi, c, rl, t, f, s = args
    R = E.run(seed=100000 * int(lid) + s, N=300, T=2000, burn=1000, phi=phi, cost=c, rm=rl, rp=1.0, tpr=t, fpr=f,
              prior_mode=0, ptrue_mode=0, seed_mode=0, gamma=1.0, kappa=2.0)
    N = R["N"]; post = np.arange(2000) >= 1000
    fk = R["c_fake"]; sz = R["c_size"] / N
    J = R["J"][:, 0]
    ver = R["n_checks"][:, 0] > 0
    vr = R["c_checks"][:, 0] / R["c_size"]
    return dict(lhs_id=lid, frac_verified_ever=ver.mean(), bc_J=bc(J),
                avg_true_cascade=sz[post & ~fk].mean() if (post & ~fk).any() else 0.0,
                avg_fake_cascade=sz[post & fk].mean() if (post & fk).any() else 0.0,
                avg_verify_rate=vr[post].mean(), mean_true_belief=R["pi"].mean(),
                mean_tpr_belief_verifiers=R["that"][ver, 0].mean() if ver.any() else np.nan,
                mean_J_nonverifiers=J[~ver].mean() if (~ver).any() else np.nan)


if __name__ == "__main__":
    jobs = [(r.lhs_id, r.p_fake, r.v_cost, r.loss, r.tpr, r.fpr, s) for _, r in pick.iterrows() for s in range(REPS)]
    E.run(seed=0, T=10)  # compile
    t0 = time.time()
    with Pool(1) as p:
        res = p.map(one, jobs, chunksize=10)
    print("secs", round(time.time() - t0, 1))
    P = pd.DataFrame(res)
    py = P.groupby("lhs_id")[METRICS].agg(["mean", "sem"])
    jl = S[S.lhs_id.isin(pick.lhs_id)].groupby("lhs_id")[METRICS].agg(["mean", "sem"])
    rows = []
    for m in METRICS:
        d = py[(m, "mean")] - jl[(m, "mean")]
        se = np.sqrt(py[(m, "sem")] ** 2 + jl[(m, "sem")] ** 2)
        z = (d / se).replace([np.inf, -np.inf], np.nan)
        rows.append(dict(metric=m, mean_julia=jl[(m, "mean")].mean(), mean_numba=py[(m, "mean")].mean(),
                         corr=np.corrcoef(py[(m, "mean")], jl[(m, "mean")])[0, 1],
                         mean_abs_diff=d.abs().mean(), frac_abs_z_gt_2=(z.abs() > 2).mean(), frac_abs_z_gt_3=(z.abs() > 3).mean()))
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "validation_summary.csv"), index=False)
    M = py.copy(); M.columns = [f"numba_{a}_{b}" for a, b in M.columns]
    J = jl.copy(); J.columns = [f"julia_{a}_{b}" for a, b in J.columns]
    M.join(J).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "validation_points.csv"))
