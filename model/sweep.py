"""Run many simulations in parallel and summarise each run (no large arrays kept)."""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E
import diptest
from metrics import bc, dip_p, otsu_split


def summarize(R, with_dip=False, snap_metrics=False):
    P = R["params"]; N = R["N"]; K = R["K"]; T = int(P["T"]); burn = int(P["burn"])
    out = {}
    post = np.arange(T) >= burn
    fk = R["c_fake"]; sz = R["c_size"] / N
    out["reach_true"] = sz[post & ~fk].mean() if (post & ~fk).any() else np.nan
    out["reach_fake"] = sz[post & fk].mean() if (post & fk).any() else np.nan
    out["delta"] = out["reach_true"] - out["reach_fake"]
    out["check_rate"] = (R["c_checks"].sum(1)[post] / R["c_size"][post]).mean()
    chk = R["c_checks"].sum(1)
    out["q_fake_checks"] = chk[fk].sum() / chk.sum() if chk.sum() > 0 else np.nan
    out["n_checks_total"] = int(chk.sum())
    # welfare: payoffs normalised by each agent's best/worst possible outcome (as in the seminar), Sen = mean*(1-Gini)
    pay, best, worst = R["payoff"], R["best"], R["worst"]
    norm = np.zeros_like(pay)
    m1 = (pay < 0) & (worst != 0); norm[m1] = -(pay[m1] / worst[m1])
    m2 = (pay > 0) & (best != 0); norm[m2] = pay[m2] / best[m2]
    x = np.sort(norm - min(norm.min(), 0)); n = len(x)
    gini = (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n) if x.sum() > 0 else 0.0
    out["welfare_mean"] = norm.mean(); out["welfare_gini"] = gini; out["welfare_sen"] = norm.mean() * (1 - gini)
    out["payoff_mean"] = pay.mean() / max(T - burn, 1)
    out["mean_pi"] = R["pi"].mean()
    any_chk = R["c_checks"].sum(1) > 0
    out["t_first_check"] = int(np.argmax(any_chk)) if any_chk.any() else -1           # first item that anyone checked
    st, sf = R["shares_true"].astype(float), R["shares_fake"].astype(float); ns = st + sf; ok = ns >= 5
    if ok.sum() > 10 and np.std(sf[ok] / ns[ok]) > 0:                                  # model analogue of the data's
        from scipy.stats import spearmanr                                              # degree-misinformation link
        out["rho_deg_fakeshare"] = float(spearmanr(R["deg"][ok], sf[ok] / ns[ok])[0])
    else:
        out["rho_deg_fakeshare"] = np.nan
    out["frac_both"] = (R["n_both"] > 0).mean() if K == 2 else 0.0
    tp = np.atleast_1d(np.asarray(P["tpr"], float)); fp = np.atleast_1d(np.asarray(P["fpr"], float))
    if tp.size == 1: tp = np.repeat(tp, K)
    if fp.size == 1: fp = np.repeat(fp, K)
    for c in range(K):
        sfx = "" if K == 1 else "_" + "AB"[c]
        J = R["J"][:, c]; th = R["theta"][:, c]; nch = R["n_checks"][:, c]
        ever = nch > 0; locked = J <= th
        Jtrue = tp[c] - fp[c]
        worth = Jtrue > th                      # checking would pay for some items if beliefs were correct
        m = otsu_split(J)
        out.update({
            "dip4" + sfx: 4 * diptest.dipstat(J) if J.std() > 1e-12 else 0.0,
            "frac_worth" + sfx: worth.mean(), "frac_trapped" + sfx: (worth & locked).mean(),
            "frac_users" + sfx: (~locked).mean(),
            "bc" + sfx: bc(J), "s_low" + sfx: m["s_low"], "gap" + sfx: m["gap"], "eta" + sfx: m["eta"], "er" + sfx: m["er"],
            "cut" + sfx: m["cut"],
            "frac_ever" + sfx: ever.mean(), "frac_locked" + sfx: locked.mean(), "frac_burned" + sfx: (ever & locked).mean(),
            "mean_J" + sfx: J.mean(), "sd_J" + sfx: J.std(),
            "mean_J_ever" + sfx: J[ever].mean() if ever.any() else np.nan,
            "mean_J_never" + sfx: J[~ever].mean() if (~ever).any() else np.nan,
            "mean_checks" + sfx: nch.mean(),
            "first_bad_rate" + sfx: (R["first_bad"][ever, c] == 1).mean() if ever.any() else np.nan,
        })
        if with_dip:
            out["dip_p" + sfx] = dip_p(J)
    if K == 2:
        JA, JB = R["J"][:, 0], R["J"][:, 1]
        thA, thB = R["theta"][:, 0], R["theta"][:, 1]
        a_ok, b_ok = JA > thA, JB > thB
        nA, nB = R["n_checks"][:, 0], R["n_checks"][:, 1]
        uA, uB = nA > 0, nB > 0
        users = uA | uB
        out.update(corr_AB=np.corrcoef(JA, JB)[0, 1] if JA.std() > 0 and JB.std() > 0 else np.nan,
                   frac_A_only=(a_ok & ~b_ok).mean(), frac_B_only=(~a_ok & b_ok).mean(),
                   frac_both_ok=(a_ok & b_ok).mean(), frac_none_ok=(~a_ok & ~b_ok).mean(),
                   used_onlyA=(uA & ~uB).mean(), used_onlyB=(~uA & uB).mean(), used_both=(uA & uB).mean(),
                   used_none=(~users).mean(),
                   excl_share=((uA ^ uB)[users]).mean() if users.any() else np.nan,
                   dominance=(np.maximum(nA, nB)[users] / (nA + nB)[users]).mean() if users.any() else np.nan,
                   absdiff_AB=np.abs(JA - JB).mean())
    if snap_metrics:
        for si, tt in enumerate(R["snap_times"]):
            J = R["snapJ"][si, :, 0].astype(float)
            out[f"bc_t{tt}"] = bc(J)
            out[f"dip4_t{tt}"] = 4 * diptest.dipstat(J) if J.std() > 1e-12 else 0.0
            out[f"locked_t{tt}"] = (J <= R["theta"][:, 0]).mean()
            out[f"ever_t{tt}"] = (R["snapNC"][si, :, 0] > 0).mean()
            out[f"er_t{tt}"] = otsu_split(J)["er"]
    return out


def _job(args):
    kw, extra, opts = args
    R = E.run(**kw)
    s = summarize(R, **opts)
    s.update(extra)
    return s


def run_many(jobs, n_proc=None, opts=None, chunksize=4, verbose=True):
    """jobs: list of (kwargs for engine.run, dict of labels to attach)."""
    opts = opts or {}
    import config as C
    n_proc = n_proc or C.NPROC
    E.run(seed=0, T=5, N=50)     # compile in parent (cache)
    t0 = time.time()
    with Pool(n_proc) as p:
        res = p.map(_job, [(kw, extra, opts) for kw, extra in jobs], chunksize=chunksize)
    if verbose:
        print(f"{len(jobs)} runs in {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(res)
