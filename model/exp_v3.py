"""All experiments for the v3 re-run, in one place.

Each experiment is a list of jobs (engine kwargs, labels, kind). The kind says what is kept from a run:
  summary : sweep.summarize(R)                      (one row per run)
  snap    : summarize(R, snap_metrics=True)         (adds locked/dip/ever at each snapshot time)
  time    : summary + snapshot histograms by group and sample trajectories (Fig. 2)
  mech    : summary + per-agent degree, threshold, trust, checks (Fig. 1)
  two     : summary + per-agent trust in / use of both checkers (Fig. 4)
  agents  : summary + per-agent final trust, ever checked, locked out (checker comparison)
v3 defaults (config.py): spread-out starting guesses about the share of genuine items; main setting phi = 0.3,
c/g = 0.05, l/g = 2; phase diagrams on a hexagonal lattice (config.hex_lattice).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import engine as E
import synthnets as SN
from sweep import summarize
from gsa import lhs

PRO = (0.95, 0.03)                       # professional fact-checker (two-checker asymmetric design)
TIME_BINS = np.linspace(-1, 1, 81)
ROBUST_SNAPS = [0, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
INT_SNAPS = [0, 250, 500, 750, 1000, 1050, 1100, 1200, 1500, 2000, 2500, 3000, 3500, 4000]
GSA_BOUNDS = {
    "phi": (0.05, 0.95, "lin"), "cost": (0.005, 1.0, "log"), "lam": (1.0, 5.0, "lin"), "J0": (0.05, 0.65, "lin"),
    "kappa": (2.0, 30.0, "log"), "spread": (0.0, 0.3, "lin"), "tpr": (0.60, 0.95, "lin"), "fpr": (0.02, 0.30, "lin"),
}
ROBUST_VARIANTS = {
    "baseline": {},
    "explore 1e-4": dict(eps_explore=1e-4), "explore 1e-3": dict(eps_explore=1e-3), "explore 1e-2": dict(eps_explore=1e-2),
    "feedback 0.5": dict(feedback=0.5), "feedback 0.2": dict(feedback=0.2),
    "posterior v1 (double count)": dict(ptrue_mode=0, seed_mode=0, n0=0.5),
    "stakes indep. of degree": dict(gamma=0.0),
    "stakes sqrt(degree)": dict(gamma=0.5),
    "prior strength x3 (kappa=9)": dict(kappa=9.0),
    "loss stake sqrt(degree)": dict(gamma_loss=0.5),       # safeguard for item 2: loss grows more slowly than gain
    "loss stake indep. of degree": dict(gamma_loss=0.0),
    "feedback 0.05": dict(feedback=0.05),                   # added 5 Nov (appended: earlier job indices stay valid)
}


def snaps_time(T):
    s = sorted(set([0] + list(range(1, 50)) + list(range(50, 200, 5)) + list(range(200, 1000, 20)) + list(range(1000, T + 1, 50))))
    return [x for x in s if x <= T]


def ident(J0, kap=3.0):
    return dict(prior_mode=2, t0=0.5 + J0 / 2, f0=0.5 - J0 / 2, kappa=kap)


# ---------------------------------------------------------------------------------------------------------------
# job lists
# ---------------------------------------------------------------------------------------------------------------

def phase(net, lams=(1.0, 2.0, 4.0), reps=None):
    """Hexagonal lattice over (phi, c/g) for each loss/gain ratio."""
    reps = reps or {1.0: 2, 2.0: 3, 4.0: 2}
    jobs = []
    for li, lam in enumerate(lams):
        for ci, (row, phi, c) in enumerate(C.hex_lattice()):
            for r in range(reps[lam]):
                kw = C.kwargs(net, phi=phi, cost=c, rm=lam, seed=int(1_000_000 * (li + 1) + 1000 * r + ci))
                jobs.append((kw, dict(net=net, lam=lam, row=row, phi=phi, cost=c, rep=r), "summary"))
    return jobs


def phase_lines(net, lam=2.0, costs=(0.001, 0.01, 0.1, 1.0), reps_from=3, reps_to=10):
    """Added 6 Nov: the four cost rows drawn as lines in Fig. 4F-G (l/g = 2, phi = 0.05, 0.10, ..., 0.95), topped up
    from 3 to 10 runs per cell. Same seed scheme as phase() for l/g = 2 (li = 1), continuing the run index r, so the
    runs add to those in phase_<net> (figures concatenate the two files)."""
    li = (1.0, 2.0, 4.0).index(lam); jobs = []
    for ci, (row, phi, c) in enumerate(C.hex_lattice()):
        if not any(np.isclose(c, cc) for cc in costs) or not np.isclose(round(phi * 1000) % 50, 0):
            continue
        for r in range(reps_from, reps_to):
            kw = C.kwargs(net, phi=phi, cost=c, rm=lam, seed=int(1_000_000 * (li + 1) + 1000 * r + ci))
            jobs.append((kw, dict(net=net, lam=lam, row=row, phi=phi, cost=c, rep=r), "summary"))
    return jobs


def gsa(net, n=1000):
    X = lhs(n, GSA_BOUNDS, np.random.default_rng(2026)); jobs = []
    for i, r in X.iterrows():
        t0, f0, w = 0.5 + r.J0 / 2, 0.5 - r.J0 / 2, r.spread
        kw = C.kwargs(net, phi=r.phi, cost=r.cost, rm=r.lam, tpr=r.tpr, fpr=r.fpr, seed=int(60000 + i), prior_mode=3,
                      t_lo=max(t0 - w, 0.01), t_hi=min(t0 + w, 0.99), f_lo=max(f0 - w, 0.01), f_hi=min(f0 + w, 0.99), kappa=r.kappa)
        jobs.append((kw, dict(i=int(i), **r.to_dict()), "summary"))
    return jobs


def time_exp(net, costs=(0.01, 0.05, 0.2), reps=None, T=2000):
    """One checker over time. 20 runs per cost: the costs 0.01 and 0.2 started with 10 runs and were topped up to 20 on
    4 Nov; the extra runs are appended so that earlier job indices stay valid."""
    reps = reps or {0.01: 10, 0.05: 20, 0.2: 10}
    jobs = [(C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=100 + r, T=T, snap_times=snaps_time(T)),
             dict(cost=c, phi=C.MAIN["phi"], rep=r), "time") for c in costs for r in range(reps[c])]
    jobs += [(C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=100 + r, T=T, snap_times=snaps_time(T)),
              dict(cost=c, phi=C.MAIN["phi"], rep=r), "time") for c in (0.01, 0.2) for r in range(10, 20)]
    return jobs


def snaps_long(T):
    """Snapshots for long runs: the Fig. 2 schedule up to 2,000 items, then 40 log-spaced points up to T."""
    return sorted(set(snaps_time(2000) + np.unique(np.round(np.geomspace(2000, T, 40)).astype(int)).tolist()))


DEG_CUTS = (35, 210)                     # terciles of the Twitter network's degrees (14,128 agents)


def time_deg(net, cost=0.05, reps=20, T=4000):
    """Added 6 Nov: the Fig. 2 runs at the main setting (same seeds and settings as time_long, so identical dynamics)
    with trust tracked by audience size: degree terciles (fewer than 35, 35-209, 210 or more neighbours), for the
    comparison with the GSS (Fig. S12)."""
    return [(C.kwargs(net, phi=C.MAIN["phi"], cost=cost, seed=100 + r, T=T, burn=T - 1000, snap_times=snaps_long(T)),
             dict(cost=cost, phi=C.MAIN["phi"], rep=r, T=T), "bydeg") for r in range(reps)]


def time_long(net, costs=(0.01, 0.05, 0.2), reps=20, T=4000):
    """Added 6 Nov: the Fig. 2 runs continued until the split of trust levels off at every cost (same seeds as
    time_exp, so the first 2,000 items are identical to those runs); burn-in: the last 1,000 items."""
    return [(C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=100 + r, T=T, burn=T - 1000, snap_times=snaps_long(T)),
             dict(cost=c, phi=C.MAIN["phi"], rep=r, T=T), "time") for c in costs for r in range(reps)]


def mech(net, costs=(0.01, 0.02, 0.05, 0.1), reps=6, n_top=10):
    jobs = [(C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=3000 + r), dict(cost=c, phi=C.MAIN["phi"], rep=r), "mech")
            for c in costs for r in range(reps)]
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    jobs += [(C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=3000 + r), dict(cost=c, phi=C.MAIN["phi"], rep=r), "mech")
             for c in costs for r in range(reps, n_top)]
    return jobs


def two(net, reps_sym=5, reps_asym=3, keep_cost=0.05, n_top=10, top_asym=True):
    costs = [c for c in C.C125 if c <= 0.5]; jobs = []
    for c in costs:
        for r in range(reps_sym):
            jobs.append((C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=c, seed=40000 + r),
                         dict(design="symmetric", cost=c, ratio=1.0, rep=r, keep=bool(c == keep_cost)), "two"))
    for ratio in (3.0, 10.0):
        for c in costs:
            for r in range(reps_asym):
                jobs.append((C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=[c, ratio * c], tpr=[C.CHECKER["tpr"], PRO[0]],
                                      fpr=[C.CHECKER["fpr"], PRO[1]], seed=50000 + r),
                             dict(design="asymmetric", cost=c, ratio=ratio, rep=r, keep=bool(c == keep_cost and ratio == 3.0)), "two"))
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    for c in costs:
        for r in range(reps_sym, n_top):
            jobs.append((C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=c, seed=40000 + r),
                         dict(design="symmetric", cost=c, ratio=1.0, rep=r, keep=bool(c == keep_cost)), "two"))
    if top_asym:                                                       # only the shown design: 3x the cost, c/g = 0.05
        for r in range(reps_asym, n_top):
            jobs.append((C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=[keep_cost, 3.0 * keep_cost], tpr=[C.CHECKER["tpr"], PRO[0]],
                                  fpr=[C.CHECKER["fpr"], PRO[1]], seed=50000 + r),
                         dict(design="asymmetric", cost=keep_cost, ratio=3.0, rep=r, keep=True), "two"))
    return jobs


def priors(net, reps=2):
    J0S = np.round(np.linspace(0.0, 0.6, 7), 4); costs = [c for c in C.C125 if c <= 0.5]; c0 = C.MAIN["cost"]; jobs = []
    for J0 in J0S:
        for c in costs:
            for r in range(reps):
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=7000 + r, **ident(J0)),
                             dict(block="J0xcost", design="identical", cost=c, J0=float(J0), kappa=3.0, width=0.0, rep=r), "summary"))
    for J0 in J0S:
        for kap in (2.0, 10.0, 30.0):
            for r in range(reps + 1):
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c0, seed=7500 + r, **ident(J0, kap)),
                             dict(block="J0xkappa", design="identical", cost=c0, J0=float(J0), kappa=kap, width=0.0, rep=r), "summary"))
    for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
        for r in range(reps + 1):
            pk = dict(prior_mode=3, t_lo=2 / 3 - w, t_hi=2 / 3 + w, f_lo=1 / 3 - w, f_hi=1 / 3 + w, kappa=3.0)
            jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c0, seed=8000 + r, **pk),
                         dict(block="spread", design="spread", cost=c0, J0=1 / 3, kappa=3.0, width=w, rep=r), "summary"))
    for name, pk in [("seminar prior U[0.3,0.7]", dict(prior_mode=3, t_lo=0.3, t_hi=0.7, f_lo=0.3, f_hi=0.7, kappa=2.0)),
                     ("max-ignorance U(0,1)", dict(prior_mode=3, t_lo=0.01, t_hi=0.99, f_lo=0.01, f_hi=0.99, kappa=2.0))]:
        for r in range(reps + 1):
            jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c0, seed=9000 + r, **pk),
                         dict(block="alt", design=name, cost=c0, J0=np.nan, kappa=pk["kappa"], width=np.nan, rep=r), "summary"))
    # added 5 Nov (appended, so earlier job indices stay valid): 10 runs per point for the line plots of Figs. S1B-D and
    # S5C-D, so their 95% CIs (Student's t) rest on 10 runs; same seeds continued (7000 + r, 7500 + r, 8000 + r)
    n_top = 10
    for c in (0.01, c0, 0.2):                                          # J0 x cost: S1B (kappa = 3) and S5C
        for J0 in J0S:
            for r in range(reps, n_top):
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=7000 + r, **ident(J0)),
                             dict(block="J0xcost", design="identical", cost=c, J0=float(J0), kappa=3.0, width=0.0, rep=r), "summary"))
    for J0 in J0S:                                                     # J0 x kappa: S1B
        for kap in (2.0, 10.0, 30.0):
            for r in range(reps + 1, n_top):
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c0, seed=7500 + r, **ident(J0, kap)),
                             dict(block="J0xkappa", design="identical", cost=c0, J0=float(J0), kappa=kap, width=0.0, rep=r), "summary"))
    for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):                   # spread: S1C-D and the main row of S5D
        for r in range(reps + 1, n_top):
            pk = dict(prior_mode=3, t_lo=2 / 3 - w, t_hi=2 / 3 + w, f_lo=1 / 3 - w, f_hi=1 / 3 + w, kappa=3.0)
            jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c0, seed=8000 + r, **pk),
                         dict(block="spread", design="spread", cost=c0, J0=1 / 3, kappa=3.0, width=w, rep=r), "summary"))
    return jobs


def checkers(net, costs=(0.01, 0.05, 0.2), reps=3, fpr_sweep=0.10):
    """Real-world checkers R1-R4, plus a sweep over checker quality J = TPR - FPR (FPR fixed at 0.10)."""
    jobs = []
    for ref, (t, f) in C.CHECKERS.items():
        for c in costs:
            for r in range(reps):
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c, tpr=t, fpr=f, seed=70000 + r),
                             dict(block="reference", ref=ref, tpr=t, fpr=f, cost=c, rep=r), "agents"))
    for J in np.round(np.arange(0.1, 0.81, 0.1), 2):
        t = float(np.round(fpr_sweep + J, 3))
        for c in costs:
            for r in range(reps):
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c, tpr=t, fpr=fpr_sweep, seed=71000 + r),
                             dict(block="quality", ref=f"J={J:.1f}", tpr=t, fpr=fpr_sweep, cost=c, rep=r), "summary"))
    # added 5 Nov (appended, so earlier job indices stay valid): two real-world checkers in the middle of the range
    for ref, (t, f) in C.CHECKERS_EXTRA.items():
        for c in costs:
            for r in range(reps):
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c, tpr=t, fpr=f, seed=70000 + r),
                             dict(block="reference", ref=ref, tpr=t, fpr=f, cost=c, rep=r), "agents"))
    # added 5 Nov (appended): 10 runs per point for the lines and markers of Fig. S2 (B, C) and its histograms (A), so the
    # 95% CIs (Student's t) rest on 10 runs; same seeds continued (71000 + r, 70000 + r)
    n_top = 10
    for J in np.round(np.arange(0.1, 0.81, 0.1), 2):
        t = float(np.round(fpr_sweep + J, 3))
        for c in costs:
            for r in range(reps, n_top):
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c, tpr=t, fpr=fpr_sweep, seed=71000 + r),
                             dict(block="quality", ref=f"J={J:.1f}", tpr=t, fpr=fpr_sweep, cost=c, rep=r), "summary"))
    shown = {**{k: C.CHECKERS[k] for k in ("R2", "R4")}, **C.CHECKERS_EXTRA}          # the four checkers in Fig. S2
    for ref, (t, f) in shown.items():
        for r in range(reps, n_top):
            jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=C.MAIN["cost"], tpr=t, fpr=f, seed=70000 + r),
                         dict(block="reference", ref=ref, tpr=t, fpr=f, cost=C.MAIN["cost"], rep=r), "agents"))
    return jobs


def robust(net, costs=(0.01, 0.05), reps=3, T=8000):
    jobs = []
    for vname, v in ROBUST_VARIANTS.items():
        for c in costs:
            for r in range(reps):
                kw = C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=500 + r, T=T, burn=T // 2,
                              snap_times=[s for s in ROBUST_SNAPS if s <= T])
                kw.update(v)
                jobs.append((kw, dict(variant=vname, cost=c, rep=r), "snap"))
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    shown = ["baseline", "explore 1e-4", "explore 1e-3", "explore 1e-2", "feedback 0.5", "feedback 0.2", "feedback 0.05"]
    for vname in shown:
        for r in range(reps, 10):
            kw = C.kwargs(net, phi=C.MAIN["phi"], cost=C.MAIN["cost"], seed=500 + r, T=T, burn=T // 2,
                          snap_times=[s for s in ROBUST_SNAPS if s <= T])
            kw.update(ROBUST_VARIANTS[vname])
            jobs.append((kw, dict(variant=vname, cost=C.MAIN["cost"], rep=r), "snap"))
    return jobs


def intervention(net, reps=5, T=4000, start=1000):
    """Temporary cut in checking costs (to 0, 10% or 50%) for 50, 200 or 1,000 items, for everyone or only for
    accounts with a below-median audience; same seeds as the control, so runs are identical until the cut starts."""
    jobs = []
    conds = [("control", 1.0, 0, "all")] + [(f"x{m:g} for {d} ({tg})", m, d, tg) for m in (0.0, 0.1, 0.5)
                                            for d in (50, 200, 1000) for tg in ("all", "below_median")]
    for name, m, d, tg in conds:
        for r in range(reps):
            kw = C.kwargs(net, **C.MAIN, seed=900 + r, T=T, burn=T - 1000, snap_times=INT_SNAPS,
                          int_start=start, int_end=start + d, int_mult=m, int_target=tg)
            jobs.append((kw, dict(condition=name, mult=m, duration=d, target=tg, rep=r), "snap"))
    return jobs


def baserate(net, reps=3):
    """Starting belief about the share of genuine items: uninformed (0.5 for all, v2), informed (1 - phi), spread (v3)."""
    sets = [(2.0, 0.3, 0.01), (2.0, 0.3, 0.05), (2.0, 0.3, 0.2), (2.0, 0.55, 0.05), (2.0, 0.75, 0.05), (4.0, 0.25, 0.01), (4.0, 0.35, 0.01)]
    jobs = []
    for lam, phi, c in sets:
        for name in ("uninformed", "informed", "spread"):
            pk = {"uninformed": dict(g_lo=None, g_hi=None, g0=0.5), "informed": dict(g_lo=None, g_hi=None, g0=1 - phi),
                  "spread": {}}[name]
            for r in range(reps):
                jobs.append((C.kwargs(net, phi=phi, cost=c, rm=lam, seed=900 + r, **pk),
                             dict(prior=name, lam=lam, phi=phi, cost=c, rep=r), "summary"))
    return jobs



# --- added 4 Nov: two checkers over time (main text) and other settings (SI) ---------------------------------------
SETTINGS = [(0.3, 0.01), (0.3, 0.2), (0.6, 0.05)]      # (phi, c/g): two other costs and a higher share of fakes


def two_time(net, T=2000):
    """Two identical checkers with snapshots over time: main setting (10 runs) and the three other settings (6 runs,
    topped up to 10 on 4 Nov; the extra runs are appended so that earlier job indices stay valid)."""
    sets = [(C.MAIN["phi"], C.MAIN["cost"], 10)] + [(phi, c, 6) for phi, c in SETTINGS]
    jobs = [(C.kwargs(net, K=2, phi=phi, cost=c, seed=42000 + r, T=T, snap_times=snaps_time(T)),
             dict(phi=phi, cost=c, rep=r), "two_time") for phi, c, n in sets for r in range(n)]
    jobs += [(C.kwargs(net, K=2, phi=phi, cost=c, seed=42000 + r, T=T, snap_times=snaps_time(T)),
              dict(phi=phi, cost=c, rep=r), "two_time") for phi, c in SETTINGS for r in range(6, 10)]
    return jobs


def two_sym(net, reps=5, keep_cost=0.05, n_top=10):
    """Two identical checkers only (same seeds as two()), for the other networks in Fig. S6 (added 5 Nov)."""
    costs = [c for c in C.C125 if c <= 0.5]
    jobs = [(C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=c, seed=40000 + r),
             dict(design="symmetric", cost=c, ratio=1.0, rep=r, keep=bool(c == keep_cost)), "two")
            for c in costs for r in range(reps)]
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    jobs += [(C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=c, seed=40000 + r),
              dict(design="symmetric", cost=c, ratio=1.0, rep=r, keep=bool(c == keep_cost)), "two")
             for c in costs for r in range(reps, n_top)]
    return jobs


def intervention_agents(net, reps=5, T=4000, start=1000):
    """Cost cut with per-agent payoffs (added 5 Nov for the cost-cut figure, dropped from the SI the same day): control, checking free for 50, 200 or 1,000 items, and
    10% of the cost for 200 items, for everyone; same seeds as intervention(), so each run equals its control until
    the cut starts. Payoffs are summed over the last 1,000 items (after the cut has ended)."""
    conds = [("control", 1.0, 0)] + [(f"x0 for {d} (all)", 0.0, d) for d in (50, 200, 1000)] + [("x0.1 for 200 (all)", 0.1, 200)]
    jobs = []
    for name, m, d in conds:
        for r in range(reps):
            kw = C.kwargs(net, **C.MAIN, seed=900 + r, T=T, burn=T - 1000, snap_times=INT_SNAPS,
                          int_start=start, int_end=start + d, int_mult=m, int_target="all")
            jobs.append((kw, dict(condition=name, mult=m, duration=d, target="all", rep=r), "cut"))
    return jobs


def priors_hex(net, reps=2, dJ=0.05, rows_per_decade=6, cmax=0.5):
    """Shared starting trust J0 x cost on a hexagonal lattice (Fig. S1A, added 5 Nov; replaces the 7 x 9 rectangular
    grid of priors()): rows of cost as in the phase diagrams (6 per decade, 0.001-0.46), J0 from 0 to 0.6 in steps of
    dJ, odd rows shifted by dJ/2; identical starting beliefs (kappa = 3), same seeds as priors(). The loop over runs is
    outermost, so the first run of every cell is done before any second run."""
    nrows = int(np.floor(np.log10(cmax / 1e-3) * rows_per_decade)) + 1
    jobs = []
    for r in range(reps):
        for i in range(nrows):
            c = float(np.round(10 ** (-3 + i / rows_per_decade), 8))
            J0s = np.arange(0, 0.6 + 1e-9, dJ) if i % 2 == 0 else np.arange(dJ / 2, 0.6 - dJ / 2 + 1e-9, dJ)
            for J0 in J0s:
                J0 = float(np.round(J0, 4))
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=7000 + r, **ident(J0)),
                             dict(block="hex", row=i, J0=J0, cost=c, kappa=3.0, rep=r), "summary"))
    return jobs


def time_phi(net, phi=0.6, c=0.05, reps=10, T=2000):
    """One checker over time at a higher share of fakes (the costs 0.01 and 0.2 are in time_exp)."""
    return [(C.kwargs(net, phi=phi, cost=c, seed=100 + r, T=T, snap_times=snaps_time(T)),
             dict(cost=c, phi=phi, rep=r), "time") for r in range(reps)]


def two_asym_settings(net, reps=3, n_top=10):
    """Cheap AI detector vs professional (3x cost) at the other settings, keeping per-agent data."""
    jobs = [(C.kwargs(net, K=2, phi=phi, cost=[c, 3.0 * c], tpr=[C.CHECKER["tpr"], PRO[0]], fpr=[C.CHECKER["fpr"], PRO[1]],
                      seed=50000 + r),
             dict(design="asymmetric", phi=phi, cost=c, ratio=3.0, rep=r, keep=True), "two")
            for phi, c in SETTINGS for r in range(reps)]
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    jobs += [(C.kwargs(net, K=2, phi=phi, cost=[c, 3.0 * c], tpr=[C.CHECKER["tpr"], PRO[0]], fpr=[C.CHECKER["fpr"], PRO[1]],
                       seed=50000 + r),
              dict(design="asymmetric", phi=phi, cost=c, ratio=3.0, rep=r, keep=True), "two")
             for phi, c in SETTINGS for r in range(reps, n_top)]
    return jobs


def priors_settings(net, reps=3):
    """Spread of starting beliefs about the checker at the other settings; shared starting trust J0 at phi = 0.6."""
    jobs = []
    for phi, c in SETTINGS:
        for w in (0.0, 0.1, 0.2, 0.3):
            pk = dict(prior_mode=3, t_lo=2 / 3 - w, t_hi=2 / 3 + w, f_lo=1 / 3 - w, f_hi=1 / 3 + w, kappa=3.0)
            for r in range(reps):
                jobs.append((C.kwargs(net, phi=phi, cost=c, seed=8000 + r, **pk),
                             dict(block="spread", phi=phi, cost=c, J0=1 / 3, width=w, rep=r), "summary"))
    for J0 in np.round(np.linspace(0.0, 0.6, 7), 4):
        for r in range(2):
            jobs.append((C.kwargs(net, phi=0.6, cost=0.05, seed=7000 + r, **ident(J0)),
                         dict(block="J0", phi=0.6, cost=0.05, J0=float(J0), width=0.0, rep=r), "summary"))
    # added 5 Nov (appended, so earlier job indices stay valid): 10 runs per point for the line plots of Fig. S5C-D
    n_top = 10
    for phi, c in SETTINGS:
        for w in (0.0, 0.1, 0.2, 0.3):
            pk = dict(prior_mode=3, t_lo=2 / 3 - w, t_hi=2 / 3 + w, f_lo=1 / 3 - w, f_hi=1 / 3 + w, kappa=3.0)
            for r in range(reps, n_top):
                jobs.append((C.kwargs(net, phi=phi, cost=c, seed=8000 + r, **pk),
                             dict(block="spread", phi=phi, cost=c, J0=1 / 3, width=w, rep=r), "summary"))
    for J0 in np.round(np.linspace(0.0, 0.6, 7), 4):
        for r in range(2, n_top):
            jobs.append((C.kwargs(net, phi=0.6, cost=0.05, seed=7000 + r, **ident(J0)),
                         dict(block="J0", phi=0.6, cost=0.05, J0=float(J0), width=0.0, rep=r), "summary"))
    return jobs


# --- added 4 Nov: how the penalty for sharing a fake grows with the audience (SI) ----------------------------------
# loss stake l * s^gamma_l with s = k / <k>; the gain stake stays g * s. All three forms agree for an average audience.
PENALTY_FORMS = [(1.0, "∝ audience (baseline)"), (0.5, "∝ √audience"), (0.0, "constant")]


def penalty(net, costs=(0.01, 0.05, 0.2), reps=5, T=2000):
    """One checker, per-agent data kept; same seeds for every form (the baseline repeats the mech runs)."""
    jobs = [(C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=3000 + r, T=T, snap_times=[T // 2, T], gamma_loss=gl),
             dict(gamma_loss=gl, cost=c, phi=C.MAIN["phi"], rep=r), "penalty")
            for gl, _ in PENALTY_FORMS for c in costs for r in range(reps)]
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    jobs += [(C.kwargs(net, phi=C.MAIN["phi"], cost=C.MAIN["cost"], seed=3000 + r, T=T, snap_times=[T // 2, T], gamma_loss=gl),
              dict(gamma_loss=gl, cost=C.MAIN["cost"], phi=C.MAIN["phi"], rep=r), "penalty")
             for gl, _ in PENALTY_FORMS for r in range(reps, 10)]
    return jobs


def penalty_two(net, costs=(0.01, 0.05, 0.2), reps=5, T=2000):
    """Two identical checkers, snapshots over time; same seeds as two_time (the baseline repeats those runs)."""
    jobs = [(C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=c, seed=42000 + r, T=T, snap_times=snaps_time(T), gamma_loss=gl),
             dict(gamma_loss=gl, cost=c, phi=C.MAIN["phi"], rep=r), "two_time")
            for gl, _ in PENALTY_FORMS for c in costs for r in range(reps)]
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    jobs += [(C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=C.MAIN["cost"], seed=42000 + r, T=T, snap_times=snaps_time(T), gamma_loss=gl),
              dict(gamma_loss=gl, cost=C.MAIN["cost"], phi=C.MAIN["phi"], rep=r), "two_time")
             for gl, _ in PENALTY_FORMS for r in range(reps, 10)]
    return jobs


# --- added 4 Nov (afternoon): both stakes scaled the same way (for Table S1 next to the penalty forms) --------------
STAKE_FORMS = [(0.5, "both stakes ∝ √audience"), (0.0, "both stakes constant")]


def stakes(net, costs=(0.01, 0.05, 0.2), reps=5, T=2000):
    """One checker, gain and loss stakes both ∝ s^gamma; same seeds and payload as penalty()."""
    jobs = [(C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=3000 + r, T=T, snap_times=[T // 2, T], gamma=g),
             dict(gamma=g, cost=c, phi=C.MAIN["phi"], rep=r), "penalty")
            for g, _ in STAKE_FORMS for c in costs for r in range(reps)]
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    jobs += [(C.kwargs(net, phi=C.MAIN["phi"], cost=C.MAIN["cost"], seed=3000 + r, T=T, snap_times=[T // 2, T], gamma=g),
              dict(gamma=g, cost=C.MAIN["cost"], phi=C.MAIN["phi"], rep=r), "penalty")
             for g, _ in STAKE_FORMS for r in range(reps, 10)]
    return jobs


def stakes_two(net, costs=(0.01, 0.05, 0.2), reps=5, T=2000):
    """Two identical checkers, both stakes ∝ s^gamma; same seeds as penalty_two()."""
    jobs = [(C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=c, seed=42000 + r, T=T, snap_times=snaps_time(T), gamma=g),
             dict(gamma=g, cost=c, phi=C.MAIN["phi"], rep=r), "two_time")
            for g, _ in STAKE_FORMS for c in costs for r in range(reps)]
    # added 6 Nov (appended, so earlier job indices stay valid): at least 10 runs per point wherever a figure shows a 95% CI
    jobs += [(C.kwargs(net, K=2, phi=C.MAIN["phi"], cost=C.MAIN["cost"], seed=42000 + r, T=T, snap_times=snaps_time(T), gamma=g),
              dict(gamma=g, cost=C.MAIN["cost"], phi=C.MAIN["phi"], rep=r), "two_time")
             for g, _ in STAKE_FORMS for r in range(reps, 10)]
    return jobs


INEQ_COSTS = [float(np.round(10 ** (-3 + i / 6), 8)) for i in range(19)]   # 0.001 ... 1, six per decade (as the hex rows)


def ineq(levels, reps=10):
    """Added 6 Nov: audience-inequality test on random networks (synthnets.py; N = 5,000). levels: (mean degree, CV of
    degrees). Run r uses wiring r of a level at every cost, with the same engine seed at every cost and level (common
    random numbers, so each run traces a smooth curve over cost); main setting otherwise (phi = 0.3, l/g = 2)."""
    jobs = []
    for kbar, cv in levels:
        for r in range(reps):
            net = SN.name(kbar, cv, r)
            for c in INEQ_COSTS:
                jobs.append((C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=int(7_000_000 + 1000 * r)),
                             dict(net=net, kbar=kbar, cv=cv, cost=c, rep=r), "summary"))
    return jobs


def senders(net, costs=(0.01, 0.05, 0.2), reps=10):
    """Added 11 Nov (your comment on first senders): after the truth is revealed, agents update the record of every
    neighbour who shared the item, not only the first sender's (engine all_senders=True). Paired with the baseline
    (same seeds, common random numbers); main setting otherwise (phi = 0.3, l/g = 2)."""
    jobs = []
    for variant, flag in (("first sender", False), ("all senders", True)):
        for c in costs:
            for r in range(reps):
                kw = C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=8_000_000 + r, all_senders=flag)
                jobs.append((kw, dict(variant=variant, cost=c, rep=r), "summary"))
    return jobs


def kappas(net, values=(0.5, 11.0), costs=(0.01, 0.05, 0.2), reps=10):
    """Added 11 Nov (your questions on kappa_s): the weight of pi_i in a sender's record (engine kappa_tau; 2 in the
    baseline) set to 0.5 and to 11, the value whose spread matches how much senders really differ at the main setting.
    Same seeds as senders_twitter, whose 'first sender' runs are the paired baseline (kappa_tau = 2)."""
    jobs = []
    for k in values:
        for c in costs:
            for r in range(reps):
                kw = C.kwargs(net, phi=C.MAIN["phi"], cost=c, seed=8_000_000 + r, kappa_tau=k)
                jobs.append((kw, dict(kappa_s=k, cost=c, rep=r), "summary"))
    return jobs


# run order: small, figure-critical experiments first, then the big sweeps
EXPERIMENTS = [
    ("mech_twitter", lambda: mech("twitter")),
    ("time_twitter", lambda: time_exp("twitter")),
    ("two_twitter", lambda: two("twitter")),
    ("priors_twitter", lambda: priors("twitter")),
    ("checkers_twitter", lambda: checkers("twitter")),
    ("intervention_twitter", lambda: intervention("twitter")),
    ("baserate_twitter", lambda: baserate("twitter")),
    ("phase_twitter", lambda: phase("twitter")),
    ("robust_twitter", lambda: robust("twitter")),
    ("gsa_twitter", lambda: gsa("twitter")),
    ("phase_twitter_rewired", lambda: phase("twitter_rewired", lams=(2.0,), reps={2.0: 1})),
    ("phase_fb_jhu", lambda: phase("fb_jhu", lams=(2.0,), reps={2.0: 2})),
    ("phase_twitch", lambda: phase("twitch", lams=(2.0,), reps={2.0: 2})),
    ("phase_hk", lambda: phase("hk", lams=(2.0,), reps={2.0: 2})),
    ("two_hk", lambda: two("hk", reps_sym=6, reps_asym=3, top_asym=False)),
    ("two_time_twitter", lambda: two_time("twitter")),
    ("time_twitter_phi06", lambda: time_phi("twitter")),
    ("two_asym_settings_twitter", lambda: two_asym_settings("twitter")),
    ("priors_settings_twitter", lambda: priors_settings("twitter")),
    ("penalty_twitter", lambda: penalty("twitter")),
    ("penalty_two_twitter", lambda: penalty_two("twitter")),
    ("stakes_twitter", lambda: stakes("twitter")),
    ("stakes_two_twitter", lambda: stakes_two("twitter")),
    ("intervention_agents_twitter", lambda: intervention_agents("twitter")),
    ("two_fb_jhu", lambda: two_sym("fb_jhu")),
    ("two_twitch", lambda: two_sym("twitch")),
    ("priors_hex_twitter", lambda: priors_hex("twitter")),
    ("phase_lines_twitter", lambda: phase_lines("twitter")),
    ("time_long_twitter", lambda: time_long("twitter")),
    ("ineq_cv", lambda: ineq(SN.ARM_CV)),                                        # mean degree 20, rising inequality
    ("ineq_k", lambda: ineq([lv for lv in SN.ARM_K if lv not in SN.ARM_CV])),   # CV 1, other mean degrees
    ("time_deg_twitter", lambda: time_deg("twitter")),                           # trust by audience size (Fig. S12)
    ("senders_twitter", lambda: senders("twitter")),                             # every sender's record updated (11 Nov)
    ("kappas_twitter", lambda: kappas("twitter")),                               # kappa_s = 0.5 and 11 (11 Nov)
]


# ---------------------------------------------------------------------------------------------------------------
# what to keep from a run
# ---------------------------------------------------------------------------------------------------------------

def payload_time(R, kw, n_traj=40):
    import diptest
    S = R["snap_times"]; th = R["theta"][:, 0]; N = R["N"]
    rng = np.random.default_rng(int(kw["seed"]) + 99)
    idx = rng.choice(N, min(n_traj, N), replace=False)
    out = dict(times=S, hist=np.zeros((len(S), 3, len(TIME_BINS) - 1), np.int32), dip4=np.zeros(len(S)),
               ever=np.zeros(len(S)), locked=np.zeros(len(S)), burned=np.zeros(len(S)),
               mean_by_group=np.full((len(S), 3), np.nan), traj=np.zeros((len(S), len(idx)), np.float32),
               traj_deg=R["deg"][idx], traj_nchecks=R["n_checks"][idx, 0], deg=R["deg"],
               J_end=R["J"][:, 0].astype(np.float32), ever_end=R["n_checks"][:, 0] > 0, theta=th,
               first_bad=R["first_bad"][:, 0], first_check=R["first_check"][:, 0])
    for si in range(len(S)):
        J = R["snapJ"][si, :, 0].astype(float); ever = R["snapNC"][si, :, 0] > 0; lock = J <= th
        for g, m in enumerate([~ever, ever & lock, ever & ~lock]):
            out["hist"][si, g] = np.histogram(J[m], TIME_BINS)[0]
            if m.any():
                out["mean_by_group"][si, g] = J[m].mean()
        out["dip4"][si] = 4 * diptest.dipstat(J) if J.std() > 1e-12 else 0.0
        out["ever"][si] = ever.mean(); out["locked"][si] = lock.mean(); out["burned"][si] = (ever & lock).mean()
        out["traj"][si] = J[idx]
    return out


def payload_mech(R, kw):
    chk = R["c_checks"][:, 0]
    return dict(q=chk[R["c_fake"]].sum() / max(chk.sum(), 1), deg=R["deg"], theta=R["theta"][:, 0],
                J=R["J"][:, 0].astype(np.float32), nchecks=R["n_checks"][:, 0], first_bad=R["first_bad"][:, 0],
                first_check=R["first_check"][:, 0], stake=R["stake"])


def payload_two(R, kw):
    return dict(JA=R["J"][:, 0].astype(np.float32), JB=R["J"][:, 1].astype(np.float32), nA=R["n_checks"][:, 0],
                nB=R["n_checks"][:, 1], nboth=R["n_both"], deg=R["deg"], thA=R["theta"][:, 0], thB=R["theta"][:, 1],
                firstA=R["first_check"][:, 0], firstB=R["first_check"][:, 1], badA=R["first_bad"][:, 0], badB=R["first_bad"][:, 1])


D_BINS = np.linspace(-1, 1, 81)


def payload_two_time(R, kw):
    """Two checkers over time. At each snapshot, among agents who can still check (above threshold for at least one
    checker): histogram of D = trust in A - trust in B by use (both / only A / only B / not yet), 4 x dip of D and the
    share with a clear favourite (|D| > 0.2); among all agents: share who used both / only A / only B / neither, and
    corr(trust in A, trust in B) among agents who checked at least once. Plus the final per-agent arrays."""
    import diptest
    S = R["snap_times"]; thA, thB = R["theta"][:, 0], R["theta"][:, 1]
    out = dict(times=S, histD=np.zeros((len(S), 4, len(D_BINS) - 1), np.int32), dip4D=np.zeros(len(S)), fav=np.zeros(len(S)),
               active=np.zeros(len(S)), use=np.zeros((len(S), 4)), corr=np.full(len(S), np.nan), **payload_two(R, kw))
    for si in range(len(S)):
        JA = R["snapJ"][si, :, 0].astype(float); JB = R["snapJ"][si, :, 1].astype(float)
        nA = R["snapNC"][si, :, 0]; nB = R["snapNC"][si, :, 1]
        act = (JA > thA) | (JB > thB); D = JA - JB
        for g, m in enumerate([(nA > 0) & (nB > 0), (nA > 0) & (nB == 0), (nA == 0) & (nB > 0), (nA == 0) & (nB == 0)]):
            out["histD"][si, g] = np.histogram(D[act & m], D_BINS)[0]
            out["use"][si, g] = m.mean()
        d = D[act]; out["active"][si] = act.mean()
        if len(d) > 10 and d.std() > 1e-12:
            out["dip4D"][si] = 4 * diptest.dipstat(d); out["fav"][si] = (np.abs(d) > 0.2).mean()
        ev = (nA + nB) > 0
        if ev.sum() > 10 and JA[ev].std() > 0 and JB[ev].std() > 0:
            out["corr"][si] = np.corrcoef(JA[ev], JB[ev])[0, 1]
    return out


def payload_penalty(R, kw):
    """Per-agent behaviour for the penalty-scaling scenario: checks in the whole run and in its first half, items
    received, shared, and shared genuine/fake, final trust and threshold."""
    S = list(R["snap_times"]); h = S.index(int(kw["T"]) // 2)
    return dict(deg=R["deg"], theta=R["theta"][:, 0].astype(np.float32), J=R["J"][:, 0].astype(np.float32),
                n_checks=R["n_checks"][:, 0], n_checks_half=R["snapNC"][h, :, 0], n_recv=R["n_recv"],
                n_shared=R["n_shared"], shares_true=R["shares_true"], shares_fake=R["shares_fake"],
                first_bad=R["first_bad"][:, 0])


def payload_cut(R, kw):
    """Per-agent payoffs in the last 1,000 items (checking costs included), best and worst possible payoffs, checks,
    final trust and threshold (cost-cut figure, not in the paper)."""
    return dict(deg=R["deg"], theta=R["theta"][:, 0].astype(np.float32), J=R["J"][:, 0].astype(np.float32),
                n_checks=R["n_checks"][:, 0], payoff=R["payoff"].astype(np.float32), best=R["best"].astype(np.float32),
                worst=R["worst"].astype(np.float32), costpaid=R["costpaid"].astype(np.float32), n_recv=R["n_recv"],
                shares_true=R["shares_true"], shares_fake=R["shares_fake"])


def payload_bydeg(R, kw, cuts=DEG_CUTS):
    """Per snapshot and degree tercile (fewer than cuts[0], up to cuts[1], more neighbours): mean trust in the checker,
    share not checking (trust at or below one's threshold) and share who ever checked."""
    S = R["snap_times"]; th = R["theta"][:, 0]
    grp = np.digitize(R["deg"], cuts)
    out = dict(times=S, cuts=np.array(cuts), n=np.bincount(grp, minlength=3),
               meanJ=np.zeros((len(S), 3)), notchk=np.zeros((len(S), 3)), ever=np.zeros((len(S), 3)))
    for si in range(len(S)):
        J = R["snapJ"][si, :, 0].astype(float); ev = R["snapNC"][si, :, 0] > 0
        for g in range(3):
            m = grp == g
            out["meanJ"][si, g] = J[m].mean(); out["notchk"][si, g] = (J[m] <= th[m]).mean(); out["ever"][si, g] = ev[m].mean()
    return out


def payload_agents(R, kw):
    return dict(J=R["J"][:, 0].astype(np.float32), ever=R["n_checks"][:, 0] > 0, lock=R["J"][:, 0] <= R["theta"][:, 0],
                deg=R["deg"])


def run_job(args):
    """Worker: (index, engine kwargs, labels, kind) -> (index, summary row, payload or None)."""
    idx, kw, meta, kind = args
    R = E.run(**kw)
    s = summarize(R, snap_metrics=(kind == "snap"))
    s.update(meta); s["job_idx"] = idx
    pay = None
    if kind == "time":
        pay = payload_time(R, kw)
    elif kind == "mech":
        pay = payload_mech(R, kw)
    elif kind == "two" and meta.get("keep"):
        pay = payload_two(R, kw)
    elif kind == "two_time":
        pay = payload_two_time(R, kw)
    elif kind == "penalty":
        pay = payload_penalty(R, kw)
    elif kind == "cut":
        pay = payload_cut(R, kw)
    elif kind == "agents" and meta.get("cost") == C.MAIN["cost"]:
        pay = payload_agents(R, kw)
    elif kind == "bydeg":
        pay = payload_bydeg(R, kw)
    return idx, s, pay
