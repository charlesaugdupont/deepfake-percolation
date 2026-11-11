"""Evidence figure for the modelling decisions (not a paper figure)."""
import sys, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/work")
import engine as E, diptest
from metrics import bc
from sweep import run_many, summarize
from trapmodel import p_trap
import figstyle as fs
fs.setup()
OUT = sys.argv[1]

econ = dict(T=2000, burn=1000, tpr=0.85, fpr=0.10, rp=1.0, rm=2.0, phi=0.35, kscale="mean", ptrue_mode=2, seed_mode=1,
            g0=0.5, kappa_g=2.0)
PRIORS = {"original prior\nmeans U[0.3,0.7], κ=2": dict(prior_mode=1, prior_lo=0.3, prior_hi=0.7, kappa=2.0),
          "identical prior\nTPR~Beta(2,1), FPR~Beta(1,2)": dict(prior_mode=2, t0=2 / 3, f0=1 / 3, kappa=3.0),
          "max-ignorance prior\nmeans U(0,1), κ=2": dict(prior_mode=3, t_lo=0.01, t_hi=0.99, f_lo=0.01, f_hi=0.99, kappa=2.0)}
nets = {}
for name in ["lastfm", "twitch"]:
    nets[name] = E.edgelist_to_csr(pd.read_csv(f"/home/claude/netdata/{name}/edges.csv").values)

fig = plt.figure(figsize=(fs.W2, 5.3))
gs = fig.add_gridspec(2, 4, hspace=0.95, wspace=0.6)
# A: histograms under three priors
for j, (pn, pk) in enumerate(PRIORS.items()):
    ax = fig.add_subplot(gs[0, j])
    kw = dict(econ); kw.update(pk); kw.update(N=1000, cost=0.1, seed=4)
    R = E.run(**kw); J = R["J"][:, 0]; ever = R["n_checks"][:, 0] > 0; lock = J <= R["theta"][:, 0]
    bins = np.linspace(-1, 1, 51)
    ax.hist([J[~ever], J[ever & lock], J[ever & ~lock]], bins=bins, stacked=True, color=[fs.GREY, fs.ORANGE, fs.BLUE],
            label=["never checked", "checked, now locked out", "still checks"])
    ax.axvline(0.75, color=fs.GREEN, lw=0.8)
    ax.set_title(pn, fontsize=6.3)
    ax.text(0.02, 0.97, f"BC={bc(J):.2f}\n4·dip={4 * diptest.dipstat(J):.2f}", transform=ax.transAxes, va="top", fontsize=6)
    ax.set_xlabel("trust in checker Ĵ"); ax.set_yticks([])
    if j == 0:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.38), ncol=3, fontsize=6)
        fs.panel_label(ax, "A", x=-0.1, y=1.22)
# B: BC vs dip across cost (identical prior)
ax = fig.add_subplot(gs[0, 3]); fs.panel_label(ax, "B", y=1.22)
jobs = []
for c in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    for r in range(6):
        kw = dict(econ); kw.update(PRIORS["identical prior\nTPR~Beta(2,1), FPR~Beta(1,2)"]); kw.update(N=1000, cost=c, seed=r)
        jobs.append((kw, dict(cost=c, rep=r)))
D = run_many(jobs, verbose=False)
g = D.groupby("cost").agg(bc=("bc", "mean"), dip=("dip4", "mean"), ever=("frac_ever", "mean"), bcs=("bc", "sem"), dips=("dip4", "sem"))
ax.errorbar(g.index, g.bc, yerr=1.96 * g.bcs, color=fs.PURPLE, marker="o", ms=2.5, label="BC")
ax.errorbar(g.index, g.dip, yerr=1.96 * g.dips, color=fs.BLUE, marker="o", ms=2.5, label="4·dip")
ax.plot(g.index, g.ever, color=fs.GREY, ls="--", label="share ever checked")
ax.axhline(5 / 9, color=fs.PURPLE, lw=0.5, ls=":")
ax.set_xscale("log"); ax.set_xlabel("cost / gain (avg. agent)"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=5.5, loc="center left"); ax.set_title("identical prior:\nBC stays high for lopsided splits", fontsize=6.3)
ax.set_xticks([0.02, 0.1, 0.5]); ax.set_xticklabels(["0.02", "0.1", "0.5"]); ax.minorticks_off()
# C: locked fraction vs cost on three networks
ax = fig.add_subplot(gs[1, 0]); fs.panel_label(ax, "C", y=1.22)
jobs = []
costs = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
for net in ["HK N=1000 (m=3)", "LastFM Asia (N=7,624)", "Twitch EN (N=7,126)"]:
    for c in costs:
        for r in range(4):
            kw = dict(econ); kw.update(PRIORS["identical prior\nTPR~Beta(2,1), FPR~Beta(1,2)"]); kw.update(cost=c, seed=r)
            if net.startswith("HK"): kw.update(N=1000)
            else: kw.update(graph=nets["lastfm" if "LastFM" in net else "twitch"])
            jobs.append((kw, dict(net=net, cost=c, rep=r)))
D2 = run_many(jobs, verbose=False, chunksize=1)
for net, col in zip(["HK N=1000 (m=3)", "LastFM Asia (N=7,624)", "Twitch EN (N=7,126)"], [fs.GREY, fs.BLUE, fs.ORANGE]):
    g = D2[D2.net == net].groupby("cost").agg(m=("frac_locked", "mean"), s=("frac_locked", "sem"), d=("dip4", "mean"), ds=("dip4", "sem"))
    ax.errorbar(g.index, g.m, yerr=1.96 * g.s, color=col, marker="o", ms=2.5, label=net + ": locked out")
    ax.errorbar(g.index, g.d, yerr=1.96 * g.ds, color=col, marker="s", ms=2.5, ls="--", label=net + ": 4·dip")
ax.set_xscale("log"); ax.set_xlabel("cost / gain (avg. agent)"); ax.set_ylabel("share / index"); ax.set_ylim(0, 1)
ax.legend(fontsize=5.3, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"]); ax.minorticks_off()
ax.set_title("degree distribution matters\n(identical priors)", fontsize=6.3)
# D: trap model vs simulation by degree
ax = fig.add_subplot(gs[1, 2]); fs.panel_label(ax, "D", y=1.22)
rows = []
for c in [0.1, 0.2]:
    for r in range(6):
        kw = dict(econ); kw.update(PRIORS["identical prior\nTPR~Beta(2,1), FPR~Beta(1,2)"]); kw.update(N=1000, cost=c, seed=r)
        R = E.run(**kw); chk = R["c_checks"][:, 0]; q = chk[R["c_fake"]].sum() / max(chk.sum(), 1)
        J = R["J"][:, 0]; th = R["theta"][:, 0]
        for i in range(R["N"]):
            rows.append(dict(cost=c, q=q, deg=R["deg"][i], theta=th[i], locked=J[i] <= th[i]))
A = pd.DataFrame(rows)
A["dbin"] = pd.cut(A.deg, [2, 3, 4, 5, 7, 10, 15, 25, 1000])
for c, col in [(0.1, fs.BLUE), (0.2, fs.ORANGE)]:
    d = A[A.cost == c]; g = d.groupby("dbin", observed=True).agg(deg=("deg", "median"), theta=("theta", "mean"), locked=("locked", "mean"))
    pm = [p_trap(x, q=d.q.mean(), n_mc=8000)[0] for x in g.theta]
    ax.plot(g.deg, g.locked, "o", color=col, ms=3, label=f"simulation, cost {c}")
    ax.plot(g.deg, pm, "-", color=col, lw=1, label=f"first-passage model, cost {c}")
ax.set_xscale("log"); ax.set_xlabel("degree"); ax.set_ylabel("share locked out"); ax.legend(fontsize=5, loc="center right")
ax.set_xticks([3, 5, 10, 20, 50]); ax.set_xticklabels(["3", "5", "10", "20", "50"]); ax.minorticks_off()
ax.set_title("who gets trapped:\nfirst-passage model vs simulation", fontsize=6.3)
# E: two identical checkers
ax = fig.add_subplot(gs[1, 3]); fs.panel_label(ax, "E", y=1.22)
kw = dict(econ); kw.update(PRIORS["identical prior\nTPR~Beta(2,1), FPR~Beta(1,2)"]); kw.update(N=1000, cost=0.1, K=2, seed=3)
R = E.run(**kw); s = summarize(R)
rng = np.random.default_rng(0)
ax.scatter(R["J"][:, 0] + rng.normal(0, 0.012, R["N"]), R["J"][:, 1] + rng.normal(0, 0.012, R["N"]), s=1.5, alpha=0.35, color=fs.INK)
ax.set_xlim(-0.2, 1); ax.set_ylim(-0.2, 1); ax.set_xlabel("trust in checker A"); ax.set_ylabel("trust in checker B")
ax.set_title(f"two identical checkers\ncorr(A,B) = {s['corr_AB']:.2f}", fontsize=6.3)
fig.savefig(OUT, bbox_inches="tight", dpi=200)
print("saved")
