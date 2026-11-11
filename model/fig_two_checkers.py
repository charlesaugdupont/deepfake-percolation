"""v3 main figure: two identical checkers, and how trust splits between them.
Everyone starts with the same beliefs about two identical checkers A and B (TPR 0.80, FPR 0.10).
(A) trust in A minus trust in B among agents who can still check (above threshold for at least one checker) at five
    moments, stacked by what they have used; main setting (phi = 0.3, l/g = 2, c/g = 0.05), log scale;
(B) share of agents who have used both checkers, only A, only B or neither, over time (main setting);
(C) the split between checkers over time: share of the agents who can still check whose trust in one checker exceeds
    their trust in the other by more than 0.2, for three costs;
(D) loyalty against cost: share of checks done with the preferred checker and share of checking agents who never tried
    the other one;
(E) what agents used, by number of neighbours (main setting);
(F) a cheap AI detector vs a costly accurate professional (3x the cost, TPR 0.95, FPR 0.03), by number of neighbours;
(G) share not using each checker against the AI detector's cost.
Inputs: <data>/two_time_twitter_agents.pkl, <data>/two_twitter.parquet, <data>/two_twitter_agents.pkl."""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from figstyle import ci, degree_axis, by_degree
fs.setup()

D_BINS = np.linspace(-1, 1, 81)
USE = [("used both", fs.BOTH), ("only A", fs.CHK_A), ("only B", fs.CHK_B), ("not yet checked", fs.NEVER)]
OVERLAY_ORDER = [0, 3, 1, 2]      # draw 'used both' first, then 'not yet checked', then the two small one-checker groups
HIST_ALPHA = fs.HIST_ALPHA


def ci95(a):
    a = np.asarray(a, float); m = np.nanmean(a, 0)
    h = fs.ci95_hw(np.nanstd(a, 0, ddof=1), np.sum(~np.isnan(a), 0))   # 95% CI, Student's t over runs (5 Nov)
    return m, m - h, m + h


def line(ax, x, m, e, i, color, label, markevery=None, lw=1.2):
    ax.fill_between(x, m - e, m + e, color=color, alpha=0.18, lw=0)
    ax.plot(x, m, color=color, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.8, mec="white", mew=0.3, label=label,
            markevery=markevery, lw=lw)


def legend_below(ax, ncol=2, y=-0.3):
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=2.2, columnspacing=0.9,
              frameon=False)


def runs_at(TT, phi, cost):
    return [p for m, p in TT if abs(m["phi"] - phi) < 1e-9 and abs(m["cost"] - cost) < 1e-9]


def time_axis(ax, tmax):
    ax.set_xscale("log"); ax.set_xlim(1, tmax); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"])


def histograms(fig, spec, runs, moments, J_true=0.70, J0=1 / 3, label="A", ylab=True):
    top = spec.subgridspec(1, len(moments), wspace=0.08)
    tt = runs[0]["times"]
    H = np.sum([r["histD"] for r in runs], 0).astype(float)          # snapshots x groups x bins (counts)
    axes = []
    for k, t in enumerate(moments):
        ax = fig.add_subplot(top[0, k], sharey=axes[0] if axes else None); axes.append(ax)
        si = int(np.searchsorted(tt, t))
        h = 100 * H[si] / max(H[si].sum(), 1)
        for g in OVERLAY_ORDER:                      # overlaid (not stacked), semi-transparent, small groups drawn last
            lab, col = USE[g]
            ax.bar(D_BINS[:-1], h[g], width=np.diff(D_BINS), align="edge", color=col, alpha=HIST_ALPHA, edgecolor="none",
                   linewidth=0, label=lab)
        ax.axvline(0, color=fs.INK2, lw=0.5, ls=":")
        ax.set_yscale("log"); ax.set_ylim(0.01, 150); ax.set_xlim(-0.75, 0.75)
        ax.set_xticks([-0.5, 0, 0.5]); ax.set_xticklabels(["−0.5", "0", "0.5"])
        ax.set_title(f"after {tt[si]:,} items" if t > 0 else "start", fontsize=6.3)
        if k == 0:
            if ylab:
                ax.set_ylabel("% of agents who\ncan still check (log)")
            fs.panel_label(ax, label, x=-0.62, y=1.08)
        else:
            ax.tick_params(labelleft=False)
        if k == len(moments) // 2:
            ax.set_xlabel("trust in A − trust in B")
        if k == len(moments) - 1:
            ax.text(J_true - J0, 90, "trusts A,\nnever\ntried B", fontsize=4.8, va="top", ha="center", color=fs.CHK_A,
                    fontweight="bold")
            ax.text(-(J_true - J0), 90, "trusts B,\nnever\ntried A", fontsize=4.8, va="top", ha="center", color=fs.CHK_B,
                    fontweight="bold")
    axes[0].set_yticks([0.01, 0.1, 1, 10, 100]); axes[0].set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    for ax in axes:
        ax.minorticks_off()
    return axes


def usage_over_time(ax, runs, title=None, end_labels=True):
    tt = np.maximum(runs[0]["times"], 1)
    for i, (lab, col) in enumerate([("used both", fs.BOTH), ("only A", fs.CHK_A), ("only B", fs.CHK_B), ("neither", fs.NEVER)]):
        g = {"used both": 0, "only A": 1, "only B": 2, "neither": 3}[lab]
        m, lo, hi = ci95([r["use"][:, g] for r in runs])
        ax.fill_between(tt, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(tt, m, color=col, ls=fs.DASHES[i], lw=1.2, label=lab)
        ax.plot(tt[::12], m[::12], ls="none", marker=fs.MARKERS[i], color=col, ms=2.6, mec="white", mew=0.3,
                markevery=(i % 2, 2) if g in (1, 2) else None)
        dy = {0: 0.035, 3: -0.035, 1: 0.03, 2: -0.03}[g]
        if end_labels:
            ax.text(tt[-1] * 1.12, m[-1] + dy, f"{100 * m[-1]:.0f}%", fontsize=5.2, va="center", color=fs.INK)
    time_axis(ax, tt[-1]); ax.set_ylim(0, 1.02)
    ax.set_xlabel("items circulated"); ax.set_ylabel("share of agents")
    if title:
        ax.set_title(title, fontsize=6.3)


def make(data, out, phi=0.3, cost=0.05, moments=(0, 10, 50, 300, 2000)):
    TT = pickle.load(open(os.path.join(data, "two_time_twitter_agents.pkl"), "rb"))
    A2 = pickle.load(open(os.path.join(data, "two_twitter_agents.pkl"), "rb"))
    runs = runs_at(TT, phi, cost)
    fig = plt.figure(figsize=(fs.W2, 6.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.0, 1.0], hspace=0.62, top=0.93)
    axs = histograms(fig, outer[0], runs, moments)
    h, l = axs[0].get_legend_handles_labels()
    order = [l.index(x) for x in [u for u, _ in USE]]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 0.955), ncol=4,
               fontsize=6.0, frameon=False, columnspacing=1.6)

    mid = outer[1].subgridspec(1, 2, wspace=0.42)
    # B: what agents used, over time
    ax = fig.add_subplot(mid[0, 0]); fs.panel_label(ax, "B", x=-0.2, y=1.06)
    usage_over_time(ax, runs, title=f"what agents used, c/g = {cost:g}")
    ax.legend(fontsize=5.6, loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=2, handlelength=2.2, columnspacing=1.0)
    # C: the split between checkers over time, three costs
    ax = fig.add_subplot(mid[0, 1]); fs.panel_label(ax, "C", x=-0.2, y=1.06)
    costs = sorted({m["cost"] for m, p in TT if abs(m["phi"] - phi) < 1e-9}); cols = fs.blues(len(costs))
    for i, (c, col) in enumerate(zip(costs, cols)):
        rr = runs_at(TT, phi, c); tt = np.maximum(rr[0]["times"], 1)
        m, lo, hi = ci95([r["fav"] for r in rr])
        ax.fill_between(tt, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(tt, m, color=col, ls=fs.DASHES[i], lw=1.2)
        ax.plot(tt[::12], m[::12], ls="none", marker=fs.MARKERS[i], color=col, ms=2.6, mec="white", mew=0.3)
        ax.plot([], [], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.6, label=f"c/g = {c:g} ({len(rr)} simulations)")
    time_axis(ax, tt[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel("items circulated"); ax.set_ylabel("share with a clear favourite\n(|trust in A − B| > 0.2)")
    ax.set_title("a clear favourite, among agents who can still check", fontsize=6.3)
    ax.legend(fontsize=5.6, loc="upper left", handlelength=2.2)

    bot = outer[2].subgridspec(1, 2, wspace=0.42)
    # D: by degree, identical checkers
    ax = fig.add_subplot(bot[0, 0]); fs.panel_label(ax, "D", x=-0.2, y=1.06)
    fns = [lambda a: (a["nA"] > 0) ^ (a["nB"] > 0), lambda a: (a["nA"] > 0) & (a["nB"] > 0), lambda a: (a["nA"] == 0) & (a["nB"] == 0)]
    cen, mu, er = by_degree(runs, 12, fns)
    for i, (c_, lab) in enumerate([(fs.ONLY_ONE, "only one checker"), (fs.BOTH, "both"), (fs.NEVER, "neither")]):
        line(ax, cen, mu[:, i], er[:, i], i, c_, lab)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("number of neighbours k"); ax.set_ylabel("share of agents who used…")
    degree_axis(ax, cen.min(), cen.max()); ax.set_title(f"identical checkers, c/g = {cost:g}", fontsize=6.3)
    ax.legend(fontsize=5.6, loc="center left", bbox_to_anchor=(0.02, 0.5), handlelength=2.2)
    # E: AI detector vs professional, by degree
    ax = fig.add_subplot(bot[0, 1]); fs.panel_label(ax, "E", x=-0.2, y=1.06)
    asy = [a for e, a in A2 if e["design"] == "asymmetric" and abs(e["cost"] - cost) < 1e-9 and e["ratio"] == 3.0]
    # nobody used the professional without also using the detector, so 'used the professional' = 'used both'
    fns = [lambda a: (a["nA"] > 0) & (a["nB"] == 0), lambda a: a["nB"] > 0, lambda a: (a["nA"] == 0) & (a["nB"] == 0)]
    cen, mu, er = by_degree(asy, 9, fns)
    for i, (c_, lab) in enumerate([(fs.CHK_A, "AI detector only"), (fs.CHK_B, "both (detector and\nprofessional)"),
                                   (fs.NEVER, "neither")]):
        line(ax, cen, mu[:, i], er[:, i], i, c_, lab)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("number of neighbours k"); ax.set_ylabel("share of agents who used…")
    degree_axis(ax, cen.min(), cen.max())
    ax.set_title(f"AI detector vs professional (3× the cost, TPR 0.95, FPR 0.03)", fontsize=6.3)
    ax.legend(fontsize=5.6, loc="center left", bbox_to_anchor=(0.02, 0.55), handlelength=2.2)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
