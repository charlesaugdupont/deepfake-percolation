"""v3 Figure 2: one checker over time at the main setting (phi = 0.3, l/g = 2, c/g = 0.05).
(A) distribution of trust in the checker at five moments (stacked by group), (B) shares of the three groups over time,
(C) split of trust (Hartigan's dip, 0-0.25) over time for three checking costs (20 runs each). Since 6 Nov the runs
continue to 16,000 items (time_long_twitter: same seeds, so the first 2,000 items are those of time_twitter) and the
triangles marking the ceiling set by the group sizes are gone (ceiling() still computes it for the text).
Input: <data>/time_long_twitter_agents.pkl, or <data>/time_twitter_agents.pkl (2,000 items) if the former is missing."""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

BINS = np.linspace(-1, 1, 81)
GROUPS = [("never checked", fs.NEVER), ("checked, then stopped", fs.LOCKED), ("still checks", fs.USER)]


def ci95(a):
    a = np.asarray(a, float); m = a.mean(0); h = fs.ci95_hw(a.std(0, ddof=1), len(a))   # Student's t over runs
    return m, m - h, m + h


def ceiling(runs):
    """Dip at the end of the run if every agent who still checks held the same trust (their median): the most the split
    can reach given the sizes of the groups (about half the share of the smaller group; 0.25 for two equal groups)."""
    import diptest
    v = []
    for r in runs:
        J = r["J_end"].astype(float); u = r["ever_end"] & (J > r["theta"])
        J2 = J.copy(); J2[u] = np.median(J[u]); v.append(diptest.dipstat(J2))
    return float(np.mean(v))


def make(agents_pkl, out, c_main=0.05, moments=(0, 10, 50, 300, 2000), J_true=0.70, J0=1 / 3):
    A = pickle.load(open(agents_pkl, "rb"))
    runs = [p for m, p in A if abs(m["cost"] - c_main) < 1e-9]
    tt = runs[0]["times"]
    fig = plt.figure(figsize=(fs.W2, 4.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.0], hspace=0.62)
    top = outer[0].subgridspec(1, len(moments), wspace=0.08)
    H = np.mean([r["hist"] for r in runs], 0)                       # snapshots x groups x bins (counts)
    N = H[0].sum()
    w = np.diff(BINS); left = BINS[:-1]
    first = None
    for k, t in enumerate(moments):
        ax = fig.add_subplot(top[0, k], sharey=first); si = int(np.searchsorted(tt, t)); first = first or ax
        for g in (0, 2, 1):                          # overlaid (not stacked), semi-transparent; the small group drawn last
            lab, col = GROUPS[g]
            ax.bar(left, 100 * H[si, g] / N, width=w, align="edge", color=col, alpha=fs.HIST_ALPHA, edgecolor="none", linewidth=0,
                   label=lab)
        ax.axvline(J_true, color=fs.ACTUAL, lw=0.7, ls="--")
        ax.set_xlim(-0.3, 1.0); ax.set_yscale("log"); ax.set_ylim(0.01, 150)
        ax.set_title(f"after {tt[si]:,} items" if t > 0 else "start", fontsize=6.3)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0.01, 0.1, 1, 10, 100]); ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"]); ax.minorticks_off()
        if k == 0:
            ax.set_ylabel("% of agents (log scale)"); fs.panel_label(ax, "A", x=-0.62, y=1.08)
        else:
            ax.tick_params(labelleft=False)
        if k == len(moments) // 2:
            ax.set_xlabel("trust in the checker Ĵ")
        if k == len(moments) - 1:
            ax.text(J_true + 0.03, 120, "checker's\nactual J", fontsize=5.0, va="top", color=fs.INK)
    h, l = fig.axes[0].get_legend_handles_labels()
    order = [l.index(g[0]) for g in GROUPS]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 0.965), ncol=3,
               fontsize=6.0, frameon=False, columnspacing=1.6)

    bot = outer[1].subgridspec(1, 2, wspace=0.32)
    # B: group shares over time
    ax = fig.add_subplot(bot[0, 0]); fs.panel_label(ax, "B", x=-0.16)
    x = np.maximum(tt, 1)
    never = [1 - r["ever"] for r in runs]; burned = [r["burned"] for r in runs]; active = [r["ever"] - r["burned"] for r in runs]
    for i, (arr, (lab, col)) in enumerate(zip([never, burned, active], GROUPS)):
        m, lo, hi = ci95(arr)
        ax.fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(x, m, color=col, ls=fs.DASHES[i], lw=1.3, label=lab)
        ax.plot(x[::12], m[::12], ls="none", marker=fs.MARKERS[i], color=col, ms=2.8, mec="white", mew=0.3)
        ax.text(x[-1] * 1.08, m[-1], f"{100 * m[-1]:.0f}%", fontsize=5.6, va="center", color=fs.INK)
    ax.set_xscale("log"); ax.set_xlim(1, tt[-1]); ax.set_ylim(0, 1.02)
    tk = [t for t in (1, 10, 100, 1000, 10000) if t <= tt[-1]]
    ax.set_xticks(tk); ax.set_xticklabels([f"{t:,}" for t in tk])
    ax.set_xlabel("items circulated"); ax.set_ylabel("share of agents")
    ax.set_title("who checks, over time", fontsize=6.5)
    # C: the dip over time for three costs (no ceiling markers since 6 Nov: the runs now reach the plateau, except the
    # slow approach at c/g = 0.2; ceiling() gives the values for the text)
    ax = fig.add_subplot(bot[0, 1]); fs.panel_label(ax, "C", x=-0.16)
    costs = sorted({m["cost"] for m, p in A})
    cols = fs.blues(len(costs)); nrun = []
    for i, (c, col) in enumerate(zip(costs, cols)):
        rr = [p for m, p in A if abs(m["cost"] - c) < 1e-9]; nrun.append(len(rr))
        m_, lo, hi = ci95([r["dip4"] / 4 for r in rr])                # the payload stores 4 x dip
        ax.fill_between(x, lo, hi, color=col, alpha=0.25, lw=0)
        ax.plot(x, m_, color=col, ls=fs.DASHES[i], lw=1.3)
        ax.plot(x[::12], m_[::12], ls="none", marker=fs.MARKERS[i], color=col, ms=2.8, mec="white", mew=0.3)
        ax.plot([], [], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.8, label=f"c/g = {c:g}")
    h, l = ax.get_legend_handles_labels()
    ax.set_xscale("log"); ax.set_xlim(1, tt[-1]); ax.set_ylim(0, 0.25)
    tk = [t for t in (1, 10, 100, 1000, 10000) if t <= tt[-1]]
    ax.set_xticks(tk); ax.set_xticklabels([f"{t:,}" for t in tk])
    ax.set_xlabel("items circulated"); ax.set_ylabel("split of trust (dip)")
    ax.set_title("split of trust, over time", fontsize=6.5)
    ax.legend(h, l, loc="upper left", fontsize=5.6)                    # runs per cost (20 each) go in the caption
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
