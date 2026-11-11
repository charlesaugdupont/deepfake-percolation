"""SI figure: the split follows the inequality of audiences, not the average audience (added 6 Nov; four panels since
6 Nov 12:35). Random networks (synthnets.py; N = 5,000), main setting otherwise (phi = 0.3, l/g = 2); 10 runs per point,
each on its own wiring of the network (same degree sequence).
Top row: average audience fixed at 20 neighbours, inequality of audiences rising (Gini of the number of neighbours
0, 0.26, 0.45, 0.64, 0.73 shown; 0.14 and 0.56 were also run). (A) split of trust (dip) and (B) share not checking
against the relative cost of checking.
Bottom row: inequality fixed (CV 1, Gini 0.45), average audience rising (4.35, 20 and 92.8 neighbours shown; 9.28 and
43.1 were also run). (C) split and (D) share not checking against the relative cost, as in the model (stakes relative
to the average audience); 92.8 is drawn with hollow markers on top of 20, because the two curves coincide (the point
of the row; 6 Nov 12:38).
In the text only (`--numbers`): the largest split and the range of costs with a clear split (dip > 0.1) for every
inequality level, and the absolute-stakes reading of the bottom-row runs. Decisions depend only on cost/stake, so with
stakes k_i/20 (the same gain for every neighbour reached) an agent at cost c acts as under relative stakes k_i/<k> at
cost c x 20/<k>; each fixed absolute cost is therefore a different relative cost on each network (exact; the sparsest
level, whose largest component has <k> = 4.35 instead of 4.31, sits 1% off).
Bands: 95% CI (Student's t) over runs. The dip and share not checking are averaged over runs in which items circulate:
on the (nearly) equal networks (Gini 0-0.26) at c/g >= 0.2-0.46, where checking never pays, nothing circulates in 2-8
of the 10 runs (nobody checks and too few share); there the dip is 0 and the share not checking 1 in every run that
circulates. On the sparsest network (4.35 neighbours) a fifth of the agents have one neighbour: they hear from a single
sender, learn its record fast and rarely meet an item doubtful enough to check (median 2 checks in 2,000 items), so
their trust stays near the start, a small split of another kind (rare vs frequent checkers) at low cost.
Usage: si_inequality.py <data dir> <out.png> [--numbers]"""
import sys, os, glob, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

SHOW_CV = (0.0, 0.5, 1.0, 2.0, 3.0)         # top row: Gini 0, 0.26, 0.45, 0.64, 0.73 (of 7 levels run)
SHOW_K = (4.31, 20.0, 92.83)                # bottom row: 4.35, 20, 92.8 neighbours (of 5 levels run)
W15 = 11.4 * fs.CM                          # PNAS 1.5-column width
THR = 0.1                                   # a clear split (as in the network comparison, Fig. S8)
K_REF = 20.0                                # absolute stakes: c/g relative to the gain of an agent with 20 neighbours
ABS_COSTS = (0.01, 0.0464, 0.2154)          # fixed absolute costs reported in the text (on the cost grid for every level)
COSTTICKS = ([0.001, 0.01, 0.1, 1], ["0.001", "0.01", "0.1", "1"])


def load(data, name):
    """Merged results, or (while runs are still going) the chunk files."""
    p = os.path.join(data, f"{name}.parquet")
    if os.path.exists(p):
        D = pd.read_parquet(p)
    else:
        D = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(data, name, "chunk_*.parquet"))], ignore_index=True)
    D["dip"] = D.dip4 / 4                    # runs store 4 x dip
    return D


def levels(data):
    """Runs and level descriptions of both arms (the 20-neighbour, CV 1 level belongs to both)."""
    D1 = load(data, "ineq_cv"); D2 = load(data, "ineq_k")
    NET = pd.read_csv(os.path.join(data, "synthetic_networks.csv"))
    lev1 = NET[np.isclose(NET.kbar_target, 20.0)].groupby("cv_target").agg(gini=("gini", "mean"), mean=("mean", "mean"))
    lev1 = lev1[lev1.index.isin(D1.cv.unique())]
    lev2 = NET[np.isclose(NET.cv_target, 1.0)].groupby("kbar_target").agg(gini=("gini", "mean"), mean=("mean", "mean"))
    B = pd.concat([D1[np.isclose(D1.cv, 1.0)], D2], ignore_index=True)
    lev2 = lev2[lev2.index.isin(B.kbar.unique())]
    return D1, B, lev1, lev2


def ramp(cmap, n):
    return [cmap(x) for x in np.linspace(0.38, 1.0, n)]


def width_above(x, y, thr):
    """Length of {x: y(x) > thr} for y linear between grid points x (here log10 cost)."""
    w = 0.0
    for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if y0 > thr and y1 > thr:
            w += x1 - x0
        elif (y0 > thr) != (y1 > thr):
            w += (x1 - x0) * (max(y0, y1) - thr) / abs(y1 - y0)
    return w


def per_run(sub):
    """Per run (= wiring): the largest dip over costs and the width (decades) of the cost range with dip > THR."""
    rows = []
    for r, d in fs.circulating(sub).groupby("rep"):
        d = d.sort_values("cost")
        rows.append(dict(rep=r, peak=d.dip.max(), width=width_above(np.log10(d.cost.values), d.dip.values, THR)))
    return pd.DataFrame(rows)


def numbers(data):
    """The numbers reported in the text (RESULTS_v3.md, Fig. S11)."""
    D1, B, lev1, lev2 = levels(data)
    print("largest split and cost range with dip > 0.1 (decades), mean ± 95% CI over 10 runs:")
    for cv, L in lev1.iterrows():
        pr = per_run(D1[np.isclose(D1.cv, cv)])
        print(f"  Gini {L.gini:.2f} (CV {cv:g}): peak {pr.peak.mean():.3f} ± {fs.ci95_hw(pr.peak.std(ddof=1), len(pr)):.3f}, "
              f"range {pr.width.mean():.2f} ± {fs.ci95_hw(pr.width.std(ddof=1), len(pr)):.2f}")
    print("absolute stakes (gain per neighbour; c/g relative to an agent with 20 neighbours): dip / share not checking")
    L2 = fs.circulating(B)
    for ca in ABS_COSTS:
        cells = []
        for kb, L in lev2.iterrows():
            s = L2[np.isclose(L2.kbar, kb) & np.isclose(L2.cost, ca * K_REF / kb, rtol=2e-3)]
            cells.append(f"{L['mean']:.3g}: {s.dip.mean():.3f} / {s.frac_locked.mean():.2f}")
        print(f"  c/g = {ca:g}: " + "; ".join(cells))


def curve(ax, sub, m, col, i, label, hollow=False):
    g = fs.agg_ci(fs.circulating(sub), "cost", m)
    ax.fill_between(g.index, g["mean"] - g["hw"], g["mean"] + g["hw"], color=col, alpha=0.22, lw=0)
    mk = dict(ms=3.2, mfc="white", mec=col, mew=0.7) if hollow else dict(ms=2.4, mec="white", mew=0.3)
    ax.plot(g.index, g["mean"], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], lw=1.0, label=label, zorder=3 + i, **mk)


def cost_axis(ax):
    ax.set_xscale("log"); ax.set_xlim(8e-4, 1.25); ax.set_xticks(COSTTICKS[0]); ax.set_xticklabels(COSTTICKS[1])
    ax.minorticks_off(); ax.set_xlabel("relative cost c/g")


def make(data, out):
    D1, B, lev1, lev2 = levels(data)
    lev1 = lev1[[any(np.isclose(cv, x) for x in SHOW_CV) for cv in lev1.index]]
    lev2 = lev2[[any(np.isclose(kb, x) for x in SHOW_K) for kb in lev2.index]]
    fig = plt.figure(figsize=(W15, 4.5))
    gs = fig.add_gridspec(2, 2, hspace=1.1, wspace=0.42, top=0.845, bottom=0.095, left=0.13, right=0.98)
    ax = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]
    rows = [(D1, "cv", lev1, ramp(fs.CM_GEN, len(lev1)),
             lambda k, L: "0 (all 20)" if k == 0 else f"{L.gini:.2f}"),
            (B, "kbar", lev2, ramp(fs.CM_GAP, len(lev2)), lambda k, L: f"{L['mean']:.3g}")]
    for r, (D, col, lev, cols, lab) in enumerate(rows):
        for i, (k, L) in enumerate(lev.iterrows()):
            sub = D[np.isclose(D[col], k)]
            hollow = r == 1 and i == len(lev) - 1               # 92.8 drawn hollow on top of 20: the two curves coincide
            curve(ax[r][0], sub, "dip", cols[i], i, lab(k, L), hollow)
            curve(ax[r][1], sub, "frac_locked", cols[i], i, lab(k, L), hollow)
        for c, (ylab, ymax) in enumerate([("split of trust (dip)", 0.25), ("share not checking", 1.0)]):
            a = ax[r][c]; cost_axis(a); a.set_ylim(0, ymax); a.set_ylabel(ylab)
            fs.panel_label(a, "ABCD"[2 * r + c], x=-0.33, y=1.04)
    # row titles and legends (one line each, above the row)
    for r, (title, ltitle, ytitle, yleg) in enumerate([
            ("same average audience (20 neighbours), rising inequality", "Gini of number of neighbours", 0.985, 0.958),
            ("same inequality (Gini 0.45), rising average audience", "average number of neighbours", 0.49, 0.463)]):
        fig.text(0.555, ytitle, title, ha="center", va="top", fontsize=6.4, fontweight="bold", color=fs.INK)
        h, l = ax[r][0].get_legend_handles_labels()
        fig.legend(h, l, title=ltitle, loc="upper center", bbox_to_anchor=(0.555, yleg), ncol=len(l), fontsize=5.3,
                   title_fontsize=5.5, frameon=False, columnspacing=1.0, handlelength=2.3, handletextpad=0.4,
                   borderaxespad=0.0, borderpad=0.0)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    if "--numbers" in sys.argv:
        numbers(sys.argv[1])
    else:
        make(sys.argv[1], sys.argv[2])
