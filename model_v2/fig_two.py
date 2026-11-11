"""Figure: two identical checkers, divergent loyalties (+ asymmetric case)."""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()


def ci(df, col, by="cost"):
    g = fs.agg_ci(df, by, col)                                          # 95% CI, Student's t over runs (5 Nov)
    return g.index.values, g["mean"].values, g["hw"].values


def degree_axis(ax, kmin, kmax):
    """Plain integer ticks on a log degree axis (avoids crowded 2 × 10¹-style labels on narrow ranges)."""
    from matplotlib.ticker import NullFormatter
    cand = [1, 10, 100, 1000, 10000]
    if kmax / max(kmin, 1) < 30:
        cand = [3, 5, 10, 20, 30, 50, 100]
    tk = [t for t in cand if kmin * 0.8 <= t <= kmax * 1.25]
    ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk]); ax.xaxis.set_minor_formatter(NullFormatter())


def by_degree(runs, n_bins, fns, min_n=30):
    """Share of agents per degree bin for each function in fns, as mean and 95% CI half-width over runs."""
    deg_all = np.concatenate([a["deg"] for a in runs])
    edges = np.unique(np.round(np.geomspace(1, deg_all.max() + 1, n_bins)))
    cen, mus, errs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if ((deg_all >= lo) & (deg_all < hi)).sum() < min_n:
            continue
        V = np.array([[f(a)[(a["deg"] >= lo) & (a["deg"] < hi)].mean() if ((a["deg"] >= lo) & (a["deg"] < hi)).any() else np.nan
                       for f in fns] for a in runs])
        n = np.sum(~np.isnan(V), axis=0)
        sd = np.nanstd(V, axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(fns))
        cen.append(np.sqrt(lo * max(hi - 1, lo))); mus.append(np.nanmean(V, axis=0)); errs.append(fs.ci95_hw(sd, n))
    return np.array(cen), np.array(mus), np.array(errs)


def make(prefix, out, rep_cost=0.05):
    S = pd.read_parquet(prefix + ".parquet"); A = pickle.load(open(prefix + "_agents.pkl", "rb"))
    sym = S[S.design == "symmetric"]
    fig = plt.figure(figsize=(fs.W2, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=1.0, wspace=0.75)

    # A: joint distribution of trust in A and B (symmetric, representative cost), pooled over runs
    ax = fig.add_subplot(gs[0, 0]); fs.panel_label(ax, "A", x=-0.3)
    JA = np.concatenate([a["JA"] for e, a in A if e["design"] == "symmetric" and abs(e["cost"] - rep_cost) < 1e-9])
    JB = np.concatenate([a["JB"] for e, a in A if e["design"] == "symmetric" and abs(e["cost"] - rep_cost) < 1e-9])
    bins = np.linspace(-0.3, 1.0, 66)
    H, xe, ye = np.histogram2d(JA, JB, bins=[bins, bins])
    cmap = LinearSegmentedColormap.from_list("k", ["#eef3fb", "#9cc2ef", "#2a78d6", "#154f99", "#0b2e5c"])
    pc = ax.pcolormesh(xe, ye, (H / H.sum()).T, norm=LogNorm(vmin=1e-4, vmax=0.3), cmap=cmap, rasterized=True)
    ax.set_aspect("equal"); ax.set_xlabel("trust in checker A  Ĵ_A"); ax.set_ylabel("trust in checker B  Ĵ_B")
    ax.axvline(0.70, color=fs.GREEN, lw=0.6, ls="--"); ax.axhline(0.70, color=fs.GREEN, lw=0.6, ls="--")
    ax.text(0.72, 0.93, "actual", color=fs.GREEN, fontsize=5.6)
    ax.annotate("relies on A,\nnever tried B", xy=(0.70, 0.333), xytext=(-0.27, -0.2), fontsize=5.6, arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK2))
    ax.annotate("relies on B,\nnever tried A", xy=(0.333, 0.70), xytext=(-0.25, 0.86), fontsize=5.6, arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK2))
    ax.set_title(f"identical checkers, c/g = {rep_cost:.2g}", fontsize=6.2)
    cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0]); cb = fig.colorbar(pc, cax=cax); cb.ax.tick_params(labelsize=5)
    cb.set_label("share of agents", fontsize=5.2, labelpad=1)

    # B: who uses what, vs cost
    ax = fig.add_subplot(gs[0, 1]); fs.panel_label(ax, "B", x=-0.3)
    for col, c_, lab in [("used_onlyA", fs.CHK_A, "only A"), ("used_onlyB", fs.CHK_B, "only B"), ("used_both", fs.GREEN, "both"),
                         ("used_none", fs.GREY, "neither")]:
        x, m, e = ci(sym, col)
        ax.fill_between(x, m - e, m + e, color=c_, alpha=0.2, lw=0); ax.plot(x, m, color=c_, marker="o", ms=1.8, label=lab)
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("checking cost / gain  c/g"); ax.set_ylabel("share of agents who used…")
    ax.legend(fontsize=5.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3), columnspacing=0.8, handlelength=1.4, frameon=False)

    # C: loyalty among agents who check
    ax = fig.add_subplot(gs[0, 2]); fs.panel_label(ax, "C")
    x, m, e = ci(sym, "dominance"); ax.fill_between(x, m - e, m + e, color=fs.INK, alpha=0.15, lw=0); ax.plot(x, m, color=fs.INK, label="checks via preferred checker")
    x, m, e = ci(sym, "excl_share"); ax.fill_between(x, m - e, m + e, color=fs.PURPLE, alpha=0.15, lw=0); ax.plot(x, m, color=fs.PURPLE, label="never tried the other")
    x, m, e = ci(sym, "corr_AB"); ax.fill_between(x, m - e, m + e, color=fs.RED, alpha=0.15, lw=0); ax.plot(x, m, color=fs.RED, label="corr(Ĵ_A, Ĵ_B)")
    ax.axhline(0, color=fs.INK2, lw=0.5); ax.axhline(0.5, color=fs.INK2, lw=0.4, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("checking cost / gain  c/g"); ax.set_ylim(-0.7, 1.05)
    ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.3), handlelength=1.4, frameon=False)
    ax.set_ylabel("loyalty among checkers")

    # D: asymmetric: cheap AI detector vs costly accurate professional checker, by degree
    ax = fig.add_subplot(gs[1, 0]); fs.panel_label(ax, "D", x=-0.3, y=1.12)
    asy = [(e, a) for e, a in A if e["design"] == "asymmetric" and abs(e["cost"] - rep_cost) < 1e-9 and e["ratio"] == 3.0]
    if asy:
        fns = [lambda a: (a["nA"] > 0) & (a["nB"] == 0), lambda a: a["nB"] > 0, lambda a: (a["nA"] == 0) & (a["nB"] == 0)]
        cen, mu, er = by_degree([a for e, a in asy], 9, fns)
        for k, (c_, lab) in enumerate([(fs.CHK_A, "AI only"), (fs.CHK_B, "professional"), (fs.GREY, "never")]):
            ax.fill_between(cen, mu[:, k] - er[:, k], mu[:, k] + er[:, k], color=c_, alpha=0.2, lw=0)
            ax.plot(cen, mu[:, k], color=c_, marker="o", ms=2, label=lab)
        ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("number of neighbours k"); ax.set_ylabel("share of agents")
        degree_axis(ax, min(cen), max(cen))
        ax.set_title("AI detector (cost c) vs professional\n(cost 3c, TPR 0.95, FPR 0.03)", fontsize=6)
        ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, handlelength=1.4, columnspacing=0.8, frameon=False)
    # E: asymmetric: share locked out of each checker vs cost
    ax = fig.add_subplot(gs[1, 1]); fs.panel_label(ax, "E", x=-0.3, y=1.12)
    asyS = S[S.design == "asymmetric"]
    for ratio, ls in [(3.0, "-"), (10.0, "--")]:
        sub = asyS[asyS.ratio == ratio]
        for col, c_, lab in [("frac_locked_A", fs.CHK_A, "AI detector"), ("frac_locked_B", fs.CHK_B, "professional")]:
            x, m, e = ci(sub, col)
            ax.fill_between(x, m - e, m + e, color=c_, alpha=0.15, lw=0)
            ax.plot(x, m, color=c_, ls=ls, label=(f"professional ({ratio:g}× cost)" if col.endswith("B") else ("AI detector" if ratio == 3.0 else None)))
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("AI detector cost / gain"); ax.set_ylabel("share locked out of each checker")
    ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1.6, columnspacing=0.8, frameon=False)
    # F: symmetric case by degree: who relies on one checker, who uses both
    ax = fig.add_subplot(gs[1, 2]); fs.panel_label(ax, "F", x=-0.3, y=1.12)
    sy = [(e, a) for e, a in A if e["design"] == "symmetric" and abs(e["cost"] - rep_cost) < 1e-9]
    fns = [lambda a: (a["nA"] > 0) ^ (a["nB"] > 0), lambda a: (a["nA"] > 0) & (a["nB"] > 0), lambda a: (a["nA"] == 0) & (a["nB"] == 0)]
    cen, mu, er = by_degree([a for e, a in sy], 10, fns)
    for k, (c_, lab) in enumerate([(fs.PURPLE, "only one"), (fs.GREEN, "both"), (fs.GREY, "neither")]):
        ax.fill_between(cen, mu[:, k] - er[:, k], mu[:, k] + er[:, k], color=c_, alpha=0.2, lw=0)
        ax.plot(cen, mu[:, k], color=c_, marker="o", ms=2, label=lab)
    R_ = cen[:, None]
    ax.set_xscale("log"); ax.set_ylim(0, 1); ax.set_xlabel("number of neighbours k"); ax.set_ylabel("share of agents who used…")
    degree_axis(ax, R_[:, 0].min(), R_[:, 0].max())
    ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, handlelength=1.4, columnspacing=0.8, frameon=False); ax.set_title(f"identical checkers, c/g = {rep_cost:.2g}", fontsize=6)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.05)
