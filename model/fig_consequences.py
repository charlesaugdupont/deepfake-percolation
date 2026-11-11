"""v3 Figure 4: consequences (phase-diagram runs, l/g = 2 unless given).
(A) how much further genuine items travel than fakes: reach of genuine items minus reach of fakes (share of agents
    reached), against the share of fakes, for four relative costs of checking;
(B) welfare: Sen index (mean normalised payoff x (1 - Gini)), checking costs included;
(C) believed against actual share of fakes (mean over agents at the end of the run);
(D) the reach gap against the share not checking (cells with phi = 0.2-0.6 where items circulate; colour = c/g).
Usage: fig_consequences.py <phase parquet> <out.png> [lam]"""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

COSTS = [0.001, 0.01, 0.1, 1.0]


def make(parquet, out, lam=2.0):
    D = pd.read_parquet(parquet); D = D[np.isclose(D.lam, lam)].copy()
    D["believed_fake"] = 1 - D.mean_pi
    cols = fs.blues(len(COSTS))
    fig, axs = plt.subplots(1, 4, figsize=(fs.W2, 2.15), gridspec_kw=dict(wspace=0.62, width_ratios=[1, 1, 1, 1.08]))
    panels = [("delta", "reach of genuine items\nminus reach of fakes"), ("welfare_sen", "welfare (Sen index)"),
              ("believed_fake", "believed share of fakes")]
    for ax, (m, lab) in zip(axs[:3], panels):
        for i, (c, col) in enumerate(zip(COSTS, cols)):
            sub = D[np.isclose(D.cost, c)]
            if len(sub) == 0:
                continue
            g = fs.agg_ci(sub, "phi", m)
            g = g[np.isclose((g.index * 1000).round() % 50, 0)]               # even lattice rows only (phi = 0.05, 0.10, ...)
            ax.fill_between(g.index, g["mean"] - g["hw"], g["mean"] + g["hw"], color=col, alpha=0.25, lw=0)
            ax.plot(g.index, g["mean"], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.4, mec="white", mew=0.3,
                    markevery=2, label=f"c/g = {c:g}")
        ax.set_xlabel("share of fakes φ"); ax.set_ylabel(lab); ax.set_xlim(0, 1); ax.set_xticks([0.1, 0.5, 0.9])
    axs[0].set_ylim(-0.05, 1.0); axs[1].set_ylim(-0.05, 1.0); axs[2].set_ylim(0, 1)
    axs[2].plot([0, 1], [0, 1], color=fs.INK2, lw=0.6, ls=":")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.4, 0.97), ncol=4, fontsize=5.8, frameon=False, columnspacing=1.6,
               handlelength=2.6, title=f"relative cost of checking (ℓ/g = {lam:g})", title_fontsize=5.8)
    # D: reach gap vs share not checking
    ax = axs[3]
    P = D[D.phi.between(0.19, 0.61)].groupby(["phi", "cost"])[["delta", "frac_locked", "reach_true"]].mean().reset_index()
    P = P[P.reach_true >= 0.05]
    cmap = LinearSegmentedColormap.from_list("cost", [fs.CM_LOCK(x) for x in np.linspace(0.3, 1.0, 8)])
    sc = ax.scatter(P.frac_locked, P.delta, c=np.log10(P.cost), s=5, cmap=cmap, vmin=-3, vmax=0, lw=0)
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8]); cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([-3, -2, -1, 0]); cb.set_ticklabels(["0.001", "0.01", "0.1", "1"]); cb.set_label("c/g", fontsize=5.6, labelpad=1)
    cb.ax.tick_params(labelsize=5)
    r = np.corrcoef(P.frac_locked, P.delta)[0, 1]
    ax.text(0.97, 0.97, f"φ = 0.2–0.6\nr = {r:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.4)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("share not checking"); ax.set_ylabel("reach of genuine items\nminus reach of fakes")
    for a, L in zip(axs, "ABCD"):
        fs.panel_label(a, L, x=-0.36, y=1.03)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


def cost_lines(ax, D, m, costs=COSTS):
    """One line per relative cost of checking: mean over runs of metric m against the share of fakes, 95% CI band
    (Student's t). Reach and welfare over all runs (where nothing circulates welfare is ~0); behaviour and beliefs over
    the runs in which items circulate, and only in cells that are not hatched in the maps (5 Nov)."""
    cols = fs.blues(len(costs))
    for i, (c, col) in enumerate(zip(costs, cols)):
        sub = D[np.isclose(D.cost, c)]
        if len(sub) == 0:
            continue
        if m not in fs.ALL_RUNS:
            live = sub.groupby("phi").reach_true.mean() >= fs.MIN_REACH
            sub = fs.circulating(sub[sub.phi.map(live)])
        g = fs.agg_ci(sub, "phi", m)
        g = g[np.isclose((g.index * 1000).round() % 50, 0)]               # even lattice rows only (phi = 0.05, 0.10, ...)
        ax.fill_between(g.index, g["mean"] - g["hw"], g["mean"] + g["hw"], color=col, alpha=0.25, lw=0)
        ax.plot(g.index, g["mean"], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.4, mec="white", mew=0.3,
                markevery=2, label=f"c/g = {c:g}")
    ax.set_xlabel("share of fakes φ"); ax.set_xlim(0, 1); ax.set_xticks([0.1, 0.5, 0.9])


def make_si(parquet, out, lam=2.0):
    """SI figure (the reach gap is in main Fig. 4): (A) welfare (Sen index, checking costs included) and (B) believed
    against actual share of fakes, against the share of fakes, for four relative costs of checking."""
    D = pd.read_parquet(parquet); D = D[np.isclose(D.lam, lam)].copy()
    D["believed_fake"] = 1 - D.mean_pi
    cols = fs.blues(len(COSTS))
    fig, axs = plt.subplots(1, 2, figsize=(fs.W2 * 0.62, 2.2), gridspec_kw=dict(wspace=0.5))
    for ax, (m, lab) in zip(axs, [("welfare_sen", "welfare (Sen index)"), ("believed_fake", "believed share of fakes")]):
        for i, (c, col) in enumerate(zip(COSTS, cols)):
            sub = D[np.isclose(D.cost, c)]
            g = fs.agg_ci(sub, "phi", m)
            g = g[np.isclose((g.index * 1000).round() % 50, 0)]               # even lattice rows only (phi = 0.05, 0.10, ...)
            ax.fill_between(g.index, g["mean"] - g["hw"], g["mean"] + g["hw"], color=col, alpha=0.25, lw=0)
            ax.plot(g.index, g["mean"], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.4, mec="white", mew=0.3,
                    markevery=2, label=f"c/g = {c:g}")
        ax.set_xlabel("share of fakes φ"); ax.set_ylabel(lab); ax.set_xlim(0, 1); ax.set_xticks([0.1, 0.5, 0.9])
    axs[0].set_ylim(-0.05, 1.0); axs[1].set_ylim(0, 1)
    axs[1].plot([0, 1], [0, 1], color=fs.INK2, lw=0.6, ls=":")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=4, fontsize=5.8, frameon=False, columnspacing=1.4,
               handlelength=2.6, title=f"relative cost of checking (ℓ/g = {lam:g})", title_fontsize=5.8)
    for a, L in zip(axs, "AB"):
        fs.panel_label(a, L, x=-0.36, y=1.03)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make_si(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
