"""Figure: consequences of the split for what spreads and for welfare (from the phase-diagram runs)."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()


def make(phase_parquet, out, lam=2.0, costs_show=None):
    D = pd.read_parquet(phase_parquet); D = D[D.lam == lam]
    costs = np.sort(D.cost.unique())
    if costs_show is None:
        costs_show = [costs[np.argmin(np.abs(costs - c))] for c in [0.01, 0.05, 0.15, 0.5]]
    shades = ["#0b2e5c", fs.BLUE, "#7fb0ea", "#c9d8ee"]
    fig, axs = plt.subplots(1, 4, figsize=(fs.W2, 2.0), gridspec_kw=dict(wspace=0.65, width_ratios=[1, 1, 1, 1.1]))
    for ax, (m, lab) in zip(axs[:3], [("delta", "truth's edge in reach Δ\n(genuine − fake)"), ("welfare_sen", "welfare (Sen)"),
                                       ("mean_pi", "believed share of genuine items")]):
        for c, col in zip(costs_show, shades):
            g = D[D.cost == c].groupby("phi")[m].agg(["mean", "sem"])
            ax.fill_between(g.index, g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"], color=col, alpha=0.25, lw=0)
            ax.plot(g.index, g["mean"], color=col, label=f"c/g = {c:.2g}")
        ax.set_xlabel("share of fakes φ"); ax.set_ylabel(lab); ax.set_xlim(0, 1); ax.set_xticks([0.1, 0.5, 0.9])
    axs[2].plot([0, 1], [1, 0], color=fs.INK2, lw=0.6, ls=":"); axs[2].text(0.55, 0.32, "actual", fontsize=5.6, color=fs.INK2, rotation=-33)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.4, 0.98), ncol=4, fontsize=5.8, frameon=False, columnspacing=1.6)
    # D: truth's edge vs share locked out, all parameter points (phi between 0.2 and 0.6)
    ax = axs[3]
    P = D[D.phi.between(0.19, 0.61)].groupby(["phi", "cost"])[["delta", "frac_locked", "reach_true"]].mean().reset_index()
    P = P[P.reach_true >= 0.05]                                   # skip cells where nothing circulates
    sc = ax.scatter(P.frac_locked, P.delta, c=np.log10(P.cost), s=5, cmap="viridis", lw=0)
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8]); cb = fig.colorbar(sc, cax=cax)
    cb.set_ticks([-3, -2, -1, 0]); cb.set_ticklabels(["0.001", "0.01", "0.1", "1"]); cb.set_label("c/g", fontsize=5.6, labelpad=1)
    cb.ax.tick_params(labelsize=5)
    r = np.corrcoef(P.frac_locked, P.delta)[0, 1]
    ax.text(0.97, 0.95, f"φ = 0.2–0.6\nr = {r:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.4)
    ax.set_xlabel("share locked out"); ax.set_ylabel("truth's edge Δ")
    for a, L in zip(axs, "ABCD"):
        fs.panel_label(a, L, x=-0.32)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
