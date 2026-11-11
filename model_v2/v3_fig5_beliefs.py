"""Starting beliefs about the checker (SI; drawn by v3_figS_starting.py). (A) split of trust (Hartigan's dip) over the
starting trust J0 (the same for every agent) and the relative cost of checking; dashed: half of the agents start
checking (J0 equals the threshold of the median account), so to its left most start and to its right most never do; (B) share not checking vs J0 for four prior
strengths (c/g = 0.05); (C, D) spreading the starting beliefs between agents (c/g = 0.05): split of trust (C), shares
not checking and ever checked (D).
(A) is a hexagonal lattice (x: J0 from 0 to 0.6 in steps of 0.05, y: 6 cost rows per decade, 2 runs per cell) when
<data>/priors_hex_twitter.parquet exists; otherwise the earlier 7 x 9 rectangular grid.
Input: <data>/priors_twitter.parquet (and priors_hex_twitter.parquet)."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()


DXJ, DYC = 0.05, 1 / 6                              # hexagonal lattice: J0 step, cost rows per decade


def hex_panel(fig, ax, H, lam, kr):
    """Hexagonal map (x: starting trust J0, y: log10 c/g, as the phase diagrams); colour: dip, mean over runs.
    Dashed: half of the agents start checking (J0 equals the threshold of the median account)."""
    g = fs.circulating(H).groupby(["row", "J0", "cost"]).dip4.mean().reset_index(); g["dip"] = g.dip4 / 4   # runs store 4 x dip
    pc = fs.hexmap(ax, g.J0.values, np.log10(g.cost.values), g.dip.values, DXJ, DYC, fs.CM_SPLIT, 0, 0.2)
    ytop = -3 + np.floor(np.log10(0.5 / 1e-3) * 6) / 6                # top row of the lattice (c/g = 0.46)
    ax.set_xlim(-0.6 * DXJ, 0.6 + 0.6 * DXJ); ax.set_ylim(-3 - 0.75 * DYC, ytop + 0.75 * DYC)
    ax.set_aspect(1.5 / np.sqrt(3) * DXJ / DYC, adjustable="box"); ax.set_anchor("W")   # left edge in line with C below
    cax = ax.inset_axes([1.05, 0.0, 0.06, 1.0]); cb = fig.colorbar(pc, cax=cax, ticks=[0, 0.05, 0.1, 0.15, 0.2])
    cb.ax.tick_params(labelsize=5.5); cax.set_title("dip", fontsize=6, pad=3)
    jj = np.linspace(0.004, 0.6, 80)
    ax.plot(jj, np.log10(jj / ((1 + 1 / lam) * kr)), color=fs.INK, lw=0.8, ls="--")
    ax.annotate("half of the agents\nstart checking", xy=(0.45, np.log10(0.45 / ((1 + 1 / lam) * kr))), xytext=(0.56, -2.05),
                ha="right", va="center", fontsize=5.0, color=fs.INK, arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK),
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
    ax.text(0.585, -2.97, "most start checking", fontsize=5.0, ha="right", va="bottom", color=fs.INK,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.3))
    ax.text(0.015, ytop + 0.05, "most never start", fontsize=5.0, ha="left", va="top", color=fs.INK,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.3))
    ax.set_yticks([-3, -2, -1]); ax.set_yticklabels(["0.001", "0.01", "0.1"]); ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("starting trust J₀ (same for all)"); ax.set_ylabel("relative cost c/g")
    ax.set_title("split of trust (dip)", fontsize=6.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def rect_panel(fig, ax, D, lam, kr):
    """Earlier rectangular version (7 x 9 cells), used when the hexagonal runs are missing."""
    A = D[D.block == "J0xcost"].groupby(["J0", "cost"]).dip4.mean().reset_index(); A["dip4"] = A.dip4 / 4
    J0s = np.sort(A.J0.unique()); cs = np.sort(A.cost.unique())
    Z = A.pivot(index="J0", columns="cost", values="dip4").loc[J0s, cs].values
    lc = np.log10(cs); dl = np.diff(lc); edges = np.r_[lc[0] - dl[0] / 2, lc[:-1] + dl / 2, lc[-1] + dl[-1] / 2]
    dj = J0s[1] - J0s[0]
    pc = ax.pcolormesh(edges, np.r_[J0s - dj / 2, J0s[-1] + dj / 2], Z, cmap=fs.CM_SPLIT, vmin=0, vmax=0.2, rasterized=True,
                       edgecolor="face", linewidth=0.2)
    cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0]); cb = fig.colorbar(pc, cax=cax); cb.ax.tick_params(labelsize=5)
    cax.set_title("dip", fontsize=5.4, pad=3)
    ax.set_title("split of trust (dip)", fontsize=6.5)
    xx = np.linspace(edges[0], edges[-1], 60)
    ax.plot(xx, 10 ** xx * (1 + 1 / lam) * kr, color=fs.INK, lw=0.8, ls="--")
    ax.set_xticks([-3, -2, -1]); ax.set_xticklabels(["0.001", "0.01", "0.1"]); ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(J0s[0] - dj / 2, J0s[-1] + dj / 2)
    ax.set_xlabel("relative cost of checking  c/g"); ax.set_ylabel("starting trust J₀ (same for all)")


def draw(fig, spec, D, c_show=0.05, lam=2.0, letters="ABCD", net="twitter", H=None):
    """Draw the four panels into the SubplotSpec `spec`; returns the axes. H: the hexagonal J0 x cost runs.
    (C and D were one panel until 5 Nov; they split when the figures switched from 4 x dip to the dip itself, which
    no longer shares a 0-1 axis with the shares.)"""
    sub = spec.subgridspec(2, 2, wspace=0.42, hspace=0.5, height_ratios=[1.4, 1])   # A B / C D; A is aspect-locked
    axs = [fig.add_subplot(sub[i // 2, i % 2]) for i in range(4)]
    # A: split of trust over the starting trust and the cost of checking
    ax = axs[0]; fs.panel_label(ax, letters[0], x=-0.29, y=1.04)                # in line with C (A is narrower)
    import config as C
    deg = np.diff(C.graph(net)[0]); kr = deg.mean() / np.median(deg)        # <k> / k_median = 2.7 on the Twitter network
    if H is not None:
        hex_panel(fig, ax, H, lam, kr)
    else:
        rect_panel(fig, ax, D, lam, kr)
    # B: share not checking vs J0 for prior strengths
    ax = axs[1]; fs.panel_label(ax, letters[1], x=-0.2, y=1.04)
    D = fs.circulating(D)                                             # behaviour: runs in which items circulate (5 Nov)
    B = D[(D.design == "identical") & np.isclose(D.cost, c_show)]
    for i, kap in enumerate([2.0, 3.0, 10.0, 30.0]):
        g = fs.agg_ci(B[np.isclose(B.kappa, kap)], "J0", "frac_locked")
        if len(g) == 0:
            continue
        col = fs.blues(4)[i]
        ax.fill_between(g.index, g["mean"] - g["hw"], g["mean"] + g["hw"], color=col, alpha=0.25, lw=0)
        ax.plot(g.index, g["mean"], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.8, mec="white", mew=0.3,
                label=f"κ = {kap:g}" + (" (default)" if kap == 3.0 else ""))
    ax.set_xlabel("starting trust J₀ (same for all)"); ax.set_ylabel("share not checking"); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=5.0, title="prior strength", title_fontsize=5.0, loc="lower left", ncol=1, handlelength=1.8,
              borderaxespad=0.3)
    ax.set_title(f"c/g = {c_show:g}", fontsize=6.5)
    # C, D: spreading the starting beliefs about the checker (± w around J0 = 1/3)
    S = D[(D.block == "spread") & np.isclose(D.cost, c_show)].assign(dip=lambda x: x.dip4 / 4)   # runs store 4 x dip
    ax = axs[2]; fs.panel_label(ax, letters[2], x=-0.2, y=1.04)
    g = fs.agg_ci(S, "width", "dip")
    ax.fill_between(g.index, g["mean"] - g["hw"], g["mean"] + g["hw"], color=fs.VIOLET, alpha=0.2, lw=0)
    ax.plot(g.index, g["mean"], color=fs.VIOLET, marker="o", ms=2.8, mec="white", mew=0.3)
    ax.set_xlabel("spread of starting beliefs\nabout the checker (± w)"); ax.set_ylim(0, 0.25)
    ax.set_ylabel("split of trust (dip)"); ax.set_title(f"c/g = {c_show:g}", fontsize=6.5)
    ax = axs[3]; fs.panel_label(ax, letters[3], x=-0.2, y=1.04)
    for i, (m, col, lab) in enumerate([("frac_locked", fs.BLUE, "not checking"), ("frac_ever", fs.GREY, "ever checked")]):
        g = fs.agg_ci(S, "width", m)
        ax.fill_between(g.index, g["mean"] - g["hw"], g["mean"] + g["hw"], color=col, alpha=0.2, lw=0)
        ax.plot(g.index, g["mean"], color=col, ls=fs.DASHES[i + 1], marker=fs.MARKERS[i + 1], ms=2.8, mec="white", mew=0.3,
                label=lab)
    ax.set_xlabel("spread of starting beliefs\nabout the checker (± w)"); ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("share of agents"); ax.set_title(f"c/g = {c_show:g}", fontsize=6.5)
    ax.legend(fontsize=5.0, loc="upper right", ncol=1, handlelength=2.2, borderaxespad=0.3)
    return axs


def make(priors_parquet, out, c_show=0.05, lam=2.0):
    fig = plt.figure(figsize=(fs.W2, 4.9))
    draw(fig, fig.add_gridspec(1, 1)[0], pd.read_parquet(priors_parquet), c_show, lam)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
