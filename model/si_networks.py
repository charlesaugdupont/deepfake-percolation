"""SI figure: other networks (l/g = 2). (A) split of trust (Hartigan's dip) and (B) share trapped (checking would pay
off for them, but their trust is at or below their threshold), over the share of fakes and the relative cost of checking,
for four networks (hatched: nothing circulates; the degree-preserving rewired Twitter network gives the same results as
the original and is reported in the text); (C) share not checking and (D) share trapped against cost (phi = 0.2-0.5): simulation (markers)
and the one-agent model for each network's degree distribution (bands).
Usage: si_networks.py <data dir> <out.png>"""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import config as C, engine as E
from fig_regimes import cellmeans, theory, sim_curve, read_phase, DX, DY, ASPECT, J0
fs.setup()

NETS = [("twitter", "Twitter news sharers\n(main; N = 14,128)", fs.INK),
        ("fb_jhu", "Facebook100, Johns\nHopkins (N = 5,157)", fs.CAT[1]),
        ("twitch", "Twitch EN\n(N = 7,126)", fs.CAT[2]),
        ("hk", "Holme–Kim\n(N = 1,000, m = 3)", fs.CAT[3])]


def make(data, out, lam=2.0):
    fig = plt.figure(figsize=(fs.W2, 5.3))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.25], hspace=0.38)
    nn = len(NETS)
    top = outer[0].subgridspec(2, nn + 1, width_ratios=[1] * nn + [0.06], hspace=0.45, wspace=0.1)
    bot = outer[1].subgridspec(1, 2, wspace=0.3)
    axl = [fig.add_subplot(bot[0, k]) for k in range(2)]
    for j, (net, lab, col) in enumerate(NETS):
        path = os.path.join(data, f"phase_{net}.parquet")
        if not os.path.exists(path):
            continue
        D = read_phase(path); D = D[np.isclose(D.lam, lam)]
        g = cellmeans(D, lam)
        for i, (m, cmap, vmax, rowlab) in enumerate([("dip", fs.CM_SPLIT, 0.2, "split of trust (dip)"),
                                                     ("frac_trapped", fs.CM_TRAP, 0.5, "share trapped")]):
            ax = fig.add_subplot(top[i, j])
            pc = fs.hexmap(ax, g.phi.values, np.log10(g.cost.values), g[m].values, DX, DY, cmap, 0, vmax,
                           hatch_mask=g.dead.values)
            ax.set_xlim(0.05 - 0.6 * DX, 0.95 + 0.6 * DX); ax.set_ylim(-3 - 0.75 * DY, 0.75 * DY)
            ax.set_aspect(ASPECT, adjustable="box")
            ax.set_xticks([0.1, 0.5, 0.9]); ax.set_yticks([-3, -2, -1, 0])
            ax.set_xticklabels(["0.1", "0.5", "0.9"])
            ax.set_xlabel("share of fakes φ", fontsize=6.0)                 # both rows (4 Nov feedback)
            if j == 0:
                ax.set_yticklabels(["0.001", "0.01", "0.1", "1"]); ax.set_ylabel("relative cost c/g", fontsize=6.0)
                fs.panel_label(ax, "AB"[i], x=-0.75, y=0.98)
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(lab, fontsize=5.6, pad=13)
            if j == len(NETS) - 1:
                cax = fig.add_subplot(top[i, nn]); cb = fig.colorbar(pc, cax=cax, extend="max" if i == 1 else "neither")
                cb.ax.tick_params(labelsize=5); cb.set_label(rowlab, fontsize=6.0)   # the metric names the colour bar
        # theory vs simulation
        gr = C.graph(net) if net != "hk" else E.holme_kim(1000, 3, 0.5, 1)
        deg = np.diff(gr[0])
        th = theory(D, deg, [lam], os.path.join(data, f"theory_{net}.json"))[lam]
        for k, m in enumerate(["frac_locked", "frac_trapped"]):
            s = sim_curve(D, lam, m)
            axl[k].plot(th[:, 0], th[:, 1 + k], color=col, lw=2.4, alpha=0.3, solid_capstyle="round", zorder=1)
            main = net == "twitter"
            axl[k].errorbar(s.index, s["mean"], yerr=s["hw"], fmt="none", ecolor=col, elinewidth=0.7, capsize=0,
                            zorder=4 if main else 2)                        # 95% CIs (see sim_curve), mostly hidden by markers
            axl[k].plot(s.index, s["mean"], ls="none", marker=fs.MARKERS[j], ms=3.4 if main else 2.6, color=col,
                        mfc="none" if main else col, mec=col if main else "white", mew=0.7 if main else 0.3,
                        zorder=5 if main else 3, label=lab.replace("\n", " ").replace("- ", "-"))
    for k, (lab, ymax) in enumerate([("share not checking", 1.0), ("share trapped", 0.8)]):
        ax = axl[k]; fs.panel_label(ax, "CD"[k], x=-0.2, y=1.02)
        ax.set_xscale("log"); ax.set_xlim(8e-4, 1.25); ax.set_xticks([0.001, 0.01, 0.1, 1]); ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
        ax.set_ylim(0, ymax); ax.set_xlabel("relative cost c/g"); ax.set_ylabel(lab); ax.set_title("φ = 0.2–0.5", fontsize=6.0)
    h, l = axl[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=fs.INK2, lw=2.4, alpha=0.3)); l.append("theory (one-agent model)")
    h.append(Patch(facecolor="#e4e2dc", edgecolor="#9a978f", hatch="//////")); l.append("nothing circulates (A, B)")
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=4, fontsize=5.4, frameon=False, columnspacing=1.2)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
