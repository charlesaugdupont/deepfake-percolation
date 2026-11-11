"""SI figure: two checkers in more detail. make_networks (Fig. S6 since 5 Nov; S7 before the renumbering): one row per network, panels A-C below.
make (older layout, Twitter news sharers; optional third row for the Holme-Kim network):
Top row, two identical checkers: (A) joint trust in A and B, (B) who used which checker vs cost, (C) loyalty vs cost.
Middle row: (D) identical checkers by number of neighbours, (E) cheap AI detector vs costly accurate professional by
number of neighbours, (F) share not using each checker vs cost. Third row (with hk=<prefix>): panels A-C on the
Holme-Kim network (G-I).
Input: <data>/two_<net>.parquet and <data>/two_<net>_agents.pkl."""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from figstyle import ci, degree_axis, by_degree
fs.setup()

KMAP = LinearSegmentedColormap.from_list("k", ["#eef3fb", "#9cc2ef", "#2a78d6", "#154f99", "#0b2e5c"])


def line(ax, x, m, e, i, color, label, markevery=None):
    ax.fill_between(x, m - e, m + e, color=color, alpha=0.18, lw=0)
    ax.plot(x, m, color=color, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.8, mec="white", mew=0.3, label=label,
            markevery=markevery)


def cost_axis(ax):
    ax.set_xscale("log"); ax.set_xticks([0.001, 0.01, 0.1]); ax.set_xticklabels(["0.001", "0.01", "0.1"]); ax.minorticks_off()


def legend_below(ax, ncol=2, y=-0.3):
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.8, columnspacing=0.9, frameon=False)


def load(prefix):
    S = pd.read_parquet(prefix + ".parquet"); A = pickle.load(open(prefix + "_agents.pkl", "rb"))
    return S, A


def panel_joint(fig, ax, A, rep_cost, J_true, title):
    sy = [a for e, a in A if e["design"] == "symmetric" and abs(e["cost"] - rep_cost) < 1e-9]
    JA = np.concatenate([a["JA"] for a in sy]); JB = np.concatenate([a["JB"] for a in sy])
    bins = np.linspace(-0.3, 1.0, 66)
    H, xe, ye = np.histogram2d(JA, JB, bins=[bins, bins])
    pc = ax.pcolormesh(xe, ye, (H / H.sum()).T, norm=LogNorm(vmin=1e-4, vmax=0.3), cmap=KMAP, rasterized=True)
    ax.set_aspect("equal"); ax.set_xlabel("trust in checker A  Ĵ_A"); ax.set_ylabel("trust in checker B  Ĵ_B")
    ax.axvline(J_true, color=fs.ACTUAL, lw=0.6, ls="--"); ax.axhline(J_true, color=fs.ACTUAL, lw=0.6, ls="--")
    ax.text(J_true + 0.02, 0.93, "actual", color=fs.INK, fontsize=5.4)
    ax.annotate("relies on A,\nnever tried B", xy=(J_true, 1 / 3), xytext=(-0.27, -0.2), fontsize=5.4,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK2))
    ax.annotate("relies on B,\nnever tried A", xy=(1 / 3, J_true), xytext=(-0.25, 0.86), fontsize=5.4,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK2))
    ax.set_title(title, fontsize=6.2)
    cax = ax.inset_axes([1.04, 0.0, 0.05, 1.0]); cb = fig.colorbar(pc, cax=cax); cb.ax.tick_params(labelsize=5)
    cb.set_label("share of agents", fontsize=5.2, labelpad=1)
    return sy


def panel_use(ax, sym, title):
    for i, (col, c_, lab) in enumerate([("used_onlyA", fs.CHK_A, "only A"), ("used_onlyB", fs.CHK_B, "only B"),
                                        ("used_both", fs.BOTH, "both"), ("used_none", fs.NEVER, "neither")]):
        x, m, e = ci(sym, col)
        line(ax, x, m, e, i, c_, lab, markevery=(i, 2) if i < 2 else None)   # A and B coincide: alternate their markers
    cost_axis(ax); ax.set_ylim(0, 1); ax.set_xlabel("relative cost of checking  c/g"); ax.set_ylabel("share who used…")
    ax.set_title(title, fontsize=6.2); legend_below(ax)


def panel_loyalty(ax, sym, title):
    for i, (col, c_, lab) in enumerate([("dominance", fs.INK, "checks via preferred checker"), ("excl_share", fs.VIOLET, "never tried the other")]):
        x, m, e = ci(sym, col); line(ax, x, m, e, i, c_, lab)
    cost_axis(ax); ax.set_xlabel("relative cost of checking  c/g"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("loyalty (share)"); ax.set_title(title, fontsize=6.2); legend_below(ax, ncol=1)


def make(prefix, out, rep_cost=0.05, J_true=0.70, label=None, hk=None, hk_label="Holme–Kim network (N = 1,000, m = 3)"):
    S, A = load(prefix)
    sym = S[S.design == "symmetric"]
    rows = 3 if hk else 2
    fig = plt.figure(figsize=(fs.W2, 2.5 * rows + (0.3 if hk else 0)))
    gs = fig.add_gridspec(rows, 3, hspace=1.15 if hk else 1.0, wspace=0.85)

    ax = fig.add_subplot(gs[0, 0]); fs.panel_label(ax, "A", x=-0.3, y=1.1)
    sy = panel_joint(fig, ax, A, rep_cost, J_true, f"two identical checkers, c/g = {rep_cost:g}")
    ax = fig.add_subplot(gs[0, 1]); fs.panel_label(ax, "B", x=-0.3, y=1.1); panel_use(ax, sym, "identical checkers")
    ax = fig.add_subplot(gs[0, 2]); fs.panel_label(ax, "C", x=-0.3, y=1.1); panel_loyalty(ax, sym, "identical checkers")

    # D: identical checkers by number of neighbours
    ax = fig.add_subplot(gs[1, 0]); fs.panel_label(ax, "D", x=-0.3, y=1.1)
    fns = [lambda a: (a["nA"] > 0) ^ (a["nB"] > 0), lambda a: (a["nA"] > 0) & (a["nB"] > 0), lambda a: (a["nA"] == 0) & (a["nB"] == 0)]
    cen, mu, er = by_degree(sy, 10, fns)
    for i, (c_, lab) in enumerate([(fs.ONLY_ONE, "only one"), (fs.BOTH, "both"), (fs.NEVER, "neither")]):
        line(ax, cen, mu[:, i], er[:, i], i, c_, lab)
    ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("number of neighbours k"); ax.set_ylabel("share of agents who used…")
    degree_axis(ax, cen.min(), cen.max()); ax.set_title(f"identical checkers, c/g = {rep_cost:g}", fontsize=6.2)
    legend_below(ax, ncol=3)

    # E: asymmetric by degree
    ax = fig.add_subplot(gs[1, 1]); fs.panel_label(ax, "E", x=-0.3, y=1.1)
    asy = [a for e, a in A if e["design"] == "asymmetric" and abs(e["cost"] - rep_cost) < 1e-9 and e["ratio"] == 3.0]
    if asy:
        fns = [lambda a: (a["nA"] > 0) & (a["nB"] == 0), lambda a: a["nB"] > 0, lambda a: (a["nA"] == 0) & (a["nB"] == 0)]
        cen, mu, er = by_degree(asy, 9, fns)
        for i, (c_, lab) in enumerate([(fs.CHK_A, "AI detector only"), (fs.CHK_B, "professional"), (fs.NEVER, "neither")]):
            line(ax, cen, mu[:, i], er[:, i], i, c_, lab)
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("number of neighbours k"); ax.set_ylabel("share of agents who used…")
        degree_axis(ax, cen.min(), cen.max())
        ax.set_title(f"AI detector vs professional (3× cost,\nTPR 0.95, FPR 0.03), c/g = {rep_cost:g}", fontsize=6.0)
        legend_below(ax, ncol=2)

    # F: asymmetric: share not using each checker vs cost
    ax = fig.add_subplot(gs[1, 2]); fs.panel_label(ax, "F", x=-0.3, y=1.1)
    asyS = S[S.design == "asymmetric"]
    k = 0
    for ratio in (3.0, 10.0):
        sub = asyS[asyS.ratio == ratio]
        for col, c_, lab in [("frac_locked_A", fs.CHK_A, "AI detector"), ("frac_locked_B", fs.CHK_B, "professional")]:
            if col.endswith("A") and ratio == 10.0:
                continue                                   # the detector's curve hardly depends on the professional's price
            x, m, e = ci(sub, col)
            line(ax, x, m, e, k, c_, f"professional ({ratio:g}× cost)" if col.endswith("B") else lab); k += 1
    cost_axis(ax); ax.set_ylim(0, 1); ax.set_xlabel("AI detector's cost  c/g"); ax.set_ylabel("share not using\neach checker")
    ax.set_title("AI detector vs professional", fontsize=6.2); legend_below(ax, ncol=1)

    if hk:
        Sh, Ah = load(hk); symh = Sh[Sh.design == "symmetric"]
        ax = fig.add_subplot(gs[2, 0]); fs.panel_label(ax, "G", x=-0.3, y=1.1)
        panel_joint(fig, ax, Ah, rep_cost, J_true, f"two identical checkers, c/g = {rep_cost:g}")
        ax = fig.add_subplot(gs[2, 1]); fs.panel_label(ax, "H", x=-0.3, y=1.1); panel_use(ax, symh, "identical checkers")
        ax = fig.add_subplot(gs[2, 2]); fs.panel_label(ax, "I", x=-0.3, y=1.1); panel_loyalty(ax, symh, "identical checkers")
        y_top = fig.axes[0].get_position().y1; y_hk = ax.get_position().y1
        fig.text(0.0, y_top + 0.045, "Twitter news sharers (main network)", fontsize=7.2, fontweight="bold", ha="left", va="bottom")
        fig.text(0.0, y_hk + 0.045, hk_label, fontsize=7.2, fontweight="bold", ha="left", va="bottom")
    elif label:
        fig.suptitle(label, fontsize=7.2, y=0.99)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


NETS = [("two_twitter", "Twitter news sharers (main network; N = 14,128)"),
        ("two_fb_jhu", "Facebook100, Johns Hopkins (N = 5,157)"),
        ("two_twitch", "Twitch EN (N = 7,126)"),
        ("two_hk", "Holme–Kim (N = 1,000, m = 3)")]


def make_networks(data, out, rep_cost=0.05, J_true=0.70, nets=NETS):
    """SI figure (5 Nov): two identical checkers on four networks, one row per network: joint trust in A and B at
    c/g = rep_cost; who used which checker vs cost; loyalty vs cost. (By number of neighbours and the AI detector vs
    professional are in main Fig. 3D-E.)"""
    nets = [(f, lab) for f, lab in nets if os.path.exists(os.path.join(data, f + ".parquet"))]
    fig = plt.figure(figsize=(fs.W2, 1.95 * len(nets) + 0.5))
    gs = fig.add_gridspec(len(nets), 3, hspace=1.2, wspace=0.85)
    L = "ABCDEFGHIJKLMNOP"
    for r, (f, lab) in enumerate(nets):
        S, A = load(os.path.join(data, f)); sym = S[S.design == "symmetric"]
        n_runs = sym.groupby("cost").size().min()
        ax = fig.add_subplot(gs[r, 0]); fs.panel_label(ax, L[3 * r], x=-0.3, y=1.1)
        panel_joint(fig, ax, A, rep_cost, J_true, f"joint trust, c/g = {rep_cost:g}" if r == 0 else "")
        ax = fig.add_subplot(gs[r, 1]); fs.panel_label(ax, L[3 * r + 1], x=-0.3, y=1.1)
        panel_use(ax, sym, "who used which checker" if r == 0 else "")
        if r < len(nets) - 1:
            ax.get_legend().remove(); ax.set_xlabel("")
        ax = fig.add_subplot(gs[r, 2]); fs.panel_label(ax, L[3 * r + 2], x=-0.3, y=1.1)
        panel_loyalty(ax, sym, "loyalty" if r == 0 else "")
        if r < len(nets) - 1:
            ax.get_legend().remove(); ax.set_xlabel("")
        y1 = fig.axes[-1].get_position().y1
        fig.text(0.0, y1 + 0.036, f"{lab}; {n_runs} simulations per cost", fontsize=6.8, fontweight="bold", ha="left", va="bottom")
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.05)
