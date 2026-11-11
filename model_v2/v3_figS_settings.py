"""SI figures: the single-setting panels repeated at other settings. Rows: main setting (phi = 0.3, c/g = 0.05), two other
costs (c/g = 0.01 and 0.2) and a higher share of fakes (phi = 0.6, c/g = 0.05); l/g = 2 throughout.
make_one: one checker - (a) trust after 2,000 items by group (log scale), (b) groups over time, (c) share not checking
          against the shared starting trust J0, (d) spreading the starting beliefs about the checker (+- w): split of trust
          (Hartigan's dip), with the share not checking at the narrowest and widest spread written in the panel.
make_two: two identical checkers - (a) trust in A minus trust in B after 2,000 items among agents who can still check
          (log scale), (b) what agents used over time, (c) by number of neighbours, (d) cheap AI detector vs professional
          (3x the cost) by number of neighbours.
Usage: v3_figS_settings.py <data dir> <out one.png> <out two.png>"""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from fig_two import degree_axis, by_degree
from v3_fig_two_time import D_BINS, USE, ci95, line, runs_at, time_axis, usage_over_time
fs.setup()

ROWS = [(0.3, 0.05, "main setting\nφ = 0.3, c/g = 0.05"), (0.3, 0.01, "cheaper checking\nφ = 0.3, c/g = 0.01"),
        (0.3, 0.2, "more expensive checking\nφ = 0.3, c/g = 0.2"), (0.6, 0.05, "more fakes\nφ = 0.6, c/g = 0.05")]
BINS = np.linspace(-1, 1, 81)
GROUPS = [("never checked", fs.NEVER), ("checked, then stopped", fs.LOCKED), ("still checks", fs.USER)]


def log_hist(ax, H, groups, bins, xlim, xticks):
    h = 100 * H / max(H.sum(), 1)
    order = sorted(range(len(groups)), key=lambda g: -h[g].sum())          # largest first, small groups on top
    for g in order:                                                         # overlaid (not stacked), semi-transparent
        lab, col = groups[g]
        ax.bar(bins[:-1], h[g], width=np.diff(bins), align="edge", color=col, alpha=fs.HIST_ALPHA, edgecolor="none", linewidth=0,
               label=lab)
    ax.set_yscale("log"); ax.set_ylim(0.01, 150); ax.set_xlim(*xlim); ax.set_xticks(xticks)
    ax.set_yticks([0.01, 1, 100]); ax.set_yticklabels(["0.01", "1", "100"]); ax.minorticks_off()


def ordered(h, l, groups):
    """Legend handles in the groups' own order (histograms are drawn largest first)."""
    idx = [l.index(g[0]) for g in groups if g[0] in l] + [i for i, x in enumerate(l) if x not in [g[0] for g in groups]]
    return [h[i] for i in idx], [l[i] for i in idx]


def row_label(ax, text):
    ax.text(-0.66, 0.5, text, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.0, color=fs.INK)


def make_one(data, out):
    T1 = pickle.load(open(os.path.join(data, "time_twitter_agents.pkl"), "rb"))
    T6 = pickle.load(open(os.path.join(data, "time_twitter_phi06_agents.pkl"), "rb"))
    P = pd.read_parquet(os.path.join(data, "priors_twitter.parquet"))
    PS = pd.read_parquet(os.path.join(data, "priors_settings_twitter.parquet"))
    fig, axs = plt.subplots(len(ROWS), 4, figsize=(fs.W2, 7.4), gridspec_kw=dict(hspace=0.55, wspace=0.5, top=0.93))
    for r, (phi, c, lab) in enumerate(ROWS):
        src = T1 if phi == 0.3 else T6
        runs = [p for m, p in src if abs(m["cost"] - c) < 1e-9 and abs(m.get("phi", 0.3) - phi) < 1e-9]
        # (a) distribution after 2,000 items
        ax = axs[r, 0]
        H = np.mean([p["hist"][-1] for p in runs], 0)
        log_hist(ax, H, GROUPS, BINS, (-0.3, 1.0), [0, 0.5, 1])
        ax.axvline(0.70, color=fs.ACTUAL, lw=0.7, ls="--")
        ax.set_ylabel("% of agents (log)"); row_label(ax, lab)
        if r == len(ROWS) - 1:
            ax.set_xlabel("trust in the checker Ĵ")
        # (b) groups over time
        ax = axs[r, 1]; tt = np.maximum(runs[0]["times"], 1)
        never = [1 - p["ever"] for p in runs]; burned = [p["burned"] for p in runs]; active = [p["ever"] - p["burned"] for p in runs]
        for i, (arr, (gl, col)) in enumerate(zip([never, burned, active], GROUPS)):
            m, lo, hi = ci95(arr)
            ax.fill_between(tt, lo, hi, color=col, alpha=0.25, lw=0); ax.plot(tt, m, color=col, ls=fs.DASHES[i], lw=1.1)
            ax.plot(tt[::14], m[::14], ls="none", marker=fs.MARKERS[i], color=col, ms=2.4, mec="white", mew=0.3)
        time_axis(ax, tt[-1]); ax.set_ylim(0, 1.02); ax.set_ylabel("share of agents")
        if r == len(ROWS) - 1:
            ax.set_xlabel("items circulated")
        # (c) not checking against J0
        ax = axs[r, 2]
        if phi == 0.3:
            B = P[(P.block == "J0xcost") & np.isclose(P.cost, c)]
        else:
            B = PS[(PS.block == "J0") & np.isclose(PS.phi, phi) & np.isclose(PS.cost, c)]
        g = fs.agg_ci(fs.circulating(B), "J0", "frac_locked")             # behaviour: circulating runs
        line(ax, g.index.values, g["mean"].values, g["hw"].values, 0, fs.BLUE, "share not checking")
        gone = sorted(set(np.round(B.J0, 4)) - set(np.round(g.index.values, 4)))   # no run circulates there
        if gone:
            ax.text(0.01, 0.98, "J₀ = " + ", ".join(f"{x:g}" for x in gone) + ":\nnothing circulates", fontsize=4.8,
                    ha="left", va="top", color=fs.INK2)
        never_pays = (1 - B.frac_worth).mean()
        ax.axhline(never_pays, color=fs.INK2, lw=0.6, ls=":")
        ax.text(0.0, never_pays - 0.03, "priced out", fontsize=4.8, ha="left", va="top", color=fs.INK2)
        ax.set_ylim(-0.02, 1.02); ax.set_xlim(-0.02, 0.62); ax.set_ylabel("share not checking")
        if r == len(ROWS) - 1:
            ax.set_xlabel("starting trust J₀ (same for all)")
        # (d) spread of starting beliefs
        ax = axs[r, 3]
        if (phi, c) == (0.3, 0.05):
            Sd = P[(P.block == "spread") & np.isclose(P.cost, c)]
        else:
            Sd = PS[(PS.block == "spread") & np.isclose(PS.phi, phi) & np.isclose(PS.cost, c)]
        Sd = fs.circulating(Sd)                                                                # behaviour: circulating runs
        g = fs.agg_ci(Sd.assign(dip=Sd.dip4 / 4), "width", "dip")                           # runs store 4 x dip
        line(ax, g.index.values, g["mean"].values, g["hw"].values, 0, fs.VIOLET, "split of trust (dip)")
        ax.set_ylim(0, 0.25); ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_ylabel("split of trust (dip)")
        nl = Sd.groupby("width").frac_locked.mean()                      # the behaviour barely moves: say so in the panel
        ax.text(0.97, 0.95, f"not checking:\n{nl.iloc[0]:.0%} at w = 0,\n{nl.iloc[-1]:.0%} at w = {nl.index[-1]:g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=4.8, color=fs.INK2, linespacing=1.15)
        if r == len(ROWS) - 1:
            ax.set_xlabel("spread of starting beliefs\nabout the checker (± w)")
        if r == 0:
            for k, t in enumerate(["trust after 2,000 items", "groups over time", "starting trust", "spread of starting beliefs"]):
                axs[0, k].set_title(t, fontsize=6.4)
    h, l = ordered(*axs[0, 0].get_legend_handles_labels(), GROUPS)
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.968), ncol=3, fontsize=5.6, frameon=False,
               columnspacing=1.2, handlelength=1.8)
    for k, L in enumerate("ABCD"):
        fs.panel_label(axs[0, k], L, x=-0.3, y=1.12)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


def make_two(data, out):
    TT = pickle.load(open(os.path.join(data, "two_time_twitter_agents.pkl"), "rb"))
    A2 = pickle.load(open(os.path.join(data, "two_twitter_agents.pkl"), "rb"))
    AS = pickle.load(open(os.path.join(data, "two_asym_settings_twitter_agents.pkl"), "rb"))
    fig, axs = plt.subplots(len(ROWS), 4, figsize=(fs.W2, 7.4), gridspec_kw=dict(hspace=0.55, wspace=0.5, top=0.93))
    for r, (phi, c, lab) in enumerate(ROWS):
        runs = runs_at(TT, phi, c)
        ax = axs[r, 0]
        H = np.sum([p["histD"][-1] for p in runs], 0).astype(float)
        log_hist(ax, H, USE, D_BINS, (-0.75, 0.75), [-0.5, 0, 0.5]); ax.set_xticklabels(["−0.5", "0", "0.5"])
        ax.axvline(0, color=fs.INK2, lw=0.5, ls=":")
        ax.set_ylabel("% (log scale)"); row_label(ax, lab)
        if r == len(ROWS) - 1:
            ax.set_xlabel("trust in A − trust in B")
        ax = axs[r, 1]; usage_over_time(ax, runs, end_labels=False)
        if r != len(ROWS) - 1:
            ax.set_xlabel("")
        ax = axs[r, 2]
        fns = [lambda a: (a["nA"] > 0) ^ (a["nB"] > 0), lambda a: (a["nA"] > 0) & (a["nB"] > 0), lambda a: (a["nA"] == 0) & (a["nB"] == 0)]
        cen, mu, er = by_degree(runs, 12, fns)
        for i, (col, gl) in enumerate([(fs.ONLY_ONE, "only one checker"), (fs.BOTH, "both"), (fs.NEVER, "neither")]):
            line(ax, cen, mu[:, i], er[:, i], i, col, gl)
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); degree_axis(ax, cen.min(), cen.max()); ax.set_ylabel("share who used…")
        if r == len(ROWS) - 1:
            ax.set_xlabel("number of neighbours k")
        ax = axs[r, 3]
        if (phi, c) == (0.3, 0.05):
            asy = [a for e, a in A2 if e["design"] == "asymmetric" and abs(e["cost"] - c) < 1e-9 and e["ratio"] == 3.0]
        else:
            asy = [a for e, a in AS if abs(e["phi"] - phi) < 1e-9 and abs(e["cost"] - c) < 1e-9]
        fns = [lambda a: (a["nA"] > 0) & (a["nB"] == 0), lambda a: a["nB"] > 0, lambda a: (a["nA"] == 0) & (a["nB"] == 0)]
        cen, mu, er = by_degree(asy, 9, fns)
        for i, (col, gl) in enumerate([(fs.CHK_A, "AI detector only"), (fs.CHK_B, "professional"), (fs.NEVER, "neither ")]):
            line(ax, cen, mu[:, i], er[:, i], i, col, gl)
        ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); degree_axis(ax, cen.min(), cen.max()); ax.set_ylabel("share who used…")
        if r == len(ROWS) - 1:
            ax.set_xlabel("number of neighbours k")
        if r == 0:
            for k, t in enumerate(["trust in A − B, after 2,000 items", "what agents used, over time", "identical checkers",
                                   "AI detector vs professional"]):
                axs[0, k].set_title(t, fontsize=6.2)
    for k in range(4):                                   # one legend under each column
        h, l = axs[-1, k].get_legend_handles_labels()
        if k == 0:
            h, l = ordered(h, l, USE)
        axs[-1, k].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1 if k else 2, fontsize=5.4,
                          frameon=False, handlelength=2.0, columnspacing=0.8)
    for k, L in enumerate("ABCD"):
        fs.panel_label(axs[0, k], L, x=-0.3, y=1.12)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make_one(sys.argv[1], sys.argv[2]); make_two(sys.argv[1], sys.argv[3])
