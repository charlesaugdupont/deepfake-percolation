"""SI figure: how the penalty for sharing a fake grows with the audience (phi = 0.3, l/g = 2; 2,000 items; 5 runs each).
The gain stake is g·s and the loss stake l·s^gamma, with s = k/<k> the number of neighbours relative to the average:
gamma = 1 (baseline, penalty grows with the audience), 0.5 (grows with its square root) and 0 (constant). The three
agree for an average audience.
What changes: (A) who checks (share of received items checked, by number of neighbours, c/g = 0.05) and (B) who shares
fakes (Spearman correlation between number of neighbours and share of fakes among items shared, against c/g, with the
value in the Twitter data). What holds: (C) share locked out and (D) with two identical checkers, the share with a clear
favourite (|trust in A - trust in B| > 0.2) among agents who can still check.
Usage: v3_figS_penalty.py <data dir> <out.png>"""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from fig_two import degree_axis
fs.setup()

RHO_DATA = 0.36            # number of neighbours vs share of low-credibility links shared, Twitter news sharers
COSTS = [0.01, 0.05, 0.2]
REP_COST = 0.05
PENALTY_FORMS = [1.0, 0.5, 0.0]          # gamma_loss, as in exp_v3.PENALTY_FORMS
FORM = {1.0: ("penalty ∝ audience (baseline)", fs.INK, "-", "o"),
        0.5: ("penalty ∝ √audience", fs.CAT[0], "--", "s"),
        0.0: ("constant penalty", fs.CAT[1], ":", "^")}


def per_bin(runs, edges, min_n=30):
    """Share of received items checked: mean over the agents in each degree bin per run; mean and 95% CI over runs."""
    deg_all = np.concatenate([a["deg"] for a in runs])
    cen, mu, er = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if ((deg_all >= lo) & (deg_all < hi)).sum() < min_n:
            continue
        vals = []
        for a in runs:
            m = (a["n_recv"] > 0) & (a["deg"] >= lo) & (a["deg"] < hi)
            vals.append((a["n_checks"][m] / a["n_recv"][m]).mean() if m.any() else np.nan)
        vals = np.array(vals, float); n = int(np.sum(~np.isnan(vals)))
        if n == 0:
            continue
        cen.append(np.sqrt(lo * max(hi - 1, lo))); mu.append(np.nanmean(vals))
        er.append(1.96 * np.nanstd(vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0)
    return np.array(cen), np.array(mu), np.array(er)


def draw(ax, x, m, e, gl):
    lab, col, ls, mk = FORM[gl]
    ax.fill_between(x, m - e, m + e, color=col, alpha=0.16, lw=0)
    ax.plot(x, m, color=col, ls=ls, lw=1.1, marker=mk, ms=3.0, mec="white", mew=0.3, label=lab)


def cost_axis(ax):
    ax.set_xscale("log"); ax.set_xlim(0.0075, 0.27); ax.set_xticks(COSTS)
    ax.set_xticklabels([f"{c:g}" for c in COSTS]); ax.minorticks_off()
    ax.set_xlabel("relative cost c/g")


def make(data, out):
    S1 = pd.read_parquet(os.path.join(data, "penalty_twitter.parquet"))
    A1 = pickle.load(open(os.path.join(data, "penalty_twitter_agents.pkl"), "rb"))
    A2 = pickle.load(open(os.path.join(data, "penalty_two_twitter_agents.pkl"), "rb"))
    kbar = float(np.mean(A1[0][1]["deg"]))
    deg_all = np.concatenate([p["deg"] for m, p in A1 if m["rep"] == 0 and m["cost"] == REP_COST and m["gamma_loss"] == 1.0])
    edges = np.unique(np.round(np.geomspace(1, deg_all.max() + 1, 13)))
    fig, axs = plt.subplots(1, 4, figsize=(fs.W2, 2.25), gridspec_kw=dict(wspace=0.78, width_ratios=[1.35, 1, 1, 1]))
    ax = dict(zip("ABCD", axs))
    for gl in PENALTY_FORMS:
        # A: who checks, by number of neighbours
        runs = [p for m, p in A1 if m["gamma_loss"] == gl and abs(m["cost"] - REP_COST) < 1e-9]
        cen, mu, er = per_bin(runs, edges)
        draw(ax["A"], cen, mu, er, gl)
        # B, C: against cost (one checker)
        g = S1[np.isclose(S1.gamma_loss, gl)].groupby("cost")
        for k, col_ in [("B", "rho_deg_fakeshare"), ("C", "frac_trapped")]:
            draw(ax[k], np.array(COSTS), g[col_].mean().reindex(COSTS).values,
                 1.96 * g[col_].sem().reindex(COSTS).fillna(0).values, gl)
        # D: two identical checkers, share with a clear favourite
        favs = np.array([[p["fav"][-1] for m, p in A2 if m["gamma_loss"] == gl and abs(m["cost"] - c) < 1e-9]
                         for c in COSTS], float)
        draw(ax["D"], np.array(COSTS), favs.mean(1), 1.96 * favs.std(1, ddof=1) / np.sqrt(favs.shape[1]), gl)
    a = ax["A"]
    a.set_xscale("log"); degree_axis(a, cen.min(), cen.max()); a.set_xlabel("number of neighbours k")
    a.axvline(kbar, color=fs.INK2, lw=0.5, ls=(0, (1, 2)), zorder=0)
    a.text(kbar * 1.15, 1.0, "average\naudience", fontsize=5.0, color=fs.INK2, ha="left", va="top")
    a.set_ylim(-0.02, 1.02); a.set_ylabel("share of received\nitems checked")
    a = ax["B"]; cost_axis(a); a.set_ylim(-1, 1)
    a.axhline(0, color=fs.INK2, lw=0.5)
    a.axhline(RHO_DATA, color=fs.ACTUAL, lw=0.8, ls="--")
    a.text(0.25, -0.93, "dashed: Twitter data", fontsize=5.0, color=fs.INK2, va="bottom", ha="right")
    a.set_yticks([-1, -0.5, 0, 0.5, 1]); a.set_yticklabels(["−1", "−0.5", "0", "0.5", "1"])
    a.set_ylabel("correlation: neighbours vs\nshare of fakes shared")
    a = ax["C"]; cost_axis(a); a.set_ylim(0, 0.4); a.set_ylabel("share locked out")
    a = ax["D"]; cost_axis(a); a.set_ylim(0, 1); a.set_ylabel("share with a\nclear favourite")
    for k, t in {"A": "who checks (c/g = 0.05)", "B": "who shares fakes", "C": "locked out",
                 "D": "two checkers: clear favourite"}.items():
        ax[k].set_title(t, fontsize=6.2, pad=4)
        fs.panel_label(ax[k], k, x=-0.42, y=1.06)
    h, l = ax["A"].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6.0,
               handlelength=2.6, columnspacing=1.6)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
