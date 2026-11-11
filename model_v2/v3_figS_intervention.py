"""SI figure: temporary cuts in the cost of checking (main setting phi = 0.3, l/g = 2, c/g = 0.05; 5 runs per condition,
same seeds as the control, so each run equals its control until the cut starts after 1,000 items; then the cost goes
back to normal). Outcomes after the cut are measured over the last 1,000 items (3,000-4,000).
(A) share not checking over time when checking is free for 50, 200 or 1,000 items;
(B) change in how much further genuine items travel than fakes (reach gap), vs the control run with the same seed;
(C) change in welfare (Sen index, checking costs included), same comparison;
(D) change in mean normalised payoff by group (groups as in the control run): trapped, priced out, still checking.
Input: <data>/intervention_twitter.parquet and <data>/intervention_agents_twitter_agents.pkl."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()


def norm_payoff(p):
    """Payoff normalised by each agent's best / worst possible outcome (as in the welfare measure, sweep.summarize)."""
    pay, best, worst = p["payoff"].astype(float), p["best"].astype(float), p["worst"].astype(float)
    n = np.zeros_like(pay)
    m1 = (pay < 0) & (worst != 0); n[m1] = -(pay[m1] / worst[m1])
    m2 = (pay > 0) & (best != 0); n[m2] = pay[m2] / best[m2]
    return n


def by_group(agents_pkl, J_true=0.70):
    """Per run: mean normalised payoff in the last 1,000 items for groups defined in the control run with the same
    seed (trapped: theta < J and trust <= theta; priced out: theta >= J; users: the rest), control vs cut."""
    import pickle
    runs = {(m["condition"], m["rep"]): p for m, p in pickle.load(open(agents_pkl, "rb"))}
    rows = []
    for (cond, rep), p in runs.items():
        if cond == "control":
            continue
        c = runs[("control", rep)]
        trapped = (c["theta"] < J_true) & (c["J"] <= c["theta"]); priced = c["theta"] >= J_true
        nc, nx = norm_payoff(c), norm_payoff(p)
        for g, m in [("trapped", trapped), ("priced out", priced), ("users", ~trapped & ~priced)]:
            rows.append(dict(cond=cond, rep=rep, group=g, ctrl=nc[m].mean(), cut=nx[m].mean(), d=nx[m].mean() - nc[m].mean()))
    return pd.DataFrame(rows)


def make(parquet, out, start=1000, agents_pkl=None):
    D = pd.read_parquet(parquet)
    D["never_pays"] = 1 - D.frac_worth
    D = D[(D.condition == "control") | (D.target == "all")]                      # the cut for everyone only
    snaps = sorted(int(c.split("_t")[1]) for c in D.columns if c.startswith("locked_t"))
    fig, axs = plt.subplots(2, 2, figsize=(fs.W2 * 0.78, 4.1), gridspec_kw=dict(wspace=0.45, hspace=0.6))
    axs = axs.ravel()
    # A: time course
    ax = axs[0]; fs.panel_label(ax, "A", x=-0.25, y=1.06)
    conds = [("control", "no cut", fs.INK), ("x0 for 50 (all)", "free for 50 items", fs.blues(3)[0]),
             ("x0 for 200 (all)", "free for 200 items", fs.blues(3)[1]), ("x0 for 1000 (all)", "free for 1,000 items", fs.blues(3)[2])]
    for i, (cond, lab, col) in enumerate(conds):
        M = D[D.condition == cond][[f"locked_t{t}" for t in snaps]].values
        m = M.mean(0); e = 1.96 * M.std(0, ddof=1) / np.sqrt(len(M))
        ax.fill_between(snaps, m - e, m + e, color=col, alpha=0.2, lw=0)
        ax.plot(snaps, m, color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.6, mec="white", mew=0.3, label=lab)
    ax.axvspan(start, start + 1000, color="#ecebe7", lw=0, zorder=0)
    ax.text(start + 500, 0.585, "checking free\n(50, 200 or\n1,000 items)", ha="center", va="top", fontsize=5.0, color=fs.INK2)
    nev = D[D.condition == "control"].never_pays.mean()
    ax.axhline(nev, color=fs.INK2, lw=0.6, ls=":"); ax.text(3950, nev - 0.012, "priced out", ha="right", va="top", fontsize=5.2, color=fs.INK2)
    ax.set_xlim(0, 4000); ax.set_ylim(0, 0.6); ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000", "4,000"])
    ax.set_xlabel("items circulated"); ax.set_ylabel("share not checking")
    ax.legend(fontsize=5.2, loc="lower right", ncol=1, handlelength=1.8)
    # B, C: change after the cut, paired with the control run of the same seed
    ctrl = D[D.condition == "control"].set_index("rep")
    depth = [(0.0, "free"), (0.1, "10% of the cost"), (0.5, "half price")]
    colors = {0.0: fs.blues(3)[2], 0.1: fs.blues(3)[1], 0.5: fs.blues(3)[0]}
    for k, (m_, lab_y, title) in enumerate([("delta", "change in reach gap\n(genuine − fake)", "what spreads"),
                                            ("welfare_sen", "change in welfare\n(Sen index)", "welfare, everyone")]):
        ax = axs[1 + k]; fs.panel_label(ax, "BC"[k], x=-0.3, y=1.06)
        for i, (mult, lab) in enumerate(depth):
            sub = D[(D["mult"] == mult) & (D.condition != "control")]
            rows = []
            for dur, g in sub.groupby("duration"):
                g = g.set_index("rep"); d = g[m_] - ctrl.loc[g.index, m_]
                rows.append((dur, d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))))
            x, mu, e = map(np.array, zip(*rows))
            xo = x * (1.0 + 0.09 * (i - 1))                                     # small offsets so error bars do not overlap
            ax.errorbar(xo, mu, yerr=e, color=colors[mult], ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.8, mec="white",
                        mew=0.3, capsize=1, lw=1.0, label=lab)
        ax.axhline(0, color=fs.INK2, lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_xticks([50, 200, 1000]); ax.set_xticklabels(["50", "200", "1,000"]); ax.minorticks_off()
        ax.set_xlim(35, 1450); ax.set_ylim(-0.035, 0.11)
        ax.set_xlabel("length of the cut (items)"); ax.set_ylabel(lab_y)
        ax.set_title(f"{title} (control: {ctrl[m_].mean():.2f})", fontsize=6.0)
        ax.legend(fontsize=5.0, loc="upper left", handlelength=2.2, title="cost during the cut", title_fontsize=5.0, ncol=3,
                  columnspacing=0.8, borderaxespad=0.3)                # each panel has its own legend
    # D: who gains: change in mean normalised payoff by group (groups from the control run)
    ax = axs[3]; fs.panel_label(ax, "D", x=-0.3, y=1.06)
    if agents_pkl and os.path.exists(agents_pkl):
        G = by_group(agents_pkl)
        groups = [("trapped", "trapped"), ("priced out", "priced\nout"), ("users", "still\nchecking")]
        cuts = [("x0 for 50 (all)", "free, 50 items"), ("x0 for 200 (all)", "free, 200 items"), ("x0 for 1000 (all)", "free, 1,000 items")]
        for i, (cond, lab) in enumerate(cuts):
            sub = G[G.cond == cond]
            mu = [sub[sub.group == g].d.mean() for g, _ in groups]
            e = [1.96 * sub[sub.group == g].d.std(ddof=1) / np.sqrt((sub.group == g).sum()) for g, _ in groups]
            ax.errorbar(np.arange(len(groups)) + (i - 1) * 0.18, mu, yerr=e, ls="none", marker=fs.MARKERS[i], ms=3.0,
                        color=fs.blues(3)[i], mec="white", mew=0.3, capsize=1, lw=1.0, label=lab)
        base = G[G.cond == cuts[-1][0]].groupby("group").ctrl.mean()
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{t}\n({base[g]:.2f})" for g, t in groups], fontsize=5.6)
        ax.axhline(0, color=fs.INK2, lw=0.6, ls=":")
        ax.set_xlim(-0.5, len(groups) - 0.5); ax.set_ylim(-0.035, 0.11)
        ax.set_ylabel("change in mean\nnormalised payoff"); ax.set_title("who gains (control payoff in brackets)", fontsize=6.0)
        ax.legend(fontsize=5.2, loc="upper right", handlelength=1.2)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], agents_pkl=sys.argv[3] if len(sys.argv) > 3 else None)
