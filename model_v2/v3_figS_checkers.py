"""SI figure: how good the checker is. Main setting (phi = 0.3, l/g = 2).
(A) final trust in four real-world checkers spread over the quality range (J = 0.70, 0.48, 0.32, 0.11) at c/g = 0.05
    (groups overlaid, semi-transparent); R1 and R3 (J = 0.10 and 0.70) behave like R4 and R2 and are left out;
(B) who does not check vs checker quality J = TPR - FPR (FPR fixed at 0.10), c/g = 0.05: priced out (grey) and
    trapped (orange); the four checkers as markers;
(C) split of trust (Hartigan's dip) vs J for three costs. (The share trapped among agents for whom checking would pay, the
    former panel C, is reported in the text.)
Input: <data>/checkers_twitter.parquet and <data>/checkers_twitter_agents.pkl."""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

BINS = np.linspace(-1, 1, 81)
HIST_ALPHA = fs.HIST_ALPHA
GROUPS = [("never checked", fs.NEVER), ("checked, then stopped", fs.LOCKED), ("still checks", fs.USER)]
NAMES = {"R2": "commercial AI detector", "R3": "LLM fact-checker", "R1": "open-source AI detector", "R4": "people",
         "R5": "fine-tuned open-source detector", "R6": "audio detector in the wild"}
SHORT = {"R2": "commercial\ndetector", "R5": "fine-tuned\ndetector", "R6": "audio detector\nin the wild", "R4": "people"}
SHOWN = ["R2", "R5", "R6", "R4"]                       # ordered by quality J


def make(prefix, out, c_show=0.05, J0=1 / 3):
    S = pd.read_parquet(prefix + ".parquet"); A = pickle.load(open(prefix + "_agents.pkl", "rb"))
    S["J"] = (S.tpr - S.fpr).round(2); S["never_pays"] = 1 - S.frac_worth; S["trap_rel"] = S.frac_trapped / S.frac_worth
    # behaviour and beliefs over the runs in which items circulate (5 Nov): e.g. one of the first three runs with the
    # 'people' checker at c/g = 0.05 collapsed (genuine items reached 0.4% of agents)
    dead = {(r.ref, round(r.cost, 6), int(r.rep)) for r in S[S.reach_true < fs.MIN_REACH].itertuples()}
    A = [(m, p) for m, p in A if (m["ref"], round(m["cost"], 6), int(m["rep"])) not in dead]
    S = fs.circulating(S)
    fig = plt.figure(figsize=(fs.W2, 4.3))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.0], hspace=0.75)
    # A: final trust for the four real-world checkers
    top = outer[0].subgridspec(1, 4, wspace=0.1)
    first = None
    for k, ref in enumerate(SHOWN):
        ax = fig.add_subplot(top[0, k], sharey=first); first = first or ax
        runs = [p for m, p in A if m["ref"] == ref and abs(m["cost"] - c_show) < 1e-9]
        m0 = [m for m, p in A if m["ref"] == ref][0]
        H = np.zeros((3, len(BINS) - 1))
        for p in runs:
            J = p["J"].astype(float); ever = p["ever"]; lock = p["lock"]
            for g, sel in enumerate([~ever, ever & lock, ever & ~lock]):
                H[g] += np.histogram(J[sel], BINS)[0]
        H = 100 * H / (len(runs) * len(runs[0]["J"]))
        for g in (0, 2, 1):                          # overlaid (not stacked), semi-transparent; the small group drawn last
            lab, col = GROUPS[g]
            ax.bar(BINS[:-1], H[g], width=np.diff(BINS), align="edge", color=col, alpha=HIST_ALPHA, edgecolor="none",
                   linewidth=0, label=lab)
        bottom = H.max(0)
        Jt = m0["tpr"] - m0["fpr"]
        ax.axvline(Jt, color=fs.ACTUAL, lw=0.7, ls="--")
        ax.set_xlim(-0.3, 1.0); ax.set_ylim(0, 25); ax.set_yticks([0, 10, 20])
        if bottom.max() > 25:
            i = int(np.argmax(bottom))
            left = 0 < Jt - BINS[i] < 0.25                                # keep the label off the actual-J line
            ax.text(BINS[i] - 0.03 if left else BINS[i] + 0.06, 24, f"{bottom.max():.0f}%", fontsize=5.2, va="top",
                    ha="right" if left else "left", color=fs.INK)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(f"{NAMES[ref]}\nTPR {m0['tpr']:.2f}, FPR {m0['fpr']:.2f} (J = {Jt:.2f})", fontsize=5.8)
        if k == 0:
            ax.set_ylabel("% of agents"); fs.panel_label(ax, "A", x=-0.42, y=1.18)
        else:
            ax.tick_params(labelleft=False)
        if k == 0:
            ax.text(Jt + 0.04, 22, "actual J", fontsize=5.0, va="top", color=fs.INK)
        ax.set_xlabel("trust in the checker Ĵ", fontsize=6.0)
    h, l = fig.axes[0].get_legend_handles_labels()
    order = [l.index(g[0]) for g in GROUPS]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 0.975), ncol=3,
               fontsize=6.0, frameon=False, columnspacing=1.6)
    fig.text(0.5, 0.955, f"c/g = {c_show:g}", ha="center", fontsize=6.0, color=fs.INK2)

    bot = outer[1].subgridspec(1, 2, wspace=0.42)
    Q = S[S.block == "quality"]
    # B: who does not check vs quality, c/g = c_show
    ax = fig.add_subplot(bot[0, 0]); fs.panel_label(ax, "B", x=-0.22, y=1.06)
    g = Q[np.isclose(Q.cost, c_show)].groupby("J")[["never_pays", "frac_locked"]].mean()
    ax.fill_between(g.index, 0, g.never_pays, color=fs.GREY, lw=0, label="priced out (checking never pays)")
    ax.fill_between(g.index, g.never_pays, g.frac_locked, color=fs.ORANGE, lw=0, label="trapped (it would pay, trust too low)")
    gl = fs.agg_ci(Q[np.isclose(Q.cost, c_show)], "J", "frac_locked")      # the priced-out share is the same in every run
    ax.fill_between(gl.index, gl["mean"] - gl["hw"], gl["mean"] + gl["hw"], color=fs.INK, alpha=0.3, lw=0, zorder=3)
    ax.plot(g.index, g.frac_locked, color=fs.INK, lw=0.8, label="not checking (total)")
    R = S[(S.block == "reference") & np.isclose(S.cost, c_show)].groupby("ref")[["J", "frac_locked"]].mean()
    Rh = fs.agg_ci(S[(S.block == "reference") & np.isclose(S.cost, c_show)], "ref", "frac_locked")
    for ref in SHOWN:
        x, y = R.loc[ref, "J"], R.loc[ref, "frac_locked"]
        ax.plot([x, x], [0, y], color=fs.INK, lw=0.6, ls=(0, (2.5, 2)), alpha=0.45, zorder=2.5)   # faint drop line (5 Nov)
        ax.errorbar([x], [y], yerr=[Rh.loc[ref, "hw"]], fmt="none", ecolor=fs.INK, elinewidth=0.7, capsize=0, zorder=4,
                    clip_on=False)
        ax.plot(x, y, ls="none", marker="o", ms=3.0, mfc="white", mec=fs.INK, mew=0.7, zorder=5, clip_on=False)
        ax.annotate(SHORT[ref], xy=(x, y + 0.02), xytext=(x + (0.03 if ref == "R4" else 0.0), y + 0.13), fontsize=5.0,
                    ha="left" if ref == "R4" else "center", va="bottom", color=fs.INK,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK2, shrinkA=0, shrinkB=0))
    ax.set_xlim(0.1, 0.8); ax.set_ylim(0, 1); ax.set_xlabel("checker quality J = TPR − FPR")
    ax.set_ylabel("share not checking"); ax.set_title(f"c/g = {c_show:g}", fontsize=6.2)
    ax.legend(fontsize=5.3, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, frameon=False)
    # C: split of trust
    costs = sorted(Q.cost.unique()); cols = fs.blues(len(costs))
    ax = fig.add_subplot(bot[0, 1]); fs.panel_label(ax, "C", x=-0.22, y=1.06)
    for i, (c, col) in enumerate(zip(costs, cols)):
        g = fs.agg_ci(Q[np.isclose(Q.cost, c)].assign(dip=lambda x: x.dip4 / 4), "J", "dip")
        ax.fill_between(g.index, g["mean"] - g["hw"], g["mean"] + g["hw"], color=col, alpha=0.2, lw=0)
        ax.plot(g.index, g["mean"], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.8, mec="white", mew=0.3,
                label=f"c/g = {c:g}")
    ax.axvline(J0, color=fs.INK2, lw=0.6, ls=":")
    ax.text(J0 + 0.015, 0.2425, "everyone's\nstarting trust", fontsize=5.0, va="top", color=fs.INK2)
    ax.set_xlim(0.1, 0.8); ax.set_ylim(0, 0.25); ax.set_xlabel("checker quality J = TPR − FPR")
    ax.set_ylabel("split of trust (dip)")
    ax.legend(fontsize=5.3, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, handlelength=2.6,
              columnspacing=1.2)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
