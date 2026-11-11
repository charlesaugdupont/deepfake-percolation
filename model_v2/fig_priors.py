"""Figure: does the split depend on where trust starts? (A) bifurcation over starting trust x cost,
(B) share locked out vs starting trust for several prior strengths, (C) individual variation in starting beliefs,
(D) reference checkers: a near-useless checker gives uniform (correct) distrust, not a split."""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()
CM_DIP = LinearSegmentedColormap.from_list("dip", ["#fbfaf7", "#f6d7c3", "#ec9a6c", "#d4572a", "#8f2a12", "#4a1406"])


def make(priors_parquet, checkers_prefix, out, c_show=None):
    D = pd.read_parquet(priors_parquet)
    if c_show is None:
        c_show = float(D[D.block == "J0xkappa"].cost.iloc[0])
    fig = plt.figure(figsize=(fs.W2, 4.2))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.62)
    # A
    ax = fig.add_subplot(gs[0, 0]); fs.panel_label(ax, "A")
    A = D[D.block == "J0xcost"].groupby(["J0", "cost"]).dip4.mean().reset_index()
    J0s = np.sort(A.J0.unique()); cs = np.sort(A.cost.unique())
    Z = A.pivot(index="J0", columns="cost", values="dip4").loc[J0s, cs].values
    lc = np.log10(cs); dl = lc[1] - lc[0]; dj = J0s[1] - J0s[0]
    pc = ax.pcolormesh(np.r_[lc - dl / 2, lc[-1] + dl / 2], np.r_[J0s - dj / 2, J0s[-1] + dj / 2], Z, cmap=CM_DIP, vmin=0, vmax=0.7, rasterized=True)
    cax = ax.inset_axes([1.03, 0.0, 0.05, 1.0]); cb = fig.colorbar(pc, cax=cax); cb.ax.tick_params(labelsize=5); ax.set_title("bifurcation (4 × dip), shared prior", fontsize=6.3)
    lam = 2.0; xx = np.linspace(lc[0] - dl / 2, lc[-1] + dl / 2, 50)
    ax.plot(xx, 10 ** xx * (1 + 1 / lam), color=fs.INK, lw=0.8, ls="--")       # average agent's threshold
    ax.annotate("J₀ = θ of an\naverage agent", xy=(np.log10(0.25), 0.375), xytext=(lc[-1] + dl / 2 - 0.05, 0.0), ha="right", va="center",
                fontsize=5.0, color=fs.INK, arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK))
    ax.set_xticks([-2, -1]); ax.set_xticklabels(["0.01", "0.1"]); ax.set_ylim(J0s[0] - dj / 2, J0s[-1] + dj / 2)
    ax.set_xlabel("checking cost / gain  c/g"); ax.set_ylabel("shared starting trust  J₀")
    # B
    ax = fig.add_subplot(gs[0, 1]); fs.panel_label(ax, "B", x=-0.25)
    B = D[(D.design == "identical") & np.isclose(D.cost, c_show)]
    for kap, col in zip([2.0, 3.0, 10.0, 30.0], ["#c9d8ee", "#7fb0ea", fs.BLUE, "#0b2e5c"]):
        g = B[np.isclose(B.kappa, kap)].groupby("J0").frac_locked.agg(["mean", "sem"])
        if len(g) == 0: continue
        ax.fill_between(g.index, g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"], color=col, alpha=0.3, lw=0)
        ax.plot(g.index, g["mean"], color=col, marker="o", ms=1.8, label=f"κ = {kap:g}")
    ax.set_xlabel("shared starting trust  J₀"); ax.set_ylabel("share locked out"); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=5.6, title="prior strength", title_fontsize=5.6); ax.set_title(f"c/g = {c_show:.2g}", fontsize=6.5)
    # C
    ax = fig.add_subplot(gs[0, 2]); fs.panel_label(ax, "C")
    S = D[(D.block == "spread") & np.isclose(D.cost, c_show)]
    for m, col, lab in [("dip4", fs.RED, "bifurcation (4 × dip)"), ("frac_locked", fs.BLUE, "share locked out"), ("frac_ever", fs.GREY, "share ever checked")]:
        g = S.groupby("width")[m].agg(["mean", "sem"])
        ax.fill_between(g.index, g["mean"] - 1.96 * g["sem"], g["mean"] + 1.96 * g["sem"], color=col, alpha=0.2, lw=0)
        ax.plot(g.index, g["mean"], color=col, marker="o", ms=1.8, label=lab)
    ax.set_xlabel("spread of starting beliefs (± w)"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=5.4, loc="upper right")
    ax.set_title(f"c/g = {c_show:.2g}", fontsize=6.5)
    # D: reference checkers
    if checkers_prefix:
        CS = pd.read_parquet(checkers_prefix + ".parquet"); Jd = pickle.load(open(checkers_prefix + "_J.pkl", "rb"))
        cs2 = np.sort(CS.cost.unique()); cpick = cs2[np.argmin(np.abs(cs2 - c_show))]
        names = {"R1": "R1 open-source AI detector\n(0.55, 0.45)", "R2": "R2 commercial AI detector\n(0.80, 0.10)",
                 "R3": "R3 LLM fact-checker\n(0.85, 0.15)", "R4": "R4 people\n(0.43, 0.32)"}
        sub = gs[1, :].subgridspec(1, 4, wspace=0.12)
        bins = np.linspace(-0.4, 1.0, 57)
        for k, ref in enumerate(["R2", "R3", "R1", "R4"]):
            ax = fig.add_subplot(sub[0, k])
            Js = [(J, ever, lock) for e, (J, ever, lock) in Jd if e["ref"] == ref and abs(e["cost"] - cpick) < 1e-9]
            J = np.concatenate([x[0] for x in Js]); ever = np.concatenate([x[1] for x in Js]); lock = np.concatenate([x[2] for x in Js])
            w = np.ones_like(J) * 100 / len(J)
            ax.hist([J[~ever], J[ever & lock], J[ever & ~lock]], bins=bins, weights=[w[~ever], w[ever & lock], w[ever & ~lock]],
                    stacked=True, color=[fs.GREY, fs.ORANGE, fs.BLUE], label=["never checked", "locked out", "still checks"])
            t, f = CS[CS.ref == ref].tpr.iloc[0], CS[CS.ref == ref].fpr.iloc[0]
            ax.axvline(t - f, color=fs.GREEN, lw=0.8, ls="--")
            r = CS[(CS.ref == ref) & np.isclose(CS.cost, cpick)]
            ax.set_title(names[ref], fontsize=5.8)
            left = (t - f) > 0.4                      # put the numbers on the side away from the checker's actual J
            ax.text(0.03 if left else 0.97, 0.62, f"4·dip {r.dip4.mean():.2f}\nΔ {r.delta.mean():+.2f}", transform=ax.transAxes,
                    ha="left" if left else "right", va="top", fontsize=5.4)
            ax.set_xlabel("trust in the checker Ĵ"); ax.set_yticks([] if k else ax.get_yticks())
            if k == 0:
                ax.set_ylabel("% of agents"); fs.panel_label(ax, "D", x=-0.3)
                ax.legend(fontsize=5.2, loc="upper left")
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2] if len(sys.argv) > 3 else None, sys.argv[-1])
