"""SI figure: PAWN sensitivity indices (median KS, 95% bootstrap CI) for several outputs; dummy input = noise floor."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from gsa import pawn_bootstrap
fs.setup()

LABELS = {"phi": "share of fakes φ", "cost": "checking cost c/g", "lam": "loss / gain l/g", "J0": "starting trust J₀",
          "kappa": "prior strength κ", "spread": "spread of starting beliefs", "tpr": "checker hit rate (TPR)",
          "fpr": "checker false-alarm rate (FPR)", "dummy": "dummy (noise floor)"}
OUTS = [("dead", "nothing circulates\n(all runs)"), ("dip4", "bifurcation\n(4 × dip)"), ("frac_locked", "share\nlocked out"),
        ("mean_J", "mean trust\nĴ"), ("delta", "truth's edge\nΔ"), ("welfare_sen", "welfare\n(Sen)")]
MIN_REACH = 0.05     # runs where genuine items reach < 5% of agents: nobody shares or checks (see fig_phase.dead_cells)


def make(gsa_parquet, out, n_boot=200):
    D = pd.read_parquet(gsa_parquet)
    D["dead"] = (D.reach_true < MIN_REACH).astype(float)
    live = D.dead.values == 0
    X = D[list(LABELS)].copy(); X["cost"] = np.log10(X["cost"]); X["kappa"] = np.log10(X["kappa"])
    outs = [(m, l) for m, l in OUTS if m in D.columns]
    fig, axs = plt.subplots(1, len(outs), figsize=(fs.W2, 2.4), sharey=True)
    res = {}
    for ax, (m, lab) in zip(axs, outs):
        y = D[m].values.astype(float); ok = ~np.isnan(y)
        if m != "dead":
            ok &= live                                             # outputs other than circulation: runs where items circulate
        else:
            lab = lab.replace("(all runs)", f"({100 * D.dead.mean():.0f}% of all runs)")
        P = pawn_bootstrap(X[ok].reset_index(drop=True), y[ok], n_slices=10, n_boot=n_boot)
        res[m] = P
        order = list(LABELS)[::-1]
        yy = np.arange(len(order))
        vals = P.loc[order, "ks_median"].values; lo = P.loc[order, "lo"].values; hi = P.loc[order, "hi"].values
        colors = [fs.GREY if k == "dummy" else fs.BLUE for k in order]
        ax.barh(yy, vals, color=colors, height=0.6)
        ax.errorbar(vals, yy, xerr=[np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)], fmt="none", ecolor=fs.INK, lw=0.6, capsize=1)
        ax.axvline(P.loc["dummy", "hi"], color=fs.INK2, lw=0.6, ls=":")
        ax.set_title(lab, fontsize=5.8); ax.set_xlim(0, max(0.6, hi.max() * 1.05))
    axs[0].set_yticks(np.arange(len(LABELS))); axs[0].set_yticklabels([LABELS[k] for k in list(LABELS)[::-1]])
    fig.supxlabel("PAWN sensitivity index (median Kolmogorov–Smirnov distance; bars: 95% bootstrap CI). "
                  f"First panel: all {len(D):,} runs; other panels: the {int(live.sum()):,} runs in which items circulate.", fontsize=6.0, y=0.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("saved", out)
    return res


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
