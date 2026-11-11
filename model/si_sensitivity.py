"""SI figure: global sensitivity analysis (PAWN indices: median Kolmogorov-Smirnov distance between conditional and
unconditional output distributions; bars: 95% bootstrap CIs; dotted line: upper CI of a dummy input = noise floor).
First panel: whether anything circulates (all runs); other panels: runs in which items circulate.
Usage: si_sensitivity.py <gsa parquet> <out.png>"""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from gsa import pawn_bootstrap
fs.setup()

LABELS = {"phi": "share of fakes φ", "cost": "relative cost c/g", "lam": "loss / gain ℓ/g", "J0": "starting trust J₀",
          "kappa": "prior strength κ", "spread": "spread of starting trust (± w)", "tpr": "checker hit rate (TPR)",
          "fpr": "checker false-alarm rate (FPR)", "dummy": "dummy input (noise floor)"}
OUTS = [("dead", "nothing circulates"), ("dip4", "split of trust\n(dip)"), ("frac_locked", "share\nnot checking"),
        ("frac_trapped", "share\ntrapped"), ("delta", "reach gap\n(genuine − fake)"), ("welfare_sen", "welfare\n(Sen index)")]
MIN_REACH = 0.05


def make(gsa_parquet, out, n_boot=200):
    D = pd.read_parquet(gsa_parquet)
    D["dead"] = (D.reach_true < MIN_REACH).astype(float)
    live = D.dead.values == 0
    X = D[list(LABELS)].copy(); X["cost"] = np.log10(X["cost"]); X["kappa"] = np.log10(X["kappa"])
    fig, axs = plt.subplots(1, len(OUTS), figsize=(fs.W2, 2.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    res = {}
    order = list(LABELS)[::-1]; yy = np.arange(len(order))
    for k, (ax, (m, lab)) in enumerate(zip(axs, OUTS)):
        y = D[m].values.astype(float); ok = ~np.isnan(y)
        if m != "dead":
            ok &= live
        else:
            lab = f"{lab}\n({100 * D.dead.mean():.0f}% of all simulations)"
        P = pawn_bootstrap(X[ok].reset_index(drop=True), y[ok], n_slices=10, n_boot=n_boot)
        res[m] = P
        vals = P.loc[order, "ks_median"].values; lo = P.loc[order, "lo"].values; hi = P.loc[order, "hi"].values
        cols = [fs.GREY if p == "dummy" else (fs.BLUE if vals[i] > P.loc["dummy", "hi"] else "#a9c8ee") for i, p in enumerate(order)]
        ax.barh(yy, vals, color=cols, height=0.62)
        ax.errorbar(vals, yy, xerr=[np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)], fmt="none", ecolor=fs.INK,
                    lw=0.6, capsize=1)
        ax.axvline(P.loc["dummy", "hi"], color=fs.INK2, lw=0.6, ls=":")
        ax.set_title(lab, fontsize=5.8); ax.set_xlim(0, max(0.62, float(np.nanmax(hi)) * 1.05)); ax.set_xticks([0, 0.2, 0.4])
        ax.set_xticklabels(["0", "0.2", "0.4"])
        ax.tick_params(axis="y", length=0)
        fs.panel_label(ax, "ABCDEF"[k], x=-0.08, y=1.12)                 # same place for every panel (5 Nov)
    axs[0].set_yticks(yy); axs[0].set_yticklabels([LABELS[p] for p in order], fontsize=6.0)
    fig.supxlabel("PAWN index (median KS distance)", fontsize=6.3, y=0.0)
    fig.text(0.5, -0.07, f"A: all {len(D):,} simulations; B–F: the {int(live.sum()):,} in which items circulate. "
             "Pale bars: not distinguishable from the dummy input.", ha="center", fontsize=5.6, color=fs.INK2)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)
    return res


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
