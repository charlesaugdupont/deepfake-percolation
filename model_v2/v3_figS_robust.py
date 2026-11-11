"""SI figure: robustness (main setting phi = 0.3, l/g = 2; 8,000 items).
(A, B) share not checking over time and (C, D) split of trust (Hartigan's dip) over time for random exploration (each item
checked at random with a small probability) and partial feedback (the truth is revealed to only some of the agents
reached: 50%, 20% or 5%). How the stakes grow with the audience is in v3_figS_stakes.py.
Usage: v3_figS_robust.py <robust parquet> <out.png> [cost]"""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

RHO_DATA = 0.36            # degree vs share of low-credibility sources shared, Twitter news sharers (same network)
GROUPS = [("random exploration\n(items checked at random)",
           [("baseline", "none (baseline)"), ("explore 1e-4", "0.01%"), ("explore 1e-3", "0.1%"), ("explore 1e-2", "1%")]),
          ("partial feedback\n(truth revealed with prob.)",
           [("baseline", "1 (baseline)"), ("feedback 0.5", "0.5"), ("feedback 0.2", "0.2"), ("feedback 0.05", "0.05")])]


def series_style(i):
    return (fs.INK, "-", "o") if i == 0 else (fs.CAT[i - 1], fs.DASHES[i], fs.MARKERS[i])


def make(parquet, out, cost=0.05):
    D = pd.read_parquet(parquet)
    snaps = sorted(int(c.split("_t")[1]) for c in D.columns if c.startswith("locked_t"))
    x = np.maximum(snaps, 1)
    fig = plt.figure(figsize=(fs.W2 * 0.68, 3.6))
    top = fig.add_gridspec(2, len(GROUPS), hspace=0.2, wspace=0.12, top=0.86)
    Dc = fs.circulating(D[np.isclose(D.cost, cost)])              # behaviour and belief measures: circulating runs only
    nruns = set()
    for j, (title, variants) in enumerate(GROUPS):
        for r, (m, ylab, ylim) in enumerate([("locked", "share not checking", (0, 1)), ("dip4", "split of trust (dip)", (0, 0.25))]):
            ax = fig.add_subplot(top[r, j])
            for i, (v, lab) in enumerate(variants):
                sub = Dc[Dc.variant == v]
                if sub.empty:
                    continue
                M = sub[[f"{m}_t{t}" for t in snaps]].values / (4 if m == "dip4" else 1)    # runs store 4 x dip
                nruns.add(len(M)); mu = M.mean(0); se = fs.ci95_hw(M.std(0, ddof=1), len(M)) if len(M) > 1 else 0 * mu   # t-based half-width
                col, ls, mk = series_style(i)
                ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.18, lw=0)
                ax.plot(x, mu, color=col, ls=ls, marker=mk, ms=2.4, mec="white", mew=0.3, lw=1.0, label=lab)
            ax.set_xscale("log"); ax.set_xlim(1, snaps[-1] * 1.1); ax.set_ylim(*ylim)
            ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"] if r == 1 else [])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(title, fontsize=6.0, pad=6)
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("items circulated")
            ax.legend(fontsize=5.0, loc="upper left" if r == 1 else "upper right", handlelength=2.4, borderaxespad=0.2,
                      ncol=1)                                         # every panel has its own legend
            fs.panel_label(ax, "ABCD"[len(GROUPS) * r + j], x=-0.3 if j == 0 else -0.1, y=1.02 if r else 1.2)
    nr = f"{min(nruns)}" if len(nruns) == 1 else f"{min(nruns)}–{max(nruns)}"     # was hard-coded "3" before the 6 Nov top-up
    fig.text(0.5, 0.995, f"c/g = {cost:g}, φ = 0.3, ℓ/g = 2; mean of {nr} simulations (bands: 95% CI)", ha="center", va="top",
             fontsize=5.8, color=fs.INK2)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.05)
