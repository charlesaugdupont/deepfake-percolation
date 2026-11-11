"""SI figure: robustness of the split. Share locked out and bifurcation over (long) time for model variants."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

GROUPS = [("exploration (random checks per item)", ["baseline", "explore 1e-4", "explore 1e-3", "explore 1e-2"],
           ["#0b2e5c", "#2a78d6", "#7fb0ea", "#c9d8ee"], ["none", "0.01%", "0.1%", "1%"]),
          ("feedback (truth revealed with prob.)", ["baseline", "feedback 0.5", "feedback 0.2"], ["#0b2e5c", "#d9822b", "#f2c08f"], ["1", "0.5", "0.2"]),
          ("model variants", ["baseline", "posterior v1 (double count)", "prior strength x3 (kappa=9)", "stakes sqrt(degree)", "stakes indep. of degree"],
           ["#0b2e5c", "#7b5cc4", "#1a9e6e", "#d9822b", "#9a9893"], ["v2 baseline", "v1 posterior", "prior strength ×3", "stakes ∝ √k", "stakes ∝ 1"])]


def make(parquet, out, cost):
    D = pd.read_parquet(parquet); D = D[np.isclose(D.cost, cost)]
    snaps = sorted(int(c.split("_t")[1]) for c in D.columns if c.startswith("locked_t"))
    fig, axs = plt.subplots(2, 3, figsize=(fs.W2, 3.6), sharex=True)
    for j, (title, variants, cols, labs) in enumerate(GROUPS):
        for row, m, ylab in [(0, "locked", "share locked out"), (1, "dip4", "bifurcation (4 × dip)")]:
            ax = axs[row, j]
            for v, c, lab in zip(variants, cols, labs):
                sub = D[D.variant == v]
                if sub.empty: continue
                M = sub[[f"{m}_t{t}" for t in snaps]].values
                mu = M.mean(0); se = M.std(0, ddof=1) / np.sqrt(len(M))
                x = np.maximum(snaps, 1)
                ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=c, alpha=0.25, lw=0); ax.plot(x, mu, color=c, marker="o", ms=1.6, label=lab)
            ax.set_xscale("log")
            if row == 0: ax.set_title(title, fontsize=6.5)
            if row == 1: ax.legend(fontsize=5.2, loc="center left", bbox_to_anchor=(0.0, 0.62))
            fs.panel_label(ax, "ABCDEF"[3 * row + j], x=-0.22 if j == 0 else -0.14)
            if j == 0: ax.set_ylabel(ylab)
            if row == 1: ax.set_xlabel("items circulated")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], float(sys.argv[3]))
