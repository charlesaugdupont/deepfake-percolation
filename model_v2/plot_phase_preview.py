import sys, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/work"); import figstyle as fs; fs.setup()
D = pd.read_parquet(sys.argv[1]); out = sys.argv[2]
mets = [("dip4", "bifurcation (4·dip)", 0, 0.6, "magma"), ("frac_locked", "share locked out", 0, 1, "viridis"),
        ("frac_trapped", "share trapped (would gain from checking)", 0, 0.6, "viridis"), ("bc", "BC", 0.3, 1, "magma"),
        ("mean_J", "mean trust Ĵ", 0, 0.8, "cividis"), ("delta", "truth differential Δ", 0, 0.6, "cividis")]
lams = sorted(D.lam.unique())
fig, axs = plt.subplots(len(lams), len(mets), figsize=(fs.W2 * 1.3, 1.9 * len(lams)), squeeze=False)
for i, lam in enumerate(lams):
    d = D[D.lam == lam].groupby(["phi", "cost"]).mean(numeric_only=True).reset_index()
    phis = np.sort(d.phi.unique()); costs = np.sort(d.cost.unique())
    for j, (m, lab, lo, hi, cm) in enumerate(mets):
        Z = d.pivot(index="cost", columns="phi", values=m).loc[costs, phis].values
        ax = axs[i, j]
        pe = np.r_[phis - 0.025, phis[-1] + 0.025]; lc = np.log10(costs); ce = np.r_[lc - (lc[1]-lc[0])/2, lc[-1] + (lc[1]-lc[0])/2]
        pc = ax.pcolormesh(pe, ce, Z, vmin=lo, vmax=hi, cmap=cm)
        thbar = 10 ** ce  # mark where average agent's barrier equals prior J0=1/3
        ax.axhline(np.log10((1/3) / (1 + 1 / lam)), color="w", lw=0.6, ls="--")
        ax.set_yticks([-2, -1, 0]); ax.set_yticklabels(["0.01", "0.1", "1"])
        if i == 0: ax.set_title(lab, fontsize=6.5)
        if j == 0: ax.set_ylabel(f"loss/gain = {lam:g}\nchecking cost / gain")
        if i == len(lams) - 1: ax.set_xlabel("share of fakes φ")
        plt.colorbar(pc, ax=ax, fraction=0.05)
fig.tight_layout(); fig.savefig(out, dpi=150)
