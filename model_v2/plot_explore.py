import sys, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
D = pd.read_csv(sys.argv[1]); out = sys.argv[2]
metrics = [("bc", "BC (trust in checker)", 0.2, 1.0), ("er", "balance x gap  4s(1-s)Δ", 0, 0.8),
           ("frac_ever", "share ever checked", 0, 1), ("frac_locked", "share locked out (Ĵ ≤ θ)", 0, 1), ("mean_J", "mean Ĵ", -0.2, 0.8)]
V = list(dict.fromkeys(D.variant))
fig, axs = plt.subplots(len(V), len(metrics), figsize=(3.0 * len(metrics), 2.3 * len(V)), squeeze=False)
for i, v in enumerate(V):
    d = D[D.variant == v].groupby(["phi", "cost"]).mean(numeric_only=True).reset_index()
    phis = np.sort(d.phi.unique()); costs = np.sort(d.cost.unique())
    for j, (m, lab, lo, hi) in enumerate(metrics):
        Z = d.pivot(index="cost", columns="phi", values=m).loc[costs, phis].values
        ax = axs[i, j]
        pc = ax.pcolormesh(np.arange(len(phis) + 1), np.arange(len(costs) + 1), Z, vmin=lo, vmax=hi, cmap="viridis")
        if m == "bc":
            ax.contour(np.arange(len(phis)) + 0.5, np.arange(len(costs)) + 0.5, Z, levels=[5 / 9], colors="w", linewidths=1)
        ax.set_xticks(np.arange(len(phis))[::2] + 0.5); ax.set_xticklabels([f"{p:.2f}" for p in phis[::2]], fontsize=7)
        ax.set_yticks(np.arange(len(costs))[::2] + 0.5); ax.set_yticklabels([f"{c:g}" for c in costs[::2]], fontsize=7)
        if i == 0: ax.set_title(lab, fontsize=9)
        if j == 0: ax.set_ylabel(v + "\ncost / gain", fontsize=8)
        if i == len(V) - 1: ax.set_xlabel("share of fakes φ", fontsize=8)
        plt.colorbar(pc, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(out, dpi=80)
