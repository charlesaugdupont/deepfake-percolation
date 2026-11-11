"""SI figure: the numba engine (v1 switches) reproduces the Julia sweep (40 LHS points x 50 seeds each)."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()
P = pd.read_csv(sys.argv[1]); out = sys.argv[2]
mets = [("frac_verified_ever", "share ever checked"), ("bc_J", "BC of trust"), ("avg_true_cascade", "reach, genuine"),
        ("avg_fake_cascade", "reach, fake"), ("avg_verify_rate", "checks per item received"), ("mean_true_belief", "believed share genuine"),
        ("mean_tpr_belief_verifiers", "believed TPR (checkers)"), ("mean_J_nonverifiers", "trust, never checked")]
fig, axs = plt.subplots(2, 4, figsize=(fs.W2, 3.6))
for ax, (m, lab) in zip(axs.ravel(), mets):
    x = P[f"julia_{m}_mean"]; y = P[f"numba_{m}_mean"]; ex = 1.96 * P[f"julia_{m}_sem"]; ey = 1.96 * P[f"numba_{m}_sem"]
    ax.errorbar(x, y, xerr=ex, yerr=ey, fmt="o", ms=2, lw=0.5, color=fs.BLUE, capsize=0)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=fs.INK2, lw=0.6, ls=":")
    ax.set_title(f"{lab}\nr = {np.corrcoef(x, y)[0, 1]:.4f}", fontsize=6)
    ax.set_xlabel("Julia (v1 code)"); ax.set_ylabel("this engine")
fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); print("saved", out)
