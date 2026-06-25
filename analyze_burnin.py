# # Burn-in Convergence Diagnostics — Single Parameter Point
#
# Loads output from `run_burnin_diagnostic.jl` (single fixed parameter point,
# many stochastic reps) and produces convergence plots.
#
# Files expected in DATA_DIR:
#   burnin_snapshots.arrow  — per (rep_id, cascade_id, node_id)
#   burnin_cascades.arrow   — per (rep_id, cascade_id)
#   diagnostic_config.txt   — parameter values and scalar config

# %%
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr

DATA_DIR = Path("./data/burnin_diagnostic/")

snap_df = feather.read_feather(DATA_DIR / "burnin_snapshots.arrow")
casc_df = feather.read_feather(DATA_DIR / "burnin_cascades.arrow")

cfg = {}
with open(DATA_DIR / "diagnostic_config.txt") as f:
    for line in f:
        k, v = line.strip().split(" = ")
        try:    cfg[k] = float(v)
        except: cfg[k] = v

BURN_IN   = int(cfg["burn_in_length"])
TRUE_PREV = 1.0 - cfg["p_fake"]
TRUE_TPR  = cfg["tpr"]
TRUE_FPR  = cfg["fpr"]

PARAM_STR = (f"v_cost={cfg['v_cost']}  loss={cfg['rep_loss']}  "
             f"tpr={cfg['tpr']}  fpr={cfg['fpr']}  p_fake={cfg['p_fake']}")

print(f"Reps        : {snap_df['rep_id'].nunique()}")
print(f"Timepoints  : {snap_df['cascade_id'].nunique()}")
print(f"Burn-in     : {BURN_IN} cascades")
print(f"Ground truth: p_true={TRUE_PREV:.2f}  TPR={TRUE_TPR:.2f}  FPR={TRUE_FPR:.2f}")

# ── style ─────────────────────────────────────────────────────────────────────
sns.set_context("notebook", font_scale=1.1)
plt.rcParams.update({
    "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True, "figure.dpi": 120,
})
ALPHA_REP  = 0.15
ALPHA_BAND = 0.20
Path("figures").mkdir(exist_ok=True)

def shade_burnin(ax):
    ylim = ax.get_ylim()
    ax.axvspan(0, BURN_IN, color="gold", alpha=0.07, zorder=0)
    ax.axvline(BURN_IN, color="#d62728", lw=1.5, ls="--", label="Burn-in end")
    ax.set_ylim(ylim)

def add_truth(ax, value, label):
    ax.axhline(value, color="black", lw=1.3, ls=":", label=f"True = {value:.2f}")

def per_rep_mean(col):
    return (snap_df.groupby(["rep_id", "cascade_id"])[col]
            .mean().reset_index().rename(columns={col: "val"}))

def grand_agg(col):
    pr = per_rep_mean(col)
    agg = pr.groupby("cascade_id")["val"].agg(["mean", "std"]).reset_index()
    agg["lo"] = agg["mean"] - agg["std"]
    agg["hi"] = agg["mean"] + agg["std"]
    return agg

def cross_agent_var(col):
    var_per_rep = (snap_df.groupby(["rep_id", "cascade_id"])[col]
                   .var().reset_index().rename(columns={col: "var"}))
    return var_per_rep.groupby("cascade_id")["var"].mean().reset_index()

def binned_line(ax, x, y, n_bins=12, color="steelblue", label=None, lw=2.5):
    """Bin x, plot mean(y) per bin with ±1 std band."""
    lo, hi = np.nanpercentile(x, 2), np.nanpercentile(x, 98)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx   = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    cx, mn, sd = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 5:
            continue
        cx.append((edges[b] + edges[b+1]) / 2)
        mn.append(np.mean(y[m]))
        sd.append(np.std(y[m]))
    cx, mn, sd = map(np.array, (cx, mn, sd))
    ax.plot(cx, mn, color=color, lw=lw, label=label)
    ax.fill_between(cx, mn - sd, mn + sd, alpha=0.18, color=color)


# =============================================================================
# PLOT 1 — Belief means over time
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes = axes.flatten()

belief_panels = [
    ("true_mean",  "Belief: P(content is true)", TRUE_PREV, "steelblue"),
    ("tpr_mean",   "Belief: TPR",                TRUE_TPR,  "seagreen"),
    ("fpr_mean",   "Belief: FPR",                TRUE_FPR,  "crimson"),
    ("mean_trust", "Belief: Neighbour Trust",     None,      "darkorchid"),
]

for ax, (col, title, truth, color) in zip(axes, belief_panels):
    pr = per_rep_mean(col)
    for rep_id, grp in pr.groupby("rep_id"):
        ax.plot(grp["cascade_id"], grp["val"], color=color, alpha=ALPHA_REP, lw=0.8)
    agg = grand_agg(col)
    ax.fill_between(agg["cascade_id"], agg["lo"], agg["hi"], alpha=ALPHA_BAND, color=color)
    ax.plot(agg["cascade_id"], agg["mean"], color=color, lw=2.5, label="Mean across reps")
    if truth is not None:
        add_truth(ax, truth, f"True = {truth:.2f}")
        ax.legend(fontsize=10)
    shade_burnin(ax)
    ax.set_title(title)
    ax.set_ylabel("Mean belief (Beta mean)")
    ax.set_xlabel("Cascade #")

fig.suptitle(f"Belief Means Over Time\n{PARAM_STR}\n"
             f"(thin = individual reps; bold = mean; band = ±1 std)", fontsize=12)
fig.tight_layout()
fig.savefig("figures/burnin_belief_means.pdf", bbox_inches="tight")
plt.show()

# =============================================================================
# PLOT 2 — Cross-agent belief variance
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes = axes.flatten()

var_panels = [
    ("true_mean",  "Variance: P(true) belief",  "steelblue"),
    ("tpr_mean",   "Variance: TPR belief",       "seagreen"),
    ("fpr_mean",   "Variance: FPR belief",       "crimson"),
    ("mean_trust", "Variance: Neighbour Trust",  "darkorchid"),
]

for ax, (col, title, color) in zip(axes, var_panels):
    var_per_rep = (snap_df.groupby(["rep_id", "cascade_id"])[col]
                   .var().reset_index().rename(columns={col: "var"}))
    for rep_id, grp in var_per_rep.groupby("rep_id"):
        ax.plot(grp["cascade_id"], grp["var"], color=color, alpha=ALPHA_REP, lw=0.8)
    mean_var = var_per_rep.groupby("cascade_id")["var"].mean().reset_index()
    ax.plot(mean_var["cascade_id"], mean_var["var"], color=color, lw=2.5)
    shade_burnin(ax)
    ax.set_title(title)
    ax.set_ylabel("Cross-agent variance")
    ax.set_xlabel("Cascade #")

fig.suptitle(f"Belief Variance Over Time  (lower = more consensus)\n{PARAM_STR}", fontsize=12)
fig.tight_layout()
fig.savefig("figures/burnin_belief_variance.pdf", bbox_inches="tight")
plt.show()

# =============================================================================
# PLOT 3 — Verification rate over time
# =============================================================================
ROLL = 30
fig, ax = plt.subplots(figsize=(12, 5))

for rep_id, grp in casc_df.groupby("rep_id"):
    grp = grp.sort_values("cascade_id")
    ax.plot(grp["cascade_id"], grp["verify_rate"].rolling(ROLL, min_periods=1).mean(),
            color="steelblue", alpha=ALPHA_REP, lw=0.8)

grand_vr = (casc_df.groupby("cascade_id")["verify_rate"]
            .mean().reset_index().sort_values("cascade_id"))
ax.plot(grand_vr["cascade_id"],
        grand_vr["verify_rate"].rolling(ROLL, min_periods=1).mean(),
        color="steelblue", lw=2.5, label=f"Mean across reps (roll={ROLL})")

shade_burnin(ax)
ax.set_xlabel("Cascade #")
ax.set_ylabel("Verification rate")
ax.set_title(f"Verification Rate Over Time  (rolling window = {ROLL})\n{PARAM_STR}")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig("figures/burnin_verify_rate.pdf", bbox_inches="tight")
plt.show()

# =============================================================================
# PLOT 4 — True vs. fake cascade size over time
# =============================================================================
ROLL2 = 30
fig, ax = plt.subplots(figsize=(12, 5))

for label, color, is_fake_val in [
    ("True content", "steelblue", False),
    ("Fake content", "crimson",   True),
]:
    sub = casc_df[casc_df["is_fake"] == is_fake_val]
    mean_cs = (sub.groupby("cascade_id")["cascade_size"]
               .mean().reset_index().sort_values("cascade_id"))
    mean_cs["rolled"] = mean_cs["cascade_size"].rolling(ROLL2, min_periods=1).mean()
    ax.plot(mean_cs["cascade_id"], mean_cs["rolled"], color=color, lw=2.5, label=label)

shade_burnin(ax)
ax.set_xlabel("Cascade #")
ax.set_ylabel("Avg. cascade size (agents reached)")
ax.set_title(f"True vs. Fake Cascade Size Over Time  (rolling window = {ROLL2})\n{PARAM_STR}")
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig("figures/burnin_cascade_size.pdf", bbox_inches="tight")
plt.show()

# =============================================================================
# PLOT 5 — Belief distributions at three time points
# =============================================================================
all_cids        = sorted(snap_df["cascade_id"].unique())
t_early         = all_cids[0]
t_mid           = all_cids[len(all_cids) // 4]
t_burnin_actual = min(all_cids, key=lambda x: abs(x - BURN_IN))
timepoints      = [t_early, t_mid, t_burnin_actual]
t_labels        = {
    t_early:         f"t={t_early} (start)",
    t_mid:           f"t={t_mid} (~25%)",
    t_burnin_actual: f"t={t_burnin_actual} (burn-in end)",
}

dist_panels = [
    ("true_mean", "P(true) belief", TRUE_PREV, "steelblue"),
    ("tpr_mean",  "TPR belief",     TRUE_TPR,  "seagreen"),
    ("fpr_mean",  "FPR belief",     TRUE_FPR,  "crimson"),
]

fig, axes = plt.subplots(len(dist_panels), len(timepoints), figsize=(14, 10), sharey="row")

for row, (col, ylabel, truth, color) in enumerate(dist_panels):
    for col_i, t in enumerate(timepoints):
        ax = axes[row, col_i]
        data = snap_df[snap_df["cascade_id"] == t][col].dropna()
        ax.hist(data, bins=40, color=color, alpha=0.75, density=True, edgecolor="white")
        if truth is not None:
            ax.axvline(truth, color="black", lw=1.8, ls="--", label=f"True = {truth:.2f}")
            ax.legend(fontsize=9)
        if row == 0:
            ax.set_title(t_labels[t], fontsize=12)
        if col_i == 0:
            ax.set_ylabel(ylabel)
        ax.set_xlabel("Belief value")

fig.suptitle(f"Agent Belief Distributions at Three Time Points  (pooled across all reps)\n{PARAM_STR}",
             fontsize=12)
fig.tight_layout()
fig.savefig("figures/burnin_belief_distributions.pdf", bbox_inches="tight")
plt.show()

print("\nAll burn-in diagnostic plots saved to ./figures/")
