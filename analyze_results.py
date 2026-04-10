"""
Extended Analysis for Deepfake Percolation ABM
"""

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from scipy.stats import ks_2samp
from pathlib import Path

from utils import *
import os

if not os.path.exists("./figures"):
    os.mkdir("./figures")

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════════════════
DATA_DIR      = Path("./data/sweep_2026_04_10_1044/")
EDGE_SAMPLE_N = None          # None → all sims; int → sample that many sim-ids
P_FAKE_SPLIT  = "quartiles"   # "quartiles" | "tertiles" | [b1, b2, ...]
BIN_N         = 12            # bins for binned line plots
PHASE_GRID    = 25            # grid resolution for phase-transition heatmaps
RANDOM_SEED   = 42

# PAWN config
PAWN_N_SLICES      = 10       # conditioning slices per parameter
PAWN_N_BOOT        = 200      # bootstrap resamples for CIs
PAWN_CONV_STEPS    = [0.2, 0.4, 0.6, 0.8, 1.0]   # fractions of N for convergence check

PARAM_COLS = ["v_cost", "loss", "tpr", "fpr", "p_fake"]
PARAM_LABELS = {
    "v_cost": "Verif. Cost",
    "loss":   "Payoff Loss",
    "tpr":    "TPR",
    "fpr":    "FPR",
    "p_fake": "P(Fake)",
}

# ── plotting style ────────────────────────────────────────────────────────────
sns.set_context("notebook", font_scale=1.15)
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       14,
    "axes.labelsize":  15,
    "axes.titlesize":  14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi":      120,
    "axes.grid":       True,
    "grid.alpha":      0.35,
    "axes.axisbelow":  True,
})
PALETTE = sns.color_palette("muted")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def make_regimes(sim_df, split="quartiles"):
    """
    Attach a 'regime' string column to sim_df.
    Returns (sim_df_copy, ordered label list, boundary list).
    """
    pf = sim_df["p_fake"].values
    if split == "quartiles":
        boundaries = list(np.percentile(pf, [25, 50, 75]))
        prefixes   = ["Q1 (Low)", "Q2 (Mid-Low)", "Q3 (Mid-High)", "Q4 (High)"]
    elif split == "tertiles":
        boundaries = list(np.percentile(pf, [33.3, 66.7]))
        prefixes   = ["Low", "Mid", "High"]
    else:
        boundaries = list(split)
        prefixes   = [f"Bin {i+1}" for i in range(len(boundaries) + 1)]

    bounds = [-np.inf] + boundaries + [np.inf]
    n      = len(bounds) - 1
    labels = np.empty(len(pf), dtype=object)
    names  = []
    for i in range(n):
        lo, hi = bounds[i], bounds[i + 1]
        if i == 0:
            name = f"{prefixes[i]} p_fake (< {boundaries[0]:.2f})"
        elif i == n - 1:
            name = f"{prefixes[i]} p_fake (> {boundaries[-1]:.2f})"
        else:
            name = f"{prefixes[i]} p_fake ({boundaries[i-1]:.2f}-{boundaries[i]:.2f})"
        names.append(name)
        mask = (pf > lo) & (pf <= hi)
        labels[mask] = name

    out = sim_df.copy()
    out["regime"] = labels
    return out, names, boundaries


def binned_line(ax, x, y, n_bins=BIN_N, color="dodgerblue",
                label=None, lw=2.5, ls="-"):
    """
    Bin x into n_bins equal-width bins; plot mean(y) per bin as a line
    with ±1 std shaded band. Skips bins with fewer than 5 observations.
    """
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    edges  = np.linspace(lo, hi, n_bins + 1)
    idx    = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    cx, mn, sd = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 5:
            continue
        cx.append((edges[b] + edges[b + 1]) / 2)
        mn.append(np.mean(y[m]))
        sd.append(np.std(y[m]))
    cx, mn, sd = map(np.array, (cx, mn, sd))
    ax.plot(cx, mn, color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(cx, mn - sd, mn + sd, alpha=0.18, color=color)


def smooth_grid(df, xp, yp, zp, n_grid=PHASE_GRID, sigma=1.0):
    """
    Bin (xp, yp) onto a regular grid, return (xi, yi, zi_smoothed).
    NaN cells are filled with the global mean before Gaussian smoothing.
    """
    x, y, z = df[xp].values, df[yp].values, df[zp].values
    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    xi_i = np.clip(np.digitize(x, xi) - 1, 0, n_grid - 1)
    yi_i = np.clip(np.digitize(y, yi) - 1, 0, n_grid - 1)
    sums   = np.zeros((n_grid, n_grid))
    counts = np.zeros((n_grid, n_grid))
    np.add.at(sums,   (xi_i, yi_i), z)
    np.add.at(counts, (xi_i, yi_i), 1)
    with np.errstate(invalid="ignore"):
        zi = np.where(counts > 0, sums / counts, np.nan)
    zi_filled = np.where(np.isnan(zi), np.nanmean(zi), zi)
    return xi, yi, gaussian_filter(zi_filled, sigma=sigma)


# ═══════════════════════════════════════════════════════════════════════════
# PAWN
# ═══════════════════════════════════════════════════════════════════════════

def pawn_indices(X: pd.DataFrame, y: np.ndarray, n_slices: int = PAWN_N_SLICES) -> pd.DataFrame:
    """
    Compute PAWN sensitivity indices for each column of X against output y.

    Algorithm
    ---------
    For each input Xi:
      - Divide the range of Xi into n_slices equal-frequency bins.
      - For each bin, compute the KS statistic between the conditional
        distribution Y|Xi∈bin and the unconditional distribution of Y.
      - Summary statistics: median KS (robust, recommended) and max KS
        (used in original Pianosi & Wagener 2015 paper).

    Returns a DataFrame indexed by parameter name with columns:
      ks_median, ks_max, ks_per_slice (list of per-slice KS values).
    """
    rows = []
    for col in X.columns:
        xi    = X[col].values
        # equal-frequency binning so every slice has the same number of points
        edges = np.percentile(xi, np.linspace(0, 100, n_slices + 1))
        edges = np.unique(edges)          # collapse duplicates at boundaries
        ks_vals = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (xi >= lo) & (xi <= hi)
            if mask.sum() < 10:
                continue
            stat, _ = ks_2samp(y[mask], y)
            ks_vals.append(stat)
        rows.append({
            "param":         col,
            "ks_median":     float(np.median(ks_vals)),
            "ks_max":        float(np.max(ks_vals)),
            "ks_per_slice":  ks_vals,
        })
    return pd.DataFrame(rows).set_index("param")


def pawn_bootstrap_ci(X: pd.DataFrame, y: np.ndarray,
                      n_slices: int = PAWN_N_SLICES,
                      n_boot:   int = PAWN_N_BOOT,
                      ci:       float = 0.95,
                      rng_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Bootstrap confidence intervals on the PAWN median KS index.
    Resamples rows of (X, y) with replacement n_boot times.
    Returns a DataFrame with columns: ci_lo, ci_median, ci_hi.
    """
    rng      = np.random.default_rng(rng_seed)
    n        = len(y)
    alpha    = (1 - ci) / 2
    boot_med = {col: [] for col in X.columns}

    for _ in range(n_boot):
        idx    = rng.integers(0, n, n)
        X_boot = X.iloc[idx].reset_index(drop=True)
        y_boot = y[idx]
        res    = pawn_indices(X_boot, y_boot, n_slices)
        for col in X.columns:
            boot_med[col].append(res.loc[col, "ks_median"])

    rows = []
    for col in X.columns:
        vals = np.array(boot_med[col])
        rows.append({
            "param":     col,
            "ci_lo":     float(np.percentile(vals, 100 * alpha)),
            "ci_median": float(np.median(vals)),
            "ci_hi":     float(np.percentile(vals, 100 * (1 - alpha))),
        })
    return pd.DataFrame(rows).set_index("param")


def pawn_convergence(X: pd.DataFrame, y: np.ndarray,
                     fractions: list = PAWN_CONV_STEPS,
                     n_slices:  int  = PAWN_N_SLICES,
                     rng_seed:  int  = RANDOM_SEED) -> pd.DataFrame:
    """
    Check convergence of PAWN median KS indices by computing them on
    random subsets of increasing size.

    Returns a long-form DataFrame with columns:
      param, n_samples, ks_median.
    """
    rng  = np.random.default_rng(rng_seed)
    n    = len(y)
    rows = []
    for frac in fractions:
        size = int(n * frac)
        idx  = rng.choice(n, size=size, replace=False)
        res  = pawn_indices(X.iloc[idx].reset_index(drop=True), y[idx], n_slices)
        for col in X.columns:
            rows.append({
                "param":     col,
                "n_samples": size,
                "ks_median": res.loc[col, "ks_median"],
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("Loading simulation and node data")
sim_df  = feather.read_feather(DATA_DIR / "simulations.arrow")
node_df = feather.read_feather(DATA_DIR / "nodes.arrow")
print(f"  sim_df  : {sim_df.shape}")
print(f"  node_df : {node_df.shape}")

def compute_normalized_payoff(row):
    rep = row["payoff"]
    best = row["best_payoff"]
    worst = row["worst_payoff"]
    
    if rep < 0 and worst != 0:
        return - (rep / worst)
    elif rep > 0 and best != 0:
        return rep / best
    else:
        return 0.0

node_df["agent_veracity"] = node_df.apply(compute_normalized_payoff, axis=1)

# ── Sen welfare per simulation ────────────────────────────────────────────────
print("Computing Sen welfare per simulation")
sen_vals = (
    node_df.groupby("global_sim_id")["agent_veracity"]
    .apply(lambda x: sen_welfare(x.values))
    .reset_index()
    .rename(columns={"agent_veracity": "sen_welfare"})
)
sim_df = sim_df.merge(sen_vals, on="global_sim_id")

# ── p_fake regimes ────────────────────────────────────────────────────────────
sim_df, REGIME_ORDER, BOUNDARIES = make_regimes(sim_df, P_FAKE_SPLIT)
N_REGIMES     = len(REGIME_ORDER)
REGIME_COLORS = PALETTE[:N_REGIMES]
print(f"  Regimes ({P_FAKE_SPLIT}): {REGIME_ORDER}")

# ── propagate regime & lhs_id onto node_df ───────────────────────────────────
regime_map = dict(zip(sim_df["global_sim_id"], sim_df["regime"]))
lhs_map    = dict(zip(sim_df["global_sim_id"], sim_df["lhs_id"]))
node_df["regime"] = node_df["global_sim_id"].map(regime_map)
node_df["lhs_id"] = node_df["global_sim_id"].map(lhs_map)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PAWN SENSITIVITY ANALYSIS  +  CONVERGENCE DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] PAWN Sensitivity Analysis")

X_pawn = sim_df[PARAM_COLS]
y_pawn = sim_df["sen_welfare"].values

# ── main indices ──────────────────────────────────────────────────────────────
pawn_df = pawn_indices(X_pawn, y_pawn)
print(f"\n  {'Parameter':<25} {'KS median':>10} {'KS max':>10}")
for param, row in pawn_df.sort_values("ks_median", ascending=False).iterrows():
    print(f"  {PARAM_LABELS[param]:<25} {row.ks_median:>10.4f} {row.ks_max:>10.4f}")

# ── bootstrap CIs ─────────────────────────────────────────────────────────────
print(f"\n  Computing bootstrap CIs ({PAWN_N_BOOT} resamples)")
ci_df = pawn_bootstrap_ci(X_pawn, y_pawn)

# ── convergence ───────────────────────────────────────────────────────────────
print(f"  Running convergence check at fractions {PAWN_CONV_STEPS}")
conv_df = pawn_convergence(X_pawn, y_pawn)

# ── Plot 1a: bar chart with CIs ───────────────────────────────────────────────
pawn_sorted = pawn_df.join(ci_df).sort_values("ks_median", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(pawn_sorted))
ax.bar(x_pos, pawn_sorted["ks_median"],
       color=PALETTE[0], width=0.55, zorder=3, label="PAWN (median KS)")
yerr_lo = (pawn_sorted["ks_median"] - pawn_sorted["ci_lo"]).values
yerr_hi = (pawn_sorted["ci_hi"]     - pawn_sorted["ks_median"]).values
ax.errorbar(x_pos, pawn_sorted["ks_median"],
            yerr=[yerr_lo, yerr_hi],
            fmt="none", color="black", capsize=5, lw=1.5, zorder=4)
ax.set_xticks(x_pos)
ax.set_xticklabels([PARAM_LABELS[p] for p in pawn_sorted.index], rotation=0)
ax.set_ylabel("PAWN Index (KS statistic)")
ax.set_title("PAWN Global Sensitivity Analysis - Sen Welfare\n"
             "(bars = median KS ± 95% CI)")
ax.set_ylim(0, pawn_sorted["ci_hi"].max() * 1.25)
fig.tight_layout()
fig.savefig("figures/pawn_sensitivity.pdf", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 2.  PARTIAL DEPENDENCE — TPR vs. FPR → Sen Welfare
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] Partial dependence: TPR and FPR on Sen Welfare")

# ── Plot 2a: marginal PDP for TPR and FPR (collapsing over all other params) ─
fig, ax = plt.subplots(figsize=(9, 6))
binned_line(ax, sim_df["tpr"].values, sim_df["sen_welfare"].values,
            color=PALETTE[0], label="TPR", lw=3.0, ls="-")
binned_line(ax, sim_df["fpr"].values, sim_df["sen_welfare"].values,
            color="crimson",  label="FPR", lw=3.0, ls="--")
ax.set_xlabel("Parameter value", fontsize=16)
ax.set_ylabel("Sen Welfare", fontsize=16)
ax.set_title("Marginal Partial Dependence of Sen Welfare\non TPR and FPR (all simulations)", fontsize=15)
ax.legend(fontsize=14)
ax.tick_params(labelsize=13)
fig.tight_layout()
fig.savefig("figures/pdp_marginal.pdf", bbox_inches="tight")
plt.show()

# ── Plot 2b: TPR and FPR slopes as a continuous function of p_fake ────────────
#
# For each fine p_fake slice: fit a linear slope of sen_welfare ~ TPR  and
# sen_welfare ~ FPR within that slice.  Plot the two slope series against
# p_fake.  This directly answers "how does the marginal value of improving
# TPR vs. reducing FPR change as fake prevalence rises?" without any
# arbitrary binning.
#
PF_SLOPE_BINS = 10          # number of p_fake slices
pf_vals  = sim_df["p_fake"].values
pf_edges = np.linspace(pf_vals.min(), pf_vals.max(), PF_SLOPE_BINS + 1)
pf_idx   = np.clip(np.digitize(pf_vals, pf_edges) - 1, 0, PF_SLOPE_BINS - 1)

pf_centres, tpr_slopes, fpr_slopes = [], [], []
for b in range(PF_SLOPE_BINS):
    mask = pf_idx == b
    if mask.sum() < 20:
        continue
    sub = sim_df[mask]
    pf_centres.append((pf_edges[b] + pf_edges[b + 1]) / 2)
    tpr_slopes.append(np.polyfit(sub["tpr"], sub["sen_welfare"], 1)[0])
    fpr_slopes.append(np.polyfit(sub["fpr"], sub["sen_welfare"], 1)[0])

pf_centres = np.array(pf_centres)
tpr_slopes = np.array(tpr_slopes)
fpr_slopes = np.array(fpr_slopes)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(pf_centres, tpr_slopes, color=PALETTE[0], lw=3.0,
        marker="o", ms=5, label="Marginal gain from increasing TPR")
ax.plot(pf_centres, -np.array(fpr_slopes), color="crimson", lw=3.0,
        marker="o", ms=5, ls="--",
        label="Marginal gain from reducing FPR")
ax.axhline(0, color="black", lw=1.0, ls=":")
ax.set_xlabel("P(Fake)", fontsize=16)
ax.set_ylabel("Marginal welfare gain (Sen Welfare)", fontsize=16)
ax.set_title("Marginal Welfare Gain from Improving TPR vs. Reducing FPR\n"
             "as a Function of P(Fake)", fontsize=15)
ax.legend(fontsize=13)
ax.tick_params(labelsize=13)
fig.tight_layout()
fig.savefig("figures/pdp_slopes_vs_pfake.pdf", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 3.  BELIEF MISCALIBRATION DRIVERS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] Belief miscalibration")

cal_df = node_df[["global_sim_id", "node_id",
                   "true_alpha", "true_beta", "regime"]].copy()
cal_df["true_prevalence"] = (
    cal_df["true_alpha"] / (cal_df["true_alpha"] + cal_df["true_beta"])
)
p_true_map            = dict(zip(sim_df["global_sim_id"], 1 - sim_df["p_fake"]))
cal_df["p_true"]      = cal_df["global_sim_id"].map(p_true_map)
# positive = agent overestimates truth prevalence
cal_df["cal_error"]   = cal_df["true_prevalence"] - cal_df["p_true"]
cal_df["abs_cal_err"] = cal_df["cal_error"].abs()

cal_sim = (
    cal_df.groupby("global_sim_id")
    .agg(mean_cal_error=("cal_error",    "mean"),
         mean_abs_error=("abs_cal_err",  "mean"))
    .reset_index()
)
sim_df = sim_df.merge(cal_sim, on="global_sim_id")

# |calibration error| vs. each parameter ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
axes_flat = axes.flatten()

for ax, param in zip(axes_flat[:len(PARAM_COLS)], PARAM_COLS):
    binned_line(ax, sim_df[param].values, sim_df["mean_abs_error"].values,
                color=PALETTE[3], lw=2.5)
    ax.set_xlabel(PARAM_LABELS[param])
    ax.set_ylabel("|Calibration Error|")
    ax.set_title(f"|Cal. Error| vs. {PARAM_LABELS[param]}")

fig.suptitle("Belief Miscalibration Drivers", fontsize=15)
fig.tight_layout()
fig.savefig("figures/miscalibration_params.pdf")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 4.  VERACITY DIFFERENTIAL VS. SEN WELFARE
#     Are the two outcomes aligned, or can you have one without the other?
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4] Veracity differential vs. Sen welfare")

vd = sim_df["veracity_differential"].values
sw = sim_df["sen_welfare"].values

# ── scatter with binned line, coloured by p_fake ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# left panel: raw relationship with binned mean overlaid
ax = axes[0]
sc = ax.scatter(vd, sw, c=sim_df["p_fake"].values, cmap="plasma", alpha=0.25, s=6, rasterized=True)
cb = fig.colorbar(sc, ax=ax)
cb.set_label("P(Fake)", fontsize=13)
ax.set_xlabel("Veracity Differential", fontsize=15)
ax.set_ylabel("Sen Welfare", fontsize=15)
ax.set_title("Sen Welfare vs. Veracity Differential\n(coloured by P(Fake))", fontsize=14)
ax.tick_params(labelsize=12)

# right panel: binned line broken out by low / mid / high p_fake terciles
#   (using only 3 groups here purely for visual clarity on a single axes —
#    this is display grouping, not an analytical binning of the outcome)
ax  = axes[1]
pf  = sim_df["p_fake"].values
t1, t2 = np.percentile(pf, [33.3, 66.7])
groups = [
    ("Low P(Fake)",  pf <= t1,             PALETTE[0]),
    ("Mid P(Fake)",  (pf > t1) & (pf <= t2), PALETTE[1]),
    ("High P(Fake)", pf > t2,              PALETTE[2]),
]
for label, mask, color in groups:
    binned_line(ax, vd[mask], sw[mask], color=color, label=label, lw=2.5)
ax.axvline(0, color="grey", lw=1.0, ls="--")
ax.set_xlabel("Veracity Differential", fontsize=15)
ax.set_ylabel("Sen Welfare", fontsize=15)
ax.set_title("Sen Welfare vs. Veracity Differential\nby P(Fake) Tercile", fontsize=14)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)

fig.tight_layout()
fig.savefig("figures/veracity_vs_welfare.pdf", bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# 5.  VERIFICATION RATE AS MEDIATOR
#     Does verify rate explain the TPR/FPR → welfare relationship?
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] Verification rate as mediator")

vr = sim_df["avg_verify_rate"].values

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# row 1: verify rate as a function of each input parameter
for ax, param in zip(axes[0], ["v_cost", "tpr", "fpr"]):
    binned_line(ax, sim_df[param].values, vr, color=PALETTE[4], lw=2.5)
    ax.set_xlabel(PARAM_LABELS[param], fontsize=14)
    ax.set_ylabel("Avg. Verification Rate", fontsize=14)
    ax.set_title(f"Verification Rate vs. {PARAM_LABELS[param]}", fontsize=13)
    ax.tick_params(labelsize=12)

# row 2, left: verify rate → sen welfare (the mediation path)
ax = axes[1, 0]
binned_line(ax, vr, sw, color=PALETTE[4], lw=2.5)
ax.set_xlabel("Avg. Verification Rate", fontsize=14)
ax.set_ylabel("Sen Welfare", fontsize=15)
ax.set_title("Sen Welfare vs. Verification Rate", fontsize=13)
ax.tick_params(labelsize=12)

# row 2, middle: verify rate → veracity differential
ax = axes[1, 1]
binned_line(ax, vr, vd, color=PALETTE[5], lw=2.5)
ax.axhline(0, color="grey", lw=1.0, ls="--")
ax.set_xlabel("Avg. Verification Rate", fontsize=14)
ax.set_ylabel("Veracity Differential", fontsize=14)
ax.set_title("Veracity Differential vs. Verification Rate", fontsize=13)
ax.tick_params(labelsize=12)

# row 2, right: 2D surface — v_cost × p_fake → verify rate
#   shows where in parameter space agents actually bother to verify
ax  = axes[1, 2]
xi, yi, zi = smooth_grid(sim_df, "v_cost", "p_fake", "avg_verify_rate", sigma=1.0)
im = ax.pcolormesh(xi, yi, zi.T, cmap="YlOrRd", shading="auto")
fig.colorbar(im, ax=ax).set_label("Avg. Verification Rate", fontsize=12)
ax.set_xlabel(PARAM_LABELS["v_cost"], fontsize=14)
ax.set_ylabel(PARAM_LABELS["p_fake"], fontsize=14)
ax.set_title("Verification Rate:\nVerif. Cost × P(Fake)", fontsize=13)
ax.tick_params(labelsize=12)

fig.suptitle("Verification Rate as a Mediator of Technology Parameters → Welfare",
             fontsize=15)
fig.tight_layout()
fig.savefig("figures/verification_mediation.pdf", bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 6.  CASCADE SIZE ASYMMETRY
#     True vs. fake cascade sizes: distributions, gap, and parameter drivers
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] Cascade size asymmetry")

tc = sim_df["avg_true_cascade"].values
fc = sim_df["avg_fake_cascade"].values
gap = tc - fc          # positive = true cascades reach more agents

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# ── row 1: distributions ──────────────────────────────────────────────────────
ax = axes[0, 0]
ax.hist(tc, bins=40, color=PALETTE[0], alpha=0.65, label="True cascades",  density=True)
ax.hist(fc, bins=40, color="crimson",  alpha=0.65, label="Fake cascades",  density=True)
ax.set_xlabel("Avg. Cascade Size (fraction of agents)", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_title("Distribution of True vs. Fake Cascade Sizes", fontsize=13)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)

ax = axes[0, 1]
ax.hist(gap, bins=40, color="steelblue", alpha=0.8, density=True)
ax.axvline(0,              color="black", lw=1.5, ls="--", label="Zero (equal sizes)")
ax.axvline(np.mean(gap),   color="crimson", lw=2.0, ls="-",  label=f"Mean = {np.mean(gap):.3f}")
ax.set_xlabel("True − Fake Cascade Size", fontsize=14)
ax.set_ylabel("Density", fontsize=14)
ax.set_title("Cascade Size Gap Distribution", fontsize=13)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)

# fraction of sims where fake cascades are larger than true
frac_fake_larger = (fc > tc).mean()
ax = axes[0, 2]
ax.bar(["True > Fake", "Fake ≥ True"],
       [1 - frac_fake_larger, frac_fake_larger],
       color=[PALETTE[0], "crimson"], edgecolor="white", width=0.5)
ax.set_ylabel("Fraction of simulations", fontsize=14)
ax.set_title(f"How Often Do Fake Cascades Dominate?\n"
             f"({frac_fake_larger:.1%} of simulations)", fontsize=13)
ax.set_ylim(0, 1)
ax.tick_params(labelsize=12)

# ── row 2: gap as a function of key parameters ────────────────────────────────
for ax, param in zip(axes[1], ["p_fake", "tpr", "fpr"]):
    binned_line(ax, sim_df[param].values, tc,
                color=PALETTE[0], label="True cascade", lw=2.5, ls="-")
    binned_line(ax, sim_df[param].values, fc,
                color="crimson",  label="Fake cascade", lw=2.5, ls="--")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(PARAM_LABELS[param], fontsize=14)
    ax.set_ylabel("Avg. Cascade Size", fontsize=14)
    ax.set_title(f"True vs. Fake Cascade Size\nvs. {PARAM_LABELS[param]}", fontsize=13)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)

fig.suptitle("Cascade Size Asymmetry: True vs. Fake Content Spread",
             fontsize=15)
fig.tight_layout()
fig.savefig("figures/cascade_asymmetry.pdf", bbox_inches="tight")
plt.show()

print("\nAll analyses complete.")