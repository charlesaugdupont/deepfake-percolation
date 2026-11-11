"""v3 Figure 3: regimes over the share of fakes (phi) and the relative cost of checking (c/g), for three loss/gain
ratios (columns). Rows: (A) split of trust (Hartigan's dip), (B) share not checking, (C) share trapped (checking would
pay off for them given the checker's actual quality, but their trust is at or below their threshold). Hatched cells:
nothing circulates (no one shares or checks; mean reach of genuine items < 5%).
(D, E) not checking and trapped against c/g (cells with phi = 0.2-0.5): simulation (markers, 95% CI over
runs) and the one-agent model (lines; nothing fitted, the share of fakes among checked items is taken from the runs).
Usage: v3_fig3_regimes.py <phase parquet> <out.png> <network name> [main lam]"""
import sys, os, json, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

DX, DY = 0.05, 1 / 6
J0, J_TRUE = 1 / 3, 0.70
ROWS = [("dip", fs.CM_SPLIT, "split of trust (dip)", 0.2),
        ("frac_locked", fs.CM_LOCK, "share not checking", 1.0),
        ("frac_trapped", fs.CM_TRAP, "share trapped (checking would pay, trust too low)", 0.25)]
ASPECT = 1.5 / np.sqrt(3) * DX / DY          # screen ratio that makes the hexagons regular


def cellmeans(D, lam):
    """Per lattice cell: reach and the reach gap over all runs; the dip, not checking and trapped over the runs in which
    items circulate (fs.circulating; 'split by measure', 5 Nov). Hatched ('dead'): mean reach over all runs < 5%."""
    d = D[D.lam == lam]; key = ["row", "phi", "cost"]
    g = d.groupby(key).agg(reach_true=("reach_true", "mean"), delta=("delta", "mean"), n=("dip4", "size")).reset_index()
    b = fs.circulating(d).groupby(key).agg(dip4=("dip4", "mean"), frac_locked=("frac_locked", "mean"),
                                           frac_trapped=("frac_trapped", "mean")).reset_index()
    g = g.merge(b, on=key, how="left")
    g["dip"] = g.dip4 / 4                                             # runs store 4 x dip; figures show the dip (0-0.25)
    g["dead"] = g.reach_true < fs.MIN_REACH
    return g


def theory(D, deg, lams, cache, band=(0.19, 0.51)):
    """One-agent model for each loss ratio and cost row: (cost, not checking, trapped); cached as JSON."""
    if cache and os.path.exists(cache):
        T = json.load(open(cache))
        if all(str(l) in T for l in lams):
            return {float(k): np.array(v) for k, v in T.items()}
    from trapmodel import predicted_groups
    out = {}
    for lam in lams:
        g = cellmeans(D, lam); live = g[~g.dead & g.phi.between(*band)][["phi", "cost"]]
        sub = D[(D.lam == lam)].merge(live, on=["phi", "cost"])
        rows = []
        for c, s in sub.groupby("cost"):
            q = float(np.nan_to_num(s.q_fake_checks.mean(), nan=0.25))
            lo, tr = predicted_groups(deg, c, lam, q)   # trapmodel's common setting (11 Nov)
            rows.append((c, lo, tr))
        out[lam] = np.array(rows)
    if cache:
        json.dump({str(k): v.tolist() for k, v in out.items()}, open(cache, "w"))
    return out


def sim_curve(D, lam, m, band=(0.19, 0.51)):
    """Simulation points of the theory panels: per cost row, the mean of metric m over the cells with phi in the band
    where items circulate, and its 95% CI half-width `hw`. Every cell and run has its own seed, so the run-to-run
    variance is estimated within cells and pooled: Var = sum_c s_c^2 / n_c / K^2 over K cells with n_c runs each,
    Student's t with Satterthwaite's degrees of freedom (5 Nov; before, the CI came from the 2-3 run means only)."""
    g = cellmeans(D, lam); live = g[~g.dead & g.phi.between(*band)][["phi", "cost"]]
    sub = D[D.lam == lam].merge(live, on=["phi", "cost"])
    if m not in fs.ALL_RUNS:
        sub = fs.circulating(sub)                                     # behaviour: runs in which items circulate
    rows = []
    for c, s in sub.groupby("cost"):
        cell = s.groupby("phi")[m].agg(["mean", "var", "count"])
        K = len(cell); a = (cell["var"].fillna(0) / cell["count"]).values / K ** 2; v = a.sum()
        dfs = (cell["count"] - 1).values; ok = dfs > 0
        df = v ** 2 / np.sum(a[ok] ** 2 / dfs[ok]) if (v > 0 and ok.any()) else np.nan
        hw = float(fs.t95(df + 1) * np.sqrt(v)) if np.isfinite(df) else 0.0
        rows.append((c, cell["mean"].mean(), hw, K, int(cell["count"].sum())))
    return pd.DataFrame(rows, columns=["cost", "mean", "hw", "cells", "runs"]).set_index("cost")


def read_phase(parquet):
    """phase_<net>.parquet plus, when it exists next to it, phase_lines_<net>.parquet: the four cost rows of Fig. 4F-G
    topped up from 3 to 10 runs per cell on 6 Nov (same seed scheme, further run indices)."""
    D = pd.read_parquet(parquet)
    extra = os.path.join(os.path.dirname(parquet), os.path.basename(parquet).replace("phase_", "phase_lines_", 1))
    if os.path.exists(extra):
        X = pd.read_parquet(extra)
        D = pd.concat([D, X[[c for c in D.columns if c in X.columns]]], ignore_index=True)
    return D


def make(parquet, out, net, main_lam=2.0, main=(0.3, 0.05), cache=None):
    D = read_phase(parquet)
    lams = [1.0, 2.0, 4.0]
    have = sorted(D.lam.unique())
    fig = plt.figure(figsize=(fs.W2, 5.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.0], wspace=0.42)
    left = outer[0].subgridspec(3, 4, width_ratios=[1, 1, 1, 0.06], hspace=0.22, wspace=0.1)
    right = outer[1].subgridspec(2, 1, hspace=0.55)
    for i, (m, cmap, lab, vmax) in enumerate(ROWS):
        for j, lam in enumerate(lams):
            ax = fig.add_subplot(left[i, j])
            if lam in have:
                g = cellmeans(D, lam)
                pc = fs.hexmap(ax, g.phi.values, np.log10(g.cost.values), g[m].values, DX, DY, cmap, 0, vmax,
                               hatch_mask=g.dead.values)
            else:
                ax.text(0.5, -1.5, "still running", ha="center", va="center", fontsize=6, color=fs.INK2)
            ax.set_xlim(0.05 - 0.6 * DX, 0.95 + 0.6 * DX); ax.set_ylim(-3 - 0.75 * DY, 0.75 * DY)
            ax.set_aspect(ASPECT, adjustable="box")
            ax.set_yticks([-3, -2, -1, 0])
            ax.set_xticks([0.1, 0.5, 0.9])
            if j == 0:
                ax.set_yticklabels(["0.001", "0.01", "0.1", "1"]); ax.set_ylabel("relative cost c/g")
                fs.panel_label(ax, "ABC"[i], x=-0.62, y=0.98)
                ax.text(-0.02, 1.03, lab, transform=ax.transAxes, fontsize=6.2, ha="left", va="bottom", color=fs.INK,
                        fontweight="bold" if False else "normal")
            else:
                ax.set_yticklabels([])
            if i == 2:
                ax.set_xlabel("share of fakes φ")
            else:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_title(f"ℓ/g = {lam:g}", fontsize=6.8, pad=14)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        cax = fig.add_subplot(left[i, 3])
        cb = fig.colorbar(pc, cax=cax); cb.ax.tick_params(labelsize=5.2)
    # legend for map annotations, under the maps

    # D, E: theory vs simulation
    import config as C, engine as E
    gnet = C.graph(net) if net not in (None, "hk") else E.holme_kim(1000, 3, 0.5, 1)
    deg = np.diff(gnet[0])
    th = theory(D, deg, have, cache)
    cols = fs.blues(3)
    for k, (m, lab) in enumerate([("frac_locked", "share not checking"), ("frac_trapped", "share trapped")]):
        ax = fig.add_subplot(right[k]); fs.panel_label(ax, "DE"[k], x=-0.38, y=1.04)
        for i, lam in enumerate(lams):
            if lam not in have:
                continue
            s = sim_curve(D, lam, m)
            T = th[lam]
            ax.plot(T[:, 0], T[:, 1 + k], color=cols[i], lw=2.4, alpha=0.35, solid_capstyle="round", zorder=1)
            ax.errorbar(s.index, s["mean"], yerr=s["hw"], fmt="none", ecolor=cols[i], elinewidth=0.7, capsize=0, zorder=2)
            ax.plot(s.index, s["mean"], ls="none", marker=fs.MARKERS[i], ms=2.8, color=cols[i], mec="white",
                    mew=0.3, label=f"ℓ/g = {lam:g}", zorder=3)                 # 95% CIs: error bars, mostly hidden by markers
        ax.set_xscale("log"); ax.set_xlim(8e-4, 1.25); ax.set_xticks([0.001, 0.01, 0.1, 1])
        ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
        ax.set_ylim(0, 1.0 if k == 0 else 0.3); ax.set_ylabel(lab)
        ax.set_xlabel("relative cost c/g")
        ax.set_title("φ = 0.2–0.5", fontsize=6.2)
        if k == 0:
            h_lam, l_lam = ax.get_legend_handles_labels()
    hs = [Patch(facecolor="#e4e2dc", edgecolor="#9a978f", hatch="//////")] + h_lam + [Line2D([], [], color=fs.INK2, lw=2.4, alpha=0.35)]
    ls = ["nothing circulates (no one shares or checks)"] + [f"{x} (D, E: simulation)" if i == 0 else x for i, x in enumerate(l_lam)] + \
         ["theory (one-agent model)"]
    fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=5, fontsize=5.6, frameon=False,
               handlelength=1.8, columnspacing=1.4)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


MAPS = ROWS + [("delta", fs.CM_GAP, "how much further genuine items travel than fakes", 0.8)]
MAP_TITLES = {"dip": "split of trust (dip)", "frac_locked": "share not checking",
              "frac_trapped": "share trapped", "delta": "reach of genuine items\nminus reach of fakes"}


def hex_axes(ax, left_col, bottom_row, lab_y=True):
    ax.set_xlim(0.05 - 0.6 * DX, 0.95 + 0.6 * DX); ax.set_ylim(-3 - 0.75 * DY, 0.75 * DY)
    ax.set_aspect(ASPECT, adjustable="box")
    ax.set_yticks([-3, -2, -1, 0]); ax.set_xticks([0.1, 0.5, 0.9])
    if left_col:
        ax.set_yticklabels(["0.001", "0.01", "0.1", "1"]); ax.set_ylabel("relative cost c/g")
    else:
        ax.set_yticklabels([])
    if bottom_row:
        ax.set_xlabel("share of fakes φ")
    else:
        ax.set_xticklabels([])


def map_row(fig, spec, D, lam, letters=None, titles=True):
    """Four hexagonal maps for one loss/gain ratio: split, not checking, trapped, reach gap; each with a colour bar.
    Colour scales end at round numbers above the largest cell (trapped: largest cell 0.21 for every l/g)."""
    sub = spec.subgridspec(1, 4, wspace=0.62)
    g = cellmeans(D, lam); axes = []
    for j, (m, cmap, lab, vmax) in enumerate(MAPS):
        ax = fig.add_subplot(sub[0, j]); axes.append(ax)
        pc = fs.hexmap(ax, g.phi.values, np.log10(g.cost.values), g[m].values, DX, DY, cmap, 0, vmax, hatch_mask=g.dead.values)
        hex_axes(ax, j == 0, True)
        cax = ax.inset_axes([1.05, 0.0, 0.06, 1.0]); cb = fig.colorbar(pc, cax=cax)
        if m == "frac_trapped":
            cb.set_ticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
        cb.ax.tick_params(labelsize=5.0)
        if titles:
            ax.set_title(MAP_TITLES[m], fontsize=6.2, pad=5)
        if letters:
            fs.panel_label(ax, letters[j], x=-0.55 if j == 0 else -0.25, y=1.08)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    return axes


def row_labels(fig, rows, gap=0.012):
    """Rotated row labels just left of each row's y-axis label; rows = [(first axes of the row, text)]. Call last."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    for ax, text in rows:
        bb = inv.transform(ax.yaxis.label.get_window_extent(r)); y = ax.get_position()
        fig.text(bb[0, 0] - gap, (y.y0 + y.y1) / 2, text, rotation=90, ha="right", va="center", fontsize=6.4,
                 fontweight="bold")


def theory_panels(fig, axs, D, net, cache, lams=(1.0, 2.0, 4.0), letters="EF"):
    """Not checking and trapped against c/g (cells with phi = 0.2-0.5 where items circulate): simulation (markers with
    95% CI error bars, see sim_curve; mostly hidden by the markers) and the one-agent model (lines; nothing fitted).
    Returns legend handles/labels."""
    import config as C, engine as E
    have = sorted(D.lam.unique())
    gnet = C.graph(net) if net not in (None, "hk") else E.holme_kim(1000, 3, 0.5, 1)
    th = theory(D, np.diff(gnet[0]), have, cache)
    cols = fs.blues(3)
    for k, (ax, (m, lab)) in enumerate(zip(axs, [("frac_locked", "share not checking"), ("frac_trapped", "share trapped")])):
        fs.panel_label(ax, letters[k], x=-0.3, y=1.04)
        for i, lam in enumerate(lams):
            if lam not in have:
                continue
            sc = sim_curve(D, lam, m); T = th[lam]
            ax.plot(T[:, 0], T[:, 1 + k], color=cols[i], lw=2.4, alpha=0.35, solid_capstyle="round", zorder=1)
            ax.errorbar(sc.index, sc["mean"], yerr=sc["hw"], fmt="none", ecolor=cols[i], elinewidth=0.7, capsize=0, zorder=2)
            ax.plot(sc.index, sc["mean"], ls="none", marker=fs.MARKERS[i], ms=2.8, color=cols[i], mec="white",
                    mew=0.3, label=f"ℓ/g = {lam:g}", zorder=3)
        ax.set_xscale("log"); ax.set_xlim(8e-4, 1.25); ax.set_xticks([0.001, 0.01, 0.1, 1])
        ax.set_xticklabels(["0.001", "0.01", "0.1", "1"]); ax.minorticks_off()
        ax.set_ylim(0, 1.0 if k == 0 else 0.25); ax.set_ylabel(lab); ax.set_xlabel("relative cost c/g")
        ax.set_title("φ = 0.2–0.5", fontsize=6.2)
    h, l = axs[0].get_legend_handles_labels()
    return h + [Line2D([], [], color=fs.INK2, lw=2.4, alpha=0.35)], [f"{x}: simulation" for x in l] + ["theory (one-agent model)"]


def make_main(parquet, out, net, main_lam=2.0, cache=None, band=(0.19, 0.51)):
    """Main-text Fig. 4: regimes and consequences, l/g = 2. (A-D) maps: split of trust, share not checking, share
    trapped, and how much further genuine items travel than fakes; (E) that reach gap against the share not checking
    (cells with phi = 0.2-0.5 in which items circulate; colour = c/g); (F) welfare (Sen index, checking costs included)
    and (G) the believed share of fakes (mean over agents at the end of the run; dotted: believed = phi), against the
    share of fakes for four relative costs. The theory-vs-simulation curves are in the SI (make_ratios)."""
    from matplotlib.colors import LinearSegmentedColormap
    import v3_fig4_consequences as cons
    D = read_phase(parquet)
    fig = plt.figure(figsize=(fs.W2, 3.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.05], hspace=0.42)
    axes = map_row(fig, outer[0], D, main_lam, letters="ABCD")
    bot = outer[1].subgridspec(1, 3, wspace=0.5)
    Dm = D[np.isclose(D.lam, main_lam)].copy(); Dm["believed_fake"] = 1 - Dm.mean_pi
    # E: reach gap against the share not checking
    ax = fig.add_subplot(bot[0, 0]); fs.panel_label(ax, "E", x=-0.3, y=1.04)
    Pb = Dm[Dm.phi.between(*band)]
    P = Pb.groupby(["phi", "cost"])[["delta", "reach_true"]].mean().join(
        fs.circulating(Pb).groupby(["phi", "cost"])[["frac_locked"]].mean()).reset_index()   # reach: all runs; behaviour: circulating
    P = P[P.reach_true >= fs.MIN_REACH]
    cmap = LinearSegmentedColormap.from_list("cost", [fs.CM_LOCK(x) for x in np.linspace(0.35, 1.0, 8)])
    sc = ax.scatter(P.frac_locked, P.delta, c=np.log10(P.cost), s=5, cmap=cmap, vmin=-3, vmax=0, lw=0)
    cax = ax.inset_axes([0.08, 0.13, 0.44, 0.045]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([-3, -2, -1, 0]); cb.set_ticklabels(["0.001", "0.01", "0.1", "1"]); cax.set_title("c/g", fontsize=5.6, pad=2)
    cb.ax.tick_params(labelsize=5, pad=1.5)
    r = np.corrcoef(P.frac_locked, P.delta)[0, 1]
    ax.text(0.95, 0.97, f"φ = {band[0] + 0.01:.1f}–{band[1] - 0.01:.1f}\nr = {r:.2f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=5.4)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("share not checking"); ax.set_ylabel("reach of genuine items\nminus reach of fakes")
    # F: welfare; G: believed share of fakes
    ax = fig.add_subplot(bot[0, 1]); fs.panel_label(ax, "F", x=-0.3, y=1.04)
    cons.cost_lines(ax, Dm, "welfare_sen"); ax.set_ylim(-0.05, 1.0); ax.set_ylabel("welfare (Sen index)")
    h_c, l_c = ax.get_legend_handles_labels()
    ax = fig.add_subplot(bot[0, 2]); fs.panel_label(ax, "G", x=-0.3, y=1.04)
    cons.cost_lines(ax, Dm, "believed_fake"); ax.set_ylim(0, 1.0); ax.set_ylabel("believed share of fakes")
    ax.plot([0, 1], [0, 1], color=fs.INK2, lw=0.6, ls=":")
    hs = [Patch(facecolor="#e4e2dc", edgecolor="#9a978f", hatch="//////")] + h_c
    ls = ["nothing circulates (A–D)"] + [x + (" (F, G)" if i == 0 else "") for i, x in enumerate(l_c)]
    fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=5, fontsize=5.6, frameon=False,
               handlelength=2.4, columnspacing=1.4)
    row_labels(fig, [(axes[0], f"ℓ/g = {main_lam:g}")])
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


def make_ratios(parquet, out, lams=(1.0, 4.0), net="twitter", cache=None):
    """SI figure: the maps of Fig. 4A-D for the other loss/gain ratios (rows), and (bottom row) not checking and trapped
    against c/g for the three ratios: simulation vs the one-agent model (cells with phi = 0.2-0.5)."""
    D = read_phase(parquet)
    fig = plt.figure(figsize=(fs.W2, 1.75 * len(lams) + 2.1))
    outer = fig.add_gridspec(len(lams) + 1, 1, hspace=0.42, height_ratios=[1.0] * len(lams) + [1.1])
    rows = []
    for i, lam in enumerate(lams):
        axes = map_row(fig, outer[i], D, lam, letters="ABCDEFGH"[4 * i:4 * i + 4], titles=(i == 0))
        rows.append((axes[0], f"ℓ/g = {lam:g}"))
    bsub = outer[-1].subgridspec(1, 3, wspace=0.55, width_ratios=[1, 1, 0.9])
    axs = [fig.add_subplot(bsub[0, 0]), fig.add_subplot(bsub[0, 1])]
    k0 = 4 * len(lams)
    h, l = theory_panels(fig, axs, D, net, cache, letters="ABCDEFGHIJKL"[k0:k0 + 2])
    lax = fig.add_subplot(bsub[0, 2]); lax.axis("off")
    hs = h + [Patch(facecolor="#e4e2dc", edgecolor="#9a978f", hatch="//////")]
    lax.legend(hs, l + ["nothing circulates (maps)"], loc="center left", fontsize=5.8, frameon=False, handlelength=2.0)
    row_labels(fig, rows)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    a = sys.argv
    make_main(a[1], a[2], a[3], cache=a[4] if len(a) > 4 else None)
