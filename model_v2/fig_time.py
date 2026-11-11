"""Figure: the trust split forms early and then freezes (time evolution of trust in the checker)."""
import sys, os, pickle, numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()


def ci95(a, axis=0):
    a = np.asarray(a, float)
    m = np.nanmean(a, axis); s = np.nanstd(a, axis, ddof=1) / np.sqrt(np.sum(~np.isnan(a), axis))
    return m, m - 1.96 * s, m + 1.96 * s


def make(main_pkl, other_pkls, out, Jtrue=0.70, J0=1 / 3):
    d = pickle.load(open(main_pkl, "rb")); runs = d["runs"]; bins = d["bins"]
    t = runs[0]["times"].astype(float)
    H = np.array([r["hist"] for r in runs]).sum(0)            # (S, 3, B) pooled over runs
    tot = H.sum((1, 2), keepdims=True)
    fig = plt.figure(figsize=(fs.W2, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], hspace=0.55, wspace=0.42)

    # A: density over time (log time axis), all agents pooled
    ax = fig.add_subplot(gs[:, 0]); fs.panel_label(ax, "A", x=-0.16, y=1.01)
    dens = (H.sum(1) / tot[:, 0, :]).T                          # (B, S)
    tt = np.maximum(t, 1)
    cmap = LinearSegmentedColormap.from_list("w2k", ["#ffffff", "#c9d8ee", "#6d9bd6", "#23508f", "#0c203f"])
    te = np.r_[tt[0] * 0.9, np.sqrt(tt[1:] * tt[:-1]), tt[-1] * 1.05]
    pc = ax.pcolormesh(te, bins, np.sqrt(dens), cmap=cmap, shading="flat", rasterized=True)
    mg = np.nanmean(np.array([r["mean_by_group"] for r in runs]), 0)
    gsize = H.sum(2) / H.sum((1, 2))[:, None]                     # pooled share of agents in each group
    mg[gsize < 0.05] = np.nan                                        # hide group means while a group is (almost) empty
    cols = [fs.GREY, fs.ORANGE, fs.BLUE]; labs = ["never checked", "checked, then locked out", "still checks"]
    for g in range(3):
        ok = ~np.isnan(mg[:, g])
        ax.plot(tt[ok], mg[ok, g], color=cols[g], lw=1.4, label=labs[g])
    ax.axhline(Jtrue, color=fs.GREEN, lw=0.8, ls="--"); ax.text(tt[-1] * 0.9, Jtrue + 0.03, "checker's actual J", color=fs.GREEN, ha="right", fontsize=6)
    ax.set_xscale("log"); ax.set_xlim(10, tt[-1]); ax.set_ylim(-0.3, 1.0)
    ax.set_xlabel("items circulated (cascades)"); ax.set_ylabel("trust in the checker  Ĵ = t̂ − f̂")
    ax.legend(loc="lower left", fontsize=5.6)
    cax = ax.inset_axes([0.06, 0.93, 0.38, 0.025])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal"); cb.set_label("share of agents", fontsize=5.5, labelpad=1)
    cb.set_ticks(np.sqrt([0.02, 0.2, 0.6])); cb.set_ticklabels(["0.02", "0.2", "0.6"]); cb.ax.tick_params(labelsize=5, length=1.5, pad=1)

    # B: shares over time with 95% CI
    ax = fig.add_subplot(gs[0, 1]); fs.panel_label(ax, "B")
    never = [1 - r["ever"] for r in runs]; burned = [r["burned"] for r in runs]; active = [r["ever"] - r["burned"] for r in runs]
    for arr, col, lab, up in [(never, fs.GREY, "never\nchecked", False), (burned, fs.ORANGE, "locked out", True), (active, fs.BLUE, "still\nchecks", True)]:
        m, lo, hi = ci95(arr)
        ax.fill_between(tt, lo, hi, color=col, alpha=0.25, lw=0); ax.plot(tt, m, color=col, label=lab)
        ax.text(tt[-1] * 0.97, (hi[-1] + 0.025) if up else (lo[-1] - 0.025), lab, color=col, ha="right", va="bottom" if up else "top",
                fontsize=5.6, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(10, tt[-1]); ax.set_ylim(0, 1.02); ax.set_ylabel("share of agents")
    ax.set_xlabel("items circulated")

    # C: bifurcation index over time, several costs
    ax = fig.add_subplot(gs[0, 2]); fs.panel_label(ax, "C")
    allp = [(main_pkl, None)] + [(p, None) for p in other_pkls]
    shades = ["#0c203f", fs.BLUE, "#8fb5e8", "#c9d8ee"]
    for (p, _), col in zip(sorted(set(allp), key=lambda x: pickle.load(open(x[0], "rb"))["cost"]), shades):
        dd = pickle.load(open(p, "rb")); rr = dd["runs"]
        m, lo, hi = ci95([r["dip4"] for r in rr])
        ax.fill_between(tt, lo, hi, color=col, alpha=0.25, lw=0); ax.plot(tt, m, color=col, label=f"c/g = {dd['cost']:.2g}")
    ax.set_xscale("log"); ax.set_xlim(10, tt[-1]); ax.set_ylim(0, None); ax.set_ylabel("bifurcation (4 × dip)"); ax.set_xlabel("items circulated")
    ax.legend(fontsize=5.6, loc="upper left")

    # D: final distribution by group (stacked), share of agents
    ax = fig.add_subplot(gs[1, 1:]); fs.panel_label(ax, "D", x=-0.08)
    Hend = H[-1] / H[-1].sum()
    w = np.diff(bins); left = bins[:-1]; bottom = np.zeros(len(left))
    for g in range(3):
        ax.bar(left, 100 * Hend[g], width=w, bottom=100 * bottom, align="edge", color=cols[g], edgecolor="white", linewidth=0.3, label=labs[g])
        bottom += Hend[g]
    ax.axvline(Jtrue, color=fs.GREEN, lw=0.8, ls="--"); ax.axvline(J0, color=fs.INK2, lw=0.6, ls=":")
    ax.text(J0 + 0.03, ax.get_ylim()[1] * 0.95, "shared\nstarting trust", fontsize=5.8, color=fs.INK2, va="top")
    ax.set_xlim(-0.3, 1.0); ax.set_xlabel("trust in the checker at the end (Ĵ)"); ax.set_ylabel("% of agents")
    ax.legend(fontsize=5.8, loc="upper left")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2:-1], sys.argv[-1])
