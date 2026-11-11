"""SI figure: the starting belief about the share of genuine items. Three starts, seven settings:
everyone at 0.5 (as in model v2), everyone at the true share 1 - phi, and spread out, U(0.01, 0.99) (used throughout).
(A) items until the first check (the 'cold start'), (B) share who ever checked, (C) share not checking, (D) split of
trust (Hartigan's dip, 0-0.25; the runs store 4 x dip), all at the end of the run. Not in the current SI (the
comparison is reported in the text). Input: <data>/baserate_twitter.parquet."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

SETS = [(2.0, 0.3, 0.01), (2.0, 0.3, 0.05), (2.0, 0.3, 0.2), (2.0, 0.55, 0.05), (2.0, 0.75, 0.05), (4.0, 0.25, 0.01),
        (4.0, 0.35, 0.01)]
PRIORS = [("spread", "spread out, U(0.01, 0.99) (used here)", fs.BLUE, "o"),
          ("uninformed", "everyone 0.5", fs.ORANGE, "s"),
          ("informed", "everyone correct (1 − φ)", fs.AQUA, "^")]


def draw(fig, spec, D, letters="ABCD"):
    """Draw the four panels into the SubplotSpec `spec`; returns the axes and the legend handles and labels."""
    D = D.copy(); D["t1"] = np.maximum(D.t_first_check, 1); D["dip"] = D.dip4 / 4
    sub = spec.subgridspec(1, 4, wspace=0.18)
    axs = [fig.add_subplot(sub[0, 0])]
    axs += [fig.add_subplot(sub[0, i], sharey=axs[0]) for i in range(1, 4)]
    y0 = np.arange(len(SETS))[::-1]
    off = {"spread": 0.22, "uninformed": 0.0, "informed": -0.22}
    panels = [("t1", "items until the first check", "log"), ("frac_ever", "share who ever checked", None),
              ("frac_locked", "share not checking", None), ("dip", "split of trust (dip)", "dip")]
    for k, (ax, (col, lab, sc)) in enumerate(zip(axs, panels)):
        fs.panel_label(ax, letters[k], x=-0.08 if k else -1.05, y=1.03)
        if k:
            ax.tick_params(labelleft=False)
        for y, (lam, phi, c) in zip(y0, SETS):
            ax.axhspan(y - 0.45, y + 0.45, color="#f4f3f0" if (y % 2 == 0) else "white", lw=0, zorder=0)
            for name, plab, pc, mk in PRIORS:
                sub = D[(D.prior == name) & np.isclose(D.lam, lam) & np.isclose(D.phi, phi) & np.isclose(D.cost, c)]
                v = sub[col].values
                dead = sub.frac_ever.mean() < 0.01
                ax.plot([v.min(), v.max()], [y + off[name]] * 2, color=pc, lw=0.8, alpha=0.7)
                ax.plot(v.mean(), y + off[name], ls="none", marker=mk, ms=3.4, mec=pc if dead else "white",
                        mfc="white" if dead else pc, mew=0.7 if dead else 0.3, label=plab if y == y0[0] else None, zorder=4)
        if sc == "log":
            ax.set_xscale("log"); ax.set_xlim(0.7, 3000); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["≤1", "10", "100", "1,000"])
            ax.minorticks_off()
        elif sc == "dip":
            ax.set_xlim(-0.008, 0.25)
        else:
            ax.set_xlim(-0.03, 1.0)
        ax.set_xlabel(lab, fontsize=6.3)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(y0)
    axs[0].set_yticklabels([f"l/g = {lam:g}, φ = {phi:g}, c/g = {c:g}" for lam, phi, c in SETS], fontsize=5.8)
    axs[0].set_ylim(-0.6, len(SETS) - 0.4)
    yd = y0[SETS.index((2.0, 0.75, 0.05))]
    axs[1].text(0.05, yd - 0.22, "nothing circulates", fontsize=5.0, va="center", color=fs.INK)
    return axs, axs[0].get_legend_handles_labels()


def draw2(fig, spec, D, letters="DE"):
    """Two-panel summary: (first) how soon checking starts under each starting belief (one dot per setting, mean of 3
    runs; bar: median over settings); (second) the end state under the two other starts against the spread-out start,
    for three measures (on the diagonal = unchanged). Returns the axes and the legend handles and labels."""
    from matplotlib.lines import Line2D
    D = D.copy(); D["t1"] = np.maximum(D.t_first_check, 1); D["dip"] = D.dip4 / 4
    sub = spec.subgridspec(1, 2, wspace=0.45, width_ratios=[1.0, 1.0])
    ax = fig.add_subplot(sub[0, 0]); fs.panel_label(ax, letters[0], x=-0.32, y=1.04)
    rng = np.random.default_rng(3)
    for k, (name, plab, pc, mk) in enumerate(PRIORS):
        v = np.array([D[(D.prior == name) & np.isclose(D.lam, lam) & np.isclose(D.phi, phi) & np.isclose(D.cost, c)].t1.mean()
                      for lam, phi, c in SETS])
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=pc, marker=mk, edgecolor="white", lw=0.3, zorder=3)
        ax.plot([k - 0.28, k + 0.28], [np.median(v)] * 2, color=fs.INK, lw=1.0, zorder=4)
    ax.set_yscale("log"); ax.set_ylim(0.7, 3000); ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(["≤1", "10", "100", "1,000"])
    ax.minorticks_off(); ax.set_xticks(range(len(PRIORS)))
    ax.set_xticklabels(["spread out\n(used here)", "everyone\n0.5", "everyone\ncorrect"], fontsize=5.6)
    ax.set_xlim(-0.5, len(PRIORS) - 0.5); ax.set_ylabel("items until the first check")
    ax.set_title("how soon checking starts", fontsize=6.3)
    ax = fig.add_subplot(sub[0, 1]); fs.panel_label(ax, letters[1], x=-0.32, y=1.04)
    measures = [("frac_ever", "share who ever checked", "o"), ("frac_locked", "share not checking", "s"),
                ("dip", "split of trust (dip, 0–0.25)", "^")]
    base = {}
    for lam, phi, c in SETS:
        b = D[(D.prior == "spread") & np.isclose(D.lam, lam) & np.isclose(D.phi, phi) & np.isclose(D.cost, c)]
        base[(lam, phi, c)] = b
    for name, plab, pc, _ in PRIORS[1:]:
        for m, mlab, mk in measures:
            xs, ys, dead = [], [], []
            for key in SETS:
                lam, phi, c = key
                o = D[(D.prior == name) & np.isclose(D.lam, lam) & np.isclose(D.phi, phi) & np.isclose(D.cost, c)]
                xs.append(base[key][m].mean()); ys.append(o[m].mean()); dead.append(o.frac_ever.mean() < 0.01)
            xs, ys, dead = map(np.array, (xs, ys, dead))
            ax.scatter(xs[~dead], ys[~dead], s=10, color=pc, marker=mk, edgecolor="white", lw=0.3, zorder=3)
            ax.scatter(xs[dead], ys[dead], s=12, facecolor="white", edgecolor=pc, marker=mk, lw=0.7, zorder=3)
    ax.plot([0, 1], [0, 1], color=fs.INK2, lw=0.6, ls=":", zorder=1)
    ax.set_xlim(-0.03, 1.0); ax.set_ylim(-0.03, 1.0); ax.set_aspect("equal")
    ax.set_xlabel("end state, spread-out start"); ax.set_ylabel("end state, other start")
    ax.set_title("where checking ends up", fontsize=6.3)
    hs = [Line2D([], [], ls="none", marker="o", color=pc, ms=3.4) for _, _, pc, _ in PRIORS[1:]] + \
         [Line2D([], [], ls="none", marker=mk, color=fs.INK2, ms=3.4) for _, _, mk in measures] + \
         [Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.INK2, ms=3.4)]
    ls = [plab for _, plab, _, _ in PRIORS[1:]] + [mlab for _, mlab, _ in measures] + ["nothing circulates"]
    return [fig.axes[-2], ax], (hs, ls)


def make(parquet, out):
    fig = plt.figure(figsize=(fs.W2, 2.5))
    axs, (h, l) = draw(fig, fig.add_gridspec(1, 1)[0], pd.read_parquet(parquet))
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.55, 0.98), ncol=3, fontsize=5.8, frameon=False,
               title="starting belief about the share of genuine items (dots: mean of 3 runs; lines: range)",
               title_fontsize=5.8, columnspacing=1.4)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
