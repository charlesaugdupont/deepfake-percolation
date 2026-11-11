"""Figure: where does trust in the checker split? Heat maps over the share of fakes and the relative checking payoff,
for three loss/gain ratios; bifurcation (4 x dip) and share locked out; plus first-passage prediction of the locked-out
share (averaged over the degree distribution) against simulation."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

CM_DIP = LinearSegmentedColormap.from_list("dip", ["#fbfaf7", "#f6d7c3", "#ec9a6c", "#d4572a", "#8f2a12", "#4a1406"])
CM_LOCK = LinearSegmentedColormap.from_list("lock", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])


def grid(D, lam, m):
    d = D[D.lam == lam].groupby(["phi", "cost"])[m].mean().reset_index()
    phis = np.sort(d.phi.unique()); costs = np.sort(d.cost.unique())
    Z = d.pivot(index="cost", columns="phi", values=m).loc[costs, phis].values
    dp = phis[1] - phis[0]; lc = np.log10(costs); dl = lc[1] - lc[0]
    return np.r_[phis - dp / 2, phis[-1] + dp / 2], np.r_[lc - dl / 2, lc[-1] + dl / 2], phis, lc, Z


def dead_cells(D, lam, min_reach=0.05):
    """Cells where nothing circulates: mean reach of genuine items below min_reach (nobody shares or checks)."""
    d = D[D.lam == lam].groupby(["phi", "cost"]).reach_true.mean()
    return set(d[d < min_reach].index)


def hatch_dead(ax, D, lam, pe, ce, phis, lc):
    mpl.rcParams["hatch.linewidth"] = 0.35
    dead = dead_cells(D, lam)
    costs = np.sort(D[D.lam == lam].cost.unique())
    for (phi, c) in dead:
        j = int(np.argmin(np.abs(phis - phi))); i = int(np.argmin(np.abs(costs - c)))
        ax.add_patch(Rectangle((pe[j], ce[i]), pe[j + 1] - pe[j], ce[i + 1] - ce[i], fc="#e4e2dc", ec="#9a978f", hatch="//////", lw=0))
    return dead


def make(phase_parquet, out, J0=1 / 3, label=None, theory=None):
    D = pd.read_parquet(phase_parquet)
    lams = sorted(D.lam.unique())
    fig = plt.figure(figsize=(fs.W2, 4.4))
    gs = fig.add_gridspec(2, len(lams) + 3, width_ratios=[1] * len(lams) + [0.07, 0.62, 1.15], hspace=0.38, wspace=0.12)
    for i, (m, cmap, lab, vmax) in enumerate([("dip4", CM_DIP, "bifurcation of trust (4 × dip)", 0.7),
                                              ("frac_locked", CM_LOCK, "share locked out (Ĵ ≤ θ)", 1.0)]):
        for j, lam in enumerate(lams):
            ax = fig.add_subplot(gs[i, j])
            pe, ce, phis, lc, Z = grid(D, lam, m)
            pc = ax.pcolormesh(pe, ce, Z, cmap=cmap, vmin=0, vmax=vmax, rasterized=True)
            dead = hatch_dead(ax, D, lam, pe, ce, phis, lc)
            if m == "dip4":
                ax.contour(phis, lc, Z, levels=[0.2], colors=[fs.INK], linewidths=0.6)
            if i == 0 and len(dead) > 30:
                dp = np.array([p for p, _ in dead])
                ax.text(0.5 * (dp.min() + pe[-1]), np.log10(0.006), "nothing\ncirculates:\nno one shares\nor checks", ha="center",
                        va="center", fontsize=5.0, color=fs.INK, bbox=dict(fc="#e4e2dc", ec="none", pad=1.0))
            # average agent's threshold equals the shared starting trust: c/g = J0 / (1 + g/l)
            ax.axhline(np.log10(J0 / (1 + 1 / lam)), color=fs.INK2, lw=0.6, ls="--")
            ax.set_yticks([-3, -2, -1, 0]); ax.set_yticklabels(["0.001", "0.01", "0.1", "1"]); ax.set_ylim(ce[0], ce[-1])
            ax.set_xticks([0.1, 0.5, 0.9])
            if i == 0: ax.set_title(f"loss / gain  l/g = {lam:g}", fontsize=6.8)
            if i == 0 and j == 0:
                for xx, yy, txt in [(0.42, ce[-1] - 0.35, "few can afford to check"), (0.5, ce[0] + 0.25, "almost all check and learn")]:
                    ax.text(xx, yy, txt, ha="center", va="center", fontsize=5.2, color=fs.INK)
            if j == 0: ax.set_ylabel("checking cost / gain  c/g")
            else: ax.set_yticklabels([])
            if i == 1: ax.set_xlabel("share of fakes φ")
            else: ax.set_xticklabels([])
            if j == 0: fs.panel_label(ax, "AB"[i], x=-0.42, y=1.02)
        cax = fig.add_subplot(gs[i, len(lams)])
        cb = fig.colorbar(pc, cax=cax); cb.ax.tick_params(labelsize=5.5); cb.set_label(lab, fontsize=6)
    # C: locked-out share vs cost: simulation (mean over phi in [0.2, 0.5]) vs first-passage prediction
    ax = fig.add_subplot(gs[:, len(lams) + 2]); fs.panel_label(ax, "C", x=-0.3, y=1.01)
    cols = {1.0: "#9cc2ef", 2.0: fs.BLUE, 4.0: "#0b2e5c"}
    for lam in lams:
        sub = D[(D.lam == lam) & D.phi.between(0.19, 0.51)]
        dead = dead_cells(D, lam)
        sub = sub[[(p, c) not in dead for p, c in zip(sub.phi, sub.cost)]]          # skip cells where nothing circulates
        per = sub.groupby(["cost", "rep"]).frac_locked.mean().reset_index()
        g = per.groupby("cost").frac_locked.agg(["mean", "sem"])
        ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], fmt="o", ms=2.4, color=cols.get(lam, fs.GREY), capsize=1, lw=0.8,
                    label=f"l/g = {lam:g}")
        if theory is not None and lam in theory:
            cc, pr = theory[lam]
            ax.plot(cc, pr, color=cols.get(lam, fs.GREY), lw=1)
    if theory is not None:
        ax.plot([], [], color=fs.INK2, lw=1, label="first-passage model")
    ax.set_xscale("log"); ax.set_xlabel("checking cost / gain  c/g"); ax.set_ylabel("share locked out (φ = 0.2–0.5)", labelpad=1)
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=5.6, loc="upper left")
    if label:
        fig.suptitle(label, fontsize=7, y=0.995)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


def theory_curves(D, deg, lams, t=0.80, f=0.10, n_mc=3000):
    """First-passage prediction of the locked-out share (phi in [0.2, 0.5]) for each loss/gain ratio and cost."""
    from trapmodel import predicted_locked
    out = {}
    for lam in lams:
        sub = D[(D.lam == lam) & D.phi.between(0.19, 0.51)]
        dead = dead_cells(D, lam)
        sub = sub[[(p, c) not in dead for p, c in zip(sub.phi, sub.cost)]]
        cs = np.sort(sub.cost.unique())
        qs = sub.groupby("cost").q_fake_checks.mean() if "q_fake_checks" in sub else pd.Series(0.25, index=cs)
        pr = [predicted_locked(deg, c, lam, q=float(np.nan_to_num(qs.get(c, 0.25), nan=0.25)), t=t, f=f, n_mc=n_mc) for c in cs]
        out[lam] = (cs, np.array(pr))
    return out


if __name__ == "__main__":
    import engine as E, config as C
    D = pd.read_parquet(sys.argv[1]); net = sys.argv[3] if len(sys.argv) > 3 else None
    g = C.graph(net) if net not in (None, "hk") else E.holme_kim(1000, 3, 0.5, 1)
    deg = np.diff(g[0])
    th = theory_curves(D, deg, sorted(D.lam.unique()))
    make(sys.argv[1], sys.argv[2], theory=th)
