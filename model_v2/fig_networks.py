"""SI figure: the split on other networks (loss/gain = 2). Bifurcation heat maps and locked-out share vs cost with
the first-passage prediction for each network's degree distribution."""
import sys, os, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import config as C, engine as E
from trapmodel import predicted_locked
from fig_phase import CM_DIP, grid, hatch_dead, dead_cells
fs.setup()

NETS = [("twitter", "out/phase_twitter.parquet", "Twitter news sharers\n(main, N = 14,128)"),
        ("twitter_rewired", "out/phase_twitter_rewired.parquet", "Twitter, degree-preserving\nrewired"),
        ("fb_jhu", "out/phase_fb_jhu.parquet", "Facebook100 Johns Hopkins\n(N = 5,157)"),
        ("twitch", "out/phase_twitch_lam2.parquet", "Twitch EN\n(N = 7,126)"),
        ("hk", "out/phase_hk.parquet", "Holme–Kim\n(N = 1,000, m = 3)")]
DATA = os.environ.get("DFP_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
NETS = [(n, os.path.join(DATA, os.path.basename(p)), lab) for n, p, lab in NETS]
COLS = ["#0b2e5c", "#7b5cc4", "#1a9e6e", "#d9822b", "#9a9893"]


def make(out):
    fig = plt.figure(figsize=(fs.W2, 3.9))
    gs = fig.add_gridspec(2, len(NETS), height_ratios=[1, 1.1], hspace=0.6, wspace=0.35)
    axc = fig.add_subplot(gs[1, :3]); fs.panel_label(axc, "B", x=-0.07)
    for j, ((net, path, lab), col) in enumerate(zip(NETS, COLS)):
        D = pd.read_parquet(path); D = D[D.lam == 2.0]
        ax = fig.add_subplot(gs[0, j])
        pe, ce, phis, lc, Z = grid(D, 2.0, "dip4")
        ax.set_facecolor("#ecebe7")                                  # grey = not simulated
        pc = ax.pcolormesh(pe, ce, Z, cmap=CM_DIP, vmin=0, vmax=0.7, rasterized=True)
        hatch_dead(ax, D, 2.0, pe, ce, phis, lc)                     # hatched = nothing circulates
        ax.set_title(lab, fontsize=5.8); ax.set_xticks([0.1, 0.5, 0.9]); ax.set_xlabel("φ")
        ax.set_yticks([-3, -2, -1, 0]); ax.set_yticklabels(["0.001", "0.01", "0.1", "1"] if j == 0 else []); ax.set_ylim(-3.1, 0.1)
        if j == 0: ax.set_ylabel("c/g"); fs.panel_label(ax, "A", x=-0.6)
        dead = dead_cells(D, 2.0)
        sub = D[D.phi.between(0.19, 0.51)]
        sub = sub[[(p, c) not in dead for p, c in zip(sub.phi, sub.cost)]]
        per = sub.groupby(["cost", "rep"]).frac_locked.mean().reset_index()
        g = per.groupby("cost").frac_locked.agg(["mean", "sem"])
        axc.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], fmt="o", ms=2.2, color=col, capsize=1, lw=0.7, label=lab.replace("\n", " "))
        gr = C.graph(net) if net != "hk" else E.holme_kim(1000, 3, 0.5, 1)
        deg = np.diff(gr[0]); cs = np.sort(D.cost.unique())
        cs = np.sort(sub.cost.unique())
        qs = sub.groupby("cost").q_fake_checks.mean() if "q_fake_checks" in sub else pd.Series(0.25, index=cs)
        pr = [predicted_locked(deg, c, 2.0, q=float(np.nan_to_num(qs.get(c, 0.25), nan=0.25)), n_mc=2000) for c in cs]
        axc.plot(cs, pr, color=col, lw=0.8, ls="--")
    cax = fig.add_axes([0.92, 0.6, 0.008, 0.28]); cb = fig.colorbar(pc, cax=cax); cb.ax.tick_params(labelsize=5); cax.set_title("4·dip", fontsize=5.5)
    axc.plot([], [], color=fs.INK2, lw=0.8, ls="--", label="first-passage prediction")
    axc.set_xscale("log"); axc.set_xlabel("checking cost / gain  c/g"); axc.set_ylabel("share locked out (φ = 0.2–0.5)")
    axc.legend(fontsize=5.2, loc="upper left", bbox_to_anchor=(1.02, 1.0)); axc.set_ylim(-0.02, 1.02)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1])
