"""v3 Figure 1: model and mechanism. (A) schematic, (B) the checking threshold against the number of neighbours, with the
checker's actual quality and everyone's starting trust: priced out / trapped from the start / start checking,
(C) 60 identical agents (theta = 0.2): belief trajectories, some hit the absorbing barrier (trapped later),
(D) share not checking by degree for c/g = 0.01, 0.02, 0.05: simulation vs theory (one-agent model).
regions() (decision regions over the item estimate and the trust in the checker) is kept for the SI model description.
Input: <data>/mech_twitter_agents.pkl (list of (labels, per-agent arrays))."""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from trapmodel import p_trap, paths
fs.setup()

G, L = 1.0, 2.0
T_TRUE, F_TRUE = 0.80, 0.10
PRIOR = (2.0, 1.0, 1.0, 2.0)


BOXES = []


def box(ax, x, y, w, h, text, fc, ec=None, fs_=6.2, bold=False):
    pa = ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec or fc, lw=0.8))
    tx = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, fontweight="bold" if bold else None, color=fs.INK)
    BOXES.append((pa, tx))


def check_boxes(fig, margin=0.88):
    """Warn if any schematic text uses more than `margin` of its box (PDF viewers render text slightly wider)."""
    r = fig.canvas.get_renderer()
    for pa, tx in BOXES:
        bp, bt = pa.get_window_extent(r), tx.get_window_extent(r)
        if bt.width > margin * bp.width or bt.height > margin * bp.height:
            print(f"warning: text fills {bt.width / bp.width:.0%} x {bt.height / bp.height:.0%} of its box: {tx.get_text()[:40]!r}")


def arrow(ax, x0, y0, x1, y1, col=None, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col or fs.INK2, ls=ls))


def schematic(ax):
    """Wide schematic: network cascade -> decision -> learning -> absorbing barrier. Axes aspect is equal, x in [0, 4.4]."""
    ax.set_xlim(0, 4.45); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    pos = np.array([[0.10, 0.55], [0.30, 0.85], [0.35, 0.50], [0.12, 0.20], [0.58, 0.80], [0.62, 0.42], [0.40, 0.15], [0.82, 0.62], [0.80, 0.18]])
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (2, 5), (3, 6), (2, 6), (5, 7), (4, 7), (5, 8), (6, 8)]
    shared = {(0, 1), (0, 2), (2, 5), (2, 6), (5, 7), (5, 8)}
    for a, b in edges:
        on = (a, b) in shared
        ax.plot(*pos[[a, b]].T, color=fs.BLUE if on else "#d5d4d0", lw=1.3 if on else 0.6, zorder=1)
    st = {0: fs.BLUE, 1: fs.GREY, 2: fs.AQUA, 3: "#ffffff", 4: "#ffffff", 5: fs.BLUE, 6: fs.GREY, 7: fs.GREY, 8: fs.BLUE}
    for i, p in enumerate(pos):
        ax.add_patch(Circle(p, 0.05, fc=st[i], ec=fs.INK2, lw=0.5, zorder=2))
    ax.text(0.035, 0.55, "seed", fontsize=5.4, ha="right", va="center", color=fs.INK2)
    for x0, col, lab in [(0.03, fs.BLUE, "shared"), (0.33, fs.AQUA, "checked"), (0.65, fs.GREY, "discarded")]:
        ax.add_patch(Circle((x0, -0.04), 0.025, fc=col, ec=fs.INK2, lw=0.4, clip_on=False))
        ax.text(x0 + 0.045, -0.04, lab, fontsize=5, va="center", clip_on=False)
    # boxes (sized with generous margins so that the text also fits in PDF viewers)
    box(ax, 1.05, 0.24, 0.96, 0.62, "agent $i$ receives an item\nfrom neighbour $j$\n(fake with probability $\\varphi$)\n\n"
        "estimates $p$ = P(genuine)\nfrom $j$'s past items\nand the base rate $\\pi_i$", "#eef3fb", fs_=5.2)
    arrow(ax, 0.91, 0.55, 1.04, 0.55)
    box(ax, 2.15, 0.70, 0.82, 0.23, "share: $+g\\,s_i$ if genuine,\n$-\\ell\\,s_i$ if fake", "#d7e6f8", fs_=5.2)
    box(ax, 2.15, 0.42, 0.82, 0.23, "check: pay $c$, then share\nunless it is flagged", "#d3efe3", fs_=5.2)
    box(ax, 2.15, 0.14, 0.82, 0.23, "discard: 0", "#ecebe7", fs_=5.2)
    ax.text(2.56, 0.965, "highest expected payoff", ha="center", fontsize=5.3, color=fs.INK2)
    for y in [0.815, 0.535, 0.255]:
        arrow(ax, 2.02, 0.55, 2.14, y)
    box(ax, 3.10, 0.555, 1.30, 0.40, "afterwards the truth is revealed:\nall receivers update $\\pi_i$ and $j$'s record;\n"
        "only agents who checked learn\nabout the checker $(\\hat t_i,\\hat f_i)$", "#f6f5f2", fs_=5.0)
    box(ax, 3.10, 0.0, 1.30, 0.445, "checking can pay only if\n$\\hat J_i=\\hat t_i-\\hat f_i>\\theta_i=\\frac{c}{s_i}\\left(\\frac{1}{g}+\\frac{1}{\\ell}\\right)$\n"
        "with audience $s_i=k_i/\\langle k\\rangle$;\nbelow $\\theta_i$: no checks, no learning,\nno way back (absorbing)", "#fbe9dd", fs_=5.0)
    arrow(ax, 2.98, 0.60, 3.09, 0.72)
    arrow(ax, 3.75, 0.553, 3.75, 0.448)


def regions(ax, c=0.05):
    """Decision regions of an agent with an average audience (s = 1, filled) and the region where checking pays for an
    agent with an audience 1/6 of average (dotted outline). The lowest point of each check region is that agent's
    threshold theta: checking can pay only above it."""
    p = np.linspace(0, 1, 600)[None, :]; Jv = np.linspace(-0.2, 1.0, 600)[:, None]
    t = 0.5 + Jv / 2; f = 0.5 - Jv / 2

    def decide(s):
        us = s * (p * G - (1 - p) * L); uv = -c + s * (p * (1 - f) * G - (1 - p) * (1 - t) * L)
        return np.where(us > 0, np.where(us > uv, 2, 1), np.where(uv > 0, 1, 0))
    R = decide(1.0); R6 = decide(1 / 6)
    ax.pcolormesh(p.ravel(), Jv.ravel(), R, cmap=ListedColormap(["#ecebe7", "#d3efe3", "#d7e6f8"]), shading="auto", rasterized=True)
    ax.contour(p.ravel(), Jv.ravel(), (R6 == 1).astype(float), levels=[0.5], colors=[fs.INK], linewidths=0.9, linestyles=":")
    th1 = c * (1 / G + 1 / L); th6 = 6 * th1; ptip = L / (G + L)
    ax.axhline(1 / 3, color=fs.INK2, lw=0.6, ls="-.")
    ax.text(0.98, 1 / 3 - 0.015, "starting trust", fontsize=5.2, color=fs.INK2, ha="right", va="top")
    for th, lab, side in [(th1, "θ, average\naudience", 1), (th6, "θ, audience\n1/6 of average", -1)]:
        ax.plot([ptip], [th], marker="o", ms=2.6, color=fs.INK, zorder=5)
        ax.text(ptip + side * 0.035, th, lab, fontsize=5.0, ha="left" if side > 0 else "right", va="center", color=fs.INK)
    ax.text(0.6, 0.915, "check, audience\n1/6 of average", fontsize=4.8, ha="center", va="center", color=fs.INK)
    for x, y, lab in [(0.16, -0.12, "discard"), (0.22, 0.86, "check"), (0.88, -0.12, "share")]:
        ax.text(x, y, lab, fontsize=6.2, ha="center", va="center", color=fs.INK, fontweight="bold")
    ax.set_xlabel("agent's estimate that the item is genuine"); ax.set_ylabel("trust in the checker  Ĵ")
    ax.set_xlim(0, 1); ax.set_ylim(-0.2, 1.0)
    ax.set_title(f"filled: average audience; c/g = {c:g}", fontsize=5.8, color=fs.INK2)


def thresholds(ax, kbar, shares, c=0.05, kmax=5000, ylim=(-0.05, 1.0)):
    """Threshold theta against the number of neighbours k (c/g = c, l/g = L). Shaded: trust above theta, where checking
    pays. Everyone starts at the same trust J0 (horizontal line, coloured by what it means for each k): theta >= J,
    checking never pays even at the checker's actual quality (priced out); J0 <= theta < J, checking would pay but the
    starting trust is below theta, so the agent never tries (trapped from the start); theta < J0, the agent starts
    checking (and may still be trapped later by bad luck, panel C). `shares`: share of agents in each group."""
    J0 = 1 / 3; Jt = T_TRUE - F_TRUE; a = c * (1 / G + 1 / L)
    k = np.geomspace(1, kmax, 400); th = a * kbar / k
    k1, k2 = a * kbar / Jt, a * kbar / J0
    ax.fill_between(k, np.minimum(th, ylim[1]), ylim[1], color="#d3efe3", lw=0, zorder=0)
    ax.plot(k, th, color=fs.INK, lw=1.5, zorder=3)
    ax.axhline(Jt, color=fs.ACTUAL, lw=0.9, zorder=2)
    for lo, hi, col in [(1, k1, fs.NEVER), (k1, k2, fs.LOCKED), (k2, kmax, fs.USER)]:
        ax.plot([lo, hi], [J0, J0], color=col, lw=2.4, solid_capstyle="butt", zorder=4)
    ax.text(kmax * 0.85, Jt + 0.015, "checker's actual J", fontsize=5.2, ha="right", va="bottom", color=fs.INK)
    ax.text(kmax * 0.85, J0 - 0.025, "everyone's\nstarting trust", fontsize=5.0, ha="right", va="top", color=fs.INK2)
    ax.text(kmax * 0.85, 0.955, "trust above θ:\nchecking pays", fontsize=5.2, ha="right", va="top", color=fs.INK2,
            style="italic")
    kt = 300; ax.text(kmax * 0.85, a * kbar / kt + 0.025, "threshold θ", fontsize=5.4, ha="right", va="bottom", color=fs.INK,
                      fontweight="bold")                              # right-aligned like the other labels (6 Nov: θ clipped)
    ax.text(np.sqrt(1 * k1) * 0.9, J0 + 0.03, f"priced out\n{100 * shares[0]:.0f}%", fontsize=5.6, ha="center", va="bottom",
            color=fs.INK)
    ax.annotate(f"trapped from\nthe start {100 * shares[1]:.0f}%", xy=(np.sqrt(k1 * k2), J0 - 0.012), xytext=(6.5, 0.13),
                fontsize=5.6, ha="center", va="center", color=fs.INK,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK, shrinkA=1, shrinkB=0))
    ax.text(np.sqrt(k2 * kmax) * 1.3, J0 + 0.03, f"start checking\n{100 * shares[2]:.0f}%", fontsize=5.6, ha="center",
            va="bottom", color=fs.INK)
    ax.set_xscale("log"); ax.set_xlim(1, kmax); ax.set_ylim(*ylim)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"]); ax.minorticks_off()
    ax.set_xlabel("number of neighbours k"); ax.set_ylabel("trust in the checker  Ĵ")
    ax.set_title(f"who checks at the start (c/g = {c:g})", fontsize=5.8, color=fs.INK2)


def trajectories(ax, q, theta=0.2, n=60, k=40, kbar=252.5, c=0.05, ylim=(-0.05, 1.0)):
    """60 identical agents (same threshold theta, same starting trust 1/3) who differ only in the luck of their checks:
    one-agent model, checks drawn with the checker's actual TPR and FPR and the share of fakes among checked items q."""
    P = paths(n, k, prior=PRIOR, t=T_TRUE, f=F_TRUE, q=q, seed=11)
    x = np.arange(k + 1)
    for i in range(n):
        hit = np.where(P[i] <= theta)[0]
        if len(hit):
            h = hit[0]
            ax.plot(x[:h + 1], P[i, :h + 1], color=fs.LOCKED, lw=0.6, alpha=0.85)
            ax.plot([h, k], [P[i, h]] * 2, color=fs.LOCKED, lw=0.6, alpha=0.5, ls=":")
            ax.plot(h, P[i, h], "s", ms=2.0, color=fs.LOCKED)
        else:
            ax.plot(x, P[i], color=fs.USER, lw=0.5, alpha=0.45)
    ax.axhline(theta, color=fs.INK, lw=0.9, ls="--"); ax.text(k, theta + 0.015, "threshold θ", ha="right", va="bottom", fontsize=5.6)
    ax.axhline(T_TRUE - F_TRUE, color=fs.ACTUAL, lw=0.8)
    ax.text(0.6, T_TRUE - F_TRUE + 0.02, "checker's actual J", fontsize=5.4, ha="left", va="bottom", color=fs.INK,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85), zorder=5)
    ax.text(0.8, 0.985, "keep checking:\ntrust nears the actual J", fontsize=5.2, ha="left", va="top", color=fs.USER,
            fontweight="bold")
    ax.text(k - 0.5, -0.035, "hit θ: stop checking,\ntrust frozen (trapped)", fontsize=5.2, ha="right", va="bottom",
            color=fs.LOCKED, fontweight="bold")
    ax.set_xlabel("checks made by the agent"); ax.set_ylabel("trust in the checker  Ĵ"); ax.set_ylim(*ylim); ax.set_xlim(0, k)
    kk = c * (1 / G + 1 / L) * kbar / theta
    ax.set_title(f"{n} identical agents (θ = {theta:g})", fontsize=5.8, color=fs.INK2)


def load_runs(mech_pkl):
    out = []
    for meta, p in pickle.load(open(mech_pkl, "rb")):
        r = dict(p); r["cost"] = meta["cost"]; r["seed"] = meta["rep"]; out.append(r)
    return out


def by_degree(ax, mech_pkl, show_max=0.05):
    runs = load_runs(mech_pkl)
    rows = []
    for r in runs:
        lock = r["J"] <= r["theta"]
        rows.append(pd.DataFrame(dict(cost=r["cost"], seed=r["seed"], q=r["q"], deg=r["deg"], theta=r["theta"], locked=lock)))
    A = pd.concat(rows)
    edges = np.unique(np.round(np.geomspace(1, A.deg.max() + 1, 16))) - 0.5
    A["dbin"] = pd.cut(A.deg, edges)
    costs = [c for c in sorted(A.cost.unique()) if c <= show_max]
    cols = fs.blues(len(costs))
    for i, (c, col) in enumerate(zip(costs, cols)):
        sub = A[A.cost == c]
        per_run = sub.groupby(["dbin", "seed"], observed=True).agg(locked=("locked", "mean"), deg=("deg", "mean"), theta=("theta", "mean")).reset_index()
        g = per_run.groupby("dbin", observed=True).agg(m=("locked", "mean"), s=("locked", "sem"), deg=("deg", "mean"), theta=("theta", "mean"), n=("locked", "size"))
        g = g[g.n >= 3]
        mk = fs.MARKERS[i]
        ax.errorbar(g.deg, g.m, yerr=fs.t95(g.n.values) * g.s.fillna(0).values, fmt="none", ecolor=col, elinewidth=0.7,
                    capsize=0, zorder=2)                                    # 95% CIs over runs (Student's t), mostly hidden by markers
        ax.plot(g.deg, g.m, ls="none", marker=mk, ms=3.0, color=col, mec="white", mew=0.3, zorder=3)
        ax.plot([], [], color=col, lw=1, marker=mk, ms=2.8, label=f"{c:g}")
        degs = np.unique(np.round(np.geomspace(1, A.deg.max(), 40)))
        th = (c / (degs / (sub.deg.mean()))) * (1 / G + 1 / L)
        ax.plot(degs, p_trap(th, prior=PRIOR, t=T_TRUE, f=F_TRUE, q=sub.q.mean()), color=col, lw=1)
    ax.set_xscale("log"); ax.set_xlabel("number of neighbours k"); ax.set_ylabel("share not checking (Ĵ ≤ θ)")
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=5.2, loc="lower left", bbox_to_anchor=(0.0, 0.0), title="c/g", title_fontsize=5.4,
                                        borderaxespad=0.3)
    ax.set_title("markers: simulation\nlines: one-agent model (no fitting)", fontsize=5.4, color=fs.INK2)
    tk = [t for t in [1, 10, 100, 1000] if t <= A.deg.max() * 1.5]
    ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk]); ax.minorticks_off()
    return A


def make(mech_pkl, out):
    fig = plt.figure(figsize=(fs.W2, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.42)
    ax = fig.add_subplot(gs[0, :]); schematic(ax); fs.panel_label(ax, "A", x=0.0, y=0.98)
    runs = [r for r in load_runs(mech_pkl) if abs(r["cost"] - 0.05) < 1e-9]
    th = np.concatenate([r["theta"] for r in runs]); Jt = T_TRUE - F_TRUE
    shares = [(th >= Jt).mean(), ((th >= 1 / 3) & (th < Jt)).mean(), (th < 1 / 3).mean()]
    ax = fig.add_subplot(gs[1, 0]); thresholds(ax, float(np.mean(runs[0]["deg"])), shares); fs.panel_label(ax, "B")
    q = np.mean([r["q"] for r in runs])
    ax = fig.add_subplot(gs[1, 1]); trajectories(ax, q=q, kbar=float(np.mean(runs[0]["deg"]))); fs.panel_label(ax, "C")
    ax = fig.add_subplot(gs[1, 2]); by_degree(ax, mech_pkl); fs.panel_label(ax, "D")
    fig.canvas.draw(); check_boxes(fig)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
