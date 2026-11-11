"""Figure: model and mechanism. (A) schematic, (B) decision regions and the checking threshold,
(C) belief trajectories hitting the absorbing barrier, (D) share locked out by degree: simulation vs first-passage model."""
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


def box(ax, x, y, w, h, text, fc, ec=None, fs_=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec or fc, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs_, fontweight="bold" if bold else None, color=fs.INK)


def arrow(ax, x0, y0, x1, y1, col=None, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col or fs.INK2, ls=ls))


def schematic(ax):
    """Wide schematic: network cascade -> decision -> learning -> absorbing barrier. Axes aspect is equal, x in [0, 4.4]."""
    ax.set_xlim(0, 4.4); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    pos = np.array([[0.10, 0.55], [0.30, 0.85], [0.35, 0.50], [0.12, 0.20], [0.58, 0.80], [0.62, 0.42], [0.40, 0.15], [0.82, 0.62], [0.80, 0.18]])
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (2, 5), (3, 6), (2, 6), (5, 7), (4, 7), (5, 8), (6, 8)]
    shared = {(0, 1), (0, 2), (2, 5), (2, 6), (5, 7), (5, 8)}
    for a, b in edges:
        on = (a, b) in shared
        ax.plot(*pos[[a, b]].T, color=fs.BLUE if on else "#d5d4d0", lw=1.3 if on else 0.6, zorder=1)
    st = {0: fs.BLUE, 1: fs.GREY, 2: fs.GREEN, 3: "#ffffff", 4: "#ffffff", 5: fs.BLUE, 6: fs.GREY, 7: fs.GREY, 8: fs.BLUE}
    for i, p in enumerate(pos):
        ax.add_patch(Circle(p, 0.05, fc=st[i], ec=fs.INK2, lw=0.5, zorder=2))
    ax.text(0.10, 0.66, "seed", fontsize=5.4, ha="center", color=fs.INK2)
    for y, col, lab in [(0.97, fs.BLUE, "shared"), (0.97, fs.GREEN, "checked, passed"), (0.97, fs.GREY, "discarded")][:0]:
        pass
    ax.add_patch(Circle((0.06, 0.02), 0.025, fc=fs.BLUE, ec=fs.INK2, lw=0.4)); ax.text(0.10, 0.02, "shared", fontsize=5, va="center")
    ax.add_patch(Circle((0.36, 0.02), 0.025, fc=fs.GREEN, ec=fs.INK2, lw=0.4)); ax.text(0.40, 0.02, "checked", fontsize=5, va="center")
    ax.add_patch(Circle((0.68, 0.02), 0.025, fc=fs.GREY, ec=fs.INK2, lw=0.4)); ax.text(0.72, 0.02, "discarded", fontsize=5, va="center")
    # boxes
    box(ax, 1.08, 0.30, 0.86, 0.50, "agent $i$ receives an item\nfrom neighbour $j$\n(fake with prob. $\\varphi$)\n\nestimates $p$ = P(genuine)\nfrom $j$'s record and\nthe base rate $\\pi_i$", "#eef3fb", fs_=5.5)
    arrow(ax, 0.92, 0.55, 1.07, 0.55)
    box(ax, 2.10, 0.70, 0.78, 0.22, "share: $+g\\,s_i$ if genuine,\n$-l\\,s_i$ if fake", "#d7e6f8", fs_=5.3)
    box(ax, 2.10, 0.42, 0.78, 0.22, "check: pay $c$, then share\nunless the checker flags it", "#d3efe3", fs_=5.3)
    box(ax, 2.10, 0.14, 0.78, 0.22, "discard: 0", "#ecebe7", fs_=5.3)
    ax.text(2.49, 0.96, "highest expected payoff", ha="center", fontsize=5.4, color=fs.INK2)
    for y in [0.81, 0.53, 0.25]:
        arrow(ax, 1.95, 0.55, 2.09, y)
    box(ax, 3.04, 0.50, 1.32, 0.42, "truth revealed afterwards: everyone reached\nupdates $\\pi_i$ and trust in the sender $\\tau_{ij}$;\n"
        "only agents who checked update their beliefs\nabout the checker $(\\hat t_i,\\hat f_i)$", "#f6f5f2", fs_=4.9)
    box(ax, 3.04, 0.02, 1.32, 0.40, "checking can pay only if\n$\\hat J_i=\\hat t_i-\\hat f_i>\\theta_i=\\frac{c}{s_i}\\left(\\frac{1}{g}+\\frac{1}{l}\\right)$,  $s_i=\\frac{k_i}{\\langle k\\rangle}$\n"
        "below $\\theta_i$: no checks, no learning — absorbing", "#fbe9dd", fs_=4.9)
    arrow(ax, 2.89, 0.62, 3.03, 0.70); arrow(ax, 3.70, 0.50, 3.70, 0.43, col=fs.ORANGE)


def regions(ax, s_list=(1.0, 0.1), c=0.04329):
    p = np.linspace(0, 1, 500)[None, :]; Jv = np.linspace(-0.2, 1.0, 500)[:, None]
    t = 0.5 + Jv / 2; f = 0.5 - Jv / 2; s = s_list[0]
    us = s * (p * G - (1 - p) * L); uv = -c + s * (p * (1 - f) * G - (1 - p) * (1 - t) * L)
    R = np.where(us > 0, np.where(us > uv, 2, 1), np.where(uv > 0, 1, 0))
    ax.pcolormesh(p.ravel(), Jv.ravel(), R, cmap=ListedColormap(["#ecebe7", "#d3efe3", "#d7e6f8"]), shading="auto", rasterized=True)
    for sv, ls, va, dy in zip(s_list, ["--", ":"], ["bottom", "top"], [0.012, -0.012]):
        th = (c / sv) * (1 / G + 1 / L)
        ax.axhline(th, color=fs.INK, lw=0.9, ls=ls)
        ax.text(0.01, th + dy, f"θ, {'average audience' if sv == 1 else 'audience 1/10 of average'}", ha="left", va=va, fontsize=5.4)
    ax.axhline(T_TRUE - F_TRUE, color=fs.GREEN, lw=0.8); ax.text(0.99, T_TRUE - F_TRUE + 0.012, "actual J of checker", fontsize=5.4, color=fs.GREEN, ha="right", va="bottom")
    ax.axhline(1 / 3, color=fs.INK2, lw=0.6, ls="-."); ax.text(0.01, 1 / 3 + 0.012, "shared starting trust", fontsize=5.4, color=fs.INK2)
    ax.text(0.10, -0.15, "DISCARD", fontsize=6.5, fontweight="bold", color=fs.INK2)
    ax.text(0.47, 0.88, "CHECK", fontsize=6.5, fontweight="bold", color="#0f7a53")
    ax.text(0.80, -0.15, "SHARE", fontsize=6.5, fontweight="bold", color="#154f99")
    ax.set_xlabel("P(item is genuine), p"); ax.set_ylabel("trust in the checker  Ĵ")
    ax.set_xlim(0, 1); ax.set_ylim(-0.2, 1.0)


def trajectories(ax, q, theta=0.2, n=60, k=40):
    P = paths(n, k, prior=PRIOR, t=T_TRUE, f=F_TRUE, q=q, seed=11)
    x = np.arange(k + 1)
    for i in range(n):
        hit = np.where(P[i] <= theta)[0]
        if len(hit):
            h = hit[0]
            ax.plot(x[:h + 1], P[i, :h + 1], color=fs.ORANGE, lw=0.6, alpha=0.8)
            ax.plot([h, k], [P[i, h]] * 2, color=fs.ORANGE, lw=0.6, alpha=0.5, ls=":")
            ax.plot(h, P[i, h], "o", ms=1.8, color=fs.ORANGE)
        else:
            ax.plot(x, P[i], color=fs.BLUE, lw=0.5, alpha=0.45)
    ax.axhline(theta, color=fs.INK, lw=0.9, ls="--"); ax.text(k, theta + 0.015, "threshold θ", ha="right", va="bottom", fontsize=5.6)
    ax.text(k * 0.98, 0.97, "keeps checking: learns the truth", color=fs.BLUE, ha="right", va="top", fontsize=5.4)
    ax.text(k * 0.98, -0.12, "hits θ, stops checking: belief frozen", color=fs.ORANGE, ha="right", va="top", fontsize=5.4)
    ax.axhline(T_TRUE - F_TRUE, color=fs.GREEN, lw=0.8)
    ax.set_xlabel("checks made by the agent"); ax.set_ylabel("trust in the checker  Ĵ"); ax.set_ylim(-0.25, 1.0); ax.set_xlim(0, k)


def by_degree(ax, mech_pkl):
    d = pickle.load(open(mech_pkl, "rb")); runs = d["runs"]
    rows = []
    for r in runs:
        lock = r["J"] <= r["theta"]
        rows.append(pd.DataFrame(dict(cost=r["cost"], seed=r["seed"], q=r["q"], deg=r["deg"], theta=r["theta"], locked=lock)))
    A = pd.concat(rows)
    edges = np.unique(np.round(np.geomspace(1, A.deg.max() + 1, 16))) - 0.5
    A["dbin"] = pd.cut(A.deg, edges)
    costs = sorted(A.cost.unique())
    cols = ["#9cc2ef", fs.BLUE, "#154f99", "#0b2e5c"][-len(costs):]
    for c, col in zip(costs, cols):
        sub = A[A.cost == c]
        per_run = sub.groupby(["dbin", "seed"], observed=True).agg(locked=("locked", "mean"), deg=("deg", "mean"), theta=("theta", "mean")).reset_index()
        g = per_run.groupby("dbin", observed=True).agg(m=("locked", "mean"), s=("locked", "sem"), deg=("deg", "mean"), theta=("theta", "mean"), n=("locked", "size"))
        g = g[g.n >= 3]
        ax.errorbar(g.deg, g.m, yerr=1.96 * g.s.fillna(0), fmt="o", ms=2.6, color=col, capsize=1.2, lw=0.8)
        ax.plot([], [], color=col, lw=1, marker="o", ms=2.6, label=f"{c:.2g}")
        degs = np.unique(np.round(np.geomspace(1, A.deg.max(), 40)))
        th = (c / (degs / (sub.deg.mean()))) * (1 / G + 1 / L)
        ax.plot(degs, p_trap(th, prior=PRIOR, t=T_TRUE, f=F_TRUE, q=sub.q.mean(), n_mc=6000), color=col, lw=1)
    ax.set_xscale("log"); ax.set_xlabel("number of neighbours k"); ax.set_ylabel("share locked out (Ĵ ≤ θ)")
    ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=5.2, loc="center left", bbox_to_anchor=(0.0, 0.42), title="c/g", title_fontsize=5.4)
    ax.set_title("dots: simulation (mean ± 95% CI)\nlines: first-passage model", fontsize=5.4, color=fs.INK2)
    tk = [t for t in [1, 10, 100, 1000] if t <= A.deg.max() * 1.5]
    ax.set_xticks(tk); ax.set_xticklabels([str(t) for t in tk]); ax.minorticks_off()
    return A


def make(mech_pkl, out):
    fig = plt.figure(figsize=(fs.W2, 4.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.42)
    ax = fig.add_subplot(gs[0, :]); schematic(ax); fs.panel_label(ax, "A", x=0.0, y=0.98)
    ax = fig.add_subplot(gs[1, 0]); regions(ax); fs.panel_label(ax, "B")
    d = pickle.load(open(mech_pkl, "rb")); q = np.mean([r["q"] for r in d["runs"]])
    ax = fig.add_subplot(gs[1, 1]); trajectories(ax, q=q); fs.panel_label(ax, "C")
    ax = fig.add_subplot(gs[1, 2]); by_degree(ax, mech_pkl); fs.panel_label(ax, "D")
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
