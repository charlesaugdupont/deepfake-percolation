"""SI figure (added 6 Nov; provisional number S12): trust splitting over time, in survey data and in the model, side by
side (not on common axes).
(A) US General Social Survey 1973-2024 (cumulative file, release 3a; gss_press.py): share of adults with 'hardly any'
confidence in the press, by party (PARTYID: Democrats 0-2 and Republicans 4-6, leaners included; independents 3).
Weighted (WTSSPS); 95% CIs from the weighted (Kish effective) sample size, ignoring the clustered design (true CIs
somewhat wider); from 2021 the GSS was fielded by web or mixed mode (dotted line).
(B) Model at the main setting (phi = 0.3, c/g = 0.05, l/g = 2; the Fig. 2 runs: same seeds, 20 runs, 4,000 items),
agents with few, middling or many neighbours (thirds of the network: fewer than 35, 35-209, 210 or more); 95% CIs over
runs (Student's t); log axis as in Fig. 2. Violet ramp, so that no audience group shares a colour with a party.
B shows mean trust in the checker J-hat (measure="trust", the default). Dashed: everyone's starting trust (1/3);
green: the checker's actual J (0.70). Tried and set aside on 6 Nov (you preferred the trust version, 15:03):
measure="distrust", the believed error rate (1 - J-hat)/2 = ((1 - t-hat) + f-hat)/2 in % (start 33%, actual 15%; the
same curves upside down), and measure="notuse", the share whose trust is too low for their stakes (never tried the
checker, or tried it and stopped), which starts split (100 / 24 / 0%) because low stakes keep the bottom third out
from the start.
Shared pattern: groups that start alike drift apart, and the gap does not close. Differences: the survey's groups are
parties, its split is partisan and its timing follows political events; the model has no opinions or parties, and its
groups differ in what is at stake for them (audience size); the measures differ (distrust of the press vs trust in a
verification tool, so the panels point opposite ways: up is distrust in A, trust in B) and so do the clocks (years vs
items); no correspondence between parties and audience groups is implied. The gap opens differently: in the model
trust rises among agents who use the checker (it is better than they first believed) and stays where it began among
those who never try it; in the survey distrust rose in every group (1973 -> 2024: Democrats 13% -> 33%, independents
16% -> 56%, Republicans 15% -> 78%).
Model numbers (20 runs, at 4,000 items): trust 0.333 / 0.560 / 0.689 (actual J 0.70; as believed error rates 33.3 /
22.0 / 15.5%, actual 15%); nobody with fewer than 35 neighbours ever checks, 24% of the middle third and none of the top
third never check; the gap between the top and bottom thirds reaches half its final size by item 28 and 90% by item
185, then stays.
Usage: si_survey.py <data dir> <out.png> [trust|distrust|notuse]   (reads <data>/time_deg_twitter_agents.pkl and
<data>/../gss/gss_press_extract.parquet)"""
import sys, os, pickle, numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import gss_press as G
fs.setup()

J0, J_TRUE = 1 / 3, 0.70                                    # starting trust; the checker's actual J (tpr 0.80 - fpr 0.10)
PARTIES = [("Democrats", fs.CAT[0], "-", "o"), ("independents", fs.GREY, "--", "s"), ("Republicans", fs.CAT[1], ":", "^")]


def ramp(cmap, n):
    """n shades, light -> dark; the lightest keeps 3.4:1 contrast on white (violet: free of the party colours and the
    green reference line)."""
    return [cmap(x) for x in np.linspace(0.52, 1.0, n)]


def model_series(P, measure):
    """runs x snapshots x groups, plus axis settings and reference lines [(value, label, colour, above?)]."""
    M = np.array([p["meanJ"] for p in P])
    if measure == "distrust":
        return (100 * (1 - M) / 2, "believed error rate of the checker (%)", "model: distrust of the checker, by audience size",
                (0, 40), [(100 * (1 - J0) / 2, "everyone's starting belief", fs.INK2, True),
                          (100 * (1 - J_TRUE) / 2, "checker's actual error rate", fs.ACTUAL, False)])
    if measure == "trust":
        return (M, "mean trust in the checker Ĵ", "model: trust in the checker, by audience size", (0, 0.8),
                [(J0, "everyone's starting trust", fs.INK2, False), (J_TRUE, "checker's actual J", fs.ACTUAL, True)])
    if measure == "notuse":
        return (100 * np.array([p["notchk"] for p in P]), "% not using the checker", "model: not using the checker, by audience size",
                (0, 105), [])
    raise ValueError(measure)


def make(data, out, measure="trust"):
    gss = os.path.join(data, "..", "gss", "gss_press_extract.parquet")
    D = G.prepare(gss)
    P = [p for m, p in pickle.load(open(os.path.join(data, "time_deg_twitter_agents.pkl"), "rb"))]
    fig, axs = plt.subplots(1, 2, figsize=(fs.W2, 2.55))
    fig.subplots_adjust(wspace=0.3, top=0.86, bottom=0.17, left=0.08, right=0.98)

    # A: survey
    ax = axs[0]
    for grp, col, ls, mk in PARTIES:
        s = D[D.party == grp].groupby("year").apply(lambda g: G.share(g, 3))
        ax.fill_between(s.index, s.p - s.hw, s.p + s.hw, color=col, alpha=0.18, lw=0)
        ax.plot(s.index, s.p, color=col, ls=ls, marker=mk, ms=2.6, mec="white", mew=0.3, lw=1.1, label=grp)
    ax.set_xlim(1971, 2026); ax.set_xticks([1975, 1985, 1995, 2005, 2015, 2024]); ax.set_ylim(0, 100)
    ax.axvline(2019.5, color=fs.INK2, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.text(2020.0, 97, "web /\nmixed\nmode", fontsize=4.6, color=fs.INK2, va="top", ha="left")
    ax.set_xlabel("survey year"); ax.set_ylabel("% with 'hardly any' confidence")
    ax.set_title("survey: distrust of the press, by party (US, GSS)", fontsize=6.6)
    ax.grid(axis="y", color="#e6e4df", lw=0.5, zorder=0)
    ax.legend(fontsize=5.3, loc="upper left", handlelength=2.3)

    # B: model
    ax = axs[1]
    t = np.maximum(P[0]["times"], 1); cuts = P[0]["cuts"]
    Y, ylab, title, ylim, refs = model_series(P, measure)
    labels = [f"fewer than {cuts[0]} neighbours", f"{cuts[0]}–{cuts[1] - 1} neighbours", f"{cuts[1]} or more neighbours"]
    cols = ramp(fs.CM_GEN, 3)
    for g in range(3):
        m = Y[:, :, g].mean(0); hw = fs.ci95_hw(Y[:, :, g].std(0, ddof=1), len(Y))
        ax.fill_between(t, m - hw, m + hw, color=cols[g], alpha=0.22, lw=0)
        ax.plot(t, m, color=cols[g], ls=fs.DASHES[g], lw=1.2)
        ax.plot(t[::14], m[::14], ls="none", marker=fs.MARKERS[g], ms=2.6, color=cols[g], mec="white", mew=0.3)
    off = 0.02 * (ylim[1] - ylim[0])
    for k, (y, lab, col, above) in enumerate(refs):
        ax.axhline(y, color=col, lw=0.6 if k == 0 else 0.8, ls=(0, (3, 2)) if k == 0 else "-", zorder=0)
        ax.text(1.15, y + off if above else y - off, lab, fontsize=4.8, color=col, va="bottom" if above else "top",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))       # keeps grid lines out of the label
    ax.set_xscale("log"); ax.set_xlim(1, t[-1] * 1.15); ax.set_ylim(*ylim)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1,000"]); ax.minorticks_off()
    ax.set_xlabel("items circulated"); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=6.6)
    ax.grid(axis="y", color="#e6e4df", lw=0.5, zorder=0)
    h = [Line2D([], [], color=cols[g], ls=fs.DASHES[g], marker=fs.MARKERS[g], ms=2.6, mec="white", mew=0.3, lw=1.2)
         for g in range(3)]
    ax.legend(h, labels, fontsize=5.3, loc="lower right", handlelength=2.3, bbox_to_anchor=(1.0, 0.03))
    for k, a in enumerate(axs):
        fs.panel_label(a, "AB"[k], x=-0.13, y=1.04)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)
    print(measure, "model at start", Y[:, 0, :].mean(0).round(3), "at end", Y[:, -1, :].mean(0).round(3),
          "n per group", P[0]["n"], "never checked share at end", (1 - np.array([p["ever"][-1] for p in P]).mean(0)).round(2))


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or []))
