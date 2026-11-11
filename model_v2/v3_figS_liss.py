"""SI figure: three descriptive checks in the LISS panel (Politics and Values, 2008-2026; aggregates only).
(A, B) confidence in the media and in science (0-10), first and last wave; (C) share with low (0-2) and high (8-10)
confidence per wave; (D) 4 x dip per wave with the 5% critical value of the dip test; (E) probability of being in the
same band 1, 3 and 5 waves later (media); (F) share with low confidence in the media by two proxies for being active
online. Input: <data>/liss_aggregates.json (written by liss_checks.py where the LISS files are)."""
import sys, os, json, numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
fs.setup()

CRIT = 2.17          # 5% critical value of 4 x dip is about 2.17 / sqrt(n) (simulated from the uniform null)
ITEM_COL = {"media": fs.BLUE, "science": fs.VIOLET}


def make(path, out):
    d = json.load(open(path)); years = d["years"]
    fig = plt.figure(figsize=(fs.W2, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.5)
    # A, B: distributions in the first and last wave
    for k, item in enumerate(["media", "science"]):
        ax = fig.add_subplot(gs[0, k]); fs.panel_label(ax, "AB"[k], x=-0.28, y=1.06)
        W = d["waves"][item]
        for j, (w, col, lab) in enumerate([(W[0], "#b9d3f2" if item == "media" else "#cbc5f0", str(W[0]["year"])),
                                           (W[-1], ITEM_COL[item], str(W[-1]["year"]))]):
            c = np.array(w["counts"], float); c = 100 * c / c.sum()
            ax.bar(np.arange(11) + (j - 0.5) * 0.4, c, width=0.4, color=col, edgecolor="white", linewidth=0.3,
                   label=f"{lab} (n = {w['n']:,}, 4·dip = {w['dip4']:.3f})")
        ax.set_xticks(range(0, 11, 2)); ax.set_xlim(-0.6, 10.6); ax.set_ylim(0, 45)
        ax.set_xlabel(f"confidence in {item} (0–10)"); ax.set_ylabel("% of respondents")
        ax.legend(fontsize=5.2, loc="upper left", handlelength=1.2)
        ax.set_title(f"confidence in {'the media' if item == 'media' else 'science'}", fontsize=6.4)
    # C: low and high shares over time
    ax = fig.add_subplot(gs[0, 2]); fs.panel_label(ax, "C", x=-0.28, y=1.06)
    k = 0
    for item in ["media", "science"]:
        W = d["waves"][item]
        for band, lab, ls, mk in [("low", "low (0–2)", "-", "o"), ("high", "high (8–10)", "--", "s")]:
            v = [w[band] for w in W]
            ax.plot(years, v, color=ITEM_COL[item], ls=ls, marker=mk, ms=2.4, mec="white", mew=0.3)
            ax.text(2026.8, v[-1], f"{item},\n{lab}" if (item, band) == ("science", "high") else f"{item}, {lab}",
                    fontsize=5.0, va="center", color=fs.INK)
            k += 1
    ax.set_ylim(0, 0.7); ax.set_xlabel("year"); ax.set_ylabel("share of respondents")
    ax.set_xticks([2008, 2014, 2020, 2026]); ax.set_xlim(2007, 2026.5)
    # D: 4 x dip per wave and the 5% critical value
    ax = fig.add_subplot(gs[1, 0]); fs.panel_label(ax, "D", x=-0.28, y=1.06)
    for i, item in enumerate(["media", "science"]):
        W = d["waves"][item]
        ax.plot(years, [w["dip4"] for w in W], color=ITEM_COL[item], ls=fs.DASHES[i], marker=fs.MARKERS[i], ms=2.6,
                mec="white", mew=0.3, label=item)
    crit = [CRIT / np.sqrt(w["n"]) for w in d["waves"]["media"]]
    ax.plot(years, crit, color=fs.INK2, lw=0.7, ls=":")
    ax.text(2026, crit[-1] + 0.002, "5% critical value", ha="right", va="bottom", fontsize=5.0, color=fs.INK2)
    ax.set_ylim(0, 0.05); ax.set_xticks([2008, 2014, 2020, 2026]); ax.set_xlabel("year"); ax.set_ylabel("split (4 × dip)")
    ax.legend(fontsize=5.4, loc="lower left", ncol=2)
    ax.set_title("no wave is bimodal at the 5% level", fontsize=6.2)
    # E: persistence of bands (media)
    ax = fig.add_subplot(gs[1, 1]); fs.panel_label(ax, "E", x=-0.28, y=1.06)
    T = d["transitions"]["media"]
    for i, (band, col) in enumerate([("low (0-2)", fs.ORANGE), ("middle (4-6)", fs.GREY), ("high (8-10)", fs.BLUE)]):
        ax.plot([1, 3, 5], [T[f"lag{l}"][band]["stay"] for l in (1, 3, 5)], color=col, ls=fs.DASHES[i], marker=fs.MARKERS[i],
                ms=3.0, mec="white", mew=0.3, label=band.replace("-", "–"))
    ax.set_xticks([1, 3, 5]); ax.set_xlim(0.6, 5.4); ax.set_ylim(0, 1)
    ax.set_xlabel("waves later"); ax.set_ylabel("share still in the same band")
    ax.set_title("confidence in the media, by starting band", fontsize=6.2)
    ax.legend(fontsize=5.4, loc="lower left")
    # F: who has low trust in the media
    ax = fig.add_subplot(gs[1, 2]); fs.panel_label(ax, "F", x=-0.28, y=1.06)
    G = d["groups"]["media"]
    rows = [("follows news via social media", "follows news via social media (2023–26)"),
            ("political discussion online (past 5 years)", "political discussion online, past 5 years (2008–26)")]
    for r, (key, lab) in enumerate(rows):
        y = 1 - r
        for val, col, mk in [("yes", fs.ORANGE, "o"), ("no", fs.GREY, "s")]:
            g = G[f"{key}: {val}"]
            ax.plot(g["low"], y, ls="none", marker=mk, ms=3.6, color=col, mec="white", mew=0.3,
                    label=val if r == 0 else None)
        ax.plot([G[f"{key}: yes"]["low"], G[f"{key}: no"]["low"]], [y, y], color=fs.INK2, lw=0.5, zorder=0)
        ax.text(0.101, y + 0.3, lab, fontsize=5.2, va="bottom", color=fs.INK)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_ylim(-0.5, 2.1)
    ax.set_xlim(0.10, 0.22); ax.set_xticks([0.10, 0.14, 0.18, 0.22])
    ax.set_xlabel("share with low confidence\nin the media (0–2)")
    ax.legend(fontsize=5.4, loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(1.0, 1.08))
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2])
