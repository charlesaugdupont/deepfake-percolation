"""Confidence in the press in the US, 1973-2024 (General Social Survey cumulative file 1972-2024, release 3a).
Not a paper figure (yet): real-world data on trust in news splitting over time, for the discussion (6 Nov).
CONPRESS: "...would you say you have a great deal of confidence, only some confidence, or hardly any confidence at all
in [the press]?"  PARTYID: Democrats 0-2 and Republicans 4-6 (leaners included), independents 3.
(A) all adults; (B) 'hardly any' confidence by party. Shares weighted with WTSSPS; 95% CIs from the weighted (Kish
effective) sample size, ignoring the clustered design (true CIs somewhat wider). From 2021 the GSS was fielded by web or
mixed mode (marked), which can shift levels. (The newspaper-reading panels were dropped on 6 Nov, 13:25.)
Usage: python3 gss_press.py extract <gss7224_r3a.dta> <out.parquet>    (needs pyreadstat)
       python3 gss_press.py figure <extract.parquet> <out.png>"""
import sys, os, numpy as np, pandas as pd
COLS = ["year", "conpress", "contv", "news", "partyid", "polviews", "age", "degree", "wtssps", "wtssall", "mode", "ballot"]


def extract(dta, out):
    import pyreadstat
    df, _ = pyreadstat.read_dta(dta, usecols=COLS, encoding="latin1")      # value labels are not UTF-8
    df.to_parquet(out); print("wrote", out, df.shape)


def share(g, k):
    """Weighted share answering k, its 95% CI half-width (Kish effective n) and n."""
    w = g.w.values; tot = w.sum(); neff = tot ** 2 / (w ** 2).sum(); p = w[g.conpress.values == k].sum() / tot
    return pd.Series({"p": 100 * p, "hw": 100 * 1.96 * np.sqrt(p * (1 - p) / neff), "n": len(g)})


def prepare(parquet):
    d = pd.read_parquet(parquet)
    d = d[d.conpress.isin([1, 2, 3])].copy(); d["w"] = d.wtssps
    d["party"] = np.select([d.partyid.isin([0, 1, 2]), d.partyid.isin([4, 5, 6]), d.partyid == 3],
                           ["Democrats", "Republicans", "independents"], "other")
    return d


def figure(parquet, out):
    import matplotlib as mpl
    mpl.use("Agg"); import matplotlib.pyplot as plt
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import figstyle as fs
    fs.setup()
    d = prepare(parquet)
    fig, axs = plt.subplots(1, 2, figsize=(fs.W2, 2.4))
    fig.subplots_adjust(wspace=0.28, top=0.86, bottom=0.17, left=0.08, right=0.98)

    def band(ax, s, col, ls, mk, lab):
        ax.fill_between(s.index, s.p - s.hw, s.p + s.hw, color=col, alpha=0.18, lw=0)
        ax.plot(s.index, s.p, color=col, ls=ls, marker=mk, ms=2.6, mec="white", mew=0.3, lw=1.1, label=lab)

    def year_axis(ax):
        ax.set_xlim(1971, 2026); ax.set_xticks([1975, 1985, 1995, 2005, 2015, 2024]); ax.set_ylim(0, 100)
        ax.axvline(2019.5, color=fs.INK2, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.text(2020.0, 97, "web /\nmixed\nmode", fontsize=4.6, color=fs.INK2, va="top", ha="left")
        ax.set_ylabel("% of US adults"); ax.set_xlabel("survey year")
        ax.grid(axis="y", color="#e6e4df", lw=0.5, zorder=0)

    ax = axs[0]                                              # A: all adults
    for k, lab, col, ls, mk in [(1, "a great deal", fs.CAT[0], "-", "o"), (2, "only some", fs.GREY, "--", "s"),
                                (3, "hardly any", fs.CAT[1], ":", "^")]:
        band(ax, d.groupby("year").apply(lambda g: share(g, k)), col, ls, mk, lab)
    year_axis(ax); ax.set_title("confidence in the press, all adults", fontsize=6.6)
    ax.legend(fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=2.3, columnspacing=1.0)
    ax = axs[1]                                              # B: by party
    for grp, col, ls, mk in [("Democrats", fs.CAT[0], "-", "o"), ("independents", fs.GREY, "--", "s"),
                             ("Republicans", fs.CAT[1], ":", "^")]:
        band(ax, d[d.party == grp].groupby("year").apply(lambda g: share(g, 3)), col, ls, mk, grp)
    year_axis(ax); ax.set_title("'hardly any' confidence, by party", fontsize=6.6)
    ax.legend(fontsize=5.2, loc="upper left", bbox_to_anchor=(0.0, 0.86), handlelength=2.3)
    for k, a in enumerate(axs):
        fs.panel_label(a, "AB"[k], x=-0.13, y=1.04)
    fig.savefig(out, bbox_inches="tight", dpi=300); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    {"extract": extract, "figure": figure}[sys.argv[1]](*sys.argv[2:4])
