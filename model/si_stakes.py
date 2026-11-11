"""SI figure and Table S1: how the stakes grow with the audience (phi = 0.3, l/g = 2; 2,000 items; 10 runs per form at
c/g = 0.05, the cost shown, and 5 at c/g = 0.01 and 0.2). s = k/<k> is the number of neighbours relative to the average; the gain stake is g·s^a and the loss stake
l·s^b. Five forms: both stakes ∝ s (baseline); penalty ∝ √s or constant with the gain ∝ s; both stakes ∝ √s; both
stakes constant. All forms agree for an average audience.
Figure: share of received items checked, by number of neighbours (c/g = 0.05).
Table S1 (c/g = 0.05): not checking, priced out, trapped, split of trust (one checker), share with a clear favourite
(two identical checkers, among agents who can still check), reach of genuine items, and the correlation between number
of neighbours and share of fakes among the items shared (Twitter data: +0.36).
Usage: si_stakes.py <data dir> <out.png> <table prefix>"""
import sys, os, pickle, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
from figstyle import degree_axis
fs.setup()

RHO_DATA = 0.36
COST = 0.05
# (label, experiment, key in the run labels, value, colour, dash, marker)
FORMS = [("both stakes ∝ audience (baseline)", "penalty", "gamma_loss", 1.0, fs.INK, "-", "o"),
         ("penalty ∝ √audience", "penalty", "gamma_loss", 0.5, fs.CAT[0], "--", "s"),
         ("penalty constant", "penalty", "gamma_loss", 0.0, fs.CAT[1], ":", "^"),
         ("both stakes ∝ √audience", "stakes", "gamma", 0.5, fs.CAT[2], "-.", "D"),
         ("both stakes constant", "stakes", "gamma", 0.0, fs.CAT[3], (0, (5, 1.5)), "v")]


def load(data):
    out = {}
    for exp in ("penalty", "stakes"):
        out[exp] = dict(S=pd.read_parquet(os.path.join(data, f"{exp}_twitter.parquet")),
                        A=pickle.load(open(os.path.join(data, f"{exp}_twitter_agents.pkl"), "rb")),
                        A2=pickle.load(open(os.path.join(data, f"{exp}_two_twitter_agents.pkl"), "rb")))
    return out


def pick(L, key, val, cost=COST):
    return [p for m, p in L if abs(m[key] - val) < 1e-9 and abs(m["cost"] - cost) < 1e-9]


def checked_by_degree(runs, edges, min_n=30):
    deg_all = np.concatenate([a["deg"] for a in runs])
    cen, mu, er = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if ((deg_all >= lo) & (deg_all < hi)).sum() < min_n:
            continue
        v = []
        for a in runs:
            m = (a["n_recv"] > 0) & (a["deg"] >= lo) & (a["deg"] < hi)
            v.append((a["n_checks"][m] / a["n_recv"][m]).mean() if m.any() else np.nan)
        v = np.array(v); n = int(np.sum(~np.isnan(v)))
        cen.append(np.sqrt(lo * max(hi - 1, lo))); mu.append(np.nanmean(v))
        er.append(float(fs.ci95_hw(np.nanstd(v, ddof=1), n)) if n > 1 else 0.0)
    return np.array(cen), np.array(mu), np.array(er)


def table(data_d):
    rows = []
    for lab, exp, key, val, *_ in FORMS:
        S = data_d[exp]["S"]; S = S[np.isclose(S[key], val) & np.isclose(S.cost, COST)]
        two = pick(data_d[exp]["A2"], key, val)
        rows.append({"stakes": lab,
                     "not checking": S.frac_locked.mean(), "priced out": (1 - S.frac_worth).mean(),
                     "trapped": S.frac_trapped.mean(), "split of trust (dip)": S.dip4.mean() / 4,
                     "two checkers: clear favourite": np.mean([p["fav"][-1] for p in two]),
                     "reach of genuine items": S.reach_true.mean(),
                     "corr. neighbours vs fakes shared (data +0.36)": S.rho_deg_fakeshare.mean()})
    return pd.DataFrame(rows)


def fmt(col, v):
    """Correlations with a sign, the dip (0-0.25) with three decimals, shares with two."""
    return f"{v:+.2f}" if col.startswith("corr") else (f"{v:.3f}" if col.startswith("split") else f"{v:.2f}")


def write_table(T, prefix):
    T.to_csv(prefix + ".csv", index=False, float_format="%.3f")
    cols = list(T.columns)
    with open(prefix + ".md", "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in T.iterrows():
            fh.write("| " + " | ".join([r[cols[0]]] + [fmt(c, r[c]) for c in cols[1:]]) + " |\n")
    with open(prefix + ".tex", "w") as fh:
        fh.write("\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}\n\\toprule\n")
        fh.write(" & ".join(cols).replace("∝", "$\\propto$").replace("√", "$\\sqrt{}$").replace("·", "$\\cdot$") + " \\\\\n\\midrule\n")
        for _, r in T.iterrows():
            cells = [r[cols[0]].replace("∝", "$\\propto$").replace("√audience", "$\\sqrt{\\text{audience}}$")]
            cells += [fmt(c, r[c]) for c in cols[1:]]
            fh.write(" & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("saved", prefix + ".csv/.md/.tex")


def make(data, out, table_prefix=None):
    d = load(data)
    A_base = pick(d["penalty"]["A"], "gamma_loss", 1.0)
    kbar = float(np.mean(A_base[0]["deg"]))
    edges = np.unique(np.round(np.geomspace(1, np.concatenate([a["deg"] for a in A_base]).max() + 1, 13)))
    fig, ax = plt.subplots(figsize=(fs.W2 * 0.62, 2.3))
    allc = []
    for lab, exp, key, val, col, ls, mk in FORMS:
        cen, mu, er = checked_by_degree(pick(d[exp]["A"], key, val), edges)
        ax.fill_between(cen, mu - er, mu + er, color=col, alpha=0.15, lw=0)
        ax.plot(cen, mu, color=col, ls=ls, marker=mk, ms=2.8, mec="white", mew=0.3, lw=1.1, label=lab); allc.extend(cen)
    ax.set_xscale("log"); degree_axis(ax, min(allc), max(allc)); ax.set_xlabel("number of neighbours k")
    ax.axvline(kbar, color=fs.INK2, lw=0.5, ls=(0, (1, 2)), zorder=0)
    ax.text(kbar * 1.12, 0.02, "average\naudience", fontsize=5.0, color=fs.INK2, ha="left", va="bottom")
    ax.set_ylim(-0.02, 1.02); ax.set_ylabel("share of received items checked")
    ax.set_title(f"who checks (c/g = {COST:g}, φ = 0.3, ℓ/g = 2)", fontsize=6.3)
    ax.legend(fontsize=5.4, loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, handlelength=2.6,
              title="how the stakes grow\nwith the audience", title_fontsize=5.6, alignment="left")
    fig.savefig(out, bbox_inches="tight"); print("saved", out)
    if table_prefix:
        T = table(d); write_table(T, table_prefix)
        return T


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
