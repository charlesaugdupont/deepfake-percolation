"""Three descriptive checks of the model's qualitative predictions in the LISS panel (Politics and Values, 2008-2026).
Runs where the LISS files are (data/LISS); writes only aggregate tables (no individual records) to a JSON file.

1. Is trust split?  Distribution of confidence in "the media" and in "science" (0-10) per wave; 4 x Hartigan's dip
   (computed on scores jittered by U(-0.5, 0.5) to undo the 0-10 grid), share with low (0-2) and high (8-10) trust.
2. Is low trust sticky?  Wave-to-wave transitions within persons: probability of staying in the same 3-point band
   (0-2, 4-6, 8-10) next wave, and mean absolute change, by starting score; also after 3 and 5 waves.
3. Who has low trust?  2023-2026: by following the news via social media, and (all waves) by having taken part in a
   political discussion or campaign online in the past five years.
Usage: python3 liss_checks.py ../data/LISS ../data/v3/liss_aggregates.json
"""
import sys, re, glob, json
import numpy as np, pandas as pd
import diptest

BANDS = {"low (0-2)": (0, 2), "middle (4-6)": (4, 6), "high (8-10)": (8, 10)}
ITEMS = {"media": "021", "science": "025"}


def load(folder):
    waves = {}
    for f in sorted(glob.glob(f"{folder}/cv*_EN*.csv")):
        w = re.search(r"(cv\d\d[a-z])", f).group(1)
        want = {"nomem_encr"} | {f"{w}{c}" for c in ("021", "025", "071", "335")}
        D = pd.read_csv(f, sep=";", usecols=lambda c: c.lstrip("﻿") in want, low_memory=False)
        D.columns = [c.lstrip("﻿") for c in D.columns]
        out = pd.DataFrame({"id": D["nomem_encr"]})
        for name, code in [("media", "021"), ("science", "025"), ("online_discussion", "071"), ("news_social_media", "335")]:
            col = f"{w}{code}"
            v = pd.to_numeric(D[col], errors="coerce") if col in D else pd.Series(np.nan, index=D.index)
            hi = 10 if name in ("media", "science") else 1
            out[name] = v.where((v >= 0) & (v <= hi))       # -9 = don't know, 999 etc. = not asked -> missing
        waves[2000 + int(w[2:4])] = out.drop_duplicates("id")
    return waves


def dip4(x, n_jitter=5, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([4 * diptest.dipstat(x + rng.uniform(-0.5, 0.5, len(x))) for _ in range(n_jitter)]))


def band(s):
    for name, (lo, hi) in BANDS.items():
        if lo <= s <= hi:
            return name
    return None


def transitions(waves, item, lag=1, mask_fn=None):
    years = sorted(waves)
    M = np.zeros((11, 11), dtype=int)
    for i in range(len(years) - lag):
        a, b = waves[years[i]], waves[years[i + lag]]
        if mask_fn is not None:
            a = a[mask_fn(a)]
        m = a[["id", item]].merge(b[["id", item]], on="id", suffixes=("_0", "_1")).dropna()
        for s0, s1 in zip(m[f"{item}_0"].astype(int), m[f"{item}_1"].astype(int)):
            M[s0, s1] += 1
    return M


def persistence(M):
    out = {}
    for name, (lo, hi) in BANDS.items():
        n = M[lo:hi + 1].sum()
        out[name] = dict(n=int(n), stay=float(M[lo:hi + 1, lo:hi + 1].sum() / n) if n else None)
    s = np.arange(11)
    out["mean_abs_change_by_start"] = [float((M[k] * np.abs(s - k)).sum() / M[k].sum()) if M[k].sum() else None for k in range(11)]
    out["mean_change_by_start"] = [float((M[k] * (s - k)).sum() / M[k].sum()) if M[k].sum() else None for k in range(11)]
    return out


if __name__ == "__main__":
    folder, outp = sys.argv[1], sys.argv[2]
    W = load(folder); years = sorted(W)
    res = dict(years=years, waves={}, transitions={}, groups={})
    for item in ITEMS:
        res["waves"][item] = []
        for y in years:
            x = W[y][item].dropna().values
            res["waves"][item].append(dict(year=y, n=int(len(x)), mean=float(x.mean()), dip4=dip4(x),
                                           low=float((x <= 2).mean()), high=float((x >= 8).mean()),
                                           counts=np.bincount(x.astype(int), minlength=11).tolist()))
        res["transitions"][item] = {}
        for lag in (1, 3, 5):
            M = transitions(W, item, lag)
            res["transitions"][item][f"lag{lag}"] = dict(matrix=M.tolist(), **persistence(M))
    # 3. who has low trust
    for item in ITEMS:
        g = {}
        recent = [y for y in years if y >= 2023]
        for var, label in [("news_social_media", "follows news via social media"), ("online_discussion", "political discussion online (past 5 years)")]:
            for val, vlab in [(1, "yes"), (0, "no")]:
                ys = recent if var == "news_social_media" else years
                x = pd.concat([W[y].loc[W[y][var] == val, item] for y in ys]).dropna().values
                M = transitions({y: W[y] for y in ys}, item, 1, mask_fn=lambda d, v=var, k=val: d[v] == k)
                g[f"{label}: {vlab}"] = dict(n=int(len(x)), mean=float(x.mean()) if len(x) else None,
                                            low=float((x <= 2).mean()) if len(x) else None,
                                            stay_low=persistence(M)["low (0-2)"], years=[int(ys[0]), int(ys[-1])])
        res["groups"][item] = g
    json.dump(res, open(outp, "w"), indent=1)
    for item in ITEMS:
        print(f"== confidence in {item}")
        for r in res["waves"][item][::3] + [res["waves"][item][-1]]:
            print(f"  {r['year']}: n={r['n']}, mean={r['mean']:.2f}, 4*dip={r['dip4']:.3f}, low={r['low']:.3f}, high={r['high']:.3f}")
        for lag in (1, 3, 5):
            p = res["transitions"][item][f"lag{lag}"]
            print(f"  stay in band after {lag} wave(s): " + ", ".join(f"{b} {p[b]['stay']:.2f} (n={p[b]['n']})" for b in BANDS))
        p = res["transitions"][item]["lag1"]
        print("  mean |change| by starting score 0..10:", [None if v is None else round(v, 2) for v in p["mean_abs_change_by_start"]])
        print("  mean change by starting score 0..10:  ", [None if v is None else round(v, 2) for v in p["mean_change_by_start"]])
        for k, v in res["groups"][item].items():
            print(f"  {k}: n={v['n']}, mean={v['mean']:.2f}, low={v['low']:.3f}, stay low after 1 wave={v['stay_low']['stay']}")
