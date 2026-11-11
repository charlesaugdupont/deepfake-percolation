"""Synthetic networks for the audience-inequality test (added 6 Nov).

Random networks with an exact degree sequence: N nodes, a set mean degree and a set inequality of degrees (coefficient
of variation, CV = SD / mean), wired at random. Degrees are the quantiles of a lognormal distribution at (i + 0.5)/N,
rounded and at least 1; the lognormal's two parameters are tuned so that the sequence has the target mean (exactly, up
to one stub) and the target CV. CV = 0 is a random regular graph. Wiring: configuration model, with self-loops and
double edges removed by degree-preserving swaps (every node keeps its degree), node labels shuffled; then the largest
connected component is kept, as for the real networks. The degree sequence is the same for every wiring of a level;
wirings differ by seed (derived from the network's name).

Two arms: (1) mean degree 20, CV 0 to 3, i.e. Gini 0 to 0.73 (the real networks: CV 0.95-2.24, Gini 0.37-0.68);
(2) CV 1, mean degree 4.3 to 93
(20 x 10^(j/3), j = -2..2, so that fixed absolute costs fall on the cost grid). Ten wirings per level.

Each network is written to <NETDIR>/synthetic/<name>/edges_csr.npz in the format config.graph() reads, so experiments
refer to it as graph="synthetic/<name>"; a summary of all networks goes to <NETDIR>/synthetic/networks.csv.
Usage: python3 synthnets.py [--netdir DIR]          (skips networks that already exist)
"""
import os, sys, argparse, hashlib
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

N = 5000
WIRINGS = 10
ARM_CV = [(20.0, cv) for cv in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]     # same mean degree, rising inequality
# (CV 3 appended 6 Nov so the range covers Twitter's Gini of 0.68; appending keeps earlier job indices valid)
ARM_K = [(k, 1.0) for k in (4.31, 9.28, 20.0, 43.09, 92.83)]              # same inequality, rising mean degree
LEVELS = ARM_CV + [lv for lv in ARM_K if lv not in ARM_CV]


def name(kbar, cv, w):
    return f"synthetic/n{N}_k{kbar:g}_cv{cv:g}_w{w}"


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def degree_sequence(n, kbar, cv, kmin=1):
    """Sorted degree sequence with sum = n * kbar (rounded to an even number) and CV close to the target."""
    total = int(round(n * kbar / 2)) * 2
    if cv == 0:
        d = np.full(n, total // n, dtype=np.int64); d[: total - d.sum()] += 1
        return d
    z = norm.ppf((np.arange(n) + 0.5) / n)
    seq = lambda mu, s: np.maximum(kmin, np.rint(np.exp(mu + s * z))).astype(np.int64)

    def fit_mu(s):                                       # smallest mu with sum >= total (the sum rises with mu)
        lo, hi = -40.0, np.log(total / n) + 5
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if seq(mid, s).sum() < total else (lo, mid)
        return hi

    def cv_of(s):
        d = seq(fit_mu(s), s)
        return d.std() / d.mean()

    slo, shi = 0.0, 3.0
    for _ in range(50):
        mid = (slo + shi) / 2
        slo, shi = (mid, shi) if cv_of(mid) < cv else (slo, mid)
    s = (slo + shi) / 2; mu = fit_mu(s); d = seq(mu, s)
    x = np.exp(mu + s * z); diff = total - int(d.sum())       # move single stubs where rounding was closest
    if diff > 0:
        d[np.argsort(-(x - d))[:diff]] += 1
    elif diff < 0:
        cand = np.where(d > kmin)[0]; d[cand[np.argsort((x - d)[cand])][:-diff]] -= 1
    return np.sort(d)


def wire(d, seed, max_tries=100000):
    """Configuration model on degree sequence d (labels shuffled); self-loops and double edges removed by
    degree-preserving swaps with random other edges. Returns (edges, number of swaps)."""
    rng = np.random.default_rng(seed)
    d = rng.permutation(d)
    stubs = np.repeat(np.arange(len(d)), d); rng.shuffle(stubs)
    a = stubs[0::2].copy(); b = stubs[1::2].copy(); M = len(a)
    key = lambda u, v: (u, v) if u < v else (v, u)
    count = {}
    for u, v in zip(a.tolist(), b.tolist()):
        k = key(u, v); count[k] = count.get(k, 0) + 1
    bad = [i for i in range(M) if a[i] == b[i] or count[key(int(a[i]), int(b[i]))] > 1]
    swaps = 0
    while bad:
        i = bad.pop(); u, v = int(a[i]), int(b[i])
        if u != v and count[key(u, v)] == 1:
            continue                                         # the other copy of a double edge was already moved
        for _ in range(max_tries):
            j = int(rng.integers(M))
            if j == i:
                continue
            x, y = int(a[j]), int(b[j])
            if rng.random() < 0.5:
                x, y = y, x
            k1, k2 = key(u, x), key(v, y)
            if u == x or v == y or k1 == k2 or k1 in count or k2 in count:
                continue
            for k in (key(u, v), key(int(a[j]), int(b[j]))):
                count[k] -= 1
                if count[k] == 0:
                    del count[k]
            count[k1] = 1; count[k2] = 1
            a[i], b[i], a[j], b[j] = u, x, v, y; swaps += 1
            break
        else:
            raise RuntimeError("could not remove a self-loop or double edge")
    return np.column_stack([a, b]), swaps


def build(kbar, cv, w, netdir):
    nm = name(kbar, cv, w)
    seed = int(hashlib.md5(nm.encode()).hexdigest()[:8], 16)
    d = degree_sequence(N, kbar, cv)
    edges, swaps = wire(d, seed)
    A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(N, N))
    ncomp, lab = connected_components(A, directed=False)
    big = np.argmax(np.bincount(lab))
    keep = lab[edges[:, 0]] == big
    g = E.edgelist_to_csr(edges[keep])                       # relabels the kept nodes to 0..n-1
    indptr, indices, rev = g
    deg = np.diff(indptr)
    # checks: simple, symmetric, every kept node keeps its planned degree
    for i in range(len(deg)):
        nb = indices[indptr[i]:indptr[i + 1]]
        assert len(np.unique(nb)) == len(nb) and i not in nb
    assert np.all(indices[rev] == np.repeat(np.arange(len(deg)), deg))
    assert len(edges[keep]) * 2 == deg.sum()
    path = os.path.join(netdir, nm, "edges_csr.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, indptr=indptr, indices=indices, rev=rev)
    return dict(name=nm, kbar_target=kbar, cv_target=cv, wiring=w, seed=seed, N=len(deg), M=int(deg.sum() // 2),
                dropped=N - len(deg), mean=deg.mean(), cv=deg.std() / deg.mean(), gini=gini(deg), kmin=int(deg.min()),
                median=float(np.median(deg)), kmax=int(deg.max()), swaps=swaps,
                md5=hashlib.md5(indices.tobytes()).hexdigest())


if __name__ == "__main__":
    import config as C
    ap = argparse.ArgumentParser(); ap.add_argument("--netdir", default=C.NETDIR); a = ap.parse_args()
    rows = []
    csv = os.path.join(a.netdir, "synthetic", "networks.csv")
    old = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame(columns=["name"])
    for kbar, cv in LEVELS:
        for w in range(WIRINGS):
            nm = name(kbar, cv, w)
            if nm in set(old.name) and os.path.exists(os.path.join(a.netdir, nm, "edges_csr.npz")):
                rows.append(old[old.name == nm].iloc[0].to_dict()); continue
            r = build(kbar, cv, w, a.netdir); rows.append(r)
            print(f"{nm}: N={r['N']} mean={r['mean']:.2f} CV={r['cv']:.3f} Gini={r['gini']:.3f} "
                  f"max={r['kmax']} dropped={r['dropped']} swaps={r['swaps']}", flush=True)
    pd.DataFrame(rows).to_csv(csv, index=False); print("wrote", csv)
