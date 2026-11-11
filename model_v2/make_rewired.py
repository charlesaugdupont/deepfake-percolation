"""Degree-preserving randomisation (double-edge swaps, 10 x E accepted swaps) of a network; keeps the largest component."""
import sys, os, numpy as np, pandas as pd
from numba import njit


@njit(cache=True)
def _swap(e, n_nodes, n_target, seed):
    np.random.seed(seed)
    E = e.shape[0]
    keys = set()
    for k in range(E):
        a, b = e[k, 0], e[k, 1]
        if a > b: a, b = b, a
        keys.add(a * n_nodes + b)
    acc = 0; tries = 0
    while acc < n_target and tries < 50 * n_target:
        tries += 1
        i = np.random.randint(E); j = np.random.randint(E)
        if i == j: continue
        a, b = e[i, 0], e[i, 1]; c, d = e[j, 0], e[j, 1]
        if np.random.random() < 0.5: c, d = d, c
        if a == d or c == b or a == c or b == d: continue
        k1 = min(a, d) * n_nodes + max(a, d); k2 = min(c, b) * n_nodes + max(c, b)
        if k1 in keys or k2 in keys: continue
        keys.remove(min(a, b) * n_nodes + max(a, b)); keys.remove(min(c, d) * n_nodes + max(c, d))
        keys.add(k1); keys.add(k2)
        e[i, 0], e[i, 1] = a, d; e[j, 0], e[j, 1] = c, b
        acc += 1
    return acc, tries


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    ed = pd.read_csv(src).values.astype(np.int64)
    n = int(ed.max()) + 1
    acc, tries = _swap(ed, n, 10 * len(ed), 2026)
    print("accepted swaps", acc, "tries", tries)
    import networkx as nx
    G = nx.Graph(); G.add_edges_from(map(tuple, ed))
    lcc = max(nx.connected_components(G), key=len); H = G.subgraph(lcc)
    deg0 = np.bincount(pd.read_csv(src).values.ravel()); deg1 = np.array([d for _, d in sorted(G.degree())])
    print("nodes", G.number_of_nodes(), "LCC", len(lcc), "edges", H.number_of_edges(), "degree sequence preserved:", np.array_equal(np.sort(deg0[deg0 > 0]), np.sort(deg1)))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pd.DataFrame(np.array(H.edges()), columns=["id_1", "id_2"]).to_csv(dst, index=False)
