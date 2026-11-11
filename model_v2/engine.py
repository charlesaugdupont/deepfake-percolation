"""
Fast re-implementation of deepfake-percolation/run_experiment.jl (Python + numba).

Default switches reproduce the Julia model exactly (up to random-number streams):
    ptrue_mode=0  : p = tau*pi / (tau*pi + (1-tau)(1-pi))   (sender record combined with base rate)
    seed_mode=0   : the seed uses its fixed 'base' trust prior as tau
    prior_mode=0  : prior means ~ U[lo, hi], strength kappa, with the Julia quirk that the base-trust
                    alpha and beta are drawn from two independent means
    gamma=1       : stakes scale linearly with degree (k_i ** gamma)
    cost_in_payoff=False : agent payoffs exclude checking costs (as in the Julia code)

Candidate tweaks (all optional):
    ptrue_mode=1  : p = tau (sender record alone)
    ptrue_mode=2  : p = (n_genuine_ij + kappa_tau*pi_i) / (n_ij + kappa_tau)  (record shrunk to base rate)
    seed_mode=1   : the seed (no sender) uses p = pi_i
    seed_mode=2   : like 1, and the seed is drawn proportional to degree
    prior_mode=1  : as 0 but without the quirk
    prior_mode=2  : identical priors for everyone (pass t0, f0, g0, n0)
    K=2           : two checkers (check A, check B, or both; 'both' uses the better of AND / OR rules)
    tpr/fpr schedules: checker quality can change over time (shock experiments)
    gamma_loss    : separate exponent for the loss stake (loss = rm * (k_i/<k>)**gamma_loss); None = gamma
    int_start, int_end, int_mult, int_target : intervention: during items [int_start, int_end) the checking cost
                    of eligible agents is multiplied by int_mult (int_target: "all", "below_median" or a bool array)
    g_lo, g_hi with prior_mode=2 : spread-out starting beliefs about the share of genuine items (uniform means)
    all_senders   : after the truth is revealed, update the record of every neighbour who shared the item (not only the
                    first sender's); decisions still use the first sender's record
"""
import math
import random
import numpy as np
from numba import njit

# ----------------------------------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------------------------------

def holme_kim(n, m, p, seed):
    """Port of powerlaw_cluster_graph in run_experiment.jl."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    s0 = m + 1
    for i in range(s0):
        for j in range(i + 1, s0):
            adj[i].add(j); adj[j].add(i)
    pref = []
    for v in range(s0):
        pref += [v] * len(adj[v])
    for v in range(s0, n):
        u = pref[rng.randrange(len(pref))]
        adj[v].add(u); adj[u].add(v); connected = [u]
        while len(connected) < m:
            if rng.random() < p:
                nb = sorted(adj[u]); w = nb[rng.randrange(len(nb))]
            else:
                w = pref[rng.randrange(len(pref))]
            if w != v and w not in connected and w not in adj[v]:
                adj[v].add(w); adj[w].add(v); connected.append(w)
        for u2 in connected:
            pref.append(v); pref.append(u2)
    return adj_to_csr(adj)


def regular_random(n, k, seed):
    import networkx as nx
    G = nx.random_regular_graph(k, n, seed=seed)
    adj = [set(G.neighbors(i)) for i in range(n)]
    return adj_to_csr(adj)


def adj_to_csr(adj):
    n = len(adj)
    deg = np.array([len(a) for a in adj], dtype=np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64); indptr[1:] = np.cumsum(deg)
    indices = np.empty(indptr[-1], dtype=np.int64)
    for i, a in enumerate(adj):
        indices[indptr[i]:indptr[i + 1]] = sorted(a)
    rev = np.empty_like(indices)
    for i in range(n):
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            lo, hi = indptr[j], indptr[j + 1]
            rev[e] = lo + np.searchsorted(indices[lo:hi], i)
    return indptr, indices, rev


def edgelist_to_csr(edges, n=None):
    """edges: (E,2) int array of undirected edges; relabels nodes to 0..n-1, drops self loops/duplicates."""
    edges = np.asarray(edges)
    nodes, inv = np.unique(edges.ravel(), return_inverse=True)
    e = inv.reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    n = len(nodes)
    adj = [set() for _ in range(n)]
    for a, b in e:
        adj[a].add(b); adj[b].add(a)
    return adj_to_csr(adj)


# ----------------------------------------------------------------------------------------------
# core simulation
# ----------------------------------------------------------------------------------------------

@njit(cache=True)
def _simulate(indptr, indices, rev, T, burn,
              phi_s, tpr_s, fpr_s, cost, rp, rm, gamma, kdiv,
              ptrue_mode, seed_mode, kappa_tau, allow_both, cost_in_payoff,
              g_a, g_b, t_a, t_b, f_a, f_b, tr_a, tr_b, base_a, base_b,
              snap_times, seed, eps_explore, feedback,
              gamma_l, split_stakes, int_start, int_end, int_mult, elig, all_senders):
    np.random.seed(seed)
    N = indptr.shape[0] - 1
    K = t_a.shape[1]
    deg = np.empty(N, dtype=np.int64)
    kk = np.empty(N)          # gain stake
    kl = np.empty(N)          # loss stake (equal to kk unless split_stakes)
    cumdeg = np.empty(N)
    acc = 0.0
    for i in range(N):
        deg[i] = indptr[i + 1] - indptr[i]
        kk[i] = (deg[i] / kdiv) ** gamma
        kl[i] = (deg[i] / kdiv) ** gamma_l if split_stakes else kk[i]
        acc += deg[i]
        cumdeg[i] = acc

    # per-agent tracking
    payoff = np.zeros(N); best = np.zeros(N); worst = np.zeros(N); costpaid = np.zeros(N)
    n_checks = np.zeros((N, K), dtype=np.int64)
    n_both = np.zeros(N, dtype=np.int64)
    n_bad = np.zeros((N, K), dtype=np.int64)
    first_check = -np.ones((N, K), dtype=np.int64)
    first_bad = -np.ones((N, K), dtype=np.int64)     # outcome of the first check: 1 bad, 0 good
    n_recv = np.zeros(N, dtype=np.int64)
    n_shared = np.zeros(N, dtype=np.int64)
    shares_true = np.zeros(N, dtype=np.int64); shares_fake = np.zeros(N, dtype=np.int64)

    # cascade records
    c_fake = np.zeros(T, dtype=np.bool_)
    c_size = np.zeros(T, dtype=np.int64)
    c_shares = np.zeros(T, dtype=np.int64)
    c_checks = np.zeros((T, K), dtype=np.int64)
    c_both = np.zeros(T, dtype=np.int64)
    c_depth = np.zeros(T, dtype=np.int64)
    c_util = np.zeros(T)

    S = snap_times.shape[0]
    snapJ = np.zeros((S, N, K), dtype=np.float32)
    snapT = np.zeros((S, N, K), dtype=np.float32)
    snapNC = np.zeros((S, N, K), dtype=np.int32)
    sidx = 0
    while sidx < S and snap_times[sidx] == 0:
        for i in range(N):
            for c in range(K):
                th = t_a[i, c] / (t_a[i, c] + t_b[i, c]); fh = f_a[i, c] / (f_a[i, c] + f_b[i, c])
                snapJ[sidx, i, c] = th - fh; snapT[sidx, i, c] = th; snapNC[sidx, i, c] = 0
        sidx += 1

    stamp = -np.ones(N, dtype=np.int64)       # cascade id when visited
    queue = np.empty(N, dtype=np.int64)
    qdepth = np.empty(N, dtype=np.int64)
    pedge = np.empty(N, dtype=np.int64)       # CSR index of receiver->sender (trust entry), -1 for seed
    order = np.empty(N, dtype=np.int64)       # visited nodes in order
    act_log = np.zeros(N, dtype=np.int64)     # 0 none/discard, 1 shared genuine, 2 shared fake
    ver_node = np.empty(2 * N, dtype=np.int64)
    ver_chk = np.empty(2 * N, dtype=np.int64)
    ver_flag = np.empty(2 * N, dtype=np.bool_)
    revealed = np.ones(N, dtype=np.bool_)

    for t in range(T):
        phi = phi_s[t]
        fake = np.random.random() < phi
        if seed_mode == 2:
            r = np.random.random() * cumdeg[N - 1]
            s = np.searchsorted(cumdeg, r)
            if s >= N:
                s = N - 1
        else:
            s = np.random.randint(N)
        after = t >= burn
        qh = 0; qt = 0; nvis = 0; nver = 0
        queue[qt] = s; qdepth[qt] = 0; pedge[s] = -1; qt += 1
        stamp[s] = t; order[nvis] = s; nvis += 1
        n_sh = 0; util = 0.0; maxd = 0; nb_both = 0
        while qh < qt:
            cur = queue[qh]; d = qdepth[qh]; qh += 1
            if d > maxd:
                maxd = d
            n_recv[cur] += 1
            act_log[cur] = 0
            pi = g_a[cur] / (g_a[cur] + g_b[cur])
            pe = pedge[cur]
            # ---- probability the item is genuine
            if pe < 0:
                if seed_mode == 0:
                    tau = base_a[cur] / (base_a[cur] + base_b[cur])
                    den = tau * pi + (1.0 - tau) * (1.0 - pi)
                    p = 0.0 if den == 0.0 else tau * pi / den
                else:
                    p = pi
            else:
                if ptrue_mode == 0:
                    tau = tr_a[pe] / (tr_a[pe] + tr_b[pe])
                    den = tau * pi + (1.0 - tau) * (1.0 - pi)
                    p = 0.0 if den == 0.0 else tau * pi / den
                elif ptrue_mode == 1:
                    p = tr_a[pe] / (tr_a[pe] + tr_b[pe])
                else:
                    p = (tr_a[pe] + kappa_tau * pi) / (tr_a[pe] + tr_b[pe] + kappa_tau)
            k = kk[cur]; kq = kl[cur]
            cm = int_mult if (t >= int_start and t < int_end and elig[cur]) else 1.0   # intervention cost multiplier
            if after:
                if not fake:
                    best[cur] += rp * k
                else:
                    worst[cur] -= rm * kq
            if split_stakes:
                us = p * rp * k - (1.0 - p) * rm * kq
            else:
                us = k * (p * rp - (1.0 - p) * rm)
            # ---- best checking option
            bestc = -1; uv = -1e300; both_rule = 0
            for c in range(K):
                th = t_a[cur, c] / (t_a[cur, c] + t_b[cur, c])
                fh = f_a[cur, c] / (f_a[cur, c] + f_b[cur, c])
                if split_stakes:
                    u = -cost[c] * cm + p * (1.0 - fh) * rp * k - (1.0 - p) * (1.0 - th) * rm * kq
                else:
                    u = -cost[c] * cm + k * (p * (1.0 - fh) * rp - (1.0 - p) * (1.0 - th) * rm)
                if u > uv or (u == uv and np.random.random() < 0.5):
                    uv = u; bestc = c
            if K == 2 and allow_both:
                thA = t_a[cur, 0] / (t_a[cur, 0] + t_b[cur, 0]); fhA = f_a[cur, 0] / (f_a[cur, 0] + f_b[cur, 0])
                thB = t_a[cur, 1] / (t_a[cur, 1] + t_b[cur, 1]); fhB = f_a[cur, 1] / (f_a[cur, 1] + f_b[cur, 1])
                if split_stakes:
                    u_and = p * (1.0 - fhA) * (1.0 - fhB) * rp * k - (1.0 - p) * (1.0 - thA) * (1.0 - thB) * rm * kq
                    u_or = p * (1.0 - fhA * fhB) * rp * k - (1.0 - p) * (1.0 - thA * thB) * rm * kq
                else:
                    u_and = k * (p * (1.0 - fhA) * (1.0 - fhB) * rp - (1.0 - p) * (1.0 - thA) * (1.0 - thB) * rm)
                    u_or = k * (p * (1.0 - fhA * fhB) * rp - (1.0 - p) * (1.0 - thA * thB) * rm)
                uboth = -cost[0] * cm - cost[1] * cm + (u_and if u_and >= u_or else u_or)
                if uboth > uv:
                    uv = uboth; bestc = 2; both_rule = 0 if u_and >= u_or else 1
            # ---- decision (same tie-breaking as the Julia code)
            if us > 0.0:
                action = 1 if us > uv else 2
            else:
                action = 0 if 0.0 >= uv else 2
            if eps_explore > 0.0 and np.random.random() < eps_explore:
                action = 2; bestc = np.random.randint(K)
            if action == 0:
                continue
            passed = True
            if action == 2:
                if bestc < 2:
                    flag = np.random.random() < (tpr_s[t, bestc] if fake else fpr_s[t, bestc])
                    ver_node[nver] = cur; ver_chk[nver] = bestc; ver_flag[nver] = flag; nver += 1
                    util -= cost[bestc] * cm
                    if after:
                        costpaid[cur] += cost[bestc] * cm
                    c_checks[t, bestc] += 1
                    passed = not flag
                else:
                    flA = np.random.random() < (tpr_s[t, 0] if fake else fpr_s[t, 0])
                    flB = np.random.random() < (tpr_s[t, 1] if fake else fpr_s[t, 1])
                    ver_node[nver] = cur; ver_chk[nver] = 0; ver_flag[nver] = flA; nver += 1
                    ver_node[nver] = cur; ver_chk[nver] = 1; ver_flag[nver] = flB; nver += 1
                    util -= cost[0] * cm + cost[1] * cm
                    if after:
                        costpaid[cur] += cost[0] * cm + cost[1] * cm
                    c_checks[t, 0] += 1; c_checks[t, 1] += 1; nb_both += 1; n_both[cur] += 1
                    if both_rule == 0:
                        passed = (not flA) and (not flB)
                    else:
                        passed = (not flA) or (not flB)
            if passed:
                n_sh += 1
                n_shared[cur] += 1
                if fake:
                    act_log[cur] = 2; util -= rm * kq
                else:
                    act_log[cur] = 1; util += rp * k
                for e in range(indptr[cur], indptr[cur + 1]):
                    nb = indices[e]
                    if stamp[nb] != t:
                        stamp[nb] = t
                        queue[qt] = nb; qdepth[qt] = d + 1; pedge[nb] = rev[e]; qt += 1
                        order[nvis] = nb; nvis += 1
        # ---- updates after the cascade
        for q in range(nvis):
            a = order[q]
            if feedback < 1.0:
                revealed[a] = np.random.random() < feedback
            else:
                revealed[a] = True
            if after:
                if act_log[a] == 1:
                    payoff[a] += rp * kk[a]; shares_true[a] += 1
                elif act_log[a] == 2:
                    payoff[a] -= rm * kl[a]; shares_fake[a] += 1
            if not revealed[a]:
                continue
            if fake:
                g_b[a] += 1.0
            else:
                g_a[a] += 1.0
            if all_senders:
                # variant (11 Nov): the record of every neighbour who shared this item is updated, not only the first
                # sender's (the trust entry of receiver a about neighbour nb sits at a's own CSR position of nb)
                for e2 in range(indptr[a], indptr[a + 1]):
                    nb2 = indices[e2]
                    if stamp[nb2] == t and act_log[nb2] > 0:
                        if fake:
                            tr_b[e2] += 1.0
                        else:
                            tr_a[e2] += 1.0
            else:
                pe = pedge[a]
                if pe >= 0:
                    if fake:
                        tr_b[pe] += 1.0
                    else:
                        tr_a[pe] += 1.0
        for q in range(nver):
            a = ver_node[q]; c = ver_chk[q]; fl = ver_flag[q]
            bad = (fake and not fl) or ((not fake) and fl)
            if n_checks[a, c] == 0:
                first_check[a, c] = t
                first_bad[a, c] = 1 if bad else 0
            n_checks[a, c] += 1
            if bad:
                n_bad[a, c] += 1
            if not revealed[a]:
                continue
            if fake:
                if fl:
                    t_a[a, c] += 1.0
                else:
                    t_b[a, c] += 1.0
            else:
                if fl:
                    f_a[a, c] += 1.0
                else:
                    f_b[a, c] += 1.0
        c_fake[t] = fake; c_size[t] = nvis; c_shares[t] = n_sh; c_both[t] = nb_both
        c_depth[t] = maxd; c_util[t] = util
        while sidx < S and snap_times[sidx] == t + 1:
            for i in range(N):
                for c in range(K):
                    th = t_a[i, c] / (t_a[i, c] + t_b[i, c]); fh = f_a[i, c] / (f_a[i, c] + f_b[i, c])
                    snapJ[sidx, i, c] = th - fh; snapT[sidx, i, c] = th; snapNC[sidx, i, c] = n_checks[i, c]
            sidx += 1
    if cost_in_payoff:
        for i in range(N):
            payoff[i] -= costpaid[i]
    return (deg, payoff, best, worst, costpaid, n_checks, n_both, n_bad, first_check, first_bad,
            n_recv, n_shared, shares_true, shares_fake,
            c_fake, c_size, c_shares, c_checks, c_both, c_depth, c_util,
            snapJ, snapT, snapNC)


# ----------------------------------------------------------------------------------------------
# python wrapper
# ----------------------------------------------------------------------------------------------

DEFAULTS = dict(
    N=300, m=3, p_tri=0.5, network="hk", graph=None,
    T=2000, burn=1000,
    phi=0.5, tpr=0.9, fpr=0.1, cost=1.0, rp=1.0, rm=2.0, gamma=1.0,
    kscale="none",                       # "none": stake = k_i**gamma; "mean": stake = (k_i/<k>)**gamma
    K=1, allow_both=True,
    ptrue_mode=0, seed_mode=0, kappa_tau=2.0, cost_in_payoff=False,
    prior_mode=0, prior_lo=0.3, prior_hi=0.7, kappa=2.0,
    tech_lo=None, tech_hi=None,          # optional separate range for checker beliefs (prior_mode 0/1)
    t0=0.6, f0=0.4, g0=0.5, n0=0.5,      # identical start (prior_mode 2)
    t_lo=None, t_hi=None, f_lo=None, f_hi=None, g_lo=None, g_hi=None,   # prior_mode 3: separate uniform ranges of prior means
    kappa_g=None, kappa_n=None,          # optional separate strengths for prevalence / neighbour beliefs
    tpr_sched=None, fpr_sched=None, phi_sched=None,
    snap_times=None, seed=0, eps_explore=0.0, feedback=1.0,
    gamma_loss=None,                     # separate exponent for the loss stake (None: same as gamma)
    int_start=0, int_end=0, int_mult=1.0, int_target="all",   # intervention window on checking costs
    all_senders=False,                   # True: update the record of every neighbour who shared an item (11 Nov check)
)


def make_graph(P, seed):
    if isinstance(P["graph"], str):
        import config
        return config.graph(P["graph"])
    if P["graph"] is not None:
        return P["graph"]
    if P["network"] == "hk":
        return holme_kim(P["N"], P["m"], P["p_tri"], seed)
    if P["network"] == "regular":
        return regular_random(P["N"], 2 * P["m"], seed)
    raise ValueError(P["network"])


def run(**kw):
    P = dict(DEFAULTS); P.update(kw)
    seed = int(P["seed"])
    rng = np.random.default_rng(seed + 7919)
    indptr, indices, rev = make_graph(P, seed)
    N = indptr.shape[0] - 1; E = indices.shape[0]; K = int(P["K"])
    deg = np.diff(indptr).astype(float)
    kdiv = 1.0 if P["kscale"] == "none" else float(deg.mean())
    kap = P["kappa"]; kg = P["kappa_g"] or kap; kn = P["kappa_n"] or kap
    lo, hi = P["prior_lo"], P["prior_hi"]
    tlo = P["tech_lo"] if P["tech_lo"] is not None else lo
    thi = P["tech_hi"] if P["tech_hi"] is not None else hi
    pm = P["prior_mode"]
    if pm in (0, 1):
        g = lo + rng.random(N) * (hi - lo)
        t = tlo + rng.random((N, K)) * (thi - tlo)
        f = tlo + rng.random((N, K)) * (thi - tlo)
        if pm == 0:     # Julia quirk: alpha and beta from independent means
            m1 = lo + rng.random(N) * (hi - lo); m2 = lo + rng.random(N) * (hi - lo)
            na, nb = m1 * kn, (1 - m2) * kn
        else:
            m1 = lo + rng.random(N) * (hi - lo)
            na, nb = m1 * kn, (1 - m1) * kn
    elif pm == 2:
        if P["g_lo"] is not None and P["g_hi"] is not None:      # spread-out starting beliefs about the base rate
            g = P["g_lo"] + rng.random(N) * (P["g_hi"] - P["g_lo"])
        else:
            g = np.full(N, P["g0"])
        t = np.tile(np.atleast_1d(np.asarray(P["t0"], float)), (N, 1)) if np.ndim(P["t0"]) else np.full((N, K), P["t0"])
        f = np.tile(np.atleast_1d(np.asarray(P["f0"], float)), (N, 1)) if np.ndim(P["f0"]) else np.full((N, K), P["f0"])
        na = np.full(N, P["n0"] * kn); nb = np.full(N, (1 - P["n0"]) * kn)
    elif pm == 3:       # heterogeneous: prior means uniform on separate ranges, identical strength
        glo = P["g_lo"] if P["g_lo"] is not None else P["g0"]; ghi = P["g_hi"] if P["g_hi"] is not None else P["g0"]
        g = glo + rng.random(N) * (ghi - glo)
        t = P["t_lo"] + rng.random((N, K)) * (P["t_hi"] - P["t_lo"])
        f = P["f_lo"] + rng.random((N, K)) * (P["f_hi"] - P["f_lo"])
        na = np.full(N, P["n0"] * kn); nb = np.full(N, (1 - P["n0"]) * kn)
    else:
        raise ValueError(pm)
    g_a, g_b = g * kg, (1 - g) * kg
    t_a, t_b = t * kap, (1 - t) * kap
    f_a, f_b = f * kap, (1 - f) * kap
    src = np.repeat(np.arange(N), np.diff(indptr))
    if P["ptrue_mode"] == 2:
        tr_a = np.zeros(E); tr_b = np.zeros(E)
    else:
        tr_a = na[src].copy(); tr_b = nb[src].copy()
    T = int(P["T"])
    phi_s = np.full(T, float(P["phi"])) if P["phi_sched"] is None else np.asarray(P["phi_sched"], float)
    tp = np.atleast_1d(np.asarray(P["tpr"], float)); fp = np.atleast_1d(np.asarray(P["fpr"], float))
    if tp.size == 1: tp = np.repeat(tp, K)
    if fp.size == 1: fp = np.repeat(fp, K)
    tpr_s = np.tile(tp, (T, 1)) if P["tpr_sched"] is None else np.asarray(P["tpr_sched"], float).reshape(T, K)
    fpr_s = np.tile(fp, (T, 1)) if P["fpr_sched"] is None else np.asarray(P["fpr_sched"], float).reshape(T, K)
    cost = np.atleast_1d(np.asarray(P["cost"], float))
    if cost.size == 1: cost = np.repeat(cost, K)
    snaps = np.array([T] if P["snap_times"] is None else sorted(set(P["snap_times"]) | {T}), dtype=np.int64)
    J0 = t - f
    g_init = g.copy()
    split = P["gamma_loss"] is not None and float(P["gamma_loss"]) != float(P["gamma"])
    gamma_l = float(P["gamma"]) if P["gamma_loss"] is None else float(P["gamma_loss"])
    tgt = P["int_target"]
    if isinstance(tgt, str):
        if tgt == "all":
            elig = np.ones(N, dtype=np.bool_)
        elif tgt == "below_median":
            elig = deg < np.median(deg)
        else:
            raise ValueError(tgt)
    else:
        elig = np.asarray(tgt, dtype=np.bool_)
    out = _simulate(indptr, indices, rev, T, int(P["burn"]), phi_s, tpr_s, fpr_s, cost,
                    float(P["rp"]), float(P["rm"]), float(P["gamma"]), kdiv,
                    int(P["ptrue_mode"]), int(P["seed_mode"]), float(P["kappa_tau"]), bool(P["allow_both"]),
                    bool(P["cost_in_payoff"]),
                    g_a, g_b, t_a, t_b, f_a, f_b, tr_a, tr_b, na.copy(), nb.copy(), snaps, seed,
                    float(P["eps_explore"]), float(P["feedback"]),
                    gamma_l, split, int(P["int_start"]), int(P["int_end"]), float(P["int_mult"]), elig,
                    bool(P["all_senders"]))
    keys = ["deg", "payoff", "best", "worst", "costpaid", "n_checks", "n_both", "n_bad", "first_check", "first_bad",
            "n_recv", "n_shared", "shares_true", "shares_fake",
            "c_fake", "c_size", "c_shares", "c_checks", "c_both", "c_depth", "c_util",
            "snapJ", "snapT", "snapNC"]
    R = dict(zip(keys, out))
    R.update(g_a=g_a, g_b=g_b, t_a=t_a, t_b=t_b, f_a=f_a, f_b=f_b, tr_a=tr_a, tr_b=tr_b,
             J0=J0, snap_times=snaps, N=N, K=K, params=P, cost_vec=cost,
             indptr=indptr, indices=indices, kdiv=kdiv)
    R["that"] = t_a / (t_a + t_b); R["fhat"] = f_a / (f_a + f_b); R["J"] = R["that"] - R["fhat"]
    R["pi"] = g_a / (g_a + g_b)
    kk = (R["deg"].astype(float) / kdiv) ** P["gamma"]
    kl = (R["deg"].astype(float) / kdiv) ** gamma_l
    R["stake"] = kk; R["stake_loss"] = kl; R["pi0"] = g_init; R["elig"] = elig
    # checking threshold per agent/checker (normal costs): checking can pay only if J > c (1/(g s_gain) + 1/(l s_loss))
    R["theta"] = cost[None, :] * (1.0 / (P["rp"] * kk[:, None]) + 1.0 / (P["rm"] * kl[:, None]))
    return R
