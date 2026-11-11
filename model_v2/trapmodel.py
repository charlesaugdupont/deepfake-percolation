"""Single-agent first-passage model of the distrust trap.
An agent checks a sequence of items; each checked item is fake with probability q. Beliefs about the checker are Beta
counts (a,b) for the hit rate and (c,d) for the false-alarm rate. The agent stops checking for good the first time its
believed informativeness J = a/(a+b) - c/(c+d) falls to or below its threshold theta. Returns P(trapped within n checks)."""
import numpy as np
from numba import njit

# one setting for every figure (harmonised 11 Nov; before: 400 checks and 2,000-3,000 paths for Figs. 4, S8, S9,
# 500 checks and 6,000 paths for Fig. 1)
N_CHECKS, N_MC, N_GRID = 500, 6000, 80


@njit(cache=True)
def _p_trap(theta, a0, b0, c0, d0, t, f, q, n_checks, n_mc, seed):
    np.random.seed(seed)
    trapped = 0
    for m in range(n_mc):
        a = a0; b = b0; c = c0; d = d0
        if a / (a + b) - c / (c + d) <= theta:
            trapped += 1
            continue
        for k in range(n_checks):
            if np.random.random() < q:          # fake item
                if np.random.random() < t:
                    a += 1.0
                else:
                    b += 1.0
            else:
                if np.random.random() < f:
                    c += 1.0
                else:
                    d += 1.0
            if a / (a + b) - c / (c + d) <= theta:
                trapped += 1
                break
    return trapped / n_mc


def p_trap(theta, prior=(2.0, 1.0, 1.0, 2.0), t=0.85, f=0.10, q=1 / 3, n_checks=N_CHECKS, n_mc=N_MC, seed=0):
    th = np.atleast_1d(np.asarray(theta, float))
    return np.array([_p_trap(x, *prior, t, f, q, n_checks, n_mc, seed) for x in th])


def paths(n_paths, n_checks, prior=(2.0, 1.0, 1.0, 2.0), t=0.80, f=0.10, q=0.25, seed=1):
    """Belief trajectories J_k (k = 0..n_checks) of agents who keep checking regardless (no absorption applied)."""
    rng = np.random.default_rng(seed)
    a0, b0, c0, d0 = prior
    out = np.zeros((n_paths, n_checks + 1))
    for i in range(n_paths):
        a, b, c, d = a0, b0, c0, d0
        out[i, 0] = a / (a + b) - c / (c + d)
        for k in range(n_checks):
            if rng.random() < q:
                if rng.random() < t: a += 1
                else: b += 1
            else:
                if rng.random() < f: c += 1
                else: d += 1
            out[i, k + 1] = a / (a + b) - c / (c + d)
    return out


def predicted_locked(deg, cost, lam, q, prior=(2.0, 1.0, 1.0, 2.0), t=0.80, f=0.10, g=1.0, n_checks=N_CHECKS, n_mc=N_MC,
                     n_grid=N_GRID):
    """Population share locked out predicted by the first-passage model: mean over agents of P_trap(theta_i),
    theta_i = (c / s_i) (1/g + 1/l), s_i = k_i / <k>. P_trap is computed on a grid of thresholds below the starting
    trust and interpolated (it is 1 at or above the starting trust)."""
    deg = np.asarray(deg, float)
    th = (cost / (deg / deg.mean())) * (1 / g + 1 / (lam * g))
    J0 = prior[0] / (prior[0] + prior[1]) - prior[2] / (prior[2] + prior[3])
    grid = np.linspace(0.0, J0 - 1e-9, n_grid)
    pg = p_trap(grid, prior=prior, t=t, f=f, q=q, n_checks=n_checks, n_mc=n_mc)
    pt = np.where(th >= J0, 1.0, np.interp(np.minimum(th, J0), grid, pg))
    return float(pt.mean())


def predicted_groups(deg, cost, lam, q, prior=(2.0, 1.0, 1.0, 2.0), t=0.80, f=0.10, g=1.0, n_checks=N_CHECKS, n_mc=N_MC,
                     n_grid=N_GRID):
    """As predicted_locked, but returns (share locked out, share for whom checking would pay (theta_i < t - f) and who
    are locked out)."""
    deg = np.asarray(deg, float)
    th = (cost / (deg / deg.mean())) * (1 / g + 1 / (lam * g))
    J0 = prior[0] / (prior[0] + prior[1]) - prior[2] / (prior[2] + prior[3])
    grid = np.linspace(0.0, J0 - 1e-9, n_grid)
    pg = p_trap(grid, prior=prior, t=t, f=f, q=q, n_checks=n_checks, n_mc=n_mc)
    pt = np.where(th >= J0, 1.0, np.interp(np.minimum(th, J0), grid, pg))
    return float(pt.mean()), float((pt * (th < t - f)).mean())
