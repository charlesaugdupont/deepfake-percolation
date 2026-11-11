"""Model configuration for the paper experiments (model v2).

Changes relative to run_experiment.jl (v1), all switchable in engine.run():
  * posterior: p = (n_genuine_ij + kappa_tau * pi_i) / (n_ij + kappa_tau)  (sender record shrunk to the base rate;
    removes the double counting of the base rate)                                      -> ptrue_mode=2
  * seed (no sender): p = pi_i (unknown external source treated as uninformative)        -> seed_mode=1
  * stakes relative to mean degree: payoffs of sharing scale with k_i/<k>, so the checking payoff c is in units of
    an average agent's gain from sharing one genuine item                              -> kscale="mean"
  * priors about the checker: identical for all agents, TPR ~ Beta(2,1), FPR ~ Beta(1,2) (the marginals of a uniform
    prior over all better-than-chance checkers)
  * v3: starting beliefs about the share of genuine items are spread out: each agent's prior mean ~ U(0.01, 0.99),
    strength 2 (a weak individual guess; removes the cold start without assuming people know the true share)
  * welfare includes checking costs                                                    -> cost_in_payoff=True
"""
import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

_HERE = os.path.dirname(os.path.abspath(__file__))
NETDIR = os.environ.get("DFP_NETDIR") or ("/home/claude/netdata" if os.path.isdir("/home/claude/netdata")
                                          else os.path.join(_HERE, "..", "data", "networks"))
NPROC = int(os.environ.get("DFP_NPROC", "2"))
_GRAPHS = {}


NETFILES = {
    "twitter": "twitter_simsom/derived/edges.csv",          # Nikolov et al. 2021 news sharers, mutual follows, LCC
    "twitter_rewired": "twitter_rewired/edges.csv",          # degree-preserving randomisation of "twitter" (make_rewired.py)
    "fb_jhu": "fb100_linkx/derived/Johns_Hopkins55/edges.csv",
    "fb_cornell": "fb100_linkx/derived/Cornell5/edges.csv",
    "fb_reed": "fb100_linkx/derived/Reed98/edges.csv",
    "ego_facebook": "ego_facebook/derived/edges.csv",
    "lastfm": "lastfm/edges.csv", "twitch": "twitch/edges.csv",
}


def graph(name):
    """Real network by short name (fixed across runs), or None / 'hk' for a Holme-Kim graph generated per seed.
    CSR arrays are cached as .npz next to the edge list, so workers load them quickly."""
    if name is None or name.startswith("hk"):
        return None
    if name not in _GRAPHS:
        path = os.path.join(NETDIR, NETFILES.get(name, f"{name}/edges.csv"))
        cache = path.replace(".csv", "_csr.npz")
        if os.path.exists(cache):
            z = np.load(cache); _GRAPHS[name] = (z["indptr"], z["indices"], z["rev"])
        else:
            g = E.edgelist_to_csr(pd.read_csv(path).values)
            np.savez(cache, indptr=g[0], indices=g[1], rev=g[2]); _GRAPHS[name] = g
    return _GRAPHS[name]


PRIOR_IDENTICAL = dict(prior_mode=2, t0=2 / 3, f0=1 / 3, kappa=3.0, g0=0.5, kappa_g=2.0, g_lo=0.01, g_hi=0.99)
PRIOR_V2_BASERATE = dict(g_lo=None, g_hi=None, g0=0.5)       # v2: everyone starts at 0.5 (uninformed, identical)
PRIOR_V1 = dict(prior_mode=1, prior_lo=0.3, prior_hi=0.7, kappa=2.0, g0=0.5, kappa_g=2.0)

BASE = dict(T=2000, burn=1000, rp=1.0, rm=2.0, kscale="mean", gamma=1.0, ptrue_mode=2, seed_mode=1, kappa_tau=2.0,
            cost_in_payoff=True, **PRIOR_IDENTICAL)

# Reference checkers (see checker_calibration.md): R2 = best commercial AI deepfake detector in the wild (Deepfake-Eval-2024)
CHECKERS = {"R1": (0.55, 0.45), "R2": (0.80, 0.10), "R3": (0.85, 0.15), "R4": (0.43, 0.32)}
# Added 5 Nov to fill the middle of the quality range (checker_calibration.md, evidence table): R5 = best fine-tuned
# open-source video detector on in-the-wild media (Chandra et al. 2025, Table S9; J = 0.48); R6 = best lab-trained
# audio deepfake detector on in-the-wild audio (RawNet2 at its equal-error point, Mueller et al. 2022; J = 0.32)
CHECKERS_EXTRA = {"R5": (0.65, 0.17), "R6": (0.66, 0.34)}
CHECKER = dict(tpr=0.80, fpr=0.10)


def grid(net):
    """Parameter grids. The dense Twitter network (~15 s per run) gets a coarser grid that reaches lower costs."""
    if net is not None and net.startswith("twitter"):
        return dict(phis=np.round(np.arange(0.05, 0.96, 0.1), 3),
                    costs=np.round(np.exp(np.linspace(np.log(0.001), np.log(1.0), 12)), 5))
    return dict(phis=np.round(np.arange(0.05, 0.951, 0.05), 3),
                costs=np.round(np.exp(np.linspace(np.log(0.005), np.log(1.0), 15)), 5))


# ---------------------------------------------------------------------------------------------------------------
# v3 designs: round single values, and a hexagonal lattice for the phase diagrams
# ---------------------------------------------------------------------------------------------------------------
MAIN = dict(phi=0.3, cost=0.05, rm=2.0)                                   # the illustrative "main setting"
C125 = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]        # round costs for line plots


def hex_lattice(rows_per_decade=6, dphi=0.05, cmin=1e-3, cmax=1.0):
    """Cells of a hexagonal lattice over (share of fakes, log10 cost): rows evenly spaced in log10(c/g); odd rows
    shifted by half a phi step. Returns a list of (row, phi, cost). Default: 19 rows x 19/18 columns = 352 cells."""
    nrows = int(round(np.log10(cmax / cmin) * rows_per_decade)) + 1
    cells = []
    for r in range(nrows):
        c = float(np.round(10 ** (np.log10(cmin) + r / rows_per_decade), 8))
        phis = np.arange(0.05, 0.951, dphi) if r % 2 == 0 else np.arange(0.05 + dphi / 2, 0.951 - dphi / 2 + 1e-9, dphi)
        cells += [(r, float(np.round(p, 4)), c) for p in phis]
    return cells


def kwargs(net="lastfm", **over):
    kw = dict(BASE); kw.update(CHECKER)
    if net is None or net.startswith("hk"):
        kw.update(N=1000, m=3, p_tri=0.5)
    else:
        kw.update(graph=net)          # resolved (and cached) inside each worker by engine.make_graph
    kw.update(over)
    return kw
