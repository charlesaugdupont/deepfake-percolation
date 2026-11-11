# Model v2: costly checking and the trust bifurcation (paper version)

Fast Python/numba re-implementation of `run_experiment.jl`, plus the experiments and figures for the paper.
Everything that differs from the Julia model (v1) is a switch in `engine.run()`, so v1 can still be reproduced exactly.

## Files

| File | What it does |
|---|---|
| `engine.py` | The model (numba). One call to `run(**params)` = one simulation. |
| `config.py` | Paper defaults (model v2), reference checkers R1–R4, network loader. |
| `metrics.py` | Bimodality coefficient, two-group split (Otsu), dip test helper. |
| `sweep.py` | Runs many simulations in parallel and summarises each run (`summarize`). |
| `trapmodel.py` | One-agent first-passage ("absorbing barrier") model of the distrust trap. |
| `gsa.py` | Latin hypercube design and PAWN indices with bootstrap CIs and a dummy input. |
| `validate_vs_julia.py` | Checks the engine (v1 switches) against `data/sweep_2026_04_10_1044`. |
| `exp_*.py` | Experiments; each writes a parquet/pickle to `out/` (list in `RESULTS.md`, section 6). |
| `chunk_runner.py`, `merge_chunks.py` | Resumable, time-boxed runner for the big sweeps (phase diagrams, GSA, robustness), and merging its chunks. |
| `make_rewired.py` | Degree-preserving rewiring of the Twitter network (robustness). |
| `fig_*.py` | Paper figures from those outputs; `figstyle.py` = shared style (PNAS widths). |
| `make_paper_figures.py` | Renders all paper and SI figures (PDF + PNG); `DFP_OUT` points it at the outputs (e.g. `../data/v2`). |
| `RESULTS.md` | Set-up decisions, main numbers, draft captions, open decisions. |

## Model v2 in one paragraph

Agents sit on an undirected social network. Each time step one item enters at a uniformly random agent; it is fake
with probability φ. An agent who receives the item from neighbour *j* estimates the probability that it is genuine as
p = (n_genuine,ij + 2π_i) / (n_ij + 2): neighbour *j*'s track record, shrunk towards the agent's belief π_i about the
share of genuine items (the seed has no sender and uses π_i). It then picks the action with the highest expected
payoff: **share** (+g·s_i if genuine, −l·s_i if fake, with s_i = k_i/⟨k⟩ the agent's audience relative to the mean
degree), **check** (pay c, then share unless the checker flags the item) or **discard** (0). The checker flags fakes
with probability TPR and genuine items with probability FPR. After the cascade the truth is revealed to everyone it
reached: they update π_i and their trust in the sender (Beta counts); agents who checked also update their beliefs
about the checker's TPR and FPR (Beta counts). Checking can only pay if the believed informativeness
Ĵ_i = t̂_i − f̂_i exceeds θ_i = (c/s_i)(1/g + 1/l); below θ_i an agent never checks again and so never learns again
(absorbing barrier).

## Changes relative to v1 (`run_experiment.jl`)

| | v1 | v2 (paper) | switch |
|---|---|---|---|
| P(genuine) | τπ/(τπ + (1−τ)(1−π)) (base rate counted twice) | (n_genuine + 2π)/(n + 2) | `ptrue_mode=2` |
| Seed | uses a fixed random "base trust" | uses π_i | `seed_mode=1` |
| Stakes | ∝ k_i | ∝ k_i/⟨k⟩ (same model, rescaled cost) | `kscale="mean"` |
| Priors about the checker | means U[0.3, 0.7], strength 2, per agent | identical: TPR ~ Beta(2,1), FPR ~ Beta(1,2) | `prior_mode=2` |
| Prior about share genuine | means U[0.3, 0.7] | Beta(1,1) | `g0=0.5, kappa_g=2` |
| Checker | swept | R2 = best commercial AI detector (0.80, 0.10) | `tpr, fpr` |
| Welfare | excludes checking costs | includes them | `cost_in_payoff=True` |
| Bifurcation metric | bimodality coefficient | 4 × Hartigan's dip; share locked out (Ĵ ≤ θ) | `sweep.summarize` |

Extensions: two checkers (`K=2`: check A, check B or both at cost c_A + c_B; after checking both, share under the
better of the AND / OR rules, which is the Bayes-optimal rule when both are believed informative), exploration
(`eps_explore`), partial feedback (`feedback`), stakes exponent (`gamma`), time-varying φ / TPR / FPR (`*_sched`).

## Regimes worth knowing about

Besides the split, two edge cases show up in the sweeps: (i) a **cold start** of ~100–300 items in which nobody checks,
because agents first have to learn that most items are genuine (their base-rate belief starts at 0.5); and (ii) a
**nothing-circulates** region (high l/g, many fakes): with the shared starting trust, checking does not pay at the
prevailing base rate and sharing unchecked is too risky, so nobody shares or checks. Figures hatch (ii) and exclude it
from averages; `fig_phase.dead_cells` defines it (reach of genuine items < 5%). Both depend on the base-rate prior
(`g0`, `kappa_g`); see `RESULTS.md`, open decision 1.

## Running

`config.py` reads `DFP_NETDIR` (folder with the network edge lists, default `../data/networks`) and `DFP_NPROC`
(worker processes). A Twitter run of 2,000 items takes ~5–10 s per core. Long sweeps: `python3 chunk_runner.py phase
twitter <seconds> out` repeatedly until it exits with 0; then `python3 merge_chunks.py out/phase_twitter out/phase_twitter.parquet`.

## Validation

With the v1 switches the engine reproduces the April Julia sweep: 40 LHS points × 50 seeds, eight outputs; per-point
means correlate 0.996–0.9998 with Julia's and differences are within sampling noise (`validation_summary.csv`, `validation_points.csv`; Fig. S5).
