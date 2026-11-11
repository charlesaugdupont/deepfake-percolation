# Model code: costly checking and the split of trust

Python/numba agent-based model, plus every experiment and figure of the paper. One call to `engine.run(**params)` is one
simulation. The paper's settings are in `config.py`; every variant used in the paper is a keyword of `engine.run()`.

The original Julia model (v1), its analysis scripts and the v2 scripts were removed on 11 Nov 2026 and remain in the
git history (commit `96220d5`). `engine.run()` still reproduces v1 with its v1 switches (below), and
`validate_vs_julia.py` checks that against the summary of the April 2026 Julia sweep.

## Files

**Model**

| File | What it does |
|---|---|
| `engine.py` | The model (numba). `run(**params)` = one simulation; returns beliefs, groups, cascades and snapshots. |
| `config.py` | Paper defaults (`BASE`, `CHECKER`, `MAIN`), reference checkers, network loader (`DFP_NETDIR`). |
| `sweep.py` | Summary measures of one simulation (`summarize`): reach, welfare, groups, the dip. |
| `metrics.py` | Helpers for the split of trust (dip test, bimodality coefficient, two-group split). |
| `trapmodel.py` | One-agent model of the trap (first passage below the threshold); one setting for every figure. |
| `synthnets.py` | Random networks with a set mean degree and degree inequality (Fig. S11). |
| `make_rewired.py` | Degree-preserving rewiring of the Twitter network (Fig. S8). |
| `gsa.py` | Latin-hypercube design and PAWN sensitivity indices (Fig. S10). |

**Experiments**

| File | What it does |
|---|---|
| `exp_v3.py` | Every experiment of the paper as a list of jobs (`EXPERIMENTS`). |
| `run_v3.py` | Runs them, restartable (chunks in `../data/v3/<experiment>/`); `status`, `collect`, `--tiny` test. |
| `run_on_mac_v3.sh` | Runs all experiments on a Mac (own environment in `~/.venvs/dfp`, keeps the Mac awake). |

**Figures** (`make_figures_v3.py` renders all of them, PNG + PDF, from `../data/v3` into `../figures/paper_v3`)

| File | Figure |
|---|---|
| `figstyle.py` | Shared style, colours, widths and CI helpers |
| `fig_mechanism.py` | Fig. 1, mechanism and threshold |
| `fig_one_checker.py` | Fig. 2, one checker over time |
| `fig_two_checkers.py` | Fig. 3, two checkers |
| `fig_regimes.py`, `fig_consequences.py` | Fig. 4 and Fig. S9; one-agent model curves (also used by Fig. S8) |
| `si_starting_beliefs.py` | Fig. S1 |
| `si_checker_quality.py` | Fig. S2 |
| `si_stakes.py` | Fig. S3 and Table S1 |
| `si_robustness.py` | Fig. S4 |
| `si_other_settings.py` | Figs. S5 and S7 |
| `si_two_checkers_networks.py` | Fig. S6 |
| `si_networks.py` | Fig. S8 |
| `si_sensitivity.py` | Fig. S10 |
| `si_inequality.py` | Fig. S11 |
| `si_survey.py` | Fig. S12 (survey data from `gss_press.py`) |
| `si_cost_cut.py` | Temporary cost cut; dropped from the paper, not drawn by `make_figures_v3.py` |

**Data tools and notes**

| File | What it does |
|---|---|
| `gss_press.py` | Extracts confidence in the press and party from the GSS file (`../data/gss/raw/`) for Fig. S12. |
| `pew_dip_check.py` | Pew ideology check behind the comparison with Thurner et al. (discussion). |
| `liss_checks.py` | LISS panel aggregates (not in the paper; the microdata stay on this machine). |
| `validate_vs_julia.py` | Engine with v1 switches vs the April Julia sweep (`../figures/seminar_2026_10/data/sim_summary.csv`). |
| `RESULTS_v3.md` | Every decision, the numbers behind each figure, draft captions and open points. |

Renamed on 11 Nov 2026 (old → new): `v3_fig1_mechanism` → `fig_mechanism`, `v3_fig2_time` → `fig_one_checker`,
`v3_fig_two_time` → `fig_two_checkers`, `v3_fig3_regimes` → `fig_regimes`, `v3_fig4_consequences` → `fig_consequences`,
`v3_figS_starting` + `v3_fig5_beliefs` → `si_starting_beliefs`, `v3_figS_checkers` → `si_checker_quality`,
`v3_figS_stakes` → `si_stakes`, `v3_figS_robust` → `si_robustness`, `v3_figS_settings` → `si_other_settings`,
`v3_fig6_two` → `si_two_checkers_networks`, `v3_figS_networks` → `si_networks`, `v3_figS_gsa` → `si_sensitivity`,
`v3_figS_inequality` → `si_inequality`, `v3_figS_gss` → `si_survey`, `v3_figS_intervention` → `si_cost_cut`; the plotting
helpers of the v2 script `fig_two.py` moved into `figstyle.py`.

## Switches in `engine.run()`

Paper values come from `config.BASE` and `config.CHECKER`:

| Keyword | Paper value | Meaning |
|---|---|---|
| `ptrue_mode`, `kappa_tau` | 2, 2.0 | p = sender's record shrunk towards πᵢ with weight κₛ = `kappa_tau` |
| `seed_mode` | 1 | the seed (no sender) uses p = πᵢ |
| `kscale`, `gamma` | "mean", 1 | stake sᵢ = (kᵢ/⟨k⟩)^γ |
| `prior_mode`, `t0`, `f0`, `kappa` | 2, 2/3, 1/3, 3 | identical starting beliefs about the checker (J₀ = 1/3) |
| `g_lo`, `g_hi`, `kappa_g` | 0.01, 0.99, 2 | starting beliefs about the share of genuine items, spread out, weight 2 |
| `rp`, `rm`, `cost` | 1, 2, 0.05 | g, ℓ and c (main setting); only c/g and ℓ/g matter |
| `tpr`, `fpr` | 0.80, 0.10 | the checker (per checker with `K=2`) |
| `cost_in_payoff` | True | welfare includes checking costs |
| `T`, `burn` | 2000, 1000 | items per simulation; flows measured after the burn-in |

Variants: `K=2` (two checkers; `allow_both`), `eps_explore` (random checks), `feedback` (share of reached agents who
learn the truth), `gamma_loss` (separate loss stake), `tpr_sched` / `fpr_sched` / `phi_sched` (time-varying),
`int_start` / `int_end` / `int_mult` / `int_target` (temporary cost cut), `all_senders` (update the record of every
neighbour who shared an item). v1 switches: `ptrue_mode=0`, `seed_mode=0`, `prior_mode=0`, `kscale="none"`,
`cost_in_payoff=False`.

## Running

`config.py` reads `DFP_NETDIR` (the network edge lists; default `../data/networks`) and `DFP_NPROC` (worker processes).
On the Twitter network one simulation of 2,000 items takes about 5–15 s per core. How to run one simulation, every
experiment and every figure: see the top-level `README.md`.
