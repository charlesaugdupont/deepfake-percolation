# Results so far: model v2 on the Twitter news-sharer network

Status 2 November 2026. Figures: `figures/paper_v2/` (PDF + PNG, PNAS widths). Data behind them: `data/v2/`.
Code: this folder (`make_paper_figures.py` re-renders everything from `data/v2`, see "Reproduce" at the end).

## 1. Set-up decisions (agreed)

| Choice                                           | Setting                                                                                                                                                   |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Starting beliefs about the checker               | identical for everyone, better than chance: TPR ~ Beta(2,1), FPR ~ Beta(1,2) → starting trust Ĵ₀ = t̂ − f̂ = 1/3                                          |
| Starting belief about the share of genuine items | Beta(1,1) (uninformed; see open decision 1)                                                                                                               |
| Payoffs                                          | share: +g·s_i if genuine, −l·s_i if fake; check: −c, then share unless flagged; discard: 0. Everything is a ratio to g: l/g ∈ {1, 2, 4}, c/g ∈ [0.001, 1] |
| Stakes                                           | s_i = k_i/⟨k⟩ (audience relative to the average)                                                                                                          |
| Checker                                          | R2, best commercial AI detector: TPR 0.80, FPR 0.10 (actual J = 0.70). R1, R3, R4 as a comparison panel                                                   |
| Two checkers                                     | check A, check B, or both (cost c_A + c_B), then share/discard; ties broken at random                                                                     |
| Bifurcation metrics                              | 4 × Hartigan's dip (shape of the trust distribution) and share locked out, Ĵ_i ≤ θ_i (who is affected); bimodality coefficient in SI                      |
| Main network                                     | Twitter news sharers (Nikolov et al. 2021; mutual-follow largest component, N = 14,128, ⟨k⟩ = 252.5)                                                      |
| Robustness networks                              | degree-preserving rewired Twitter, Facebook100 Johns Hopkins (N = 5,157), Twitch EN (N = 7,126), Holme–Kim (N = 1,000, m = 3)                             |
| Runs                                             | 2,000 items (1,000 burn-in for the reach/welfare averages); 3–20 runs per parameter point; bands and bars are 95% CIs over runs                           |

## 2. Main findings

**Mechanism (Fig. 1).** Checking can only pay when the believed informativeness of the checker exceeds a threshold that
scales with cost over audience: Ĵ_i > θ_i = (c/s_i)(1/g + 1/l). An agent whose belief falls below θ_i never checks again,
so never learns again: an absorbing barrier. Small accounts have high thresholds. A one-agent first-passage model
(random walk of Beta updates from checks, absorbed at θ) predicts the locked-out share by degree with no fitted
parameters, at every cost, loss ratio and network we ran (Fig. 1D, 3C, S2B).

notes:
- the subfigure that shows a mini network with shared/checked/discarded colored nodes and edges sort of cuts off the legend in panel A
- s_i is the agent/s degree over the average degree in the network, is that right?
- Do we think it's an issue that there is such a strong absorbing state? maybe we should allow for agents below threshold to randomly check, with small chance? not sure if this is realistic or would make the analysis more interesting
- panel B: so basically agents with a smaller audience have a higher threshold theta for checking?
- panel C: nice, this is quite clear
- panel D: i'm confused about "first-passage model" and what is being shown here. this is the proportion of agents that are locked out, as a function of what, the number of neighbours that they have? By the way, the error bars are near impossible to see here, not sure that it's a very effective visualization. also why such random values of c/g with three decimals? it's a little ugly to look at

**The split over time (Fig. 2).** Nothing happens for the first ~100–300 items (agents first have to learn that most
items are genuine), then the population splits within a few hundred items and freezes. At c/g = 0.043, φ = 0.3,
l/g = 2 (20 runs): 57.5% still check and their trust converges on the checker's actual J (0.70); 39% never checked and
stay at the starting value 1/3; 3.5% checked, had bad luck and are locked out; 4·dip = 0.76. At c/g = 0.012 the split
is weaker (4·dip 0.45, 78% ever checked); at c/g = 0.15 too (0.46, 38%).

notes:
- do you think the ranges of values for c, g, and l were well selected and are we exploring a large enough region of the parameter space? or too small?
- it's a bit jarring how most agents just never check and remain stuck, or "locked out"
- I'm not totally understanding what the 4·dip metric is. can you help to summarize it mathematically? why the multiplication by 4? at what value does it indicate significant multimodality?
- in panel A, the legend indicates that there is an orange line for agents who checked and got locked out, but we really cannot see this line at all. I find this panel a bit confusing overall, though i do see what you're trying to do (show the moving belief distribution over time, with average belief lines)

**Where the split happens (Fig. 3).** Along cost there are three regimes: almost everyone checks and learns
(c/g ≲ 0.003), a split (≈ 0.003–0.3), and few can afford to check (≳ 0.3). The locked-out share depends almost only on
cost (and the degree distribution), hardly on the share of fakes φ or the loss ratio. 4·dip > 0.2 in 67/120 cells at
l/g = 1, 58/120 at l/g = 2, 12/120 at l/g = 4; maximum ≈ 0.78. A fourth region (hatched): **nothing circulates**.
With high losses and many fakes, the shared starting trust in the checker makes checking unattractive at the prevailing
base rate, and sharing unchecked is too risky, so nobody shares or checks (reach of genuine items < 5%): l/g = 4 at
φ ≥ 0.35 (all costs) and at high cost; l/g = 2 at φ = 0.95. This is the population-level version of the lock-out.

notes:
- these heatmaps have quite low "resolution" and look very pixelly, does this mean maybe we need more runs / a finer parameter grid design?
- in panel C, again, what is first-passage model?  error bars don't really show up for l/g=1 and l/g=2, are they just very small?
- Overall, do you think it might be confusing that we keep writing (in legends, caption titles, and axes) ratios of parameters? I mean, I do think the ratio is what's important right, since a simulation with c=1, l=2, g=1.5 should yield identical results to c=2, l=4, g=3... but maybe we need to find a way to make things a bit more intuitive?

**Two identical checkers (Fig. 4; Holme–Kim in S4).** At c/g = 0.043: 47% use both, 7.1% only A, 7.0% only B, 39%
neither. Among agents who check, 23% never tried the other checker and 83% of checks go to the preferred one;
corr(Ĵ_A, Ĵ_B) = +0.19. Exclusive loyalty rises with cost (4% at c/g = 0.001 → 57% at 0.53) and sits in a middle
band of audience sizes (k ≈ 30–300): accounts that can afford one checker but not both. On Holme–Kim the loyalties are
sharper (exclusivity ≈ 0.5, corr down to −0.50 at c/g ≈ 0.08). With a cheap AI detector (cost c) and an accurate
professional checker (cost 3c, TPR 0.95, FPR 0.03) agents sort by audience: small accounts check nothing,
middle-sized ones use only the AI detector, large ones the professional.

notes:
- i find panel A quite interesting, some agents trust checker A but not B, and some agents trust checker B but not A
- panel B: it's hard to see the blue line for "only A", I guess it's right underneath the orange line of "only B"? nice message though: as verification gets more costly, agents essentially use neither checker. most agents usually check with both (did you make sure that checking with both means paying more verification cost?), and some use only one checker, though very rarely. I'm curious: do agents start with identical beliefs about both checkers in an unbiased way?
- panel D looks a little bit cropped somehow at the top, the circle markers are cut in half horizontally. not really understanding what this figure is saying. for particular amounts of neighbors agents rely on the AI checker, but not the professional checker, but with enough neighbours, agents prefer to use the professional checker and pay more? interesting.
- Maybe we can organize the panels a little better here.
- I wonder to what extent it would be interesting to do an analysis or mini experiment to see what happens as the number of checkers gets quite large, and the distribution of agent accuracies gets also quite varied. But maybe that's for another work

**Starting beliefs and checker quality (Fig. 5).** With starting trust 0 nobody ever checks. The split is strongest
when the starting trust is close to the typical threshold (ridge along J₀ ≈ θ of the average agent); prior strength
matters little. Spreading starting beliefs (uniform ± w around the shared means) quickly erodes the dip (0.77 → 0.25
at w = 0.1) but barely changes the locked-out share (0.42 → 0.44): **the locked-out share is the robust headline, the
dip is shape-sensitive.** With per-agent priors from U(0,1) the locked-out share is 0.77 (4·dip 0.29); with the
seminar priors U[0.3, 0.7] 0.85 (0.23). With a near-useless checker (R1 open-source detector, R4 people) the split
reverses: users learn the checker is nearly useless (Ĵ ≈ 0.1) while non-users keep 1/3, and the checker no longer
gives truth an edge (Δ ≈ 0).

notes:
- interesting that split reverses for checkers R1 and R4
- do we think it's accurate that almost half of agents just never check, if we think of real world people? what proportion of people who use social media actually do fact check or try to verify information?
- in panel C where you wrote the spread of starting beliefs, is this starting beliefs about the checker?
- seems like the locked-out share should or could be a headline of the paper. i'm just maybe concerned that it's kind of "obvious" in the sense that there is a clear analytical formula for the checking threshold, and we, in a way, baked it into the model that some agents just are below that threshold? does that make the argument or result a bit weak?
- panel B is quite interesting. prior strength is not that impactful. and as the starting trust in the checker in creases, the (final?) share locked out decreases, but it seems like it converges to a non zero value, so there is always a number of agents who are locked out?

**Consequences (Fig. 6).** Truth's edge in reach, Δ = reach(genuine) − reach(fake), falls linearly with the
locked-out share: Δ ≈ 0.77 − 0.72 × locked (r = −0.99; φ = 0.2–0.6, l/g = 2). Agents live in a filtered world: they
think genuine items are more common than they are (φ = 0.55: believed 0.69, actual 0.45). Welfare falls with φ and cost.

notes:
- panel D: interesting, very strong correlation between truth edge delta and the share locked out. i guess it makes sense, the more agents are locked out the less checking happens, so there is likely quite some discarding (also due to high verification cost as we can see by the yellow color of those points at bottom right), and therefore probably the reach of genuine and reach of fake are both close to zero.
- panel C is a little confusing, one axis is share of fakes, and the other is believed share of GENUINE items. ideally both axes should be about either belief in genuine items, or belief in fakes?
- c/g values in the legend seem a bit random, but I guess it's okay.
- nice panel B

**Robustness (S2, S3).**
- Degree-preserving rewiring of Twitter changes nothing: audience size (the degree sequence), not clustering or
  communities, drives the split. Facebook, Twitch and Holme–Kim show the same mechanism; narrower degree
  distributions give a sharper, narrower split band.
- Without exploration the split is permanent (locked-out 0.42 at both 2,000 and 8,000 items; 4·dip 0.76 → 0.78). Random exploration dissolves it
  slowly: 1% random checks per item → 4·dip 0.78 → 0.01 and locked-out 0.42 → 0.29 by 8,000 items (the rest cannot
  profit from checking even at the true quality); 0.01% → 4·dip 0.46.
- Partial feedback (truth revealed with probability 0.5 or 0.2): slower, same end state. Old (v1) posterior: same.
  Prior strength × 3: locked-out 0.40.
- Stakes ∝ √k: locked-out 0.23, 4·dip 0.36; stakes independent of degree: no split (0.01). With identical
  starting beliefs, differences in audience size are what create the split.
- Informed base-rate belief (π₀ = 1 − φ instead of 0.5): same split in the main setting (locked-out 0.43 vs 0.42,
  4·dip 0.78 vs 0.76), no cold start; but the edge of the "nothing circulates" region moves (open decision 1).

notes:
- not seeing graphs or figures for this? not sure where these results are.
- nice to have some robustness results though (partial feedback, exploration, different networks, etc.)

**Global sensitivity (S1; PAWN, Latin hypercube, 1,000 runs, 8 inputs + dummy).** 36% of runs are in the
nothing-circulates region, driven by φ (0.19), l/g (0.17), starting trust, cost and TPR (≈ 0.12; dummy 0.02). Among
the 644 runs that circulate: locked-out share ← cost (0.39), then starting trust (0.19); 4·dip ← starting trust
(0.30), everything else near the noise floor (≤ 0.16 vs 0.09–0.15); mean trust ← starting trust (0.41);
truth's edge ← cost (0.24); welfare ← φ (0.28).

notes:
- maybe we have too many runs in the nothing circulates region, and we could tighten our parameter ranges a bit?
- Do you think it's a nice design to just fix the checker's FPR/TPR rates as I said earlier, and only vary c, g, and l? I think it's pretty clean I guess. just wondering if we're ignoring something that we should be experimenting with more.

**Why not the bimodality coefficient.** Across the 236 circulating cells of the Twitter phase diagram BC exceeds the usual 5/9 cut-off in 198 (84%) and is uncorrelated with 4·dip (r = 0.02); it is highest (0.90) at c/g = 0.53, where almost nobody checks and the dip is ≈ 0.09. A point mass at the shared starting trust plus a small tail is enough to push BC up, so we report BC only in the SI and use the dip (shape) and the locked-out share (who).

notes:
- ok, clear. sounds good to just focus on the 4 dip and not BC. maybe we shouldn't even put BC in the SI

**Validation (S5).** With the v1 switches the numba engine reproduces the Julia sweep: per-point means correlate
0.996–0.9998 over eight outputs (40 LHS points × 50 seeds).

notes:
- okay, nice, good enough i think.

## 3. Figures (draft captions)

- **Fig. 1 Model and mechanism.** (A) One item: estimate P(genuine) from the sender's record and the base rate, then
  share, check or discard; the truth is revealed afterwards. (B) Best action by P(genuine) p and believed checker
  informativeness Ĵ for an average agent (c/g = 0.043, l/g = 2); horizontal lines: θ for an average audience and for
  1/10 of it, the checker's actual J, and the shared starting trust. (C) First-passage model: belief trajectories of
  60 agents (θ = 0.2); hitting θ freezes the belief. (D) Share locked out by number of neighbours (Twitter, φ = 0.3,
  6 runs per cost): dots simulation, lines first-passage model.
- **Fig. 2 The split over time** (c/g = 0.043, φ = 0.3, l/g = 2, 20 runs). (A) Distribution of trust in the checker
  over time with group means (shown when a group has ≥ 5% of agents). (B) Shares of agents who never checked, are
  locked out, still check. (C) 4·dip over time for three costs. (D) Final distribution by group.
- **Fig. 3 Regimes.** (A) 4·dip and (B) share locked out over the share of fakes and c/g for l/g = 1, 2, 4 (Twitter;
  6 runs per cell at l/g = 2, 3 otherwise). Contour: 4·dip = 0.2. Dashed: cost at which an average agent's threshold
  equals the starting trust. Hatched: nothing circulates. (C) Locked-out share vs cost (φ = 0.2–0.5, circulating
  cells) against the first-passage prediction averaged over Twitter's degree distribution.
- **Fig. 4 Two checkers.** (A) Joint trust in two identical checkers (c/g = 0.043). (B) Who used which checker vs cost.
  (C) Loyalty among checkers vs cost. (D) Cheap AI detector vs costly accurate professional by number of neighbours.
  (E) Locked out of each checker vs cost (professional at 3× and 10× the detector's cost). (F) Identical checkers by
  number of neighbours. 6 runs per cost (3 for D–E).
- **Fig. 5 Starting beliefs and checker quality.** (A) 4·dip over starting trust J₀ and cost. (B) Locked out vs J₀ for
  four prior strengths. (C) Effect of spreading starting beliefs. (D) Final trust for four real-world checkers
  (R1–R4: TPR, FPR from the literature) at c/g = 0.043.
- **Fig. 6 Consequences.** (A) Truth's edge in reach, (B) welfare, (C) believed share of genuine items vs φ for four
  costs (l/g = 2). (D) Truth's edge vs locked-out share across all circulating cells with φ = 0.2–0.6.
- **Fig. S1** PAWN indices. **S2** Other networks. **S3** Exploration, feedback, model variants over 8,000 items.
  **S4** Two checkers on Holme–Kim. **S5** Validation against the Julia code.

Suggested main text: Fig. 1–5, with Fig. 6 as the one "consequences" figure (or as panels in Fig. 3).

## 4. Open decisions

1. **Starting belief about the base rate.** Currently uninformed (Beta(1,1), mean 0.5). This causes the ~100–300-item
   cold start, and makes the edge of the nothing-circulates region depend on early luck: with l/g = 4 and φ = 0.35
   nothing circulates under the uninformed start, but checking does get going if agents start from the true base
   rate; with l/g = 2 and φ = 0.75 it is the other way round. The split itself is unchanged (checked at
   c/g = 0.012/0.043/0.15). Options: keep (fewest assumptions) or start at π₀ = 1 − φ (agents know roughly how common
   fakes are). Switching means re-running the phase diagrams and GSA (~2 h on the container + Mac).
	1. *you are talking about the starting belief about the prevalence of genuine items, right? It seems like currently the setup leads to a bit of a cold start, and one option would be to start at 1 minus the fake prevalence. Is this maybe also a very strong assumption though? in reality people may not know what this prevalence actually is. Maybe beliefs should actually start more spread out? I'm not sure on this one. I do want to make as little strong assumptions as possible.*
2. **How to present "nothing circulates"**: as a result (collective lock-out caused by low starting trust) or as a
   boundary we exclude from the main analysis (current figures hatch it and exclude it from averages).
	1. *i think it is a result more than a boundary, since it says that in certain regimes where every one is distrustful, we get a society where no one verifies information or wants to share things (not sure if that's actually realistic, but seems like a genuine finding/outcome for this model*
3. **Main text vs SI** selection above.
	1. *I think the two checkers experiment should be a main text feature. First show the single checker case and results. and have a section about two identical checkers. the SI can contain the R1 and R4 checkers for robustness, as well as holme-kim and the other networks. I don't think julia validation needs to show up at all in the manuscript. Sounds good to have S1 be the PAWN stuff.*

notes:
- generally speaking, I want us to aim to provide some support/motivation for different choices in the modeling
	- for instance, why should the gain/loss payoffs scale with reach? is there evidence or hints of this in the real world?
	- what about the fact that verification cost is always at most equal to the gain or loss? does that make sense?
	- etc.

## 5. Next steps

- Intervention / hysteresis experiment: can a temporary subsidy (free checks for small accounts for a while) or a
  burst of exploration undo the split for good? The absorbing barrier suggests yes, if it lifts beliefs above θ.
	- *indeed, I do think a small experiment like this could be interesting, and should be done as an additional SI section, might be bumped to main text if it's interesting enough*
- LISS panel comparison once access is granted (trust in fact-checkers vs social media audience/activity).
	- *I now have access to this data, and I've created a folder, `LISS` in the model v2 folder with the data. they're csv format, let me know if you need a codebook or something*
- Optional: use the Twitter node attributes (partisanship, misinformation exposure) to ask who ends up locked out.
	- *not sure what this would give us, since our agents are quite abstract in the model, without opinions or partisanship?*

## 6. Reproduce

```
cd model_v2
export DFP_NETDIR=../data/networks DFP_NPROC=4                 # networks: data/networks (see its README)
DFP_OUT=../data/v2 python3 make_paper_figures.py ../figures/paper_v2            # all figures from data/v2
DFP_OUT=../data/v2 python3 make_paper_figures.py ../figures/paper_v2 fig3_regimes   # just one
python3 chunk_runner.py phase twitter 170 out                  # resumable, time-boxed sweep (repeat until exit code 0)
```

Experiments → outputs: `exp_phase` / `chunk_runner phase` → `phase_<net>.parquet`; `exp_time` → `time_twitter_c*.pkl`;
`exp_mechanism` → `mech_twitter.pkl`; `exp_two` → `two_<net>(.parquet, _agents.pkl)`; `exp_priors` →
`priors_twitter.parquet`; `exp_checkers` → `checkers_twitter(.parquet, _J.pkl)`; `exp_gsa` / `chunk_runner gsa` →
`gsa_twitter.parquet`; `chunk_runner robust` → `robust_twitter.parquet`; `exp_cstar` → `cstar_twitter.parquet`;
`validate_vs_julia` → `validation_points.csv`; base-rate check → `baserate_prior_check.csv`.
