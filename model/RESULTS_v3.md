# Results v3: model v3 on the Twitter news-sharer network

Status 11 November 2026. Repository cleaned (11 Nov, afternoon): code in `model/` (was `model_v2/`), figure scripts named by
content (`fig_*.py`, `si_*.py`; old → new names in `model/README.md`), v1/v2 code and old outputs deleted, raw GSS files in
`data/gss/raw/`. Model description, second draft (Claude Doc "Model description (v3)", your comments of 11 Nov
answered in their threads); every-sender robustness check run (section 4, point 12). Figures re-rendered 11 Nov (labels with ℓ and
"simulations"; one setting for the one-agent model), revised after your notes of 5 Nov (12:10, 13:12, 14:10, 14:26 and 14:50) and 6 Nov (09:43–15:03). On 6 Nov: every line or marker with a CI now rests on at
least 10 runs (section 1, "Uncertainty"); Fig. 2 runs to 4,000 items, without the ceiling triangles; new Fig. S11
(provisional number) tests on random networks whether the split follows the inequality of audiences or the average
audience (the inequality; section 3); new Fig. S12 (provisional) sets US survey data (GSS) beside the model (section 3).
SI renumbered on 5 Nov after dropping the cost cut: old S6–S11 are now S5–S10.
Since 14:10 the split of trust is reported as Hartigan's dip itself (0–0.25) instead of 4·dip, in every figure, table and
note. Since 14:26 every simulated line and marker has a 95% CI computed with Student's t, and runs in which nothing
circulates are handled by measure (section 1).
Figures: `figures/paper_v3/` (PDF + PNG, PNAS widths; the earlier figure sets of 3–11 Nov were deleted on 11 Nov).
Data: `data/v3/` (not in git). Code: `model/` — `run_v3.py` runs every
experiment (restartable), `make_figures_v3.py` draws every figure. Target journal: PNAS.

## 1. Decisions

| Item | Setting |
|---|---|
| Terms (4 Nov; renamed 5 Nov) | **Not checking**: trust in the checker at or below the threshold θᵢ (Ĵᵢ ≤ θᵢ); both groups below are below their threshold, they differ in whether they are right not to check. **Priced out**: θᵢ ≥ J, checking would not pay for them even if they knew exactly how good the checker is (small audience, so the stakes do not cover the cost): not checking is the right call. **Trapped** (called 'locked out' until 5 Nov; links to the 'learning trap' literature): θᵢ < J, checking would pay for them, but their trust is too low to try it (from the start, or after bad luck), and without checking they never find out (no checks, no learning) |
| Headline (4 Nov) | One checker: who stops checking and why (priced out vs trapped), how fast, and how cheaply it is undone. Two checkers: the split of trust between identical checkers (the bifurcation of epistemic trust). The dip describes the shape of the trust distribution, with its caveats; no longer a headline |
| Split of trust (5 Nov 14:10) | Hartigan's dip statistic of the trust distribution, reported as is: 0 for one hump, at most 0.25 (two equal groups, each at a single value). When two groups are well apart it is about half the share of the smaller group. Until 14:10 we reported 4·dip (0–1), only a rescaling to the 0–1 range of the shares; dropped because the dip itself is the standard statistic (Hartigan & Hartigan 1985). With 14,128 agents a dip above ~0.005 is significant (p < 0.01; 99th percentile under a uniform null, simulated). The runs still store `dip4` = 4 × dip; the figure code divides by 4 |
| Key message (4 Nov) | Checking is costly and imperfect, and people learn how good a checker is only by using it. Chance differences in what they see and how their checks turn out, amplified by how much each has at stake, then make otherwise identical people diverge in whom they trust, even between equally good checkers. The divergence lasts because those who stop checking stop learning. Checkers stand for epistemic institutions people can consult at a cost (fact-checkers, detection tools). Future work: opinions that co-evolve with trust |
| Empirical anchors (4 Nov) | No own experiment. Inputs calibrated (real network; checker error rates from detector benchmarks). Mechanism backed by experiments (hot stove effect, learning traps, algorithm aversion, trust in automation); outcomes matched to survey facts (Reuters DNR 2025; Pew 2019); (the intervention anchor, Dupas 2014, went with the cost cut on 5 Nov). Testable predictions stated explicitly (section 4.6). X Community Notes analysis kept in reserve for the revision |
| Two checkers (4 Nov) | Main text, right after the single-checker figures (Fig. 3), with the split forming over time; split measured on trust in A minus trust in B among agents who can still check |
| LISS (4 Nov) | Dropped from the paper (the survey measures institutional trust, not trust in a checker learned by use) |
| Other settings (4 Nov) | Single-setting panels repeated at c/g = 0.01, c/g = 0.2 and φ = 0.6 (Figs. S5, S7) |
| Trapped result (3 Nov) | Model unchanged; the threshold θ is the transparent backbone. Random exploration only as a robustness check and an intervention lever |
| Payoffs (kept, 4 Nov) | Gain and loss both scale with the audience sᵢ = kᵢ/⟨k⟩, read as reputation with one's audience (being right earns esteem, being wrong costs more). SI: Fig. S3 and Table S1, five forms (penalty ∝ audience, ∝ √audience or constant with the gain ∝ audience; both stakes ∝ √audience; both constant) |
| Main-text figures (4 Nov) | Four, for PNAS's standard 6 pages (about 4,000 words and 4 medium figures; up to 12 pages allowed): Fig. 1 mechanism; Fig. 2 one checker over time (trust, groups, split of trust over time); Fig. 3 two checkers (trust in A − B, use over time, clear favourite, by audience, AI detector vs professional); Fig. 4 regimes and consequences (maps at l/g = 2 incl. how much further genuine items travel; reach gap vs share not checking; welfare; believed share of fakes — the former Fig. S12, moved in on 4 Nov). Starting beliefs, l/g = 1 and 4 maps and the theory-vs-simulation curves across the phase diagram (former Fig. 4E–F, now Fig. S9I–J) in the SI |
| Exploration and cost cut (4 Nov; cost cut dropped 5 Nov) | Random exploration in the SI as a robustness check (Fig. S4). The temporary cost cut is out of the paper (5 Nov): it brings most of the trapped back for good, but what spreads and welfare barely change (runs and code kept) |
| Model in the main text (4 Nov) | Short model section with the decision rule and threshold θᵢ and the learning rules; full specification, two-checker rule, priors and parameter table in the SI. Paper outline on hold until the rest is settled |
| Social reward (4 Nov) | "Being liked" (reward from the audience resharing) left out of the paper: it muddies the message |
| Starting belief about the share of genuine items | Spread out: π₀ᵢ ~ U(0.01, 0.99), weight 2 (removes the cold start of v2). Its comparison with other starts goes in the text only (panels dropped from Fig. S1 on 4 Nov; see the notes for the main text) |
| Starting beliefs about the checker | Identical: TPR ~ Beta(2,1), FPR ~ Beta(1,2), starting trust J₀ = 1/3 |
| Ranges | c/g 0.001–1, l/g ∈ {1, 2, 4}, φ 0.05–0.95; GSA ranges as in v2 |
| Parameters (6 Nov) | The payoffs have three parameters (gain g, loss l, cost c) but only two degrees of freedom: every decision and the welfare measure (each agent's payoff normalised by its best and worst possible payoff) depend only on c/g and l/g, so g = 1 in the code. The model's other parameters: the share of fakes φ, the checker's hit and false-alarm rates, the starting beliefs (about the checker: J₀ = 1/3, prior strength κ = 3, spread w = 0; about the share of genuine items: U(0.01, 0.99), weight 2; about senders: weight 2), how the stakes grow with the audience (s = k/⟨k⟩, exponent 1) and the network. The sensitivity analysis varies eight of them (φ, c/g, l/g, J₀, κ, w, hit and false-alarm rates) |
| Measurement window (6 Nov) | Runs: 2,000 items with a burn-in of 1,000. The burn-in applies only to quantities summed over items, which are taken over items 1,000–2,000: reach of genuine items and of fakes (reach gap), welfare and payoffs, the check rate, and the share of fakes each agent shared (degree correlation, Table S1). States are read at the end of the run (item 2,000) from learning that runs from item 1: trust, not checking, priced out, trapped, ever checked, the dip, the believed share of fakes. Time plots show every item from 0. Other lengths: robustness runs 8,000 items (burn-in 4,000); Fig. 2 runs continued to 4,000 items (6 Nov; burn-in 3,000); cost-cut runs 4,000 (outcomes over items 3,000–4,000). The share of fakes among checked items (input to the one-agent model) uses the whole run |
| Grid | Hexagonal lattice: 19 rows, six per decade of c/g; φ in steps of 0.05, odd rows shifted by 0.025; 352 cells, 2–3 runs per cell |
| Main setting | φ = 0.3, l/g = 2, c/g = 0.05 (illustrative) |
| Checker | R2 (TPR 0.80, FPR 0.10, J = 0.70) throughout; quality sweep and R1–R4 in Fig. S2 |
| Two checkers, checking with both | The agent pays both costs and uses whichever rule has the higher expected payoff given its beliefs: share only if neither checker flags the item, or share unless both flag it. With equal trust in both checkers this is the Bayes-optimal use of the two flags |
| Wording and figures | c/g = "relative cost of checking"; "theory (one-agent model)"; BC dropped; Julia validation out; colour-blind-safe palette, own marker and dash per line, hexagonal maps, round cost values, boxed in-plot legends; 95% CIs on every simulated line and marker (next row) |
| Uncertainty (5 Nov 14:26) | Every line or marker from simulations shows a 95% CI over runs (band for lines, error bar for markers), computed with Student's t. Until 5 Nov all bands used 1.96 × SE, which understates the 95% CI 2.2× with 3 runs and 6.5× with 2 (7% with 20 runs). Bands are drawn even where they are narrower than the line, and the caption says so; lines of the one-agent model are calculations and have none. The band-averaged simulation points of Figs. S8C–D and S9I–J pool the run-to-run variance within cells (every cell and run has its own seed; Satterthwaite's degrees of freedom). Topped up to 10 runs per point on 5 Nov: Figs. S1B–D, S2 (B, C and the four checkers in A) and S5C–D; on 6 Nov the four cost rows of Fig. 4F–G (10 runs per cell, `phase_lines_twitter`; the maps and Figs. S8–S9 include these runs too). Also on 6 Nov: Figs. 1D (from 6 runs), 3E (from 3), S3 and Table S1 at c/g = 0.05 (from 5), S4 (from 3), S6 (from 5–6) and the AI-detector-vs-professional column of Fig. S7 (from 3). Since then every line or marker with a CI rests on at least 10 runs (20 in Fig. 2 and the time rows of Fig. S5; the band-averaged points of Figs. S8C–D and S9I–J pool at least 14). The numbers in these notes are recomputed with the added runs; nearly all are unchanged or move by about 0.01 |
| Runs where nothing circulates (5 Nov 14:50) | Genuine items reach fewer than 5% of agents (e.g. one of the first three runs with the "people" checker at c/g = 0.05; 15% of the phase-diagram runs, mostly in the hatched region). Split by measure: not checking, trapped, ever checked, the dip and beliefs are averaged over the runs in which items circulate; reach and welfare over all runs (a collapse really means that nothing spreads and nobody gains). Map cells stay hatched when the mean reach over all runs is below 5%; the lines of behaviour and beliefs stop there (Fig. 4G). The sensitivity analysis keeps its own rule (panel A: whether anything circulates; B–F: circulating runs only). The cached one-agent-model curves still take the share of fakes among checked items from all runs of the live cells (difference negligible) |
| Wording and notation (11 Nov) | "Checker" throughout (not "verification tool"); ℓ for the loss from sharing a fake (was l, which reads as 1 or I); "simulation" for a run, everywhere. Figure labels changed on 11 Nov (Figs. 1, 3, 4, S3, S4, S6, S9, S10); Fig. 1A now says "j's record" instead of τᵢⱼ, which the text does not define |
| One-agent model setting (11 Nov) | One setting in every figure: 500 checks, 6,000 paths, 80-point grid of thresholds (`trapmodel.N_CHECKS, N_MC, N_GRID`). Before: Fig. 1 used 500 checks and 6,000 paths, Figs. 4, S8 and S9 400 checks and 2,000 paths. Caches (`theory_*.json`) regenerated and Figs. 4, S8, S9 re-rendered on 11 Nov (Fig. 1 already used this setting); the fits are as good or better (Figs. S8, S9 below) |
| Forward-looking agents (11 Nov) | No extra experiment: random exploration (Fig. S4C) already shows what happens when agents sometimes check against their myopic choice; the main text cites it in one sentence |

## 2. Main-text figures (4; revised 4 Nov afternoon after your notes)

**Fig. 1 — mechanism** (`fig1_model_mechanism`). (A) What an agent does with an item; (B) who checks at the start
(c/g = 0.05, l/g = 2): the threshold θ against the number of neighbours (θ = 0.075·⟨k⟩/k, ⟨k⟩ = 252). Shaded: trust
above θ, where checking pays. Green line: the checker's actual quality J = 0.70. Coloured line: everyone's starting
trust (1/3), coloured by what it means at each k: below 27 neighbours even the actual J is below θ, so checking never
pays (priced out, grey, 30% of agents); between 27 and 57 the actual J is above θ but the starting trust is below it,
so the agent never tries (trapped from the start, orange, 11%); above 57 the starting trust is above θ and the agent
starts checking (blue, 59%; 4% of all agents are trapped later by bad luck, panel C). The y axes of B and C are the
same, so the actual-J lines line up. (C) 60 identical agents with θ = 0.2 (about 95 neighbours at c/g = 0.05) and the
same starting trust, who differ only in the luck of their checks (one-agent model: checks drawn with the checker's
TPR and FPR and the share of fakes among checked items from the runs): those who keep checking end near the actual J,
those whose trust hits θ stop for good with their trust frozen (trapped). Direct labels instead of a legend. (D) Share
not checking by number of neighbours at c/g = 0.01, 0.02, 0.05 (c/g = 0.1 dropped on 5 Nov; markers: simulation, 10
runs each since 6 Nov (from 6); lines: one-agent model, no fitting). Overall share not checking, simulation vs one-agent model: 0.225 vs
0.223, 0.306 vs 0.306, 0.449 vs 0.448. 95% CIs (Student's t over the 10 runs; at most ±0.012) are drawn as error bars
and hidden by the markers (until 5 Nov they were not drawn).

notes (6 Nov, addressed):
- Another small point, for when you get the time - figure 1B: the theta in "threshold theta" label at the bottom right sort of clips through the edge of the figure. maybe make that label a bit smaller so it fits, or move it slightly to the left. (right-aligned with the other labels, inside the panel)

notes (5 Nov, addressed):
- Fig 1B: you know, I'm still confused about priced out vs. locked out. Can you remind me what it means, and if the naming convention really makes sense? (renamed: priced out / trapped)
- Fig 1C: are these lines ONLY from agents who had a threshold of theta = 0.2? so they pretty much have the same number of neighbors and initial trust, right? (yes: 60 identical agents)
- Fig 1D: maybe let's remove one of hte four lines, for instance c/g = 0.1.
- Fig 1B and 1C : let's make the y-axes have the same sort of scale, so that "checker's actual J" green line are aligned horizontally. for Fig 1C we currently have a bunch of white space below trust = 0.0 which is not utilized. We just may have to move the legend a bit to not obstruct things.

notes (4 Nov 14:31, addressed): panel B redrawn as threshold vs number of neighbours.
- Fig 1 panel B - i don't actually see a shaded area? I'm still confused about the thetas and audiences on this figure, it's confusing me. also now the "check" text does not totally lie in the green region

notes (3 Nov, addressed; panel letters refer to the figure as it was then):
- Panel A: left most, the word "seed" is almost on top of one of the edges, maybe move it a bit to the left. Also, the text in the first blue box ("agent i receives an item from neighbour...") doesn't actually quite fit properly inside the drawn box in the pdf file (seems okay in the png somehow). please check throughout that figures are clean and legible. The little orange arrow pointing down at the far right also looks a little oddly placed, please double check
- panel B looks a bit busy, I find it hard to read since there is so much text and lines, maybe make some of the text smaller or find a more clean and intuitive way to show the different thresholds. Also the word SHARE cuts out of the figure!
- panel C: the legend is quite weirdly placed. maybe you should add a box around the legend and a drop shadow or something. please double check this, and other legends!
- panel D: if the CIs are very narrow, we can safely remove them I think

**Fig. 2 — one checker over time** (`fig2_one_checker_over_time`, c/g = 0.05, 20 runs). (A) Trust in the checker at
the start and after 10, 50, 300 and 2,000 items by group, overlaid at 72% opacity (log scale, so that small groups
show); (B) group shares over time: with the spread-out base-rate start, checking starts at once; who ever checks is
settled within ~50 items (300 at c/g = 0.2). End state (after 4,000 items; the same after 2,000): 41% never checked, 4% checked and then stopped, 55% still
check. (C) Split of trust (dip) over time for c/g = 0.01, 0.05, 0.2, 20 runs each (0.01 and 0.2 topped up from 10
on 4 Nov; the number of runs goes in the caption, not the legend); since 6 Nov B and C run to 4,000 items (the same runs
continued). The ceiling set by the group sizes (the dip if every agent who still checks held the
same trust; drawn as triangles until 6 Nov): 0.10, 0.21 and 0.14. The dip is at most about half the share of the smaller group (users vs non-users),
and it approaches that ceiling as the users' trust narrows around the actual J (the middle 80% of users span 0.38 of
trust after 100 items, 0.20 after 300, 0.07–0.10 after 2,000, at every cost). At c/g = 0.01 only ~1 in 5 agents does
not check, so the ceiling is low (0.10) and reached by ~300 items: the line goes flat. At c/g = 0.05 the two groups
are about equal (45% vs 55%), so the ceiling is high (0.21) and the split keeps rising until it levels off near it (0.111, 0.135, 0.164, 0.189, 0.202,
0.206 after 300, 500, 1,000, 2,000, 3,000, 4,000 items; 0.21 after 8,000, Fig. S4). At c/g = 0.2 the users are 27% (ceiling
0.14) and the line rises more slowly (0.056, 0.071, 0.086, 0.098, 0.104, 0.107 at the same times): it crosses the
c/g = 0.01 line at ~2,150 items and is still rising at 4,000 (users at that cost check rarely, so their trust narrows slowly). The dashed line at 4·dip = 0.2 (dip 0.05) is gone: it
was not a meaningful cutoff (with 14,128 agents a dip above about 0.005 is already significant, p < 0.01).
- The early bump and dip in C (first ~50 items): after one or two checks trust can take only a handful of values, so
  the few agents who have checked sit on a few spikes (at 3 items the three fullest bins hold 96% of them), and a
  spiky distribution counts as split. From ~8 to ~50 items many agents have checked a few times and their trust is
  spread widely (26–39 of the 80 bins used), overlapping the never-checked spike at 1/3: less split (0.058 → 0.034).
  After ~50 items the users' trust narrows around the actual J and moves away from 1/3: the split grows.
- The few agents who still check although their trust is below the starting trust (blue, left of the grey spike in
  A): mid-sized accounts (median 115 neighbours) whose first checks went badly; trust (median 0.17) just above their
  own θ (median 0.16), so checking is the best option only for rare doubtful items: median 3 checks in 2,000 items
  (other users: 1,132). On the edge: one bad check traps them. 0.13% of agents.

notes (3 Nov, addressed; panel letters refer to the figure as it was then):
- panel A: maybe a log scale would make things more legible?

notes (6 Nov, addressed):
- For figure 2C, I think maybe it would be nice to circulate a larger number of items (for instance up to 1500 or 2000) so that the curves actually reach their plateau range, instead of having the pointy ceiling arrows! (the runs already went to 2,000 items; the same 60 runs are now continued to 4,000 items, same seeds, so the first 2,000 items are identical (checked); the triangles are gone. See the next note)
- 16000 items might be a tad too much? maybe just adding another 1000-2000 or so is enough, just to see the c/g=0.05 line level off a bit, and the c/g=0.2 cross the 0.01 and get closer to its ceiling. What do you think? (agreed, 4,000 items: c/g = 0.05 levels off near its ceiling, 0.189 at 2,000 and 0.206 at 4,000 against 0.21; c/g = 0.2 crosses the c/g = 0.01 line at ~2,150 items and reaches 0.107 against its ceiling of 0.14 (a pilot run reached 0.122 by 32,000 items); c/g = 0.01 stays flat at 0.10)

notes (5 Nov, addressed):
- Fig 2C: we can remove the "20 runs each" from the legend, this will go in the caption.

notes (4 Nov 14:31, addressed):
- Fig 2 panel A: opacity looks good. Panel C the three lines have different numbers of runs, not ideal. Also, interesting, but how to explain that c/g = 0.05 line keeps increasing, whereas the others stabilize. Also why the dashed line at 4dip = 0.2? Is that a meaningful cutoff?

**Fig. 3 — two checkers: trust splits between identical checkers** (`fig3_two_checkers`; two identical checkers A and B
(TPR 0.80, FPR 0.10) and identical starting beliefs about both; 10 runs at each cost). (A) Trust in A minus trust in B
among agents who can still check (above threshold for at least one checker), at the start and after 10, 50, 300 and
2,000 items, by what they have used, overlaid at 72% opacity (log scale): one spike at 0 splits into three groups —
agents who trust both, agents who trust A and never tried B (blue), agents who trust B and never tried A (orange) —
plus agents who used both but now trust only one. (B) What agents used over time: after 2,000 items 45% have used
both checkers, 7% only A, 7% only B, 41% neither. (C) Share of the agents who can still check with a clear favourite
(trust in one checker more than 0.2 above the other): 56% after 2,000 items at c/g = 0.05 (peak 69% around 150 items),
79% at c/g = 0.2, 32% at c/g = 0.01. Among agents who checked at least once, trust in A and trust in B are negatively
correlated (−0.34 at c/g = 0.05, −0.55 at 0.2, −0.14 at 0.01). (D) By number of neighbours: small accounts use
neither, mid-sized accounts one, large accounts both. (E) A cheap AI detector vs a professional at 3× the cost
(TPR 0.95, FPR 0.03): mid-sized accounts use only the detector, only large accounts pay for the professional.
Nobody used the professional without also using the detector, so its line is now labelled "both (detector and
professional)" (5 Nov). Starting beliefs about the two checkers are identical (starting trust 1/3 each); the
professional's threshold is 3× the detector's, so at the start it pays only above ~170 neighbours, and with equal
trust the cheaper detector is tried first. Large accounts then try the professional (separately or together with the
detector) and learn that it is better: trust 0.87–0.91 in the professional vs 0.53–0.57 in the detector above 500
neighbours (E: 10 runs since 6 Nov, from 3). (Loyalty against cost, formerly D, is in Fig. S6; the share not using each checker, formerly G, is in the
text: 0.45 for the detector and 0.62 for the professional at 3× the cost, c/g = 0.05; 0.87 at 10× the cost.)

**Fig. 4 — regimes and consequences** (`fig4_regimes_and_consequences`; hexagonal lattice over φ and c/g; 3 runs per
cell at l/g = 2 (10 in the four cost rows drawn in F–G, since 6 Nov), 2 at l/g = 1 and 4; all panels l/g = 2). (A–D) Maps: split of trust (dip), share not checking,
share trapped, and how much further genuine items travel than fakes (reach of genuine items minus reach of fakes).
Hatched: nothing circulates (genuine items reach < 5% of agents). The share trapped depends almost only on the cost;
the split needs intermediate costs (maximum 0.20 around c/g ≈ 0.03–0.05; 0.19 at l/g = 1 and 4); the trapped share peaks at ~0.20
around c/g ≈ 0.2–0.3 (largest cell 0.20–0.21 for every l/g; colour scale now ends at 0.25, no arrow); at φ = 0.3 genuine
items reach 0.67 more of the agents than fakes at c/g = 0.001, 0.07 at c/g = 1. (E) The reach gap falls in step with
the share not checking (r = −0.99 over the 124 cells with φ = 0.2–0.5 in which items circulate; the band now matches
the one used for the theory curves). (F) Welfare (Sen index: mean normalised payoff × (1 − Gini), checking costs
included) against φ for c/g = 0.001, 0.01, 0.1, 1: 0.83 vs 0.32 at φ = 0.3 for c/g = 0.001 vs 1; over all runs, so
where nothing circulates welfare is ~0 and the lines run on into the hatched region. With 10 runs per cell (6 Nov) its bands are
at most ±0.03 for φ ≤ 0.4; they widen where runs start to collapse: at c/g = 1 for φ ≈ 0.45–0.6 (±0.04–0.06; 1, 1, 3
and 6 of 10 runs collapse at φ = 0.45, 0.5, 0.55, 0.6) and near φ = 0.9 in the cheaper rows (up to ±0.09). (G) Believed share
of fakes (mean over agents at the end of the run; dotted: believed = φ): where checking is cheap people believe there
are fewer fakes than are produced (0.10 believed at φ = 0.3, c/g = 0.001) because checked fakes stop early and reach
few; where nobody checks, beliefs track φ (0.28 at c/g = 1). Averaged over the runs in which items circulate, so the
lines stop where the maps are hatched (5 Nov; before, they ran on with the frozen beliefs of runs in which nothing
circulated). (Former E–F, not checking and trapped against cost
with the one-agent model for l/g = 1, 2, 4, are now Fig. S9I–J; Fig. 1D keeps theory vs simulation in the main text.)

notes (6 Nov, addressed):
- the error bars look a little crazy (wide and pointy) in panels 4F and 4G, but I guess that is fine? or do you think doing some more runs/sweeps would help smooth things out? also the c/g=1 line in 4G stops early because nothing circulates beyond the share of fakes around 0.6, right? (3 runs per cell gave Student's t factor 4.3 and a separate band per φ; the four rows are now topped up to 10 runs per cell (t factor 2.26): bands at most ±0.03 for φ ≤ 0.4. Yes: at c/g = 1 most runs collapse from φ = 0.6 on (6 of 10 at 0.6, 9 at 0.65 and 0.7, all from 0.75), so the line stops at 0.55)
- OK good, thanks for doing more runs. in general, error bars with under ~10 runs/samples is not great. so make sure we do enough runs for the various plots with CIs. (done: every CI now rests on at least 10 runs; section 1, "Uncertainty")

notes (3 Nov, addressed; panel letters refer to the figure as it was then):
- i think we could remove the start symbol denoting the main setting, looks a little weird
- what does would pay, locked out mean? they would pay for the verification, but don't cross the threshold?
- confused about the horizontal dashed lines in panels A-C
- panel A: let's remove the contouring black lines

notes (4 Nov 14:31, addressed):
- Fig 4: maps look good, although for the top row, the l/g=2 label is a bit weirdly placed, with quite some white space between it and the relative cost c/g ylabel, perhaps adjust a bit
- Fig 4: panels E and F have share of fakes between 0.2-0.5, but panel G has 0.2-0.6? why this very slight discrepancy? (no reason; now 0.2–0.5 everywhere)
- I think S12 should be main text material somehow. S12 panel B should remove the text label "believed=actual". (now Fig. 4F–G, without the label)

## 3. SI figures (12 and Table S1; revised 4 Nov evening, S11–S12 added 6 Nov; grouped by the main figure they support)

*Model and its assumptions (Fig. 1)*

**Fig. S1 — starting beliefs about the checker** (`figS1_starting_beliefs`; 2 × 2 panels since 5 Nov 14:10, when C was
split into C and D). (A) The dip over the
starting trust J₀ (the same for every agent; axis label changed from "shared starting trust" on 5 Nov) and c/g.
Since 5 Nov a hexagonal lattice like the Fig. 4 maps, with the axes swapped to match them and panel B (x: J₀ from 0
to 0.6 in steps of 0.05; y: c/g, 6 rows per decade, 0.001–0.46; 213 cells, 2 runs each, `priors_hex_twitter`; the
run-to-run SD of the dip is typically 0.0006, so the gradient is smooth). Strongest split 0.20 at J₀ ≈ 0.2, c/g ≈ 0.03; in each
column the strongest split sits just below the dashed line (at 40–60% of its cost, where 58–66% of agents start
checking). Earlier 7 × 9 rectangular grid: `priors_twitter`, block J0xcost;
dashed (redrawn 5 Nov): half of the agents start checking, i.e. J₀ equals the threshold of the median account (93
neighbours; θ = 1.5·c/g·⟨k⟩/93 = 4.1·c/g). Below it (lower cost) most agents start checking, above it most never start; the
split is strongest along this line, where the population divides into two groups of similar size (at the cell with
the largest split in each row 54–65% start checking). Before 5 Nov the line marked the average account (252
neighbours), which sits at 4–6× higher cost than the band of strong split, because 71% of accounts have fewer
neighbours than average. The label "half of the agents start checking" moved up, away from "most start checking"
(5 Nov 14:10). (B) Share not
checking vs J₀ for prior strengths κ = 2, 3 (default), 10, 30 at c/g = 0.05: prior strength matters little (0.47 at
J₀ = 0.3 for κ = 3, 0.43 for κ = 30); the share falls with J₀ but stays above the share priced out (0.30). (C, D)
Spreading the starting beliefs about the checker between agents (± w around J₀ = 1/3): the split of trust disappears
(C; dip 0.19 → 0.068 at w = 0.1 → 0.023 at w = 0.3) while the share not checking rises (D; 0.45 → 0.53), and so does
the share trapped (0.15 → 0.23). C and D were one panel until 5 Nov 14:10; the dip (0–0.25) now has its own axis
instead of sharing the 0–1 axis of the shares. B–D: 10 runs per point since 5 Nov (topped up from 2–3); the 95% CI bands
are drawn but narrower than the lines (at most ±0.004 in B, ±0.003 in C–D). At J₀ = 0 nobody ever checks, and in 6–7
of 10 runs nothing circulates (excluded from B; the share not checking is 1 either way).
The starting belief about the share of genuine items (former bottom row, D–E, removed on 4 Nov; 7 settings, 3 runs
each; everyone at 0.5 as in v2, everyone correct (1 − φ), or spread out as used here) goes in the text: it mainly
sets how soon checking starts (first check within 5 items with the spread-out start in all 7 settings; after
~325–1,200 items when everyone starts at 0.5). The end state is the same (main setting at c/g = 0.01, 0.05, 0.2,
dip spread out / everyone correct / everyone at 0.5: 0.099/0.099/0.099, 0.189/0.188/0.183, 0.095/0.098/0.098; share who
ever checked identical), except where sharing does
not get going: with l/g = 4 and c/g = 0.01, everyone starting at 0.5 still has not finished its cold start after
2,000 items (ever checked 0.55 and 0.27 vs 0.82 at φ = 0.25 and 0.35), and with φ = 0.75 and everyone starting
correct nothing circulates (also smaller split and reach with everyone at 0.5 there).

notes (3 Nov, addressed; panel letters refer to the figure as it was then):
- the grid cells in the heatmap look kind of ugly in panel A. can we avoid the white line separation maybe? or do another nice hexagonal grid thing?
- just curious: do you think it's limiting that a LOT of our figures are shown for just that default setting? maybe a reviewer or reader would be constantly wondering, but how does this look for other settings?

notes (4 Nov 14:31, addressed):
- Fig S1, panel C: the legend is a bit weirdly placed, why not also in a box , maybe at the top of the graph below the title where there is whitespace in the graph?
- Fig S1 panels D and E, i'm sort of thinking let's just remove these, and we can just briefly note in text somewhere that results end up similar, it's mostly about how soon checking begins. Right? (yes, with the two exceptions above)

notes (5 Nov, addressed):
- Let's hold on to the ready to use sentence about the old S1 D-E plots, so we don't forget. But I like the S1 plot more now so that is good. (kept in the notes for the main text)
- S1 A: Not sure what is meant by "average agent here" is this dashed line just the average initial trust? Or the average initial threshold for checking of an agent? (the threshold of an agent with an average audience; label reworded)
- S1 B: Again not sure what is meant by 'shared' starting trust. Everyone has the same starting trust? I guess that's what it means (yes; axis now says "same for all")
- S1 C: What's the y-label on this graph? (added)

notes (5 Nov 13:12, addressed):
- Only thing i'm still slightly confused about is S1 A - how to interpret this? (see the walkthrough; dashed line now "half of the agents start checking")

notes (5 Nov 13:25, addressed):
- One thing about S1 A figure, the color boxes are sort of rectangular, doesn't look the nicest. Can we maybe potentially do more runs, and do a hexgrid, so it looks a little cleaner, smoother, and more uniform in the gradient?

notes (5 Nov 14:26, addressed):
- I'm just now noticing, though, there is no shaded region of uncertainty in panels S1B through S1D (no error bars either). Why not? I think whenever we have some line plot we should have a shaded region of confidence interval, right? (the bands were drawn but thinner than the lines; now every simulated line and marker has a 95% CI computed with Student's t, and S1B–D rest on 10 runs per point; see section 1, "Uncertainty")

notes (5 Nov 14:10, addressed):
- Can we move up the text "half of the agents start checking" a bit, it's currently sitting quite close to "most start checking" (moved up)
- Another thing: throughout the figure we reported 4 x dip. But why multiply by 4? wouldn't it be simpler/cleaner to just report the dip metric? (which I guess ranges from 0 to 0.25, right? was that it?) (yes, 0–0.25; the ×4 was only a rescaling to the 0–1 range of the shares. The dip is now reported as is in every figure, Table S1 and these notes; see section 1)

**Fig. S2 — how good the checker is** (`figS2_checker_quality`; 10 runs per point since 5 Nov, topped up from 3; R1 and R3,
reported in the text only, 3 runs). (A) Final trust at c/g = 0.05 for
four real-world checkers spread over the quality range (5 Nov; groups overlaid at 72% opacity): commercial AI detector
(R2: TPR 0.80, FPR 0.10, J = 0.70), best fine-tuned open-source detector on in-the-wild media (R5: 0.65, 0.17,
J = 0.48; Chandra et al. 2025, Table S9), best lab-trained audio deepfake detector on in-the-wild audio (R6: 0.66, 0.34,
J = 0.32; Müller et al. 2022) and people asked to spot fakes (R4: 0.43, 0.32, J = 0.11). R3 (LLM fact-checker,
J = 0.70) and R1 (off-the-shelf detector, J = 0.10) behave like R2 and R4 and are reported in the text (not checking
0.46 vs 0.45; 0.87 vs 0.86). Not checking at c/g = 0.05: 0.45, 0.51, 0.65, 0.86; split (dip): 0.19, 0.092, 0.028, 0.11.
With people as the checker nothing circulates in 2 of the 10 runs (excluded; until 5 Nov one such run among 3 was
averaged in, which gave 0.75).
(B) Not checking vs checker quality J = TPR − FPR (FPR 0.10): priced out (θ ≥ J, grey) and trapped (orange); the four
checkers as markers, with faint dashed lines down to the x axis (5 Nov). They do not sit exactly on the curve because
the curve fixes FPR at 0.10: how a checker errs also matters. The audio detector and people, both with many false
alarms (FPR 0.34 and 0.32), lose more users than their J alone suggests (0.65 vs 0.55 on the curve; 0.86 vs 0.78).
(C) The dip vs J. How to read it: the dip measures how far the trust of the agents who check has moved away from that of
the agents who never check (who stay at J₀ = 1/3), not how many do not check. It is lowest (0.02) when the checker is
about as good as people first assume (J ≈ J₀): users' trust then stays near 1/3, although 58% are not checking. It
grows the more the checker's real quality differs from that first guess, in either direction: for a good checker the
users' trust rises towards J (up to 0.21 at J = 0.8, c/g = 0.05); for a poor one it falls below 1/3 (0.08–0.11 at
J = 0.1–0.2 for c/g ≤ 0.05). Its height is capped by the smaller group, as in Fig. 2C: at c/g = 0.01 only ~20% do not
check, hence the plateau at 0.10 (exactly half the never-checked share); at c/g = 0.2 few check.
The former panel C (trapped as a share of those for whom checking would pay) is dropped (5 Nov) and goes in the text:
a poor checker loses a larger share of the people it could help, 0.46 at J = 0.1 vs 0.20–0.22 at J ≥ 0.4 (c/g = 0.05);
0.37 vs 0.11–0.12 at c/g = 0.01. It is B's orange band divided by everything above the grey band, so it adds no new data.

notes (4 Nov 14:31, addressed):
- Fig S2, panel A also needs lower opacity (apply this low opacity whenever we have these sort of belief histograms please!)
- how to interpret panel D of fig S2?

notes (5 Nov 14:50, addressed):
- Sounds good to split by measure I would say, that's fair. (runs where nothing circulates: section 1)
- Also a small note about Fig S2B - for each checker point, let's add a faint vertical dashed line from the x-axis up to each point (no need to give it any legend label) (added)

notes (5 Nov, addressed):
- S2 B: I think maybe we don't need to do both R2 and R3 because they're so similar. What if instead we keep R2 (the one we use in main experiments), we use R4, and maybe we use one or two more checkers that have a J value that is more around 0.3, and 0.5 respectively. That way we test checkers along more of a spectrum. We should then also update S2 A accordingly.
- S2 C: Not sure if this adds a lot - what do you think? What's the main / new point this panel makes that is missing from other figures? (dropped; numbers in the text)

**Fig. S3 and Table S1 — how the stakes grow with the audience** (`figS3_stakes`, `tableS1_stakes.{csv,md,tex}`;
φ = 0.3, l/g = 2, c/g = 0.05, 2,000 items, 10 runs per form since 6 Nov (from 5); s = k/⟨k⟩; all forms agree for an average audience).
Figure: share of received items checked by number of neighbours for five forms: both stakes ∝ audience (baseline);
penalty ∝ √audience or constant with the gain ∝ audience; both stakes ∝ √audience; both stakes constant. Table S1:

| stakes | not checking | priced out | trapped | split (dip) | two checkers: clear favourite | reach of genuine items | neighbours vs fakes shared (data +0.36) |
|---|---|---|---|---|---|---|---|
| both ∝ audience (baseline) | 0.45 | 0.30 | 0.15 | 0.188 | 0.55 | 0.94 | −0.65 |
| penalty ∝ √audience | 0.41 | 0.26 | 0.15 | 0.171 | 0.66 | 0.61 | −0.19 |
| penalty constant | 0.39 | 0.25 | 0.14 | 0.149 | 0.70 | 0.54 | +0.40 |
| both ∝ √audience | 0.26 | 0.07 | 0.19 | 0.104 | 0.73 | 0.93 | −0.64 |
| both constant | 0.02 | 0.00 | 0.02 | 0.008 | 0.72 | 0.90 | +0.13 |

- Who checks and who spreads fakes depends on the form: checking rises with audience in the baseline, peaks at
  mid-sized audiences when the penalty grows more slowly than the gain, and is flat when both stakes are constant.
  Only the constant penalty matches the sign of the Twitter correlation.
- The two-checker split holds under every form, even when both stakes are constant (clear favourite 0.72): it needs
  only chance early experience. The one-checker split and the trapped group need differences in audience size
  (both stakes constant: almost everyone checks, 2% trapped, no split).

*One checker (Fig. 2)*

**Fig. S4 — robustness** (`figS4_robustness`, c/g = 0.05, 8,000 items, 10 runs per variant since 6 Nov (from 3); the repeat at c/g = 0.01 is
dropped and its result goes in the main text: same picture at cheaper checking). (A, B) Share not checking and (C, D)
split of trust over time for random exploration and partial feedback. Every panel has its own legend (5 Nov).
- Random exploration (each item checked at random with probability ε) slowly dissolves the trap: after 8,000 items the
  trapped share is 0.15 (ε = 0), 0.13 (0.01%), 0.06 (0.1%), 0.015 (1%), and the split of trust vanishes for
  ε ≥ 0.1% (dip 0.21 → 0.017). The share not checking falls only to the share priced out (0.30).
- Why the split first rises, then falls, with exploration (C): until random checks reach the agents who never check,
  the split forms as in the baseline. Then each random check moves a never-checker's trust off the 1/3 spike towards
  the checker's real quality (and the trapped start checking again), so the two groups merge. The peak comes when
  random checks start reaching most never-checkers: ε = 1% peaks at 100 items (80% have ever checked by then),
  0.1% at 500 items (72%), 0.01% at 2,000 items (65%); at the end 100%, 100% and 78% have ever checked.
- Partial feedback (truth revealed to 50%, 20% or 5% of the agents reached; 5% added on 5 Nov): who checks does not
  change (not checking 0.45, trapped 0.15 for every value), the split only forms more slowly: dip after 8,000 items
  0.21, 0.18 and 0.125 (baseline 0.21). With 5% feedback learning is about 20 times slower, so after the first
  discrete-value bump the split falls (0.026 at 500 items) before it grows.
- Dropped from the figure (4 Nov): the old v1 updating rule and a three times stronger prior (no change; prior strength
  is covered by Fig. S1B and the sensitivity analysis), both stakes ∝ √audience or constant (now in Table S1), and the
  end-state panels (correlation with the data now in Table S1).

notes (5 Nov, addressed):
- S4 A: Missing a legend, it's not obvious that it's shared with S4 C. Add the legend for this too
- S4 B: Also missing a legend. Show the legend here, don't share it with S4 D, as that's not very obvious. There is basically no difference with the partial feedback here. Maybe we should try a more drastic value instead of 0.2, such as 0.05
- S4 C: Nice result. Split of trust is much lower for more random exploration. Interestingly the split increases initially up to 100 items, but then starts decaying (we'll want to explain this, why it happens)

**Former Fig. S5 — temporary cuts in the cost of checking: dropped from the SI on 5 Nov** (outcomes barely change; code
`si_cost_cut.py` and data `intervention_twitter`, `intervention_agents_twitter` kept, e.g. for a reviewer).
What the runs showed (main setting, 5 runs per condition, each paired with the control run of the same seed; outcomes
over items 3,000–4,000, after the cut): checking free for 50 / 200 / 1,000 items cuts the share not checking from 45%
to 36% / 33% / 31% for good and brings back 57% / 78% / 91% of the trapped (10% of the cost works as well as free;
half price 33% / 56% / 70%); the priced out are unaffected. But outcomes hardly move: reach gap +0.02 to +0.04
(control 0.46), welfare +0.02 to +0.03 (control 0.65), within wide CIs; the trapped themselves gain most (+0.04 to
+0.05 normalised payoff, from 0.59). The trapped are mid-sized accounts who, once back, check only their most doubtful
items, and most fakes are already stopped by large accounts. This is also why Fig. 4E should be described as "in step
with", not "because of" (notes for the main text).

notes (4 Nov 14:31, addressed):
- Main takeaway of Fig S5?

notes (5 Nov, addressed):
- S5 A: Maybe we're missing some metric about whether the welfare improves due to this intervention? Or whether the reach of genuine items - reach of fake items also improves due to this intervention? Would be nice to see those as well perhaps. Maybe that could replace panel S5 B
- S5 C: Don't really love this panel either. See the above notes in S5 A and S5 B about showing truth differential and welfare. Try those out, maybe there's something interesting there.

notes (5 Nov 13:12, addressed):
- You're right about the new S5 figure, the outcomes are essentially unchanged, error bars are big and differences do not seem significant. Maybe we omit this figure / section of SI entirely.

**Fig. S5 — other settings, one checker** (`figS5_other_settings_one_checker`; S6 until 5 Nov). Rows: main setting, c/g = 0.01
("cheaper checking"), c/g = 0.2 ("more expensive checking"; was "dearer", 5 Nov), φ = 0.6 (c/g = 0.05). The time
rows at c/g = 0.01 and 0.2 now use 20 runs each (as Fig. 2C). Columns: trust after 2,000 items by group (log scale); groups over time; share not
checking against the shared starting trust J₀ (priced-out share dotted); spreading the starting beliefs about the
checker (since 5 Nov 14:10 the split of trust only, on its own 0–0.25 axis, with the share not checking at w = 0 and
w = 0.3 written in each panel: 45% → 53%, 22% → 33%, 72% → 76%, 45% → 54%; the top legend now lists only the
three groups). C and D: 10 runs per point since 5 Nov (topped up from 2–3). At φ = 0.6 with J₀ = 0 nobody checks and
nothing circulates in any of the 10 runs, so that point is missing (noted in the panel). Same picture in every row; the shares scale with the cost (not checking 22%, 45%, 72% at c/g = 0.01, 0.05,
0.2) and hardly change with the share of fakes (46% at φ = 0.6). Column B looks like an epidemic curve: the share who
never checked falls and the share still checking rises until both level off; the crossing point moves later with the
cost and never happens at c/g = 0.2 (checking spreads through the population item by item as each agent meets its
first uncertain item; unlike an epidemic, nobody infects anybody, it is each agent's own threshold).

notes (5 Nov, addressed):
- S6: The legend at the top is a smidge too close to the A, B, C, D panel labels, looks a little tight, add just a bit more space please. ... the "dearer" term is weird, don't use that ... We see in column B that the crossover point essentially moves, or never occurs for the expensive checking. These curves sort of look like SIR dynamics, it's interesting.

*Two checkers (Fig. 3)*

**Fig. S6 — two identical checkers on four networks** (`figS6_two_checkers_networks`; S7 until 5 Nov; rebuilt 5 Nov,
replacing `figS7_two_checkers_detail`). One row per network (Twitter news sharers, Facebook100 Johns Hopkins, Twitch EN,
Holme–Kim; 10 runs per cost since 6 Nov (from 5, 6 on Holme–Kim); Facebook and Twitch run on 5 Nov), columns: (A) joint trust in A and B at
c/g = 0.05 (agents who rely on one checker and never tried the other sit on the two arms); (B) who used which checker
vs cost; (C) loyalty vs cost (share of checks done with the preferred checker; share of checking agents who never
tried the other). The by-degree and AI-detector-vs-professional panels are dropped (they are main Fig. 3D–E); so is
the share not using each checker (numbers in the Fig. 3 notes).
Same picture on every network: with cheap checking almost everyone who checks uses both (used both at c/g = 0.001:
0.91, 0.96, 0.98, 1.00); as checking gets more expensive, users settle on one checker and never try the other (c/g = 0.05:
0.24, 0.23, 0.33, 0.39; c/g = 0.5: 0.55, 0.62, 0.55, 0.51) and checks go to the preferred checker (0.85, 0.88, 0.89,
0.91 at c/g = 0.05). Trust in A and B among agents who checked is negatively correlated on every network (c/g = 0.05:
−0.34, −0.44, −0.37, −0.48). Networks differ only in how many never check (c/g = 0.05: 41%, 21%, 31%, 1%), which
follows their degree distributions.
Takeaway (5 Nov): the split between two identical checkers is not a feature of the Twitter network. On four very
different networks (two real social networks, a streaming platform's and a synthetic one) agents settle on one checker
as checking gets more expensive, stay loyal to it, and their trust in the two checkers becomes negatively correlated.
What the network changes is only how many agents never check, set by the degree distribution (as for one checker in
Fig. S8). Together with Fig. S8 this shows that what matters about the network is who has how big an audience, not
how they are connected.

notes (5 Nov, addressed):
- S7 A through F: Aren't these panels using the twitter network that we're already using in the main results? Don't we already show a lot of these figures in the main text? Maybe I'm confused
- S7: So we're only comparing two different networks? Didn't we have the Facebook one, and even Twitch? I think that for figure S7, we should essentially show panels A through C for each of the different networks, always identical checkers. So you can also sort of drop panels D-F (they're in main text already pretty much)

notes (5 Nov 13:12, addressed):
- Fig S7 - what is the takeaway here? All these different empirical (and synthetic in case of holme-kim) networks behave similarly and have the same sort of emergent pattern of checking and loyalty and trust bifurcation? (yes; takeaway above)

notes (3 Nov, addressed; panel letters refer to the figure as it was then):
- do agents who use two checkers also combine their outputs in some way? like how does the calculation for the estimate that the item is genuine or fake change now that there are two different checker outputs?
- panel F: why is there a line for professional 3x cost, and another for professional 10x cost? i guess it shows how the share of locked out agents increases as the checker is more costly?

**Fig. S7 — other settings, two checkers** (`figS7_other_settings_two_checkers`; S8 until 5 Nov). Same rows as Fig. S5; 10 runs per setting (topped up from 6 on 4 Nov; the AI-detector-vs-professional column from 3 on 6 Nov). Columns: trust in A
minus trust in B after 2,000 items; what agents used over time; use by number of neighbours (identical checkers); AI
detector vs professional by number of neighbours. The three-group split appears in every row; more expensive checking
means fewer users but a larger share of them loyal to one checker. Row label "dearer checking" replaced by "more
expensive checking" (5 Nov).

*Regimes and consequences (Fig. 4)*

**Fig. S8 — other networks** (`figS8_networks`; S9 until 5 Nov; l/g = 2, 2 runs per cell, 1 for the rewired network). (A) The dip and
(B) trapped maps (5 Nov: every map has its own x axis, and the measure is named on its colour bar instead of above the
first map; the trapped colour bar keeps its arrow because Holme–Kim cells reach 0.77, above the scale's 0.5); (C, D) curves for φ = 0.2–0.5 with the one-agent model for each network's degree
distribution. Rewiring the Twitter network while keeping every node's degree changes nothing (curves within 0.004 of
the original; reported in the caption, no longer shown): beyond the degree distribution, the network's structure plays
no role. The one-agent model fits every
network (largest difference 0.003 on Facebook, 0.05 on Twitch and 0.034 on Holme–Kim, at single points on steep parts
of the curves; one-agent model with 500 checks and 6,000 paths since 11 Nov, before 0.005, 0.05 and 0.035). Narrower degree distributions give a sharper switch: almost nobody trapped below some cost, nearly
everybody above it; in between, most agents' thresholds lie between their starting trust and the checker's actual J,
so the trapped share peaks at 0.33 (Facebook), 0.34 (Twitch) and 0.77 (Holme–Kim). The split of trust
is weakest on Holme–Kim (maximum dip 0.12 vs 0.20 on Twitter). Simulation markers in C–D have 95% CI error bars
(run-to-run variance pooled within the cells of the band; at most ±0.014, hidden by the markers).

notes (3 Nov, addressed; panel letters refer to the figure as it was then):
- maybe we remove the twitter degree-preserving? seems basically identical to the twitter news sharer.

notes (5 Nov, addressed):
- S9 A: The share locked out label in Panel A is weirdly positioned. Let's put it under each heat map in that row please, similar to how share of fakes label is below each heat map in panel B
- S9 C and D: We again are seeing the prediction of the one-agent model here. Does the existence of this theoretical prediction (one agent model) mean there is no point of actually running any simulations?? (answered in section 4, point 9)

**Fig. S9 — other loss/gain ratios, and the one-agent model** (`figS9_other_loss_gain_ratios`; S10 until 5 Nov). (A–H) The maps of
Fig. 4A–D for l/g = 1 and 4: same picture; with l/g = 4 nothing circulates in a larger region (84 of 352 cells: many
fakes and high cost). Trapped colour scale 0–0.25 without the arrow (largest cell 0.21). (I, J) Not checking and
trapped against cost for φ = 0.2–0.5 and l/g = 1, 2, 4 (former Fig. 4E–F): the one-agent model matches the
simulation everywhere (largest difference 0.004, 0.002 and 0.008 since 11 Nov, before 0.004, 0.003 and 0.010; it takes the share of fakes among checked items from
the runs, nothing else). Simulation markers have 95% CI error bars as in Fig. S8C–D (at most ±0.005, hidden by the
markers).

notes (4 Nov 14:31, addressed):
- Fig S10: panels C and G, why does the color bar have some arrow at the top compared to the other heatmaps? A bit strange, also we never seem to reach the dark dark red/brown color near the 0.3 color scale? (the arrow meant "values above 0.3 shown in the darkest colour", which never happens: largest cell 0.21; scale now 0–0.25, no arrow)

**Fig. S10 — global sensitivity analysis** (`figS10_sensitivity`; S11 until 5 Nov; panel label A moved to match the others, 5 Nov; 1,000 Latin-hypercube runs over φ, c/g, l/g, J₀, κ,
spread of starting trust, TPR, FPR; ranges as in v2; PAWN indices with 95% bootstrap CIs; a dummy input gives the
noise floor). Nothing circulates in 28% of the runs (36% in v2, where everyone started at a base rate of 0.5), mostly
decided by φ. In the 715 runs where items circulate: the share not checking is driven by c/g (0.38) and J₀ (0.20); the trapped share by J₀
(0.44) — pessimistic starting beliefs keep exactly the people who would benefit from ever trying; the dip only by J₀
(0.29); the reach gap by c/g, φ and the checker's hit rate; welfare by φ, the hit rate and c/g. Prior strength, the
spread of starting trust and l/g stay at the noise floor for every outcome.

**Fig. S11 — inequality of audiences vs average audience** (new, 6 Nov; provisional number, it supports Fig. 4 and Fig.
S8; `figS11_inequality`). Your question (6 Nov, 10:35): does more connectivity mean more split, as in Thurner et al.?
Random networks (`synthnets.py`): N = 5,000, degrees from a lognormal (quantiles, rounded, at least 1) with a set mean
and inequality, wired at random (configuration model; self-loops and double edges removed by degree-preserving swaps;
largest component kept: 99.8–100% of nodes, 98.5–99% for the sparsest); main setting φ = 0.3, l/g = 2; 19 costs
c/g = 0.001–1; 10 runs per point, each on its own wiring (`ineq_cv`, `ineq_k`; 2,090 runs). Four panels since 6 Nov
12:35, on a 2 × 2 grid at 1.5-column width: (A) the dip and (B) the share not checking against c/g for 5 of the 7
inequality levels (Gini 0, 0.26, 0.45, 0.64, 0.73); (C, D) the same for 3 of the 5 average audiences (4.35, 20, 92.8).
The summaries (largest split, cost range with a clear split), the real networks' Gini values and the absolute-stakes
reading moved from the figure to the text below (`si_inequality.py <data> --numbers` prints them).
- Top row (A, B), average audience 20, inequality rising (Gini of the number of neighbours 0, 0.14, 0.26, 0.45, 0.56,
  0.64, 0.73, i.e. CV 0–3; the real networks: Holme–Kim 0.37, Facebook 0.49, Twitch 0.61, Twitter 0.68).
  With equal audiences everyone has the same threshold, so at each cost nearly everyone checks or nearly nobody does
  (not checking jumps from 0.1 to 0.9 between c/g = 0.15 and 0.32); a split appears only in that narrow window, from bad
  luck just below the cost where checking stops paying (largest dip 0.100 ± 0.012 at c/g = 0.22; dip > 0.1 over 0.03
  decades of cost). With unequal audiences thresholds spread out, the switch stretches over 1.5 decades (c/g =
  0.015–0.46 at Gini 0.64), and the split lasts: the range of costs with dip > 0.1 grows steadily, 0.03, 0.12, 0.28,
  0.48, 0.64, 0.71, 0.99 decades (CIs at most ±0.03). The largest split grows less (0.10, 0.13, 0.14, 0.16, 0.165, 0.17,
  0.165): its height is capped by the group sizes. Inequality sets over how wide a range of costs the population divides,
  and moves that range to cheaper checking (peak at c/g = 0.22 with equal audiences, 0.03 at Gini 0.73).
- Bottom row (C, D), inequality fixed (CV 1, Gini 0.45), average audience 4.35, 9.28, 20, 43.1, 92.8. In the model (stakes
  relative to the average audience) the curves coincide from 9 neighbours on (dip within 0.016, not checking within
  0.06 at every cost): giving everyone more neighbours leaves every threshold unchanged. The sparsest network differs a
  little: a fifth of its agents have one neighbour, hear from a single sender, learn its record fast and rarely meet an
  item doubtful enough to check (median 2 checks in 2,000 items, against ~600–1,000 for agents with 3 or more
  neighbours), so their trust stays near the start: a small split of another kind at cheap checking (dip 0.03 vs 0.005).
- Text only (panels G–H of the first version): if stakes were absolute instead (the same gain for every neighbour reached; c/g relative to the gain of an agent with
  20 neighbours; the same runs re-read, since choices depend only on cost/stake), more connectivity works like cheaper
  checking: the share not checking falls with the average audience (c/g = 0.046: 0.73, 0.40, 0.14, 0.03, 0.01) and the
  split first rises and then falls (dip 0.08, 0.16, 0.04, 0.006, 0.004), peaking at more neighbours when checking costs
  more (c/g = 0.22: 0.004, 0.03, 0.10, 0.15, 0.04, peak at 43 neighbours). At c/g = 0.01 the dip only falls (the 0.09
  at 4.35 neighbours is the sparse-network effect above).
- Where checking never pays on the (nearly) equal networks (Gini 0–0.26, c/g ≥ 0.2–0.46), nothing circulates in 2–8 of
  the 10 runs: nobody checks and too few share. There the dip is 0 and the share not checking 1 in every run that
  circulates, so those points rest on fewer runs without changing the picture (stated in the caption).
- Takeaway: in our model the divide follows who has a bigger audience than others, not how connected everyone is. Even
  with absolute stakes, rising connectivity would first create and then remove the split, unlike the steady rise with
  connectivity in Thurner et al. Main text: one sentence where Fig. 4 is discussed, plus the discussion line in
  section 4, point 11.

notes (6 Nov 12:35, addressed):
- Thanks. I do like this Figure S11. I think most interesting are panels A, B, E, and F. Let's only have those four panels please. For A and B, I think we can plot only 5 different Gini curves (including all 20), there are a bit too many right now. Also for E and F, let's just do maybe 3 (around 4, the 20, and around 92,.8). (done: A–D on a 2 × 2 grid; Gini 0, 0.26, 0.45, 0.64, 0.73 and 4.35, 20, 92.8 neighbours; the dropped summaries and the absolute-stakes reading are in the text above)
- Looks good I would say! althouh in the bottom row (C and D) the 20 and 92.8 basically totally overlap, perhaps not very insightful? would a different value be more interesting? (the overlap is the finding: with stakes relative to the average audience, giving everyone more neighbours leaves every threshold unchanged, so 9.28, 20, 43.1 and 92.8 all give the same curves; only the sparsest network (4.35) differs, through its one-neighbour agents, and a sparser one would mostly add that artefact. Kept 92.8, the widest range (20×), drawn with hollow markers on top of 20, plus the note "20 and 92.8 neighbours: the same curve" in C)
- Thanks. let's just remove the text label "20 and 92.8 neithbours: the same curve" from panel C please. (removed; the hollow markers stay)

**Fig. S12 — survey data and the model, side by side** (new, 6 Nov; provisional number, it supports the discussion
and Fig. 2; `figS12_survey_and_model`, `si_survey.py`). Your request: overlay the GSS data with our
model output to show they are similar; side by side chosen over common axes, since the two have no shared units.
(Request of 6 Nov, afternoon.)
- (A) US General Social Survey 1973–2024 (`gss_press.py`, extract in `data/gss/`): share of adults with "hardly any"
  confidence in the press, by party (Democrats and Republicans with leaners; independents), weighted, 95% CIs from the
  weighted (Kish) sample size; dotted line at the switch to web / mixed mode (2021). Numbers in section 4, point 6:
  the Republican–Democrat gap was 2 points in 1973, mostly 10 or less until 2000, 12–23 in 2002–2016 and 41–55 since
  2018 (78% vs 33% in 2024).
- (B) Model at the main setting (φ = 0.3, c/g = 0.05, l/g = 2), the Fig. 2 runs re-run with a per-group summary at
  every snapshot (`time_deg_twitter`: same seeds, so the same dynamics; 20 runs, 4,000 items): mean trust in the
  checker among the thirds of the network by number of neighbours (fewer than 35: 4,693 agents; 35–209: 4,721; 210 or
  more: 4,714); 95% CIs over runs (Student's t); log axis as in Fig. 2. Everyone starts at 1/3. The bottom third never
  moves: at this cost nobody with fewer than 35 neighbours ever checks, so their trust stays at 0.333. The middle third
  rises to 0.560 (24% of it never checks) and the top third to 0.689 (all check; the checker's actual J is 0.70); CIs
  ±0.002 at the end. The gap between the top and bottom thirds reaches half its final size (0.36) by item 28 and 90%
  by item 185, then stays (0.355 at 2,000 items, 0.356 at 4,000).
- Shared pattern: groups that start alike drift apart and the gap does not close. Differences, for the caption: the
  survey's groups are parties and its split is partisan, with timing that follows political events; the model has no
  opinions or parties, and its groups differ only in what is at stake for them (audience size). The measures differ
  (distrust of the press vs trust in a verification tool; up is distrust in A, trust in B), as do the clocks (years
  vs items) and the timing (the model's gap opens within the first few hundred items, the survey's after 2000). So
  does the way the gap opens: in the model trust rises among agents who use the checker, because it is better than
  they first believed, and stays where it began among those who never try it (no group's average trust falls); in the
  survey distrust rose in every group (Democrats 13% → 33%, independents 16% → 56%, Republicans 15% → 78%, 1973 →
  2024), fastest among Republicans. No correspondence between parties and audience groups is implied, and colours keep
  them apart (violet for audience groups). Use: motivation and a qualitative parallel, not evidence for the mechanism.
- Distrust versions of B, tried on 6 Nov and set aside (both drawn by `si_survey.py <data> <out> <measure>`):
  - `distrust` (14:58): the believed error rate of the checker, (1 − Ĵ)/2 = ((1 − t̂) + f̂)/2, the mean of the
    believed miss and false-alarm rates, in %: everyone starts at 33% (t₀ = 2/3, f₀ = 1/3), the bottom third stays
    there, the middle third falls to 22.0% and the top third to 15.5% (the checker's actual error rate is 15%: misses
    20%, false alarms 10%). The same curves upside down, so up is distrust in both panels but the model's fan opens
    downward. You preferred the trust version (15:03).
  - `notuse`: the share whose trust is too low for their stakes (never tried the checker, or tried it and stopped;
    your two sources of low trust), a share like the survey's: it starts split, 100% / 24% / 0%, because low stakes
    keep the bottom third out from the start, and ends at 100% / 33% / 1.3% (the middle third's rise is agents who
    tried the checker and gave up). It shows the groups differing, not drifting apart from a common start.
- Possible caption opening: "Trust splitting over time in survey data and in the model. (A) Share of US adults with
  'hardly any' confidence in the press, by party (General Social Survey, 1973–2024). (B) Mean trust in the checker
  over time for agents with few, middling or many neighbours (thirds of the network), main setting, 20 runs. In both,
  groups that start alike drift apart and the gap does not close; the groups, measures and time scales differ (see
  text)."

notes (6 Nov 14:58 and 15:03, addressed):
- any way to make the right plot the "distrust" ? (done at 14:58: B as the believed error rate of the checker)
- i think i  liked the earlier figure more (done: B back to mean trust; the distrust versions are described above)

(The earlier Fig. S12 — welfare and beliefs about the share of fakes — moved into the main text on 4 Nov as Fig. 4F–G;
its number is reused for the survey figure above.)

(The LISS checks, formerly an SI figure, are dropped; `liss_checks.py` and `data/v3/liss_aggregates.json` stay in the repo.)

## 4. Points to discuss

1. **Headline — decided (4 Nov).** See section 1. Background: with identical starting beliefs, the dip mostly measures the
   distance between agents who never check (stuck at J₀) and the users; it keeps rising with run length (0.19 after
   2,000 items, 0.21 after 8,000), vanishes when the checker is about as good as people first assume (Fig. S2C),
   shrinks when starting beliefs differ (Fig. S1C) and vanishes with a little random exploration (Fig. S4C), while the
   shares not checking and trapped hold.
2. **Exploration — decided (4 Nov): random exploration in the SI (Fig. S4); the temporary cost cut dropped from the
   paper (5 Nov).** The trap is permanent only if nobody ever checks against their own
   judgement: with 0.1% of items checked at random the trapped group shrinks by 60% within 8,000 items; with 1% it
   is gone. The cost cut works the same way (agents check again, so trust can recover) but is a deliberate,
   time-limited lever rather than an assumption about behaviour.
3. **Payoff scaling — decided (4 Nov): keep the baseline; SI: Fig. S3 and Table S1 (five forms of the stakes).**
   Background: keep both stakes ∝ audience (decided 3 Nov) and report the variants in the SI, or switch
   the loss stake to one that matches the data's positive degree–misinformation correlation (+0.46 with loss ∝ 1 vs
   +0.36 in the data; −0.68 in the baseline; main results barely change; Fig. S3). This is the paper's only
   direct comparison with data, so the choice also decides whether that comparison supports the model or counts
   against it. If the baseline stays, say that the data correlation has causes outside the model (the sample was
   selected for sharing low-credibility links; partisanship, very active accounts).
   Quick probe (4 Nov; main setting, 2,000 items, 1–2 runs per variant): with the loss fixed, the biggest accounts
   (k > 1,000) check under 6% of the items they receive (81–84% in the baseline): a false alarm costs them much
   reach, a fake only the fixed penalty, so they skip checks although they trust the checker, a third reason for
   not checking that the threshold measure (Ĵ ≤ θ) does not count. Accounts with k < 10 share 0.3% of what they
   receive (93% in the baseline). The two-checker split survives (used both 49%, only one 14%, neither 37%; trust in
   A vs B among agents who checked −0.46, vs −0.34 in the baseline). Prediction P2 (checking rises with audience
   size) holds only with the baseline payoffs; with the loss fixed, checking peaks at mid-sized audiences.
4. **Only the degree distribution matters (to state plainly).** The one-agent model fits every network; degree-preserving
   rewiring changes nothing (Fig. S8).
5. **Static vs dynamic.** At the main setting the 14.8% trapped are 11.1% who never tried (starting trust below
   their threshold) plus 3.7% who tried, had bad luck early and stopped. The temporary cost cut rescues both. With two
   checkers, *which* checker an agent trusts is decided by early experience only.
6. **Empirical anchors — decided (4 Nov).** No own online experiment. The paper is a theory paper whose ingredients
   and outcomes are each tied to existing evidence, plus explicit predictions:
   - *Inputs, calibrated*: the network is the Twitter news-sharer network (Nikolov et al. 2021); the checker's error
     rates (TPR 0.80, FPR 0.10) are those of the best commercial detectors in a 2024–25 benchmark (Chandra et al. 2025).
   - *Mechanism, from experiments*: people stop sampling options that disappointed them, so bad early experiences
     persist (hot stove effect, Denrell & March 2001; negative first impressions, Denrell 2005); learning traps when
     feedback depends on one's own choices (Rich & Gureckis 2018); people avoid an algorithm after seeing it err
     (Dietvorst et al. 2015); reliance on decision aids follows trust built from observed errors (Dzindolet et al. 2003).
   - *Outcomes, from surveys*: verification is a minority habit — 13% do not know how they would check suspicious
     information; people with less formal education and on the political right are less likely to name fact-checkers
     (Reuters DNR 2025). Trust in fact-checkers is split — 70% of Republicans vs 29% of Democrats say they favour one
     side (Pew 2019). That real split is partisan; the model shows a split arises even without opinions, which is
     the bridge to the future work.
   - *Motivation, long-run survey data (6 Nov; now beside the model in Fig. S12, `gss_press.py`, extract in `data/gss/`)*: confidence
     in the press, US General Social Survey 1973–2024 (weighted; 95% CIs from the effective sample size). "Hardly any"
     confidence rose from 15% to 55% of adults ("a great deal" 23% → 7.5%, "only some" 62% → 38%). By party, the gap in
     "hardly any" between Republicans and Democrats was 2 points in 1973, mostly 10 or less until 2000, 12–23 points in
     2002–2016 and 41–55 since 2018 (78% vs 33% in 2024). Groups that started almost alike drifted apart and the gap has
     not closed, like the model's split from identical starting beliefs; but the real split is partisan and its timing
     points to political events, so it is motivation, not evidence for the mechanism. Supporting numbers: Gallup, trust
     in the mass media 68% (1972) → 28% (2025), Republicans 8% vs Democrats 51% in 2025; Pew 2025, CNN trusted by 58% of
     Democrats and distrusted by 58% of Republicans, Fox News trusted by 56% of Republicans and distrusted by 64% of
     Democrats. Possible sentence: "Trust in news has not only declined but split: in 1973, Democrats and Republicans
     hardly differed in their distrust of the press; by 2024 the gap was 45 points. Our model shows how such a split can
     arise even without partisan differences, from identical starting beliefs, unequal stakes and the luck of early
     experiences." (The Reuters DNR US series and the newspaper-reading breakdown were looked at and set aside, 6 Nov.)
   - *Intervention, from a field experiment* (dropped with the cost cut, 5 Nov): a short free period raises long-run
     adoption through learning from experience (Dupas 2014, bed nets in Kenya).
   - *Predictions (for a box or the discussion)*: (P1) users whose first checks go wrong stop using the checker far
     more often than its average accuracy warrants, and rarely return on their own (main setting, agents for whom
     checking pays: 30% stop for good after a bad first check vs 3% after a good one); (P2) verification use rises with
     audience size (baseline payoffs only; see item 3); (P3) among people who use verification, trust in one of two similar checkers is negatively related
     to trust in the other (−0.34 among agents who ever checked, main setting), the favourite is set by early
     experience rather than quality, and loyalty to one checker is more common when checking is more expensive (clear
     favourite among those who can still check: 31% / 55% / 79% at c/g 0.01 / 0.05 / 0.2); (P4; dropped with the cost cut on 5 Nov) a temporary free or
     cheap period raises use for good, most among mid-sized audiences: at the main setting the trapped sit at
     27–120 neighbours (never tried at 27–57, tried and burned mostly at 57–120), the priced out below 27.
   - *In reserve*: X Community Notes public data (notes and ratings) if reviewers ask for a data test.
7. **Positioning (for the introduction).** Closest literatures: the hot stove effect and learning traps (Denrell &
   March 2001; Denrell 2005; Rich & Gureckis 2018), including its group version, the collective hot stove effect
   (Le Mens, Kovács, Avrahami & Kareev 2023: group ratings correct overestimates more than underestimates; model plus
   an online experiment and 160 million online ratings), algorithm aversion, network epistemology (Zollman 2010;
   O'Connor & Weatherall 2018), and costly-verification models of misinformation (Papanastasiou 2020; Acemoglu,
   Ozdaglar & Siderius 2024; Cheredina & Lukyanov 2025). None has agents learn a checker's reliability only by paying
   to use it, with stakes that scale with audience, or several checkers.
8. **Social reward ("being liked") — decided (4 Nov): left out of the paper.** Payoff from the audience resharing what one shared. Two forms:
   (a) a like-reward per share that does not depend on truth adds the same amount to both outcomes, which equals a
   smaller loss-to-gain ratio (already covered by the l/g sweep; checking dies out only if the like-reward exceeds the
   penalty for a fake); (b) a reward from reshares makes the penalty for a fake depend on the audience's own checking
   (a flagged fake is not reshared), so checking pays only if one's audience checks: a coordination game with tipping
   points, local checking cultures and communities converging on one checker. Leaning: out of scope for this paper
   (adds parameters but no data; blurs the claim that the split needs no social motive); discussion and future work,
   with opinions. Our payoff reads as reputation with one's audience: sharing fake news hurts reputation more than
   sharing true news helps (Altay, Hacquin & Mercier 2022, New Media & Society), matching l > g. Evidence that sharing
   habits follow platform rewards, and that rewarding accuracy improves sharing: Ceylan, Anderson & Wood 2023 (PNAS).

9. **Why run the network simulation if the one-agent model predicts it? (5 Nov, your question on Fig. S8C–D, numbered S9 at the time).** The
   one-agent model is not a substitute; it is the explanation. (i) It needs an input only the simulation gives: the
   share of fakes among the items agents check, which depends on what the network filters upstream (and that in turn
   on who checks). (ii) It predicts only who ends up not checking or trapped, for one agent in a steady stream of
   items. It says nothing about anything collective or dynamic: what spreads (reach gap), welfare, believed share of
   fakes, the time course, the split between two checkers, or the effect of a cost cut. (iii) Its success is itself a
   result: given that one input and the degree distribution, the network's structure beyond audience size does not
   matter for who checks (degree-preserving rewiring changes nothing, Fig. S8). Framing for the paper: the simulation
   shows the phenomena and their consequences; the one-agent model shows we understand why, with nothing fitted.
10. **Comparison with Thurner, Hofer & Korbel 2025 (PNAS 122:e2517530122) — recommendation: no panel (6 Nov).** Your
   question: add a panel next to Fig. 2C comparing our dip over time with the polarization over time in their Fig. 1D.
   Their measure is ψ = Var(a_ij), the variance of pairwise opinion alignment over 10 binary Pew items, not
   bimodality. Their repository (github.com/complexity-science-hub/ConnectivityPolarizationTransition) holds the six Pew
   surveys; recomputed per year (1999, 2004, 2011, 2014, 2015, 2017; n = 985–10,013): ψ = 0.128, 0.140, 0.149, 0.171,
   0.177, 0.200 (matches their code), but the dip of the same ideology distribution is 0.007, 0.005, 0.004, 0.002,
   0.003, 0.006: no rise, and no more than sampling noise gives a single hump of these sizes. (The ideology score is the
   mean of 10 answers coded +1/−1/0, so it takes 21 values; odd tenths need an odd number of "neither/don't know"
   answers and are rare, a sawtooth. Each value is spread over one full step to average it out; the half-step jitter
   used first kept the sawtooth, and gave 0.032, 0.017, 0.016, 0.016, 0.015, 0.020, corrected 6 Nov.) The distribution widens and flattens
   (SD 0.40 → 0.56, excess kurtosis −0.04 → −1.05, bimodality coefficient 0.35 → 0.54, below the 0.555 threshold)
   instead of splitting. The levelling-off in their Fig. 1D is their model's prediction, driven by a logistic fit of
   friendship counts; the survey points themselves still rise in 2017. A dip-vs-dip panel would show a flat real
   curve; ψ vs our dip would compare different measures of different things (political opinions vs trust in a
   checker) on different clocks (years vs items). Suggested instead: cite them in the discussion as a different,
   connectivity-driven mechanism that also yields rising polarization; our plateau has a specific explanation (group
   sizes). Script: `pew_dip_check.py` (reads the survey files in their repository). A comparison figure for discussion (not for the paper; 6 Nov): our dip and bimodality coefficient over time against the same two measures of the Pew surveys on a common time axis (share of each series' window). The Pew bimodality coefficient rises steadily (0.35 → 0.54) while its dip stays flat, i.e. the ideology distribution widens and flattens; ours crosses 5/9 within ~100 items and levels off at 0.74–0.80. The S shape in their Fig. 1D is in ψ only (per year: +0.001, +0.004, +0.011, +0.006, +0.005 between consecutive surveys) and in their model curve.

11. **Connectivity and the split (6 Nov, your question).** Agents with more neighbours check more: the threshold falls
   as 1/k because both stakes grow with the audience (Fig. 1B; by number of neighbours: Fig. 1D, Fig. S3, and Fig. 3D–E
   for two checkers). At the main setting the median degree is 12 among agents who never checked, 85 among those who
   checked and stopped, 267 among those still checking; degree and final trust correlate at ρ = 0.72 (Spearman). So
   the two humps are small accounts (stuck at their starting trust) and large accounts (trust near the checker's real
   quality). This needs unequal audiences: with both stakes constant almost everyone checks and there is no split
   (Table S1: dip 0.008); with a constant penalty, checking peaks at mid-sized audiences (Fig. S3). Unlike Thurner et
   al., overall connectivity does not drive it: stakes depend on the audience relative to the average (s = k/⟨k⟩), so
   giving everyone more neighbours leaves every threshold unchanged, and across networks the split does not follow
   the average degree (largest dip: Twitter ⟨k⟩ = 252, 0.20; Facebook 72, 0.175; Twitch 10, 0.17; Holme–Kim 6, 0.12).
   With stakes in absolute audience, more connectivity would act like cheaper checking (θ ∝ c/k): the split would
   first grow and then shrink as nearly everyone checks (the cost axis of Fig. 4A), not rise steadily. Tested on 6 Nov
   (Fig. S11, random networks): at a fixed average audience the split grows with the inequality of audiences; at a
   fixed inequality the average audience does not matter; with absolute stakes the split first rises and then falls as
   connectivity grows. Possible line for
   the discussion: in Thurner et al. polarization rises when everyone becomes more connected; here the divide follows
   who is more connected than others.

12. **Learning from every sender (11 Nov, your comment on the model description) — decided 11 Nov 14:19: keep the first-sender rule; report the variant in the SI.** In the model an
   agent updates only the record of the first neighbour who passed it an item. Variant (`engine.run(all_senders=True)`,
   experiment `senders_twitter`): once the truth is out, the agent updates the record of every neighbour who shared the
   item; decisions still use the first sender's record. 10 simulations per cost, paired with the baseline (same seeds);
   main setting otherwise; differences variant − baseline with 95% CIs over the 10 pairs:

   | c/g | not checking | trapped | ever checked | dip | check rate | reach of fakes | reach gap | welfare |
   |---|---|---|---|---|---|---|---|---|
   | 0.01 | 0.225 → 0.208 | 0.099 → 0.082 | 0.80 → 0.76 | 0.099 → 0.127 (+0.028 ± 0.014) | 0.61 → 0.48 | 0.32 → 0.46 | 0.61 → 0.47 | 0.76 → 0.63 |
   | 0.05 | 0.447 → 0.426 | 0.147 → 0.126 | 0.59 → 0.53 | 0.190 → 0.131 (−0.059 ± 0.012) | 0.48 → 0.32 | 0.49 → 0.59 | 0.46 → 0.36 | 0.65 → 0.49 |
   | 0.2 | 0.722 → 0.706 | 0.193 → 0.176 | 0.31 → 0.27 | 0.098 → 0.069 (−0.029 ± 0.009) | 0.27 → 0.17 | 0.70 → 0.78 | 0.26 → 0.19 | 0.45 → 0.33 |

   The groups survive: the priced out are unchanged and the trapped fall only by those trapped after bad luck (3.6% →
   1.6% at c/g = 0.05; the never-tried trapped are untouched). But agents see each neighbour's sharing on every item, so
   sender records become far more informative (≈ 200× more record updates), p moves away from the checking window more
   often, and agents check a third less. Fakes then reach 8–14 points more agents, the reach gap shrinks and welfare
   falls by 0.13–0.16; the split changes with the group sizes (down at c/g = 0.05 and 0.2, up at 0.01). Records carry
   real information in both versions: items passed on by neighbours who still check are 82% genuine, by neighbours who
   do not 62% (base rate 70%; main setting, baseline). Options: keep the first-sender rule (decision at first exposure)
   and report this in the SI as a robustness check whose direction is clear (more knowledge of senders crowds out
   checking); or switch the baseline and rerun every figure.

13. **κₛ, the weight of πᵢ in a sender's record (11 Nov, your questions) — check run; recommendation: keep κₛ = 2.** p = w·(n⁺/n) + (1 − w)·πᵢ
   with w = n/(n + κₛ): πᵢ counts as κₛ imagined items from the sender, so the record gets half the weight after κₛ
   items (2 now). It is partial pooling: a Beta prior over senders' genuine rates with mean πᵢ and weight κₛ; small κₛ
   means agents assume senders differ a lot. How much they really differ (main setting, 3 simulations, pairs with ≥ 20
   items): genuine rates 0.85 ± 0.10 between senders (binomial noise removed), which a Beta prior matches at κₛ ≈ 11
   (10.7–12.9); κₛ = 2 assumes a spread of ±0.21, twice the real one, so records take over sooner than the data
   warrant. Overlap with πᵢ (your second question): every item from j updates both πᵢ and j's record; the top sender
   supplies a median 37% of the evidence behind πᵢ for agents with fewer than 35 neighbours, 18% for 35–209, 12% for
   210 or more. It barely matters: shrinking towards πᵢ computed without j's own items changes p by 0.0002 on average
   (99th percentile 0.0006), because the overlap is large only where the agent has so many items from j that the record
   dominates anyway. Check run 11 Nov (`kappas_twitter`: κₛ = 0.5 and 11 at c/g = 0.01, 0.05, 0.2, 10 simulations
   each, paired by seed with the baseline runs of `senders_twitter`), differences from κₛ = 2:

   | measure | κₛ = 0.5 | κₛ = 11 |
   |---|---|---|
   | not checking, trapped, ever checked | at most 0.003 (within the CIs) | at most 0.004 |
   | dip | −0.005 at c/g = 0.05 (0.190 → 0.185), less elsewhere | +0.004 at c/g = 0.05 (→ 0.194), +0.003 at 0.2 |
   | share of received items checked | −0.006 to −0.016 | +0.025 to +0.034 |
   | reach of genuine items and fakes, reach gap | at most 0.02 (within the CIs) | at most 0.01 (within the CIs) |
   | welfare | −0.028 ± 0.022 at c/g = 0.05, within the CIs elsewhere | +0.003 to +0.013 (within the CIs) |
   | believed share of fakes | no change | no change |

   κₛ barely matters: it only changes how the first few items from each neighbour are weighed, and most items come
   from established senders. A larger κₛ keeps p near πᵢ a little longer, so agents check a little more (2.5–3.4
   points), without changing who checks, the trapped or what spreads. Unlike point 12, where agents get ≈ 200× more
   information about senders. Recommendation: keep κₛ = 2; one sentence plus these numbers in the SI.

### Figure walkthrough (5 Nov): your reading, with corrections — key points for writing

*Fig. 1.* A: agents share, discard or check; p comes from the belief about the base rate and the trust in the
neighbour who sent the item (its record); highest expected payoff wins; beliefs update when the truth is revealed;
checking can pay only if trust in the checker exceeds θ. — B: θ depends on the number of neighbours (and the cost and
stakes), not on trust; trust is compared with it. "Right of the curve" means "checks" only at the height of the
agent's own trust, which at the start is 1/3 for everyone: that is why the groups are read along the starting-trust
line (redrawn that way). — C: 60 identical agents (θ = 0.2, about 95 neighbours, same starting trust) that differ only
by luck; one-agent model, not picked from the network run. The orange ones are trapped after trying. — D: share not
*checking* (not "not sharing") by number of neighbours; the switch sits between ~10 and ~200 neighbours depending on
c/g.

*Fig. 2.* A: trust bifurcates over time; the agents who never check do not become more doubtful, they stay exactly at
the starting trust because they never update. — B: as you read it. — C: at 1,000 items the order is c/g = 0.05 (0.16)
> 0.01 (0.10) > 0.2 (0.086); 0.2 catches up with 0.01 only around 2,000 items and heads for its higher ceiling (0.14).
The plateaus are set by group sizes (triangles), and the split is largest at the intermediate cost because the two
groups are about equal there. The early bump and dip come from the first few checks giving only a handful of trust
values (Fig. 2 notes).

*Fig. 3.* A, B, C: as you read them. The slow decline of "only A" / "only B" (9% → 7%): about 3% of agents (median
189 neighbours) add their second checker after item 100, two thirds of them through one check with both checkers on a
doubtful item, and then keep both (final trust 0.67 and 0.63). The clear-favourite share (C) peaks early (~150 items
at c/g = 0.05) and declines slowly for the same reason. — D: "used both" means used each checker at least once. All
accounts above ~300 neighbours have; above 1,000 neighbours two thirds of their checks use both checkers at once
(300–1,000: 38%; 150–300: 4%). — E: nobody uses the professional alone, so that line is "both" (relabelled); starting
beliefs about the two checkers are identical; the detector is tried first because it is cheaper, the professional
pays at the start only above ~170 neighbours; those who try it learn that it is better (trust 0.87–0.91 vs 0.53–0.57).

*Fig. 4.* A: the split peaks around c/g ≈ 0.03–0.05 (below the 0.1 tick); φ matters little, except that the split is
weaker when fakes are rare (φ < ~0.15) and nothing circulates when they are very common. — B: nothing circulates only
when fakes are common *and* checking is not cheap (top right); with cheap checking items circulate even at φ = 0.95.
The hatching is the same in A–D. — C: the trapped share peaks around c/g ≈ 0.2–0.3 (halfway between the 0.1 and 1
ticks on the log axis is 0.32; largest cell 0.21). — D: largest gap with cheap checking and φ ≈ 0.3–0.9; your reason
is right: with more fakes agents are less sure, more items fall where checking pays, and fakes get stopped. — E: each
dot is a cell (φ, c/g), averaged over runs. Careful with causal language: the correlation is across costs; the cost-cut
runs (not in the paper) show that bringing back only the trapped moves the reach gap a little. — F: as you read it
(checking costs included). — G: right: a selection effect, people see fewer fakes than are produced because checked
fakes stop early. Also: where nobody checks (c/g = 1) beliefs stall near 1/3 for φ ≈ 0.3–0.55, consistent with
sharing stopping once agents believe more than a third are fake (l/g = 2: share only if p > 2/3), so items stop
reaching them and they stop learning.

*Fig. S1.* A, how to read it: each cell is one combination of the starting trust (the same for everyone, x) and the
cost of checking (y; axes swapped on 5 Nov to match the other maps); its colour is how split trust is after 2,000 items. The dashed line marks where half of the
agents start checking (the starting trust equals the threshold of the median account); below and to the right of it
(lower cost, higher starting trust) most agents start, above and to the left most never do. The split is strong along
that line, where the population divides into two groups of similar size: agents who start checking and learn, and
agents who never start and stay at J₀. It fades towards low cost (almost everyone checks, no second group), towards
high cost (almost nobody checks), at J₀ = 0 (nobody ever starts) and at J₀ = 0.6 (the checkers' trust ends near the real 0.7, too close to the non-checkers' 0.6 to form two humps).
Takeaway: the split is not a knife-edge result; it appears over about a decade of costs for any starting trust between
0.1 and 0.5, shifting to higher costs as the starting trust rises. — B: falls steeply, then levels off towards the priced-out floor (0.30), rather than
exponentially; prior strength matters little; more starting trust puts more agents above their threshold at the start
and gives those who start more margin before bad luck traps them. — C, D (one panel until 14:10): why spreading the starting beliefs lowers the
split: with identical beliefs all never-checkers sit on one value, a spike far from where users end up, and that spike
is what makes the distribution so clearly two-humped. Spread the starting beliefs and the never-checkers fill the gap,
so the distribution stops looking bimodal although *more* agents do not check (0.45 → 0.53) and more are trapped
(0.15 → 0.23). Caveat for the paper: the dip partly reflects identical starting beliefs; the groups themselves persist.

*Fig. S2.* A: R3 looks like R2 because J is the same; R1 and R4 (J ≈ 0.1) look alike too: most agents who try them
learn that they are poor and stop (rightly: at that J most are priced out). Now four checkers along J instead. J is
not everything, though: with the same J, a checker whose errors are false alarms on genuine items loses a few more
users (the audio detector and people, FPR 0.34 and 0.32, sit above the curve in B).

*Fig. S3.* As you read it: in the baseline checking rises between ~20 and ~100 neighbours, the bands of Fig. 1B.

*Fig. S4.* Exploration (C): the split rises as in the baseline until random checks reach the never-checkers, then
falls as their trust leaves the 1/3 spike; peak at ~100 / 500 / 2,000 items for ε = 1% / 0.1% / 0.01%. Partial
feedback (D): who checks does not change, only how fast the split forms (5% feedback: dip 0.13 after 8,000 items).

*Cost cut (former Fig. S5, dropped 5 Nov).* The cut changes behaviour for good, outcomes only a little, and mostly for
the agents it brings back.

*Fig. S5.* Column B does look like an epidemic curve, and the crossing moves later with the cost (none at c/g = 0.2);
but nothing is contagious: each agent starts checking at its own first doubtful item.

*Fig. S6.* The same two-checker pattern on four networks; only the share who never check differs (degree
distribution).

*Figs. S8–S10.* One-agent model vs simulation: section 4, point 9. Trapped vs priced out: both are below their
threshold; priced out are right not to check, trapped are wrong not to and cannot find out.

### Notes for writing the main text (4 Nov)
- Model section: the decision rule and threshold θᵢ = c/(g·sᵢ) + c/(l·sᵢ), and the learning rules (beliefs about the
  checker's hit and false-alarm rates update only after one's own checks). Everything else in the SI.
- Robustness at cheaper checking (c/g = 0.01; former Fig. S3b, dropped): same picture as at c/g = 0.05; one sentence
  where Fig. S4 is cited.
- Penalty scaling (Fig. S3): "who checks and who spreads fakes depends on how the penalty for a fake grows with the
  audience; the share trapped and the split of trust between two checkers do not."
- Stakes (Table S1): the two-checker split holds even when both stakes are constant (clear favourite 0.72): it needs
  only chance early experience. The one-checker split and the trapped group need differences in audience size.
- Cost cut: out of the paper (5 Nov). If the discussion needs it: free checking for 1,000 items brings back 91% of the
  trapped (10% of the cost works as well), but what spreads and welfare barely change.
- Who is trapped (Fig. 1B): at the main setting most of the trapped never tried the checker (11.1% of agents,
  27–57 neighbours, starting trust below their threshold); 3.7% tried, had bad luck early and stopped (Fig. 1C).
- Starting belief about the share of genuine items (no figure): "It mainly sets how soon checking starts (within 5
  items with the spread-out start used here, after 300–1,200 items when everyone starts at 0.5); the end state is the
  same unless sharing never gets going (large losses with everyone at 0.5; φ = 0.75 with everyone correct)."
- The dip (Figs. 2C, S2C): at most about half the share of the smaller group (users vs non-users; 0.25 for two equal
  groups); state this where the dip is introduced, so that the different plateaus across costs read as group sizes, not
  as different dynamics. Report the dip itself (Hartigan & Hartigan 1985), not a rescaled version.
- Terms (5 Nov): "Of the agents who do not check, some are priced out: checking would not pay for them even if they
  knew how good the checker is. The others are trapped: checking would pay, but they trust the checker too little to
  try it, and without trying they cannot find out otherwise." Most of the trapped never tried (11 of 15 points).
- One-agent model (section 4, point 9): "the simulation shows the phenomena and their consequences; the one-agent
  model, with nothing fitted, shows why."
- Fig. 4E: say the reach gap falls *in step with* the share not checking, not *because of* it; the cost-cut runs (not in
  the paper) show that bringing back the trapped moves the reach gap by only ~0.03.
- Identical starting beliefs exaggerate the two-humped shape (Fig. S1C); the groups do not depend on it (Fig. S1D;
  also every row of Fig. S5D).
- Captions (5 Nov): "Shaded bands (error bars for markers): 95% confidence intervals over n runs (Student's t); where
  not visible they are narrower than the line." Methods: "Runs in which genuine items reach fewer than 5% of agents are
  classed as 'nothing circulates'. Behaviour and beliefs are averaged over the other runs; reach and welfare over all
  runs." Give n per figure (section 1, "Uncertainty").
- Two sources of low trust (6 Nov; for the discussion, linking to survey distrust, e.g. the GSS motivation in section 4,
  point 6): "In the model, low trust has two sources: people who never use the verifier, whose trust stays where it
  began, and people whose first experiences went badly, who stop using it and stay distrustful because they no longer
  see evidence that could change their minds." Background for the wording: never-checkers do not distrust, they never
  update (trust stays at the start, 1/3, vs ~0.7 for users); the burned started checking because checking paid for
  them, a few early errors pushed their trust below their threshold, and it froze there (the checker is good on
  average: early bad luck, then no new evidence). Sizes at the main setting: 41% never check, 4% check and stop, so
  most low trust comes from never engaging. Limits of the analogy with surveys: survey distrust is also partisan and
  driven by elite cues, includes heavy users who distrust, and is trust in news rather than in a verifier; the analogy
  holds for any source whose reliability people learn by using it.

## 5. Reproduce

```
cd model
python synthnets.py                                   # random networks for Fig. S11 (before the ineq_* experiments)
DFP_NPROC=6 bash run_on_mac_v3.sh                     # all experiments (restartable; ~3.5 h on 6 cores)
python run_v3.py status --out ../data/v3              # progress per experiment
python run_v3.py collect --out ../data/v3             # chunk files -> <experiment>.parquet (+ _agents.pkl)
python liss_checks.py ../data/LISS ../data/v3/liss_aggregates.json   # LISS aggregates (not used in the paper)
python gss_press.py extract ../data/gss/raw/gss7224_r3a.dta ../data/gss/gss_press_extract.parquet   # for Fig. S12
python make_figures_v3.py ../data/v3 ../figures/paper_v3              # every figure, PNG + PDF (needs matplotlib)
```
