# Autonomous learner generation 3 result

## Verdict

Generation 3 is frozen as **superseded without efficacy authorization**. It did
not reach its predeclared 24-episode terminal evidence budget, so this is not a
claim that the original stopping rule proved efficacy or inefficacy. It is a
recorded human decision to stop after aggregate learner evidence demonstrated
that the estimator and deployment contract were structurally unsuitable.

No Generation-3 candidate passed fit authorization, entered Wine canary, or
entered natural-RNG full-Stage evaluation. Nothing from this generation is a
promoted gameplay policy.

## Accepted evidence

Thirteen fixed-RNG original-retail Wine Practice Stage 6 episodes completed
with HIT continuation, zero Bomb, complete transition-v9 corpora, and clean
infrastructure. The first 12 formed the declared round-1 fit. They contained
422 physical HITs, all conserved exactly by transition, factual interval,
prefix, and manifest accounting. Episode 12 completed after that fit with 27
HITs and remains valid historical training evidence.

The next episode was interrupted by the explicit supersession decision. It has
no retail report, is absent from `generation.json`, and is permanently excluded
from fitting and evaluation.

## Why round 1 was structurally ineligible

Round 1 contained 33,161 training options from nine whole episodes and 11,076
validation options from three disjoint episodes. All grouping, support,
population, finite-value, causal identity, native binding, and HIT-presence
gates passed. Both population-preserving distillation gates failed:

- held-out DR advantage RMSE: 86.1797 HIT;
- held-out zero-advantage RMSE: 86.1578 HIT;
- distillation p95 error: 3.7091 HIT versus a fixed 0.05 gate;
- distillation maximum error: 370.8625 HIT versus a fixed 0.25 gate;
- whole-episode conformal radius: 2476.2625 HIT.

The behavior policy produced 118,038 randomized boundaries in the 12 fitted
episodes but only 10,599 non-incumbent assignments (8.98%). With a 0.10
mixture divided across as many as 18 legal actions, a rare factual propensity
can be about 0.0056. The raw multi-action AIPW correction can consequently
multiply an ordinary nuisance residual by about 180. Per-action inverse-
propensity effective sample sizes were only 116--246 despite tens of thousands
of correlated option rows.

Three validation episode groups at 90% whole-episode max conformal coverage
mathematically select the maximum group score. Combined with a complete-Stage
Monte-Carlo return at every option, this made a useful runtime bound
unattainable. Merely doubling the episode count could not plausibly close the
observed 52-fold p95 and roughly 1,000-fold maximum distillation gaps.

## Superseding decision

The successor must retain Wine-only factual outcomes, native Hard authority,
HIT-only reward, immutable online weights, whole-episode grouping, a population
rather than a chosen winner, paired Wine canary, and natural complete-Stage
evaluation. It replaces the high-variance estimator with sequential
semi-Markov offline RL, an action-centered orthogonal critic, propensity-aware
autonomous exploration, policy-level cross-fitted calibration, and a full
native population whenever the latency contract permits it.

Generation-3 artifacts remain immutable under
`artifacts/autonomous-wine-generation-3/`. Historical episodes may be used as
factual offline training/development data by a declared successor, but never as
new-generation canary or promotion evidence.
