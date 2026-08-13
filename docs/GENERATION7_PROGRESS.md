# Generation-7 offline progress and corrected stop decision

Date: 2026-08-13

Status: G7-A repaired; G7-B has exploratory short-horizon signal but fails
matched OPE calibration; no Wine outcome-facing work or new collection
authorized

## Outcome

Generation 7 is not rejected as a research method. Its falsification gates
found a causal dataset bug inherited from the earlier learner path: the loader
conditioned on whether a randomized proposal survived fresh native
revalidation. The corrected unit is the randomized proposal assignment
(intention-to-treat, ITT). Native revalidation/fallback is part of the factual
deployed transition, not a row-selection rule.

The correction invalidates every earlier Generation-7 numeric result based on
167,250 accepted options. On 550,684 factual ITT assignments, raw long-horizon
signal disappears. A cross-fitted horizon-1 action-only learner finds a very
small structured score, but IPS, FQE, and DR do not calibrate. This exact policy
has not established improvement and cannot proceed to AWR/IQL promotion, Wine
evaluation, or new collection.

## Root cause: post-assignment compliance conditioning

The behavior policy randomized a proposal from the decision-time native-safe
set. A fresh publication-time native check could reject it and execute a
fallback. The old loader retained only rows where proposal and executed action
matched while continuing to use the proposal propensity. Rejection is strongly
action-dependent, so this destroyed the randomized treatment law.

The corrected v3 learner-neutral arrays preserve proposal, boundary executed
action, compliance, propensity, decision-time state, and factual downstream
outcome separately:

| behavior source | proposals | complied | compliance | reference/behavior IPW mean |
| --- | ---: | ---: | ---: | ---: |
| audited natural | 113,189 | 24,172 | 21.36% | 0.99505 |
| randomized v10 Stage 4 | 140,973 | 40,669 | 28.85% | 1.00143 |
| randomized v10 Stage 6 | 169,631 | 54,408 | 32.07% | 0.99636 |
| randomized v9 | 126,891 | 48,001 | 37.83% | 1.00058 |
| **total** | **550,684** | **167,250** | **30.37%** | **0.99836** |

The episode-equal reference/behavior ratio is `0.99858` with clustered SE
`0.00129`; every source and stage stratum is within two clustered SE of one.
This outcome-free check would have exposed the old filtering before fitting a
learner.

All 2,044 manifest HITs remain conserved across the 56 complete physical
episodes. The arrays contain 7,896,535 current native-safe candidate rows and
67 separately stored causal history/hazard features. The immutable corpus is
unchanged.

## Other bounded repairs

- Richer causal state now enters nuisance fitting and FQE as well as effect
  interactions. The previous richer-state comparison did not do this and is
  invalidated.
- Direct/IPS/FQE/DR calibration now uses paired per-episode differences, not
  the overly conservative sum of independent estimator variances.
- Factual-action null resampling respects each row's recorded propensity and
  is vectorized. The orthogonal null keeps the same propensity-aware contract.
- Predeclared horizons are 1, 2, 4, 8, 16, and 32. One-step and sequential
  importance support have separate gates.
- The deployable object is the stochastic proposal distribution composed with
  the immutable native revalidation/fallback kernel. There is no second learned
  or stochastic thinning sampler.

## Corrected raw diagnostic

The aggregate nonbaseline-proposal versus baseline-proposal IPW smoke is:

| proposal horizon | episode-equal effect | clustered SE |
| ---: | ---: | ---: |
| 1 | -0.000538 | 0.000187 |
| 2 | -0.000197 | 0.000275 |
| 4 | +0.000256 | 0.000417 |
| 8 | +0.000094 | 0.000530 |
| 16 | +0.000429 | 0.000763 |
| 32 | -0.000610 | 0.001229 |

At horizon 32, the whole-episode bootstrap keeps the observed sign in only
65.2% of replicates and its 95% interval crosses zero. The prior large negative
effect (`-0.145`) was compliance-selection bias, not evidence that deviations
improved play. Propensity-aware action and reward-suffix null diagnostics now
remain centered, and the synthetic delayed-effect control passes.

There is also an estimand mismatch at horizon 1. Nonbaseline assignment reduces
the next-boundary duration by `0.2286` frames (clustered SE `0.0284`) and native
compliance probability by `0.0624` (SE `0.0036`). Consequently, HIT per proposal
is not a fixed-exposure gameplay value: a policy can change how soon the next
proposal occurs. A rough IPW HIT-per-frame ratio remains negative, but a ratio
diagnostic is not an identified incumbent-continuation Stage value. Future work
must use a fixed physical-time outcome or a correctly specified semi-Markov
full-value estimand before treating a short-horizon result as improvement.

## Corrected horizon-1 cross-fit

The action-only learner uses five whole-episode folds and compares its exact
bounded proposal policy with the shared 0.05 incumbent/uniform reference.
Negative values mean fewer predicted HITs:

| matched estimator | episode-equal mean | clustered SE |
| --- | ---: | ---: |
| one-step direct | -0.000003313 | 0.000000928 |
| one-step DR | -0.000003473 | 0.000000997 |
| one-step IPS | -0.000001835 | 0.000000568 |
| one-step FQE | -0.000000518 | 0.000000053 |
| horizon-1 sequential DR | -0.000001505 | 0.000000436 |
| horizon-1 sequential FQE | -0.000000518 | 0.000000053 |

Direct minus DR is within one paired clustered SE and passes. IPS minus DR is
about `3.03` SE, FQE minus DR about `3.08` SE, and sequential FQE minus DR
about `2.47` SE; those gates fail. Horizon-1 overlap itself passes.

Both orthogonal nulls reached the minimum possible exploratory p-value
`1/21`. This suggests structured residual action signal, but the frozen
contract requires 100 replicates and the failed OPE gates already prevent a
positive claim. The direct direction is negative in every source and stage,
although several action-specific directions still flip by stratum.

The learned distribution is also nearly identical to the reference: mean
reference KL is only `1.34e-6`. Thus the apparently significant per-assignment
number is not yet a meaningful gameplay-sized policy change.

## Decision

1. Keep the G7 causal/deployment/data-contract repairs. They are reusable
   infrastructure and explain why earlier learner results were misleading.
2. Reject all pre-repair G7 reports and caches as evidence. Hash-bound v3 ITT
   arrays replace the v2 compliance-conditioned derivation.
3. Do not promote the current orthogonal learner; do not run IQL or claim that
   AWR can bypass failed matched-estimator calibration.
4. Do not run shadow, canary, native Wine evaluation, or G7-C collection.
5. Do not use 32-step exact sequential OPE with this corpus. Fifty-six
   independent episodes do not support repeated-policy importance weighting.
6. If research continues, freeze a new one-deviation proposal-plus-native-
   fallback estimand with fixed physical-time exposure (or a validated
   semi-Markov full value) and a training-only rule for a gameplay-meaningful
   bounded tilt. Then require formal 100-replicate nulls and paired
   direct/IPS/DR/FQE calibration before any richer model. This is a new
   falsifiable layer, not a post-hoc parameter tweak to the present result.

Ignored JSON reports under `artifacts/generation7-offline/` are reproducibility
outputs, not tracked promotion evidence. The tracked contract, code, tests,
and this decision record define their meaning.
