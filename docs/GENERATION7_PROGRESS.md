# Generation-7 offline progress and stop decision

Date: 2026-08-13

Status: G7-A implemented; G7-B stopped at failed identifiability/OPE gates;
no Wine outcome-facing work or new collection authorized

## Outcome

Generation 7 repaired the causal/deployment contract and found reproducible,
structured action-effect signal in the existing immutable corpus. It did **not**
establish an identifiable policy improvement. Direct, IPS, matched one-step
FQE, and DR do not agree closely enough, and 32-option sequential importance
weights have unusable support. Canonical IQL and G7-C collection therefore do
not start from this result.

All numbers below are diagnostics from 56 complete physical episodes (167,250
factual options, 52,448 factual nonbaseline assignments, and 2,044 manifest
HITs). They are not gameplay evidence and do not authorize deployment.

## What G7-A now enforces

- Every feature has decision/after-action/episode-end/privileged availability
  metadata and allowed-use metadata. Unknown or unavailable actor features
  fail closed.
- The three final-episode-length features are absent. The only time-like actor
  feature is causal `log1p(option_index)`.
- Physical HIT is the only cost, `gamma=1`, terminal value is zero, and tests
  conserve HITs across factual option aggregation.
- The actor objective is nonnegative weighted conditional log likelihood plus
  reference KL and L2. Extreme factual logits cannot drive the objective to
  minus infinity.
- `ResidualStochasticPolicy` is the complete bounded action distribution. It
  separates native-safe, statistically-supported, and forecast-risk masks.
  There is no proposal plus second thinning sampler.
- The common reference is a 0.05 incumbent/uniform mixture over the current
  native-safe set.
- Learner-neutral, hash-bound v2 arrays keep the original corpus independent
  from any learner. They include compact candidate geometry and separately
  stored decision-time causal history/hazard summaries.
- All splits and uncertainty summaries use complete physical episodes.

## Raw randomized diagnostic

The untreated-versus-nonbaseline IPW smoke was negative at every predeclared
horizon; at 32 options its episode-equal estimate was `-0.14515` with clustered
SE `0.02560`. Source/stage strata had the same sign. This is not a causal policy
value result: both raw nulls failed severely (action-null z approximately
`-27.9`; reward-suffix-null z approximately `-22.4`). State/time imbalance can
produce almost the same number. The synthetic delayed-effect control passed.

This result justified orthogonal residualization; it did not justify a
candidate.

## Exact-policy cross-fit result

The final conservative action-only baseline used five episode folds and a
bounded 0.05 residual policy. Negative differences mean fewer expected HITs
than the shared reference.

| Matched estimand / estimator | episode-equal mean | clustered SE |
| --- | ---: | ---: |
| one-step direct | -0.0001704 | 0.0000041 |
| one-step DR | -0.0001769 | 0.0000687 |
| one-step IPS | -0.0012995 | 0.0001962 |
| one-step behavior-FQE | -0.0000388 | 0.0000007 |
| 32-option sequential FQE | -0.0014624 | 0.0000160 |
| 32-option sequential DR | -4809.94 | 4591.69 |

The one-step direct and DR estimates agree, have the same sign, and pass the
two-standard-error calibration check. IPS is roughly seven times the DR
magnitude. Matched behavior-FQE is roughly one fifth of it. Both calibration
gates fail.

Sequential DR is unusable rather than merely noisy. Across folds its minimum
cumulative-weight ESS is about `32.5`, while the largest cumulative weight is
about `1.27e9`. Its apparent calibration with sequential FQE is vacuous because
the DR interval is enormous. The explicit support gate fails.

The orthogonal action-randomization and reward-suffix nulls both passed at the
minimum 20-replicate resolution used for exploration (`p=1/21`). The committed
contract now requires 100 replicates before any future positive claim; the
exploratory reports cannot satisfy that gate retroactively.

The exact policy's one-step direct direction was negative in all four behavior
sources and all three stages. All 18 actions had sufficient reporting strata;
16 kept their aggregate direction in all six eligible source/stage strata.
`down_left_fast` and `up_left` each flipped in one stratum. Maximum held-out
effect prediction stayed below `1.41`, within the bound of 10.

## State-sufficiency ablation

Three predeclared effect representations used the same episodes, folds,
propensities, policy bound, and OPE definitions:

| representation | one-step direct | one-step DR | IPS | behavior-FQE | result |
| --- | ---: | ---: | ---: | ---: | --- |
| action-only | -0.000170 | -0.000177 | -0.001299 | -0.0000388 | fail |
| compact bilinear | -0.000370 | -0.000256 | -0.001446 | -0.0000387 | fail |
| history + hazard bilinear | -0.000574 | -0.000333 | -0.001606 | -0.0000399 | fail |

The richer state monotonically enlarged direct effects without making the
estimators converge. It therefore does not demonstrate that the agent learned
the barrage. A preliminary compact run also revealed near-zero-variance
standardization extrapolation (`direct` about `-5.28e5`). The shared linear
model contract now floors training scales at `1e-3`; a regression test prevents
recurrence. Post-repair values above are the only interpretable ablation.

## Proper AWR challenger

A five-fold proper AWR actor was implemented and tested. It uses only 86
within-safe-set-varying action/geometry features, optimizes nonnegative weights,
decreases its objective in every fold, conserves probability, and emits the
same exact residual policy used by OPE. Its initial one-step estimates were all
negative (direct `-0.00144`, DR `-0.00154`, IPS `-0.00714`), but IPS again had a
much larger magnitude. That exploratory report predates the matched-estimand
and support gates and is not a pass. It remains a challenger implementation,
not a candidate.

## Decision

G7-B stops here. The evidence supports “there is structured randomized action
signal,” but not “this exact policy improves the reference.” In particular:

1. Do not run canonical IQL as a way around failed identifiability. IQL cannot
   repair disagreeing nuisance/value estimators or missing sequential support.
2. Do not authorize shadow, canary, A/B, or natural-RNG Wine evaluation.
3. Do not start G7-C collection. The current failure is not a localized
   action/context support gap; repeated-policy support and estimator
   calibration fail broadly.
4. The next research design should reduce the estimand before increasing model
   capacity: predeclare a one-deviation logged target with incumbent
   continuation, shorter sequential horizons, and calibration/overlap gates.
   Any new collection still requires a separate frozen contract.
5. Preserve every immutable corpus fact and the learner-neutral v2 derivation;
   future algorithms can reuse them without restoring failed generations.

Ignored JSON reports under `artifacts/generation7-offline/` are reproducibility
outputs, not tracked evidence. The tracked configuration, code, tests, and this
decision record define what those reports mean.
