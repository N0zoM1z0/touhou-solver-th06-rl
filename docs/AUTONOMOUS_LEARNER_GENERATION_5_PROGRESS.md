# Autonomous learner generation 5 progress

This append-only implementation record begins after the Generation-5 design
was committed and before any new Generation-5 Wine outcome.

## 2026-08-12: direct-Q delayed-effect smoke rejected

The first implementation used the declared terminal-zero, frozen, in-sample
cost-expectile Bellman loop but fitted factual Q directly against the common
state-plus-action target. It passed exact terminal/HIT conservation and the
null-action fixture, yet failed the delayed-effect fixture: when the physical
cost occurred beyond one backup horizon, all seven action differences were
effectively zero and the policy correctly could not publish the beneficial
action.

This is the intended pre-Wine falsification process. It exposed a generic
estimator defect, not a gameplay counterexample: the small randomized action
effect was lost inside the much larger common state value even though later
states contained the causal trace. No evidence threshold, gameplay feature,
action, reward, or data distribution was changed.

Before any Generation-5 Wine outcome, the iteration is amended to decompose
each frozen target into an offline common state outcome and an action-centered
residual Q. The residual objective uses coefficients
`1[A=a] - propensity(a|s)` over the complete safe set. Coefficients are bounded
by one and never use inverse propensity. Factual Q for the next expectile value
is common outcome plus centered residual; only residual-Q action differences
are exported. The sequential target remains physical HIT-only and in-sample.

## 2026-08-12: action-centered implicit-Q smoke passed

The amended learner passed all three deterministic contracts. Terminal-zero
eight-step targets conserve the exact sum of interval physical HITs. In the
delayed fixture, the causal cost occurs 12 option boundaries after assignment
while each individual target spans only four; after four frozen iterations,
all seven whole-episode bootstrap members recovered a negative mean effect for
the beneficial action and every member's final action-centered Q loss beat its
zero-effect loss. The deliberately strict population-plus-range rule published
the beneficial action at 25 of 3,328 interior decisions rather than changing a
large fraction of the policy.

On the same state-risk process with randomized actions but the action effect
removed, the complete population published zero overrides. The largest
centered coefficient was exactly 0.5 under the fixture's `(0.5, 0.5)` behavior
distribution; no reciprocal propensity appears. This is algorithm smoke only,
not Wine efficacy or candidate authorization.

## 2026-08-12: first frozen-Wine smoke report failed closed

The first 29-episode frozen-Wine smoke completed its loader and model work but
refused to write the final JSON. The smoke had intentionally supplied no
Generation-5 episode IDs, so the empty new-cohort comparator had zero loss and
an infinite ratio; strict JSON serialization rejected that non-finite value.
No report or partial policy was published, and no Generation-5 Wine outcome
exists.

The repaired non-authorizing smoke explicitly treats the chronologically later
16 Generation-4 episodes as a disjoint development cohort relative to the 13
older episodes. This does not make them Generation-5 authorization evidence;
it makes the development diagnostic well-defined and tests temporal cohort
generalization rather than hiding it in the aggregate.

Profiling the failed report also isolated a generic throughput cost: validating,
decompressing, parsing, and assembling all 29 corpora used one Python thread for
about 17 minutes before any model fit. A local immutable option cache now binds
each fully audited result to the corpus manifest SHA-256 and the complete loader
source contract hash. A source or manifest change creates a miss; a partial or
tampered entry fails closed. Cache miss and tamper tests prove these contracts.
The ignored cache is acceleration only and cannot change or admit a row.

The model section created many native threads but averaged only about two CPU
cores because the Python custom-objective callback remained limiting. This is
recorded as the next offline optimization target; it cannot be addressed by
reducing population size or changing the learning objective.
