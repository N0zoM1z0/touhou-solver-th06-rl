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

## 2026-08-12: frozen-Wine residual signal passed, policy stability failed

The repaired 29-episode, 102,409-option frozen Stage-6 smoke completed in
617.26 seconds and remained explicitly ineligible for evidence or
authorization. Its cross-fitted factual-Q loss was 7,530.09 versus 7,619.49 for
the common-state zero-effect comparator, a ratio of 0.988267. It improved 28 of
29 complete episodes. The chronologically later 16-episode development cohort
independently improved 16/16 episodes with ratio 0.983166. Every fold improved
and every fold's upper-order-statistic support coverage exceeded 0.99.

This is materially stronger and more stable treatment-signal evidence than
Generation 4, whose final aggregate ratio was 0.999884 and improved 14/29
episodes. It is still not deployable. The four-member calibration population's
full pessimistic rule proposed 24,819 actions, or 24.24% of held-out options,
above the frozen 10% exposure cap. Its two independent two-member panels
proposed on a union of 53,423 options and selected the same non-incumbent action
on only 11,193, giving 20.95% conditional agreement versus the fixed 80% gate.

Observation: sequential action-centered implicit Q now learns a real held-out
Wine signal, including on newer episodes, but bootstrap populations do not
agree which action realizes it. Inference: replacing the estimator again would
discard verified progress; the immediate defect is decision-level uncertainty
and action selection. Decision, before any Generation-5 Wine outcome: make
calibration match the seven-member production artifact, split members into
fixed independent 3/4 panels, require each panel's pessimistic range rule and
exact agreement on the selected action, while retaining the 10% proposal and
80% conditional-agreement gates. If that smoke still fails, use the already
declared Stage 4 -> 5 -> 6 curriculum diagnostic rather than changing a gate.

## 2026-08-12: seven-member Stage-6 smoke activates the frozen curriculum

The production-sized seven-member repeat used the fixed independent 3/4
panels and the same 29 frozen Stage-6 episodes. It again found a stable held-out
residual signal: cross-fitted Q loss was 7,533.49 versus 7,621.68 for the
zero-effect comparator, a ratio of 0.988429, with improvement in 28/29 complete
episodes. The chronologically later 16-episode development cohort improved
16/16 with ratio 0.983367. This independently reproduces the estimator-level
finding from the four-member smoke.

The stricter production decision rule proposed 6,425 of 102,409 held-out
options, or 6.27%, and therefore passed the predeclared 10% exposure ceiling.
However, either panel proposed on 38,082 options and both panels selected the
same non-incumbent action on only 6,425: 16.87% conditional agreement versus
the frozen 80% requirement. Agreement including mutual incumbent abstention
was 69.09%, which is not the declared decision metric and cannot substitute for
the failed conditional gate. The proposal action counts are retained in the
machine report for audit but are not a basis for action-specific adjustment.

Verdict: the implicit-Q estimator learns repeatable Stage-6 treatment signal,
but the data do not identify a sufficiently stable deployed decision. No
Generation-5 candidate is authorized and no Stage-6 collection schedule may
begin. This exact failure satisfies the previously declared trigger for the
fixed Stage 4 -> 5 -> 6 curriculum diagnostic. The next Wine outcome therefore
must occur only after its seeds, isolated worker assignments, canaries, and
configuration hashes are committed. No threshold, reward, feature, behavior
mixture, or action preference changes in response to this result.

## 2026-08-12: curriculum execution frozen before its first outcome

The Stage-6 decision-stability failure activated the previously declared
Stage-4 -> 5 -> 6 diagnostic. Before any curriculum Wine outcome, the repository
froze 116 unique bounded game/policy seeds, every collection worker assignment,
all fit and canary boundaries, the alternating natural Stage-6 final order, and
one non-evidence parallelism differential. Four normal-speed collection workers
share no game directory, Wine prefix, display, artifact root, or corpus root.
Canaries and final evaluation remain sequential.

The learner uses a single 32-thread host-sharing cap. Stage-4 boundaries 10 and
15 run fixed low-cost non-authorizing smokes; boundary 20 and every Stage-5/6
boundary use the complete production contract. A fit-eligible shadow population
may automatically weight later exploration through population disagreement and
ESS, but it never publishes actions during collection. No result can select an
RNG, worker, action, stage location, model size, or retry budget.

A committed execution lock binds the schedule, original game inventory, retail
executable, score template, Wine 11 binary, controller/native/scorer binaries,
learner and policy sources, worker orchestration, corpus loader, shadow audit,
and authorization code. The resumable orchestrator refuses any drift. No
curriculum outcome existed when this contract was written.

## 2026-08-12: Wine concurrency falsified, serial collection retained

The committed non-evidence differential completed three original-Wine Stage-4
runs under identical game RNG 25,154 and policy seed 23,413. The serial worker-
zero reference recorded 28 HITs; concurrent worker zero recorded 30; concurrent
worker one recorded 26. Their normalized factual-option digests were all
different. Complete-Stage provenance and isolated game/prefix/display paths
passed, so the equality failure is real closed-loop timing variation, not an
artifact mix-up.

No curriculum evidence episode had started. Per the predeclared performance
contract, the failed audit disables concurrent Wine collection without changing
or repeating it. The original worker assignment, seeds, order, learner,
behavior mixture, reward, features, support, decision gates, canaries, and final
evaluation remain frozen; assigned collection episodes now execute serially.
The failed differential corpus remains ineligible for training. Its audit and
the exact serial-fallback-only migration are hash-recorded for replay.

The first serial Stage-4 wave subsequently stopped on a pre-existing host X11
socket before episode two entered gameplay. Episodes zero and one had already
completed with 23 and 36 HITs; the failed episode-two artifact had zero trace
rows and no outcome. This is an evidenced isolation-resource collision, so an
infra-only migration moves worker IDs two and three from displays `:99`/`:100`
to unused `:105`/`:106` with fresh directories and prefixes. It preserves the
two factual episodes and every frozen seed, order, behavior, learner, and gate.
No stale socket belonging to another task is removed.

The first display-migration resume then stopped before Wine because the serial-
fallback validator still assumed the migration list contained exactly one row.
The display migration made that assumption false. A second zero-gameplay infra
migration changes lookup to the existing stable migration ID. It preserves the
same two complete episodes and all worker/display assignments established
above; no outcome-facing contract changes.
