# Autonomous learner generation 6 development

## Trigger and status

Generation 5 stopped because its action-centered scalar-tree critic improved
held-out Bellman prediction but independent populations frequently selected
widely different directions. Generation 6 is currently an offline development
program, not an authorized Wine learning generation. No new Wine gameplay may
start until one candidate passes the frozen qualification funnel and a complete
outcome-facing design is committed.

## Estimator hypothesis

The first candidate is a low-rank varying-coefficient R-critic:

`effect(s, a) = state_embedding(s) dot action_embedding(a)`

The action embedding is a learned linear projection of the adapter-declared
action and action-relative features. The state embedding is a small nonlinear
bottleneck over candidate-invariant observation, hazard-set, and factual-history
features. For each factual option, the training predictor is exactly:

`effect(s, A) - sum_a propensity(a | s) * effect(s, a)`.

Thus all actions share direction, focus, clearance, and native trajectory
geometry instead of being separated by arbitrary scalar-tree leaves. Centering
coefficients remain bounded by one and no inverse propensity enters the
treatment loss. The model receives no action name, Stage, frame, RNG, run ID,
boss, spell, or handcrafted phase.

The rest of the sequential contract is unchanged: factual semi-Markov
transitions, HIT as the only cost, `gamma = 1`, terminal value zero, frozen
n-step successor value, state-only common-outcome nuisance, lower cost
expectile value fitting, whole-episode bootstrap population, native support,
and pessimistic abstention to the incumbent.

This is a general estimator change motivated by held-out cross-episode action
instability. It does not select a movement direction or alter data according to
a gameplay failure.

## Development comparisons

Only the 31 development episodes in
`config/autonomous_generation6_qualification.json` may be used while changing
the estimator. Every experiment records source hash, parameters, effective CPU
affinity, wall time, synthetic causal/null results, Bellman loss by Stage, and
policy stability.

The old 3/4-panel exact-action diagnostic remains reported for comparison but
is not silently relaxed. Generation 6 additionally measures the policy that
would actually deploy: the complete seven-member pessimistic population and
its sensitivity to removal of each bootstrap member. This does not by itself
prove causal value; a final design must also freeze policy-level effect
uncertainty before qualification.

The 13 qualification episodes remain sample-undisclosed until one candidate's
source, seeds, hyperparameters, numerical gates, and report identity are
committed. Passing old development data is necessary, never sufficient.

## 2026-08-12: low-rank synthetic contract

The first 20-episode, three-member optimization smoke recovered the correct
effect sign but only one member beat the state-only loss. It was rejected
before Wine replay. The fixed production-strength synthetic repeat used 64
episode groups, seven whole-episode bootstrap members, four frozen Bellman
iterations, a four-option backup horizon, and a physical effect delayed by 12
options. All seven members recovered the beneficial negative effect and all
seven beat their zero-effect nuisance loss. On the matched randomized process
with the action effect removed, the full pessimistic population made zero
overrides in 4,608 decisions.

This is a model-family sanity result only. It neither selects a Wine action nor
passes development/qualification. Its immediate value is showing in seconds
that the shared action representation can learn delayed causal structure and
fail closed under a null before a costly corpus experiment.

The formal ignored report is
`artifacts/autonomous-generation-6-development/low-rank-causal-smoke-v1.json`,
SHA-256
`4eda12d7a1acc148713511a1fbfacedad3685ba2c7b377650eea1b4200334733`.
It records the complete seven effects, seven loss gates, zero null overrides,
and effective eight-CPU affinity.

## 2026-08-12: low-rank Wine development failed

The first full frozen-Wine development cross-fit used all 31 development
episodes (102,737 factual options), three complete-episode folds, seven
whole-episode bootstrap members, and two frozen Bellman iterations. Every
fold fitted its hazard representation only on training episodes. The 13
qualification episodes were not loaded.

The model failed every outcome-generalization criterion. Held-out Q loss was
1.046598 of the state-only comparator overall and improved only 11/31 episode
groups. Stage 4 independently measured 1.068562 and 4/10; Stage 6 measured
1.037371 and 7/21. The full seven-member pessimistic population proposed on
6.83% of options, but removing each member preserved the exact selected action
for only 31.09% of full-policy proposals. Conditional on any full/leave-one-out
policy proposing, all perturbations agreed only 9.28%. The old 3/4 panel metric
was 5.20%.

This is not evidence that a few more optimizer epochs will fix the learner.
Twenty of 21 member/fold fits beat their bootstrap training zero-effect loss,
typically by 3--5%, while complete held-out episodes worsened. The low-rank
varying-coefficient assumption or its regularization therefore overfits
episode-specific treatment residuals and does not express transferable Wine
action effects. It passed a known synthetic process but is falsified as the
Generation-6 Wine learner candidate.

The ignored machine report is
`artifacts/autonomous-generation-6-development/low-rank-wine-development-v1.json`,
SHA-256
`2b5cb9d1c02c374b7421fcfc0e3038cab2db144b2755d3f6a99bc0f01f18a014`.
It records `qualification_samples_loaded: false`, the exact CPU set, per-fold
bootstrap/train diagnostics, every episode loss, and all policy metrics.

The next diagnostic reuses this development data to measure the actual full
seven-member policy of the nonlinear tree R-critic under leave-one-member-out
perturbations. This distinguishes an incorrectly calibrated split-panel gate
from estimator instability before another model family or any Wine collection
is attempted. No old gate is relaxed after observing this failure.

## 2026-08-12: full tree population confirms policy-extraction failure

The old nonlinear tree R-critic was refitted on the same 31 development
episodes with five complete-episode folds, seven members, two Bellman
iterations, and the original low-cost tree sizes. This diagnostic deliberately
gave the representation the optimistic all-development unsupervised fit used
by historical Generation-5 smokes; it remains non-authorizing and cannot enter
qualification.

The critic retained real factual prediction signal: held-out Q loss was
0.986195 of the state-only comparator and improved 31/31 episodes. But the
actual deployed seven-member pessimistic rule was not hidden by the old split
gate. It proposed on 13.40% of decisions. Among its proposals, removing each of
the seven members preserved the exact action in only 14.08%. Conditional on the
full or any leave-one-out population proposing, every perturbation agreed only
6.17%. The independent 3/4 panels measured 6.69%.

Therefore Generation 5 cannot be repaired by relaxing or replacing the panel
metric. Flexible action-centered Q predicts a small factual residual but
directly taking its pessimistic argmin is an unstable policy-extraction method.
The ignored report is
`artifacts/autonomous-generation-6-development/tree-full-policy-development-v1.json`,
SHA-256
`77ccbe4c3414919cb53a22a5d4dddb4fced928cf3f363b07b7f2fbd48144a6ec`.

## Next hypothesis: cross-fitted IQL actor

Generation 5 implemented IQL-style in-sample value backup but deployed the
critic directly. Standard IQL instead extracts a behavior-supported policy by
advantage-weighted regression. The next offline candidate retains the flexible
tree Q/V signal and learns a separate listwise actor over every factual
native-safe set. For cost minimization, factual action weight is

`exp((V(s) - Q(s,A)) / temperature)`.

Weights are computed out of episode group, temperature is a predeclared robust
scale of cross-fitted advantage, and the exponential range is fixed before
Wine development output. The actor objective is factual behavior cloning
reweighted toward lower-cost actions; it uses no counterfactual successor and
no inverse propensity. Thus it remains anchored to actions that Wine actually
executed while converting the critic's noisy pointwise values into a smooth
supported policy.

The actor scorer consumes only the same generic observation, history, hazard,
candidate, and candidate-relative features. A complete bootstrap actor
population must abstain to the incumbent under disagreement. Synthetic delayed
effect/null tests run before Wine development, and the untouched 13-episode
qualification remains locked.

## 2026-08-12: action-centered IQL actor synthetic smoke passed

The first actor implementation used ordinary factual advantage-weighted
behavior cloning. Its advantage direction was correct, but the deterministic
64-episode fixture happened to contain 2,372 stay versus 2,236 left actions.
The learned 5.9% weight advantage almost exactly cancelled this 6.1% empirical
action imbalance, so every actor's mean beneficial-action logit had the wrong
sign. It was rejected before Wine development.

Complete behavior propensities make that sampling noise unnecessary. The
repaired objective computes the behavior-cloning baseline exactly over the
known safe-set distribution and estimates only the normalized
advantage-weighted residual with the factual action. This action-centered
control variate is unbiased for the listwise IQL actor objective, uses bounded
factual residuals, and never divides by propensity.

The formal repeat passed. All seven actor members assigned positive mean logit
effect to the beneficial action, and the pessimistic population exercised it
126 times in 4,608 delayed-effect decisions. On the matched randomized null,
individual actors retained finite estimation noise but the complete
pessimistic population made zero overrides. The physical effect occurred 12
options after action assignment while each critic backup spanned four.

The ignored report is
`artifacts/autonomous-generation-6-development/iql-actor-causal-smoke-v1.json`,
SHA-256
`c69a45cc273fd421d872a53bec2a179f9e1d6803a57e7ad09f74442d2b69fb55`.
This unlocks only frozen Wine development, not qualification or gameplay.

## 2026-08-12: first Wine actor replay found an estimator bug

The first five-fold replay completed all 31 development episodes and safely
made zero complete-population overrides in 102,737 held-out options. This was
not useful conservatism: the actors had moved substantially away from behavior
(`mean KL = 0.404640`) but disagreed on every proposed alternative. One member
also reported `-1.048867` for a field misleadingly labelled weighted cross
entropy. A finite-sample control-variate risk estimate may be negative, so that
number was only an audit trigger, not proof of the defect; the field is now
named `action_centered_risk_estimate`.

Audit found a generic control-variate algebra error. For factual action
`A ~ mu`, action loss `L`, and advantage weight `w`, the no-inverse-propensity
unbiased estimator is:

`w(A)L(A) - (L(A) - E_mu[L]) = E_mu[L] + (w(A) - 1)L(A)`.

The implementation had instead centered the entire `(w - 1)L` term:

`E_mu[L] + (w(A) - 1)(L(A) - E_mu[L])`.

Because weights are normalized over the training population, not separately
at every state, the extra subtraction has nonzero conditional expectation and
can make the objective negative. This is a learner implementation defect, not
a gameplay observation or a reason to tune the data distribution. A regression
test now enumerates a nonuniform propensity distribution and verifies that the
estimator expectation exactly equals the target AWR risk.

The ignored, invalid-for-comparison report is
`artifacts/autonomous-generation-6-development/iql-actor-wine-development-v1.json`.
Its SHA-256 is
`b6cd87fb555350b11c1d95d5b98af19f66941d0c8a2012022e6f4e6b7c889226`.
It remains preserved so the failure and diagnosis cannot be rewritten. After
the algebra repair, the exact same delayed-effect/null contract passed: all
seven mean beneficial-action logit effects were positive, the population made
158/4,608 beneficial-process overrides, and it made zero null-process
overrides. The repaired ignored report is
`artifacts/autonomous-generation-6-development/iql-actor-causal-smoke-v2.json`.
Its SHA-256 is
`a4fd4da219ce57fa037ce715b975a7eba25f8f9839b6b5a8dc7c748c579cb2c0`.
Only an otherwise unchanged Wine-development replay may now assess the repair.

## 2026-08-12: repaired actor still fails per-decision extraction

The otherwise identical repaired replay again used 31 development episodes,
five complete-episode folds, and 102,737 held-out options. It made no strict
seven-member range-pessimistic override. Unlike v1, the expanded diagnostics
prove that nontrivial policies exist beneath that abstention: individual actors
proposed on 6.80% of member/options, at least one member proposed on 18.88% of
options, and the mean-population policy proposed on 5.83%. However, conditional
on any individual proposing, all seven chose the same action only 0.92% of the
time. Stage 4 and Stage 6 fail in the same way.

This does not authorize relaxing the observed range threshold. It shows the
implementation still performs the Generation-5 failure mode: uncertainty is
the simultaneous extreme over every bootstrap model at every decision. The
declared Generation-6 requirement was instead policy-level cost-effect
uncertainty, with native geometry owning per-action physical safety.

There is also a second generic defect in the actor experiment: each actor's
AWR weights were generated by a critic fitted on the same bootstrap episodes.
The outer held-out policy is honest, but the actor can first imitate in-sample
critic residual noise. The next candidate must generate every actor training
weight from a critic that excluded that complete episode, then fit a policy to
those cross-fitted weights. It must evaluate the complete policy's aggregate
cost effect on distinct held-out episodes; it may not substitute looser
per-frame agreement after observing v2.

The ignored v2 report is
`artifacts/autonomous-generation-6-development/iql-actor-wine-development-v2.json`,
SHA-256
`fba933b8b112f478ea58701e9bd2e12f6f8717e4a8f6b040a9eabcee3bc57c8b`.
Warm load, representation, and five-fold critic/actor work took 11.39, 52.96,
and 310.11 seconds respectively (374.46 seconds total) on hard CPU set 0--31.
The untouched qualification corpus and Wine were not loaded.

## Next candidate: nested cross-fit policy improvement

The actor-training leakage is removed with nested complete-episode
cross-fitting. For each outer development fold, three inner critics generate
AWR weights only for episodes they did not fit. Seven whole-episode-bootstrap
actors then learn from that one immutable cross-fitted label vector. The outer
held-out episodes are excluded from the hazard representation, inner labels,
actor, support model, and evaluation nuisance. This is more than changing a
random seed: it prevents flexible critic residuals from becoming actor targets
on the same trajectories.

The candidate policy is the complete population's mean listwise score, not a
selected actor and not the extreme range at each frame. If it ranks a
native-supported candidate above the incumbent, the deployed policy is a small
stochastic intervention from incumbent to that candidate. Its probability is
the minimum of 10%, twice the recorded propensity of the candidate, and twice
the recorded propensity of the incumbent. Consequently the factual correction
coefficient for either relevant action is bounded by two; rare actions can
never create the old 180-fold pseudo-outcome spike.

On an outer held-out factual option, evaluation estimates the policy contrast
with a semi-Markov doubly robust score:

`rho * (Q(candidate) - Q(incumbent))`

`+ rho * (1[A=candidate] - 1[A=incumbent]) / mu(A) * (Y - Q(A))`.

`Y` is the unchanged physical-HIT n-step target with terminal value zero. Scores
are summed within each complete episode; confidence is computed across episode
groups for the complete policy, not across thousands of counterfactual
actions. This is a bounded, development-time policy-improvement diagnostic,
not a substitute for the original-Wine complete-Stage canary: occupancy changes
and approximation error remain possible, so only Wine owns promotion.

The formal delayed-effect/matched-null smoke passed. The delayed-effect policy
estimated `-0.4223` HIT per episode with complete-episode-bootstrap 95% upper
bound `-0.0789`; its
mean-population leave-one-member-out stability was 87.78%. The matched null
estimated `-0.0621` with bootstrap interval `[-0.1869, 0.0630]`, correctly retaining
zero despite its actor having finite-sample preferences. All inner labels were
out of episode, and the maximum factual correction was 0.2 under the fixture's
50/50 behavior. The ignored report is
`artifacts/autonomous-generation-6-development/crossfit-actor-policy-smoke-v3.json`,
SHA-256
`d682ba5c63617073485cb7f9a2a5631c0ff720c31bd0eb98f5518d8c504b3b12`.
All seven delayed-effect leave-one-actor-out policies also retained a negative
bootstrap upper bound; the worst was `-0.0342`. Every matched-null leave-one-out
policy retained a positive upper bound. Thus actor population uncertainty is
calibrated at the policy-effect level without demanding identical frame-level
actions.
These gates unlock only a 31-episode Wine development replay.

## 2026-08-12: first nested Wine replay is promising, not yet frozen

The first nested cross-fit replay used 31 development episodes and 102,737
outer-held-out options. Every outer-fold representation excluded held-out
episodes; every AWR label excluded its complete inner episode. The complete
mean policy proposed candidates on 6.27% of options but its propensity-bounded
intervention exposure was only 0.366%.

Its episode-grouped DR estimate was `-2.8284 HIT/stage`, with 4,096-bootstrap
95% interval `[-3.7499, -1.9891]`; 28/31 episode effects were negative. Stage 4
measured `-5.0203`, upper bound `-3.4538`, and Stage 6 measured `-1.7846`, upper
bound `-1.0410`. Maximum factual correction was exactly the predeclared bound
of two. The independent model term was also beneficial in all cohorts, though
smaller (`-1.0411` overall), rather than the conclusion being carried solely by
the learned Q model.

This is the first reusable-corpus learner with a clearly negative held-out
policy-level HIT signal in both Stages. It is not yet a qualification
candidate. Mean-policy exact action stability under leave-one-actor-out was
45.55%, so the next unchanged replay adds the missing correct uncertainty
question: each of the seven leave-one-actor-out *complete policies* must retain
a negative episode-bootstrap 95% HIT-effect upper bound. No per-frame threshold
is relaxed and no hyperparameter, episode, action, or data distribution changes.

The ignored intermediate report is
`artifacts/autonomous-generation-6-development/crossfit-actor-wine-development-v1.json`,
SHA-256
`43f44b8ffb90958f92d68c1b259f5b25ad20302365d8837809299877d708ccb1`.
Warm load plus the full five-fold nested replay took 266.58 seconds on hard CPU
set 0--31. Qualification and Wine were not loaded.

The otherwise unchanged population-value replay passed the missing gate. The
worst leave-one-actor-out 95% upper bound was `-1.6564` overall, `-2.7747` on
Stage 4, and `-0.8435` on Stage 6. The complete policy estimates and all other
diagnostics reproduced exactly; total time was 263.00 seconds. Candidate action
counts covered all 18 movement/focus combinations rather than collapsing onto
one manually favored direction. This establishes development-level policy
value robustness across the complete population despite low framewise exact
agreement.

The ignored report is
`artifacts/autonomous-generation-6-development/crossfit-actor-wine-development-v2.json`,
SHA-256
`a68954b0df5d285abf0feaf4f89651910d40528eb0aa820709a989e7c49436d5`.
It is the development reference for freezing qualification gates, not efficacy
evidence and not permission to inspect qualification data.

Before qualification, evaluator refactoring exposed that the fixed bootstrap
seed had still indexed episode effects in transient mapping insertion order.
JSON key sorting could therefore change a finite 4,096-resample quantile while
leaving the episode data and conclusion unchanged. Resampling now first sorts
by immutable episode ID; a regression test reverses the mapping and requires
an identical complete report. The canonical development summary is
`artifacts/autonomous-generation-6-development/crossfit-actor-wine-canonical-summary-v1.json`,
SHA-256
`497f3359542c8523e568061cb5d01f08256b6c536b6823814737f33fd4e50457`.
Canonical overall/full-policy and worst-LOO upper bounds are `-2.0157` and
`-1.6945`; Stage-4 values are `-3.4837` and `-2.7370`; Stage-6 values are
`-1.0536` and `-0.8577`. This order-only correction used saved episode scores
and did not refit or disclose qualification.

## Frozen full-development candidate preflight

All 31 development episodes were fitted once into a separate immutable
candidate artifact: one game-neutral hazard codebook, action-conditional local
support, and seven nested-cross-fit IQL actors. The fit checkpoint is distinct
from deployment validation so a native scorer defect cannot trigger outcome-
conditioned retraining. Neither artifact loaded qualification samples.

The final candidate is
`artifacts/autonomous-generation-6-candidate/candidate-v1.json`, SHA-256
`aea789ed9fe63aa4a2c0799092675fd287c9b66787ed968d82e82098fbb4ea64`.
Its pre-native fit checkpoint SHA-256 is
`4ee6a8e7a7bfeeb7f471a6b0cd0c7b5db1e46aff8c9db8073d4a62bfb1651d5a`.
It passed 64 portable/native end-to-end conformance cases, 1,200-decision p95
of 2.19 ms, zero 60 Hz misses, and the 32-bit Wine DLL kernel differential.
This permits hash-freezing one qualification attempt; it still cannot
authorize gameplay.

## Performance contract

All development launchers apply the Linux process-tree CPU affinity contract
before model imports or worker creation and use at most 32 inherited CPUs.
The low-rank teacher may be complex offline. A candidate cannot unlock Wine
until an equivalent immutable native population is below 4 ms p95 with zero
60 Hz deadline misses.

## Deployable online handoff

The offline target is now implemented without changing the fitted actor. The
immutable plug-in loads the complete seven-member population and chooses its
mean-score positive supported alternative inside the native safe set. It uses
eight-frame semi-Markov options and the history-independent bounded
intervention probability audited after qualification. Learner output has no
collision authority and Bomb remains forbidden.

The first complete Win32 resident-path measurement failed at `5.1593 ms` p95,
despite the earlier kernel-only pass. That failure was preserved rather than
relaxing the 4 ms gate. Shared-state feature construction and cache-contiguous
dense actor loops then reduced Win32 p95 to `3.2986 ms`, with exact portable,
Linux, and Windows actions on 64 factual Wine contexts and zero deadline
misses. No corpus, fit parameter, score rule, proposal, or qualification value
changed. The frozen Stage-4 active wiring canary is the next evidence boundary.

## First original-Wine wiring canary result

The frozen normal-speed, natural-RNG, complete-Stage-4 canary ran once from
commit `6908dd1` and completed cleanly with three physical HITs. It exercised
147 actor proposals and one propensity-sampled eight-frame intervention across
4,693 option boundaries. Bomb remained zero, the immutable state was unchanged,
the original executable and optimized scorer hashes matched, all selected
actions belonged to the native-safe vocabulary, no deadline was missed, and
the Wine prefix was completely cleaned. Eighteen Hard-empty observations were
handled by the existing native fail-close contract; there was no capture,
authority, delivery, trace, or corpus failure.

The canary nevertheless failed its predeclared deployment gate: real resident
policy p95 was `8.1554 ms`, with 2,143 boundaries above 4 ms, versus `3.2986 ms`
in isolated Wine preflight. It remained below the 16.67 ms controller deadline
in every decision, but the 4 ms gate is not relaxed. Report SHA-256 is
`eae09196301352e00a8d552abe5becb3b38b5ecedc2c6a7c81002e5f33e65537`;
result SHA-256 is
`5ce4f48dd85b8dca2679b1a8415ebe9fb5b042c165e60b4efd89b6e5c504c4fb`.

The three-HIT observation is wiring evidence only: one stochastic intervention
cannot estimate policy efficacy and is not compared with a historical
baseline. The demonstrated problem is deployment infrastructure under live
game contention. The next work is a generic native fused row-normalization /
support / actor path, verified against the unchanged portable candidate. A
successful repair requires a separately frozen successor canary; this failed
schedule is never replayed or reinterpreted as a pass.

The successor implementation fuses the remaining Python/FFI numerical path
while keeping final proposal selection and safety authority unchanged. A
maximum-width factual preflight (181--256 hazards) retained exact three-way
actions and measured `2.1229 ms` Wine p95, `2.4212 ms` maximum, and zero misses.
Its report SHA-256 is
`d687027508acc5787a0db846f8c5b48ce64c3ccee4b7fe9f49dfeb6a150cce2f`.
Generation-6 canary v2 is a new frozen experiment identity and separates the
game/controller CPU sets within the same 32-core host budget to test the
demonstrated live-contention defect.

Canary v2 again completed the original-Wine Stage 4 cleanly, with five physical
HITs, 118 proposals, and two sampled interventions over 5,051 boundaries. All
non-latency gates passed, including frozen CPU partitions, zero Bomb, immutable
state, native-safe action vocabulary, complete HIT accounting, zero infra
events/deadline misses, and cleanup. Resident p95 improved from `8.1554 ms` to
`4.1027 ms`, but still failed the unchanged 4 ms limit; 438 boundaries exceeded
4 ms. Report SHA-256 is
`df38fa5e14c57d8766ce39ef3e5b796ca7abd0ff4843a6ac8c94d3bc6f3bec2a`;
result SHA-256 is
`9c01abc412a33668fd97e73a666248e7c5af4479a8219d865c9036b142f41f5c`.
Five HITs remain wiring-only and are not compared with v1 or a baseline.

This second rejection narrows the generic performance defect to the tail after
native scoring: Win32 Python still materializes 126 member/action floats and
computes the population mean and proposal. The next repair fuses that exact
mean and safe-row choice into the same native policy call. The 4 ms gate and
learner are unchanged, and any new Wine run needs a third frozen identity.

The native-choice implementation passed that prerequisite. It reproduces the
seven-member double mean, positive advantage, support mask, and lexical tie
semantics exactly; 64 maximum-width factual contexts retained identical
portable/Linux/Win32 proposals. Wine p95 was `1.7113 ms`, maximum `1.8918 ms`,
with no >4 ms samples or deadline misses. Canary v3 is frozen separately at
SHA-256
`cf51538bd8ccbf266a9442579078fc5373411f704814133c326203f25c6622a1`.
