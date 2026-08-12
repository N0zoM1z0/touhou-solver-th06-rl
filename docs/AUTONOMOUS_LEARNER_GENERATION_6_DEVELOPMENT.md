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

## Performance contract

All development launchers apply the Linux process-tree CPU affinity contract
before model imports or worker creation and use at most 32 inherited CPUs.
The low-rank teacher may be complex offline. A candidate cannot unlock Wine
until an equivalent immutable native population is below 4 ms p95 with zero
60 Hz deadline misses.
