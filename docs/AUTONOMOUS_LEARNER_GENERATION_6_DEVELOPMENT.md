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

## Performance contract

All development launchers apply the Linux process-tree CPU affinity contract
before model imports or worker creation and use at most 32 inherited CPUs.
The low-rank teacher may be complex offline. A candidate cannot unlock Wine
until an equivalent immutable native population is below 4 ms p95 with zero
60 Hz deadline misses.
