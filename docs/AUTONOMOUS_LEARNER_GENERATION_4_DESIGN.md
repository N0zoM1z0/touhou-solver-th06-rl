# Autonomous learner generation 4 design

## Decision and evidence boundary

Generation 4 replaces the estimator and collection design because of the
aggregate Generation-3 result, not a TH06 location, phase, RNG seed, action, or
failure pattern. Its architecture is:

`propensity-aware Wine options -> sequential offline RL -> full native population -> cross-fitted policy gate -> Wine canary -> natural full-Stage evaluation`

Original retail Wine remains the sole source of transitions, physical HITs,
and evaluation outcomes. Native geometry remains the sole action authority.
The learner may rank or abstain inside the current native-safe set and may not
request Bomb, enlarge the set, alter collision margins, or synthesize a
successor for an action Wine did not execute.

The 13 complete Generation-3 collection episodes are historical factual
training/development data. They cannot authorize Generation 4, enter its
canary, or enter its final evaluation. No new Generation-4 Wine collection may
start until the implementation, deterministic causal recovery smoke, factual
semi-Markov accounting smoke, full-population native-equivalence/latency smoke,
seed schedule, and unattended runner are committed.

## 1. Factual semi-Markov process

Only a factually executed randomized option boundary is a decision point. For
successive factual boundaries, the transition is

`(state, native-safe set, incumbent, action, propensity, interval HIT, elapsed, next state)`.

Rejected tentative options never become treatments or successor labels. HITs
in an ordinary death/invulnerability gap remain in the factual interval that
began at the preceding decision. Prefix, interval, transition, and manifest HIT
counts conserve exactly.

The sole gameplay cost is physical HIT. With `gamma = 1`, terminal value zero,
and no reward shaping, the Bellman identity is

`Q(s, a) = interval_HIT + V(next factual boundary)`.

Generation 4 uses an eight-decision n-step target. It sums factual interval
HITs for at most eight successive decision intervals and bootstraps once from
a frozen nuisance value at the eighth successor. A terminal transition omits
the bootstrap. The recursive undiscounted sum remains exactly the complete-
Stage physical HIT objective; Hard-empty, clearance, survival time, progress,
source phase, and teacher output are not reward.

## 2. Cross-fitted frozen nuisances

Complete physical episodes are split into five deterministic folds. For every
held-out fold, nuisance value and behavior-outcome models are fitted without
that fold, frozen, and used once to construct its n-step outcome and common
state residual. No held-out episode updates its own target. A final nuisance is
fitted on all training episodes only after every cross-fitted diagnostic is
fixed.

Nuisance models may use generic transition position and remaining factual
decision count as offline-only control variates. These fields are never
exported and can never select an online action. The deployed treatment-effect
critic uses only the game-neutral observation, action, observed hazard set,
and factual history interface available unchanged to a TH08 adapter.

The nuisance population uses three members and 160 depth-six histogram trees.
These counts, five folds, eight decision steps, `gamma = 1`, and learner seed
260812 are generation constants.

## 3. Generalized action-centered R-critic

Generation 4 does not form per-action AIPW pseudo-labels. At each randomized
boundary it reconstructs the complete recorded behavior distribution over the
native-safe set. For candidate score `f(s,a)`, known propensity `e(a|s)`, and
factual action `A`, the centered prediction is

`f(s,A) - sum_a e(a|s) f(s,a)`.

The critic minimizes the generalized Robinson/R loss

`(outcome residual - centered prediction)^2`.

Its gradient for candidate `a` is proportional to
`1[A=a] - e(a|s)`, not its reciprocal. Rare assignments therefore contribute
bounded action-centered information instead of a `1 / propensity` spike. The
same scorer consumes shared direction, speed/focus, native trajectory,
hazard-set, and factual-history features for every action; there are no
handwritten action specialists.

Seven final members use the same custom objective, full feature/reward
contract, and 128 depth-six trees. They differ only by predeclared whole-
episode bootstrap counts and learner seeds. Runtime subtracts each member's
incumbent score from its candidate score and requires every member's advantage
to be negative. No member is selected as a winner.

## 4. Autonomous propensity-aware exploration

Every assignment remains inside the complete native-safe set and is freshly
recertified on every physical frame. A boundary distribution is the fixed
mixture of:

- 0.50 incumbent mass;
- 0.25 uniform mass over the current native-safe set;
- 0.25 information mass, normalized over the same set.

Before a critic exists, information mass is inversely proportional to the
square root of the action's accumulated propensity ESS. After a critic exists,
the same term is multiplied by its bounded population disagreement. There is
no phase, frame, RNG, screen region, failure case, or named-action quota. Full
boundary probability vectors, chosen probability, ESS inputs, and optional
uncertainty inputs are recorded so fitting never reconstructs an adaptive
propensity by guesswork.

The uniform component guarantees every legal action probability at least
`0.25 / |safe set|`, so inverse propensity is at most 72 for the 18-action
interface. This bound follows from the declared action interface and mixture,
not an observed gameplay exception. Policy state is immutable during an
episode; only auditable in-memory assignment statistics advance.

## 5. Cross-fitted policy/decision calibration

Hard safety already protects physical execution, so Generation 4 does not
demand simultaneous pseudo-label coverage for every counterfactual action at
every frame. Cross-fitted diagnostics instead evaluate the policy the
population would actually propose:

- held-out generalized R loss must beat the zero-effect critic globally;
- it must beat zero in a strict majority of complete episode groups;
- every proposed decision must have all seven member advantages below zero;
- proposal episode coverage, action coverage, support abstention, and member
  disagreement are reported;
- policy-level episode estimates are aggregated only across models that did
  not fit that episode.

Calibration uses at least 20 cross-fitted complete-episode groups. The support
threshold is the committed 99th percentile of cross-fitted factual distances,
not an episode maximum. These offline checks may reject a model but cannot
promote it. Only disjoint paired Wine canary outcomes can authorize natural
full-Stage evaluation.

## 6. Full native population

The default deployment artifact is the complete seven-member 128-tree critic;
mean-only or winner-only distillation is forbidden. Before Wine activation,
the host and Win32 native scorers must match every portable member on committed
conformance vectors, run 1,200 production-sized decisions, remain below 4 ms
p95, and record zero controller-deadline misses.

Only if the complete population fails this latency smoke may an explicitly new
generation declare population-preserving distillation. Generation 4 may not
silently fall back to the Generation-3 48-tree students.

## 7. Evidence schedule and stopping rule

The first learner smoke uses the 13 frozen historical episodes plus synthetic
causal fixtures. Generation-4 collection uses a new seed schedule committed
before outcomes. The first fit occurs after eight new complete Stages, giving
at least 21 complete episode groups; later fixed boundaries are 12 and 16 new
Stages. Historical episodes remain training-only. New collection, canary, and
evaluation episodes are disjoint.

Each fit-eligible candidate receives three paired fixed-RNG complete-Stage
Wine canaries. Authorization requires six clean stages, candidate exercise in
at least two pairs, strictly fewer aggregate candidate HITs, candidate no worse
in at least two pairs, and clean native/runtime evidence. An authorized
candidate receives 12 natural-RNG complete Stages per arm in fixed alternating
order, normal timing, HIT continuation, and zero Bomb. Strictly fewer aggregate
candidate HITs plus clean runtime is the efficacy verdict.

The runner stops only for reproducible infrastructure failure, a completed
effective final evaluation, or exhaustion of 16 new collection Stages without
authorization. Weak play or a failed model causes the next fixed autonomous
round, never a manual change to data, reward, action preference, or activation
region.
