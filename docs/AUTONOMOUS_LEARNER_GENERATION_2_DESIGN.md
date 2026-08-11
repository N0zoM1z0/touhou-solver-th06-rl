# Autonomous learner generation 2 design

## Decision

The Wine-only learning and promotion architecture is retained.  Generation 1
was a successful infrastructure test but an ineffective policy learner.  Its
linear factual-return regressor is replaced in generation 2; no observed
failure location, source phase, frame range, RNG seed, or hand-selected action
is added to the policy.

This document fixes the generation-2 method before new Wine outcomes are
observed.  A later algorithm change is a new generation, not an adjustment to
this one.

## Generation-1 evidence

The 10 original-retail Wine episodes contained 40,976 decisions and 30,634
complete 120-frame learning windows.  Of those targets, 29,578 (96.55%) were
the identical maximum survival return and only 0.627% were negative.  There
were no physical HIT labels; every episode ended at `Hard safe set empty`.

The round-2 factual-return RMSE improved from the constant predictor's 8.532 to
7.725, while the held-out committee produced zero policy proposals.  This is
direct evidence that outcome RMSE learned terminal proximity without
establishing a repeatable action advantage.  Factual-return RMSE is therefore
not a policy-effect gate in generation 2.

The generation-1 learner was mathematically meaningful as a regularized
estimate of a truncated behavior-policy return.  It was not a complete offline
policy-improvement procedure: it used a partial observation, a one-step
deviation followed by the behavior policy, global action counts instead of
local state-action support, and a clipped `1 / propensity` regression weight
that is not a sequential target-policy value estimator.

## Fixed generation-2 observation contract

The learner remains game-neutral.  A game adapter supplies normalized player
state and native action-trajectory measurements.  The TH06 adapter must not
expose source context, stage timer, RNG state, boss/spell identity, or a
handwritten phase feature.

For every native Hard-safe action, the adapter adds deterministic clearance
profiles at physical horizons 1, 2, 3, 4, 6, 8, 10, and 12.  It also adds
candidate rank/fraction summaries, displacement and boundary reserve, and
differences from the frozen incumbent action.  These values are computed from
the already-lowered observed Wine hazards by bounded native code.  They expose
the shape and temporal pressure of the hazard field without running a resident
tree/beam search or adding an action to the safe set.

The lossless corpus already retains every occupied bullet tail, sprite
geometry, laser, enemy body, player state, action, propensity, and factual next
snapshot.  Offline reconstruction may derive the same versioned features from
those roots.  It may not derive a successor for an action Wine did not execute.

## Fixed generation-2 objective and learner

The authoritative cost is one per factual physical HIT and zero otherwise.
`Hard safe set empty` is retained as an infrastructure/control diagnostic, not
as a substitute HIT reward.  Training episodes continue through HITs to the
natural Practice Stage completion so the corpus observes the same quantity
used by final evaluation.  Fixed original-retail RNG is allowed only for
training and paired canary variance reduction.

The offline teacher is a five-member episode-bootstrap conservative fitted-Q
ensemble for the discrete native-safe action set:

- 60-frame factual n-step HIT cost;
- six fitted-Q iterations with an undiscounted finite-Stage backup;
- nonlinear histogram tree regressors;
- clipped recorded-propensity weights;
- conservative next-action backup from ensemble uncertainty;
- no model-generated state transition;
- whole physical episodes as train, validation, bootstrap, and confidence
  units.

Each member is exported as immutable scalar trees.  Wine scores all current
safe candidates with the existing native batch tree evaluator.  The deployed
policy minimizes predicted HIT cost, but changes the incumbent action only
when every member agrees and the candidate's pessimistic bound beats the
incumbent's optimistic bound.

Local support is not inferred from a global action count.  Per-action
prototypes are fitted from training state-action vectors, and authorization
uses a distance threshold derived from held-out episode distances.  The
incumbent remains available outside learned support; unsupported alternatives
abstain.

## Hard-empty audit contract

`Hard safe set empty` means that the current four-frame certificate found no
action safe under every declared delivery delay.  It does not by itself prove
that the earlier trajectory was unavoidable or that every multi-action path
was lost.

The audit must use original-Wine roots and the same native build to check:

1. recorded and recomputed Hard masks agree before termination;
2. every rejected action has a concrete projected collision witness inside the
   four-frame authority horizon;
3. a continuous-stage Wine run reaches the corresponding physical HIT or
   recovers after the controller's fail-closed input release;
4. HIT, Hard-empty, and recovery accounting remain distinct in the corpus.

A native/safety change is allowed only if this audit produces a reproducible
contract violation.  Poor earlier positioning, short learning credit, or an
ineffective policy is not a safety defect.

The completed source audit is recorded in `docs/HARD_EMPTY_SOURCE_AUDIT.md`.
It found that 3 of 10 generation-1 roots were closures of the repo's extra
0.35 px uncertainty margin, not closures of the shipped game's exact collision
geometry. Consequently the controller now prefers the conservative set and
falls back to the source-exact native set only when the conservative set is
empty. No learner can invoke, widen, or otherwise alter this fallback.

## Evidence and promotion

Offline prediction loss may reject malformed or non-generalizing fits but
cannot establish efficacy.  Shadow checks immutable loading, native-safe-only
selection, local support, committee agreement, proposal exercise, and latency.

An exercised candidate then receives paired, complete-Stage, fixed-RNG Wine
canaries against the frozen incumbent.  Canary authorization requires clean
authority/accounting and a strictly lower aggregate physical HIT count.  Only
then may it enter alternating normal-speed, natural-RNG, complete-Stage Wine
evaluation.  Final efficacy still means strictly fewer aggregate physical
HITs; offline Q, shadow proposals, Hard-empty counts, and fixed-RNG canaries
cannot promote by themselves.

## Portability boundary

TH08 may change capture addresses, hazard decoding, playfield normalization,
movement actions, and the adapter that emits native trajectory profiles.  It
must reuse the dataset semantics, factual HIT objective, fitted-Q ensemble,
local-support rule, immutable scorer, grouped evidence, and promotion state
machine.  A TH06-specific exception in any of those shared components is
forbidden.
