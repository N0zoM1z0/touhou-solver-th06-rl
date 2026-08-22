# Immutable Wine data plane

The durable research asset is the original-retail Wine trajectory, not one
algorithm's feature matrix.

## Required factual episode data

Every admitted episode binds:

- game/executable hash, adapter/schema/code version, run and episode identity;
- coherent dense player state and lifecycle;
- instantiated bullet, laser, and enemy-body physical state with stable slot or
  object correspondence when the game exposes it;
- attacks, items, resources, HIT/Bomb events, and stage transitions;
- commanded action, witnessed input, hold/elapsed time, policy identity,
  exploration seed, and the full behavior distribution;
- when declared, randomized action-intention group, step, horizon, intended
  action, initial assignment propensity/distribution, and shield override;
- observed-shield inputs, result, horizon, margin, and publication outcome;
- factual next root for every nonterminal transition;
- immutable compressed shard hashes and byte counts.

Optional ECL, source, RNG, address, and stage-program records are forensic
streams. Their absence must not prevent the generic episode loader from reading
physical transitions. Derived tensors, capped object sets, options, rewards,
and learner features are versioned products, never raw-data authority.

## Admission

An episode is eligible only when it is complete for its declared Practice Stage
or route, continues after HIT, has zero Bomb, no dropped records, valid shard
hashes, exact frame/transition linkage, and no capture/publication/storage
failure. A HIT count of any size is valid training data.

Admission verifies facts and provenance, not the quality of the behavior policy
and not coverage of future source semantics. Data from old schemas stays
immutable but is not silently upgraded.

## Derived causal learner views

The raw episode is frame-linked because that is the factual recording unit. It
is not permission to treat every frame as a fresh policy intervention. Input
publication can remain in a lease until the game witnesses pickup.

The first learner dataset derives one decision row from an actual immutable
policy invocation to the next invocation or physical episode terminal. Each row
must retain:

- the starting and next factual root references;
- the published command and its full behavior distribution, or an explicit
  no-publication marker and exclusion;
- proposed, published, sampled, and executed action identity without aliases;
- elapsed game frames and input-lease/publication evidence;
- the exact sum of physical HIT events between decision roots;
- lifecycle, observation-gap, and learning-exclusion evidence;
- a versioned portable history and shield mask built without future facts.

For action-exposure episodes, frame/transition v14 stores the assignment and
override provenance alongside the ordinary per-root behavior distribution.
The generic loader accepts immutable v13 episodes without inventing these
fields and requires exact raw/transition agreement when v14 exposure metadata
is present. Group identity, step, assignment propensity, and override reason
are never portable actor inputs. The controller still publishes only a freshly
observed-shield-admissible action; an unsafe in-flight input is a fail-closed
control dead-end, not a successful override.

The first L2k audit exposed an important collapsed-epoch edge: a published
action may be witnessed and remain learning-eligible while a later transition
inside the same decision epoch records a control dead-end and interrupts the
unfinished group. The erratum checker therefore requires the factual
`outcome.control_dead_end` transition itself; it does not infer interruption
from group shape or relax execution eligibility. The original rejected audit
and corrected read-only artifact are both retained and hash-linked.

The L2l group-level learner view is derived without changing frame/transition
v14. One row begins only at exposure step zero. It retains the initial portable
observation, randomized intended action, assignment distribution, every
observed override/dead-end mediator, and the next 12 factual unit-frame
transitions. HIT before a new assignment is positive; a complete 12-step
no-HIT group is negative. Gaps, initial non-execution, reassignment before the
outcome, incomplete no-HIT groups, and episode-end truncation remain explicit
censoring reasons. Group IDs and propensities are audit fields, not actor
inputs.

Decision-row HIT totals must sum exactly to the raw complete-episode HIT total.
Rows with no policy intervention are context inside an interval, not new
behavior-policy samples. A policy invocation with no publication remains an
accounting/exclusion row rather than an action-conditioned sample. An
unexecuted action never receives a factual successor.

The same immutable episode may also produce a pre-first-HIT or time-since-HIT
diagnostic view and a complete physical auxiliary-learning view. These views
may measure survival-conditioned coverage or predict factual future events, but
the initial task view remains the complete route with every HIT. The first HIT
is not relabeled as the physical episode terminal, and diagnostic stop-on-HIT
runs remain inadmissible for training.

## Algorithm separation

A corpus inventory contains episode identities, provenance, hashes, and split
groups. A learner names an inventory plus a feature/target schema. A fitted
artifact names both. Replacing BC with IQL, CQL, or another method must not
require recollecting an otherwise valid factual corpus.

The goal metric is complete-route NMNB probability. The initial optimization
target is expected undiscounted physical HIT count over complete episodes. A
fitted artifact must state which claim it optimizes; HIT reduction must not be
reported as NMNB-probability improvement without complete Wine evidence.

Splits use complete episodes. Adjacent frames from one route never cross a
train/validation/test boundary; the initial learnability experiment uses only a
frozen train/validation split. Offline evaluation and final Wine evaluation are
separate evidence classes.
