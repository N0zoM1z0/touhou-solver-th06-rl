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

## Algorithm separation

A corpus inventory contains episode identities, provenance, hashes, and split
groups. A learner names an inventory plus a feature/target schema. A fitted
artifact names both. Replacing BC with IQL, CQL, or another method must not
require recollecting an otherwise valid factual corpus.

Splits use complete episodes. Adjacent frames from one route never cross a
train/validation/test boundary. Offline evaluation and final Wine evaluation
are separate evidence classes.
