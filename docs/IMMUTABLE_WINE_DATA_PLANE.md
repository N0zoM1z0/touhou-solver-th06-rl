# Source-complete Wine data plane

## Current admission state

There is no legacy corpus authorized for current offline-RL training. The old
transition-v6/v9/v10 registry was removed because a transition schema alone
does not prove the current online safety and factual-state contract. Those runs
predate source-complete Hard lowering, same-paused-epoch publication, raw source
commitments, comprehensive offline facts, and the causal successor audit.
Historical artifacts remain quarantined evidence and must not be scanned or
silently admitted by a learner.

The first admissible inventory will be built only from newly completed,
source-complete original-Wine episodes. Until such an inventory is created,
offline fitting has zero authorized training episodes.

## Permanent separation

Collected data, learning algorithms, and fitted policy artifacts are separate:

`original-Wine facts -> admitted immutable inventory -> offline learner -> immutable fit`

- Data identity binds the exact retail executable, native library, code commit,
  schemas, episode identity, run/manifest hashes, and per-run audit.
- Algorithm identity declares the factual capabilities it consumes and never
  changes or filters the admitted inventory based on observed outcomes.
- Fit identity binds the exact inventory, whole-episode split, implementation,
  parameters, and seeds. It cannot become a factual corpus.
- Evaluation creates new normal-speed original-Wine evidence. Offline replay
  may reject a candidate but cannot promote it.

## Required admission evidence

Every admitted episode must satisfy all of these conditions. Missing evidence
fails closed; a compatible-looking transition version is insufficient.

1. It is a complete declared Practice Stage or full route with HIT continuation,
   zero Bomb, zero dropped rows, and exact HIT conservation.
2. Its run metadata declares the current authoritative frame/object/anchor
   schemas, `source-complete-hard-v1`, and same-paused-root publication.
3. Dense roots retain raw bullet, laser, enemy, manager, player-attack, item,
   score/graze, RNG/rank, and NMNB resource state independently of learner
   features.
4. Every published action retains the exact four-frame AABB/laser primitives
   certified online; unknown or incomplete source coverage is absent.
5. The ordinary run audit has no integrity error, dense native replay agrees,
   and the one-sided retained-next-root audit finds no uncovered factual AABB or
   laser hazard.
6. Artifacts are immutable and every run path and hash is unique. Partial,
   first-HIT, time-bounded, authority-failed, or cleanup-failed runs are never
   selected for training.

The admission builder must validate the complete source root rather than skip
bad outcomes. Training/validation roles are assigned later by whole physical
episode, never by adjacent frames.

`scripts/collect_wine_parallel.py` writes the predeclared schedule before any
episode and publishes `th06-rl-source-complete-parallel-collection-v1` only
after every row is clean. Each row binds run, manifest, audit, normalized
factual digest, policy, code, and pool hashes. This ledger is the current
admission boundary; arbitrary directory scans are forbidden.

## Parallel collection boundary

Parallel Wine workers may append facts only after the fixed-seed serial versus
two-worker compatibility gate passes exactly. Each worker owns its game copy,
Wine prefix, display, CPU partition, artifact root, and corpus root. A worker
failure cancels and reaps the complete batch; only individually audited,
immutable complete episodes can enter the next inventory. Canary and promotion
runs remain sequential and natural-speed.
