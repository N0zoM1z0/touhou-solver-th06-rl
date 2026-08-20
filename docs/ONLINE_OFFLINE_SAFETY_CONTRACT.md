# Online safety and offline data contract

## Boundary

Online and offline work have different jobs.

- Online collision authority is deliberately small and bounded. A published
  action needs a source-complete envelope for every physical frame on which
  that input may remain active. Unknown coverage fails closed.
- The exact Wine process remains paused from coherent source-root capture
  through certification and input publication. Publication therefore belongs
  to the certified epoch; pausing never lengthens the four-frame pickup proof.
- The immutable online policy only ranks actions already certified by that
  authority. It cannot change geometry, margins, delivery delays, or source
  coverage.
- Offline jobs may be expensive and may use wide temporal and spatial context.
  They train only from factual original-retail Wine transitions. They may not
  manufacture an unexecuted successor state.

“Source-complete” includes current bullets, enemy bodies, and lasers; spawn
animation fallthrough; enemy clamps; and any timeline/ECL birth or body/laser
mutation that can occur before the input lease expires. Repeating an
observed-only forecast during the fresh issue check does not cover a future
source event.

## Dense corpus authority

`control-v3` retains three independent layers:

1. compact decoded values used by the resident controller and learner feature
   adapter;
2. raw factual hazard-source records queued to the offline writer;
3. decoded offline-only Player attack, occupied Item, effect/RNG witness, and
   run/resource state from the same paused epoch.

Every occupied bullet retains its collision/motion tail and visual dimensions.
Every state-2/3/4 bullet additionally retains the complete retail Bullet struct
so the ANM completion tick can be decoded offline. Every occupied Enemy and
Laser retains its complete struct, and the dynamic EnemyManager tail retains
timeline timers, boss pointers, random-item cursors, and spell state. The root
also records the exact reachable-bullet slot linkage and the collision margin
used for its Hard set. Authoritative anchors retain immutable stage/ECL graphs.

The offline layer retains active player-shot geometry/timers, occupied item
positions/states/timers, item allocation cursors, power, score, graze, deaths,
bomb counters, spell captures, point-item counters, rank/RNG, and the effect
state needed to preserve source RNG consumption. These fields never enter the
online Hard gate.

This remains a factual combat/resource root, not a portable save state or
permission to simulate unexecuted actions. Full visual-effect and rendering
state is intentionally omitted and schema-visible. Any later task requiring
those raw subsystems must introduce a new explicit tier instead of silently
inferring them; the factual outcome is always the next Wine root.

## Audit verdict that introduced this contract

The audit is bound to GensokyoClub/th06 commit
`cc475a0bc3fef38683b0f02224c87ddba0a021d9`. It found:

- the native player, bullet, enemy-body scaling, laser midpoint behavior, and
  inclusive AABB geometry agree with the shipped source;
- pure 0x2/0x4/0x8 spawn bullets were under-projected on their completion
  update and could be removed by both reachability prefilters;
- collidable enemy projection omitted `Enemy::ClampPos`;
- the former resident observed-only gate did not cover future ECL/timeline
  births or body/laser mutations, so it was not a complete Hard authority;
- a rejected calc-phase root was retried while the exact process remained
  suspended, so BulletManager could never finish that phase; paused capture
  now resumes between bounded attempts and retains only the successful pause
  through publication;
- the corpus omitted the Hard fallback margin and its audit consequently
  reported a false unsafe divergence;
- dense learner hazard features are intentionally capped/lossy and therefore
  cannot serve as geometry evidence.

The spawn, clamp, data, and replay-accounting defects have direct regression
tests. The correctness-first controller now evaluates a bounded four-frame
source commitment from the exhaustive retail root and records the exact
committed primitives. Promotion remains blocked until live all-stage coverage,
latency, and replay parity pass; an observed-only action set must never be
described or promoted as a complete retail-safe set.

The run audit also performs an offline causal cross-check. For every adjacent
`control-v3` root within Hard-4, it asks whether the preceding committed frame
contains every retained bullet, lethal enemy body, and laser collision
geometry that can intersect the player's preceding reachable envelope. This
is a one-sided falsifier: a retained uncovered hazard rejects the run, while a
hazard that retired during the source update is not silently counted as
verified. Ambiguous zero-delay laser retirement is reported separately. The
small float32 comparison tolerance in this audit never changes the online
collision margin or certified action set.
