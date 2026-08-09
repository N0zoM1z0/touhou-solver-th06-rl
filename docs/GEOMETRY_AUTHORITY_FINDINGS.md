# Geometry authority findings

This log records exact-state cases where source-physical replay separated a
native geometry limitation from a learned-policy error. It is intentionally a
correctness record, not a model leaderboard. All paths under `artifacts/` are
ignored local evidence; the counts and provenance needed to reproduce the
experiments are retained here.

## 2026-08-09: Stage 2 late-failure boundary

Baseline provenance:

- solver implementation: `054c05d`;
- authoritative portable source: `1350819f396b9db93eb9891c107e651be70c83f6`;
- source binary SHA-256:
  `402f7d89a2cdbed0ad9b32b121177a345a18bb4dfa79ac92219a2ed163edc873`;
- scope: Lunatic / Reimu-A / Stage 2;
- continuations: generic clearance, native local horizons 4/12/30/60,
  incumbent `a4245bdfc8d3`, and rejected LambdaRank `1418c9a8fd8a`;
- branch bound: 1,200 physical ticks.

Two exact checkpoints at the end of fresh seed-113/114 failures produced no
witness:

| seed | sequence / tick | native first actions | audited branches | result |
|---|---:|---:|---:|---|
| 113 | 3306 / 3307 | 3 | 21 | every branch lost native authority after 1 tick |
| 114 | 4243 / 4244 | 4 | 28 | every branch lost native authority after 1 tick |

The independent audit accepted both files and recomputed 49/49 branches as
`authority-failure`, with zero physical deaths and zero Bomb use. This is
`oracle-no-witness`, not a proof of physical impossibility.

An ungated source-physical diagnostic then retained each native first action,
enumerated all 18 Bomb-free constant second actions, and let the authoritative
game decide HIT for 60 ticks. Seed 113 had 0/54 witnesses and a maximum physical
survival of 17 ticks; seed 114 had 0/72 and a maximum of 9 ticks. These late
states are therefore genuinely poor correction targets even though the exact
unrecoverability of arbitrary non-constant sequences remains unproved.

## Root cause: player-aim turn over-enclosure

The seed-113 post-action state exposed a solver false negative in addition to
the late-state problem:

- zero delivery delay, zero extra collision margin, and a four-frame horizon
  still certified 0/18 actions;
- the authoritative source nevertheless had constant-action branches that
  survived 17 ticks;
- removing fired bullets with source `ex_flags & 0x080` restored all 18
  actions, while those player-aim bullets alone rejected all 18;
- 72 fired player-aim bullets were present in the measured snapshot.

`BulletManager::OnUpdate` in the authoritative checkout retargets a `0x080`
bullet to `g_Player.AngleToPlayer(...) + direction_rotation` when its direction
timer fires. The old `headless_geometry._project_fired` could not express that
candidate-dependent angle in one shared hazard view, so it enclosed every
direction from the bullet position. That 360-degree box was safe but invented
trajectories aimed at physically unreachable player positions.

The correction bounds the target by the complete source-kinematic reachable
rectangle during the fixed four-frame Hard window. For the common one-shot
player-aim form, the rectangle subtends one angular interval; its endpoints and
contained cardinal angles give constant-work velocity-component extrema.
Unsupported multi-turn/dynamic combinations and retargets beyond the Hard
window keep the prior fail-close arbitrary-direction envelope.

Measured on the same 156-bullet exact state:

| check | old 360-degree enclosure | corrected reachable cone |
|---|---:|---:|
| native legal actions | 3 | 7 |
| mean lower + certify time | 9.73 ms | 9.86 ms |
| relative overhead | - | 1.3% |

All seven corrected actions were replayed for four constant source-physical
ticks: 7/7 reached the tick bound with zero deaths and zero Bomb. A dynamically
recertified generic continuation extended the best authority-preserving prefix
from one tick to 33 ticks, but still found no 60-tick witness. The correction
therefore fixes a real geometry false negative without pretending that this
already-late state became globally recoverable.

The roughly 10 ms absolute time is pre-existing Python hazard-lowering cost on
this extreme bullet count. Moving that lowering into the fixed native kernel is
a separate resident-latency task; the correction itself did not introduce a
material hot-path regression.

## Consequences for collection and learning

Last-frame corrective labels can be unsatisfiable or dominated by authority
conservatism. Failure collection must sweep backward from the first HIT or
authority release at several offsets and retain the transition where a
constructive witness first appears. Samples after the witness boundary are
geometry/search diagnostics, not ordinary ranking supervision.

Geometry changes also change the native action set. The feasibility generator
therefore refuses such a comparison by default. An explicit
`--allow-native-set-revision` A/B records both the corpus and recomputed legal
sets, and the independent auditor rejects undeclared set drift.

The next acceptance gate is not training loss. It is a clean before/after
exact-state audit at earlier failure offsets, followed by fresh natural Stage
rollouts. Windows physical play remains the final promotion authority.
