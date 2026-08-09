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

## Second differential: cross-candidate aim contamination

The shared reachable cone was safe, but it was still not exact enough for a
native *set* of mutually exclusive first actions. A trajectory aimed at a
left-moving candidate remained in the common hazard view used to certify a
right-moving candidate, and vice versa. A candidate-coupled diagnostic at the
same seed-113 checkpoint separated the layers:

| Hard authority | certified actions |
|---|---:|
| old arbitrary 360-degree enclosure | 3 |
| shared reachable-target cone | 7 |
| candidate/delivery-coupled native turn | 11 |
| authoritative source, immediate constant action for 4 ticks | 12 |

Every one of the 11 candidate-coupled actions is in the 12-action source-safe
set. The remaining source-safe action is conservatively rejected by the fixed
0--3 frame delivery coverage and 0.35 collision margin; an immediate source
branch alone is not evidence that those robustness terms may be weakened.
Seed 114 remains unchanged at four certified actions.

The final common `0x080` fired-bullet turn is now passed as compact source state
to the native Hard kernel. For each of the fixed 18 actions, four delivery
delays, bounded `Keyboard::_sync` prefixes, and four Hard frames, the kernel
advances the player first and then retargets the bullet exactly as the source
calc-chain ordering requires. Unsupported combinations, spawning bullets,
multi-turn forms, and longer offline lookahead retain the shared fail-close
projection.

On 60 repetitions of the same 156-bullet snapshot, combined lowering plus
certification changed from a 9.776 ms median for the shared cone to 8.686 ms
for candidate-coupled native aim. The correction therefore removed four more
false negatives while reducing this Python-path benchmark by 11.2%; native
candidate certification itself took about 0.65 ms for 72 aimed bullets. The
remaining absolute cost is common Python lowering for other hazard classes.

## Third differential: headless delivery is not Windows delivery

The 1,200-tick extension of the corrected seed-113 branch did not remain a
witness: `up_left` plus generic clearance reached 133 ticks before Hard became
empty at tick 3440. Replaying that exact 260-bullet state in the authoritative
runtime found 13/18 immediate constant actions source-safe for four ticks.
The factor audit was exact:

- delay 0, margin 0 certified the same 13 actions as source replay;
- delay 0, margin 0.35 retained 10 actions;
- delays 0--3 certified no actions even with margin 0;
- 224 ordinary fired bullets (`ex_flags` motion mask zero) caused the closure;
  the 36 candidate-coupled player-aim bullets did not.

This is not another bullet-geometry error. The Linux `STEP` protocol receives
one action and publishes it before the same authoritative `RunTick`, so its
complete delivery set is exactly `{0}`. Reusing the asynchronous Windows
0--3-frame pickup envelope in that synchronous environment invented delivery
paths which cannot occur and stopped otherwise source-valid learning runs.

Headless Hard now declares `synchronous-step-v1` and certifies delay 0 only.
The collision margin, fresh issue check, native ownership, unknown-state
fail-close behavior, and Bomb prohibition remain unchanged. The Windows
adapter retains its measured bounded delivery envelope and input-lease checks;
a learned action is still intersected with that physical native set before it
can be published. Corpus, feasibility, DAgger, and model manifests record the
delivery contract, and training/auditing refuses to silently mix contracts.

This authority correction does not by itself improve the existing fallback.
With the same initial `up_left`, generic clearance over the enlarged exact set
survived only 16 ticks instead of the former 133 because its later choices
changed. The old 0--3 envelope had accidentally acted as a ranking regularizer;
it was never a valid source-physical restriction for synchronous headless play.
Any useful robust-delivery preference must therefore be an explicitly logged
non-authoritative feature or ranker term, while the exact delay-0 set remains
the collision authority. Promotion requires oracle and held-out route evidence
over that exact set rather than the smaller-set trajectory.

## Fourth differential: margin closure is not geometry mismatch

The corrected seed-114 route reached another empty configured set at tick
4254. Its exact fingerprint was
`51fea335ac942e9dd6c65c5c7186931a8d31936f17ab7110bfd840ec67f2beb0`.
The isolated COW differential tried every ordinary action directly in the
authoritative runtime for the four-tick Hard window:

- configured margin 0.35 certified no action;
- native margin 0 certified only `down_left`;
- source execution also kept only `down_left` no-HIT for all four ticks;
- the other 17 actions physically HIT within one to three ticks.

Margin-zero native and source execution therefore agree exactly. This is a
conservative-margin closure, not permission to weaken collision authority.
The route had already spent its recovery reserve and left only a knife-edge
source path. Training must move the correction to an earlier state instead of
teaching the ranker to rely on a sub-margin action.

Two later pure-h12-teacher failures close the remaining geometry suspicion.
After their final certified actions, seed 113 tick 3484 and seed 114 tick 3330
had neither a configured nor margin-zero native action, and all 18 direct
source trials physically HIT within four ticks. These terminal rows are true
tested constant-action dead ends. They remain evidence about an earlier
planning failure, not useful last-frame action labels.

## Fifth differential: newborn laser history was a false terminal

The corrected-contract incumbent repeatedly reached Stage 2 decision 12,932
and then failed closed on `laser slot 0 lacks angular history`. At seed 114,
an exact oracle confirmed that all 36 action/continuation branches stopped one
tick later on that same observation-authority error. The isolated source
differential then executed every ordinary action for the complete four-tick
Hard window: all 18/18 remained no-HIT and no-Bomb. This state was neither a
policy dead end nor a collision closure.

The authoritative `Laser` object has no intrinsic angular-velocity member.
Commit `63afca914939bc6dfcd78aa913a0fc55173b1fe5` therefore adds an observation
fact, `angle_initialized`, only while the source laser timer is initialized or
reset; its binary SHA-256 is
`10c408012bcd556e2b5187b21cc7c8feb3cd24820af7d250e21cf3e3adc32b0e`.
The native lowering accepts zero angular velocity only when that flag, timer
bound, and zero value agree. An untracked lethal laser without the explicit
source initialization flag still fails closed. Replaying the seed-114 onset
with the patched runtime produced three initialized lasers and an 18-action
native set, matching the 18 source-safe trials.

This is an additive headless observation ABI correction, not a learned-policy
permission. Old rankers may be copied to the new source identity only by the
bounded compatibility-extension tool and the immutable differential evidence;
the resulting copy remains unpromoted until new-source full-stage rollout.

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
