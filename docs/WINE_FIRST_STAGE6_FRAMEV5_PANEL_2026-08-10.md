# Stage 6 frame-v5 Wine-first panel (2026-08-10)

## Decision

The fixed three-episode frame-v5 frozen-UCB original-retail Wine panel is
complete.  It validates the dialogue-delivery evidence and exact replay
plumbing, identifies two repeated generic fallback-opportunity regions, and
finds **zero** residual candidates after the robust independent-prefix COW
gate.  Frozen UCB remains the sole incumbent; no shadow or active policy is
authorized.

The panel also exposed and fixed a diagnostic contract bug.  Retail Wine Hard
certification covers asynchronous input pickup delays `(0, 1, 2, 3)`, while a
source STEP branch publishes synchronously with `(0,)`.  The first COW
preflight compared the synchronous source safe set with the retail safe set
and falsely rejected two exact boundary checkpoints.  Re-certifying the same
source observation with the retail delivery set reproduces the recorded
retail action set.  The corrected COW-v2 audit keeps the two delivery domains
separate; it does not weaken delivery coverage or change the resident gate.

Exact sampled input delivery is now audited separately from RNG and physical
state. Both reconstructed-source domains reproduced every retained retail
dialogue current-input sample, yet MinGW source-under-Wine had already diverged
from retail RNG before dialogue began. Equal action delivery is therefore
necessary but is not evidence of retail-source equivalence. Rendering/draw
work, compiler arithmetic, platform libraries, and other source-versus-retail
paths may change RNG-consuming branches; the audit records the first observed
divergence without assigning unsupported causality.

## First original-retail Wine episode

Run `20260810T124531Z-310133600` used Lunatic / Reimu-A / Stage 6, immutable
frozen UCB, exploration zero, natural Practice, no life patch, and default
first-failure stopping. It ended at retail frame 3303 with
`authority-stop:Hard safe set empty`:

- zero physical HITs and zero Bomb use/request;
- about 90 seconds wall time;
- 2,912 frames, 2,911 transitions, and three anchors;
- complete lossless storage and an intentionally incomplete Stage trajectory;
- frame schema v5 with 251 dialogue-delivery records covering 238 unique
  frames from 2568 through 2805;
- exact input/PID/Wine-prefix cleanup and no leftover trial process.

The run is one independent episode. Its 2,911 adjacent transitions are not
2,911 independent examples and no fit was made from them.

## Completed three-episode panel

Two further runs were collected before any fit or policy change.  All three
used the same immutable retail executable, native kernel, frozen-UCB policy,
scope, exploration-zero setting, first-failure stop, and frame-v5 recorder.

| Run | Terminal frame | Frames / transitions | Dialogue records / unique frames | Physical HIT / Bomb | Terminal |
| --- | ---: | ---: | ---: | ---: | --- |
| `20260810T124531Z-310133600` | 3303 | 2,912 / 2,911 | 251 / 238 | 0 / 0 | Hard safe set empty |
| `20260810T131002Z-681278300` | 3573 | 3,201 / 3,200 | 256 / 239 | 0 / 0 | Hard safe set empty |
| `20260810T131344Z-316672800` | 3722 | 3,337 / 3,336 | 253 / 238 | 0 / 0 | Hard safe set empty |

Every retained dialogue current-input frame matched both reconstructed-source
domains: 238/238, 239/239, and 238/238 respectively.  Runs two and three had
no RNG, game-state, hazard-count, or input divergence from retail at any
coherent snapshot in either source domain.  Their first player half-width
difference above `1e-6` occurred at frames 709 and 2438.  Run one retains the
independent pre-dialogue MinGW RNG divergence at frame 1206 described below.
Thus pixel/draw/platform effects remain an explicit possibility, but they are
not inferred where the trace only proves floating-point or RNG divergence.

The exact frozen-incumbent replay covered 9,440 policy calls across the three
runs with zero recorded-incumbent mismatch, zero policy mismatch, and zero
shadow action-contract violation.  Episode grouping then reduced 69 correlated
positive rows to three physical failure units.  All terminal failures were the
same sub10 boundary/dense-bullet family.  Two action opportunities had support
from two independent runs:

- incumbent `down_right`, native baseline `down_fast` in runs one and two;
- incumbent `up_fast`, native baseline `down_fast` in runs two and three.

Adjacent frames and multiple rows inside one run never add independent support.

## Exact dialogue replay

The exporter now distinguishes three clocks:

1. retained retail `game_frame`, whose current/previous input globals are the
   evidence;
2. source runtime input-request tick N;
3. source observation/game frame N+1, where that requested input is visible.

The first implementation incorrectly keyed the exact stream directly by
retail game frame. The new dialogue audit caught the one-tick error at frame
2569. The corrected exporter sends each retail input at `game_frame - 1` and
has a focused regression test for that mapping.

The final stream contains 15 Bomb-free RLE segments. Current, previous, and
held-counter evidence establishes 240 consecutive retail input frames from
2566 through 2805. Of those, 238 frames are directly sampled current inputs;
one additional frame is established only by the held counter after accounting
for overlap with previous-input evidence. Other unobserved battle-gap inputs
remain explicitly inferred.

The portable source runtime accepts this diagnostic stream through
`--retail-dialogue-inputs PATH`. It validates ordered non-overlapping segments,
allowed bits, and Bomb prohibition, then overrides the generic dialogue
approximation only on established ticks. This does not enter the resident
controller or select movement by frame/seed/Boss/phase. The implementation is
pushed to `headless/th06-rl-headless-spike`; the final clean source revision is
`86648a449d2f5db5dcd5ef52c724c6e19c4416c9`.

## Differential result

The corrected action stream was replayed through native Linux source and the
32-bit MinGW source build under Wine. Both returned zero, reached tick 3303,
reported no HIT, and left the dedicated source Wine prefix clean.

Dialogue delivery is exact in both domains:

| Audit | Linux source | MinGW source under Wine |
| --- | ---: | ---: |
| retained sample records | 251 | 251 |
| unique sampled frames | 238 | 238 |
| matched sampled frames | 238 | 238 |
| missing frames | 0 | 0 |
| first input divergence | none | none |

Independent state comparisons still reject equivalence:

- Linux versus MinGW source exact physical state first differed at tick 441
  (`enemies[0].hitbox_width`);
- their first birth-event difference was at tick 446;
- their first discrete difference was RNG seed at tick 1206
  (`26340` versus `27319`);
- MinGW source likewise first differed from original-retail RNG at retail frame
  1206, then differed in bullet count at frame 1270;
- both divergences precede the first retained dialogue sample at frame 2568;
- native Linux source matched retail RNG, game state, hazard counts, and input
  at every coherent corpus snapshot, but player half-width differed by more
  than `1e-6` at frame 3025.

This closes the old unknown-dialogue-delivery defect for this one episode. It
does not make a later checkpoint COW-valid if RNG, game state, hazard counts,
geometry, or the native hard set has already diverged.

## Delivery-contract correction and targeted COW

At run-one sequence 2904 / frame 3296, retail and Linux source both contained
524 live bullets.  Bullet state, hitbox size, sprite size, and ordering matched;
the largest observed bullet-position difference was about `1.61e-4`.  The
source synchronous `(0,)` Hard set nevertheless contained 16 actions while the
recorded retail set contained 14.  The discrepancy was exactly the three-tick
input-pickup envelope: applying `(0, 1, 2, 3)` to the same source observation
produced the same 14 action names as retail, with clearance differences around
`3e-5` at that checkpoint.

`label_retail_replay_cow.py` now emits schema
`th06-rl-retail-replay-cow-v2`.  Its preflight:

1. matches retail/source shared physical state at `1e-6`;
2. reconstructs the recorded retail Hard set using `(0, 1, 2, 3)`;
3. requires every requested first action to be retail-Hard-safe;
4. retains `(0,)` only for the diagnostic source STEP branch.

The strongest repeated pair, `down_right` versus `down_fast`, then completed
two independent 600-tick COW branches:

| Wine anchor | `down_right` | `down_fast` | Robust result |
| --- | --- | --- | --- |
| run one, seq 2904 / frame 3296 | survives 600; min safe width 2; reserve 32.998 | survives 600; min safe width 3; reserve 30.091 | incumbent `down_right` better |
| run two, seq 3193 / frame 3566 | survives 600; min safe width 3; reserve 17.154 | survives 600; min safe width 3; reserve 35.799 | `down_fast` better |

The raw diagnostic rank selects `down_fast` in both rows, because it uses the
exact safe-set width before reserve.  Candidate construction deliberately uses
the predeclared robust rank: safe-set width and boundary reserve are bucketed
to avoid learning from pixel-level accidental argmaxes.  Under that rank the
two independent anchors disagree, so the alternative is rejected and the
candidate count is zero.  The second repeated pair already has an exact anchor
favoring incumbent `up_fast`; no additional branch can make the alternative
unanimous, so it also creates no candidate.

## Evidence

- retail run JSON SHA-256:
  `619cf2ade46cab9ebce49deba695deaa29beb4e1cd2e877b6131e4b5c3951f50`;
- retail manifest SHA-256:
  `00e3791b1974bcdf1de754abc505e0c6c24c374c39c2da48ce2e31fcd716da60`;
- action stream SHA-256:
  `d4427f1f40754c935762a0cf40f7555c096c179dab9e36a4af59f273c8f0ab37`;
- exact dialogue-input text SHA-256:
  `203140550fdc837c9e492a6f07ef26a2777e784c518cc956f968396901b81564`;
- source-platform report SHA-256:
  `d16200a9e6edc74f16e3007f952affd431624330191aa80d63c3408492052476`;
- retail/source audit SHA-256:
  `d197b76cd42fdb8763620f53b63ae875ede8ac1217d41928e15ab863865a560a`;
- tested native Linux source binary SHA-256:
  `f1afd4fae66f5af4f913ec251f8ae66bbe46a5219814a85b39352a43888c0ba0`;
- tested MinGW source binary SHA-256:
  `92dc7184acb010d6a8c2fd85799f2e81460c1436cad00c89fb221d6f6745d5fd`.

Three-episode and COW evidence:

- factual action replay SHA-256:
  `b1984a370914f458a98f8d0b33796fdff2c7af745387b9cf4ca3d1f396d6e566`;
- episode-grouped failure-region audit SHA-256:
  `9f80aa102d37e555043ccce15ce7182262307f230232a53c78056713f7eef694`;
- run-one COW-v2 SHA-256:
  `b7c9c0b2092d64debd92c2bf34fa9f6cd02b6bf329edc24f8a60251add146b27`;
- run-two COW-v2 SHA-256:
  `1a67be59418b1cc2d0965a5db6a20806f4d9f469a94a276f76b9d29e67f723ec`;
- robust COW aggregate SHA-256:
  `efd6509bfeb33bdfe66c1b16ce02fc7ce57acf18068faa654146d9cff39f66a6`.

Ignored local evidence is under
`artifacts/wine-first-stage6/framev5-panel-r1-exact-v2/`.

## Next gate

This fixed panel is closed and must not be repeatedly mined until a favorable
threshold appears.  It authorizes no fit or shadow run.  The next bounded
experiment is another predeclared small frozen-UCB Wine first-failure panel,
with no model changes between episodes.  Its purpose is to discover whether a
new generic failure opportunity repeats; the two rejected action pairs above
remain rejected regardless of later favorable rows.  For each new episode:

1. export and audit exact dialogue delivery;
2. record first RNG, game, hazard, geometry, and native-hard-set divergence
   separately for Linux and MinGW source;
3. group failure regions by episode using only Wine-reproducible generic
   features;
4. compare native hard sets under the correct retail delivery contract;
5. allow targeted COW only for a new repeated region whose chosen checkpoint
   is still exact in the relevant reconstructed-source domain.

Do not train from this panel, do not repair RNG drift by relaxing a tolerance,
and do not use full-Stage HIT continuation as training data.
