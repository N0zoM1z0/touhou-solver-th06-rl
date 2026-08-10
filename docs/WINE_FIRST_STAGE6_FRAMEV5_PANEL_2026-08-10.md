# Stage 6 frame-v5 Wine-first panel (2026-08-10)

## Decision

The first new frame-v5 frozen-UCB original-retail Wine episode validates the
dialogue-delivery evidence and exact replay plumbing, but it does not authorize
a residual candidate or COW label. Frozen UCB remains the sole incumbent.

Exact sampled input delivery is now audited separately from RNG and physical
state. Both reconstructed-source domains reproduced every retained retail
dialogue current-input sample, yet MinGW source-under-Wine had already diverged
from retail RNG before dialogue began. Equal action delivery is therefore
necessary but is not evidence of retail-source equivalence. Rendering/draw
work, compiler arithmetic, platform libraries, and other source-versus-retail
paths may change RNG-consuming branches; the audit records the first observed
divergence without assigning unsupported causality.

## Original-retail Wine episode

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

Ignored local evidence is under
`artifacts/wine-first-stage6/framev5-panel-r1-exact-v2/`.

## Next gate

Collect a few additional independent frame-v5 frozen-UCB original-retail Wine
first-failure episodes under the same safety contract. For each episode:

1. export and audit exact dialogue delivery;
2. record first RNG, game, hazard, geometry, and native-hard-set divergence
   separately for Linux and MinGW source;
3. group failure regions by episode using only Wine-reproducible generic
   features;
4. allow targeted COW only for a repeated region whose chosen checkpoint is
   still exact in the relevant reconstructed-source domain.

Do not train from this single episode, do not repair RNG drift by relaxing a
tolerance, and do not use full-Stage HIT continuation as training data.
