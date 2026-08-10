# Stage 6 Wine-anchored targeted COW audit (2026-08-10)

## Decision

The first Stage 6 residual hypothesis is rejected.  The repeated original-retail
Wine sub10 failure region suggested replacing incumbent `right_fast` with
`left_fast`.  Three independently collected Wine prefixes were reconstructed
in the source runtime and branched for 600 frames.  The incumbent was better on
two prefixes and tied on one.  No sub10 residual candidate, shadow policy, or
active canary is authorized.

This is a useful negative result.  It prevents a correction inferred from
correlated Wine terminal frames or unrelated headless seeds from entering the
live controller.  Frozen UCB remains the incumbent.  The next bounded audit is
the repeated sub31 family, followed by sub18 if sub31 also supplies no
independently supported intervention.

## Why the first generic headless audit was insufficient

The episode-grouped Wine audit queued three repeated physical families:

- sub10: boundary, dense bullets, broad native-safe set;
- sub31: interior, lasers present, broad native-safe set;
- sub18: boundary, medium bullet density, broad native-safe set.

The existing headless corpus supplied 75 unique checkpoints in 81 COW evidence
files, all with a 600-frame horizon.  The sub10 family contained 14 checkpoints
from seeds 153, 157, 158, 161, 165, and 166.  On the six checkpoints that
exactly represented the proposed `right_fast` versus `left_fast` pair, the
seed-level result was one `left_fast` win, two `right_fast` wins, and three
ties.  Other fixed alternatives were similarly mixed.

An odd-seed development/even-seed confirmation committee produced zero
confirmation activations.  A separate seed-167-through-174 coverage batch was
Bomb-free and HIT-free but had zero exact overlap with the Wine-defined region.
It therefore could not rescue the hypothesis.  This older evidence remains a
fast rejection and hypothesis audit; it is not a causal Wine checkpoint test.

Evidence:
`artifacts/wine-first-stage6/targeted-cow-v1-report.json`, SHA-256
`9dddf452cec83e707becf282830dbe9c64c01ecceda5899f24e6000858c089a1`.

## Deterministic retail replay contract

The reconstructed source runtime now exposes two diagnostic-only controls:

- `--stage-rng-seed` restores the recovered pre-Stage RNG state immediately
  before Stage registration;
- `--auto-shoot-after-tick` delays automatic Shoot until the retail controller
  actually published its first battle action.

These changes are in the ignored source checkout
`reference/GensokyoClub-th06-portable` on branch
`th06-rl-headless-spike`:

- `e92ff98 Add retail stage RNG replay control`;
- `666398e Align delayed Shoot for retail replay`.

Both commits are pushed to `headless/th06-rl-headless-spike`.  Native and
MinGW builds passed, the MinGW executable ran under an owned dedicated Wine
prefix, and cleanup left no source runtime or wineserver process.  The exact
source commit used below is
`666398ee6b1aed713e214305d4370344c30c7e6b`; the native binary SHA-256 is
`131a7020a230fcedf69805d5a528ea6fe7a8856e41659915037b8b04609c2a58`.

`scripts/export_wine_action_stream.py` strictly verifies an immutable Wine
first-failure prefix, reconstructs the pre-Stage RNG seed, rejects Bomb or
unsupported input, and run-length-encodes the recorded action delivery.
`scripts/audit_retail_source_replay.py` then compares source traces with retail
snapshots.  A checkpoint may enter COW only when
`scripts/label_retail_replay_cow.py` verifies both of the following:

1. the reconstructed source state matches the Wine physical state within
   `1e-6`;
2. the source native hard-action set exactly equals the set recorded in Wine.

This is deterministic replay from a retail anchor, not a copied retail memory
snapshot and not original-retail execution.  It can reject a residual
hypothesis.  It cannot promote one.

## Delivery bug found during replay

The first exporter version started automatic Shoot at the first captured
snapshot.  That is wrong when coherent capture is followed by stale retries:
the controller has observed a frame but has not yet published input.  In run
`20260810T055603Z-598480400`, capture began at frame 132, stale retries occurred
at frames 132 and 136, and the first non-null `published_action` was at frame
137.  Starting Shoot at 132 caused input divergence at frame 136, RNG
divergence at frame 450, and false source HITs at tick 634.

The exporter now derives the Shoot threshold from the first actual published
action.  The invalid v1 audit is retained only as negative delivery evidence at
`artifacts/wine-first-stage6/retail-replay-drift-20260810T055603Z-v1/retail-source-audit.json`,
SHA-256
`63cc945e80fbcde367d48195482ed2b82cc7cc91fcd160ebe989aa9c504c0cc2`.
It must not be used for COW, training, or platform conclusions.

## Full-prefix retail/source validation

Two independent frozen-UCB sub10 Wine prefixes were replayed through both the
native Linux source runtime and the MinGW source runtime under Wine:

| Retail run | Retail frames | Common snapshots | Shared dynamics | Source result |
| --- | ---: | ---: | --- | --- |
| `20260810T053928Z-269933800` | 127--3117 | 2,736 | RNG, game, hazard counts, and player geometry match at `1e-6` | Linux and MinGW reach tick 3117, no HIT |
| `20260810T055603Z-598480400` | 132--3199 | 2,808 | RNG, game, hazard counts, and player geometry match at `1e-6` | Linux and MinGW reach tick 3199, no HIT |

Both source-platform pairs retained exact discrete delivery through their full
tick bounds.  Linux/MinGW floating-point geometry still begins to drift at
tick 441 and exceeds `1e-6` at tick 457 or 476 depending on the route.  Against
original retail, the only categorical discrepancy is a known dialogue/control
gap at target frame 2805: retail releases or changes focus while the source
action stream continues the battle input.  Physical player geometry remains
equal at `1e-6`, so checkpoints before and after the gap remain individually
verified rather than assumed.

The source trace ends at `tick-limit` because it replays the recorded actions;
it does not execute the original-retail controller's native authority-failure
stop.  The retail `control-dead-end` remains the authoritative outcome.

Evidence:

- first replay audit:
  `artifacts/wine-first-stage6/retail-replay-v1/retail-source-audit-v2.json`,
  SHA-256
  `832dcd0886c301d31839a96fe389df5c9a8abd3ad1cb619c8bc904bdb22125fa`;
- second replay audit:
  `artifacts/wine-first-stage6/retail-replay-drift-20260810T055603Z-v2/retail-source-audit-v2.json`,
  SHA-256
  `1524fb502fc303c6646f9c3e54afcf8a272234b500a6ef50552049b042e8741e`.

## Exact sub10 pair result

Only `right_fast` and `left_fast` were branched.  An exhaustive 18-action
600-frame sweep was unnecessary once the declared pair could be decided; the
stopped exploratory process and all child source processes were cleaned up.

| Wine prefix | Policy evidence | `right_fast` | `left_fast` | Result |
| --- | --- | --- | --- | --- |
| `20260810T053928Z-269933800`, sequence 2650, frame 3032 | frozen UCB | authority failure at 268 ticks | authority failure at 268 ticks | tie |
| `20260810T055603Z-598480400`, sequence 2717, frame 3109 | frozen UCB | survives 600 ticks, reserve 5.657 | authority failure at 90 ticks | right better |
| `20260810T074759Z-903969600`, sequence 2538, frame 3105 | replay-proven Wine shadow | survives 600 ticks, reserve 31.598 | authority failure at 35 ticks | right better |

The first row is deliberately a tie: both actions reach the same authority
failure at the same time.  A larger intermediate legal set for `left_fast`
does not override the conservative survival-first outcome ranking.

Aggregate evidence:
`artifacts/wine-first-stage6/retail-replay-cow-v1/report.json`, SHA-256
`933493ba072368642fb0792b9a7f260162e013e9d9f146ab49a96045f47925c6`.

## Gate and next work

The sub10 leftward residual has zero candidates.  Do not train a classifier to
recover it, loosen the region, or select a favorable seed.  The incumbent owns
all sub10 decisions.

For sub31, reuse the same contract:

1. select independent original-retail Wine prefixes in the repeated sub31
   family;
2. export and validate their full deterministic source replays;
3. derive at most one concrete action-pair hypothesis from the existing COW
   evidence;
4. branch only that declared pair at exact Wine-verified checkpoints;
5. return to the incumbent on disagreement or insufficient support.

No residual model is fit until a family produces the same native-safe
alternative across independent Wine anchors.  Even then it enters replay and
Wine shadow before any active canary.
