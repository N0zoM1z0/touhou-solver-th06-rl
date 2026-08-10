# Stage 6 source-platform differential (2026-08-10)

This record closes the first Wine-first implementation gate from
`WINE_FIRST_LEARNING_PLAN_2026-08-10.md`: replay one bounded, immutable action
stream through the native Linux reconstructed source runtime and the 32-bit
MinGW reconstructed source runtime under Wine, then report the first physical,
event, discrete-delivery, and HIT divergence.

This is reconstructed-source platform evidence only.  It is not equivalence
evidence for original retail TH06 and cannot promote a learned policy.

## Implemented contract

`scripts/run_source_platform_differential.py` accepts
`th06-rl-source-action-stream-v1`.  The stream fixes all of:

- difficulty, character, shot type, and Stage;
- the 16-bit initial source RNG seed;
- a positive maximum tick bound;
- auto-shoot state;
- a run-length sequence drawn only from the 18 ordinary movement actions.

The schema cannot represent Bomb and refuses an action stream shorter than the
tick bound.  Both runtimes receive the same generated `actions.txt`.  The
runner uses ordinary pipes without a PTY, owns a marked dedicated Wine prefix,
stops its wineserver, and requires a stable three-second no-process audit after
both wineboot and the trial.

Every report retains the source commit/dirty state, Linux and PE/DLL hashes,
game-data hashes, commands, stdout/stderr, raw traces, exact physical and event
comparisons, a discrete delivery projection, a tolerance ladder, terminal HIT
geometry, and cleanup evidence.  Float tolerance is diagnostic only and never
changes the exact result.

`scripts/export_headless_action_stream.py` records an existing headless policy
prefix into the same schema.  Export requires a complete manifest and matching
transition SHA-256, refuses HIT-continuation data, mixed or discontinuous
scope/sequence, benchmark-forced actions, unknown actions, nonzero Bomb delta,
and a prefix that ends before the requested bound.  It preserves the corpus,
source, and ranker identities as provenance before run-length encoding.

The paired runtime source was clean commit
`2e0c41653c1698667f848986a783b090a90c23fb`.  The tested Linux binary SHA-256
was `d40b98d22b20248f98baa6c12a311be976bca4332ed890b484a9e054fb4fffff`;
the tested PE SHA-256 was
`44da2e76ee07a920d2863ad2fe79b3c79f3028298be63f26a7ce15bd0a412d7b`.

## Experiments

All artifacts below are ignored local evidence under
`artifacts/source-platform-differential/`.

### Plumbing smoke

Stage 6, Lunatic, Reimu-A, source seed 7, 300 `stay` ticks:

- both runs returned zero and terminated at tick 300 with `tick-limit`;
- all 300 parsed physical snapshots matched exactly;
- all 300 event snapshots matched exactly;
- raw files differed only because Linux emitted LF and Win32 emitted CRLF;
- LF-normalized JSON bytes matched;
- the dedicated Wine prefix had no leftover process.

Evidence: `stage6-seed7-stay-smoke-v2/report.json`.

### Stationary first-HIT window

Stage 6, Lunatic, Reimu-A, source seed 7, 1200-tick bound, constant `stay`:

- both runs stopped on the first physical HIT at tick 848;
- all 848 discrete delivery rows matched, including tick, game frame, input,
  RNG seed/generation, lives, bombs, deaths, score, rank, and terminal reason;
- exact physical state first differed at tick 441:
  `enemies[0].hitbox_width` was `18.666666` on Linux and `18.6666667` under
  Wine;
- the first event difference was bullet-birth `vx` at tick 457,
  `0.114039101` versus `0.114038944`;
- physical drift first exceeded `1e-6` at tick 462, `1e-5` at tick 489,
  `1e-4` at tick 592, and `1e-3` at tick 752; every physical value remained
  within `1e-2` through the common terminal;
- both terminals named bullet slot 306, state 1, flags 4, timer 251 at the
  same center and with the same death/Bomb totals; only the recorded angle
  differed (`0.914369345` versus `0.914369583`);
- both runs returned zero and the dedicated Wine prefix was clean after a
  stable post-run audit.

Evidence: `stage6-seed7-stay-first-hit-v2/report.json`.

### Recorded dynamic policy prefix

The exporter froze the first 1200 verified actions from the existing Stage 6
seed-73 headless corpus at source commit `1350819f...`, ranker SHA-256
`2631f33b...`.  The immutable stream contains 153 run-length segments.

When replayed against both current source builds:

- both completed all 1200 ticks with no HIT and a common `tick-limit`;
- all 1200 discrete delivery rows matched exactly;
- exact physical drift again began at tick 441 in the same enemy hitbox-width
  field;
- event drift again began at tick 457;
- drift first exceeded `1e-6` at tick 476, `1e-5` at tick 483, `1e-4` at tick
  495, and `1e-3` at tick 668; it remained within `1e-2` through tick 1200;
- both runs returned zero and cleanup was complete.

Evidence:
`stage6-seed73-frozen-headless-policy-1200-v1/report.json`.

## Decision

These panels found no early action-delivery or RNG divergence.  They did find
repeatable compiler/platform floating-point drift beginning as soon as active
Stage geometry appears, then accumulating in bullet positions.  On the tested
stationary path that drift did not change the first-HIT tick or collision
identity, and on the dynamic prefix it did not change the 1200-tick outcome.

Therefore Linux headless remains useful for fast rejection, deterministic
counterfactual generation, and bounded geometry experiments.  Exact Linux
snapshot identity after tick 440 must not be assumed, and a margin observed
only in Linux headless cannot be promoted.  Candidate intervention regions
must use Wine-reproducible features and be replayed/shadowed in original retail
Wine.  Two action streams do not prove that later dense patterns, alternate
seeds, or original retail preserve the same outcome.

No training is authorized by this result.  The next gate is the
episode-grouped audit of existing frozen-UCB original-retail Wine first-failure
prefixes.

## Retail-anchored supplement

Later work added diagnostic pre-Stage RNG restoration and delayed Shoot to the
source runtime, then reproduced two complete original-retail Wine prefixes in
both source builds.  This confirmed full-prefix discrete source-platform
delivery and matched retail RNG, game state, hazard counts, and player geometry
at `1e-6`, apart from a known dialogue input gap.  It also exposed and fixed an
exporter error that had treated a captured-but-stale frame as published input.

The resulting exact-checkpoint sub10 COW rejected the proposed alternative and
created no residual candidate.  See
`WINE_FIRST_STAGE6_TARGETED_COW_2026-08-10.md` for the stricter contract,
negative result, hashes, and next gate.  This supplement does not change the
original rule: reconstructed source evidence cannot promote a policy.

A later sub31 audit corrected a second exporter assumption: successful input
publication is not proof that original retail sampled the new mask on the next
game frame.  The exporter now uses each coherent target snapshot's observed
input for every observed frame and uses publication only to fill truly
unobserved interior frames.  With this correction the second sub31 source
prefix no longer died at tick 1724 and matched every observed retail input.

The corrected prefix still diverged after a 246-frame dialogue gap: source RNG
and timeline first differed at retail frame 4675 and both Linux and MinGW
source builds physically HIT at tick 5170, well before retail's frame-6481
authority failure.  Linux and MinGW retained exact discrete delivery and the
same terminal, so this is not a Linux-versus-Wine source-platform disagreement.
The missing variable is the original-retail Ctrl/Shoot edge sequence inside
the old corpus's dialogue gap.  Future frame-v5 Wine corpus records that tiny
delivery stream separately from movement and learning.  See
`WINE_FIRST_STAGE6_LATE_FAILURE_AUDIT_2026-08-10.md`.

## Reproduction

```bash
PYTHONPATH=src python3 scripts/export_headless_action_stream.py \
  artifacts/headless-dagger-1350819-stage6-prehit-v2-r5-unique/20260809T040211Z-dagger-d3-c0-s0-stage6-seed73 \
  --max-ticks 1200 \
  --output artifacts/source-platform-differential-inputs/stage6-seed73-frozen-headless-policy-1200.json

PYTHONPATH=src python3 scripts/run_source_platform_differential.py \
  --action-stream artifacts/source-platform-differential-inputs/stage6-seed73-frozen-headless-policy-1200.json \
  --output artifacts/source-platform-differential/stage6-seed73-frozen-headless-policy-1200-v1
```

Use a new empty output directory for every run.  Do not point the runner at a
retail or otherwise shared Wine prefix; unmarked non-dedicated prefixes are
refused.
