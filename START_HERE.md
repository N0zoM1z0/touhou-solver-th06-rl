# Start here

`th06-rl` is a clean-room architecture reset for the TH06 agent. It keeps the
trusted source-model/safety boundary, but replaces phase-by-phase movement
scripts and online planning with a small native hazard gate plus learning.

## Build order

1. Define movement/action value objects and a generic reactive fallback.
2. Prove simple dodge, boundary-reserve, and no-legal-action behavior in
   synthetic tests.
3. Add a narrow adapter for TH06 coherent capture, observed-hazard projection,
   native first-action certification, fresh issue certification, and input
   release.
4. Validate the background-capable reactive baseline physically.
5. Record comprehensive compressed observations and raw outcome terms.
6. Learn only inside the native survival frontier, with independent
   difficulty/character/shot/stage/source-phase state.

For Linux-VPS validation against the original retail executable, read
`docs/WINE_RETAIL_VALIDATION.md`. The first complete natural Stage 6-to-1 and
Start-to-Ending measurements are frozen in
`docs/WINE_RETAIL_BASELINE_2026-08-09.md`. Those HIT-continuation runs are
benchmark evidence only and are never training corpus.

The current project state, the distinction between offline fitting,
Linux-headless closed-loop behavior, original-retail Wine validation, and real
Windows evidence, plus the ordered continuation plan are frozen in
`docs/HANDOFF_2026-08-10.md`.

The first real immutable offline-ranker-to-original-retail Wine A/B is recorded
in `docs/WINE_OFFLINE_RANKER_AB_2026-08-10.md`. The adapter works, but the
tested Stage 6 XGBoost selector was not promoted: frozen UCB scored 8/9 HITs
and the offline selector scored 12/9 across two natural trials each.

The current continuation contract is Wine-first and is authoritative over the
older broad-training recommendations in the handoff. Read
`docs/WINE_FIRST_LEARNING_PLAN_2026-08-10.md` before starting another fit. It
records the failed offline assumptions, the episode-grouped data funnel, the
small residual-population rule, and the ordered Stage 6-to-1 promotion gates.

The architecture deliberately has no handwritten phase movement table. A
source context label is metadata, not control flow.

## Repository layout

```text
src/th06_rl/core/       game-independent movement/value objects
src/th06_rl/th06/       narrow TH06 runtime adapter
src/th06_rl/corpus.py   lossless gzip shards and raw transitions
tests/                  synthetic and recorded focused tests
docs/                   source contracts and physical evidence
```

The active collection strata are Lunatic / Reimu-A / Stages 1 through 6. From
Windows, run `run_lunatic_stage123456_learning.bat`; create
`artifacts\pause-lunatic-stage123456` to stop cleanly between stages. Each
non-mastered stage runs in a three-trial block; three latest trustworthy
no-HIT clears mark it mastered and skip it on later cycles. Every stage has
independent policy and corpus scope. Hard remains preserved as non-sharing
baseline evidence only. Pre-hit-credit checkpoints are diagnostic history
only and are not valid hot starts for the active online reward version.

## Donor policy

- TH08 is a geometry reference, not an online planning architecture to clone.
- The old TH06 tree provides verified capture, source physics, and input
  contracts while they are extracted behind the narrow adapter.
- Any temporary sibling import must be explicit and removable; the planner
  core must not import either donor repository.
