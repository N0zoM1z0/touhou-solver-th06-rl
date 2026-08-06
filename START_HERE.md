# Start here

`th06-rl` is a clean-room architecture reset for the TH06 agent. It keeps the
trusted source-model/safety boundary, but replaces phase-by-phase movement
scripts with one small reactive local planner.

## Build order

1. Define dependency-free planner value objects and a deterministic beam.
2. Prove simple dodge, boundary-reserve, and no-legal-action behavior in
   synthetic tests.
3. Add a narrow adapter for TH06 coherent capture, source-grounded hazard
   projection, Hard first-action certification, fresh issue certification,
   and input release.
4. Validate the background-capable reactive baseline physically.
5. Record comprehensive compressed observations and raw outcome terms.
6. Learn only inside the native/local survival frontier, with independent
   difficulty/character/shot/stage/source-phase state.

The architecture deliberately has no handwritten phase movement table. A
source context label is metadata, not control flow.

## Repository layout

```text
src/th06_rl/core/       game-independent local planner
src/th06_rl/th06/       narrow TH06 runtime adapter
src/th06_rl/corpus.py   lossless gzip shards and raw transitions
tests/                  synthetic and recorded focused tests
docs/                   source contracts and physical evidence
```

The active collection stratum is Hard / Reimu-A / Stage 4. From Windows, run
`run_hard_stage4_learning.bat`; create
`artifacts\pause-hard-stage4` to stop cleanly between trials.

## Donor policy

- TH08 provides local beam/search ideas, not an architecture to clone.
- The old TH06 tree provides verified capture, source physics, and input
  contracts while they are extracted behind the narrow adapter.
- Any temporary sibling import must be explicit and removable; the planner
  core must not import either donor repository.
