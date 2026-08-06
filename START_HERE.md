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
4. Validate Normal / Reimu-A / Stage 1 physically with learning disabled.
5. Record comprehensive compressed observations and raw outcome terms.
6. Add human demonstration and offline learning as optional rankers inside the
   reactive planner's survival-equivalent frontier.

The architecture deliberately has no handwritten phase movement table. A
source context label is metadata, not control flow.

## Repository layout

```text
src/th06_rl/core/       game-independent local planner
src/th06_rl/th06/       narrow TH06 runtime adapter
tests/                  synthetic and recorded focused tests
docs/                   source contracts and physical evidence
```

## Donor policy

- TH08 provides local beam/search ideas, not an architecture to clone.
- The old TH06 tree provides verified capture, source physics, and input
  contracts while they are extracted behind the narrow adapter.
- Any temporary sibling import must be explicit and removable; the planner
  core must not import either donor repository.

