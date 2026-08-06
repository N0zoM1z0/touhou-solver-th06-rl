# TH06-RL Working Rules

Read `START_HERE.md` before changing code. The authoritative source clone is
the ignored sibling checkout at
`../th06/reference/GensokyoClub-th06/`. Source and shipped-game claims must be
traceable to it. Do not use REA, REA-provided tools, or LeanToken.

## Product boundary

The first product is a phase-agnostic reactive baseline that can dodge ordinary
TH06 patterns without learning. Imitation and RL are later ranking layers; they
must never own collision authority.

The runtime boundary is fixed:

1. capture one coherent physical snapshot;
2. construct the authoritative safe first-action set;
3. let the short-horizon local planner rank only that set;
4. recapture/revalidate immediately before input publication;
5. publish one action, with Bomb bit `0x02` forbidden in every mode.

The planner may not weaken geometry, uncertainty, delivery coverage,
fail-close behavior, or the fresh issue check. Unknown or incoherent hazards
fail closed.

## No scripted play

Do not add movement branches keyed to a captured frame, RNG seed,
counterexample identity, boss name, or hand-authored phase state. Source phase
identity may be derived automatically for corpus partitioning, episode
boundaries, evaluation, and later model conditioning; it must not select a
handwritten movement script.

Difficulty, character, shot type, stage, and automatically derived source
context are separate learning scopes. Data and models never silently mix
across those keys.

## Scope restraint

Port only the useful TH08 local-planning ideas: short receding-horizon search,
state deduplication, collision/clearance-first ranking, maneuver reserve,
boundary dead-end cost, and movement hysteresis. Do not port TH08's global
corridor machinery, stage scripts, plugin framework, event bus, or policy
service.

The old `../th06` and `../th08` trees are read-only donors. Copy no game,
authoritative source clone, DAT archive, corpus, trace, log, cache, binary, or
generated artifact into this repository.

## Physical-run safety

- Stop on the first HIT, authority failure, or Bomb request by default.
- Menu/dialogue control stays separate from battle movement.
- Never launch the Windows game path through a PTY.
- Release every input, stop the exact trial PID, and check for leftover game,
  controller, or high-CPU processes after each run.
- Physical play is final evidence; offline tests and replay are acceleration.

