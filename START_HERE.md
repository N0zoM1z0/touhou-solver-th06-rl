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
The first paired reconstructed-source Linux/MinGW-under-Wine action-stream
result is recorded in
`docs/SOURCE_PLATFORM_DIFFERENTIAL_STAGE6_2026-08-10.md`; it found exact
discrete delivery but accumulating floating-point geometry drift.
The next episode-grouped original-retail audit is in
`docs/WINE_FAILURE_REGION_AUDIT_STAGE6_2026-08-10.md`; it is the authority for
which repeated Stage 6 regions may enter targeted COW.
The first Wine-anchored targeted COW result is recorded in
`docs/WINE_FIRST_STAGE6_TARGETED_COW_2026-08-10.md`.  Deterministic retail
replay is now validated, but the proposed sub10 `right_fast` to `left_fast`
residual was rejected and no candidate was created.  The completed late-family
audit is in `docs/WINE_FIRST_STAGE6_LATE_FAILURE_AUDIT_2026-08-10.md`: sub31
favored the incumbent at its one exact checkpoint, while a second sub31 anchor
and both repeated sub18 anchors exposed missing retail dialogue-delivery edges
in the old corpus.  All three Stage 6 families currently have zero residual
candidates.  Collect a small new frozen-UCB Wine first-failure panel with frame
schema v5 dialogue-delivery evidence; do not resume broad training.
The completed three-episode v5 panel and its corrected exact-delivery
differential are in
`docs/WINE_FIRST_STAGE6_FRAMEV5_PANEL_2026-08-10.md`.  All 238 unique retained
dialogue-input frames in run one matched both source domains, as did 239/239
and 238/238 in runs two and three.  Run one retained a pre-dialogue MinGW RNG
divergence at frame 1206; the other two matched retail discrete state through
their full prefixes.  The panel found and fixed a COW audit bug that confused
Wine delivery delays `(0,1,2,3)` with synchronous source STEP `(0,)`.  Corrected
targeted COW still produced disagreement under the robust rank, so the panel
has zero residual candidates.  Do not fit or enter shadow from it.
The second predeclared three-run result is recorded in
`docs/WINE_FIRST_STAGE6_PANEL2_RESULT_2026-08-10.md`. All three frozen-UCB
prefixes stopped 0-HIT at a sub31 native authority failure. Their only new
cross-episode residual pair strongly favored the incumbent at its first exact
COW checkpoint, so panel 2 also has zero candidates. Exact dialogue delivery
and RNG matched both source domains, yet both source domains temporarily had
zero bullets where retail had 289--308. The resulting geometry audit is in
`docs/WINE_FIRST_STAGE6_ENEMY_MOTION_AUDIT_2026-08-10.md`: the old source trace
omitted enemy interpolation state and produced 24/33/26 false native-set
differences. After the additive source fix, all 19,120 same-frame native sets
matched retail in each source domain, including every terminal window. The
corrected COW still favored the incumbent and creates no candidate. This is a
local reject-only repair, not proof of whole-runtime equivalence. Complete
natural original-retail Wine Stage HIT count remains the final promotion
metric; stopped 0-HIT prefixes and headless COW are filters only.
The next one-shot exploratory COW is predeclared in
`docs/WINE_FIRST_STAGE6_FIRST_ACTION_SCAN_PROTOCOL_2026-08-10.md`; its fixed
robust selection rule must not be changed after discovery.
That scan is complete in
`docs/WINE_FIRST_STAGE6_FIRST_ACTION_SCAN_RESULT_2026-08-10.md`:
`down_right` was the unique robust winner across all 14 retail-native-safe
first actions, so the conditional r6 confirmation was not run and the result
creates zero candidates. Do not mine another Panel-2 frame. The next audit is
whether source COW can restore and advance the immutable frozen-UCB state after
one substituted action; the current offline-teacher continuation is reject-only.

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
