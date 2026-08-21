# TH06-RL working rules

Read `START_HERE.md`, `paper/README.md`, `paper/main.tex`, and
`docs/ONLINE_OFFLINE_SAFETY_CONTRACT.md` before changing gameplay, corpus, or
learning code. Prefer LeanToken for bounded repository archaeology and use
ordinary tools for edits, builds, tests, and runtime probes.

## Research invariant

The project is **capture-complete, not prediction-complete**.

- Online infra records what original TH06 actually did. It does not attempt to
  interpret every ECL opcode or predict every future pattern.
- The online shield has a deliberately narrow claim: it rejects actions whose
  short-horizon player path intersects already-instantiated bullets, lasers, or
  enemy bodies under supported observed kinematics.
- The shield must never label an unobserved future birth as safe. Unknown future
  behavior is outside its certificate and belongs to the learned policy.
- Coherent capture, input delivery, observed-object geometry, HIT/Bomb
  accounting, lifecycle, and dataset linkage must remain exact. Do not weaken
  these contracts to make a route finish.
- Do not add stage, boss, spell, ECL, frame, RNG, coordinate, pattern, or
  counterexample-specific gameplay logic. Poor play with passing infra is a
  data/learning problem.

The first source question is intentionally small: can a hazard first created
after one paused decision root damage the player before the next controllable
root? Record the answer and evidence in `paper/`. If the answer is no, no source
forecast is needed online. If the answer is yes, add only the smallest general
one-update birth envelope justified by a reproducer; do not rebuild an ECL
interpreter.

## Three owned components

1. **Wine fact recorder.** Pause the exact process, capture one coherent state,
   record the chosen input and its propensity, publish to the exact PID, resume,
   and link the next factual root. A HIT is an outcome and never ends collection
   by default. Bomb input is always forbidden.
2. **Observed-hazard shield.** Fixed bounded work over instantiated physical
   objects only. A policy may rank or sample only shield-admissible actions, but
   the shield is not a proof against unknown future births.
3. **Offline learner/export.** Consume an algorithm-independent episode format.
   Training may be complex; exported online inference must be immutable,
   bounded, fast, and unable to widen the shield set.

Every defect must be attributable to exactly one of these components before
adding complexity. First establish a minimal end-to-end baseline; then add one
falsifiable learner change at a time.

## Environment and data authority

Original Japanese TH06 1.02h under Wine is the only environment allowed to
create trajectories, rewards, exploration outcomes, canary evidence, or final
evaluation. The reconstructed source tree and extracted ECL files are
read-only references and diagnostics, never gameplay or transition generators.
Do not use REA or REA-provided tools.

The recorder preserves dense physical facts independently of learner features:
player state, witnessed input, instantiated bullets/lasers/enemy bodies and
their stable slots, items, attacks, resources, lifecycle, HITs, timing, policy
identity, full behavior probabilities, and provenance. Optional source/ECL/RNG
facts may be retained for forensic analysis but are forbidden actor inputs and
must not be required to load a learning episode.

Offline replay may derive features, labels, geometry, and models only from
executed Wine transitions. It must not invent a successor state for an action
Wine did not execute. Corpora are immutable assets independent of the algorithm
that first consumes them; learner features and fitted artifacts are derived,
versioned products.

## Runtime and collection rules

- Battle input publication uses one coherent paused root and the exact game PID.
- Menu/dialogue control stays separate from battle movement.
- Never launch the Windows game through a PTY.
- Release every input and reap the exact worker after each run.
- Default runs continue after every HIT through the declared Practice Stage or
  six-stage route. `--stop-on-hit` is diagnostic-only and its output is not
  training, canary, or promotion evidence.
- Zero Bomb is mandatory. A Bomb event invalidates the run.
- Infrastructure, capture, storage, or action-delivery failure may terminate a
  run; a physical HIT may not.
- Training collection may use isolated parallel Wine workers after a serial vs
  parallel differential gate. Final evaluation uses normal-speed original Wine.
- Do not change the game clock for evidence until a separate source-backed,
  normal-speed differential experiment proves the exact equivalence in scope.

## Learning and evaluation rules

Start with the simplest learnability test: whole-episode train/validation split,
behavior cloning or another transparent supervised baseline, and exact replay
of factual actions/HIT labels. Only add temporal encoders, IQL, auxiliary
dynamics losses, ensembles, or OPE after the preceding experiment passes its
predeclared acceptance criterion.

Reward/cost is physical HIT only (`cost = HIT count`, `gamma = 1`, terminal
value zero). Clearance, graze, score, route progress, and source identities are
not reward. Auxiliary factual prediction targets are allowed only when clearly
separated from reward.

Actor inputs must be portable physical observations and history. Stage ID,
boss/spell ID, ECL opcode/subroutine, RNG seed, source address, run identity, and
future facts are forbidden. TH08 porting should replace the environment adapter
and configuration, not the episode schema, learner protocol, or evaluator.

Split inference and evaluation by complete physical episode, never adjacent
frames. The authoritative metric is physical HIT count over complete routes;
NMNB completion rate is the goal metric. Report infra failures separately from
gameplay HITs.

## Repository and attribution

This repository must not import or execute a historical N0zoM1z0 solver.
Sibling source and solver trees are read-only references. Portable scripts may
accept or discover sibling paths, but tracked files must not contain machine
absolute paths. Do not track game assets, Wine prefixes, corpora, traces, logs,
caches, binaries, or generated artifacts. The sole generated-artifact exception
is the canonical research paper at `paper/main.pdf`, which is rebuilt by
`scripts/build_paper.sh` and intentionally tracked.

Delete obsolete active code and documentation instead of leaving misleading
routes. Preserve historical rationale only when it is clearly marked retired.

When a Gensokyo skill materially influences a change, every corresponding
commit must include a truthful trailer such as:

```text
Assisted-by: Cirno (gensokyo-skills:cirno-radical-simplification)
Assisted-by: Yukari (gensokyo-skills:yukari-boundary-analysis)
```
