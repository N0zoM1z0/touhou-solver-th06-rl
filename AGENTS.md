# TH06-RL working rules

Read `START_HERE.md` and `docs/WINE_ONLY_AUTONOMOUS_LEARNING.md` before
changing code. The authoritative source clone is the ignored checkout at
`reference/GensokyoClub-th06/`. Source and shipped-game claims must be
traceable to it. Do not use REA, REA-provided tools, or LeanToken.

## Wine-only environment boundary

Original Japanese TH06 1.02h under Wine is the only environment allowed to
create trajectories, rewards, exploration outcomes, counterfactual branch
labels, or promotion evidence. Do not use the reconstructed Linux/headless
runtime for training, evaluation, action proposals, or compatibility claims.
Tracked headless code has been removed. Historical commits and ignored
artifacts are quarantine history, not an available backend; do not restore
them into the active tree.

Offline code may replay observations captured from original-retail Wine,
recompute native geometry, construct features, fit models, and score candidate
policies. Such replay must not invent a successor state for an action that Wine
did not execute.

Dense physical frames retain the raw state needed to reconstruct collision
geometry and hazard-producer state plus factual player attacks, items, and
NMNB resource counters independently of learner features. Any intentional
omission must be schema-visible and must not be described as a lossless
full-game state. Low-frequency authoritative anchors retain immutable
stage/ECL graphs; compact roots retain factual occupied bullet, laser, enemy,
manager, player-attack, item, and resource state. Derived, capped, or lossy
feature tensors are never audit authority.

## Runtime boundary

1. pause the exact Wine process and capture one coherent physical snapshot;
2. construct a source-complete Hard-horizon collision envelope and a native
   safe first-action set with fixed bounded work;
3. let an immutable policy rank only that set;
4. revalidate and publish against that same paused source epoch;
5. resume only after publishing one action, with Bomb bit `0x02` forbidden in
   every mode.

Learning never owns collision authority, uncertainty margins, delivery
coverage, fail-close behavior, source commitments, or the fresh issue check.
The envelope must include already-observed hazards plus every retail birth,
body mutation, clamp, and laser mutation possible before the input lease
expires. Unknown, incoherent, or uncovered source state fails closed. A small,
bounded, source-verified commitment evaluator belongs to the environment
adapter; unbounded ECL interpretation and tree/beam search do not belong in the
resident hot path.

## Learning boundary

The active method is:

`Wine exploration -> grouped offline learning -> immutable candidate -> Wine canary`

Online policy state is immutable. Data collection may make predeclared,
propensity-recorded randomized choices inside the native-safe set; it may not
update weights. Split training and validation by complete physical episode,
never by adjacent frame. A learned policy defaults to the frozen incumbent
outside supported physical features and abstains on model disagreement.

Failed Generation-1--6 learners and runners have been pruned from the active
tree. Do not restore them from Git history or ignored artifacts. The terminal
action-centered actor objective was unbounded below under empirical
optimization; its proof and the successor boundary are recorded in
`docs/LEARNER_AUDIT_AND_GENERATION7_DECISION.md`. Every successor must use a
bounded proper objective and pass an extreme-logit anti-exploitation smoke
before Wine gameplay.

## Autonomous-learning boundary

Gameplay improvement belongs to the fixed learning algorithm and repeated
Wine data rounds, not to human case-by-case policy edits. Do not tune collection
eligibility, reward terms, feature thresholds, activation regions, or action
preferences after inspecting a failure location. Do not add stage, boss, spell,
frame-window, RNG-seed, bullet-pattern, or counterexample-specific logic.

Humans may change gameplay-facing code only to repair a demonstrated
infrastructure defect: incoherent capture, incorrect memory semantics, action
delivery, native geometry/safety, factual label alignment, process isolation,
or evaluation accounting. Every such repair needs a reproducer and a contract
test. If contracts pass and play is poor, collect more Wine experience and let
the unchanged learner update. The unattended round runner, not a human, decides
when to fit, shadow, canary, evaluate, continue collecting, or stop at a
predeclared evidence limit.

The learning interface must be game-agnostic: observations, native-safe action
sets, chosen-action propensities, transitions, episode groups, rewards, and HIT
outcomes. TH06-specific memory and input details stay in the environment
adapter. Porting to TH08 should replace that adapter and configuration, not the
dataset, fitting, validation, or orchestration algorithm.

RNG control, accelerated Wine, snapshots, and parallel workers are diagnostic
or training accelerators only. Final comparison uses normal-speed original
Wine, natural full Practice Stages, HIT continuation, zero Bomb, immutable
policies, and alternating incumbent/candidate trials. The per-run physical HIT
count is authoritative.

## No scripted play

Do not key movement to a frame number, RNG seed, run ID, counterexample,
boss name, or handwritten phase state. Difficulty, character, shot type,
stage, and automatically derived source context are separate scopes. Source
context may partition evidence but may not select a handwritten route.

## Physical-run safety

- Stop on the first HIT, authority failure, or Bomb request unless running an
  explicitly marked complete-Stage HIT-continuation training or evaluation.
- Menu/dialogue control stays separate from battle movement.
- Never launch the Windows game through a PTY.
- Release every input, stop the exact trial PID, and check for leftover game,
  controller, display, or high-CPU processes after every run.
- Do not run canonical promotion trials concurrently or in accelerated mode.

## Gensokyo skill attribution

When a Gensokyo skill materially influences a change, attribute it in every
corresponding commit with an `Assisted-by:` trailer naming both the character
and skill. Keep the trailer specific to the help actually used; do not add it
to unrelated commits. Example:

```text
Assisted-by: Nitori (gensokyo-skills:nitori-reverse-engineering)
```

Sibling source and historical solver trees are read-only references, never
runtime or import dependencies. Portable setup scripts may discover a sibling
checkout as a convenience, but must accept an explicit path and must not bake
absolute workspace paths into tracked files. Do not copy game assets, source
clones, corpora, traces, logs, caches, binaries, or generated artifacts into
tracked files.
