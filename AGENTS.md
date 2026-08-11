# TH06-RL working rules

Read `START_HERE.md` and `docs/WINE_ONLY_INTERVENTION_LEARNING.md` before
changing code. The authoritative source clone is the ignored checkout at
`reference/GensokyoClub-th06/`. Source and shipped-game claims must be
traceable to it. Do not use REA, REA-provided tools, or LeanToken.

## Wine-only environment boundary

Original Japanese TH06 1.02h under Wine is the only environment allowed to
create trajectories, rewards, intervention outcomes, counterfactual branch
labels, or promotion evidence. Do not use the reconstructed Linux/headless
runtime for training, evaluation, action proposals, or compatibility claims.
Historical headless code and ignored artifacts are quarantined history.

Offline code may replay observations captured from original-retail Wine,
recompute native geometry, construct features, fit models, and score candidate
policies. Such replay must not invent a successor state for an action that Wine
did not execute.

## Runtime boundary

1. capture one coherent physical Wine snapshot;
2. project already-observed hazards and construct a native safe first-action
   set with fixed bounded work;
3. let an immutable policy rank only that set;
4. recapture and revalidate immediately before input publication;
5. publish one action, with Bomb bit `0x02` forbidden in every mode.

Learning never owns collision authority, uncertainty margins, delivery
coverage, fail-close behavior, or the fresh issue check. Unknown or incoherent
hazards fail closed. Do not interpret ECL births or run tree/beam search in the
resident hot path.

## Learning boundary

The active method is:

`Wine intervention -> episode-grouped offline fit/replay -> small residual -> Wine canary`

Online policy state is immutable. Data collection may make a small number of
predeclared, propensity-recorded randomized choices inside the native-safe set;
it may not update weights. Split training and validation by complete physical
episode or fixed-RNG pair, never by adjacent frame. A residual defaults to the
frozen incumbent outside supported physical features and abstains on model
disagreement.

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
  explicitly marked full-Stage HIT-continuation evaluation.
- Menu/dialogue control stays separate from battle movement.
- Never launch the Windows game through a PTY.
- Release every input, stop the exact trial PID, and check for leftover game,
  controller, display, or high-CPU processes after every run.
- Do not run canonical promotion trials concurrently or in accelerated mode.

The old `../th06` and `../th08` trees are read-only donors. Do not copy game
assets, source clones, corpora, traces, logs, caches, binaries, or generated
artifacts into tracked files.
