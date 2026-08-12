# Autonomous learner generation 3 progress

This is an append-only implementation and execution record for the predeclared
Generation 3 contract. It is not authority to change reward, data distribution,
options, thresholds, evidence seeds, or evaluation rules after outcomes.

## 2026-08-12: correctness and option treatment

- Corrected the Generation 2 Bellman export off-by-one and replaced its
  mismatched 60-frame validation comparison with matched-horizon targets.
- Corrected online selection to examine every supported action and to apply the
  declared uncertainty scale.
- Added eight-physical-frame safe options with fresh native certification on
  every publication, including Windows input-lease frames.
- Added transition v9 option identity, boundary/conditional propensity,
  physical elapsed time, and termination accounting.

## 2026-08-12: learner population and causal smoke

Generation 3 now constructs undiscounted option-boundary HIT returns, produces
multi-action AIPW/doubly robust advantages with three whole-episode cross-fit
folds and three nuisance members, and fits seven whole-episode-bootstrap
population members. The baseline pseudo-advantage is identically zero.

The deterministic pre-collection causal smoke was executed with the production
tree counts. Its known candidate advantage was -1 HIT while an independent
state-risk feature changed between two levels. Result:

- all seven members predicted a negative advantage at both risk levels;
- population prediction range: -1.0334 to -0.9656 HIT;
- aggregate prediction mean: -0.9990 HIT;
- maximum state-risk leakage: 0.00166 HIT;
- held-out DR-advantage RMSE: 0.0587;
- held-out zero-advantage comparator RMSE: 0.7287;
- all predeclared causal-smoke gates passed.

This proves the deterministic learner path can recover the requested relative
effect without confusing it with common state risk. It is not Wine gameplay
evidence. The short non-evidence Wine option-pipeline smoke remains pending
until the Generation 3 runtime scorer and runner are connected.

Repository tests after this checkpoint: 371 passed.

## 2026-08-12: learned hazard set and history representation

The controller, transition-v9 corpus, and learner now share a game-neutral
bounded representation: up to 256 player-relative observed hazard primitives,
four factual observation/action records, and a 24-prototype codebook learned
only from training episodes. Set pooling is permutation invariant. The adapter
does not expose source phase, ECL, spell, RNG, frame windows, flags, or slots.

The production causal smoke was rerun after adding the learned representation:

- aggregate known-effect prediction: -0.9957 HIT for a true -1 HIT effect;
- population prediction range: 0.0409 HIT;
- maximum state-risk leakage: 0.00295 HIT;
- held-out DR-advantage RMSE: 0.0377;
- held-out zero-advantage comparator RMSE: 0.7202;
- all causal-smoke gates passed.

Repository tests after this checkpoint: 374 passed.

## 2026-08-12: population-preserving native deployment smoke

Each of the seven 128-tree teacher members is now distilled separately into a
48-tree depth-four student. Runtime selection uses the maximum member advantage
plus a whole-episode one-sided conformal residual; it does not use a selected
winner or population mean. Unobserved factual actions and locally unsupported
states fail closed to the incumbent.

The host native scorer, support kernel, and new bounded native hazard-codebook
encoder compiled successfully. A production-sized deterministic state was
loaded through the complete native runtime and evaluated for 1,200 decisions:

- native decision p95: 0.440 ms;
- native decision maximum: 0.613 ms;
- decisions over 4 ms: 0;
- controller deadline misses: 0;
- held-out distillation p95 absolute error: 0.0298 HIT;
- held-out distillation maximum absolute error: 0.0321 HIT;
- one-sided conformal radius on the fixture: 0.00173 HIT.

The host and fully static Win32 scorer libraries both compile after adding the
encoder. These are deterministic infrastructure smokes, not Wine outcome
evidence.

Repository tests after this checkpoint: 377 passed.

## 2026-08-12: fail-fast preflight contract

The fixed collection/canary seed schedule is now a tracked immutable artifact.
Long collection is gated by a causal learner smoke and a 45-second retail-Wine
option wiring smoke. The latter has a dedicated time-bounded corpus mode,
patched HIT continuation, fixed diagnostic RNG, and an explicit non-evidence
label. Its rows cannot enter fitting or effect evaluation.

The first real smoke correctly rejected the pipeline before long collection:
the corpus had conflated `published_action` (a new key-delivery event) with the
physical action that remained held during a stale retry. Generation 3 now uses
transition v9 with an explicit factual `executed_action`. A tentative option
whose intent was not physically executed is marked `publication-rejected` and
cannot become a learner sample; a held, observed matching action remains
factual. This is a label/delivery infrastructure repair backed by the smoke,
not an outcome-driven change to collection or reward.

Every fit is now followed by an exact held-out native shadow replay. It loads
the complete seven-member population and learned hazard encoder through the
host native library, replays every factual held-out option boundary, asserts
baseline-only publication in shadow mode, and gates p95 latency plus deadline
misses. Active canary state is hash-bound to that exact clean audit. The host
and Win32 scorer hashes are both bound at fit time; this is binary portability,
not a second policy.

The crash-resumable Generation-3 runner now owns the complete fixed protocol:
preflight, up to 24 fixed-RNG collection Stages, fits at 12/16/20/24, exact
held-out native shadow, three paired fixed-RNG canaries per fit, and—only after
authorization—12 natural-RNG complete Stages per arm in alternating order.
Its CLI deliberately exposes no gameplay, reward, seed, threshold, round-size,
canary-count, or evaluation-count knobs. Natural evaluation always runs all 24
trials unless infrastructure itself fails, and reports aggregate physical HITs,
candidate exercise, total overrides, and an approximate HIT-rate-ratio interval.

## 2026-08-12: real Wine preflight passed and collection launched

The hash-bound committed preflight passed before the first evidence Stage:

- causal fixture prediction mean: -0.9690 HIT for a true -1 HIT effect;
- held-out DR RMSE: 0.0536 versus zero-advantage RMSE 0.7202;
- all seven population members recovered the negative effect with zero measured
  state-risk leakage;
- 45-second Wine smoke: 2,078 option rows, 1,441 factually executed rows,
  637 explicitly rejected rows, 405 factual boundaries, 21 non-incumbent
  boundaries, 93 horizon terminations, and five input-lease rows;
- every causal and Wine wiring gate passed; the smoke remains non-evidence.

The unattended fixed experiment then launched. The first two complete
fixed-RNG collection Stages passed full corpus/HIT/infrastructure validation:
episode 0 recorded 38 physical HITs and episode 1 recorded 32. These are
training outcomes only, not an interim efficacy comparison and not authority
to alter the committed schedule. Collection continues toward the first
12-episode grouped fit.
