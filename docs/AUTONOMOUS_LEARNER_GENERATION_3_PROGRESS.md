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
- Added transition v8 option identity, boundary/conditional propensity,
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

The controller, transition-v8 corpus, and learner now share a game-neutral
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
