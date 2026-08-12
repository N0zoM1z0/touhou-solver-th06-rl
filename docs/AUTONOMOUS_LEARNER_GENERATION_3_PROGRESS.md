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
- Added transition v7 option identity, boundary/conditional propensity,
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
