# Wine-only autonomous learning

The improvement loop is:

```text
original Wine facts -> immutable episode corpus -> offline fit
                    -> frozen lightweight policy -> original Wine evaluation
```

Online code never updates weights. Collection policies may make predeclared,
propensity-recorded randomized choices inside the observed-shield set. HITs do
not stop a run. Humans do not patch individual stages or patterns after looking
at failures.

The first learner milestone is deliberately modest: prove that a portable
physical observation/history representation can reproduce held-out behavior
better than trivial action-frequency baselines without using forbidden source
identity. That validates episode linkage, features, splits, targets, and export
before offline RL adds value estimation.

Only after that test passes may the project introduce one advanced component at
a time. Candidate examples are a temporal encoder, object-set encoder, HIT-risk
head, IQL critic/AWR actor, and episode ensemble. Each addition needs a frozen
ablation and must improve held-out episode evidence or complete Wine routes.

Physical HIT count is the only gameplay cost. Auxiliary next-object,
birth/death, occupancy, or hit-horizon predictions are representation losses,
not reward shaping. Final policy evidence is complete natural original-Wine
routes at normal speed, with zero Bomb and infra failures reported separately.

Parallel collection is a throughput tool, not a semantic change. Workers are
isolated copies of an immutable Wine template and must first pass a serial vs
parallel differential for control facts, lifecycle, schemas, and policy output.
Partial failed waves remain quarantined.
