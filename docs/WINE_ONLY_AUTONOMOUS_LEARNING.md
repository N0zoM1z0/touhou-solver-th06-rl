# Wine-only learning loop

The improvement loop is deliberately small:

```text
original Wine facts -> immutable episode corpus -> frozen offline fit
                    -> immutable lightweight policy -> original Wine evaluation
```

Original Japanese TH06 1.02h under Wine is the only environment that may
create transitions, costs, exploration outcomes, or policy evidence. Online
code never updates weights. Collection policies may make only predeclared,
propensity-recorded randomized choices inside the observed-hazard shield set.
HITs do not stop a run, Bomb is forbidden, and humans do not patch individual
stages or patterns after inspecting failures.

## Frozen objective

For a complete, infrastructure-valid, zero-Bomb physical episode, let
`C = sum_t HIT_t`, where each `HIT_t` is an observed physical life loss. The
scientific goal is the complete-route NMNB probability `P(C = 0)`. The first
optimization target is expected undiscounted HIT count `E[C]`, with `gamma = 1`
and terminal value zero only at the declared physical episode terminal.
Reward-maximizing code must use exactly `r_k = -C_k`; cost-form code may
minimize `C_k` directly. Neither convention permits a stage terminal, first-HIT
terminal, discount, or progress-shaped term.

These claims are different. Lower expected HIT count does not by itself prove
higher NMNB probability, although `P(C >= 1) <= E[C]` makes HIT count a useful
dense early surrogate. Clearance, graze, score, power, rank, route progress,
and source identities are not reward. A first-HIT survival target may be a
later diagnostic or candidate objective; it may not truncate the initial task
or discard later HITs. Infrastructure-invalid and incomplete runs are reported
separately and cannot be silently removed from evaluation.

## Causal learner unit

The recorder's immutable factual unit remains a frame-linked physical episode.
The learner's causal unit is a derived decision epoch: one actual policy
invocation to the next invocation or the physical episode terminal. An input
lease can span several raw frames, so treating every frame as a new action
would fabricate interventions.

Each derived row must preserve the starting and successor factual roots,
proposed, published, commanded, sampled, and executed action facts, the full
behavior distribution, elapsed game frames, interval HIT cost, terminal state,
and every exclusion reason. Rows with no publication remain auditable but are
not action-conditioned behavior samples. No learner may invent the successor
of an action Wine did not execute. Interval costs must sum exactly to the
episode HIT total.

## Admission ladder

Only one rung changes at a time. Corpus inventory, whole-episode split, feature
and target schema, seeds, acceptance statistic, and artifact hashes are frozen
before fitting.

1. **L0 — causal audit (completed).** Build and test the decision-epoch view,
   action/propensity reconstruction, HIT conservation, exclusions,
   forbidden-feature rejection, and offline/online feature parity. No model
   result is meaningful before this passes.
2. **L1 — transparent supervised baseline (active).** After L0, predeclare and collect
   the first training inventory serially with unchanged retail clock/update
   semantics, coherent capture suspension, the frozen 20% uniform
   observed-shield mixture, and complete episodes. Freeze its episode
   count, seeds, stopping rule, and split before the first fit. Compare an
   admissible-mask action-frequency control, the reactive control, and a
   current-observation behavior clone. The fitted baseline must beat action
   frequency on held-out complete-episode log loss with a frozen
   episode-bootstrap criterion and acceptable calibration. Accuracy alone is
   insufficient.
3. **L2 — one representation ablation.** Add one fixed short physical/action
   history and simple factual HIT-horizon or observed-shield-collapse probes.
   Retain it only after a preregistered held-out proper-score improvement. A
   bounded object-set encoder is considered only if a transparent vector
   summary fails; recurrence only if short history has already shown value.
4. **L3 — one offline value method.** On the identical frozen data and
   representation, compare BC with one expected-HIT value learner. IQL is the
   first candidate. CQL is admitted only if IQL demonstrates unsupported-action
   overestimation; it is not a parallel hyperparameter arm. Offline evidence
   can reject a candidate but cannot promote it without complete original-Wine
   evaluation.
5. **L4–L6 — evidence-gated alternatives.** Survival/distributional objectives,
   ensembles, OPE, adaptive collection, learned future-birth models, MPC, and
   exact-prefix Wine branching remain inactive until the distinct failure they
   address has been measured. Real-Wine branching additionally requires a
   serial same-seed, same-policy, exact-action-prefix replay gate with identical
   factual and input-delivery digests. Predictions never widen the shield's
   observed-object certificate.

The initial runnable learner therefore has exactly three moving parts: a
deterministic causal dataset builder, a transparent current-observation BC
scorer, and an immutable bounded exporter. The existing online policy context
is not expanded before offline evidence earns a precisely specified portable
feature and a parity test.

## Current evidence

- E0 completed Practice Stages 4–6 and one six-stage route. The route recorded
  108 physical HITs and zero Bomb; this validates infrastructure, not policy
  quality.
- E5's fixed-seed serial/concurrent Stage 4 runs individually completed but
  produced 16, 10, and 13 HITs and different factual digests. Shared-host
  parallel collection remains disabled and those runs are not the L1 training
  inventory.
- E2/L0 passed at commit `2818861f4079bebfbd8443638ed0cb34236bd5e0`.
  Canonical E0 Stage 4 and complete-route replay conserved 18/18 and 108/108
  physical HITs respectively while retaining every excluded decision row.
- E3/L1 is preregistered in `experiments/l1-stage4-bc-v1.json`: 12 serial
  complete natural-RNG Lunatic Practice Stage 4 episodes, eight train and four
  validation, under independent streams of the frozen 20% uniform observed-
  shield mixture. No L1 episode has yet been collected, no research model has
  been fitted, and no learned online candidate has been evaluated.

Auxiliary next-object, birth/death, occupancy, or HIT-horizon predictions may
test representation information, but they are never reward shaping. Final
policy evidence consists of predeclared serial, non-suspending normal-speed,
complete original-Wine routes with zero Bomb; NMNB rate and HIT count are both
reported, and infrastructure failures remain separate.

Parallel collection is only a possible throughput tool. It remains prohibited
until an unchanged serial-versus-parallel differential proves equality of
control facts, lifecycle, schemas, action delivery, and factual traces. Failed
or partial waves remain quarantined.
