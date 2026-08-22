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

For a preregistered multi-root randomized intention, the raw frame and
transition additionally retain a run-local group ID, step, horizon, intended
action, initial assignment propensity/distribution, and override reason. These
are intervention provenance, not actor observations. The per-root behavior
distribution remains the exact conditional publication distribution. A
decision group never erases commanded, sampled, or executed action evidence.

## Admission ladder

Only one rung changes at a time. Corpus inventory, whole-episode split, feature
and target schema, seeds, acceptance statistic, and artifact hashes are frozen
before fitting.

1. **L0 — causal audit (completed).** Build and test the decision-epoch view,
   action/propensity reconstruction, HIT conservation, exclusions,
   forbidden-feature rejection, and offline/online feature parity. No model
   result is meaningful before this passes.
2. **L1 — transparent supervised baseline (completed negative).** The frozen
   serial inventory contains 12 complete episodes split eight/four before
   collection. The current-observation BC beat action frequency on held-out
   complete-episode log loss, but failed the separately frozen calibration
   criterion. It was not exported to Wine and the withheld canary was not run.
   A refit or calibration method is a new preregistered L1 experiment, not a
   reinterpretation of this result.
   The read-only diagnosis reproduced the failure and found systematic
   under-confidence: validation accuracy was 0.461290 while mean top
   confidence was 0.337672, and the frozen train gradient remained 19.1% of
   its initial norm. Portable roots reproduced every recorded reactive
   baseline, so missing capture is not the failure. L1b reused the exact
   corpus/split and changed only train-only optimization convergence plus a
   corrected ECE implementation. It exhausted the frozen 2,000-update timebox
   at gradient ratio 0.0196 rather than 0.01, so its result is inconclusive as
   collector BC. The separately gated factual-probe line nevertheless
   established current-root h16 physical-HIT predictability without admitting
   a policy.
3. **L2 — factual hazard and action exposure (active collection boundary).**
   Current scalar, weighted, fixed-history, observed-primitive, and log-score
   hazard candidates were tested and rejected by their frozen joint gates.
   BCE-with-logits repaired almost all saturated missed positives but did not
   stabilize object or exploratory-action gain. A factual audit then found
   that 99.45% of h16 positives occurred after the initially published action
   ended. The label is valid behavior-continuation risk but unsupported as
   current-action value. Learner sweeps on that target are stopped. L2k now
   preregisters a four-policy-root uniformly randomized intention with a fresh
   observed-shield check at every root, exact assignment/override provenance,
   and two serial complete Stage 4 pilot-train episodes. Recurrence and value
   learning remain blocked.
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

The active runnable experiment has exactly three moving parts: a uniform
four-root intention policy, the unchanged per-root observed shield/controller,
and a versioned factual exposure recorder/auditor. It fits no learner. The
online actor observation is unchanged; exposure identifiers and propensities
are recorder provenance only.

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
- E3/L1 completed at commit `559572a8f3699c06eb41080e6061e579a1156c33`.
  All 12 serial complete natural-RNG Lunatic Practice Stage 4 episodes passed
  admission on attempt 1, yielding 193,331 training and 94,162 validation
  rows. Validation BC NLL was 2.014976 versus 2.286394 for action frequency;
  the episode-bootstrap delta interval was `[-0.287520, -0.254559]`. Validation
  ECE was 0.124251 versus the admissible bound 0.028749, so the joint gate
  failed. The result is `stop-l1-bc-learnability`; no Wine canary was run and
  there is no authorized learned candidate.
- The post-hoc reliability audit found validation mean top confidence 0.337672
  against accuracy 0.461290 in all four held-out episodes. The model recovered
  the deterministic reactive baseline from the permitted facts on only 56.19%
  of rows, although a diagnostic replay of those same facts recovered the
  recorded baseline on 100%. The 100-update fit was not stationary: final
  train gradient L2 was 19.1% of its initial value and increasing logit scale
  locally reduced train NLL. A floating ECE edge omitted exact confidence 0.6,
  but corrected binning changed the comparator ECE by only 0.000017 and left
  the L1 decision negative.
- L1b ran from commit `7285ee76fe36eda1470844b3635eaddd64292d23` and
  reused all 12 episode identities and the eight/four split. It reached all
  2,000 full-batch updates with final/initial train gradient ratio 0.019566,
  missing the frozen 0.01 threshold. Validation NLL, Brier, accuracy, and ECE
  improved to 1.720755, 0.647767, 0.528111, and 0.055148 respectively. The
  corrected action-frequency ECE was 0.008732, so calibration still exceeded
  the 0.028732 bound. Proper-score passed, optimization and calibration failed,
  and the exact decision is `inconclusive-l1b-optimization-not-converged`.
  L1b performed no Wine run and created no authorized learned candidate.
- L2h ran from commit `bbf6e81a3203fb00443060a215949a37fdb13352` on
  exact unit-frame 16-root histories. History-full Brier was 0.00479407 versus
  0.00483026 for same-row current-only, but the temporal interval crossed zero
  with five of eight favorable episodes. Its overall current-action ablation
  passed, while low-propensity, nonbaseline, and pre-first-HIT intervals crossed
  zero and the raw surface clipped 2.43%. The immutable decision is
  `reject-fixed-history-h16-hazard`; no fresh confirmation or Wine run occurred.
- L2i and L2j both failed their frozen joint gates. L2j improved object-full
  Brier from 0.00473906 to 0.00464423 and reduced saturated missed positives
  from 89 to one, but scalar-only remained better and exploratory/lifecycle
  action gates did not stabilize. The immutable L2j decision is
  `reject-logscore-observed-primitive-h16-hazard`.
- The subsequent action/target audit found median initial action duration one
  frame and 1,079/1,085 h16 positives after that action ended. L2k is frozen but
  not run: two serial complete Stage 4 pilot-train episodes test uniform
  four-root action intentions, exact per-root shield/override behavior, and
  action-exposure schema integrity. No model or learned policy runs.

Auxiliary next-object, birth/death, occupancy, or HIT-horizon predictions may
test representation information, but they are never reward shaping. Final
policy evidence consists of predeclared serial, non-suspending normal-speed,
complete original-Wine routes with zero Bomb; NMNB rate and HIT count are both
reported, and infrastructure failures remain separate.

The stateless uniform-mixture collector already passed its declared parallel
infrastructure gate. L2k nevertheless starts serially because its policy adds
stateful multi-root semantics. Scaled L2k collection remains prohibited until
a separate stateful serial-versus-parallel differential proves equality of
control facts, group lifecycle, schemas, action delivery, and factual traces.
Failed or partial waves remain quarantined.
