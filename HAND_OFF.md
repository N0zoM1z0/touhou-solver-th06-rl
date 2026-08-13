# TH06 offline-RL hand-off

## Current state

`main` is a pruned infrastructure baseline. The active
`generation7-causal-policy-contract` branch implements G7-A and a stopped G7-B
offline investigation. The repository intentionally has no authorized
learner, candidate, canary, or final-evaluation runner. Generations 1--6 failed
to demonstrate a repeatable reduction in complete-Stage physical HIT count;
their algorithm code and scattered experiment contracts were removed. Their
original-Wine corpora were not changed and remain reusable through
`config/wine_corpus_registry.json`.

The terminal learner diagnosis, independent audit, disputed points, corpus
statistics, and Generation-7 decisions are consolidated in
`docs/LEARNER_AUDIT_AND_GENERATION7_DECISION.md`.
G7-A/B implementation results and the current stop decision are in
`docs/GENERATION7_PROGRESS.md`.

## Product objective

Build an autonomous loop in which original retail Wine generates factual
experience, a replaceable offline algorithm learns from every compatible
episode, and a small immutable online policy ranks only the native-safe action
set. Humans repair infrastructure and specify general algorithms; they do not
write routes or tune named stages, spells, frames, RNG seeds, or failure sites.

```text
Wine facts -> immutable corpus -> replaceable offline learner
           -> exact deployable stochastic policy -> Wine shadow/canary/A-B
           -> append only independently authorized training facts
```

## Valid retained foundation

- coherent original-Wine capture, delivery, zero-Bomb enforcement, and exact
  process cleanup;
- source-grounded native collision geometry, Hard-empty fail-close behavior,
  and fresh pre-publication revalidation;
- immutable capability-indexed corpus registration and whole-episode groups;
- factual eight-frame option metadata, complete behavior propensities, and HIT
  conservation in recorded transition schemas;
- adapter-provided observation, action-conditional geometry, bounded hazard
  primitives, and causal short history;
- generic native tree/support/hazard-codebook scoring primitives;
- immutable policy loading, CPU/resource controls, and latency measurement;
- rejection-only complete-block evidence accounting.

The prior fused actor scorer showed that sub-frame native inference is
feasible, but no rejected actor artifact or objective is authorized for reuse.

## Corpus snapshot

The registry contains 71 clean complete Stage runs. Fifty-six episodes have
sequential-offline-RL capabilities: 19 Stage 4, 4 Stage 5, and 33 Stage 6.
The audited replay snapshot contains 167,250 factual options, 2,044 manifest
HITs, and 52,448 factual nonbaseline assignments. Counts are diagnostics, not
hard-coded contracts; code must query capabilities and bind inventory hashes.

The data comes from several behavior policies. A pooled row is therefore not
automatically a sample from one well-defined behavior policy `mu`. Any value or
advantage learner must condition nuisance estimation on source/cohort (and
stage where needed), or explicitly define a shared deployable reference policy.

## Generation-7 status

1. G7-A causal feature, outcome, proper-objective, and exact-policy contracts
   are implemented and tested.
2. Raw IPW signal failed action/reward nulls; cross-fitted orthogonal signal
   passed its exploratory nulls and was directionally stable across source and
   stage.
3. One-step direct and DR agree, but IPS and matched behavior-FQE disagree.
4. Thirty-two-option sequential DR has catastrophic cumulative-weight support
   (minimum fold ESS about 32.5; maximum weight about 1.27e9).
5. Compact and history/hazard bilinear state ablations enlarge direct effects
   without repairing estimator calibration.
6. Proper AWR exists as a bounded challenger, but is not a candidate.
7. Canonical IQL, G7-C collection, Wine shadow/canary/A-B, and deployment are
   stopped and unauthorized.

The next research contract should narrow the estimand and solve calibration
and overlap rather than add model capacity. Do not run original-Wine
outcome-facing experiments merely because an offline metric improves.

## Operational discipline

- Keep repository-wide CPU use at or below 32 and do not overlap canonical
  Wine gameplay with heavy fitting.
- Concurrent Wine collection is currently scientifically unauthorized: the
  fixed-seed serial/concurrent differential changed HIT, frame, and digest
  outcomes. Offline parsing and fitting may parallelize.
- Never delete or commit ignored corpora, artifacts, game binaries, Wine
  prefixes, or the ignored source checkout.
- Commit as `N0zoM1z0 <161784452+N0zoM1z0@users.noreply.github.com>`.
