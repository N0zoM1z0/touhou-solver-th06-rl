# Start here

The repository contains a Wine-only environment/data/safety foundation and no
authorized gameplay learner on `main`. Historical learner generations 1--6
were removed after their terminal audit; their immutable Wine facts remain in
the capability-indexed corpus registry.

Read in this order:

1. `HAND_OFF.md` for the current boundary and exact next work;
2. `AGENTS.md` for safety and scientific rules;
3. `docs/PORTABLE_WINE_RUNTIME.md` to provision and smoke-test a new host;
4. `docs/LEARNER_AUDIT_AND_GENERATION7_DECISION.md` for the consolidated
   failure analysis and the frozen Generation-7 research direction;
5. `docs/WINE_ONLY_AUTONOMOUS_LEARNING.md` for the end-to-end contract;
6. `docs/IMMUTABLE_WINE_DATA_PLANE.md` for current source-complete admission;
7. `docs/REPOSITORY_PRUNE.md` for the removal/retention boundary;
8. `docs/TRAINING_INFRA_PERFORMANCE.md` for historical performance and the
   failed concurrent-Wine differential;
9. `docs/WINE_RETAIL_VALIDATION.md` before gameplay-facing work; use
   `scripts/audit_run.py` and `scripts/verify_baseline_route.py` for current
   causal/source-successor audits.

## Non-negotiable separation

- Only original Japanese TH06 1.02h under Wine creates transitions, rewards,
  or evaluation outcomes.
- Offline replay may recompute features from observed facts but may not invent
  an unexecuted action's successor.
- Source-complete bounded hazard lowering plus native geometry own collision
  authority and the publishable safe set; observed-only projection is not a
  complete safety certificate.
- Coherent capture, Hard certification, and input publication share one paused
  physical source epoch; this does not extend the four-frame pickup envelope.
- The learner ranks or abstains inside that safe set; Bomb is forbidden.
- Corpus, learner implementation, fitted artifact, and evaluation evidence
  remain separate.
- Training/validation splits use complete physical episodes.
- Offline training may be complex; the immutable online policy must be small,
  bounded, deterministic in runtime cost, and fast.
- Dense corpus roots preserve factual raw hazard-producer state plus player
  attacks, items, and NMNB resource counters separately from capped learner
  features; features never replace authority evidence.
- Concurrent Wine collection remains disabled until the exact fixed-seed
  serial/two-worker differential passes.

## Active layout

```text
src/th06_rl/core/       native-safe movement value objects and planning
src/th06_rl/retail/     self-contained TH06 process/source-semantics package
src/th06_rl/th06/       original-retail Wine capture/control adapter
src/th06_rl/corpus.py   lossless factual recorder
src/th06_rl/wine_*.py   immutable registry, validation, and worker infra
src/th06_rl/policies/   generic safe exploration primitives only
native/                 bounded geometry and retained scorer primitives
scripts/                environment, replay, audit, and infra commands
tests/                  active contract tests
docs/                   current contracts and consolidated audit evidence
```

Generation 7 is developed on a separate branch. No old generation command is
an available fallback, and ignored artifacts or historical Git commits must
not be restored into the active tree.
