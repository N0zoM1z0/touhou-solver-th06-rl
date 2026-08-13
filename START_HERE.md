# Start here

The repository contains a Wine-only environment/data/safety foundation and no
authorized gameplay learner on `main`. Historical learner generations 1--6
were removed after their terminal audit; their immutable Wine facts remain in
the capability-indexed corpus registry.

Read in this order:

1. `HAND_OFF.md` for the current boundary and exact next work;
2. `AGENTS.md` for safety and scientific rules;
3. `docs/LEARNER_AUDIT_AND_GENERATION7_DECISION.md` for the consolidated
   failure analysis and the frozen Generation-7 research direction;
4. `docs/WINE_ONLY_AUTONOMOUS_LEARNING.md` for the end-to-end contract;
5. `docs/IMMUTABLE_WINE_DATA_PLANE.md` and
   `config/wine_corpus_registry.json` for reusable factual data;
6. `docs/REPOSITORY_PRUNE.md` for the removal/retention boundary;
7. `docs/TRAINING_INFRA_PERFORMANCE.md` for historical performance and the
   failed concurrent-Wine differential;
8. `docs/HARD_EMPTY_SOURCE_AUDIT.md` and `docs/WINE_RETAIL_VALIDATION.md`
   before gameplay-facing work.

## Non-negotiable separation

- Only original Japanese TH06 1.02h under Wine creates transitions, rewards,
  or evaluation outcomes.
- Offline replay may recompute features from observed facts but may not invent
  an unexecuted action's successor.
- Native geometry owns collision authority and the publishable safe set.
- The learner ranks or abstains inside that safe set; Bomb is forbidden.
- Corpus, learner implementation, fitted artifact, and evaluation evidence
  remain separate.
- Training/validation splits use complete physical episodes.
- Offline training may be complex; the immutable online policy must be small,
  bounded, deterministic in runtime cost, and fast.
- Concurrent Wine collection remains disabled until a new differential passes.

## Active layout

```text
src/th06_rl/core/       native-safe movement value objects and planning
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
