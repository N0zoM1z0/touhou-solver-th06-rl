# Start here

The project now has one learning path: original-retail Wine exploration feeds
episode-grouped offline learning, followed by Wine shadow, canary, and
complete-Stage HIT-count evaluation. A resumable runner owns repeated rounds;
people repair infrastructure but do not hand-tune gameplay cases.

Read in this order:

1. `HAND_OFF.md` for the current terminal result, non-negotiable requirements,
   corpus inventory, learner history, and exact next boundary;
2. `AGENTS.md` for safety and product boundaries;
3. `docs/WINE_ONLY_AUTONOMOUS_LEARNING.md` for the data, training, and
   promotion contract;
4. `docs/IMMUTABLE_WINE_DATA_PLANE.md` for the permanent capability-indexed
   corpus registry and the separation of data, learner, and fitted artifact;
5. `docs/AUTONOMOUS_LEARNER_GENERATION_2_DESIGN.md` for the frozen
   generation-2 learner, observation, Hard-empty audit, and evidence contract;
6. `docs/AUTONOMOUS_LEARNER_GENERATION_2_RESULT.md` for its completed Wine
   evidence and ineffective verdict;
7. `docs/AUTONOMOUS_LEARNER_GENERATION_3_DESIGN.md`, its progress record, and
   `docs/AUTONOMOUS_LEARNER_GENERATION_3_RESULT.md` for the frozen superseded
   complete-return learner;
8. `docs/AUTONOMOUS_LEARNER_GENERATION_4_DESIGN.md` for the frozen sequential
   semi-Markov learner, action-centered critic, exploration, and evidence
   contract;
9. `docs/AUTONOMOUS_LEARNER_GENERATION_4_PROGRESS.md` for its append-only
   implementation and execution record, followed by
   `docs/AUTONOMOUS_LEARNER_GENERATION_4_RESULT.md` for its ineffective verdict;
10. `docs/OFFLINE_RL_REFERENCES.md` for the ignored, exactly reproducible paper
   and upstream-repository cache;
11. `docs/AUTONOMOUS_LEARNER_GENERATION_5_DESIGN.md` for the frozen in-sample
   implicit-Q learner and its predeclared smoke/evidence gates, followed by
   `docs/AUTONOMOUS_LEARNER_GENERATION_5_PROGRESS.md` and
   `docs/AUTONOMOUS_LEARNER_GENERATION_5_RESULT.md` for its Stage-4 fail-fast
   result and ineffective-for-continuation verdict;
12. `docs/AUTONOMOUS_LEARNER_QUALIFICATION.md` for the frozen-corpus rejection
   funnel that must pass before another Wine collection wave;
13. `docs/AUTONOMOUS_LEARNER_GENERATION_6_DEVELOPMENT.md` for the IQL actor
   development record, followed by its Stage-6 pilot design and
   `docs/AUTONOMOUS_LEARNER_GENERATION_6_RESULT.md` for the completed positive
   directional result and non-promotion boundary, then
   `docs/AUTONOMOUS_LEARNER_GENERATION_6_ROUND_1.md` for the current frozen
   all-corpus autonomous collection/refit/evidence state machine, and
   `docs/GENERATION6_LATENCY_TAIL_AUDIT.md` for its immutable latency failure,
   controlled reproducer, and generic successor repair, followed by
   `docs/AUTONOMOUS_LEARNER_GENERATION_6_ROUND_2.md` for the separately frozen
   startup-aborted successor and
   `docs/AUTONOMOUS_LEARNER_GENERATION_6_ROUND_3.md` for the fully audited
   current successor, and `docs/GENERATION6_NATIVE_EQUIVALENCE_AUDIT.md` for
   its offline rejection and the required learner-only successor boundary,
   followed by `docs/GENERATION6_DECISION_NUMERIC_SUCCESSOR.md` for the new
   baseline-centred decision-level conformance contract, and
   `docs/GENERATION6_DECISION_GAMEPLAY_RESULT.md` for its clean but
   conclusively ineffective Wine result and the discovered unbounded actor
   objective;
14. `docs/WINE_EXACT_ACCELERATION.md` for normal-speed isolated collection
   parallelism and the compatibility gate;
15. `docs/TRAINING_INFRA_PERFORMANCE.md` for append-only profiles,
   optimizations, failed attempts, and correctness differentials;
16. `docs/HARD_EMPTY_SOURCE_AUDIT.md` for the source-bound Hard-empty verdict
   and conservative-to-source-exact fallback;
17. `docs/WINE_RETAIL_VALIDATION.md` before launching the game.

Generations 1 through 6 are frozen without a promoted candidate. Generation 6
did pass frozen-corpus qualification, native serving, and a small directional
pilot, but that historical promise was superseded by its larger clean Wine
confirmation. After four complete Stage-6 blocks the candidate had 42 HIT
versus the incumbent's 34 and the frozen positive rule was mathematically
impossible. Offline diagnosis then proved that the fitted action-centered
actor ERM objective is unbounded below and had diverged as corpus grew. The
learner is rejected and another Generation-6 refit is forbidden. Its float64
serving, numeric differential, scheduler/PID repairs, and corpus remain valid.
The next step is a separately frozen bounded proper learner on the existing
registry, as specified in `HAND_OFF.md`.

## Non-negotiable separation

- Original retail under Wine creates all gameplay outcomes.
- Offline replay may analyze Wine observations but may not simulate an unseen
  successor action.
- Fixed RNG and accelerated Wine are training diagnostics, not final evidence.
- Normal-speed full-Stage Wine HIT count is the final metric.
- Native geometry owns the safe set; learning only ranks it.
- Bomb is forbidden.
- Poor play means more autonomous Wine learning, not a handwritten exception.
- TH06-specific capture/control stays behind an adapter so the same learner and
  round orchestration can be reused for TH08.

## Repository layout

```text
src/th06_rl/core/       movement and native-safe value objects
src/th06_rl/th06/       original-retail Wine capture and control adapter
src/th06_rl/policies/   immutable incumbent and residual policies
src/th06_rl/corpus.py   lossless Wine trajectory recorder
native/                 bounded geometry and compact model scorers
scripts/                Wine runner, offline replay, fitting, and audits
tests/                  synthetic and recorded contract tests
docs/                   current contracts only
```

Historical headless scripts and ignored artifacts are not an available
learning backend. Do not use them while implementing or evaluating the active
method.
