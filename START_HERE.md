# Start here

The project now has one learning path: original-retail Wine exploration feeds
episode-grouped offline learning, followed by Wine shadow, canary, and
complete-Stage HIT-count evaluation. A resumable runner owns repeated rounds;
people repair infrastructure but do not hand-tune gameplay cases.

Read in this order:

1. `AGENTS.md` for safety and product boundaries;
2. `docs/WINE_ONLY_AUTONOMOUS_LEARNING.md` for the data, training, and
   promotion contract;
3. `docs/IMMUTABLE_WINE_DATA_PLANE.md` for the permanent capability-indexed
   corpus registry and the separation of data, learner, and fitted artifact;
4. `docs/AUTONOMOUS_LEARNER_GENERATION_2_DESIGN.md` for the frozen
   generation-2 learner, observation, Hard-empty audit, and evidence contract;
5. `docs/AUTONOMOUS_LEARNER_GENERATION_2_RESULT.md` for its completed Wine
   evidence and ineffective verdict;
6. `docs/AUTONOMOUS_LEARNER_GENERATION_3_DESIGN.md`, its progress record, and
   `docs/AUTONOMOUS_LEARNER_GENERATION_3_RESULT.md` for the frozen superseded
   complete-return learner;
7. `docs/AUTONOMOUS_LEARNER_GENERATION_4_DESIGN.md` for the frozen sequential
   semi-Markov learner, action-centered critic, exploration, and evidence
   contract;
8. `docs/AUTONOMOUS_LEARNER_GENERATION_4_PROGRESS.md` for its append-only
   implementation and execution record, followed by
   `docs/AUTONOMOUS_LEARNER_GENERATION_4_RESULT.md` for its ineffective verdict;
9. `docs/OFFLINE_RL_REFERENCES.md` for the ignored, exactly reproducible paper
   and upstream-repository cache;
10. `docs/AUTONOMOUS_LEARNER_GENERATION_5_DESIGN.md` for the frozen in-sample
   implicit-Q learner and its predeclared smoke/evidence gates, followed by
   `docs/AUTONOMOUS_LEARNER_GENERATION_5_PROGRESS.md` and
   `docs/AUTONOMOUS_LEARNER_GENERATION_5_RESULT.md` for its Stage-4 fail-fast
   result and ineffective-for-continuation verdict;
11. `docs/AUTONOMOUS_LEARNER_QUALIFICATION.md` for the frozen-corpus rejection
   funnel that must pass before another Wine collection wave;
12. `docs/AUTONOMOUS_LEARNER_GENERATION_6_DEVELOPMENT.md` for the IQL actor
   development record, followed by its Stage-6 pilot design and
   `docs/AUTONOMOUS_LEARNER_GENERATION_6_RESULT.md` for the completed positive
   directional result and non-promotion boundary;
13. `docs/WINE_EXACT_ACCELERATION.md` for normal-speed isolated collection
   parallelism and the compatibility gate;
14. `docs/TRAINING_INFRA_PERFORMANCE.md` for append-only profiles,
   optimizations, failed attempts, and correctness differentials;
15. `docs/HARD_EMPTY_SOURCE_AUDIT.md` for the source-bound Hard-empty verdict
   and conservative-to-source-exact fallback;
16. `docs/WINE_RETAIL_VALIDATION.md` before launching the game.

Generations 3, 4, and 5 are frozen. Generation 6 passed frozen-corpus
qualification, native serving, and a six-run original-Wine Stage-6 directional
pilot: candidate aggregate HIT was 25 versus incumbent 28. The small panel is
promising but not promotion evidence; a larger confirmation or autonomous
learning round must be separately frozen before more outcome-facing play.

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
