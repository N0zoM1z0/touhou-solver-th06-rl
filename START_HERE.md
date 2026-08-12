# Start here

The project now has one learning path: original-retail Wine exploration feeds
episode-grouped offline learning, followed by Wine shadow, canary, and
complete-Stage HIT-count evaluation. A resumable runner owns repeated rounds;
people repair infrastructure but do not hand-tune gameplay cases.

Read in this order:

1. `AGENTS.md` for safety and product boundaries;
2. `docs/WINE_ONLY_AUTONOMOUS_LEARNING.md` for the data, training, and
   promotion contract;
3. `docs/AUTONOMOUS_LEARNER_GENERATION_2_DESIGN.md` for the frozen
   generation-2 learner, observation, Hard-empty audit, and evidence contract;
4. `docs/AUTONOMOUS_LEARNER_GENERATION_2_RESULT.md` for its completed Wine
   evidence and ineffective verdict;
5. `docs/AUTONOMOUS_LEARNER_GENERATION_3_DESIGN.md`, its progress record, and
   `docs/AUTONOMOUS_LEARNER_GENERATION_3_RESULT.md` for the frozen superseded
   complete-return learner;
6. `docs/AUTONOMOUS_LEARNER_GENERATION_4_DESIGN.md` for the frozen sequential
   semi-Markov learner, action-centered critic, exploration, and evidence
   contract;
7. `docs/AUTONOMOUS_LEARNER_GENERATION_4_PROGRESS.md` for its append-only
   implementation and execution record, followed by
   `docs/AUTONOMOUS_LEARNER_GENERATION_4_RESULT.md` for its ineffective verdict;
8. `docs/OFFLINE_RL_REFERENCES.md` for the ignored, exactly reproducible paper
   and upstream-repository cache;
9. `docs/AUTONOMOUS_LEARNER_GENERATION_5_DESIGN.md` for the active in-sample
   implicit-Q learner and its predeclared smoke/evidence gates, followed by
   `docs/AUTONOMOUS_LEARNER_GENERATION_5_PROGRESS.md` for append-only execution
   and estimator findings;
10. `docs/WINE_EXACT_ACCELERATION.md` for normal-speed isolated collection
   parallelism and the compatibility gate;
11. `docs/TRAINING_INFRA_PERFORMANCE.md` for append-only profiles,
   optimizations, failed attempts, and correctness differentials;
12. `docs/HARD_EMPTY_SOURCE_AUDIT.md` for the source-bound Hard-empty verdict
   and conservative-to-source-exact fallback;
13. `docs/WINE_RETAIL_VALIDATION.md` before launching the game.

Generations 3 and 4 are frozen. Generation 4 exhausted its evidence budget
without canary authorization. Generation 5 is declared, but no new Wine
outcome may launch before its smoke, native, seed, and worker-isolation
contracts are implemented and committed.

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
