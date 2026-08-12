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
6. `docs/AUTONOMOUS_LEARNER_GENERATION_4_DESIGN.md` for the active sequential
   semi-Markov learner, action-centered critic, exploration, and evidence
   contract;
7. `docs/AUTONOMOUS_LEARNER_GENERATION_4_PROGRESS.md` for its append-only
   implementation and execution record;
8. `docs/HARD_EMPTY_SOURCE_AUDIT.md` for the source-bound Hard-empty verdict
   and conservative-to-source-exact fallback;
9. `docs/WINE_RETAIL_VALIDATION.md` before launching the game.

Generation 3 is frozen. No Generation-4 Wine evidence may launch until its
committed preflight and unattended entry point pass the declared smoke gates.

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
