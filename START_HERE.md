# Start here

The project now has one learning path: original-retail Wine interventions feed
episode-grouped offline residual learning, followed by Wine shadow, canary, and
complete-Stage HIT-count evaluation.

Read in this order:

1. `AGENTS.md` for safety and product boundaries;
2. `docs/WINE_ONLY_INTERVENTION_LEARNING.md` for the data, training, and
   promotion contract;
3. `docs/WINE_RETAIL_VALIDATION.md` before launching the game.

## Non-negotiable separation

- Original retail under Wine creates all gameplay outcomes.
- Offline replay may analyze Wine observations but may not simulate an unseen
  successor action.
- Fixed RNG and accelerated Wine are training diagnostics, not final evidence.
- Normal-speed full-Stage Wine HIT count is the final metric.
- Native geometry owns the safe set; learning only ranks it.
- Bomb is forbidden.

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
