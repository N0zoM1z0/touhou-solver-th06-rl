# Repository prune boundary

This document prevents obsolete implementation paths from returning after the
Wine-only autonomous-learning decision. The prune removes misleading active
code, not scientific history. Every Generation 1--6 design, runner, frozen
contract, result, reusable learner implementation, and performance optimization
remains tracked.

## Removed paths

The following families were deleted because they contradict the current
contract or duplicate a superseded data plane:

- the reconstructed Linux/headless client, forkserver, geometry lowering,
  corpus generation, COW branching, feasibility oracle, teacher/distillation,
  DAgger, paired panels, compatibility extension, and Stage evaluation;
- retail/headless source differentials and retail-COW tools that could create
  simulated successors or counterfactual labels for actions Wine did not
  execute;
- targeted failure-region, frame/action-family, and manual mastery tooling;
- the mutable `adaptive.py` online UCB, its shaped survival/reserve/edge/phase
  rewards, offline risk guards, consensus promotion, and replay/export tools;
- the old Hugging Face dataset learner/audit/sync path and pre-generation
  shaped offline FQI command line;
- Windows batch launchers that repeatedly invoked mutable online learning;
- resident policy hot reload, online outcome feedback, and Stage checkpoint
  transactions;
- tests whose only purpose was to authorize or preserve one of those paths.

All removed source remains recoverable through Git history for audit. It must
not be copied back into the active tree as a shortcut.

## Retained paths

- all Generation 1--6 learner, orchestrator, configuration, design, progress,
  and result records, including negative results;
- the immutable Wine corpus registry and complete-episode loaders;
- factual semi-Markov option construction and the successive conservative,
  doubly robust, R-critic, implicit-Q, low-rank, and IQL actor implementations;
- the Generation-6 unbounded-objective reproducer and regression test;
- original-retail Wine capture, menu/input delivery, natural/fixed-RNG
  diagnostics, full-Stage HIT accounting, and cleanup;
- native safety, Hard-empty source audit, fused tree/population scorer,
  float64 decision conformance, process isolation, CPU limits, caches, and
  performance benchmarks;
- synthetic learner smokes and tests for every retained generation or reusable
  infrastructure contract;
- the legacy portable tree feature schema, retained only as a read-only model
  serving format. Its old shaped-label learner was deleted.

## Active entry-point rule

The Wine runner and Windows controller now require all of these explicitly:

1. a policy plug-in;
2. its immutable state artifact;
3. `--immutable-policy`;
4. controller-level exploration exactly zero.

There is no default gameplay policy. Autonomous exploration probabilities live
inside the frozen behavior-policy state and are recorded in the corpus. The
resident controller cannot learn, reload weights, checkpoint a policy, or
receive reward feedback.

Historical Generation runners remain source-level audit records, but their
hash-bound completed contracts must not be edited or resumed as current
experiments. Common infrastructure has advanced since those hashes were
recorded; reproduce an exact old run from its named Git commit, never by
weakening a historical manifest in the current tree. A new learner uses a new
frozen generation contract.

## Future prune test

Delete a file only when its complete use is obsolete and no retained
generation or reusable infra depends on it. A path is not obsolete merely
because its learner failed: failed generations are required evidence. A path
is obsolete when it enables a forbidden environment/data/online-learning
contract, has been replaced by the capability-indexed Wine data plane, or is a
test exclusively for such a path.

After pruning, require:

- no tracked headless gameplay/COW branching, targeted distribution,
  mutable online-policy, or old risk-guard entry point;
- no import of a removed module;
- exact retention of native scorer/safety/performance tests;
- full pytest success;
- documentation that names the retained replacement.

`tests/test_repository_prune.py` makes the retired-path and import boundary an
executable regression contract.

Copy-on-write process workers used solely to share immutable offline learner
matrices are a retained performance optimization. They do not execute gameplay
or create counterfactual outcomes and must not be confused with the removed
headless COW environment.
