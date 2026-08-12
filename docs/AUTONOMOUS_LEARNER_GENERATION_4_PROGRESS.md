# Autonomous learner generation 4 progress

This append-only record begins after Generation 3 was explicitly stopped and
before any Generation-4 Wine outcome. It cannot amend the design contract
during an evidence run.

## 2026-08-12: generation declared

Generation 3 was stopped after 13 clean complete historical episodes and one
fit-ineligible round. Its next in-progress episode was terminated, has no
retail report, and is excluded. Exact process-group shutdown left no game,
controller, Wine runner, or Xvfb process.

Generation 4 is declared in
`AUTONOMOUS_LEARNER_GENERATION_4_DESIGN.md`. It replaces complete-Stage raw
AIPW labels with factual semi-Markov n-step Bellman targets and a generalized
action-centered R objective; adds recorded ESS/uncertainty-aware propensities;
calibrates the cross-fitted policy rather than the maximum counterfactual row;
and requires the full seven-member population to meet the native latency gate.
No new-generation Wine outcome has been created.
