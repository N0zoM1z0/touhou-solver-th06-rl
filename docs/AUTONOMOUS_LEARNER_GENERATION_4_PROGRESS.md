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

## 2026-08-12: sequential causal learner smoke passed

The first complete implementation slice constructs factual option intervals,
eight-decision undiscounted Bellman targets, five-fold frozen value/outcome
nuisances, and a seven-member generalized action-centered R critic. An initial
smoke failure exposed an implementation-scale bug: equal episode weights had
been normalized so far below the tree learner's fixed minimum child weight
that the nuisances remained at their base prediction. Correcting the weight
scale while preserving equal whole-episode influence made the full pipeline
learnable; no gameplay parameter or evidence threshold was involved.

The deterministic production-sized causal fixture then passed on 160 complete
episode groups. Its true candidate effect was -1 HIT while common state risk
changed independently. Results:

- all seven members predicted negative advantage at both risk levels;
- aggregate prediction: -1.0149 HIT;
- cross-fitted critic R loss: 12,675.41 versus zero-effect 14,415.47;
- aggregate state-risk leakage: 0.3535 HIT;
- maximum individual-member state-risk leakage: 0.7179 HIT;
- every centered objective coefficient was bounded by 0.75; no inverse
  propensity appeared in the objective.

Unit contracts separately prove recursive interval-HIT conservation at the
eight-decision/terminal boundary and direct recovery of -1 action effect from
the centered objective. This is synthetic algorithm evidence, not Wine
gameplay evidence or authorization.
