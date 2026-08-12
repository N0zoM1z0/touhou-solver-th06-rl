# Autonomous learner generation 5 design

## Decision boundary

Generation 5 is declared after Generation 4 exhausted its fixed evidence
budget without canary authorization. The trigger is its aggregate estimator
behavior across 29 complete episode groups: effectively zero and unstable
held-out treatment signal plus a 17%/61%/52% proposal-rate trajectory. No
stage, spell, frame, RNG seed, failure region, or preferred action is used in
this redesign.

The architecture is:

`propensity-recorded Wine options -> in-sample implicit fitted Q -> supported pessimistic population -> Wine canary -> natural complete-Stage evaluation`

Original retail Wine remains the only environment. The sole cost is physical
HIT, `gamma = 1`, and terminal value is zero. Native geometry remains the sole
publisher of the Bomb-free safe set. The learner can only rank or abstain
inside that set.

Generation-3 and Generation-4 factual episodes are development and training
experience. They cannot by themselves authorize a Generation-5 candidate.
Authorization diagnostics must pass separately on newly collected
Generation-5 episodes whose schedule is committed before any outcome.

## 1. Factual in-sample semi-Markov Bellman iteration

Only factually executed option boundaries enter the learner. For each boundary
the dataset contains the observed state, complete native-safe action set,
incumbent, factual action and propensity, interval HIT cost, and the next
factual boundary or terminal. Rejected tentative actions have neither cost nor
successor labels.

Generation 5 minimizes cost with an implicit-Q iteration. Starting from zero
terminal/successor value, each frozen iteration fits factual action value to:

`Q_k(s_t, A_t) = sum(i=0..h-1) HIT_(t+i) + V_(k-1)(s_(t+h))`

where `h` is at most eight factual option intervals and no bootstrap is used
past terminal. It then fits `V_k(s)` to the lower cost expectile of the factual
`Q_k(s,A)` distribution. A 0.10 cost expectile is the reward-maximization IQL
expectile mirrored for cost minimization: low-cost factual actions receive the
larger asymmetric squared-error weight. Backups never maximize or minimize
over unobserved successor actions.

The frozen constants are eight Bellman iterations, eight factual intervals per
target, cost expectile 0.10, `gamma = 1`, 128 depth-six Q trees, and 96 depth-six
offline V trees. Each iteration refits from a frozen predecessor. Complete-
episode weights prevent a long Stage from silently dominating another Stage.

The offline V nuisance may use generic normalized sequence position and
remaining-option count as control variates. Those fields are never exported.
The deployed Q population receives only the adapter-neutral observation,
candidate action, hazard-set encoding, and factual observation history already
available online. It may not read frame number, run identity, RNG, stage, or
phase.

## 2. Population and baseline bootstrapping

The final artifact contains seven independently whole-episode-bootstrapped Q
members. No member is selected as a winner and no mean-only student replaces
the population. For candidate `a`, member `j` supplies the action-relative cost

`D_j(s,a) = Q_j(s,a) - Q_j(s,incumbent)`.

A candidate is eligible only when its factual-action local support passes and

`max_j D_j + (max_j D_j - min_j D_j) < 0`.

Thus the entire population must prefer the candidate and its full bootstrap
range is paid once more as a pessimistic uncertainty margin. Otherwise the
policy bootstraps to the incumbent. This is decision uncertainty, not physical
safety; native geometry still owns safety.

Support uses action-conditional prototypes learned from factual training rows
only. Its 99% threshold is the upper observed order statistic, guaranteeing at
least declared finite-sample coverage. Unsupported actions always abstain.

## 3. Cross-fitting and required smoke evidence

Five deterministic folds are split only by complete episode. Every held-out
row is scored by models that saw none of its episode. The zero-effect comparator
uses the learned state value `V(s)`; the Q learner must improve factual Bellman
target squared error globally and in a strict majority of episode groups.

Four whole-episode-bootstrap calibration members are divided into two
independent halves. On held-out observations the report records exact policy
agreement, agreement conditional on either half proposing, proposal rate,
actions, support abstentions, and member range. A fit must satisfy all of:

- at least 20 complete episode groups overall;
- Q Bellman loss below the state-only V loss globally and in a strict episode
  majority;
- the same two conditions on the new Generation-5 cohort alone;
- at least one held-out proposal, no more than 10% of held-out decisions
  proposed, and at least 80% exact action agreement where either independent
  half proposes;
- complete seven-member artifact, at least 99% factual support calibration,
  finite diagnostics, native equivalence, p95 below 4 ms, and zero 60 Hz
  deadline misses.

The 10% policy exposure ceiling and agreement threshold are generation-level
conservative deployment constraints, not tuned activation regions. Failure
starts another declared algorithm generation or autonomous collection round;
it cannot be repaired by choosing actions or examples by hand.

Before any corpus is used, two deterministic environments must pass:

1. a delayed-effect semi-Markov process in which an action changes a physical
   cost several factual boundaries later; the fitted population must recover
   the direction without a shaped intermediate reward;
2. a null-effect process with state risk and randomized actions but no action
   effect; the deployed selection rule must abstain.

Both fixtures also enforce factual HIT conservation, terminal-zero backup,
episode isolation, and absence of inverse-propensity coefficients. A small
frozen-Wine development fit must complete before costly new collection, but it
cannot authorize gameplay.

## 4. Evidence schedule

Generation-5 collection keeps the Generation-4 propensity-aware mixture inside
the native-safe set: 0.50 incumbent, 0.25 uniform, and 0.25 ESS/model-
uncertainty information mass, with its full probability vector recorded. It
does not target a gameplay location or observed failure.

Fit boundaries are eight, twelve, and sixteen new complete Stages. Historical
experience may fit the learner, but every authorization gate is additionally
computed on the new cohort. A fit-eligible immutable population receives three
paired fixed-RNG complete-Stage canaries. Candidate exercise in at least two
pairs, clean runtime, strictly fewer aggregate candidate HITs, and candidate no
worse in at least two pairs are required.

Only a passing canary permits the final alternating normal-speed natural-RNG
evaluation: 12 complete Stages per arm, HIT continuation, zero Bomb, and the
original executable. Strictly fewer aggregate candidate physical HITs with no
safety/runtime regression is the effective verdict. Exhausting 16 new Stages
without authorization is ineffective.

## 5. Falsification and allowed changes

Generation 5 is falsified if it cannot recover delayed effects, acts in the
null process, lacks new-cohort Bellman improvement, has unstable held-out
policy decisions, fails native latency, loses a Wine canary, or fails to lower
the natural complete-Stage HIT aggregate. Offline loss alone never proves
efficacy.

Human repair remains limited to demonstrated infrastructure errors with a
reproducer, contract test, and audit note: capture/action timing, memory
semantics, factual alignment, process isolation, artifact integrity, native
geometry, or HIT accounting. Changing data quotas, reward, action preferences,
or activation around a gameplay failure is forbidden.
