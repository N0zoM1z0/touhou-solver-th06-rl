# TH06 autonomous-learning hand-off

This is the authoritative short path for the next agent. It records the
project intent, the evidence accumulated through 2026-08-13, the terminal
Generation-6 diagnosis, and the boundary for the next learner. Read this file,
then `AGENTS.md`, before running code or Wine. The detailed immutable records
linked below remain authoritative for exact experiment contracts and hashes.

## Current decision

No learner generation has yet demonstrated a repeatable reduction in physical
HIT count under the final original-Wine evaluation contract. Offline RL remains
the project direction, but the fitted Generation-6 actor is rejected and must
not be refit or extended. Its final clean Wine confirmation reached 42
candidate HITs versus 34 incumbent HITs after four paired Stage-6 blocks. The
candidate was no worse in only one block; even winning both remaining blocks
could reach only three of the required four, so the rejection-only fail-fast
correctly stopped the experiment.

That negative learner result does **not** invalidate the reusable Wine corpus,
native safety, Wine capture/control, the sequential transition schema, the
float64 native scorer, or the performance work. The next task is learner-only
research on the existing corpus, frozen as a new generation. Do not collect
more Wine or resume the old Generation-6 evaluation merely to try another fit.

There is no authorized Generation-6 candidate and no partial ninth trial that
may be used as evidence or training data. The historical ledger deliberately
has a null decision because the old runner expected all six blocks; the
complete-block impossibility proof is the conclusive result.

## The product we are building

The core research question is whether an agent can improve Touhou play through
**autonomous learning**, not whether humans can gradually script a good TH06
route. The intended loop is:

```text
original Wine exploration
  -> immutable capability-indexed corpus
  -> grouped offline RL
  -> immutable fast native candidate
  -> original Wine shadow/canary
  -> natural-RNG complete-Stage paired HIT evaluation
  -> algorithmic continue/reject/stop decision
```

People may repair demonstrated infrastructure defects. The algorithm decides
how to learn from repeated episodes and which native-safe action to prefer.
Poor play is evidence about the learner; it is not permission to add a spell,
frame, RNG, failure-location, or action exception.

The offline learner may be sophisticated, expensive, and population-based.
The online component must remain immutable and fast: it receives one coherent
observation and the current native-safe action set, ranks only that set, and
returns before the control deadline. The same data/learner/orchestrator
interfaces should work for TH08 after replacing the game-specific adapter.

## Non-negotiable scientific and safety contract

1. **Wine is the environment.** Only the original Japanese TH06 1.02h retail
   executable under Wine may create trajectories, rewards, branch outcomes, or
   promotion evidence. Linux/headless simulation is quarantined history and
   may not train, label, propose, evaluate, or make compatibility claims.
2. **The final objective is physical HIT count.** Serious learners use HIT as
   the only cost, `gamma = 1`, and terminal value zero. No survival, phase,
   progress, graze, spell, or route reward shaping is allowed.
3. **The final evaluation is the real game.** Use normal-speed original Wine,
   natural unread RNG, a complete Practice Stage, continue-on-HIT, zero Bomb,
   immutable policies, and alternating incumbent/candidate trials. A fixed-RNG
   canary, accelerated run, snapshot, offline estimate, or wiring test can
   reject a candidate but can never prove efficacy.
4. **Native code owns safety.** Capture one coherent snapshot, build a bounded
   native-safe first-action set, let learning rank only that set, then recapture
   and revalidate before publishing input. Learning may not enlarge the safe
   set, alter collision geometry, control Bomb, or bypass fail-close behavior.
5. **No scripted play or manual distribution repair.** Do not select data,
   collection eligibility, features, rewards, thresholds, exploration, or
   actions by boss, spell, frame window, RNG, observed HIT location, run ID, or
   a hand-picked counterexample. Do not manually oversample where a run looked
   bad. Autonomous, predeclared, propensity-recorded exploration inside the
   native-safe set is allowed.
6. **Humans repair only general infrastructure.** Valid areas include memory
   semantics/coherent capture, input delivery, source-accurate safety geometry,
   factual transition and HIT accounting, artifact integrity, process
   isolation, and performance. A repair needs a reproducer, a contract test,
   and an audit showing that semantics were preserved.
7. **Complete episodes are the statistical groups.** Never split adjacent
   frames/options from one physical episode across train and validation.
   Offline replay may construct features and factual semi-Markov targets, but
   it may never invent the successor of an action that Wine did not execute.
8. **Collection and fitting are separate.** Online collection records the
   chosen action and its complete behavior propensity; it never updates model
   weights. Fit only at declared episode boundaries. A fitted model binds the
   exact corpus inventory, code, parameters, folds, and random seeds.
9. **Every round is auditable and resumable.** Preserve the hash-bound state
   machine `Collect -> Validate -> Fit -> Shadow -> Canary -> Evaluate ->
   Decide`. Only complete accepted reports advance it. Reject early only at a
   complete predeclared boundary when a monotone gate is mathematically
   impossible; never accept early.
10. **Do not consume the whole host.** Keep the repository-wide CPU ceiling at
    32 because other CPU-heavy jobs share the machine. Do not overlap canonical
    Wine gameplay with learner load. Native C/C++ hot paths and bounded
    copy-on-write worker pools are welcome when exact differentials pass.

These are project constraints, not tunable hyperparameters.

## Permanent data-plane separation

Corpus, learner code, fitted artifacts, and evaluation evidence are four
different things:

- The corpus contains immutable physical facts produced by original Wine.
- The registry describes what each source can support; learner code queries
  capabilities instead of assuming that every episode has every field.
- A learner is a replaceable algorithm that may reuse every compatible old
  episode. Replacing it does not copy, relabel, mutate, or recollect the data.
- A fit is a derived, hash-bound artifact. It may be deleted and reproduced
  without changing the corpus.
- Evaluation trajectories are promotion evidence and are not silently fed
  back into the same candidate that they evaluated.

`config/wine_corpus_registry.json` is the authority. At this hand-off it lists:

| Source | Access/capability | Complete episodes |
| --- | --- | ---: |
| `deterministic-complete-stage-v6` | training for representation/state value; not sequential options | 12 |
| `randomized-option-v9` | sequential RL; reconstructible complete propensity | 13 |
| `randomized-option-v10-stage6` | sequential RL; recorded complete propensity | 16 |
| `randomized-option-v10-stage4` | sequential RL; recorded complete propensity | 15 |
| `generation6-round3-startup-audited-natural` | sequential RL; natural RNG; Generation-6 behavior | 12 |
| `failed-parallelism-differential` | infrastructure regression only; never learner input | 3 |

Therefore there are 71 registered clean complete Stage runs in total, 68 with
some training access, and 56 sequential-offline-RL-compatible episodes. The
latest all-registry sequential replay has 167,250 factual options and
2,415,808 candidate rows. These counts are a snapshot, not a hard-coded learner
contract; always query capabilities and validate inventory hashes.

The corpus roots, game binaries, reference source, logs, models, and reports are
ignored local evidence, not source files. Do not commit or copy them. The
tracked registry binds their expected inventory hashes so missing local
artifacts must be restored, not regenerated with a different policy under the
same source ID.

New collection is justified only by a predeclared autonomous round or a real
capability/coverage gap. A poor offline or Wine result is not itself a data gap.

## Learner history and why each generation stopped

The experiments below are negative results about particular estimators and
policy-extraction contracts. They are not evidence that offline RL cannot work.

### Generation 1: grouped linear factual-return committee

- Ten fixed-RNG Stage-6 Wine episodes; 0.10 uniform exploration inside the
  native-safe set.
- Factual 120-frame discounted return (`gamma = 0.99`) and a
  clipped-propensity grouped ridge committee with whole-episode holdout.
- Held-out factual RMSE ratios improved from 0.955 to 0.905, but action-level
  uncertainty did not improve: the two shadows produced four and then zero
  proposals against a minimum of ten.
- It stopped safely on genuine Hard-empty authority outcomes and never entered
  active canary or A/B evaluation.

Why it failed: predicting behavior-policy return was easier than identifying a
stable counterfactual action advantage. Better factual prediction did not
produce a deployable decision rule.

### Generation 2: nonlinear conservative fitted Q

- Eight complete Stage-6 Wine collection episodes.
- Five episode-bootstrap members, 60-frame n-step HIT targets, six fitted-Q
  iterations, histogram trees, local action-support prototypes, and a
  pessimistic ensemble.
- A fixed-RNG canary passed, but authoritative natural-RNG evaluation was
  baseline 17 HITs versus candidate 18. The candidate made only four overrides
  across 63,055 decisions.

Why it failed: the sparse conservative overrides did not yield a repeatable
physical advantage; a small fixed-RNG win did not transfer to natural RNG.
The source-bound Hard-empty audit also showed that remaining Hard-empty events
were legitimate fail-close semantics, not the learner's missing fix.

The historical 17 is a two-run Generation-2 aggregate, **not** a universal
single-Stage baseline. Do not compare it with randomized collection policies,
different stage panels, or later multi-block totals.

### Generation 3: complete-return multi-action AIPW

- Safe eight-frame options, cross-fitted multi-action doubly robust/AIPW
  pseudo-outcomes, an ensemble, and distilled online scorer.
- Thirteen Wine episodes were accepted; the first fit used twelve.
- Rare action propensity near 0.0056 created inverse-propensity factors around
  180, while every option inherited a noisy complete-Stage Monte-Carlo return.
- Held-out advantage RMSE was 86.1797 HIT versus 86.1578 for zero advantage;
  distillation p95 was 3.7091 versus a 0.05 gate, maximum 370.8625 versus 0.25,
  and the three-group conformal radius was 2476.2625 HIT.

Why it failed: raw multi-action inverse propensity, long-horizon complete
returns, and too few independent episode groups produced unusable variance.
It was frozen before canary; no outcome-facing candidate was authorized.

### Generation 4: semi-Markov R-critic

- Eight-option factual semi-Markov transitions, frozen nuisance targets,
  generalized action-centered Robinson/R learning, a seven-member 128-tree
  native population, and propensity/ESS-aware autonomous exploration.
- It reused 13 episodes and collected 16 new Stage-6 episodes. The three critic
  loss ratios versus zero effect were 1.002472, 1.001182, and 0.999884; the
  final fit improved only 14 of 29 episodes versus the required 15.
- Proposal rates moved 17.0% -> 61.0% -> 52.1%.

Why it failed: the one-pass frozen nuisance was not a true iterative Bellman
fixed point; long common risk still dominated small action effects, and
unanimous bootstrap signs did not calibrate decision uncertainty. No canary
was authorized. This generation did yield reusable general repairs: an exact
empirical order-statistic support threshold and state hazard encoding once per
option instead of once per action.

### Generation 5: implicit-Q/IQL-style factual critic

- Factual in-sample semi-Markov Bellman iteration, a state-only nuisance,
  centered residual Q, cost expectile 0.10, population pessimism, and support
  fallback.
- Synthetic delayed/null tests passed. Fifteen Stage-4 curriculum episodes
  improved held-out Q-loss ratio from 0.976178 to 0.970393, and all episode
  groups improved.
- Independent 3/4-member panels chose the same proposed non-incumbent action
  on only 7.00% then 8.94% of relevant rows versus an 80% target. A larger
  Stage-6 capacity smoke reached only 14.47%.

Why it failed: the learner found outcome structure but the pointwise critic
argmin/pessimism did not extract a stable physical action ordering. The
disagreement covered genuinely different directions, not merely focus or
near-equivalent controls. The runner stopped at boundary 15 without canary.

### Generation 6: policy extraction experiments

Generation 6 deliberately reused the frozen corpus to shorten algorithm
iteration before more Wine collection.

1. A low-rank varying-coefficient R-critic passed synthetic tests but had
   held-out Q loss 1.0466 of zero, improved only 11/31 development episodes,
   and preserved leave-one-out full-policy proposals on 31.09% of cases.
2. A full nonlinear tree critic improved Q loss to 0.986 and all 31 episode
   groups, but full-policy removal stability was 14.08% and conditional
   stability only 6.17%. This isolated policy extraction as the problem.
3. Ordinary empirical advantage-weighted regression initially canceled the
   treatment signal under action imbalance. A known-propensity action-centered
   control variate fixed the synthetic mean and, after correcting an algebra
   error, became the actor objective.
4. Nested cross-fitting then used episode-excluded critic labels, outer policy
   evaluation, seven actor members, and a small propensity-bounded intervention
   sampler. It passed development/qualification and reported favorable
   Stage-4/Stage-6 offline estimates. A six-run Stage-6 pilot was directionally
   positive (candidate 25 versus incumbent 28) but was explicitly too small for
   promotion.
5. Autonomous round 1 was invalidated by two scorer deadline misses reproduced
   as CFS scheduling tails under unrelated host contention. Round 2 stopped
   before gameplay because a privileged monitor watched the wrapper PID instead
   of the exec child. Round 3 bound the generic repairs (exact CPU affinity,
   bounded `nice -10`, PID attestation/cleanup), reused only the clean prefix,
   collected the two missing Stage-5/6 episodes, and registered 12 accepted
   natural-RNG runs. These were infrastructure failures and repairs, not
   algorithm evidence.
6. The all-registry 56-episode fit predicted Stage-6 effect -4.9632 HIT with a
   bootstrap upper endpoint -3.7991. A float32 native export initially failed
   strict numeric conformance. Decision-level analysis found two actual action
   differences, so the gate was not relaxed. Keeping float32 parameters but
   using float64 intermediates made support and action decisions exact on all
   167,250 options and on the Win32 panel, while remaining fast enough.
7. The resulting clean Wine confirmation contradicted the offline forecast:
   candidate 42 versus incumbent 34 after four blocks, making the frozen
   positive rule impossible.

The confirmed root learner defect is mathematical. Generation 6 optimized

```text
L_hat = E_mu[L] + (w(A) - 1) L(A)
```

as empirical actor loss. It is unbiased for a **fixed** model, but its factual
coefficient can be negative. The actor can therefore reduce the objective
without bound by sending the factual action probability toward zero. A direct
probe produced losses -1.837, -5.526, -11.052, and -27.631 as that probability
fell from `1e-2` to `1e-30`; a proper weighted cross-entropy stayed positive.

The production fits display exactly this exploitation:

| Fit | Episodes / options | Mean centered actor risk | Stage-6 behavior KL | Offline Stage-6 estimate |
| --- | ---: | ---: | ---: | ---: |
| initial qualification | 31 / 102,737 | +0.188 | 2.83 | -1.3405 HIT |
| all-registry preflight | 44 / 143,078 | -9.579 | 6.52 | -3.6075 HIT |
| round-3 production | 56 / 167,250 | -56.638 | 46.13 | -4.9632 HIT |

A true weighted cross-entropy is nonnegative. More data made this invalid
objective look more optimistic while its policy diverged. In addition, using
the mean of seven actors discarded the uncertainty that the population was
supposed to express: on Stage 6 the independent 3/4 split agreed on only 1.42%
of contexts where either proposed, while the fully pessimistic population
proposed on only 28 of 114,580 options.

**Do not optimize `action_centered_actor_losses` again.** It remains in the
repository solely for history, tests, and diagnosis.

## Reusable infrastructure that is currently valid

Keep these improvements when replacing the learner:

- Original-retail Wine full-Stage continue-on-HIT collection and evaluation,
  exact executable binding, zero-Bomb checks, coherent capture, safe-action
  revalidation, process-prefix cleanup, and immutable episode reports.
- Capability-indexed immutable corpus reuse with whole-episode groups and
  recorded/reconstructible complete behavior propensities.
- Generic eight-frame factual semi-Markov options and HIT conservation.
- Native safety/Hard-empty behavior. The authoritative source audit found the
  extra uncertainty-margin fallback issue and repaired it; remaining
  Hard-empty events are source-consistent fail-close decisions.
- Fused seven-member native scorer. Float64 intermediate accumulation with
  float32 parameters is the frozen numeric contract; do not revert to
  NumPy/OpenBLAS final-logit closeness as the sole oracle.
- Decision conformance checks compare baseline-centered advantages, support
  masks, proposal, tie-break, and final action. The complete Linux audit was
  exact on 167,250 options / 2,415,808 candidate rows; the frozen wide Win32
  panel was exact as well.
- Full-corpus audit uses 16 deterministic copy-on-write workers, one BLAS
  thread per worker, and the 32-CPU host cap. It completed in 109.84 seconds
  under the 180-second gate.
- Online gates remain p95 below 4 ms and zero deadline misses. The accepted
  float64 path measured about 0.488 ms Linux p95 and 1.221 ms isolated Wine
  p95; the final gameplay panel's largest resident p95 was 3.2269 ms with zero
  misses.
- Rejection-only mathematical fail-fast at complete paired-block boundaries.

Do not confuse historical scorer revisions with learner failures. The first
Generation-6 Stage-4 wiring builds measured p95 8.155, 4.1027, and 4.0678 ms;
SSE2/fusion reduced the next build to 2.8751 ms. The later numeric successor
passed. Latency and portable/native numerical consistency are no longer the
current blocker.

## Required boundary for Generation 7

The next agent should begin with offline replay, not Wine collection:

1. Freeze a new learner design and its rejection gates before viewing any new
   outcome-facing Wine result. Reuse all compatible registry episodes through
   capability queries.
2. Use a bounded proper actor/value objective. Every per-sample optimization
   weight must be nonnegative, and the empirical objective must have a finite
   lower bound. An unbiased control variate is not sufficient.
3. Add an extreme-logit smoke that proves the loss cannot improve by assigning
   vanishing probability to a low-weight factual action. Also retain causal
   null, delayed-effect, propensity, HIT-conservation, and episode-leakage
   smokes.
4. Cross-fit the complete policy-construction pipeline by physical episode,
   not only the critic. Report held-out proper loss, policy value, proposal
   stability, behavior KL, and intervention exposure.
5. Define and evaluate the exact deployable stochastic policy. Do not validate
   a pessimistic population and then deploy an uncalibrated mean, or estimate a
   teacher and deploy a materially different distilled student.
6. Calibrate uncertainty at the policy/decision level. Hard safety already
   handles physical legality; statistical uncertainty should determine whether
   an estimated action effect is trustworthy. A model that nearly always
   abstains has not solved the task merely because it is safe.
7. Use offline metrics only to reject candidates and choose among
   predeclared learner variants. Once a candidate is frozen, rerun the full
   native Linux/Win32 decision differential and latency smokes.
8. Before an active canary, require a fresh incumbent-occupancy Wine shadow
   panel to satisfy a predeclared proposal/exposure calibration rule. Shadow
   play checks deployment distribution and wiring; it is not efficacy proof.
9. Only then run a small original-Wine canary and a separately frozen natural-
   RNG complete-Stage paired evaluation. Preserve all safety, cleanup, and
   fail-fast gates.
10. If the result is poor but infra contracts pass, analyze the learner and
    improve the general algorithm. Do not patch a TH06 situation or manually
    reshape the corpus.

Promising directions are allowed to be advanced and complex, but the choice
must be justified by sequential offline-RL identification and the available
behavior support. In particular, a proper nonnegative advantage-weighted/KL-
regularized policy objective can be investigated, as can a different bounded
value/policy method. The repository does not mandate one named paper. It does
mandate falsifiable synthetic tests, complete-episode cross-fitting, exact
deployment fidelity, and original-Wine confirmation.

## Files to read and entry points

Read in this order:

1. `HAND_OFF.md` -- this snapshot and next boundary.
2. `AGENTS.md` -- hard working rules enforced for every change.
3. `docs/WINE_ONLY_AUTONOMOUS_LEARNING.md` -- end-to-end scientific contract.
4. `docs/IMMUTABLE_WINE_DATA_PLANE.md` and
   `config/wine_corpus_registry.json` -- corpus/learner separation.
5. `docs/GENERATION6_DECISION_GAMEPLAY_RESULT.md` -- current terminal learner
   verdict and objective proof.
6. `docs/GENERATION6_DECISION_NUMERIC_SUCCESSOR.md` and
   `docs/TRAINING_INFRA_PERFORMANCE.md` -- serving and throughput contracts
   that must not regress.
7. `docs/AUTONOMOUS_LEARNER_GENERATION_6_DEVELOPMENT.md` and the prior
   generation result documents -- detailed algorithm trail.
8. `docs/HARD_EMPTY_SOURCE_AUDIT.md` and
   `docs/WINE_RETAIL_VALIDATION.md` before gameplay-facing changes.
9. `docs/OFFLINE_RL_REFERENCES.md` for the reproducible ignored paper/repo
   cache and source URLs.

Relevant code:

- `src/th06_rl/wine_corpus_registry.py` -- capability-aware immutable inputs.
- `src/th06_rl/iql_actor_learning.py` -- historical Generation-6 learner;
  `action_centered_actor_losses` is forbidden as an optimization objective.
- `src/th06_rl/native_decision_conformance.py` and
  `scripts/audit_generation6_native_decisions.py` -- full decision differential.
- `native/src/th06_rl_ranker.cpp` -- valid fused float64-intermediate serving.
- `src/th06_rl/evidence.py` -- reusable rejection-only count impossibility.
- `scripts/run_generation6_decision_successor.py` -- historical exact runner
  and evidence contract; do not resume it as though the candidate were live.
- `scripts/run_generation6_autonomous_round.py` -- historical resumable round
  infrastructure; a new learner needs a new frozen contract, not edited old
  outcome gates.

The authoritative source checkout is the ignored
`reference/GensokyoClub-th06/`. Trace shipped-game memory, collision, and
Hard-empty claims to it. Do not use REA, REA-provided tools, or LeanToken.

## Operational discipline

- Preserve user changes and ignored evidence. Never delete a corpus, artifact,
  Wine prefix, or source checkout merely because it is untracked/ignored.
- Do not kill generic Wine processes without proving the exact prefix and PID;
  other host tasks may be using Wine.
- Never launch the Windows game through a PTY. Release all input and attest
  exact cleanup after every physical run.
- Commit meaningful checkpoints as `N0zoM1z0
  <161784452+N0zoM1z0@users.noreply.github.com>`, push the working branch, and
  use reviewable PRs. Keep analysis, design decisions, failed attempts,
  performance changes, and final evidence in tracked docs.
- Tests and offline replay are cheap rejection gates. Do not wait through many
  full Wine episodes to discover an algebra, export, latency, or wiring defect.

## Bottom line

The environment/data/safety/serving foundation is usable. The current blocker
is policy learning and trustworthy offline-to-online calibration. Generation 6
looked increasingly strong offline because its actor ERM was unbounded below;
the real original-Wine experiment exposed the contradiction. Start Generation
7 by replacing that objective with a bounded proper, fully cross-fitted policy
learner and prove on the existing corpus that it learns the intended action
effect without exploiting its loss. Only after it passes the entire offline
and native rejection funnel should the project spend more original-Wine
gameplay time.
