# Autonomous learner generation 4 result

## Verdict

Generation 4 is frozen as **ineffective**. Its unattended state machine used
the complete predeclared evidence budget: 13 frozen historical episodes plus
16 new fixed-RNG original-retail Wine complete Stages. No fit earned canary
authorization, so no canary or natural-RNG evaluation was launched.

This is an algorithm verdict, not an environment verdict. All 16 new runs were
clean Stage 6 Lunatic completions in continue-on-HIT mode, with complete v10
propensities, factual options, native-safe execution, zero Bomb, and exact HIT
accounting. Their physical HIT counts were:

`40, 34, 41, 38, 42, 26, 31, 38, 47, 36, 35, 34, 46, 42, 38, 30`

The mean was 37.375 and the range was 26--47. These are randomized collection
policy outcomes, not candidate-policy scores and not comparable to the
Generation-2 two-run natural baseline aggregate of 17 HITs.

## Fit evidence

The three immutable fit boundaries produced:

| New / total episodes | zero R loss | critic R loss | ratio | episodes improved | proposals / options |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 / 21 | 80,279.60 | 80,478.03 | 1.002472 | 4 / 21 | 12,871 / 75,623 |
| 12 / 25 | 101,489.80 | 101,609.76 | 1.001182 | 13 / 25 | 54,488 / 89,267 |
| 16 / 29 | 69,716.67 | 69,708.57 | 0.999884 | 14 / 29 | 53,390 / 102,409 |

The final aggregate improvement is only 8.10 squared-error units out of
69,716.67, or 0.0116%, and it fails the strict episode-majority gate. On the new
Wine cohort alone, critic/zero ratios at the three boundaries were 1.006534,
1.006312, and 1.000823. The final first eight new episodes were worse at
1.003197 while the last eight were better at 0.998471. There is no stable
generalization trend.

The selected-policy surface was also unstable: proposal rate moved from 17.0%
to 61.0% to 52.1%, with proposals in every episode. Unanimous agreement among
seven bootstrap members therefore did not behave as useful decision-level
uncertainty. More Wine episodes under the unchanged estimator are not justified
until its sequential credit and abstention behavior pass stronger smoke tests.

## Demonstrated infrastructure findings

The support threshold used NumPy's interpolated 99th percentile. With a finite
sample that threshold can lie below the next observed distance, so measured
coverage was 0.989990, 0.989996, and 0.989991 rather than at least 0.99. This is
a generic calibration defect. It is repaired by selecting the upper empirical
order statistic and is covered by a finite-sample contract test.

That defect did not cause the ineffective verdict: Round 3 independently
failed the strict 15-of-29 episode improvement requirement, and Rounds 1 and 2
had critic loss above zero-effect loss. The completed result is not recomputed
or retroactively authorized.

Offline profiling also found that hazard features were identically recomputed
for every safe candidate although hazards are state-level inputs. Encoding once
per option preserves every feature bit while removing up to 18 repetitions.
This is a semantics-preserving generic throughput repair with an exact-vector
test.

## Observation, inference, decision, falsification

Observation: eight-step frozen nuisance targets drastically reduced the raw
complete-return variance, but the action-centered critic could not recover a
stable held-out treatment effect and changed a very large fraction of actions.

Inference: Generation 4 is not a genuinely iterative fitted offline-RL
algorithm. Its value nuisance starts from observed complete return and is used
once to form targets; it does not repeatedly solve an in-sample Bellman fixed
point for the policy being evaluated. Long-horizon common risk still dominates
the small action effect, while unanimous bootstrap sign is poorly calibrated
as a policy decision bound.

Decision: Generation 5 must use factual in-sample semi-Markov Bellman
iterations, separate an offline-only common state value from a deployable
action-relative scorer, and bootstrap to the incumbent under unsupported or
uncertain decisions. It must test delayed action effects, null effects, HIT
conservation, held-out Bellman error, policy stability, and native latency
before collecting new evidence.

Falsification: reject Generation 5 before Wine canary if it cannot recover a
known delayed effect, proposes under a known null effect, fails to improve on a
disjoint new-episode cohort, has high cross-fold/round proposal churn, or cannot
execute its complete immutable population within the native deadline. Only a
predeclared full Wine canary and natural-RNG complete-Stage HIT aggregate can
establish gameplay efficacy.
