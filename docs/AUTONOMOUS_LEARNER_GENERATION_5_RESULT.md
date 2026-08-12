# Autonomous learner generation 5 result

## Verdict

Generation 5 is frozen without a candidate and without Wine canary or natural
Stage-6 evaluation. Its Stage-4 curriculum probe stopped after the predeclared
boundary-15 fail-fast smoke. This is an `ineffective-for-continuation` learner
verdict, not evidence that any candidate policy is better or worse than the
incumbent.

The estimator learned a repeatable held-out Bellman signal, but it did not
learn a sufficiently stable action effect. Spending the remaining five
Stage-4 Wine episodes would test the same structurally unstable decision rule,
so the user-requested boundary review stopped collection and starts a new
algorithm generation. No HIT, action, RNG, frame, or failure location selected
this decision or the successor design.

## Accepted Wine evidence

Fifteen complete original-retail Wine Stage-4 episodes passed the immutable
corpus, physical-HIT conservation, zero-Bomb, propensity, authority, and
cleanup gates. Their physical HIT counts, in frozen schedule order, were:

`23, 36, 48, 29, 21, 27, 34, 26, 32, 26, 21, 27, 31, 36, 30`.

One earlier 22-HIT attempt had an input-pickup timeout. The whole attempt was
strictly rejected and is absent from the learner and this result. A separate
display failure had zero gameplay rows. Both remain archived as infrastructure
evidence and neither was selected according to its gameplay outcome.

The boundary reports are ignored runtime artifacts bound here by SHA-256:

- boundary 10:
  `5c0424666905a978cfa917613755705586957dc25ea9d839cd4d6ebf650bc9d9`;
- boundary 15:
  `bd1fa7bf8ee82c246df9fd63f706fd774f9788483916aa6e9a43dec5c5ceb82a`.

Both reports are explicitly non-authorizing development smokes.

## What was learned

At boundary 10 the cross-fitted factual-Q loss was 0.976178 of the state-only
zero-effect loss and improved all 10 held-out episode groups. At boundary 15
the ratio improved to 0.970393 and improved all 15 groups. Thus the sequential
HIT-only Bellman target, common-state nuisance, and action-centered residual do
extract outcome structure from factual Wine experience.

Decision identification did not improve enough. The two disjoint 3/4-member
panels selected the same non-incumbent action on only 7.00% of the boundary-10
rows where either panel proposed and 8.94% at boundary 15. The frozen
qualification target was 80%. Exact agreement including mutual incumbent
abstention fell from 67.43% to 66.17% as more data made the panels propose more
often. This is not a deployable trend.

A post-stop action-geometry diagnostic classified the 13,759 unequal panel
choices at boundary 15 without inspecting a Stage location or changing a gate:

- 24.0% differed by 45 degrees;
- 21.1% differed by 90 degrees;
- 14.9% differed by 135 degrees;
- 14.5% were opposite directions;
- 13.9% had the same direction and differed only in focus;
- 11.5% were stationary versus movement.

The disagreement is therefore not mainly an overly literal distinction among
near-equivalent controls. The scalar tree treatment head has not identified a
stable physical action ordering. More capacity alone is also not a credible
repair: the earlier 29-episode Stage-6 production-sized smoke had 14.47%
conditional panel agreement, still far below 80%.

## Successor requirement

Generation 6 must shorten learner development before collecting more Wine.
Existing factual Wine episodes become a frozen, episode-grouped qualification
corpus. New learners must first pass deterministic causal/null fixtures, then
unseen complete-episode Wine qualification and native latency. Those checks
may reject an algorithm but can never authorize or promote it. A held-out
qualification partition is disclosed only after architecture choices are
frozen so repeated development cannot tune to all existing episodes.

The estimator redesign must share statistical strength across action geometry
and calibrate the actual full-population pessimistic policy. It may not repair
this result by lowering the old gate, selecting actions, adding shaped reward,
or changing the Wine data distribution. Only a learner that passes the frozen
qualification funnel may spend new Wine episodes, and only normal-speed
original-Wine complete-Stage HIT-continuation evaluation can establish final
effectiveness.
