# Learning retrospective and experiment ledger

This document records why the TH06 solver is moving beyond an online UCB-only
learner, what remains useful about UCB, and which headless experiments justify
each change. It is an evidence log, not a claim that the current replacement
has already cleared Lunatic or passed Windows validation.

## Why UCB was a reasonable first learner

The original environment was dominated by slow Windows physical runs, limited
safe exploration, expensive restarts, and a tiny resident control budget. A
bounded UCB table fit those constraints:

- constant-time lookup and update;
- easy persistence and inspection;
- natural online adaptation without a GPU or model service;
- no ability to enlarge the native action set or bypass collision authority;
- graceful cold start through coarse state backoff.

Those properties remain valuable. UCB was not selected because it was expected
to solve long-horizon routing alone; it was the safest learner that could make
progress under the original collection bottleneck.

## Observed limitations of UCB-only learning

### 1. Rare failures make credit assignment too slow

An authority dead-end may be caused by a movement choice hundreds of ticks
earlier. Online UCB mainly observes the selected action and delayed trajectory
result. On a physical Windows loop, repeating enough comparable trials to
attribute that failure is slow and noisy.

The Linux headless loop found this pattern in minutes. For example, the first
Stage 6 distilled candidate failed at tick 940; an iterative candidate reached
1800; a later candidate regressed to 1613 and was rejected. The final 120
decisions before that regression disagreed with the local teacher 115 times as
the native legal set collapsed. This feedback would have required many slow
physical episodes to isolate.

### 2. Coarse UCB keys alias physically different futures

Phase/position/threat/reserve buckets are deliberately small enough for online
sample efficiency, but two states with the same bucket can have very different
bullet geometry and future maneuverability. More exact mask/context backoff
reduces aliasing but does not create counterfactual evidence for actions that
were not selected.

### 3. UCB explores one factual trajectory at a time

At a decision point, online UCB can observe only the chosen action's successor.
It cannot know whether another native-safe first action would have prevented a
dead-end without revisiting a sufficiently identical state. Deterministic COW
checkpoints can instead test every native-legal first action from the same
physical state and dynamically revalidate every continuation tick.

At Stage 3 / seed 9 / tick 1951, both the factual policy and 12-frame local
teacher chose `stay_fast` and reached an authority dead-end 101 ticks later.
Changing only the first action to `up`, then returning control to the same
dynamic local teacher, survived the full 180-tick branch with all 18 actions
remaining native-legal and about 63.46 pixels of terminal boundary reserve.

### 4. Sparse online samples hide RNG-specific failure distributions

Stage-isolated training is necessary but not sufficient. In the seed15–22
12-frame-teacher benchmark, Stage 3 had five authority failures and three
3000-tick runs; Stage 4 had two failures and six 3000-tick runs; Stage 5 split
four and four. A model or UCB table that looks good on one RNG route can fail
much earlier on another.

### 5. UCB has no efficient way to learn a shared action-value surface

Table entries do not naturally share evidence between nearby candidate
geometries. Tree rankers can learn reusable relationships between candidate
clearance, endpoint, boundary reserve, current movement, counts, and automatic
source context while still ranking only the native-safe set. This higher
capacity also creates OOD risk, so it must be paired with complete-seed
holdouts and rollout gates.

## What the first offline models taught us

The first LightGBM teacher distillation achieved approximately 99% held-out
teacher top-1 on early Stage 6 data, but its rollout still failed at tick 940.
DAgger states extended the best candidate to 1800, and failure-distance
weighting later produced two audited 3000-tick Stage 6 runs. Offline top-1 was
therefore useful for experiment ordering but not sufficient for promotion.

The same lesson repeated across stages. Initial Stage 1–5 models used five
1200-tick seeds per stage. On a seed-12 3000-tick rollout, Stages 1, 2, and 4
reached the bound while Stages 3 and 5 failed at 1586 and 2763. On new seeds,
those apparent wins did not all generalize. Only Stages 1 and 2 later met the
two-seed 3000-tick candidate gate; Stage 3–5 models were rejected.

Increasing the local teacher horizon was not a universal fix. Stage 3 horizon
30 did not improve over horizon 12; horizon 60 moved failures from roughly
2052/2097 to 2230/2225 but did not reach 3000. Stage 4 seed 9 improved from a
horizon-12 failure at 1427 to a horizon-30 3000-tick run. Horizon is therefore
an experimental dimension, not a single global constant to hand tune per
phase in the resident policy.

Point-wise COW overrides were also insufficient. Five Stage 4 counterfactual
labels corrected four local-teacher choices, but the resulting model failed on
new seeds 15 and 16 at 1883 and 2680. This motivated dense state-neighborhood
coverage and an explicit action-value ranking objective.

## Current learning flywheel

1. Run deterministic, scope-isolated headless episodes with Bomb impossible.
2. Audit every factual transition, digest chain, propensity, native action, and
   sparse full-state anchor.
3. Use complete-seed splits; never randomly split rows from the same route.
4. Treat HIT and native-authority failure as terminal negative evidence.
5. Replay failure/success neighborhoods into immutable COW checkpoints.
6. Fork every native-legal first action and continue with dynamic per-tick
   native revalidation. Static factual continuation is not trusted.
7. Audit that each checkpoint covers exactly its native legal set and recompute
   the outcome ranking independently.
8. Train and compare:
   - local-teacher imitation with failure-distance weighting;
   - COW-corrected imitation for conservative full-route coverage;
   - LambdaMART grouped action value using all branch outcomes;
   - later fitted-Q/CQL/IQL only when sufficiently diverse terminal returns and
     behavior support exist.
9. Rank only the native-safe set and revalidate immediately before publication.
10. Promote by unseen-seed full-stage headless rollout, then by Windows
    differential and physical evidence. A 3000-tick bound is not a stage clear.

## Current counterfactual evidence

The first dense seed9/14 set contains 150 audited checkpoints and 2649 action
outcomes across Stages 3–5. Eighty percent of checkpoints have a unique best
long-horizon action. The factual/local-teacher action belongs to the best set
at only 9.3% overall: 19.6% for Stage 3, 9.1% for Stage 4, and 1.7% for Stage 5.

A LambdaMART value smoke trained on only one seed fit its training checkpoints
perfectly but generalized poorly to the other seed: Stage 3 held-out top-1 was
34.8%, and Stage 4 was 4.5%. These models are diagnostic failures and are not
deployed. The response is to expand complete seed groups, not to report the
training score or weaken the rollout gate.

### 2026-08-08 multi-seed COW result

The formal expansion used eight complete factual seeds per stage (15--22) and
the final 600 transitions of each trajectory at stride 40. All 360 COW files
passed an independent audit: 360 checkpoints, 6340 native-legal action
outcomes, and no missing action branches. A unique best action existed at
82.2% of checkpoints, while the factual/local-teacher action was in the best
set at only 7.8%. Per-stage factual-best ratios were 7.5%, 7.5%, and 8.3% for
Stages 3, 4, and 5.

Complete-seed holdout exposed severe generalization error despite perfect
training ranking. The plain-feature LambdaMART held-out top-1 results were
23.3% / 3.3% / 0% for Stages 3 / 4 / 5. COW-corrected imitation did not rescue
closed-loop behavior:

- Stage 3 unseen seeds 23/24 failed closed at ticks 2484/1216;
- Stage 4 unseen seeds 23/24 failed closed at ticks 838/895;
- Stage 5 unseen seed 23 reached the 3000-tick bound, but seed 24 failed closed
  at tick 1583.

All runs remained HIT-free and Bomb-free because native authority failed
closed. None of these candidates was promoted.

The first representation response added a fixed 8-sector summary of already
observed bullets (near/approaching counts and current/projected surface
distance). Replaying the same 24 runs produced 64,820 audited transitions with
identical physical termination distribution: 11 authority failures and 13
tick-limit runs. It modestly raised value top-1 to 23.3% / 6.7% / 6.7%, but
unseen-seed rollouts still failed at Stage 3 ticks 1216/1210, Stage 4 tick 2764
and one 3000-tick run, and Stage 5 ticks 2452/1837. This representation is
retained as a bounded portable primitive, not claimed as a solution. The next
experiment should expose candidate-relative multi-time clearance profiles or
a fixed hazard field instead of increasing tree capacity.

The multi-time profile experiment confirmed that unique-best is also the wrong
primary target. Across the 360 checkpoints, 84.2% / 93.8% / 86.0% of candidate
actions survived the full 180-tick branch in Stages 3 / 4 / 5. The local
teacher survived at 83.3% / 93.3% / 86.7% even though it was lexicographically
best at only 7.5% / 7.5% / 8.3%. A survivable-set classifier raised held-out
COW survivable top-1 to 86.7% / 93.3% / 73.3%.

That label correction was necessary but not sufficient. On unseen seeds 23/24
the survivable-set candidates still failed closed at ticks 2198/1371,
2003/1394, and 2452/2300 for Stages 3/4/5. They remain rejected. The next
policy experiment should retain a quality-diverse archive and allow a COW
correction only under physical-support and cross-model agreement, keeping the
incumbent elsewhere. Archive membership is decided by full-stage clear/HIT,
worst survival, reserve, seed coverage, diversity, and CPU latency rather than
latest training time.

## Intended role of UCB after offline training

UCB should remain as a small, safe online correction layer rather than the sole
source of long-horizon knowledge. A likely deployment is:

1. native geometry constructs the only publishable action set;
2. a distilled offline ranker supplies a strong prior inside that set;
3. contextual UCB adapts among those safe candidates using real Windows
   outcomes and detects simulator mismatch;
4. the fresh issue gate remains final authority.

UCB must not override collision geometry, add an action, request Bomb, select a
handwritten phase route, or turn a failed authority check into exploration.

## Success does not require a perfect solver

NMNB is an existential route objective, not a requirement to identify the
globally best action at every state. A policy may rank many COW checkpoints
incorrectly yet clear because several actions remain survivable. Conversely,
99% imitation accuracy can fail at the one irreversible corridor decision.
The primary metric is therefore a complete no-HIT/no-Bomb stage clear, then
repeatability across target seeds and worst-case reserve. COW top-1, teacher
accuracy, and loss are diagnostic metrics only.

The repository currently contains no committed Stage 1--3 physical clear IDs.
`main:PHYSICAL_EVIDENCE.md` records complete Lunatic evidence for Stages 4--6,
all with HITs, while the three-clear Stage 1--6 mastery logic in `START_HERE.md`
defines a gate rather than proving it was met. Reports that older Stage 1--3
behavior "basically cleared" remain useful historical context, but require the
ignored Windows run artifacts or fresh paired trials before qualification.

## Windows non-regression boundary

Linux ranking never replaces the installed Windows policy. A candidate remains
side-by-side with the incumbent until paired Windows stage trials show no
regression in clears, HIT, authority failure, survival point, and control
latency. Validation starts with shadow scoring, then one-stage canaries; the
first HIT or authority anomaly stops the run and preserves the incumbent.
This guarantees that an unvalidated offline model is not promoted, but it does
not pretend Linux evidence can guarantee Windows performance equivalence.

Every candidate also receives a secondary per-stage continuation benchmark,
matching the Windows learning loop's ability to respawn and finish after a
HIT. It reports total HITs, clear status, HIT ticks, authority-empty frames,
Bomb use, and termination. Forced fail-close release rows and post-HIT states
are benchmark-only and `training_eligible=false`; they may identify new COW
neighborhoods but never enter the factual training corpus directly.

## Preserve a policy population, not one offline winner

The COW audit showed why a single argmax target is too narrow: 84.25%, 93.77%,
and 86.02% of the sampled Stage 3/4/5 actions respectively survived the branch
horizon, with roughly 15--17 survivable actions per checkpoint. A high-quality
population should retain different safe tradeoffs instead of collapsing all of
that support into one label.

`scripts/build_headless_policy_population.py` links each immutable model hash
to its exact scope, compatible clean source commit/binary, offline report, and
closed-loop manifests. Within each exact source and
difficulty/character/shot/stage scope it retains the non-dominated candidates
over worst observed survival, tick-limit rate, authority-failure rate, and,
when actually measured, HITs per 1000 ticks. Offline accuracy and action
entropy explain and diversify the archive; they do not dominate a candidate or
promote it. Incompatible weights remain historical evidence and are rejected
by the active queue. The online consumer may use consensus, uncertainty, or a
contextual selector only to rank the current native safe set.

Evidence tiers are intentionally strict:

- `offline-only` has no rollout claim;
- `first-failure-only` measures where a weight stopped but does not know its
  full-stage HIT rate;
- `continuation-evidenced` has HIT-counting rollouts after respawn;
- Windows shadow/canary remains the only promotion gate.

`high_quality_population` is stricter than the continuation evidence tier. A
candidate enters it only after at least two distinct seeds both finish as
natural no-HIT/no-Bomb stage clears. Interrupted runs, bounded survival, and
continued clears with any HIT stay in the research/Pareto evidence archive.
Any benchmark forced-release or authority-empty event also rejects NMNB status,
even when the continued physical trajectory happens to record zero HITs.

First-failure DAgger trajectories remain training-eligible because they end
before benchmark continuation or forced release. Corrective-pair weighting
applies to both an observed physical HIT and an authority-empty termination;
continued-HIT trajectories remain evaluation-only regardless of this signal.

The current archive contains 41 model artifacts and 12 historical
first-failure Pareto members across source builds. Only six Stage 3--5 members
are compatible with the active `8c3de1de63fd` / `ca4b2e7cb05e` runtime; the
older Stage 1/2/6 weights correctly fail the exact-source gate. There are still
zero high-quality continuation-evidenced members. This is an evaluation gap,
not a zero-HIT result. The six compatible candidates form the active
continuation evaluation queue; no teacher-baseline rerun is required.

| Stage | Weight (short SHA) | Candidate | Runs | Worst stop tick | Authority-failure rate | Tick-limit rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 3 | `1f029e289d8f` | COW survivable | 2 | 1,370 | 1.0 | 0 |
| 3 | `5a5df4c75081` | COW profile | 2 | 1,284 | 0.5 | 0.5 |
| 4 | `38c6cb48f49d` | COW spatial | 2 | 2,763 | 0.5 | 0.5 |
| 4 | `6290afae140e` | COW corrective-r2 | 2 | 1,882 | 1.0 | 0 |
| 5 | `0e7de641a0ed` | COW multiseed | 2 | 1,582 | 0.5 | 0.5 |
| 5 | `71a3688705dd` | COW survivable | 2 | 2,299 | 1.0 | 0 |

These are actual closed-loop weight outcomes, but they used the default
first-failure protocol. Consequently their HIT count is unknown after the stop
point; `tick-limit` means reaching the 3,000-tick evaluation budget, not stage
clear. Future continuation runs fill that missing column without changing or
overwriting the historical evidence.

## Preserved Linux HIT-continuation baseline

The six concurrent seed-23 teacher baselines were intentionally stopped and
not rerun. `summarize_headless_continuation.py` recovered every complete JSON
record from their interrupted gzip streams. All six remain
`training_eligible=false` and termination is `interrupted-partial`.

| Stage | Observed ticks | HITs | HIT ticks | HIT/1000 ticks | Longest no-HIT interval | Forced release rows |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 88,296 | 2 | 11,302; 16,150 | 0.0227 | 72,147 | 25 |
| 2 | 16,682 | 1 | 10,191 | 0.0599 | 10,190 | 34 |
| 3 | 15,744 | 1 | 2,283 | 0.0635 | 13,462 | 6 |
| 4 | 8,175 | 0 | -- | 0 | 8,175 | 1 |
| 5 | 20,184 | 3 | 4,069; 16,545; 17,051 | 0.1486 | 12,476 | 18 |
| 6 | 28,583 | 3 | 17,955; 20,104; 21,926 | 0.1050 | 17,954 | 6,672 |

The aggregate is 177,664 transitions, 10 HITs, zero Bombs, and 6,756 forced
release rows. Stage 4's zero means zero within 8,175 observed ticks, not a
stage clear. Stage 6's large forced-release count is a strong model/authority
improvement signal and must not be hidden by its relatively low HIT rate.

## Natural Practice completion and continuation semantics

Headless runtime `1350819` exports the authoritative message interpreter's
`STAGERESULTS` event as `stage-clear-success`. It does not infer a phase, boss,
frame, or route. A Stage 1--6 × seeds 7/23 differential compared the preceding
runtime over 3,000 ticks per case: all 12 physical-observation streams and
timeline clocks were identical. The same audit found that the old diagnostic
`source_context.next` could read beyond the coherent timeline section; the new
runtime validates the ECL section boundary/opcode/size and emits `null` for an
unknown next instruction.

`--continue-after-hit` now reserves one simulation-only life before Game Over.
Every `Player::Die` still increments the physical death counter; the reserve
only lets the authoritative respawn path reach the end of the same Practice
stage. A fixed stationary action cleared all six infrastructure trials at
18,306 / 21,455 / 29,467 / 33,734 / 33,135 / 44,236 ticks while recording
32 / 41 / 54 / 60 / 47 / 73 HITs and zero Bombs. Default mode remained
first-HIT fail-close. Continued-HIT runs are evaluation-only and never factual
training data.

Continuation rollout no longer computes the unused offline teacher label. A
same-model Stage 1 seed-23 differential compared the old and optimized paths:
the first 2,998 complete transition records had the same physical digest and
all 2,999 published actions were identical. Only the bounded trial's final
successor differed because `max_ticks=3000` marks that observation terminal.
The optimized 3,000-tick run took 44.97 CPU seconds on the shared VPS. It still
computes candidate-relative profiles and performs both native certification
and the fresh issue check.

This also invalidates the apparent millions-of-ticks survival denominator in
the deadline-stopped old-runtime runs. Each of those six weights incurred
exactly three battle HITs before Game Over, then idled in menu/empty state.
Their first-HIT ticks and forced rows remain useful diagnostics, but their
post-Game-Over HIT rate is not policy evidence.

## Current-source six-stage baseline and corpus yield

Runtime `1350819` generated eight factual seed runs (15--22) per exact
Reimu-A/Lunatic stage scope. All 48 runs and 210,180 transitions passed the
independent compact-corpus audit: factual successor, native-legal action,
propensity, and Bomb-free ratios were all 100%. This is row validity, not route
coverage. With a 6,000-tick collection bound, native authority stopped 0 / 4 /
5 / 4 / 8 / 6 runs in Stages 1--6 respectively. In particular, every Stage 5
factual run ended at an early authority dead-end, so the corpus supplies no
teacher-supported late-stage distribution.

The first current-source distilled weights were then run with HIT continuation
to natural stage completion on unseen seeds 23/24. Every run reached
`stage-clear-success` and used zero Bombs, but none qualified as a two-seed
NMNB stage candidate:

| Stage | Natural ticks (seed 23 / 24) | HITs | Forced release rows |
| --- | --- | --- | --- |
| 1 | 18,306 / 18,306 | 6 / 6 | 98 / 97 |
| 2 | 21,455 / 20,650 | 5 / 5 | 75 / 80 |
| 3 | 29,638 / 29,370 | 7 / 6 | 71 / 47 |
| 4 | 35,055 / 35,263 | 14 / 15 | 126 / 104 |
| 5 | 33,135 / 32,961 | 22 / 16 | 202 / 143 |
| 6 | 44,236 / 44,236 | 25 / 25 | 270 / 407 |

The bootstrap policies are substantially better than the fixed stationary
infrastructure baseline, but imitation accuracy did not solve the sparse
irreversible decisions. The first iterative Stage 1 DAgger weight moved one
new seed from six HITs to a full 18,306-tick no-HIT/no-Bomb clear; its other
new seed still had four HITs. This is useful causal evidence for iterative
failure-neighborhood aggregation, but one seed is not promotion or
high-quality-population evidence.

The paired factual-tail COW expansion completed all 48 seed runs: 432 audited
checkpoints and 7,331 complete native-action outcomes. A unique strict best
action existed at 84.5% of checkpoints, while the local teacher/factual action
was in that best set at only 8.3%. Per-stage factual-best ratios were 0%,
2.8%, 8.3%, 9.7%, 18.1%, and 11.1%. These labels support survivable-set and
grouped-value experiments, but the first value rankers again fit training
groups perfectly while reaching only 0--27.8% held-out top-1; they were
discarded by first-failure rollout rather than promoted from training fit.

As of the next exact-source iteration, the best measured continuation member
for each stage is still a research incumbent, not a promoted policy:

| Stage | Ranker SHA-256 prefix | Unseen seeds | HITs | Forced release rows |
| --- | --- | --- | --- | --- |
| 1 | `7a816b1f4e3e` | 73 / 74 | 2 / 0 | 24 / 4 |
| 2 | `a4245bdfc8d3` | 71 / 72 | 0 / 0 | 20 / 14 |
| 3 | `4b4e9235f816` | 75 / 76 | 1 / 2 | 4 / 9 |
| 4 | `02a85a9fdf7d` | 75 / 76 | 3 / 3 | 27 / 17 |
| 5 | `911a325c4468` | 79 / 80 | 9 / 12 | 60 / 120 |
| 6 | `3bc9e35ea979` | 51 / 52 | 9 / 12 | 83 / 146 |

All twelve runs completed their natural Practice Stage with zero Bombs, but
none of the six two-seed stage candidates is NMNB. A different Stage 1 member
did produce one strict natural 0-HIT/0-forced clear on seed 26; its paired seed
had four HITs. A later 600-tick unique-best COW member independently produced
a strict 0-HIT/0-forced clear on seed 43, but its paired seed had two HITs and
40 forced releases. A Borda ensemble of two strict-clear-bearing members then
produced another strict clear on seed 56, but its paired seed had three HITs
and 66 forced releases. These three distinct model/seed successes remain
population evidence rather than promotion; consensus generated a new feasible
mode but did not make success repeatable.
The Stage 1 table member matches the preceding `269675ae8805` member's 1/1 HIT
pair while reducing total forced releases from 38 to 32. The table intentionally
uses closed-loop HIT and forced-release counts instead of offline accuracy to
select the current experiment order.

The Stage 2 row is the first two-seed natural 0-HIT continuation result, but its
20/14 forced-release rows keep it outside strict NMNB: at those states the
native safe set was empty and the benchmark released input to let the physical
stage continue. The same weight's preceding seeds 61/62 had 1/0 HIT and 60/15
forced rows, so the fresh pair is stronger evidence but still an authority-gap
target rather than a clear. The preceding `37d75f340fd1` Stage 2 member (0/3
HIT, 27/48 forced) remains a different per-seed trade-off, and `ac7edd49d803`
(3/1 HIT, 53/62 forced) retains additional trajectory diversity. The preceding
`8c7a94fd3b4f` Stage 4 member
(7/10 HIT, 54/70 forced) retains trajectory diversity but no longer leads
either closed-loop objective. The later Stage 4 HIT-primary member
`f8a94d1689c4` had 2/7 HIT and 8/78 forced rows. Its sibling
`3763b2518d4a` had 7/5 HIT and 39/42 forced rows. The repaired first-failure
unique-best successor `02a85a9fdf7d` then reached 3/3 HIT and 27/17 forced
rows. It reduces aggregate HIT from nine to six and forced rows from 86 to 44,
so it supersedes both older table members on aggregate metrics. The older
2-HIT single-seed mode remains population diversity rather than being deleted.
These are explicit trade-offs, not overwritten checkpoints.
The corresponding full-label Stage 4 candidates regressed to 4/11 HIT with
14/73 forced rows and 6/11 HIT with 65/83 forced rows. More labels at the same
counterfactual weight therefore did not dominate the partial correction.

The Stage 3 bootstrap-DAgger COW member `d8472ed568fe` is a strict closed-loop
improvement over the preceding bootstrap on these metrics: total HIT fell from
13 to 9 and total forced releases from 118 to 56 on its unseen seed pair. It
was superseded in both metrics by `b9b3c842d98e`, which reduced the pair to
3/2 HIT and 21/30 forced rows. A sibling `07eadef52e0b` unique-best candidate
also had five total HITs, including a zero-HIT seed, but that seed still
required one forced-release row and its pair required 61. It is retained as a
different feasible mode rather than mistaken for a strict clear. The repaired
survivable-source successor `4b4e9235f816` then reduced the pair to 1/2 HIT
and 4/9 forced rows. It strictly improves the preceding table row on both
aggregate closed-loop metrics and becomes the current Stage 3 incumbent.

The Stage 6 bootstrap-DAgger COW member `3d4f9b3e5599` first reduced its
unseen-pair HIT total from 37 to 32 (13/19 rather than 18/19), but raised forced
releases from 224 to 347. The later `3bc9e35ea979` unique-best member reduced
the pair again to 9/12 HIT with 83/146 forced rows. It dominates the bootstrap
COW member and becomes the HIT-primary table row, while the original
`85471d1dbe2a` weight remains lower by five aggregate forced rows.

The newer Stage 5 survivable-set member `88b334b4ad78` retained the previous
pair's 22 total HITs while reducing total forced rows from 167 to 146. Its
8/14 HIT split is less balanced than the preceding member's 11/11, so the old
weight remains a robustness/diversity candidate even though the new member is
better on the aggregate continuation Pareto metrics.
The sibling unique-best member returned 11/11 HIT and 86/105 forced rows, so
it did not improve either the old balanced member or the new aggregate member.
The later `911a325c4468` survivable member reduced the pair to 9/12 HIT on
seeds 79/80, becoming the new HIT-primary Stage 5 incumbent despite raising
forced rows to 60/120. Its unique-best sibling regressed to 10/14 HIT with
75/123 forced rows. Both completed naturally with zero Bombs; neither is a
strict clear, and the older lower-forced and balanced members remain in the
population.

A second COW batch targeted the last 600 decisions of the bootstrap policy's
first-failure trajectories. All 10 files were valid: 90 checkpoints and 1,441
complete outcomes. A unique strict best action existed at 83.3% of checkpoints,
while the local teacher and factual action were strict-best at only 20.0% and
15.6%. Per-stage unique-best ratios for Stages 2--6 were 83.3%, 88.9%, 83.3%,
66.7%, and 94.4%. Early 240-tick survivable-set models changed few local labels
and did not improve Stage 5 or 6 continuation HITs, motivating event-centered
600/1200-tick branches rather than more short-horizon imitation weight.

The first event-centered 600-tick cycle then sharded each two-seed failure
pair into 14 independently resumable checkpoint tasks. Every completed file
passed the same dynamic-branch audit:

| Stage | Checkpoints | Outcomes | Unique strict best | Local teacher best | Factual best |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 14 | 200 | 78.6% | 28.6% | 35.7% |
| 2 | 14 | 233 | 100.0% | 7.1% | 7.1% |
| 3 | 14 | 232 | 92.9% | 7.1% | 0.0% |
| 4 | 14 | 229 | 92.9% | 7.1% | 7.1% |
| 5 | 14 | 221 | 92.9% | 0.0% | 14.3% |
| 6 | 14 | 229 | 92.9% | 0.0% | 0.0% |

These are deliberately failure-neighborhood yields, not corpus-wide action
accuracy. They show that the native-factual/local choices are especially weak
near irreversible failures and justify collecting more such neighborhoods.
They do not show that a pointwise strict-best classifier will improve a whole
stage: Stage 4 first-failure checks regressed despite 92.9% unique targets, so
both survivable-set and unique-best candidates still require complete unseen
seed continuation rollout.

A subsequent Stage 6 cycle targeted the failure neighborhoods of the lower-HIT
but higher-forced bootstrap-DAgger member. All 14 sharded files were valid and
contained 221 complete outcomes. Only 71.4% of checkpoints had a unique strict
best action; the local teacher and factual action were best at 21.4% and 7.1%,
respectively. This distribution is materially more ambiguous than the first
Stage 6 event batch and is retained as a distinct population stratum rather
than merged into a single assumed ground truth. Its survivable-set and
unique-best classifiers changed 6/14 and 10/10 usable local labels,
respectively; unseen-seed continuation remains the promotion gate.

The same two Stage 6 generations also supplied 28 grouped action-value
checkpoints for an exact-source LambdaMART diagnostic. It fit the 21 training
groups perfectly but achieved only 14.3% top-1 on the seven checkpoints of a
complete held-out seed. This is direct evidence that sparse counterfactual
value fit still does not generalize reliably; the model remains only a diverse
rollout candidate, not evidence that value learning has surpassed corrective
imitation.

The first 1200-tick Stage 1 corrective candidate then completed two new
continuation seeds with 1/1 HIT and 36/19 forced-release rows. Both HITs were
late (around tick 15.8k of 18.3k), making those event neighborhoods useful for
the next COW cycle, but the member is dominated by the existing 1/1-HIT
incumbent's 24/8 forced rows and is not promoted.

Repeating that Stage 1 incumbent on fresh seeds 71/72 produced 3/2 HIT and
26/57 forced-release rows. This rejects the tempting interpretation that its
earlier 1/1 pair had nearly solved the stage. The corresponding first-failure
runs still contributed 10,779 and 15,762 trustworthy pre-failure decisions,
and their first event neighborhoods entered a separate 1200-tick COW batch.
The following 445,904-decision survivable model changed 11 of 14 long-COW
labels and reached 2/0 HIT with 24/4 forced rows on seeds 73/74. It preserves
the population's two-HIT total while reducing aggregate forced rows from 32 to
28, so it becomes the aggregate table member; the old 1/1 split remains the
more balanced robustness member. The sibling unique-best model reached 1/1
HIT with 22/10 forced rows.

The next Stage 2 live-snapshot corrections also regressed and were rejected:
the survivable target produced 2/3 HIT with 46/88 forced rows on seeds 69/70,
while unique-best produced 2/2 HIT with 39/68 forced rows. In Stage 6, the
newer survivable and unique-best pair both totaled 27 HIT (16/11 and 15/12),
with 296 and 271 forced rows respectively. The sparse grouped-value candidate
was substantially worse at 30/33 HIT and 333/384 forced rows. These outcomes
reinforce that COW fit and offline top-1 only order experiments; closed-loop
continuation remains the decision criterion.

The subsequent Stage 2 45-file snapshots reached 2/1 HIT with 69/31 forced
rows (survivable) and 1/0 HIT with 54/21 forced rows (unique-best). Their
56-file siblings reached 1/1 HIT with 59/62 forced rows and 2/2 HIT with 66/57
forced rows. All four are rejected relative to the same incumbent's fresh
0/0-HIT, 20/14-forced pair. Most labels in those directories occurred after
the first benchmark-forced action and correctly remained unmatched by the
trainer.

Stage 3's next unique-source siblings reached 2/4 HIT with 14/35 forced rows
and 3/2 HIT with 29/27 forced rows. The latter matches the incumbent's five
total HIT but has five more aggregate forced rows; the former trades one extra
HIT for two fewer forced rows and remains diversity evidence only. Stage 4's
siblings reached 3/9 HIT with 17/105 forced rows and 8/5 HIT with 61/40 forced
rows, both worse than the existing 2/7-HIT, 8/78-forced table member. Stage 5
reached 15/10 HIT with 101/74 forced rows and 15/7 HIT with 109/54 forced rows;
the unique model ties the old 22-HIT total but has 17 more forced rows. Stage 6
reached 15/18 HIT with 147/152 forced rows and 18/20 HIT with 233/169 forced
rows. None is promoted.

### First-HIT reconstruction boundary

The first continuation-derived event selector included every HIT in a run.
That was invalid for correction: the forkserver's factual prefix stops at the
first physical HIT, so checkpoints after it cannot be reconstructed from the
same physical trajectory. A later audit tightened the same rule around native
authority: once the benchmark publishes its first forced release, every later
state depends on that evaluation-only action. Event selection therefore stops
at whichever occurs first: a physical HIT or forced authority release. The old
partial directories remain diagnostic-only and are not training inputs.

Two model rollouts can also have identical timestamp/stage/seed basenames.
The first multi-model batch consequently reported 16 successful tasks while
only eight files survived: different factual runs had overwritten the same
output path. Those directories are diagnostic-only. Conflicting basenames now
receive a stable full-run-path digest, and worker submission is round-robin
across runs so an early frozen snapshot covers multiple seeds rather than one
long run prefix.

The first repaired `prehit-v2` batches completed without a replay failure and
passed independent native-action-table audits:

| Source stage/policy | Checkpoints | Outcomes | Unique strict best | Local/factual best |
| --- | ---: | ---: | ---: | ---: |
| Stage 2 unique-best r4 | 56 | 995 | 94.6% | 10.7% |
| Stage 3 unique-best r3 | 12 | 187 | 75.0% | 8.3% |
| Stage 4 unique-best r4 partial | 8 | 119 | 75.0% | 25.0% |
| Stage 5 survivable r4 | 11 | 152 | 72.7% | 18.2% |
| Stage 6 unique-best r3 | 8 | 121 | 87.5% | 12.5% |

Stage 2 also shows why an event count is not the same as trainable yield. Only
15 of its 56 audited checkpoints match first-failure corpus observations; the
remaining post-authority states are preserved as diagnostics rather than
silently treated as factual training data. Exact COW file paths and hashes are
frozen in every new training report before decoding begins.

The zero-HIT Stage 2 seeds 71/72 made the efficiency gap especially visible.
The old selector launched 118 valid but mostly post-forced diagnostic branches;
the corrected first-failure selector retained eight balanced, trainable
checkpoints with 134 complete outcomes. A unique best existed at 87.5%, while
the local teacher and factual action were best at 0%. Both ordinary and
strong-weight corrective populations are evaluated independently rather than
assuming that more correction weight must improve a route.

The ordinary corrected Stage 2 r8 population illustrates why the strict gate
also counts forced releases. Its survivable member reached 3/0 HIT with 67/22
forced rows on seeds 77/78. The unique-best member reached 0/0 HIT but still
needed 27/16 forced rows. It is retained as a second reproducible zero-HIT
population member, but it does not replace the 0/0-HIT incumbent because its
aggregate forced count is 43 rather than 34. The native authority gap remains
the limiting objective; zero observed collision alone is not NMNB evidence.

The incumbent's 34 forced rows are now attributed rather than treated as one
opaque policy metric. All 34 are newly lethal lasers without angular history:
12 each from slots 0 and 3, and 10 from slot 6. None are an empty native safe
set. The authoritative snapshot at the first such failure contains a newly
spawned active laser with `start_time=0`, `graze_delay=0`, and no preceding
physical angle sample. Assuming zero rotation would weaken the uncertainty
contract, so this remains an observation-authority gap rather than a learned
policy correction target. Continuation summaries now aggregate
`benchmark_forced_reason_rows` and separately count unattributed rows from
interrupted partial streams.

Increasing the same correction weight from 32 to 128 did not close this gap.
The strong-weight survivable member reached 2/2 HIT with 40/84 forced rows,
and its unique-best sibling reached 3/1 HIT with 103/41. A Borda ensemble of
the incumbent and ordinary r8 unique member also regressed to 1/3 HIT with
44/75 forced rows. All remain archived population evidence, not promotions.

The repaired Stage 5 r5-pair COW batch contributed 16/16 valid checkpoints
and 240 complete branches. A unique strict best action existed at 87.5% of
checkpoints, while both the local teacher and factual action were best at only
18.75%. These labels entered separate survivable-set and unique-best r6
training jobs; they are not promoted until both natural unseen-seed
continuations complete.

The next completed population checks did not replace their incumbents. Stage
1 r7 survivable reached 3/1 HIT with 76/15 forced rows and unique-best reached
3/2 with 38/36. Stage 3 r6 survivable matched the incumbent's three aggregate
HITs at 2/1 but raised forced rows from 13 to 17; unique-best reached 3/2 with
19/44. These late failures also reject promotion from an encouraging partial
snapshot: the Stage 1 unique pair was still 0-HIT/0-forced after roughly 10.7k
ticks each, then failed before the natural 18,306-tick stage end.

The following generation remained a negative but useful population sweep.
Stage 2 r9 reached 1/1 HIT with 59/60 forced rows (survivable) and 1/2 with
20/35 (unique-best), so neither replaced the 0/0-HIT incumbent. Stage 3 r7
reached 4/2 with 19/11 and 4/4 with 16/24. Stage 4 r7 reached 4/6 with 27/71
and 9/4 with 107/27. Stage 5 r6 kept 22 aggregate HITs in both modes, at
12/10 with 73/85 and 10/12 with 81/146. Stage 6 r6 regressed to 19/15 with
234/132 and 14/21 with 114/163. These closed-loop results reject the models;
their completed first-failure trajectories remain eligible inputs for the
next causal-label generation.

Stage 1 r8 also failed its unseen-seed continuation gate. Its survivable
member reached 3/3 HIT with 58/54 forced rows on seeds 77/78, while the
unique-best sibling reached 2/2 HIT with 40/41 forced rows. Both are dominated
by the 2-HIT, 28-forced incumbent and remain archived only as corrective-data
sources and diversity evidence.

An exact ordinal action-return experiment then trained LambdaMART on 56 Stage
1 COW groups and 913 complete branch outcomes. Splitting by seed rather than
by checkpoint exposed severe generalization failure: the model achieved 100%
top-1 and 1.0 MRR on seeds 73/74, but only 17.9% top-1 and 0.305 MRR on held-out
seeds 75/76. This is a useful modern offline-RL negative result, not a rollout
candidate. Exact simulated return is still too sparse and seed-specific to
override the corrective classifiers, and it will not be promoted or ensembled
without materially better complete-seed validation.

The concurrent Stage 2 r10 and Stage 4 r8 checks were also rejected. Stage 2
reached 2/0 HIT with 70/27 forced rows (survivable) and 2/1 with 71/38
(unique-best), both worse than the 0/0-HIT incumbent. Stage 4 reached 6/7 HIT
with 39/37 forced rows and 2/12 with 13/91, both worse than its 3/3-HIT
incumbent. All used zero Bombs. Their completed first-failure trajectories
remain causal-label sources; the population table is unchanged.

The next completed COW batches remained useful despite those rollout losses.
Stage 2 r10 contained 16 valid checkpoints and 241 outcomes, with 87.5% unique
strict best, 12.5% local-teacher best, and 12.5% factual-action best. Stage 3
r8 contained 16 valid checkpoints and 246 outcomes, with corresponding ratios
of 93.75%, 25%, and 12.5%. Stage 4 r7 contained 16 valid checkpoints and 244
outcomes, with ratios of 75%, 25%, and 18.75%. These are new supervision
signals, but only their unseen-seed descendants can establish improvement.

That Stage 3 r8 descendant did not establish an improvement: survivable
reached 2/2 HIT with 18/14 forced rows, and unique-best reached 3/5 with
37/78. Both used zero Bombs and are rejected relative to the 1/2-HIT,
4/9-forced incumbent. The next Stage 5 survivable member did improve the
HIT-primary objective from 22 to 21 aggregate HITs at 9/12, although its
60/120 forced rows are worse than the previous incumbent's 58/88. It becomes
the experiment-order incumbent while both lower-forced and balanced members
remain in the Pareto population. Its unique sibling reached 10/14 HIT and
75/123 forced rows and is rejected.

Two more complete COW batches quantify the continuing mismatch. Stage 1 r8
contained 28 valid checkpoints and 462 outcomes, with 85.7% unique strict
best and only 10.7% each local-teacher and factual-action best. Stage 6 r7
contained 16 valid checkpoints and 227 outcomes, with ratios of 68.75%,
18.75%, and 18.75%. They are valuable corrective corpora, but their descendants
still require full natural continuation before any population update.

The Stage 6 r7 descendants were then decisively rejected. Survivable reached
19/20 HIT with 147/150 forced rows, and unique-best reached 14/14 HIT with
104/101; both used zero Bombs and are worse than the 9/12-HIT incumbent. The
Stage 5 r7 COW batch completed 16 valid checkpoints and 250 outcomes, with
68.75% unique strict best, 18.75% local-teacher best, and only 6.25% factual
best. Its r8 descendants remain behind the natural-continuation gate.

## Portability to TH08

The COW mechanism itself is a TH06 adapter. The learning contract is portable:
coherent digest, explicit scope, native legal candidates, dynamic branch
continuation, outcome table, complete-seed groups, grouped value ranker, and
rollout promotion. TH08 can provide a different snapshot/simulator adapter and
feature encoder while reusing the outcome schema, auditor, training split, and
promotion logic. See `COUNTERFACTUAL_LEARNING.md` for the strict adapter
boundary.

## Open gates

- Finish the multi-seed counterfactual corpus and measure held-out value
  ranking on at least two complete seeds.
- Test COW-corrected imitation and value candidates on entirely unseen RNG
  seeds; keep best-by-rollout, not latest-by-training.
- Run the per-stage Pareto queue with ranker HIT continuation; attach HIT ticks,
  rates, forced-release counts, and immutable weight hashes to the archive.
- Extend successful 3000-tick candidates to actual headless stage clears.
- Compose independently promoted Stage 1–6 policies into the Reimu-A Lunatic
  route only after every stage clears headlessly without HIT/Bomb.
- Differentially replay identical action streams against shipped Windows TH06.
- Use Windows physical play as the final NMNB evidence gate.
