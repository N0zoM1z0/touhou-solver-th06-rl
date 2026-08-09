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
