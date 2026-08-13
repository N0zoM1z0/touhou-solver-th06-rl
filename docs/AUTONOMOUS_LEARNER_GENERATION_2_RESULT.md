# Autonomous learner generation 2 result

## Decision

Generation 2 is **ineffective under its predeclared evidence budget**. The
round-2 candidate passed grouped offline validation, native shadow, and a
paired fixed-RNG Wine canary, but it did not lower aggregate physical HITs in
the authoritative normal-speed, natural-RNG, complete-Stage Wine evaluation:

- baseline: `10 + 7 = 17` physical HITs;
- candidate: `7 + 11 = 18` physical HITs;
- declared effect: `17 - 18 = -1` HIT;
- verdict: `ineffective` because candidate aggregate HITs were not strictly
  lower.

This is a negative learner result, not evidence of a safety or collection
failure. The Wine-only data path, grouped fit, compact native scorer, shadow
gate, canary gate, resumable orchestration, and final evaluator all completed
their declared contracts. Generation 2 is now frozen. Its failure may not be
converted into a hand-authored action, RNG, frame, phase, reward, threshold,
or data-distribution exception.

The ignored, machine-readable decision is
`artifacts/autonomous-wine-generation-2/generation.json`; the authoritative
final report is
`artifacts/autonomous-wine-generation-2/full-stage/report.json`.

## Evidence contract

The experiment used the original retail TH06 executable under Wine 11.0 on
Lunatic Practice Stage 6. Training and fixed-RNG canaries continued after
physical HITs until the complete stage ended. Final evaluation omitted a
diagnostic RNG seed, ran at normal Wine timing, disabled corpus collection,
alternated baseline and candidate, and retained continue-on-HIT lives only.

The collection budget was fixed at eight complete stages. The first fit
boundary was six stages; after its canary rejection, exactly two more stages
were collected for the second and final fit. Each fit held out the final two
whole episodes. No fit occurred inside a stage and no canary or final-evaluation
trajectory was fed back into the learner.

The accepted collection outcomes were:

| Episode | Fixed game RNG | Physical HITs | Decisions | Safe exploration |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 25245 | 21 | 29,065 | 2,622 |
| 1 | 103 | 25 | 27,780 | 2,557 |
| 2 | 1082 | 30 | 26,351 | 2,351 |
| 3 | 36448 | 35 | 27,623 | 2,509 |
| 4 | 54677 | 25 | 25,541 | 2,344 |
| 5 | 21944 | 24 | 29,408 | 2,680 |
| 6 | 21285 | 30 | 28,534 | 2,667 |
| 7 | 33289 | 31 | 25,724 | 2,389 |

Together they contain 220,026 policy decisions, 20,119 non-incumbent safe
exploration decisions, and 221 factual physical HITs. A complete collection
stage took 578 to 741 seconds, with a mean of 658 seconds. The complete
generation-2 artifact set occupies about 4.8 GiB and remains ignored rather
than being committed as source.

Episode 0 is bound to repository commit `1ab6c4a`; episodes 1 through 7 and all
fit/evaluation runs are bound to `4c5ee10`. The only intervening source change
reclassified a next-frame source-unsafe input lease from infrastructure loss to
a fail-closed control dead-end. Episode 0 recorded neither an authority
infrastructure failure nor an in-flight unsafe lease, and the native library
and behavior policy were unchanged. The prior partial episode 0 and rejected
episode 1 remain archived as `.incomplete-001` evidence; neither is in the
accepted learner set.

## Learner and promotion evidence

| Gate | Round 1: 6 episodes | Round 2: 8 episodes |
| --- | ---: | ---: |
| grouped train / validation rows | 51,536 / 30,068 | 81,604 / 23,992 |
| local-support validation coverage | 98.999% | 99.000% |
| held-out factual-cost RMSE | 0.4334 | 0.3705 |
| held-out constant RMSE | 0.1653 | 0.2025 |
| native shadow decisions | 30,068 | 23,992 |
| native shadow proposals | 45 | 28 |
| invalid publications | 0 | 0 |
| scorer p95 per-decision latency | 1.83 ms | 2.90 ms |
| fixed-RNG baseline / candidate HITs | 36 / 40 | 33 / 27 |
| candidate canary overrides | 18 | 42 |
| canary decision | rejected | passed |

Both offline fits satisfied the predeclared integrity/support gates, but their
held-out factual-cost RMSE was worse than the constant comparator. RMSE was
deliberately not a promotion metric, so this did not authorize or reject the
policy by itself. The outcome gates behaved as intended: round 1 was rejected
after losing its fixed-RNG canary by four HITs; round 2 was authorized after
winning a different fixed-RNG canary by six HITs.

The round-2 fixed-RNG win did not reproduce in final natural-RNG aggregate.
The final candidate made only two high-confidence overrides in each complete
stage, four across 63,055 decisions. This establishes that compact online
deployment and conservative abstention work, but it does not establish a
repeatable policy advantage. With only two natural-RNG stages per arm, the
effect estimate is noisy; nevertheless the predeclared stopping rule is
unambiguous and the generation-2 verdict remains `ineffective`.

## Hard-empty source follow-up

The source-bound audit was run on all eight accepted complete-stage corpora
against GensokyoClub/th06 commit
`cc475a0bc3fef38683b0f02224c87ddba0a021d9`. It examined 1,468 recorded true
Hard-empty decision roots:

- recorded and recomputed conservative masks agreed at every root;
- all 1,468 roots remained empty with the shipped game's margin-0 geometry;
- zero roots were closures caused only by the repo's 0.35 px uncertainty
  margin;
- 809 roots were followed by a physical HIT after 1 to 9 frames, mean 2.42;
- 659 roots recovered a native safe set after 1 to 76 frames, mean 6.20;
- every run was a complete stage and every audit gate passed.

These rows are not independent failure episodes; consecutive control frames
may each be a root. They do demonstrate that the conservative-to-source-exact
fallback removed margin-only false Hard-empty reports and that the remaining
events have the intended fail-closed semantics. There is no evidence-backed
Hard-empty infrastructure repair to make after this experiment.

The ignored detailed report is
`artifacts/autonomous-wine-generation-2/hard-empty-source-audit-v2.json`.

## What the result permits

The evidence supports retaining the Wine-only outcome source, grouped episode
boundaries, hard safety ownership, immutable compact scorer, and promotion
state machine. It does **not** support promoting the round-2 model or extending
the generation-2 evidence budget after seeing its result.

A future attempt must be declared as a new learner generation with a general,
game-neutral algorithmic hypothesis and frozen gates before collecting its
evaluation outcomes. It may improve representation, offline value estimation,
uncertainty, exploration, or sequential credit assignment as an algorithm. It
may not mine these failed trajectories to add TH06 spell, frame, RNG, action,
or failure-location exceptions. The same boundary must remain reusable for a
future TH08 adapter.
