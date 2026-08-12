# Autonomous learner generation 6 result

## Verdict

Generation 6 produced an `effective-pilot-signal` in its first predeclared
complete-Stage-6 deployment panel. The frozen IQL actor target recorded 25
physical HITs over three natural-RNG original-Wine Stages, versus 28 for the
exact incumbent over three Stages. This is a directional improvement of three
HITs in aggregate, or one HIT per Stage.

This result rejects the conclusion that the new learner is already ineffective.
It is the first current-generation policy whose offline sign, deployable target,
native implementation, and fresh original-Wine Stage-6 sign agree. It does not
promote the policy: three Stages per arm cannot separate a one-HIT mean effect
from the observed natural-RNG outcome variance. Generation 6 is therefore
`promising; confirmation required`, not `proven effective`.

## Frozen original-Wine evidence

The pilot contract SHA-256 was
`b3ecb0720fa1ac0fb0f4ab8dbb90b8f286812b6468f3028d00a9f0672e6301cd`.
It ran from source commit `2043271`, at original 60 Hz pacing, with natural
unread game RNG, complete Practice Stage 6, HIT continuation, zero Bomb, no
corpus output, and the immutable seven-member actor in both arms. The incumbent
arm shadow-scored but published the reactive baseline; the candidate arm used
the already qualified propensity-bounded target.

| Block | First arm | Second arm | Incumbent HIT | Candidate HIT | Incumbent minus candidate |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | incumbent | candidate | 3 | 10 | -7 |
| 1 | candidate | incumbent | 11 | 8 | +3 |
| 2 | incumbent | candidate | 14 | 7 | +7 |
| **Total** | | | **28** | **25** | **+3** |

All six runs passed every frozen runtime and integrity gate. Candidate Stages
made 111/100/115 proposals and 3/2/3 actual interventions, so all three Stages
exercised the treatment. Incumbent intervention was zero. Across both arms the
resident policy scored 25,963 option boundaries; per-run p95 ranged from
`2.9619` to `3.0019 ms`, only 12 decisions exceeded 4 ms, and none missed the
60 Hz deadline. All actions remained in the native safe vocabulary, immutable
state and binary hashes matched, physical HIT accounting completed, and every
private Wine prefix cleaned.

The result report SHA-256 is
`d4eba15f9db2881a39f6809072267eb22a0602bcdc6519a30b8a451a20a97b43`;
the resumable ledger SHA-256 is
`2864f3a1f55d911d52ba1e0fff2a679dcce40220ff9bd33bfa56795affb9cf44`.
The six report hashes are preserved in the ledger and in the pilot design
record.

## What was learned

The fresh deployed sign agrees with the reusable-corpus deployable audit,
which estimated `-1.3405 HIT/Stage` on historical Stage 6. The live estimate is
`-1.0 HIT/Stage`. That agreement is encouraging because the offline learner
was fitted without these six trajectories and the live schedule was committed
before any of their outcomes existed.

The uncertainty is nevertheless material. Incumbent outcomes ranged from 3 to
14 HIT and one incumbent Stage beat every candidate Stage. Block contrasts
were `-7`, `+3`, and `+7`; only two of three blocks favored the candidate. The
small aggregate rule was intentionally directional and the machine result sets
`authorization_eligible: false`. Neither the historical single 17-HIT run nor
the Stage-4 wiring-canary HIT counts are valid controls for this panel.

The correct decision is therefore to retain the Generation-6 learner and its
native serving path for a separately frozen larger confirmation or autonomous
learning round. It would be incorrect either to discard the learner after this
positive panel or to deploy it as a promoted policy. A confirmation must keep
natural RNG and complete Stage HIT count as final evidence and must predeclare
its sample/stopping rule before play.

## Data-plane consequence

The pilot created no training corpus and changed no learner, fit, propensity,
support, action, or reward. Its outcome is a new evaluation fact, not a private
training split. The permanent relation remains:

`one immutable Wine corpus -> many learner/framework experiments -> many immutable fits`

Every compatible registered episode remains reusable by future learners.
Changing or repairing an offline RL framework must first replay those same
facts; it must not copy, relabel, resample, or recollect them because a model
result was inconvenient. New Wine collection is justified only by a declared
capability gap or a predeclared autonomous evidence round.
