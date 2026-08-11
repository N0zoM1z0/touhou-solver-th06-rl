# Stage 6 enemy-motion and native-set audit (2026-08-10)

## Decision

The panel-2 source replay had a real offline-infrastructure error: its enemy
records omitted already-observed interpolation state, and the solver therefore
projected the current enemy velocity as constant. The source export and
headless geometry have been corrected. This restores local native-safe-set
agreement for the retained panel, but it creates no policy candidate and does
not make reconstructed source a promotion domain.

The final policy gate remains lower HIT count in alternating complete natural
original-retail Wine Stage 6 runs against frozen UCB, ultimately including
repeatable 0-HIT clears. First-failure prefixes, this differential, and COW are
reject-only filters. Full-Stage HIT continuation remains evaluation-only and
cannot enter training.

## Symptom before correction

The first same-frame audit reconstructed every recorded retail Hard set under
the retail publication envelope `(0,1,2,3)`. Linux source and MinGW
source-under-Wine produced the same result, but both disagreed with retail in a
short repeatable interval:

| Run | Common snapshots | Equal | Different | Divergent frames | Audit SHA-256 |
| --- | ---: | ---: | ---: | --- | --- |
| r4 | 6,535 | 6,511 | 24 | 3661--3684 | `38e9f7c47f6b5e0d21953423e7059185dad3309fad9a1354c636b650e1b36665` |
| r5 | 6,184 | 6,151 | 33 | 3647--3679 | `7c3276d8c907a7e84f576af54bdbd049e692bc436780c9cad294683fa4f6a53c` |
| r6 | 6,401 | 6,375 | 26 | 3767--3792 | `c778986b647bd0a07199b4fb1793fa8b08678abf6166b78c8e4a3aa973c59e63` |

This was not Linux-versus-Wine numeric noise because the same action was
removed in both source domains at the same frames.

At the first r5 divergence, the observed boss body had a current vertical
velocity near `19.6466`, but it was in movement mode 2 and the next projected
vertical increment was only about `0.2598`. The old portable record exposed
only `x/y/vx/vy`, so its conservative legacy path repeated `19.6466` across
the lookahead horizon, invented an enemy-body collision, and removed upward
actions that original retail retained.

## Correction and compatibility boundary

Portable source commit
`ed8c0730480b6cc15add99167c519f58d96421ff` adds observed enemy axis velocity,
polar motion, acceleration, interpolation mode/ease, inversion, endpoints,
and timer state to schema-v2 records. It interprets no future ECL and adds no
resident planning. The commit is pushed on `headless/th06-rl-headless-spike`.

The rebuilt binaries are:

- Linux source: `45f933a87837afcd4c73973385dbf2fb52ddadb6174c80a4674f2bac66f11724`;
- MinGW source-under-Wine: `bb0fa058ae74edae21c1e1ceb2d293570e466cd6df28a53edfd89a56b4d4545e`.

The solver now projects observed enemy movement modes 0, 1, and 2 with the
same bounded horizon used by native authority. An old record with none of the
new fields retains the conservative constant-velocity behavior so historical
artifacts remain readable. A partially populated new record fails closed; it
cannot silently mix the two contracts.

## Corrected differential

The exact r4--r6 action streams were replayed from their original seeds in both
rebuilt source domains. Input and RNG rows remain equal between Linux source
and MinGW source-under-Wine. Reconstructing retail native authority on every
retained snapshot now gives:

| Run | Equal / common snapshots per source domain | Terminal window | Audit SHA-256 | Platform report SHA-256 |
| --- | ---: | ---: | --- | --- |
| r4 | 6,535 / 6,535 | 120 / 120 | `891e126780fe4ff7a74635d896b7f64231dd1a1c57584bf02c001d7be9b6507d` | `f8634284a6d877d7cef56b3b2599465cf415d956ff0b95a95215147f86a8ad9c` |
| r5 | 6,184 / 6,184 | 120 / 120 | `f86cb016180805db148f15e690b6d2b8efc9a76cf110c486546236a7fb38a302` | `2f25670c3bfb5e72910884c468c9e4bd2b5016b16475eb56066597b1212a5d0a` |
| r6 | 6,401 / 6,401 | 120 / 120 | `31da93d50c82c1f2c4e52b4c22634c76bad448c168a940ec17ff312396192e47` | `399c79f39c72c9584f212f9f1b9c80615896e713ed803d37ea145a4e1b44156a` |

That is 19,120/19,120 same-frame sets in each source domain, or 38,240 exact
comparisons total. There were no missing snapshots and no source authority
failures.

This does not prove whole-runtime equivalence. Original retail still has a
dialogue/transition interval with 289--308 bullets where both source domains
have zero, and Linux versus MinGW source still accumulates physical floating
point drift despite equal discrete input and RNG. Correct safe-set equality in
this panel means only that locally matched snapshots may remain eligible for
reject-only COW.

## Corrected COW regression

The one predeclared r5 checkpoint at sequence 6175 / retail frame 6834 was
replayed with the repaired trace and teacher. Both first actions survived 600
ticks, and the incumbent remained the unique robust winner:

| First action | Minimum future native-safe width | Terminal boundary reserve |
| --- | ---: | ---: |
| `down_right` incumbent | 5 | 112.4410 |
| `down_left` alternative | 3 | 71.2033 |

The corrected COW SHA-256 is
`9ee7134b89d56887634638a813f361bc0425cf5f1c7e86196d7318a1b638ee7a`.
The corrected aggregate remains `left-alternative-rejected`, contains zero
residual candidates, and has SHA-256
`7d0804e851a3cf811ab245dd099d6d9c35becae629c713d1189299ebeec717f3`.

## Next bounded experiment

Panel 2 is closed. A separate predeclared exploratory audit may branch every
retail-native-safe first action once at the already-fixed r5 discovery anchor.
Only a unique robust non-incumbent winner may be checked at the already-fixed
r6 confirmation anchor. This isolates a small hypothesis without a broad fit
or hyperparameter sweep.

The branch continuation uses the offline native teacher after the substituted
first action, not the frozen UCB resident continuation. Its result therefore
ranks hypotheses only; it is not a causal estimate of an active residual's
complete Wine Stage HIT count. At most one confirmed action may proceed to a
new, disjoint original-retail Wine shadow episode, followed by an active
first-failure canary and only then an alternating complete-Stage HIT-count A/B.
