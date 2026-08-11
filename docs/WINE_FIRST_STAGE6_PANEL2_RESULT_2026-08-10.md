# Stage 6 Wine-first panel 2 result (2026-08-10)

## Decision

Panel 2 creates no residual candidate. Frozen
`phase-local-hierarchical-ucb-v4` remains the incumbent. No policy fit,
threshold search, shadow intervention, or active publication is authorized by
this panel.

This is a useful negative result, not a Stage 6 performance claim. The final
promotion metric remains the HIT count in complete natural original-retail
Wine Stage 6 runs, with a 0-HIT clear as the target. First-failure survival,
shadow activations, source replay, and headless COW are only cheaper filters.
They cannot replace an alternating full-Stage Wine HIT-count A/B. HIT
continuation is a quarantined evaluation benchmark and never enters training.

## Frozen original-retail Wine panel

The protocol in `WINE_FIRST_STAGE6_PANEL2_PROTOCOL_2026-08-10.md` was committed
as `078cdaee5e83848d2317679a1831115d1b283998` before collection. Exactly r4,
r5, and r6 then ran with Lunatic / Reimu-A / Stage 6 Practice, retail executable
SHA-256 `9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`,
native kernel SHA-256
`71b27bef942928dfc04b6a489ef261cb6485cd5bab2cd4d4ebf4081991a22b5b`,
policy plugin SHA-256
`4d7f10925731d7f83389aaa8c2aa942d7ed156de54791767ff2ced802483bbf2`,
and immutable policy-state SHA-256
`e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`.

| Run | Run ID | Terminal frame | Frames / transitions | Anchors | Physical result | Wall time |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| r4 | `20260810T134144Z-438145900` | 7178 | 6535 / 6534 | 9 | 0 HIT, 0 Bomb, Hard empty | 187.59 s |
| r5 | `20260810T134540Z-909617800` | 6842 | 6184 / 6183 | 8 | 0 HIT, 0 Bomb, Hard empty | 176.22 s |
| r6 | `20260810T134906Z-860850500` | 7045 | 6401 / 6400 | 7 | 0 HIT, 0 Bomb, Hard empty | 186.51 s |

Each run preserved equal before/after policy, score, and config hashes, returned
through the exact trial wrapper, and reported an empty leftover-prefix process
list. The run-report SHA-256 values are, in r4-r6 order:

- `a4a4a886d0876d0f373e83fa9e80df50f994d2e50615aa341478b52ff70a239b`;
- `f30ebebf4ea3e571b44503702e31a18621f3e64297cb8c1e529046e64d45a830`;
- `9b32e2de4303de498b481d4948849eca2c06c101e9ec9b36c9c771c8d59549d4`.

These runs are not complete-Stage results: the safety contract stopped each
one on the first native authority failure. Their 0-HIT prefixes therefore do
not demonstrate NMNB and do not improve the historical full-Stage baseline.

## Episode-grouped audit

The immutable factual replay audited 19,097 policy calls across all three
prefixes and found zero recorded-incumbent mismatch, zero recorded-policy
mismatch, and zero shadow action-contract violation. Its report is
`artifacts/wine-first-stage6/framev5-panel2-r4-r6-factual-action-audit.json`,
SHA-256 `d8bf4b08b51fb777467388ffc65884a2afee76fe4fa5908385b0fad6474abc07`.
Candidate counters emitted by the old installed shadow model are incidental;
the policy was immutable and none of those actions was published or used to
select this panel.

The grouped audit contains 18,606 eligible rows, 332 positive-window rows, 70
fallback-opportunity rows, 12 per-episode-deduplicated events, and three
independent episodes. All three failures are in
`boss:0:sub31:life_cb31:timer_cb19:spell`. After excluding both action pairs
already rejected by panel 1, only one new opportunity had support in at least
two panel-2 episodes:

`interior / lasers-present / broad-5-plus / incumbent=down_right / baseline=down_left`

The report is
`artifacts/wine-first-stage6/framev5-panel2-r4-r6-failure-regions.json`,
SHA-256 `c5395f44ec66cd17e50365a6d66374c19bb9879af388a8fe7ba744b00d9f651b`.

## Bounded COW result

The declared pair was tested at r5 sequence 6175 / retail frame 6834, eight
frames before terminal. Exact source replay matched the retail checkpoint and
reconstructed the recorded native Hard set under the retail delivery envelope
`(0,1,2,3)`. Both 600-tick branches survived without a source death, but the
robust outcome favored the incumbent:

| First action | Minimum future native-safe width | Terminal boundary reserve |
| --- | ---: | ---: |
| `down_right` incumbent | 5 | 112.4410 |
| `down_left` alternative | 3 | 71.2033 |

The COW document SHA-256 is
`9ee7134b89d56887634638a813f361bc0425cf5f1c7e86196d7318a1b638ee7a`.
Once one independent exact checkpoint favored the incumbent, the alternative
could no longer satisfy the predeclared unanimous multi-prefix gate, so no
additional branch compute was spent on r6. The aggregate audit concludes
`left-alternative-rejected`, retains one incumbent win, and creates zero
candidates. Its SHA-256 is
`7d0804e851a3cf811ab245dd099d6d9c35becae629c713d1189299ebeec717f3`.

These numbers supersede the first COW rendering. The decision did not change,
but the earlier trace omitted observed enemy interpolation state and therefore
used a false constant-velocity contact-body projection. The repaired COW is
bound to source commit `ed8c0730480b6cc15add99167c519f58d96421ff`
and Linux source binary SHA-256
`45f933a87837afcd4c73973385dbf2fb52ddadb6174c80a4674f2bac66f11724`.

## Delivery, RNG, and platform drift

The exact action streams contain 7178, 6842, and 7045 actions with recovered
Stage RNG seeds 48856, 45688, and 61877. Linux source and MinGW
source-under-Wine agreed on every discrete input/RNG row in each full stream.
They nevertheless diverged in exact physical state at source tick 441 and in
bullet-birth event geometry at ticks 455, 453, and 450 respectively. This
confirms ordinary platform numeric drift even when discrete delivery agrees.

More importantly, both reconstructed source domains diverged from original
retail in the same dialogue/transition interval despite retaining equal RNG
and exact sampled dialogue inputs:

| Run | Exact dialogue frames matched per source domain | First retail/source hazard-count divergence | First player geometry divergence above `1e-6` |
| --- | ---: | --- | ---: |
| r4 | 484 / 484 | frame 4466: source 0 bullets, retail 289 | 2047 |
| r5 | 483 / 483 | frame 4461: source 0 bullets, retail 298 | 4478 |
| r6 | 487 / 487 | frame 4574: source 0 bullets, retail 308 | 3050 |

The r5 source state later reconverged closely enough to validate the local
frame-6834 checkpoint. That local equality must not be extrapolated into
whole-prefix or future-dynamics equivalence. Dialogue delivery and RNG
equality are necessary diagnostics, but neither proves hazard-state equality;
render/update ordering and shipped-versus-reconstructed runtime behavior remain
separate causes.

## Consequence for the next experiment

The required safe-set audit and correction are complete in
`WINE_FIRST_STAGE6_ENEMY_MOTION_AUDIT_2026-08-10.md`. It restored exact
same-frame native-set agreement across every r4--r6 snapshot in both source
domains, while preserving the evidence that the reconstructed and retail
runtimes are not globally equivalent. Panel 2 remains closed with zero
candidates.

Do not resume a large offline fit or repeatedly mine panel 2. The next bounded
experiment must be committed before execution and may use the fixed r5 anchor
to scan all retail-native-safe first actions once, with the fixed r6 anchor as
a confirmation only if r5 has a unique robust non-incumbent winner. Any result
is hypothesis-only and requires a disjoint new original-retail Wine shadow
episode before activation.

If a later residual passes disjoint Wine shadow and alternating first-failure
canary trials, run alternating complete natural Stage 6 benchmarks and compare
HIT count directly against frozen UCB. A lower complete-Stage HIT count, and
ultimately repeatable 0-HIT clears, is the criterion that matters; an improved
headless score or longer stopped prefix alone is not promotion.
