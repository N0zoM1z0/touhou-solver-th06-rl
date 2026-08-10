# Stage 6 sub10 source support result (2026-08-10)

## Decision

The fixed source support panel fails its candidate-construction gate.  It
creates zero residual candidates, authorizes no Wine shadow or active input,
and leaves frozen `phase-local-hierarchical-ucb-v4` as the incumbent.

This is reject-only headless evidence.  It does not change the complete
natural original-retail Wine Stage 6 baseline or establish NMNB.

## Bound execution

The predeclared seeds 201--208 all produced one valid checkpoint under source
commit `604235a13140999e7f7239aafe8c7fd0a22ff51d`, Linux binary SHA-256
`9369474727800457299a1fae1ca963dc073d474cac21e89bd4d62c55f21d6ae2`,
and immutable policy-state SHA-256
`e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`.
Every checkpoint matched its replay digest and every factual COW branch
reproduced the source root action suffix and terminal.  Native certification
retained the retail delivery envelope `(0,1,2,3)` and Bomb remained
unrepresentable.

The machine report is
`artifacts/wine-first-stage6/framev5-sub10-headless-source-support-seeds201-208-v1.json`,
SHA-256
`0a08ea1aadda431009bf09566f0d4a4a6be77f74fae4975e7e94efda18f93d84`.

| Seed | Split | Root terminal | Checkpoint / local width | Factual | Unique robust winner |
| ---: | --- | --- | --- | --- | --- |
| 201 | development | authority failure 3396 | 3395 / 8 | `down_left` | tie |
| 202 | confirmation | authority failure 3083 | 3079 / 11 | `left_fast` | tie |
| 203 | development | authority failure 3324 | 3323 / 12 | `down_right` | tie |
| 204 | confirmation | authority failure 3079 | 3078 / 8 | `down_right` | tie |
| 205 | development | tick limit 8000 | 3818 / 13 | `right_fast` | tie |
| 206 | confirmation | authority failure 3153 | 3144 / 8 | `left_fast` | `left` |
| 207 | development | tick limit 8000 | 3820 / 6 | `up` | tie |
| 208 | confirmation | authority failure 3074 | 3062 / 7 | `down_left_fast` | tie |

Only seed 206 favored a unique non-incumbent action: `left` survived 12 ticks
versus factual `left_fast` for 9 ticks.  The other seven seeds had tied robust
top tiers.  Support is therefore 0/4 development episodes and 1/4
confirmation episodes, below the fixed requirement of at least 2/4 in each
split.  All eight checkpoints being available satisfies coverage but cannot
substitute for consistent action-value support.

## Consequence

Do not fit an exact lookup, fixed direction, sub10-wide override, or small
classifier from these rows.  In particular, the differing Wine winners and
the seven source ties show that the broad sub10 geometry is not itself enough
to choose an intervention.

The next physical metric should be measured directly rather than inferred
from more headless score.  A current-kernel frozen-incumbent complete Stage 6
calibration may run as evaluation-only HIT continuation.  It cannot enter
training and cannot serve as the sole future A/B control; any eventual
candidate still requires alternating fresh full-Stage control/candidate Wine
runs.

