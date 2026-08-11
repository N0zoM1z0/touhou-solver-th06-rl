# Stage 6 sub10 action-relative label panel (predeclared 2026-08-10)

## Question

Do multiple current-kernel sub10 Wine anchors contain unique native-safe
first-action improvements even though the improving direction is not fixed
across episodes? This panel collects labels only. It does not fit or activate a
policy.

The fixed `up_right` hypothesis is already rejected and remains rejected. The
new unit is an action relative to the native-safe set and current physical
features, not a frame, seed, run ID, or hard-coded direction.

## Fixed episode panel

Use the four independent frame-v5 frozen-UCB episodes in the repeated
`boss:0:sub10:life_cb14:timer_cb13:nonspell` failure context. Each anchor is
exactly seven frames before its first Hard-empty terminal and is selected once:

| Role | Run | Sequence / frame | Factual action | Recorded Hard actions |
| --- | --- | --- | --- | --- |
| known development | `20260810T124531Z-310133600` | 2904 / 3296 | `down_right` | `stay, down, left, right, up_left, up_right, down_left, down_right, stay_fast, down_fast, left_fast, right_fast, down_left_fast, down_right_fast` |
| unopened panel | `20260810T131002Z-681278300` | 3193 / 3566 | `down_right` | `stay, up, down, left, up_left, down_left, down_right, stay_fast, up_fast, down_fast, left_fast, up_left_fast, down_left_fast, down_right_fast` |
| unopened panel | `20260810T131344Z-316672800` | 3329 / 3715 | `down_left` | `stay, up, down, left, right, up_left, up_right, down_left, down_right, stay_fast, down_fast, left_fast, up_left_fast, down_left_fast` |
| unopened panel | `20260810T151039Z-362645700` | 2839 / 3238 | `down_left` | `stay, up, down, left, up_left, down_left, stay_fast, up_fast, down_fast, left_fast, up_left_fast, down_left_fast` |

r1 reuses the already frozen exhaustive document and is never a holdout. Run
r2, r3, and r8 with no code, threshold, action-list, or policy change between
them, and do not inspect any of their outcomes until all three documents
exist. No neighboring checkpoint or replacement episode is allowed.

## COW contract

Every row uses restored immutable UCB after the substituted first action,
policy horizon 12, 600 source ticks, policy-state SHA-256
`e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`,
source commit `604235a13140999e7f7239aafe8c7fd0a22ff51d`, and Linux binary SHA-256
`9369474727800457299a1fae1ca963dc073d474cac21e89bd4d62c55f21d6ae2`.

Each exact factual branch must reproduce the recorded suffix, terminal kind,
and terminal frame; source state must match retail at `1e-6`; every requested
action must remain native-admissible. Any failed document makes the panel
insufficient. Native authority and retail delivery coverage `(0,1,2,3)` stay
unchanged, and Bomb is forbidden.

## Data-support gate

After all unopened documents finish, run
`scripts/audit_retail_policy_cow_panel.py` with the unchanged
`robust-outcome-rank-v1`. At least three of the four physical episodes must
have exactly one robust winner different from their factual incumbent. Ties
and incumbent wins remain valid negative episode outcomes but do not count as
positive support.

If fewer than three pass, fit zero models. If at least three pass, this panel
authorizes only a separate, predeclared targeted action-relative fit with at
most three small candidates and episode-grouped validation. It does not choose
features, thresholds, or a winner after the fact and creates zero active
candidates by itself.

Any later fitted candidate defaults to frozen UCB, ranks only the current
native-safe set, forbids Bomb, and must pass a still-new original-retail Wine
shadow plus alternating active first-failure canary. The final authority is an
alternating complete natural Wine Stage 6 HIT-count A/B, not these COW scores.
