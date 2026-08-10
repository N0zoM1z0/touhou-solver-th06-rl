# Stage 6 sub10 native-safe scan protocol (predeclared 2026-08-10)

## Purpose

Run one final, exhaustive policy-faithful first-action scan at the unchanged
sub10 r1 checkpoint. This prevents serial post-hoc pair selection after the
fixed native-baseline pair tied. It is the last discovery use of this anchor.

Complete natural original-retail Wine Stage 6 HIT count remains the final
metric. This scan is reject-only headless evidence and cannot activate a
policy, train a model, or consume HIT-continuation data.

## Frozen discovery

Discovery is run `20260810T124531Z-310133600`, sequence 2904 / retail frame
3296. Branch every action in the already recorded retail native Hard set,
exactly once and in this fixed order:

1. `stay`
2. `down`
3. `left`
4. `right`
5. `up_left`
6. `up_right`
7. `down_left`
8. `down_right`
9. `stay_fast`
10. `down_fast`
11. `left_fast`
12. `right_fast`
13. `down_left_fast`
14. `down_right_fast`

Use immutable policy state SHA-256
`e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`,
source commit `604235a13140999e7f7239aafe8c7fd0a22ff51d`, Linux source binary
SHA-256 `9369474727800457299a1fae1ca963dc073d474cac21e89bd4d62c55f21d6ae2`,
policy horizon 12, and branch horizon 600 ticks. All later decisions belong to
the restored immutable UCB policy. Native authority and the retail delivery
envelope `(0,1,2,3)` are unchanged; Bomb is forbidden.

The `down_right` factual branch must reproduce the retained suffix and exact
frame-3303 Hard-empty terminal. Source and retail checkpoint state must match
at `1e-6`, and restored UCB must select `down_right`. Any failed prerequisite
closes the scan with zero candidates.

## Unique-winner and conditional confirmation gate

Apply `robust-outcome-rank-v1` without changing buckets after observation. A
discovery candidate exists only when exactly one action has the best robust
rank and it is not factual `down_right`. A tie or incumbent win permanently
closes this r1 anchor; do not inspect a nearby frame, alter the rank, or add a
second discovery episode.

Only if discovery produces one candidate, run conditional confirmation at
r2 `20260810T131002Z-681278300`, sequence 3193 / retail frame 3566, branching
exactly `down_right` then that candidate. The candidate must be native-safe at
the r2 checkpoint, all factual prerequisites must pass, and it must strictly
beat `down_right` under the same robust rank. Failure or disagreement rejects
it. No r3/r8 fallback confirmation is allowed.

At most one headless hypothesis may survive. It still requires a new
original-retail Wine shadow episode, then an alternating active first-failure
canary, before the decisive alternating complete natural Stage HIT-count A/B.
