# Stage 6 sub10 policy-faithful pair result (2026-08-10)

## Decision

The fixed `down_right`/`down_fast` pair creates zero candidates. The two
actions tied under the predeclared robust outcome rank, so r2 confirmation was
not run. Frozen UCB remains the incumbent.

## Evidence

The protocol was committed as `f5ff48d` before discovery. At r1
`20260810T124531Z-310133600`, sequence 2904 / retail frame 3296, source state
matched retail at `1e-6`, restored UCB selected the recorded `down_right`, and
the factual branch reproduced the exact seven-action suffix and frame-3303
Hard-empty terminal.

Both first actions then survived seven ticks and reached the same authority
failure with minimum native-safe width 10, zero physical death, and terminal
boundary-reserve bucket zero. `down_fast` changed the later UCB action stream,
but it did not delay or avoid the terminal. Both robust ranks are
`(0, 1, 7, 0, 0)`.

The COW document SHA-256 is
`f5e51e09aba991603a876ec86974613536c0edd88269b18f8f4c8c3b6a0b25ba`.
The gate report concludes `discovery-robust-tie-rejected`, exposes zero
headless hypotheses and zero active candidates, and has SHA-256
`2ee546e54aa4c494cc6522be5899d4a0481d156d69d4413c5f123f88b01c1b1b`.

## Boundary and next step

This result neither improves nor measures complete-Stage performance. The
decisive metric remains HIT count in complete natural original-retail Wine
Stage 6 runs, with repeatable 0-HIT clears as the target. Headless branches do
not train a model or substitute for that gate.

Do not run r2 for this pair or retest it at another row. A final separately
predeclared exhaustive scan may branch every action in the already recorded
r1 native-safe set once under policy-faithful continuation. That avoids
serially choosing another pair after each negative result. If it does not
produce a unique non-incumbent robust winner, the r1 sub10 anchor is closed.
