# Stage 6 policy-faithful COW result (2026-08-10)

## Decision

The policy-faithful discovery creates zero candidates. `stay` and `stay_fast`
tied for the best robust outcome, so the predeclared unique-winner gate failed.
The conditional r6 confirmation was not run. No action may enter Wine shadow
from this result.

The tie is useful evidence: both neutral first actions changed the restored
frozen-UCB continuation from an eight-tick Hard-empty stop to at least 600
0-HIT source ticks. But they differ in the focus bit, which can change shooting
and later boss timing. They cannot be merged or selected after observing the
result.

Complete natural original-retail Wine Stage HIT count remains the decisive
metric. No candidate has reached that gate; full-Stage HIT continuation is
evaluation-only and remains excluded from training.

## Frozen protocol and factual regression

The protocol and implementation were committed as `1f1825d` before any branch
ran. Discovery used panel-2 r5 `20260810T134540Z-909617800`, sequence 6175 /
frame 6834, source commit `604235a13140999e7f7239aafe8c7fd0a22ff51d`,
Linux binary SHA-256
`9369474727800457299a1fae1ca963dc073d474cac21e89bd4d62c55f21d6ae2`,
and immutable UCB state SHA-256
`e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`.

Before accepting counterfactuals, the `down_right` factual branch reproduced
the recorded suffix exactly: `down_right`, seven `down_left` actions, then the
same retail-envelope Hard-empty stop at frame 6842. This preserves the factual
continuation prerequisite established by the standalone audit.

The COW document is
`artifacts/wine-first-stage6/framev5-panel2-r5-seq6175-policy-cow-discovery-v1.json`,
SHA-256 `696e28b6f0c1675485e9da621f4bc5c1631ddbab1635991ea5bc7db07ab656c0`.

## Robust outcomes

| First action | Terminal | Survival ticks | Minimum safe width | Reserve bucket | Robust result |
| --- | --- | ---: | ---: | ---: | --- |
| `stay` | tick limit | 600 | 3 | 3 | best tie |
| `stay_fast` | tick limit | 600 | 3 | 3 | best tie |
| `right` | tick limit | 600 | 3 | 0 | below best |
| `up_left` | tick limit | 600 | 3 | 0 | below best |
| `up_right` | tick limit | 600 | 3 | 0 | below best |
| `up` | tick limit | 600 | 1 | 0 | below best |
| `down`, `left`, `down_left`, `down_right`, `left_fast`, `down_left_fast` | authority failure | 8 | 1--3 | n/a | rejected |
| `up_right_fast` | authority failure | 4 | 1 | n/a | rejected |
| `down_fast` | authority failure | 3 | 1 | n/a | rejected |

Every branch had zero physical deaths. The robust gate report concludes
`discovery-robust-tie-rejected`, exposes zero headless hypotheses and zero
active candidates, and has SHA-256
`4d7375cb8441de5045e931fe10ead958cbd68c8e72ec470d3a2454f13a82374b`.

## Next evidence

Do not run the conditional r6 branch, choose a tied winner, merge the focus
variants, inspect a runner-up, or move to an adjacent Panel-2 frame. Panel 2 is
closed.

The next efficient experiment is a newly predeclared, disjoint frozen-UCB Wine
panel. It may carry the fixed two-member neutral-action family (`stay`,
`stay_fast`) as a hypothesis, not as candidates. At deterministic generic
failure anchors in the new episodes, policy-faithful COW must produce the same
unique neutral winner across independent episodes before at most one action is
allowed into a still-new Wine shadow run. If the variants remain tied or
disagree, the family is rejected without a broad model fit.
