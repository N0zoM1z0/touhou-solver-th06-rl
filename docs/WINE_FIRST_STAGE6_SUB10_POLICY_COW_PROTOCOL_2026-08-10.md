# Stage 6 sub10 policy-faithful COW protocol (predeclared 2026-08-10)

## Question

At the fixed current-kernel sub10 opportunity, does replacing frozen UCB's
first `down_right` with native-safe `down_fast`, then immediately returning
all later choices to the same immutable UCB policy, improve the robust source
outcome in two independent Wine-anchored episodes?

This is not a new fit and does not reinterpret the old offline-teacher COW.
The first action, pair, episode order, and checkpoints were all selected by the
closed r1--r3 panel before policy-faithful continuation existed.

## Frozen inputs

- scope: Lunatic / Reimu-A / Stage 6;
- retail executable SHA-256:
  `9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`;
- retail native kernel SHA-256:
  `71b27bef942928dfc04b6a489ef261cb6485cd5bab2cd4d4ebf4081991a22b5b`;
- immutable UCB state SHA-256:
  `e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`;
- portable source commit:
  `604235a13140999e7f7239aafe8c7fd0a22ff51d`;
- Linux source binary SHA-256:
  `9369474727800457299a1fae1ca963dc073d474cac21e89bd4d62c55f21d6ae2`;
- policy horizon 12, branch horizon 600 source ticks;
- actions in fixed order: factual `down_right`, alternative `down_fast`.

Discovery is run r1 `20260810T124531Z-310133600`, sequence 2904 / retail
frame 3296. Only if `down_fast` is its unique robust winner is confirmation
run r2 `20260810T131002Z-681278300`, sequence 3193 / retail frame 3566
allowed. Do not swap episode order, inspect r3/r8, change the frame, or add an
action after observing discovery.

## Required factual regression

At each allowed checkpoint, source state must match the retained retail
physical snapshot at `1e-6`; restored policy state must select the recorded
`down_right`; and the factual branch must reproduce the exact recorded action
suffix, terminal kind, and terminal frame. Failure of any prerequisite closes
the experiment with zero candidates. Retail native certification retains the
delivery envelope `(0,1,2,3)`; synchronous source stepping remains `(0,)`.
Native authority is unchanged and Bomb is forbidden.

## Fixed gate

Use `scripts/label_retail_policy_cow.py` and the already frozen
`robust-outcome-rank-v1`. Discovery must have the single robust winner
`down_fast`; a tie or incumbent win rejects it and forbids confirmation. If
discovery passes, r2 must also rank `down_fast` strictly above `down_right`.

At most one headless hypothesis can result. It cannot become active from this
COW, cannot train on these branches, and must first enter a still-new
original-retail Wine shadow episode. After shadow, it needs alternating
frozen-UCB/active-canary first-failure trials. Only a later alternating
complete natural original-retail Wine Stage 6 A/B can answer the core question:
whether HIT count falls and repeatable 0-HIT clears become more frequent.
HIT-continuation is evaluation-only and cannot flow back into training.
