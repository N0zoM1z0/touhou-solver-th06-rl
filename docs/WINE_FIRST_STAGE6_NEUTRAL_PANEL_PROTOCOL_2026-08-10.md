# Stage 6 neutral-family Wine panel protocol (predeclared 2026-08-10)

## Question and final metric

Does the Panel-2 neutral first-action family (`stay`, `stay_fast`) reproduce on
new physical episodes strongly enough to justify one later Wine shadow test?
This panel cannot activate a residual. Complete natural original-retail Wine
Stage 6 HIT count, including repeatable 0-HIT clears, remains the decisive
metric after shadow and active-canary filters. Full-Stage HIT continuation is
evaluation-only and never training data.

## Frozen Wine collection

Collect exactly two new, disjoint original-retail Wine first-failure episodes,
named r7 and r8, before inspecting either one. Both use:

- Lunatic / Reimu-A / Stage 6 Practice;
- original retail 1.02h SHA-256
  `9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`;
- native kernel SHA-256
  `71b27bef942928dfc04b6a489ef261cb6485cd5bab2cd4d4ebf4081991a22b5b`;
- `phase-local-hierarchical-ucb-v4`, immutable state SHA-256
  `e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`,
  exploration `0`;
- frame schema v5, natural Practice start, no life patch and no continuation;
- stop on first physical HIT, authority failure, or Bomb request.

There is no fit, state update, threshold adjustment, anchor selection, or
policy change between r7 and r8. Each wrapper report must show equal policy,
score, and config hashes plus no leftover process in its dedicated prefix.

## Deterministic anchor rule

After both runs exist, `scripts/select_retail_neutral_anchors.py` independently
selects one row per episode. A row is eligible only when all of these hold:

- it is in the contiguous 120-frame window before the first failure;
- source context is exactly `boss:0:sub31:life_cb31:timer_cb19:spell`;
- the fixed generic bin is interior, lasers present, native Hard width at least
  five;
- frozen UCB selected `down_right` and the generic baseline was `down_left`;
- `down_right`, `stay`, and `stay_fast` are all native Hard-safe and locally
  legal.

Select the eligible row with minimum frames-to-failure; break an exact tie by
maximum sequence. If either episode does not end in the declared Hard-empty
context or has no eligible row, the panel creates zero candidates and no COW
is run. No alternate context, neighboring row, runner-up, or third episode may
replace a missing anchor.

## Policy-faithful COW gate

Only if both anchors exist, branch exactly these actions in this order at each
anchor: `down_right`, `stay`, `stay_fast`. Use source commit
`604235a13140999e7f7239aafe8c7fd0a22ff51d`, Linux binary SHA-256
`9369474727800457299a1fae1ca963dc073d474cac21e89bd4d62c55f21d6ae2`,
the immutable UCB state above, horizon 12, and 600 source ticks. The factual
`down_right` branch must reproduce the recorded Wine suffix and terminal frame
exactly. All later actions are selected by the restored immutable UCB policy;
native authority is unchanged and Bomb remains forbidden.

`scripts/audit_retail_neutral_policy_cow.py` uses the already frozen robust
rank. Exactly one headless hypothesis exists only if the same single neutral
action is the unique robust winner over both the incumbent and the other focus
variant in both independent episodes. A tie, disagreement, incumbent win,
checkpoint mismatch, factual-regression failure, source failure, or physical
death rejects the entire family. Do not refine the rank or add another action.

Even a unanimous result is reject-only headless evidence. It may enter only a
still-new original-retail Wine shadow episode; r7 and r8 cannot double as that
shadow because they selected the hypothesis. Active first-failure A/B comes
after shadow. Only then is an alternating complete natural Stage 6 HIT-count
A/B worth its roughly seven-to-eight-minute trials.
