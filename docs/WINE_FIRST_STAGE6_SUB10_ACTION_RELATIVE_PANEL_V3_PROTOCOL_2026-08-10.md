# Stage 6 sub10 action-relative panel v3 (predeclared 2026-08-10)

## Purpose

Complete one authority-correct four-episode label audit without rerunning the
two individually valid corrected documents from closed v2. v3 changes no
anchor, outcome rank, support threshold, source binary, or policy state. It
only respects the recorded local action set at r3 and r8.

This panel cannot fit or activate a policy. The eventual promotion metric is
still complete natural original-retail Wine Stage 6 HIT count.

## Fixed documents

Reuse these corrected v2 documents by exact hash, without reading the sealed
r2 outcomes before the new documents exist:

- r1 `20260810T124531Z-310133600`, sequence 2904, 14 local actions,
  `framev5-sub10-r1-seq2904-policy-cow-safe-panel-v2.json`, SHA-256
  `e4709ecc4e731467d3120bdfd4054c22ea955b49b741af1e657dc57644e93479`;
- r2 `20260810T131002Z-681278300`, sequence 3193, 14 local actions,
  `framev5-sub10-r2-seq3193-policy-cow-safe-panel-v2.json`, SHA-256
  `c4e4492cc33ac7625d5ca6e7e9f5ba12b12d6b84d4020f09527f4278d1ecd405`.

Generate exactly two new documents:

- r3 `20260810T131344Z-316672800`, sequence 3329 / frame 3715, factual and
  complete local set `down_left` only, output
  `framev5-sub10-r3-seq3329-policy-cow-local-panel-v3.json`;
- r8 `20260810T151039Z-362645700`, sequence 2839 / frame 3238, factual
  `down_left`, complete local set in order
  `stay, up, down, left, up_left, down_left, stay_fast, up_fast, down_fast,
  left_fast, up_left_fast, down_left_fast`, output
  `framev5-sub10-r8-seq2839-policy-cow-local-panel-v3.json`.

Do not inspect r2 or r8 outcomes until both new documents exist. Stop on any
preflight, checkpoint, factual-regression, native-authority, or process
failure. No replacement episode or action is allowed.

## Frozen COW contract and gate

Use restored immutable UCB, horizon 12, 600 source ticks, policy-state
SHA-256 `e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`,
source commit `604235a13140999e7f7239aafe8c7fd0a22ff51d`, and Linux binary SHA-256
`9369474727800457299a1fae1ca963dc073d474cac21e89bd4d62c55f21d6ae2`.
Each factual branch must exactly reproduce its retained suffix and terminal.
Native authority, local-set restriction, delivery envelope `(0,1,2,3)`, and
Bomb prohibition are unchanged.

Audit all four documents with `scripts/audit_retail_policy_cow_panel.py` and
`robust-outcome-rank-v1`. The threshold remains at least three physical
episodes with one unique robust winner different from their factual action.
Because r3 has no alternative, passing requires r1, r2, and r8 all to have a
unique non-incumbent winner. A tie or incumbent win in either unopened
multi-action episode yields zero fits.

Passing authorizes only a new, separately committed action-relative fit
protocol capped at three small residual candidates. It does not choose or
activate a model. Any model must default to frozen UCB, rank only the current
native-safe local set, pass new Wine shadow and active first-failure canary,
then improve complete natural Stage HIT count in alternating Wine A/B trials.
