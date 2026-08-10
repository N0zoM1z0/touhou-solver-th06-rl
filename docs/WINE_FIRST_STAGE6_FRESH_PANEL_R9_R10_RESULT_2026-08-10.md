# Stage 6 fresh Wine panel r9-r10 result (2026-08-10)

## Result

The predeclared r9-r10 panel creates **zero residual candidates**.  Both fresh
original-retail Wine episodes ended fail-closed on a native Hard-empty safe
set, with zero physical HITs and zero Bomb requests before termination.  r9
added a singleton sub11 failure; r10 added another sub10 failure.  Neither
episode contained a native-baseline alternative opportunity, so neither can
open a new counterfactual action region.

The existing repeated opportunity regions remain the three regions already
seen in r1-r8 and already closed by their fixed counterfactual panels:

- sub10, incumbent `down_right`, native baseline `down_fast`;
- sub10, incumbent `up_fast`, native baseline `down_fast`;
- sub31, incumbent `down_right`, native baseline `down_left`.

Do not mine another row from r9/r10, reopen any of those pairs, or fit a broad
sub10 override from the increased context count.

## Physical evidence

r9:

- artifact: `artifacts/wine-retail-stage6-framev5-frozen-ucb-firstfailure-r9`;
- report SHA-256:
  `70125846c1c429443f2b5238ddb7bb229c32327f0e46983aa24bd8dfc662683d`;
- corpus run: `20260810T163220Z-994484700`;
- 2,825 transition rows, terminal frame 3,211;
- terminal context
  `boss:0:sub11:life_cb14:timer_cb13:nonspell`.

r10:

- artifact: `artifacts/wine-retail-stage6-framev5-frozen-ucb-firstfailure-r10`;
- report SHA-256:
  `a6a4725e2cf0e058c3da01183d23c7ded91f688a36553d0cd84ad5fbd46af5f6`;
- corpus run: `20260810T163356Z-619867900`;
- 2,743 transition rows, terminal frame 3,138;
- terminal context
  `boss:0:sub10:life_cb14:timer_cb13:nonspell`.

Both wrapper reports have return code 12, the expected native-authority
fail-close result, with no wrapper error.  Retail, native DLL, policy plug-in,
config, and immutable policy-state checks passed; the policy state remained
byte-identical and each dedicated prefix had no leftover process.  Both
corpus manifests are transaction-complete with no dropped rows.

## Fixed post-collection audit

The r1-r10 factual replay artifact is
`artifacts/wine-first-stage6/framev5-frozen-ucb-r1-r10-factual-action-audit.json`
(SHA-256
`f0bc710f65c5248f61aa06288de21ac354a664a4597488ff8011a3f4090eef86`).
It passed across 10 independent physical episodes and 41,666 policy calls:

- recorded-incumbent mismatches: 0;
- recorded-policy mismatches: 0;
- shadow action-contract violations: 0.

The grouped audit is
`artifacts/wine-first-stage6/framev5-frozen-ucb-r1-r10-failure-regions.json`
(SHA-256
`dcc9b8a2b6f538bb66945cf68ecdd6f7cc1a6e95c53817311934ab0c05dfe541`).
It contains 10 effective episode units, 575 positive-window rows, 88 fallback
opportunity rows in seven episodes, eight episodes in repeated contexts, and
only the three previously closed repeated action-relative regions above.

## Consequence

The incumbent remains the only active policy.  There is no justified Wine
shadow, active canary, or candidate full-stage A/B from this panel.  The core
metric remains complete natural original-retail Wine Stage 6 HIT count: the
current-kernel frozen-incumbent point baseline is 10 HITs.

Before spending more physical or headless compute, perform a deterministic,
evaluation-only audit of the ten incumbent HIT windows.  Its sole question is
whether Hard-set collapse exposes a useful earlier decision window inside the
native-safe set.  The complete-stage continuation trace remains quarantined:
it may define a future collection hypothesis but may not supply fitting rows,
labels, candidate selection, or promotion evidence.
