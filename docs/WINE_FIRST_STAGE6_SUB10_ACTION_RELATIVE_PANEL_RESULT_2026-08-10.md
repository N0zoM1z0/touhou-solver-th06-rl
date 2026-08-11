# Stage 6 sub10 action-relative panel result (2026-08-10)

## Decision

The first action-relative label panel is insufficient and authorizes zero
models. r2 produced a sealed 14-action document, but r3 failed before any
branch outcome because the policy restore incorrectly treated a historical
input-lease delivery row as a policy call. Per protocol, collection stopped;
r8 was not run and the r2 outcomes were not inspected.

No residual candidate, shadow policy, or active canary exists. Complete
natural original-retail Wine Stage 6 HIT count remains the final metric, but
there is no candidate worth spending a full-Stage trial on yet.

## Sealed evidence and failure

The protocol was committed as `1343c88` before the unopened rows were run.
r2 `20260810T131002Z-681278300`, sequence 3193, produced a factual-regression-
passing document containing all 14 declared actions, SHA-256
`6e22616c69e3e57f5f055b8475eb2953d76ae4efff224bf4cb0772231baed3d0`.
Its per-action outcomes remain uninterpreted and it is not a partial panel.

r3 stopped during restoration at historical sequence 1888. That frame has
decision reason `input-lease`, a carried `proposed_action=down_left`, no new
reactive baseline, and local delivery action `up_fast`. The old restore loop
tested only whether `proposed_action` was non-null, created a fictitious UCB
call with `baseline=None`, and failed with
`reactive baseline is outside the local safe set`. No r3 COW document was
written and no r3 counterfactual outcome was observed.

## Infrastructure correction

Both policy-continuation and policy-COW restoration now count a recorded row
as a policy call only when it has a proposal and decision reason `ok`.
Input-lease rows remain delivery evidence but no longer mutate replayed UCB
call order. A focused regression test covers the carried-proposal case.

After the fix, a diagnostic factual-only r3 continuation restored 3,327 real
policy calls with zero mismatch and reproduced the exact frame-3722
retail-envelope Hard-empty terminal. This validates the correction; it does
not retroactively complete or rescue the failed label panel. The full test
suite passes.

## Next boundary

The failed panel remains closed. A new version may be separately predeclared
and rerun from clean output paths after the restore correction, because no r3
or r8 action outcomes were observed and r2 remains sealed. Such a rerun must
repeat every declared document under the corrected implementation, keep r1
development-only, and retain the same three-of-four episode support gate. It
still cannot directly create an active policy.
