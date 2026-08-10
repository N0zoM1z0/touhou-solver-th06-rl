# Stage 6 sub10 action-relative panel v2 result (2026-08-10)

## Decision

v2 is insufficient and authorizes zero models. Corrected r1 and r2 documents
completed, but r3 exposed a protocol error before alternative outcomes: the
declared list used the 4-frame retail Hard set, while the learned policy may
rank only the narrower local lookahead set. r8 was not run.

This is not strategy evidence against r3 alternatives. They were outside the
policy authority boundary at that checkpoint and must never have been offered
to a learner.

## Valid corrected documents

The v2 protocol was committed as `b93f53b`. Corrected restore produced:

- r1, 14 outcomes, factual regression passed, SHA-256
  `e4709ecc4e731467d3120bdfd4054c22ea955b49b741af1e657dc57644e93479`;
- r2, 14 outcomes, factual regression passed, SHA-256
  `c4e4492cc33ac7625d5ca6e7e9f5ba12b12d6b84d4020f09527f4278d1ecd405`.

r1 remains known development evidence. r2's per-action outcomes remain sealed
and uninterpreted. Both documents used the corrected restore implementation
and may be named explicitly by a later protocol; they do not form a passing
panel on their own.

## r3 authority-boundary failure

At r3 sequence 3329 / frame 3715, the recorded Hard set contains 14 actions,
but the recorded local lookahead set contains only factual `down_left`. The v2
protocol incorrectly requested all 14 Hard actions. Source reconstruction
therefore rejected the first requested action with
`first action stay is not locally admissible`; no r3 document was written and
no r3 alternative outcome was observed.

`label_retail_policy_cow.py` now validates every requested action against the
recorded local set before source branching. A regression test covers a
Hard-only `stay` next to locally admissible `down_left`. This makes the product
boundary explicit: native Hard owns immediate safety, while a learned ranker
can select only from the resident controller's local set.

## Next boundary

v2 remains closed. A separately predeclared v3 may reuse the individually
valid corrected r1/r2 documents, generate a factual-only r3 negative row, and
scan the complete recorded local set at r8. The unchanged three-of-four gate
then requires r1, r2, and r8 all to have unique non-incumbent winners because
r3 has no admissible alternative. Any other result means zero fits.

No complete Wine Stage is warranted yet. Complete natural original-retail
Wine Stage 6 HIT count remains the final gate after a real candidate survives
shadow and active canary.
