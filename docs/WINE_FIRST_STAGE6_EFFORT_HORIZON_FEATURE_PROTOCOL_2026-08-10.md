# Stage 6 effort-horizon feature audit (predeclared 2026-08-10)

## Purpose

Close one generic feature alias identified by the evaluation-only complete
Stage HIT-window audit.  The controller already computes whether at least one
constant action survives the full observed lookahead; when none does, it
retains the four-frame native Hard set so the policy may re-decide.  The policy
and compact replay context currently receive the resulting action sets but not
which horizon produced them.

This is an additive evidence-contract change.  It must not change frozen UCB's
keys, scores, chosen actions, native authority, delivery, or hot-path work.

## Fixed implementation

1. Add `effort_horizon` to `PolicyContext` with `-1` meaning unavailable in a
   legacy record.
2. Supply the already-computed value from the resident controller and source
   policy-continuation path.
3. Add the value to exact compact transition `policy_context`; replay readers
   accept old rows as unknown rather than guessing 12 or 4.
4. Keep `phase-local-hierarchical-ucb-v4` context keys and decisions unchanged.
5. Define a new Wine risk feature contract containing numeric
   `effort_horizon`; do not silently mutate v1 or v2 feature order.
6. Add focused unit tests for live/offline feature parity, legacy unknown
   handling, and frozen-UCB key invariance.

No model is fitted or exported in this change.  No residual is created.

## Acceptance gates

- the complete test suite passes;
- the current source/retail policy-continuation tests retain factual suffix
  equality;
- replay of all r1-r10 physical prefixes with the immutable UCB state has zero
  recorded-incumbent mismatch and zero policy mismatch;
- old v1/v2 model feature contracts and decoding remain byte-order compatible;
- no Wine game is launched for this infrastructure audit.

If any frozen action changes, reject the change rather than relabeling the old
corpus.  Passing only authorizes a separately predeclared fresh Wine data
collection whose recorder/anchor trigger is three consecutive `ok` decisions
with `effort_horizon < 12`.  That trigger does not select an action, and final
promotion remains fresh interleaved complete natural original-retail Wine
Stage HIT count against the frozen incumbent.
