# Stage 6 full-stage HIT-window audit (2026-08-10)

## Result

The current-kernel frozen-incumbent full-Stage trace exposes a reusable
warning signal that the current policy feature boundary omits: every one of
the 10 physical HITs followed a transition from the normal 12-frame native
lookahead to the four-frame emergency fallback.

The final contiguous degraded-horizon run began 6--14 frames before each HIT
(median 11).  By contrast, the native Hard set became empty only 2--5 frames
before the HIT (median 4), after no action remained for a residual to rank.
The useful collection point is therefore the sustained lookahead degradation,
not the Hard-empty event itself.

This is an evaluation-only diagnostic from one complete Stage trajectory.  It
does not label an action, authorize a residual, or demonstrate improvement.

## Bound artifact

Input trace:
`artifacts/wine-retail-stage6-framev5-frozen-ucb-current-natural-r1/trace.jsonl`
(SHA-256
`4ea975c5c788cb21b1baf4fce834b1786053ed506ed98bde9cffedd22fc6f96e`).

Audit artifact:
`artifacts/wine-first-stage6/framev5-frozen-ucb-current-natural-r1-hit-window-audit.json`
(SHA-256
`22c02419fb403cd6998032c7ceee6bbcaf0edcbd6e7d23472d10460291036139`).

The deterministic implementation is
`scripts/audit_full_stage_hit_windows.py`.  Its 120-frame windows never cross
a preceding physical HIT, and its output declares that it may not supply
training rows, counterfactual labels, candidate selection, or promotion.

## Findings

- 10/10 HITs had a final contiguous `effort_horizon < 12` warning run;
- 10/10 then entered a final native Hard-empty interval;
- the last still-legal decision before Hard-empty had a median Hard width of
  3.5, but widths ranged from 1 to 12;
- only 7/10 HIT windows contained an `ok` row with a 1--4-action Hard set;
- 9/10 contained a 5--9-action row;
- one HIT window contained a stale retry and none contained an input lease.

Thus a trigger based only on a narrow Hard set would miss three of ten HITs.
The emergency horizon is more general: some trajectories collapse directly
from 10 or 12 Hard actions to zero because no constant action survives the
longer lookahead even while many actions remain four-frame safe.

Across the whole trace there were 19 contiguous degraded-horizon `ok` runs.
For a fixed diagnostic trigger at the third consecutive degraded row, there
were 15 activations: 10 preceded the 10 distinct HITs within 15 frames and 5
did not.  This is acceptable overhead for targeted data acquisition, but it
is not an active-policy rule and its apparent coverage must not be treated as
an independently confirmed estimate.

## Identified feature alias

The resident policy receives the Hard and locally admissible action masks but
not `effort_horizon`.  When the 12-frame constant-action lookahead is empty,
the controller deliberately falls back to the four-frame Hard set.  The
resulting Hard and legal masks can be identical both when all Hard actions
survive the long lookahead and when none do.  Frozen UCB and the existing
offline risk features therefore cannot distinguish these materially different
states.

`effort_horizon` is already native-computed, bounded, recorded in frame
evidence and trace output, and does not interpret future ECL births.  Exposing
it additively at the policy/corpus boundary preserves collision authority and
adds no search to the hot path.

## Next gate

Before any new training:

1. add exact `effort_horizon` to `PolicyContext` and compact transition
   `policy_context`, with an explicit unknown value for legacy rows;
2. keep frozen UCB's keys and action selection unchanged;
3. add a versioned Wine risk feature contract rather than silently changing
   an old schema;
4. require r1-r10 frozen action replay to remain at zero mismatch;
5. only then predeclare a small fresh Wine collection using the third
   consecutive degraded-horizon decision as a recorder/anchor trigger.

The trigger collects evidence; it never selects a movement action.  A future
candidate must still pass Wine shadow, active canary, and fresh interleaved
complete natural Stage HIT-count A/B against the incumbent point baseline of
10 HITs.
