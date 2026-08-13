# Generation 6 decision-numeric successor

## Scope and invariant

This is a learner-only successor to autonomous round 3. It reuses, without
mutation or selection:

- all 56 registered original-Wine episodes / 167,250 factual options;
- the complete seven-member fit checkpoint with SHA-256
  `e689b9b08902c2ecc761648898d682fca641468ff282aba495da74e1df2dc219`;
- the passed original-Wine startup, process-priority, cleanup, and latency-tail
  evidence.

It collects no gameplay data and does not refit. Reward, actor weights,
propensity, native-safe membership, intervention cap, option horizon, Bomb
prohibition, natural RNG, and complete-Stage HIT continuation are unchanged.
Round 3 remains historically rejected under its frozen raw-logit gate. A pass
here can authorize only a new, predeclared contract.

## Why the serving quantity changes

The deployed policy never consumes an absolute actor logit. For each member it
uses only the score difference from the native baseline action, then averages
those seven advantages. The old scorer rounded two potentially large logits
before subtracting them. Common action bias and state-dependent offsets could
therefore amplify harmless float32 representation differences even though
they cancel algebraically from every decision.

The first successor kernel computed the baseline action tower once and directly
accumulates:

`(action_hidden - baseline_hidden) * action_score_weight`

and

`(action_latent - baseline_latent) * state_latent`.

The baseline is exactly zero. This is the same mathematical policy, not a new
learner or a gameplay correction. It is generic to any baseline-centred
low-rank actor. Both Linux and Win32 builds disable multiply/add contraction so
the declared scalar float32 operation order is the serving identity.

## First contract rejection and float64 serving successor

The first full development audit rejected the float32-centered implementation.
It covered all 167,250 factual options / 2,415,808 candidate rows in 113.8
seconds and found:

- exact support masks at all 167,250 boundaries;
- two portable/Linux action differences;
- all 320 selected panel target errors inside their calculated envelopes;
- only 254/320 panel decisions with margin above the envelope, plus one
  canonical/Linux action difference;
- a smallest margin/envelope ratio of `0.000488`.

Some selected margins were only `3.4e-6`, while a legitimate float32 target
difference reached `2.4e-4`. This is real decision ambiguity, not a reason to
relax the envelope. That first frozen contract remains rejected.

The next implementation preserves every serialized float32 fitted parameter
and preserves float32 representation/support inputs, but evaluates the actor
towers, centering, low-rank dot, population mean, and comparison with scalar
float64 intermediates. Its deterministic reference and envelope are likewise
float64. This is an export/runtime precision repair, not refitting or action
shaping. A pre-freeze full development smoke then produced:

- 167,250/167,250 exact reference/Linux actions;
- 167,250/167,250 exact support masks;
- 320/320 exact wide-panel actions, covered target errors, and certified
  margins;
- minimum panel margin/envelope ratio `501,946`;
- 106.5 seconds total with 16 bounded worker processes.

The old hazard-component `2e-5` diagnostic held for 125,707/167,250 rows, with
maximum component difference `1.1562e-4`; nevertheless every support mask was
exact and maximum support-distance difference was `1.9538e-4`. The next
contract therefore treats finite hazard encoding plus exact support membership
as the decision authority. The raw component threshold remains reported but is
not allowed to reject an unchanged support mask. This rule is generic and was
frozen only in a successor after the first contract failed.

## Predeclared numerical contract

The audit has three conjunctive layers.

1. **Full registered-corpus Linux differential.** Every factual boundary must
   retain a finite fixed-width row, the same portable/native support mask, and
   the same final mean-population action. Hazard components must remain finite;
   the historical `2e-5` component bound is reported as a diagnostic while
   exact support membership is the authoritative gate. Selection includes all 167,250 options;
   there is no Stage, spell, frame, RNG, HIT, action, or failure filtering.
2. **Deterministic serving-precision reference and forward envelope.** A scalar reference
   follows the native normalization, feature-major affine reductions, tanh,
   low-rank dot, centering, population mean, positivity boundary, and lexical
   tie break. The float64 portability envelope allows eight unit roundoffs at
   each target `tanh`, then propagates that perturbation through every
   subsequent multiply/add using intermediate magnitudes and local ULPs. It is
   not a constant based on the final logit. Every selected panel action must
   have decision margin strictly above this envelope.
3. **Frozen Linux and Win32/Wine panel.** The panel is selected automatically
   from the full corpus by numerical/input stress only: smallest action margin,
   maximum observed-hazard count, maximum candidate width, maximum feature
   magnitude, and an identity-hash sample. The exact panel is frozen before
   Win32 execution. Both fused policies must match the reference action,
   preserve support and tie-break behavior, run below 4 ms p95, and record zero
   60 Hz deadline misses.

Exact actions do not alone waive a failed envelope, and an envelope does not
waive an action mismatch. If either fails, the scorer/export representation is
repaired and tested under another predeclared contract; a tolerance constant
is not enlarged after observing the result.

## Wine boundary

Only after the numeric contract, synthetic result, existing grouped cross-fit,
and native performance preflight pass may a candidate be exported. Online
evidence then starts fresh:

1. two complete original-Wine Stage-4 active canaries, with at least one actual
   intervention across them;
2. six alternating incumbent/candidate Stage-6 blocks (twelve complete Stages),
   natural RNG and continue-on-HIT;
3. an effective signal only if all runs are clean, the candidate is exercised
   in at least four blocks, aggregate candidate HIT is strictly lower, and it
   is no worse in at least four of six blocks.

This remains confirmation evidence rather than automatic promotion. A negative
result returns to learner/general-infrastructure analysis and autonomous data
collection, never manual scenario-specific data repair.
