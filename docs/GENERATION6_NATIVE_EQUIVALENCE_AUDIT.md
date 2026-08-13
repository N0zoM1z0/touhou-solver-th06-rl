# Generation 6 native-equivalence audit

## Verdict

Generation-6 round 3 is operationally closed as an offline rejection. Its
frozen production-fit gate required every raw portable/native actor score to
stay inside the predeclared float32 scale tolerance. The maximum observed
ratio was `1.4844`, above the immutable limit `1.0`, so no candidate artifact,
active state, Wine canary, or Stage-6 efficacy run was authorized. This gate is
not relaxed after seeing the result.

The rejection does not contradict the scheduler repair or the learner's
offline direction:

- both newly collected complete Wine Stages passed latency and every other
  collection gate over 56,957 frames with zero deadline misses;
- synthetic smoke passed;
- grouped cross-fit found an overall physical-HIT effect of `-8.0895`, with
  bootstrap upper 95% `-6.5683` and negative results in all three Stage
  cohorts;
- a separate checkpoint replay found 64/64 final Linux-native actions equal
  to the portable actions, with minimum selected-action margin `0.02292`.

The immediate issue is the definition and orchestration of numeric
conformance, not evidence that the learned policy is effective in Wine.

## Retained evidence

The complete 56-episode production fit was not discarded. Its immutable
checkpoint is
`artifacts/autonomous-generation-6-round-3/offline/candidate.fit.json`,
SHA-256
`e689b9b08902c2ecc761648898d682fca641468ff282aba495da74e1df2dc219`.
It contains all seven actor members, bootstraps, representation, support, and
out-of-episode advantage diagnostics. The fit took about 53 minutes, used no
Wine, remained within CPUs 0--31, peaked around 30.6 GiB RSS, and is reusable
for conformance development without retraining.

The frozen fit process reported:

- maximum raw score error: `0.0009765625`;
- maximum raw-score tolerance ratio: `1.48442326`;
- maximum support-distance error: `4.96896977e-6`.

Those first two maxima need not be the same element. An independent replay of
the same 64 contexts found 64 exact final choices, support error
`4.96896977e-6`, and a worst raw-score ratio `1.29764`: portable
`2.97855759`, native `2.97868919`, error `1.31607e-4`, allowed tolerance
`1.01420e-4`. The difference between run maxima is itself evidence that raw
NumPy/BLAS logit comparison is not a stable serving decision identity.

## Root cause

`IqlActorModel.predict` evaluates dense layers through NumPy linked to
Haswell/FMA OpenBLAS. The deployed C scorer evaluates the same frozen float32
arrays with explicit feature-major scalar accumulation and `std::tanh`; the
Win32 build is deliberately SSE2. All are legitimate float32 evaluations, but
their reduction order, fused multiply-add availability, and tanh
implementation differ.

The current tolerance is:

`1e-4 + 4 * float32_epsilon * abs(final_logit)`

That is not a valid forward-error bound for this network. State/action affine
layers, tanh, low-rank projections, the action-score dot product, and the
state/action latent dot product can contain large intermediate terms that
cancel in the final logit. A small final logit therefore does not imply a
small accumulation-error bound. Conversely, raw logits contain additive
components that cancel when the deployed policy subtracts the baseline mean.
The heuristic can reject an exact robust action without demonstrating a
decision change.

This is general to dense float32 native serving; it is not tied to TH06,
Stage, frame, spell, RNG, HIT location, or a manually selected action.

## Next contract

The next learner-only successor should consume the already registered round-3
corpus and reuse the exact fit checkpoint. It must not collect more Wine merely
to iterate on a numeric audit. Before any active canary, it should predeclare
and test a generic conformance contract with all of the following:

1. a deterministic scalar float32 reference matching the documented native
   layer/reduction order, tested separately on Linux and Win32/SSE2;
2. finite outputs and a conservative forward-error envelope based on the
   absolute intermediate products and operation counts, not final-logit
   magnitude alone;
3. exact support masks, exact mean-population proposal actions, and exact
   tie-break behavior on the frozen factual context panel;
4. an explicit lower bound between the selected proposal advantage and the
   numerical error envelope;
5. full fused Linux/Win32 policy exactness, p95 below 4 ms, and zero frame
   deadline misses;
6. unchanged native safety, reward, learner, intervention cap, and complete
   Wine canary/evaluation gates.

If the deterministic reference or error envelope cannot validate this
checkpoint, the candidate remains rejected. If they validate it, that is a
new predeclared infrastructure contract—not a retroactive pass for round 3.

## Orchestration repair

The old fitter raised before writing its negative candidate report, and the
round runner therefore retained `status: offline` rather than a clean
`offline-rejected` decision. The generic repair makes the fitter always write
its complete finite conformance/latency report and return status 1 for failed
gates. The runner recognizes only that explicit report as a model rejection,
skips online preflight, runs the conjunctive offline auditor, and records a
terminal rejection. It also resumes a content-bound `.fit.json` checkpoint
after a crash instead of spending another 53 minutes retraining. Unexpected
return codes, missing reports, or return/report disagreement still crash
fail-closed.
