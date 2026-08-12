# Generation-6 complete-Stage-6 efficacy pilot

## Question and evidence boundary

The pilot asks one deployment question: does the frozen Generation-6
propensity-bounded actor target reduce aggregate physical HITs relative to its
exact incumbent in normal-speed, natural-RNG, complete original-retail Wine
Stage 6?

This is deliberately separate from corpus collection, offline qualification,
and the Stage-4 wiring canary. It creates no training corpus and cannot change
the learner, fitted population, support, reward, option horizon, intervention
probability, or data registry. The Stage-4 canary's HIT count and the historical
Generation-2 17-HIT baseline are not controls for this experiment.

## Arms

Both arms use the same immutable Generation-6 plug-in, complete seven-member
native scorer, reactive incumbent, native hard-safe-set publisher, original
controller, CPU partition, and latency gates.

- `incumbent` loads shadow mode. It still scores every option boundary but
  publishes the reactive baseline with probability one.
- `candidate` loads active mode. It may replace the incumbent only with the
  supported actor proposal sampled under the already qualified
  `min(0.10, 2 * 0.10 / native_safe_set_size)` target.

Bomb is unavailable in both arms. Every trial has a separately predeclared
policy seed; the game RNG is natural and neither read nor fixed.

## Pilot size and ordering

The first decision panel is three temporal blocks, six complete Stages. Arm
order reverses in the middle block:

1. incumbent, candidate;
2. candidate, incumbent;
3. incumbent, candidate.

This small panel is a cost-controlled directional pilot, not final policy
promotion. It is larger than the old two-pair Generation-2 observation and
uses the exact current incumbent, but it is not presented as a high-power
frequentist proof. If it shows an effective signal, a separately frozen larger
confirmation is still required before promotion.

## Predeclared decision rule

All six trials must complete with immutable state, original hashes, natural
RNG, native-safe-only publication, zero Bomb/corpus/infrastructure events,
clean prefix teardown, live p95 below 4 ms, and zero deadline misses. Candidate
intervention must be exercised in at least two of its three Stages.

Conditional on those validity and exposure gates:

- candidate aggregate HITs strictly below incumbent is
  `effective-pilot-signal`;
- equal or greater candidate aggregate HITs is
  `no-effective-pilot-signal`;
- insufficient candidate exposure is `inconclusive`;
- any runtime/integrity failure is `invalid`.

No spell, frame, RNG, failure location, action, or interim HIT result may alter
the schedule or rule. The resumable ledger appends each complete trial before
the next starts; an audited failed row stops the panel.
