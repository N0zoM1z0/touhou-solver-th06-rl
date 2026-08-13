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

## Frozen identity

The executable schedule is
`config/autonomous_generation6_stage6_pilot.json`, SHA-256
`b3ecb0720fa1ac0fb0f4ab8dbb90b8f286812b6468f3028d00a9f0672e6301cd`.
It binds implementation commit `909e201f8b7a84725970fa751b6564fea5def332`,
the successful v4 wiring evidence, all candidate/qualification/scorer hashes,
six policy seeds, arm order, Wine inputs, CPU partitions, run gates, and the
decision rule before any pilot game starts.

The exact trial-1 candidate state passed the 64-context portable/Linux/Win32
preflight before schedule commit: all proposals matched, Wine p95 was
`1.2417 ms`, maximum `1.3555 ms`, and >4 ms/deadline counts were zero. The
ignored report SHA-256 is
`a15bcf6e2de717dea978e2fcbf1338c0c43abe9dc411362453a2d9d9e2b16622`.

## Completed result

All six trials completed and passed every frozen gate. Incumbent HITs were
`[3, 11, 14]` for 28 total; candidate HITs were `[10, 8, 7]` for 25 total.
Candidate intervention counts were `[3, 2, 3]`, so all three candidate Stages
were exercised. The predeclared verdict is `effective-pilot-signal`, with an
aggregate reduction of three HITs and `authorization_eligible: false`.

Per-run live policy p95 stayed between `2.9619` and `3.0019 ms`; all six runs
had zero deadline misses. Native-safe-only publication, zero Bomb/corpus/infra
events, immutable hashes, complete Stage accounting, natural RNG, resource
partitions, and prefix cleanup all passed. The result SHA-256 is
`d4eba15f9db2881a39f6809072267eb22a0602bcdc6519a30b8a451a20a97b43`;
the complete ledger SHA-256 is
`2864f3a1f55d911d52ba1e0fff2a679dcce40220ff9bd33bfa56795affb9cf44`.

This is the positive directional evidence the small panel was designed to
detect, not a promotion result. The one-HIT-per-Stage mean is aligned with the
offline Stage-6 deployable estimate, while the 3--14 incumbent range shows why
a separately frozen larger confirmation is still required. The interpretation
and continuation boundary are recorded in
`docs/AUTONOMOUS_LEARNER_GENERATION_6_RESULT.md`.
