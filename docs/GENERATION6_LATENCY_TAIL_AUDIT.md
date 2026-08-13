# Generation 6 scheduler tail-latency audit

## Verdict

Generation-6 autonomous round 1 remains immutably invalid. Episode 11 crossed
the frozen 60 Hz scorer deadline twice, so its old contract is not relaxed and
its ledger is not resumed. The failure is a demonstrated host-scheduling and
telemetry infrastructure defect, not a model-width, Stage-specific, safety, or
factual-corpus defect.

The generic repair is:

1. keep the existing disjoint 0--7 game / 8--31 controller CPU partition;
2. run both latency-sensitive Wine process trees under bounded Linux
   `SCHED_OTHER` priority at exact nice `-10`;
3. drop sudo authority before executing Wine and attest the effective UID,
   GID, CPU set, scheduler class, and nice value in every run report;
4. materialize complete policy metrics once per game second plus one exact
   final snapshot, instead of sorting and copying diagnostic state on every
   60 Hz factual frame;
5. retain the unchanged p95-below-4-ms and zero-16.67-ms-miss gates.

This changes no game timing, native-safe action, actor score, exploration
probability, factual option, reward, HIT count, learner, or promotion rule. It
does not use a real-time scheduler and never occupies more than the authorized
32 logical CPUs.

## Failure reconstruction

The immutable round ledger is SHA-256
`5313b872ff92344271ef312e644d71ed96d7cce977170301754c5573957cdbb9`.
Ten episodes passed. Episode 11 completed a natural-RNG original-Wine Stage 6,
conserved all 55 physical HITs, loaded the immutable collection policy once,
recorded complete propensities, stayed inside native safety, used no Bomb, and
cleaned its prefix. Its report is SHA-256
`162f6b5777623133b8cfb8823681d51707d90308ddaaca01b7b454c4e8f04549`.

The actor p95 was still `2.9630 ms`, but two calls exceeded `16.6667 ms`:

| Frame | Actor miss count | Capture | Whole solve | Nearby whole-loop tail |
|---:|---:|---:|---:|---:|
| 3458 | 1 | 8.112 ms | 24.010 ms | frame 3453 solve 21.623 ms |
| 3580 | 2 | 8.299 ms | 23.021 ms | frames 3577/3584 capture 22.128/21.179 ms; frame 3584 solve 52.992 ms |

Both occurred in one roughly 2.1-second wall-clock window. Capture and other
controller work stalled at the same time as the scorer. Input width did not
grow discontinuously, and the fused scorer has fixed bounded work. The prior
six-run Stage-6 pilot processed 25,963 actor boundaries with no deadline miss;
the first ten accepted round episodes likewise had none. A rejected earlier
capture-incoherent attempt had one same-shaped scheduling tail, which is
retained as infrastructure evidence but was never learner-visible.

An isolated 20,000-call Win32 factual-width replay produced p95 `1.2508 ms`,
maximum `1.4047 ms`, and zero misses. This rules out a deterministic slow
context or an algorithmic complexity spike.

## Controlled reproducer

`scripts/audit_generation6_latency_tail.py` executes the exact frozen
candidate, 64 maximum-computational-width factual Wine contexts, the canonical
SSE2 Win32 DLL, and 10,000 calls while 32 ordinary CFS workers contend on the
same allowed CPU set. It then repeats the identical scorer inputs under the
bounded priority wrapper. It never launches TH06 and is not gameplay or
promotion evidence.

The formal report is SHA-256
`8526220a0fc1d467bee4b9c24d4e6fa8b786560093a03ca91fabcb63ee5c591f`:

| Condition | p50 | p95 | max | >4 ms | >16.67 ms |
|---|---:|---:|---:|---:|---:|
| Equal-priority contention | 1.0164 ms | 1.3199 ms | 24.6883 ms | 53 | 29 |
| Exact nice -10, same contention | 1.0174 ms | 1.3137 ms | 9.4206 ms | 12 | 0 |

Portable, Linux, and Win32 proposals remained exact in both conditions. The
near-identical medians and p95 values, large baseline-only maximum, and removal
of all misses under bounded priority reproduce the Linux CFS preemption tail.
They do not support changing actor math or a gameplay feature.

Reproduction command:

```bash
.venv/bin/python scripts/audit_generation6_latency_tail.py \
  --output artifacts/autonomous-generation-6-latency-audit/recheck.json \
  --repetitions 10000 --stress-seconds 45
```

## Repair contract

`scripts/exec_bounded_priority.py` is deliberately narrower than arbitrary
sudo execution. It accepts only nice `-15..0` and an explicit inherited CPU
set, applies `setpriority` and `sched_setaffinity`, initializes the invoking
user's groups, drops to the non-root sudo UID/GID, writes an exclusive
attestation, and then executes the child. TH06 and the controller therefore do
not retain elevated authority. Every complete-run audit validates the
attestation rather than trusting command-line intent.

The controller's per-frame factual record still contains immutable policy
identity. Only redundant diagnostic aggregation is sampled: full counters are
written every 60 frames and once at finalization. The final snapshot remains
authoritative for intervention, action, p95, and deadline audits. Corpus
features, policy decisions, and option traces are unchanged.

Round 1 is not rewritten. A successor contract must bind the repaired source,
priority values, formal stress report, new identity, and new Wine attempts
before any further outcome-facing run.
