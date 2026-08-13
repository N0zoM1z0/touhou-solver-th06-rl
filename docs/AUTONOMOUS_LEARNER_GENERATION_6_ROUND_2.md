# Generation 6 autonomous round 2

## Frozen decision

Round 1 remains invalid under its original contract. It stopped after its
eleventh collection Stage because two scorer calls missed the frozen 60 Hz
deadline. The failure is not reclassified and the old threshold is not
relaxed. The controlled diagnosis and generic repair are recorded in
`GENERATION6_LATENCY_TAIL_AUDIT.md`.

Round 2 is a separately frozen successor. Its machine contract is
`config/autonomous_generation6_round2.json`, SHA-256
`e49e363ba0da7a2b89ddb78116612a9ca164d022188db787376c0e0390c09c4f`.
It binds implementation checkpoint `b2b0b4e`, the immutable round-1 ledger,
each reused report and corpus run, the formal scheduler stress audit, the
unmodified learner and reward, two new collection rows, every later policy
seed, and every runtime input.

The successor changes only proven infrastructure:

- the game and controller remain on disjoint CPUs 0--7 and 8--31;
- both Wine process trees use exact Linux `SCHED_OTHER` nice `-10`;
- the wrapper drops sudo authority before Wine starts and every report must
  attest the exact UID, GID, CPU set, scheduler, and nice value;
- redundant aggregate policy telemetry is materialized once per 60 factual
  frames and at finalization instead of being sorted/copied every frame.

The original game remains normal speed. Native safety, observation, action
scores, option horizon, exploration probabilities, corpus transitions,
physical-HIT-only reward, learner, intervention limit, and evidence gates are
unchanged. Real-time scheduling is forbidden.

## Outcome-blind prefix reuse

The first ten round-1 episodes passed every conjunctive gate under their
frozen contract. Replaying them would spend hours drawing different natural
RNG outcomes and would discard valid immutable facts. Round 2 therefore reuses
exactly the entire consecutive `passed=true` prefix before the first failed
ledger row. This rule is evaluated mechanically and cannot omit a passing row,
skip around a failure, or inspect HIT count, spell, phase, frame, action, RNG,
or failure location.

For each of those ten episodes the new contract binds:

- the prior invalid ledger and its SHA-256;
- episode, Stage, and behavior-policy seed;
- the original complete-Stage report and SHA-256;
- the corpus path, manifest SHA-256, and run SHA-256;
- the complete transition-schema and chunk-integrity audit.

The runner creates hard-linked, content-validated entries under the new
accepted-corpus root. This avoids a second 7+ GiB copy; neither the registry nor
training mutates corpus files. The failed old episode 11 is quarantined and is
never learner-visible. Reuse is a data-plane operation, not a favorable-outcome
selection.

The passing prefix contains Stage counts 4/3/3 for Stages 4/5/6. The only new
collection rows are therefore predeclared episode 11 Stage 6 and episode 12
Stage 5, producing the same balanced 4/4/4 cohort intended by round 1. They use
fresh behavior seeds, original-Wine natural RNG, complete-Stage HIT
continuation, zero Bomb, complete propensities, and the unchanged automatic
actor/uniform/inverse-ESS mixture.

## State machine and stopping rules

The state machine remains:

`10 audited reused facts + 2 new Wine Stages -> append one 12-run registry source -> all-corpus refit -> synthetic/cross-fit/native/Wine smoke -> 2 Stage-4 Wine canaries -> 6 paired Stage-6 blocks`

Every new Wine collection, canary, and evaluation report must pass both the old
latency rule—p95 below 4 ms and zero 16.67 ms deadline misses—and the new exact
priority attestation. The formal stress audit is only infrastructure evidence;
it is not gameplay or promotion evidence.

If either new collection row fails any frozen gate, the successor becomes
invalid. It is not resumed under a looser standard. If offline smoke rejects
the fit, no active Wine candidate is launched. If the canary fails, evaluation
does not start. Final efficacy still requires all twelve paired reports,
candidate exposure in at least four blocks, fewer aggregate candidate HITs,
and no worse HIT count in at least four blocks. Promotion remains forbidden
regardless of sign.

No result authorizes manual distribution repair. A weak or negative outcome is
evidence about the learner or general infrastructure and leads to another
predeclared autonomous iteration.

## Execution result: startup-aborted

Round 2 did not enter a new collection episode. Its first game launch exposed
that sudo returned a monitor PID while the priority wrapper attested a distinct
Wine exec-child PID. The startup GDB script attached to the monitor and failed
closed before controller launch. Two immediate infra retries saw the same
private prefix's transient `dbus-launch` helper and also refused to proceed.

The retained ledger, SHA-256
`5bfd8d521e65ee673be5ea13661fc64903aed95e0d99c49dbb73eb0f8c60eb22`,
contains only the ten mechanically reused round-1 rows. No new trajectory,
option, HIT outcome, or registry source was produced. It remains a historical
`collecting` crash ledger rather than being edited after the fact. Round 2 is
operationally closed and must not be resumed because its frozen runner no
longer matches the additional generic repair.

After repairing attested child-PID selection and private-X cleanup ordering, a
0.25-second original-Wine startup smoke completed with GDB normalization,
exact game/controller priority attestations, controller return code zero,
immutable policy state, zero decisions, zero corpus IDs, and zero leftover
prefix processes. Its report SHA-256 is
`dfd0515f8c5b282783013c3ffc90eccbbf4e4ed6d95e27dc90d4ad5762902915`.
This is infrastructure evidence only. A new round must bind it before any
complete-Stage attempt.
