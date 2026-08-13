# Generation 6 autonomous round 3

## Authorization

Round 3 is the first successor whose complete host path passed both the formal
scheduler-tail stress audit and a real original-Wine no-corpus startup smoke.
Its machine contract is `config/autonomous_generation6_round3.json`, SHA-256
`a4276c321d92ccf8ee17aa8cf7cad57934e7c0742135402bc781ea048ff6e960`.
It binds implementation checkpoint `74774b0` and is frozen before any new
complete-Stage outcome.

Round 1 remains invalid after its two deadline misses. Round 2 remains
startup-aborted after exposing sudo monitor/exec-child PID ambiguity; it
produced no new corpus and is not resumed. Their standards and ledgers are not
edited.

## Bound infrastructure evidence

The scheduler stress report remains SHA-256
`8526220a0fc1d467bee4b9c24d4e6fa8b786560093a03ca91fabcb63ee5c591f`.
Under identical 32-worker CFS contention, equal-priority Win32 scoring had 29
frame-deadline misses while bounded `SCHED_OTHER` nice `-10` had zero. Proposal
actions remained exact.

The separately bound startup report is SHA-256
`dfd0515f8c5b282783013c3ffc90eccbbf4e4ed6d95e27dc90d4ad5762902915`.
It is a 0.25-second, zero-decision, zero-corpus, non-efficacy smoke. It proves:

- GDB normalized the actual attested Wine exec-child, not sudo's monitor;
- game CPUs 0--7 and controller CPUs 8--31 both ran exact
  `SCHED_OTHER/-10` after complete privilege drop;
- the immutable collection policy loaded and the controller returned zero;
- no corpus run ID was written and no prefix process remained.

The round runner validates these semantics, not only the report hash, before
materializing or launching anything.

## Data and learning contract

The data rule is unchanged from the audited round-2 design. The runner reuses
only the entire ten-row consecutive `passed=true` prefix from immutable round
1. Every source report, manifest, run, and ledger is content-bound and fully
revalidated. Failed round-1 episode 11 and all round-2 startup artifacts are
excluded from training.

The prefix has Stage counts 4/3/3 for Stages 4/5/6. New episode 11 is Stage 6
and new episode 12 is Stage 5, restoring the predeclared balanced 4/4/4 cohort.
Both use fresh policy seeds, natural unread RNG, normal-speed original Wine,
complete-Stage HIT continuation, zero Bomb, native-safe actions only, and the
unchanged automatic 0.50 actor / 0.25 uniform / 0.25 inverse-ESS exploration.
There is no spell, phase, frame, RNG, HIT, action, or failure-location
targeting.

After both rows pass every gate, the twelve-run source is appended once to the
immutable registry. The fit consumes all sequential training sources. Reward
remains physical HIT only, with factual semi-Markov transitions, `gamma=1`,
terminal value zero, grouped cross-fit, the seven-member IQL population, and
no shaping. Synthetic, cross-fit, Linux native, and actual Win32/Wine smoke
must pass before two active Stage-4 canaries.

Final evidence remains six alternating Stage-6 incumbent/candidate blocks. An
effective signal requires all reports valid, candidate exposure in at least
four blocks, strictly fewer aggregate candidate HITs, and candidate no worse
in at least four blocks. Promotion is forbidden. A negative result leads to
learner/general-infrastructure analysis, never manual data-distribution repair.

## 2026-08-13 execution progress

The two new complete-Stage collection rows both passed every frozen gate:

| Episode | Stage | HIT | Options | Actor p95 | >4 ms | Deadline misses |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 6 | 39 | 2,879 | 3.1006 ms | 23 | 0 |
| 12 | 5 | 32 | 1,421 | 3.1105 ms | 11 | 0 |

Both used natural RNG, conserved physical HITs across the Wine report,
manifest, and factual intervals, recorded complete behavior propensities,
selected only native-safe actions, used no Bomb, preserved the immutable
policy, attested both process partitions at exact `SCHED_OTHER/-10`, and left
no prefix process. Together they covered 56,957 observed game frames without
a scorer deadline miss. The old failed episode had two misses by frame 3,580;
this long-run differential supports the scheduling diagnosis and repair.

The new registry source contains exactly twelve audited runs and has inventory
SHA-256
`81e2891e5e15b07cf10f06e15842d593d8ceff1d5b89728012691629aeb20357`.
It was checkpointed in commit `53da64a`; the ten reused rows and two new rows
remain one immutable source for future learner variants.

All-registry grouped cross-fit then consumed 56 episodes and 167,250 factual
options. Its physical-HIT-only policy DR estimates were:

| Cohort | Episodes | Mean HIT effect | Bootstrap upper 95% | Beneficial episodes |
|---|---:|---:|---:|---:|
| Overall | 56 | -8.0895 | -6.5683 | 94.6% |
| Stage 4 | 19 | -12.2189 | -9.2508 | 100% |
| Stage 5 | 4 | -14.2670 | -10.5715 | 100% |
| Stage 6 | 33 | -4.9632 | -3.7991 | 90.9% |

The worst seven-member leave-one-out upper bound remained `-6.3652`, and
synthetic causal/null smoke passed. These are encouraging offline diagnostics,
not Wine efficacy. Only 37 of 167,250 options met the complete-population
proposal rule in cross-fit, so active exposure remains a material uncertainty
that the native smoke and Wine canary must test.
