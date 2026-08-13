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
