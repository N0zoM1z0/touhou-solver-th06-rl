# Physical evidence ledger

This ledger records interpretation without rewriting immutable corpus runs.
HIT counts are comparable only for a complete Practice Stage with zero control
capture and infrastructure failures.

## Lunatic / Reimu-A / Stage 5

- `20260806T114830Z-235638200`: Supervisor reached Practice completion and
  recorded 5 HIT, but the controller had 10,917 coherent-capture failures.
  Large late-Stage intervals were unobserved, so HIT transitions in those
  intervals could not be counted or learned. This run is retained as bug and
  partial-trajectory evidence; **5 HIT is not a valid baseline**.
- `20260806T131607Z-870984500` at `cb9e851`: first complete Stage 5 run with
  zero control-capture and infrastructure failures. It recorded 19 HIT across
  19,495 learning-eligible elapsed frames, 543 Hard-empty frames, capture p95
  6.05 ms, and solve p95 1.02 ms. Dense native replay checked 64 samples with
  no unsafe divergence. This is the first trustworthy full-observation Stage
  5 baseline, not evidence of regression against the blind 5-HIT run.

The full-observation run expanded the online Stage 5 policy from 743 to 1,877
trained context-actions. Fourteen of its nineteen HITs occurred after frame
11,000, inside the interval that the earlier run did not observe coherently.
The completed policy update is retained so the next run can test whether this
new failure evidence improves play.

- `20260806T133608Z-895242200` at `8ace718`: deliberately switching focus
  exposed an infrastructure bug in the first background activity lease at
  frame 8,940. The lease called a nonexistent donor process method, the
  controller failed closed and stopped the exact game PID, and the incomplete
  Stage policy transaction was rolled back. This is infrastructure evidence
  only and is not a learning or HIT comparison run.
- `20260806T134224Z-637968400` at `a72d31a`: complete Stage 5 with 16 HIT,
  zero capture and infrastructure failures, and a committed policy update.
  Capture p95 was 5.76 ms, solve p95 was 0.97 ms, peak controller private
  memory was 65.3 MiB, and the compressed run was 196.2 MiB. Dense native
  replay checked 63 samples with no divergence. No focus-loss event occurred
  in this run (`background_reactivations=0`). The reduction from 19 to 16 HIT
  is chronological evidence, not yet a causal learning estimate.
- `20260806T135011Z-455869400` at `515ceb2`: complete Stage 5 with 17 HIT,
  zero capture and infrastructure failures, and a committed policy update.
  Capture p95 was 5.85 ms, solve p95 was 1.00 ms, and dense native replay
  checked 64 samples with no divergence. Two physical focus-loss events were
  repaired (`background_reactivations=2`) and the Stage still completed. This
  supplies the missing physical proof that background play survives switching
  away from TH06.

## Lunatic / Reimu-A / Stage 6

- `20260806T115616Z-821716900` at `756fa7b`: complete Stage 6 with 16 HIT,
  zero capture and infrastructure failures. Capture p95 was 6.08 ms, solve p95
  was 1.36 ms, and dense native replay checked 62 samples with no divergence.
  This is a trustworthy earlier Stage 6 baseline, although it predates Stage
  policy transactions and background-activity instrumentation.
- `20260806T135930Z-958959700` at `58b27b2`: complete Stage 6 with 15 HIT,
  zero capture and infrastructure failures, and a committed policy update.
  Capture p95 was 6.97 ms, solve p95 was 1.22 ms, stale retry rate was 0.417%,
  and peak controller private memory was 80.8 MiB. Dense native replay checked
  57 samples with no divergence. Six focus-loss events were repaired and the
  Stage still completed. The compressed run was 280.4 MiB.
- `20260806T140919Z-138042700` at `7e48836`: complete Stage 6 with 20 HIT,
  zero capture and infrastructure failures, and a committed policy update.
  Eighteen HITs followed a Hard-empty control dead end; two were classified as
  latency observation gaps. Capture p95 was 7.43 ms, solve p95 was 1.37 ms,
  solve p99 was 6.99 ms, and stale retry rate was 0.840%. Peak controller
  private memory was 74.1 MiB and the compressed run was 317.1 MiB. Dense
  native replay checked 61 samples with no divergence, and two focus-loss
  events were repaired. The boss sub31 spell context exposed a concentrated
  geometry hotspot: despite only 0--170 live bullets, it carried as many as 48
  lasers, with solve p95 11.68 ms and stale retry rate 11.55% across 2,286
  frames. This is the pre-optimization comparison for prepared hazard buffers.
- `20260806T143017Z-003184000` at `245bc81`: complete Stage 6 with 17 HIT,
  zero capture and infrastructure failures, and a committed policy update.
  Sixteen HITs followed a Hard-empty control dead end and one was a latency
  observation gap. Overall capture p95 was 6.96 ms, solve p95 was 1.21 ms,
  solve p99 was 3.51 ms, and stale retry rate was 0.322%. Peak controller
  private memory stayed bounded, the corpus was 286 MiB, and dense native
  replay checked 60 samples without divergence. In the sub31 spell's exact
  48-live-laser stratum, prepared buffers plus scalar source-equivalent laser
  projection reduced solve p50 from 5.29 to 2.34 ms, solve p95 from 14.26 to
  9.56 ms, stale retry rate from 30.48% to 9.56%, and observation-gap rate
  from 12.51% to 5.29%. This is physical performance evidence; the 20-to-17
  HIT change remains chronological rather than a causal learning estimate.

  A post-run state-alias audit found 29,851 learning-eligible one-frame
  transitions. Under the original coarse UCB, 97.84% belonged to a
  context-action group containing multiple current/baseline/Hard/legal/
  clearance signatures, and 22 repeated groups mixed a safe next Hard set
  with a next-Hard-empty outcome. Counterfactually partitioning the same
  behavior by the hierarchical v2 features reduced those figures to 4.59%
  and 9 groups. This is evidence for the new fine partition, not an off-policy
  claim that v2 would have selected a different action in the recorded run.
- `20260806T145106Z-285711500` at `eb290c7`: first complete physical Stage 6
  run collecting hierarchical-v2 feedback. It recorded 21 HIT, all after a
  Hard-empty control dead end, with zero capture/infrastructure failures and a
  committed policy update. Capture p95 was 7.05 ms, solve p95 was 1.31 ms,
  stale retry rate was 0.106%, peak controller private memory was 118.4 MiB,
  and the corpus was 289 MiB. The 500+ live-bullet control p95 was 9.70 ms
  with zero stale retries; dense native replay checked 64 samples without
  divergence. Three focus-loss events were repaired.

  This run contributed 30,486 fine feedback transitions over 16,055 trained
  fine context-actions. A same-run audit reduced the multiple-physical-
  signature record rate from 96.66% under the coarse key to 4.64% under v2.
  Nineteen of the 26 next-Hard-empty precursor transitions were singletons in
  their fine group, so the HIT count is first-coverage data rather than an
  estimate of learned improvement. The run also exposed 215,700 fine legal-
  opportunity keys that did not participate in decisions; omitting that
  redundant checkpoint table retains every reward/trial and cuts the same
  packed state from 2.69 MiB to 1.20 MiB.
- `20260806T151806Z-653426100` at `ddd762f`: complete Stage 6 with 12 HIT,
  all after a Hard-empty control dead end, zero capture/infrastructure
  failures, and a committed policy update. Capture p95 was 6.58 ms, solve p95
  was 1.32 ms, solve p99 was 2.75 ms, and stale retry rate was 0.0408%.
  The 500+ live-bullet stratum had control p95 8.90 ms and zero stale retries.
  Dense native replay checked 60 samples with no divergence; four focus-loss
  events were repaired. Peak controller private memory was 116.2 MiB after
  the fine policy grew to 61,286 observations, the packed policy was 1.94 MiB,
  and the compressed corpus was 309 MiB.

  In the exact sub31/48-live-laser stratum, precomputed laser bases and
  deferred exact distance evaluation reduced physical solve p50 from 3.71 to
  2.52 ms, p95 from 8.84 to 3.71 ms, p99 from 10.43 to 6.33 ms, and stale
  retry rate from 2.62% to 0.118% relative to the preceding hierarchical-v2
  run. Overall solve p95 stayed nearly flat because ordinary frames already
  dominated that percentile; overall solve p99 fell from 5.96 to 2.75 ms.
  The 21-to-12 HIT change is chronological evidence only and was not used to
  tune policy parameters.

  Cross-run support against `20260806T145106Z-285711500` exposed the next
  learning-infrastructure limit. Coarse state reused prior context-actions for
  73.21% of records but aliased multiple physical signatures for 96.77%.
  Exact hierarchical-v2 state reduced aliasing to 5.45% but reused prior
  support for only 12.01% of records and none of 19 next-Hard-empty precursors.
  A counterfactual phase-clock/current/baseline middle key reused 29.41% of
  records while cutting the coarse alias rate to 47.42%. These descriptive
  corpus figures motivate a three-level coarse -> phase/control -> exact
  frontier backoff; they are not an off-policy claim about HIT reduction.
- `20260806T153702Z-009110000` at `846b17f`: first complete physical Stage 6
  using the three-level v3 policy. It recorded 19 HIT, all after a Hard-empty
  control dead end, with zero capture/infrastructure failures and a committed
  v3 checkpoint. Capture p95 was 6.74 ms, solve p95 was 1.46 ms, solve p99 was
  2.83 ms, and stale retry rate was 0.0296%. The 500+ bullet control p95 was
  10.18 ms with a 0.257% stale rate, still well inside the 16.67 ms / 2%
  acceptance gate. Dense native parity checked 64 samples without divergence;
  three focus-loss events were repaired. Peak controller private memory was
  91.2 MiB, the corpus was 308 MiB, and the checkpoint held 93,627 middle and
  93,627 fine observations.

  Against the union of the two v2 corpus runs loaded at startup, v3's current
  records had prior support rates of 80.84% coarse, 34.68% middle, and 15.64%
  exact-fine. Their within-run multiple-signature alias rates were 96.36%,
  45.48%, and 5.32%, respectively. Two of 25 next-Hard-empty precursors had
  middle support versus none at exact-fine; this is sparse descriptive
  evidence that the intended intermediate reuse path exists. It is not a
  causal interpretation of the 12-to-19 HIT change.
- `20260807T033012Z-128212400` at `bd26231`: first complete physical Stage 6
  using `survival-reserve-hit-trace-v2` after replaying 63 exact-scope,
  complete logged-action runs into the same online UCB. It recorded 10 HIT,
  zero capture/infrastructure failures, and a committed Stage policy update.
  All 10 physical HIT events credited recent actions in the same source phase:
  checkpoint counters moved from 745 to 755 physical HIT and from 728 to 738
  credited HIT, while uncredited HIT remained 17. The prior interrupted run
  `20260807T032052Z-684850600` was rolled back and is not learning evidence.

  Capture p95 was 6.42 ms, solve p95 was 1.58 ms, observation-gap rate was
  0.0514%, and stale-retry rate was 0.0857%. Dense native parity checked 52
  samples with no unsafe or conservative divergence; the audit found no scope
  pollution or integrity errors and classified the infrastructure as stable
  for learning. Around the first HIT, the 59-frame inactive interval had a
  maximum 32.85 ms controller gap and control resumed immediately afterward.
  The next 60-second maintenance event was a 9.16 ms reload poll instead of
  the former 6-second checkpoint write. This physically validates that HIT
  feedback and ordinary maintenance no longer stall play. The 10-HIT result
  is chronological evidence, not yet a frozen-checkpoint causal comparison.
