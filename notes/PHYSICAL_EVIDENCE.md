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
