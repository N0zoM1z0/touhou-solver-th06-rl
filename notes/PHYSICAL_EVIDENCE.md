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
