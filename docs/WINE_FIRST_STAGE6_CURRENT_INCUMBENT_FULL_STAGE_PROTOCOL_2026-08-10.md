# Current-kernel Stage 6 incumbent full-stage calibration (predeclared 2026-08-10)

## Purpose

Measure the metric that ultimately matters: physical HIT count over one
complete original-retail Wine Stage 6 trajectory with the current native
kernel and frozen incumbent.  Historical comparable frozen-UCB runs recorded
8 and 9 HITs, but used native DLL SHA-256
`d5c79c30b4d46c72f0521d9653d5d99693c0fbc966e241f554732ad3ade3a37e`.
The current first-failure corpus uses a different rebuilt DLL identity, so one
current calibration is useful before further residual design.

This is evaluation-only.  HIT continuation, post-HIT actions, and the trace
may not enter training, COW anchor selection, policy-state updates, or model
promotion.  It is a point estimate, not a powered A/B and not a reusable sole
control for a later candidate.

## Frozen run

Run exactly one complete Lunatic / Reimu-A / Practice Stage 6 attempt named
`wine-retail-stage6-framev5-frozen-ucb-current-natural-r1` with:

- original retail 1.02h executable SHA-256
  `9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`;
- native DLL SHA-256
  `71b27bef942928dfc04b6a489ef261cb6485cd5bab2cd4d4ebf4081991a22b5b`;
- policy plug-in SHA-256
  `4d7f10925731d7f83389aaa8c2aa942d7ed156de54791767ff2ced802483bbf2`;
- immutable policy state SHA-256
  `e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`;
- exploration zero, no corpus, no post-run learning audit;
- natural Practice timing and stage completion, with evaluation-only life/HIT
  continuation so all physical HITs can be counted;
- ordinary pipes, never a PTY, exact input release and exact trial PID cleanup.

The wrapper must return zero, preserve policy/config hashes, reach the natural
Practice result, record zero Bombs, and report an empty dedicated-prefix
process list.  The score file is restored from its fixed template before the
trial but original retail may write normal score progress during play; its
post-run hash is evidence, not an immutability gate.  Any authority failure is
counted and retained rather than hidden.  Do not rerun merely to improve the
HIT count.

## Interpretation

Report complete-stage physical HIT count first, followed by completion,
Bombs, authority failures, stale delivery, and timing.  A 0-HIT result is one
successful calibration, not yet repeatable NMNB.  A nonzero result becomes the
current point baseline.  A future candidate reaches this gate only after
Wine shadow and alternating first-failure canary; final evidence must be fresh
alternating complete-Stage incumbent/candidate HIT-count trials.
