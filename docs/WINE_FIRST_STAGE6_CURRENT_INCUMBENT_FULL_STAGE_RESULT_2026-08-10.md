# Current-kernel Stage 6 incumbent full-stage result (2026-08-10)

## Core result

Frozen `phase-local-hierarchical-ucb-v4` completed the full natural
original-retail Wine Stage 6 trajectory with **10 physical HITs and zero
Bombs**.  This is the current-kernel point baseline and is not NMNB.

Historical frozen-UCB complete runs recorded 8 and 9 HITs with a different
native DLL identity.  The new value is one additional RNG trajectory, not
evidence that the native rebuild itself caused a regression and not a powered
comparison.  Any future residual must be tested against fresh interleaved
complete-Stage incumbent controls; it may not claim improvement merely by
beating this one value.

## Bound evidence

Run artifact:
`artifacts/wine-retail-stage6-framev5-frozen-ucb-current-natural-r1`.

- report SHA-256:
  `1d6811b9b95abd31039fb6ed562c807b6b88f00974275f8775c28a83d79247b5`;
- trace SHA-256:
  `4ea975c5c788cb21b1baf4fce834b1786053ed506ed98bde9cffedd22fc6f96e`;
- original executable SHA-256:
  `9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`;
- current native DLL SHA-256:
  `71b27bef942928dfc04b6a489ef261cb6485cd5bab2cd4d4ebf4081991a22b5b`;
- policy plug-in SHA-256:
  `4d7f10925731d7f83389aaa8c2aa942d7ed156de54791767ff2ced802483bbf2`;
- immutable policy-state SHA-256 before and after:
  `e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`;
- controller return code 0, no wrapper error, natural Practice result,
  31,075 trace rows through frame 31,597, and no leftover dedicated-prefix
  process.

The run was explicitly marked `hit-continuation-benchmark`, used `--no-corpus`,
and made no policy update.  The score file changed because original retail
wrote gameplay score progress; it was restored from the fixed template before
the trial and is not an immutable policy input.  The protocol's request to
preserve a post-run score hash was therefore incorrect; policy and config
hashes are the relevant immutable checks.

The report also exposes `physical_hit_events=8` from the policy metrics.  That
field is the counter already present in the imported frozen state: immutable
evaluation suppresses failure feedback, so it did not change during this run.
The authoritative per-run metric is `physical_hits_in_run=10`, independently
reproduced by ten trace rows with `reason=physical-hit`.

## Failure distribution

HIT frames were:

`3060, 3529, 8540, 9216, 10162, 17680, 22389, 22946, 23487, 31109`.

Automatically derived source-context counts were:

- sub10 nonspell: 2;
- sub31 spell: 1;
- sub19 nonspell: 1;
- sub20 nonspell: 1;
- sub35 spell: 1;
- sub39 spell: 3;
- sub41 spell: 1.

All 10 HITs had at least one `control-dead-end:Hard safe set empty` row in the
preceding 30 frames.  The trace contains 38 Hard-empty rows and 110 stale
retries.  This does not transfer collision authority to a learned policy: a
residual can only try to preserve future maneuverability by choosing earlier
inside the already native-safe local set.

## Consequence

The complete-stage metric validates the Wine-first priority: current frozen
UCB is roughly in the same non-NMNB range as the historical 8/9-HIT controls,
and broad offline/headless gains have not solved the physical task.  Because
the sub10 source support gate created zero candidates, do not run a candidate
full Stage yet.  Acquire a small new first-failure Wine panel, group by
episode, and use only new repeated generic regions to propose another bounded
residual.  Complete HIT continuation remains quarantined from training.
