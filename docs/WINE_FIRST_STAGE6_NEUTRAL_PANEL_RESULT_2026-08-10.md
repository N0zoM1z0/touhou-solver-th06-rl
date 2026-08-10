# Stage 6 neutral-family Wine panel result (2026-08-10)

## Decision

The neutral family is rejected with zero candidates. Neither of the two new
episodes reached the predeclared sub31 failure context, so the fixed selector
found zero anchors and policy-faithful COW was forbidden. No neighboring
frame, alternate context, or replacement episode was inspected.

This result does not change the Stage 6 performance baseline. Complete natural
original-retail Wine Stage HIT count, with repeatable 0-HIT clears as the
target, remains the decisive metric. These stopped 0-HIT prefixes are filters,
not full-Stage clears, and HIT-continuation remains evaluation-only.

## Physical r7/r8 evidence

The protocol and selector were committed as
`b18046b849cf5a4569091105b9464d68e1d31419` before collection. Both runs used
the declared original retail executable, native kernel, plugin, immutable UCB
state, exploration zero, and frame-v5 first-failure contract.

| Run | Run ID | Terminal frame/context | Result | Report SHA-256 |
| --- | --- | --- | --- | --- |
| r7 | `20260810T150822Z-018087200` | 5382 / `boss:0:sub18:life_cb31:timer_cb31:nonspell` | 0 HIT, Hard empty | `3dbfbb08de9d05ee7c71e5ca1e0c3cab23826b94b34d1e72d1f2af12acc6cd76` |
| r8 | `20260810T151039Z-362645700` | 3245 / `boss:0:sub10:life_cb14:timer_cb13:nonspell` | 0 HIT, Hard empty | `1541d1c921e5b60f1ed41e062c4467c309731f8bc0894694de1a0dab20893db9` |

Each wrapper reported equal before/after policy, controller-policy, score, and
config hashes, no error, and an empty leftover-prefix process list. Policy
state SHA-256 remained
`e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0`.

The deterministic anchor report has SHA-256
`c8c1b71caeec141b7245a5a3aa094670379e4eb91e6f3694b8b9b932a00399d4`.
It reports 0/2 eligible anchors, `headless_cow_allowed=false`, and zero active
candidates. Therefore neither `stay` nor `stay_fast` was branched.

## Episode-grouped audits

Exact factual replay across r7/r8 covered 7,569 frozen-UCB calls with zero
recorded-incumbent mismatch, zero policy mismatch, and zero shadow action
contract violation. Its SHA-256 is
`b790076bec8a4f5abedd1cad9584003a77e2336ef35dd2689a0fbdea7a47b643`.
The two contexts are singletons and there is no repeated incumbent/baseline
opportunity; the grouped audit SHA-256 is
`30054bc54e36ed0f4df0658b3fa840860ec1b32083705aca2799ed562f98f4bd`.

A second read-only audit combined all eight frame-v5 episodes collected with
the same current native kernel and frozen state. It covered 36,106 policy
calls with zero mismatches. The episode support is sub10 four, sub31 three,
and sub18 one. Its factual-replay and grouped-audit SHA-256 values are:

- `c61cc4b07e657d5f143646baa507ead546401ed9c70c1e1bcc235c3955fbd646`;
- `0e717e186eb7f879c4aa9092e6b995fc3911468ca7072e3de4af652fc69e0ef4`.

The only repeated action opportunities are the three already closed by the
fixed r1--r3 and r4--r6 panels. r8 adds no new repeated pair. Older 23-episode
audits used a different native-kernel revision and remain historical
hypothesis evidence; they are not silently pooled with this current panel.

## Next bounded experiment

Do not collect more Wine episodes merely to rescue the neutral family and do
not fit another global classifier. The newly restored policy-faithful source
continuation defines a different counterfactual from the old offline-teacher
COW. It may be tested once on the already fixed sub10 r1/r2 anchors and the
already fixed `down_right`/`down_fast` pair, under a separately committed
protocol. This does not reopen or reinterpret the old teacher result: it asks
whether one substituted first action helps when the actual frozen UCB policy,
not the offline teacher, owns every later decision.
