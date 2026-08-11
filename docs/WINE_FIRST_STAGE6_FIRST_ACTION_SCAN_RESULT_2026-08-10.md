# Stage 6 first-action scan result (2026-08-10)

## Decision

The predeclared discovery scan creates zero candidates. `down_right`, the
frozen-UCB incumbent action, is the unique robust winner across all 14
retail-native-safe first actions. The conditional r6 confirmation is therefore
forbidden and was not run.

This result changes no resident policy. The decisive policy metric remains HIT
count in alternating complete natural original-retail Wine Stage 6 runs,
ultimately including repeatable 0-HIT clears. No candidate has reached that
gate. Full-Stage HIT continuation remains evaluation-only and cannot enter
training.

## Frozen protocol and evidence

The protocol and mechanical selection rule were committed as `9dc2083` before
the scan. Discovery used panel-2 r5 run
`20260810T134540Z-909617800`, sequence 6175 / retail frame 6834, source commit
`ed8c0730480b6cc15add99167c519f58d96421ff`, Linux source binary SHA-256
`45f933a87837afcd4c73973385dbf2fb52ddadb6174c80a4674f2bac66f11724`,
a 600-tick branch horizon, and teacher horizon 12.

The COW document is
`artifacts/wine-first-stage6/framev5-panel2-r5-seq6175-all-retail-safe-enemy-motion-v2.json`,
SHA-256 `6a580de57a3448b289645d6a75802b33893bf76bd0ab8bb29b6ce9b442d746b7`.
It verified exact retail/source checkpoint state and the retail native Hard set
under delivery `(0,1,2,3)` before branching. Source continuation retained
synchronous delivery `(0,)`, native revalidation, and Bomb prohibition.

## Outcomes

The robust rank is `(complete, no-death, survival, width-bucket,
reserve-bucket)`. It deliberately discards exact pixel differences inside a
bucket.

| First action | Terminal | Ticks | Minimum safe width | Reserve | Robust rank |
| --- | --- | ---: | ---: | ---: | --- |
| `stay` | tick limit | 600 | 3 | 71.2033 | `(1,1,600,1,5)` |
| `up` | authority failure | 230 | 2 | 77.2450 | `(0,1,230,0,0)` |
| `down` | tick limit | 600 | 3 | 71.2033 | `(1,1,600,1,5)` |
| `left` | tick limit | 600 | 5 | 15.5808 | `(1,1,600,2,2)` |
| `right` | tick limit | 600 | 5 | 18.1619 | `(1,1,600,2,3)` |
| `up_left` | authority failure | 231 | 1 | 75.3038 | `(0,1,231,0,0)` |
| `up_right` | authority failure | 230 | 2 | 77.2450 | `(0,1,230,0,0)` |
| `down_left` | tick limit | 600 | 3 | 71.2033 | `(1,1,600,1,5)` |
| `down_right` | tick limit | 600 | 5 | 112.4410 | `(1,1,600,2,5)` |
| `stay_fast` | tick limit | 600 | 3 | 71.2033 | `(1,1,600,1,5)` |
| `down_fast` | tick limit | 600 | 3 | 71.2033 | `(1,1,600,1,5)` |
| `left_fast` | authority failure | 228 | 4 | 1.2158 | `(0,1,228,0,0)` |
| `up_right_fast` | tick limit | 600 | 3 | 71.2033 | `(1,1,600,1,5)` |
| `down_left_fast` | tick limit | 600 | 5 | 18.1619 | `(1,1,600,2,3)` |

The mechanical audit concludes `discovery-incumbent-wins`, with zero headless
hypotheses and zero active candidates. Its SHA-256 is
`d9ebdeb2fb0e677c9e821972dbbd7fe2ab0ebb38fe4e3c7a2c10ad30e104522c`.

## Interpretation and next boundary

The negative result must stand. Do not inspect the runner-up, change buckets,
move to a nearby panel-2 frame, or run the conditional r6 branch.

The scan also clarifies what the current COW can and cannot answer. It changes
one first action and then lets `NativeOfflineTeacher` own every later decision.
It is useful for rejecting locally poor actions, but it does not reproduce the
counterfactual trajectory of a one-step residual followed by the frozen UCB
incumbent. The factual `down_right` branch surviving 600 source ticks therefore
does not mean frozen UCB would have survived; the recorded Wine prefix reached
native fail-close eight frames later under its actual subsequent decisions.

Before collecting more near-duplicate Wine prefixes or spending another broad
COW grid, audit whether the immutable frozen-UCB decision state can be restored
at the retail checkpoint and advanced inside source COW. If it can, predeclare
a small policy-faithful first-action comparison on independent Wine anchors.
If it cannot be reproduced exactly, retain current COW as reject-only and use
new original-retail Wine shadow episodes for any future intervention instead.
