# Original-retail Wine baseline, 2026-08-09

## Outcome

The original Japanese TH06 1.02h executable completed every natural Practice
Stage and one ordinary Reimu-A Lunatic Start-to-Ending route under Wine. The
infrastructure breakthrough is real: the Linux VPS can now exercise original
retail collision/HIT behavior without an interactive desktop and retain exact
per-frame evidence.

The policy result is not NMNB. The six independent Practice runs accumulated
58 physical HITs. The full route reached Ending with 68 physical HITs. The
earlier 25-second Stage 6 smoke had zero HIT, but the complete natural Stage 6
run had 8; the smoke must not be cited as a Stage clear.

## Contract and provenance

- Scope: Lunatic, Reimu-A, Stages 6, 5, 4, 3, 2, 1, then ordinary full route.
- Completion: natural Practice result or original Supervisor Ending; no shared
  timeout (`seconds=0`).
- Retail executable SHA-256:
  `9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245`.
- Native DLL SHA-256:
  `d5c79c30b4d46c72f0521d9653d5d99693c0fbc966e241f554732ad3ade3a37e`.
- Unlock template SHA-256:
  `54cd436d5d8a7a904190c792a977bf270ab1cb759fd72101e51e94d26b749c71`.
- Runtime: Wine 11.0, Xvfb, 32-bit Windows Python 3.11.9, exact-PID
  background input, GDB startup normalization, Bomb-free input.
- Continuation: the verified one-byte life-exhaustion patch permits play after
  HIT. It does not suppress the original physical HIT signal.
- Policy: `phase-local-hierarchical-ucb-v4`, plug-in SHA-256
  `4d7f10925731d7f83389aaa8c2aa942d7ed156de54791767ff2ced802483bbf2`,
  default 3% exploration.

These are operational online-adaptive UCB baselines, not evaluations of the
offline distilled policy population. Each standalone Stage used and updated
its own Stage policy state. The full route used and updated a separate route
state. Therefore the standalone-versus-route differences are useful
variability diagnostics, not strict same-model deltas.

The run reports recorded base commit `29ba7b7`; the Wine validation worktree
was not yet committed during execution. The implementation is preserved by the
commits containing this report. Post-run hardening now records explicit policy
state hashes and the controller command, fields absent from these first
reports. Stage 6 used `/home/c/.wine`; Stages 5-to-1 and full route used the
dedicated `/home/c/.wine-th06-rl-retail` prefix. All reports recorded controller
return code 0, successful startup normalization, and zero leftover processes
in their exact prefix.

## Independent natural Practice runs

`Dead-end→HIT` counts HITs with at least one `control-dead-end:Hard safe set
empty` observation in the preceding 30 game frames. This is an association,
not proof that the dead-end caused the death.

| Order | Stage | Physical HITs | Decisions | Frame range | Max bullets | Capture fail-close | Native dead-ends | Dead-end→HIT | Wall time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 8 | 31,575 | 120–32,908 | 640 | 6 | 37 | 8 | 487.402 s |
| 2 | 5 | 9 | 20,221 | 93–23,624 | 640 | 3 | 50 | 9 | 284.373 s |
| 3 | 4 | 18 | 23,202 | 88–26,466 | 639 | 73 | 92 | 15 | 306.691 s |
| 4 | 3 | 9 | 18,033 | 96–21,868 | 640 | 169 | 35 | 6 | 217.242 s |
| 5 | 2 | 8 | 14,228 | 99–17,198 | 552 | 135 | 21 | 5 | 154.774 s |
| 6 | 1 | 6 | 9,338 | 86–12,381 | 326 | 258 | 13 | 4 | 129.474 s |
| **Total** | **1–6** | **58** | **116,597** | — | **640** | **644** | **248** | **47** | **1,579.954 s** |

Capture and solve latency stayed small relative to a 60 Hz frame budget in the
ordinary case, although long tails exist:

| Stage | Capture p50 / p95 / p99 / max | Solve p50 / p95 / p99 / max |
|---:|---:|---:|
| 6 | 6.350 / 11.836 / 15.588 / 56.224 ms | 0.930 / 2.916 / 4.442 / 14.767 ms |
| 5 | 5.460 / 9.348 / 13.473 / 129.138 ms | 0.797 / 1.541 / 2.220 / 21.436 ms |
| 4 | 5.489 / 12.928 / 19.092 / 68.153 ms | 0.826 / 1.907 / 6.393 / 41.259 ms |
| 3 | 5.092 / 9.548 / 19.343 / 74.260 ms | 0.785 / 1.795 / 3.886 / 64.007 ms |
| 2 | 5.032 / 8.290 / 21.544 / 64.508 ms | 0.855 / 1.596 / 6.064 / 15.247 ms |
| 1 | 6.140 / 9.426 / 17.587 / 57.048 ms | 0.823 / 1.539 / 8.285 / 34.191 ms |

## Full Start route

The route emitted all five legal source-scope transitions and reached the
original Ending state after 1,485.097 seconds and 119,062 policy decisions.

| Stage | Route HITs | Standalone HITs | Route minus standalone | Trace rows | Max frame | Native dead-ends | Dead-end→HIT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 6 | -1 | 11,359 | 12,380 | 18 | 4 |
| 2 | 6 | 8 | -2 | 15,390 | 17,199 | 22 | 4 |
| 3 | 14 | 9 | +5 | 19,723 | 23,337 | 53 | 9 |
| 4 | 22 | 18 | +4 | 23,158 | 27,141 | 82 | 21 |
| 5 | 9 | 9 | 0 | 23,006 | 23,694 | 36 | 9 |
| 6 | 12 | 8 | +4 | 33,048 | 33,719 | 44 | 12 |
| **Total** | **68** | **58** | **+10** | **125,684** | — | **255** | **59** |

The route observed at most 640 bullets, 548 incoherent-capture fail-closes, and
capture latency p50/p95/p99/max of 5.449/10.791/16.777/125.935 ms. Solve
latency was 0.833/1.766/4.214/62.145 ms.

The adaptive states after these runs are retained locally for diagnosis. Their
hashes identify the observed UCB tables, but they are not immutable input-model
hashes because the policy learned and checkpointed during each run:

| Scope | Final state SHA-256 | Bytes | Observed result |
|---|---|---:|---:|
| Stage 1 | `28481f81f8079158554c0dbef50800b68a6e715416a7fd537938bb9f08c94c5e` | 337,509 | 6 HIT |
| Stage 2 | `8d28e2f0069af9db501c4a1bd4c42d5da439342f04602c6fd68b3b78c90a10eb` | 528,605 | 8 HIT |
| Stage 3 | `705deac9c8f85e93e51bfd14cdb34c516d2d4fdafb381b806428516e4af3f33a` | 605,653 | 9 HIT |
| Stage 4 | `c817ffee79396471b8b1b865ac302332d5a370a486cc18281528b3e55a6fead7d` | 880,589 | 18 HIT |
| Stage 5 | `35ba907e088ecd475744b802f7d2d3c1eeb0f88c0d508136c88cb570b6396748` | 734,941 | 9 HIT |
| Stage 6 | `e2c28f8e9c0bb1cf917c8204809f8cb163fe359bf7b71df4dc1f90619e3bf6a0` | 1,124,661 | 8 HIT |
| Full route | `3e6b01a0136b88a25ade603eae5c4017275a79b9c0e1906bcde6b0c197c5504a` | 4,296,901 | 68 HIT |

## Evidence manifest

Artifacts are ignored locally. These hashes make accidental replacement
detectable:

| Run | Artifact directory | `report.json` SHA-256 | `trace.jsonl` SHA-256 |
|---|---|---|---|
| Stage 6 | `artifacts/wine-retail-baseline-stage6-natural-r1` | `85bfe4e95338de00252dabeeb0e0498c71cd3a943694f825d4f61f560d40985b` | `12e8da2f8ba554cb9ca380ac915a5d66e8ae6e021133fbe4d0a5794d0c1687e9` |
| Stage 5 | `artifacts/wine-retail-baseline-stage5-natural-r3` | `4058af78d5bc665052fca85aa561ac54a475f29d14dc646bf42f9b856df7e224` | `9f34b176608fb713d07c725c1f346c165cd5fcea6a8eda859d215ef2e9faca87` |
| Stage 4 | `artifacts/wine-retail-baseline-stage4-natural-r1` | `952bc02e493d87dc44d01baad5aa209bca7ec74216a8d1bc3ff0d57d366d678e` | `c062dec46f25577e2b8fcd87bfdd92487141d871faad404b63e8bd0c816c043d` |
| Stage 3 | `artifacts/wine-retail-baseline-stage3-natural-r1` | `8a98633f7b207e6eafb72c2482f93a4d5679b87e2efff4aef22e0d68dd662592` | `add76944a20d783b41f5252b2c43d2f03b070cfb4fc735a7a31755968e80db11` |
| Stage 2 | `artifacts/wine-retail-baseline-stage2-natural-r1` | `c73fbdae13892b684b1e39cc5ef75d5202b1c8831caf9118d87b3258a82137c5` | `fa16951da33e2f23a685d013a2351ed9a2c69936d9453c96a986b2319e5e7828` |
| Stage 1 | `artifacts/wine-retail-baseline-stage1-natural-r2` | `806d1b1d9f158b82a48bd4257a7a307ec43f1ee4234ba6c38f5e5a3929fabe5e` | `a132f2528a3788fea995f8d5c3537ddc55c3744da0defcb3ce6756c8e93329cf` |
| Full route | `artifacts/wine-retail-baseline-full-route-r1` | `38c0a10f482db59f0b481ce0fd086fd1b1543e346d39a5e296e31261049b3ced` | `74e828a2cf8bb1821ff56f8a1d68508b644e5b84911298fde0bf61d9ab40a954` |

Stage 1 `r1` was intentionally interrupted when the requested order changed.
Stage 5 `r1` stopped during prefix initialization and `r2` exposed a corrupt
prefix; neither is included. No natural run was replaced or rerun merely to
improve its HIT count.

## What the data says

1. Original-retail headless automation is viable. Capture, background input,
   continuation, stage transitions, and exact cleanup all worked through an
   entire game.
2. Current policy quality is far from NMNB. Reaching Ending with the life patch
   validates infrastructure, not survival.
3. Native dead-ends are the strongest immediate diagnostic. They precede
   47/58 standalone HITs and 59/68 route HITs within 30 frames. All Stage 5 and
   Stage 6 route HITs have that signature. The next investigation must separate
   genuinely unavoidable states from late arrival caused by earlier ranking,
   capture gaps, and delivery differences.
4. Stage 4 is the largest measured bottleneck: 18 standalone and 22 route HITs.
   It should be audited alongside Stage 5/6 rather than assuming later stages
   alone dominate.
5. Capture incoherence remains operationally relevant. Releasing input is the
   correct authority behavior, but hundreds of releases can affect survival.
   Better coherent capture may improve outcomes without weakening fail-close.
6. This experiment says nothing yet about the original-retail performance of
   the offline distilled candidates. Their Linux ranking remains experiment
   ordering, not Windows/Wine promotion evidence.

## Decision gates for a later session

No further runs are started by this report. The next session can choose among:

- add an immutable policy-evaluation mode: exploration zero, read-only copied
  state, before/after hash equality, and repeated paired trials;
- connect selected offline population members to the original-retail runner;
- audit the Stage 4-to-6 pre-HIT dead-end neighborhoods against exact source
  geometry and fresh issue delivery;
- reduce coherent-capture failures without relaxing authority;
- compare identical action streams across native Linux, Win32 reconstructed
  source under Wine, and original retail Wine before real-Windows promotion.
